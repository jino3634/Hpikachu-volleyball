// ppo_policy_v1.js
'use strict';

/**
 * PPO Policy (Actor-Critic) for tuple action space:
 *   xDirection: -1|0|1   (3-class categorical)
 *   yDirection: -1|0|1   (3-class categorical)  (y=-1 is jump)
 *   powerHit: 0|1        (2-class categorical)
 *
 * This policy is intentionally dependency-free and uses simple SGD.
 *
 * Notes:
 * - We keep the same feature builder as TuplePolicyV1 (featureLen = 16 by default).
 * - We keep basic action validity constraints (can't act while lying/diving, etc).
 * - PPO update uses clipped objective, value loss, and (optional) entropy bonus.
 */

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function tanh(x) {
  // stable tanh
  if (x > 10) return 1;
  if (x < -10) return -1;
  const e2x = Math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    exps[i] = v;
    sum += v;
  }
  if (sum <= 0) {
    const p = 1 / logits.length;
    for (let i = 0; i < exps.length; i++) exps[i] = p;
    return exps;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

function sampleCategorical(probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

function logProbFromProbs(probs, idx) {
  const p = Math.max(1e-12, Math.min(1, probs[idx] ?? 0));
  return Math.log(p);
}


function clamp01(x) {
  return (x < 0) ? 0 : (x > 1 ? 1 : x);
}
function renorm3(a, b, c) {
  const s = a + b + c;
  if (s <= 1e-12) return [1/3, 1/3, 1/3];
  return [a / s, b / s, c / s];
}
function entropyFromProbs(p) {
  let e = 0;
  for (let i = 0; i < p.length; i++) {
    const v = p[i];
    if (v > 0) e += -v * Math.log(v);
  }
  return e;
}
function maxProb(p) {
  let m = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > m) m = p[i];
  return m;
}

function mapClassToXDir(ax) {
  // 0->-1, 1->0, 2->+1
  return (ax === 0) ? -1 : (ax === 2 ? 1 : 0);
}
function mapClassToYDir(ay) {
  // 0->-1, 1->0, 2->+1
  return (ay === 0) ? -1 : (ay === 2 ? 1 : 0);
}
function mapXDirToClass(x) {
  return x < 0 ? 0 : (x > 0 ? 2 : 1);
}
function mapYDirToClass(y) {
  return y < 0 ? 0 : (y > 0 ? 2 : 1);
}

function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

export class PpoPolicyV1 {
  /**
   * @param {{
   *   featureLen?: number,
   *   learningRate?: number,
   *   imitationLr?: number,
   *   hidden1?: number,
   *   hidden2?: number,
   *   initStd?: number,
   *   activation?: 'tanh'|'linear',
   *   clipEps?: number,
   *   vfCoef?: number,
   *   entCoef?: number,
   *   gamma?: number,
   *   gaeLambda?: number,
   * }} opts
   */
  constructor(opts = {}) {
    this.kind = 'ppo_policy_v1';
    // buildFeatures가 0..23까지 쓰므로 최소 24는 보장해야 함
    this.featureLen = Math.max(24, (opts.featureLen ?? 24) | 0);

    // debug/diagnostics counters (for training stability)
    this.debug = {
      nanFeatures: 0,
      invalidFeatureSteps: 0,
      forcedIdle: 0,
      forcedIdleNoAct: 0,
      forcedIdleLying: 0,
      forcedIdleDiving: 0,
      powerHitSampled: 0,
      powerMasked: 0,
      powerGate: {
        // counts
        requested: 0,       // power=1 sampled/selected (before gating)
        allowed: 0,         // power=1 actually applied
        blockedNotAir: 0,
        blockedDX: 0,
        blockedDY: 0,
        blockedTTL: 0,

        // raw attempt stats (requested)
        sumDX_req: 0, sumDY_req: 0, sumTTL_req: 0, count_req: 0,

        // allowed stats
        sumDX_allow: 0, sumDY_allow: 0, sumTTL_allow: 0, count_allow: 0,

        // blocked stats
        sumDX_block: 0, sumDY_block: 0, sumTTL_block: 0, count_block: 0,
      },
    
// Observation sanity stats (per act call, for debugging schema issues)
obsStats: {
  count: 0,
  sumAbsMeX: 0, sumAbsMeY: 0,
  sumAbsBallX: 0, sumAbsBallY: 0,
  sumAbsDx: 0, sumAbsDy: 0,
},
};
    // Extended diagnostics (rolling; reset by Trainer after each flush)
    this.debug.featStats = null; // lazily initialized in buildFeatures()
    this.debug.actionStats = {
      n: 0,
      entX: 0, entY: 0, entP: 0,
      maxX: 0, maxY: 0, maxP: 0,
      axCounts: [0,0,0],
      ayCounts: [0,0,0],
      apCounts: [0,0],
      powerHitTotal: 0,
      powerHitNearBall: 0,
    };

    this.debug.maskStats = {
      n: 0,

      // environment split
      groundN: 0,
      airN: 0,

      // chosen y class counts (after masking)
      yChosenCounts: [0,0,0], // classes 0(-1),1(0),2(+1)

      // mass removed by masks (sum of prob mass masked away)
      yNegMaskedMass: 0,
      yNegMaskedCount: 0,
      xZeroMaskedMass: 0,
      xZeroMaskedCount: 0,

      // how often final sampled action would have been illegal without renorm (diagnostic; should stay 0)
      illegalYSampledPrevented: 0,
      illegalXSampledPrevented: 0,
    };
    this.debug.lastUpdate = null; // {policyLoss,valueLoss,clipFrac,gradNorm,wNorm,advMean,advStd,retMean,retStd,rewMean,rewStd}

  this.hidden1 = Math.max(1, (opts.hidden1 ?? 64) | 0);
    this.hidden2 = Math.max(1, (opts.hidden2 ?? 64) | 0);
    this.activation = (opts.activation === 'linear') ? 'linear' : 'tanh';

    this.learningRate = Number(opts.learningRate ?? 3e-4);
    // imitation(BC) 전용 lr. 지정 안 하면 learningRate 사용
     this.imitationLr = Number(opts.imitationLr ?? this.learningRate);
    this.initStd = Number(opts.initStd ?? 0.02);

    // PPO hyperparams
    this.clipEps = Number(opts.clipEps ?? 0.2);
    this.vfCoef = Number(opts.vfCoef ?? 0.5);
    this.entCoef = Number(opts.entCoef ?? 0.0); // set >0 if you also implement entropy grads
    this.gamma = Number(opts.gamma ?? 0.995);
    this.gaeLambda = Number(opts.gaeLambda ?? 0.95);

    // trunk weights
    this.W1 = []; // [H1][F]
    this.b1 = new Float32Array(this.hidden1);
    this.W2 = []; // [H2][H1]
    this.b2 = new Float32Array(this.hidden2);

    // actor heads
    this.Wx = []; // [3][H2]
    this.bx = new Float32Array(3);
    this.Wy = []; // [3][H2]
    this.by = new Float32Array(3);
    this.Wp = []; // [2][H2]
    this.bp = new Float32Array(2);

    // critic head
    this.Wv = new Float32Array(this.hidden2); // [H2]
    this.bv = 0;

    this._initWeights();
  }

  _initWeights() {
    const s = this.initStd;

    // W1
    this.W1 = new Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) {
      const row = new Float32Array(this.featureLen);
      for (let j = 0; j < this.featureLen; j++) row[j] = randn() * s;
      this.W1[i] = row;
      this.b1[i] = 0;
    }

    // W2
    this.W2 = new Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) {
      const row = new Float32Array(this.hidden1);
      for (let j = 0; j < this.hidden1; j++) row[j] = randn() * s;
      this.W2[i] = row;
      this.b2[i] = 0;
    }

    // heads
    this.Wx = new Array(3);
    for (let k = 0; k < 3; k++) {
      const row = new Float32Array(this.hidden2);
      for (let j = 0; j < this.hidden2; j++) row[j] = randn() * s;
      this.Wx[k] = row;
      this.bx[k] = 0;
    }

    this.Wy = new Array(3);
    for (let k = 0; k < 3; k++) {
      const row = new Float32Array(this.hidden2);
      for (let j = 0; j < this.hidden2; j++) row[j] = randn() * s;
      this.Wy[k] = row;
      this.by[k] = 0;
    }

    this.Wp = new Array(2);
    for (let k = 0; k < 2; k++) {
      const row = new Float32Array(this.hidden2);
      for (let j = 0; j < this.hidden2; j++) row[j] = randn() * s;
      this.Wp[k] = row;
      this.bp[k] = 0;
    }

    for (let j = 0; j < this.hidden2; j++) this.Wv[j] = randn() * s;
    this.bv = 0;
  }

  /**
   * Feature builder (compatible with previous TuplePolicyV1 feature schema).
   * featureLen = 16.
   * @param {any} obs
   * @param {1|2} playerIndex
   * @returns {Float32Array}
   */
  buildFeatures(obs, playerIndex) {
    const me = obs?.me ?? {};
    const opp = obs?.opp ?? {};
    const ball = obs?.ball ?? {};

    // Expanded feature vector (default 24):
    // 0  me.x
    // 1  me.y
    // 2  me.yV
    // 3  me.isAir
    // 4  me.isDiving
    // 5  me.isLying
    // 6  me.canAct
    // 7  me.stateNorm (0..1)
    // 8  opp.x
    // 9  opp.y
    // 10 opp.yV
    // 11 opp.isAir
    // 12 opp.isDiving
    // 13 opp.isLying
    // 14 ball.x
    // 15 ball.y
    // 16 ball.xV
    // 17 ball.yV
    // 18 ball.landingX
    // 19 ball.timeToLand
    // 20 ball.isPowerHit
    // 21 me.divingDirNorm (-1..1)
    // 22 opp.stateNorm (0..1)
    // 23 me.isServe (0/1) if provided else 0
    const f = new Float32Array(this.featureLen);

    const meState = Number(me.state ?? 0);
    const oppState = Number(opp.state ?? 0);

    // Booleans are encoded as 0/1.
    const meIsLying = !!me.isLying || (Number(me.lying ?? 0) > 0) || meState === 4;
    const meIsDiving = !!me.isDiving || meState === 3;
    const meCanAct = (me.canAct !== undefined) ? !!me.canAct : (!meIsLying && !meIsDiving);
    const meIsAir = (me.isAir !== undefined) ? !!me.isAir : (meState === 1 || meState === 2);

    const oppIsLying = !!opp.isLying || (Number(opp.lying ?? 0) > 0) || oppState === 4;
    const oppIsDiving = !!opp.isDiving || oppState === 3;
    const oppIsAir = (opp.isAir !== undefined) ? !!opp.isAir : (oppState === 1 || oppState === 2);

    // Fill (guarded)
    f[0] = Number(me.x ?? 0);
    f[1] = Number(me.y ?? 0);
    f[2] = Number(me.yV ?? me.yv ?? 0);
    f[3] = meIsAir ? 1 : 0;
    f[4] = meIsDiving ? 1 : 0;
    f[5] = meIsLying ? 1 : 0;
    f[6] = meCanAct ? 1 : 0;
    f[7] = Math.max(0, Math.min(1, meState / 4));

    f[8] = Number(opp.x ?? 0);
    f[9] = Number(opp.y ?? 0);
    f[10] = Number(opp.yV ?? opp.yv ?? 0);
    f[11] = oppIsAir ? 1 : 0;
    f[12] = oppIsDiving ? 1 : 0;
    f[13] = oppIsLying ? 1 : 0;

    f[14] = Number(ball.x ?? 0);
    f[15] = Number(ball.y ?? 0);
    f[16] = Number(ball.xV ?? ball.xv ?? 0);
    f[17] = Number(ball.yV ?? ball.yv ?? 0);
    f[18] = Number(ball.landingX ?? ball.expectedX ?? 0);
    f[19] = Number(ball.timeToLand ?? 0);
    f[20] = (ball.isPowerHit !== undefined) ? (ball.isPowerHit ? 1 : 0) : 0;

    // divingDir is typically -1/0/1 (player-centric). Clamp to [-1,1].
    const dd = Number(me.divingDir ?? 0);
    f[21] = Math.max(-1, Math.min(1, dd));

    f[22] = Math.max(0, Math.min(1, oppState / 4));
    f[23] = (me.isServe !== undefined) ? (me.isServe ? 1 : 0) : 0;

    // NaN/Inf guard: replace with 0 and count.
    let invalid = false;
    for (let i = 0; i < this.featureLen; i++) {
      const v = f[i];
      if (!Number.isFinite(v)) {
        invalid = true;
        this.debug.nanFeatures++;
        f[i] = 0;
      }
    }
    if (invalid) this.debug.invalidFeatureSteps++;

    // Rolling feature statistics (raw; helpful to detect saturation / scaling issues)
    // Structure: { n, min[], max[], sum[], sumsq[] }
    if (!this.debug.featStats) {
      const min = new Float64Array(this.featureLen);
      const max = new Float64Array(this.featureLen);
      const sum = new Float64Array(this.featureLen);
      const sumsq = new Float64Array(this.featureLen);
      for (let i = 0; i < this.featureLen; i++) { min[i] = Infinity; max[i] = -Infinity; sum[i] = 0; sumsq[i] = 0; }
      this.debug.featStats = { n: 0, min, max, sum, sumsq };
    }
    const fs = this.debug.featStats;
    fs.n++;
    for (let i = 0; i < this.featureLen; i++) {
      const v = f[i];
      if (v < fs.min[i]) fs.min[i] = v;
      if (v > fs.max[i]) fs.max[i] = v;
      fs.sum[i] += v;
      fs.sumsq[i] += v * v;
    }
    return f;
  }

  _forward(feat) {
    // trunk: feat -> h1 -> h2
    const h1 = new Float32Array(this.hidden1);
    const z1 = new Float32Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) {
      let sum = this.b1[i];
      const w = this.W1[i];
      for (let j = 0; j < this.featureLen; j++) sum += w[j] * feat[j];
      z1[i] = sum;
      h1[i] = (this.activation === 'linear') ? sum : tanh(sum);
    }

    const h2 = new Float32Array(this.hidden2);
    const z2 = new Float32Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) {
      let sum = this.b2[i];
      const w = this.W2[i];
      for (let j = 0; j < this.hidden1; j++) sum += w[j] * h1[j];
      z2[i] = sum;
      h2[i] = (this.activation === 'linear') ? sum : tanh(sum);
    }

    // logits
    const lx = new Float32Array(3);
    const ly = new Float32Array(3);
    const lp = new Float32Array(2);

    for (let k = 0; k < 3; k++) {
      let sum = this.bx[k];
      const w = this.Wx[k];
      for (let j = 0; j < this.hidden2; j++) sum += w[j] * h2[j];
      lx[k] = sum;
    }
    for (let k = 0; k < 3; k++) {
      let sum = this.by[k];
      const w = this.Wy[k];
      for (let j = 0; j < this.hidden2; j++) sum += w[j] * h2[j];
      ly[k] = sum;
    }
    for (let k = 0; k < 2; k++) {
      let sum = this.bp[k];
      const w = this.Wp[k];
      for (let j = 0; j < this.hidden2; j++) sum += w[j] * h2[j];
      lp[k] = sum;
    }

    // value
    let v = this.bv;
    for (let j = 0; j < this.hidden2; j++) v += this.Wv[j] * h2[j];

    return { feat, h1, z1, h2, z2, lx, ly, lp, v };
  }

  /**
   * Apply action constraints by masking logits.
   * @param {Float32Array} logitsX
   * @param {Float32Array} logitsY
   * @param {Float32Array} logitsP
   * @param {any} obs
   * @returns {{px:Float32Array, py:Float32Array, pp:Float32Array}}
   */
_maskedProbs(logitsX, logitsY, logitsP, obs) {
  const me = obs?.me ?? {};
  const state = Number(me.state ?? 0);
  const isLying = !!me.isLying || state === 4;
  const isDiving = !!me.isDiving || state === 3;
  const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
  const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

  const ball = obs?.ball ? obs.ball : {};
  const meX = Number(me.x ?? 0);
  const meY = Number(me.y ?? 0);
  const ballX = Number(ball.x ?? 0);
  const ballY = Number(ball.y ?? 0);
  const dx = Math.abs(ballX - meX);
  const dy = Math.abs(ballY - meY);

  // NOTE: 너 코드에서는 ball.timeToLand를 쓰고 있는데,
  // 이 함수에서는 더 이상 timeToLand / dxOk / dyOk gate를 쓰지 않으니 제거해도 됨.
  // const timeToLand = Number(ball.timeToLand ?? 1);
  // timeToLand는 ball에 들어온다(정규화 관측에서 채워줌)
  const timeToLand = Number(ball.timeToLand ?? 1);

  // --- Obs sanity accumulation (detect constant/zeroed observations) ---
  {
    const os = this.debug.obsStats;
    os.count++;
    os.sumAbsMeX += Math.abs(meX);
    os.sumAbsMeY += Math.abs(meY);
    os.sumAbsBallX += Math.abs(ballX);
    os.sumAbsBallY += Math.abs(ballY);
    os.sumAbsDx += (typeof dx === 'number' ? dx : 0);
    os.sumAbsDy += (typeof dy === 'number' ? dy : 0);
  }

  // If cannot act at all, force IDLE effectively.
  if (!canAct) {
    const px = new Float32Array([0, 1, 0]);
    const py = new Float32Array([0, 1, 0]);
    const pp = new Float32Array([1, 0]);
    return { px, py, pp };
  }

  const mx = new Float32Array(logitsX);
  const my = new Float32Array(logitsY);
  const mp = new Float32Array(logitsP);

  // Base constraint: on ground, forbid DOWN (+1). Jump is y=-1.
  if (!isAir) {
    // y classes: 0->-1 (jump), 1->0 (idle), 2->+1 (down)
    my[2] = -1e9;
  }

  // ✅ 우리가 합의한 하드 금지 규칙(소프트 억제 없음):
  // - lying 또는 diving 상태에서는 powerHit=1을 하드 금지
  // - 그 외(지상/공중 포함)는 powerHit 완전 허용
  if (isLying || isDiving) {
    mp[1] = -1e9;
  }

  const px0 = softmax(mx);
  const py0 = softmax(my);
  const pp0 = softmax(mp);
  return { px: px0, py: py0, pp: pp0 };
}


  /**
   * Evaluate: compute action distribution + value for obs.
   * @param {any} obs
   * @param {1|2} playerIndex
   */
  evaluate(obs, playerIndex) {
    const feat = this.buildFeatures(obs, playerIndex);
    const fwd = this._forward(feat);
    const { px, py, pp } = this._maskedProbs(fwd.lx, fwd.ly, fwd.lp, obs);
    return { px, py, pp, value: fwd.v, feat, fwd };
  }

  /**
   * Act and return {action, logp, value}.
   * @param {any} obs
   * @param {1|2} playerIndex
   * @param {{deterministic?:boolean, epsilon?:number}} [opts]
   */
act(obs, playerIndex, opts = {}) {
  const deterministic = !!opts.deterministic;
  const epsRand = Number(opts.epsilon ?? 0);

  const me = obs?.me ?? {};
  const state = Number(me.state ?? 0);
  const isLying = !!me.isLying || state === 4;
  const isDiving = !!me.isDiving || state === 3;
  const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
  const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

  if (!canAct) {
    // Truly cannot act: force IDLE and skip learning.
    this.debug.forcedIdle++;
    this.debug.forcedIdleNoAct++;
    return {
      action: { xDirection: 0, yDirection: 0, powerHit: 0 },
      logp: 0,
      value: 0,
      meta: { forcedIdle: true, reason: 'noAct' },
    };
  }

  // Diving / lying: do not force IDLE. Apply soft mask later.
  // Mark as forcedIdle for learning-skip (episode_builder will skip these steps).
  const skipLearn = (isLying || isDiving);
  if (skipLearn) {
    this.debug.forcedIdle++;
    if (isLying) this.debug.forcedIdleLying++;
    if (isDiving) this.debug.forcedIdleDiving++;
  }

  const { px, py, pp, value } = this.evaluate(obs, playerIndex);
  if (this.debug.maskStats) this.debug.maskStats.n++;
  const as = this.debug.actionStats;
  const recordActionStats = (pxEff, pyEff, ppEff, ax, ay, ap) => {
    as.n++;
    as.entX += entropyFromProbs(pxEff);
    as.entY += entropyFromProbs(pyEff);
    as.entP += entropyFromProbs(ppEff);
    as.maxX += maxProb(pxEff);
    as.maxY += maxProb(pyEff);
    as.maxP += maxProb(ppEff);

    as.axCounts[ax] = (as.axCounts[ax] ?? 0) + 1;
    as.ayCounts[ay] = (as.ayCounts[ay] ?? 0) + 1;
    as.apCounts[ap] = (as.apCounts[ap] ?? 0) + 1;

    // ✅ (보완 2) act(): powerGate 디버그 블록을 새 allowPowerHit 정의(=lying/diving만 금지)에 맞게 "최소 수정"
    //
    // 기존 powerGate 블록의 blockedNotAir / blockedDX / blockedDY / blockedTTL 는 의미가 깨졌으니,
    // 아래처럼 "blockedLying / blockedDiving"만 집계하게 바꿔.
    // (필드가 없을 수 있으니 0 초기화도 안전하게 포함)

    if (ap === 1) {

      if (this.debug && this.debug.powerGate) {
        const pg = this.debug.powerGate;

        // 안전 초기화(없으면 생성)
        pg.requested = pg.requested ?? 0;
        pg.allowed = pg.allowed ?? 0;
        pg.blockedLying = pg.blockedLying ?? 0;
        pg.blockedDiving = pg.blockedDiving ?? 0;

        // 기존 통계가 남아있어도 무방(원하면 지워도 됨)
        pg.sumDX_req = pg.sumDX_req ?? 0;
        pg.sumDY_req = pg.sumDY_req ?? 0;
        pg.sumTTL_req = pg.sumTTL_req ?? 0;
        pg.count_req = pg.count_req ?? 0;

        pg.sumDX_allow = pg.sumDX_allow ?? 0;
        pg.sumDY_allow = pg.sumDY_allow ?? 0;
        pg.sumTTL_allow = pg.sumTTL_allow ?? 0;
        pg.count_allow = pg.count_allow ?? 0;

        pg.sumDX_block = pg.sumDX_block ?? 0;
        pg.sumDY_block = pg.sumDY_block ?? 0;
        pg.sumTTL_block = pg.sumTTL_block ?? 0;
        pg.count_block = pg.count_block ?? 0;

        // 집계(여기서 dxN/dyN/tLandN는 이미 위에서 계산해둔 값 그대로 사용)
        pg.requested++;
        pg.sumDX_req += dxN; pg.sumDY_req += dyN; pg.sumTTL_req += tLandN; pg.count_req++;

        if (allowPowerHit) {
          pg.allowed++;
          pg.sumDX_allow += dxN; pg.sumDY_allow += dyN; pg.sumTTL_allow += tLandN; pg.count_allow++;
        } else {
          // allowPowerHit=false는 이제 (isLying||isDiving) 뿐
          pg.sumDX_block += dxN; pg.sumDY_block += dyN; pg.sumTTL_block += tLandN; pg.count_block++;
          if (isLying) pg.blockedLying++;
          if (isDiving) pg.blockedDiving++;
        }
      }
    }

  };

  // ✅ Power-hit gate 제거(소프트 억제/근접/TTL gate 전부 제거)
  // ✅ 하드 금지 조건은 오직 lying/diving
  const allowPowerHit = !(isLying || isDiving);

  // (아래 dxN/dyN/tLandN는 기존 디버그/통계에서 쓰이므로 "남겨도 됨")
  // powerGate 디버그는 기존 allowPowerHit(근접+TTL) 기준을 전제로 만들어졌으니,
  // 이 버전에서는 의미가 바뀐다. (원하면 아래 디버그 블록도 같이 정리해야 함)
  const ballN = obs?.ball ?? {};
  const dxN = Math.abs(Number(ballN.x ?? 0) - Number(me.x ?? 0));
  const dyN = Math.abs(Number(ballN.y ?? 0) - Number(me.y ?? 0));
  const tLandN = Number(ballN.timeToLand ?? 1);

  // epsilon random exploration (still valid)
  if (!deterministic && epsRand > 0 && Math.random() < epsRand) {
    // 0) epsilon 랜덤 액션 먼저 결정 (유효성 최소 보장)
    let ax = (Math.random() * 3) | 0; // 0..2
    let ay = (Math.random() * 3) | 0; // 0..2
    let ap = (Math.random() < 0.5) ? 1 : 0;

    // 공중에서는 y=-1(클래스0) 금지(네 정책과 동일)
    if (isAir && ay === 0) ay = 1;

    // ✅ powerHit gate: lying/diving일 때만 막기
    if (!allowPowerHit) ap = 0;

    const powerHit = ap ? 1 : 0;

    // 1) logp 계산용 eff 분포 구성(메인 경로와 일치)
    const ppEff = (!allowPowerHit) ? [1, 0] : (() => {
      const a0 = clamp01(pp[0] ?? 0);
      const b0 = clamp01(pp[1] ?? 0);
      const s0 = a0 + b0;
      return (s0 > 1e-12) ? [a0 / s0, b0 / s0] : [0.5, 0.5];
    })();

    const pxEff = (() => {
      const a0 = clamp01(px[0] ?? 0), b0 = clamp01(px[1] ?? 0), c0 = clamp01(px[2] ?? 0);
      // ✅ 지상 powerHit=1 & x=0 하드 금지(이미 너가 채택한 룰)
      if (!isAir && ap === 1) return renorm3(a0, 0, c0);
      const s0 = a0 + b0 + c0;
      return (s0 > 1e-12) ? [a0 / s0, b0 / s0, c0 / s0] : [1/3, 1/3, 1/3];
    })();

    const pyEff = (() => {
      const a0 = clamp01(py[0] ?? 0), b0 = clamp01(py[1] ?? 0), c0 = clamp01(py[2] ?? 0);
      if (isAir) {
        const s0 = b0 + c0;
        return (s0 > 1e-12) ? [0, b0 / s0, c0 / s0] : [0, 0.5, 0.5];
      }
      const s0 = a0 + b0 + c0;
      return (s0 > 1e-12) ? [a0 / s0, b0 / s0, c0 / s0] : [1/3, 1/3, 1/3];
    })();

    // 2) 이제 안전하게 기록/로그확률 계산 가능
    recordActionStats(pxEff, pyEff, ppEff, ax, ay, ap);
    const logp =
      logProbFromProbs(pxEff, ax) +
      logProbFromProbs(pyEff, ay) +
      logProbFromProbs(ppEff, ap);

    return {
      action: { xDirection: mapClassToXDir(ax), yDirection: mapClassToYDir(ay), powerHit },
      logp,
      value,
      meta: { epsRandom: true, ax, ay, ap, forcedIdle: skipLearn, reason: (skipLearn ? (isLying ? 'lying' : 'diving') : null) },
    };
  }

  // deterministic = argmax, else sample (with conditional masks applied consistently to sampling + logp)
  let ax = 1, ay = 1, ap = 0;

  // ✅ Power gate probabilities: 이제 lying/diving일 때만 [1,0]
  const ppEff = (!allowPowerHit) ? [1, 0] : (() => {
    const a = clamp01(pp[0] ?? 0);
    const b = clamp01(pp[1] ?? 0);
    const s = a + b;
    return (s > 1e-12) ? [a / s, b / s] : [0.5, 0.5];
  })();

  // 1) sample/choose ap first
  if (deterministic) {
    ap = (ppEff[1] > ppEff[0]) ? 1 : 0;
  } else {
    ap = sampleCategorical(ppEff);
    // ✅ 사실상 allowPowerHit=false일 때는 ppEff=[1,0]이라 ap=1이 안 나오지만 안전장치 유지
    if (ap === 1 && !allowPowerHit) { ap = 0; this.debug.powerMasked++; }
  }

  // 2) conditional X distribution: if ground && ap==1 => forbid x=0 (class 1)
  const pxEff = (() => {
    const a = clamp01(px[0] ?? 0);
    const b = clamp01(px[1] ?? 0);
    const c = clamp01(px[2] ?? 0);
    if (!isAir && ap === 1) {
      if (this.debug.maskStats) { this.debug.maskStats.xZeroMaskedCount++; this.debug.maskStats.xZeroMaskedMass += (px[1] ?? 0); }
      return renorm3(a, 0, c);
    }
    const s = a + b + c;
    return (s > 1e-12) ? [a / s, b / s, c / s] : [1/3, 1/3, 1/3];
  })();

  if (deterministic) {
    ax = (pxEff[1] >= pxEff[0] && pxEff[1] >= pxEff[2]) ? 1 : ((pxEff[2] > pxEff[0]) ? 2 : 0);
    if (!isAir && ap === 1 && ax === 1 && this.debug.maskStats) this.debug.maskStats.illegalXSampledPrevented++;
  } else {
    ax = sampleCategorical(pxEff);
    if (!isAir && ap === 1 && ax === 1 && this.debug.maskStats) this.debug.maskStats.illegalXSampledPrevented++;
  }

  // 3) Y distribution (keep current behavior: if air forbid y=-1 to avoid double-jump)
  const pyEff = (() => {
    const a = clamp01(py[0] ?? 0);
    const b = clamp01(py[1] ?? 0);
    const c = clamp01(py[2] ?? 0);
    if (isAir) {
      if (this.debug.maskStats) { this.debug.maskStats.yNegMaskedCount++; this.debug.maskStats.yNegMaskedMass += (py[0] ?? 0); }
      const s = b + c;
      return (s > 1e-12) ? [0, b / s, c / s] : [0, 0.5, 0.5];
    }
    const s = a + b + c;
    return (s > 1e-12) ? [a / s, b / s, c / s] : [1/3, 1/3, 1/3];
  })();

  if (deterministic) {
    ay = (pyEff[1] >= pyEff[0] && pyEff[1] >= pyEff[2]) ? 1 : ((pyEff[2] > pyEff[0]) ? 2 : 0);
    if (isAir && ay === 0 && this.debug.maskStats) this.debug.maskStats.illegalYSampledPrevented++;
  } else {
    ay = sampleCategorical(pyEff);
    if (isAir && ay === 0 && this.debug.maskStats) this.debug.maskStats.illegalYSampledPrevented++;
  }

  if (this.debug.maskStats) {
    if (!isAir) this.debug.maskStats.groundN++; else this.debug.maskStats.airN++;
    const yc = this.debug.maskStats.yChosenCounts;
    yc[ay] = (yc[ay] ?? 0) + 1;
  }

  const logp = logProbFromProbs(pxEff, ax) + logProbFromProbs(pyEff, ay) + logProbFromProbs(ppEff, ap);
  const action = { xDirection: mapClassToXDir(ax), yDirection: mapClassToYDir(ay), powerHit: ap ? 1 : 0 };

  // diagnostics: count sampled power-hit actions (main policy path)
  if (ap === 1) {
    this.debug.powerHitSampled++; // requested power-hit (pre-gate)

    // NOTE: 기존 powerGate 디버그는 "근접+TTL+공중" gate를 전제로 함.
    // 지금은 allowPowerHit 의미가 "lying/diving이 아님"으로 바뀌었으니,
    // 아래 블록은 숫자 해석이 달라진다. (원하면 통째로 제거해도 됨)
    if (this.debug && this.debug.powerGate) {
      const pg = this.debug.powerGate;
      const dxv = dxN;
      const dyv = dyN;
      const ttlv = tLandN;

      pg.requested++;
      pg.sumDX_req += dxv; pg.sumDY_req += dyv; pg.sumTTL_req += ttlv; pg.count_req++;

      if (allowPowerHit) {
        pg.allowed++;
        pg.sumDX_allow += dxv; pg.sumDY_allow += dyv; pg.sumTTL_allow += ttlv; pg.count_allow++;
      } else {
        pg.sumDX_block += dxv; pg.sumDY_block += dyv; pg.sumTTL_block += ttlv; pg.count_block++;
        // blockedNotAir/dx/dy/ttl 등은 더 이상 의미 없음(원하면 지워라)
      }
    }
  }

  recordActionStats(pxEff, pyEff, ppEff, ax, ay, ap);
  return { action, logp, value, meta: { ax, ay, ap, forcedIdle: skipLearn, reason: (skipLearn ? (isLying ? 'lying' : 'diving') : null) } };
}


  /**
   * Compute log-prob and value under current params for a given (obs, action).
   * @param {any} obs
   * @param {1|2} playerIndex
   * @param {{xDirection:number,yDirection:number,powerHit:number}|number} action
   */
logpValue(obs, playerIndex, action) {
  const me = obs?.me ?? {};
  const state = Number(me.state ?? 0);
  const isLying = !!me.isLying || state === 4;
  const isDiving = !!me.isDiving || state === 3;
  const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

  // ✅ act()와 동일한 규칙:
  // - powerHit 하드 금지는 오직 lying/diving
  // - 지상/공중, 근접/TTL 조건으로는 막지 않음
  const allowPowerHit = !(isLying || isDiving);

  // (dxN/dyN/tLandN는 더 이상 gate에 쓰지 않으므로 삭제해도 됨)
  // const ballN = obs?.ball ?? {};
  // const dxN = Math.abs(Number(ballN.x ?? 0) - Number(me.x ?? 0));
  // const dyN = Math.abs(Number(ballN.y ?? 0) - Number(me.y ?? 0));
  // const tLandN = Number(ballN.timeToLand ?? 1);

  const a = (typeof action === 'number')
    ? { xDirection: 0, yDirection: 0, powerHit: 0 }
    : action;

  const ax = mapXDirToClass(Number(a.xDirection ?? 0));
  const ay = mapYDirToClass(Number(a.yDirection ?? 0));
  const ap = Number(a.powerHit ?? 0) ? 1 : 0;

  const { px, py, pp, value } = this.evaluate(obs, playerIndex);

  // ✅ allowPowerHit=false(=lying/diving)일 때만 powerHit=1을 확률 0으로 처리
  const ppEff = (!allowPowerHit) ? [1, 0] : (() => {
    const a0 = clamp01(pp[0] ?? 0), b0 = clamp01(pp[1] ?? 0);
    const s0 = a0 + b0;
    return (s0 > 1e-12) ? [a0 / s0, b0 / s0] : [0.5, 0.5];
  })();

  const pxEff = (() => {
    const a0 = clamp01(px[0] ?? 0), b0 = clamp01(px[1] ?? 0), c0 = clamp01(px[2] ?? 0);
    // ✅ 지상 powerHit=1 & x=0 하드 금지(네가 채택한 룰 유지)
    if (!isAir && ap === 1) return renorm3(a0, 0, c0);
    const s0 = a0 + b0 + c0;
    return (s0 > 1e-12) ? [a0 / s0, b0 / s0, c0 / s0] : [1/3, 1/3, 1/3];
  })();

  const pyEff = (() => {
    const a0 = clamp01(py[0] ?? 0), b0 = clamp01(py[1] ?? 0), c0 = clamp01(py[2] ?? 0);
    // ✅ 기존 정책 유지: 공중에서는 y=-1(클래스0) 금지
    if (isAir) {
      const s0 = b0 + c0;
      return (s0 > 1e-12) ? [0, b0 / s0, c0 / s0] : [0, 0.5, 0.5];
    }
    const s0 = a0 + b0 + c0;
    return (s0 > 1e-12) ? [a0 / s0, b0 / s0, c0 / s0] : [1/3, 1/3, 1/3];
  })();

  const logp =
    logProbFromProbs(pxEff, ax) +
    logProbFromProbs(pyEff, ay) +
    logProbFromProbs(ppEff, ap);

  return { logp, value, px, py, pp };
}



  /**
   * Compute GAE advantages and returns from a rollout (in-place).
   * Rollout item must contain: reward, done, value.
   * Adds: adv, ret.
   * @param {Array<any>} rollout
   */
  computeGAE(rollout) {
    const gamma = this.gamma;
    const lam = this.gaeLambda;
    let advNext = 0;
    let vNext = 0;
    for (let i = rollout.length - 1; i >= 0; i--) {
      const it = rollout[i];
      const done = !!it.done;
      const r = Number(it.reward ?? 0);
      const v = Number(it.value ?? 0);
      vNext = done ? 0 : Number(rollout[i + 1]?.value ?? 0);
      const delta = r + gamma * (done ? 0 : vNext) - v;
      const adv = delta + gamma * lam * (done ? 0 : advNext);
      it.adv = adv;
      it.ret = adv + v;
      advNext = adv;
    }

    // normalize advantages
    let mean = 0;
    for (const it of rollout) mean += it.adv;
    mean /= Math.max(1, rollout.length);
    let varSum = 0;
    for (const it of rollout) { const d = it.adv - mean; varSum += d * d; }
    const std = Math.sqrt(varSum / Math.max(1, rollout.length)) || 1;
    for (const it of rollout) it.adv = (it.adv - mean) / (std + 1e-8);
  }

  /**
   * PPO update on a rollout batch.
   * Each item must have: obs, action, oldLogp, adv, ret.
   * @param {Array<any>} batch
   * @param {{epochs?:number, minibatch?:number}} [opts]
   */
  ppoUpdate(batch, opts = {}) {
    const epochs = Math.max(1, (opts.epochs ?? 4) | 0);
    const mb = Math.max(8, (opts.minibatch ?? 256) | 0);

    // Value diagnostics accumulators (reset each ppoUpdate)
    this._vSum=0; this._rSum=0; this._vSum2=0; this._rSum2=0; this._vrSum=0; this._vrCount=0;
    this._advPos=0; this._advNeg=0; this._advZero=0;
    let stats = { steps: batch.length, updates: 0, approxKl: 0, policyLoss: 0, valueLoss: 0, clipFrac: 0, gradNorm: 0, wNorm: 0, advMean: 0, advStd: 0, retMean: 0, retStd: 0, rewMean: 0, rewStd: 0, vrCorr: 0 };

    for (let ep = 0; ep < epochs; ep++) {
      // shuffle indices
      const idxs = new Array(batch.length);
      for (let i = 0; i < idxs.length; i++) idxs[i] = i;
      for (let i = idxs.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const t = idxs[i]; idxs[i] = idxs[j]; idxs[j] = t;
      }

      for (let start = 0; start < idxs.length; start += mb) {
        const end = Math.min(idxs.length, start + mb);
        const slice = idxs.slice(start, end);

        // accumulate grads
        const g = this._zeroGrads();

        let klSum = 0;

        // minibatch metrics
        let pLoss = 0;
        let vLoss = 0;
        let clipCount = 0;
        let advSum = 0, advSum2 = 0;
        let retSum = 0, retSum2 = 0;
        let rewSum = 0, rewSum2 = 0;

        for (const id of slice) {
          const it = batch[id];
          const obs = it.obs;
          const action = it.action;
          const oldLogp = Number(it.oldLogp ?? 0);
          const adv = Number(it.adv ?? 0);
          if (adv > 0) this._advPos++; else if (adv < 0) this._advNeg++; else this._advZero++;
          const ret = Number(it.ret ?? 0);

          const evalNow = this.logpValue(obs, it.playerIndex ?? 1, action);
          const logp = evalNow.logp;
          const v = evalNow.value;
          this._vSum += v; this._vSum2 += v*v;
          this._rSum += ret; this._rSum2 += ret*ret;
          this._vrSum += v * ret;
          this._vrCount++;
          const ratio = Math.exp(clip(logp - oldLogp, -10, 10));

          // Accumulate advantage/return/reward stats
          advSum += adv; advSum2 += adv * adv;
          retSum += ret; retSum2 += ret * ret;
          const rwd = Number(it.reward ?? 0);
          rewSum += rwd; rewSum2 += rwd * rwd;


          // PPO clipped objective: L = -min(ratio*adv, clip(ratio)*adv)
          const clipEps = this.clipEps;
          const rClipped = clip(ratio, 1 - clipEps, 1 + clipEps);

          // determine if gradient flows
          let dL_dlogp = 0;
          const useClipped = (adv >= 0 && ratio > 1 + clipEps) || (adv < 0 && ratio < 1 - clipEps);

          // Policy loss (for logging): -min(ratio*adv, clippedRatio*adv)
          const unclippedObj = ratio * adv;
          const clippedObj = rClipped * adv;
          const chosenObj = (unclippedObj < clippedObj) ? unclippedObj : clippedObj;
          pLoss += -chosenObj;

          // Value loss (for logging): 0.5*(v-ret)^2
          const vErr = (v - ret);
          vLoss += 0.5 * vErr * vErr;

          if (Math.abs(ratio - rClipped) > 1e-12) clipCount++;

          if (!useClipped) {
            // L = -ratio*adv => dL/dlogp = -ratio*adv
            dL_dlogp = -ratio * adv;
          } else {
            dL_dlogp = 0;
          }

          // value loss: 0.5*(v-ret)^2
          const dL_dv = this.vfCoef * (v - ret);

          // accumulate KL for logging (approx)
          klSum += (oldLogp - logp);

          // backprop for policy (via chosen actions) and value
          this._accumulateGrads(g, obs, it.playerIndex ?? 1, action, dL_dlogp, dL_dv);
        }

        // apply grads
        const scale = 1 / Math.max(1, slice.length);
        this._applyGrads(g, scale);

        // finalize minibatch metrics
        const nS = Math.max(1, slice.length);
        const advMean = advSum / nS;
        const retMean = retSum / nS;
        const rewMean = rewSum / nS;
        const advVar = Math.max(0, advSum2 / nS - advMean * advMean);
        const retVar = Math.max(0, retSum2 / nS - retMean * retMean);
        const rewVar = Math.max(0, rewSum2 / nS - rewMean * rewMean);
        const advStd = Math.sqrt(advVar);
        const retStd = Math.sqrt(retVar);
        const rewStd = Math.sqrt(rewVar);

        const gradNorm = this._l2Grads(g) * (1 / Math.max(1e-12, nS)); // scaled similarly to apply step
        const wNorm = this._l2Weights();

        stats.policyLoss += pLoss / nS;
        stats.valueLoss += vLoss / nS;
        stats.clipFrac += clipCount / nS;
        stats.advMean += advMean;
        stats.advStd += advStd;
        stats.retMean += retMean;
        stats.retStd += retStd;
        stats.rewMean += rewMean;
        stats.rewStd += rewStd;
        stats.gradNorm += gradNorm;
        stats.wNorm += wNorm;

        stats.updates++;
        stats.approxKl += klSum / Math.max(1, slice.length);
      }
    }

    stats.approxKl /= Math.max(1, stats.updates);
    // average extra metrics across updates
    const u = Math.max(1, stats.updates);
    stats.policyLoss /= u;
    stats.valueLoss /= u;
    stats.clipFrac /= u;
    stats.gradNorm /= u;
    stats.wNorm /= u;
    stats.advMean /= u;
    stats.advStd /= u;
    stats.retMean /= u;
    stats.retStd /= u;
    stats.rewMean /= u;
    stats.rewStd /= u;

    this.debug.lastUpdate = {
      policyLoss: stats.policyLoss,
      valueLoss: stats.valueLoss,
      clipFrac: stats.clipFrac,
      gradNorm: stats.gradNorm,
      wNorm: stats.wNorm,
      advMean: stats.advMean,
      advStd: stats.advStd,
      retMean: stats.retMean,
      retStd: stats.retStd,
      rewMean: stats.rewMean,
      rewStd: stats.rewStd,
    };

    
    // value-return correlation
    let corr = 0;
    if (this._vrCount > 1) {
      const cov = (this._vrSum - (this._vSum * this._rSum) / this._vrCount) / this._vrCount;
      const vVar = Math.max(1e-12, this._vSum2 / this._vrCount - (this._vSum / this._vrCount) ** 2);
      const rVar = Math.max(1e-12, this._rSum2 / this._vrCount - (this._rSum / this._vrCount) ** 2);
      corr = cov / Math.sqrt(vVar * rVar);
    }
    stats.vrCorr = corr;
    stats.advPos = this._advPos; stats.advNeg = this._advNeg; stats.advZero = this._advZero;

    return stats;
  }

  _l2Weights() {
    let s = 0;
    const accArr = (arr) => { for (let i = 0; i < arr.length; i++) { const v = arr[i]; s += v * v; } };
    for (let i = 0; i < this.hidden1; i++) accArr(this.W1[i]);
    accArr(this.b1);
    for (let i = 0; i < this.hidden2; i++) accArr(this.W2[i]);
    accArr(this.b2);
    for (let i = 0; i < 3; i++) accArr(this.Wx[i]);
    accArr(this.bx);
    for (let i = 0; i < 3; i++) accArr(this.Wy[i]);
    accArr(this.by);
    for (let i = 0; i < 2; i++) accArr(this.Wp[i]);
    accArr(this.bp);
    accArr(this.Wv);
    s += this.bv * this.bv;
    return Math.sqrt(s);
  }

  _l2Grads(g) {
    let s = 0;
    const accArr = (arr) => { for (let i = 0; i < arr.length; i++) { const v = arr[i]; s += v * v; } };
    for (let i = 0; i < g.dW1.length; i++) accArr(g.dW1[i]);
    accArr(g.db1);
    for (let i = 0; i < g.dW2.length; i++) accArr(g.dW2[i]);
    accArr(g.db2);
    for (let i = 0; i < g.dWx.length; i++) accArr(g.dWx[i]);
    accArr(g.dbx);
    for (let i = 0; i < g.dWy.length; i++) accArr(g.dWy[i]);
    accArr(g.dby);
    for (let i = 0; i < g.dWp.length; i++) accArr(g.dWp[i]);
    accArr(g.dbp);
    accArr(g.dWv);
    s += g.dbv * g.dbv;
    return Math.sqrt(s);
  }

  _zeroGrads() {
    const g = {
      dW1: new Array(this.hidden1),
      db1: new Float32Array(this.hidden1),
      dW2: new Array(this.hidden2),
      db2: new Float32Array(this.hidden2),
      dWx: new Array(3),
      dbx: new Float32Array(3),
      dWy: new Array(3),
      dby: new Float32Array(3),
      dWp: new Array(2),
      dbp: new Float32Array(2),
      dWv: new Float32Array(this.hidden2),
      dbv: 0,
    };
    for (let i = 0; i < this.hidden1; i++) g.dW1[i] = new Float32Array(this.featureLen);
    for (let i = 0; i < this.hidden2; i++) g.dW2[i] = new Float32Array(this.hidden1);
    for (let k = 0; k < 3; k++) g.dWx[k] = new Float32Array(this.hidden2);
    for (let k = 0; k < 3; k++) g.dWy[k] = new Float32Array(this.hidden2);
    for (let k = 0; k < 2; k++) g.dWp[k] = new Float32Array(this.hidden2);
    return g;
  }

  _accumulateGrads(g, obs, playerIndex, action, dL_dlogp, dL_dv) {
    // forward with caches
    const feat = this.buildFeatures(obs, playerIndex);
    const fwd = this._forward(feat);
    const { px, py, pp } = this._maskedProbs(fwd.lx, fwd.ly, fwd.lp, obs);

    // action classes
    const a = (typeof action === 'number') ? { xDirection: 0, yDirection: 0, powerHit: 0 } : action;
    const ax = mapXDirToClass(Number(a.xDirection ?? 0));
    const ay = mapYDirToClass(Number(a.yDirection ?? 0));
    const ap = Number(a.powerHit ?? 0) ? 1 : 0;

    // dL/dlogits for each head from dL/dlogp:
    // d logp / d logits = onehot - probs
    const dlogitsX = new Float32Array(3);
    const dlogitsY = new Float32Array(3);
    const dlogitsP = new Float32Array(2);

    if (dL_dlogp !== 0) {
      // d logp / d logits = onehot - probs
      // Therefore: dL/dlogits = dL/dlogp * (onehot - probs)
      for (let i = 0; i < 3; i++) dlogitsX[i] = dL_dlogp * ((i === ax ? 1 : 0) - px[i]);
      for (let i = 0; i < 3; i++) dlogitsY[i] = dL_dlogp * ((i === ay ? 1 : 0) - py[i]);
      for (let i = 0; i < 2; i++) dlogitsP[i] = dL_dlogp * ((i === ap ? 1 : 0) - pp[i]);
    }

    // grads for heads + accumulate dh2
    const dh2 = new Float32Array(this.hidden2);

    for (let k = 0; k < 3; k++) {
      const grad = dlogitsX[k];
      g.dbx[k] += grad;
      const w = this.Wx[k];
      const gw = g.dWx[k];
      for (let j = 0; j < this.hidden2; j++) {
        gw[j] += grad * fwd.h2[j];
        dh2[j] += grad * w[j];
      }
    }
    for (let k = 0; k < 3; k++) {
      const grad = dlogitsY[k];
      g.dby[k] += grad;
      const w = this.Wy[k];
      const gw = g.dWy[k];
      for (let j = 0; j < this.hidden2; j++) {
        gw[j] += grad * fwd.h2[j];
        dh2[j] += grad * w[j];
      }
    }
    for (let k = 0; k < 2; k++) {
      const grad = dlogitsP[k];
      g.dbp[k] += grad;
      const w = this.Wp[k];
      const gw = g.dWp[k];
      for (let j = 0; j < this.hidden2; j++) {
        gw[j] += grad * fwd.h2[j];
        dh2[j] += grad * w[j];
      }
    }

    // critic head grads
    g.dbv += dL_dv;
    for (let j = 0; j < this.hidden2; j++) {
      g.dWv[j] += dL_dv * fwd.h2[j];
      dh2[j] += dL_dv * this.Wv[j];
    }

    // backprop through tanh at layer2
    const dz2 = new Float32Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) {
      const act = fwd.h2[i];
      const der = (this.activation === 'linear') ? 1 : (1 - act * act);
      dz2[i] = dh2[i] * der;
      g.db2[i] += dz2[i];
    }

    // W2 grads and dh1
    const dh1 = new Float32Array(this.hidden1);
    for (let i = 0; i < this.hidden2; i++) {
      const grad = dz2[i];
      const w = this.W2[i];
      const gw = g.dW2[i];
      for (let j = 0; j < this.hidden1; j++) {
        gw[j] += grad * fwd.h1[j];
        dh1[j] += grad * w[j];
      }
    }

    // backprop through tanh at layer1
    const dz1 = new Float32Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) {
      const act = fwd.h1[i];
      const der = (this.activation === 'linear') ? 1 : (1 - act * act);
      dz1[i] = dh1[i] * der;
      g.db1[i] += dz1[i];
    }

    // W1 grads
    for (let i = 0; i < this.hidden1; i++) {
      const grad = dz1[i];
      const gw = g.dW1[i];
      for (let j = 0; j < this.featureLen; j++) {
        gw[j] += grad * feat[j];
      }
    }
  }

  _applyGrads(g, scale) {
    const lr = this.learningRate * scale;

    for (let i = 0; i < this.hidden1; i++) {
      const w = this.W1[i];
      const dw = g.dW1[i];
      for (let j = 0; j < this.featureLen; j++) w[j] -= lr * dw[j];
      this.b1[i] -= lr * g.db1[i];
    }

    for (let i = 0; i < this.hidden2; i++) {
      const w = this.W2[i];
      const dw = g.dW2[i];
      for (let j = 0; j < this.hidden1; j++) w[j] -= lr * dw[j];
      this.b2[i] -= lr * g.db2[i];
    }

    for (let k = 0; k < 3; k++) {
      const w = this.Wx[k];
      const dw = g.dWx[k];
      for (let j = 0; j < this.hidden2; j++) w[j] -= lr * dw[j];
      this.bx[k] -= lr * g.dbx[k];
    }
    for (let k = 0; k < 3; k++) {
      const w = this.Wy[k];
      const dw = g.dWy[k];
      for (let j = 0; j < this.hidden2; j++) w[j] -= lr * dw[j];
      this.by[k] -= lr * g.dby[k];
    }
    for (let k = 0; k < 2; k++) {
      const w = this.Wp[k];
      const dw = g.dWp[k];
      for (let j = 0; j < this.hidden2; j++) w[j] -= lr * dw[j];
      this.bp[k] -= lr * g.dbp[k];
    }

    for (let j = 0; j < this.hidden2; j++) this.Wv[j] -= lr * g.dWv[j];
    this.bv -= lr * g.dbv;
  }


  /**
   * Supervised warmup (behavior cloning) update from an imitation sample.
   * Trainer calls this during warmup-training stage.
   * We only train the actor heads to maximize log-prob of demonstrated action.
   */
  updateImitationSample(obs, playerIndex, label, weight = 1) {
    const w = Number(weight ?? 1);
    if (!isFinite(w) || w === 0) return { loss: 0 };

    // Forward (with masking based on obs) to compute NLL
    const ev = this.evaluate(obs, playerIndex);
    const a = label ?? {};
    const ax = mapXDirToClass(Number(a.xDirection ?? 0));
    const ay = mapYDirToClass(Number(a.yDirection ?? 0));
    const ap = Number(a.powerHit ?? 0) ? 1 : 0;

    const nll =
      -logProbFromProbs(ev.px, ax) +
      -logProbFromProbs(ev.py, ay) +
      -logProbFromProbs(ev.pp, ap);

    // Predicted classes (for logging/diagnostics)
    const argmax = (arr) => {
      let mi = 0;
      let mv = arr?.[0] ?? -Infinity;
      for (let i = 1; i < (arr?.length ?? 0); i++) {
        const v = arr[i];
        if (v > mv) { mv = v; mi = i; }
      }
      return mi;
    };
    const predAx = argmax(ev.px);
    const predAy = argmax(ev.py);
    const predAp = argmax(ev.pp);

    // Backprop through the same accumulator used by PPO.
    // For CE loss L = -log p(a):
    //   dL/dlogp = -1
    // and since d logp / d logits = (onehot - probs):
    //   dL/dlogits = -1 * (onehot - probs) = (probs - onehot)
    // Therefore we pass -w so SGD increases probability of the demonstrated action.
    const g = this._zeroGrads();
    this._accumulateGrads(g, obs, playerIndex, label, -w, 0);

    // Apply grads using an imitation learning-rate.
    // _applyGrads expects a *scale* relative to this.learningRate.
    const imitationLr = Number(this.imitationLr ?? this.learningRate);
    const base = Number(this.learningRate) || 1;
    const scale = imitationLr / base;
    this._applyGrads(g, scale);

    return { loss: nll, ax: predAx, ay: predAy, ap: predAp };
  }

  saveState() {
    const toArr2D = (A) => A.map((row) => Array.from(row));
    return {
      kind: this.kind,
      featureLen: this.featureLen,
      hidden1: this.hidden1,
      hidden2: this.hidden2,
      activation: this.activation,
      learningRate: this.learningRate,
      initStd: this.initStd,
      clipEps: this.clipEps,
      vfCoef: this.vfCoef,
      entCoef: this.entCoef,
      gamma: this.gamma,
      gaeLambda: this.gaeLambda,
      W1: toArr2D(this.W1),
      b1: Array.from(this.b1),
      W2: toArr2D(this.W2),
      b2: Array.from(this.b2),
      Wx: toArr2D(this.Wx),
      bx: Array.from(this.bx),
      Wy: toArr2D(this.Wy),
      by: Array.from(this.by),
      Wp: toArr2D(this.Wp),
      bp: Array.from(this.bp),
      Wv: Array.from(this.Wv),
      bv: this.bv,
    };
  }

  loadState(state) {
    if (!state || state.kind !== this.kind) return false;
    if ((state.featureLen | 0) !== (this.featureLen | 0)) return false;
    if ((state.hidden1 | 0) !== (this.hidden1 | 0)) return false;
    if ((state.hidden2 | 0) !== (this.hidden2 | 0)) return false;

    const fromArr2D = (A, rows, cols) => {
      const out = new Array(rows);
      for (let i = 0; i < rows; i++) {
        const row = new Float32Array(cols);
        const src = A?.[i] ?? [];
        for (let j = 0; j < cols; j++) row[j] = Number(src[j] ?? 0);
        out[i] = row;
      }
      return out;
    };

    this.activation = (state.activation === 'linear') ? 'linear' : 'tanh';
    this.learningRate = Number(state.learningRate ?? this.learningRate);
    this.clipEps = Number(state.clipEps ?? this.clipEps);
    this.vfCoef = Number(state.vfCoef ?? this.vfCoef);
    this.entCoef = Number(state.entCoef ?? this.entCoef);
    this.gamma = Number(state.gamma ?? this.gamma);
    this.gaeLambda = Number(state.gaeLambda ?? this.gaeLambda);

    this.W1 = fromArr2D(state.W1, this.hidden1, this.featureLen);
    this.b1 = new Float32Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) this.b1[i] = Number(state.b1?.[i] ?? 0);

    this.W2 = fromArr2D(state.W2, this.hidden2, this.hidden1);
    this.b2 = new Float32Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) this.b2[i] = Number(state.b2?.[i] ?? 0);

    this.Wx = fromArr2D(state.Wx, 3, this.hidden2);
    this.bx = new Float32Array(3);
    for (let i = 0; i < 3; i++) this.bx[i] = Number(state.bx?.[i] ?? 0);

    this.Wy = fromArr2D(state.Wy, 3, this.hidden2);
    this.by = new Float32Array(3);
    for (let i = 0; i < 3; i++) this.by[i] = Number(state.by?.[i] ?? 0);

    this.Wp = fromArr2D(state.Wp, 2, this.hidden2);
    this.bp = new Float32Array(2);
    for (let i = 0; i < 2; i++) this.bp[i] = Number(state.bp?.[i] ?? 0);

    this.Wv = new Float32Array(this.hidden2);
    for (let j = 0; j < this.hidden2; j++) this.Wv[j] = Number(state.Wv?.[j] ?? 0);
    this.bv = Number(state.bv ?? 0);

    return true;
  }
}
