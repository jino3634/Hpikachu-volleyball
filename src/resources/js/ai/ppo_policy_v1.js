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

function softmaxInto(logits, out) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];

  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    out[i] = v;
    sum += v;
  }

  if (sum <= 0) {
    const p = 1 / logits.length;
    for (let i = 0; i < logits.length; i++) out[i] = p;
    return out;
  }

  const inv = 1 / sum;
  for (let i = 0; i < logits.length; i++) out[i] *= inv;
  return out;
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

// ------------------------------------------------------------
// alloc-free renorm helpers (write into reusable buffers)
// ------------------------------------------------------------
function renorm2Into(out, a, b) {
  const s = a + b;
  if (s <= 1e-12) {
    out[0] = 0.5; out[1] = 0.5;
    return out;
  }
  const inv = 1 / s;
  out[0] = a * inv;
  out[1] = b * inv;
  return out;
}

function renorm3Into(out, a, b, c) {
  const s = a + b + c;
  if (s <= 1e-12) {
    out[0] = 1/3; out[1] = 1/3; out[2] = 1/3;
    return out;
  }
  const inv = 1 / s;
  out[0] = a * inv;
  out[1] = b * inv;
  out[2] = c * inv;
  return out;
}

// top1-top2 margin without alloc/sort (3-way)
function top2Margin3(a, b, c) {
  let m1 = a, m2 = b;
  if (m2 > m1) { const t = m1; m1 = m2; m2 = t; }
  if (c > m1) { m2 = m1; m1 = c; }
  else if (c > m2) { m2 = c; }
  return m1 - m2;
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

function powerHitGate(obs, playerIndex, gateParams) {
  // ✅ 기본값(유전자 없을 때)
  const g = gateParams ?? {
    // P0-2: 충돌 예측 gate 파라미터
    kFrames: 8,          // 몇 프레임 ahead로 충돌 가능성 볼지
    dxMarginPx: 12,       // 충돌 박스 여유(px)
    dyMarginPx: 16,      // 충돌 박스 여유(px)
  };

  const toFinite = (v, fb = 0) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fb;
  };

  const me = obs?.me ?? {};
  const ball = obs?.ball ?? {};

  const state = toFinite(me.state, 0) | 0;
  const isLying = !!me.isLying || state === 4;
  const isDiving = !!me.isDiving || state === 3;

  const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
  const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

  // ✅ diagnostics용(정규화 공간): 기존 통계가 NaN 안 나게 유지
  const mxN = toFinite(me.x, 0);
  const myN = toFinite(me.y, 0);
  const bxN = toFinite(ball.x, 0);
  const byN = toFinite(ball.y, 0);
  const dx = Math.abs(bxN - mxN);
  const dy = Math.abs(byN - myN);
  const tLand = toFinite(ball.timeToLand, 1);

  // ✅ raw 픽셀 기반 충돌 예측
  const raw = obs?.raw ?? null;
  const mePx = raw?.me ?? null;
  const bPx = raw?.ball ?? null;

  const meX = toFinite(mePx?.x, 0);
  const meY = toFinite(mePx?.y, 0);
  const bX0 = toFinite(bPx?.x, 0);
  const bY0 = toFinite(bPx?.y, 0);
  const bVX = toFinite(bPx?.xV, 0);
  const bVY = toFinite(bPx?.yV, 0);

  const k = Math.max(0, Math.min(12, toFinite(g.kFrames, 4) | 0));
  const mx = Math.max(0, Math.min(32, toFinite(g.dxMarginPx, 6)));
  const my = Math.max(0, Math.min(40, toFinite(g.dyMarginPx, 10)));

  // physics.js: PLAYER_HALF_LENGTH = 32
  const PH = 32;
  const thrX = PH + mx;
  const thrY = PH + my;

  let airOK = false;
  if (isAir && canAct && !isLying && !isDiving) {
    for (let i = 0; i <= k; i++) {
      const bx = bX0 + bVX * i;
      const by = bY0 + bVY * i;
      const dxPx = Math.abs(bx - meX);
      const dyPx = Math.abs(by - meY);
      if (dxPx <= thrX && dyPx <= thrY) {
        airOK = true;
        break;
      }
    }
  }

  // ✅ 지상 파워는 "서있는 상태(대개 state=0)"에서만 아주 제한적으로 허용
  let groundOK = false;
  if (!isAir && canAct && !isLying && !isDiving && state === 0) {
    // 지상에서는 너무 멀리 예측하면 노이즈니까, k를 줄여서(예: 최대 3프레임)만 본다
    const kg = Math.min(3, k);
    for (let i = 0; i <= kg; i++) {
      const bx = bX0 + bVX * i;
      const by = bY0 + bVY * i;
      const dxPx = Math.abs(bx - meX);
      const dyPx = Math.abs(by - meY);
      if (dxPx <= thrX && dyPx <= thrY) {
        groundOK = true;
        break;
      }
    }
  }

  // ✅ allow는 air 또는 ground
  const allow = (airOK || groundOK);


  // dLand/ballOnMySide는 기존 구조 유지용 더미(통계 NaN 방지)
  const dLand = 0;
  const ballOnMySide = true;

  return { allow, airOK, groundOK, canAct, isLying, isDiving, isAir, dx, dy, tLand, dLand, ballOnMySide };
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
   *   genome?: { powerHitGate?: any }, // ✅ 추가
   * }} opts
   */
  constructor(opts = {}) {
    this.kind = 'ppo_policy_v1';
    // buildFeatures가 0..23까지 쓰므로 최소 24는 보장해야 함
    this.featureLen = Math.max(24, (opts.featureLen ?? 24) | 0);

    // ✅ 주입 지점에서 들어온 genome 저장 (없으면 기본값)
    // powerHitGate는 "픽셀 충돌 예측 gate"와 동일 스키마로 통일
    this.genome = opts.genome ?? {
      powerHitGate: {
        kFrames: 4,
        dxMarginPx: 6,
        dyMarginPx: 10,
      },
    };


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
        // 모집단(= act 호출 단위) 기반
        frames: 0,               // act 호출 수 (canAct true인 프레임)
        eligibleFrames: 0,       // gate.allow === true 인 프레임 수
        blockedFrames: 0,        // gate.allow === false 인 프레임 수 (=frames-eligibleFrames)

        // 정책의 power 선호도(샘플이 아니라 확률 평균)
        sumPPower: 0,            // Σ ppRaw[1]
        sumPPowerEligible: 0,    // Σ ppRaw[1] where eligible

        // “기댓값 기준” 실제 적용량
        expectedApplied: 0,      // Σ (eligible ? ppRaw[1] : 0)

        // “샘플 기준” 실제 정책 샘플(ap==1)
        sampledApplied: 0,       // Σ (ap==1)

        // (선택) blocked 원인(현재 gate는 사실상 no-contact-window 하나뿐)
        blockedNoContactWindow: 0,
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

      tieX: 0, tieY: 0, tieP: 0,

      // ✅ near-tie / margin (deterministic 분석용)
      nearTieX: 0,
      nearTieY: 0,
      nearTieP: 0,
      marginXSum: 0,
      marginYSum: 0,
      marginPSum: 0,

      // ✅ power gate 관찰용
      gateAllowN: 0,     // allowPowerHit=true 프레임 수
      gateBlockN: 0,     // allowPowerHit=false 프레임 수
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

    // ------------------------------------------------------------
    // Inference scratch buffers (avoid per-frame allocations)
    // ------------------------------------------------------------
    this._infer = {
      feat: new Float32Array(this.featureLen),

      // forward caches
      fwd: {
        feat: null,
        h1: new Float32Array(this.hidden1),
        z1: new Float32Array(this.hidden1),
        h2: new Float32Array(this.hidden2),
        z2: new Float32Array(this.hidden2),
        lx: new Float32Array(3),
        ly: new Float32Array(3),
        lp: new Float32Array(2),
        v: 0,
      },

      // masking temp logits + probs (reused)
      tmp: {
        mx: new Float32Array(3),
        my: new Float32Array(3),
        mp: new Float32Array(2),
        px: new Float32Array(3),
        py: new Float32Array(3),
        pp: new Float32Array(2),
      },
    };

    // ------------------------------------------------------------
    // Training scratch buffers (avoid per-sample allocations)
    // ------------------------------------------------------------
    this._train = {
      dlogitsX: new Float32Array(3),
      dlogitsY: new Float32Array(3),
      dlogitsP: new Float32Array(2),

      dh2: new Float32Array(this.hidden2),
      dz2: new Float32Array(this.hidden2),
      dh1: new Float32Array(this.hidden1),
      dz1: new Float32Array(this.hidden1),

      // aux uses X only (can reuse dlogitsX but keeping separate is clearer)
      auxDlogitsX: new Float32Array(3),
    };

    // ------------------------------------------------------------
    // Act/logpValue scratch buffers (avoid per-decision allocations)
    // NOTE: act() returns immediately, but logpValue() is called inside
    // PPO loops; keep separate buffers to avoid accidental aliasing.
    // ------------------------------------------------------------
    this._act = {
      ppRaw: new Float32Array(2),
      ppEff: new Float32Array(2),
      pxEff: new Float32Array(3),
      pyEff: new Float32Array(3),
    };
    this._lp = {
      ppRaw: new Float32Array(2),
      ppEff: new Float32Array(2),
      pxEff: new Float32Array(3),
      pyEff: new Float32Array(3),
    };

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

    // ✅ Warmup(BC) 때는 하드룰/마스크를 끌 수 있게 스위치 추가
    this.hardRulesEnabled = true;

    this._initWeights();
    
  }

    setHardRulesEnabled(v) {
      this.hardRulesEnabled = !!v;
    }

    /**
     * ✅ PBT/진화용: 런타임 genome 적용(가중치 변경 없음)
     * - Trainer는 변이 후 이 함수만 호출한다.
     * - policy.saveState()/loadState()는 가중치 복구용, genome은 별도 저장/복구.
     * @param {{
     *   learningRate?: number,
     *   clipEps?: number,
     *   vfCoef?: number,
     *   powerHitGate?: { kFrames?: number, dxMarginPx?: number, dyMarginPx?: number }
     * }|any} genome
     */
    applyGenome(genome) {
      if (!genome || typeof genome !== 'object') return;

      // hyper-params
      if (genome.learningRate !== undefined) this.learningRate = Number(genome.learningRate);
      if (genome.clipEps !== undefined) this.clipEps = Number(genome.clipEps);
      if (genome.vfCoef !== undefined) this.vfCoef = Number(genome.vfCoef);

      // gate
      const cur = (this.genome && typeof this.genome === 'object') ? this.genome : {};
      const curGate = (cur.powerHitGate && typeof cur.powerHitGate === 'object') ? cur.powerHitGate : {};
      const g = (genome.powerHitGate && typeof genome.powerHitGate === 'object') ? genome.powerHitGate : null;

      const nextGate = {
        kFrames: (g?.kFrames !== undefined) ? (g.kFrames | 0) : (curGate.kFrames | 0),
        dxMarginPx: (g?.dxMarginPx !== undefined) ? (g.dxMarginPx | 0) : (curGate.dxMarginPx | 0),
        dyMarginPx: (g?.dyMarginPx !== undefined) ? (g.dyMarginPx | 0) : (curGate.dyMarginPx | 0),
      };

      this.genome = {
        ...cur,
        powerHitGate: nextGate,
      };

      console.log(
        `[POLICY-GENOME] applied ` +
        `lr=${this.learningRate} clip=${this.clipEps} vf=${this.vfCoef} ` +
        `gate=${JSON.stringify(this.genome?.powerHitGate ?? {})}`
      );
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
  buildFeatures(obs, playerIndex, out = null) {
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
    // ✅ out이 있으면 재사용, 없으면 기존처럼 새로 생성(학습 경로 호환)
    const f = (out && out.length === this.featureLen) ? out : new Float32Array(this.featureLen);


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

    // ------------------------------------------------------------
    // Feature scaling helpers
    // - If obs is already normalized (-1..1), DO NOT normalize again.
    // - If obs is raw pixels, normalize here.
    // ------------------------------------------------------------
    const GROUND_W = 432;
    const GROUND_H = 304;

    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const nxRaw = (x) => (Number(x) / GROUND_W) * 2 - 1; // 0..432 -> -1..1
    const nyRaw = (y) => (Number(y) / GROUND_H) * 2 - 1; // 0..304 -> -1..1
    const nvRaw = (v, scale) => clamp(Number(v) / scale, -1, 1);

    // Heuristic: normalized obs has positions roughly in [-1..1].
    const meX = Number(me.x ?? 0);
    const oppX = Number(opp.x ?? 0);
    const ballX = Number(ball.x ?? 0);
    const landX = Number(ball.landingX ?? ball.expectedX ?? 0);
    const isNormalizedObs =
      Math.abs(meX) <= 1.5 &&
      Math.abs(oppX) <= 1.5 &&
      Math.abs(ballX) <= 1.5 &&
      Math.abs(landX) <= 1.5;

    // Unified accessors
    const posX = (x) => isNormalizedObs ? clamp(Number(x ?? 0), -1, 1) : nxRaw(x ?? 0);
    const posY = (y) => isNormalizedObs ? clamp(Number(y ?? 0), -1, 1) : nyRaw(y ?? 0);
    const velN = (v, scale) => isNormalizedObs ? clamp(Number(v ?? 0), -1, 1) : nvRaw(v ?? 0, scale);

    // Fill (guarded)
    // positions: [-1, 1]
    f[0] = posX(me.x ?? 0);
    f[1] = posY(me.y ?? 0);

    // velocities: already normalized in getObservationNormalized()
    f[2] = velN(me.yV ?? me.yv ?? 0, 20);

    f[8]  = posX(opp.x ?? 0);
    f[9]  = posY(opp.y ?? 0);
    f[10] = velN(opp.yV ?? opp.yv ?? 0, 20);

    f[14] = posX(ball.x ?? 0);
    f[15] = posY(ball.y ?? 0);
    f[16] = velN(ball.xV ?? ball.xv ?? 0, 25);
    f[17] = velN(ball.yV ?? ball.yv ?? 0, 35);

    // landingX: [-1, 1]
    f[18] = posX(ball.landingX ?? ball.expectedX ?? 0);

    // timeToLand: convert 0..1 -> -1..1 (and clamp)
    const ttl = clamp(Number(ball.timeToLand ?? 0), 0, 1);
    f[19] = ttl * 2 - 1;

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

  _forward(feat, out = null) {
    const h1 = out ? out.h1 : new Float32Array(this.hidden1);
    const z1 = out ? out.z1 : new Float32Array(this.hidden1);
    const h2 = out ? out.h2 : new Float32Array(this.hidden2);
    const z2 = out ? out.z2 : new Float32Array(this.hidden2);
    const lx = out ? out.lx : new Float32Array(3);
    const ly = out ? out.ly : new Float32Array(3);
    const lp = out ? out.lp : new Float32Array(2);

    // trunk: feat -> h1 -> h2
    for (let i = 0; i < this.hidden1; i++) {
      let sum = this.b1[i];
      const w = this.W1[i];
      for (let j = 0; j < this.featureLen; j++) sum += w[j] * feat[j];
      z1[i] = sum;
      h1[i] = (this.activation === 'linear') ? sum : tanh(sum);
    }

    for (let i = 0; i < this.hidden2; i++) {
      let sum = this.b2[i];
      const w = this.W2[i];
      for (let j = 0; j < this.hidden1; j++) sum += w[j] * h1[j];
      z2[i] = sum;
      h2[i] = (this.activation === 'linear') ? sum : tanh(sum);
    }

    // logits
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

    if (out) {
      out.feat = feat;
      out.v = v;
      return out;
    }
    return { feat, h1, z1, h2, z2, lx, ly, lp, v };
  }

  _rawProbs(logitsX, logitsY, logitsP) {
    const t = this._infer.tmp;
    const px = softmaxInto(logitsX, t.px);
    const py = softmaxInto(logitsY, t.py);
    const pp = softmaxInto(logitsP, t.pp);
    return { px, py, pp };
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

  const t = this._infer.tmp;

  // If cannot act at all, force IDLE effectively.
  if (!canAct) {
    t.px[0]=0; t.px[1]=1; t.px[2]=0;
    t.py[0]=0; t.py[1]=1; t.py[2]=0;
    t.pp[0]=1; t.pp[1]=0;
    return { px: t.px, py: t.py, pp: t.pp };
  }

  // copy logits into tmp (because we will mutate for masks)
  for (let i=0;i<3;i++) t.mx[i] = logitsX[i];
  for (let i=0;i<3;i++) t.my[i] = logitsY[i];
  for (let i=0;i<2;i++) t.mp[i] = logitsP[i];

  // Base constraint: on ground, forbid DOWN (+1).
  if (!isAir) {
    t.my[2] = -1e9;
  }

  // lying/diving: forbid powerHit=1
  if (isLying || isDiving) {
    t.mp[1] = -1e9;
  }

  const px0 = softmaxInto(t.mx, t.px);
  const py0 = softmaxInto(t.my, t.py);
  const pp0 = softmaxInto(t.mp, t.pp);
  return { px: px0, py: py0, pp: pp0 };
}


  /**
   * Evaluate: compute action distribution + value for obs.
   * @param {any} obs
   * @param {1|2} playerIndex
   */
  evaluate(obs, playerIndex) {
    const feat = this.buildFeatures(obs, playerIndex, this._infer.feat);
    const fwd = this._forward(feat, this._infer.fwd);

    const probs = this.hardRulesEnabled
      ? this._maskedProbs(fwd.lx, fwd.ly, fwd.lp, obs)
      : this._rawProbs(fwd.lx, fwd.ly, fwd.lp);

    return { px: probs.px, py: probs.py, pp: probs.pp, value: fwd.v, feat, fwd };
  }

  // ------------------------------------------------------------
  // alloc-free stats recording helper (called from act())
  // ------------------------------------------------------------
  _recordActionStats(pxEff, pyEff, ppEff, ax, ay, ap) {
    const as = this.debug.actionStats;
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

  // deterministic step counter (결정론이지만 step별 변화를 주기 위함)
  this._detStep = (this._detStep ?? 0) + 1;

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


  // ✅ Power-hit gate 제거(소프트 억제/근접/TTL gate 전부 제거)
  // ✅ 하드 금지 조건은 오직 lying/diving
  const gate = powerHitGate(obs, playerIndex, this.genome?.powerHitGate);
  const allowPowerHit = gate.allow;


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

    // 1) logp 계산용 eff 분포 구성(메인 경로와 일치) - alloc-free
    const ppEff = this._act.ppEff;
    const pxEff = this._act.pxEff;
    const pyEff = this._act.pyEff;

    // ppEff
    if (!allowPowerHit) {
      ppEff[0] = 1; ppEff[1] = 0;
    } else {
      const a0 = clamp01(pp[0] ?? 0);
      const b0 = clamp01(pp[1] ?? 0);
      renorm2Into(ppEff, a0, b0);
    }

    // pxEff
    {
      const a0 = clamp01(px[0] ?? 0), b0 = clamp01(px[1] ?? 0), c0 = clamp01(px[2] ?? 0);
      // ✅ 지상 powerHit=1 & x=0 하드 금지(이미 너가 채택한 룰)
      if (!isAir && ap === 1) renorm3Into(pxEff, a0, 0, c0);
      else renorm3Into(pxEff, a0, b0, c0);
    }

    // pyEff
    {
      const a0 = clamp01(py[0] ?? 0), b0 = clamp01(py[1] ?? 0), c0 = clamp01(py[2] ?? 0);
      if (isAir) {
        const s = b0 + c0;
        pyEff[0] = 0;
        if (s > 1e-12) {
          const inv = 1 / s;
          pyEff[1] = b0 * inv;
          pyEff[2] = c0 * inv;
        } else {
          pyEff[1] = 0.5;
          pyEff[2] = 0.5;
        }
      } else {
        renorm3Into(pyEff, a0, b0, c0);
      }
    }

    // 2) 이제 안전하게 기록/로그확률 계산 가능
    this._recordActionStats(pxEff, pyEff, ppEff, ax, ay, ap);
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

  // 0) 마스킹 전 "원본" pp 정규화 (계측용) - alloc-free
  const ppRaw = this._act.ppRaw;
  {
    const a = clamp01(pp[0] ?? 0);
    const b = clamp01(pp[1] ?? 0);
    renorm2Into(ppRaw, a, b);
  }

  // ✅ allowPowerHit일 때도 power=0으로 붕괴하니까, 행동분포를 혼합으로 만든다.
  // mix=0.15면: 85%는 모델, 15%는 50:50 탐색(=power도 가끔 눌러봄)
  const POWER_MIX = 0.15;

  const ppEff = this._act.ppEff;
  if (!allowPowerHit) {
    ppEff[0] = 1; ppEff[1] = 0;
  } else {
    const p0 = ppRaw[0];
    const p1 = ppRaw[1];
    const m = POWER_MIX;
    const q0 = p0 * (1 - m) + 0.5 * m;
    const q1 = p1 * (1 - m) + 0.5 * m;
    renorm2Into(ppEff, q0, q1);
  }


  // --- TIE-DIAG (POWER) INSERT HERE ---
  const as2 = this.debug?.actionStats;

  // ✅ EPS=1e-6은 너무 빡빡해서 사실상 0만 나옴
  const EPS_TIE = 1e-6;   // "완전 동점"용(그대로 둬도 됨)
  const EPS_NEAR = 1e-3;  // ✅ "거의 동점"용(이게 핵심)

  if (as2) {
    // gate 관찰
    if (allowPowerHit) as2.gateAllowN = (as2.gateAllowN | 0) + 1;
    else as2.gateBlockN = (as2.gateBlockN | 0) + 1;

    if (deterministic) {
      const marginP = Math.abs(ppEff[1] - ppEff[0]);
      as2.marginPSum += marginP;

      if (marginP < EPS_TIE) as2.tieP = (as2.tieP | 0) + 1;
      if (marginP < EPS_NEAR) as2.nearTieP = (as2.nearTieP | 0) + 1;
    }
  }
  // --- END ---
  // ✅ deterministic에서 near-tie면 argmax 고정 대신 "고정 seed" 샘플링(결정론 유지)
  const TIE_EPS_P = 1e-3; // power near-tie
  const TIE_EPS_X = 1e-3; // x near-tie
  const TIE_EPS_Y = 1e-3; // y near-tie

  function _mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function _hashObs32(obs, playerIndex) {
    const me2 = obs?.me ?? {};
    const ball2 = obs?.ball ?? {};
    const q = (v, s) => (Math.floor((Number(v) || 0) * s) | 0);

    let h = 2166136261 | 0;
    const mix = (x) => { h ^= (x | 0); h = Math.imul(h, 16777619); };

    mix(playerIndex | 0);
    mix(q(me2.x, 1000));
    mix(q(me2.y, 1000));
    mix(q(ball2.x, 1000));
    mix(q(ball2.y, 1000));
    mix(q(ball2.timeToLand, 1000));
    mix(q(ball2.landingX, 1000));
    return h >>> 0;
  }

  const rng = _mulberry32(
    (_hashObs32(obs, playerIndex) ^ (this._detStep | 0)) >>> 0
  );

  function _sampleCategoricalRng(probs) {
    let sum = 0;
    for (let i = 0; i < probs.length; i++) sum += Math.max(0, Number(probs[i] ?? 0));
    if (!(sum > 0)) return 0;

    let r = rng() * sum;
    for (let i = 0; i < probs.length; i++) {
      r -= Math.max(0, Number(probs[i] ?? 0));
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  function _top2Margin(arr) {
    let a = -Infinity, b = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      const v = Number(arr[i] ?? -Infinity);
      if (v > a) { b = a; a = v; }
      else if (v > b) { b = v; }
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
    return a - b;
  }

  function _argmax(arr) {
    let mx = -Infinity, mi = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = Number(arr[i] ?? -Infinity);
      if (v > mx) { mx = v; mi = i; }
    }
    return mi;
  }

  function _chooseDeterministic(probs, tieEps) {
    const m = _top2Margin(probs);
    if (m < tieEps) return _sampleCategoricalRng(probs); // ✅ near-tie면 seed 샘플
    return _argmax(probs);                               // ✅ 아니면 argmax
  }

  // 1) (계측용) apRaw: "원래 정책이 원했을" powerHit 샘플
  let apRaw = 0;
  if (deterministic) {
    apRaw = (ppRaw[1] > ppRaw[0]) ? 1 : 0;
  } else {
    apRaw = sampleCategorical(ppRaw);
  }

  // 2) (실제 행동용) ap: gate 반영된 분포에서 샘플
  if (deterministic) {
    // ✅ deterministic에서도 power는 argmax 고정 대신 "seed 고정 샘플링" 사용
    // allowPowerHit=false면 ppEff=[1,0]이라 어차피 0만 나옴
    ap = _sampleCategoricalRng(ppEff);
  } else {
    ap = sampleCategorical(ppEff);
  }


  // 2) conditional X distribution: if ground && ap==1 => forbid x=0 (class 1) - alloc-free
  const pxEff = this._act.pxEff;
  {
    const a = clamp01(px[0] ?? 0);
    const b = clamp01(px[1] ?? 0);
    const c = clamp01(px[2] ?? 0);
    if (!isAir && ap === 1) {
      if (this.debug.maskStats) { this.debug.maskStats.xZeroMaskedCount++; this.debug.maskStats.xZeroMaskedMass += (px[1] ?? 0); }
      renorm3Into(pxEff, a, 0, c);
    } else {
      renorm3Into(pxEff, a, b, c);
    }
  }

  // --- TIE-DIAG (X) INSERT HERE ---
  if (as2 && deterministic) {
    // top1-top2 margin
    const marginX = top2Margin3(pxEff[0], pxEff[1], pxEff[2]);
    as2.marginXSum += marginX;

    // 완전 동점(거의 안 나옴)
    if (marginX < 1e-6) as2.tieX = (as2.tieX | 0) + 1;

    // ✅ near-tie (이게 의미 있음)
    if (marginX < 1e-3) as2.nearTieX = (as2.nearTieX | 0) + 1;
  }
  // --- END ---

  if (deterministic) {
    ax = _chooseDeterministic(pxEff, TIE_EPS_X);
    if (!isAir && ap === 1 && ax === 1 && this.debug.maskStats) this.debug.maskStats.illegalXSampledPrevented++;
  } else {
    ax = sampleCategorical(pxEff);
    if (!isAir && ap === 1 && ax === 1 && this.debug.maskStats) this.debug.maskStats.illegalXSampledPrevented++;
  }

  // 3) Y distribution (keep current behavior: if air forbid y=-1 to avoid double-jump) - alloc-free
  const pyEff = this._act.pyEff;
  {
    const a = clamp01(py[0] ?? 0);
    const b = clamp01(py[1] ?? 0);
    const c = clamp01(py[2] ?? 0);
    if (isAir) {
      if (this.debug.maskStats) { this.debug.maskStats.yNegMaskedCount++; this.debug.maskStats.yNegMaskedMass += (py[0] ?? 0); }
      const s = b + c;
      pyEff[0] = 0;
      if (s > 1e-12) {
        const inv = 1 / s;
        pyEff[1] = b * inv;
        pyEff[2] = c * inv;
      } else {
        pyEff[1] = 0.5;
        pyEff[2] = 0.5;
      }
    } else {
      renorm3Into(pyEff, a, b, c);
    }
  }

  // --- TIE-DIAG (Y) INSERT HERE ---
  if (as2 && deterministic) {
    const marginY = top2Margin3(pyEff[0], pyEff[1], pyEff[2]);
    as2.marginYSum += marginY;

    if (marginY < 1e-6) as2.tieY = (as2.tieY | 0) + 1;
    if (marginY < 1e-3) as2.nearTieY = (as2.nearTieY | 0) + 1;
  }
  // --- END ---

  if (deterministic) {
    ay = _chooseDeterministic(pyEff, TIE_EPS_Y);
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
  // (선택) powerHit gate 디버그 카운트 개선
  function makeEmptyPowerGate() {
    return {
      frames: 0,
      eligibleFrames: 0,
      blockedFrames: 0,
      sumPPower: 0,
      sumPPowerEligible: 0,
      expectedApplied: 0,
      sampledApplied: 0,
      blockedNoContactWindow: 0,
    };
  }

  // --- P0-2 power gate diagnostics: per-frame, expectation-based ---
  if (!this.debug.powerGate) this.debug.powerGate = makeEmptyPowerGate();
  const pg = this.debug.powerGate;

  // 모집단: act 호출 프레임
  pg.frames += 1;

  if (allowPowerHit) {
    pg.eligibleFrames += 1;
  } else {
    pg.blockedFrames += 1;
    pg.blockedNoContactWindow += 1; // 현재 gate는 사실상 이 이유 하나
  }

  // 정책의 power 선호도(샘플이 아니라 확률)
  const pPower = Number(ppRaw[1] ?? 0);
  pg.sumPPower += pPower;
  if (allowPowerHit) pg.sumPPowerEligible += pPower;

  // 기댓값 기준 실제 적용량
  pg.expectedApplied += allowPowerHit ? pPower : 0;

  // 샘플 기준 실제 행동(ap==1)
  if (ap === 1) pg.sampledApplied += 1;


  // ✅ "실제로 실행된" powerHit=1 샘플은 기존처럼 따로 유지
  if (ap === 1) {
    this.debug.powerHitSampled = (this.debug.powerHitSampled ?? 0) + 1;

    const as = this.debug.actionStats;
    as.powerHitAllowed = (as.powerHitAllowed ?? 0) + (allowPowerHit ? 1 : 0);
    as.powerHitAir = (as.powerHitAir ?? 0) + (gate.airOK ? 1 : 0);
    as.powerHitGround = (as.powerHitGround ?? 0) + (gate.groundOK ? 1 : 0);
  }

  this._recordActionStats(pxEff, pyEff, ppEff, ax, ay, ap);
  return { action, logp, value, meta: { ax, ay, ap, forcedIdle: skipLearn, reason: (skipLearn ? (isLying ? 'lying' : 'diving') : null) } };
}


  /**
   * Compute log-prob and value under current params for a given (obs, action).
   * @param {any|Float32Array} obsOrFeat
   * @param {1|2} playerIndex
   * @param {{xDirection:number,yDirection:number,powerHit:number}|number} action
   */
logpValue(obsOrFeat, playerIndex, action) {
  const isFeat = (obsOrFeat instanceof Float32Array);
  const obs = isFeat ? null : obsOrFeat;
  const feat = isFeat ? obsOrFeat : null;

  // State flags (obs or feat)
  // buildFeatures indices:
  // 3 me.isAir, 4 me.isDiving, 5 me.isLying, 6 me.canAct
  let isLying = false;
  let isDiving = false;
  let isAir = false;
  if (isFeat) {
    isAir = (feat[3] > 0.5);
    isDiving = (feat[4] > 0.5);
    isLying = (feat[5] > 0.5);
  } else {
    const me = obs?.me ?? {};
    const state = Number(me.state ?? 0);
    isLying = !!me.isLying || state === 4;
    isDiving = !!me.isDiving || state === 3;
    isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);
  }

  // ✅ act()와 동일한 규칙:
  // - powerHit 하드 금지는 오직 lying/diving
  // - 나머지 powerHit gate는 (obs가 없으면) 평가에서 재현 불가 → feat-only일 때는 allow=true로 둔다.
  let allowPowerHit = true;
  if (!isFeat) {
    const gate = powerHitGate(obs, playerIndex, this.genome?.powerHitGate);
    allowPowerHit = !!gate.allow;
  }

  const a = (typeof action === 'number')
    ? { xDirection: 0, yDirection: 0, powerHit: 0 }
    : action;

  const ax = mapXDirToClass(Number(a.xDirection ?? 0));
  const ay = mapYDirToClass(Number(a.yDirection ?? 0));
  const ap = Number(a.powerHit ?? 0) ? 1 : 0;

  // Evaluate under current params
  let px, py, pp, value;
  if (!isFeat) {
    ({ px, py, pp, value } = this.evaluate(obs, playerIndex));
  } else {
    const fwd = this._forward(feat, this._infer.fwd);
    value = fwd.v;
    const probs = this._rawProbs(fwd.lx, fwd.ly, fwd.lp);
    px = probs.px;
    py = probs.py;
    pp = probs.pp;

    // 최소 하드룰(환경 합법성): ground면 DOWN(+1) 금지, lying/diving이면 power=1 금지
    if (this.hardRulesEnabled) {
      if (!isAir) {
        py[2] = 0;
        const s = py[0] + py[1];
        if (s > 1e-12) {
          const inv = 1 / s;
          py[0] *= inv;
          py[1] *= inv;
        } else {
          py[0] = 0.5;
          py[1] = 0.5;
        }
      }
      if (isLying || isDiving) {
        pp[1] = 0;
        pp[0] = 1;
      }
    }
  }

  // alloc-free: reuse scratch buffers (logpValue is called heavily during PPO)
  const ppRaw2 = this._lp.ppRaw;
  const ppEff = this._lp.ppEff;
  const pxEff = this._lp.pxEff;
  const pyEff = this._lp.pyEff;

  // (계측용) 원본 pp 정규화
  if (!allowPowerHit) {
    ppRaw2[0] = 1; ppRaw2[1] = 0;
  } else {
    const a0 = clamp01(pp[0] ?? 0);
    const b0 = clamp01(pp[1] ?? 0);
    renorm2Into(ppRaw2, a0, b0);
  }

  // act()와 동일한 POWER_MIX
  const POWER_MIX = 0.15;
  if (!allowPowerHit) {
    ppEff[0] = 1; ppEff[1] = 0;
  } else {
    const p0 = ppRaw2[0], p1 = ppRaw2[1];
    const m = POWER_MIX;
    const q0 = p0 * (1 - m) + 0.5 * m;
    const q1 = p1 * (1 - m) + 0.5 * m;
    renorm2Into(ppEff, q0, q1);
  }

  // X: ground & power=1이면 x=0 금지
  {
    const a0 = clamp01(px[0] ?? 0), b0 = clamp01(px[1] ?? 0), c0 = clamp01(px[2] ?? 0);
    if (!isAir && ap === 1) renorm3Into(pxEff, a0, 0, c0);
    else renorm3Into(pxEff, a0, b0, c0);
  }

  // Y: 공중이면 y=-1 금지
  {
    const a0 = clamp01(py[0] ?? 0), b0 = clamp01(py[1] ?? 0), c0 = clamp01(py[2] ?? 0);
    if (isAir) {
      const s = b0 + c0;
      pyEff[0] = 0;
      if (s > 1e-12) {
        const inv = 1 / s;
        pyEff[1] = b0 * inv;
        pyEff[2] = c0 * inv;
      } else {
        pyEff[1] = 0.5;
        pyEff[2] = 0.5;
      }
    } else {
      renorm3Into(pyEff, a0, b0, c0);
    }
  }

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
    this._vSum = 0; this._rSum = 0; this._vSum2 = 0; this._rSum2 = 0; this._vrSum = 0; this._vrCount = 0;
    this._advPos = 0; this._advNeg = 0; this._advZero = 0;
    let stats = { steps: batch.length, updates: 0, approxKl: 0, policyLoss: 0, valueLoss: 0, clipFrac: 0, gradNorm: 0, wNorm: 0, advMean: 0, advStd: 0, retMean: 0, retStd: 0, rewMean: 0, rewStd: 0, vrCorr: 0, auxLoss: 0, auxCount: 0 };

    // ------------------------------------------------------------
    // ✅ 1단계 alloc 제거: index buffer 재사용 + slice 제거
    // ------------------------------------------------------------
    const n = batch.length | 0;
    let idxs = this._ppoIdxs;
    if (!idxs || idxs.length !== n) {
      idxs = new Int32Array(n);
      this._ppoIdxs = idxs;
    }
    for (let i = 0; i < n; i++) idxs[i] = i;

    for (let ep = 0; ep < epochs; ep++) {
      // shuffle indices in-place (Fisher–Yates)
      for (let i = n - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const t = idxs[i]; idxs[i] = idxs[j]; idxs[j] = t;
      }

      for (let start = 0; start < n; start += mb) {
        const end = Math.min(n, start + mb);
        const nMb = Math.max(1, end - start);

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
        let auxLoss = 0;
        let auxCount = 0;

        for (let t = start; t < end; t++) {
          const id = idxs[t];
          const it = batch[id];
          const obsOrFeat = (it.feat ?? it.obs);
          const action = it.action;
          const oldLogp = Number(it.oldLogp ?? 0);
          const adv = Number(it.adv ?? 0);
          if (adv > 0) this._advPos++; else if (adv < 0) this._advNeg++; else this._advZero++;
          const ret = Number(it.ret ?? 0);

          const evalNow = this.logpValue(obsOrFeat, it.playerIndex ?? 1, action);
          const logp = evalNow.logp;
          const v = evalNow.value;
          this._vSum += v; this._vSum2 += v * v;
          this._rSum += ret; this._rSum2 += ret * ret;
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

          // ------------------------------------------------------------
          // [STEP4] Aux teacher loss (X-only)
          // ------------------------------------------------------------
          const AUX_COEF = 0.03;
          const aux = it.aux;

          let auxTeacherCls = -1;
          let auxW = 0;

          if (AUX_COEF > 0 && aux && aux.mask === 1) {
            const tt = aux.moveToLandingX;
            auxTeacherCls = (tt === 0 || tt === 1 || tt === 2) ? tt : 1;
            auxW = AUX_COEF;

            // log only (이미 evalNow.px 사용)
            auxLoss += -logProbFromProbs(evalNow.px, auxTeacherCls);
            auxCount++;
          }

          // ✅ 한 번의 forward/backprop에서 PPO+aux를 같이 누적
          this._accumulateGrads(
            g,
            obsOrFeat,
            it.playerIndex ?? 1,
            action,
            dL_dlogp,
            dL_dv,
            auxTeacherCls,
            auxW
          );
        }

        // apply grads
        const scale = 1 / Math.max(1, nMb);
        this._applyGrads(g, scale);

        // finalize minibatch metrics
        const nS = Math.max(1, nMb);
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

        stats.auxLoss += (auxCount > 0) ? (auxLoss / auxCount) : 0;
        stats.auxCount += auxCount;

        stats.updates++;
        stats.approxKl += klSum / Math.max(1, nMb);
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
    stats.auxLoss /= u;

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

  _accumulateGrads(g, obs, playerIndex, action, dL_dlogp, dL_dv, auxTeacherCls, auxW) {
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
    const tmp = this._train;
    const dlogitsX = tmp.dlogitsX; dlogitsX.fill(0);
    const dlogitsY = tmp.dlogitsY; dlogitsY.fill(0);
    const dlogitsP = tmp.dlogitsP; dlogitsP.fill(0);

    if (dL_dlogp !== 0) {
      // d logp / d logits = onehot - probs
      // Therefore: dL/dlogits = dL/dlogp * (onehot - probs)
      for (let i = 0; i < 3; i++) dlogitsX[i] = dL_dlogp * ((i === ax ? 1 : 0) - px[i]);
      for (let i = 0; i < 3; i++) dlogitsY[i] = dL_dlogp * ((i === ay ? 1 : 0) - py[i]);
      for (let i = 0; i < 2; i++) dlogitsP[i] = dL_dlogp * ((i === ap ? 1 : 0) - pp[i]);
    }

    // ------------------------------------------------------------
    // ✅ Aux CE on X head only: dL/dlogitsX += w * (px - onehot(teacher))
    // (기존 _accumulateAuxX와 수학 동일)
    // ------------------------------------------------------------
    const wAux = Number(auxW ?? 0);
    const tAux = auxTeacherCls | 0;
    if (wAux !== 0 && (tAux === 0 || tAux === 1 || tAux === 2)) {
      for (let i = 0; i < 3; i++) {
        dlogitsX[i] += wAux * (px[i] - (i === tAux ? 1 : 0));
      }
    }

    // grads for heads + accumulate dh2
    const dh2 = tmp.dh2; dh2.fill(0);

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
    const dz2 = tmp.dz2; dz2.fill(0);
    for (let i = 0; i < this.hidden2; i++) {
      const act = fwd.h2[i];
      const der = (this.activation === 'linear') ? 1 : (1 - act * act);
      dz2[i] = dh2[i] * der;
      g.db2[i] += dz2[i];
    }

    // W2 grads and dh1
    const dh1 = tmp.dh1; dh1.fill(0);
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
    const dz1 = tmp.dz1; dz1.fill(0);
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

  // ------------------------------------------------------------
  // [STEP4] Auxiliary supervised loss on X head only (CE)
  // dL/dlogits = (probs - onehot)
  // ------------------------------------------------------------
  _accumulateAuxX(g, obs, playerIndex, teacherCls, weight) {
    const w = Number(weight ?? 0);
    if (!isFinite(w) || w === 0) return;

    // forward with caches
    const feat = this.buildFeatures(obs, playerIndex);
    const fwd = this._forward(feat);

    // use the SAME masking rule as PPO (hardRulesEnabled path uses masked probs in _accumulateGrads)
    const { px } = this._maskedProbs(fwd.lx, fwd.ly, fwd.lp, obs);

    // dL/dlogitsX = w * (px - onehot(teacher))
    const tmp = this._train;
    const dlogitsX = tmp.auxDlogitsX; dlogitsX.fill(0);
    for (let i = 0; i < 3; i++) {
      dlogitsX[i] = w * (px[i] - (i === teacherCls ? 1 : 0));
    }

    // grads for X head + accumulate dh2
    const dh2 = tmp.dh2; dh2.fill(0);

    for (let k = 0; k < 3; k++) {
      const grad = dlogitsX[k];
      g.dbx[k] += grad;
      const wRow = this.Wx[k];
      const gw = g.dWx[k];
      for (let j = 0; j < this.hidden2; j++) {
        gw[j] += grad * fwd.h2[j];
        dh2[j] += grad * wRow[j];
      }
    }

    // backprop through tanh at layer2
    const dz2 = tmp.dz2; dz2.fill(0);
    for (let i = 0; i < this.hidden2; i++) {
      const act = fwd.h2[i];
      const der = (this.activation === 'linear') ? 1 : (1 - act * act);
      dz2[i] = dh2[i] * der;
      g.db2[i] += dz2[i];
    }

    // W2 grads and dh1
    const dh1 = tmp.dh1; dh1.fill(0);
    for (let i = 0; i < this.hidden2; i++) {
      const grad = dz2[i];
      const wRow = this.W2[i];
      const gw = g.dW2[i];
      for (let j = 0; j < this.hidden1; j++) {
        gw[j] += grad * fwd.h1[j];
        dh1[j] += grad * wRow[j];
      }
    }

    // backprop through tanh at layer1
    const dz1 = tmp.dz1; dz1.fill(0);
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
