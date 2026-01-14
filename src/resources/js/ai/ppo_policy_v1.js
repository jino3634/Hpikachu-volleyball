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
    this.featureLen = Math.max(1, (opts.featureLen ?? 16) | 0);

    this.hidden1 = Math.max(1, (opts.hidden1 ?? 64) | 0);
    this.hidden2 = Math.max(1, (opts.hidden2 ?? 64) | 0);
    this.activation = (opts.activation === 'linear') ? 'linear' : 'tanh';

    this.learningRate = Number(opts.learningRate ?? 3e-4);
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

    // Expected normalized inputs from getObservation():
    // positions in [-1,1], velocities roughly [-1,1] after clip/scale.
    const f = new Float32Array(this.featureLen);

    // 0..3 me
    // Note: getObservation() provides {x, y, yV, ...} (no xV for players).
    f[0] = Number(me.x ?? 0);
    f[1] = Number(me.y ?? 0);
    f[2] = 0;
    f[3] = Number(me.yV ?? me.yv ?? 0);

    // 4..7 opp
    f[4] = Number(opp.x ?? 0);
    f[5] = Number(opp.y ?? 0);
    f[6] = 0;
    f[7] = Number(opp.yV ?? opp.yv ?? 0);

    // 8..11 ball
    // getObservation() provides ball.xV/ball.yV (camel-case V).
    f[8] = Number(ball.x ?? 0);
    f[9] = Number(ball.y ?? 0);
    f[10] = Number(ball.xV ?? ball.xv ?? 0);
    f[11] = Number(ball.yV ?? ball.yv ?? 0);

    // 12..13 prediction helpers (ball)
    // landingX: expected landing x in player-centric coords ([-1,1])
    // timeToLand: normalized frames until landing (0..1)
    f[12] = Number((ball.landingX ?? ball.expectedX ?? 0));
    f[13] = Number((ball.timeToLand ?? 0));

    // 14..15 helper signals (agent)
    // serve / canAct like flags (if missing, 0)
    f[14] = (me.canAct !== undefined) ? (me.canAct ? 1 : 0) : 0;
    f[15] = (me.isServe !== undefined) ? (me.isServe ? 1 : 0) : 0;
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
    const isLying = !!me.isLying || (Number(me.lying ?? 0) > 0) || state === 4;
    const isDiving = !!me.isDiving || state === 3;
    const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
    const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

    // if cannot act, force IDLE effectively
    if (!canAct || isLying || isDiving) {
      const px = new Float32Array([0, 1, 0]);
      const py = new Float32Array([0, 1, 0]);
      const pp = new Float32Array([1, 0]);
      return { px, py, pp };
    }

    const mx = new Float32Array(logitsX);
    const my = new Float32Array(logitsY);
    const mp = new Float32Array(logitsP);

    // constraint: if ground & powerHit=1, forbid x=0 (force dive direction)
    // We'll enforce this later by re-masking x after sampling power=1 on ground.
    // Here we only restrict y on ground: forbid DOWN (+1). Jump is y=-1.
    if (!isAir) {
      // y classes: 0->-1 (jump), 1->0 (idle), 2->+1 (down)
      my[2] = -1e9;
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
    const isLying = !!me.isLying || (Number(me.lying ?? 0) > 0) || state === 4;
    const isDiving = !!me.isDiving || state === 3;
    const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
    const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

    if (!canAct || isLying || isDiving) {
      return {
        action: { xDirection: 0, yDirection: 0, powerHit: 0 },
        logp: 0,
        value: 0,
        meta: { forcedIdle: true },
      };
    }

    const { px, py, pp, value } = this.evaluate(obs, playerIndex);

    // epsilon random exploration (still valid)
    if (!deterministic && epsRand > 0 && Math.random() < epsRand) {
      const powerHit = (Math.random() < 0.5) ? 1 : 0;
      let axChoices = [0, 1, 2];
      if (!isAir && powerHit === 1) axChoices = [0, 2];
      const ayChoices = (!isAir) ? [0, 1, 2] : [0, 1, 2];
      const ax = axChoices[(Math.random() * axChoices.length) | 0];
      const ay = ayChoices[(Math.random() * ayChoices.length) | 0];
      const ap = powerHit ? 1 : 0;

      const logp = logProbFromProbs(px, ax) + logProbFromProbs(py, ay) + logProbFromProbs(pp, ap);
      return {
        action: { xDirection: mapClassToXDir(ax), yDirection: mapClassToYDir(ay), powerHit },
        logp,
        value,
        meta: { epsRandom: true, ax, ay, ap },
      };
    }

    // deterministic = argmax, else sample
    let ax = 1, ay = 1, ap = 0;
    if (deterministic) {
      let best = 0, bestv = -Infinity;
      for (let i = 0; i < 3; i++) { if (px[i] > bestv) { bestv = px[i]; best = i; } }
      ax = best;
      best = 0; bestv = -Infinity;
      for (let i = 0; i < 3; i++) { if (py[i] > bestv) { bestv = py[i]; best = i; } }
      ay = best;
      ap = (pp[1] > pp[0]) ? 1 : 0;
    } else {
      ap = sampleCategorical(pp);
      // if ground and power=1, mask x=0
      if (!isAir && ap === 1) {
        const px2 = new Float32Array(px);
        px2[1] = 0;
        const s = px2[0] + px2[2];
        if (s > 0) { px2[0] /= s; px2[2] /= s; }
        ax = sampleCategorical(px2);
      } else {
        ax = sampleCategorical(px);
      }
      ay = sampleCategorical(py);
    }

    const logp = logProbFromProbs(px, ax) + logProbFromProbs(py, ay) + logProbFromProbs(pp, ap);
    const action = { xDirection: mapClassToXDir(ax), yDirection: mapClassToYDir(ay), powerHit: ap ? 1 : 0 };
    return { action, logp, value, meta: { ax, ay, ap } };
  }

  /**
   * Compute log-prob and value under current params for a given (obs, action).
   * @param {any} obs
   * @param {1|2} playerIndex
   * @param {{xDirection:number,yDirection:number,powerHit:number}|number} action
   */
  logpValue(obs, playerIndex, action) {
    const a = (typeof action === 'number') ? { xDirection: 0, yDirection: 0, powerHit: 0 } : action;
    const ax = mapXDirToClass(Number(a.xDirection ?? 0));
    const ay = mapYDirToClass(Number(a.yDirection ?? 0));
    const ap = Number(a.powerHit ?? 0) ? 1 : 0;
    const { px, py, pp, value } = this.evaluate(obs, playerIndex);
    const logp = logProbFromProbs(px, ax) + logProbFromProbs(py, ay) + logProbFromProbs(pp, ap);
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

    let stats = { steps: batch.length, updates: 0, approxKl: 0 };

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
        for (const id of slice) {
          const it = batch[id];
          const obs = it.obs;
          const action = it.action;
          const oldLogp = Number(it.oldLogp ?? 0);
          const adv = Number(it.adv ?? 0);
          const ret = Number(it.ret ?? 0);

          const evalNow = this.logpValue(obs, it.playerIndex ?? 1, action);
          const logp = evalNow.logp;
          const v = evalNow.value;
          const ratio = Math.exp(clip(logp - oldLogp, -10, 10));

          // PPO clipped objective: L = -min(ratio*adv, clip(ratio)*adv)
          const clipEps = this.clipEps;
          const rClipped = clip(ratio, 1 - clipEps, 1 + clipEps);

          // determine if gradient flows
          let dL_dlogp = 0;
          const useClipped = (adv >= 0 && ratio > 1 + clipEps) || (adv < 0 && ratio < 1 - clipEps);
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

        stats.updates++;
        stats.approxKl += klSum / Math.max(1, slice.length);
      }
    }

    stats.approxKl /= Math.max(1, stats.updates);
    return stats;
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
      for (let i = 0; i < 3; i++) dlogitsX[i] = -dL_dlogp * ((i === ax ? 1 : 0) - px[i]);
      for (let i = 0; i < 3; i++) dlogitsY[i] = -dL_dlogp * ((i === ay ? 1 : 0) - py[i]);
      for (let i = 0; i < 2; i++) dlogitsP[i] = -dL_dlogp * ((i === ap ? 1 : 0) - pp[i]);
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

    // Backprop through existing policy-gradient accumulator.
    // NOTE: _accumulateGrads uses:
    //   dlogits = -dL_dlogp * (onehot - probs)
    // For CE loss L = -log p(a), the correct gradient w.r.t logits is:
    //   dL/dlogits = (probs - onehot)
    // which corresponds to dL_dlogp = +1.
    // Therefore we pass +w (NOT -w) to *increase* probability of the demonstrated action.
    const g = this._zeroGrads();
    this._accumulateGrads(g, obs, playerIndex, label, +w, 0);

    // Apply grads using imitation LR (fallback to actor LR)
    const lr = (this.imitationLr !== undefined) ? Number(this.imitationLr) : (this.lrActor ?? this.lr);
    this._applyGrads(g, lr);

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
