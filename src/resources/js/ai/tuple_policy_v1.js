// tuple_policy_v1.js
'use strict';

/**
 * TuplePolicyV1 (MLP): (xDirection, yDirection, powerHit) 를 직접 출력하는 확률 정책.
 *
 * 구조:
 *   features(F) -> MLP trunk (H1,H2) -> 3 heads
 *     - headX: 3-class softmax for xDirection ∈ {-1,0,+1}
 *     - headY: 3-class softmax for yDirection ∈ {-1,0,+1}
 *     - headP: 2-class softmax for powerHit ∈ {0,1}
 *
 * 기존(선형) 체크포인트(kind: 'tuple_policy_v1')도 로드 가능:
 *   - trunk를 identity(linear activation)로 구성해 완전히 동일한 동작을 재현.
 */

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function softmax(arr) {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  const exps = new Float32Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const e = Math.exp(arr[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < arr.length; i++) exps[i] /= (sum || 1);
  return exps;
}

function sampleCategorical(probs) {
  let r = Math.random();
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (r <= c) return i;
  }
  return probs.length - 1;
}

function dot(Wa, feat) {
  let s = 0;
  for (let j = 0; j < feat.length; j++) s += Wa[j] * feat[j];
  return s;
}

function matVec(W, x, b) {
  const out = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) {
    out[i] = dot(W[i], x) + (b ? b[i] : 0);
  }
  return out;
}

function addScaledOuter(W, a, x, scale) {
  // W -= scale * a ⊗ x
  for (let i = 0; i < W.length; i++) {
    const Wi = W[i];
    const ai = a[i];
    if (ai === 0) continue;
    const s = scale * ai;
    for (let j = 0; j < x.length; j++) Wi[j] -= s * x[j];
  }
}

function tanhVec(z) {
  const out = new Float32Array(z.length);
  for (let i = 0; i < z.length; i++) out[i] = Math.tanh(z[i]);
  return out;
}

function tanhGradFromAct(a) {
  // d/dz tanh(z) = 1 - tanh(z)^2 = 1 - a^2
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    out[i] = 1 - v * v;
  }
  return out;
}

function hadamardInPlace(a, b) {
  for (let i = 0; i < a.length; i++) a[i] *= b[i];
  return a;
}

function matTVec(W, g) {
  // W^T * g
  const nOut = W.length;
  const nIn = (W[0] ? W[0].length : 0);
  const out = new Float32Array(nIn);
  for (let i = 0; i < nOut; i++) {
    const Wi = W[i];
    const gi = g[i];
    if (gi === 0) continue;
    for (let j = 0; j < nIn; j++) out[j] += Wi[j] * gi;
  }
  return out;
}

function clampInt(x, lo, hi) {
  const v = x | 0;
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// label mapping helpers
export function mapXDirToClass(x) {
  const v = (x | 0);
  if (v < 0) return 0; // -1
  if (v > 0) return 2; // +1
  return 1; // 0
}
export function mapYDirToClass(y) {
  const v = (y | 0);
  if (v < 0) return 0; // -1
  if (v > 0) return 2; // +1
  return 1; // 0
}
export function mapClassToXDir(c) {
  if ((c | 0) === 0) return -1;
  if ((c | 0) === 2) return +1;
  return 0;
}
export function mapClassToYDir(c) {
  if ((c | 0) === 0) return -1;
  if ((c | 0) === 2) return +1;
  return 0;
}

export class TuplePolicyV1 {
  /**
   * @param {{featureLen?:number, learningRate?:number, epsilon?:number, initStd?:number, hidden1?:number, hidden2?:number, activation?:'tanh'|'linear'}} opts
   */
  constructor(opts = {}) {
    // Feature length is tied to observation schema. Keep in sync with PolicyV1.buildFeatures.
    this.featureLen = Math.max(1, (opts.featureLen ?? 1) | 0);
    this.learningRate = Number(opts.learningRate ?? 0.001);
    this.epsilon = Number(opts.epsilon ?? 0.05);
    this.initStd = Number(opts.initStd ?? 0.02);

    // MLP sizes
    this.hidden1 = Math.max(1, (opts.hidden1 ?? 64) | 0);
    this.hidden2 = Math.max(1, (opts.hidden2 ?? 64) | 0);
    this.activation = (opts.activation === 'linear') ? 'linear' : 'tanh';

    // trunk
    this.W1 = []; // [H1][F]
    this.b1 = new Float32Array(this.hidden1);
    this.W2 = []; // [H2][H1]
    this.b2 = new Float32Array(this.hidden2);

    // heads
    this.Wx = []; // [3][H2]
    this.bx = new Float32Array(3);
    this.Wy = []; // [3][H2]
    this.by = new Float32Array(3);
    this.Wp = []; // [2][H2]
    this.bp = new Float32Array(2);

    this._initParams();
  }

  _initMatrix(rows, cols, std) {
    const out = [];
    for (let i = 0; i < rows; i++) {
      const r = new Float32Array(cols);
      for (let j = 0; j < cols; j++) r[j] = randn() * std;
      out.push(r);
    }
    return out;
  }

  _initParams() {
    const std = this.initStd;
    this.W1 = this._initMatrix(this.hidden1, this.featureLen, std);
    this.b1 = new Float32Array(this.hidden1);
    this.W2 = this._initMatrix(this.hidden2, this.hidden1, std);
    this.b2 = new Float32Array(this.hidden2);

    this.Wx = this._initMatrix(3, this.hidden2, std);
    this.bx = new Float32Array(3);
    this.Wy = this._initMatrix(3, this.hidden2, std);
    this.by = new Float32Array(3);
    this.Wp = this._initMatrix(2, this.hidden2, std);
    this.bp = new Float32Array(2);
  }

  /**
   * 동일한 feature 정의를 사용 (PolicyV1과 sync)
   * @param {any} obs
   * @param {1|2} playerIndex
   * @returns {Float32Array}
   */
  buildFeatures(obs, playerIndex) {
    const me = obs?.me ?? {};
    const opp = obs?.opp ?? {};
    const ball = obs?.ball ?? {};

    const feat = new Float32Array(this.featureLen);
    let k = 0;

    // bias
    feat[k++] = 1;

    // me
    feat[k++] = Number(me.x ?? 0);
    feat[k++] = Number(me.y ?? 0);
    feat[k++] = Number(me.yV ?? 0);

    // opp
    feat[k++] = Number(opp.x ?? 0);
    feat[k++] = Number(opp.y ?? 0);
    feat[k++] = Number(opp.yV ?? 0);

    // ball
    feat[k++] = Number(ball.x ?? 0);
    feat[k++] = Number(ball.y ?? 0);
    feat[k++] = Number(ball.xV ?? 0);
    feat[k++] = Number(ball.yV ?? 0);
    feat[k++] = Number(ball.expectedX ?? ball.x ?? 0);

    // serve / power flag
    feat[k++] = Number(obs?.isPlayer2Serve ?? 0);
    feat[k++] = Number(ball.isPowerHit ?? 0);

    // If featureLen is larger than 14 (future extension), remaining entries stay 0.
    return feat;
  }

  _activate(z) {
    if (this.activation === 'linear') return z;
    return tanhVec(z);
  }

  _activationGradFromAct(a) {
    if (this.activation === 'linear') {
      const out = new Float32Array(a.length);
      out.fill(1);
      return out;
    }
    return tanhGradFromAct(a);
  }

  _forward(feat) {
    const z1 = matVec(this.W1, feat, this.b1);
    const h1 = this._activate(z1);
    const z2 = matVec(this.W2, h1, this.b2);
    const h2 = this._activate(z2);
    return { feat, h1, h2 };
  }

  _logitsHead(W, b, h2) {
    const out = new Float32Array(b.length);
    for (let a = 0; a < b.length; a++) out[a] = dot(W[a], h2) + b[a];
    return out;
  }

    /**
   * Deterministic (argmax) input with action validity masking.
   * - Ground: forbid y=+1 (POWER_DOWN)
   * - If lying/diving/can't act: force IDLE
   * - If ground & powerHit=1: forbid x=0 (force dive direction)
   */
  actDeterministic(obs, playerIndex) {
    const me = obs?.me ?? {};
    const state = Number(me.state ?? 0);
    const isLying = !!me.isLying || (Number(me.lying ?? 0) > 0) || state === 4;
    const isDiving = !!me.isDiving || state === 3;
    const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
    const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

    if (!canAct || isLying || isDiving) {
      return { xDirection: 0, yDirection: 0, powerHit: 0 };
    }

    const feat = this.buildFeatures(obs, playerIndex);
    const { h2 } = this._forward(feat);

    const lx = this._logitsHead(this.Wx, this.bx, h2);
    const ly = this._logitsHead(this.Wy, this.by, h2);
    const lp = this._logitsHead(this.Wp, this.bp, h2);

    // Choose powerHit first (argmax)
    let ap = 0;
    for (let i = 1; i < 2; i++) if (lp[i] > lp[ap]) ap = i;
    const powerHit = ap | 0;

    // X selection (mask x=0 if ground & powerHit=1 to avoid wasted "power only")
    let ax = 0;
    if (!isAir && powerHit === 1) {
      // argmax among classes {0,2} -> xDir {-1,+1}
      ax = (lx[2] > lx[0]) ? 2 : 0;
    } else {
      for (let i = 1; i < 3; i++) if (lx[i] > lx[ax]) ax = i;
    }

    // Y selection (mask y=+1 on ground)
    let ay = 0;
    if (!isAir) {
      // choose among {0,1} -> yDir {-1,0}
      ay = (ly[1] > ly[0]) ? 1 : 0;
    } else {
      for (let i = 1; i < 3; i++) if (ly[i] > ly[ay]) ay = i;
    }

    return {
      xDirection: mapClassToXDir(ax),
      yDirection: mapClassToYDir(ay),
      powerHit,
    };
  }

  /**
   * Stochastic input (softmax sampling + epsilon random) with action validity masking.
   * - Ground: forbid y=+1 (POWER_DOWN)
   * - If lying/diving/can't act: force IDLE
   * - If ground & powerHit=1: forbid x=0 (force dive direction)
   */
  actStochastic(obs, playerIndex) {
    const me = obs?.me ?? {};
    const state = Number(me.state ?? 0);
    const isLying = !!me.isLying || (Number(me.lying ?? 0) > 0) || state === 4;
    const isDiving = !!me.isDiving || state === 3;
    const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
    const isAir = (me.isAir !== undefined) ? !!me.isAir : (state === 1 || state === 2);

    if (!canAct || isLying || isDiving) {
      return { xDirection: 0, yDirection: 0, powerHit: 0 };
    }

    const randChoice = (arr) => arr[(Math.random() * arr.length) | 0];

    // epsilon random (but still valid)
    if (Math.random() < this.epsilon) {
      const powerHit = randChoice([0, 1]);
      const xClasses = (!isAir && powerHit === 1) ? [0, 2] : [0, 1, 2];
      const yClasses = (!isAir) ? [0, 1] : [0, 1, 2];
      const ax = randChoice(xClasses);
      const ay = randChoice(yClasses);
      return {
        xDirection: mapClassToXDir(ax),
        yDirection: mapClassToYDir(ay),
        powerHit,
      };
    }

    const feat = this.buildFeatures(obs, playerIndex);
    const { h2 } = this._forward(feat);

    const px = softmax(Array.from(this._logitsHead(this.Wx, this.bx, h2)));
    const py = softmax(Array.from(this._logitsHead(this.Wy, this.by, h2)));
    const pp = softmax(Array.from(this._logitsHead(this.Wp, this.bp, h2)));

    // sample powerHit first
    const powerHit = sampleCategorical(pp);

    // sample x with mask if needed
    let ax = 0;
    if (!isAir && powerHit === 1) {
      // normalize probabilities over {0,2}
      const a0 = px[0], a2 = px[2];
      const s = a0 + a2;
      const r = Math.random() * (s > 0 ? s : 1);
      ax = (r < a0) ? 0 : 2;
    } else {
      ax = sampleCategorical(px);
    }

    // sample y with ground mask (forbid +1)
    let ay = 0;
    if (!isAir) {
      const a0 = py[0], a1 = py[1];
      const s = a0 + a1;
      const r = Math.random() * (s > 0 ? s : 1);
      ay = (r < a0) ? 0 : 1;
    } else {
      ay = sampleCategorical(py);
    }

    return {
      xDirection: mapClassToXDir(ax),
      yDirection: mapClassToYDir(ay),
      powerHit,
    };
  }

/**
   * Supervised/RL-lite update for one sample.
   * sign=+1 강화, sign=-1 억제
   * @param {Float32Array} feat
   * @param {{xDirection:number, yDirection:number, powerHit:number}} label
   * @param {number} sign
   * @returns {{loss:number, ax:number, ay:number, ap:number}}
   */
  updateImitation(feat, label, sign = 1) {
    const tx = clampInt(mapXDirToClass(label.xDirection), 0, 2);
    const ty = clampInt(mapYDirToClass(label.yDirection), 0, 2);
    const tp = (label.powerHit ?? 0) ? 1 : 0;

    // forward trunk
    const { h1, h2 } = this._forward(feat);

    // heads forward
    const lx = this._logitsHead(this.Wx, this.bx, h2);
    const ly = this._logitsHead(this.Wy, this.by, h2);
    const lp = this._logitsHead(this.Wp, this.bp, h2);

    const px = softmax(Array.from(lx));
    const py = softmax(Array.from(ly));
    const pp = softmax(Array.from(lp));

    const eps = 1e-8;
    const loss = -Math.log(px[tx] + eps) - Math.log(py[ty] + eps) - Math.log(pp[tp] + eps);

    // gradients at logits: (p - y)
    const gx = new Float32Array(3);
    const gy = new Float32Array(3);
    const gp = new Float32Array(2);
    for (let i = 0; i < 3; i++) gx[i] = px[i] - (i === tx ? 1 : 0);
    for (let i = 0; i < 3; i++) gy[i] = py[i] - (i === ty ? 1 : 0);
    for (let i = 0; i < 2; i++) gp[i] = pp[i] - (i === tp ? 1 : 0);

    const lr = this.learningRate * (Number(sign) || 0);
    if (lr === 0) return { loss, ax: tx, ay: ty, ap: tp };

    // --- head updates + accumulate dL/dh2
    const dh2 = new Float32Array(this.hidden2);
    // headX
    for (let a = 0; a < 3; a++) {
      const ga = gx[a];
      const Wa = this.Wx[a];
      for (let j = 0; j < dh2.length; j++) dh2[j] += Wa[j] * ga;
      for (let j = 0; j < h2.length; j++) Wa[j] -= lr * ga * h2[j];
      this.bx[a] -= lr * ga;
    }
    // headY
    for (let a = 0; a < 3; a++) {
      const ga = gy[a];
      const Wa = this.Wy[a];
      for (let j = 0; j < dh2.length; j++) dh2[j] += Wa[j] * ga;
      for (let j = 0; j < h2.length; j++) Wa[j] -= lr * ga * h2[j];
      this.by[a] -= lr * ga;
    }
    // headP
    for (let a = 0; a < 2; a++) {
      const ga = gp[a];
      const Wa = this.Wp[a];
      for (let j = 0; j < dh2.length; j++) dh2[j] += Wa[j] * ga;
      for (let j = 0; j < h2.length; j++) Wa[j] -= lr * ga * h2[j];
      this.bp[a] -= lr * ga;
    }

    // --- backprop through trunk (h2 -> h1 -> feat)
    // dL/dz2 = dL/dh2 * act'(z2) ; we only have h2, so use act' from activation output
    const dz2 = new Float32Array(dh2);
    hadamardInPlace(dz2, this._activationGradFromAct(h2));

    // W2,b2 update
    // W2 -= lr * dz2 ⊗ h1
    addScaledOuter(this.W2, dz2, h1, lr);
    for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * dz2[i];

    // dL/dh1 = W2^T * dz2
    const dh1 = matTVec(this.W2, dz2);
    const dz1 = new Float32Array(dh1);
    hadamardInPlace(dz1, this._activationGradFromAct(h1));

    // W1,b1 update
    addScaledOuter(this.W1, dz1, feat, lr);
    for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * dz1[i];

    // predictions (argmax)
    let ax = 0, ay = 0, ap = 0;
    for (let i = 1; i < 3; i++) if (px[i] > px[ax]) ax = i;
    for (let i = 1; i < 3; i++) if (py[i] > py[ay]) ay = i;
    if (pp[1] > pp[0]) ap = 1;

    return { loss, ax, ay, ap };
  }

  /**
   * Lightweight on-policy improvement from an episode.
   * @param {any} episode
   * @returns {{updated:number, avgLoss:number}}
   */
  learnFromEpisode(episode) {
    const trans = episode?.transitions;
    if (!Array.isArray(trans) || trans.length === 0) return { updated: 0, avgLoss: 0 };

    const learningPlayer = /** @type {1|2} */ (episode.learningPlayer ?? 1);

    // --- Collect usable transitions (must have action object) and their rewards.
    const usable = [];
    const rewards = [];
    for (const tr of trans) {
      const a = tr?.action;
      if (!a || typeof a !== 'object') continue;
      const r = Number(tr?.reward ?? 0);
      usable.push(tr);
      rewards.push(r);
    }
    const T = usable.length;
    if (T === 0) return { updated: 0, avgLoss: 0 };

    // --- Baseline (mean) + scale (std) to stabilize updates across episodes.
    let mean = 0;
    for (let i = 0; i < rewards.length; i++) mean += rewards[i];
    mean /= T;

    let varSum = 0;
    for (let i = 0; i < rewards.length; i++) {
      const d = rewards[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / T);
    const scale = std > 1e-6 ? std : 1; // avoid division by 0

    // --- Length normalization: make long/short episodes have similar total impact.
    const lenNorm = 1 / Math.sqrt(T);

    // --- Clip advantages to avoid rare spikes dominating updates.
    const ADV_CLIP = 3;

    let updated = 0;
    let lossSum = 0;

    for (let i = 0; i < T; i++) {
      const tr = usable[i];
      const a = tr.action;

      const r = rewards[i];
      // Advantage: centered, scaled, and length-normalized.
      let advantage = ((r - mean) / scale) * lenNorm;
      if (advantage > ADV_CLIP) advantage = ADV_CLIP;
      else if (advantage < -ADV_CLIP) advantage = -ADV_CLIP;

      // If advantage is ~0, skip to save compute (no meaningful update).
      if (Math.abs(advantage) < 1e-12) continue;

      const feat = this.buildFeatures(tr?.obs, learningPlayer);
      const out = this.updateImitation(feat, a, advantage);
      updated++;
      lossSum += out.loss;
    }

    return { updated, avgLoss: updated ? (lossSum / updated) : 0 };
  }

  clone() {
    const c = new TuplePolicyV1({
      featureLen: this.featureLen,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      initStd: this.initStd,
      hidden1: this.hidden1,
      hidden2: this.hidden2,
      activation: this.activation,
    });
    c.W1 = this.W1.map(w => new Float32Array(w));
    c.b1 = new Float32Array(this.b1);
    c.W2 = this.W2.map(w => new Float32Array(w));
    c.b2 = new Float32Array(this.b2);

    c.Wx = this.Wx.map(w => new Float32Array(w));
    c.bx = new Float32Array(this.bx);
    c.Wy = this.Wy.map(w => new Float32Array(w));
    c.by = new Float32Array(this.by);
    c.Wp = this.Wp.map(w => new Float32Array(w));
    c.bp = new Float32Array(this.bp);
    return c;
  }

  addNoise(std = 0.01) {
    const s = Number(std);
    const addNoiseMat = (M) => {
      for (const row of M) for (let j = 0; j < row.length; j++) row[j] += randn() * s;
    };
    addNoiseMat(this.W1);
    addNoiseMat(this.W2);
    addNoiseMat(this.Wx);
    addNoiseMat(this.Wy);
    addNoiseMat(this.Wp);
    for (let i = 0; i < this.b1.length; i++) this.b1[i] += randn() * s;
    for (let i = 0; i < this.b2.length; i++) this.b2[i] += randn() * s;
    for (let i = 0; i < this.bx.length; i++) this.bx[i] += randn() * s;
    for (let i = 0; i < this.by.length; i++) this.by[i] += randn() * s;
    for (let i = 0; i < this.bp.length; i++) this.bp[i] += randn() * s;
  }

  saveState() {
    return {
      kind: 'tuple_policy_mlp_v1',
      featureLen: this.featureLen,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      initStd: this.initStd,
      hidden1: this.hidden1,
      hidden2: this.hidden2,
      activation: this.activation,
      W1: this.W1.map(w => Array.from(w)),
      b1: Array.from(this.b1),
      W2: this.W2.map(w => Array.from(w)),
      b2: Array.from(this.b2),
      Wx: this.Wx.map(w => Array.from(w)),
      bx: Array.from(this.bx),
      Wy: this.Wy.map(w => Array.from(w)),
      by: Array.from(this.by),
      Wp: this.Wp.map(w => Array.from(w)),
      bp: Array.from(this.bp),
      updatedAt: Date.now(),
    };
  }

  loadState(state) {
    if (!state || typeof state !== 'object') return;

    // --- Backward compatibility: old linear state (kind: 'tuple_policy_v1')
    if (state.kind === 'tuple_policy_v1' && state.Wx && state.Wy && state.Wp) {
      // Build identity trunk so old heads act directly on features.
      this.featureLen = Math.max(1, state.featureLen | 0);
      this.hidden1 = this.featureLen;
      this.hidden2 = this.featureLen;
      this.learningRate = Number(state.learningRate ?? this.learningRate);
      this.epsilon = Number(state.epsilon ?? this.epsilon);
      this.initStd = Number(state.initStd ?? this.initStd);
      this.activation = 'linear';

      // identity matrices
      this.W1 = [];
      for (let i = 0; i < this.hidden1; i++) {
        const row = new Float32Array(this.featureLen);
        row[i] = 1;
        this.W1.push(row);
      }
      this.b1 = new Float32Array(this.hidden1);

      this.W2 = [];
      for (let i = 0; i < this.hidden2; i++) {
        const row = new Float32Array(this.hidden1);
        row[i] = 1;
        this.W2.push(row);
      }
      this.b2 = new Float32Array(this.hidden2);

      const toF32 = (arr) => new Float32Array(Array.isArray(arr) ? arr.map(Number) : []);
      const toHead = (head, nClass) => {
        const out = [];
        for (let a = 0; a < nClass; a++) {
          const row = head?.[a] ?? [];
          const w = toF32(row);
          const ww = new Float32Array(this.hidden2);
          ww.set(w.subarray(0, this.hidden2));
          out.push(ww);
        }
        return out;
      };
      this.Wx = toHead(state.Wx, 3);
      this.bx = toF32(state.bx ?? []);
      if (this.bx.length !== 3) this.bx = new Float32Array(3);
      this.Wy = toHead(state.Wy, 3);
      this.by = toF32(state.by ?? []);
      if (this.by.length !== 3) this.by = new Float32Array(3);
      this.Wp = toHead(state.Wp, 2);
      this.bp = toF32(state.bp ?? []);
      if (this.bp.length !== 2) this.bp = new Float32Array(2);
      return;
    }

    if (state.kind !== 'tuple_policy_mlp_v1') return;

    this.featureLen = Math.max(1, state.featureLen | 0);
    this.hidden1 = Math.max(1, (state.hidden1 ?? this.hidden1) | 0);
    this.hidden2 = Math.max(1, (state.hidden2 ?? this.hidden2) | 0);
    this.learningRate = Number(state.learningRate ?? this.learningRate);
    this.epsilon = Number(state.epsilon ?? this.epsilon);
    this.initStd = Number(state.initStd ?? this.initStd);
    this.activation = (state.activation === 'linear') ? 'linear' : 'tanh';

    const toF32 = (arr) => new Float32Array(Array.isArray(arr) ? arr.map(Number) : []);
    const toMat = (mat, rows, cols) => {
      const out = [];
      for (let i = 0; i < rows; i++) {
        const row = mat?.[i] ?? [];
        const w = toF32(row);
        const ww = new Float32Array(cols);
        ww.set(w.subarray(0, cols));
        out.push(ww);
      }
      return out;
    };

    this.W1 = toMat(state.W1, this.hidden1, this.featureLen);
    this.b1 = toF32(state.b1 ?? []);
    if (this.b1.length !== this.hidden1) this.b1 = new Float32Array(this.hidden1);

    this.W2 = toMat(state.W2, this.hidden2, this.hidden1);
    this.b2 = toF32(state.b2 ?? []);
    if (this.b2.length !== this.hidden2) this.b2 = new Float32Array(this.hidden2);

    this.Wx = toMat(state.Wx, 3, this.hidden2);
    this.bx = toF32(state.bx ?? []);
    if (this.bx.length !== 3) this.bx = new Float32Array(3);

    this.Wy = toMat(state.Wy, 3, this.hidden2);
    this.by = toF32(state.by ?? []);
    if (this.by.length !== 3) this.by = new Float32Array(3);

    this.Wp = toMat(state.Wp, 2, this.hidden2);
    this.bp = toF32(state.bp ?? []);
    if (this.bp.length !== 2) this.bp = new Float32Array(2);
  }
}


function sanitizeAction(action, obs, phase) {
  const me = obs.me || {};
  const isAir = !!me.isAir;
  const isLying = !!me.isLying;
  const isDiving = !!me.isDiving;
  const canAct = me.canAct !== false;
  if (isLying || isDiving || !canAct || obs.roundEnded || obs.gameEnded || me.state===5 || me.state===6) {
    return {xDirection:0,yDirection:0,powerHit:0};
  }
  // phase safety
  if (phase !== 0) action.powerHit = 0;
  // ground constraints
  if (!isAir) {
    if (action.yDirection === 1) action.yDirection = 0;
    if (action.powerHit === 1) {
      action.yDirection = 0; // critical fix
      if (action.xDirection === 0) action.xDirection = Math.random()<0.5?-1:1;
    }
  }
  return sanitizeAction(action, obs, phaseInDecisionInterval);
}
