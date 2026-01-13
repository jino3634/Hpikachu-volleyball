// tuple_policy_v1.js
'use strict';

/**
 * TuplePolicyV1: (xDirection, yDirection, powerHit) 를 직접 출력하는 간단한 선형 정책.
 * - headX: 3-class softmax for xDirection ∈ {-1,0,+1}
 * - headY: 3-class softmax for yDirection ∈ {-1,0,+1}
 * - headP: 2-class softmax for powerHit ∈ {0,1}
 *
 * Warmup imitation(지도학습)과, 이후 self-play/탐색용 확률 정책에 사용.
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
   * @param {{featureLen?:number, learningRate?:number, epsilon?:number, initStd?:number}} opts
   */
  constructor(opts = {}) {
    // Feature length is tied to observation schema. Keep in sync with PolicyV1.buildFeatures.
    this.featureLen = Math.max(1, (opts.featureLen ?? 1) | 0);
    this.learningRate = Number(opts.learningRate ?? 0.001);
    this.epsilon = Number(opts.epsilon ?? 0.05);
    this.initStd = Number(opts.initStd ?? 0.01);

    // weights
    this.Wx = []; // [3][F]
    this.bx = new Float32Array(3);
    this.Wy = []; // [3][F]
    this.by = new Float32Array(3);
    this.Wp = []; // [2][F]
    this.bp = new Float32Array(2);

    this._initParams();
  }

  _initParams() {
    this.Wx = [];
    this.Wy = [];
    this.Wp = [];
    for (let a = 0; a < 3; a++) {
      const w = new Float32Array(this.featureLen);
      for (let j = 0; j < this.featureLen; j++) w[j] = randn() * this.initStd;
      this.Wx.push(w);
    }
    for (let a = 0; a < 3; a++) {
      const w = new Float32Array(this.featureLen);
      for (let j = 0; j < this.featureLen; j++) w[j] = randn() * this.initStd;
      this.Wy.push(w);
    }
    for (let a = 0; a < 2; a++) {
      const w = new Float32Array(this.featureLen);
      for (let j = 0; j < this.featureLen; j++) w[j] = randn() * this.initStd;
      this.Wp.push(w);
    }
    this.bx = new Float32Array(3);
    this.by = new Float32Array(3);
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

    // if featureLen mismatch, rebuild a vector to match.
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

    return feat;
  }

  _logitsHead(W, b, feat) {
    const out = new Float32Array(b.length);
    for (let a = 0; a < b.length; a++) out[a] = dot(W[a], feat) + b[a];
    return out;
  }

  /**
   * Deterministic (argmax) input
   */
  actDeterministic(obs, playerIndex) {
    const feat = this.buildFeatures(obs, playerIndex);

    const lx = this._logitsHead(this.Wx, this.bx, feat);
    const ly = this._logitsHead(this.Wy, this.by, feat);
    const lp = this._logitsHead(this.Wp, this.bp, feat);

    let ax = 0, ay = 0, ap = 0;
    for (let i = 1; i < 3; i++) if (lx[i] > lx[ax]) ax = i;
    for (let i = 1; i < 3; i++) if (ly[i] > ly[ay]) ay = i;
    for (let i = 1; i < 2; i++) if (lp[i] > lp[ap]) ap = i;

    return {
      xDirection: mapClassToXDir(ax),
      yDirection: mapClassToYDir(ay),
      powerHit: ap | 0,
    };
  }

  /**
   * Stochastic input (softmax sampling + epsilon random)
   */
  actStochastic(obs, playerIndex) {
    if (Math.random() < this.epsilon) {
      const ax = (Math.random() * 3) | 0;
      const ay = (Math.random() * 3) | 0;
      const ap = (Math.random() * 2) | 0;
      return {
        xDirection: mapClassToXDir(ax),
        yDirection: mapClassToYDir(ay),
        powerHit: ap,
      };
    }

    const feat = this.buildFeatures(obs, playerIndex);
    const px = softmax(Array.from(this._logitsHead(this.Wx, this.bx, feat)));
    const py = softmax(Array.from(this._logitsHead(this.Wy, this.by, feat)));
    const pp = softmax(Array.from(this._logitsHead(this.Wp, this.bp, feat)));

    const ax = sampleCategorical(px);
    const ay = sampleCategorical(py);
    const ap = sampleCategorical(pp);

    return {
      xDirection: mapClassToXDir(ax),
      yDirection: mapClassToYDir(ay),
      powerHit: ap,
    };
  }

  /**
   * Supervised update for one sample.
   * @param {Float32Array} feat
   * @param {{xDirection:number, yDirection:number, powerHit:number}} label
   * @returns {{loss:number, ax:number, ay:number, ap:number}}
   */
  updateImitation(feat, label, sign = 1) {
    const tx = mapXDirToClass(label.xDirection);
    const ty = mapYDirToClass(label.yDirection);
    const tp = (label.powerHit ?? 0) ? 1 : 0;

    // forward
    const lx = this._logitsHead(this.Wx, this.bx, feat);
    const ly = this._logitsHead(this.Wy, this.by, feat);
    const lp = this._logitsHead(this.Wp, this.bp, feat);

    const px = softmax(Array.from(lx));
    const py = softmax(Array.from(ly));
    const pp = softmax(Array.from(lp));

    const eps = 1e-8;
    const loss = -Math.log(px[tx] + eps) - Math.log(py[ty] + eps) - Math.log(pp[tp] + eps);

    // gradients: (p - y)
    const gx = new Float32Array(3);
    const gy = new Float32Array(3);
    const gp = new Float32Array(2);
    for (let i = 0; i < 3; i++) gx[i] = px[i] - (i === tx ? 1 : 0);
    for (let i = 0; i < 3; i++) gy[i] = py[i] - (i === ty ? 1 : 0);
    for (let i = 0; i < 2; i++) gp[i] = pp[i] - (i === tp ? 1 : 0);

    const lr = this.learningRate * (Number(sign) || 0);

    // update weights
    for (let a = 0; a < 3; a++) {
      const Wa = this.Wx[a];
      const ga = gx[a];
      for (let j = 0; j < feat.length; j++) Wa[j] -= lr * ga * feat[j];
      this.bx[a] -= lr * ga;
    }
    for (let a = 0; a < 3; a++) {
      const Wa = this.Wy[a];
      const ga = gy[a];
      for (let j = 0; j < feat.length; j++) Wa[j] -= lr * ga * feat[j];
      this.by[a] -= lr * ga;
    }
    for (let a = 0; a < 2; a++) {
      const Wa = this.Wp[a];
      const ga = gp[a];
      for (let j = 0; j < feat.length; j++) Wa[j] -= lr * ga * feat[j];
      this.bp[a] -= lr * ga;
    }

    // predictions
    let ax = 0, ay = 0, ap = 0;
    for (let i = 1; i < 3; i++) if (px[i] > px[ax]) ax = i;
    for (let i = 1; i < 3; i++) if (py[i] > py[ay]) ay = i;
    if (pp[1] > pp[0]) ap = 1;

    return { loss, ax, ay, ap };
  }

  /**
   * Lightweight on-policy improvement from an episode.
   * - If actions in transitions are tuple objects, treat them as labels.
   * - We only reinforce positive-reward transitions to avoid destabilizing early training.
   *
   * This is intentionally simple: it keeps the pipeline running and improves
   * the policy without introducing a full RL algorithm yet.
   *
   * @param {any} episode
   * @returns {{updated:number, avgLoss:number}}
   */
  learnFromEpisode(episode) {
    const trans = episode?.transitions;
    if (!Array.isArray(trans) || trans.length === 0) return { updated: 0, avgLoss: 0 };

    const learningPlayer = /** @type {1|2} */ (episode.learningPlayer ?? 1);
    let updated = 0;
    let lossSum = 0;

    for (const tr of trans) {
      // Only support tuple actions here (the new pipeline)
      const a = tr?.action;
      if (!a || typeof a !== 'object') continue;

      const r = Number(tr?.reward ?? 0);
      if (r === 0) continue;
      const sign = r > 0 ? 1 : -1;

      const feat = this.buildFeatures(tr?.obs, learningPlayer);
      const out = this.updateImitation(feat, a, sign);
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
    });
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
    for (const head of [this.Wx, this.Wy, this.Wp]) {
      for (const w of head) {
        for (let j = 0; j < w.length; j++) w[j] += randn() * s;
      }
    }
    for (let i = 0; i < this.bx.length; i++) this.bx[i] += randn() * s;
    for (let i = 0; i < this.by.length; i++) this.by[i] += randn() * s;
    for (let i = 0; i < this.bp.length; i++) this.bp[i] += randn() * s;
  }

  saveState() {
    return {
      kind: 'tuple_policy_v1',
      featureLen: this.featureLen,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      initStd: this.initStd,
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
    if (!state || state.kind !== 'tuple_policy_v1') return;
    this.featureLen = Math.max(1, state.featureLen | 0);
    this.learningRate = Number(state.learningRate ?? this.learningRate);
    this.epsilon = Number(state.epsilon ?? this.epsilon);
    this.initStd = Number(state.initStd ?? this.initStd);

    const toF32 = (arr) => new Float32Array(Array.isArray(arr) ? arr.map(Number) : []);
    const toHead = (head, nClass) => {
      const out = [];
      for (let a = 0; a < nClass; a++) {
        const row = head?.[a] ?? [];
        const w = toF32(row);
        // ensure length
        const ww = new Float32Array(this.featureLen);
        ww.set(w.subarray(0, this.featureLen));
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
  }
}
