// policy_v1.js
'use strict';

/**
 * 아주 단순한 정책:
 * - logits[a] = dot(W[a], feat) + b[a]
 * - action ~ softmax(logits) (탐험 epsilon)
 * - 에피소드 끝 reward(+1/-1)를 전체 step에 동일하게 REINFORCE로 업데이트
 *
 * "최고 AI"로 가려면 나중에:
 * - 더 좋은 feature / 더 많은 액션 / 커리큘럼 / self-play / value baseline 등 추가
 */

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum || 1;
  return exps;
}

function sampleCategorical(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function clampInt(x, lo, hi) {
  x |= 0;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

export class PolicyV1 {
  /**
   * @param {{
   *   numActions?: number,
   *   learningRate?: number,
   *   epsilon?: number,
   *   initStd?: number,
   * }} [opts]
   */
  constructor(opts = {}) {
    this.numActions = (opts.numActions ?? 10) | 0;
    this.learningRate = Number(opts.learningRate ?? 0.0005);
    this.epsilon = Number(opts.epsilon ?? 0.08);
    this.initStd = Number(opts.initStd ?? 0.01);

    // feature length: buildFeatures() 기준
    this.featureLen = 1 +  // bias
      3 + // me: x,y,yV
      3 + // opp: x,y,yV
      5 + // ball: x,y,xV,yV,expectedX
      2;  // serve,isPower

    this.W = []; // [A][F]
    this.b = []; // [A]
    this._initParams();
  }

  _initParams() {
    this.W = new Array(this.numActions);
    this.b = new Array(this.numActions);
    for (let a = 0; a < this.numActions; a++) {
      const row = new Float32Array(this.featureLen);
      for (let j = 0; j < this.featureLen; j++) row[j] = randn() * this.initStd;
      this.W[a] = row;
      this.b[a] = 0;
    }
  }

  /**
   * checkpoint로부터 로드
   * @param {any} state
   */
  loadState(state) {
    if (!state || state.kind !== 'policy_v1') return false;
    if (!state.W || !state.b) return false;
    if ((state.numActions | 0) !== this.numActions) return false;
    if ((state.featureLen | 0) !== this.featureLen) return false;

    // restore
    this.b = state.b.slice();
    this.W = state.W.map(arr => Float32Array.from(arr));
    return true;
  }

  /**
   * checkpoint로 저장
   */
  saveState() {
    return {
      kind: 'policy_v1',
      updatedAt: Date.now(),
      numActions: this.numActions,
      featureLen: this.featureLen,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      W: this.W.map(row => Array.from(row)),
      b: this.b.slice(),
    };
  }

  /**
   * @param {any} obs (game.getObservation() 반환)
   * @param {1|2} playerIndex
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

    return feat;
  }

  /**
   * @param {Float32Array} feat
   */
  _logits(feat) {
    const logits = new Float32Array(this.numActions);
    for (let a = 0; a < this.numActions; a++) {
      let s = this.b[a];
      const Wa = this.W[a];
      for (let j = 0; j < this.featureLen; j++) s += Wa[j] * feat[j];
      logits[a] = s;
    }
    return logits;
  }

  /**
   * 행동 선택 (agent.chooseAction에서 호출)
   * @param {any} obs
   * @param {1|2} playerIndex
   */
  act(obs, playerIndex) {
    // epsilon-greedy (softmax 샘플링 + 가끔 랜덤)
    if (Math.random() < this.epsilon) {
      return (Math.random() * this.numActions) | 0;
    }
    const feat = this.buildFeatures(obs, playerIndex);
    const logits = this._logits(feat);
    const probs = softmax(Array.from(logits));
    return sampleCategorical(probs);
  }

  /**
   * REINFORCE 업데이트 (baseline 없음)
   * @param {import('./rl_schema.js').Episode} episode
   */
  learnFromEpisode(episode) {
    if (!episode || !Array.isArray(episode.transitions)) return;

    // 최종 보상: 마지막 transition의 reward (우리가 builder에서 done에서만 reward 부여)
    let finalR = 0;
    for (let i = episode.transitions.length - 1; i >= 0; i--) {
      const r = episode.transitions[i].reward;
      if (typeof r === 'number' && r !== 0) { finalR = r; break; }
    }
    if (finalR === 0) return; // 무승부/판정불가면 학습 스킵

    const lr = this.learningRate;
    const playerIndex = episode.learningPlayer;

    // 각 step에 대해: grad = (onehot(a)-prob) * feat * finalR
    for (const tr of episode.transitions) {
      const obs = tr.obs;
      const action = clampInt(tr.action, 0, this.numActions - 1);

      const feat = this.buildFeatures(obs, playerIndex);
      const logits = this._logits(feat);
      const probs = softmax(Array.from(logits));

      for (let a = 0; a < this.numActions; a++) {
        const coeff = ( (a === action ? 1 : 0) - probs[a] ) * finalR * lr;
        const Wa = this.W[a];
        for (let j = 0; j < this.featureLen; j++) {
          Wa[j] += coeff * feat[j];
        }
        this.b[a] += coeff;
      }
    }
  }
}

/**
 * 게임에 붙일 agent wrapper
 */
export class PolicyAgentV1 {
  /**
   * @param {PolicyV1} policy
   * @param {{playerIndex:1|2}} opts
   */
  constructor(policy, opts) {
    this.policy = policy;
    this.playerIndex = opts.playerIndex;
  }

  chooseAction(obs, playerIndex /*ignored*/, game /*ignored*/) {
    return this.policy.act(obs, this.playerIndex) | 0;
  }
}
