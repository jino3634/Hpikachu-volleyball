// rl_episode_builder.js
'use strict';

import { SCHEMA_VERSION } from './rl_schema.js';
import { computeSparseReward } from './rl_reward.js';

/** @returns {string} */
function makeEpisodeId() {
  // 충돌 가능성 낮게
  return `ep_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/**
 * EpisodeBuilder
 * - runner 루프에서 매 프레임 snapshot을 넣고,
 * - 점수 발생(done) 시 episode로 finalize
 */
export class EpisodeBuilder {
  /**
   * @param {{learningPlayer?: 1|2}} [opts]
   */
  constructor(opts = {}) {
    /** @type {1|2} */
    this.learningPlayer = (opts.learningPlayer ?? 1);

    this.reset();
  }

  reset() {
    this.episodeId = makeEpisodeId();
    this.startedAt = Date.now();
    this.endedAt = 0;

    /** @type {any[]} */
    this.transitions = [];
    this.frames = 0;

    /** @type {0|1|2} */
    this.scoredBy = 0;
    /** @type {0|1|2} */
    this.loser = 0;
    /** @type {string} */
    this.loseReason = '';
  }

  /**
   * Transition을 1개 추가한다.
   *
   * @param {{
   *   t: number,
   *   obs: any,
 *   feat?: Float32Array|null,
 *   action: any,
 *   nextObs: any,
 *   nextFeat?: Float32Array|null,
   *   done: boolean,
   *   info?: any,
   *   roundEvents?: any
   * }} step
   */
  addStep(step) {
    // step is required; silently ignore null/undefined to keep training loop robust
    if (!step) return;
    // NOTE:
    // 런타임에서 addStep(undefined)가 들어오면 번들 에러로 학습이 전부 중단될 수 있다.
    // 파라미터를 안전하게 디폴트 처리해서 학습 루프를 계속 진행하게 한다.
    const {
      t = 0,
      obs = null,
      feat = null,
      action = 0,
      nextObs = null,
      nextFeat = null,
      done = false,
      info = null,
      roundEvents = null,
    } = step;

    const reward = computeSparseReward(roundEvents, this.learningPlayer);

    // Terminal detection (supports both {scoredBy} and {scored}).
    const scoredByNow = (roundEvents && typeof roundEvents.scoredBy === 'number')
      ? roundEvents.scoredBy
      : ((roundEvents && typeof roundEvents.scored === 'number') ? roundEvents.scored : 0);
    const isTerminalEvent = (scoredByNow === 1 || scoredByNow === 2);

    // action can be a number (legacy actionId) or an input tuple object
    const storedAction = (typeof action === 'number') ? (action | 0) : action;

    this.transitions.push({
      t,
      obs: obs ?? null,
      feat: feat ?? null,
      action: storedAction,
      reward,
      nextObs: nextObs ?? null,
      nextFeat: nextFeat ?? null,
      done: !!done,
      info: info ?? null,
      // Internal marker: used to avoid double-counting terminal reward when
      // the score happens exactly on a decision frame.
      _terminalEvent: isTerminalEvent,
    });

    this.frames = this.transitions.length;
  }

  /**
   * Ensure the episode has a proper terminal transition.
   * - Guarantee the last transition has done=true.
   * - If the point ended on a non-decision frame, add terminal reward to the last transition.
   *   (Avoid double-counting when the last transition already saw terminal event.)
   * @param {any} terminalRoundEvents
   */
  applyTerminal(terminalRoundEvents) {
    if (!this.transitions || this.transitions.length <= 0) return;
    const last = this.transitions[this.transitions.length - 1];
    if (!last) return;

    // Always mark done on the last transition.
    last.done = true;

    // If the last transition did not already observe the terminal event,
    // inject terminal reward so PPO sees it.
    if (!last._terminalEvent) {
      const tr = computeSparseReward(terminalRoundEvents, this.learningPlayer);
      if (Number.isFinite(tr) && tr !== 0) last.reward = Number(last.reward ?? 0) + tr;
      last._terminalEvent = true;
    }
  }

  /**
   * runner 결과를 반영해 episode 메타를 확정한다.
   * @param {{
   *   scoredBy: 0|1|2,
   *   loser: 0|1|2,
   *   loseReason: string
   * }} resultMeta
   */
  finalize(resultMeta) {
    this.endedAt = Date.now();
    this.scoredBy = resultMeta.scoredBy;
    this.loser = resultMeta.loser;
    this.loseReason = resultMeta.loseReason || '';
  }

  /**
   * @returns {import('./rl_schema.js').Episode}
   */
  toEpisode() {

    // === reward redistribution over all frames (win/loss credit assignment) ===
    if (this.scoredBy !== null && this.scoredBy !== undefined) {
      // PPO core uses terminal sparse reward; keep per-step rewards as-is (no redistribution)
    }

    // Strip internal fields that are not part of the public schema.
    // (Keep in-memory markers during building, but don't persist them.)
    for (const tr of this.transitions) {
      if (tr && typeof tr === 'object' && ('_terminalEvent' in tr)) {
        try { delete tr._terminalEvent; } catch (_) { /* ignore */ }
      }
    }

    return {
      schemaVersion: SCHEMA_VERSION,
      episodeId: this.episodeId,
      startedAt: this.startedAt,
      endedAt: this.endedAt || Date.now(),
      frames: this.frames,
      learningPlayer: this.learningPlayer,
      scoredBy: this.scoredBy,
      loser: this.loser,
      loseReason: this.loseReason,
      transitions: this.transitions,
    };
  }
}
