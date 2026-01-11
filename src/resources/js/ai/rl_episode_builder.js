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
   *   action: number,
   *   nextObs: any,
   *   done: boolean,
   *   info?: any,
   *   roundEvents?: any
   * }} param0
   */
  addStep({ t, obs, action, nextObs, done, info = null, roundEvents = null }) {
    const reward = computeSparseReward(roundEvents, this.learningPlayer);

    this.transitions.push({
      t,
      obs: obs ?? null,
      action: action | 0,
      reward,
      nextObs: nextObs ?? null,
      done: !!done,
      info: info ?? null,
    });

    this.frames = this.transitions.length;
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
