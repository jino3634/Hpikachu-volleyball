// episode_runner.js
'use strict';

import { EpisodeBuilder } from './rl_episode_builder.js';

// ActionId convention:
// 0 idle
// 1 left, 2 right, 3 jump, 4 jump_left, 5 jump_right
// 6.. : power family
function isPowerAction(actionId) {
  return actionId >= 6;
}

// ---------- Ring buffer ----------
class TraceBuffer {
  constructor(capacity) {
    this.capacity = Math.max(1, capacity | 0);
    this.arr = new Array(this.capacity);
    this.size = 0;
    this.head = 0;
  }

  push(item) {
    this.arr[this.head] = item;
    this.head = (this.head + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  toArray() {
    const out = [];
    if (this.size === 0) return out;
    const start = (this.head - this.size + this.capacity) % this.capacity;
    for (let i = 0; i < this.size; i++) {
      out.push(this.arr[(start + i) % this.capacity]);
    }
    return out;
  }

  last() {
    if (this.size === 0) return null;
    const idx = (this.head - 1 + this.capacity) % this.capacity;
    return this.arr[idx];
  }
}

// ---------- Lose reason heuristics ----------
function classifyLoseReason(traceArr, scoredBy) {
  const loser = scoredBy === 1 ? 2 : 1;

  const tail = traceArr.slice(Math.max(0, traceArr.length - 20));
  const last = traceArr.length ? traceArr[traceArr.length - 1] : null;
  if (!last) {
    return { loser, loseReason: 'UNKNOWN' };
  }

  const me = last.obs?.[`p${loser}`]?.me;
  const ball = last.obs?.ball;

  if (me && ball) {
    const dist = Math.abs((me.x ?? 0) - (ball.expectedX ?? (ball.x ?? 0)));
    if (dist > 60 && (me.state === 0 || me.state === 1 || me.state === 2)) {
      return { loser, loseReason: 'MISPOSITION' };
    }
  }

  let powerCount = 0;
  for (const t of tail) {
    const a = loser === 1 ? t.actionP1 : t.actionP2;
    if (isPowerAction(a)) powerCount++;
  }
  if (powerCount >= 6) {
    return { loser, loseReason: 'BAD_POWER_SPAM' };
  }

  if (ball) {
    const nearNet = Math.abs((ball.x ?? 0) - 216) < 40;
    const low = (ball.y ?? 999) > 160;
    if (nearNet && low) {
      return { loser, loseReason: 'NET_TRAP' };
    }
  }

  return { loser, loseReason: 'OTHER' };
}

// ---------- Runner ----------
const DEFAULT_MAX_FRAMES = 60 * 60;
const DEFAULT_TRACE_LEN = 180;
const DEFAULT_WARMUP_FRAMES = 2000;

export class OnePointEpisodeRunner {
  /**
   * @param {import('../pikavolley.js').PikachuVolleyball} game
   * @param {{
   *   traceLen?: number,
   *   maxFrames?: number,
   *   warmupFrames?: number,
   *   learningPlayer?: 1|2
   * }} [opts]
   */
  constructor(game, opts = {}) {
    this.game = game;
    this.traceLen = (opts.traceLen ?? DEFAULT_TRACE_LEN) | 0;
    this.maxFrames = (opts.maxFrames ?? DEFAULT_MAX_FRAMES) | 0;
    this.warmupFrames = (opts.warmupFrames ?? DEFAULT_WARMUP_FRAMES) | 0;
    /** @type {1|2} */
    this.learningPlayer = (opts.learningPlayer ?? 1);

    this.trace = new TraceBuffer(this.traceLen);
  }

  warmupToPlayable() {
    if (typeof this.game.setControlMode === 'function') {
      this.game.setControlMode('external');
    } else {
      this.game.controlMode = 'external';
    }

    for (let i = 0; i < this.warmupFrames; i++) {
      this.game.stepLogic();
      const s = this.game.state;
      if (
        s === this.game.round ||
        s === this.game.afterEndOfRound ||
        s === this.game.beforeStartOfNextRound
      ) {
        return true;
      }
    }
    return false;
  }

  /**
   * @returns {{
   *   ok: boolean,
   *   scoredBy: 0|1|2,
   *   loser: 0|1|2,
   *   loseReason: string,
   *   frames: number,
   *   trace: any[],
   *   last: any,
   *   episode: import('./rl_schema.js').Episode
   * }}
   */
  runOnePoint() {
    this.trace = new TraceBuffer(this.traceLen);

    this.warmupToPlayable();

    const builder = new EpisodeBuilder({ learningPlayer: this.learningPlayer });

    let frames = 0;
    /** @type {0|1|2} */
    let scoredBy = 0;

    while (frames < this.maxFrames) {
      const obs1 = this.game.getObservation ? this.game.getObservation(1) : null;
      const obs2 = this.game.getObservation ? this.game.getObservation(2) : null;

      const actionP1 = (this.game._heldActionP1 ?? 0) | 0;
      const actionP2 = (this.game._heldActionP2 ?? 0) | 0;

      this.trace.push({
        frame: frames,
        actionP1,
        actionP2,
        obs: {
          p1: obs1,
          p2: obs2,
          ball: obs1?.ball ?? obs2?.ball ?? null,
        },
        stateName: this._guessStateName(),
      });

      // step
      this.game.stepLogic();
      frames++;

      const nextObs1 = this.game.getObservation ? this.game.getObservation(1) : null;
      const nextObs2 = this.game.getObservation ? this.game.getObservation(2) : null;

      const ev = this.game._lastRoundEvents;

      // learning player의 transition만 기록
      const myObs = (this.learningPlayer === 1) ? obs1 : obs2;
      const myNextObs = (this.learningPlayer === 1) ? nextObs1 : nextObs2;
      const myAction = (this.learningPlayer === 1) ? actionP1 : actionP2;

      const done = !!(ev && (ev.scored === 1 || ev.scored === 2));

      builder.addStep({
        t: frames,
        obs: myObs,
        action: myAction,
        nextObs: myNextObs,
        done,
        info: { state: this._guessStateName() },
        roundEvents: ev ?? null,
      });

      if (done) {
        scoredBy = /** @type {0|1|2} */ (ev.scored);
        break;
      }

      // fallback
      if (this.game.roundEnded === true && this.game.state === this.game.afterEndOfRound) {
        const punchX = this.game.physics?.ball?.punchEffectX;
        if (typeof punchX === 'number') {
          scoredBy = punchX < 216 ? 2 : 1;
        } else {
          scoredBy = 0;
        }
        break;
      }
    }

    const traceArr = this.trace.toArray();
    const last = this.trace.last();

    if (scoredBy === 0) {
      builder.finalize({ scoredBy: 0, loser: 0, loseReason: 'TIMEOUT' });
      return {
        ok: false,
        scoredBy: 0,
        loser: 0,
        loseReason: 'TIMEOUT',
        frames,
        trace: traceArr,
        last,
        episode: builder.toEpisode(),
      };
    }

    const { loser, loseReason } = classifyLoseReason(traceArr, scoredBy);
    builder.finalize({ scoredBy, loser, loseReason, frames });

    return {
      ok: true,
      scoredBy,
      loser,
      loseReason,
      frames,
      trace: traceArr,
      last,
      episode: builder.toEpisode(),
    };
  }

  _guessStateName() {
    const s = this.game.state;
    if (s === this.game.intro) return 'intro';
    if (s === this.game.menu) return 'menu';
    if (s === this.game.afterMenuSelection) return 'afterMenuSelection';
    if (s === this.game.beforeStartOfNewGame) return 'beforeStartOfNewGame';
    if (s === this.game.startOfNewGame) return 'startOfNewGame';
    if (s === this.game.round) return 'round';
    if (s === this.game.afterEndOfRound) return 'afterEndOfRound';
    if (s === this.game.beforeStartOfNextRound) return 'beforeStartOfNextRound';
    return 'unknown';
  }
}
