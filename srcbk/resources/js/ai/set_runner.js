// set_runner.js
'use strict';

/**
 * Run ONE set (to winningScore=15 by default) using your existing game loop.
 *
 * Assumptions about your game:
 * - game.stepLogic() advances one logical frame
 * - game._lastRoundEvents is updated each round() call (you already set it)
 * - game.scores exists: [p1Score, p2Score]
 * - game.winningScore exists: number (15)
 * - game.gameEnded becomes true when someone reaches winningScore
 * - game.physics.player1.isWinner / player2.isWinner set when game ends
 */

const DEFAULT_MAX_FRAMES = 60 * 60 * 10; // safety: ~10 minutes worth of logical frames

function inferWinnerFromScores(scores) {
  if (!scores || scores.length < 2) return 0;
  if (scores[0] > scores[1]) return 1;
  if (scores[1] > scores[0]) return 2;
  return 0;
}

export class OneSetRunner {
  /**
   * @param {import('../pikavolley.js').PikachuVolleyball} game
   * @param {{
   *   maxFrames?: number,
   *   warmupFrames?: number,
   *   traceLen?: number
   * }} [opts]
   */
  constructor(game, opts = {}) {
    this.game = game;
    this.maxFrames = (opts.maxFrames ?? DEFAULT_MAX_FRAMES) | 0;
    this.warmupFrames = (opts.warmupFrames ?? 3000) | 0;

    // optional trace
    this.traceLen = (opts.traceLen ?? 0) | 0;
    this._trace = this.traceLen > 0 ? new TraceBuffer(this.traceLen) : null;
  }

  /**
   * Put game into playable-ish state.
   * We do NOT “force menu clicks”; we just run frames until round-like.
   */
  warmupToRoundLike() {
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
   * Run until the set ends (someone reaches winningScore).
   *
   * @returns {{
   *   ok: boolean,
   *   winner: 0|1|2,
   *   finalScores: [number, number],
   *   frames: number,
   *   trace?: any[]
   * }}
   */
  runOneSet() {
    // optional: reset trace
    if (this._trace) this._trace.clear();

    // warmup into playable
    this.warmupToRoundLike();

    let frames = 0;

    while (frames < this.maxFrames) {
      // optional trace snapshot
      if (this._trace) {
        const obs1 = this.game.getObservation ? this.game.getObservation(1) : null;
        const obs2 = this.game.getObservation ? this.game.getObservation(2) : null;
        this._trace.push({
          frame: frames,
          scores: [this.game.scores?.[0] ?? 0, this.game.scores?.[1] ?? 0],
          actionP1: (this.game._heldActionP1 ?? 0) | 0,
          actionP2: (this.game._heldActionP2 ?? 0) | 0,
          obs: { p1: obs1, p2: obs2, ball: obs1?.ball ?? obs2?.ball ?? null },
          stateName: guessStateName(this.game),
          lastRoundEvents: this.game._lastRoundEvents ?? null,
        });
      }

      this.game.stepLogic();
      frames++;

      // 세트 종료 감지: 점수 도달 시점은 roundLogic 내부에서 gameEnded=true로 세팅됨
      const p1 = this.game.scores?.[0] ?? 0;
      const p2 = this.game.scores?.[1] ?? 0;
      const ws = this.game.winningScore ?? 15;

      if (p1 >= ws || p2 >= ws || this.game.gameEnded === true) {
        // winner 판정
        /** @type {0|1|2} */
        let winner = 0;

        // 가능하면 physics winner 플래그를 우선
        const isP1Winner = !!this.game.physics?.player1?.isWinner;
        const isP2Winner = !!this.game.physics?.player2?.isWinner;
        if (isP1Winner && !isP2Winner) winner = 1;
        else if (isP2Winner && !isP1Winner) winner = 2;
        else winner = /** @type {0|1|2} */ (inferWinnerFromScores([p1, p2]) || 0);

        return {
          ok: true,
          winner,
          finalScores: [p1, p2],
          frames,
          ...(this._trace ? { trace: this._trace.toArray() } : {}),
        };
      }
    }

    return {
      ok: false,
      winner: 0,
      finalScores: [this.game.scores?.[0] ?? 0, this.game.scores?.[1] ?? 0],
      frames,
      ...(this._trace ? { trace: this._trace.toArray() } : {}),
    };
  }
}

/* ---------------- helpers ---------------- */

function guessStateName(game) {
  const s = game.state;
  if (s === game.intro) return 'intro';
  if (s === game.menu) return 'menu';
  if (s === game.afterMenuSelection) return 'afterMenuSelection';
  if (s === game.beforeStartOfNewGame) return 'beforeStartOfNewGame';
  if (s === game.startOfNewGame) return 'startOfNewGame';
  if (s === game.round) return 'round';
  if (s === game.afterEndOfRound) return 'afterEndOfRound';
  if (s === game.beforeStartOfNextRound) return 'beforeStartOfNextRound';
  return 'unknown';
}

class TraceBuffer {
  constructor(capacity) {
    this.capacity = Math.max(1, capacity | 0);
    this.arr = new Array(this.capacity);
    this.size = 0;
    this.head = 0;
  }
  clear() {
    this.size = 0;
    this.head = 0;
    this.arr = new Array(this.capacity);
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
}
