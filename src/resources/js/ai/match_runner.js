// match_runner.js
'use strict';

import { OnePointEpisodeRunner } from './episode_runner.js';

/**
 * MatchRunner
 * - P1: external agent (your AI)
 * - P2: builtin AI (physics player2.isComputer = true)
 * - 15 points = 1 set
 * - If P1 wins 3 sets in a row => graduate
 */
export class MatchRunner {
  /**
   * @param {import('../pikavolley.js').PikachuVolleyball} game
   * @param {object} agentP1 - must have chooseAction(obs, playerIndex, game)
   * @param {{
   *   winningScore?: number,
   *   requiredWinStreak?: number,
   *   maxSets?: number,
   *   pointMaxFrames?: number,
   *   pointWarmupFrames?: number,
   *   pointTraceLen?: number,
   *   verbose?: boolean
   * }} [opts]
   */
  constructor(game, agentP1, opts = {}) {
    this.game = game;
    this.agentP1 = agentP1;

    this.winningScore = (opts.winningScore ?? 15) | 0;
    this.requiredWinStreak = (opts.requiredWinStreak ?? 3) | 0;
    this.maxSets = (opts.maxSets ?? 50) | 0;

    this.verbose = opts.verbose ?? true;

    // point runner options
    this.pointMaxFrames = (opts.pointMaxFrames ?? 60 * 60) | 0;
    this.pointWarmupFrames = (opts.pointWarmupFrames ?? 2000) | 0;
    this.pointTraceLen = (opts.pointTraceLen ?? 180) | 0;

    this._pointRunner = new OnePointEpisodeRunner(this.game, {
      maxFrames: this.pointMaxFrames,
      warmupFrames: this.pointWarmupFrames,
      traceLen: this.pointTraceLen,
    });

    // stats
    this.setsPlayed = 0;
    this.p1SetWins = 0;
    this.p2SetWins = 0;
    this.p1WinStreak = 0;
    this.graduated = false;

    this.history = []; // set summaries
  }

  // ------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------

  /**
   * Run sets until:
   * - P1 gets requiredWinStreak (graduate), OR
   * - reaches maxSets (stop)
   *
   * @returns {{
   *   ok: boolean,
   *   graduated: boolean,
   *   setsPlayed: number,
   *   p1SetWins: number,
   *   p2SetWins: number,
   *   p1WinStreak: number,
   *   history: any[]
   * }}
   */
  runUntilGraduate() {
    this._resetMatchStats();

    for (let setNo = 1; setNo <= this.maxSets; setNo++) {
      const setResult = this.runOneSet(setNo);
      this.history.push(setResult);

      if (setResult.winner === 1) {
        this.p1SetWins++;
        this.p1WinStreak++;
      } else if (setResult.winner === 2) {
        this.p2SetWins++;
        this.p1WinStreak = 0;
      } else {
        // unknown/timeout -> streak reset (너가 원하는 방향이면 이렇게 두는 게 안전)
        this.p1WinStreak = 0;
      }

      this.setsPlayed++;

      if (this.verbose) {
        this._logSetSummary(setResult);
        console.log(`Streak: P1 ${this.p1WinStreak} / ${this.requiredWinStreak}`);
        console.log('--------------------------------------------------');
      }

      if (this.p1WinStreak >= this.requiredWinStreak) {
        this.graduated = true;
        break;
      }
    }

    return {
      ok: true,
      graduated: this.graduated,
      setsPlayed: this.setsPlayed,
      p1SetWins: this.p1SetWins,
      p2SetWins: this.p2SetWins,
      p1WinStreak: this.p1WinStreak,
      history: this.history,
    };
  }

  /**
   * Run exactly ONE set to winningScore.
   * - auto restart game
   * - force external vs builtin
   * - keep playing points until someone reaches winningScore
   *
   * @param {number} setNo
   * @returns {{
   *   setNo: number,
   *   winner: 0|1|2,
   *   finalScore: [number, number],
   *   points: number,
   *   pointResults: any[],
   *   notes: string[]
   * }}
   */
  runOneSet(setNo = 1) {
    const notes = [];

    // 1) Restart / reset flow
    this._hardResetToIntro();

    // 2) Warmup into round-like state (menu -> start game -> round)
    const okPlayable = this._warmupToRoundLike();
    if (!okPlayable) {
      notes.push('WARMUP_FAILED: could not reach round-like state');
    }

    // 3) Force P1 external vs P2 builtin + attach agent
    this._applyExternalVsBuiltin();

    // 4) Set winning score (15) + practice mode off
    this.game.winningScore = this.winningScore;
    this.game.isPracticeMode = false;

    // 5) Loop points until set ends
    const pointResults = [];
    let points = 0;

    // 안전장치: 세트당 너무 오래 걸리면 중단 (원하면 늘려)
    //const maxPointsSafety = this.winningScore * 5;
    const maxPointsSafety = 1e9; // 사실상 무제한

    while (points < maxPointsSafety) {
      points++;

      // run one point
      const pr = this._pointRunner.runOnePoint();
      pointResults.push(pr);

      // set ended?
      const s1 = this.game.scores?.[0] ?? 0;
      const s2 = this.game.scores?.[1] ?? 0;

      if (this.verbose) {
        const who = pr.ok ? `scoredBy=${pr.scoredBy}` : `timeout`;
        console.log(`[Set ${setNo}] Point ${points}: ${who} | score ${s1}-${s2} | loseReason=${pr.loseReason}`);
      }

      if (s1 >= this.winningScore || s2 >= this.winningScore) {
        break;
      }

      // 만약 point가 timeout으로 계속 끊기면, 그냥 진행되게 둠(실수하면서 배우기)
      // 필요하면 여기서 "timeout 연속" 감지해서 강제 리셋도 가능.
    }

    const finalScore = /** @type {[number,number]} */ ([
      this.game.scores?.[0] ?? 0,
      this.game.scores?.[1] ?? 0,
    ]);

    /** @type {0|1|2} */
    let winner = 0;
    if (finalScore[0] >= this.winningScore) winner = 1;
    else if (finalScore[1] >= this.winningScore) winner = 2;
    else {
      notes.push('SET_TIMEOUT_OR_STUCK: no one reached winningScore');
      winner = 0;
    }

    return {
      setNo,
      winner,
      finalScore,
      points,
      pointResults,
      notes,
    };
  }

  // ------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------

  _resetMatchStats() {
    this.setsPlayed = 0;
    this.p1SetWins = 0;
    this.p2SetWins = 0;
    this.p1WinStreak = 0;
    this.graduated = false;
    this.history = [];
  }

  _logSetSummary(setResult) {
    const w = setResult.winner === 1 ? 'P1(external) WIN'
            : setResult.winner === 2 ? 'P2(builtin) WIN'
            : 'UNKNOWN';
    console.log(`SET ${setResult.setNo} RESULT: ${w} | Final ${setResult.finalScore[0]}-${setResult.finalScore[1]} | points=${setResult.points}`);
    if (setResult.notes?.length) {
      console.log(`Notes: ${setResult.notes.join(' | ')}`);
    }
  }

  _hardResetToIntro() {
    // game.restart() already sets state=intro and hides views.
    if (typeof this.game.restart === 'function') {
      this.game.restart();
    } else {
      // fallback
      this.game.frameCounter = 0;
      this.game.noInputFrameCounter = 0;
      this.game.slowMotionFramesLeft = 0;
      this.game.slowMotionNumOfSkippedFrames = 0;
      this.game.state = this.game.intro;
    }

    // Make sure no overrides remain
    try {
      this.game.keyboardArray?.[0]?.clearOverrideInput();
      this.game.keyboardArray?.[1]?.clearOverrideInput();
    } catch (_) {}

    // detach agents temporarily during warmup (optional)
    // (we’ll reattach after reaching round-like)
    this.game.agent1 = null;
    this.game.agent2 = null;

    // default: let menu auto-progress
    this.game.setControlMode?.('builtin');
  }

  _warmupToRoundLike() {
    // Use point runner warmup, but keep it simple:
    // stepLogic 반복해서 round-like state 도달만 하면 OK
    const warmupN = this.pointWarmupFrames;

    for (let i = 0; i < warmupN; i++) {
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

  _applyExternalVsBuiltin() {
    // 핵심: P1 external / P2 builtin
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(true, false);
    } else {
      // fallback (older code)
      this.game.setControlMode?.('external');
      this.game.externalEnabledP1 = true;
      this.game.externalEnabledP2 = false;
      this.game.physics.player1.isComputer = false;
      this.game.physics.player2.isComputer = true;
      this.game.keyboardArray?.[1]?.clearOverrideInput();
    }

    // Attach agent only to P1
    if (typeof this.game.setAgents === 'function') {
      this.game.setAgents(this.agentP1, null);
    } else {
      this.game.agent1 = this.agentP1;
      this.game.agent2 = null;
    }

    // Optional: make sure decision interval is what you want
    // (you can tune later; keep as current)
    // this.game.decisionInterval = 2;
  }
}
