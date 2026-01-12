// train_loop.js
'use strict';

import { MatchRunner } from './match_runner.js';
import { HeuristicAgentV1 } from './agents.js';

/**
 * Browser Training Loop (requestAnimationFrame)
 * - Fast training: run many stepLogic() per RAF tick
 * - Watch mode: run 1 stepLogic() per RAF tick (or a small number)
 *
 * Assumptions:
 * - You already created `game = new PikachuVolleyball(stage, resources)` somewhere.
 * - That creation file can call `startTrainingLoop(game)` at the end.
 *
 * This module only needs the `game` instance.
 */

// -----------------------------
// Config defaults
// -----------------------------
const DEFAULTS = {
  // training speed
  stepsPerFrameTrain: 400,   // 50~500 recommended, can go 1000+
  stepsPerFrameWatch: 1,     // watch/play speed

  // match rules
  winningScore: 15,
  requiredWinStreak: 3,
  maxSets: 30,

  // runner safety
  pointMaxFrames: 60 * 60,
  pointWarmupFrames: 2000,
  pointTraceLen: 180,

  // logs
  verbose: true,
};

// -----------------------------
// Public entry
// -----------------------------
export function startTrainingLoop(game, opts = {}) {
  const cfg = { ...DEFAULTS, ...opts };

  // External agent for P1
  const agent = new HeuristicAgentV1({ playerIndex: 1 });

  // Match runner (set loop)
  const mr = new MatchRunner(game, agent, {
    winningScore: cfg.winningScore,
    requiredWinStreak: cfg.requiredWinStreak,
    maxSets: cfg.maxSets,
    pointMaxFrames: cfg.pointMaxFrames,
    pointWarmupFrames: cfg.pointWarmupFrames,
    pointTraceLen: cfg.pointTraceLen,
    verbose: cfg.verbose,
  });

  // Loop state
  const state = {
    mode: 'train', // 'train' | 'watch' | 'paused'
    stepsPerFrameTrain: cfg.stepsPerFrameTrain | 0,
    stepsPerFrameWatch: cfg.stepsPerFrameWatch | 0,

    // running match state
    running: true,
    lastResult: null,

    // If you want repeated “graduate cycles”
    autoRestartAfterGraduate: true,

    // FPS control (optional cap)
    lastRAFTime: 0,
    minMsPerFrame: 0, // 0 = no cap, set 16 for ~60fps cap
  };

  // Optional: You can expose it globally for debugging
  // window.__train = { mr, state, game };

  // Controls (keyboard)
  attachHotkeys(state);

  // Start first run
  console.log('[TRAIN] starting runUntilGraduate() in accelerated loop');
  // We do NOT call mr.runUntilGraduate() directly because it blocks.
  // Instead, we run incrementally by stepping points/sets ourselves.
  // For simplest integration with your current runner, we will run one set at a time per “macro step”.
  const macro = makeMacroStepper(mr, state);

  function rafLoop(ts) {
    if (!state.running) return;

    if (state.minMsPerFrame > 0) {
      if (ts - state.lastRAFTime < state.minMsPerFrame) {
        requestAnimationFrame(rafLoop);
        return;
      }
      state.lastRAFTime = ts;
    }

    if (state.mode === 'paused') {
      requestAnimationFrame(rafLoop);
      return;
    }

    const steps =
      state.mode === 'train'
        ? Math.max(1, state.stepsPerFrameTrain | 0)
        : Math.max(1, state.stepsPerFrameWatch | 0);

    // In each RAF, do "steps" macro-steps (not stepLogic frames).
    // A macro-step = advance training progress (e.g., point/set progression).
    for (let i = 0; i < steps; i++) {
      const done = macro.stepOnce();
      if (done) break; // graduated or finished max sets
    }

    requestAnimationFrame(rafLoop);
  }

  requestAnimationFrame(rafLoop);

  return { mr, state };
}

// -----------------------------
// Macro-stepper: non-blocking runner
// -----------------------------
function makeMacroStepper(mr, state) {
  // We run set-by-set to avoid huge blocking loops.
  // Each macro step runs ONE POINT (via OnePointEpisodeRunner) to keep UI responsive.
  //
  // But your MatchRunner currently calls runOneSet() which internally calls runOnePoint() in a loop.
  // That also blocks for a while if called naively.
  //
  // So here we replicate a small “incremental set runner” using mr.game + mr._pointRunner.
  // This stays compatible with your existing OnePointEpisodeRunner + pikavolley hooks.

  const game = mr.game;

  // local set state
  const ctx = {
    inited: false,
    setNo: 0,

    // set progress
    currentSetWinner: /** @type {0|1|2} */ (0),
    setNotes: [],
    pointCount: 0,

    // match progress
    graduated: false,
    finished: false,
  };

  function beginNewSet() {
    ctx.setNo++;
    ctx.currentSetWinner = 0;
    ctx.setNotes = [];
    ctx.pointCount = 0;

    // hard reset to intro/menu flow
    if (typeof game.restart === 'function') game.restart();
    else game.state = game.intro;

    // warm up to round-like
    const ok = warmupToRoundLike(game, mr.pointWarmupFrames || 2000);
    if (!ok) ctx.setNotes.push('WARMUP_FAILED');

    // force P1 external vs P2 builtin + attach agent
    if (typeof game.setExternalVsBuiltin === 'function') {
      game.setExternalVsBuiltin(true, false);
    } else {
      // fallback
      game.setControlMode?.('external');
      game.physics.player1.isComputer = false;
      game.physics.player2.isComputer = true;
    }

    if (typeof game.setAgents === 'function') game.setAgents(mr.agentP1, null);
    else game.agent1 = mr.agentP1;

    // match rules
    game.winningScore = mr.winningScore;
    game.isPracticeMode = false;

    if (state.mode !== 'train') {
      console.log(`[MATCH] Begin SET ${ctx.setNo}`);
    }
  }

  // Initialize match stats
  function initMatch() {
    mr._resetMatchStats?.(); // if exists
    mr.setsPlayed = 0;
    mr.p1SetWins = 0;
    mr.p2SetWins = 0;
    mr.p1WinStreak = 0;
    mr.graduated = false;
    mr.history = [];

    ctx.setNo = 0;
    ctx.graduated = false;
    ctx.finished = false;
    ctx.inited = true;

    beginNewSet();
  }

  function endSetAndUpdateMatch() {
    const s1 = game.scores?.[0] ?? 0;
    const s2 = game.scores?.[1] ?? 0;

    /** @type {0|1|2} */
    let winner = 0;
    if (s1 >= mr.winningScore) winner = 1;
    else if (s2 >= mr.winningScore) winner = 2;

    ctx.currentSetWinner = winner;

    const setSummary = {
      setNo: ctx.setNo,
      winner,
      finalScore: /** @type {[number,number]} */ ([s1, s2]),
      points: ctx.pointCount,
      notes: ctx.setNotes.slice(),
    };
    mr.history.push(setSummary);

    mr.setsPlayed++;
    if (winner === 1) {
      mr.p1SetWins++;
      mr.p1WinStreak++;
    } else if (winner === 2) {
      mr.p2SetWins++;
      mr.p1WinStreak = 0;
    } else {
      mr.p1WinStreak = 0;
    }

    if (state.mode !== 'train') {
      console.log(
        `[MATCH] SET ${ctx.setNo} END | winner=${winner} | score ${s1}-${s2} | streak=${mr.p1WinStreak}/${mr.requiredWinStreak}`
      );
    }

    if (mr.p1WinStreak >= mr.requiredWinStreak) {
      mr.graduated = true;
      ctx.graduated = true;
      ctx.finished = true;
      console.log(`[GRADUATE] P1 won ${mr.requiredWinStreak} sets in a row. DONE.`);
      return;
    }

    if (ctx.setNo >= mr.maxSets) {
      ctx.finished = true;
      console.log(`[STOP] reached maxSets=${mr.maxSets}. DONE.`);
      return;
    }

    beginNewSet();
  }

  // One macro step = run one point (non-blocking-ish) and check set end
  function stepOnce() {
    if (!ctx.inited) initMatch();
    if (ctx.finished) {
      if (state.autoRestartAfterGraduate && ctx.graduated) {
        // restart fresh cycle automatically (optional)
        if (state.mode !== 'train') console.log('[TRAIN] auto-restarting new match cycle after graduation');
        initMatch();
      }
      return true; // done for now
    }

    // Run 1 point
    ctx.pointCount++;

    const pr = mr._pointRunner.runOnePoint();
    // logs (keep minimal in train mode)
    if (state.mode !== 'train') {
      console.log(
        `[Set ${ctx.setNo}] Point ${ctx.pointCount}: ok=${pr.ok} scoredBy=${pr.scoredBy} loser=${pr.loser} reason=${pr.loseReason}`
      );
    }

    // Check if set ended
    const s1 = game.scores?.[0] ?? 0;
    const s2 = game.scores?.[1] ?? 0;
    if (s1 >= mr.winningScore || s2 >= mr.winningScore) {
      endSetAndUpdateMatch();
    }

    return false;
  }

  return { stepOnce };
}

// -----------------------------
// Warmup helper
// -----------------------------
function warmupToRoundLike(game, warmupFrames) {
  for (let i = 0; i < warmupFrames; i++) {
    game.stepLogic();
    const s = game.state;
    if (
      s === game.round ||
      s === game.afterEndOfRound ||
      s === game.beforeStartOfNextRound
    ) {
      return true;
    }
  }
  return false;
}

// -----------------------------
// Hotkeys
// -----------------------------
function attachHotkeys(state) {
  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyT') {
      state.mode = 'train';
      console.log(`[MODE] train (stepsPerFrame=${state.stepsPerFrameTrain})`);
    } else if (e.code === 'KeyW') {
      state.mode = 'watch';
      console.log(`[MODE] watch (stepsPerFrame=${state.stepsPerFrameWatch})`);
    } else if (e.code === 'Space') {
      state.mode = (state.mode === 'paused') ? 'watch' : 'paused';
      console.log(`[MODE] ${state.mode}`);
    } else if (e.code === 'BracketRight') {
      // ] : speed up
      state.stepsPerFrameTrain = Math.min(5000, state.stepsPerFrameTrain + 50);
      console.log(`[TRAIN SPEED] stepsPerFrameTrain=${state.stepsPerFrameTrain}`);
    } else if (e.code === 'BracketLeft') {
      // [ : speed down
      state.stepsPerFrameTrain = Math.max(1, state.stepsPerFrameTrain - 50);
      console.log(`[TRAIN SPEED] stepsPerFrameTrain=${state.stepsPerFrameTrain}`);
    } else if (e.code === 'KeyC') {
      // cap toggle
      state.minMsPerFrame = state.minMsPerFrame === 0 ? 16 : 0;
      console.log(`[CAP] minMsPerFrame=${state.minMsPerFrame}`);
    }
  });

  console.log(
    [
      '[HOTKEYS]',
      'T: train mode (fast)',
      'W: watch mode (slow)',
      'Space: pause/resume',
      '[: train speed down',
      ']: train speed up',
      'C: toggle RAF cap (~60fps)',
    ].join(' | ')
  );
}
