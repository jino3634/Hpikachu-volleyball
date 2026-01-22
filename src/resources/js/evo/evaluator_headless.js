'use strict';

import { PikaPhysics, PikaUserInput, GROUND_HALF_WIDTH } from '../physics.js';
import { setCustomRng } from '../rand.js';
import { makeXorShift32 } from './prng.js';
import { makeObservation } from './observation.js';
import { chooseAction } from './policy_weighted.js';
import { decidePhysicsAI } from '../physics_ai.js';
import { evoLogger } from './logger.js';

let __matchSeq = 0;

/**
 * @param {any} g
 * @returns {boolean}
 */
function isPhysicsGenome(g) {
  return !!(g && typeof g === 'object' && g.__opp === 'physics');
}

/**
 * Run a headless match (no rendering) and return diagnostics.
 *
 * IMPORTANT: scoring logic mirrors pikavolley.js round() as closely as possible:
 * when ball touches ground, point is awarded based on ball.punchEffectX.
 *
 * @param {{
 *  seed:number,
 *  genomeP1:any,
 *  genomeP2:any,
 *  winningScore?:number,
 *  maxFrames?:number,
 *  decisionInterval?:number,
 *  servePlayer2First?:boolean,
 *  initialServeMode?:('alternate'|'p1'|'p2'),
 *  deterministic?:boolean,
 *  collectReplay?:boolean
 * }} opts
 */
export function runMatch(opts) {
  const seed = (Number.isFinite(opts?.seed) ? (opts.seed | 0) : 1);
  const winningScore = Math.max(1, (opts?.winningScore ?? 11) | 0);
  const maxFrames = Math.max(1, (opts?.maxFrames ?? (60 * 30)) | 0);
  const decisionInterval = Math.max(1, (opts?.decisionInterval ?? 3) | 0);
  const initialServeMode = /** @type {'alternate'|'p1'|'p2'} */ (opts?.initialServeMode || 'alternate');
  // servePlayer2First:
  // - if explicitly provided, respect it
  // - else if initialServeMode is p1/p2, follow it
  // - else (alternate): deterministically alternate by seed parity so runMatch is reproducible
  const servePlayer2First = (typeof opts?.servePlayer2First === 'boolean')
    ? !!opts.servePlayer2First
    : (initialServeMode === 'p2')
      ? true
      : (initialServeMode === 'p1')
        ? false
        : ((seed & 1) === 1);

  const collectReplay = !!opts?.collectReplay;
  /** @type {number[]|null} */
  const replayArr = collectReplay ? [] : null;

  const matchId = `m${(++__matchSeq)}_s${seed}`;
  const routeP1 = isPhysicsGenome(opts?.genomeP1) ? 'physics' : 'genome';
  const routeP2 = isPhysicsGenome(opts?.genomeP2) ? 'physics' : 'genome';
  const decisionCalls = [0, 0];
  const routeCalls = { p1: { physics: 0, genome: 0 }, p2: { physics: 0, genome: 0 } };
  const held = [ { x: 0, y: 0, power: 0 }, { x: 0, y: 0, power: 0 } ];
  const warn = { inputChanged: 0, powerOnNonDecision: 0, outOfRangeInput: 0 };

  // Deterministic RNG for this match
  setCustomRng(makeXorShift32(seed));

  const physics = new PikaPhysics(true, true);
  physics.setDecisionInterval(decisionInterval);
  // Speed: skip expensive ball prediction on non-decision frames.
  physics.setFastEvalMode(true);

  // ---- Stage A3: expose match context for downstream anomaly logs (obs/physics) ----
  try {
    const physAny = /** @type {any} */ (physics);
    physAny.__evoMatchId = matchId;
    physAny.__evoSeed = seed;
    physAny.__evoObsWarnMatchId = matchId;
    physAny.__evoObsWarnCount = 0;
  } catch {}
  try {
    // @ts-ignore
    if (typeof window !== 'undefined') window.__evoLogCtx = { matchId, seed };
  } catch {}

  // Stage 3: route confirmation (once per match)
  try {
    evoLogger.log('physics_route', 'INFO', { matchId, seed, decisionInterval, winningScore, maxFrames, routeP1, routeP2 });
  } catch {}

  // External AI controller (per-player genome)
  physics.setAIController((playerIndex, player, ball, otherPlayer, userInput, frameCtx) => {

    const genome = (playerIndex === 1) ? opts.genomeP1 : opts.genomeP2;

    const isP1 = (playerIndex === 1);
    const idx = isP1 ? 0 : 1;
    const isDecision = !!(frameCtx && frameCtx.decisionFrame);

    // Stage A2: On non-decision frames, keep input held constant.
    // This prevents false-positive "input changed" logs and avoids wasting compute on chooseAction().
    if (!isDecision) {
      const h = held[idx];
      const ax = userInput.xDirection | 0;
      const ay = userInput.yDirection | 0;
      const ap = userInput.powerHit ? 1 : 0;

      // If anything changed between decisions, log (limited per match) then overwrite with held input.
      if (ax !== (h.x | 0) || ay !== (h.y | 0)) {
        warn.inputChanged = (warn.inputChanged + 1) | 0;
        if (warn.inputChanged <= 10) {
          try {
            evoLogger.log('input', 'WARN', {
              matchId,
              seed,
              frame: frameCtx?.frame,
              playerIndex,
              note: 'inputChangedBetweenDecisions',
              expected: { x: h.x | 0, y: h.y | 0 },
              actual: { x: ax, y: ay },
            });
          } catch {}
        }
      }
      if (ap !== 0) {
        warn.powerOnNonDecision = (warn.powerOnNonDecision + 1) | 0;
        if (warn.powerOnNonDecision <= 10) {
          try {
            evoLogger.log('input', 'WARN', {
              matchId,
              seed,
              frame: frameCtx?.frame,
              playerIndex,
              note: 'powerOnNonDecision',
              actualPower: ap,
            });
          } catch {}
        }
      }

      userInput.xDirection = h.x | 0;
      userInput.yDirection = h.y | 0;
      // powerHit should be edge-triggered on decision frames only.
      userInput.powerHit = 0;
      return;
    }

    // Decision frame: compute action.
    decisionCalls[idx] = (decisionCalls[idx] + 1) | 0;

    if (isPhysicsGenome(genome)) {
      // Scripted physics opponent (expected to write into userInput).
      if (isP1) routeCalls.p1.physics = (routeCalls.p1.physics + 1) | 0;
      else routeCalls.p2.physics = (routeCalls.p2.physics + 1) | 0;

      decidePhysicsAI(playerIndex, player, ball, otherPlayer, userInput, frameCtx);
    } else {
      if (isP1) routeCalls.p1.genome = (routeCalls.p1.genome + 1) | 0;
      else routeCalls.p2.genome = (routeCalls.p2.genome + 1) | 0;

      const obs = makeObservation(physics, playerIndex);
      const act = chooseAction(obs, genome);
      userInput.xDirection = act.xDirection | 0;
      userInput.yDirection = act.yDirection | 0;
      userInput.powerHit = act.powerHit ? 1 : 0;
    }

    // Snapshot held input as decided (decision frames only).
    held[idx].x = userInput.xDirection | 0;
    held[idx].y = userInput.yDirection | 0;
    held[idx].power = userInput.powerHit ? 1 : 0;

    // Basic range validation (should always be within [-1,1] and power 0/1).
    const x = userInput.xDirection | 0;
    const y = userInput.yDirection | 0;
    const p = userInput.powerHit ? 1 : 0;
    if (x < -1 || x > 1 || y < -1 || y > 1 || (p !== 0 && p !== 1)) {
      warn.outOfRangeInput = (warn.outOfRangeInput + 1) | 0;
      try {
        evoLogger.log('input', 'WARN', { matchId, seed, frame: frameCtx?.frame, playerIndex, x, y, p, note: 'outOfRange' });
      } catch {}
    }

  });

  const inputs = [new PikaUserInput(), new PikaUserInput()];

  // initialize round like pikavolley.js startOfNewGame
  let isPlayer2Serve = servePlayer2First;
  physics.player1.initializeForNewRound();
  physics.player2.initializeForNewRound();
  physics.ball.initializeForNewRound(isPlayer2Serve);
  physics.resetAIState();

  const score = [0, 0];
  let rounds = 0;
  let frames = 0;
  let reason = 'maxFrames';

  // main loop
  for (frames = 0; frames < maxFrames; frames++) {
    const touchingGround = physics.runEngineForNextFrame(inputs);

    // Record the *actual* inputs used by the physics engine for this frame.
    // (Both AIs write into `inputs` via physics.setAIController.)
    if (replayArr) {
      const p1 = inputs[0];
      const p2 = inputs[1];
      replayArr.push(
        (p1.xDirection | 0),
        (p1.yDirection | 0),
        (p1.powerHit ? 1 : 0),
        (p2.xDirection | 0),
        (p2.yDirection | 0),
        (p2.powerHit ? 1 : 0)
      );
    }

    if (touchingGround) {
      rounds++;
      // scoring rule mirrors pikavolley.js:
      if (physics.ball.punchEffectX < GROUND_HALF_WIDTH) {
        // left side touched => player2 gets point
        isPlayer2Serve = true;
        score[1]++;
      } else {
        isPlayer2Serve = false;
        score[0]++;
      }

      if (score[0] >= winningScore || score[1] >= winningScore) {
        reason = 'winningScore';
        break;
      }

      // next round init mirrors beforeStartOfNextRound -> startOfNewRound path
      physics.player1.initializeForNewRound();
      physics.player2.initializeForNewRound();
      physics.ball.initializeForNewRound(isPlayer2Serve);
      physics.resetAIState();
    }
  }

  const winner = (score[0] === score[1]) ? 0 : (score[0] > score[1] ? 1 : 2);
  const result = {
    matchId,
    seed,
    winner,
    scoreP1: score[0],
    scoreP2: score[1],
    rounds,
    frames: (frames | 0),
    reason,
  };

  // Stage 3 diagnostic: match summary + anomaly checks (no replay payload).
  // Enable from UI (Stage 2) or console: window.__evoLogger.setEnabled(true)
  try {
    const framesOut = (frames | 0);
    const scoreP1 = score[0] | 0;
    const scoreP2 = score[1] | 0;
    const sumScore = (scoreP1 + scoreP2) | 0;
    const maxScore = (scoreP1 > scoreP2) ? scoreP1 : scoreP2;

    // Anomaly checks (log as WARN/ERROR but keep returning result).
    if (sumScore !== (rounds | 0)) {
      evoLogger.log('match_anomaly', 'WARN', { matchId, seed, note: 'scoreSumNotRounds', rounds, scoreP1, scoreP2 });
    }
    if (reason === 'winningScore' && maxScore < (winningScore | 0)) {
      evoLogger.log('match_anomaly', 'ERROR', { matchId, seed, note: 'reasonWinningButScoreLow', winningScore, scoreP1, scoreP2, rounds, frames: framesOut });
    }
    if (reason === 'maxFrames' && maxScore >= (winningScore | 0)) {
      evoLogger.log('match_anomaly', 'WARN', { matchId, seed, note: 'reasonMaxFramesButReachedWinning', winningScore, scoreP1, scoreP2, rounds, frames: framesOut });
    }
    if (framesOut <= 0) {
      evoLogger.log('match_anomaly', 'WARN', { matchId, seed, note: 'framesNonPositive', frames: framesOut, rounds, scoreP1, scoreP2 });
    }
    if (framesOut < 60) {
      evoLogger.log('match_anomaly', 'WARN', { matchId, seed, note: 'matchTooShort', frames: framesOut, rounds, scoreP1, scoreP2, reason });
    }

    evoLogger.log('match', 'INFO', {
      matchId,
      seed,
      winner,
      scoreP1,
      scoreP2,
      rounds,
      frames: framesOut,
      reason,
      winningScore,
      maxFrames,
      decisionInterval,
      initialServeMode,
      servePlayer2First,
      routeP1,
      routeP2,
      decisionCallsP1: decisionCalls[0] | 0,
      decisionCallsP2: decisionCalls[1] | 0,
      routeCalls,
      warn,
      collectReplay: !!collectReplay,
    });
  } catch {}

  if (replayArr) {
    // Fixed-size packed input stream: 6 int8s per frame.
    // [p1.x, p1.y, p1.power, p2.x, p2.y, p2.power]
    const packed = Int8Array.from(replayArr);
    // NOTE: We intentionally keep this as a typed array here (no base64 yet).
    // Stage 2/4 will decide the best UI/storage representation.
    result.replay = {
      seed,
      winningScore,
      maxFrames,
      decisionInterval,
      initialServeMode,
      servePlayer2First,
      frames: (frames | 0),
      packed,
    };
  }
  return result;
}

/**
 * Run a batch of matches over a fixed seed list.
 * Returns aggregated stats for fitness.
 *
 * @param {{
 *   seeds:(number[]|readonly number[]),
 *   genomeP1:any,
 *   genomeP2:any,
 *   winningScore?:number,
 *   maxFrames?:number,
 *   decisionInterval?:number,
 *   servePlayer2First?:boolean,
 *   initialServeMode?:('alternate'|'p1'|'p2')
 * }} opts
 */
export function runBatch(opts) {
  const seeds = Array.isArray(opts?.seeds) ? opts.seeds : [];
  const agg = { wins: 0, losses: 0, draws: 0, scoreDiff: 0, matches: 0 };
  const di = opts?.decisionInterval;
  const initialServeMode = /** @type {'alternate'|'p1'|'p2'} */ (opts?.initialServeMode || 'alternate');
  for (let i = 0; i < seeds.length; i++) {
    const serveP2 = (typeof opts?.servePlayer2First === 'boolean')
      ? !!opts.servePlayer2First
      : (initialServeMode === 'alternate' ? ((i & 1) === 1) : (initialServeMode === 'p2'));
    const r = runMatch({
      seed: seeds[i],
      genomeP1: opts.genomeP1,
      genomeP2: opts.genomeP2,
      winningScore: opts.winningScore,
      maxFrames: opts.maxFrames,
      decisionInterval: di,
      servePlayer2First: serveP2,
      initialServeMode,
    });
    agg.matches++;
    agg.scoreDiff += (r.scoreP1 - r.scoreP2);
    if (r.winner === 1) agg.wins++;
    else if (r.winner === 2) agg.losses++;
    else agg.draws++;
  }
  return agg;
}
