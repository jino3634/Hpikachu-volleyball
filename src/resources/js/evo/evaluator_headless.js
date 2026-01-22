'use strict';

import { PikaPhysics, PikaUserInput, GROUND_HALF_WIDTH } from '../physics.js';
import { setCustomRng } from '../rand.js';
import { makeXorShift32 } from './prng.js';
import { makeObservation } from './observation.js';
import { chooseAction } from './policy_weighted.js';
import { decidePhysicsAI } from '../physics_ai.js';

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
  const seed = (opts?.seed | 0) || 1;
  const winningScore = Math.max(1, (opts?.winningScore ?? 11) | 0);
  const maxFrames = Math.max(1, (opts?.maxFrames ?? (60 * 30)) | 0);
  const decisionInterval = Math.max(1, (opts?.decisionInterval ?? 3) | 0);
  const initialServeMode = /** @type {'alternate'|'p1'|'p2'} */ (opts?.initialServeMode || 'alternate');
  const servePlayer2First = (typeof opts?.servePlayer2First === 'boolean')
    ? !!opts.servePlayer2First
    : (initialServeMode === 'p2');

  const collectReplay = !!opts?.collectReplay;
  /** @type {number[]|null} */
  const replayArr = collectReplay ? [] : null;

  // Deterministic RNG for this match
  setCustomRng(makeXorShift32(seed));

  const physics = new PikaPhysics(true, true);
  physics.setDecisionInterval(decisionInterval);
  // Speed: skip expensive ball prediction on non-decision frames.
  physics.setFastEvalMode(true);

  // External AI controller (per-player genome)
  physics.setAIController((playerIndex, player, ball, otherPlayer, userInput, frameCtx) => {
    const obs = makeObservation(physics, playerIndex);
    const genome = (playerIndex === 1) ? opts.genomeP1 : opts.genomeP2;
    if (isPhysicsGenome(genome)) {
      // Scripted physics opponent
      decidePhysicsAI(playerIndex, player, ball, otherPlayer, userInput, frameCtx);
    } else {
      const act = chooseAction(obs, genome);
      userInput.xDirection = act.xDirection | 0;
      userInput.yDirection = act.yDirection | 0;
      userInput.powerHit = act.powerHit ? 1 : 0;
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
    seed,
    winner,
    scoreP1: score[0],
    scoreP2: score[1],
    rounds,
    frames: (frames | 0),
    reason,
  };

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
