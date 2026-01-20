/**
 * Headless evaluator (no rendering).
 *
 * Notes:
 * - This is meant for fast evolution runs.
 * - Scoring rule used here: if ball touches ground on left side => right scores, and vice versa.
 */
'use strict';

import { PikaPhysics, PikaUserInput, GROUND_HALF_WIDTH } from '../physics.js';
import { setCustomRng } from '../rand.js';
import { createXorshift32 } from './prng.js';
import { makeObservation } from './observation.js';
import { chooseAction } from './policy_weighted.js';

/**
 * Run a headless match.
 * @param {{seed:number, genomeP1:any, genomeP2:any, winningScore?:number, maxFrames?:number, decisionInterval?:number}} opts
 * @returns {{p1:number, p2:number, frames:number, seed:number}}
 */
export function runMatch(opts) {
  const seed = (opts?.seed | 0) || 1;
  const winningScore = Math.max(1, (opts?.winningScore ?? 15) | 0);
  const maxFrames = Math.max(60, (opts?.maxFrames ?? (60 * 60)) | 0);
  const decisionInterval = Math.max(1, (opts?.decisionInterval ?? 3) | 0);

  const rng = createXorshift32(seed);
  setCustomRng(rng);

  const physics = new PikaPhysics(true, true);
  physics.setDecisionInterval(decisionInterval);

  // Controller uses held input, updated only on decision frames by physics.js
  physics.setAIController((playerIndex, _player, _ball, _other, held) => {
    const obs = makeObservation(physics, /** @type {1|2} */ (playerIndex));
    const genome = (playerIndex === 1) ? opts.genomeP1 : opts.genomeP2;
    const act = chooseAction(obs, genome);
    held.xDirection = act.xDir;
    held.yDirection = act.yDir;
    held.powerHit = act.powerHit;
  });

  const inputs = [new PikaUserInput(), new PikaUserInput()];

  let p1 = 0;
  let p2 = 0;
  let frames = 0;

  // Start with P1 serve
  physics.player1.initializeForNewRound();
  physics.player2.initializeForNewRound();
  physics.ball.initializeForNewRound(false);
  physics.resetAIState();

  while (frames < maxFrames && p1 < winningScore && p2 < winningScore) {
    const touched = physics.runEngineForNextFrame(inputs);
    frames++;

    if (touched) {
      // If ball lands on left side, right scores.
      const rightScores = (physics.ball.x < GROUND_HALF_WIDTH);
      if (rightScores) p2++;
      else p1++;

      const isPlayer2Serve = rightScores;
      physics.player1.initializeForNewRound();
      physics.player2.initializeForNewRound();
      physics.ball.initializeForNewRound(isPlayer2Serve);
      physics.resetAIState();
    }
  }

  // Restore default rng
  setCustomRng(null);

  return { p1, p2, frames, seed };
}
