'use strict';

import { runMatch } from './evaluator_headless.js';
import { DEFAULT_SEEDS } from './scenarios.js';
import { computeFitness } from './fitness.js';

/**
 * Evaluate a genome over a fixed seed set.
 * @param {any} genome
 * @param {{seeds?:number[], opponentGenome?:any}} [opts]
 */
export function evaluateGenome(genome, opts = {}) {
  const seeds = opts.seeds || DEFAULT_SEEDS;
  const opp = opts.opponentGenome || genome;
  const agg = { wins: 0, losses: 0, draws: 0, scoreDiff: 0 };

  for (const seed of seeds) {
    const r = runMatch({ seed, genomeP1: genome, genomeP2: opp, winningScore: 11, maxFrames: 60 * 30, decisionInterval: 3 });
    const diff = (r.scoreP1 - r.scoreP2) | 0;
    agg.scoreDiff += diff;
    if (diff > 0) agg.wins++;
    else if (diff < 0) agg.losses++;
    else agg.draws++;
  }

  return {
    ...agg,
    fitness: computeFitness(agg),
    winRate: agg.wins / Math.max(1, agg.wins + agg.losses + agg.draws),
  };
}
