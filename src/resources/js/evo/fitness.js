'use strict';

/**
 * Compute fitness from a batch of match results.
 * @param {{wins:number, losses:number, draws:number, scoreDiff:number}} agg
 */
export function computeFitness(agg) {
  const total = Math.max(1, agg.wins + agg.losses + agg.draws);
  const winRate = agg.wins / total;
  // small tie-break using scoreDiff
  const scoreBonus = Math.max(-1, Math.min(1, (agg.scoreDiff || 0) / (total * 5)));
  return winRate + 0.02 * scoreBonus;
}
