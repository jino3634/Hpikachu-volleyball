'use strict';

/**
 * Compute fitness from a batch of match results.
 * Keep it mostly win-rate, but add small shaping to favor "cleaner" wins.
 * @param {{wins:number, losses:number, draws:number, scoreDiff:number}} agg
 */
export function computeFitness(agg) {
  const total = Math.max(1, agg.wins + agg.losses + agg.draws);
  const winRate = agg.wins / total;

  // Encourage larger scoreDiff a bit (scaled by total games and winningScore-like range).
  const scoreBonus = Math.max(-1, Math.min(1, (agg.scoreDiff || 0) / (total * 5)));

  // Very small penalty for draws (avoid "stalling" policies when enabled).
  const drawPenalty = Math.max(0, Math.min(1, (agg.draws || 0) / total));

  return winRate + 0.05 * scoreBonus - 0.01 * drawPenalty;
}
