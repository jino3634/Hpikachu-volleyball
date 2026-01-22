'use strict';

import { TRAIN_SEEDS, EVAL_SEEDS } from './scenarios.js';

/**
 * Default evolution configuration used by the UI runner.
 * You can override any field when calling startEvolution().
 */
export const EVO_DEFAULTS = {
  // Training opponent mode.
  // - 'self': self-play vs previous best snapshot (default)
  // - 'baseline': vs default baseline genome
  // - 'physics': (reserved) vs physics/rule-based opponent (wired in later)
  opponentMode: 'self',

  // Fair evaluation: fixed seed sets
  trainSeeds: TRAIN_SEEDS,
  evalSeeds: EVAL_SEEDS,
  // Evaluate best-on-eval every N generations (1 = every gen)
  evalEveryGenerations: 1,

  // Genetic algorithm
  populationSize: 64,
  eliteFraction: 0.15,
  mutationRate: 0.9,
  mutationSigma: 0.18,

  // Opponent pool (Hall of Fame)
  // Keep a small list of past best genomes and evaluate against them.
  hofSize: 8,
  // When true, evaluation seeds are split across opponents so total match count stays similar.
  splitSeedsAcrossOpponents: true,

  // Match settings
  winningScore: 11,
  maxFrames: 60 * 30,
  decisionInterval: 3,
  initialServeMode: 'alternate',

  // Runner
  yieldEveryGenerationMs: 0,

  // Deterministic evolution randomness (selection/mutation) for reproducibility.
  evoSeed: 1337,
};
