'use strict';

import { DEFAULT_SEEDS } from './scenarios.js';

/**
 * Default evolution configuration used by the UI runner.
 * You can override any field when calling startEvolution().
 */
export const EVO_DEFAULTS = {
  // Fair evaluation: fixed seed set
  seeds: DEFAULT_SEEDS,

  // Genetic algorithm
  populationSize: 64,
  eliteFraction: 0.15,
  mutationRate: 0.9,
  mutationSigma: 0.18,

  // Match settings
  winningScore: 11,
  maxFrames: 60 * 30,
  decisionInterval: 3,

  // Runner
  yieldEveryGenerationMs: 0,
};
