'use strict';

import { EVO_DEFAULTS } from './config.js';
import { defaultGenome } from './policy_weighted.js';
import { evolveOneGeneration, initPopulation } from './evolver_ga.js';
import { evaluateGenome } from './evolver_ga.js';
import { loadBest, saveBest } from './storage.js';

/**
 * Evolution runner state (single instance).
 */
const state = {
  running: false,
  generation: 0,
  bestGenome: null,
  bestWinRate: 0,
  bestEvalWinRate: 0,
  bestFitness: 0,
  lastSavedAt: 0,
  population: null,
  config: null,
};

/**
 * @returns {boolean}
 */
export function isEvolutionRunning() {
  return !!state.running;
}

/**
 * Start evolution loop.
 * @param {Partial<typeof EVO_DEFAULTS>} [opts]
 * @param {(s:any)=>void} [onUpdate]
 */
export async function startEvolution(opts = {}, onUpdate = null) {
  if (state.running) return;

  const cfg = { ...EVO_DEFAULTS, ...(opts || {}) };
  state.config = cfg;

  // Load saved best if exists
  const saved = loadBest();
  if (saved?.genome) {
    state.bestGenome = saved.genome;
    state.bestWinRate = Number(saved.bestWinRate ?? 0);
    state.bestEvalWinRate = Number(saved.bestEvalWinRate ?? saved.bestWinRate ?? 0);
    state.bestFitness = Number(saved.bestFitness ?? 0);
    state.generation = Math.max(0, (saved.generation ?? 0) | 0);
    state.lastSavedAt = Number(saved.savedAt ?? 0);
  } else {
    state.bestGenome = defaultGenome();
    state.bestWinRate = 0;
    state.bestEvalWinRate = 0;
    state.bestFitness = 0;
    state.generation = 0;
    state.lastSavedAt = 0;
  }

  // Init population around best
  const base = state.bestGenome || defaultGenome();
  state.population = initPopulation(cfg.populationSize, {
    baseGenome: base,
    sigma: cfg.mutationSigma,
  });

  state.running = true;
  emit(onUpdate, { ...state, event: 'started' });

  // Fixed opponent for absolute progress: baseline genome.
  const baselineOpp = defaultGenome();

  try {
    while (state.running) {
      const t0 = Date.now();

      const res = evolveOneGeneration(state.population, {
        seeds: cfg.trainSeeds,
        opponentGenome: baselineOpp,
        eliteFraction: cfg.eliteFraction,
        mutationRate: cfg.mutationRate,
        mutationSigma: cfg.mutationSigma,
        winningScore: cfg.winningScore,
        maxFrames: cfg.maxFrames,
        decisionInterval: cfg.decisionInterval,
        initialServeMode: cfg.initialServeMode,
      });

      state.population = res.nextPop;
      state.generation += 1;

      const bestEval = res.best.eval || {};
      const bestWin = Number(bestEval.winRate ?? 0);
      const bestFit = Number(bestEval.fitness ?? 0);

      // Optionally evaluate the best on a separate eval seed set for reporting/saving.
      let bestEvalWinRate = state.bestEvalWinRate;
      if (cfg.evalSeeds && cfg.evalSeeds.length && ((state.generation % Math.max(1, (cfg.evalEveryGenerations ?? 1) | 0)) === 0)) {
        const evalRes = evaluateGenome(res.best.genome, {
          seeds: cfg.evalSeeds,
          opponentGenome: baselineOpp,
          winningScore: cfg.winningScore,
          maxFrames: cfg.maxFrames,
          decisionInterval: cfg.decisionInterval,
          initialServeMode: cfg.initialServeMode,
        });
        bestEvalWinRate = Number(evalRes.winRate ?? 0);
      }

      let savedNow = false;
      // Save based on eval-winrate (preferred) when available; fall back to train-winrate.
      const saveScore = (Number.isFinite(bestEvalWinRate) ? bestEvalWinRate : bestWin);
      const prevSaveScore = (Number.isFinite(state.bestEvalWinRate) ? state.bestEvalWinRate : state.bestWinRate);
      if (saveScore > (prevSaveScore + 1e-9)) {
        state.bestGenome = res.best.genome;
        state.bestWinRate = bestWin;
        state.bestEvalWinRate = bestEvalWinRate;
        state.bestFitness = bestFit;
        state.lastSavedAt = Date.now();
        saveBest({
          genome: state.bestGenome,
          bestWinRate: state.bestWinRate,
          bestEvalWinRate: state.bestEvalWinRate,
          bestFitness: state.bestFitness,
          generation: state.generation,
          savedAt: state.lastSavedAt,
        });
        savedNow = true;
      }

      const elapsedMs = Date.now() - t0;
      emit(onUpdate, {
        ...state,
        event: 'generation',
        generationBestWinRate: bestWin,
        generationBestEvalWinRate: bestEvalWinRate,
        generationBestFitness: bestFit,
        avgWinRate: res.avgWinRate,
        avgFitness: res.avgFitness,
        eliteCount: res.eliteCount,
        elapsedMs,
        savedNow,
      });

      // Yield to UI
      const sleepMs = Math.max(0, cfg.yieldEveryGenerationMs | 0);
      if (sleepMs > 0) {
        await sleep(sleepMs);
      } else {
        await sleep(0);
      }
    }
  } finally {
    state.running = false;
    emit(onUpdate, { ...state, event: 'stopped' });
  }
}

/**
 * Stop evolution loop.
 */
export function stopEvolution() {
  state.running = false;
}

/**
 * Return current runner snapshot (for UI init).
 */
export function getEvolutionState() {
  return { ...state };
}

function emit(cb, payload) {
  if (typeof cb === 'function') {
    try {
      cb(payload);
    } catch (e) {
      // ignore UI errors
      console.error(e);
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
