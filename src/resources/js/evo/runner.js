'use strict';

import { EVO_DEFAULTS } from './config.js';
import { defaultGenome } from './policy_weighted.js';
import { evolveOneGeneration, initPopulation } from './evolver_ga.js';
import { evaluateGenome } from './evolver_ga.js';
import { loadBest, saveBest, loadHof, saveHof } from './storage.js';
import { makeXorShift32 } from './prng.js';

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
  hof: [],
  rng: null,
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
  state.rng = makeXorShift32(Number(cfg.evoSeed ?? 1337) | 0);

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

  // Load Hall of Fame (past best opponents)
  state.hof = loadHof();

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
        opponentGenomes: buildOpponentPool(baselineOpp, state.bestGenome, state.hof, cfg.hofSize),
        rng: state.rng,
        eliteFraction: cfg.eliteFraction,
        mutationRate: cfg.mutationRate,
        mutationSigma: cfg.mutationSigma,
        winningScore: cfg.winningScore,
        maxFrames: cfg.maxFrames,
        decisionInterval: cfg.decisionInterval,
        splitSeedsAcrossOpponents: !!cfg.splitSeedsAcrossOpponents,
        rng: state.rng,
        // JSDoc inference may widen config.initialServeMode to string; cast to the intended union.
        initialServeMode: /** @type {'alternate'|'p1'|'p2'} */ (cfg.initialServeMode || 'alternate'),
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
          opponentGenomes: buildOpponentPool(baselineOpp, state.bestGenome, state.hof, cfg.hofSize),
          winningScore: cfg.winningScore,
          maxFrames: cfg.maxFrames,
          decisionInterval: cfg.decisionInterval,
          splitSeedsAcrossOpponents: !!cfg.splitSeedsAcrossOpponents,
          // Cast for the same reason as above.
          initialServeMode: /** @type {'alternate'|'p1'|'p2'} */ (cfg.initialServeMode || 'alternate'),
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

        // Update Hall of Fame with this new best (for stronger, less exploitable evolution).
        state.hof = updateHof(state.hof, {
          genome: state.bestGenome,
          bestWinRate: state.bestWinRate,
          bestEvalWinRate: state.bestEvalWinRate,
          bestFitness: state.bestFitness,
          generation: state.generation,
          savedAt: state.lastSavedAt,
        }, Math.max(1, (cfg.hofSize ?? 8) | 0));
        saveHof(state.hof);
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
 * Build opponent pool for evaluation.
 * Order matters a bit: baseline first, then current best, then HOF.
 * @param {any} baseline
 * @param {any} best
 * @param {Array<{genome:any}>} hof
 * @param {number} maxHof
 * @returns {any[]}
 */
function buildOpponentPool(baseline, best, hof, maxHof) {
  /** @type {any[]} */
  const out = [];
  if (baseline) out.push(baseline);
  if (best) out.push(best);

  const lim = Math.max(0, (maxHof ?? 0) | 0);
  if (lim > 0 && Array.isArray(hof) && hof.length) {
    for (let i = 0; i < hof.length && out.length < (2 + lim); i++) {
      const g = hof[i]?.genome;
      if (g) out.push(g);
    }
  }
  return out;
}

/**
 * Insert a new best into the Hall of Fame, keeping list small and de-duplicated.
 * @param {Array<any>} list
 * @param {any} entry
 * @param {number} maxSize
 */
function updateHof(list, entry, maxSize) {
  const maxN = Math.max(1, maxSize | 0);
  const arr = Array.isArray(list) ? [...list] : [];
  const key = safeKey(entry?.genome);

  // de-dup by genome JSON
  for (let i = arr.length - 1; i >= 0; i--) {
    if (safeKey(arr[i]?.genome) === key) {
      arr.splice(i, 1);
    }
  }
  arr.unshift(entry);

  // sort by eval-winrate (desc), then by fitness
  arr.sort((a, b) => {
    const aw = Number(a?.bestEvalWinRate ?? a?.bestWinRate ?? 0);
    const bw = Number(b?.bestEvalWinRate ?? b?.bestWinRate ?? 0);
    if (bw !== aw) return bw - aw;
    return Number(b?.bestFitness ?? 0) - Number(a?.bestFitness ?? 0);
  });

  // trim
  if (arr.length > maxN) arr.length = maxN;
  return arr;
}

function safeKey(genome) {
  try {
    return JSON.stringify(genome);
  } catch {
    return String(genome);
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
