'use strict';

import { runMatch } from './evaluator_headless.js';
import { TRAIN_SEEDS } from './scenarios.js';
import { computeFitness } from './fitness.js';
import { defaultGenome } from './policy_weighted.js';

/**
 * @typedef {{genome:any, eval:any}} Individual
 */

/**
 * Evaluate a genome over a fixed seed set.
 * @param {any} genome
 * @param {{
 *   seeds?:(number[]|readonly number[]),
 *   opponentGenome?:any,
 *   opponentGenomes?:any[],
 *   winningScore?:number,
 *   maxFrames?:number,
 *   decisionInterval?:number,
 *   initialServeMode?:('alternate'|'p1'|'p2'),
 *   splitSeedsAcrossOpponents?:boolean
 * }} [opts]
 */
export function evaluateGenome(genome, opts = {}) {
  const seeds = opts.seeds || TRAIN_SEEDS;
  const splitSeedsAcrossOpponents = (opts.splitSeedsAcrossOpponents !== undefined)
    ? !!opts.splitSeedsAcrossOpponents
    : true;
  /** @type {any[]} */
  const opps = Array.isArray(opts.opponentGenomes) ? opts.opponentGenomes.filter(Boolean) : [];
  if (!opps.length) opps.push(opts.opponentGenome || genome);
  const winningScore = Math.max(1, (opts.winningScore ?? 11) | 0);
  const maxFrames = Math.max(60, (opts.maxFrames ?? (60 * 30)) | 0);
  const decisionInterval = Math.max(1, (opts.decisionInterval ?? 3) | 0);
  const initialServeMode = /** @type {'alternate'|'p1'|'p2'} */ (opts.initialServeMode || 'alternate');
  const agg = { wins: 0, losses: 0, draws: 0, scoreDiff: 0 };

  // To keep evaluation cost bounded when using multiple opponents,
  // optionally split the seed list across opponents.
  const seedArr = Array.isArray(seeds) ? seeds : Array.from(seeds);
  const m = Math.max(1, opps.length);
  for (let oi = 0; oi < m; oi++) {
    const opp = opps[oi];
    let start = 0;
    let end = seedArr.length;
    if (splitSeedsAcrossOpponents && m > 1) {
      const chunk = Math.floor(seedArr.length / m);
      const rem = seedArr.length - chunk * m;
      // distribute remainder to earlier chunks
      start = oi * chunk + Math.min(oi, rem);
      end = start + chunk + (oi < rem ? 1 : 0);
      if (end <= start) {
        // if we have fewer seeds than opponents, fall back to a single seed
        start = Math.min(seedArr.length - 1, 0);
        end = start + 1;
      }
    }
    for (let si = start; si < end; si++) {
      const seed = seedArr[si];
      const r = runMatch({
        seed,
        genomeP1: genome,
        genomeP2: opp,
        winningScore,
        maxFrames,
        decisionInterval,
        initialServeMode,
      });
      const diff = (r.scoreP1 - r.scoreP2) | 0;
      agg.scoreDiff += diff;
      if (diff > 0) agg.wins++;
      else if (diff < 0) agg.losses++;
      else agg.draws++;
    }
  }

  return {
    ...agg,
    fitness: computeFitness(agg),
    winRate: agg.wins / Math.max(1, agg.wins + agg.losses + agg.draws),
  };
}

/**
 * Create a fresh population around a base genome.
 * @param {number} size
 * @param {{baseGenome?:any, sigma?:number}} [opts]
 * @returns {Individual[]}
 */
export function initPopulation(size, opts = {}) {
  const n = Math.max(2, size | 0);
  const base = opts.baseGenome || defaultGenome();
  const sigma = Number(opts.sigma ?? 0.18);

  /** @type {Individual[]} */
  const pop = [];
  // First individual = exact base
  pop.push({ genome: cloneGenome(base), eval: null });
  // Rest = mutated base
  for (let i = 1; i < n; i++) {
    pop.push({ genome: mutateGenome(base, { sigma, rate: 1.0 }), eval: null });
  }
  return pop;
}

/**
 * Evolve one generation using elitism + mutation.
 * @param {Individual[]} population
 * @param {{
 *   seeds?:(number[]|readonly number[]),
 *   opponentGenome?:any,
 *   opponentGenomes?:any[],
 *   eliteFraction?:number,
 *   mutationRate?:number,
 *   mutationSigma?:number,
 *   rng?:() => number,
 *   winningScore?:number,
 *   maxFrames?:number,
 *   decisionInterval?:number,
 *   initialServeMode?:('alternate'|'p1'|'p2'),
 *   splitSeedsAcrossOpponents?:boolean
 * }} opts
 */
export function evolveOneGeneration(population, opts = {}) {
  const seeds = opts.seeds || TRAIN_SEEDS;
  const opponentGenome = opts.opponentGenome;
  const opponentGenomes = Array.isArray(opts.opponentGenomes) ? opts.opponentGenomes : null;
  const eliteFraction = clamp01(Number(opts.eliteFraction ?? 0.15));
  const mutationRate = clamp01(Number(opts.mutationRate ?? 0.9));
  const mutationSigma = Math.max(0, Number(opts.mutationSigma ?? 0.18));
  const rng = (typeof opts.rng === 'function') ? opts.rng : Math.random;

  const splitSeedsAcrossOpponents = (opts.splitSeedsAcrossOpponents !== undefined)
    ? !!opts.splitSeedsAcrossOpponents
    : true;

  const initialServeMode = /** @type {'alternate'|'p1'|'p2'} */ (opts.initialServeMode || 'alternate');

  // 1) evaluate
  for (const ind of population) {
    ind.eval = evaluateGenome(ind.genome, {
      seeds,
      opponentGenome,
      opponentGenomes,
      winningScore: opts.winningScore,
      maxFrames: opts.maxFrames,
      decisionInterval: opts.decisionInterval,
      initialServeMode,
      splitSeedsAcrossOpponents,
    });
  }

  // 2) sort by fitness
  const ranked = [...population].sort((a, b) => (b.eval?.fitness ?? -1e9) - (a.eval?.fitness ?? -1e9));
  const best = ranked[0];

  // 3) stats
  let sumWin = 0;
  let sumFit = 0;
  for (const ind of ranked) {
    sumWin += Number(ind.eval?.winRate ?? 0);
    sumFit += Number(ind.eval?.fitness ?? 0);
  }
  const avgWinRate = sumWin / Math.max(1, ranked.length);
  const avgFitness = sumFit / Math.max(1, ranked.length);

  // 4) next population
  const eliteCount = Math.max(1, Math.floor(ranked.length * eliteFraction));
  /** @type {Individual[]} */
  const nextPop = [];
  for (let i = 0; i < eliteCount; i++) {
    nextPop.push({ genome: cloneGenome(ranked[i].genome), eval: null });
  }
  while (nextPop.length < ranked.length) {
    const parent = ranked[(rng() * eliteCount) | 0].genome;
    const child = mutateGenome(parent, { sigma: mutationSigma, rate: mutationRate, rng });
    nextPop.push({ genome: child, eval: null });
  }

  return {
    nextPop,
    ranked,
    best: { genome: best.genome, eval: best.eval },
    avgWinRate,
    avgFitness,
    eliteCount,
  };
}

function cloneGenome(g) {
  return JSON.parse(JSON.stringify(g));
}

/**
 * Mutate numeric fields in a genome.
 * @param {any} genome
 * @param {{sigma:number, rate:number, rng?:() => number}} opts
 */
export function mutateGenome(genome, opts) {
  const sigma = Math.max(0, Number(opts?.sigma ?? 0.18));
  const rate = clamp01(Number(opts?.rate ?? 0.9));
  const rng = (typeof opts?.rng === 'function') ? opts.rng : Math.random;
  const out = cloneGenome(genome);
  for (const k of Object.keys(out)) {
    const v = out[k];
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    if (rng() > rate) continue;
    const n = v + randn(rng) * sigma;
    out[k] = sanitizeParam(k, n);
  }
  return out;
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function randn(rng) {
  // Box-Muller
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sanitizeParam(key, val) {
  // Simple safety bounds for known thresholds.
  if (key === 'jumpMinBallY') return clamp(val, 20, 200);
  if (key === 'powerMinBallY') return clamp(val, 20, 220);
  if (key === 'powerMaxDX') return clamp(val, 20, 200);
  if (key === 'deadZoneX') return clamp(val, 0, 60);
  if (key === 'netAvoidBand') return clamp(val, 8, 180);
  if (key === 'minPowerScore') return clamp(val, -2, 2);
  // weights
  return clamp(val, -5, 5);
}

function clamp(x, lo, hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}
