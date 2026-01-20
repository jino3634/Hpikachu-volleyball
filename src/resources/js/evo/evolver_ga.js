/**
 * Simple GA loop (skeleton).
 *
 * This file is intentionally minimal: it provides the shape of the evolution loop.
 * You can extend fitness, seeds, elitism, mutation rules later.
 */
'use strict';

import { defaultGenome } from './policy_weighted.js';
import { runMatch } from './evaluator_headless.js';

/**
 * @param {any} g
 */
function cloneGenome(g) {
  return JSON.parse(JSON.stringify(g));
}

/**
 * Mutate genome in-place.
 * @param {any} g
 * @param {number} sigma
 */
function mutate(g, sigma) {
  const s = Math.max(0, Number(sigma ?? 0.1));
  for (const k of Object.keys(g)) {
    const v = g[k];
    if (typeof v === 'number') {
      // gaussian-ish via sum of uniforms
      let n = 0;
      for (let i = 0; i < 6; i++) n += (Math.random() - 0.5);
      g[k] = v + n * s;
    }
  }
  // keep some bounds sensible
  g.moveDeadband = Math.max(0, Number(g.moveDeadband ?? 0));
  g.jumpBallDX = Math.max(0, Number(g.jumpBallDX ?? 0));
  g.powerBallDX = Math.max(0, Number(g.powerBallDX ?? 0));
  return g;
}

/**
 * Create initial population.
 * @param {number} size
 */
export function createPopulation(size) {
  const n = Math.max(2, (size | 0) || 32);
  const base = defaultGenome();
  const pop = [];
  for (let i = 0; i < n; i++) {
    const g = cloneGenome(base);
    mutate(g, 0.25);
    pop.push(g);
  }
  return pop;
}

/**
 * Evaluate a genome against a fixed opponent genome.
 * @param {any} genome
 * @param {any} opponent
 * @param {number[]} seeds
 * @returns {number} winrate in [0,1]
 */
export function evalWinrate(genome, opponent, seeds) {
  const ss = (seeds && seeds.length) ? seeds : [1,2,3,4,5];
  let wins = 0;
  let total = 0;
  for (const seed of ss) {
    const r = runMatch({ seed, genomeP1: genome, genomeP2: opponent, winningScore: 11, maxFrames: 60*45 });
    if (r.p1 > r.p2) wins++;
    total++;
  }
  return total ? (wins / total) : 0;
}

/**
 * Evolve one generation.
 * @param {any[]} pop
 * @param {number[]} seeds
 * @returns {{next:any[], best:any, bestScore:number}}
 */
export function evolveOneGeneration(pop, seeds) {
  const population = pop && pop.length ? pop : createPopulation(32);
  const eliteN = Math.max(1, Math.floor(population.length * 0.1));

  // Use current best as opponent for quick pressure.
  const baseline = defaultGenome();

  const scored = population.map((g) => ({
    g,
    score: evalWinrate(g, baseline, seeds),
  }));
  scored.sort((a,b) => b.score - a.score);

  const best = scored[0]?.g;
  const bestScore = scored[0]?.score ?? 0;

  const next = [];
  // keep elites
  for (let i = 0; i < eliteN; i++) next.push(cloneGenome(scored[i].g));

  // fill rest with mutated elites
  while (next.length < population.length) {
    const parent = scored[(Math.random() * eliteN) | 0].g;
    const child = cloneGenome(parent);
    mutate(child, 0.15);
    next.push(child);
  }

  return { next, best: cloneGenome(best), bestScore };
}
