/**
 * Local storage helpers for evolution.
 */
'use strict';

const KEY_BEST = 'evo_best_genome_v1';
const KEY_GEN = 'evo_generation_v1';

export function saveBestGenome(genome, gen, score) {
  try {
    localStorage.setItem(KEY_BEST, JSON.stringify({ genome, gen: gen|0, score: Number(score ?? 0) }));
    localStorage.setItem(KEY_GEN, String(gen|0));
  } catch (_) {
    // ignore
  }
}

export function loadBestGenome() {
  try {
    const s = localStorage.getItem(KEY_BEST);
    if (!s) return null;
    return JSON.parse(s);
  } catch (_) {
    return null;
  }
}
