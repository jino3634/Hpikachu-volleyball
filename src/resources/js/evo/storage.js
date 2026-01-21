'use strict';

const KEY_PREFIX = 'evo_';

export function saveJson(key, value) {
  try {
    localStorage.setItem(KEY_PREFIX + key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function loadJson(key, fallback = null) {
  try {
    const v = localStorage.getItem(KEY_PREFIX + key);
    if (v == null) return fallback;
    return JSON.parse(v);
  } catch {
    return fallback;
  }
}

// ---------------------------------------------
// Convenience helpers for the evolution runner
// ---------------------------------------------

const BEST_KEY = 'best';
const HOF_KEY = 'hof';

/**
 * @param {{
 *  genome:any,
 *  bestWinRate:number,
 *  bestEvalWinRate?:number,
 *  bestFitness:number,
 *  generation:number,
 *  savedAt:number
 * }} payload
 */
export function saveBest(payload) {
  return saveJson(BEST_KEY, payload);
}

/**
 * @returns {{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number} | null}
 */
export function loadBest() {
  return loadJson(BEST_KEY, null);
}

// -------------------------------------------------------------
// Hall of Fame (opponent pool)
// -------------------------------------------------------------

/**
 * Load Hall of Fame list.
 * @returns {Array<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number}>}
 */
export function loadHof() {
  const v = loadJson(HOF_KEY, []);
  return Array.isArray(v) ? v : [];
}

/**
 * Save Hall of Fame list.
 * @param {Array<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number}>} list
 */
export function saveHof(list) {
  return saveJson(HOF_KEY, Array.isArray(list) ? list : []);
}
