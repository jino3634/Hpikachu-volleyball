/**
 * Deterministic PRNG helpers for headless evaluation.
 *
 * This is intentionally simple (fast + reproducible).
 */
'use strict';

/**
 * Create a xorshift32 PRNG.
 * @param {number} seed
 * @returns {() => number} rng that returns float in [0,1)
 */
export function createXorshift32(seed) {
  let x = (seed | 0) || 123456789;
  // Avoid zero state
  if (x === 0) x = 123456789;

  return function rng() {
    // xorshift32
    x ^= (x << 13);
    x ^= (x >>> 17);
    x ^= (x << 5);
    // Convert to [0,1)
    // >>> 0 makes it uint32
    return ((x >>> 0) / 4294967296);
  };
}
