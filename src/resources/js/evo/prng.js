'use strict';

/**
 * Deterministic PRNG for fair evaluation.
 * Returns a function that yields floats in [0,1).
 * @param {number} seed
 * @returns {() => number}
 */
export function makeXorShift32(seed) {
  let x = (seed | 0) || 1;
  return function rng() {
    // xorshift32
    x ^= (x << 13);
    x |= 0;
    x ^= (x >>> 17);
    x |= 0;
    x ^= (x << 5);
    x |= 0;
    // to [0,1)
    return ((x >>> 0) / 4294967296);
  };
}
