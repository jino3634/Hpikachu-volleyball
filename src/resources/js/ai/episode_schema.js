// src/resources/js/ai/episode_schema.js
//
// Episode(=one-point) result contract helpers.
// This file is intentionally small and dependency-free so other modules can import it safely.

export const EPISODE_SCHEMA_VERSION = 1;

/**
 * @typedef {Object} OnePointResult
 * @property {boolean} ok
 * @property {number} scoredBy
 * @property {number} loser
 * @property {string} loseReason
 * @property {number} frames
 * @property {any[]} trace
 * @property {any|null} episode
 * @property {number} [schemaVersion]
 */

/**
 * Normalize OnePointResult so downstream code can rely on required fields.
 * Never returns null/undefined.
 * @param {any} res
 * @returns {OnePointResult}
 */
export function normalizeOnePointResult(res) {
  const r = res || {};
  return {
    ok: !!r.ok,
    scoredBy: Number.isFinite(r.scoredBy) ? r.scoredBy : 0,
    loser: Number.isFinite(r.loser) ? r.loser : 0,
    loseReason: typeof r.loseReason === 'string' ? r.loseReason : 'UNKNOWN',
    frames: Number.isFinite(r.frames) ? r.frames : 0,
    trace: Array.isArray(r.trace) ? r.trace : [],
    episode: r.episode ?? null,
    schemaVersion: EPISODE_SCHEMA_VERSION,
  };
}

/**
 * Quick predicate for serve-miss style episodes (project-specific heuristic).
 * If your project guarantees serve-miss ends at 22 frames, this detects it.
 * @param {OnePointResult} res
 */
export function isServeMissEpisode(res) {
  return !!res && res.frames === 22;
}
