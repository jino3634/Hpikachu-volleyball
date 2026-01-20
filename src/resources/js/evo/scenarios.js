'use strict';

/**
 * Fixed scenario seeds for fair evaluation.
 * Keep this constant across generations.
 */
// ------------------------------------------------------------
// Seed sets
// ------------------------------------------------------------
// We split seeds into TRAIN vs EVAL to reduce overfitting.
// - TRAIN_SEEDS: used for selection inside evolution.
// - EVAL_SEEDS: used only for reporting / saving "true" best.
// Keep these constant across generations.

export const TRAIN_SEEDS = Object.freeze([
  1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,
  1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,
  1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,
  1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,
  1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,
]);

export const EVAL_SEEDS = Object.freeze([
  2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
  2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,
  2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,
  2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,
  2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,
]);

// Backward-compatible name (kept for older imports)
export const DEFAULT_SEEDS = TRAIN_SEEDS;

export const DEFAULT_MATCH_OPTS = Object.freeze({
  winningScore: 11,
  maxFrames: 60 * 30, // ~30 seconds at 60fps
  decisionInterval: 3,
  // Match-start serve control:
  // - 'alternate' : P1 then P2 alternating by seed index
  // - 'p1'        : always start with P1 serve
  // - 'p2'        : always start with P2 serve
  initialServeMode: 'alternate',
});
