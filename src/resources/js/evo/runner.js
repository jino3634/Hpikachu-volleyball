'use strict';

import { EVO_DEFAULTS } from './config.js';
import { defaultGenome } from './policy_weighted.js';
import { evolveOneGeneration, initPopulation } from './evolver_ga.js';
import { evaluateGenome } from './evolver_ga.js';
import { initStorage, loadBest, saveBest, loadHof, saveHof, appendHistory, loadOppStats, saveOppStats, loadReplays, saveReplays } from './storage.js';
import { makeXorShift32 } from './prng.js';
import { evoLogger } from './logger.js';


const PHYSICS_OPPONENT = Object.freeze({ __opp: 'physics' });
const REPLAY_RECENT_GAMES_N = 10;
const REPLAY_RECENT_WINS_N = 3;

let __saveReplaysTimer = null;
function _scheduleSaveReplays() {
  try {
    if (__saveReplaysTimer) clearTimeout(__saveReplaysTimer);
    __saveReplaysTimer = setTimeout(() => {
      __saveReplaysTimer = null;
      try { saveReplays(state.replays); } catch {}
    }, 250);
  } catch {}
}



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
  // Opponent-mode recent outcomes (for winrate last-N).
  oppStats: {
    self: [],
    baseline: [],
    physics: [],
  },
  oppStatsN: 100,
  oppStatsSummary: {
    self: { n: 0, winRate: 0 },
    baseline: { n: 0, winRate: 0 },
    physics: { n: 0, winRate: 0 },
  },
  // Recent replays captured during opponent-mode evaluation (memory only).
  replays: {
    recentGames: [],
    recentWins: [],
  }
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

  // Ensure storage is ready (IndexedDB open, etc.)
  await initStorage();

  // Load persisted replays (recent lists) if available.
  try {
    const rep = await loadReplays();
    if (rep) {
      state.replays.recentGames = Array.isArray(rep.recentGames) ? rep.recentGames : [];
      state.replays.recentWins = Array.isArray(rep.recentWins) ? rep.recentWins : [];
    }
  } catch {}

  const cfg = { ...EVO_DEFAULTS, ...(opts || {}) };
  state.config = cfg;
  state.rng = makeXorShift32(Number(cfg.evoSeed ?? 1337) | 0);

  // Opponent mode selected by UI.
    const opponentMode = normalizeOpponentMode(cfg.opponentMode);

  // Emit a single meta/header line for diagnostics (best-effort).
  try {
    evoLogger.emitMetaOnce({
      app: 'Hpikachu-volleyball',
      mode: 'evo',
      t: Date.now(),
      ua: (typeof navigator !== 'undefined' ? navigator.userAgent : ''),
      href: (typeof location !== 'undefined' ? location.href : ''),
      cfg: {
        populationSize: cfg.populationSize,
        eliteFraction: cfg.eliteFraction,
        mutationRate: cfg.mutationRate,
        mutationSigma: cfg.mutationSigma,
        winningScore: cfg.winningScore,
        maxFrames: cfg.maxFrames,
        decisionInterval: cfg.decisionInterval,
        opponentMode,
        trainSeeds: Array.isArray(cfg.trainSeeds) ? cfg.trainSeeds.length : undefined,
        evalSeeds: Array.isArray(cfg.evalSeeds) ? cfg.evalSeeds.length : undefined,
        splitSeedsAcrossOpponents: !!cfg.splitSeedsAcrossOpponents,
        initialServeMode: cfg.initialServeMode,
        hofSize: cfg.hofSize,
      },
    });
  } catch {}

  // Load saved best if exists
  const saved = await loadBest();
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
  state.hof = await loadHof();

  // Load opponent-mode recent outcomes (for winrate last-N)
  try {
    const loadedOpp = await loadOppStats();
    if (loadedOpp && typeof loadedOpp === 'object') {
      for (const k of ['self','baseline','physics']) {
        const arr = loadedOpp[k];
        if (Array.isArray(arr)) {
          state.oppStats[k] = arr.map((v) => (v | 0));
          if (state.oppStats[k].length > state.oppStatsN) state.oppStats[k].splice(0, state.oppStats[k].length - state.oppStatsN);
        }
      }
    }
  } catch {}
  // Recompute summaries
  for (const k of ['self','baseline','physics']) {
    const buf = state.oppStats[k];
    let w = 0;
    for (let i = 0; i < buf.length; i++) if (buf[i] > 0) w++;
    const n = buf.length;
    state.oppStatsSummary[k] = { n, winRate: n ? (w / n) : 0 };
  }

  // Init population around best
  const base = state.bestGenome || defaultGenome();
  state.population = initPopulation(cfg.populationSize, {
    baseGenome: base,
    sigma: cfg.mutationSigma,
  });

  state.running = true;
  emit(onUpdate, { ...state, event: 'started' });

  // Fixed opponent snapshot used for evaluation/training.
  // - self: previous best snapshot (updated each generation)
  // - baseline: default baseline genome (fixed)
  // - physics: (reserved) fallback to baseline for now
  let baselineOpp = resolveOpponent(opponentMode, state.bestGenome);

  try {
    while (state.running) {
      const t0 = Date.now();

      const res = evolveOneGeneration(state.population, {
        seeds: cfg.trainSeeds,
        opponentGenomes: buildOpponentPool(baselineOpp, null, state.hof, cfg.hofSize),
        rng: state.rng,
        eliteFraction: cfg.eliteFraction,
        mutationRate: cfg.mutationRate,
        mutationSigma: cfg.mutationSigma,
        winningScore: cfg.winningScore,
        maxFrames: cfg.maxFrames,
        decisionInterval: cfg.decisionInterval,
        splitSeedsAcrossOpponents: !!cfg.splitSeedsAcrossOpponents,
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
          opponentGenomes: buildOpponentPool(baselineOpp, null, state.hof, cfg.hofSize),
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
        await saveBest({
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
        await saveHof(state.hof);
        savedNow = true;
      }

      // Update opponent-mode recent-100 stats (internal only; UI will display later).
      try {
        const replayMatches = [];
        const statsEval = evaluateGenome(res.best.genome, {
          seeds: cfg.trainSeeds,
          opponentGenomes: [baselineOpp],
          winningScore: cfg.winningScore,
          maxFrames: cfg.maxFrames,
          decisionInterval: cfg.decisionInterval,
          splitSeedsAcrossOpponents: false,
          initialServeMode: /** @type {'alternate'|'p1'|'p2'} */ (cfg.initialServeMode || 'alternate'),
          collectOutcomes: true,
          collectReplay: true,
          onMatch: (r) => { try { replayMatches.push(r); } catch {} },
        });
        _pushReplays(opponentMode, replayMatches);
        _pushOutcomes(opponentMode, statsEval.outcomes, state.oppStatsN);
        try { await saveOppStats(state.oppStats); } catch {}
      } catch {}

      // ✅ Self-play: for next generation, use the latest best as the fixed opponent snapshot.
      // Baseline mode keeps using the fixed default opponent.
      // Physics mode keeps using the physics marker.
      if (opponentMode === 'self') {
        baselineOpp = cloneGenome(state.bestGenome);
      } else if (opponentMode === 'physics') {
        baselineOpp = PHYSICS_OPPONENT;
      }

      // (B-4) History row: record generation metrics (newest-first list via IndexedDB index)
      try {
        await appendHistory({
          createdAt: Date.now(),
          generation: state.generation,
          bestWinRate: Number(bestWin ?? 0),
          bestEvalWinRate: Number(bestEval ?? bestWin ?? 0),
          bestFitness: Number(state.bestFitness ?? 0),
          savedNow: !!savedNow,
        });
      } catch {
        // ignore
      }


      const elapsedMs = Date.now() - t0;

      // Stage 1 diagnostic: generation summary (enable via window.__evoLogger.setEnabled(true)).
      try {
        const oppPool = buildOpponentPool(baselineOpp, null, state.hof, cfg.hofSize);
        const oppPoolPhysics = oppPool.filter((g) => (g && g.__opp === 'physics')).length;
        const bestMeta = res.best?.eval?.meta || null;
        const bestPerOpp = res.best?.eval?.perOpp || null;
        evoLogger.log('generation', 'INFO', {
          generation: state.generation,
          opponentMode,
          savedNow: !!savedNow,

          // Best of this generation (train/eval)
          bestWinRate: Number(bestWin ?? 0),
          bestEvalWinRate: Number(bestEvalWinRate ?? 0),
          bestFitness: Number(bestFit ?? 0),

          // Population stats (diversity/collapse diagnostics)
          avgWinRate: Number(res.avgWinRate ?? 0),
          stdWinRate: Number(res.stdWinRate ?? 0),
          avgFitness: Number(res.avgFitness ?? 0),
          stdFitness: Number(res.stdFitness ?? 0),
          eliteCount: Number(res.eliteCount ?? 0),

          // Opponent pool / seed split diagnostics
          oppPoolSize: oppPool.length,
          oppPoolPhysics,
          splitSeedsAcrossOpponents: !!cfg.splitSeedsAcrossOpponents,
          trainSeedCount: Array.isArray(cfg.trainSeeds) ? (cfg.trainSeeds.length | 0) : 0,
          evalSeedCount: Array.isArray(cfg.evalSeeds) ? (cfg.evalSeeds.length | 0) : 0,
          bestEvalMeta: bestMeta,
          bestPerOpp,

          // Recent opponent-mode last-N (for UI later)
          oppStatsSummary: state.oppStatsSummary,

          elapsedMs: Number(elapsedMs ?? 0),
        });
      } catch {}

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
 * @param {any} mode
 * @returns {'self'|'baseline'|'physics'}
 */
function normalizeOpponentMode(mode) {
  const v = String(mode || '').toLowerCase();
  if (v === 'baseline') return 'baseline';
  if (v === 'physics') return 'physics';
  return 'self';
}

/**
 * @param {'self'|'baseline'|'physics'} mode
 * @param {any} bestGenome
 * @returns {any}
 */
function resolveOpponent(mode, bestGenome) {
  if (mode === 'baseline') {
    return cloneGenome(defaultGenome());
  }
  if (mode === 'physics') {
    return PHYSICS_OPPONENT;
  }
  // self
  return cloneGenome(bestGenome || defaultGenome());
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
 * Push match outcomes into ring buffer for the selected opponent mode.
 * outcome: 1 win, -1 loss, 0 draw.
 */
function _pushOutcomes(mode, outcomes, maxN) {
  if (!outcomes || !outcomes.length) return;
  const key = (mode === 'baseline' || mode === 'physics') ? mode : 'self';
  const buf = state.oppStats[key];
  const lim = Math.max(1, (maxN ?? state.oppStatsN ?? 100) | 0);
  for (let i = 0; i < outcomes.length; i++) {
    buf.push((outcomes[i] | 0));
  }
  if (buf.length > lim) buf.splice(0, buf.length - lim);
  // recompute summary
  let w = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] > 0) w++;
  const n = buf.length;
  state.oppStatsSummary[key] = { n, winRate: n ? (w / n) : 0 };
}

/**
 * Push replay items into recent ring buffers.
 * Keeps:
 *  - recentGames: last N games (all outcomes)
 *  - recentWins: last N wins by P1 (training AI)
 */
function _pushReplays(mode, matchResults) {
  if (!matchResults || !matchResults.length) return;
  const key = (mode === 'baseline' || mode === 'physics') ? mode : 'self';
  const gamesBuf = state.replays.recentGames;
  const winsBuf = state.replays.recentWins;
  let changed = false;

  for (let i = 0; i < matchResults.length; i++) {
    const r = matchResults[i];
    if (!r || !r.replay) continue;
    const item = {
      t: Date.now(),
      mode: key,
      seed: r.seed,
      winner: r.winner,
      scoreP1: r.scoreP1,
      scoreP2: r.scoreP2,
      frames: r.frames,
      replay: r.replay,
    };
    gamesBuf.push(item);
    changed = true;
    if (gamesBuf.length > REPLAY_RECENT_GAMES_N) gamesBuf.splice(0, gamesBuf.length - REPLAY_RECENT_GAMES_N);

    if ((r.winner | 0) === 1) {
      winsBuf.push(item);
      if (winsBuf.length > REPLAY_RECENT_WINS_N) winsBuf.splice(0, winsBuf.length - REPLAY_RECENT_WINS_N);
    }
  }
  if (changed) _scheduleSaveReplays();
}

/**
 * @returns {{recentGames:any[], recentWins:any[]}}
 */
export function getRecentReplays() {
  return state.replays;
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

// Self-play baseline opponent: deep-clone to avoid mutation side-effects.
function cloneGenome(g) {
  return g ? JSON.parse(JSON.stringify(g)) : g;
}
