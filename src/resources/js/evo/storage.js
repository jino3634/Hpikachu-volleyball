'use strict';

import { LocalStorageStorage } from '../storage/storage_localstorage.js';
import { IndexedDBStorage } from '../storage/storage_indexeddb.js';
import { evoLogger } from './logger.js';

const KEY_PREFIX = 'evo_';

/** @type {import('../storage/storage_iface.js').StorageIface|null} */
let __storage = null;

/**
 * Initialize storage backend.
 * Stage-2 default: IndexedDB, with localStorage fallback.
 */
/** @returns {Promise<import('../storage/storage_iface.js').StorageIface>} */
export async function initStorage() {
  if (__storage) {
    if (__storage.init) await __storage.init();
    return __storage;
  }

  if (IndexedDBStorage.isSupported()) {
    __storage = new IndexedDBStorage({
      dbName: 'hpika_evo_db',
      dbVersion: 3, // B-4: include HOF+History row stores
      storeName: 'kv',
      keyPrefix: KEY_PREFIX,
    });
  } else {
    __storage = new LocalStorageStorage({ keyPrefix: KEY_PREFIX });
  }

  try {
    await __storage.init();
  } catch {
    // Fallback to localStorage if IDB init fails (private mode, quota, etc.)
    __storage = new LocalStorageStorage({ keyPrefix: KEY_PREFIX });
    await __storage.init();
  }
  return __storage;
}


function _approxBytesOfJson(value) {
  try {
    const s = JSON.stringify(value);
    if (typeof TextEncoder !== 'undefined') {
      return (new TextEncoder().encode(s)).length;
    }
    return s.length;
  } catch {
    return -1;
  }
}

async function ensureInit() {
  if (!__storage) await initStorage();
}

export async function saveJson(key, value) {
  await ensureInit();
  try {
    const ok = await __storage.saveJson(key, value);
    const bytes = evoLogger.wouldLog('storage', 'INFO') ? _approxBytesOfJson(value) : undefined;
    evoLogger.log('storage', 'INFO', { op: 'saveJson', key: String(key||''), ok: !!ok, bytes });
    return ok;
  } catch (e) {
    evoLogger.log('storage', 'WARN', { op: 'saveJson', key: String(key||''), err: String(e) });
    throw e;
  }
}

export async function loadJson(key, fallback = null) {
  await ensureInit();
  try {
    const v = await __storage.loadJson(key, fallback);
    evoLogger.log('storage', 'INFO', { op: 'loadJson', key: String(key||''), hit: v != null });
    return v;
  } catch (e) {
    evoLogger.log('storage', 'WARN', { op: 'loadJson', key: String(key||''), err: String(e) });
    return fallback;
  }
}

// ---------------------------------------------
// Convenience helpers for the evolution runner
// ---------------------------------------------

const BEST_KEY = 'best';
const HOF_KEY = 'hof';
const OPP_STATS_KEY = 'oppstats';
const REPLAYS_KEY = 'replays_v1';

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
export async function saveBest(payload) {
  return saveJson(BEST_KEY, payload);
}

/**
 * @returns {Promise<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number} | null>}
 */
export async function loadBest() {
  return loadJson(BEST_KEY, null);
}

/**
 * Save opponent-mode recent outcomes (for winrate last-N).
 * @param {{self:number[], baseline:number[], physics:number[]}} payload
 */
export async function saveOppStats(payload) {
  return saveJson(OPP_STATS_KEY, payload);
}

/**
 * Load opponent-mode recent outcomes (for winrate last-N).
 * @returns {Promise<{self:number[], baseline:number[], physics:number[]} | null>}
 */
export async function loadOppStats() {
  return loadJson(OPP_STATS_KEY, null);
}

// -------------------------------------------------------------
// Replays (recent 10 games / recent 3 wins)
// Stored as JSON with base64-packed int8 arrays.
// -------------------------------------------------------------

function _u8ToB64(u8) {
  try {
    if (!u8 || !u8.length) return '';
    let s = '';
    const CHUNK = 0x8000; // 32k
    for (let i = 0; i < u8.length; i += CHUNK) {
      const sub = u8.subarray(i, i + CHUNK);
      // Convert to binary string in chunks.
      let part = '';
      for (let j = 0; j < sub.length; j++) part += String.fromCharCode(sub[j]);
      s += part;
    }
    return btoa(s);
  } catch {
    return '';
  }
}

function _b64ToI8(b64) {
  try {
    if (!b64) return new Int8Array(0);
    const bin = atob(String(b64));
    const len = bin.length;
    const u8 = new Uint8Array(len);
    for (let i = 0; i < len; i++) u8[i] = bin.charCodeAt(i) & 255;
    return new Int8Array(u8.buffer);
  } catch {
    return new Int8Array(0);
  }
}

function _serializeReplayItem(item) {
  try {
    const r = item && item.replay ? item.replay : null;
    const packed = r && r.packed ? r.packed : null;
    const u8 = packed ? new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength) : null;
    return {
      t: item.t,
      mode: item.mode,
      seed: item.seed,
      winner: item.winner,
      scoreP1: item.scoreP1,
      scoreP2: item.scoreP2,
      frames: item.frames,
      replay: r ? {
        frames: r.frames,
        packedB64: u8 ? _u8ToB64(u8) : '',
      } : null,
    };
  } catch {
    return null;
  }
}

function _deserializeReplayItem(obj) {
  try {
    if (!obj || !obj.replay) return null;
    const packedI8 = _b64ToI8(obj.replay.packedB64);
    return {
      t: Number(obj.t || 0),
      mode: obj.mode || 'self',
      seed: Number(obj.seed || 0),
      winner: Number(obj.winner || 0),
      scoreP1: Number(obj.scoreP1 || 0),
      scoreP2: Number(obj.scoreP2 || 0),
      frames: Number(obj.frames || obj.replay.frames || 0),
      replay: {
        frames: Number(obj.replay.frames || 0),
        packed: packedI8,
      },
    };
  } catch {
    return null;
  }
}

/**
 * Save recent replays (ring buffers).
 * @param {{recentGames:any[], recentWins:any[]}} payload
 * @returns {Promise<boolean>}
 */
export async function saveReplays(payload) {
  try {
    const recentGames = Array.isArray(payload?.recentGames) ? payload.recentGames : [];
    const recentWins = Array.isArray(payload?.recentWins) ? payload.recentWins : [];
    const ser = {
      v: 1,
      savedAt: Date.now(),
      recentGames: recentGames.map(_serializeReplayItem).filter(Boolean),
      recentWins: recentWins.map(_serializeReplayItem).filter(Boolean),
    };
    await saveJson(REPLAYS_KEY, ser);
    return true;
  } catch {
    return false;
  }
}

/**
 * Load recent replays (ring buffers).
 * @returns {Promise<{recentGames:any[], recentWins:any[]} | null>}
 */
export async function loadReplays() {
  try {
    const data = await loadJson(REPLAYS_KEY, null);
    if (!data || (!Array.isArray(data.recentGames) && !Array.isArray(data.recentWins))) return null;
    const recentGames = Array.isArray(data.recentGames) ? data.recentGames.map(_deserializeReplayItem).filter(Boolean) : [];
    const recentWins = Array.isArray(data.recentWins) ? data.recentWins.map(_deserializeReplayItem).filter(Boolean) : [];
    return { recentGames, recentWins };
  } catch {
    return null;
  }
}




// -------------------------------------------------------------
// Hall of Fame (opponent pool)
// -------------------------------------------------------------

/**
 * Load Hall of Fame list.
 * @returns {Promise<Array<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number}>>}
 */
export async function loadHof() {
  const v = await loadJson(HOF_KEY, []);
  return Array.isArray(v) ? v : [];
}

/**
 * Save Hall of Fame list.
 * @param {Array<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number}>} list
 */
export async function saveHof(list) {
  const arr = Array.isArray(list) ? list : [];
  // Keep legacy array storage for now (Stage B-1 keeps reads unchanged)
  await saveJson(HOF_KEY, arr);

  // (B-1) Also write HOF as rows when backend supports it.
  try {
    await ensureInit();
    const st = __storage;
    if (st && typeof st.replaceHofRows === 'function') {
      await st.replaceHofRows(arr);
    }
  } catch (e) {
    // ignore: row-store is optional in this stage
  }
  return true;
}


/**
 * List HOF entries (newest -> oldest) from row-store when available.
 * @param {number} [limit]
 * @returns {Promise<any[]>}
 */
export async function listHof(limit = 200) {
  // HOF는 "best-first" 성격이라 row-store(createdAt) 정렬을 소스로 쓰면 안 됨.
  const v = await loadHof(); // HOF_KEY JSON 배열(이미 runner가 정렬해 저장함)
  const lim = Math.max(0, Number(limit ?? 200) | 0);
  return (lim > 0) ? v.slice(0, lim) : [];
}

/**
 * Prune old HOF entries keeping only newest maxKeep rows.
 * @param {number} maxKeep
 * @returns {Promise<number>}
 */
export async function pruneHof(maxKeep) {
  // HOF는 newest가 아니라 best를 유지해야 함.
  const v = await loadHof();
  const keep = Math.max(0, Number(maxKeep ?? 0) | 0);

  if (keep <= 0) {
    await saveHof([]);       // saveHof가 row 미러링까지 같이 처리
    return v.length;
  }
  if (v.length <= keep) return 0;

  await saveHof(v.slice(0, keep));
  return v.length - keep;
}


/**
 * Export all evo-related persisted data (best/hof and any future keys).
 * @returns {Promise<{format:string, version:number, savedAt:number, data:Record<string, any>}>}
 */

// --------------------
// (B-4) History helpers (row store in IndexedDB, array in localStorage fallback)
// --------------------
/**
 * Append one history entry.
 * @param {any} entry
 * @returns {Promise<boolean>}
 */
export async function appendHistory(entry) {
  await ensureInit();
  const st = __storage;
  return await st.appendHistory(entry);
}

export async function listHistory(limit = 200) {
  await ensureInit();
  const st = __storage;
  return st.listHistory(limit);
}

export async function pruneHistory(maxKeep) {
  await ensureInit();
  const st = __storage;
  return st.pruneHistory(maxKeep);
}

export async function exportEvoData() {
  await initStorage();
  if (!__storage || !__storage.exportAll) {
    return { format: 'hpika_storage_export', version: 1, savedAt: Date.now(), data: {} };
  }
  return __storage.exportAll();
}

/**
 * Import evo persisted data created by exportEvoData().
 * @param {any} payload
 * @param {{clearBefore?:boolean}} [opts]
 * @returns {Promise<boolean>}
 */
export async function importEvoData(payload, opts = {}) {
  await initStorage();
  if (!__storage || !__storage.importAll) return false;
  return __storage.importAll(payload, opts);
}