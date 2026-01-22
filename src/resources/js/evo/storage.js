'use strict';

import { LocalStorageStorage } from '../storage/storage_localstorage.js';
import { IndexedDBStorage } from '../storage/storage_indexeddb.js';

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

async function ensureInit() {
  if (!__storage) await initStorage();
}

export async function saveJson(key, value) {
  await ensureInit();
  return __storage.saveJson(key, value);
}

export async function loadJson(key, fallback = null) {
  await ensureInit();
  return __storage.loadJson(key, fallback);
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
export async function saveBest(payload) {
  return saveJson(BEST_KEY, payload);
}

/**
 * @returns {Promise<{genome:any, bestWinRate:number, bestEvalWinRate?:number, bestFitness:number, generation:number, savedAt:number} | null>}
 */
export async function loadBest() {
  return loadJson(BEST_KEY, null);
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
  await ensureInit();
  const st = __storage;
  if (st && typeof st.listHof === 'function') {
    return st.listHof(limit);
  }
  const v = await loadHof();
  const lim = Math.max(0, Number(limit ?? 200) | 0);
  return (lim > 0) ? v.slice(0, lim) : [];
}

/**
 * Prune old HOF entries keeping only newest maxKeep rows.
 * @param {number} maxKeep
 * @returns {Promise<number>}
 */
export async function pruneHof(maxKeep) {
  await ensureInit();
  const st = __storage;
  if (st && typeof st.pruneHof === 'function') {
    return st.pruneHof(maxKeep);
  }
  // fallback: prune array storage
  const v = await loadHof();
  const keep = Math.max(0, Number(maxKeep ?? 0) | 0);
  if (keep <= 0) {
    await saveHof([]);
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