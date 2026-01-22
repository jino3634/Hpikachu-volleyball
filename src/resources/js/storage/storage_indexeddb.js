'use strict';

import { StorageIface } from './storage_iface.js';

/**
 * Minimal IndexedDB implementation of StorageIface.
 * Stage-2 goal: key/value JSON storage for evo best/hof.
 */
export class IndexedDBStorage extends StorageIface {
  /**
   * @param {{dbName?:string, dbVersion?:number, storeName?:string, hofStoreName?:string, historyStoreName?:string, keyPrefix?:string}} [opts]
   */
  constructor(opts = {}) {
    super();
    this.dbName = String(opts.dbName ?? 'hpika_evo_db');
    this.dbVersion = Math.max(1, (opts.dbVersion ?? 3) | 0); // B-4: bump default schema version // B-1: bump default schema version
    this.hofStoreName = String(opts.hofStoreName ?? 'hof');
    this.historyStoreName = String(opts.historyStoreName ?? 'history');
    this.storeName = String(opts.storeName ?? 'kv');
    this.keyPrefix = String(opts.keyPrefix ?? '');
    /** @type {IDBDatabase|null} */
    this._db = null;
    /** @type {Promise<void>|null} */
    this._initPromise = null;
  }

  static isSupported() {
    try {
      return typeof indexedDB !== 'undefined' && !!indexedDB;
    } catch {
      return false;
    }
  }

  /** @returns {string} */
  _k(key) {
    return this.keyPrefix + String(key ?? '');
  }

  async init() {
    if (this._db) return;
    if (this._initPromise) return this._initPromise;
    this._initPromise = new Promise((resolve, reject) => {
      let req;
      try {
        req = indexedDB.open(this.dbName, this.dbVersion);
      } catch (e) {
        reject(e);
        return;
      }

      req.onupgradeneeded = () => {
  const db = req.result;

  // KV store (used by saveJson/loadJson)
  if (!db.objectStoreNames.contains(this.storeName)) {
    db.createObjectStore(this.storeName, { keyPath: 'k' });
  }

  // (B-1) HOF row store (dev2-style)
  if (!db.objectStoreNames.contains(this.hofStoreName)) {
    const hof = db.createObjectStore(this.hofStoreName, { keyPath: 'id' });
    hof.createIndex('createdAt', 'createdAt', { unique: false });
  } else {
    // ensure index exists (best-effort)
    try {
      const tx = req.transaction;
      const hof = tx.objectStore(this.hofStoreName);
      if (!hof.indexNames.contains('createdAt')) {
        hof.createIndex('createdAt', 'createdAt', { unique: false });
      }
    } catch (e) {
      // ignore
    }
  }

  
  // (B-4) History row store (dev2-style)
  if (!db.objectStoreNames.contains(this.historyStoreName)) {
    const hist = db.createObjectStore(this.historyStoreName, { keyPath: 'id' });
    hist.createIndex('createdAt', 'createdAt', { unique: false });
    hist.createIndex('generation', 'generation', { unique: false });
  } else {
    try {
      const tx = req.transaction;
      if (tx) {
        const hist = tx.objectStore(this.historyStoreName);
        try { if (!hist.indexNames.contains('createdAt')) hist.createIndex('createdAt', 'createdAt', { unique: false }); } catch {}
        try { if (!hist.indexNames.contains('generation')) hist.createIndex('generation', 'generation', { unique: false }); } catch {}
      }
    } catch {
      // ignore
    }
  }

// Best-effort one-time migration: if KV has HOF array but row store empty, populate rows.
  try {
    const tx = req.transaction;
    const kv = tx.objectStore(this.storeName);
    const hof = tx.objectStore(this.hofStoreName);
    const getReq = kv.get(this._k('hof'));
    getReq.onsuccess = () => {
      try {
        const row = getReq.result;
        const arr = row && row.v ? JSON.parse(row.v) : null;
        if (!Array.isArray(arr) || arr.length === 0) return;

        const countReq = hof.count();
        countReq.onsuccess = () => {
          const cnt = Number(countReq.result ?? 0);
          if (cnt > 0) return;

          const now = Date.now();
          for (let i = 0; i < arr.length; i++) {
            const it = arr[i];
            const createdAt = Number(it?.savedAt ?? it?.createdAt ?? (now - i));
            const id = String(it?.id ?? `${createdAt}_${i}`);
            hof.put({ id, createdAt, item: it });
          }
        };
      } catch (e) {
        // ignore
      }
    };
  } catch (e) {
    // ignore
  }
};

      req.onsuccess = () => {
        this._db = req.result;
        // If the connection is closed by browser (versionchange), we will re-init on next call.
        this._db.onversionchange = () => {
          try { this._db?.close(); } catch {}
          this._db = null;
          this._initPromise = null;
        };
        resolve();
      };

      req.onerror = () => {
        reject(req.error ?? new Error('IndexedDB open failed'));
      };
    });
    return this._initPromise;
  }

  /**
   * @param {'readonly'|'readwrite'} mode
   * @returns {IDBObjectStore}
   */
  _store(mode) {
    if (!this._db) throw new Error('IndexedDBStorage not initialized');
    const tx = this._db.transaction([this.storeName], mode);
    return tx.objectStore(this.storeName);
  }

  /**
   * Open a transaction for an arbitrary store name.
   * @param {string} storeName
   * @param {'readonly'|'readwrite'} mode
   */
  _storeNamed(storeName, mode) {
    if (!this._db) throw new Error('IndexedDBStorage not initialized');
    const tx = this._db.transaction([storeName], mode);
    return tx.objectStore(storeName);
  }

  /**
   * @param {IDBRequest<any>} req
   * @returns {Promise<any>}
   */
  _req(req) {
    return new Promise((resolve, reject) => {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
    });
  }

  async saveJson(key, value) {
    await this.init();
    try {
      const k = this._k(key);
      const v = JSON.stringify(value);
      const store = this._store('readwrite');
      await this._req(store.put({ k, v }));
      return true;
    } catch {
      return false;
    }
  }

  async loadJson(key, fallback = null) {
    await this.init();
    try {
      const k = this._k(key);
      const store = this._store('readonly');
      const row = await this._req(store.get(k));
      if (!row || row.v == null) return fallback;
      return JSON.parse(String(row.v));
    } catch {
      return fallback;
    }
  }
/**
 * (B-1) Replace the entire HOF rows store with the provided list.
 * This does not change the KV 'hof' value; higher-level code may dual-write during transition.
 * @param {any[]} list
 * @returns {Promise<boolean>}
 */
async replaceHofRows(list) {
  await this.init();
  const arr = Array.isArray(list) ? list : [];

  return new Promise((resolve, reject) => {
    try {
      const tx = this._db.transaction([this.hofStoreName], 'readwrite');
      const store = tx.objectStore(this.hofStoreName);

      // Clear then re-add (simple and deterministic for stage B-1).
      store.clear();

      const now = Date.now();
      for (let i = 0; i < arr.length; i++) {
        const it = arr[i];
        const createdAt = Number(it?.savedAt ?? it?.createdAt ?? (now - i));
        const id = String(it?.id ?? `${createdAt}_${i}`);
        store.put({ id, createdAt, item: it });
      }

      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error ?? new Error('replaceHofRows tx error'));
      tx.onabort = () => reject(tx.error ?? new Error('replaceHofRows tx abort'));
    } catch (e) {
      reject(e);
    }
  });
}

/**
 * (B-1) Append one HOF entry as a row.
 * @param {any} entry
 * @returns {Promise<boolean>}
 */
async appendHofRow(entry) {
  await this.init();
  const it = entry ?? null;
  const createdAt = Number(it?.savedAt ?? it?.createdAt ?? Date.now());
  const id = String(it?.id ?? `${createdAt}_${Math.random().toString(16).slice(2)}`);

  return new Promise((resolve, reject) => {
    try {
      const tx = this._db.transaction([this.hofStoreName], 'readwrite');
      const store = tx.objectStore(this.hofStoreName);
      store.put({ id, createdAt, item: it });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error ?? new Error('appendHofRow tx error'));
      tx.onabort = () => reject(tx.error ?? new Error('appendHofRow tx abort'));
    } catch (e) {
      reject(e);
    }
  });
}



  
  async listHof(limit = 200) {
    await this.init();
    const lim = Math.max(0, Number(limit ?? 200) | 0);
    if (lim <= 0) return [];
  
    return new Promise((resolve, reject) => {
      try {
        const tx = this._db.transaction([this.hofStoreName], 'readonly');
        const store = tx.objectStore(this.hofStoreName);
        const idx = store.index('createdAt');
  
        /** @type {any[]} */
        const out = [];
  
        const req = idx.openCursor(null, 'prev'); // newest -> oldest
        req.onsuccess = () => {
          const cursor = req.result;
          if (!cursor) return; // done
          const v = cursor.value;
          const it = v?.item ?? null;
          if (it && typeof it === 'object') {
            try { it.__createdAt = v?.createdAt ?? 0; } catch {}
          }
          out.push(it);
          if (out.length >= lim) {
            // stop early
            resolve(out.filter(x => x != null));
            try { tx.abort(); } catch (e) {}
            return;
          }
          cursor.continue();
        };
        req.onerror = () => reject(req.error ?? new Error('listHof cursor error'));
        tx.oncomplete = () => resolve(out.filter(x => x != null));
        tx.onerror = () => reject(tx.error ?? new Error('listHof tx error'));
        tx.onabort = () => resolve(out.filter(x => x != null)); // aborted due to early stop
      } catch (e) {
        reject(e);
      }
    });
  }
  
  async pruneHof(maxKeep) {
    await this.init();
    const keep = Math.max(0, Number(maxKeep ?? 0) | 0);
  
    return new Promise((resolve, reject) => {
      try {
        const tx = this._db.transaction([this.hofStoreName], 'readwrite');
        const store = tx.objectStore(this.hofStoreName);
        const idx = store.index('createdAt');
  
        let kept = 0;
        let deleted = 0;
  
        const req = idx.openCursor(null, 'prev'); // newest first
        req.onsuccess = () => {
          const cursor = req.result;
          if (!cursor) return;
          kept++;
          if (keep <= 0 || kept > keep) {
            try {
              cursor.delete();
              deleted++;
            } catch (e) {
              // ignore per-row delete error
            }
          }
          cursor.continue();
        };
        req.onerror = () => reject(req.error ?? new Error('pruneHof cursor error'));
        tx.oncomplete = () => resolve(deleted);
        tx.onerror = () => reject(tx.error ?? new Error('pruneHof tx error'));
        tx.onabort = () => reject(tx.error ?? new Error('pruneHof tx abort'));
      } catch (e) {
        reject(e);
      }
    });
  }


  
  /**
   * (B-4) Append one history row into row store.
   * @param {any} entry
   * @returns {Promise<boolean>}
   */
  async appendHistory(entry) {
    await this.init();
    const createdAt = Number(entry?.createdAt ?? Date.now());
    const generation = Number(entry?.generation ?? entry?.gen ?? 0);
    const id = String(entry?.id ?? `${createdAt}_${Math.floor(Math.random()*1e9)}`);
    const row = { id, createdAt, generation, item: entry };
    try {
      const store = this._storeNamed(this.historyStoreName, 'readwrite');
      await this._req(store.put(row));
      return true;
    } catch {
      return false;
    }
  }

  /**
   * (B-4) List newest history rows.
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async listHistory(limit = 200) {
    await this.init();
    const lim = Math.max(0, Number(limit ?? 0) | 0);
    if (lim <= 0) return [];
    const db = this._db;
    if (!db) return [];
    return await new Promise((resolve, reject) => {
      try {
        const tx = db.transaction([this.historyStoreName], 'readonly');
        const store = tx.objectStore(this.historyStoreName);
        let idx;
        try { idx = store.index('createdAt'); } catch { idx = null; }
        const req = (idx ? idx.openCursor(null, 'prev') : store.openCursor(null, 'prev'));
        /** @type {any[]} */
        const out = [];
        req.onerror = () => reject(req.error ?? new Error('listHistory cursor failed'));
        req.onsuccess = () => {
          const cur = req.result;
          if (!cur) { resolve(out); return; }
          const row = cur.value;
          const item = row?.item ?? row;
          if (item && typeof item === 'object') {
            // attach createdAt for UI (non-invasive)
            try { item.__createdAt = Number(row?.createdAt ?? item.__createdAt ?? 0); } catch {}
          }
          out.push(item);
          if (out.length >= lim) { resolve(out); return; }
          cur.continue();
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  /**
   * (B-4) Prune old history rows, keeping only newest maxKeep.
   * @param {number} maxKeep
   * @returns {Promise<number>}
   */
  async pruneHistory(maxKeep) {
    await this.init();
    const keep = Math.max(0, Number(maxKeep ?? 0) | 0);
    const db = this._db;
    if (!db) return 0;
    return await new Promise((resolve, reject) => {
      try {
        const tx = db.transaction([this.historyStoreName], 'readwrite');
        const store = tx.objectStore(this.historyStoreName);
        let idx;
        try { idx = store.index('createdAt'); } catch { idx = null; }
        // count total
        const countReq = store.count();
        countReq.onerror = () => reject(countReq.error ?? new Error('pruneHistory count failed'));
        countReq.onsuccess = () => {
          const total = Number(countReq.result ?? 0);
          if (keep <= 0) {
            const clearReq = store.clear();
            clearReq.onerror = () => reject(clearReq.error ?? new Error('pruneHistory clear failed'));
            clearReq.onsuccess = () => resolve(total);
            return;
          }
          if (total <= keep) { resolve(0); return; }
          let seen = 0;
          let deleted = 0;
          const req = (idx ? idx.openCursor(null, 'prev') : store.openCursor(null, 'prev'));
          req.onerror = () => reject(req.error ?? new Error('pruneHistory cursor failed'));
          req.onsuccess = () => {
            const cur = req.result;
            if (!cur) { resolve(deleted); return; }
            seen++;
            if (seen > keep) {
              try { cur.delete(); deleted++; } catch {}
            }
            cur.continue();
          };
        };
        tx.onerror = () => reject(tx.error ?? new Error('pruneHistory tx error'));
        tx.onabort = () => reject(tx.error ?? new Error('pruneHistory tx abort'));
      } catch (e) {
        reject(e);
      }
    });
  }

async exportAll() {
    await this.init();
    /** @type {{kv: Record<string, any>, hofRows: any[], historyRows: any[]}} */
    const data = { kv: {}, hofRows: [], historyRows: [] };

    // Export KV (prefix-filtered)
    try {
      const store = this._storeNamed(this.storeName, 'readonly');
      const req = store.openCursor();
      await new Promise((resolve, reject) => {
        req.onerror = () => reject(req.error ?? new Error('IndexedDB kv cursor failed'));
        req.onsuccess = () => {
          const cur = req.result;
          if (!cur) { resolve(); return; }
          const row = cur.value;
          const fullKey = String(row?.k ?? '');
          if (fullKey.startsWith(this.keyPrefix)) {
            const k = fullKey.slice(this.keyPrefix.length);
            try { data.kv[k] = JSON.parse(String(row?.v ?? 'null')); } catch { /* ignore */ }
          }
          cur.continue();
        };
      });
    } catch { /* ignore */ }

    // Export HOF rows
    try {
      const store = this._storeNamed(this.hofStoreName, 'readonly');
      const req = store.openCursor();
      await new Promise((resolve, reject) => {
        req.onerror = () => reject(req.error ?? new Error('IndexedDB hof cursor failed'));
        req.onsuccess = () => {
          const cur = req.result;
          if (!cur) { resolve(); return; }
          const row = cur.value;
          if (row) data.hofRows.push(row);
          cur.continue();
        };
      });
    } catch { /* ignore */ }

    // Export History rows
    try {
      const store = this._storeNamed(this.historyStoreName, 'readonly');
      const req = store.openCursor();
      await new Promise((resolve, reject) => {
        req.onerror = () => reject(req.error ?? new Error('IndexedDB history cursor failed'));
        req.onsuccess = () => {
          const cur = req.result;
          if (!cur) { resolve(); return; }
          const row = cur.value;
          if (row) data.historyRows.push(row);
          cur.continue();
        };
      });
    } catch { /* ignore */ }

    return { format: 'hpika_storage_export', version: 2, savedAt: Date.now(), data };
  }

  async importAll(payload, opts = {}) {
    await this.init();
    try {
      const clearBefore = !!opts.clearBefore;

      // Accept both:
      // - v2: {format, version:2, data:{kv, hofRows, historyRows}}
      // - v1/legacy: {format, version:1, data:{k:v}} or plain {k:v}
      const dataObj = (payload && typeof payload === 'object') ? payload : null;
      const data = (dataObj && dataObj.data && typeof dataObj.data === 'object') ? dataObj.data : dataObj;

      /** @type {{kv?:Record<string,any>, hofRows?:any[], historyRows?:any[]}} */
      const v2 = (data && typeof data === 'object' && ('kv' in data || 'hofRows' in data || 'historyRows' in data)) ? data : null;

      /** @type {Record<string, any>} */
      const kvData = v2 ? (v2.kv && typeof v2.kv === 'object' ? v2.kv : {}) : (data && typeof data === 'object' ? data : null);
      if (!kvData || typeof kvData !== 'object') return false;

      // KV store
      const kvStore = this._storeNamed(this.storeName, 'readwrite');

      if (clearBefore) {
        // Delete keys for this prefix by cursor scan.
        const req = kvStore.openCursor();
        await new Promise((resolve, reject) => {
          req.onerror = () => reject(req.error ?? new Error('IndexedDB kv cursor failed'));
          req.onsuccess = () => {
            const cur = req.result;
            if (!cur) { resolve(); return; }
            const row = cur.value;
            const fullKey = String(row?.k ?? '');
            if (fullKey.startsWith(this.keyPrefix)) {
              try { cur.delete(); } catch { /* ignore */ }
            }
            cur.continue();
          };
        });

        // Clear row stores too (best-effort)
        try {
          const tx = this._db.transaction([this.hofStoreName, this.historyStoreName], 'readwrite');
          try { tx.objectStore(this.hofStoreName).clear(); } catch {}
          try { tx.objectStore(this.historyStoreName).clear(); } catch {}
        } catch { /* ignore */ }
      }

      // Restore KV
      for (const [k, v] of Object.entries(kvData)) {
        const fullKey = this._k(k);
        const vv = JSON.stringify(v);
        await this._req(kvStore.put({ k: fullKey, v: vv }));
      }

      // Restore row stores if v2
      if (v2) {
        // HOF rows
        try {
          const rows = Array.isArray(v2.hofRows) ? v2.hofRows : [];
          const tx = this._db.transaction([this.hofStoreName], 'readwrite');
          const store = tx.objectStore(this.hofStoreName);
          try { store.clear(); } catch {}
          for (let i = 0; i < rows.length; i++) {
            const r = rows[i];
            if (r && typeof r === 'object') {
              try { store.put(r); } catch {}
            }
          }
        } catch { /* ignore */ }

        // History rows
        try {
          const rows = Array.isArray(v2.historyRows) ? v2.historyRows : [];
          const tx = this._db.transaction([this.historyStoreName], 'readwrite');
          const store = tx.objectStore(this.historyStoreName);
          try { store.clear(); } catch {}
          for (let i = 0; i < rows.length; i++) {
            const r = rows[i];
            if (r && typeof r === 'object') {
              try { store.put(r); } catch {}
            }
          }
        } catch { /* ignore */ }
      }

      return true;
    } catch (e) {
      console.error(e);
      return false;
    }
  }

}
