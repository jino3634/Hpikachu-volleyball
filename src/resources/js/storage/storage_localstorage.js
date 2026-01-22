'use strict';

import { StorageIface } from './storage_iface.js';

/**
 * localStorage implementation of StorageIface.
 * Stage-1: JSON via localStorage with a fixed key prefix.
 */
export class LocalStorageStorage extends StorageIface {
  /**
   * @param {{keyPrefix?:string}} [opts]
   */
  constructor(opts = {}) {
    super();
    this.keyPrefix = String(opts.keyPrefix ?? '');
  }

  async saveJson(key, value) {
    try {
      localStorage.setItem(this.keyPrefix + key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  }

  async loadJson(key, fallback = null) {
    try {
      const v = localStorage.getItem(this.keyPrefix + key);
      if (v == null) return fallback;
      return JSON.parse(v);
    } catch {
      return fallback;
    }
  }

  async exportAll() {
    /** @type {Record<string, any>} */
    const data = {};
    try {
      const prefix = this.keyPrefix;
      for (let i = 0; i < localStorage.length; i++) {
        const fullKey = localStorage.key(i);
        if (!fullKey) continue;
        if (!fullKey.startsWith(prefix)) continue;
        const k = fullKey.slice(prefix.length);
        const v = localStorage.getItem(fullKey);
        if (v == null) continue;
        try { data[k] = JSON.parse(v); } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
    return { format: 'hpika_storage_export', version: 2, savedAt: Date.now(), data: { kv: data, hofRows: [], historyRows: [] } };
  }

  async importAll(payload, opts = {}) {
    try {
      const clearBefore = !!opts.clearBefore;
      const prefix = this.keyPrefix;
      /** @type {Record<string, any>} */
      const dataObj = (payload && typeof payload === 'object') ? payload : null;
      const data = (dataObj && dataObj.data && typeof dataObj.data === 'object') ? dataObj.data : dataObj;
      const kvData = (data && typeof data === 'object' && 'kv' in data) ? (data.kv || {}) : data;
      if (!kvData || typeof kvData !== 'object') return false;

      if (clearBefore) {
        const del = [];
        for (let i = 0; i < localStorage.length; i++) {
          const fullKey = localStorage.key(i);
          if (fullKey && fullKey.startsWith(prefix)) del.push(fullKey);
        }
        for (const k of del) localStorage.removeItem(k);
      }

      for (const [k, v] of Object.entries(kvData)) {
        localStorage.setItem(prefix + k, JSON.stringify(v));
      }
      return true;
    } catch {
      return false;
    }
  }


async replaceHofRows(list) { // eslint-disable-line no-unused-vars
  // localStorage backend has no row store; keep array storage only.
  return false;
}

async appendHofRow(entry) { // eslint-disable-line no-unused-vars
  // localStorage backend has no row store; keep array storage only.
  return false;
}



  /**
   * (B-2) List HOF entries newest->oldest.
   * localStorage backend stores HOF as a single array under key 'hof'.
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async listHof(limit = 200) {
    const v = await this.loadJson('hof', []);
    const arr = Array.isArray(v) ? v : [];
    const lim = Math.max(0, Number(limit ?? 200) | 0);
    return (lim > 0) ? arr.slice(0, lim) : [];
  }

  /**
   * (B-2) Prune HOF array to newest maxKeep items.
   * @param {number} maxKeep
   * @returns {Promise<number>}
   */
  async pruneHof(maxKeep) {
    const v = await this.loadJson('hof', []);
    const arr = Array.isArray(v) ? v : [];
    const keep = Math.max(0, Number(maxKeep ?? 0) | 0);
    if (keep <= 0) {
      const del = arr.length;
      await this.saveJson('hof', []);
      return del;
    }
    if (arr.length <= keep) return 0;
    const pruned = arr.slice(0, keep);
    const del = arr.length - pruned.length;
    await this.saveJson('hof', pruned);
    return del;
  }

  /**
   * (B-4) Append history entry into a KV-backed array.
   * @param {any} entry
   * @returns {Promise<boolean>}
   */
  async appendHistory(entry) {
    const arr = await this.loadJson('history', []);
    const list = Array.isArray(arr) ? arr : [];
    const createdAt = Number(entry?.createdAt ?? Date.now());
    const item = { ...(entry || {}), createdAt };
    list.unshift(item);
    await this.saveJson('history', list);
    return true;
  }

  /**
   * (B-4) List newest history entries.
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async listHistory(limit = 200) {
    const v = await this.loadJson('history', []);
    const arr = Array.isArray(v) ? v : [];
    const lim = Math.max(0, Number(limit ?? 0) | 0);
    return (lim > 0) ? arr.slice(0, lim) : [];
  }

  /**
   * (B-4) Prune history array to newest maxKeep items.
   * @param {number} maxKeep
   * @returns {Promise<number>}
   */
  async pruneHistory(maxKeep) {
    const v = await this.loadJson('history', []);
    const arr = Array.isArray(v) ? v : [];
    const keep = Math.max(0, Number(maxKeep ?? 0) | 0);
    if (keep <= 0) {
      const del = arr.length;
      await this.saveJson('history', []);
      return del;
    }
    if (arr.length <= keep) return 0;
    const pruned = arr.slice(0, keep);
    const del = arr.length - pruned.length;
    await this.saveJson('history', pruned);
    return del;
  }

}
