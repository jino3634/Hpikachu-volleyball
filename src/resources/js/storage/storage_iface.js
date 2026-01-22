'use strict';

/**
 * Storage interface (contract) used by higher-level code.
 * Implementations can use localStorage, IndexedDB, server APIs, etc.
 *
 * All methods are async so implementations like IndexedDB can be used.
 */
export class StorageIface {
  /**
   * Optional init hook (e.g., open IndexedDB).
   * @returns {Promise<void>}
   */
  async init() {}

  /**
   * Save a JSON-serializable value under a key.
   * @param {string} key
   * @param {any} value
   * @returns {Promise<boolean>}
   */
  async saveJson(key, value) { // eslint-disable-line no-unused-vars
    throw new Error('StorageIface.saveJson not implemented');
  }

  /**
   * Load a JSON value by key.
   * @template T
   * @param {string} key
   * @param {T} [fallback]
   * @returns {Promise<T>}
   */
  async loadJson(key, fallback = null) { // eslint-disable-line no-unused-vars
    throw new Error('StorageIface.loadJson not implemented');
  }

  /**
   * Export all stored keys for this backend/prefix.
   * @returns {Promise<{format:string, version:number, savedAt:number, data:Record<string, any>}>}
   */
  async exportAll() {
    throw new Error('StorageIface.exportAll not implemented');
  }

  /**
   * Import values (typically previously exported).
   * Implementations should accept either {data:{...}} or a plain object map.
   * @param {any} payload
   * @param {{clearBefore?:boolean}} [opts]
   * @returns {Promise<boolean>}
   */
  async importAll(payload, opts = {}) { // eslint-disable-line no-unused-vars
    throw new Error('StorageIface.importAll not implemented');
  }

  // ------------------------------------------------------------
  // HOF (row store) - optional in some backends
  // ------------------------------------------------------------

  /**
   * Replace all HOF rows (clear then insert).
   * @param {any[]} rows
   * @returns {Promise<boolean>}
   */
  async replaceHofRows(rows) { // eslint-disable-line no-unused-vars
    return false;
  }

  /**
   * Append one HOF row.
   * @param {any} entry
   * @returns {Promise<boolean>}
   */
  async appendHofRow(entry) { // eslint-disable-line no-unused-vars
    return false;
  }

  /**
   * List HOF rows newest -> oldest.
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async listHof(limit = 200) { // eslint-disable-line no-unused-vars
    return [];
  }

  /**
   * Prune old HOF rows, keep newest maxKeep.
   * @param {number} maxKeep
   * @returns {Promise<number>} deleted count
   */
  async pruneHof(maxKeep) { // eslint-disable-line no-unused-vars
    return 0;
  }

  // ------------------------------------------------------------
  // History (row store)
  // ------------------------------------------------------------

  /**
   * Append a history row.
   * @param {any} entry
   * @returns {Promise<boolean>}
   */
  async appendHistory(entry) { // eslint-disable-line no-unused-vars
    return false;
  }

  /**
   * List history rows newest -> oldest.
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async listHistory(limit = 200) { // eslint-disable-line no-unused-vars
    return [];
  }

  /**
   * Prune history rows, keep newest maxKeep.
   * @param {number} maxKeep
   * @returns {Promise<number>} deleted count
   */
  async pruneHistory(maxKeep) { // eslint-disable-line no-unused-vars
    return 0;
  }
}
