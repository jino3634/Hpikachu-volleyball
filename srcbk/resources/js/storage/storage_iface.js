// storage_iface.js
'use strict';

/**
 * 저장소 인터페이스(추상). 구현체만 교체 가능하게.
 * - IndexedDB 구현은 storage_indexeddb.js
 * - 나중에 파일/로컬/서버 등으로 교체 가능
 */

export class StorageIface {
  async init() { throw new Error('not implemented'); }

  /** @returns {Promise<any>} */
  async getMeta() { throw new Error('not implemented'); }

  /** @param {any} meta */
  async setMeta(meta) { throw new Error('not implemented'); }

  /** @param {import('../ai/rl_schema.js').Episode} episode */
  async appendEpisode(episode) { throw new Error('not implemented'); }

  /** @returns {Promise<number>} */
  async countEpisodes() { throw new Error('not implemented'); }

  /**
   * @param {{limit?:number, offset?:number}} [opts]
   * @returns {Promise<import('../ai/rl_schema.js').Episode[]>}
   */
  async listEpisodes(opts) { throw new Error('not implemented'); }

  /** @param {string} key */
  async getCheckpoint(key) { throw new Error('not implemented'); }

  /** @param {string} key @param {any} value */
  async setCheckpoint(key, value) { throw new Error('not implemented'); }

  async clearAll() { throw new Error('not implemented'); }

  /** @returns {Promise<any>} */
  async exportAll() { throw new Error('not implemented'); }

  /** @param {any} data */
  async importAll(data) { throw new Error('not implemented'); }

  /** @param {any} replay */
  async appendReplay(replay) { throw new Error('not implemented'); }

  /** @returns {Promise<any[]>} */
  async listReplays(opts = {}) { throw new Error('not implemented'); }

  /** @param {number} maxKeep */
  async pruneReplays(maxKeep) { throw new Error('not implemented'); }

}
