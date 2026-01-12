// storage_indexeddb.js
'use strict';

import { StorageIface } from './storage_iface.js';

const DB_NAME = 'pika_rl_db';
const DB_VERSION = 2;

const STORE_META = 'meta';
const STORE_EPISODES = 'episodes';
const STORE_CHECKPOINTS = 'checkpoints';
const STORE_REPLAYS = 'replays';

function reqToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

async function openDB() {
  const req = indexedDB.open(DB_NAME, DB_VERSION);

  req.onupgradeneeded = (e) => {
    const db = req.result;

    if (!db.objectStoreNames.contains(STORE_META)) {
      db.createObjectStore(STORE_META); // key-value
    }
    if (!db.objectStoreNames.contains(STORE_EPISODES)) {
      const os = db.createObjectStore(STORE_EPISODES, { keyPath: 'id' });
      os.createIndex('createdAt', 'createdAt', { unique: false });
    }
    if (!db.objectStoreNames.contains(STORE_CHECKPOINTS)) {
      db.createObjectStore(STORE_CHECKPOINTS); // key-value
    }
    if (!db.objectStoreNames.contains(STORE_REPLAYS)) {
    const os = db.createObjectStore(STORE_REPLAYS, { keyPath: 'id' });
    os.createIndex('createdAt', 'createdAt', { unique: false });
    }
  };

  const db = await reqToPromise(req);
  return db;
}

export class IndexedDBStorage extends StorageIface {
  constructor() {
    super();
    this.db = null;
  }

  async init() {
    this.db = await openDB();
    // meta 기본값 보장
    const meta = await this.getMeta();
    if (!meta) {
      await this.setMeta({
        createdAt: Date.now(),
        version: 1,
        totalEpisodes: 0,
        notes: '',
      });
    }
  }

  _assert() {
    if (!this.db) throw new Error('IndexedDBStorage not initialized. Call init() first.');
  }

  async getMeta() {
    this._assert();
    const tx = this.db.transaction([STORE_META], 'readonly');
    const os = tx.objectStore(STORE_META);
    const meta = await reqToPromise(os.get('meta'));
    await txDone(tx);
    return meta ?? null;
  }

  async setMeta(meta) {
    this._assert();
    const tx = this.db.transaction([STORE_META], 'readwrite');
    const os = tx.objectStore(STORE_META);
    os.put(meta, 'meta');
    await txDone(tx);
  }

  async appendEpisode(episode) {
    this._assert();
    const tx = this.db.transaction([STORE_EPISODES, STORE_META], 'readwrite');
    const epOS = tx.objectStore(STORE_EPISODES);
    const metaOS = tx.objectStore(STORE_META);

    // episode 저장
    epOS.put(episode);

    // meta 갱신(대략 카운터)
    const meta = (await reqToPromise(metaOS.get('meta'))) ?? {
      createdAt: Date.now(),
      version: 1,
      totalEpisodes: 0,
      notes: '',
    };
    meta.totalEpisodes = (meta.totalEpisodes | 0) + 1;
    meta.lastEpisodeAt = episode.createdAt;
    metaOS.put(meta, 'meta');

    await txDone(tx);
  }

  async countEpisodes() {
    this._assert();
    const tx = this.db.transaction([STORE_EPISODES], 'readonly');
    const os = tx.objectStore(STORE_EPISODES);
    const countReq = os.count();
    const n = await reqToPromise(countReq);
    await txDone(tx);
    return n | 0;
  }

  async listEpisodes(opts = {}) {
    this._assert();
    const limit = Math.max(0, (opts.limit ?? 100) | 0);
    const offset = Math.max(0, (opts.offset ?? 0) | 0);

    const tx = this.db.transaction([STORE_EPISODES], 'readonly');
    const os = tx.objectStore(STORE_EPISODES);
    const idx = os.index('createdAt');

    const out = [];
    let skipped = 0;

    // createdAt 순으로
    await new Promise((resolve, reject) => {
      const cursorReq = idx.openCursor();
      cursorReq.onerror = () => reject(cursorReq.error);
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (!cursor) return resolve();

        if (skipped < offset) {
          skipped++;
          cursor.continue();
          return;
        }

        out.push(cursor.value);
        if (limit > 0 && out.length >= limit) return resolve();

        cursor.continue();
      };
    });

    await txDone(tx);
    return out;
  }

  async appendReplay(replay) {
    this._assert();
    const tx = this.db.transaction([STORE_REPLAYS], 'readwrite');
    const os = tx.objectStore(STORE_REPLAYS);
    os.put(replay);
    await txDone(tx);
  }

  async listReplays(opts = {}) {
    this._assert();
    const limit = Math.max(0, (opts.limit ?? 10) | 0);
    const offset = Math.max(0, (opts.offset ?? 0) | 0);

    const tx = this.db.transaction([STORE_REPLAYS], 'readonly');
    const os = tx.objectStore(STORE_REPLAYS);
    const idx = os.index('createdAt');

    const out = [];
    let skipped = 0;

    await new Promise((resolve, reject) => {
      // 최신순(내림차순)
      const req = idx.openCursor(null, 'prev');
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return resolve();

        if (skipped < offset) {
          skipped++;
          cursor.continue();
          return;
        }

        out.push(cursor.value);
        if (limit > 0 && out.length >= limit) return resolve();
        cursor.continue();
      };
    });

    await txDone(tx);
    return out;
  }

  async pruneReplays(maxKeep = 10) {
    this._assert();
    const keep = Math.max(0, maxKeep | 0);

    const tx = this.db.transaction([STORE_REPLAYS], 'readwrite');
    const os = tx.objectStore(STORE_REPLAYS);
    const idx = os.index('createdAt');

    const total = await reqToPromise(os.count());
    const excess = total - keep;
    if (excess <= 0) {
      await txDone(tx);
      return;
    }

    let removed = 0;

    await new Promise((resolve, reject) => {
      // 오래된 것부터(오름차순) 삭제
      const req = idx.openCursor(null, 'next');
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return resolve();

        os.delete(cursor.primaryKey);
        removed++;
        if (removed >= excess) return resolve();

        cursor.continue();
      };
    });

    await txDone(tx);
  }

  async getCheckpoint(key) {
    this._assert();
    const tx = this.db.transaction([STORE_CHECKPOINTS], 'readonly');
    const os = tx.objectStore(STORE_CHECKPOINTS);
    const v = await reqToPromise(os.get(key));
    await txDone(tx);
    return v ?? null;
  }

  async setCheckpoint(key, value) {
    this._assert();
    const tx = this.db.transaction([STORE_CHECKPOINTS], 'readwrite');
    const os = tx.objectStore(STORE_CHECKPOINTS);
    os.put(value, key);
    await txDone(tx);
  }

  async clearAll() {
    this._assert();
    const tx = this.db.transaction([STORE_META, STORE_EPISODES, STORE_CHECKPOINTS], 'readwrite');
    tx.objectStore(STORE_META).clear();
    tx.objectStore(STORE_EPISODES).clear();
    tx.objectStore(STORE_CHECKPOINTS).clear();
    await txDone(tx);
  }

  async exportAll() {
    this._assert();

    const meta = await this.getMeta();
    const episodes = await this.listEpisodes({ limit: 0, offset: 0 }); // limit=0 => 전부
    const ck_model = await this.getCheckpoint('model_state');
    const ck_stats = await this.getCheckpoint('train_stats');
    const replays = await this.listReplays({ limit: 0, offset: 0 });

    return {
      format: 'pika_rl_export_v1',
      exportedAt: Date.now(),
      meta,
      checkpoints: {
        model_state: ck_model ?? null,
        train_stats: ck_stats ?? null,
      },
      episodes,
      replays,
    };
  }

  async importAll(data) {
    this._assert();

    if (!data || data.format !== 'pika_rl_export_v1') {
      throw new Error('Invalid import format');
    }

    const episodes = Array.isArray(data.episodes) ? data.episodes : [];
    const meta = data.meta ?? null;
    const checkpoints = data.checkpoints ?? {};
    const replays = Array.isArray(data.replays) ? data.replays : [];

    // merge import: 같은 id는 overwrite
    const tx = this.db.transaction([STORE_EPISODES, STORE_META, STORE_CHECKPOINTS, STORE_REPLAYS], 'readwrite');
    const epOS = tx.objectStore(STORE_EPISODES);
    const metaOS = tx.objectStore(STORE_META);
    const ckOS = tx.objectStore(STORE_CHECKPOINTS);
    const rpOS = tx.objectStore(STORE_REPLAYS);

    for (const ep of episodes) {
      if (!ep || !ep.id) continue;
      epOS.put(ep);
    }

    if (meta) {
      metaOS.put(meta, 'meta');
    }
    for (const rp of replays) {
      if (!rp || !rp.id) continue;
      rpOS.put(rp);
    }

    if (checkpoints.model_state != null) ckOS.put(checkpoints.model_state, 'model_state');
    if (checkpoints.train_stats != null) ckOS.put(checkpoints.train_stats, 'train_stats');

    await txDone(tx);
  }
}
