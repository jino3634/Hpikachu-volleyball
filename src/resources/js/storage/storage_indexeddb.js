// storage_indexeddb.js
'use strict';

import { StorageIface } from './storage_iface.js';

const DB_NAME = 'pika_rl_db';
// DB_VERSION: bump when adding new object stores
const DB_VERSION = 3;

const STORE_META = 'meta';
const STORE_EPISODES = 'episodes';
const STORE_CHECKPOINTS = 'checkpoints';
const STORE_REPLAYS = 'replays';
const STORE_IMIT_SAMPLES = 'imit_samples';

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

    // imitation learning samples (frame-level)
    if (!db.objectStoreNames.contains(STORE_IMIT_SAMPLES)) {
      const os = db.createObjectStore(STORE_IMIT_SAMPLES, { keyPath: 'k', autoIncrement: true });
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

    // imitation sample counters (best-effort, not authoritative)
    this._imitSincePrune = 0;
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

  async clearImitationSamples() {
    this._assert();
    const tx = this.db.transaction(['imit_samples'], 'readwrite');
    const os = tx.objectStore('imit_samples');
    os.clear();
    await new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
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

    if (!episode) {
      console.warn('[IndexedDBStorage] appendEpisode: episode is null/undefined');
      return;
    }

    // 학습 루프에서 예외가 나도 전체가 멈추지 않도록 방어
    if (!episode || typeof episode !== 'object') {
      console.warn('[IndexedDBStorage] appendEpisode: invalid episode', episode);
      return;
    }
    const tx = this.db.transaction([STORE_EPISODES, STORE_META], 'readwrite');
    const epOS = tx.objectStore(STORE_EPISODES);
    const metaOS = tx.objectStore(STORE_META);

    // episode 저장 (keyPath: 'id' 요구사항 충족)
    const id = episode.id ?? episode.episodeId ?? null;
    if (!id) {
      console.warn('[IndexedDBStorage] appendEpisode: missing id/episodeId', episode);
      return;
    }

    const ep = {
      ...episode,
      id,
      createdAt: episode.createdAt ?? episode.endedAt ?? episode.startedAt ?? Date.now(),
    };

    epOS.put(ep);

    // meta 갱신(대략 카운터)
    const meta = (await reqToPromise(metaOS.get('meta'))) ?? {
      createdAt: Date.now(),
      version: 1,
      totalEpisodes: 0,
      notes: '',
    };

    meta.totalEpisodes = (meta.totalEpisodes | 0) + 1;
    meta.lastEpisodeAt = ep.createdAt;   // ✅ createdAt을 확실히 사용
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

  /**
   * Append ONE imitation sample.
   * @param {{ obs:any, label:{xDirection:number,yDirection:number,powerHit:number}, createdAt?:number }} sample
   */
  async appendImitationSample(sample) {
    this._assert();
    if (!sample || typeof sample !== 'object') return;
    if (!sample.label) return;

    const createdAt = sample.createdAt ?? Date.now();

    const tx = this.db.transaction([STORE_IMIT_SAMPLES], 'readwrite');
    const os = tx.objectStore(STORE_IMIT_SAMPLES);
    os.add({
      createdAt,
      obs: sample.obs ?? null,
      label: {
        xDirection: (sample.label.xDirection ?? 0) | 0,
        yDirection: (sample.label.yDirection ?? 0) | 0,
        powerHit: (sample.label.powerHit ?? 0) | 0,
      },
    });
    await txDone(tx);
  }

  /**
   * Append multiple imitation samples in ONE transaction (much faster).
   * @param {Array<{ obs:any, label:{xDirection:number,yDirection:number,powerHit:number}, createdAt?:number }>} samples
   */
  async appendImitationSamples(samples) {
    this._assert();
    if (!Array.isArray(samples) || samples.length === 0) return;

    const tx = this.db.transaction([STORE_IMIT_SAMPLES], 'readwrite');
    const os = tx.objectStore(STORE_IMIT_SAMPLES);

    for (const sample of samples) {
      if (!sample || typeof sample !== 'object' || !sample.label) continue;
      const createdAt = sample.createdAt ?? Date.now();
      os.add({
        createdAt,
        obs: sample.obs ?? null,
        label: {
          xDirection: (sample.label.xDirection ?? 0) | 0,
          yDirection: (sample.label.yDirection ?? 0) | 0,
          powerHit: (sample.label.powerHit ?? 0) | 0,
        },
      });
    }

    await txDone(tx);
  }

  async countImitationSamples() {
    this._assert();
    const tx = this.db.transaction([STORE_IMIT_SAMPLES], 'readonly');
    const os = tx.objectStore(STORE_IMIT_SAMPLES);
    const n = await reqToPromise(os.count());
    await txDone(tx);
    return n | 0;
  }


  /**
   * List imitation samples (newest first by createdAt).
   * @param {{limit?:number, offset?:number}} opts
   * @returns {Promise<Array<{k:number, obs:any, label:{xDirection:number,yDirection:number,powerHit:number}, createdAt:number}>>}
   */
  async listImitationSamples(opts = {}) {
    this._assert();
    const limit = Math.max(0, (opts.limit ?? 1000) | 0);
    const offset = Math.max(0, (opts.offset ?? 0) | 0);

    const tx = this.db.transaction([STORE_IMIT_SAMPLES], 'readonly');
    const os = tx.objectStore(STORE_IMIT_SAMPLES);
    const idx = os.index('createdAt');

    /** @type {any[]} */
    const out = [];
    let skipped = 0;

    await new Promise((resolve, reject) => {
      const req = idx.openCursor(null, 'prev'); // newest first
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

        const v = cursor.value;
        const id = v?.id ?? '';
        // 기본: "최근 10개" 목록은 win: 슬롯을 숨김
        if (typeof id === 'string' && id.startsWith('win:')) {
          cursor.continue();
          return;
        }

        if (skipped < offset) {
          skipped++;
          cursor.continue();
          return;
        }

        out.push(v);
        if (limit > 0 && out.length >= limit) return resolve();
        cursor.continue();
      };
    });

    await txDone(tx);
    return out;
  }

  async listWinReplays(opts = {}) {
    this._assert();
    const limit = Math.max(0, (opts.limit ?? 3) | 0);
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

        const v = cursor.value;
        const id = v?.id ?? '';
        if (!(typeof id === 'string' && id.startsWith('win:'))) {
          cursor.continue();
          return;
        }

        if (skipped < offset) {
          skipped++;
          cursor.continue();
          return;
        }

        out.push(v);
        if (limit > 0 && out.length >= limit) return resolve();
        cursor.continue();
      };
    });

    await txDone(tx);
    return out;
  }

  async pruneWinReplays(maxKeep = 3) {
    this._assert();
    const keep = Math.max(0, maxKeep | 0);

    const tx = this.db.transaction([STORE_REPLAYS], 'readwrite');
    const os = tx.objectStore(STORE_REPLAYS);
    const idx = os.index('createdAt');

    let kept = 0;

    await new Promise((resolve, reject) => {
      const req = idx.openCursor(null, 'prev');
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return resolve();

        const v = cursor.value;
        const id = v?.id ?? '';

        if (!(typeof id === 'string' && id.startsWith('win:'))) {
          cursor.continue();
          return;
        }

        if (kept < keep) {
          kept++;
          cursor.continue();
          return;
        }

        os.delete(cursor.primaryKey);
        cursor.continue();
      };
    });

    await txDone(tx);
  }

  async pruneReplays(maxKeep = 10) {
    this._assert();
    const keep = Math.max(0, maxKeep | 0);

    const tx = this.db.transaction([STORE_REPLAYS], 'readwrite');
    const os = tx.objectStore(STORE_REPLAYS);
    const idx = os.index('createdAt');

    let kept = 0;

    await new Promise((resolve, reject) => {
      // 최신순으로 훑으면서 keep 넘는 "old" 항목을 삭제
      const req = idx.openCursor(null, 'prev');
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return resolve();

        const v = cursor.value;
        const id = v?.id ?? '';

        // recent 목록에서 win: 슬롯은 제외(보존/삭제 별도)
        if (typeof id === 'string' && id.startsWith('win:')) {
          cursor.continue();
          return;
        }

        if (kept < keep) {
          kept++;
          cursor.continue();
          return;
        }

        os.delete(cursor.primaryKey);
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
    const tx = this.db.transaction([STORE_META, STORE_EPISODES, STORE_CHECKPOINTS, STORE_IMIT_SAMPLES], 'readwrite');
    tx.objectStore(STORE_META).clear();
    tx.objectStore(STORE_EPISODES).clear();
    tx.objectStore(STORE_CHECKPOINTS).clear();
    tx.objectStore(STORE_IMIT_SAMPLES).clear();
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
