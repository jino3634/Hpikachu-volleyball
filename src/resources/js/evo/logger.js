'use strict';

/**
 * In-memory JSONL logger for evo diagnostics.
 *
 * Stage 4:
 * - UI-controlled enable/level/categories/sample-rate/buffer limits.
 * - When disabled, avoid stringify costs.
 */

/** @typedef {'ERROR'|'WARN'|'INFO'|'DEBUG'} LogLevel */

const LEVEL_NUM = /** @type {const} */ ({ ERROR: 0, WARN: 1, INFO: 2, DEBUG: 3 });

function _now() {
  try { return Date.now(); } catch { return 0; }
}

function _pad2(n) {
  const v = (n | 0);
  return (v < 10 ? '0' : '') + String(v);
}

function _tsForFile(ms) {
  try {
    const d = new Date(ms);
    const y = d.getFullYear();
    const m = _pad2(d.getMonth() + 1);
    const day = _pad2(d.getDate());
    const hh = _pad2(d.getHours());
    const mm = _pad2(d.getMinutes());
    const ss = _pad2(d.getSeconds());
    return `${y}${m}${day}_${hh}${mm}${ss}`;
  } catch {
    return String(ms | 0);
  }
}

/**
 * @param {any} v
 * @param {number} defV
 */
function _toPosInt(v, defV) {
  const n = Number(v);
  const x = Number.isFinite(n) ? (n | 0) : defV;
  return Math.max(1, x);
}

class EvoLogger {
  constructor() {
    /** @type {boolean} */
    this.enabled = false;
    /** @type {LogLevel} */
    this.level = 'INFO';

    /** @type {number} */
    this.maxLines = 5000;
    /** @type {number} */
    this.maxBytes = 2 * 1024 * 1024; // ~2MB text cap (best-effort)

    /** @type {Set<string>|null} */
    this._cats = null; // null => allow all

    /** @type {{input:number, obs:number}} */
    this.sampleEvery = { input: 20, obs: 20 };

    /** @type {Record<string, number>} */
    this._catCounters = Object.create(null);

    /** @type {string[]} */
    this._lines = [];
    /** @type {number} */
    this._bytes = 0;

    /** @type {boolean} */
    this._metaEmitted = false;
  }

  /** @returns {{enabled:boolean, level:LogLevel, maxLines:number, maxBytes:number, cats:string[]|null, sampleEvery:{input:number, obs:number}}} */
  getConfig() {
    return {
      enabled: !!this.enabled,
      level: this.level,
      maxLines: this.maxLines | 0,
      maxBytes: this.maxBytes | 0,
      cats: this._cats ? Array.from(this._cats.values()) : null,
      sampleEvery: { input: this.sampleEvery.input | 0, obs: this.sampleEvery.obs | 0 },
    };
  }

  /** @param {boolean} v */
  setEnabled(v) {
    this.enabled = !!v;
  }

  /** @param {LogLevel} lvl */
  setLevel(lvl) {
    const s = String(lvl || '').toUpperCase();
    if (s === 'ERROR' || s === 'WARN' || s === 'INFO' || s === 'DEBUG') {
      this.level = /** @type {LogLevel} */ (s);
    }
  }

  /**
   * Lightweight check to avoid stringify/alloc costs when a log would be filtered out.
   * @param {string} cat
   * @param {LogLevel} lvl
   * @returns {boolean}
   */
  wouldLog(cat, lvl = 'INFO') {
    if (!this.enabled) return false;
    const lv = String(lvl || 'INFO').toUpperCase();
    const cur = LEVEL_NUM[this.level] ?? 2;
    const n = LEVEL_NUM[/** @type {LogLevel} */(lv)] ?? 2;
    if (n > cur) return false;
    if (this._cats && !this._cats.has(String(cat || ''))) return false;
    return true;
  }

  /**
   * Emit a single "meta" line once per page load (best-effort).
   * @param {any} meta
   */
  emitMetaOnce(meta) {
    if (this._metaEmitted) return;
    this._metaEmitted = true;
    try {
      this.log('meta', 'INFO', meta || {});
    } catch {}
  }

  /** @param {number} n */
  setMaxLines(n) {
    const m = Math.max(100, _toPosInt(n, 5000));
    this.maxLines = m;
    this._trim();
  }

  /** @param {number} n */
  setMaxBytes(n) {
    const m = Math.max(32 * 1024, _toPosInt(n, 2 * 1024 * 1024));
    this.maxBytes = m;
    this._trim();
  }

  /**
   * @param {string[]|null} cats - null/empty => allow all
   */
  setCategories(cats) {
    const arr = Array.isArray(cats) ? cats.map((x) => String(x || '').trim()).filter(Boolean) : [];
    if (arr.length === 0) {
      this._cats = null;
      return;
    }
    this._cats = new Set(arr);
  }

  /** @param {{input?:number, obs?:number}} cfg */
  setSampleEvery(cfg) {
    if (!cfg || typeof cfg !== 'object') return;
    if (cfg.input != null) this.sampleEvery.input = Math.max(1, _toPosInt(cfg.input, 20));
    if (cfg.obs != null) this.sampleEvery.obs = Math.max(1, _toPosInt(cfg.obs, 20));
  }

  clear() {
    this._lines.length = 0;
    this._bytes = 0;
    this._catCounters = Object.create(null);
    // Do NOT reset _metaEmitted here by default: it's intended "once per page load".
    // If you want meta again, call emitMetaOnce(...) after calling clear().
  }

  /** @param {number} n */
  getTail(n = 200) {
    const k = Math.max(1, (n | 0));
    const start = Math.max(0, this._lines.length - k);
    return this._lines.slice(start);
  }

  /**
   * @param {string} cat
   * @param {LogLevel} lvl
   * @param {any} obj
   */
  log(cat, lvl, obj) {
    if (!this.enabled) return;

    const lv = String(lvl || 'INFO').toUpperCase();
    const num = LEVEL_NUM[/** @type {LogLevel} */(lv)] ?? 2;
    const cur = LEVEL_NUM[this.level] ?? 2;
    if (num > cur) return;

    const c = String(cat || 'misc');

    // Category filter
    if (this._cats && !this._cats.has(c)) return;

    // Sampling (only for noisy categories; applied for DEBUG level logs)
    if (c === 'input') {
      const every = Math.max(1, this.sampleEvery.input | 0);
      const k = (this._catCounters[c] = (this._catCounters[c] || 0) + 1);
      if ((k % every) !== 0 && num >= LEVEL_NUM.DEBUG) return;
    } else if (c === 'obs') {
      const every = Math.max(1, this.sampleEvery.obs | 0);
      const k = (this._catCounters[c] = (this._catCounters[c] || 0) + 1);
      if ((k % every) !== 0 && num >= LEVEL_NUM.DEBUG) return;
    }

    const t = _now();
    /** @type {any} */
    const rec = (obj && typeof obj === 'object') ? { ...obj } : { msg: obj };
    rec.t = t;
    rec.lvl = lv;
    rec.cat = c;

    let line = '';
    try {
      line = JSON.stringify(rec);
    } catch {
      try {
        line = JSON.stringify({ t, lvl: lv, cat: c, msg: String(obj) });
      } catch {
        line = `{"t":${t},"lvl":"${lv}","cat":"${c}","msg":"[unstringifiable]"}`;
      }
    }

    this._lines.push(line);
    this._bytes += line.length + 1;
    this._trim();
  }

  _trim() {
    // Lines cap
    if (this._lines.length > this.maxLines) {
      const drop = this._lines.length - this.maxLines;
      for (let i = 0; i < drop; i++) this._bytes -= (this._lines[i].length + 1);
      this._lines.splice(0, drop);
    }
    // Bytes cap (best-effort)
    if (this._bytes > this.maxBytes) {
      while (this._lines.length > 0 && this._bytes > this.maxBytes) {
        const first = this._lines.shift();
        this._bytes -= (first ? first.length + 1 : 0);
      }
    }
    if (this._bytes < 0) this._bytes = 0;
  }

  /**
   * Download logs as a JSONL file.
   * @param {string} [filename]
   */
  download(filename = '') {
    const lines = this._lines;
    const text = lines.join('\n') + (lines.length ? '\n' : '');
    const t = _now();
    const fn = filename && String(filename).trim()
      ? String(filename).trim()
      : `hpikachu_evo_log_${_tsForFile(t)}.jsonl`;

    try {
      const blob = new Blob([text], { type: 'application/jsonl;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fn;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => { try { URL.revokeObjectURL(url); } catch {} }, 1000);
    } catch {
      try { console.log(text); } catch {}
    }
  }
}

export const evoLogger = new EvoLogger();

try {
  if (typeof window !== 'undefined') {
    // @ts-ignore
    window.__evoLogger = evoLogger;
  }
} catch {}
