// debug_log.js
'use strict';

/**
 * Full-session debug log capture.
 * - Captures console.log / console.warn / console.error into an in-memory buffer.
 * - Provides a "Download log" helper to export everything as a .txt file.
 *
 * This prevents loss of important diagnostics due to the DevTools console buffer limit.
 */

export const DEBUG_LOG = [];
export const DEBUG_LOG_MAX_LINES = 300000; // safety cap to avoid unbounded memory growth

let _HOOKED = false;
let _ORIG = null;

function _toLine(args) {
  return args.map((a) => {
    if (typeof a === 'string') return a;
    try { return JSON.stringify(a); } catch (e) { return String(a); }
  }).join(' ');
}

function _pushLine(prefix, args) {
  const line = prefix ? `${prefix} ${_toLine(args)}` : _toLine(args);
  DEBUG_LOG.push(line);
  if (DEBUG_LOG.length > DEBUG_LOG_MAX_LINES) {
    DEBUG_LOG.splice(0, DEBUG_LOG.length - DEBUG_LOG_MAX_LINES);
  }
}

export function hookConsole() {
  if (_HOOKED) return;
  _HOOKED = true;

  _ORIG = {
    log: console.log ? console.log.bind(console) : null,
    warn: console.warn ? console.warn.bind(console) : null,
    error: console.error ? console.error.bind(console) : null,
  };

  if (_ORIG.log) {
    console.log = (...args) => {
      _pushLine('', args);
      _ORIG.log(...args);
    };
  }
  if (_ORIG.warn) {
    console.warn = (...args) => {
      _pushLine('[WARN]', args);
      _ORIG.warn(...args);
    };
  }
  if (_ORIG.error) {
    console.error = (...args) => {
      _pushLine('[ERROR]', args);
      _ORIG.error(...args);
    };
  }
}

export function logDebug(...args) {
  // Prefer writing through console.log so it remains visible,
  // but ensure we don't depend on our override.
  if (_ORIG && _ORIG.log) {
    _pushLine('', args);
    _ORIG.log(...args);
  } else {
    // if hook not active yet
    _pushLine('', args);
    console.log(...args);
  }
}

export function clearDebugLog() {
  DEBUG_LOG.length = 0;
}

export function downloadDebugLog(filename = null) {
  const ts = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const name = filename ?? `hpikachu_debug_${ts.getFullYear()}${pad(ts.getMonth()+1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}.txt`;

  const text = DEBUG_LOG.join('\n');
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Auto-hook on import
hookConsole();

// Expose helpers for manual use from the console and UI.
if (typeof window !== 'undefined') {
  window.__HPIKACHU_DEBUG_LOG__ = DEBUG_LOG;
  window.downloadDebugLog = downloadDebugLog;
  window.clearDebugLog = clearDebugLog;
}
