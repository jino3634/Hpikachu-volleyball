'use strict';

const KEY_PREFIX = 'evo_';

export function saveJson(key, value) {
  try {
    localStorage.setItem(KEY_PREFIX + key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function loadJson(key, fallback = null) {
  try {
    const v = localStorage.getItem(KEY_PREFIX + key);
    if (v == null) return fallback;
    return JSON.parse(v);
  } catch {
    return fallback;
  }
}
