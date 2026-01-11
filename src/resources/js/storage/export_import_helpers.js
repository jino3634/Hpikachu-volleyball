// storage/export_import_helpers.js
'use strict';

/**
 * Trigger download of a Blob in browser
 * @param {Blob} blob
 * @param {string} filename
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Read a File from <input type="file"> to Blob
 * @param {File} file
 * @returns {Blob}
 */
export function fileToBlob(file) {
  // File is a Blob already
  return file;
}
