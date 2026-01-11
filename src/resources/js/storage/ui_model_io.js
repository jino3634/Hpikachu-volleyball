// ui_model_io.js
'use strict';

import { downloadBlob } from './storage/export_import_helpers.js';

export async function bindModelIO(agent, {
  exportBtnId,
  importInputId,
  filename = 'pikavolley_model.json',
}) {
  const exportBtn = document.getElementById(exportBtnId);
  const importInput = document.getElementById(importInputId);

  exportBtn.addEventListener('click', async () => {
    const blob = await agent.exportToBlob();
    downloadBlob(blob, filename);
  });

  importInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 덮어쓰기 import (merge 원하면 opts로)
    await agent.importFrom(file, { merge: false });

    // 즉시 저장해두는 게 안전
    await agent.save();

    // input reset
    e.target.value = '';
  });
}
