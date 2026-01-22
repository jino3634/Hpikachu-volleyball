/**
 * Always-on Evolution panel (static UI in Step 2; Step 3 wires Start/Stop/Apply best).
 */
'use strict';

/**
 * @typedef {{
 *   mount?: HTMLElement,
 *   onStart?: ()=>void|Promise<void>,
 *   onStop?: ()=>void,
 *   onApplyBest?: ()=>void|Promise<void>,
 *   onExport?: ()=>void|Promise<void>,
 *   onImportFile?: (file: File)=>void|Promise<void>,
 *   onRefreshHof?: ()=>void|Promise<void>,
 *   onPruneHof?: ()=>void|Promise<void>,
 *   onRefreshHist?: ()=>void|Promise<void>,
 *   onPruneHist?: ()=>void|Promise<void>,
 * }} EvoPanelOpts
 */


// panel.js 상단 근처(헬퍼 함수 위) 추가
/** @typedef {{ getItem:(k:string)=>string|null; setItem:(k:string,v:string)=>void }} StorageLike */
/** @type {Window & { localStorageWrapper?: StorageLike }} */
const _win = /** @type {any} */ (typeof window !== 'undefined' ? window : {});

function _lsGet(key) {
  try {
    const ls = _win.localStorageWrapper || _win.localStorage;
    return ls ? ls.getItem(key) : null;
  } catch {
    return null;
  }
}
function _lsSet(key, val) {
  try {
    const ls = _win.localStorageWrapper || _win.localStorage;
    if (ls) ls.setItem(key, String(val));
  } catch {}
}

function readNumSetting(key, defV) {
  const raw = _lsGet(key);
  const v = Number(raw);
  return Number.isFinite(v) && v > 0 ? v : defV;
}
function bindNumSetting(inputEl, key, defV) {
  // init
  inputEl.value = String(readNumSetting(key, defV));
  inputEl.addEventListener('change', () => {
    const n = Number(inputEl.value);
    const v = Number.isFinite(n) && n > 0 ? (n | 0) : defV;
    inputEl.value = String(v);
    _lsSet(key, v);
  });
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function hr() {
  const d = document.createElement('div');
  d.style.height = '1px';
  d.style.background = 'rgba(255,255,255,0.18)';
  d.style.margin = '8px 0';
  return d;
}

function mkBtn(label) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  b.style.padding = '6px 8px';
  b.style.borderRadius = '8px';
  b.style.border = '1px solid rgba(255,255,255,0.22)';
  b.style.background = 'rgba(255,255,255,0.08)';
  b.style.color = '#fff';
  b.style.cursor = 'pointer';
  b.style.fontSize = '12px';
  return b;
}

function mkNumInput(defVal) {
  const i = document.createElement('input');
  i.type = 'number';
  i.value = String(defVal);
  i.style.width = '84px';
  i.style.padding = '4px 6px';
  i.style.borderRadius = '8px';
  i.style.border = '1px solid rgba(255,255,255,0.22)';
  i.style.background = 'rgba(0,0,0,0.25)';
  i.style.color = '#fff';
  i.style.fontSize = '12px';
  return i;
}

function mkRadio(name, value, labelText, checked = false) {
  const wrap = el('label');
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '6px';
  wrap.style.marginRight = '10px';
  wrap.style.cursor = 'pointer';
  wrap.style.userSelect = 'none';

  const r = /** @type {HTMLInputElement} */ (document.createElement('input'));
  r.type = 'radio';
  r.name = name;
  r.value = value;
  r.checked = !!checked;

  const t = el('span', null, labelText);
  t.style.fontSize = '12px';
  t.style.opacity = '0.95';

  wrap.append(r, t);
  return { wrap, input: r };
}

export class EvoPanel {
  /** @param {EvoPanelOpts} [opts] */
  constructor(opts = {}) {
    const mount = (opts && opts.mount) ? opts.mount : document.body;

    /** @type {HTMLDivElement} */
    const panel = document.createElement('div');
    panel.id = 'evo-panel';
    panel.style.position = 'absolute';
    panel.style.right = '12px';
    panel.style.bottom = '12px';
    panel.style.zIndex = '9999';
    panel.style.minWidth = '320px';
    panel.style.maxWidth = '420px';
    panel.style.padding = '10px';
    panel.style.border = '1px solid rgba(255,255,255,0.35)';
    panel.style.borderRadius = '12px';
    panel.style.background = 'rgba(0,0,0,0.60)';
    panel.style.color = '#fff';
    panel.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
    panel.style.fontSize = '13px';
    panel.style.backdropFilter = 'blur(6px)';

    // Header
    const header = el('div');
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';

    const title = el('div', null, 'Evolution');
    title.style.fontWeight = '800';
    title.style.letterSpacing = '0.2px';

    const status = el('div', null, 'Status: idle');
    status.style.opacity = '0.85';
    status.style.fontSize = '12px';

    header.appendChild(title);
    header.appendChild(status);

    // Row: main buttons
    const row1 = el('div');
    row1.style.display = 'flex';
    row1.style.gap = '6px';
    row1.style.flexWrap = 'wrap';
    row1.style.marginTop = '8px';

    const btnStart = mkBtn('Start');
    const btnStop = mkBtn('Stop');
    const btnApply = mkBtn('Apply best');
    const btnExport = mkBtn('Export');
    const btnImport = mkBtn('Import');
    const importFile = document.createElement('input');
    importFile.type = 'file';
    importFile.accept = 'application/json';
    importFile.style.display = 'none';
    btnStop.disabled = true;

    row1.append(btnStart, btnStop, btnApply, btnExport, btnImport);

    // Opponent mode
    const oppBox = el('div');
    oppBox.style.marginTop = '8px';

    const oppTitle = el('div', null, 'Opponent');
    oppTitle.style.fontWeight = '700';
    oppTitle.style.marginBottom = '4px';

    const oppRow = el('div');
    oppRow.style.display = 'flex';
    oppRow.style.flexWrap = 'wrap';

    const rSelf = mkRadio('evo-opp-mode-panel', 'self', 'Self', true);
    const rBase = mkRadio('evo-opp-mode-panel', 'baseline', 'Baseline', false);
    const rPhys = mkRadio('evo-opp-mode-panel', 'physics', 'Physics', false);

    oppRow.append(rSelf.wrap, rBase.wrap, rPhys.wrap);

    oppBox.append(oppTitle, oppRow);

    // HOF section (static)
    const hofBox = el('div');
    hofBox.style.marginTop = '8px';
    const hofTitle = el('div', null, 'Hall of Fame');
    hofTitle.style.fontWeight = '700';
    hofTitle.style.marginBottom = '4px';

    const hofRow = el('div');
    hofRow.style.display = 'flex';
    hofRow.style.alignItems = 'center';
    hofRow.style.gap = '6px';
    hofRow.style.flexWrap = 'wrap';

    const HOF_LIMIT_KEY = 'evo_hof_list_limit';
    const HOF_MAXKEEP_KEY = 'evo_hof_maxkeep';
    const hofLimit = mkNumInput(20);
    const hofMaxKeep = mkNumInput(200);
    const hofRefresh = mkBtn('Refresh');
    const hofPrune = mkBtn('Prune');

    bindNumSetting(hofLimit, HOF_LIMIT_KEY, 20);
    bindNumSetting(hofMaxKeep, HOF_MAXKEEP_KEY, 200);

    const lab1 = el('span', null, 'limit');
    lab1.style.opacity = '0.8';
    lab1.style.fontSize = '12px';
    const lab2 = el('span', null, 'maxKeep');
    lab2.style.opacity = '0.8';
    lab2.style.fontSize = '12px';

    hofRow.append(lab1, hofLimit, lab2, hofMaxKeep, hofRefresh, hofPrune);

    const hofList = el('div');
    hofList.style.marginTop = '6px';
    hofList.style.maxHeight = '120px';
    hofList.style.overflow = 'auto';
    hofList.style.border = '1px solid rgba(255,255,255,0.18)';
    hofList.style.borderRadius = '10px';
    hofList.style.padding = '6px';
    hofList.style.fontSize = '12px';
    hofList.style.opacity = '0.9';
    hofList.textContent = '(HOF list will appear here)';

    hofBox.append(hofTitle, hofRow, hofList);

    // History section (static)
    const histBox = el('div');
    histBox.style.marginTop = '8px';
    const histTitle = el('div', null, 'History');
    histTitle.style.fontWeight = '700';
    histTitle.style.marginBottom = '4px';

    const histRow = el('div');
    histRow.style.display = 'flex';
    histRow.style.alignItems = 'center';
    histRow.style.gap = '6px';
    histRow.style.flexWrap = 'wrap';

    const HIST_LIMIT_KEY = 'evo_hist_list_limit';
    const HIST_MAXKEEP_KEY = 'evo_hist_maxkeep';
    const histLimit = mkNumInput(50);
    const histMaxKeep = mkNumInput(2000);
    const histRefresh = mkBtn('Refresh');
    const histPrune = mkBtn('Prune');

    bindNumSetting(histLimit, HIST_LIMIT_KEY, 50);
    bindNumSetting(histMaxKeep, HIST_MAXKEEP_KEY, 2000);

    const lab3 = el('span', null, 'limit');
    lab3.style.opacity = '0.8';
    lab3.style.fontSize = '12px';
    const lab4 = el('span', null, 'maxKeep');
    lab4.style.opacity = '0.8';
    lab4.style.fontSize = '12px';

    histRow.append(lab3, histLimit, lab4, histMaxKeep, histRefresh, histPrune);

    const histList = el('div');
    histList.style.marginTop = '6px';
    histList.style.maxHeight = '120px';
    histList.style.overflow = 'auto';
    histList.style.border = '1px solid rgba(255,255,255,0.18)';
    histList.style.borderRadius = '10px';
    histList.style.padding = '6px';
    histList.style.fontSize = '12px';
    histList.style.opacity = '0.9';
    histList.textContent = '(History list will appear here)';

    histBox.append(histTitle, histRow, histList);

    panel.appendChild(importFile);

    panel.append(header, row1, hr(), oppBox, hr(), hofBox, hr(), histBox);

    mount.appendChild(panel);

    // Store refs
    this.panel = panel;
    this._statusEl = status;
    this._btnStart = btnStart;
    this._btnStop = btnStop;
    this._btnApply = btnApply;
    this._btnExport = btnExport;
    this._btnImport = btnImport;
    this._importFile = importFile;
    this._oppRadios = [rSelf.input, rBase.input, rPhys.input];
    this._hofLimit = hofLimit;
    this._hofMaxKeep = hofMaxKeep;
    this._hofRefresh = hofRefresh;
    this._hofPrune = hofPrune;
    this._hofListEl = hofList;

    this._histLimit = histLimit;
    this._histMaxKeep = histMaxKeep;
    this._histRefresh = histRefresh;
    this._histPrune = histPrune;
    this._histListEl = histList;

    // Wire Step 3 buttons
    btnStart.addEventListener('click', async () => {
      if (opts.onStart) await opts.onStart();
    });
    btnStop.addEventListener('click', () => {
      if (opts.onStop) opts.onStop();
    });
    btnApply.addEventListener('click', async () => {
      if (opts.onApplyBest) await opts.onApplyBest();
    });

    btnExport.addEventListener('click', async () => {
      if (opts.onExport) await opts.onExport();
    });
    btnImport.addEventListener('click', () => {
      try {
        importFile.value = '';
        importFile.click();
      } catch {}
    });
    importFile.addEventListener('change', async () => {
      const file = importFile.files && importFile.files[0];
      if (!file) return;
      if (opts.onImportFile) await opts.onImportFile(file);
    });

    // Step 5: list refresh/prune wiring
    hofRefresh.addEventListener('click', async () => {
      if (opts.onRefreshHof) await opts.onRefreshHof();
    });
    hofPrune.addEventListener('click', async () => {
      if (opts.onPruneHof) await opts.onPruneHof();
    });
    histRefresh.addEventListener('click', async () => {
      if (opts.onRefreshHist) await opts.onRefreshHist();
    });
    histPrune.addEventListener('click', async () => {
      if (opts.onPruneHist) await opts.onPruneHist();
    });

  }

  show() {
    this.panel.style.display = 'block';
  }

  /** @param {string} text */
  setStatus(text) {
    if (this._statusEl) this._statusEl.textContent = String(text || '');
  }

  /** @param {boolean} running */
  setRunning(running) {
    const r = !!running;
    if (this._btnStart) this._btnStart.disabled = r;
    if (this._btnStop) this._btnStop.disabled = !r;
  }

  /** @param {any[]} list */
  renderHof(list) {
    const box = this._hofListEl;
    if (!box) return;
    box.textContent = '';
    const arr = Array.isArray(list) ? list : [];
    if (arr.length === 0) {
      box.textContent = '(empty)';
      return;
    }
    for (let i = 0; i < arr.length; i++) {
      const it = arr[i] || {};
      const row = document.createElement('div');
      row.style.padding = '4px 4px';
      row.style.borderBottom = '1px solid rgba(255,255,255,0.10)';
      row.style.display = 'flex';
      row.style.gap = '8px';
      row.style.alignItems = 'baseline';
      row.style.justifyContent = 'space-between';

      const left = document.createElement('div');
      left.style.display = 'flex';
      left.style.flexWrap = 'wrap';
      left.style.gap = '6px';

      const gen = (it.generation != null) ? String(it.generation) : '?';
      const wr = (it.bestEvalWinRate ?? it.bestWinRate);
      const wrPct = (Number(wr || 0) * 100).toFixed(1) + '%';
      const fit = (it.bestFitness != null) ? Number(it.bestFitness).toFixed(3) : '';
      const at = (it.savedAt != null) ? new Date(Number(it.savedAt)).toLocaleString() : '';

      const t1 = document.createElement('span'); t1.textContent = `#${i+1}`;
      const t2 = document.createElement('span'); t2.textContent = `gen ${gen}`;
      const t3 = document.createElement('span'); t3.textContent = `wr ${wrPct}`;
      const t4 = document.createElement('span'); t4.textContent = fit ? `fit ${fit}` : '';
      [t1,t2,t3,t4].forEach(t => { t.style.opacity='0.92'; t.style.fontSize='12px'; });

      left.appendChild(t1); left.appendChild(t2); left.appendChild(t3);
      if (fit) left.appendChild(t4);

      const right = document.createElement('span');
      right.textContent = at ? at : '';
      right.style.opacity = '0.65';
      right.style.fontSize = '10px';

      row.append(left, right);
      box.appendChild(row);
    }
  }

  /** @param {any[]} list */
  renderHistory(list) {
    const box = this._histListEl;
    if (!box) return;
    box.textContent = '';
    const arr = Array.isArray(list) ? list : [];
    if (arr.length === 0) {
      box.textContent = '(empty)';
      return;
    }
    for (let i = 0; i < arr.length; i++) {
      const it = arr[i] || {};
      const row = document.createElement('div');
      row.style.padding = '4px 4px';
      row.style.borderBottom = '1px solid rgba(255,255,255,0.10)';
      row.style.display = 'flex';
      row.style.gap = '8px';
      row.style.alignItems = 'baseline';
      row.style.justifyContent = 'space-between';

      const left = document.createElement('div');
      left.style.display = 'flex';
      left.style.flexWrap = 'wrap';
      left.style.gap = '6px';

      const gen = (it.generation != null) ? String(it.generation) : '?';
      const wr = (it.evalWinRate ?? it.bestEvalWinRate ?? it.bestWinRate ?? it.winRate);
      const wrPct = (Number(wr || 0) * 100).toFixed(1) + '%';
      const fit = (it.fitness != null) ? Number(it.fitness).toFixed(3) : ((it.bestFitness != null) ? Number(it.bestFitness).toFixed(3) : '');
      const at = (it.savedAt != null) ? new Date(Number(it.savedAt)).toLocaleString() : ((it.time != null) ? new Date(Number(it.time)).toLocaleString() : '');

      const t1 = document.createElement('span'); t1.textContent = `#${i+1}`;
      const t2 = document.createElement('span'); t2.textContent = `gen ${gen}`;
      const t3 = document.createElement('span'); t3.textContent = `wr ${wrPct}`;
      const t4 = document.createElement('span'); t4.textContent = fit ? `fit ${fit}` : '';
      [t1,t2,t3,t4].forEach(t => { t.style.opacity='0.92'; t.style.fontSize='12px'; });

      left.appendChild(t1); left.appendChild(t2); left.appendChild(t3);
      if (fit) left.appendChild(t4);

      const right = document.createElement('span');
      right.textContent = at ? at : '';
      right.style.opacity = '0.65';
      right.style.fontSize = '10px';

      row.append(left, right);
      box.appendChild(row);
    }
  }

  /** @returns {{hofLimit:number, hofMaxKeep:number, histLimit:number, histMaxKeep:number}} */
  getListSettings() {
    const num = (x, defV) => {
      const v = Number(x && x.value);
      const n = Number.isFinite(v) ? (v | 0) : defV;
      return Math.max(1, n);
    };
    return {
      hofLimit: num(this._hofLimit, 20),
      hofMaxKeep: num(this._hofMaxKeep, 200),
      histLimit: num(this._histLimit, 50),
      histMaxKeep: num(this._histMaxKeep, 2000),
    };
  }

  /** @returns {'self'|'baseline'|'physics'} */
  getOpponentMode() {
    const r = this._oppRadios || [];
    for (const it of r) {
      if (it && it.checked) {
        const v = String(it.value || '');
        if (v === 'baseline') return 'baseline';
        if (v === 'physics') return 'physics';
        return 'self';
      }
    }
    return 'self';
  }
}
