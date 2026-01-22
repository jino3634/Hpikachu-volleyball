/**
 * Always-on Evolution panel (static UI in Step 2; Step 3 wires Start/Stop/Apply best).
 */
import { evoLogger } from './logger.js';

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
    // Persist opponent mode selection (so refresh keeps your choice)
    const OPP_MODE_KEY = 'evo_opp_mode';
    const savedOppMode = _lsGet(OPP_MODE_KEY);
    if (savedOppMode === 'self' || savedOppMode === 'baseline' || savedOppMode === 'physics') {
      rSelf.input.checked = (savedOppMode === 'self');
      rBase.input.checked = (savedOppMode === 'baseline');
      rPhys.input.checked = (savedOppMode === 'physics');
    }
    [rSelf.input, rBase.input, rPhys.input].forEach((r) => {
      r.addEventListener('change', () => {
        if (r.checked) _lsSet(OPP_MODE_KEY, r.value);
      });
    });


    // Opponent winrate (last 100) display (filled by runner updates)
    const wrRow = el('div');
    wrRow.style.marginTop = '4px';
    wrRow.style.fontSize = '12px';
    wrRow.style.opacity = '0.9';
    wrRow.style.display = 'flex';
    wrRow.style.flexWrap = 'wrap';
    wrRow.style.gap = '8px';

    const wrSelf = el('span', null, 'Self: —');
    const wrBase = el('span', null, 'Baseline: —');
    const wrPhys = el('span', null, 'Physics: —');

    wrRow.append(el('span', null, 'WR100'), wrSelf, wrBase, wrPhys);

    oppBox.append(oppTitle, oppRow, wrRow);

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
    // ---- Logs (Stage 4: level/category/sample/buffer controls) ----
    const LOG_ENABLED_KEY = 'EVO_LOG_ENABLED';
    const LOG_LEVEL_KEY = 'EVO_LOG_LEVEL';
    const LOG_MAXLINES_KEY = 'EVO_LOG_MAXLINES';
    const LOG_MAXBYTES_KEY = 'EVO_LOG_MAXBYTES';
    const LOG_SAMPLE_INPUT_KEY = 'EVO_LOG_SAMPLE_INPUT';
    const LOG_SAMPLE_OBS_KEY = 'EVO_LOG_SAMPLE_OBS';
    const LOG_CATS_KEY = 'EVO_LOG_CATS';

    const logBox = el('div');
    const logTitle = el('div', null, 'Logs');
    logTitle.style.fontWeight = '700';
    logTitle.style.marginBottom = '6px';

    const logRow1 = el('div');
    logRow1.style.display = 'flex';
    logRow1.style.alignItems = 'center';
    logRow1.style.gap = '8px';
    logRow1.style.flexWrap = 'wrap';

    const logEnableWrap = el('label');
    logEnableWrap.style.display = 'inline-flex';
    logEnableWrap.style.alignItems = 'center';
    logEnableWrap.style.gap = '6px';
    logEnableWrap.style.cursor = 'pointer';

    const logEnable = /** @type {HTMLInputElement} */ (document.createElement('input'));
    logEnable.type = 'checkbox';
    logEnable.checked = (_lsGet(LOG_ENABLED_KEY) === '1');

    const logEnableText = el('span', null, 'Enable');
    logEnableText.style.fontSize = '12px';
    logEnableText.style.opacity = '0.95';
    logEnableWrap.append(logEnable, logEnableText);

    const levelSel = /** @type {HTMLSelectElement} */ (document.createElement('select'));
    ['ERROR','WARN','INFO','DEBUG'].forEach((s) => {
      const o = document.createElement('option');
      o.value = s; o.textContent = s;
      levelSel.appendChild(o);
    });
    levelSel.style.padding = '4px 6px';
    levelSel.style.borderRadius = '8px';
    levelSel.style.border = '1px solid rgba(255,255,255,0.22)';
    levelSel.style.background = 'rgba(0,0,0,0.25)';
    levelSel.style.color = '#fff';
    levelSel.style.fontSize = '12px';

    const levelLab = el('span', null, 'Level');
    levelLab.style.opacity = '0.8';
    levelLab.style.fontSize = '12px';

    const maxLinesLab = el('span', null, 'MaxLines');
    maxLinesLab.style.opacity = '0.8';
    maxLinesLab.style.fontSize = '12px';
    const maxLinesInput = mkNumInput(5000);
    maxLinesInput.style.width = '90px';

    const maxKbLab = el('span', null, 'MaxKB');
    maxKbLab.style.opacity = '0.8';
    maxKbLab.style.fontSize = '12px';
    const maxKbInput = mkNumInput(2048);
    maxKbInput.style.width = '84px';

    const logRefresh = mkBtn('Refresh');
    const logDownload = mkBtn('Download');
    const logClear = mkBtn('Clear');

    logRow1.append(logEnableWrap, levelLab, levelSel, maxLinesLab, maxLinesInput, maxKbLab, maxKbInput, logRefresh, logDownload, logClear);

    // Categories row
    const logRow2 = el('div');
    logRow2.style.display = 'flex';
    logRow2.style.alignItems = 'center';
    logRow2.style.gap = '10px';
    logRow2.style.flexWrap = 'wrap';
    logRow2.style.marginTop = '6px';

    const catLab = el('span', null, 'Cats');
    catLab.style.opacity = '0.8';
    catLab.style.fontSize = '12px';

    const mkCat = (key, label) => {
      const wrap = el('label');
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '6px';
      wrap.style.cursor = 'pointer';
      wrap.style.userSelect = 'none';
      const cb = /** @type {HTMLInputElement} */ (document.createElement('input'));
      cb.type = 'checkbox';
      cb.dataset.cat = key;
      const t = el('span', null, label);
      t.style.fontSize = '12px';
      t.style.opacity = '0.95';
      wrap.append(cb, t);
      return { wrap, cb };
    };

    const cats = [
      mkCat('meta', 'meta'),
      mkCat('match', 'match'),
      mkCat('match_anomaly', 'anomaly'),
      mkCat('physics_route', 'physics'),
      mkCat('input', 'input'),
      mkCat('obs', 'obs'),
      mkCat('generation', 'gen'),
      mkCat('storage', 'storage'),
    ];

    logRow2.append(catLab, ...cats.map((x) => x.wrap));

    // Sampling row
    const logRow3 = el('div');
    logRow3.style.display = 'flex';
    logRow3.style.alignItems = 'center';
    logRow3.style.gap = '10px';
    logRow3.style.flexWrap = 'wrap';
    logRow3.style.marginTop = '6px';

    const sampLab = el('span', null, 'Sample');
    sampLab.style.opacity = '0.8';
    sampLab.style.fontSize = '12px';

    const sampInputLab = el('span', null, 'input/DEBUG every');
    sampInputLab.style.opacity = '0.8';
    sampInputLab.style.fontSize = '12px';
    const sampInput = mkNumInput(20);
    sampInput.style.width = '72px';

    const sampObsLab = el('span', null, 'obs/DEBUG every');
    sampObsLab.style.opacity = '0.8';
    sampObsLab.style.fontSize = '12px';
    const sampObs = mkNumInput(20);
    sampObs.style.width = '72px';

    logRow3.append(sampLab, sampInputLab, sampInput, sampObsLab, sampObs);

    const logTail = /** @type {HTMLTextAreaElement} */ (document.createElement('textarea'));
    logTail.readOnly = true;
    logTail.rows = 8;
    logTail.placeholder = 'Logs will appear here (JSONL)...';
    logTail.style.width = '100%';
    logTail.style.marginTop = '6px';
    logTail.style.padding = '6px';
    logTail.style.borderRadius = '10px';
    logTail.style.border = '1px solid rgba(255,255,255,0.18)';
    logTail.style.background = 'rgba(0,0,0,0.25)';
    logTail.style.color = '#fff';
    logTail.style.fontSize = '11px';
    logTail.style.opacity = '0.92';
    logTail.style.resize = 'vertical';

    const refreshTail = () => {
      try {
        const lines = evoLogger.getTail(200);
        logTail.value = lines.join('\n');
        logTail.scrollTop = logTail.scrollHeight;
      } catch (e) {
        try { logTail.value = String(e); } catch {}
      }
    };

    const applyLogSettingsFromUI = () => {
      const on = !!logEnable.checked;
      evoLogger.setEnabled(on);
      _lsSet(LOG_ENABLED_KEY, on ? '1' : '0');

      const lvl = String(levelSel.value || 'INFO');
      evoLogger.setLevel(/** @type {any} */ (lvl));
      _lsSet(LOG_LEVEL_KEY, lvl);

      const ml = Number(maxLinesInput.value);
      evoLogger.setMaxLines(Number.isFinite(ml) ? (ml | 0) : 5000);
      _lsSet(LOG_MAXLINES_KEY, String(evoLogger.maxLines | 0));

      const kb = Number(maxKbInput.value);
      const bytes = (Number.isFinite(kb) ? (kb | 0) : 2048) * 1024;
      evoLogger.setMaxBytes(bytes);
      _lsSet(LOG_MAXBYTES_KEY, String(evoLogger.maxBytes | 0));

      const si = Number(sampInput.value);
      const so = Number(sampObs.value);
      evoLogger.setSampleEvery({ input: (Number.isFinite(si) ? (si | 0) : 20), obs: (Number.isFinite(so) ? (so | 0) : 20) });
      _lsSet(LOG_SAMPLE_INPUT_KEY, String(evoLogger.sampleEvery.input | 0));
      _lsSet(LOG_SAMPLE_OBS_KEY, String(evoLogger.sampleEvery.obs | 0));

      const enabledCats = cats.filter((x) => x.cb.checked).map((x) => x.cb.dataset.cat || '').filter(Boolean);
      // If none checked => allow all (null)
      evoLogger.setCategories(enabledCats.length ? enabledCats : null);
      _lsSet(LOG_CATS_KEY, enabledCats.join(','));
    };

    const initLogUIFromStorage = () => {
      // enabled
      logEnable.checked = (_lsGet(LOG_ENABLED_KEY) === '1');

      // level
      const savedLvl = String(_lsGet(LOG_LEVEL_KEY) || 'INFO').toUpperCase();
      levelSel.value = (savedLvl === 'ERROR' || savedLvl === 'WARN' || savedLvl === 'INFO' || savedLvl === 'DEBUG') ? savedLvl : 'INFO';

      // max lines
      const savedML = Number(_lsGet(LOG_MAXLINES_KEY));
      maxLinesInput.value = String(Number.isFinite(savedML) && savedML > 0 ? (savedML | 0) : 5000);

      // max bytes (KB shown)
      const savedMB = Number(_lsGet(LOG_MAXBYTES_KEY));
      const mb = (Number.isFinite(savedMB) && savedMB > 0) ? (savedMB | 0) : (2 * 1024 * 1024);
      maxKbInput.value = String(Math.max(32, (mb / 1024) | 0));

      // sampling
      const sIn = Number(_lsGet(LOG_SAMPLE_INPUT_KEY));
      const sOb = Number(_lsGet(LOG_SAMPLE_OBS_KEY));
      sampInput.value = String(Number.isFinite(sIn) && sIn > 0 ? (sIn | 0) : 20);
      sampObs.value = String(Number.isFinite(sOb) && sOb > 0 ? (sOb | 0) : 20);

      // cats
      const catStr = String(_lsGet(LOG_CATS_KEY) || '').trim();
      const set = new Set(catStr ? catStr.split(',').map(s => s.trim()).filter(Boolean) : []);
      if (set.size === 0) {
        // default: enable common categories
        ['match','match_anomaly','physics_route','generation','storage'].forEach((k) => set.add(k));
      }
      cats.forEach((x) => { x.cb.checked = set.has(String(x.cb.dataset.cat || '')); });
    };

    initLogUIFromStorage();
    applyLogSettingsFromUI();

    // Wire events
    [logEnable, levelSel, maxLinesInput, maxKbInput, sampInput, sampObs, ...cats.map(x=>x.cb)].forEach((node) => {
      node.addEventListener('change', () => {
        applyLogSettingsFromUI();
        refreshTail();
      });
    });

    logRefresh.addEventListener('click', () => refreshTail());
    logDownload.addEventListener('click', () => { try { evoLogger.download(); } catch {} refreshTail(); });
    logClear.addEventListener('click', () => { try { evoLogger.clear(); } catch {} refreshTail(); });

    logBox.append(logTitle, logRow1, logRow2, logRow3, logTail);
// Initial fill
    refreshTail();


    panel.appendChild(importFile);

    panel.append(header, row1, hr(), oppBox, hr(), hofBox, hr(), histBox, hr(), logBox);

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
    this._oppWrSelf = wrSelf;
    this._oppWrBase = wrBase;
    this._oppWrPhys = wrPhys;
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


  /**
   * Update opponent winrates display.
   * @param {{self?:{n?:number, winRate?:number}, baseline?:{n?:number, winRate?:number}, physics?:{n?:number, winRate?:number}}} summary
   */
  setOpponentWinrates(summary) {
    const fmt = (label, obj) => {
      const n = obj && Number.isFinite(obj.n) ? (obj.n | 0) : 0;
      const wr = obj && Number.isFinite(obj.winRate) ? obj.winRate : null;
      if (!n || wr == null) return `${label}: —`;
      return `${label}: ${(wr * 100).toFixed(1)}% (n=${n})`;
    };
    if (this._oppWrSelf) this._oppWrSelf.textContent = fmt('Self', summary && summary.self);
    if (this._oppWrBase) this._oppWrBase.textContent = fmt('Baseline', summary && summary.baseline);
    if (this._oppWrPhys) this._oppWrPhys.textContent = fmt('Physics', summary && summary.physics);
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