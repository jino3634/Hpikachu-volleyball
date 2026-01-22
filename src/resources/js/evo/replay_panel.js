/**
 * Replay panel (bottom-left) - shows recent wins and recent games.
 */
'use strict';

/**
 * @typedef {{
 *  mount?: HTMLElement,
 *  onPlay?: (item:any)=>void,
 *  onStop?: ()=>void,
 * }} ReplayPanelOptions
 */

function _fmtMode(mode) {
  if (mode === 'physics') return 'physics';
  if (mode === 'baseline') return 'baseline';
  return 'self';
}
function _fmtTimeAgo(t) {
  const dt = Date.now() - (Number(t) || 0);
  if (!Number.isFinite(dt) || dt < 0) return '';
  const s = Math.floor(dt / 1000);
  if (s < 60) return s + 's ago';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm ago';
  const h = Math.floor(m / 60);
  if (h < 48) return h + 'h ago';
  const d = Math.floor(h / 24);
  return d + 'd ago';
}
function _safeText(v) {
  try { return String(v ?? ''); } catch { return ''; }
}

export class ReplayPanel {
  /** @param {ReplayPanelOptions} opts */
  constructor(opts = {}) {
    this._opts = opts;
    const mount = opts.mount ? opts.mount : document.body;

    /** @type {HTMLDivElement} */
    const panel = document.createElement('div');
    panel.id = 'replay-panel';
    panel.style.position = 'absolute';
    panel.style.left = '12px';
    panel.style.bottom = '12px';
    panel.style.zIndex = '9998';
    panel.style.minWidth = '320px';
    panel.style.maxWidth = '460px';
    panel.style.padding = '10px';
    panel.style.border = '1px solid rgba(255,255,255,0.35)';
    panel.style.borderRadius = '10px';
    panel.style.background = 'rgba(0,0,0,0.55)';
    panel.style.backdropFilter = 'blur(6px)';
    panel.style.color = '#fff';
    panel.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
    panel.style.fontSize = '12px';
    panel.style.pointerEvents = 'auto';

    const titleRow = document.createElement('div');
    titleRow.style.display = 'flex';
    titleRow.style.alignItems = 'center';
    titleRow.style.justifyContent = 'space-between';

    const title = document.createElement('div');
    title.textContent = 'Replay';
    title.style.fontWeight = '700';
    title.style.fontSize = '13px';

    const btnRow = document.createElement('div');
    btnRow.style.display = 'flex';
    btnRow.style.gap = '6px';

    const stopBtn = document.createElement('button');
    stopBtn.textContent = 'Stop';
    stopBtn.style.padding = '4px 8px';
    stopBtn.style.cursor = 'pointer';
    stopBtn.addEventListener('click', () => {
      try { this._opts.onStop && this._opts.onStop(); } catch (e) { console.error(e); }
    });

    btnRow.appendChild(stopBtn);
    titleRow.appendChild(title);
    titleRow.appendChild(btnRow);

    const hr = document.createElement('div');
    hr.style.height = '1px';
    hr.style.margin = '8px 0';
    hr.style.background = 'rgba(255,255,255,0.18)';

    const winsTitle = document.createElement('div');
    winsTitle.textContent = 'Recent 3 Wins (P1)';
    winsTitle.style.fontWeight = '600';

    const winsList = document.createElement('div');
    winsList.style.marginTop = '4px';
    winsList.style.display = 'flex';
    winsList.style.flexDirection = 'column';
    winsList.style.gap = '4px';

    const gamesTitle = document.createElement('div');
    gamesTitle.textContent = 'Recent 10 Games';
    gamesTitle.style.fontWeight = '600';
    gamesTitle.style.marginTop = '8px';

    const gamesList = document.createElement('div');
    gamesList.style.marginTop = '4px';
    gamesList.style.display = 'flex';
    gamesList.style.flexDirection = 'column';
    gamesList.style.gap = '4px';
    gamesList.style.maxHeight = '220px';
    gamesList.style.overflow = 'auto';

    panel.appendChild(titleRow);
    panel.appendChild(hr);
    panel.appendChild(winsTitle);
    panel.appendChild(winsList);
    panel.appendChild(gamesTitle);
    panel.appendChild(gamesList);

    mount.appendChild(panel);

    this.el = panel;
    this._winsListEl = winsList;
    this._gamesListEl = gamesList;

    this.setData({ recentWins: [], recentGames: [] });
  }

  /** @param {{recentWins:any[], recentGames:any[]}} data */
  setData(data) {
    const wins = (data && Array.isArray(data.recentWins)) ? data.recentWins : [];
    const games = (data && Array.isArray(data.recentGames)) ? data.recentGames : [];

    this._renderList(this._winsListEl, wins, 3);
    this._renderList(this._gamesListEl, games, 10);
  }

  _renderList(rootEl, items, limit) {
    if (!rootEl) return;
    rootEl.innerHTML = '';
    const arr = Array.isArray(items) ? items : [];
    const show = arr.slice(Math.max(0, arr.length - limit)); // keep newest end
    if (!show.length) {
      const empty = document.createElement('div');
      empty.textContent = '(empty)';
      empty.style.opacity = '0.75';
      rootEl.appendChild(empty);
      return;
    }
    // newest first
    show.reverse();
    for (let i = 0; i < show.length; i++) {
      const item = show[i];
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.justifyContent = 'space-between';
      row.style.gap = '8px';
      row.style.padding = '4px 6px';
      row.style.border = '1px solid rgba(255,255,255,0.14)';
      row.style.borderRadius = '8px';
      row.style.background = 'rgba(255,255,255,0.06)';

      const left = document.createElement('div');
      left.style.display = 'flex';
      left.style.flexDirection = 'column';
      left.style.gap = '2px';
      left.style.minWidth = '0';

      const line1 = document.createElement('div');
      const mode = _fmtMode(item?.mode);
      const score = `${_safeText(item?.scoreP1)}-${_safeText(item?.scoreP2)}`;
      const winner = (Number(item?.winner) === 1) ? 'W' : (Number(item?.winner) === 2 ? 'L' : '?');
      line1.textContent = `${winner} [${mode}] ${score}`;
      line1.style.fontWeight = '600';
      line1.style.whiteSpace = 'nowrap';
      line1.style.overflow = 'hidden';
      line1.style.textOverflow = 'ellipsis';

      const line2 = document.createElement('div');
      const ago = _fmtTimeAgo(item?.t);
      const seed = _safeText(item?.seed);
      const frames = _safeText(item?.frames);
      line2.textContent = `${ago}  seed:${seed}  frames:${frames}`;
      line2.style.opacity = '0.8';
      line2.style.whiteSpace = 'nowrap';
      line2.style.overflow = 'hidden';
      line2.style.textOverflow = 'ellipsis';

      left.appendChild(line1);
      left.appendChild(line2);

      const playBtn = document.createElement('button');
      playBtn.textContent = 'Play';
      playBtn.style.padding = '4px 10px';
      playBtn.style.cursor = 'pointer';
      playBtn.addEventListener('click', () => {
        try { this._opts.onPlay && this._opts.onPlay(item); } catch (e) { console.error(e); }
      });

      row.appendChild(left);
      row.appendChild(playBtn);
      rootEl.appendChild(row);
    }
  }
}
