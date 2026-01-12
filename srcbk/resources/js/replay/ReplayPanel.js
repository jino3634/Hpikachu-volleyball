export class ReplayPanel {
  /**
   * @param {{ mount: HTMLElement, onPlay:Function, onPause:Function, onStop:Function, onSpeed:Function, onSelect:Function, onRefresh:Function }} args
   */
  constructor({ mount, onPlay, onPause, onStop, onSpeed, onSelect, onRefresh }) {
    const panel = document.createElement('div');
    panel.id = 'replay-panel';
    panel.style.position = 'absolute';
    panel.style.left = '12px';
    panel.style.bottom = '12px';
    panel.style.zIndex = '9999';
    panel.style.padding = '10px 12px';
    panel.style.background = 'rgba(0,0,0,0.75)';
    panel.style.border = '1px solid rgba(255,255,255,0.2)';
    panel.style.borderRadius = '10px';
    panel.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, Arial';
    panel.style.fontSize = '12px';
    panel.style.color = '#fff';
    panel.style.minWidth = '260px';
    panel.style.userSelect = 'none';

    const title = document.createElement('div');
    title.textContent = 'REPLAY';
    title.style.fontWeight = '700';
    title.style.letterSpacing = '0.08em';
    title.style.marginBottom = '6px';
    panel.appendChild(title);

    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '8px';
    row1.style.marginBottom = '8px';

    const btnRefresh = document.createElement('button');
    btnRefresh.textContent = 'Refresh';
    btnRefresh.onclick = () => onRefresh();

    const btnPlay = document.createElement('button');
    btnPlay.textContent = 'Play';
    btnPlay.onclick = () => onPlay();

    [btnRefresh, btnPlay].forEach((b) => {
      b.style.flex = '1';
      b.style.cursor = 'pointer';
    });

    row1.appendChild(btnRefresh);
    row1.appendChild(btnPlay);
    panel.appendChild(row1);

    const row2 = document.createElement('div');
    row2.style.display = 'flex';
    row2.style.gap = '8px';
    row2.style.marginBottom = '8px';

    const btnPause = document.createElement('button');
    btnPause.textContent = 'Pause';
    btnPause.onclick = () => onPause();

    const btnStop = document.createElement('button');
    btnStop.textContent = 'Stop';
    btnStop.onclick = () => onStop();

    const speedSel = document.createElement('select');
    ['0.5', '1', '2', '4', '8'].forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = `${s}x`;
      speedSel.appendChild(opt);
    });
    speedSel.value = '1';
    speedSel.onchange = () => onSpeed(Number(speedSel.value));

    [btnPause, btnStop].forEach((b) => (b.style.cursor = 'pointer'));

    row2.appendChild(btnPause);
    row2.appendChild(btnStop);
    row2.appendChild(speedSel);
    panel.appendChild(row2);

    const list = document.createElement('div');
    list.style.maxHeight = '220px';
    list.style.overflow = 'auto';
    list.style.borderTop = '1px solid rgba(255,255,255,0.12)';
    list.style.paddingTop = '6px';
    panel.appendChild(list);

    mount.style.position = mount.style.position || 'relative';
    mount.appendChild(panel);

    this.panel = panel;
    this.list = list;
    this.speedSel = speedSel;

    this._selectedId = null;
    this._onSelect = onSelect;
  }

  setSpeedValue(v) {
    this.speedSel.value = String(v);
  }

  /**
   * @param {Array<{id:string, createdAt:number, scoredBy:number, loser:number, loseReason:string, frames:number}>} items
   */
  renderList(items) {
    const list = this.list;
    list.innerHTML = '';

    if (!items.length) {
      const empty = document.createElement('div');
      empty.textContent = '(no replays)';
      empty.style.opacity = '0.75';
      list.appendChild(empty);
      return;
    }

    items.forEach((it) => {
      const row = document.createElement('div');
      row.style.padding = '6px 6px';
      row.style.cursor = 'pointer';
      row.style.borderRadius = '6px';
      row.style.marginBottom = '4px';
      row.style.background = (it.id === this._selectedId) ? 'rgba(255,255,255,0.12)' : 'transparent';

      const t = new Date(it.createdAt);
      const who = (it.scoredBy === 1) ? 'P1' : (it.scoredBy === 2) ? 'P2' : '?';
      row.textContent = `${t.toLocaleString()}  ${who}  frames=${it.frames ?? 0}  reason=${it.loseReason ?? ''}`;

      row.onclick = () => {
        this._selectedId = it.id;
        this._onSelect(it.id);
        // rerender selection highlight
        this.renderList(items);
      };

      list.appendChild(row);
    });
  }

  getSelectedId() {
    return this._selectedId;
  }

  getSpeed() {
    return Number(this.speedSel.value) || 1;
  }
}
