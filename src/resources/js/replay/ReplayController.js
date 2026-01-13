// src/resources/js/replay/ReplayController.js

import { ReplayOverlay2D } from './ReplayOverlay2D.js';
import { ReplayPanel } from './ReplayPanel.js';
import { pauseGameForReplay, resumeGameForReplay } from '../ui.js';

export class ReplayController {
  /**
   * @param {{ container: HTMLElement, storage: any, pikaVolley: any, width: number, height: number }} args
   */
  constructor({ container, storage, pikaVolley, width, height }) {
    this.container = container;
    this.storage = storage;
    this.pikaVolley = pikaVolley;

    this._selectedId = null;
    this._cached = [];

    this._isReplayActive = false;

    const targetCanvasEl = document.getElementById('game-canvas');
    if (!(targetCanvasEl instanceof HTMLCanvasElement)) {
      throw new Error('#game-canvas is not a canvas element');
    }

    // Overlay는 1번만 생성(반복 생성 금지)
    this.overlay = new ReplayOverlay2D({
      baseWidth: width,
      baseHeight: height,
      mountEl: this.container,
      targetCanvas: targetCanvasEl,
      resolution: 2,
      onFinish: () => {
        // replay가 자연 종료되면 자동 복구
        this._restoreAfterReplay();
      },
    });

    // 패널
    this.panel = new ReplayPanel({
      mount: this.container,
      onPlay: () => this.playSelected(),
      onPause: () => this.overlay.togglePause(),
      onStop: () => this.stop(),
      onSpeed: (v) => this.overlay.setSpeed(v),
      onSelect: (id) => {
        this._selectedId = id;
      },
      onRefresh: () => this.refresh(),
    });

    // 초기 목록 로드
    void this.refresh();
  }

  async refresh() {
    if (!this.storage?.listReplays) return;

    const wins = (this.storage.listWinReplays)
      ? await this.storage.listWinReplays({ limit: 3, offset: 0 })
      : [];

    const recent = await this.storage.listReplays({ limit: 10, offset: 0 });

    // merge: wins first, then recent (dedupe by baseId/id)
    const seen = new Set();
    const merged = [];

    for (const it of wins) {
      const key = it?.baseId ?? (typeof it?.id === 'string' ? it.id.replace(/^win:/, '') : it?.id);
      if (key != null) seen.add(String(key));
      merged.push(it);
    }

    for (const it of recent) {
      const key = it?.baseId ?? (typeof it?.id === 'string' ? it.id.replace(/^win:/, '') : it?.id);
      const k = String(key);
      if (seen.has(k)) continue;
      seen.add(k);
      merged.push(it);
    }

    this._cached = merged;
    this.panel.renderList(this._cached);
  }

  _freezeForReplay() {
    if (this._isReplayActive) return;
    this._isReplayActive = true;

    // 게임 캔버스는 보여준다(배경으로 유지)
    const canvasEl = document.getElementById('game-canvas');
    if (canvasEl) canvasEl.style.display = 'block';

    // ✅ 기존 게임 pause 시스템을 사용
    pauseGameForReplay(this.pikaVolley);

    // 프로젝트 관례: replay 중엔 음소거(필요 없으면 제거 가능)
    window.__PV_TRAINING_MUTE__ = true;
  }

  _restoreAfterReplay() {
    if (!this._isReplayActive) return;
    this._isReplayActive = false;

    // overlay는 onFinish에서 stop()을 이미 호출하므로 여기서 stop을 강제하지 않음
    // (Stop 버튼으로 종료한 경우엔 stop()에서 overlay.stop()이 호출됨)

    resumeGameForReplay(this.pikaVolley);
    window.__PV_TRAINING_MUTE__ = false;
  }

  async playSelected() {
    const id = this._selectedId || this.panel.getSelectedId();
    const it = this._cached.find((x) => x.id === id);
    if (!it) return;

    this._freezeForReplay();

    // 속도 적용
    this.overlay.setSpeed(this.panel.getSpeed());

    // ✅ overlay는 trace 시각화만 수행
    this.overlay.play({
      trace: it.trace,
      scoredBy: it.scoredBy,
      loser: it.loser,
      loseReason: it.loseReason,
    });
  }

  stop() {
    this.overlay.stop();
    this._restoreAfterReplay();
  }

  // 개발 중 핫리로드/재생성 대비(선택)
  destroy() {
    try {
      this.stop();
    } catch {}

    try {
      this.overlay.destroy?.();
    } catch {}

    try {
      // ReplayPanel 내부 구조에 따라 조정
      this.panel.panel?.remove?.();
    } catch {}
  }
}
