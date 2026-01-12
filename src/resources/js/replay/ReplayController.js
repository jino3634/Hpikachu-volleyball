import { ReplayOverlay2D } from './ReplayOverlay2D.js';
import { ReplayPanel } from './ReplayPanel.js';

export class ReplayController {
  /**
   * @param {{ container: HTMLElement, storage: any, pikaVolley?: any, width:number, height:number }} args
   */
  constructor({ container, storage, pikaVolley, width, height }) {
    this.container = container;
    this.storage = storage;
    this.pikaVolley = pikaVolley;

    const targetCanvasEl = document.getElementById('game-canvas');
    if (!(targetCanvasEl instanceof HTMLCanvasElement)) {
    throw new Error('#game-canvas is not a canvas element');
    }

    this.overlay = new ReplayOverlay2D({
    baseWidth: width,
    baseHeight: height,
    mountEl: this.container,
    targetCanvas: targetCanvasEl,
    resolution: 2,
    });



    this._selectedId = null;
    this._cached = [];

    // gameLoop freeze/restore
    this._originalGameLoop = pikaVolley?.gameLoop?.bind(pikaVolley) ?? null;

    this.panel = new ReplayPanel({
      mount: container,
      onPlay: () => this.playSelected(),
      onPause: () => this.overlay.togglePause(),
      onStop: () => this.stop(),
      onSpeed: (v) => this.overlay.setSpeed(v),
      onSelect: (id) => { this._selectedId = id; },
      onRefresh: () => this.refresh(),
    });

    this.refresh();
  }

  async refresh() {
    if (!this.storage?.listReplays) return;
    this._cached = await this.storage.listReplays({ limit: 10, offset: 0 });
    this.panel.renderList(this._cached);
  }

  _freezeGameLoop() {
    if (!this.pikaVolley || !this._originalGameLoop) return;
    this.pikaVolley.gameLoop = () => {
      // no-op: 물리/사운드 진행 차단
    };
  }

  _restoreGameLoop() {
    if (!this.pikaVolley || !this._originalGameLoop) return;
    this.pikaVolley.gameLoop = this._originalGameLoop;
  }

async playSelected() {
  const id = this._selectedId || this.panel.getSelectedId();
  const it = this._cached.find((x) => x.id === id);
  if (!it) return;

  window.__PV_TRAINING_MUTE__ = false;

  const canvasEl = document.getElementById('game-canvas');
  if (canvasEl) canvasEl.style.display = 'block';

  this._freezeGameLoop();
  this.overlay.setSpeed(this.panel.getSpeed());

  // ✅ listReplays()가 trace 포함 replay 객체를 이미 줌 (trainer._buildPointReplay 참고)
  this.overlay.play({
    trace: it.trace,
    scoredBy: it.scoredBy,
    loser: it.loser,
    loseReason: it.loseReason,
  });
}


  stop() {
    this.overlay.stop();
    this._restoreGameLoop();
    window.__PV_TRAINING_MUTE__ = false;
  }
  // ✅ 페이지 이동/핫리로드/패널 재생성 등 대비용
destroy() {
    try { this.overlay.destroy?.(); } catch {}
    try { this.panel.panel?.remove(); } catch {} // panel DOM 제거(구조에 따라 조정)
    }
}
