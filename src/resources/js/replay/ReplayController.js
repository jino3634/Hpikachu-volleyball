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
    baseWidth: 432,
    baseHeight: 304,
    mountEl: this.container,      // 또는 생성자 param으로 받은 값이면 그걸 써도 됨
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

    // 리플레이는 시각화 전용: 학습은 멈추는 쪽이 안전(필요하면 main에서 trainer.stop 호출)
    // mute 플래그는 프로젝트 기존 관례 유지
    window.__PV_TRAINING_MUTE__ = false;

    // 캔버스 표시
    const canvasEl = document.getElementById('game-canvas');
    if (canvasEl) canvasEl.style.display = 'block';

    this._freezeGameLoop();
    this.overlay.setSpeed(this.panel.getSpeed());

    const targetCanvas =
    /** @type {HTMLCanvasElement} */ (
        document.getElementById('game-canvas')
    );


    this.overlay = new ReplayOverlay2D({
    baseWidth: 432,
    baseHeight: 304,
    mountEl: this.container,
    targetCanvas,
    resolution: 2, // settings.RESOLUTION과 맞추기
    });

  }

  stop() {
    this.overlay.stop();
    this._restoreGameLoop();
    window.__PV_TRAINING_MUTE__ = false;
  }
}
