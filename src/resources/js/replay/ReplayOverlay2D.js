// src/resources/js/replay/ReplayOverlay2D.js

export class ReplayOverlay2D {
  /**
   * @param {{
   *   baseWidth: number,
   *   baseHeight: number,
   *   mountEl: HTMLElement,
   *   targetCanvas: HTMLCanvasElement,
   *   resolution?: number,
   *   onFinish?: (() => void) | null
   * }} args
   */
  constructor({
    baseWidth,
    baseHeight,
    mountEl,
    targetCanvas,
    resolution = 2,
    onFinish = null,
  }) {
    this.baseWidth = baseWidth;     // trace 좌표계 폭(예: 432)
    this.baseHeight = baseHeight;   // trace 좌표계 높이(예: 304)
    this.mountEl = mountEl;
    this.targetCanvas = targetCanvas;
    this.resolution = resolution;

    this.onFinish = typeof onFinish === 'function' ? onFinish : null;

    const canvas = document.createElement('canvas');

    // 내부 픽셀(고해상도). draw는 base 좌표계로 하고 transform으로 resolution 반영
    canvas.width = Math.round(baseWidth * this.resolution);
    canvas.height = Math.round(baseHeight * this.resolution);

    canvas.style.position = 'absolute';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '5000';

    mountEl.style.position = mountEl.style.position || 'relative';
    mountEl.appendChild(canvas);

    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    this.running = false;
    this.paused = false;
    this.speed = 1;

    this._trace = [];
    this._meta = null;
    this._framePos = 0;
    this._raf = 0;

    // 초기 transform + 위치 동기화
    this._applyTransform();
    this._syncLayout();

    // resize 대응 (target canvas 크기/위치 바뀌면 overlay도 따라감)
    this._ro = new ResizeObserver(() => this._syncLayout());
    this._ro.observe(this.targetCanvas);
    window.addEventListener('resize', this._syncLayout, { passive: true });
  }

  destroy() {
    this.stop();
    try {
      this._ro?.disconnect();
    } catch {}
    try {
      window.removeEventListener('resize', this._syncLayout);
    } catch {}
    try {
      this.canvas?.remove();
    } catch {}
  }

  _applyTransform() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.setTransform(this.resolution, 0, 0, this.resolution, 0, 0);
  }

  // ✅ 핵심: overlay가 #game-canvas의 화면상의 위치/크기를 그대로 따라감
  _syncLayout = () => {
    const mountRect = this.mountEl.getBoundingClientRect();
    const targetRect = this.targetCanvas.getBoundingClientRect();

    const left = targetRect.left - mountRect.left;
    const top = targetRect.top - mountRect.top;

    this.canvas.style.left = `${left}px`;
    this.canvas.style.top = `${top}px`;
    this.canvas.style.width = `${targetRect.width}px`;
    this.canvas.style.height = `${targetRect.height}px`;
  };

  setSpeed(n) {
    const v = Number(n);
    this.speed = Number.isFinite(v) && v > 0 ? v : 1;
  }

  togglePause() {
    this.paused = !this.paused;
  }

  /**
   * @param {{ trace: any[], scoredBy?: number, loser?: number, loseReason?: string } | any[]} replay
   */
  play(replay) {
    let trace = [];
    let meta = null;

    if (Array.isArray(replay)) {
    trace = replay;
    } else if (replay && typeof replay === 'object' && Array.isArray(replay.trace)) {
    trace = replay.trace;
    meta = replay;
    }

    this.stop();

    this._trace = trace;
    this._meta = meta;
    this._framePos = 0;

    this.running = true;
    this.paused = false;

    this._syncLayout();
    this._loop();
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;

    this._trace = [];
    this._meta = null;
    this._framePos = 0;

    this._clear();
  }

  _clear() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.setTransform(this.resolution, 0, 0, this.resolution, 0, 0);
    ctx.clearRect(0, 0, this.baseWidth, this.baseHeight);
  }

  _loop = () => {
    if (!this.running) return;

    // 화면 위치/크기 안전 동기화
    this._syncLayout();

    if (!this.paused) this._framePos += this.speed;

    const idx = Math.floor(this._framePos);
    if (idx >= this._trace.length) {
      this.stop();
      try {
        this.onFinish?.(); // ✅ 자연 종료 콜백
      } catch {}
      return;
    }

    this._draw(this._trace[idx], idx);
    this._raf = requestAnimationFrame(this._loop);
  };

  _draw(frame, idx) {
    const ctx = this.ctx;
    if (!ctx) return;

    ctx.setTransform(this.resolution, 0, 0, this.resolution, 0, 0);
    ctx.clearRect(0, 0, this.baseWidth, this.baseHeight);

    // (선택) 살짝 어둡게 깔아주면 실제 화면 위에서 더 잘 보임
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, this.baseWidth, this.baseHeight);
    ctx.globalAlpha = 1;

    // 디버그 기준선(네트 중앙)
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(this.baseWidth / 2, 0);
    ctx.lineTo(this.baseWidth / 2, this.baseHeight);
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    ctx.globalAlpha = 1;

    const p1 = frame?.p1;
    const p2 = frame?.p2;
    const ball = frame?.ball;

    if (p1) this._drawCircle(p1.x, p1.y, 14, '#00ffcc');
    if (p2) this._drawCircle(p2.x, p2.y, 14, '#ffcc00');
    if (ball) this._drawCircle(ball.x, ball.y, 6, '#ffffff');

    // HUD
    const scores = frame?.scores ?? null;
    ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    ctx.fillStyle = 'rgba(255,255,255,0.95)';

    const scoreText = scores ? `Score: P1 ${scores.p1 ?? '?'} - P2 ${scores.p2 ?? '?'}` : 'Score: ?';
    const infoText = `Replay frame ${idx + 1}/${this._trace.length}  speed ${this.speed}x  ${this.paused ? '[PAUSED]' : ''}`;

    ctx.fillText(scoreText, 10, 16);
    ctx.fillText(infoText, 10, 32);

    if (this._meta) {
      const who =
        (this._meta.scoredBy === 1) ? 'P1 scored' :
        (this._meta.scoredBy === 2) ? 'P2 scored' : 'unknown';
      ctx.fillText(`${who}  reason=${this._meta.loseReason ?? ''}`, 10, 48);
    }
  }

  _drawCircle(x, y, r, color) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}
