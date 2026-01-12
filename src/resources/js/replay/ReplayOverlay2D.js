export class ReplayOverlay2D {
  /**
   * @param {{ baseWidth:number, baseHeight:number, mountEl: HTMLElement, targetCanvas: HTMLCanvasElement, resolution?: number }} args
   */
  constructor({ baseWidth, baseHeight, mountEl, targetCanvas, resolution = 2 }) {
    this.baseWidth = baseWidth;   // 432
    this.baseHeight = baseHeight; // 304
    this.resolution = resolution; // settings.RESOLUTION과 맞추기(현재 2)
    this.targetCanvas = targetCanvas;
    this.mountEl = mountEl;

    const canvas = document.createElement('canvas');

    // 내부 해상도는 Pixi처럼 고해상도로
    canvas.width = Math.round(baseWidth * this.resolution);
    canvas.height = Math.round(baseHeight * this.resolution);

    // 화면상 크기/위치는 targetCanvas(#game-canvas)에 맞춰서 매 프레임/리사이즈마다 갱신
    canvas.style.position = 'absolute';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '5000';

    // mount는 relative여야 absolute가 먹음
    mountEl.style.position = mountEl.style.position || 'relative';
    mountEl.appendChild(canvas);

    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // 논리 좌표계(432x304)로 그릴 수 있게 transform 고정
    this._applyTransform();

    this.running = false;
    this.paused = false;
    this.speed = 1;

    this._trace = [];
    this._meta = null;
    this._framePos = 0;
    this._raf = 0;

    // 캔버스 위치/크기 동기화
    this._syncLayout();

    // 리사이즈 대응(가장 확실한 방식)
    this._ro = new ResizeObserver(() => this._syncLayout());
    this._ro.observe(this.targetCanvas);
    window.addEventListener('resize', this._syncLayout, { passive: true });
  }

  destroy() {
    this.stop();
    try { this._ro?.disconnect(); } catch {}
    window.removeEventListener('resize', this._syncLayout);
    this.canvas.remove();
  }

  _applyTransform() {
    const ctx = this.ctx;
    if (!ctx) return;
    // 매번 clear 전에 변형 유지되도록 reset
    ctx.setTransform(this.resolution, 0, 0, this.resolution, 0, 0);
  }

  // ✅ 핵심: overlay가 #game-canvas의 실제 화면 위치/크기를 따라감
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

  play(replayObj) {
    // replayObj: { trace, scoredBy, loser, loseReason ... } 또는 trace+meta 따로
    const trace = Array.isArray(replayObj?.trace) ? replayObj.trace : (Array.isArray(replayObj) ? replayObj : []);
    this._meta = replayObj && !Array.isArray(replayObj) ? replayObj : null;

    this.stop();
    this._trace = trace;
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

    // 레이아웃은 “가끔”만 맞춰도 되지만, 안전하게 프레임마다 한번(비용 거의 없음)
    this._syncLayout();

    if (!this.paused) this._framePos += this.speed;

    const idx = Math.floor(this._framePos);
    if (idx >= this._trace.length) {
      this.stop();
      return;
    }

    this._draw(this._trace[idx], idx);

    this._raf = requestAnimationFrame(this._loop);
  };

  _draw(frame, idx) {
    const ctx = this.ctx;
    if (!ctx) return;

    // transform 다시 걸어두기(외부 코드가 건드려도 안전)
    ctx.setTransform(this.resolution, 0, 0, this.resolution, 0, 0);
    ctx.clearRect(0, 0, this.baseWidth, this.baseHeight);

    // ✅ 디버그 가이드(정확히 중앙)
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(this.baseWidth / 2, 0);
    ctx.lineTo(this.baseWidth / 2, this.baseHeight);
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    ctx.globalAlpha = 1;

    // trace 포맷: trainer._buildPointReplay가 만든 compact 포맷 기준
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
      const who = (this._meta.scoredBy === 1) ? 'P1 scored' : (this._meta.scoredBy === 2) ? 'P2 scored' : 'unknown';
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
