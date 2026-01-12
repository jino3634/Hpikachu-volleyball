/**
 * This is the main script which executes the game.
 * General explanations for the all source code files of the game are following.
 *
 ********************************************************************************************************************
 * This web version of the Pikachu Volleyball is made by
 * reverse engineering the core part of the original Pikachu Volleyball game
 * which is developed by "1997 (C) SACHI SOFT / SAWAYAKAN Programmers" & "1997 (C) Satoshi Takenouchi".
 *
 * "physics.js", "cloud_and_wave.js", and some codes in "view.js" are the results of this reverse engineering.
 * Refer to the comments in each file for the machine code addresses of the original functions.
 ********************************************************************************************************************
 *
 * This web version game is mainly composed of three parts which follows MVC pattern.
 *  1) "physics.js" (Model): The physics engine which takes charge of the dynamics of the ball and the players (Pikachus).
 *                           It is gained by reverse engineering the machine code of the original game.
 *  2) "view.js" (View): The rendering part of the game which depends on pixi.js (https://www.pixijs.com/, https://github.com/pixijs/pixi.js) library.
 *                       Some codes in this part is gained by reverse engineering the original machine code.
 *  3) "pikavolley.js" (Controller): Make the game work by controlling the Model and the View according to the user input.
 *
 * And explanations for other source files are below.
 *  - "cloud_and_wave.js": This is also a Model part which takes charge of the clouds and wave motion in the game. Of course, it is also rendered by "view.js".
 *                         It is also gained by reverse engineering the original machine code.
 *  - "keyboard.js": Support the Controller("pikavolley.js") to get a user input via keyboard.
 *  - "audio.js": The game audio or sounds. It depends on pixi-sound (https://github.com/pixijs/pixi-sound) library.
 *  - "rand.js": For the random function used in the Models ("physics.js", "cloud_and_wave.js").
 *  - "assets_path.js": For the assets (image files, sound files) locations.
 *  - "ui.js": For the user interface (menu bar, buttons etc.) of the html page.
 */
'use strict';
import { settings } from '@pixi/settings';
import { SCALE_MODES } from '@pixi/constants';
import { Renderer, BatchRenderer, autoDetectRenderer } from '@pixi/core';
import { Prepare } from '@pixi/prepare';
import { Container } from '@pixi/display';
import { Loader } from '@pixi/loaders';
import { SpritesheetLoader } from '@pixi/spritesheet';
import { Ticker } from '@pixi/ticker';
import { CanvasRenderer } from '@pixi/canvas-renderer';
import { CanvasSpriteRenderer } from '@pixi/canvas-sprite';
import { CanvasPrepare } from '@pixi/canvas-prepare';
import '@pixi/canvas-display';
import { PikachuVolleyball } from './pikavolley.js';
import { ASSETS_PATH } from './assets_path.js';
import { setUpUI } from './ui.js';

// ✅ 추가
import { Trainer } from './ai/trainer.js';

// Reference for how to use Renderer.registerPlugin:
Renderer.registerPlugin('prepare', Prepare);
Renderer.registerPlugin('batch', BatchRenderer);
// Reference for how to use CanvasRenderer.registerPlugin:
CanvasRenderer.registerPlugin('prepare', CanvasPrepare);
CanvasRenderer.registerPlugin('sprite', CanvasSpriteRenderer);
Loader.registerPlugin(SpritesheetLoader);

// Settings
settings.RESOLUTION = 2;
settings.SCALE_MODE = SCALE_MODES.NEAREST;
settings.ROUND_PIXELS = true;

const renderer = autoDetectRenderer({
  width: 432,
  height: 304,
  antialias: false,
  backgroundColor: 0x000000,
  backgroundAlpha: 1,
  forceCanvas: true,
});

const stage = new Container();
const ticker = new Ticker();
const loader = new Loader();

renderer.view.setAttribute('id', 'game-canvas');
document.getElementById('game-canvas-container').appendChild(renderer.view);
renderer.render(stage);

loader.add(ASSETS_PATH.SPRITE_SHEET);
for (const prop in ASSETS_PATH.SOUNDS) {
  loader.add(ASSETS_PATH.SOUNDS[prop]);
}

setUpInitialUI();

function setUpInitialUI() {
  const loadingBox = document.getElementById('loading-box');
  const progressBar = document.getElementById('progress-bar');
  loader.onProgress.add(() => {
    progressBar.style.width = `${loader.progress}%`;
  });
  loader.onComplete.add(() => {
    loadingBox.classList.add('hidden');
  });

  const aboutBox = document.getElementById('about-box');
  const aboutBtn = document.getElementById('about-btn');
  const closeAboutBtn = document.getElementById('close-about-btn');
  const gameDropdownBtn = document.getElementById('game-dropdown-btn');
  const optionsDropdownBtn = document.getElementById('options-dropdown-btn');
  // @ts-ignore
  gameDropdownBtn.disabled = true;
  // @ts-ignore
  optionsDropdownBtn.disabled = true;

  const closeAboutBox = () => {
    if (!aboutBox.classList.contains('hidden')) {
      aboutBox.classList.add('hidden');
      // @ts-ignore
      aboutBtn.disabled = true;
    }
    aboutBtn.getElementsByClassName('text-play')[0].classList.add('hidden');
    aboutBtn.getElementsByClassName('text-about')[0].classList.remove('hidden');
    aboutBtn.classList.remove('glow');
    closeAboutBtn.getElementsByClassName('text-play')[0].classList.add('hidden');
    closeAboutBtn.getElementsByClassName('text-close')[0].classList.remove('hidden');
    closeAboutBtn.classList.remove('glow');

    loader.load(setup); // ✅ setup is called after loader finishes loading
    loadingBox.classList.remove('hidden');
    aboutBtn.removeEventListener('click', closeAboutBox);
    closeAboutBtn.removeEventListener('click', closeAboutBox);
  };

  aboutBtn.addEventListener('click', closeAboutBox);
  closeAboutBtn.addEventListener('click', closeAboutBox);
}

/**
 * Set up the game and the full UI, and start the game.
 * ✅ Trainer init 때문에 async로 변경
 */
async function setup() {
  const pikaVolley = new PikachuVolleyball(stage, loader.resources);
  setUpUI(pikaVolley, ticker);

  const trainer = new Trainer(pikaVolley, { /* ... */ });
  await trainer.init();

  window.train = window.train || {};
  window.train.trainer = trainer;
  window.train.storage = trainer.storage;

  createTrainingControlPanel({ trainer, ticker });
  createReplayPanel({ trainer, ticker, pikaVolley });

  // 콘솔용 병합도 여기서 하고
  window.train = window.train || {};
  Object.assign(window.train, {
    start: () => trainer.start(),
    stop: () => trainer.stop(),
    status: () => trainer.status(),
    export: () => trainer.exportToFile(),
    import: (opts) => trainer.importFromFile(opts),
    clear: () => trainer.clearAll(),
    setSpeed: (n) => { trainer.pointsPerTick = Math.max(1, n | 0); },
    trainer,
    storage: trainer.storage,
  });

  start(pikaVolley);   // ✅ 여기서 게임 시작
}                    // ✅ setup 끝!



  /**
 * Minimal in-page UI for training.
 * - Start/Stop
 * - Status readout
 * - Hide canvas while training (default)
 *
 * @param {{ trainer: import('./ai/trainer.js').Trainer, ticker: import('@pixi/ticker').Ticker }} args
 */
function createTrainingControlPanel({ trainer, ticker }) {
  const mount = document.getElementById('game-canvas-container') ?? document.body;

  const panel = document.createElement('div');
  panel.id = 'training-panel';
  panel.style.position = 'absolute';
  panel.style.right = '12px';
  panel.style.bottom = '12px';
  panel.style.zIndex = '9999';
  panel.style.padding = '10px 12px';
  panel.style.background = 'rgba(0,0,0,0.75)';
  panel.style.border = '1px solid rgba(255,255,255,0.2)';
  panel.style.borderRadius = '10px';
  panel.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, Arial';
  panel.style.fontSize = '12px';
  panel.style.color = '#fff';
  panel.style.minWidth = '240px';
  panel.style.userSelect = 'none';

  const title = document.createElement('div');
  title.textContent = 'TRAINING';
  title.style.fontWeight = '700';
  title.style.letterSpacing = '0.08em';
  title.style.marginBottom = '6px';
  panel.appendChild(title);

  const row1 = document.createElement('div');
  row1.style.display = 'flex';
  row1.style.gap = '8px';
  row1.style.marginBottom = '8px';

  const btnStart = document.createElement('button');
  btnStart.textContent = 'Start';
  btnStart.style.flex = '1';
  const btnStop = document.createElement('button');
  btnStop.textContent = 'Stop';
  btnStop.style.flex = '1';

  for (const b of [btnStart, btnStop]) {
    b.style.padding = '6px 8px';
    b.style.borderRadius = '8px';
    b.style.border = '1px solid rgba(255,255,255,0.25)';
    b.style.background = 'rgba(255,255,255,0.08)';
    b.style.color = '#fff';
    b.style.cursor = 'pointer';
  }
  row1.appendChild(btnStart);
  row1.appendChild(btnStop);
  panel.appendChild(row1);

  const options = document.createElement('div');
  options.style.display = 'flex';
  options.style.alignItems = 'center';
  options.style.gap = '8px';
  options.style.marginBottom = '8px';

  const hideLabel = document.createElement('label');
  hideLabel.style.display = 'flex';
  hideLabel.style.alignItems = 'center';
  hideLabel.style.gap = '6px';
  const hideChk = document.createElement('input');
  hideChk.type = 'checkbox';
  hideChk.checked = true;
  hideLabel.appendChild(hideChk);
  const hideText = document.createElement('span');
  hideText.textContent = 'Hide game while training';
  hideLabel.appendChild(hideText);
  options.appendChild(hideLabel);
  panel.appendChild(options);

  const status = document.createElement('pre');
  status.style.margin = '0';
  status.style.whiteSpace = 'pre-wrap';
  status.style.lineHeight = '1.25';
  status.style.opacity = '0.9';
  panel.appendChild(status);

  mount.style.position = mount.style.position || 'relative';
  mount.appendChild(panel);

  const canvasEl = document.getElementById('game-canvas');

  const setCanvasVisible = (visible) => {
    if (!canvasEl) return;
    canvasEl.style.display = visible ? 'block' : 'none';
  };

  let trainingPromise = null;

  const startTraining = () => {
    const s = trainer.status();
    if (s.graduated) return;
    if (s.running) return;

    // Prevent double stepping: stop the visual loop
    ticker.stop();
    window.__PV_TRAINING_MUTE__ = true;
    if (hideChk.checked) setCanvasVisible(false);

    trainingPromise = trainer.start().finally(() => {
      trainingPromise = null;
      refresh();
    });
    refresh();
  };

  const stopTraining = () => {
    trainer.stop();
    window.__PV_TRAINING_MUTE__ = false;
    // Resume visuals
    setCanvasVisible(true);
    ticker.start();
    refresh();
  };

  btnStart.onclick = startTraining;
  btnStop.onclick = stopTraining;

  hideChk.onchange = () => {
    const s = trainer.status();
    if (s.running) {
      setCanvasVisible(!hideChk.checked);
    }
  };

  const refresh = () => {
    const s = trainer.status();
    const winratePct = (s.totalEpisodes > 0) ? (s.winrate * 100).toFixed(1) : '0.0';
    status.textContent = [
      `running: ${s.running ? 'ON' : 'OFF'}`,
      `graduated: ${s.graduated ? 'YES' : 'NO'}`,
      `episodes: ${s.totalEpisodes} (W ${s.wins} / L ${s.losses}, ${winratePct}%)`,
      `set target: ${s.setWinTarget}, streak: ${s.consecutiveSetWins}/${s.consecutiveSetWinsToGraduate}`,
      `current set: P1 ${s.currentSet?.p1 ?? 0} - P2 ${s.currentSet?.p2 ?? 0} (set #${s.currentSet?.index ?? 1})`,
      `speed: ${s.pointsPerTick} point(s)/tick, delay: ${s.tickDelayMs}ms`,
    ].join('\n');

    btnStart.disabled = s.running || s.graduated;
    btnStop.disabled = !s.running;
  };

  refresh();
  setInterval(refresh, 500);

  return { refresh };
}

class ReplayOverlay2D {
  /**
   * @param {{ width:number, height:number, mountEl: HTMLElement }} args
   */
  constructor({ width, height, mountEl }) {
    this.width = width;
    this.height = height;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.style.position = 'absolute';
    canvas.style.left = '0';
    canvas.style.top = '0';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '5000'; // pixi canvas 위
    canvas.style.imageRendering = 'pixelated';

    // mountEl은 game-canvas-container 같은 상대 위치 컨테이너
    mountEl.style.position = mountEl.style.position || 'relative';
    mountEl.appendChild(canvas);

    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    this.running = false;
    this.paused = false;
    this.speed = 1; // 1x,2x,4x...
    this._raf = 0;
    this._framePos = 0; // float
    this._trace = [];
    this._meta = null;
  }

  setSpeed(n) {
    this.speed = Math.max(0.25, Number(n) || 1);
  }

  start(replay) {
    this.stop();

    this._meta = replay;
    this._trace = Array.isArray(replay.trace) ? replay.trace : [];
    this._framePos = 0;
    this.running = true;
    this.paused = false;

    this._loop();
  }

  togglePause() {
    if (!this.running) return;
    this.paused = !this.paused;
  }

  stop() {
    this.running = false;
    this.paused = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
    this._trace = [];
    this._meta = null;
    this._clear();
  }

  _clear() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, this.width, this.height);
  }

  _loop = () => {
    if (!this.running) return;

    if (!this.paused) {
      // 60fps 기준으로 speed만큼 진행 (간단/안정)
      this._framePos += this.speed;
    }

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

    ctx.clearRect(0, 0, this.width, this.height);

    // 배경 가이드(선택): 중앙 네트 라인
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(this.width / 2, 0);
    ctx.lineTo(this.width / 2, this.height);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // 좌표
    const p1 = frame.p1;
    const p2 = frame.p2;
    const ball = frame.ball;

    // player draw (단순 원)
    if (p1) this._drawCircle(p1.x, p1.y, 14, '#00ffcc');
    if (p2) this._drawCircle(p2.x, p2.y, 14, '#ffcc00');

    // ball draw
    if (ball) this._drawCircle(ball.x, ball.y, 6, '#ffffff');

    // 상단 텍스트 HUD
    const scores = frame.scores;
    ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    const scoreText = scores ? `Score: P1 ${scores.p1 ?? '?'} - P2 ${scores.p2 ?? '?'}` : 'Score: ?';
    const infoText = `Replay frame ${idx+1}/${this._trace.length}  speed ${this.speed}x  ${this.paused ? '[PAUSED]' : ''}`;
    ctx.fillText(scoreText, 10, 16);
    ctx.fillText(infoText, 10, 32);

    // 메타 표시(포인트 결과)
    if (this._meta) {
      const who = (this._meta.scoredBy === 1) ? 'P1 scored' : (this._meta.scoredBy === 2) ? 'P2 scored' : 'unknown';
      ctx.fillText(`${who}  reason=${this._meta.loseReason}`, 10, 48);
    }
  }

  _drawCircle(x, y, r, color) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000';
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

function createReplayPanel({ trainer, ticker, pikaVolley }) {
  const mount = document.getElementById('game-canvas-container') ?? document.body;

  // ✅ overlay 생성 (게임 캔버스 위)
  const overlay = new ReplayOverlay2D({ width: 432, height: 304, mountEl: mount });

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
  panel.style.minWidth = '360px';
  panel.style.maxWidth = '460px';
  panel.style.userSelect = 'none';

  const title = document.createElement('div');
  title.textContent = 'REPLAYS (last 10 points)';
  title.style.fontWeight = '700';
  title.style.letterSpacing = '0.04em';
  title.style.marginBottom = '6px';
  panel.appendChild(title);

  // row1: refresh/play
  const row1 = document.createElement('div');
  row1.style.display = 'flex';
  row1.style.gap = '8px';
  row1.style.marginBottom = '8px';

  const btnRefresh = document.createElement('button');
  btnRefresh.textContent = 'Refresh';
  btnRefresh.style.flex = '1';

  const btnPlay = document.createElement('button');
  btnPlay.textContent = 'Play';
  btnPlay.style.flex = '1';

  for (const b of [btnRefresh, btnPlay]) {
    b.style.padding = '6px 8px';
    b.style.borderRadius = '8px';
    b.style.border = '1px solid rgba(255,255,255,0.25)';
    b.style.background = 'rgba(255,255,255,0.08)';
    b.style.color = '#fff';
    b.style.cursor = 'pointer';
  }

  row1.appendChild(btnRefresh);
  row1.appendChild(btnPlay);
  panel.appendChild(row1);

  // row2: pause/stop/speed
  const row2 = document.createElement('div');
  row2.style.display = 'flex';
  row2.style.gap = '8px';
  row2.style.marginBottom = '8px';

  const btnPause = document.createElement('button');
  btnPause.textContent = 'Pause/Resume';
  btnPause.style.flex = '1';

  const btnStop = document.createElement('button');
  btnStop.textContent = 'Stop';
  btnStop.style.flex = '1';

  const speedSel = document.createElement('select');
  speedSel.style.flex = '1';
  speedSel.style.padding = '6px 8px';
  speedSel.style.borderRadius = '8px';
  speedSel.style.border = '1px solid rgba(255,255,255,0.25)';
  speedSel.style.background = 'rgba(255,255,255,0.08)';
  speedSel.style.color = '#fff';
  for (const s of [1, 2, 4, 8]) {
    const opt = document.createElement('option');
    opt.value = String(s);
    opt.textContent = `${s}x`;
    speedSel.appendChild(opt);
  }
  speedSel.value = '2'; // 기본 2x

  for (const b of [btnPause, btnStop]) {
    b.style.padding = '6px 8px';
    b.style.borderRadius = '8px';
    b.style.border = '1px solid rgba(255,255,255,0.25)';
    b.style.background = 'rgba(255,255,255,0.08)';
    b.style.color = '#fff';
    b.style.cursor = 'pointer';
  }

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

  let selectedId = null;
  let cached = [];

  // ✅ replay 중엔 게임 로직을 멈추고 렌더만 유지(오버레이가 주인공)
  const originalGameLoop = pikaVolley?.gameLoop?.bind(pikaVolley);
  const freezeGameLoop = () => {
    if (!pikaVolley || !originalGameLoop) return;
    pikaVolley.gameLoop = () => {
      // no-op: renderer.render는 ticker 콜백에서 계속 돌지만
      // gameLoop가 physics 진행/사운드 유발을 막아준다
    };
  };
  const restoreGameLoop = () => {
    if (!pikaVolley || !originalGameLoop) return;
    pikaVolley.gameLoop = originalGameLoop;
  };

  const renderList = (items) => {
    list.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.style.opacity = '0.8';
      empty.textContent = 'No replays saved yet.';
      list.appendChild(empty);
      return;
    }

    for (const it of items) {
      const row = document.createElement('div');
      row.style.display = 'grid';
      row.style.gridTemplateColumns = '18px 1fr';
      row.style.gap = '8px';
      row.style.padding = '6px 4px';
      row.style.borderRadius = '8px';
      row.style.cursor = 'pointer';
      row.style.background = (it.id === selectedId) ? 'rgba(255,255,255,0.12)' : 'transparent';

      const bullet = document.createElement('div');
      bullet.textContent = (it.id === selectedId) ? '▶' : '•';
      bullet.style.opacity = '0.9';

      const text = document.createElement('div');
      const t = new Date(it.createdAt).toLocaleString();
      const who = (it.scoredBy === 1) ? 'P1 scored' : (it.scoredBy === 2) ? 'P2 scored' : 'unknown';
      text.textContent = `[${t}] ${who} | ${it.loseReason} | frames=${it.frames}`;

      row.onclick = () => {
        selectedId = it.id;
        renderList(items);
      };

      row.appendChild(bullet);
      row.appendChild(text);
      list.appendChild(row);
    }
  };

  const refresh = async () => {
    cached = await trainer.storage.listReplays({ limit: 10, offset: 0 });
    renderList(cached);
  };

  btnRefresh.onclick = refresh;

  btnPlay.onclick = async () => {
    const it = cached.find(x => x.id === selectedId);
    if (!it) return;

    // 학습 중이면 멈추고, 사운드 mute 해제(리플레이는 시각화만; 사운드는 아직 안 씀)
    trainer.stop();
    window.__PV_TRAINING_MUTE__ = false;

    // 캔버스 표시
    const canvasEl = document.getElementById('game-canvas');
    if (canvasEl) canvasEl.style.display = 'block';

    // ✅ 게임 물리 진행 차단
    freezeGameLoop();

    overlay.setSpeed(Number(speedSel.value));
    overlay.start(it);
  };

  btnPause.onclick = () => overlay.togglePause();

  btnStop.onclick = () => {
    overlay.stop();
    restoreGameLoop();
  };

  speedSel.onchange = () => overlay.setSpeed(Number(speedSel.value));

  refresh();
}


/**
 * Start the game.
 * @param {PikachuVolleyball} pikaVolley
 */
function start(pikaVolley) {
  ticker.maxFPS = pikaVolley.normalFPS;
  ticker.add(() => {
    pikaVolley.gameLoop();
    renderer.render(stage);
  });
  ticker.start();
}
