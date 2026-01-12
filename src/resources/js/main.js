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

  // ✅ Trainer 연결 (P1 학습, P2 builtin AI)
  const trainer = new Trainer(pikaVolley, {
    learningPlayer: 1,
    pointsPerTick: 1,
    tickDelayMs: 0,
    autosaveEveryEpisodes: 50,
    setWinTarget: 15,
    consecutiveSetWinsToGraduate: 3,
  });
  await trainer.init();
  createTrainingControlPanel({ trainer, ticker });

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
    if (hideChk.checked) setCanvasVisible(false);

    trainingPromise = trainer.start().finally(() => {
      trainingPromise = null;
      refresh();
    });
    refresh();
  };

  const stopTraining = () => {
    trainer.stop();
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



  // ✅ 콘솔에서 window.train으로 조작
  /** @type {any} */ (window).train = {
    start: () => trainer.start(),
    stop: () => trainer.stop(),
    status: () => trainer.status(),
    export: () => trainer.exportToFile(),
    import: (opts) => trainer.importFromFile(opts),
    clear: () => trainer.clearAll(),
    setSpeed: (n) => { trainer.pointsPerTick = Math.max(1, n | 0); },
  };

  start(pikaVolley);
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
