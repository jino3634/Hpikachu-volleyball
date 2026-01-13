/**
 * This is the main script which executes the game.
 * General explanations for the all source code files of the game are following.
 *
 ********************************************************************************************************************
 * This web version of the Pikachu Volleyball is made by
 * reverse engineering the core part of the original Pikachu Volleyball game
 * which is developed by "1997 (C) SACHI SOFT / SAWAYAKAN Programmers" & "1997 (C) Satoshi Takenouchi".
 *
 * This game is for education purposes and no commercial purposes.
 * This game is open to the public and the source code is open.
 *
 * If you want to contact the original developer, please contact to
 * - 1997 (C) SACHI SOFT / SAWAYAKAN Programmers
 * - 1997 (C) Satoshi Takenouchi
 *
 * If you want to contact the developer of this web version, please contact to
 * - GitHub: https://github.com/gjbae1212/pikachu-volleyball
 *
 ********************************************************************************************************************
 *
 * For the all files (including this file), in order to easily find modified and newly added source code,
 * some special comments are used as following.
 * - // MODIFIED:    The original code is modified.
 * - // ADDED:       The code is newly added.
 * - // DELETED:     The original code is deleted.
 *
 ********************************************************************************************************************
 *
 * This file contains the main script which executes the game.
 * This file is loaded by index.html and executed when the html is loaded.
 *
 * The source code is written with JavaScript ES6 and uses Pixi.js library.
 *
 * This file can be thought as a main function (like main() in C language).
 *
 ********************************************************************************************************************
 *
 * Most important classes of this game are following.
 * - PikachuVolleyball:  The main class of the game.
 * - Player:             The player class.
 * - Ball:               The ball class.
 * - Game:               The game class.
 * - State:              The state class.
 *
 ********************************************************************************************************************
 *
 * The game is executed as following order.
 * 1. Initialize Pixi.js renderer.
 * 2. Load resources (images, sounds, etc.).
 * 3. Setup game (create game objects, setup UI, etc.).
 * 4. Start the game loop (ticker).
 *
 * The game loop is executed by Pixi.js ticker.
 * The ticker calls gameLoop() method of PikachuVolleyball class.
 * The gameLoop() method updates the game objects and renders the game.
 *
 ********************************************************************************************************************
 *
 * The source code is open and can be modified.
 * However, the source code is complex and hard to understand.
 * Therefore, the code is modified and added with special comments.
 *
 * Enjoy the game!
 */

/**
 * NOTE:
 * This file has been modified to integrate training/replay features.
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

// ✅ Training
import { Trainer } from './ai/trainer.js';

// ✅ Replay 구조정리 버전(별도 파일로 분리해뒀다는 전제)
// (아직 파일을 안 만들었으면, 일단 import/생성부를 주석 처리해도 컴파일은 됨)
import { ReplayController } from './replay/ReplayController.js';

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

// ✅ 기존 프로젝트 방식 유지 (Canvas 기반)
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

  // ✅ Trainer
  const trainer = new Trainer(pikaVolley, { /* ... 네가 쓰던 옵션 유지 ... */ });
  await trainer.init();

  // window.train 준비
  window.train = window.train || {};
  window.train.trainer = trainer;
  window.train.storage = trainer.storage;

  // 패널
  const trainingPanel = createTrainingControlPanel({ trainer, ticker });

  // ✅ Replay 구조 정리: Controller가 Overlay+Panel lifecycle 소유
  // (ReplayController.js 파일을 아직 안 만들었으면 여기 3줄을 주석처리하면 됨)
  const replayMount = document.getElementById('game-canvas-container') ?? document.body;
  const replayController = new ReplayController({
    container: replayMount,
    storage: trainer.storage,
    pikaVolley,
    width: 432,
    height: 304,
  });
  window.train.replay = replayController;

  // 콘솔용 병합
  Object.assign(window.train, {
    start: () => trainer.start(),
    stop: () => trainer.stop(),
    status: () => trainer.status(),
    export: () => trainer.exportToFile(),
    import: (opts) => trainer.importFromFile(opts),
    clear: () => trainer.clearAll(),
    setSpeed: (n) => { trainer.pointsPerTick = Math.max(1, n | 0); },

    // 패널 refresh 바로 호출하고 싶으면
    refreshPanels: () => {
      try { trainingPanel.refresh(); } catch {}
      try { replayController.refresh?.(); } catch {}
    },
  });

  start(pikaVolley);
}

/**
 * Minimal in-page UI for training.
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
  hideText.style.opacity = '0.9';
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

  const setCanvasVisible = (on) => {
    const canvas = document.getElementById('game-canvas');
    if (!canvas) return;
    canvas.style.display = on ? 'block' : 'none';
  };

  const startTraining = () => {
    window.__PV_TRAINING_MUTE__ = true;

    // 학습 중 화면 멈춤(원래 네 구조 유지)
    ticker.stop();
    trainer.start();

    if (hideChk.checked) setCanvasVisible(false);
    refresh();
  };

  const stopTraining = () => {
    trainer.stop();
    window.__PV_TRAINING_MUTE__ = false;

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
    const last1000Pct = (s.last1000Count > 0) ? (s.last1000Winrate * 100).toFixed(1) : '0.0';

    // ✅ status.textContent는 string만
    status.textContent = [
      `running: ${s.running ? 'ON' : 'OFF'}`,
      `graduated: ${s.graduated ? 'YES' : 'NO'}`,
      `episodes: ${s.totalEpisodes} (W ${s.wins} / L ${s.losses}, ${winratePct}%)`,
      `last1000: ${s.last1000Wins}/${s.last1000Count} (${last1000Pct}%)`,
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

function start(pikaVolley) {
  ticker.maxFPS = pikaVolley.normalFPS;
  ticker.add(() => {
    pikaVolley.gameLoop();
    renderer.render(stage);
  });
  ticker.start();
}
