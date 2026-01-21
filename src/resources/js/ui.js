/**
 * Manages event listeners relevant to the UI (menu bar, buttons, etc.) of the web page
 */
'use strict';

import { localStorageWrapper } from './utils/local_storage_wrapper.js';
import {
  startEvolution,
  stopEvolution,
  getEvolutionState,
  isEvolutionRunning,
} from './evo/runner.js';
import { loadBest, saveBest } from './evo/storage.js';
import { chooseAction, defaultGenome } from './evo/policy_weighted.js';
import { makeObservation } from './evo/observation.js';
import { decidePhysicsAI } from './physics_ai.js';


/** @typedef {import('./pikavolley.js').PikachuVolleyball} PikachuVolleyball */
/** @typedef {import('@pixi/ticker').Ticker} Ticker */
/** @typedef {{graphic?: string, bgm?: string, sfx?: string, speed?: string, winningScore?: string}} Options */

/**
 * Enum for "game paused by what?".
 * The greater the number, the higher the precedence.
 *
 * @readonly
 * @enum {number}
 */
const PauseResumePrecedence = {
  pauseBtn: 3,
  messageBox: 2,
  dropdown: 1,
  notPaused: 0,
};

/**
 * Manages pausing and resuming of the game
 */
const pauseResumeManager = {
  /** @type {number} PauseResumePrecedence enum */
  _precedence: PauseResumePrecedence.notPaused,
  /**
   * Pause game
   * @param {PikachuVolleyball} pikaVolley
   * @param {number} precedence PauseResumePrecedence enum
   */
  pause: function (pikaVolley, precedence) {
    // @ts-ignore
    if (precedence > this._precedence) {
      pikaVolley.paused = true;
      this._precedence = precedence;
    }
  },
  /**
   * Resume game
   * @param {PikachuVolleyball} pikaVolley
   * @param {number} precedence PauseResumePrecedence enum
   */
  resume: function (pikaVolley, precedence) {
    if (precedence === this._precedence) {
      pikaVolley.paused = false;
      this._precedence = PauseResumePrecedence.notPaused;
    }
  },
};

/**
 * Set up the user interface: menu bar, buttons, dropdowns, submenus, etc.
 * @param {PikachuVolleyball} pikaVolley
 * @param {Ticker} ticker
 */
export function setUpUI(pikaVolley, ticker) {
  /**
   * Apply options
   * @param {Options} options
   */
  const applyOptions = (options) => {
    setSelectedOptionsBtn(options);
    switch (options.graphic) {
      case 'sharp':
        document.getElementById('game-canvas').classList.remove('graphic-soft');
        break;
      case 'soft':
        document.getElementById('game-canvas').classList.add('graphic-soft');
        break;
    }
    switch (options.bgm) {
      case 'on':
        pikaVolley.audio.turnBGMVolume(true);
        break;
      case 'off':
        pikaVolley.audio.turnBGMVolume(false);
        break;
    }
    switch (options.sfx) {
      case 'stereo':
        pikaVolley.audio.turnSFXVolume(true);
        pikaVolley.isStereoSound = true;
        break;
      case 'mono':
        pikaVolley.audio.turnSFXVolume(true);
        pikaVolley.isStereoSound = false;
        break;
      case 'off':
        pikaVolley.audio.turnSFXVolume(false);
        break;
    }
    switch (options.speed) {
      case 'slow':
        pikaVolley.normalFPS = 20;
        ticker.maxFPS = pikaVolley.normalFPS;
        break;
      case 'medium':
        pikaVolley.normalFPS = 25;
        ticker.maxFPS = pikaVolley.normalFPS;
        break;
      case 'fast':
        pikaVolley.normalFPS = 30;
        ticker.maxFPS = pikaVolley.normalFPS;
        break;
    }
    switch (options.winningScore) {
      case '5':
        pikaVolley.winningScore = 5;
        break;
      case '10':
        pikaVolley.winningScore = 10;
        break;
      case '15':
        pikaVolley.winningScore = 15;
        break;
    }
  };

  /**
   * Save options
   * @param {Options} options
   */
  const saveOptions = (options) => {
    setSelectedOptionsBtn(options);
    if (options.graphic) {
      localStorageWrapper.set('pv-offline-graphic', options.graphic);
    }
    if (options.bgm) {
      localStorageWrapper.set('pv-offline-bgm', options.bgm);
    }
    if (options.sfx) {
      localStorageWrapper.set('pv-offline-sfx', options.sfx);
    }
    if (options.speed) {
      localStorageWrapper.set('pv-offline-speed', options.speed);
    }
    if (options.winningScore) {
      localStorageWrapper.set('pv-offline-winningScore', options.winningScore);
    }
  };

  /**
   * Load options
   * @returns {Options}
   */
  const loadOptions = () => ({
    graphic: localStorageWrapper.get('pv-offline-graphic'),
    bgm: localStorageWrapper.get('pv-offline-bgm'),
    sfx: localStorageWrapper.get('pv-offline-sfx'),
    speed: localStorageWrapper.get('pv-offline-speed'),
    winningScore: localStorageWrapper.get('pv-offline-winningScore'),
  });

  /**
   * Apply and save options
   * @param {Options} options
   */
  const applyAndSaveOptions = (options) => {
    applyOptions(options);
    saveOptions(options);
  };

  // Load and apply saved options
  applyOptions(loadOptions());

  setUpBtns(pikaVolley, applyAndSaveOptions);
  setUpToShowDropdownsAndSubmenus(pikaVolley);
  setUpEvoStatusOnLoad();

  // hide or show menubar if the user presses the "esc" key
  window.addEventListener('keydown', (event) => {
    if (event.code === 'Escape') {
      const menuBar = document.getElementById('menu-bar');
      if (menuBar.classList.contains('hidden')) {
        menuBar.classList.remove('hidden');
      } else {
        menuBar.classList.add('hidden');
      }
      event.preventDefault();
    } else if (event.code === 'Space') {
      const aboutBox = document.getElementById('about-box');
      if (aboutBox.classList.contains('hidden')) {
        event.preventDefault();
      }
    }
  });

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') {
      pikaVolley.audio.unmuteAll();
    } else {
      pikaVolley.audio.muteAll();
    }
  });
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = String(text);
}

function fmtPct(x) {
  return (Number(x || 0) * 100).toFixed(1) + '%';
}


function fmtNum(x, digits = 3) {
  const n = Number(x);
  if (!Number.isFinite(n)) return '0';
  return n.toFixed(digits);
}

/**
 * Apply an external AI controller to the live game.
 * This will control BOTH players as computer players:
 *   - P1 uses best genome
 *   - P2 uses baseline genome (or provided genomeP2)
 *
 * @param {import('./pikavolley.js').PikachuVolleyball} pikaVolley
 * @param {any} genomeP1
 * @param {any} genomeP2
 * @param {number} decisionInterval
 */
function applyExternalBestVsBaseline(pikaVolley, genomeP1, genomeP2, decisionInterval = 3) {
  const physics = pikaVolley && pikaVolley.physics;
  if (!physics) return false;

  const g1 = genomeP1 || defaultGenome();
  // Default baseline: self-play (P2 = P1) when genomeP2 is not provided.
  const g2 = (genomeP2 !== undefined && genomeP2 !== null) ? genomeP2 : g1;

  // Ensure both are computer players so physics will accept external control.
  physics.player1.isComputer = true;
  physics.player2.isComputer = true;

  // Ensure external control is enabled for both players.
  physics.player1.__useExternalAI = true;
  physics.player2.__useExternalAI = true;

  // Let physics schedule decision frames and hold inputs on non-decision frames.
  physics.decisionInterval = Math.max(1, (decisionInterval | 0));
  physics._aiFrameCounter = 0;

  // ✅ Must match Physics.aiController signature:
  // (playerIndex, me, ball, other, userInputForPlayer, meta)
  physics.aiController = (playerIndex, me, ball, other, userInput, meta) => {
    const ui = userInput;
    if (!ui) return;

    // We only want to decide on decision frames (physics will call us only then,
    // but keep this guard to be safe across versions).
    if (meta && meta.decisionFrame === false) return;

    const genome = (playerIndex === 1) ? g1 : g2;

    // Build observation from current physics state.
    const obs = makeObservation((meta && meta.physics) ? meta.physics : physics, playerIndex);
    if (!obs) {
      ui.xDirection = 0;
      ui.yDirection = 0;
      ui.powerHit = 0;
      return;
    }

    const act = chooseAction(obs, genome);

    ui.xDirection = act.xDirection | 0;
    ui.yDirection = act.yDirection | 0;
    ui.powerHit = act.powerHit ? 1 : 0;
  };

  return true;
}

function applyExternalBestVsBuiltin(pikaVolley, genomeP1, decisionInterval = 3) {
  const physics = pikaVolley && pikaVolley.physics;
  if (!physics) return false;

  const g1 = genomeP1 || defaultGenome();

  // P1 is controlled by the external best genome.
  // P2 is controlled by the built-in game AI.
  physics.player1.isComputer = true;
  physics.player2.isComputer = true;

  // Tell Physics to use external control only for P1.
  physics.player1.__useExternalAI = true;
  physics.player2.__useExternalAI = false;

  // Let physics schedule decision frames and hold inputs on non-decision frames.
  physics.decisionInterval = Math.max(1, (decisionInterval | 0));
  physics._aiFrameCounter = 0;

  // External AI hook: only fill inputs for P1. P2 stays on built-in AI path.
  physics.aiController = (playerIndex, me, ball, other, userInput, meta) => {
    if (playerIndex !== 1) return;
    const ui = userInput;
    if (!ui) return;
    if (meta && meta.decisionFrame === false) return;

    const obs = makeObservation((meta && meta.physics) ? meta.physics : physics, 1);
    if (!obs) {
      ui.xDirection = 0;
      ui.yDirection = 0;
      ui.powerHit = 0;
      return;
    }

    const act = chooseAction(obs, g1);
    ui.xDirection = act.xDirection | 0;
    ui.yDirection = act.yDirection | 0;
    ui.powerHit = act.powerHit ? 1 : 0;
  };

  return true;
}


function applyExternalBestVsPhysicsAI(pikaVolley, genomeP1, decisionInterval = 3) {
  const physics = pikaVolley && pikaVolley.physics;
  if (!physics) return false;

  const g1 = genomeP1 || defaultGenome();

  // P1: external best genome
  // P2: physics-based scripted AI (decidePhysicsAI)
  physics.player1.isComputer = true;
  physics.player2.isComputer = true;

  // Both use external controller (P2 uses scripted).
  physics.player1.__useExternalAI = true;
  physics.player2.__useExternalAI = true;

  // Ensure expectedLandingPointX is computed BEFORE callback on decision frames
  // (needed by physics-based AI). This is safe for normal gameplay too.
  physics.fastEvalMode = true;

  physics.decisionInterval = Math.max(1, (decisionInterval | 0));
  physics._aiFrameCounter = 0;

  physics.aiController = (playerIndex, me, ball, other, userInput, meta) => {
    if (!userInput) return;
    if (meta && meta.decisionFrame === false) return;

    if (playerIndex === 1) {
      const obs = makeObservation((meta && meta.physics) ? meta.physics : physics, 1);
      if (!obs) {
        userInput.xDirection = 0;
        userInput.yDirection = 0;
        userInput.powerHit = 0;
        return;
      }
      const act = chooseAction(obs, g1);
      userInput.xDirection = act.xDirection | 0;
      userInput.yDirection = act.yDirection | 0;
      userInput.powerHit = act.powerHit ? 1 : 0;
      return;
    }

    // Player 2 = physics scripted AI
    decidePhysicsAI(2, me, ball, other, userInput, meta);
  };

  return true;
}




function setUpEvoStatusOnLoad() {
  const s = getEvolutionState();
  if (!document.getElementById('evo-gen')) return; // UI not present
  setText('evo-gen', s.generation ?? 0);
  // Prefer eval winrate for display when available
  setText('evo-best-eval', fmtPct(s.bestEvalWinRate ?? s.bestWinRate ?? 0));
  setText('evo-best-train', fmtPct(s.bestWinRate ?? 0));
  setText('evo-best-fit', fmtNum(s.bestFitness ?? 0));
  setText('evo-hof', (s.hof && s.hof.length) ? s.hof.length : 0);
  setText('evo-running', s.running ? 'on' : 'off');
  setText('evo-avg-train', fmtPct(0));
  setText('evo-last', (s.lastSavedAt ? new Date(s.lastSavedAt).toLocaleString() : '-'));
}

function updateEvoStatus(payload) {
  setText('evo-gen', payload.generation ?? 0);
  setText('evo-best-eval', fmtPct(payload.bestEvalWinRate ?? payload.bestWinRate ?? 0));
  setText('evo-best-train', fmtPct(payload.bestWinRate ?? 0));
  setText('evo-best-fit', fmtNum(payload.bestFitness ?? 0));
  setText('evo-hof', (payload.hof && payload.hof.length) ? payload.hof.length : 0);
  setText('evo-running', payload.running ? 'on' : 'off');
  setText('evo-avg-train', fmtPct(payload.avgWinRate ?? 0));
  setText('evo-last', (payload.lastSavedAt ? new Date(payload.lastSavedAt).toLocaleString() : '-'));
}

/**
 * Attach event listeners to the buttons
 * @param {PikachuVolleyball} pikaVolley
 * @param {(options: Options) => void} applyAndSaveOptions
 */
function setUpBtns(pikaVolley, applyAndSaveOptions) {
  const gameDropdownBtn = document.getElementById('game-dropdown-btn');
  const optionsDropdownBtn = document.getElementById('options-dropdown-btn');
  const aboutBtn = document.getElementById('about-btn');
  // @ts-ignore
  gameDropdownBtn.disabled = false;
  // @ts-ignore
  optionsDropdownBtn.disabled = false;
  // @ts-ignore
  aboutBtn.disabled = false;

  const pauseBtn = document.getElementById('pause-btn');
  pauseBtn.addEventListener('click', () => {
    if (pauseBtn.classList.contains('selected')) {
      pauseBtn.classList.remove('selected');
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.pauseBtn);
    } else {
      pauseBtn.classList.add('selected');
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.pauseBtn);
    }
  });

  const restartBtn = document.getElementById('restart-btn');
  restartBtn.addEventListener('click', () => {
    if (pauseBtn.classList.contains('selected')) {
      pauseBtn.classList.remove('selected');
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.pauseBtn);
    }
    pikaVolley.restart();
  });

  const graphicSharpBtn = document.getElementById('graphic-sharp-btn');
  const graphicSoftBtn = document.getElementById('graphic-soft-btn');
  graphicSharpBtn.addEventListener('click', () => {
    applyAndSaveOptions({ graphic: 'sharp' });
  });
  graphicSoftBtn.addEventListener('click', () => {
    applyAndSaveOptions({ graphic: 'soft' });
  });

  const bgmOnBtn = document.getElementById('bgm-on-btn');
  const bgmOffBtn = document.getElementById('bgm-off-btn');
  bgmOnBtn.addEventListener('click', () => {
    applyAndSaveOptions({ bgm: 'on' });
  });
  bgmOffBtn.addEventListener('click', () => {
    applyAndSaveOptions({ bgm: 'off' });
  });

  const stereoBtn = document.getElementById('stereo-btn');
  const monoBtn = document.getElementById('mono-btn');
  const sfxOffBtn = document.getElementById('sfx-off-btn');
  stereoBtn.addEventListener('click', () => {
    applyAndSaveOptions({ sfx: 'stereo' });
  });
  monoBtn.addEventListener('click', () => {
    applyAndSaveOptions({ sfx: 'mono' });
  });
  sfxOffBtn.addEventListener('click', () => {
    applyAndSaveOptions({ sfx: 'off' });
  });

  // Game speed:
  //   slow: 1 frame per 50ms = 20 FPS
  //   medium: 1 frame per 40ms = 25 FPS
  //   fast: 1 frame per 33ms = 30.303030... FPS
  const slowSpeedBtn = document.getElementById('slow-speed-btn');
  const mediumSpeedBtn = document.getElementById('medium-speed-btn');
  const fastSpeedBtn = document.getElementById('fast-speed-btn');
  slowSpeedBtn.addEventListener('click', () => {
    applyAndSaveOptions({ speed: 'slow' });
  });
  mediumSpeedBtn.addEventListener('click', () => {
    applyAndSaveOptions({ speed: 'medium' });
  });
  fastSpeedBtn.addEventListener('click', () => {
    applyAndSaveOptions({ speed: 'fast' });
  });

  const winningScore5Btn = document.getElementById('winning-score-5-btn');
  const winningScore10Btn = document.getElementById('winning-score-10-btn');
  const winningScore15Btn = document.getElementById('winning-score-15-btn');
  const noticeBox1 = document.getElementById('notice-box-1');
  const noticeOKBtn1 = document.getElementById('notice-ok-btn-1');
  const winningScoreInNoticeBox1 = document.getElementById(
    'winning-score-in-notice-box-1'
  );
  function isWinningScoreAlreadyReached(winningScore) {
    const isGamePlaying =
      pikaVolley.state === pikaVolley.round ||
      pikaVolley.state === pikaVolley.afterEndOfRound ||
      pikaVolley.state === pikaVolley.beforeStartOfNextRound;
    if (
      isGamePlaying &&
      (pikaVolley.scores[0] >= winningScore ||
        pikaVolley.scores[1] >= winningScore)
    ) {
      return true;
    }
    return false;
  }
  const noticeBox2 = document.getElementById('notice-box-2');
  const noticeOKBtn2 = document.getElementById('notice-ok-btn-2');
  winningScore5Btn.addEventListener('click', () => {
    if (winningScore5Btn.classList.contains('selected')) {
      return;
    }
    if (pikaVolley.isPracticeMode === true) {
      noticeBox2.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    if (isWinningScoreAlreadyReached(5)) {
      winningScoreInNoticeBox1.textContent = '5';
      noticeBox1.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    applyAndSaveOptions({ winningScore: '5' });
  });
  winningScore10Btn.addEventListener('click', () => {
    if (winningScore10Btn.classList.contains('selected')) {
      return;
    }
    if (pikaVolley.isPracticeMode === true) {
      noticeBox2.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    if (isWinningScoreAlreadyReached(10)) {
      winningScoreInNoticeBox1.textContent = '10';
      noticeBox1.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    applyAndSaveOptions({ winningScore: '10' });
  });
  winningScore15Btn.addEventListener('click', () => {
    if (winningScore15Btn.classList.contains('selected')) {
      return;
    }
    if (pikaVolley.isPracticeMode === true) {
      noticeBox2.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    if (isWinningScoreAlreadyReached(15)) {
      winningScoreInNoticeBox1.textContent = '15';
      noticeBox1.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      // @ts-ignore
      aboutBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
      return;
    }
    applyAndSaveOptions({ winningScore: '15' });
  });
  noticeOKBtn1.addEventListener('click', () => {
    if (!noticeBox1.classList.contains('hidden')) {
      noticeBox1.classList.add('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = false;
      // @ts-ignore
      optionsDropdownBtn.disabled = false;
      // @ts-ignore
      aboutBtn.disabled = false;
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.messageBox);
    }
  });
  noticeOKBtn2.addEventListener('click', () => {
    if (!noticeBox2.classList.contains('hidden')) {
      noticeBox2.classList.add('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = false;
      // @ts-ignore
      optionsDropdownBtn.disabled = false;
      // @ts-ignore
      aboutBtn.disabled = false;
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.messageBox);
    }
  });

  const practiceModeOnBtn = document.getElementById('practice-mode-on-btn');
  const practiceModeOffBtn = document.getElementById('practice-mode-off-btn');
  practiceModeOnBtn.addEventListener('click', () => {
    practiceModeOffBtn.classList.remove('selected');
    practiceModeOnBtn.classList.add('selected');
    pikaVolley.isPracticeMode = true;
  });
  practiceModeOffBtn.addEventListener('click', () => {
    practiceModeOnBtn.classList.remove('selected');
    practiceModeOffBtn.classList.add('selected');
    pikaVolley.isPracticeMode = false;
  });

  const aboutBox = document.getElementById('about-box');
  const closeAboutBtn = document.getElementById('close-about-btn');
  aboutBtn.addEventListener('click', () => {
    if (aboutBox.classList.contains('hidden')) {
      aboutBox.classList.remove('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = true;
      // @ts-ignore
      optionsDropdownBtn.disabled = true;
      pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.messageBox);
    } else {
      aboutBox.classList.add('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = false;
      // @ts-ignore
      optionsDropdownBtn.disabled = false;
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.messageBox);
    }
  });
  closeAboutBtn.addEventListener('click', () => {
    if (!aboutBox.classList.contains('hidden')) {
      aboutBox.classList.add('hidden');
      // @ts-ignore
      gameDropdownBtn.disabled = false;
      // @ts-ignore
      optionsDropdownBtn.disabled = false;
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.messageBox);
    }
  });

  const resetToDefaultBtn = document.getElementById('reset-to-default-btn');
  resetToDefaultBtn.addEventListener('click', () => {
    // turn off practice mode
    practiceModeOffBtn.click();

    // and restore the reset options to default
    const defaultOptions = {
      graphic: 'sharp',
      bgm: 'on',
      sfx: 'stereo',
      speed: 'medium',
      winningScore: '15',
    };
    applyAndSaveOptions(defaultOptions);
  });

  // --------------------
  // Evolution controls
  // --------------------
  const evoStartBtn = document.getElementById('evo-start-btn');
  const evoStopBtn = document.getElementById('evo-stop-btn');
  if (evoStartBtn && evoStopBtn) {
    // init button state
    // @ts-ignore
    evoStopBtn.disabled = !isEvolutionRunning();

    evoStartBtn.addEventListener('click', async () => {
      // @ts-ignore
      evoStartBtn.disabled = true;
      // @ts-ignore
      evoStopBtn.disabled = false;

      // Evolution is headless; keep the game paused to avoid confusion.
      const pauseBtn = document.getElementById('pause-btn');
      if (pauseBtn && !pauseBtn.classList.contains('selected')) {
        pauseBtn.classList.add('selected');
        pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.pauseBtn);
      }

      await startEvolution({}, (s) => {
        updateEvoStatus(s);
      });
    });

    evoStopBtn.addEventListener('click', () => {
      stopEvolution();
      // @ts-ignore
      evoStartBtn.disabled = false;
      // @ts-ignore
      evoStopBtn.disabled = true;
    });


    // Opponent mode for "Apply best"
    // - self: best vs best (P2 = P1)
    // - baseline: best vs default baseline genome (P2 = defaultGenome())
    function getEvoOpponentMode() {
      const el = document.querySelector('input[name="evo-opp-mode"]:checked');
      const v = el ? String(el.value || '') : '';
      if (v === 'baseline') return 'baseline';
      if (v === 'builtin') return 'builtin';
      if (v === 'physics') return 'physics';
      return 'self';
    }

    // Extra evo buttons (apply/save/load)
    const evoApplyBtn = document.getElementById('evo-apply-btn');
    const evoSaveBtn = document.getElementById('evo-save-btn');
    const evoLoadBtn = document.getElementById('evo-load-btn');

    if (evoApplyBtn) {
      evoApplyBtn.addEventListener('click', () => {
        const s = getEvolutionState();
        const loaded = loadBest();
        const gBest = (s && s.bestGenome) ? s.bestGenome : (loaded ? loaded.genome : null);
        if (!gBest) {
          alert('No best genome yet. Run evolution or load a saved best first.');
          return;
        }
        const di = Number((s && s.config && s.config.decisionInterval) ? s.config.decisionInterval : 3);
        const oppMode = getEvoOpponentMode();
        const di2 = Math.max(1, di | 0);
        let ok = false;
        if (oppMode === 'builtin') {
          ok = applyExternalBestVsBuiltin(pikaVolley, gBest, di2);
        } else if (oppMode === 'physics') {
          ok = applyExternalBestVsPhysicsAI(pikaVolley, gBest, di2);
        } else {
          const g2 = (oppMode === 'baseline') ? defaultGenome() : null;
          ok = applyExternalBestVsBaseline(pikaVolley, gBest, g2, di2);
        }
        if (!ok) {
          alert('Apply failed (physics not ready).');
          return;
        }
        // Restart to apply cleanly.
        try { pikaVolley.restart(); } catch {}
      });
    }

    if (evoSaveBtn) {
      evoSaveBtn.addEventListener('click', () => {
        const s = getEvolutionState();
        if (!s || !s.bestGenome) {
          alert('No best genome to save yet.');
          return;
        }
        const savedAt = Date.now();
        const ok = saveBest({
          genome: s.bestGenome,
          bestWinRate: Number(s.bestWinRate ?? 0),
          bestEvalWinRate: Number(s.bestEvalWinRate ?? 0),
          bestFitness: Number(s.bestFitness ?? 0),
          generation: Number(s.generation ?? 0),
          savedAt,
        });
        if (!ok) alert('Save failed.');
        setText('evo-last', new Date(savedAt).toLocaleString());
      });
    }

    if (evoLoadBtn) {
      evoLoadBtn.addEventListener('click', () => {
        const b = loadBest();
        if (!b || !b.genome) {
          alert('No saved best found.');
          return;
        }
        const s = getEvolutionState();
        const di = Number((s && s.config && s.config.decisionInterval) ? s.config.decisionInterval : 3);
        const oppMode = getEvoOpponentMode();
        const di2 = Math.max(1, di | 0);
        let ok = false;
        if (oppMode === 'builtin') {
          ok = applyExternalBestVsBuiltin(pikaVolley, b.genome, di2);
        } else {
          const g2 = (oppMode === 'baseline') ? defaultGenome() : null;
          ok = applyExternalBestVsBaseline(pikaVolley, b.genome, g2, di2);
        }
        if (!ok) {
          alert('Load/apply failed (physics not ready).');
          return;
        }
        // Reflect loaded stats in UI for convenience.
        setText('evo-best-eval', fmtPct(b.bestEvalWinRate ?? b.bestWinRate ?? 0));
        setText('evo-best-train', fmtPct(b.bestWinRate ?? 0));
        setText('evo-best-fit', fmtNum(b.bestFitness ?? 0));
        setText('evo-gen', b.generation ?? 0);
        setText('evo-last', (b.savedAt ? new Date(b.savedAt).toLocaleString() : '-'));
        try { pikaVolley.restart(); } catch {}
      });
    }

  }
}

/**
 * Set selected (checked) options btn fit to options
 * @param {Options} options
 */
function setSelectedOptionsBtn(options) {
  if (options.graphic) {
    const graphicSharpBtn = document.getElementById('graphic-sharp-btn');
    const graphicSoftBtn = document.getElementById('graphic-soft-btn');
    switch (options.graphic) {
      case 'sharp':
        graphicSoftBtn.classList.remove('selected');
        graphicSharpBtn.classList.add('selected');
        break;
      case 'soft':
        graphicSharpBtn.classList.remove('selected');
        graphicSoftBtn.classList.add('selected');
        break;
    }
  }
  if (options.bgm) {
    const bgmOnBtn = document.getElementById('bgm-on-btn');
    const bgmOffBtn = document.getElementById('bgm-off-btn');
    switch (options.bgm) {
      case 'on':
        bgmOffBtn.classList.remove('selected');
        bgmOnBtn.classList.add('selected');
        break;
      case 'off':
        bgmOnBtn.classList.remove('selected');
        bgmOffBtn.classList.add('selected');
        break;
    }
  }
  if (options.sfx) {
    const stereoBtn = document.getElementById('stereo-btn');
    const monoBtn = document.getElementById('mono-btn');
    const sfxOffBtn = document.getElementById('sfx-off-btn');
    switch (options.sfx) {
      case 'stereo':
        monoBtn.classList.remove('selected');
        sfxOffBtn.classList.remove('selected');
        stereoBtn.classList.add('selected');
        break;
      case 'mono':
        sfxOffBtn.classList.remove('selected');
        stereoBtn.classList.remove('selected');
        monoBtn.classList.add('selected');
        break;
      case 'off':
        stereoBtn.classList.remove('selected');
        monoBtn.classList.remove('selected');
        sfxOffBtn.classList.add('selected');
        break;
    }
  }
  if (options.speed) {
    const slowSpeedBtn = document.getElementById('slow-speed-btn');
    const mediumSpeedBtn = document.getElementById('medium-speed-btn');
    const fastSpeedBtn = document.getElementById('fast-speed-btn');
    switch (options.speed) {
      case 'slow':
        mediumSpeedBtn.classList.remove('selected');
        fastSpeedBtn.classList.remove('selected');
        slowSpeedBtn.classList.add('selected');
        break;
      case 'medium':
        fastSpeedBtn.classList.remove('selected');
        slowSpeedBtn.classList.remove('selected');
        mediumSpeedBtn.classList.add('selected');
        break;
      case 'fast':
        slowSpeedBtn.classList.remove('selected');
        mediumSpeedBtn.classList.remove('selected');
        fastSpeedBtn.classList.add('selected');
        break;
    }
  }
  if (options.winningScore) {
    const winningScore5Btn = document.getElementById('winning-score-5-btn');
    const winningScore10Btn = document.getElementById('winning-score-10-btn');
    const winningScore15Btn = document.getElementById('winning-score-15-btn');
    switch (options.winningScore) {
      case '5':
        winningScore10Btn.classList.remove('selected');
        winningScore15Btn.classList.remove('selected');
        winningScore5Btn.classList.add('selected');
        break;
      case '10':
        winningScore15Btn.classList.remove('selected');
        winningScore5Btn.classList.remove('selected');
        winningScore10Btn.classList.add('selected');
        break;
      case '15':
        winningScore5Btn.classList.remove('selected');
        winningScore10Btn.classList.remove('selected');
        winningScore15Btn.classList.add('selected');
        break;
    }
  }
}

/**
 * Attach event listeners to show dropdowns and submenus properly
 * @param {PikachuVolleyball} pikaVolley
 */
function setUpToShowDropdownsAndSubmenus(pikaVolley) {
  // hide dropdowns and submenus if the user clicks outside of these
  window.addEventListener('click', (event) => {
    // @ts-ignore
    if (!event.target.matches('.dropdown-btn, .submenu-btn')) {
      hideSubmenus();
      hideDropdownsExcept('');
      pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.dropdown);
    }
  });

  // set up to show dropdowns
  document.getElementById('game-dropdown-btn').addEventListener('click', () => {
    toggleDropdown('game-dropdown', pikaVolley);
  });
  document
    .getElementById('options-dropdown-btn')
    .addEventListener('click', () => {
      toggleDropdown('options-dropdown', pikaVolley);
    });

  const evoDropdownBtn = document.getElementById('evo-dropdown-btn');
  if (evoDropdownBtn) {
    evoDropdownBtn.addEventListener('click', () => {
      toggleDropdown('evo-dropdown', pikaVolley);
    });
  }

  // set up to show submenus on mouseover event
  document
    .getElementById('graphic-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('graphic-submenu-btn', 'graphic-submenu');
    });
  document
    .getElementById('bgm-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('bgm-submenu-btn', 'bgm-submenu');
    });
  document
    .getElementById('sfx-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('sfx-submenu-btn', 'sfx-submenu');
    });
  document
    .getElementById('speed-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('speed-submenu-btn', 'speed-submenu');
    });
  document
    .getElementById('winning-score-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('winning-score-submenu-btn', 'winning-score-submenu');
    });
  document
    .getElementById('practice-mode-submenu-btn')
    .addEventListener('mouseover', () => {
      showSubmenu('practice-mode-submenu-btn', 'practice-mode-submenu');
    });
  document
    .getElementById('reset-to-default-btn')
    .addEventListener('mouseover', () => {
      hideSubmenus();
    });

  // set up to show submenus on click event
  // (it is for touch device equipped with physical keyboard)
  document.getElementById('bgm-submenu-btn').addEventListener('click', () => {
    showSubmenu('bgm-submenu-btn', 'bgm-submenu');
  });
  document.getElementById('sfx-submenu-btn').addEventListener('click', () => {
    showSubmenu('sfx-submenu-btn', 'sfx-submenu');
  });
  document.getElementById('speed-submenu-btn').addEventListener('click', () => {
    showSubmenu('speed-submenu-btn', 'speed-submenu');
  });
  document
    .getElementById('winning-score-submenu-btn')
    .addEventListener('click', () => {
      showSubmenu('winning-score-submenu-btn', 'winning-score-submenu');
    });
  document
    .getElementById('practice-mode-submenu-btn')
    .addEventListener('click', () => {
      showSubmenu('practice-mode-submenu-btn', 'practice-mode-submenu');
    });
  document
    .getElementById('reset-to-default-btn')
    .addEventListener('click', () => {
      hideSubmenus();
    });
}

/**
 * Toggle (show or hide) the dropdown menu
 * @param {string} dropdownID html element id of the dropdown to toggle
 * @param {PikachuVolleyball} pikaVolley
 */
function toggleDropdown(dropdownID, pikaVolley) {
  hideSubmenus();
  hideDropdownsExcept(dropdownID);
  const willShow = document.getElementById(dropdownID).classList.toggle('show');
  if (willShow) {
    pauseResumeManager.pause(pikaVolley, PauseResumePrecedence.dropdown);
  } else {
    pauseResumeManager.resume(pikaVolley, PauseResumePrecedence.dropdown);
  }
}

/**
 * Show the submenu
 * @param {string} submenuBtnID html element id of the submenu button whose submenu to show
 * @param {string} subMenuID html element id of the submenu to show
 */
function showSubmenu(submenuBtnID, subMenuID) {
  hideSubmenus();
  document.getElementById(submenuBtnID).classList.add('open');
  document.getElementById(subMenuID).classList.add('show');
}

/**
 * Hide all other dropdowns except the dropdown
 * @param {string} dropdownID html element id of the dropdown
 */
function hideDropdownsExcept(dropdownID) {
  const dropdowns = document.getElementsByClassName('dropdown');
  for (let i = 0; i < dropdowns.length; i++) {
    if (dropdowns[i].id !== dropdownID) {
      dropdowns[i].classList.remove('show');
    }
  }
}

/**
 * Hide all submenus
 */
function hideSubmenus() {
  const submenus = document.getElementsByClassName('submenu');
  for (let i = 0; i < submenus.length; i++) {
    submenus[i].classList.remove('show');
  }
  const submenuBtns = document.getElementsByClassName('submenu-btn');
  for (let i = 0; i < submenuBtns.length; i++) {
    submenuBtns[i].classList.remove('open');
  }
}