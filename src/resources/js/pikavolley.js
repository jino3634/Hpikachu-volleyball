/**
 * The Controller part in MVC pattern
 */
'use strict';
import { GROUND_HALF_WIDTH, PikaPhysics } from './physics.js';
import { MenuView, GameView, FadeInOut, IntroView } from './view.js';
import { PikaKeyboard } from './keyboard.js';
import { PikaAudio } from './audio.js';

/** @typedef {import('@pixi/display').Container} Container */
/** @typedef {import('@pixi/loaders').LoaderResource} LoaderResource */

/** @typedef GameState @type {function():void} */

/**
 * Class representing Pikachu Volleyball game
 */
export class PikachuVolleyball {
  /**
   * Create a Pikachu Volleyball game which includes physics, view, audio
   * @param {Container} stage container which is rendered by PIXI.Renderer or PIXI.CanvasRenderer
   * @param {Object.<string,LoaderResource>} resources resources property of the PIXI.Loader object which is used for loading the game resources
   */
  constructor(stage, resources) {
    this.view = {
      intro: new IntroView(resources),
      menu: new MenuView(resources),
      game: new GameView(resources),
      fadeInOut: new FadeInOut(resources),
    };
    stage.addChild(this.view.intro.container);
    stage.addChild(this.view.menu.container);
    stage.addChild(this.view.game.container);
    stage.addChild(this.view.fadeInOut.black);
    this.view.intro.visible = false;
    this.view.menu.visible = false;
    this.view.game.visible = false;
    this.view.fadeInOut.visible = false;

    this.audio = new PikaAudio(resources);
    this.physics = new PikaPhysics(true, true);
    this.keyboardArray = [
      new PikaKeyboard('KeyD', 'KeyG', 'KeyR', 'KeyV', 'KeyZ', 'KeyF'), // for player1
      new PikaKeyboard(
        'ArrowLeft',
        'ArrowRight',
        'ArrowUp',
        'ArrowDown',
        'Enter'
      ),
    ];

    /** @type {number} game fps */
    this.normalFPS = 25;
    /** @type {number} fps for slow motion */
    this.slowMotionFPS = 5;

    /** @constant @type {number} number of frames for slow motion */
    this.SLOW_MOTION_FRAMES_NUM = 6;
    /** @type {number} number of frames left for slow motion */
    this.slowMotionFramesLeft = 0;
    /** @type {number} number of elapsed normal fps frames for rendering slow motion */
    this.slowMotionNumOfSkippedFrames = 0;

    /** @type {number} 0: with computer, 1: with friend */
    this.selectedWithWho = 0;

    /** @type {number[]} [0] for player 1 score, [1] for player 2 score */
    this.scores = [0, 0];
    /** @type {number} winning score: if either one of the players reaches this score, game ends */
    this.winningScore = 15;

    /** @type {boolean} Is the game ended? */
    this.gameEnded = false;
    /** @type {boolean} Is the round ended? */
    this.roundEnded = false;
    /** @type {boolean} Will player 2 serve? */
    this.isPlayer2Serve = false;

    /** @type {number} frame counter */
    this.frameCounter = 0;

    /**
     * Optional hook for external tools (trainer, recorder, etc.).
     * Called after each physics step if set.
     * @type {null | ((info: any) => void)}
     */
    this.onAfterPhysicsFrame = null;
    /** @type {Object.<string,number>} total number of frames for each game state */
    this.frameTotal = {
      intro: 165,
      afterMenuSelection: 15,
      beforeStartOfNewGame: 15,
      startOfNewGame: 71,
      afterEndOfRound: 5,
      beforeStartOfNextRound: 30,
      gameEnd: 211,
    };

    /** @type {number} counter for frames while there is no input from keyboard */
    this.noInputFrameCounter = 0;
    /** @type {Object.<string,number>} total number of frames to be rendered while there is no input */
    this.noInputFrameTotal = {
      menu: 225,
    };

    /** @type {boolean} true: paused, false: not paused */
    this.paused = false;

    /** @type {boolean} true: stereo, false: mono */
    this.isStereoSound = true;

    /** @type {boolean} true: practice mode on, false: practice mode off */
    this._isPracticeMode = false;

    /**
     * The game state which is being rendered now
     * @type {GameState}
     */
    this.state = this.intro;

    // ------------------------------------------------------------
    // ✅ 컨트롤 관련 (여기부터가 "다음 단계" 핵심)
    // ------------------------------------------------------------

    this.controlMode = 'builtin'; // 'builtin' | 'external'
    this.decisionInterval = 2;
    this._decisionPhase = 0;
    // legacy: actionId(0..9)
    this._heldActionP1 = 0;
    this._heldActionP2 = 0;
    // preferred: input tuple { xDirection, yDirection, powerHit }
    this._heldInputP1 = { xDirection: 0, yDirection: 0, powerHit: 0 };
    this._heldInputP2 = { xDirection: 0, yDirection: 0, powerHit: 0 };

    this._prevExternalRoundLike = false;

    // external agent hooks
    this.agent1 = null;
    this.agent2 = null;

    // ✅ 플레이어별 external on/off (혼합 모드 핵심)
    // - externalEnabled=true면: agent -> keyboard override 적용
    // - false면: override 제거만 하고(=입력 개입 X), 내장AI/사람 입력이 동작
    this.externalEnabledP1 = true;
    this.externalEnabledP2 = true;

    // 학습/관측용: 마지막 step에서 나온 round events 저장
    this._lastRoundEvents = null;

    this.headless = false; // 로직 전용(나중에 렌더 스킵 등에 쓸 수 있음)
  }

  _isRoundLikeState() {
    return (
      this.state === this.round ||
      this.state === this.afterEndOfRound ||
      this.state === this.beforeStartOfNextRound
    );
  }

  _actionIdToInputTuple(actionId) {
    // NOTE: This mapping is kept for backward compatibility only.
    // New agents should output {xDirection, yDirection, powerHit} directly.
    let x = 0, y = 0, p = 0;
    switch (actionId | 0) {
      case 0: break;                 // IDLE
      case 1: x = -1; break;         // LEFT
      case 2: x = 1; break;          // RIGHT
      case 3: y = -1; break;         // JUMP
      case 4: x = -1; y = -1; break; // JUMP_LEFT
      case 5: x = 1; y = -1; break;  // JUMP_RIGHT
      case 6: p = 1; break;                  // POWER_NEUTRAL
      case 7: x = -1; p = 1; break;          // POWER_LEFT
      case 8: x = 1; p = 1; break;           // POWER_RIGHT
      case 9: y = 1; p = 1; break;           // POWER_DOWN (air only)
      default: break;
    }
    return { xDirection: x, yDirection: y, powerHit: p };
  }

  _applyInputToKeyboard(kb, inputTuple, phaseInDecisionInterval, playerIndex) {
    // Normalize
    let x = (inputTuple && typeof inputTuple.xDirection === 'number') ? (inputTuple.xDirection | 0) : 0;
    let y = (inputTuple && typeof inputTuple.yDirection === 'number') ? (inputTuple.yDirection | 0) : 0;
    let p = (inputTuple && typeof inputTuple.powerHit === 'number') ? (inputTuple.powerHit | 0) : 0;

    // powerHit is a 1-frame trigger at the beginning of each decision interval
    if (p === 1 && phaseInDecisionInterval !== 0) {
      p = 0;
    }

    // POWER_DOWN style (y=+1) is only meaningful in air
    if (y === 1) {
      const player = this.physics[`player${playerIndex}`];
      const isAir = player.y < 244; // physics.js: PLAYER_TOUCHING_GROUND_Y_COORD = 244
      if (!isAir) y = 0;
    }

    kb.setOverrideInput(x, y, p);
  }

  // Backward-compatible wrapper
  _applyActionToKeyboard(kb, actionId, phaseInDecisionInterval, playerIndex) {
    const tup = this._actionIdToInputTuple(actionId);
    this._applyInputToKeyboard(kb, tup, phaseInDecisionInterval, playerIndex);
  }

  /**
   * Game loop
   */
  gameLoop() {
    this.stepLogic();
  }

  /**
   * Advance exactly "one logical frame"
   * @returns {boolean} true if processed, false if skipped/paused
   */
  stepLogic() {
    if (this.paused === true) {
      return false;
    }

    // Slow motion frame skipping (keeps original behavior)
    if (this.slowMotionFramesLeft > 0) {
      this.slowMotionNumOfSkippedFrames++;

      const skipMod = Math.round(this.normalFPS / this.slowMotionFPS);
      if (this.slowMotionNumOfSkippedFrames % skipMod !== 0) {
        return false;
      }

      this.slowMotionFramesLeft--;
      this.slowMotionNumOfSkippedFrames = 0;
    }

    const roundLike = this._isRoundLikeState();
    const externalRoundLike = (this.controlMode === 'external' && roundLike);

    // externalRoundLike 모드 진입 시 phase 정렬
    if (externalRoundLike && this._decisionPhase !== 0 && this._prevExternalRoundLike !== true) {
      this._decisionPhase = 0;
    }
    this._prevExternalRoundLike = externalRoundLike;

    if (externalRoundLike) {
      const phase = this._decisionPhase;

      if (phase === 0) {
        // ✅ externalEnabled인 쪽만 chooseAction 수행
        if (this.externalEnabledP1 && this.agent1 && (typeof this.agent1.chooseInput === 'function' || typeof this.agent1.chooseAction === 'function')) {
          const obs1 = this.getObservation(1);
          // Prefer tuple-based agent API when available
          if (typeof this.agent1.chooseInput === 'function') {
            this._heldInputP1 = this.agent1.chooseInput(obs1, 1, this) || { xDirection: 0, yDirection: 0, powerHit: 0 };
          } else {
            this._heldActionP1 = (this.agent1.chooseAction(obs1, 1, this) | 0);
            this._heldInputP1 = this._actionIdToInputTuple(this._heldActionP1);
          }
        }
        if (this.externalEnabledP2 && this.agent2 && (typeof this.agent2.chooseInput === 'function' || typeof this.agent2.chooseAction === 'function')) {
          const obs2 = this.getObservation(2);
          if (typeof this.agent2.chooseInput === 'function') {
            this._heldInputP2 = this.agent2.chooseInput(obs2, 2, this) || { xDirection: 0, yDirection: 0, powerHit: 0 };
          } else {
            this._heldActionP2 = (this.agent2.chooseAction(obs2, 2, this) | 0);
            this._heldInputP2 = this._actionIdToInputTuple(this._heldActionP2);
          }
        }
      }

      // ✅ externalEnabled인 쪽만 override 주입
      if (this.externalEnabledP1) {
        this._applyInputToKeyboard(this.keyboardArray[0], this._heldInputP1, phase, 1);
      } else {
        // builtin/사람 입력이 쓰도록 override만 제거
        this.keyboardArray[0].clearOverrideInput();
      }

      if (this.externalEnabledP2) {
        this._applyInputToKeyboard(this.keyboardArray[1], this._heldInputP2, phase, 2);
      } else {
        this.keyboardArray[1].clearOverrideInput();
      }

      this._decisionPhase = (this._decisionPhase + 1) % this.decisionInterval;
    } else {
      // builtin 모드거나 round가 아니면 override 끔
      this.keyboardArray[0].clearOverrideInput();
      this.keyboardArray[1].clearOverrideInput();
      this._decisionPhase = 0;
    }

    // ✅ 사람 입력/override를 실제 x/y/powerHit로 확정
    // (override가 있으면 keyboard.js에서 override 우선 적용됨)
    this.keyboardArray[0].getInput();
    this.keyboardArray[1].getInput();

    // 상태 실행
    this.state();

    return true;
  }

  /**
   * Intro: a man with a brief case
   */
  intro() {
    if (this.frameCounter === 0) {
      this.view.intro.visible = true;
      this.view.fadeInOut.setBlackAlphaTo(0);
      this.audio.sounds.bgm.stop();
    }
    this.view.intro.drawMark(this.frameCounter);
    this.frameCounter++;

    if (
      this.keyboardArray[0].powerHit === 1 ||
      this.keyboardArray[1].powerHit === 1
    ) {
      this.frameCounter = 0;
      this.view.intro.visible = false;
      this.state = this.menu;
    }

    if (this.frameCounter >= this.frameTotal.intro) {
      this.frameCounter = 0;
      this.view.intro.visible = false;
      this.state = this.menu;
    }
  }

  /**
   * Menu: select who do you want to play. With computer? With friend?
   */
  menu() {
    if (this.frameCounter === 0) {
      this.view.menu.visible = true;
      this.view.fadeInOut.setBlackAlphaTo(0);
      this.selectedWithWho = 0;
      this.view.menu.selectWithWho(this.selectedWithWho);
    }
    this.view.menu.drawFightMessage(this.frameCounter);
    this.view.menu.drawSachisoft(this.frameCounter);
    this.view.menu.drawSittingPikachuTiles(this.frameCounter);
    this.view.menu.drawPikachuVolleyballMessage(this.frameCounter);
    this.view.menu.drawPokemonMessage(this.frameCounter);
    this.view.menu.drawWithWhoMessages(this.frameCounter);
    this.frameCounter++;

    if (
      this.frameCounter < 71 &&
      (this.keyboardArray[0].powerHit === 1 ||
        this.keyboardArray[1].powerHit === 1)
    ) {
      this.frameCounter = 71;
      return;
    }

    if (this.frameCounter <= 71) {
      return;
    }

    if (
      (this.keyboardArray[0].yDirection === -1 ||
        this.keyboardArray[1].yDirection === -1) &&
      this.selectedWithWho === 1
    ) {
      this.noInputFrameCounter = 0;
      this.selectedWithWho = 0;
      this.view.menu.selectWithWho(this.selectedWithWho);
      this.audio.sounds.pi.play();
    } else if (
      (this.keyboardArray[0].yDirection === 1 ||
        this.keyboardArray[1].yDirection === 1) &&
      this.selectedWithWho === 0
    ) {
      this.noInputFrameCounter = 0;
      this.selectedWithWho = 1;
      this.view.menu.selectWithWho(this.selectedWithWho);
      this.audio.sounds.pi.play();
    } else {
      this.noInputFrameCounter++;
    }

    if (
      this.keyboardArray[0].powerHit === 1 ||
      this.keyboardArray[1].powerHit === 1
    ) {
      if (this.selectedWithWho === 1) {
        this.physics.player1.isComputer = false;
        this.physics.player2.isComputer = false;
      } else {
        if (this.keyboardArray[0].powerHit === 1) {
          this.physics.player1.isComputer = false;
          this.physics.player2.isComputer = true;
        } else if (this.keyboardArray[1].powerHit === 1) {
          this.physics.player1.isComputer = true;
          this.physics.player2.isComputer = false;
        }
      }
      this.audio.sounds.pikachu.play();
      this.frameCounter = 0;
      this.noInputFrameCounter = 0;
      this.state = this.afterMenuSelection;
      return;
    }

    if (this.noInputFrameCounter >= this.noInputFrameTotal.menu) {
      this.physics.player1.isComputer = true;
      this.physics.player2.isComputer = true;
      this.frameCounter = 0;
      this.noInputFrameCounter = 0;
      this.state = this.afterMenuSelection;
    }
  }

  /**
   * Fade out after menu selection
   */
  afterMenuSelection() {
    this.view.fadeInOut.changeBlackAlphaBy(1 / 16);
    this.frameCounter++;
    if (this.frameCounter >= this.frameTotal.afterMenuSelection) {
      this.frameCounter = 0;
      this.state = this.beforeStartOfNewGame;
    }
  }

  /**
   * Delay before start of new game
   */
  beforeStartOfNewGame() {
    this.frameCounter++;
    if (this.frameCounter >= this.frameTotal.beforeStartOfNewGame) {
      this.frameCounter = 0;
      this.view.menu.visible = false;
      this.state = this.startOfNewGame;
    }
  }

  /**
   * Start of new game
   */
  startOfNewGame() {
    const events = this.startOfNewGameLogic();
    this.startOfNewGameView(events);
  }

  startOfNewGameLogic() {
    const isFirstFrame = this.frameCounter === 0;

    if (isFirstFrame) {
      // Flags
      this.gameEnded = false;
      this.roundEnded = false;
      this.isPlayer2Serve = false;

      // Physics flags
      this.physics.player1.gameEnded = false;
      this.physics.player1.isWinner = false;
      this.physics.player2.gameEnded = false;
      this.physics.player2.isWinner = false;

      // Scores
      this.scores[0] = 0;
      this.scores[1] = 0;

      // Init physics state
      this.physics.player1.initializeForNewRound();
      this.physics.player2.initializeForNewRound();
      this.physics.ball.initializeForNewRound(this.isPlayer2Serve);
    }

    this.frameCounter++;

    const viewFrame = this.frameCounter;
    const shouldTransitionToRound =
      this.frameCounter >= this.frameTotal.startOfNewGame;

    if (shouldTransitionToRound) {
      this.frameCounter = 0;
      this.state = this.round;
    }

    return {
      isFirstFrame,
      shouldTransitionToRound,
      viewFrame,
    };
  }

  startOfNewGameView(events) {
    if (events.isFirstFrame) {
      this.view.game.visible = true;
      this.view.game.drawScoresToScoreBoards(this.scores);
      this.view.game.drawPlayersAndBall(this.physics);
      this.view.fadeInOut.setBlackAlphaTo(1);
      this.audio.sounds.bgm.play();
    }

    this.view.game.drawGameStartMessage(
      events.viewFrame,
      this.frameTotal.startOfNewGame
    );

    this.view.game.drawCloudsAndWave();
    this.view.fadeInOut.changeBlackAlphaBy(-(1 / 17));

    if (events.shouldTransitionToRound) {
      this.view.fadeInOut.setBlackAlphaTo(0);
    }
  }

  /**
   * Round
   */
  round() {
    const events = this.roundLogic();
    this._lastRoundEvents = events;
    this.roundView(events);
  }

  roundLogic() {
    const pressedPowerHit =
      this.keyboardArray[0].powerHit === 1 ||
      this.keyboardArray[1].powerHit === 1;

    if (
      this.physics.player1.isComputer === true &&
      this.physics.player2.isComputer === true &&
      pressedPowerHit
    ) {
      this.frameCounter = 0;
      this.state = this.intro;
      return {
        pressedPowerHit,
        isBallTouchingGround: false,
        exitToIntro: true,
        becameGameEnded: false,
        gameEnded: this.gameEnded,
        roundEnded: this.roundEnded,
        scored: 0,
        scorerPunchX: null,
        shouldDrawGameEndMessage: false,
        shouldFadeOutToAfterEndOfRound: false,
        shouldHideGameView: true,
      };
    }

    const isBallTouchingGround = this.physics.runEngineForNextFrame(
      this.keyboardArray
    );

    // Optional frame hook: used for imitation dataset collection.
    // NOTE: keyboardArray may be mutated by physics (builtin AI path).
    if (typeof this.onAfterPhysicsFrame === 'function') {
      try {
        this.onAfterPhysicsFrame({
          frame: this.totalFrame ?? this.frameCounter ?? 0,
          scores: [this.scores?.[0] ?? 0, this.scores?.[1] ?? 0],
          inputP1: {
            xDirection: this.keyboardArray?.[0]?.xDirection ?? 0,
            yDirection: this.keyboardArray?.[0]?.yDirection ?? 0,
            powerHit: this.keyboardArray?.[0]?.powerHit ?? 0,
          },
          inputP2: {
            xDirection: this.keyboardArray?.[1]?.xDirection ?? 0,
            yDirection: this.keyboardArray?.[1]?.yDirection ?? 0,
            powerHit: this.keyboardArray?.[1]?.powerHit ?? 0,
          },
          obsP1: this.getObservation ? this.getObservation(1) : null,
          obsP2: this.getObservation ? this.getObservation(2) : null,
          stateName: (this.state === this.round) ? 'round' : 'non_round',
        });
      } catch (e) {
        // ignore hook errors
      }
    }

    if (this.gameEnded === true) {
      this.frameCounter++;
      if (
        this.frameCounter >= this.frameTotal.gameEnd ||
        (this.frameCounter >= 70 && pressedPowerHit)
      ) {
        this.frameCounter = 0;
        this.state = this.intro;
        return {
          pressedPowerHit,
          isBallTouchingGround,
          exitToIntro: true,
          becameGameEnded: false,
          gameEnded: true,
          roundEnded: this.roundEnded,
          scored: 0,
          scorerPunchX: null,
          shouldDrawGameEndMessage: true,
          shouldFadeOutToAfterEndOfRound: false,
          shouldHideGameView: true,
        };
      }

      return {
        pressedPowerHit,
        isBallTouchingGround,
        exitToIntro: false,
        becameGameEnded: false,
        gameEnded: true,
        roundEnded: this.roundEnded,
        scored: 0,
        scorerPunchX: null,
        shouldDrawGameEndMessage: true,
        shouldFadeOutToAfterEndOfRound: false,
        shouldHideGameView: false,
      };
    }

    /** @type {0|1|2} */
    let scored = 0;
    /** @type {boolean} */
    let becameGameEnded = false;

    if (
      isBallTouchingGround &&
      this._isPracticeMode === false &&
      this.roundEnded === false &&
      this.gameEnded === false
    ) {
      const punchX = this.physics.ball.punchEffectX;

      if (punchX < GROUND_HALF_WIDTH) {
        this.isPlayer2Serve = true;
        this.scores[1] += 1;
        scored = 2;

        if (this.scores[1] >= this.winningScore) {
          this.gameEnded = true;
          becameGameEnded = true;
          this.physics.player1.isWinner = false;
          this.physics.player2.isWinner = true;
          this.physics.player1.gameEnded = true;
          this.physics.player2.gameEnded = true;
        }
      } else {
        this.isPlayer2Serve = false;
        this.scores[0] += 1;
        scored = 1;

        if (this.scores[0] >= this.winningScore) {
          this.gameEnded = true;
          becameGameEnded = true;
          this.physics.player1.isWinner = true;
          this.physics.player2.isWinner = false;
          this.physics.player1.gameEnded = true;
          this.physics.player2.gameEnded = true;
        }
      }

      if (this.roundEnded === false && this.gameEnded === false) {
        this.slowMotionFramesLeft = this.SLOW_MOTION_FRAMES_NUM;
      }

      this.roundEnded = true;

      return {
        pressedPowerHit,
        isBallTouchingGround,
        exitToIntro: false,
        becameGameEnded,
        gameEnded: this.gameEnded,
        roundEnded: this.roundEnded,
        scored,
        scorerPunchX: this.physics.ball.punchEffectX,
        shouldDrawGameEndMessage: false,
        shouldFadeOutToAfterEndOfRound: false,
        shouldHideGameView: false,
      };
    }

    if (this.roundEnded && !this.gameEnded && this.slowMotionFramesLeft === 0) {
      this.frameCounter = 0;
      this.state = this.afterEndOfRound;
      return {
        pressedPowerHit,
        isBallTouchingGround,
        exitToIntro: false,
        becameGameEnded: false,
        gameEnded: false,
        roundEnded: true,
        scored: 0,
        scorerPunchX: null,
        shouldDrawGameEndMessage: false,
        shouldFadeOutToAfterEndOfRound: true,
        shouldHideGameView: false,
      };
    }

    this.frameCounter++;

    return {
      pressedPowerHit,
      isBallTouchingGround,
      exitToIntro: false,
      becameGameEnded: false,
      gameEnded: this.gameEnded,
      roundEnded: this.roundEnded,
      scored: 0,
      scorerPunchX: null,
      shouldDrawGameEndMessage: false,
      shouldFadeOutToAfterEndOfRound: false,
      shouldHideGameView: false,
    };
  }

  roundView(events) {
    this.playSoundEffect();
    this.view.game.drawPlayersAndBall(this.physics);
    this.view.game.drawCloudsAndWave();

    if (events.shouldDrawGameEndMessage === true) {
      this.view.game.drawGameEndMessage(this.frameCounter);
    }

    if (events.scored !== 0) {
      this.view.game.drawScoresToScoreBoards(this.scores);
    }

    if (events.shouldFadeOutToAfterEndOfRound === true) {
      this.view.fadeInOut.changeBlackAlphaBy(1 / 16);
    }

    if (events.shouldHideGameView === true) {
      this.view.game.visible = false;
    }
  }

  afterEndOfRound() {
    this.view.fadeInOut.changeBlackAlphaBy(1 / 16);
    this.frameCounter++;
    if (this.frameCounter >= this.frameTotal.afterEndOfRound) {
      this.frameCounter = 0;
      this.state = this.beforeStartOfNextRound;
    }
  }

  beforeStartOfNextRound() {
    const events = this.beforeStartOfNextRoundLogic();
    this.beforeStartOfNextRoundView(events);
  }

  beforeStartOfNextRoundLogic() {
    const isFirstFrame = this.frameCounter === 0;

    if (isFirstFrame) {
      this.physics.player1.initializeForNewRound();
      this.physics.player2.initializeForNewRound();
      this.physics.ball.initializeForNewRound(this.isPlayer2Serve);
    }

    this.frameCounter++;

    const shouldToggleReady = this.frameCounter % 5 === 0;

    const shouldTransitionToRound =
      this.frameCounter >= this.frameTotal.beforeStartOfNextRound;

    if (shouldTransitionToRound) {
      this.frameCounter = 0;
      this.roundEnded = false;
      this.state = this.round;
    }

    return {
      isFirstFrame,
      shouldToggleReady,
      shouldTransitionToRound,
    };
  }

  beforeStartOfNextRoundView(events) {
    if (events.isFirstFrame) {
      this.view.fadeInOut.setBlackAlphaTo(1);
      this.view.game.drawReadyMessage(false);
      this.view.game.drawPlayersAndBall(this.physics);
    }

    this.view.game.drawCloudsAndWave();
    this.view.fadeInOut.changeBlackAlphaBy(-(1 / 16));

    if (events.shouldToggleReady) {
      this.view.game.toggleReadyMessage();
    }

    if (events.shouldTransitionToRound) {
      this.view.game.drawReadyMessage(false);
      this.view.fadeInOut.setBlackAlphaTo(0);
    }
  }

  playSoundEffect() {
    const audio = this.audio;
    for (let i = 0; i < 2; i++) {
      const player = this.physics[`player${i + 1}`];
      const sound = player.sound;
      let leftOrCenterOrRight = 0;
      if (this.isStereoSound) {
        leftOrCenterOrRight = i === 0 ? -1 : 1;
      }
      if (sound.pipikachu === true) {
        audio.sounds.pipikachu.play(leftOrCenterOrRight);
        sound.pipikachu = false;
      }
      if (sound.pika === true) {
        audio.sounds.pika.play(leftOrCenterOrRight);
        sound.pika = false;
      }
      if (sound.chu === true) {
        audio.sounds.chu.play(leftOrCenterOrRight);
        sound.chu = false;
      }
    }
    const ball = this.physics.ball;
    const sound = ball.sound;
    let leftOrCenterOrRight = 0;
    if (this.isStereoSound) {
      if (ball.punchEffectX < GROUND_HALF_WIDTH) {
        leftOrCenterOrRight = -1;
      } else if (ball.punchEffectX > GROUND_HALF_WIDTH) {
        leftOrCenterOrRight = 1;
      }
    }
    if (sound.powerHit === true) {
      this.audio.sounds.powerHit.play(leftOrCenterOrRight);
      sound.powerHit = false;
    }
    if (sound.ballTouchesGround === true) {
      this.audio.sounds.ballTouchesGround.play(leftOrCenterOrRight);
      sound.ballTouchesGround = false;
    }
  }

  // ------------------------------------------------------------
  // ✅ 외부/내장 혼합 제어용 API (여기가 다음 단계 핵심)
  // ------------------------------------------------------------

  /**
   * external/builtin 모드 전환
   * @param {'builtin'|'external'} mode
   */
  setControlMode(mode) {
    this.controlMode = mode;
  }

  /**
   * 플레이어별 external on/off
   * @param {boolean} p1Enabled
   * @param {boolean} p2Enabled
   */
  setExternalEnabled(p1Enabled, p2Enabled) {
    this.externalEnabledP1 = !!p1Enabled;
    this.externalEnabledP2 = !!p2Enabled;

    // 꺼진 쪽은 override 즉시 제거
    if (!this.externalEnabledP1) this.keyboardArray[0].clearOverrideInput();
    if (!this.externalEnabledP2) this.keyboardArray[1].clearOverrideInput();
  }

  /**
   * 내장 AI(physics의 isComputer)와 externalEnabled를 같이 맞추는 헬퍼
   * - 예: P1 external vs P2 builtin
   */
  setExternalVsBuiltin(p1External, p2External) {
    this.setControlMode('external');
    this.setExternalEnabled(!!p1External, !!p2External);

    // external이면 내장AI는 확실히 끔
    this.physics.player1.isComputer = p1External ? false : true;
    this.physics.player2.isComputer = p2External ? false : true;
  }


  /**
   * 외부에서 actionId를 직접 주입(에이전트 없이도 가능)
   */
  setExternalActions(p1ActionId, p2ActionId) {
    this._heldActionP1 = (p1ActionId | 0);
    this._heldActionP2 = (p2ActionId | 0);
    this._heldInputP1 = this._actionIdToInputTuple(this._heldActionP1);
    this._heldInputP2 = this._actionIdToInputTuple(this._heldActionP2);
  }

  /**
   * 외부에서 입력 튜플을 직접 주입(추천)
   * @param {{xDirection:number,yDirection:number,powerHit:number}} p1Input
   * @param {{xDirection:number,yDirection:number,powerHit:number}} p2Input
   */
  setExternalInputs(p1Input, p2Input) {
    this._heldInputP1 = p1Input || { xDirection: 0, yDirection: 0, powerHit: 0 };
    this._heldInputP2 = p2Input || { xDirection: 0, yDirection: 0, powerHit: 0 };
  }

  /**
   * 외부 에이전트 연결
   * agent는 chooseAction(obs, playerIndex, game) 메서드만 있으면 됨.
   */
  setAgents(agent1, agent2) {
    this.agent1 = agent1;
    this.agent2 = agent2;
  }

  /**
   * 학습용 관측(최소)
   */
  getObservation(playerIndex) {
    const me = this.physics[`player${playerIndex}`];
    const opp = this.physics[`player${playerIndex === 1 ? 2 : 1}`];
    const b = this.physics.ball;

    // --- Observation post-processing (player-centric, learning-friendly) ---
    // Keep raw snapshot for debugging/compat.
    const raw = {
      me: {
        x: me.x, y: me.y,
        yV: me.yVelocity,
        state: me.state,
        divingDir: me.divingDirection,
        lying: me.lyingDownDurationLeft,
        isP2: me.isPlayer2 ? 1 : 0,
        bold: me.computerBoldness,
      },
      opp: {
        x: opp.x, y: opp.y,
        yV: opp.yVelocity,
        state: opp.state,
        divingDir: opp.divingDirection,
        lying: opp.lyingDownDurationLeft,
        isP2: opp.isPlayer2 ? 1 : 0,
      },
      ball: {
        x: b.x, y: b.y,
        xV: b.xVelocity, yV: b.yVelocity,
        expectedX: b.expectedLandingPointX,
        
        timeToLand: b.expectedLandingFrames,
isPowerHit: b.isPowerHit ? 1 : 0,
      },
    };

    // Court constants (from physics.js): GROUND_WIDTH=432, BALL_TOUCHING_GROUND_Y_COORD=252
    const W = 432;
    const H = 252;

    const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));
    const norm01 = (v, max) => (max <= 0 ? 0 : (v / max));
    const norm11 = (v, max) => (norm01(v, max) * 2 - 1);

    // Flip to make the observation always "me is on the left".
    const needFlip = Boolean(raw.me.isP2);
    const flipX = (x) => (W - x);
    const xTo = (x) => (needFlip ? flipX(x) : x);
    const xVTo = (xv) => (needFlip ? -xv : xv);

    // Normalize positions to [-1, 1]
    const nx = (x) => norm11(clamp(x, 0, W), W);
    const ny = (y) => norm11(clamp(y, 0, H), H);

    // Velocity clipping (empirical-safe ranges based on physics.js assignments)
    // - ball.xV can spike from hits, keep a wide clip
    // - y velocities are generally smaller
    const clipBallXV = 160;
    const clipBallYV = 60;
    const clipPlayerYV = 25;
    const nv = (v, clip) => (clip <= 0 ? 0 : (clamp(v, -clip, clip) / clip));

    const meIsLying = raw.me.state === 4 ? 1 : 0;
    const meIsDiving = raw.me.state === 3 ? 1 : 0;
    const meIsAir = (raw.me.y < 244 || raw.me.state === 1 || raw.me.state === 2 || raw.me.state === 3) ? 1 : 0;
    const meCanAct = (raw.me.state <= 3) ? 1 : 0;

    const oppIsLying = raw.opp.state === 4 ? 1 : 0;
    const oppIsAir = (raw.opp.y < 244 || raw.opp.state === 1 || raw.opp.state === 2 || raw.opp.state === 3) ? 1 : 0;

    // Build processed obs while keeping original field names.
    const meP = {
      x: nx(xTo(raw.me.x)),
      y: ny(raw.me.y),
      yV: nv(raw.me.yV, clipPlayerYV),
      state: raw.me.state,
      divingDir: raw.me.divingDir,
      lying: raw.me.lying,
      isP2: 0, // after flipping, "me" is always treated as left-side
      bold: raw.me.bold,
      isAir: meIsAir,
      isDiving: meIsDiving,
      isLying: meIsLying,
      canAct: meCanAct,
    };

    const oppP = {
      x: nx(xTo(raw.opp.x)),
      y: ny(raw.opp.y),
      yV: nv(raw.opp.yV, clipPlayerYV),
      state: raw.opp.state,
      divingDir: raw.opp.divingDir,
      lying: raw.opp.lying,
      isP2: 1, // opponent is always right-side in this player-centric view
      isAir: oppIsAir,
      isLying: oppIsLying,
    };

    const ballP = {
      x: nx(xTo(raw.ball.x)),
      y: ny(raw.ball.y),
      xV: nv(xVTo(raw.ball.xV), clipBallXV),
      yV: nv(raw.ball.yV, clipBallYV),
      expectedX: nx(xTo(raw.ball.expectedX)),
      
      landingX: nx(xTo(raw.ball.expectedX)),
      timeToLand: Math.max(0, Math.min(1, (raw.ball.timeToLand ?? 0) / 180)),
isPowerHit: raw.ball.isPowerHit,
    };

    return {
      me: meP,
      opp: oppP,
      ball: ballP,
      // keep high-level flags as-is (do NOT flip scores/serve flags here)
      scores: [this.scores[0], this.scores[1]],
      isPlayer2Serve: this.isPlayer2Serve ? 1 : 0,
      roundEnded: this.roundEnded ? 1 : 0,
      gameEnded: this.gameEnded ? 1 : 0,
      // raw snapshot for debugging/backward-compat usage
      raw,
    };
  }

  restart() {
    this.frameCounter = 0;
    this.noInputFrameCounter = 0;
    this.slowMotionFramesLeft = 0;
    this.slowMotionNumOfSkippedFrames = 0;
    this.view.menu.visible = false;
    this.view.game.visible = false;
    this.state = this.intro;
  }

  get isPracticeMode() {
    return this._isPracticeMode;
  }

  set isPracticeMode(bool) {
    this._isPracticeMode = bool;
    this.view.game.scoreBoards[0].visible = !bool;
    this.view.game.scoreBoards[1].visible = !bool;
  }
}
