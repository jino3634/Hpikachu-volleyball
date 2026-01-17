/**
 * The Controller part in MVC pattern
 */
'use strict';
import { GROUND_HALF_WIDTH, PikaPhysics } from './physics.js';
import { MenuView, GameView, FadeInOut, IntroView } from './view.js';
import { PikaKeyboard } from './keyboard.js';
import { PikaAudio } from './audio.js';
import { TouchTracker } from './ai/touch_tracker.js';

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
    
    /** @type {{ decisions: number, forcedIdle: number, powerHitRequested: number, powerHitApplied: number }} */
    debugStats;
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
    this.touchTracker = new TouchTracker();
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
    this._heldActionP1 = 0;
    this._heldActionP2 = 0;

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

  _applyActionToKeyboard(kb, actionId, phaseInDecisionInterval, playerIndex) {
    let x = 0, y = 0, p = 0;

    switch (actionId) {
      case 0: break;                 // IDLE
      case 1: x = -1; break;         // LEFT
      case 2: x = 1; break;          // RIGHT
      case 3: y = -1; break;         // JUMP
      case 4: x = -1; y = -1; break; // JUMP_LEFT
      case 5: x = 1; y = -1; break;  // JUMP_RIGHT

      // POWER 계열 (powerHit는 1프레임 트리거)
      case 6: p = 1; break;                  // POWER_NEUTRAL
      case 7: x = -1; p = 1; break;          // POWER_LEFT
      case 8: x = 1; p = 1; break;           // POWER_RIGHT
      case 9: y = 1; p = 1; break;           // POWER_DOWN (공중에서만 의미)
      default: break;
    }

    // ✅ powerHit는 decision 구간의 "첫 프레임(phase=0)"에만 발생시키기
    if (p === 1 && phaseInDecisionInterval !== 0) {
      p = 0;
    }

    // ✅ POWER_DOWN은 공중에서만 의미: 지상이면 DOWN 제거
    if (actionId === 9) {
      const player = this.physics[`player${playerIndex}`];
      const isAir = player.y < 244; // physics.js의 PLAYER_TOUCHING_GROUND_Y_COORD = 244
      if (!isAir) y = 0;
    }

    // ✅ powerHitRequested/powerHitApplied: "실제로 override에 powerHit=1이 주입되는 순간"만 카운트
    if (p === 1) {
      const selfAny = /** @type {any} */ (this);
      const ds = (selfAny.debugStats ??= {
        decisions: 0,
        forcedIdle: 0,

        powerHitRequested: 0,
        powerHitApplied: 0,

        // ✅ P0-2: ground truth 계측
        powerHitContact: 0,   // 최근 powerHit 적용 직후 실제 충돌 발생
        powerHitSuccess: 0,   // 그 충돌이 state=2라서 ball.isPowerHit=true였음

        // ✅ “최근 powerHit 적용” 타이밍 버퍼(TTL, 프레임 단위)
        _phTTL1: 0,
        _phTTL2: 0,
      });

      ds.powerHitRequested = (ds.powerHitRequested | 0) + 1;
      ds.powerHitApplied = (ds.powerHitApplied | 0) + 1;

      // ✅ 최근 적용 버퍼: 이번 프레임 포함해서 2프레임 정도 유효
      if (playerIndex === 1) ds._phTTL1 = 2;
      else ds._phTTL2 = 2;
    }



    kb.setOverrideInput(x, y, p);
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
        if (this.externalEnabledP1 && this.agent1 && typeof this.agent1.chooseAction === 'function') {
          const obs1 = this.getObservation(1);
          this._heldActionP1 = (this.agent1.chooseAction(obs1, 1, this) | 0);
        }
        if (this.externalEnabledP2 && this.agent2 && typeof this.agent2.chooseAction === 'function') {
          const obs2 = this.getObservation(2);
          this._heldActionP2 = (this.agent2.chooseAction(obs2, 2, this) | 0);
        }
      }

      // ✅ externalEnabled인 쪽만 override 주입
      if (this.externalEnabledP1) {
        this._applyActionToKeyboard(this.keyboardArray[0], this._heldActionP1, phase, 1);
      } else {
        // builtin/사람 입력이 쓰도록 override만 제거
        this.keyboardArray[0].clearOverrideInput();
      }

      if (this.externalEnabledP2) {
        this._applyActionToKeyboard(this.keyboardArray[1], this._heldActionP2, phase, 2);
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

    // ✅ after-physics hook (warmup/diagnostics)
    const gameAny = /** @type {any} */ (this);
    if (typeof gameAny.onAfterPhysicsFrame === 'function') {
      // 현재 state 함수로부터 stateName 결정
      let stateName = 'other';
      if (this.state === this.round) stateName = 'round';
      else if (this.state === this.afterEndOfRound) stateName = 'afterEndOfRound';
      else if (this.state === this.beforeStartOfNextRound) stateName = 'beforeStartOfNextRound';
      else if (this.state === this.startOfNewGame) stateName = 'startOfNewGame';
      else if (this.state === this.menu) stateName = 'menu';
      else if (this.state === this.intro) stateName = 'intro';

      // round-like일 때만 관측/라벨 제공 (다른 상태는 null)
      const roundLikeNow = this._isRoundLikeState();

      // builtin / external 모두에서 "이번 프레임에 실제로 적용된 입력" 라벨
      const kb1 = this.keyboardArray[0];
      const kb2 = this.keyboardArray[1];

      gameAny.onAfterPhysicsFrame({
        stateName,
        // 2단계에서 쓰려고 같이 넘김 (지금은 값이 0일 수도 있음)
        decisionPhase: this._decisionPhase | 0,
        decisionInterval: this.decisionInterval | 0,

        obsP1: roundLikeNow ? this.getObservation(1) : null,
        inputP1: roundLikeNow ? {
          xDirection: kb1.xDirection | 0,
          yDirection: kb1.yDirection | 0,
          powerHit: kb1.powerHit | 0,
        } : null,

        // (있어도 해 안 됨: 나중에 확장용)
        obsP2: roundLikeNow ? this.getObservation(2) : null,
        inputP2: roundLikeNow ? {
          xDirection: kb2.xDirection | 0,
          yDirection: kb2.yDirection | 0,
          powerHit: kb2.powerHit | 0,
        } : null,
      });
    }

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

      this.touchTracker.resetPoint();
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

    // ✅ P0-2: powerHit "진짜 성공" ground truth 계측 (충돌 + isPowerHit)
    // 정합 규칙: applied 1회당 contact/success는 최대 1회만 카운트
    {
      const gameAny = /** @type {any} */ (this);
      const ds = gameAny.debugStats;
      if (ds) {
        const ttl1 = ds._phTTL1 | 0;
        const ttl2 = ds._phTTL2 | 0;

        const c1 = !!this.physics.player1.isCollisionWithBallHappened;
        const c2 = !!this.physics.player2.isCollisionWithBallHappened;

        // ball.isPowerHit는 "그 충돌 프레임에서 playerState===2"일 때만 true
        const isPH = !!this.physics.ball.isPowerHit;

        // player1 window
        if (ttl1 > 0) {
          if (c1) {
            ds.powerHitContact = (ds.powerHitContact | 0) + 1;
            if (isPH) ds.powerHitSuccess = (ds.powerHitSuccess | 0) + 1;

            // ✅ 한 번 카운트했으면 이 applied window 종료(중복 contact 방지)
            ds._phTTL1 = 0;
          } else {
            ds._phTTL1 = ttl1 - 1;
          }
        }

        // player2 window
        if (ttl2 > 0) {
          if (c2) {
            ds.powerHitContact = (ds.powerHitContact | 0) + 1;
            if (isPH) ds.powerHitSuccess = (ds.powerHitSuccess | 0) + 1;

            // ✅ 한 번 카운트했으면 이 applied window 종료(중복 contact 방지)
            ds._phTTL2 = 0;
          } else {
            ds._phTTL2 = ttl2 - 1;
          }
        }
      }
    }

    // ✅ 바로 여기
    this.touchTracker.observePhysics(this.physics);
    this.touchTracker.commitFrame();


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

      // ✅ 포인트 시작마다 lastTouch 초기화
      this.touchTracker.resetPoint();
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
    if (window.__PV_TRAINING_MUTE__) return; // ✅ 학습 중 사운드 완전 차단
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

    return {
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
        isPowerHit: b.isPowerHit ? 1 : 0,
      },
      scores: [this.scores[0], this.scores[1]],
      isPlayer2Serve: this.isPlayer2Serve ? 1 : 0,
      roundEnded: this.roundEnded ? 1 : 0,
      gameEnded: this.gameEnded ? 1 : 0,
    };
  }

  getObservationNormalized(playerIndex) {
    // raw(픽셀) 관측
    const raw0 = this.getObservation(playerIndex);

    // physics에서 TTL 계산에 필요한 값 취득
    const b = this.physics.ball;

    // expectedLandingFrames(프레임)를 0..1로 정규화
    // 120프레임 ≈ 2초(60fps 기준). 너무 크면 1로 클램프됨.
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const timeToLand = clamp01(Number(b.expectedLandingFrames ?? 0) / 120);

    // raw에도 TTL/landingX를 넣어 두면 obs.raw 참조하는 코드에서도 편함(픽셀X는 유지)
    const raw = {
      ...raw0,
      ball: {
        ...raw0.ball,
        landingX: Number(raw0.ball.expectedX ?? 0),
        timeToLand,
      },
    };

    const GROUND_WIDTH = GROUND_HALF_WIDTH * 2; // 432
    const GROUND_HEIGHT = 304;

    const nx = (x) => (x / GROUND_WIDTH) * 2 - 1;  // 0..432 -> -1..1
    const ny = (y) => (y / GROUND_HEIGHT) * 2 - 1; // 0..304 -> -1..1

    const clamp = (v, a, b2) => Math.max(a, Math.min(b2, v));
    const nv = (v, scale) => clamp(v / scale, -1, 1);

    return {
      raw,

      me: {
        ...raw.me,
        x: nx(raw.me.x),
        y: ny(raw.me.y),
        yV: nv(raw.me.yV, 20),
      },
      opp: {
        ...raw.opp,
        x: nx(raw.opp.x),
        y: ny(raw.opp.y),
        yV: nv(raw.opp.yV, 20),
      },
      ball: {
        ...raw.ball,
        x: nx(raw.ball.x),
        y: ny(raw.ball.y),
        xV: nv(raw.ball.xV, 25),
        yV: nv(raw.ball.yV, 35),
        expectedX: nx(raw.ball.expectedX),
        landingX: nx(raw.ball.landingX),
        timeToLand, // ✅ 이제 obs.ball.timeToLand가 항상 존재
      },

      scores: raw.scores,
      isPlayer2Serve: raw.isPlayer2Serve,
      roundEnded: raw.roundEnded,
      gameEnded: raw.gameEnded,
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
