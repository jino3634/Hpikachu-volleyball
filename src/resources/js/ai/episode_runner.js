// src/resources/js/ai/episode_runner.js
//
// 목적:
// - 학습 시작 시 intro/menu를 거치지 않고 "바로 round"로 진입 (powerHit 자동입력 없음)
// - 학습/trace는 오직 round에서만 쌓는다
// - 포인트가 끝나면 엔진이 자연스럽게 nextRound로 넘어가고, 세트가 끝나면 startOfNewGame -> round로 다시 진입
// - hardSafety는 "학습 프레임(round 프레임)" 기준으로만 카운트

import { EpisodeBuilder } from './rl_episode_builder.js';

// 프로젝트에 따라 존재할 수 있어: 없으면 이 파일 내 trace 배열로만 유지
// import { TraceBuffer } from '../replay/TraceBuffer.js';

export class OnePointEpisodeRunner {
/**
 * @param {any} game
 * @param {{
 *   learningPlayer?: 1|2,
 *   maxTraceFrames?: number,
 *   hardSafetyFrames?: number
 * }} [opts]
 */
  constructor(game, opts = {}) {
    this.game = game;
    this.learningPlayer = (opts.learningPlayer ?? 1);

    this.maxTraceFrames = opts.maxTraceFrames ?? 2400;
    this.hardSafetyFrames = opts.hardSafetyFrames ?? 36000;
  }

  // ─────────────────────────────────────────────────────────────
  // 상태/입력 헬퍼
  // ─────────────────────────────────────────────────────────────

  _setControlModeExternal() {
    if (typeof this.game.setControlMode === 'function') this.game.setControlMode('external');
    else this.game.controlMode = 'external';
  }

  _setNeutralInput() {
    // Prefer tuple input API if available
    if (typeof this.game.setExternalInputs === 'function') {
      this.game.setExternalInputs(
        { xDirection: 0, yDirection: 0, powerHit: 0 },
        { xDirection: 0, yDirection: 0, powerHit: 0 },
      );
      return;
    }
    if (typeof this.game.setExternalActions === 'function') {
      this.game.setExternalActions(0, 0);
      return;
    }
    this.game._heldActionP1 = 0;
    this.game._heldActionP2 = 0;
  }

  _applyInputs(p1Input, p2Input) {
    // Prefer tuple input API if available
    if (typeof this.game.setExternalInputs === 'function') {
      this.game.setExternalInputs(p1Input, p2Input);
      return;
    }
    // Fallback to actionId injection if tuple API is missing
    // (Older versions may only support actionId.)
    if (typeof this.game.setExternalActions === 'function') {
      const p1Action = p1Input?.actionId ?? 0;
      const p2Action = p2Input?.actionId ?? 0;
      this.game.setExternalActions(p1Action, p2Action);
      return;
    }
    this.game._heldActionP1 = p1Input?.actionId ?? 0;
    this.game._heldActionP2 = p2Input?.actionId ?? 0;
  }

  _getStateName() {
    try {
      const s = this.game.state;
      if (s === this.game.round) return 'round';
      if (s === this.game.intro) return 'intro';
      if (s === this.game.menu) return 'menu';
      if (s === this.game.beforeStartOfNewGame) return 'beforeStartOfNewGame';
      if (s === this.game.startOfNewGame) return 'startOfNewGame';
      if (s === this.game.beforeStartOfNextRound) return 'beforeStartOfNextRound';
      if (s === this.game.afterEndOfRound) return 'afterEndOfRound';
      return 'unknown';
    } catch {
      return 'unknown';
    }
  }

  // ─────────────────────────────────────────────────────────────
  // 핵심: 메뉴 입력 없이 "바로 round 진입"
  // ─────────────────────────────────────────────────────────────

  _forceEnterRoundNoMenu() {
    // 학습은 외부제어로 고정
    this._setControlModeExternal();
    this.game.paused = false;

    // 메뉴에서 하던 CPU/HUMAN 설정을 직접 반영
    // learningPlayer = 외부 제어(사람 취급), 상대 = CPU
    // (프로퍼티 명이 다르면 여기만 조정)
    if (this.game.player1 && this.game.player2) {
      if (this.learningPlayer === 1) {
        this.game.player1.isComputer = false;
        this.game.player2.isComputer = true;
      } else {
        this.game.player1.isComputer = true;
        this.game.player2.isComputer = false;
      }
    }

    // intro/menu를 건너뛰고 새 게임 시작 단계로 점프
    if (this.game.beforeStartOfNewGame) {
      this.game.state = this.game.beforeStartOfNewGame;
    } else if (this.game.startOfNewGame) {
      this.game.state = this.game.startOfNewGame;
    }
    this.game.frameCounter = 0;

    // powerHit 등 어떤 입력도 주지 않고 상태만 진행시켜 round까지 진입
    for (let i = 0; i < 3000; i++) {
      this._setNeutralInput();
      this.game.stepLogic();
      if (this.game.state === this.game.round) return true;
    }

    console.warn('[EpisodeRunner] _forceEnterRoundNoMenu failed', {
      state: this._getStateName(),
      frameCounter: this.game.frameCounter,
    });
    return false;
  }

  // ─────────────────────────────────────────────────────────────
  // 관측/리플레이(trace) 구성 (프로젝트 포맷에 맞게 필요하면 수정)
  // ─────────────────────────────────────────────────────────────

  _captureFrame() {
    // 너 프로젝트에서 쓰는 포맷에 맞춰 최대한 무난하게 구성
    // (정확한 필드명은 네 trace 저장 로직과 맞춰야 함)
    const p1 = this.game.player1;
    const p2 = this.game.player2;
    const b = this.game.ball;

    return {
      p1: p1 ? { x: p1.x, y: p1.y } : null,
      p2: p2 ? { x: p2.x, y: p2.y } : null,
      ball: b ? { x: b.x, y: b.y } : null,
      // 있으면 점수도 저장(없으면 overlay에서 ?로 표시됨)
      scores: (this.game.score && typeof this.game.score === 'object')
        ? { p1: this.game.score.p1, p2: this.game.score.p2 }
        : null,
    };
  }

  _isPointEnded(ev) {
    // 너가 stepLogic 분리하면서 roundLogic이 반환하는 이벤트 객체를 쓰는 구조라면
    // ev.scoredBy / ev.loser / ev.loseReason 같은 걸 여기서 본다
    return !!(ev && (ev.scoredBy || ev.loser || ev.loseReason));
  }

  // ─────────────────────────────────────────────────────────────
  // Public API: 포인트 1개 수행
  // ─────────────────────────────────────────────────────────────

/**
 * Run a single point episode.
 * @returns {{
 *   ok: boolean,
 *   scoredBy: number,
 *   loser: number,
 *   loseReason: string,
 *   frames: number,
 *   trace: any[],
 *   episode: any
 * }}
 */
  runOnePoint() {
    // 학습 루프가 끊기지 않도록: runOnePoint는 "항상" 구조화된 결과를 반환한다.
    // (예외는 trainer에서 unhandled promise rejection로 이어져 전체 중단될 수 있음)
    const trace = [];
    let frames = 0;

    try {
      // 0) round 진입 강제(메뉴/intro 스킵)
      const ok = this._forceEnterRoundNoMenu();
      if (!ok) {
          return {
            ok: false,
            scoredBy: 0,
            loser: 0,
            loseReason: 'ENTER_ROUND_FAILED',
            frames,
            trace,
            episode: null,
          };
        }

      const builder = new EpisodeBuilder({ learningPlayer: this.learningPlayer });

      // agent는 trainer.init()에서 game.setAgents(...)로 이미 붙어있음
      const agent = (this.learningPlayer === 1) ? this.game.agent1 : this.game.agent2;
      const hasChooseInput = !!(agent && typeof agent.chooseInput === 'function');
      const hasChooseAction = !!(agent && typeof agent.chooseAction === 'function');
      if (!agent || (!hasChooseInput && !hasChooseAction)) {
        return {
          ok: false,
          scoredBy: 0,
          loser: 0,
          loseReason: `MISSING_AGENT_P${this.learningPlayer}`,
          frames,
          trace,
          episode: null,
        };
      }

      // IMPORTANT:
      // In external mode, pikavolley may also call agent.chooseInput/chooseAction at phase=0.
      // To make (obs -> action) pairing deterministic and avoid double-decisions,
      // runner temporarily disables game agents and injects the chosen input itself.
      const prevAgent1 = this.game.agent1;
      const prevAgent2 = this.game.agent2;
      this.game.setAgents?.(null, null);
      this.game.agent1 = null;
      this.game.agent2 = null;

      let result = null;
      try {
    // --- minimal shaping state: serve/return only (PPO-friendly) ---
    const NET_X = 216; // court half (432/2)
    const isP2ServeAtStart = !!this.game.isPlayer2Serve;
    const learningServing = (this.learningPlayer === 1) ? !isP2ServeAtStart : isP2ServeAtStart;
    let serveCrossedNet = false;
    let sawBallOnMySide = false; // for return shaping when receiving
    let returnCrossedNet = false;

        while (true) {
      // round가 아니면 학습 프레임 카운트/trace 누적 안 하고 진행만
      if (this.game.state !== this.game.round) {
        this._setNeutralInput();
        this.game.stepLogic();
        continue;
      }

      this.game._lastRoundEvents = null;

      // obs(learningPlayer 기준)
      const obs1 = this.game.getObservation(1);
      const obs2 = this.game.getObservation(2);
      const obs = (this.learningPlayer === 1) ? obs1 : obs2;

      // action: prefer tuple-based API
      let aLearn = 0;
      let inputTuple = { xDirection: 0, yDirection: 0, powerHit: 0 };
      try {
        if (hasChooseInput) {
          inputTuple = agent.chooseInput(obs, this.learningPlayer, this.game) || inputTuple;
        } else {
          // Backward-compatible: some agents implement chooseAction(obs, playerIndex, game),
          // others implement chooseAction(physics, playerIndex, game). Try obs first, then physics.
          try {
            aLearn = agent.chooseAction(obs, this.learningPlayer, this.game);
          } catch (_) {
            aLearn = undefined;
          }
          if (typeof aLearn !== 'number') {
            try {
              aLearn = agent.chooseAction(this.game.physics, this.learningPlayer, this.game);
            } catch (_) {
              aLearn = 0;
            }
          }
          // map actionId -> tuple if helper exists
          if (typeof this.game._actionIdToInputTuple === 'function') {
            inputTuple = this.game._actionIdToInputTuple(aLearn | 0);
          }
        }
      } catch (_) {
        // keep defaults
      }

      // opponent is builtin (externalEnabled=false) so we only inject learning side
      const p1Input = (this.learningPlayer === 1) ? inputTuple : { xDirection: 0, yDirection: 0, powerHit: 0 };
      const p2Input = (this.learningPlayer === 2) ? inputTuple : { xDirection: 0, yDirection: 0, powerHit: 0 };
      this._applyInputs(p1Input, p2Input);

      // 한 프레임 진행
      // stepLogic()가 이벤트를 반환하지 않는 구현도 있어서,
      // 반환값과 game._lastRoundEvents 둘 다를 확인한다.
      const stepRet = this.game.stepLogic();
      // stepLogic()는 보통 boolean을 반환하지만, 일부 구현은 이벤트 객체를 반환할 수 있다.
      // pikavolley.js에서는 round()에서 this._lastRoundEvents에 이벤트를 저장한다.
      const ev = (stepRet && typeof stepRet === 'object')
        ? stepRet
        : (this.game._lastRoundEvents ?? null);

      // nextObs
      const nextObs1 = this.game.getObservation(1);
      const nextObs2 = this.game.getObservation(2);
      const nextObs = (this.learningPlayer === 1) ? nextObs1 : nextObs2;

      // === minimal shaping: serve/return only ===
      // Uses raw ball x (0..432). Reward is small and does not replace terminal +/-1.
      let shapingReward = 0;
      try {
        const bx0 = obs?.raw?.ball?.x;
        const bx1 = nextObs?.raw?.ball?.x;
        if (typeof bx0 === 'number' && typeof bx1 === 'number') {
          // Serve success: when serving, first time the ball crosses the net into opponent side
          if (learningServing && !serveCrossedNet) {
            const crossed = (this.learningPlayer === 1) ? (bx0 <= NET_X && bx1 > NET_X) : (bx0 >= NET_X && bx1 < NET_X);
            if (crossed) {
              serveCrossedNet = true;
              shapingReward += 0.10;
            }
          }

          // Return success: when receiving, first time we send the ball back across the net
          if (!learningServing) {
            const onMySide = (this.learningPlayer === 1) ? (bx1 < NET_X) : (bx1 > NET_X);
            if (onMySide) sawBallOnMySide = true;
            if (sawBallOnMySide && !returnCrossedNet) {
              const crossedBack = (this.learningPlayer === 1) ? (bx0 <= NET_X && bx1 > NET_X) : (bx0 >= NET_X && bx1 < NET_X);
              if (crossedBack) {
                returnCrossedNet = true;
                shapingReward += 0.10;
              }
            }
          }
        }
      } catch (_) {
        // ignore shaping errors
      }

      // Attach shaping to roundEvents so builder reward function can use it.
      if (ev && typeof ev === 'object') {
        ev.shapingReward = shapingReward;
      }

      // Terminal shaping penalties (serve/return failure) are attached on the terminal frame.
      const scoredByNow = (ev && typeof ev.scoredBy === 'number') ? ev.scoredBy : ((ev && typeof ev.scored === 'number') ? ev.scored : 0);
      if ((scoredByNow === 1 || scoredByNow === 2) && ev && typeof ev === 'object') {
        let terminalShaping = 0;
        if (learningServing && !serveCrossedNet && scoredByNow !== this.learningPlayer) terminalShaping -= 0.10;
        if (!learningServing && sawBallOnMySide && !returnCrossedNet && scoredByNow !== this.learningPlayer) terminalShaping -= 0.10;
        if (terminalShaping !== 0) ev.terminalShapingReward = terminalShaping;
      }

      // trace 포맷: trainer._buildPointReplay가 기대하는 형태(t.obs.p1/p2)
      if (trace.length < this.maxTraceFrames) {
        trace.push({
          t: frames,
          obs: { p1: obs1, p2: obs2, ball: (obs1?.raw?.ball ?? obs2?.raw?.ball ?? obs1?.ball ?? obs2?.ball ?? null) },
          action: { p1: p1Input, p2: p2Input },
          roundEvents: ev,
        });
      }

      // builder step (reward는 builder가 roundEvents로 내부 계산)
      const decisionInfo = (hasChooseInput && agent && agent.lastDecision)
        ? { logp: agent.lastDecision.logp ?? 0, value: agent.lastDecision.value ?? 0 }
        : null;

      builder.addStep({
        t: frames,
        obs,
        action: hasChooseInput ? inputTuple : (aLearn | 0),
        nextObs,
        done: false,
        info: decisionInfo,
        roundEvents: ev,
      });

      frames++;

      // 포인트 종료 감지: 프로젝트에 따라 필드명이 다를 수 있어 둘 다 지원
      // - scoredBy: 1|2
      // - scored: 1|2
      const scoredBy = (ev && typeof ev.scoredBy === 'number')
        ? ev.scoredBy
        : ((ev && typeof ev.scored === 'number') ? ev.scored : 0);
      if (scoredBy === 1 || scoredBy === 2) {
        const loser = (scoredBy === 1) ? 2 : 1;

        builder.finalize({
          scoredBy,
          loser,
          loseReason: 'SCORE',
        });

        const episode = builder.toEpisode();
        result = {
          ok: true,
          scoredBy,
          loser,
          loseReason: 'SCORE',
          frames,
          trace,
          episode,
        };
        break;
      }

      // hard safety (round 프레임 기준)
      if (frames >= this.hardSafetyFrames) {
        
        // HARD_SAFETY로 끊길 때는 다음 포인트를 위해 게임 상태를 한 번 리셋해준다.
        // (roundEnded=true인데 state가 round에 머무는 등, 후속 포인트가 영원히 점수 안 나는 상태를 방지)
        try {
          if (this.game.beforeStartOfNewGame) this.game.state = this.game.beforeStartOfNewGame;
          else if (this.game.startOfNewGame) this.game.state = this.game.startOfNewGame;
          this.game.frameCounter = 0;
          if (typeof this.game.roundEnded === 'boolean') this.game.roundEnded = false;
          if (typeof this.game.gameEnded === 'boolean') this.game.gameEnded = false;
        } catch (_) {}
builder.finalize({
          scoredBy: 0,
          loser: 0,
          loseReason: 'HARD_SAFETY',
        });
        const episode = builder.toEpisode();
        result = {
          ok: false,
          scoredBy: 0,
          loser: 0,
          loseReason: 'HARD_SAFETY',
          frames,
          trace,
          episode,
        };
        break;
      }
        }
      } finally {
        // restore agents
        this.game.setAgents?.(prevAgent1, prevAgent2);
        this.game.agent1 = prevAgent1;
        this.game.agent2 = prevAgent2;
      }

      return result;

    } catch (err) {
      console.error('[EpisodeRunner] runOnePoint failed', err);
      return {
        ok: false,
        scoredBy: 0,
        loser: 0,
        loseReason: 'EXCEPTION',
        frames,
        trace,
        episode: null,
      };
    }
  }

}