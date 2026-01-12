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
   * @param {{
   *   game: any,                 // pikaVolley instance
   *   learningPlayer: 1|2,
   *   maxTraceFrames?: number,
   *   hardSafetyFrames?: number
   * }} args
   */
  constructor({ game, learningPlayer, maxTraceFrames = 2400, hardSafetyFrames = 36000 }) {
    this.game = game;
    this.learningPlayer = learningPlayer;

    // 리플레이(trace) 길이 제한(너무 길면 메모리/저장 부담)
    this.maxTraceFrames = maxTraceFrames;

    // "진짜 무한 랠리/스테이트 꼬임" 감지용 비상장치
    // - round 프레임만 카운트
    this.hardSafetyFrames = hardSafetyFrames;
  }

  // ─────────────────────────────────────────────────────────────
  // 상태/입력 헬퍼
  // ─────────────────────────────────────────────────────────────

  _setControlModeExternal() {
    if (typeof this.game.setControlMode === 'function') this.game.setControlMode('external');
    else this.game.controlMode = 'external';
  }

  _setNeutralInput() {
    if (typeof this.game.setExternalActions === 'function') {
      this.game.setExternalActions(0, 0);
    } else {
      this.game._heldActionP1 = 0;
      this.game._heldActionP2 = 0;
    }
  }

  _applyActions(p1Action, p2Action) {
    if (typeof this.game.setExternalActions === 'function') {
      this.game.setExternalActions(p1Action, p2Action);
    } else {
      this.game._heldActionP1 = p1Action;
      this.game._heldActionP2 = p2Action;
    }
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
   * @param {{
   *   agentStep: (obs:any) => number,  // learning player action id
   *   opponentStep?: (obs:any) => number, // (선택) 상대도 외부제어면 사용
   * }} args
   */
  async runOnePoint({ agentStep, opponentStep = null }) {
    // 0) 시작 시 바로 round로 진입(메뉴 자동 입력 없음)
    if (this.game.state !== this.game.round) {
      const ok = this._forceEnterRoundNoMenu();
      if (!ok) {
        throw new Error('Failed to enter round without menu (no powerHit automation)');
      }
    }

    const builder = new EpisodeBuilder({ learningPlayer: this.learningPlayer });

    // trace는 배열로 관리 (프로젝트에 TraceBuffer가 있으면 그걸 써도 됨)
    const trace = [];

    let frames = 0; // ✅ round 프레임만 카운트
    let last = null;
    let lastEv = null;

    while (true) {
      // ✅ round가 아니면: 입력 0으로 상태만 진행, 학습/trace/frames 누적 금지
      if (this.game.state !== this.game.round) {
        this._setNeutralInput();
        this.game.stepLogic();
        continue;
      }

      // 1) 관측 만들기(너 프로젝트 관측 포맷에 맞춰 수정 가능)
      const obs = this.game.getObservation
        ? this.game.getObservation()
        : {}; // 없으면 최소로

      // 2) 행동 결정
      const aLearn = agentStep(obs) | 0;

      let a1 = 0;
      let a2 = 0;

      if (this.learningPlayer === 1) {
        a1 = aLearn;
        a2 = opponentStep ? (opponentStep(obs) | 0) : 0; // 상대가 CPU면 0 유지(엔진이 내부 처리)
      } else {
        a2 = aLearn;
        a1 = opponentStep ? (opponentStep(obs) | 0) : 0;
      }

      // 3) 입력 적용
      this._applyActions(a1, a2);

      // 4) 한 프레임 진행
      const ev = this.game.stepLogic(); // roundLogic 이벤트 객체를 기대
      lastEv = ev;

      // 5) trace 저장(길이 제한)
      if (trace.length < this.maxTraceFrames) {
        trace.push(this._captureFrame());
      }

      // 6) 학습 프레임 카운트(여기서만 증가)
      frames++;

      // 7) builder에 step 추가 (보상/terminal은 너 프로젝트 기존 로직을 유지해야 함)
      //    아래는 "틀"만 제공. 실제 reward 계산/terminal 처리 방식에 맞춰 고쳐 써.
      if (builder && typeof builder.addStep === 'function') {
        const obs = this.game.getObservation();
        const action = aLearn;

        this._applyActions(a1, a2);
        const ev = this.game.stepLogic();

        const nextObs = this.game.getObservation();
        builder.addStep({
          t: frames,
          obs,
          action: aLearn,
          nextObs,
          done: false,
          info: { reward: 0 },     // ✅ 타입 OK
        });
      }

      // 8) 포인트 종료 감지
      if (this._isPointEnded(ev)) {
        // builder finalize (프로젝트 형식에 맞게)
        if (builder && typeof builder.finalize === 'function') {
          builder.finalize({
            scoredBy: ev.scoredBy || 0,
            loser: ev.loser || 0,
            loseReason: ev.loseReason || '',
          });
        }

        return {
          ok: true,
          scoredBy: ev.scoredBy || 0,
          loser: ev.loser || 0,
          loseReason: ev.loseReason || '',
          frames,
          trace,
          last: lastEv,
          episode: builder && typeof builder.toEpisode === 'function' ? builder.toEpisode() : null,
        };
      }

      // 9) 비상장치(throw 대신 “강제 실패 반환” 권장)
      if (frames >= this.hardSafetyFrames) {
        console.warn('[EpisodeRunner] Hard safety hit (round did not end).', {
          frames,
          state: this._getStateName(),
        });

        if (builder && typeof builder.finalize === 'function') {
          builder.finalize({
            scoredBy: 0,
            loser: 0,
            loseReason: 'UNKNOWN_END',
          });
        }

        return {
          ok: false,
          scoredBy: 0,
          loser: 0,
          loseReason: 'UNKNOWN_END',
          frames,
          trace,
          last: lastEv,
          episode: builder && typeof builder.toEpisode === 'function' ? builder.toEpisode() : null,
        };
      }
    }
  }
}
