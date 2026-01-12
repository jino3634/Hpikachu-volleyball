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
    // 0) round 진입 강제(메뉴/intro 스킵) — 너가 이미 추가한 로직 사용
    if (this.game.state !== this.game.round) {
      const ok = this._forceEnterRoundNoMenu();
      if (!ok) throw new Error('Failed to enter round without menu');
    }

    const builder = new EpisodeBuilder({ learningPlayer: this.learningPlayer });

    const trace = [];
    let frames = 0;

    // agent는 trainer.init()에서 game.setAgents(this.agent, null)로 이미 붙어있음
    const agent = (this.learningPlayer === 1) ? this.game.agent1 : this.game.agent2;
    if (!agent || typeof agent.chooseAction !== 'function') {
      throw new Error(`Missing agent for learningPlayer=${this.learningPlayer}`);
    }

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

      // action
      const aLearn = (agent.chooseAction(obs, this.learningPlayer, this.game) | 0);

      // opponent는 builtin이면 0 (trainer가 externalEnabledP2=false로 해둠)
      const a1 = (this.learningPlayer === 1) ? aLearn : 0;
      const a2 = (this.learningPlayer === 2) ? aLearn : 0;

      this._applyActions(a1, a2);

      // 한 프레임 진행
      // 한 프레임 진행: 반환 이벤트를 우선 신뢰
      const ev = this.game.stepLogic() ?? null;

      // nextObs
      const nextObs1 = this.game.getObservation(1);
      const nextObs2 = this.game.getObservation(2);
      const nextObs = (this.learningPlayer === 1) ? nextObs1 : nextObs2;

      // trace 포맷: trainer._buildPointReplay가 기대하는 형태(t.obs.p1/p2)
      if (trace.length < this.maxTraceFrames) {
        trace.push({
          t: frames,
          obs: { p1: obs1, p2: obs2, ball: obs1?.ball ?? obs2?.ball ?? null },
          action: { p1: a1, p2: a2 },
          roundEvents: ev,
        });
      }

      // builder step (reward는 builder가 roundEvents로 내부 계산)
      builder.addStep({
        t: frames,
        obs,
        action: aLearn,
        nextObs,
        done: false,
        roundEvents: ev,
      });

      frames++;

      // 포인트 종료 감지: roundEvents.scored (pikavolley roundLogic 기준)
      const scoredBy = (ev && typeof ev.scored === 'number') ? ev.scored : 0;
      if (scoredBy === 1 || scoredBy === 2) {
        const loser = (scoredBy === 1) ? 2 : 1;

        builder.finalize({
          scoredBy,
          loser,
          loseReason: 'SCORE',
        });

        const episode = builder.toEpisode();
        return {
          ok: true,
          scoredBy,
          loser,
          loseReason: 'SCORE',
          frames,
          trace,
          episode,
        };
      }

      // hard safety (round 프레임 기준)
      if (frames >= this.hardSafetyFrames) {
        builder.finalize({
          scoredBy: 0,
          loser: 0,
          loseReason: 'HARD_SAFETY',
        });
        const episode = builder.toEpisode();
        return {
          ok: false,
          scoredBy: 0,
          loser: 0,
          loseReason: 'HARD_SAFETY',
          frames,
          trace,
          episode,
        };
      }
    }
  }

}
