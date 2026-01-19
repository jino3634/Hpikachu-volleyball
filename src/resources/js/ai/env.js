// ai/env.js
'use strict';

import { GROUND_HALF_WIDTH, PikaPhysics } from '../physics.js';
import fs from 'node:fs';

/**
 * Bit layout (6 bits):
 *  bit0: P1 Left
 *  bit1: P1 Right
 *  bit2: P1 Up
 *  bit3: P2 Left
 *  bit4: P2 Right
 *  bit5: P2 Up
 */
function maskToControls(mask) {
  const m = mask | 0;
  return {
    left: (m & 1) !== 0,
    right: (m & 2) !== 0,
    up: (m & 4) !== 0,
  };
}

/**
 * Mimics the subset of PikaKeyboard that physics expects.
 * physics.runEngineForNextFrame(keyboardArray) reads:
 * - xDirection (-1, 0, 1)
 * - yDirection (-1, 0, 1)  (jump)
 * - powerHit (0/1)         (we can keep 0 for now, since you're doing A-only)
 */
class FakeKeyboard {
  constructor() {
    this.xDirection = 0;
    this.yDirection = 0;
    this.powerHit = 0;
    // for edge-trigger power button
    this._prevPowerRequested = 0;
  }

  /**
   * Apply an action mask (Left/Right/Up only)
   * @param {number} mask3bit bit0 left, bit1 right, bit2 up
   */
  apply3(mask3bit) {
    const c = maskToControls(mask3bit);
    // left+right => neutral (0)
    this.xDirection = c.left === c.right ? 0 : (c.left ? -1 : 1);
    // up => jump (-1 matches your menu logic convention)
    this.yDirection = c.up ? -1 : 0;
    // A-only 정책이니까 powerHit은 0 고정
    this.powerHit = 0;
    // for edge-trigger power button
    this._prevPowerRequested = 0;
  }

  /**
   * Apply tuple input (xDirection, yDirection, powerHit).
   * - xDirection: -1/0/1
   * - yDirection: -1/0/1  (up/neutral/down)
   * - powerHit: 0/1       (edge-trigger: 0->1 only)
   * @param {{xDirection:number, yDirection:number, powerHit:number}} t
   */
  applyTuple(t) {
    const xRaw = (Number(t?.xDirection ?? 0) | 0);
    const yRaw = (Number(t?.yDirection ?? 0) | 0);
    const pReq = (Number(t?.powerHit ?? 0) ? 1 : 0);

    this.xDirection = xRaw < 0 ? -1 : (xRaw > 0 ? 1 : 0);
    this.yDirection = yRaw < 0 ? -1 : (yRaw > 0 ? 1 : 0);

    // edge-trigger (no auto-repeat while held)
    this.powerHit = (pReq == 1 && this._prevPowerRequested == 0) ? 1 : 0;
    this._prevPowerRequested = pReq;
  }

}

/**
 * Replay ring buffer for most recent N games.
 */
export class ReplayRingBuffer {
  /**
   * @param {number} capacity number of games to keep
   */
  constructor(capacity = 50) {
    this.capacity = Math.max(1, capacity | 0);
    this.items = new Array(this.capacity);
    this.size = 0;
    this.head = 0; // next write index
  }

  push(gameReplay) {
    this.items[this.head] = gameReplay;
    this.head = (this.head + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  /**
   * @returns {any[]} in chronological order (oldest -> newest)
   */
  toArray() {
    const out = [];
    const start = (this.head - this.size + this.capacity) % this.capacity;
    for (let i = 0; i < this.size; i++) {
      out.push(this.items[(start + i) % this.capacity]);
    }
    return out;
  }

  last() {
    if (this.size === 0) return null;
    const idx = (this.head - 1 + this.capacity) % this.capacity;
    return this.items[idx];
  }
}

/**
 * Headless environment for training.
 * - Deterministic if physics is deterministic and you avoid Math.random in logic.
 * - Reward: +1 for scoring, -1 for conceding, else 0. (from P1 perspective)
 */
export class PikaEnv {
  /**
   * @param {{
   *   winningScore?: number,
   *   practiceMode?: boolean,
   *   maxFramesPerGame?: number,
   *   keepReplays?: number,
   * }} opts
   */
  constructor(opts = {}) {
    this.winningScore = opts.winningScore ?? 15;
    this.practiceMode = opts.practiceMode ?? false;
    this.maxFramesPerGame = opts.maxFramesPerGame ?? 60 * 60; // safety cap
    this.replays = new ReplayRingBuffer(opts.keepReplays ?? 21);

    // physics options (true,true) matches your browser ctor
    this.physics = new PikaPhysics(true, true);
    // Make sure training is not using built-in computer AI
    this.physics.player1.isComputer = false;
    this.physics.player2.isComputer = false;

    this.kb1 = new FakeKeyboard();
    this.kb2 = new FakeKeyboard();
    this.keyboardArray = [this.kb1, this.kb2];
    this.prev = {
    p1x: this.physics.player1.x,
    p1y: this.physics.player1.y,
    p2x: this.physics.player2.x,
    p2y: this.physics.player2.y,
    };

    this.reset();
  }

  /**
   * Reset a full game (scores=0, new round init, P1 serves initially)
   * @param {number} [seed] optional; stored in replay for reproducibility
   */
  reset(seed = 0) {
    // NOTE: your physics may use Math.random internally; if so, you need to patch it for real determinism.
    this.seed = seed | 0;

    this.scores = [0, 0];
    this.gameEnded = false;
    this.roundEnded = false;
    this.isPlayer2Serve = false;

    this.frame = 0;

    // init physics for new round
    this.physics.player1.gameEnded = false;
    this.physics.player1.isWinner = false;
    this.physics.player2.gameEnded = false;
    this.physics.player2.isWinner = false;

    this.physics.player1.initializeForNewRound();
    this.physics.player2.initializeForNewRound();
    this.physics.ball.initializeForNewRound(this.isPlayer2Serve);

    //prev 동기화
    this.prev.p1x = this.physics.player1.x;
    this.prev.p1y = this.physics.player1.y;
    this.prev.p2x = this.physics.player2.x;
    this.prev.p2y = this.physics.player2.y;

    // start a new replay recording
    this.currentReplay = {
      seed: this.seed,
      winningScore: this.winningScore,
      frames: [], // {p1Mask, p2Mask}
      finalScore: null,
      winner: null, // 1 or 2
      endedBy: null, // 'score' | 'maxFrames'
      _pushed: false,
    };

    return this.getObs();
  }

  /**
   * Observation for learning
   */
getObs() {
  const b  = this.physics.ball;
  const p1 = this.physics.player1;
  const p2 = this.physics.player2;

  // ------------------------------------------------------------
  // ✅ 상태/가능여부 파생값 (학습 편의)
  // ------------------------------------------------------------
  const derive = (p) => {
    const state = Number(p?.state ?? 0) | 0;
    const lying = Number(p?.lyingDownDurationLeft ?? 0) | 0;
    const isDiving = (state === 3) ? 1 : 0;
    const isLying = (state === 4 || lying > 0) ? 1 : 0;
    const isAir = (state === 1 || state === 2) ? 1 : 0;
    const canAct = (!isDiving && !isLying) ? 1 : 0;
    return { state, canAct, isAir, isDiving, isLying };
  };
  const s1 = derive(p1);
  const s2 = derive(p2);

  const vx1 = p1.x - this.prev.p1x;
  const vy1 = p1.y - this.prev.p1y;
  const vx2 = p2.x - this.prev.p2x;
  const vy2 = p2.y - this.prev.p2y;

  return {
    frame: this.frame,
    ball: {
      x: b.x,
      y: b.y,
      xVelocity: b.xVelocity,
      yVelocity: b.yVelocity,
      isPowerHit: b.isPowerHit ? 1 : 0,
    },
    players: [
      {
        x: p1.x,
        y: p1.y,
        vx: vx1,
        vy: vy1,
        yVelocity: p1.yVelocity,
        state: s1.state,
        canAct: s1.canAct,
        isAir: s1.isAir,
        isDiving: s1.isDiving,
        isLying: s1.isLying,
      },
      {
        x: p2.x,
        y: p2.y,
        vx: vx2,
        vy: vy2,
        yVelocity: p2.yVelocity,
        state: s2.state,
        canAct: s2.canAct,
        isAir: s2.isAir,
        isDiving: s2.isDiving,
        isLying: s2.isLying,
      },
    ],
    tactical: {
      ballSide: b.x < GROUND_HALF_WIDTH ? 1 : 2,
      server: this.isPlayer2Serve ? 2 : 1,
    },
    score: { p1: this.scores[0], p2: this.scores[1] },
    done: this.gameEnded,
  };
}



  /**
   * Step one frame with action masks.
   * @param {number} p1Mask 3-bit (left,right,up) or full 6-bit; we'll read low3.
   * @param {number} p2Mask 3-bit (left,right,up) or full 6-bit; we'll read low3.
   * @returns {{obs:any, reward:number, done:boolean, info:any}}
   */
  step(p1Mask, p2Mask) {
  if (this.gameEnded) {
    return { obs: this.getObs(), reward: 0, done: true, info: { alreadyDone: true } };
  }

  const p1Full = (p1Mask | 0) & 0x7;
  const p2Full = (p2Mask | 0) & 0x7;
  this.currentReplay.frames.push({ p1Mask: p1Full, p2Mask: p2Full });

  this.kb1.apply3(p1Full);
  this.kb2.apply3(p2Full);

  const isBallTouchingGround = this.physics.runEngineForNextFrame(this.keyboardArray);

  // scoring 로직 (physics 결과 기반이므로 여기 둬도 됨)
  let reward = 0;
  if (isBallTouchingGround && !this.practiceMode && !this.roundEnded && !this.gameEnded) {
    const punchX = this.physics.ball.punchEffectX;
    if (punchX < GROUND_HALF_WIDTH) {
      this.isPlayer2Serve = true;
      this.scores[1] += 1;
      reward = -1;
      if (this.scores[1] >= this.winningScore) this._setWinner(2);
      else this.roundEnded = true;
    } else {
      this.isPlayer2Serve = false;
      this.scores[0] += 1;
      reward = +1;
      if (this.scores[0] >= this.winningScore) this._setWinner(1);
      else this.roundEnded = true;
    }

    if (this.roundEnded && !this.gameEnded) {
      this._startNextRound(); // 여기서 prev 동기화까지 해줄 예정
    }
  }

  // ✅ 프레임 증가를 obs 생성 전에 해서 obs.frame이 "현재 상태"와 일치하게
  this.frame++;

  // ✅ obs는 prev(이전 프레임) 기준으로 속도를 계산해야 하므로, prev 업데이트 전에 만든다
  const obs = this.getObs();

  // ✅ prev는 마지막에 갱신 (다음 step의 속도 계산용)
  this.prev.p1x = this.physics.player1.x;
  this.prev.p1y = this.physics.player1.y;
  this.prev.p2x = this.physics.player2.x;
  this.prev.p2y = this.physics.player2.y;

  // safety cap
  if (!this.gameEnded && this.frame >= this.maxFramesPerGame) {
    this.gameEnded = true;
    this.currentReplay.endedBy = 'maxFrames';
    this.currentReplay.finalScore = { p1: this.scores[0], p2: this.scores[1] };
    this.replays.push(this.currentReplay);
  }

  const done = this.gameEnded;
  const info = {
    score: { p1: this.scores[0], p2: this.scores[1] },
    winner: this.currentReplay.winner,
    endedBy: this.currentReplay.endedBy,
    lastAction: { p1Mask: p1Full, p2Mask: p2Full },
  };

  if (done && !this.currentReplay._pushed) {
    this.replays.push(this.currentReplay);
    this.currentReplay._pushed = true;
  }

  return { obs, reward, done, info };
}

  /**
   * Step one frame with tuple inputs.
   * This supports yDirection down (+1) and powerHit (edge-trigger).
   * @param {{xDirection:number, yDirection:number, powerHit:number}} p1Input
   * @param {{xDirection:number, yDirection:number, powerHit:number}} p2Input
   * @returns {{obs:any, reward:number, done:boolean, info:any}}
   */
  stepTuple(p1Input, p2Input) {
    if (this.gameEnded) {
      return { obs: this.getObs(), reward: 0, done: true, info: { alreadyDone: true } };
    }

    // record (optional)
    this.currentReplay.frames.push({
      p1Tuple: { xDirection: p1Input?.xDirection ?? 0, yDirection: p1Input?.yDirection ?? 0, powerHit: p1Input?.powerHit ?? 0 },
      p2Tuple: { xDirection: p2Input?.xDirection ?? 0, yDirection: p2Input?.yDirection ?? 0, powerHit: p2Input?.powerHit ?? 0 },
    });

    this.kb1.applyTuple(p1Input);
    this.kb2.applyTuple(p2Input);

    const isBallTouchingGround = this.physics.runEngineForNextFrame(this.keyboardArray);

    let reward = 0;
    if (isBallTouchingGround && !this.practiceMode && !this.roundEnded && !this.gameEnded) {
      const punchX = this.physics.ball.punchEffectX;
      if (punchX < GROUND_HALF_WIDTH) {
        this.isPlayer2Serve = true;
        this.scores[1] += 1;
        reward = -1;
        if (this.scores[1] >= this.winningScore) this._setWinner(2);
        else this.roundEnded = true;
      } else {
        this.isPlayer2Serve = false;
        this.scores[0] += 1;
        reward = +1;
        if (this.scores[0] >= this.winningScore) this._setWinner(1);
        else this.roundEnded = true;
      }

      if (this.roundEnded && !this.gameEnded) {
        this._startNextRound();
      }
    }

    this.frame++;
    const obs = this.getObs();

    this.prev.p1x = this.physics.player1.x;
    this.prev.p1y = this.physics.player1.y;
    this.prev.p2x = this.physics.player2.x;
    this.prev.p2y = this.physics.player2.y;

    if (!this.gameEnded && this.frame >= this.maxFramesPerGame) {
      this.gameEnded = true;
      this.currentReplay.endedBy = 'maxFrames';
      this.currentReplay.finalScore = { p1: this.scores[0], p2: this.scores[1] };
      this.replays.push(this.currentReplay);
    }

    const done = this.gameEnded;
    const info = {
      score: { p1: this.scores[0], p2: this.scores[1] },
      winner: this.currentReplay.winner,
      endedBy: this.currentReplay.endedBy,
      lastAction: { p1Input, p2Input },
    };

    if (done && !this.currentReplay._pushed) {
      this.replays.push(this.currentReplay);
      this.currentReplay._pushed = true;
    }

    return { obs, reward, done, info };
  }


  _setWinner(winner /* 1|2 */) {
    this.gameEnded = true;
    this.currentReplay.endedBy = 'score';
    this.currentReplay.winner = winner;
    this.currentReplay.finalScore = { p1: this.scores[0], p2: this.scores[1] };

    // Keep flags consistent with browser logic (optional)
    this.physics.player1.isWinner = winner === 1;
    this.physics.player2.isWinner = winner === 2;
    this.physics.player1.gameEnded = true;
    this.physics.player2.gameEnded = true;
  }

_startNextRound() {
  this.physics.player1.initializeForNewRound();
  this.physics.player2.initializeForNewRound();
  this.physics.ball.initializeForNewRound(this.isPlayer2Serve);

  // ✅ 라운드 넘어갈 때 위치가 "순간이동"하므로 prev를 즉시 맞춰서 vx/vy 튐 방지
  this.prev.p1x = this.physics.player1.x;
  this.prev.p1y = this.physics.player1.y;
  this.prev.p2x = this.physics.player2.x;
  this.prev.p2y = this.physics.player2.y;

  this.roundEnded = false;
}

  /**
   * Get the latest N replays (oldest->newest)
   */
  getRecentReplays() {
    return this.replays.toArray();
  }

  getLatestReplay() {
    return this.replays.last();
  }

saveRecentReplaysToFile(filepath = 'ai/replays.json') {
  const data = {
    createdAt: new Date().toISOString(),
    replays: this.getRecentReplays(),
  };
  fs.mkdirSync(filepath.split('/').slice(0, -1).join('/'), { recursive: true });
  fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf-8');
}

saveLatestReplayToFile(filepath = 'ai/replay_latest.json') {
  const rep = this.getLatestReplay();
  fs.mkdirSync(filepath.split('/').slice(0, -1).join('/'), { recursive: true });
  fs.writeFileSync(filepath, JSON.stringify(rep, null, 2), 'utf-8');
}


}
