/**
 * This module takes charge of the user input via keyboard
 */
'use strict';
import { PikaUserInput } from './physics.js';

/**
 * Class representing a keyboard used to control a player
 */
export class PikaKeyboard extends PikaUserInput {
  constructor(left, right, up, down, powerHit, downRight = null) {
    super();

    /** @type {boolean} */
    this.powerHitKeyIsDownPrevious = false;

    /** @type {Key} */
    this.leftKey = new Key(left);
    /** @type {Key} */
    this.rightKey = new Key(right);
    /** @type {Key} */
    this.upKey = new Key(up);
    /** @type {Key} */
    this.downKey = new Key(down);
    /** @type {Key} */
    this.powerHitKey = new Key(powerHit);

    /** @type {Key|null} */
    this.downRightKey = downRight ? new Key(downRight) : null;

    // override: {xDirection, yDirection, powerHit} or null
    this._override = null;
  }

  /** AI/Replay에서 입력 강제 주입 */
  setOverrideInput(xDirection, yDirection, powerHit) {
    this._override = {
      xDirection: xDirection | 0,
      yDirection: yDirection | 0,
      powerHit: powerHit ? 1 : 0,
    };
  }

  clearOverrideInput() {
    this._override = null;
  }

  /**
   * Freeze input snapshot for one game frame
   */
  getInput() {
    // ✅ override 우선
    if (this._override) {
      this.xDirection = this._override.xDirection;
      this.yDirection = this._override.yDirection;
      this.powerHit = this._override.powerHit;

      // ✅ POWER는 1프레임 트리거 보장 (적용 후 자동 0)
      this._override.powerHit = 0;

      // ✅ 사람 입력으로 돌아왔을 때 꼬임 방지
      this.powerHitKeyIsDownPrevious = false;
      return;
    }

    // ---- 사람 입력 로직 ----
    if (this.leftKey.isDown) {
      this.xDirection = -1;
    } else if (this.rightKey.isDown || (this.downRightKey && this.downRightKey.isDown)) {
      this.xDirection = 1;
    } else {
      this.xDirection = 0;
    }

    if (this.upKey.isDown) {
      this.yDirection = -1;
    } else if (this.downKey.isDown || (this.downRightKey && this.downRightKey.isDown)) {
      this.yDirection = 1;
    } else {
      this.yDirection = 0;
    }

    const isDown = this.powerHitKey.isDown;
    if (!this.powerHitKeyIsDownPrevious && isDown) {
      this.powerHit = 1;
    } else {
      this.powerHit = 0;
    }
    this.powerHitKeyIsDownPrevious = isDown;
  }

  /** Subscribe keydown, keyup event listeners */
  subscribe() {
    this.leftKey.subscribe();
    this.rightKey.subscribe();
    this.upKey.subscribe();
    this.downKey.subscribe();
    this.powerHitKey.subscribe();
    if (this.downRightKey) this.downRightKey.subscribe();
  }

  /** Unsubscribe keydown, keyup event listeners */
  unsubscribe() {
    this.leftKey.unsubscribe();
    this.rightKey.unsubscribe();
    this.upKey.unsubscribe();
    this.downKey.unsubscribe();
    this.powerHitKey.unsubscribe();
    if (this.downRightKey) this.downRightKey.unsubscribe();
  }
}

/**
 * Class representing a key on a keyboard
 * referred to: https://github.com/kittykatattack/learningPixi
 */
class Key {
  constructor(value) {
    this.value = value;
    this.isDown = false;
    this.isUp = true;

    this._subscribed = false;

    this.downListener = this.downHandler.bind(this);
    this.upListener = this.upHandler.bind(this);
  }

  downHandler(event) {
    if (event.code === this.value) {
      this.isDown = true;
      this.isUp = false;
      event.preventDefault();
    }
  }

  upHandler(event) {
    if (event.code === this.value) {
      this.isDown = false;
      this.isUp = true;
      event.preventDefault();
    }
  }

  subscribe() {
    if (this._subscribed) return;
    // keyup 먼저 붙여 “짧게 눌렀다 떼는” 케이스에서 꼬임 방지
    window.addEventListener('keyup', this.upListener);
    window.addEventListener('keydown', this.downListener);
    this._subscribed = true;
  }

  unsubscribe() {
    if (!this._subscribed) return;
    window.removeEventListener('keydown', this.downListener);
    window.removeEventListener('keyup', this.upListener);
    this.isDown = false;
    this.isUp = true;
    this._subscribed = false;
  }
}
