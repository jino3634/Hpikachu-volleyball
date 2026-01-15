// src/resources/js/ai/touch_tracker.js
export class TouchTracker {
  constructor() {
    this.resetPoint();
  }

  resetPoint() {
    this.lastTouch = 0;      // 0=none, 1=P1, 2=P2
    this.prevLastTouch = 0;  // 변화 감지용(episode_runner에서 사용하면 편함)
  }

  // 물리 실행 "직후"에 호출
  observePhysics(physics) {
    // 동시에 둘 다 true일 가능성은 매우 낮지만, 생기면 더 '최근' 쪽이 덮어씀
    if (physics.player1?.isCollisionWithBallHappened) this.lastTouch = 1;
    if (physics.player2?.isCollisionWithBallHappened) this.lastTouch = 2;
  }

  // lastTouch 변화 감지용(옵션)
  commitFrame() {
    this.prevLastTouch = this.lastTouch;
  }
}
