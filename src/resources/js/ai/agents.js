// agents.js
'use strict';

const GROUND_WIDTH = 432;
const GROUND_HALF_WIDTH = 216;

const BALL_RADIUS = 20;
const BALL_TOUCHING_GROUND_Y = 252;

const NET_PILLAR_HALF_WIDTH = 25;
const NET_PILLAR_TOP_TOP_Y = 176;

const INFINITE_LOOP_LIMIT = 1000;

// ------------------------------------------------------------
// Action IDs (내부용). 표의 A6..는 개념만 맞추면 됨.
// ------------------------------------------------------------
export const ACTION = {
  IDLE: 0,
  LEFT: 1,
  RIGHT: 2,
  JUMP: 3,
  JUMP_LEFT: 4,
  JUMP_RIGHT: 5,

  // ---- POWER ACTIONS ----
  POWER_NEUTRAL: 6,        // (0, 0, 1)
  POWER_LEFT: 7,           // (-1, 0, 1)
  POWER_RIGHT: 8,          // (1, 0, 1)

  POWER_UP: 9,             // (0, -1, 1)
  POWER_UP_LEFT: 10,       // (-1, -1, 1)
  POWER_UP_RIGHT: 11,      // (1, -1, 1)

  POWER_DOWN: 12,          // (0, 1, 1)  ✅ 추가(중립 아래)
  POWER_DOWN_LEFT: 13,     // (-1, 1, 1)
  POWER_DOWN_RIGHT: 14,    // (1, 1, 1)
};
const ACTION_SIZE = 15;

// ------------------------------------------------------------
// Action -> (xDir, yDir, powerHit) 변환
// (pikavolley.js의 _applyActionToKeyboard도 이 정의를 따라야 함)
// ------------------------------------------------------------
export function actionToInput(action) {
  switch (action) {
    case ACTION.LEFT:            return { x: -1, y: 0,  p: 0 };
    case ACTION.RIGHT:           return { x: 1,  y: 0,  p: 0 };
    case ACTION.JUMP:            return { x: 0,  y: -1, p: 0 };
    case ACTION.JUMP_LEFT:       return { x: -1, y: -1, p: 0 };
    case ACTION.JUMP_RIGHT:      return { x: 1,  y: -1, p: 0 };

    case ACTION.POWER_NEUTRAL:   return { x: 0,  y: 0,  p: 1 };
    case ACTION.POWER_LEFT:      return { x: -1, y: 0,  p: 1 };
    case ACTION.POWER_RIGHT:     return { x: 1,  y: 0,  p: 1 };

    case ACTION.POWER_UP:        return { x: 0,  y: -1, p: 1 };
    case ACTION.POWER_UP_LEFT:   return { x: -1, y: -1, p: 1 };
    case ACTION.POWER_UP_RIGHT:  return { x: 1,  y: -1, p: 1 };

    case ACTION.POWER_DOWN:      return { x: 0,  y: 1,  p: 1 };  // ✅ 추가
    case ACTION.POWER_DOWN_LEFT: return { x: -1, y: 1,  p: 1 };
    case ACTION.POWER_DOWN_RIGHT:return { x: 1,  y: 1,  p: 1 };

    case ACTION.IDLE:
    default:                     return { x: 0,  y: 0,  p: 0 };
  }
}

// ------------------------------------------------------------
// 액션 마스크
// ------------------------------------------------------------
export function getActionMask(physics, playerIndex) {
  const player = physics[`player${playerIndex}`];
  const mask = Array(ACTION_SIZE).fill(false);

  const s = player.state;

  // 4: lying_down_after_diving -> 입력 무의미
  if (s === 4) {
    mask[ACTION.IDLE] = true;
    return mask;
  }

  // 3: diving -> 입력 무의미(방향/파워 등)
  if (s === 3) {
    mask[ACTION.IDLE] = true;
    return mask;
  }

  // 0: ground normal
  if (s === 0) {
    mask[ACTION.IDLE] = true;
    mask[ACTION.LEFT] = true;
    mask[ACTION.RIGHT] = true;
    mask[ACTION.JUMP] = true;
    mask[ACTION.JUMP_LEFT] = true;
    mask[ACTION.JUMP_RIGHT] = true;

    // 지상 파워는 "다이브" 용도로만 (power+좌/우)
    mask[ACTION.POWER_LEFT] = true;
    mask[ACTION.POWER_RIGHT] = true;

    // 지상에서 의미없는 파워 조합은 노이즈라서 막는다:
    // POWER_NEUTRAL / POWER_UP / POWER_DOWN / POWER_UP_* / POWER_DOWN_*
    return mask;
  }

  // 1: jumping, 2: jumping_and_power_hitting
  if (s === 1 || s === 2) {
    mask[ACTION.IDLE] = true;
    mask[ACTION.LEFT] = true;
    mask[ACTION.RIGHT] = true;

    // 공중 파워 기술 전부 허용(네가 말한 "위+파워" 포함)
    mask[ACTION.POWER_NEUTRAL] = true;
    mask[ACTION.POWER_LEFT] = true;
    mask[ACTION.POWER_RIGHT] = true;

    mask[ACTION.POWER_UP] = true;
    mask[ACTION.POWER_UP_LEFT] = true;
    mask[ACTION.POWER_UP_RIGHT] = true;

    mask[ACTION.POWER_DOWN] = true;          // ✅ 추가(중립 아래)
    mask[ACTION.POWER_DOWN_LEFT] = true;
    mask[ACTION.POWER_DOWN_RIGHT] = true;

    return mask;
  }

  // 5/6 game end motions
  mask[ACTION.IDLE] = true;
  return mask;
}

// ------------------------------------------------------------
// 관측(정규화 포함)
// ------------------------------------------------------------
export function buildObsNormalized(physics, playerIndex) {
  const me = physics[`player${playerIndex}`];
  const opp = physics[`player${playerIndex === 1 ? 2 : 1}`];
  const ball = physics.ball;

  const nx = (x) => (x / GROUND_WIDTH) * 2 - 1; // 0..432 -> -1..1
  const ny = (y) => (y / 304) * 2 - 1;          // 0..304 -> -1..1

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const nv = (v, scale) => clamp(v / scale, -1, 1);

  return {
    me: {
      x: nx(me.x),
      y: ny(me.y),
      yVelocity: nv(me.yVelocity, 20),
      state: me.state,
      isPlayer2: me.isPlayer2 ? 1 : 0,
    },
    opp: {
      x: nx(opp.x),
      y: ny(opp.y),
      yVelocity: nv(opp.yVelocity, 20),
      state: opp.state,
    },
    ball: {
      x: nx(ball.x),
      y: ny(ball.y),
      xVelocity: nv(ball.xVelocity, 25),
      yVelocity: nv(ball.yVelocity, 35),
      expectedX: nx(ball.expectedLandingPointX),
      isPowerHit: ball.isPowerHit ? 1 : 0,
    },
  };
}

// ------------------------------------------------------------
// 파워히트 후 착지 X 시뮬(physics.js expectedLandingPointXWhenPowerHit 복제)
// "실수 유도" 버전(= net 처리 단순 버전) 유지
// ------------------------------------------------------------
function simulateExpectedLandingXWhenPowerHit(userInputX, userInputY, ball) {
  const copy = {
    x: ball.x,
    y: ball.y,
    xVelocity: ball.xVelocity,
    yVelocity: ball.yVelocity,
  };

  if (copy.x < GROUND_HALF_WIDTH) {
    copy.xVelocity = (Math.abs(userInputX) + 1) * 10;
  } else {
    copy.xVelocity = -(Math.abs(userInputX) + 1) * 10;
  }
  copy.yVelocity = Math.abs(copy.yVelocity) * userInputY * 2;

  let loopCounter = 0;
  while (true) {
    loopCounter++;

    const futureX = copy.x + copy.xVelocity;
    if (futureX < BALL_RADIUS || futureX > GROUND_WIDTH) {
      copy.xVelocity = -copy.xVelocity;
    }
    if (copy.y + copy.yVelocity < 0) {
      copy.yVelocity = 1;
    }

    if (
      Math.abs(copy.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH &&
      copy.y > NET_PILLAR_TOP_TOP_Y
    ) {
      if (copy.yVelocity > 0) {
        copy.yVelocity = -copy.yVelocity;
      }
    }

    copy.y = copy.y + copy.yVelocity;
    if (copy.y > BALL_TOUCHING_GROUND_Y || loopCounter >= INFINITE_LOOP_LIMIT) {
      return copy.x;
    }
    copy.x = copy.x + copy.xVelocity;
    copy.yVelocity += 1;
  }
}

// ------------------------------------------------------------
// 휴리스틱 에이전트 V1 (승률 최우선 1차)
// ------------------------------------------------------------
export class HeuristicAgentV1 {
  constructor({ playerIndex }) {
    this.playerIndex = playerIndex; // 1 or 2
  }

  chooseAction(physics) {
    const me = physics[`player${this.playerIndex}`];
    const opp = physics[`player${this.playerIndex === 1 ? 2 : 1}`];
    const ball = physics.ball;

    const mask = getActionMask(physics, this.playerIndex);

    // diving/lying -> IDLE
    if (me.state === 3 || me.state === 4) return ACTION.IDLE;

    // 내 코트 경계
    const leftBoundary = me.isPlayer2 ? GROUND_HALF_WIDTH : 0;
    const rightBoundary = me.isPlayer2 ? GROUND_WIDTH : GROUND_HALF_WIDTH;

    const ballOnMySide =
      ball.expectedLandingPointX > leftBoundary &&
      ball.expectedLandingPointX < rightBoundary;

    // ------------------------------------------------------------
    // 1) 지상(state=0): expectedX로 이동 + 점프/다이브
    // ------------------------------------------------------------
    if (me.state === 0) {
      // (A) 목표: 내 쪽이면 expectedX, 아니면 코트 중앙 대기
      const targetX = ballOnMySide
        ? ball.expectedLandingPointX
        : leftBoundary + (GROUND_HALF_WIDTH / 2);

      const dx = targetX - me.x;
      const absDx = Math.abs(dx);

      // (B) 점프 조건(원본 AI 느낌)
      const shouldJump =
        Math.abs(ball.xVelocity) <= 6 &&
        Math.abs(ball.x - me.x) < 32 &&
        ball.y > 0 && ball.y < 110 &&
        ball.yVelocity > 0;

      if (shouldJump) {
        if (dx > 10 && mask[ACTION.JUMP_RIGHT]) return ACTION.JUMP_RIGHT;
        if (dx < -10 && mask[ACTION.JUMP_LEFT]) return ACTION.JUMP_LEFT;
        if (mask[ACTION.JUMP]) return ACTION.JUMP;
      }

      // (C) 다이브 조건
      const shouldDive =
        ballOnMySide &&
        Math.abs(ball.x - me.x) > 64 + 5 &&
        ball.x > leftBoundary && ball.x < rightBoundary &&
        ball.y > 174;

      if (shouldDive) {
        if (ball.x > me.x && mask[ACTION.POWER_RIGHT]) return ACTION.POWER_RIGHT; // ground: dive right
        if (ball.x < me.x && mask[ACTION.POWER_LEFT]) return ACTION.POWER_LEFT;   // ground: dive left
      }

      // (D) 이동
      if (absDx > 10) {
        if (dx > 0 && mask[ACTION.RIGHT]) return ACTION.RIGHT;
        if (dx < 0 && mask[ACTION.LEFT]) return ACTION.LEFT;
      }
      return ACTION.IDLE;
    }

    // ------------------------------------------------------------
    // 2) 공중(state=1/2): 볼 쪽으로 이동 + 파워히트 방향 결정
    // ------------------------------------------------------------
    if (me.state === 1 || me.state === 2) {
      const dxBall = ball.x - me.x;
      if (Math.abs(dxBall) > 8) {
        if (dxBall > 0 && mask[ACTION.RIGHT]) return ACTION.RIGHT;
        if (dxBall < 0 && mask[ACTION.LEFT]) return ACTION.LEFT;
      }

        // 파워히트 트리거: 충돌 범위 근처
        const nearBall =
        Math.abs(ball.x - me.x) < 44 &&
        Math.abs(ball.y - me.y) < 36 &&
        ball.yVelocity > -2; // 너무 상승 중이면 보류(헛파워 감소)


      if (nearBall) {
        // 후보: (xDir, yDir) + power
        const candidates = [
          { a: ACTION.POWER_RIGHT,        x:  1, y:  0 },
          { a: ACTION.POWER_LEFT,         x: -1, y:  0 },
          { a: ACTION.POWER_NEUTRAL,      x:  0, y:  0 },

          { a: ACTION.POWER_UP_RIGHT,     x:  1, y: -1 },
          { a: ACTION.POWER_UP_LEFT,      x: -1, y: -1 },
          { a: ACTION.POWER_UP,           x:  0, y: -1 },

          { a: ACTION.POWER_DOWN_RIGHT,   x:  1, y:  1 },
          { a: ACTION.POWER_DOWN_LEFT,    x: -1, y:  1 },
          { a: ACTION.POWER_DOWN,         x:  0, y:  1 }, // ✅ 추가(중립 아래)
        ].filter(c => mask[c.a]);

        let best = null;
        let bestScore = -1e9;

        for (const c of candidates) {
          const landingX = simulateExpectedLandingXWhenPowerHit(c.x, c.y, ball);

            // ---- 코트 경계 ----
            const myLeft  = me.isPlayer2 ? GROUND_HALF_WIDTH : 0;
            const myRight = me.isPlayer2 ? GROUND_WIDTH : GROUND_HALF_WIDTH;
            const oppLeft  = me.isPlayer2 ? 0 : GROUND_HALF_WIDTH;
            const oppRight = me.isPlayer2 ? GROUND_HALF_WIDTH : GROUND_WIDTH;

            const inMyCourt = (landingX > myLeft && landingX < myRight);
            const inOppCourt = (landingX > oppLeft && landingX < oppRight);

            // ---- "승률 우선" 페널티/보너스 ----
            // 1) 최우선: 내 코트에 떨어지면 거의 자살급 → 매우 큰 페널티
            let score = 0;
            if (inMyCourt) score -= 2000;

            // 2) 상대 코트 밖(아웃)도 보통은 실점 유도 못 하고 그냥 득점 기회 날림 → 페널티
            // (단, 상대가 공중/누워있음 등 확실히 못 받는 상황이면 예외 가능. 지금은 보수적으로.)
            const outOfAll = (landingX <= 0 || landingX >= GROUND_WIDTH);
            if (outOfAll) score -= 1200;

            // 3) 상대 코트 안이면 기본 가점
            if (inOppCourt) score += 300;

            // 4) 상대 위치와 거리: 멀수록 받기 어려움 → 가점
            const distToOpp = Math.abs(landingX - opp.x);
            score += distToOpp * 2.0;

            // 5) "상대가 있는 쪽 반대편" 가점: opp.x 기준 반대편일수록 더 어려움
            // (상대가 왼쪽이면 오른쪽으로, 오른쪽이면 왼쪽으로)
            const oppCenter = (oppLeft + oppRight) * 0.5;
            const landingSideSign = (landingX - oppCenter);
            const oppSideSign = (opp.x - oppCenter);
            if (landingSideSign * oppSideSign < 0) score += 250; // 반대편이면 +

            /**
             * 6) 라인 근처는 리스크(조금만 틀어져도 아웃) → 약간 감점
             *   - 단, 너무 중앙만 치면 AI가 쉽게 받으니까 약하게만
             */
            const lineMargin = 18; // 안전 여유(픽셀). 너무 작게 하면 아웃 남발
            const distToOppLeftLine = Math.abs(landingX - oppLeft);
            const distToOppRightLine = Math.abs(landingX - oppRight);
            const nearOppLine = (distToOppLeftLine < lineMargin || distToOppRightLine < lineMargin);
            if (nearOppLine) score -= 120;

            // 7) (선택) 상대가 누워있거나 다이빙 중이면 “아웃 페널티”를 조금 완화해도 됨
            // 지금은 생략(안정 우선)


            // // 방향 보정치(승률 우선)
            // // y = -1(UP): 공격적, y = 0: 기본, y = 1(DOWN): 리스크(공중에서만 의미 있지만 자살도 있음)
            // if (c.y === -1) score += 80;
            // else if (c.y === 0) score += 0;
            // else if (c.y === 1) score -= 60;

            // // 대각(좌/우 + UP)은 더 강하게 밀어도 됨
            // if (c.y === -1 && c.x !== 0) score += 40;

            // // DOWN 대각은 특히 리스크 → 추가 감점
            // if (c.y === 1 && c.x !== 0) score -= 40;



          if (score > bestScore) {
            bestScore = score;
            best = c.a;
          }
        }

        if (best != null) return best;
      }

      return ACTION.IDLE;
    }

    return ACTION.IDLE;
  }
}
