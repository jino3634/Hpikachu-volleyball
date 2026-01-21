/**
 * Lightweight physics-based opponent AI (inspired by duckll.tw physics.js AI),
 * adapted to this codebase. It uses the engine's predicted landing point.
 *
 * Exported as a pure function for use from ui.js.
 */

import { GROUND_HALF_WIDTH, PLAYER_GROUND_Y } from './physics.js';

const MARGIN_X = 28;
const HIT_RANGE_X = 36;
const HIT_RANGE_Y = 46;

/**
 * Clamp x to player's own half (keeps AI from crossing the net).
 * @param {number} x
 * @param {boolean} isPlayer2
 */
function clampToHalf(x, isPlayer2) {
  if (isPlayer2) {
    const lo = GROUND_HALF_WIDTH + MARGIN_X;
    const hi = (GROUND_HALF_WIDTH * 2) - MARGIN_X;
    return Math.max(lo, Math.min(hi, x));
  }
  const lo = MARGIN_X;
  const hi = GROUND_HALF_WIDTH - MARGIN_X;
  return Math.max(lo, Math.min(hi, x));
}

/**
 * Decide userInput for a physics-based AI.
 * Signature matches Physics.aiController callback.
 *
 * @param {1|2} playerIndex
 * @param {any} me
 * @param {any} ball
 * @param {any} other
 * @param {any} userInput
 * @param {any} meta
 */
export function decidePhysicsAI(playerIndex, me, ball, other, userInput, meta) {
  const ui = userInput;
  if (!ui || !me || !ball) return;

  // Only decide on decision frames. (Physics will hold inputs between decisions.)
  if (meta && meta.decisionFrame === false) return;

  const isP2 = !!me.isPlayer2;

  // Target: predicted landing point (engine-maintained).
  let targetX = Number(ball.expectedLandingPointX ?? ball.x ?? (isP2 ? (GROUND_HALF_WIDTH + 80) : (GROUND_HALF_WIDTH - 80)));
  if (!Number.isFinite(targetX)) targetX = isP2 ? (GROUND_HALF_WIDTH + 80) : (GROUND_HALF_WIDTH - 80);
  targetX = clampToHalf(targetX, isP2);

  const meX = Number(me.x ?? (isP2 ? (GROUND_HALF_WIDTH + 120) : (GROUND_HALF_WIDTH - 120)));
  const meY = Number(me.y ?? PLAYER_GROUND_Y);

  // Horizontal movement toward target.
  const dx = targetX - meX;
  let xDir = 0;
  if (dx > 6) xDir = 1;
  else if (dx < -6) xDir = -1;

  // Vertical action:
  // - If ball is about to be reachable above the player, jump.
  // - Otherwise, stay grounded.
  const bx = Number(ball.x ?? targetX);
  const by = Number(ball.y ?? 0);
  const bvy = Number(ball.yVelocity ?? 0);

  let yDir = 0;
  const closeX = Math.abs(bx - meX) <= HIT_RANGE_X;
  const inFront = isP2 ? (bx >= GROUND_HALF_WIDTH) : (bx <= GROUND_HALF_WIDTH);

  // Jump when ball is coming down near us (simple heuristic).
  const comingDown = (bvy > 0) && (by >= 40) && (by <= 170);
  if (inFront && closeX && comingDown) yDir = -1;

  // Power hit (edge-trigger) when we're in the air and ball is in hit window.
  // We infer "in air" by meY < ground.
  const inAir = (meY < (PLAYER_GROUND_Y - 2));
  let power = 0;
  if (inAir && inFront && closeX && (Math.abs(by - meY) <= HIT_RANGE_Y) && (by <= 200)) {
    power = 1;
    // While power-hitting, aim toward opponent side away from opponent.
    const ox = Number(other && other.x);
    if (Number.isFinite(ox)) {
      // steer away from opponent x
      xDir = (ox < GROUND_HALF_WIDTH) ? 1 : -1;
    } else {
      xDir = isP2 ? -1 : 1;
    }
  }

  ui.xDirection = xDir | 0;
  ui.yDirection = yDir | 0;
  ui.powerHit = power ? 1 : 0;
}
