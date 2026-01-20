'use strict';
import { GROUND_HALF_WIDTH, PLAYER_GROUND_Y } from '../physics.js';

/**
 * Simple weighted-rule policy for evolution.
 * Genome is a bag of weights/thresholds.
 */

export function defaultGenome() {
  return {
    // defense
    wMoveToLanding: 1.0,
    wStayCenter: 0.15,
    // attack / power usage
    wPower: 0.35,
    // penalties
    wAvoidNet: 1.0,
    // thresholds
    jumpMinBallY: 110,
    powerMinBallY: 90,
    powerMaxDX: 90,
  };
}

/**
 * @param {any} obs
 * @param {Record<string, number>} genome
 * @returns {{xDirection:-1|0|1, yDirection:-1|0|1, powerHit:0|1}}
 */
export function chooseAction(obs, genome) {
  const g = genome || defaultGenome();
  const me = obs.me;
  const opp = obs.opp;
  const ball = obs.ball;

  // --- target positions ---
  const landingX = Number(ball.expectedLandingX ?? ball.expectedLandingPointX ?? ball.expectedLandingPoint ?? me.x);
  const centerX = me.isPlayer2
  ? (GROUND_HALF_WIDTH + (GROUND_HALF_WIDTH / 2))
  : (GROUND_HALF_WIDTH / 2);

  // basic defense desire: move toward landingX when ball is coming to my side
  const mySide = me.isPlayer2 ? 2 : 1;
  const ballSide = (ball.x < GROUND_HALF_WIDTH) ? 1 : 2;
  const danger = (ballSide === mySide);

  const targetX = danger ? landingX : centerX;
  const dx = targetX - me.x;

  let xDir = 0;
  if (dx > 8) xDir = 1;
  else if (dx < -8) xDir = -1;

  // jump/power heuristic
  const nearBall = (Math.abs(ball.x - me.x) <= 72);
  const ballAbove = (ball.y <= (g.jumpMinBallY || 110));
  const canJump = (me.state === 0) && (me.y >= PLAYER_GROUND_Y);

  let yDir = 0;
  if (canJump && nearBall && ballAbove) yDir = -1;

  // powerHit: only if jumping (state 2 is power state in original); we approximate: request power when ball is near and above threshold
  const wantPower = (nearBall && (ball.y <= (g.powerMinBallY || 90)) && (Math.abs(ball.x - me.x) <= (g.powerMaxDX || 90)));
  const powerHit = wantPower ? 1 : 0;

  return { xDirection: /** @type {-1|0|1} */ (xDir), yDirection: /** @type {-1|0|1} */ (yDir), powerHit: /** @type {0|1} */ (powerHit) };
}
