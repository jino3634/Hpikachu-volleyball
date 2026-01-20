/**
 * Observation builder for evolution/RL.
 */
'use strict';

/** @typedef {import('../physics.js').PikaPhysics} PikaPhysics */

/**
 * Build observation object.
 *
 * Keep it small + stable; future steps can add more fields.
 *
 * @param {PikaPhysics} physics
 * @param {1|2} playerIndex
 */
export function makeObservation(physics, playerIndex) {
  const p1 = physics.player1;
  const p2 = physics.player2;
  const ball = physics.ball;

  const me = (playerIndex === 1) ? p1 : p2;
  const opp = (playerIndex === 1) ? p2 : p1;

  return {
    me: {
      x: me.x,
      y: me.y,
      yVelocity: me.yVelocity,
      state: me.state,
      isPlayer2: !!me.isPlayer2,
    },
    opp: {
      x: opp.x,
      y: opp.y,
      yVelocity: opp.yVelocity,
      state: opp.state,
      isPlayer2: !!opp.isPlayer2,
    },
    ball: {
      x: ball.x,
      y: ball.y,
      xVelocity: ball.xVelocity,
      yVelocity: ball.yVelocity,
      expectedLandingX: ball.expectedLandingPointX,
      isPowerHit: !!ball.isPowerHit,
    },
  };
}
