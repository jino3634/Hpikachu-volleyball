'use strict';

/**
 * Build a stable observation object from current physics state.
 * Keep this small and consistent: it will be used by evolved policies.
 *
 * @param {import('../physics.js').PikaPhysics} physics
 * @param {1|2} playerIndex
 */
export function makeObservation(physics, playerIndex) {
  const me = (playerIndex === 1) ? physics.player1 : physics.player2;
  const opp = (playerIndex === 1) ? physics.player2 : physics.player1;
  const b = physics.ball;

  return {
    me: {
      isPlayer2: !!me.isPlayer2,
      x: me.x,
      y: me.y,
      xVelocity: me.xVelocity ?? 0,
      yVelocity: me.yVelocity ?? 0,
      state: me.state ?? 0,
    },
    opp: {
      isPlayer2: !!opp.isPlayer2,
      x: opp.x,
      y: opp.y,
      xVelocity: opp.xVelocity ?? 0,
      yVelocity: opp.yVelocity ?? 0,
      state: opp.state ?? 0,
    },
    ball: {
      x: b.x,
      y: b.y,
      xVelocity: b.xVelocity,
      yVelocity: b.yVelocity,
      expectedLandingX: b.expectedLandingPointX,
      isPowerHit: !!b.isPowerHit,
    },
  };
}
