'use strict';

import { expectedLandingPointXWhenPowerHit } from '../physics.js';

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
  // Note: Player class in this codebase does not officially expose xVelocity.
  // Some forks add it. For type-safety under TS/JSDoc checking, read via any.
  const meAny = /** @type {any} */ (me);
  const oppAny = /** @type {any} */ (opp);
  const b = physics.ball;

  // Heuristic: only compute power landing candidates when ball is close enough.
  // (Computing these runs a mini-sim loop; keep it decision-frame only.)
  const dx = Math.abs((b.x ?? 0) - (me.x ?? 0));
  const dy = Math.abs((b.y ?? 0) - (me.y ?? 0));
  const canPower = (dx <= 90 && dy <= 90);

  /** @type {Float32Array|null} */
  let powerLandingX = null;
  if (canPower) {
    // index = (xDir+1)*3 + (yDir+1), xDir,yDir in {-1,0,1}
    // Perf: reuse a fixed buffer per-physics/player to avoid GC churn during evolution.
    const physAny = /** @type {any} */ (physics);
    const kName = (playerIndex === 1) ? '__evoPowerBufP1' : '__evoPowerBufP2';
    let buf = physAny[kName];
    if (!(buf instanceof Float32Array) || buf.length !== 9) {
      buf = new Float32Array(9);
      physAny[kName] = buf;
    }
    powerLandingX = buf;
    let k = 0;
    for (let xDir = -1; xDir <= 1; xDir++) {
      for (let yDir = -1; yDir <= 1; yDir++) {
        powerLandingX[k++] = expectedLandingPointXWhenPowerHit(xDir, yDir, b);
      }
    }
  }

  return {
    me: {
      x: me.x,
      y: me.y,
      xVelocity: (meAny.xVelocity ?? 0),
      yVelocity: me.yVelocity ?? 0,
      state: me.state ?? 0,
      isPlayer2: !!me.isPlayer2,
    },
    opp: {
      x: opp.x,
      y: opp.y,
      xVelocity: (oppAny.xVelocity ?? 0),
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
      canPower,
      powerLandingX, // Float32Array(9) or null
    },
  };
}
