/**
 * Weighted heuristic policy (genome = weights + thresholds).
 *
 * This is a strong starting point for evolution because it's:
 * - fast
 * - explainable
 * - easy to mutate
 */
'use strict';

/**
 * @returns {any} default genome
 */
export function defaultGenome() {
  return {
    // movement
    wMoveToLanding: 1.0,
    moveDeadband: 10,

    // jump / dive tendencies
    jumpBallY: 140,
    jumpBallDX: 60,

    // power hit tendencies
    powerBallY: 200,
    powerBallDX: 40,

    // tiny randomness (mutation can change this)
    epsilon: 0.0,
  };
}

/**
 * Choose action from observation.
 * @param {any} obs
 * @param {any} genome
 * @returns {{xDir:-1|0|1, yDir:-1|0|1, powerHit:0|1}}
 */
export function chooseAction(obs, genome) {
  const g = genome || defaultGenome();

  const meX = Number(obs?.me?.x ?? 0);
  const meY = Number(obs?.me?.y ?? 0);
  const meState = Number(obs?.me?.state ?? 0);

  const bX = Number(obs?.ball?.x ?? 0);
  const bY = Number(obs?.ball?.y ?? 0);

  const landingX = Number(obs?.ball?.expectedLandingX ?? bX);

  let xDir = 0;
  const dx = landingX - meX;
  const db = Math.max(0, Number(g.moveDeadband ?? 10));
  if (dx > db) xDir = 1;
  else if (dx < -db) xDir = -1;

  // jump if ball is above and near (simple defense/offense starter)
  let yDir = 0;
  const jumpBallY = Number(g.jumpBallY ?? 140);
  const jumpBallDX = Number(g.jumpBallDX ?? 60);
  const nearBall = Math.abs(bX - meX) <= jumpBallDX;
  const onGround = (meY >= 244);

  if (onGround && bY < jumpBallY && nearBall) {
    yDir = -1;
  }

  // power hit one-shot
  let powerHit = 0;
  const powerBallY = Number(g.powerBallY ?? 200);
  const powerBallDX = Number(g.powerBallDX ?? 40);

  const canPower = (meState === 1 || meState === 2) && Math.abs(bX - meX) <= powerBallDX && bY <= powerBallY;
  if (canPower) powerHit = 1;

  // optional epsilon randomness for exploration
  const eps = Math.max(0, Math.min(0.5, Number(g.epsilon ?? 0)));
  if (eps > 0 && Math.random() < eps) {
    // random small perturbation
    const r = (Math.random() * 3) | 0;
    xDir = (r === 0) ? -1 : (r === 1) ? 0 : 1;
  }

  return { xDir: /** @type {-1|0|1} */ (xDir), yDir: /** @type {-1|0|1} */ (yDir), powerHit: /** @type {0|1} */ (powerHit) };
}
