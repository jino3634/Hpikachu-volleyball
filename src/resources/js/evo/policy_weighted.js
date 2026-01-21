'use strict';
import { GROUND_HALF_WIDTH, PLAYER_GROUND_Y } from '../physics.js';

/**
 * Simple weighted-rule policy for evolution.
 * Genome is a bag of weights/thresholds.
 *
 * 4C upgrade: score power candidates using predicted landing X for each (xDir,yDir).
 */

export function defaultGenome() {
  return {
    // defense
    wMoveToLanding: 1.0,
    wStayCenter: 0.15,

    // power / attack
    wPower: 0.25,          // base tendency to try power when possible
    wAttackFar: 0.90,      // prefer landing far from opponent
    wAttackCorner: 0.25,   // prefer landing near corners
    wPowerOnOppSide: 1.20, // penalize power that lands on my side

    // penalties
    wAvoidNet: 1.0,

    // additional control
    deadZoneX: 10,          // px: stop moving when within this distance
    netAvoidBand: 48,       // px: avoid landing near the net line when power-hitting
    minPowerScore: -0.10,   // require at least this score to actually power

    // thresholds
    jumpMinBallY: 110,
    powerMinBallY: 95,
    powerMaxDX: 90,
  };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** @param {number} x */
function norm01(x) { return clamp(x, 0, 1); }

export function chooseAction(obs, genome) {
  const g = genome || defaultGenome();
  const me = obs.me;
  const opp = obs.opp;
  const ball = obs.ball;

  // --- target positions ---
  const landingX = Number(ball.expectedLandingX ?? me.x);
  const centerX = me.isPlayer2
    ? (GROUND_HALF_WIDTH + (GROUND_HALF_WIDTH / 2))
    : (GROUND_HALF_WIDTH / 2);

  // ball side (left=1, right=2)
  const mySide = me.isPlayer2 ? 2 : 1;
  const ballSide = (ball.x < GROUND_HALF_WIDTH) ? 1 : 2;
  const danger = (ballSide === mySide);

  // desired X: landingX when danger, otherwise drift to center
  const desiredX = danger ? landingX : centerX;

  // move decision
  const dxToDesired = desiredX - me.x;
  let xDir = 0;
  const deadZoneX = Math.max(0, Number(g.deadZoneX ?? 8));
  if (Math.abs(dxToDesired) > deadZoneX) xDir = (dxToDesired > 0) ? 1 : -1;

  // jump heuristic
  const nearBall = (Math.abs(ball.x - me.x) <= 72);
  const ballAbove = (ball.y <= (g.jumpMinBallY || 110));
  const canJump = (me.state === 0) && (me.y >= PLAYER_GROUND_Y);

  let yDir = 0;
  if (canJump && nearBall && ballAbove) yDir = -1;

  // --- power candidate scoring (4C) ---
  // Only consider power when ball is near and above threshold.
  const wantPowerBase =
    nearBall &&
    (ball.y <= (g.powerMinBallY || 95)) &&
    (Math.abs(ball.x - me.x) <= (g.powerMaxDX || 90));

  if (wantPowerBase && ball && ball.canPower && ball.powerLandingX && ball.powerLandingX.length === 9) {
    const pLX = ball.powerLandingX;

    // Opponent side definition:
    // - P1(left) wants landingX > half
    // - P2(right) wants landingX < half
    const oppSideSign = me.isPlayer2 ? -1 : 1; // +1 means want larger-than-half, -1 means want smaller-than-half

    let bestScore = -1e9;
    let bestX = 0;
    let bestY = 0;

    for (let xi = 0; xi < 3; xi++) {
      for (let yi = 0; yi < 3; yi++) {
        const idx = xi * 3 + yi;
        const lx = Number(pLX[idx]);
        if (!Number.isFinite(lx)) continue;

        // score components
        const distToOpp = Math.abs(lx - opp.x) / GROUND_HALF_WIDTH; // ~0..2
        const farScore = distToOpp;

        // corner preference: nearer to left/right wall is better
        const distToLeft = lx / (GROUND_HALF_WIDTH * 2);
        const distToRight = 1 - distToLeft;
        const cornerScore = 1 - Math.min(distToLeft, distToRight) * 2; // 0 center -> 1 corner

        // must land on opponent side; otherwise penalize hard
        const isOnOppSide = (oppSideSign > 0) ? (lx > GROUND_HALF_WIDTH) : (lx < GROUND_HALF_WIDTH);
        const sidePenalty = isOnOppSide ? 0 : 1;

        // net risk: landing too close to the net is often a free ball / self-risk
        const netBand = Math.max(8, Number(g.netAvoidBand ?? 48));
        const netDist = Math.abs(lx - GROUND_HALF_WIDTH);
        const netPenalty = (netDist >= netBand) ? 0 : (1 - (netDist / netBand));

        // combine
        let s = 0;
        s += (g.wAttackFar || 0) * farScore;
        s += (g.wAttackCorner || 0) * cornerScore;
        s -= (g.wPowerOnOppSide || 0) * sidePenalty;
        s -= (g.wAvoidNet || 0) * netPenalty;

        // mild bias towards "trying power" if it isn't catastrophic
        s += (g.wPower || 0);

        if (s > bestScore) {
          bestScore = s;
          bestX = xi - 1; // -1,0,1
          bestY = yi - 1; // -1,0,1
        }
      }
    }

    // If best is not terrible, choose it.
    const minPowerScore = Number(g.minPowerScore ?? -0.25);
    if (bestScore > minPowerScore) {
      return {
        xDirection: /** @type {-1|0|1} */ (bestX),
        yDirection: /** @type {-1|0|1} */ (bestY),
        powerHit: /** @type {0|1} */ (1),
      };
    }
  }

  // fallback: no power, standard movement/jump only
  return {
    xDirection: /** @type {-1|0|1} */ (xDir),
    yDirection: /** @type {-1|0|1} */ (yDir),
    powerHit: /** @type {0|1} */ (0),
  };
}
