// rl_reward.js
'use strict';

/**
 * Sparse reward:
 * - 득점하면 +1
 * - 실점하면 -1
 * - 그 외 0
 *
 * 여기서 "학습 플레이어" 기준으로 보상을 준다.
 *
 * @param {Object|null} roundEvents  game._lastRoundEvents (episode_runner가 읽는 그거)
 * @param {1|2} learningPlayer
 * @returns {number} reward (-1,0,+1)
 */
export function computeSparseReward(roundEvents, learningPlayer) {
  if (!roundEvents) return 0;

  const scored = roundEvents.scored; // 0|1|2 in your code
  if (scored !== 1 && scored !== 2) return 0;

  // learningPlayer가 득점했으면 +1, 아니면 -1
  return scored === learningPlayer ? 1 : -1;
}
