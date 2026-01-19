// rl_schema.js
'use strict';

/**
 * Action space (확정)
 * - 기존 이동/점프(0~5)
 * - 파워 계열(6~13): 너가 준 조합 반영 + "아래" 포함
 *
 * 주의:
 * - powerHit는 실제로는 1프레임 트리거(이미 controller에서 처리중)
 * - 여기서는 "의도"를 actionId로만 표현
 */
export const Action = Object.freeze({
  IDLE: 0,
  LEFT: 1,
  RIGHT: 2,
  JUMP: 3,
  JUMP_LEFT: 4,
  JUMP_RIGHT: 5,

  // power family
  POWER_NEUTRAL: 6,        // power only
  POWER_LEFT: 7,           // power + left
  POWER_RIGHT: 8,          // power + right
  POWER_UP: 9,             // power + up
  POWER_LEFT_UP: 10,       // power + left + up
  POWER_RIGHT_UP: 11,      // power + right + up
  POWER_LEFT_DOWN: 12,     // power + left + down
  POWER_RIGHT_DOWN: 13,    // power + right + down
});

/** @returns {boolean} */
export function isPowerAction(actionId) {
  return (actionId | 0) >= 6;
}

/**
 * 학습/저장 스키마 버전
 * - 저장된 데이터와 코드가 어긋날 수 있으니 반드시 박아둔다.
 */
export const SCHEMA_VERSION = 1;

/**
 * Transition (s, a, r, s', done) 스키마
 * @typedef {Object} Transition
 * @property {number} t                 // global frame counter inside episode
 * @property {Object|null} obs          // s  (player-centric observation; can be null in feat-only mode)
 * @property {Float32Array|null} feat   // compact feature vector for s (preferred)
 * @property {number|{xDirection:number,yDirection:number,powerHit:number}|null} action // a (Action id or input tuple)
 * @property {number} reward            // r  (-1,0,+1)
 * @property {Object|null} nextObs      // s' (optional; often null)
 * @property {Float32Array|null} nextFeat // compact feature vector for s'
 * @property {boolean} done             // terminal?
 * @property {Object|null} info         // debug info (optional)
 */

/**
 * Episode 스키마
 * @typedef {Object} Episode
 * @property {number} schemaVersion
 * @property {string} episodeId         // unique id
 * @property {number} startedAt         // Date.now()
 * @property {number} endedAt           // Date.now()
 * @property {number} frames            // number of transitions
 * @property {1|2} learningPlayer       // who is learning in this episode (너가 "2 -> 1" 했으니 기본 1)
 * @property {0|1|2} scoredBy           // 0 none/timeout, 1 p1, 2 p2
 * @property {0|1|2} loser              // 0 none/timeout, 1 p1, 2 p2
 * @property {string} loseReason        // debug string
 * @property {Transition[]} transitions
 */
