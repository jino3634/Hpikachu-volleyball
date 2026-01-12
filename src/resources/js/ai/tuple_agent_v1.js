// tuple_agent_v1.js
'use strict';

import { TuplePolicyV1 } from './tuple_policy_v1.js';

/**
 * Game engine prefers agent.chooseInput(obs, playerIndex, game).
 * This agent wraps TuplePolicyV1.
 */
export class TuplePolicyAgentV1 {
  /**
   * @param {TuplePolicyV1} policy
   * @param {{playerIndex?:1|2, deterministic?:boolean}} opts
   */
  constructor(policy, opts = {}) {
    this.policy = policy;
    this.playerIndex = (opts.playerIndex ?? 1);
    this.deterministic = !!opts.deterministic;
  }

  /**
   * @param {any} obs
   * @param {1|2} playerIndex
   */
  chooseInput(obs, playerIndex) {
    const pi = (playerIndex ?? this.playerIndex);
    if (this.deterministic) return this.policy.actDeterministic(obs, pi);
    return this.policy.actStochastic(obs, pi);
  }
}
