// ppo_agent_v1.js
'use strict';

import { PpoPolicyV1 } from './ppo_policy_v1.js';

/**
 * Game engine calls agent.chooseInput(obs, playerIndex, game).
 * We return a tuple input, and keep last decision info for the runner to log.
 */
export class PpoPolicyAgentV1 {
  /**
   * @param {PpoPolicyV1} policy
   * @param {{playerIndex?:1|2, deterministic?:boolean, epsilon?:number}} opts
   */
  constructor(policy, opts = {}) {
    this.policy = policy;
    this.playerIndex = (opts.playerIndex ?? 1);
    this.deterministic = !!opts.deterministic;
    this.epsilon = Number(opts.epsilon ?? 0); // exploration during training
    /** @type {any|null} */
    this.lastDecision = null;
  }

  /**
   * @param {any} obs
   * @param {1|2} playerIndex
   */
  chooseInput(obs, playerIndex) {
    const pi = (playerIndex ?? this.playerIndex);
    const out = this.policy.act(obs, pi, { deterministic: this.deterministic, epsilon: this.epsilon });
    this.lastDecision = out;
    return out.action;
  }
}
