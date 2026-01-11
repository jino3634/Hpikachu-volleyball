// trainer.js
'use strict';

import { OnePointEpisodeRunner } from './episode_runner.js';
import { IndexedDBStorage } from '../storage/storage_indexeddb.js';
import { PolicyV1, PolicyAgentV1 } from './policy_v1.js';

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function downloadJSON(obj, filename) {
  const json = JSON.stringify(obj);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function pickJSONFile() {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = async () => {
      try {
        const file = input.files && input.files[0];
        if (!file) return resolve(null);
        const text = await file.text();
        resolve(JSON.parse(text));
      } catch (e) {
        reject(e);
      }
    };
    input.click();
  });
}

export class Trainer {
  /**
   * @param {import('../pikavolley.js').PikachuVolleyball} game
   * @param {{
   *   learningPlayer?: 1|2,
   *   pointsPerTick?: number,
   *   tickDelayMs?: number,
   *   autosaveEveryEpisodes?: number,
   *   setWinTarget?: number,
   *   consecutiveSetWinsToGraduate?: number
   * }} [opts]
   */
  constructor(game, opts = {}) {
    this.game = game;

    /** @type {1|2} */
    this.learningPlayer = (opts.learningPlayer ?? 1);

    this.pointsPerTick = Math.max(1, (opts.pointsPerTick ?? 1) | 0);
    this.tickDelayMs = Math.max(0, (opts.tickDelayMs ?? 0) | 0);
    this.autosaveEveryEpisodes = Math.max(1, (opts.autosaveEveryEpisodes ?? 50) | 0);

    // “15점 1세트”
    this.setWinTarget = Math.max(1, (opts.setWinTarget ?? 15) | 0);

    // “3세트 연속 승리하면 졸업”
    this.consecutiveSetWinsToGraduate = Math.max(1, (opts.consecutiveSetWinsToGraduate ?? 3) | 0);

    this.storage = new IndexedDBStorage();

    this.running = false;
    this.graduated = false;

    // global stats
    this.totalEpisodes = 0;
    this.totalWins = 0;
    this.totalLosses = 0;

    // set tracking
    this.currentSet = {
      p1: 0,
      p2: 0,
      index: 1,
    };
    this.consecutiveSetWins = 0;

    this.lastResult = null;

    this._runner = null;

    // policy
    this.policy = new PolicyV1({
      numActions: 10,
      learningRate: 0.0005,
      epsilon: 0.08,
      initStd: 0.01,
    });
    this.agent = new PolicyAgentV1(this.policy, { playerIndex: this.learningPlayer });
  }

  async init() {
    await this.storage.init();

    // stats checkpoint
    const stats = await this.storage.getCheckpoint('train_stats');
    if (stats) {
      this.totalEpisodes = stats.totalEpisodes | 0;
      this.totalWins = stats.totalWins | 0;
      this.totalLosses = stats.totalLosses | 0;
      this.graduated = !!stats.graduated;
      this.consecutiveSetWins = stats.consecutiveSetWins | 0;
      this.currentSet = stats.currentSet ?? this.currentSet;
    } else {
      await this.storage.setCheckpoint('train_stats', {
        totalEpisodes: 0,
        totalWins: 0,
        totalLosses: 0,
        graduated: false,
        consecutiveSetWins: 0,
        currentSet: this.currentSet,
      });
    }

    // model_state
    const model = await this.storage.getCheckpoint('model_state');
    if (model && model.kind === 'policy_v1') {
      this.policy.loadState(model);
    } else {
      // 최초 생성
      await this.storage.setCheckpoint('model_state', this.policy.saveState());
    }

    this._runner = new OnePointEpisodeRunner(this.game, {
      learningPlayer: this.learningPlayer,
    });

    // ✅ P1 external / P2 builtin 확정 + agent 연결
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(true, false); // P2 builtin
    } else {
      // fallback
      this.game.setControlMode('external');
      this.game.externalEnabledP1 = true;
      this.game.externalEnabledP2 = false;
      this.game.physics.player1.isComputer = false;
      this.game.physics.player2.isComputer = true;
    }

    if (typeof this.game.setAgents === 'function') {
      // P1만 외부 agent, P2는 builtin이라 agent 불필요
      this.game.setAgents(this.agent, null);
    } else {
      this.game.agent1 = this.agent;
      this.game.agent2 = null;
    }
  }

  status() {
    const total = (this.totalEpisodes | 0);
    const wins = (this.totalWins | 0);
    const losses = (this.totalLosses | 0);
    const winrate = total > 0 ? (wins / total) : 0;

    return {
      running: this.running,
      graduated: this.graduated,
      learningPlayer: this.learningPlayer,
      totalEpisodes: total,
      wins,
      losses,
      winrate,
      pointsPerTick: this.pointsPerTick,
      tickDelayMs: this.tickDelayMs,
      consecutiveSetWins: this.consecutiveSetWins,
      setWinTarget: this.setWinTarget,
      consecutiveSetWinsToGraduate: this.consecutiveSetWinsToGraduate,
      currentSet: this.currentSet,
      lastResult: this.lastResult,
    };
  }

  async _saveStats() {
    await this.storage.setCheckpoint('train_stats', {
      totalEpisodes: this.totalEpisodes,
      totalWins: this.totalWins,
      totalLosses: this.totalLosses,
      graduated: this.graduated,
      consecutiveSetWins: this.consecutiveSetWins,
      currentSet: this.currentSet,
      updatedAt: Date.now(),
    });
  }

  _resetSet() {
    this.currentSet = { p1: 0, p2: 0, index: (this.currentSet.index | 0) + 1 };
  }

  _isSetFinished() {
    return (this.currentSet.p1 >= this.setWinTarget) || (this.currentSet.p2 >= this.setWinTarget);
  }

  _didP1WinSet() {
    return this.currentSet.p1 >= this.setWinTarget;
  }

  async start() {
    if (this.running) return;
    if (this.graduated) return;

    this.running = true;

    // 다시 한번 확정
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(true, false);
    }

    while (this.running && !this.graduated) {
      for (let i = 0; i < this.pointsPerTick; i++) {
        // 한 포인트 실행
        const res = this._runner.runOnePoint();
        this.lastResult = res;

        // episode 저장(무조건)
        await this.storage.appendEpisode(res.episode);

        // policy 업데이트(승/패만)
        this.policy.learnFromEpisode(res.episode);

        // global stats
        this.totalEpisodes++;
        if (res.ok && res.scoredBy === this.learningPlayer) this.totalWins++;
        else if (res.ok) this.totalLosses++;

        // set score 업데이트
        if (res.ok && (res.scoredBy === 1 || res.scoredBy === 2)) {
          if (res.scoredBy === 1) this.currentSet.p1++;
          else this.currentSet.p2++;
        }

        // 세트 종료 처리
        if (this._isSetFinished()) {
          const p1Won = this._didP1WinSet();
          if (p1Won) this.consecutiveSetWins++;
          else this.consecutiveSetWins = 0;

          // 졸업 조건
          if (this.consecutiveSetWins >= this.consecutiveSetWinsToGraduate) {
            this.graduated = true;
            this.running = false;
          }

          // 다음 세트로
          this._resetSet();
        }

        // autosave: stats + model_state
        if ((this.totalEpisodes % this.autosaveEveryEpisodes) === 0) {
          await this._saveStats();
          await this.storage.setCheckpoint('model_state', this.policy.saveState());
        }
      }

      // UI 프리즈 방지
      await sleep(this.tickDelayMs > 0 ? this.tickDelayMs : 0);
    }

    // 종료 시점에도 저장
    await this._saveStats();
    await this.storage.setCheckpoint('model_state', this.policy.saveState());
  }

  stop() {
    this.running = false;
  }

  async exportToFile() {
    // 최신 model_state도 저장하고 export
    await this.storage.setCheckpoint('model_state', this.policy.saveState());
    await this._saveStats();

    const data = await this.storage.exportAll();
    const filename = `pika_rl_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    downloadJSON(data, filename);
  }

  async importFromFile({ merge = true } = {}) {
    const data = await pickJSONFile();
    if (!data) return;

    if (!merge) {
      this.stop();
      await this.storage.clearAll();
      await this.storage.init();
    }

    await this.storage.importAll(data);

    // reload stats
    const stats = await this.storage.getCheckpoint('train_stats');
    if (stats) {
      this.totalEpisodes = stats.totalEpisodes | 0;
      this.totalWins = stats.totalWins | 0;
      this.totalLosses = stats.totalLosses | 0;
      this.graduated = !!stats.graduated;
      this.consecutiveSetWins = stats.consecutiveSetWins | 0;
      this.currentSet = stats.currentSet ?? this.currentSet;
    }

    // reload model
    const model = await this.storage.getCheckpoint('model_state');
    if (model && model.kind === 'policy_v1') {
      this.policy.loadState(model);
    } else {
      await this.storage.setCheckpoint('model_state', this.policy.saveState());
    }
  }

  async clearAll() {
    this.stop();
    await this.storage.clearAll();
    await this.storage.init();

    this.totalEpisodes = 0;
    this.totalWins = 0;
    this.totalLosses = 0;

    this.graduated = false;
    this.consecutiveSetWins = 0;
    this.currentSet = { p1: 0, p2: 0, index: 1 };

    await this.storage.setCheckpoint('model_state', this.policy.saveState());
    await this._saveStats();
  }
}
