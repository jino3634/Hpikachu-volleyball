// trainer.js
'use strict';

import { OnePointEpisodeRunner } from './episode_runner.js';
import { IndexedDBStorage } from '../storage/storage_indexeddb.js';
import { PpoPolicyV1 } from './ppo_policy_v1.js';
import { PpoPolicyAgentV1 } from './ppo_agent_v1.js';

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

    // warmup (imitation) dataset collection
    this.warmup = {
      done: false,
      trained: false,
      targetSamples: 1000000,
      trainEpochs: 2,
      trainBatch: 256,
    };

    this.running = false;
    this.graduated = false;
    // training mode: PHASE1 (builtin) -> PHASE2 (self-play)
    this.mode = 'PHASE1';
    // Phase1: track best margin (p1Score - p2Score) for current snapshot
    this.phase1BestMargin = -999;

    
    // batch update (N points accumulate)
    this.batchPoints = 64;
    this.maxBufferPoints = 256;
    this.pointBuffer = [];
    this.flushCount = 0;
    this.bufferSkipped = 0;

// global stats
    this.totalEpisodes = 0;
    this.totalWins = 0;
    this.totalLosses = 0;

    // recent-1000 point winrate (persisted)
    this.recent1000 = { idx: 0, filled: 0, wins: 0, buf: new Array(1000).fill(0) };
    this._recent1000Dirty = 0;

    // set tracking
    this.currentSet = {
      p1: 0,
      p2: 0,
      index: 1,
    };
    this.consecutiveSetWins = 0;

    this.lastResult = null;

    this._runner = null;

    // policy (tuple outputs)
    this.policy = new PpoPolicyV1({
      featureLen: 16,
      learningRate: 0.0003,
      hidden1: 64,
      hidden2: 64,
      initStd: 0.02,
      clipEps: 0.2,
      vfCoef: 0.5,
      gamma: 0.995,
      gaeLambda: 0.95,
    });

    // PPO needs stochastic sampling during training; keep a little epsilon exploration.
    this.agent = new PpoPolicyAgentV1(this.policy, {
      playerIndex: this.learningPlayer,
      deterministic: false,
      epsilon: 0.02,
    });

    // PPO rollout settings
    this.rollout = [];
    this.rolloutSteps = 2048;
    this.minRolloutToUpdate = 512; // flush threshold for remaining rollout steps

    this.ppoEpochs = 4;
    this.ppoMinibatch = 256;
  }

  async init() {
    await this.storage.init();

    // recent1000 checkpoint
    const recent1000 = await this.storage.getCheckpoint('recent1000');
    if (recent1000 && typeof recent1000 === 'object') {
      const filled = Math.max(0, Math.min(1000, (recent1000.filled ?? 0) | 0));
      const idx = Math.max(0, Math.min(999, (recent1000.idx ?? 0) | 0));
      const wins = Math.max(0, (recent1000.wins ?? 0) | 0);
      const buf = Array.isArray(recent1000.buf) ? recent1000.buf.slice(0, 1000).map((x) => (x ? 1 : 0)) : null;
      if (buf && buf.length === 1000) {
        this.recent1000 = { idx, filled, wins, buf };
      }
    }

    // warmup checkpoint
    const warmup = await this.storage.getCheckpoint('warmup');
    if (warmup) {
      this.warmup.done = !!warmup.done;
      if (warmup.targetSamples) this.warmup.targetSamples = warmup.targetSamples | 0;
    } else {
      await this.storage.setCheckpoint('warmup', {
        done: false,
        targetSamples: this.warmup.targetSamples,
        createdAt: Date.now(),
      });
    }

    // warmup training checkpoint
    const warmupTrain = await this.storage.getCheckpoint('warmup_train');
    if (warmupTrain) {
      this.warmup.trained = !!warmupTrain.trained;
      if (warmupTrain.trainEpochs) this.warmup.trainEpochs = warmupTrain.trainEpochs | 0;
      if (warmupTrain.trainBatch) this.warmup.trainBatch = warmupTrain.trainBatch | 0;
    } else {
      await this.storage.setCheckpoint('warmup_train', {
        trained: false,
        trainEpochs: this.warmup.trainEpochs,
        trainBatch: this.warmup.trainBatch,
        createdAt: Date.now(),
      });
    }

    // stats checkpoint
    const stats = await this.storage.getCheckpoint('train_stats');
    if (stats) {
      this.totalEpisodes = stats.totalEpisodes | 0;
      this.totalWins = stats.totalWins | 0;
      this.totalLosses = stats.totalLosses | 0;
      this.graduated = !!stats.graduated;
      this.consecutiveSetWins = stats.consecutiveSetWins | 0;
      this.currentSet = stats.currentSet ?? this.currentSet;
      this.mode = stats.mode ?? (this.graduated ? 'PHASE2' : 'PHASE1');
      this.phase1BestMargin = (stats.phase1BestMargin ?? this.phase1BestMargin) | 0;
    } else {
      await this.storage.setCheckpoint('train_stats', {
        totalEpisodes: 0,
        totalWins: 0,
        totalLosses: 0,
        graduated: false,
        consecutiveSetWins: 0,
        mode: 'PHASE1',
        phase1BestMargin: this.phase1BestMargin,
        currentSet: this.currentSet,
      });
    }

    // model_state
    const model = await this.storage.getCheckpoint('model_state');
    if (model && (model.kind === 'tuple_policy_mlp_v1' || model.kind === 'tuple_policy_v1' || model.kind === 'ppo_policy_v1')) {
      // tuple_policy_v1 (old linear) is still loadable (it will be wrapped as an identity-trunk MLP)
      this.policy.loadState(model);
    } else {
      // 최초 생성
      await this.storage.setCheckpoint('model_state', this.policy.saveState());
      await this._maybePersistRecent1000(true);
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
      last1000Count: this.recent1000?.filled ?? 0,
      last1000Wins: this.recent1000?.wins ?? 0,
      last1000Winrate: (this.recent1000 && (this.recent1000.filled > 0)) ? (this.recent1000.wins / this.recent1000.filled) : 0,
      pointsPerTick: this.pointsPerTick,
      tickDelayMs: this.tickDelayMs,
      consecutiveSetWins: this.consecutiveSetWins,
      setWinTarget: this.setWinTarget,
      consecutiveSetWinsToGraduate: this.consecutiveSetWinsToGraduate,
      currentSet: this.currentSet,
      lastResult: this.lastResult,
    };
  }

  /**
   * Export a warmup snapshot (model weights + warmup checkpoints) so you can skip warmup next runs.
   * Returned object is JSON-serializable.
   */
  async exportWarmupSnapshot() {
    const warmup = await this.storage.getCheckpoint('warmup');
    const warmupTrain = await this.storage.getCheckpoint('warmup_train');
    return {
      kind: 'warmup_snapshot_v1',
      createdAt: Date.now(),
      learningPlayer: this.learningPlayer,
      modelState: this.policy.saveState(),
      warmup: warmup ?? null,
      warmupTrain: warmupTrain ?? null,
    };
  }

  /**
   * Import a warmup snapshot produced by exportWarmupSnapshot().
   * This overwrites current in-memory model + persisted checkpoints.
   */
  async importWarmupSnapshot(snapshot) {
    if (!snapshot || snapshot.kind !== 'warmup_snapshot_v1') {
      throw new Error('Invalid warmup snapshot');
    }

    if (snapshot.modelState) {
      this.policy.loadState(snapshot.modelState);
      await this.storage.setCheckpoint('model_state', this.policy.saveState());
    }

    if (snapshot.warmup) {
      await this.storage.setCheckpoint('warmup', snapshot.warmup);
      this.warmup.done = !!snapshot.warmup.done;
      if (snapshot.warmup.targetSamples) this.warmup.targetSamples = snapshot.warmup.targetSamples | 0;
    }

    if (snapshot.warmupTrain) {
      await this.storage.setCheckpoint('warmup_train', snapshot.warmupTrain);
      this.warmup.trained = !!snapshot.warmupTrain.trained;
      if (snapshot.warmupTrain.trainEpochs) this.warmup.trainEpochs = snapshot.warmupTrain.trainEpochs | 0;
      if (snapshot.warmupTrain.trainBatch) this.warmup.trainBatch = snapshot.warmupTrain.trainBatch | 0;
    }
  }

  async _saveStats() {
    await this.storage.setCheckpoint('train_stats', {
      totalEpisodes: this.totalEpisodes,
      totalWins: this.totalWins,
      totalLosses: this.totalLosses,
      graduated: this.graduated,
      consecutiveSetWins: this.consecutiveSetWins,
      currentSet: this.currentSet,
      mode: this.mode,
      phase1BestMargin: this.phase1BestMargin,
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

  /**
   * Phase0: collect imitation samples from builtin AI (frame-level).
   * We run builtin vs builtin and store (obsP1, inputP1) pairs.
   */
  async _collectWarmupSamples() {
    const target = Math.max(1000, this.warmup.targetSamples | 0);
    const already = await this.storage.countImitationSamples();
    if (already >= target) {
      this.warmup.done = true;
      await this.storage.setCheckpoint('warmup', { done: true, targetSamples: target, updatedAt: Date.now() });
      console.log(`[WARMUP] dataset already ready: samples=${already}`);
      return;
    }

    console.log(`[WARMUP] collecting imitation samples... target=${target}, current=${already}`);

    const gameAny = /** @type {any} */ (this.game);
    // Save previous control config
    const prev = {
      externalEnabledP1: !!this.game.externalEnabledP1,
      externalEnabledP2: !!this.game.externalEnabledP2,
      isComp1: !!this.game.physics?.player1?.isComputer,
      isComp2: !!this.game.physics?.player2?.isComputer,
      onAfterPhysicsFrame: gameAny.onAfterPhysicsFrame,
    };

    // builtin vs builtin
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(false, false);
    } else {
      this.game.externalEnabledP1 = false;
      this.game.externalEnabledP2 = false;
      if (this.game.physics?.player1) this.game.physics.player1.isComputer = true;
      if (this.game.physics?.player2) this.game.physics.player2.isComputer = true;
    }

    // Detach agents if possible
    if (typeof this.game.setAgents === 'function') {
      this.game.setAgents(null, null);
    } else {
      this.game.agent1 = null;
      this.game.agent2 = null;
    }

    let collected = already;
    let frames = 0;
    const buffer = [];

    gameAny.onAfterPhysicsFrame = (info) => {
      // collect only during round-like state to avoid menu noise
      if (!info || info.stateName !== 'round') return;
      if (!info.obsP1) return;

      buffer.push({
        obs: info.obsP1,
        label: info.inputP1,
        createdAt: Date.now(),
      });
      collected++;
    };

    // Run frames until we have enough samples
    while (collected < target) {
      this.game.stepLogic();
      frames++;

      if (buffer.length >= 512) {
        const chunk = buffer.splice(0, buffer.length);
        await this.storage.appendImitationSamples(chunk);
      }

      if (frames % 2000 === 0) {
        if (buffer.length > 0) {
          const chunk = buffer.splice(0, buffer.length);
          await this.storage.appendImitationSamples(chunk);
        }
        const dbCount = await this.storage.countImitationSamples();
        console.log(`[WARMUP] frames=${frames} samples=${dbCount}/${target}`);
        await sleep(0);
      }
    }

    if (buffer.length > 0) {
      await this.storage.appendImitationSamples(buffer);
      buffer.length = 0;
    }

    const finalCount = await this.storage.countImitationSamples();
    console.log(`[WARMUP] done. frames=${frames} samples=${finalCount}`);

    this.warmup.done = true;
    await this.storage.setCheckpoint('warmup', { done: true, targetSamples: target, updatedAt: Date.now() });

    // Restore previous control config
    gameAny.onAfterPhysicsFrame = prev.onAfterPhysicsFrame;
    this.game.externalEnabledP1 = prev.externalEnabledP1;
    this.game.externalEnabledP2 = prev.externalEnabledP2;
    if (this.game.physics?.player1) this.game.physics.player1.isComputer = prev.isComp1;
    if (this.game.physics?.player2) this.game.physics.player2.isComputer = prev.isComp2;
  }

  /**
   * Phase0: train policy to imitate builtin inputs (supervised).
   * Uses samples stored in IndexedDB (imit_samples).
   */
  async _trainWarmupImitation() {
    if (this.warmup.trained) return;

    const total = await this.storage.countImitationSamples();
    if (total <= 0) {
      console.warn('[WARMUP-TRAIN] no imitation samples; skip');
      this.warmup.trained = true;
      await this.storage.setCheckpoint('warmup_train', { trained: true, updatedAt: Date.now() });
      return;
    }

    const epochs = Math.max(1, this.warmup.trainEpochs | 0);
    const batchSize = Math.max(8, this.warmup.trainBatch | 0);

    // Load all samples (newest-first) then shuffle indices
    const samples = await this.storage.listImitationSamples({ limit: 0, offset: 0 });
    // samples are newest-first; reverse to get chronological (not required, but stable)
    samples.reverse();

    const N = samples.length;
    console.log(`[WARMUP-TRAIN] start. samples=${N}, epochs=${epochs}, batch=${batchSize}`);

    let globalStep = 0;

    for (let ep = 0; ep < epochs; ep++) {
      // shuffle indices
      const idx = new Array(N);
      for (let i = 0; i < N; i++) idx[i] = i;
      for (let i = N - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }

      let lossSum = 0;
      let nSeen = 0;
      let accX = 0, accY = 0, accP = 0;

      for (let i = 0; i < N; i += batchSize) {
        const end = Math.min(N, i + batchSize);

        for (let k = i; k < end; k++) {
          const s = samples[idx[k]];
          const feat = this.policy.buildFeatures(s.obs, this.learningPlayer);
          let r;
          if (typeof this.policy.updateImitationSample === 'function') {
            r = this.policy.updateImitationSample(s.obs, this.learningPlayer, s.label);
          } else {
            r = this.policy.updateImitation(feat, s.label);
          }

          lossSum += r.loss;
          nSeen++;

          // label classes
          const lx = (s.label.xDirection ?? 0) | 0;
          const ly = (s.label.yDirection ?? 0) | 0;
          const lp = (s.label.powerHit ?? 0) ? 1 : 0;

          // map predicted classes -> tuple for quick acc
          const px = (r.ax === 0 ? -1 : (r.ax === 2 ? 1 : 0));
          const py = (r.ay === 0 ? -1 : (r.ay === 2 ? 1 : 0));
          accX += (px === (lx < 0 ? -1 : (lx > 0 ? 1 : 0))) ? 1 : 0;
          accY += (py === (ly < 0 ? -1 : (ly > 0 ? 1 : 0))) ? 1 : 0;
          accP += (r.ap === lp) ? 1 : 0;
        }

        globalStep++;
        if (globalStep % 50 === 0) {
          const meanLoss = lossSum / Math.max(1, nSeen);
          const ax = accX / Math.max(1, nSeen);
          const ay = accY / Math.max(1, nSeen);
          const ap = accP / Math.max(1, nSeen);
          console.log(`[WARMUP-TRAIN] ep=${ep + 1}/${epochs} step=${globalStep} loss=${meanLoss.toFixed(3)} acc(x,y,p)=${ax.toFixed(3)},${ay.toFixed(3)},${ap.toFixed(3)}`);
        }
      }

      const meanLoss = lossSum / Math.max(1, nSeen);
      const ax = accX / Math.max(1, nSeen);
      const ay = accY / Math.max(1, nSeen);
      const ap = accP / Math.max(1, nSeen);
      console.log(`[WARMUP-TRAIN] epoch done ${ep + 1}/${epochs}. loss=${meanLoss.toFixed(3)} acc(x,y,p)=${ax.toFixed(3)},${ay.toFixed(3)},${ap.toFixed(3)}`);
    }

    // Save updated model state
    await this.storage.setCheckpoint('model_state', this.policy.saveState());

    this.warmup.trained = true;
    await this.storage.setCheckpoint('warmup_train', { trained: true, trainEpochs: epochs, trainBatch: batchSize, updatedAt: Date.now() });

    console.log('[WARMUP-TRAIN] done. model_state saved.');
  }



  async start() {
    if (this.running) return;

    // Phase0: imitation dataset collection (builtin vs builtin)
    if (!this.warmup.done) {
      await this._collectWarmupSamples();
    }
    // Phase0.5: imitation training (supervised)
    if (this.warmup.done && !this.warmup.trained) {
      await this._trainWarmupImitation();
    }

    if (this.graduated) return;

    this.running = true;

    // 다시 한번 확정
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(true, false);
    }

    while (this.running && !this.graduated) {
      for (let i = 0; i < this.pointsPerTick; i++) {
        // 한 포인트 실행
        const res = await this._runner.runOnePoint();
        this.lastResult = res;

        if (!res) {
          console.warn('[TRAIN] runOnePoint returned null/undefined');
          continue;
        }

        // episode 저장(가능한 경우만)
        if (res.episode) {
          await this.storage.appendEpisode(res.episode);
        } else {
          console.warn('[TRAIN] missing episode; skip save/learn', { loseReason: res.loseReason });
        }

        // point replay 저장(최근 10개 + 최근 승리 3개 유지)
        const replay = this._buildPointReplay(res);

        // recent 10
        await this.storage.appendReplay(replay);
        await this.storage.pruneReplays(10);

        // win 3 (별도 슬롯: win: 접두사로 중복 저장)
        const isWinPoint = (replay.ok && replay.scoredBy === this.learningPlayer);
        if (isWinPoint) {
          const winReplay = {
            ...replay,
            id: `win:${replay.id}`,
            baseId: replay.id,
            bucket: 'win',
          };
          await this.storage.appendReplay(winReplay);
          if (this.storage.pruneWinReplays) {
            await this.storage.pruneWinReplays(3);
          }
        }

        // PPO rollout collection + update
        const canLearn =
          res.ok === true &&
          res.frames > 0 &&
          res.loseReason !== 'EXCEPTION' &&
          !!res.episode;

        if (canLearn) {
          const transitions = res.episode.transitions ?? [];
          for (const tr of transitions) {
            const info = tr.info ?? {};
            // For PPO we need old log-prob and value from the moment we acted.
            if (typeof info.logp !== 'number' || typeof info.value !== 'number') continue;

            this.rollout.push({
              obs: tr.obs ?? null,
              action: tr.action ?? null,
              reward: Number(tr.reward ?? 0),
              done: !!tr.done,
              oldLogp: Number(info.logp ?? 0),
              value: Number(info.value ?? 0),
              playerIndex: this.learningPlayer,
            });
          }

          // update when enough rollout steps are collected
          while (this.rollout.length >= this.rolloutSteps) {
            const batch = this.rollout.splice(0, this.rolloutSteps);

            // compute GAE + returns
            this.policy.computeGAE(batch);

            const stats = this.policy.ppoUpdate(batch, {
              epochs: this.ppoEpochs,
              minibatch: this.ppoMinibatch,
            });

            console.log(`[PPO] steps=${stats.steps} updates=${stats.updates} approxKL=${stats.approxKl.toFixed(6)}`);
          }
        } else {
          this.bufferSkipped++;
        }

        // global stats
        this.totalEpisodes++;
        const isWin = (res.ok && res.scoredBy === this.learningPlayer);
        if (isWin) this.totalWins++;
        else if (res.ok) this.totalLosses++;

        // recent1000 point stats
        if (res.ok) {
          this._recordRecent1000(isWin);
          this._maybePersistRecent1000(false);
        }

        // 🔍 학습 진행 확인용 단일 로그
        console.log(
        `[TRAIN] ep=${this.totalEpisodes} W=${this.totalWins} L=${this.totalLosses} ` +
        `WR=${(this.totalWins / Math.max(1, this.totalEpisodes)).toFixed(3)} ` +
        `last={scoredBy:${res.scoredBy}, loser:${res.loser}, frames:${res.frames}}`
        );

        // set score 업데이트
        if (res.ok && (res.scoredBy === 1 || res.scoredBy === 2)) {
          if (res.scoredBy === 1) this.currentSet.p1++;
          else this.currentSet.p2++;
        }

        // 세트 종료 처리
        if (this._isSetFinished()) {
          const p1Won = this._didP1WinSet();

          // Phase1: margin-based "current" snapshot rule
          const margin = (this.currentSet.p1 | 0) - (this.currentSet.p2 | 0);
          if (margin > (this.phase1BestMargin | 0)) {
            this.phase1BestMargin = margin;
            // save immediately so the best margin snapshot is persisted
            await this.storage.setCheckpoint('model_state', this.policy.saveState());
            await this._saveStats();
            console.log(`[P1] new bestMargin=${this.phase1BestMargin} (score ${this.currentSet.p1}-${this.currentSet.p2}) -> saved current`);
          }

          if (p1Won) this.consecutiveSetWins++;
          else this.consecutiveSetWins = 0;

          // 졸업 조건: 3세트 연속 승리 -> Phase2 준비 완료(아직 Phase2는 미구현)
          if (this.consecutiveSetWins >= this.consecutiveSetWinsToGraduate) {
            this.graduated = true;
            this.mode = 'PHASE2';
            this.running = false;
            // flush remaining rollout buffer before finalize
            await this._flushRollout(true);
            await this._saveStats();
            console.log('[P1->P2] graduated via 3 consecutive set wins; mode set to PHASE2');
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

    // flush remaining rollout steps (PPO)
    await this._flushRollout(true);

    // 종료 시점에도 저장
    await this._saveStats();
    await this.storage.setCheckpoint('model_state', this.policy.saveState());
  }

_flushBatch(allRemaining = false) {
  const n = allRemaining ? this.pointBuffer.length : this.batchPoints;
  if (!n || n <= 0) return;

  const batch = this.pointBuffer.splice(0, n);
  let updatedTotal = 0;
  for (const item of batch) {
    try {
      const out = this.policy.learnFromEpisode(item.episode);
      if (out && typeof out.updated === 'number') updatedTotal += out.updated;
    } catch (e) {
      console.warn('[BATCH] learnFromEpisode failed; skip item', e);
    }
  }

  this.flushCount++;
  console.log(`[BATCH] flush=${this.flushCount} size=${batch.length} updated=${updatedTotal} skipped=${this.bufferSkipped}`);
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
      this.mode = stats.mode ?? (this.graduated ? 'PHASE2' : 'PHASE1');
      this.phase1BestMargin = (stats.phase1BestMargin ?? this.phase1BestMargin) | 0;
    }

    // reload model
    const model = await this.storage.getCheckpoint('model_state');
    if (model && (model.kind === 'tuple_policy_mlp_v1' || model.kind === 'tuple_policy_v1' || model.kind === 'ppo_policy_v1')) {
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
    // training mode: PHASE1 (builtin) -> PHASE2 (self-play)
    this.mode = 'PHASE1';
    // Phase1: track best margin (p1Score - p2Score) for current snapshot
    this.phase1BestMargin = -999;
    this.consecutiveSetWins = 0;
    this.currentSet = { p1: 0, p2: 0, index: 1 };

    await this.storage.setCheckpoint('model_state', this.policy.saveState());
    await this._saveStats();
  }
    
  _recordRecent1000(isWin) {
    const s = this.recent1000;
    if (!s || !s.buf) return;

    const idx = s.idx | 0;
    const old = s.buf[idx] ? 1 : 0;
    const nw = isWin ? 1 : 0;

    if (old !== nw) {
      s.wins = Math.max(0, (s.wins - old + nw) | 0);
      s.buf[idx] = nw;
    }

    s.idx = (idx + 1) % 1000;
    s.filled = Math.min(1000, (s.filled + 1) | 0);

    this._recent1000Dirty = (this._recent1000Dirty | 0) + 1;
  }

  async _maybePersistRecent1000(force) {
    if (!this.storage?.setCheckpoint) return;
    const dirty = this._recent1000Dirty | 0;
    if (!force && dirty < 25) return;

    this._recent1000Dirty = 0;
    try {
      await this.storage.setCheckpoint('recent1000', {
        idx: this.recent1000.idx,
        filled: this.recent1000.filled,
        wins: this.recent1000.wins,
        buf: this.recent1000.buf,
        updatedAt: Date.now(),
      });
    } catch (e) {
      // ignore persistence errors (training can continue)
    }
  }

_buildPointReplay(res) {
    const now = Date.now();
    const id = `rp_${now}_${Math.random().toString(16).slice(2)}`;

    const opponentType = (this.game?.externalEnabledP2) ? 'external' : 'builtin';

    // trace를 "재생 친화적인" compact 포맷으로 줄임
    const trace = Array.isArray(res.trace) ? res.trace.map((t) => {
      const obsP1 = t?.obs?.p1 ?? null;
      const obsP2 = t?.obs?.p2 ?? null;

      // p1 관측 기준으로 p1/p2 상태를 잡는 게 가장 안정적
      const p1 = obsP1?.me ?? obsP2?.opp ?? null;
      const p2 = obsP1?.opp ?? obsP2?.me ?? null;
      const ball = t?.obs?.ball ?? obsP1?.ball ?? obsP2?.ball ?? null;

      // obs_patch(정규화 관측)에서도 리플레이는 픽셀 좌표(raw)를 우선 사용
      const p1Pix = obsP1?.raw?.me ?? obsP2?.raw?.opp ?? p1;
      const p2Pix = obsP1?.raw?.opp ?? obsP2?.raw?.me ?? p2;
      // obs_patch에서는 t.obs.ball이 정규화일 수 있으니 raw.ball을 최우선 사용
      const ballPix = obsP1?.raw?.ball ?? obsP2?.raw?.ball ?? t?.obs?.ball ?? ball;

      return {
        frame: ((t.frame ?? t.t ?? 0) | 0),
        a1: (t.actionP1 ?? 0) | 0,
        a2: (t.actionP2 ?? 0) | 0,
        p1: p1Pix ? { x: p1Pix.x, y: p1Pix.y, yV: p1Pix.yV } : null,
        p2: p2Pix ? { x: p2Pix.x, y: p2Pix.y, yV: p2Pix.yV } : null,
        ball: ballPix ? { x: ballPix.x, y: ballPix.y, xV: ballPix.xV, yV: ballPix.yV, isPowerHit: ballPix.isPowerHit } : null,
        scores: obsP1?.scores ?? obsP2?.scores ?? null,
        serve: obsP1?.isPlayer2Serve ?? obsP2?.isPlayer2Serve ?? null,
        state: t.stateName ?? null,
      };
    }) : [];

    return {
      id,
      createdAt: now,
      type: 'point_v1',
      ok: !!res.ok,
      learningPlayer: this.learningPlayer,
      opponentType,
      scoredBy: res.scoredBy ?? 0,
      loser: res.loser ?? 0,
      loseReason: res.loseReason ?? 'UNKNOWN',
      frames: res.frames ?? 0,
      trace,
    };
  }

  async _flushRollout(force = false) {
    // PPO rollout updates are performed in start() loop when rolloutSteps are met.
    // This helper flushes remaining partial rollout (e.g., on stop/graduation).
    const threshold = force ? this.minRolloutToUpdate : this.rolloutSteps;
    while (this.rollout.length >= threshold) {
      const take = force ? this.rollout.length : this.rolloutSteps;
      const batch = this.rollout.splice(0, take);
      this.policy.computeGAE(batch);
      const stats = this.policy.ppoUpdate(batch, {
        epochs: this.ppoEpochs,
        minibatch: this.ppoMinibatch,
      });
      console.log(`[PPO] flush steps=${stats.steps} updates=${stats.updates} approxKL=${stats.approxKl.toFixed(6)}`);
      if (!force) break;
      if (this.rollout.length < this.minRolloutToUpdate) break;
    }
  }
}
