// trainer.js
'use strict';

import { OnePointEpisodeRunner } from './episode_runner.js';
import { IndexedDBStorage } from '../storage/storage_indexeddb.js';
import { PpoPolicyV1 } from './ppo_policy_v1.js';
import { PpoPolicyAgentV1 } from './ppo_agent_v1.js';
import { logDebug, downloadDebugLog, clearDebugLog } from './debug_log.js';

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
   *   consecutiveSetWinsToGraduate?: number,
   *   pbtEnabled?: boolean,
   *   disableWarmup?: boolean
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

    // ---- 학습 OFF 검증용 카운터 ----
    this.ppoFlushCount = 0;     // [PPO] flush 몇 번 했는지
    this.ppoFlushSteps = 0;     // flush로 소비한 step 합

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
      featureLen: 24,
      learningRate: 0.001,
      hidden1: 64,
      hidden2: 64,
      initStd: 0.02,
      clipEps: 0.2,
      vfCoef: 0.5,
      gamma: 0.995,
      gaeLambda: 0.95,

      // ✅ 단 하나의 주입 지점
      genome: {
        powerHitGate: {
          kFrames: 4,
          dxMarginPx: 6,
          dyMarginPx: 10,
        },
      },


    });

    // PPO needs stochastic sampling during training; keep a little epsilon exploration.
    this.agent = new PpoPolicyAgentV1(this.policy, {
      playerIndex: this.learningPlayer,
      deterministic: false,
      epsilon: 0.02,
    });

    // PPO rollout settings
    this.rollout = [];
    this.learnDiag = {
      episodes: 0,
      transitions: 0,
      // how many transitions were actually pushed into rollout buffer
      pushedSteps: 0,
      // how many steps were consumed into PPO batches
      consumedSteps: 0,
      skippedNoInfo: 0,
      skippedBadFields: 0,

      // episode_runner diag aggregate
      ep_framesTotal: 0,
      ep_framesCanActFalse: 0,
      ep_framesDecisionSampled: 0,
      ep_framesLegacyAction: 0,
      ep_stepsAdded: 0,
      ep_stepsSkippedNoDecisionInfo: 0,
      ep_stepsSkippedForcedIdle: 0,

      stateLearn: { canAct:0, air:0, ground:0, diving:0, lying:0 },
    };
    this.rolloutSteps = 2048;
    this.minRolloutToUpdate = 256; // flush threshold for remaining rollout steps

    this.ppoEpochs = 4;
    this.ppoMinibatch = 256;

    // ✅ GAME-DIAG에서 나온 값을 PPO-DIAG 로그 시점에 같이 찍기 위해 저장
    this._lastPowerHitRequested = 0;
    this._lastPowerHitApplied = 0;

    // ---- PBT(P1) ----
    this.pbtEnabled = (opts.pbtEnabled !== undefined) ? !!opts.pbtEnabled : true;
    this.pbtCycle = 0;

    // ✅ BEST 초기값은 0으로 (avg=0일 때 SAVE 방지)
    this.pbtBestScore = 0.0;

    this.pbtBestGenome = null;
    this.pbtCurrentGenome = null;

    this.disableWarmup = !!opts.disableWarmup;
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

    // ---- PBT checkpoints (best/current genome + best score + best weights) ----
    // NOTE: BEST 롤백은 "가중치(model_state) + genome"만 되돌린다.
    const bestScore = await this.storage.getCheckpoint('pbt_best_score');
    if (typeof bestScore === 'number') {
      // ✅ 과거 -1 같은 값이 남아있으면 0으로 정규화
      this.pbtBestScore = Math.max(0, bestScore);
      if (bestScore < 0) {
        await this.storage.setCheckpoint('pbt_best_score', this.pbtBestScore);
      }
    }

    const bestGenome = await this.storage.getCheckpoint('pbt_best_genome');
    if (bestGenome && typeof bestGenome === 'object') this.pbtBestGenome = bestGenome;

    const currentGenome = await this.storage.getCheckpoint('pbt_current_genome');
    if (currentGenome && typeof currentGenome === 'object') {
      this.pbtCurrentGenome = currentGenome;
      this._applyGenome(this.pbtCurrentGenome);
    } else {
      // 처음 실행이면 "현재 policy"에서 genome 생성해서 저장
      this.pbtCurrentGenome = this._makeGenomeFromPolicy();
      await this.storage.setCheckpoint('pbt_current_genome', this.pbtCurrentGenome);
    }

    // init()에서 pbt_current_genome 로드/생성 처리 직후
    console.log(
      `[PBT-INIT] currentGenome=${currentGenome ? 'LOADED' : 'CREATED'} ` +
      `lr=${Number(this.pbtCurrentGenome?.learningRate ?? 0).toExponential?.(2)} ` +
      `clip=${Number(this.pbtCurrentGenome?.clipEps ?? 0).toFixed?.(3)} ` +
      `vf=${Number(this.pbtCurrentGenome?.vfCoef ?? 0).toFixed?.(3)} ` +
      `gate=${JSON.stringify(this.pbtCurrentGenome?.powerHitGate ?? {})}`
    );
    const bestState = await this.storage.getCheckpoint('pbt_best_model_state'); // ✅ 추가
    console.log(
      `[PBT-INIT] bestScore=${this.pbtBestScore} ` +
      `bestGenome=${this.pbtBestGenome ? 'YES' : 'NO'} ` +
      `bestState=${bestState ? 'YES' : 'NO'}`
    );

    // best model_state는 "pbt_best_model_state"로 별도 보관
    // (없어도 OK: 첫 SAVE 때 생성됨)

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

      // ✅ UI 표시용
      pbtCycle: (this.pbtCycle ?? 0) | 0,
      bestScore: Number(this.pbtBestScore ?? 0),

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

  // ==============================
  // PBT(P1) helpers
  // ==============================
  _makeGenomeFromPolicy() {
    const gate = this.policy?.genome?.powerHitGate ?? {};
    return {
      learningRate: Number(this.policy.learningRate ?? 0.001),
      clipEps: Number(this.policy.clipEps ?? 0.2),
      vfCoef: Number(this.policy.vfCoef ?? 0.5),
      powerHitGate: {
        kFrames: (gate.kFrames ?? 4) | 0,
        dxMarginPx: (gate.dxMarginPx ?? 6) | 0,
        dyMarginPx: (gate.dyMarginPx ?? 10) | 0,
      },
    };
  }

  _policySig() {
    const p = this.policy;

    // 샘플 weight(기존 유지)
    const W1 = Number(p?.W1?.[0]?.[0] ?? 0);
    const W2 = Number(p?.W2?.[0]?.[0] ?? 0);
    const Wx = Number(p?.Wx?.[0]?.[0] ?? 0);
    const Wy = Number(p?.Wy?.[0]?.[0] ?? 0);
    const Wp = Number(p?.Wp?.[0]?.[0] ?? 0);
    const Wv = Number(p?.Wv?.[0] ?? 0);

    // bias 샘플(추가: weight 변화가 안 보일 때 bias가 변하는 케이스 잡음)
    const b1 = Number(p?.b1?.[0] ?? 0);
    const b2 = Number(p?.b2?.[0] ?? 0);
    const bp = Number(p?.bp?.[0] ?? 0);
    const bv = Number(p?.bv ?? 0);

    // 학습 직후 진단값(추가: 가장 민감)
    const lu = p?.debug?.lastUpdate ?? null;
    const wNorm = Number(lu?.wNorm ?? NaN);
    const gNorm = Number(lu?.gradNorm ?? NaN);
    const clipF = Number(lu?.clipFrac ?? NaN);

    // 하이퍼파라미터(추가: rollback/mutate가 적용됐는지 즉시 보임)
    const lr = Number(p?.learningRate ?? NaN);
    const clip = Number(p?.clipEps ?? NaN);
    const vf = Number(p?.vfCoef ?? NaN);

    const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4) : 'n/a');

    return (
      `lr=${Number.isFinite(lr) ? lr.toExponential(2) : 'n/a'} ` +
      `clip=${Number.isFinite(clip) ? clip.toFixed(3) : 'n/a'} vf=${Number.isFinite(vf) ? vf.toFixed(2) : 'n/a'} ` +
      `W1=${fmt(W1)} W2=${fmt(W2)} Wx=${fmt(Wx)} Wy=${fmt(Wy)} Wp=${fmt(Wp)} Wv=${fmt(Wv)} ` +
      `b1=${fmt(b1)} b2=${fmt(b2)} bp=${fmt(bp)} bv=${fmt(bv)} ` +
      `wNorm=${fmt(wNorm)} gNorm=${fmt(gNorm)} clipF=${fmt(clipF)}`
    );
  }


  _genomeSig(genome) {
    const g = genome ?? this.pbtCurrentGenome ?? this._makeGenomeFromPolicy();
    const gate = g?.powerHitGate ?? {};
    return (
      `lr=${Number(g.learningRate ?? 0).toExponential?.(2) ?? g.learningRate} ` +
      `clip=${Number(g.clipEps ?? 0).toFixed?.(3) ?? g.clipEps} ` +
      `vf=${Number(g.vfCoef ?? 0).toFixed?.(2) ?? g.vfCoef} ` +
      `gate(dx=${gate.dxMarginPx},dy=${gate.dyMarginPx},k=${gate.kFrames})`
    );
  }


  _applyGenome(genome) {
    if (!genome || typeof genome !== 'object') return;

    const before = this._makeGenomeFromPolicy();
    logDebug(`[PBT-GENOME] apply BEGIN sig=${this._policySig()} before=${JSON.stringify(before)} next=${JSON.stringify(genome)}`);

    // ✅ 단일 주입 지점
    this.policy.applyGenome({
      learningRate: genome.learningRate,
      clipEps: genome.clipEps,
      vfCoef: genome.vfCoef,
      powerHitGate: genome.powerHitGate,
    });

    const after = this._makeGenomeFromPolicy();
    logDebug(`[PBT-GENOME] apply END   sig=${this._policySig()} after=${JSON.stringify(after)}`);
  }

  _mutateGenome(baseGenome) {
    const g = JSON.parse(JSON.stringify(baseGenome ?? this._makeGenomeFromPolicy()));
    const before = JSON.parse(JSON.stringify(baseGenome ?? this._makeGenomeFromPolicy()));

    // learningRate: ×{0.8, 1.25}
    const lrScale = (Math.random() < 0.5) ? 0.8 : 1.25;
    g.learningRate = Math.max(1e-6, Math.min(1e-2, Number(g.learningRate || 1e-3) * lrScale));

    // clipEps: ±0.02
    g.clipEps = Number(g.clipEps || 0.2) + ((Math.random() < 0.5) ? -0.02 : 0.02);
    g.clipEps = Math.max(0.05, Math.min(0.4, g.clipEps));

    // vfCoef: ±0.1
    g.vfCoef = Number(g.vfCoef || 0.5) + ((Math.random() < 0.5) ? -0.1 : 0.1);
    g.vfCoef = Math.max(0, Math.min(2.0, g.vfCoef));

    // gate dx ±1px, dy ±2px (kFrames 고정)
    const gate = g.powerHitGate ?? (g.powerHitGate = {});
    const dx = (gate.dxMarginPx ?? 6) | 0;
    const dy = (gate.dyMarginPx ?? 10) | 0;
    gate.dxMarginPx = Math.max(0, Math.min(30, dx + ((Math.random() < 0.5) ? -1 : 1)));
    gate.dyMarginPx = Math.max(0, Math.min(40, dy + ((Math.random() < 0.5) ? -2 : 2)));
    gate.kFrames = (gate.kFrames ?? 4) | 0;

    console.log(`[PBT-MUTATE] from=${JSON.stringify(before)} to=${JSON.stringify(g)}`);
    logDebug(`[PBT-MUTATE] from=${JSON.stringify(baseGenome)} to=${JSON.stringify(g)}`);
    return g;
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
      logDebug(`[WARMUP] dataset already ready: samples=${already}`);
      return;
    }

    logDebug(`[WARMUP] collecting imitation samples... target=${target}, current=${already}`);

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
      if (!info.inputP1) return;

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
        logDebug(`[WARMUP] frames=${frames} samples=${dbCount}/${target}`);
        await sleep(0);
      }
    }

    if (buffer.length > 0) {
      await this.storage.appendImitationSamples(buffer);
      buffer.length = 0;
    }

    const finalCount = await this.storage.countImitationSamples();
    logDebug(`[WARMUP] done. frames=${frames} samples=${finalCount}`);

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
      logDebug('[WARN]', '[WARMUP-TRAIN] no imitation samples; skip');
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
    logDebug(`[WARMUP-TRAIN] start. samples=${N}, epochs=${epochs}, batch=${batchSize}`);
    // ---- BC/WARMUP DIAGNOSTICS ----
    // Label distribution / validity check (helps catch mapping or off-by-one bugs)
    {
      const histX = { '-1': 0, '0': 0, '1': 0, invalid: 0 };
      const histY = { '-1': 0, '0': 0, '1': 0, invalid: 0 };
      const histP = { '0': 0, '1': 0, invalid: 0 };
      for (let i = 0; i < N; i++) {
        const s = samples[i];
        const lx = (s?.label?.xDirection ?? 0);
        const ly = (s?.label?.yDirection ?? 0);
        const lp = (s?.label?.powerHit ?? 0);
        if (lx === -1 || lx === 0 || lx === 1) histX[String(lx)]++; else histX.invalid++;
        if (ly === -1 || ly === 0 || ly === 1) histY[String(ly)]++; else histY.invalid++;
        // Normalize lp to 0/1 if possible, otherwise mark invalid.
        // Avoid boolean comparisons because TS may infer lp as number-only.
        const lpNum = (lp === 0 || lp === 1) ? lp : Number.isFinite(Number(lp)) ? (Number(lp) ? 1 : 0) : -1;

        if (lpNum === 0 || lpNum === 1) histP[String(lpNum)]++;
        else histP.invalid++;

      }
      logDebug(`[WARMUP-DIST] xDir(-1,0,1)=${histX['-1']},${histX['0']},${histX['1']} invalid=${histX.invalid}`);
      logDebug(`[WARMUP-DIST] yDir(-1,0,1)=${histY['-1']},${histY['0']},${histY['1']} invalid=${histY.invalid}`);
      logDebug(`[WARMUP-DIST] power(0,1)=${histP['0']},${histP['1']} invalid=${histP.invalid}`);
    }

    // Print a few raw samples before training to sanity-check label/action conventions
    // (If these look wrong, BC will NEVER reach 40%+ no matter how long you train.)
    {
      const policyAny = /** @type {any} */ (this.policy);
      const K = Math.min(10, N);
      for (let i = 0; i < K; i++) {
        const s = samples[(Math.random() * N) | 0];
        if (!s?.obs) continue;
        if (typeof policyAny.evaluate !== 'function') break;
        const ev = policyAny.evaluate(s.obs, this.learningPlayer);
        const lx = (s.label?.xDirection ?? 0) | 0;
        const ly = (s.label?.yDirection ?? 0) | 0;
        const lp = (s.label?.powerHit ?? 0) ? 1 : 0;
        const fmt = (arr) => Array.from(arr).map(v => Number(v).toFixed(3)).join(',');
        logDebug(`[WARMUP-SAMPLE] lx=${lx} ly=${ly} lp=${lp}  px=[${fmt(ev.px)}] py=[${fmt(ev.py)}] pp=[${fmt(ev.pp)}]`);
      }
    }
    // ---- /BC/WARMUP DIAGNOSTICS ----

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
      // Confusion matrices (labelClass -> predClass) to diagnose mapping errors
      const cmX = [ [0,0,0], [0,0,0], [0,0,0] ];
      const cmY = [ [0,0,0], [0,0,0], [0,0,0] ];
      const cmP = [ [0,0], [0,0] ];
      let invalidLabel = 0;
      let mismatchShown = 0;

      for (let i = 0; i < N; i += batchSize) {
        const end = Math.min(N, i + batchSize);

        for (let k = i; k < end; k++) {
          const s = samples[idx[k]];
          // ✅ 필수 가드: obs/label 없는 샘플은 학습 불가 -> 스킵
          if (!s || !s.obs || !s.label) { invalidLabel++; continue; }
          const feat = this.policy.buildFeatures(s.obs, this.learningPlayer);
          // NOTE: policy implementation may expose warmup helper methods that are not declared in typings.
          // Cast to any to keep type-checking happy while preserving runtime behavior.
          const policyAny = /** @type {any} */ (this.policy);
          let r;
          if (typeof policyAny.updateImitationSample === 'function') {
            r = policyAny.updateImitationSample(s.obs, this.learningPlayer, s.label);
          } else {
            r = policyAny.updateImitation(feat, s.label);
          }
          if (!r || typeof r.loss !== 'number') { invalidLabel++; continue; }
          lossSum += r.loss;
          nSeen++;

          // label classes
          const lx = (s.label.xDirection ?? 0) | 0;
          const ly = (s.label.yDirection ?? 0) | 0;
          const lp = (s.label.powerHit ?? 0) ? 1 : 0;

          const lxs = (lx < 0 ? -1 : (lx > 0 ? 1 : 0));
          const lys = (ly < 0 ? -1 : (ly > 0 ? 1 : 0));
          const lxc = (lxs < 0 ? 0 : (lxs > 0 ? 2 : 1));
          const lyc = (lys < 0 ? 0 : (lys > 0 ? 2 : 1));
          if (!((lx === -1 || lx === 0 || lx === 1) && (ly === -1 || ly === 0 || ly === 1))) {
            invalidLabel++;
          } else {
            cmX[lxc][r.ax]++;
            cmY[lyc][r.ay]++;
            cmP[lp][r.ap]++;
          }

          // Show a few mismatches with predicted probabilities (helps catch mirrored/shifted labels)
          if (mismatchShown < 5) {
            const pxDir = (r.ax === 0 ? -1 : (r.ax === 2 ? 1 : 0));
            const pyDir = (r.ay === 0 ? -1 : (r.ay === 2 ? 1 : 0));
            const mx = (pxDir !== lxs);
            const my = (pyDir !== lys);
            const mp = (r.ap !== lp);
            if (mx || my || mp) {
              mismatchShown++;
              const policyAny2 = /** @type {any} */ (this.policy);
              if (typeof policyAny2.evaluate === 'function') {
                const ev2 = policyAny2.evaluate(s.obs, this.learningPlayer);
                const fmt = (arr) => Array.from(arr).map(v => Number(v).toFixed(3)).join(',');
                logDebug(`[WARMUP-MISMATCH] lx=${lx} ly=${ly} lp=${lp}  pred(ax,ay,ap)=${r.ax},${r.ay},${r.ap}  px=[${fmt(ev2.px)}] py=[${fmt(ev2.py)}] pp=[${fmt(ev2.pp)}]`);
              } else {
                logDebug(`[WARMUP-MISMATCH] lx=${lx} ly=${ly} lp=${lp}  pred(ax,ay,ap)=${r.ax},${r.ay},${r.ap}`);
              }
            }
          }

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
          logDebug(`[WARMUP-TRAIN] ep=${ep + 1}/${epochs} step=${globalStep} loss=${meanLoss.toFixed(3)} acc(x,y,p)=${ax.toFixed(3)},${ay.toFixed(3)},${ap.toFixed(3)}`);
        }
      }

      const meanLoss = lossSum / Math.max(1, nSeen);
      const ax = accX / Math.max(1, nSeen);
      const ay = accY / Math.max(1, nSeen);
      const ap = accP / Math.max(1, nSeen);
      logDebug(`[WARMUP-TRAIN] epoch done ${ep + 1}/${epochs}. loss=${meanLoss.toFixed(3)} acc(x,y,p)=${ax.toFixed(3)},${ay.toFixed(3)},${ap.toFixed(3)}`);
      // Confusion matrix summary (rows=labelClass, cols=predClass) for this epoch
      const cmRow = (row) => row.map(v => String(v)).join(',');
      logDebug(`[WARMUP-CM-X] rows(label -1,0,1): [${cmRow(cmX[0])}] [${cmRow(cmX[1])}] [${cmRow(cmX[2])}] invalidLabel=${invalidLabel}`);
      logDebug(`[WARMUP-CM-Y] rows(label -1,0,1): [${cmRow(cmY[0])}] [${cmRow(cmY[1])}] [${cmRow(cmY[2])}] invalidLabel=${invalidLabel}`);
      logDebug(`[WARMUP-CM-P] rows(label 0,1): [${cmRow(cmP[0])}] [${cmRow(cmP[1])}]`);
    }

    // Save updated model state
    await this.storage.setCheckpoint('model_state', this.policy.saveState());

    this.warmup.trained = true;
    await this.storage.setCheckpoint('warmup_train', { trained: true, trainEpochs: epochs, trainBatch: batchSize, updatedAt: Date.now() });

    logDebug('[WARMUP-TRAIN] done. model_state saved.');
  }



  async start() {
    if (this.running) return;

    // Phase0: imitation dataset collection (builtin vs builtin)
    if (!this.disableWarmup) {
      if (!this.warmup.done) await this._collectWarmupSamples();
      if (this.warmup.done && !this.warmup.trained) await this._trainWarmupImitation();
    } else {
      logDebug('[WARMUP] disabled (skip collect/train)');
    }

    if (this.graduated) return;

    this.running = true;

    // 다시 한번 확정
    if (typeof this.game.setExternalVsBuiltin === 'function') {
      this.game.setExternalVsBuiltin(true, false);
    }

    while (this.running && !this.graduated) {
      if (this.pbtEnabled) {
        await this._runPbtLoop();
      } else {
        while (this.running && !this.graduated) {
          await this.runPoints(this.pointsPerTick, 'train');
          await sleep(this.tickDelayMs > 0 ? this.tickDelayMs : 0);
        }
      }
    }


    // flush remaining rollout steps (PPO)
    await this._flushRollout(true);

    // 종료 시점에도 저장
    await this._saveStats();
    await this.storage.setCheckpoint('model_state', this.policy.saveState());
  }

  async _runPbtLoop() {
    // PBT loop: Train30 -> Eval30 x3 -> compare -> SAVE/ROLLBACK/HOLD -> (rollback이면 mutate)
    const TRAIN_N = 30;
    const EVAL_N = 30;

    if (!this.pbtCurrentGenome) {
      this.pbtCurrentGenome = this._makeGenomeFromPolicy();
      await this.storage.setCheckpoint('pbt_current_genome', this.pbtCurrentGenome);
    }

    while (this.running && !this.graduated) {
      this.pbtCycle++;

      // 1) train 30
      const trainRes = await this.runPoints(TRAIN_N, 'train');

      // ✅ (추가) Train→Eval 경계에서 남은 rollout을 강제 flush해서
      // "평가가 최신 가중치 기준"이 되게 한다.
      const rolloutBeforeEval = (this.rollout?.length ?? 0);
      const ppoFlushBeforeEval = this.ppoFlushCount || 0;
      const ppoStepsBeforeEval = this.ppoFlushSteps || 0;

      // ✅ force-flush 정책:
      // - rollout이 충분하면(>=minRolloutToUpdate) 업데이트해서 최신 가중치로 만든다
      // - 부족하면(<minRolloutToUpdate) 평가 일관성을 위해 rollout을 DROP(업데이트 없이 비움)
      let flushAction = 'NONE';
      let dropped = 0;

      if (rolloutBeforeEval >= this.minRolloutToUpdate) {
        flushAction = 'UPDATE';
        await this._flushRollout(true);
      } else if (rolloutBeforeEval > 0) {
        flushAction = 'DROP';
        dropped = rolloutBeforeEval;
        this.rollout.length = 0;
      }

      const ppoFlushAfterEval = this.ppoFlushCount || 0;
      const ppoStepsAfterEval = this.ppoFlushSteps || 0;
      const dFlush = ppoFlushAfterEval - ppoFlushBeforeEval;
      const dSteps = ppoStepsAfterEval - ppoStepsBeforeEval;

      logDebug(
        `[PBT-FLUSH] cycle=${this.pbtCycle} afterTrain ` +
        `rolloutBefore=${rolloutBeforeEval} action=${flushAction} dropped=${dropped} ` +
        `dPpoFlush=${dFlush} dPpoSteps=${dSteps} sig=${this._policySig()}`
      );

      // 2) eval 30 x3 (learning OFF)
      const eval1 = await this.runPoints(EVAL_N, 'eval');
      const eval2 = await this.runPoints(EVAL_N, 'eval');
      const eval3 = await this.runPoints(EVAL_N, 'eval');
      const avg = (eval1.winrate + eval2.winrate + eval3.winrate) / 3;

      // 3) decision
      const prevBest = Number(this.pbtBestScore ?? -1);
      let decision = 'HOLD';

      if (avg >= (prevBest + 0.05)) {
        decision = 'SAVE';

        // (optional) SAVE 직전 sig
        const sigBeforeSave = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';

        this.pbtBestScore = avg;
        this.pbtBestGenome = JSON.parse(JSON.stringify(this.pbtCurrentGenome ?? this._makeGenomeFromPolicy()));

        await this.storage.setCheckpoint('pbt_best_score', this.pbtBestScore);
        await this.storage.setCheckpoint('pbt_best_genome', this.pbtBestGenome);
        await this.storage.setCheckpoint('pbt_best_model_state', this.policy.saveState());

        const sigAfterSave = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
        const gSig = (typeof this._genomeSig === 'function') ? this._genomeSig(this.pbtBestGenome) : JSON.stringify(this.pbtBestGenome);

        console.log(
          `[PBT-SAVE] cycle=${this.pbtCycle} prevBest=${prevBest.toFixed(4)} avg=${avg.toFixed(4)} ` +
          `bestAfter=${Number(this.pbtBestScore).toFixed(4)} ` +
          `savedKeys=pbt_best_score,pbt_best_genome,pbt_best_model_state ` +
          `sigBefore=${sigBeforeSave} sigAfter=${sigAfterSave} genome=${gSig}`
        );

      } else if (avg <= (prevBest - 0.10)) {
        decision = 'ROLLBACK';

        // rollback: weights + genome
        const bestState = await this.storage.getCheckpoint('pbt_best_model_state');
        const okState = !!(bestState && typeof bestState === 'object');

        const sigBefore = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';

        if (okState) {
          this.policy.loadState(bestState);
          // 기존 로그 유지 + 강화
          console.log(`[PBT-ROLLBACK] loadState OK. wNorm=${this.policy?.debug?.lastUpdate?.wNorm ?? 'n/a'}`);

          if (this.pbtBestGenome) this._applyGenome(this.pbtBestGenome);
          await this.storage.setCheckpoint('model_state', this.policy.saveState());
        }

        const sigAfter = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
        const bestGSig = (typeof this._genomeSig === 'function') ? this._genomeSig(this.pbtBestGenome) : JSON.stringify(this.pbtBestGenome);

        console.log(
          `[PBT-ROLLBACK] cycle=${this.pbtCycle} prevBest=${prevBest.toFixed(4)} avg=${avg.toFixed(4)} ` +
          `bestState=${okState ? 'YES' : 'NO'} bestGenome=${this.pbtBestGenome ? 'YES' : 'NO'} ` +
          `sigBefore=${sigBefore} sigAfter=${sigAfter} bestGenomeSig=${bestGSig}`
        );

        // mutate from BEST genome (없으면 current)
        const baseG = this.pbtBestGenome ?? this.pbtCurrentGenome ?? this._makeGenomeFromPolicy();
        const baseGSig = (typeof this._genomeSig === 'function') ? this._genomeSig(baseG) : JSON.stringify(baseG);

        this.pbtCurrentGenome = this._mutateGenome(baseG);

        const mutGSig = (typeof this._genomeSig === 'function') ? this._genomeSig(this.pbtCurrentGenome) : JSON.stringify(this.pbtCurrentGenome);
        console.log(`[PBT-MUTATE] cycle=${this.pbtCycle} base=${baseGSig} -> mutated=${mutGSig}`);

        this._applyGenome(this.pbtCurrentGenome);
        await this.storage.setCheckpoint('pbt_current_genome', this.pbtCurrentGenome);

        const sigAfterMutApply = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
        console.log(`[PBT-MUT-APPLIED] cycle=${this.pbtCycle} sig=${sigAfterMutApply} genome=${mutGSig}`);

      } else {
        decision = 'HOLD';
        const sigHold = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
        const gHold = (typeof this._genomeSig === 'function') ? this._genomeSig(this.pbtCurrentGenome) : JSON.stringify(this.pbtCurrentGenome);
        console.log(`[PBT-HOLD] cycle=${this.pbtCycle} prevBest=${prevBest.toFixed(4)} avg=${avg.toFixed(4)} sig=${sigHold} genome=${gHold}`);
      }

      // 4) cycle log (요약 1줄)
      const g = this.pbtCurrentGenome ?? {};
      const gate = g.powerHitGate ?? {};
      const genomeStr =
        `lr=${Number(g.learningRate ?? 0).toExponential?.(2) ?? g.learningRate} ` +
        `clip=${Number(g.clipEps ?? 0).toFixed?.(3) ?? g.clipEps} ` +
        `vf=${Number(g.vfCoef ?? 0).toFixed?.(2) ?? g.vfCoef} ` +
        `gate(dx=${gate.dxMarginPx},dy=${gate.dyMarginPx},k=${gate.kFrames})`;

      const sig = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';

      logDebug(
        `[PBT] cycle=${this.pbtCycle} ` +
        `train=${trainRes.wins}/${TRAIN_N} ` +
        `eval=${eval1.wins}/${EVAL_N},${eval2.wins}/${EVAL_N},${eval3.wins}/${EVAL_N} ` +
        `avg=${avg.toFixed(4)} prevBest=${prevBest.toFixed(4)} best=${Number(this.pbtBestScore ?? -1).toFixed(4)} ` +
        `decision=${decision} ${genomeStr} sig=${sig}`
      );

      await sleep(this.tickDelayMs > 0 ? this.tickDelayMs : 0);
    }
  }


  async _runOnePointWithMode(cfg) {
  const res = await this._runner.runOnePoint();
  this.lastResult = res;

  if (!res) {
    logDebug('[WARN]', `[${cfg.tag}] runOnePoint returned null/undefined`);
    return null;
  }

  // 1) episode 저장
  if (cfg.saveEpisode) {
    if (res.episode) {
      await this.storage.appendEpisode(res.episode);
    } else {
      logDebug('[WARN]', `[${cfg.tag}] missing episode; skip save/learn`, { loseReason: res.loseReason });
    }
  }

  // 2) replay 저장
  if (cfg.saveReplay) {
    const replay = this._buildPointReplay(res);
    await this.storage.appendReplay(replay);
    await this.storage.pruneReplays(10);

    const isWinPoint = (replay.ok && replay.scoredBy === this.learningPlayer);
    if (isWinPoint) {
      const winReplay = { ...replay, id: `win:${replay.id}`, baseId: replay.id, bucket: 'win' };
      await this.storage.appendReplay(winReplay);
      if (this.storage.pruneWinReplays) await this.storage.pruneWinReplays(3);
    }
  }

  // 3) PPO 학습(rollout push + update)  <-- Eval에서는 절대 실행되면 안 됨
  if (cfg.learn) {
    // 여기 블록은 기존 start()의 canLearn~PPO flush~GAME-DIAG reset까지를 통째로 “그대로” 옮겨오면 됨.
    // 단, Eval에서 ppoUpdate가 0회가 되려면 이 if(cfg.learn) 밖으로 새어나오는 호출이 없어야 함.
    // PPO rollout collection + update
    const canLearn =
      res.ok === true &&
      res.frames > 0 &&
      res.loseReason !== 'EXCEPTION' &&
      !!res.episode;

    if (canLearn) {
      const transitions = res.episode.transitions ?? [];

      // Track how many steps we actually push for THIS point (helps diagnose “why pushed is tiny”).
      const rolloutLenBeforePoint = this.rollout.length;
      let pushedThisPoint = 0;
      let okInfoThisPoint = 0;
      

      // Pull episode-runner diagnostics if present
      const epAny = /** @type {any} */ (res.episode);
      const d = epAny?.diag;
      if (d && this.learnDiag) {
        this.learnDiag.ep_framesTotal += (d.framesTotal | 0);
        this.learnDiag.ep_framesCanActFalse += (d.framesCanActFalse | 0);
        this.learnDiag.ep_framesDecisionSampled += (d.framesDecisionSampled | 0);
        this.learnDiag.ep_framesLegacyAction += (d.framesLegacyAction | 0);
        this.learnDiag.ep_stepsAdded += (d.stepsAdded | 0);
        this.learnDiag.ep_stepsSkippedNoDecisionInfo += (d.stepsSkippedNoDecisionInfo | 0);
        this.learnDiag.ep_stepsSkippedForcedIdle += (d.stepsSkippedForcedIdle | 0);
      }

      // Spot-check: transitions length vs episode_runner stepsAdded (helps detect builder issues)
      if (d && typeof d.stepsAdded === 'number') {
        const ta = transitions.length | 0;
        const sa = d.stepsAdded | 0;
        if (ta > 0) {
          const ratio = sa / Math.max(1, ta);
          if (ratio < 0.95 || ratio > 1.05) {
            logDebug(`[EP-MISMATCH] ep=${this.totalEpisodes} transitions=${ta} stepsAdded=${sa} ratio=${ratio.toFixed(4)}`);
          }
        }
      }
      this.learnDiag.episodes++;
      this.learnDiag.transitions += transitions.length;
        let skippedNoInfo = 0;
        let skippedBadFields = 0;
      for (const tr of transitions) {
        const info = tr.info; // keep null/undefined as-is
        if (!info) { skippedNoInfo++; continue; }
        if (typeof info.logp !== 'number' || typeof info.value !== 'number') { skippedBadFields++; continue; }

        okInfoThisPoint++;

        const me = tr.obs?.me ?? {};
          const st = Number(me.state ?? 0);
          const isLying = !!me.isLying || st === 4;
          const isDiving = !!me.isDiving || st === 3;
          const isAir = (me.isAir !== undefined) ? !!me.isAir : (st === 1 || st === 2);
          const canAct = (me.canAct !== undefined) ? !!me.canAct : (!isLying && !isDiving);
          if (canAct) this.learnDiag.stateLearn.canAct++;
          if (isAir) this.learnDiag.stateLearn.air++;
          if (!isAir && canAct) this.learnDiag.stateLearn.ground++;
          if (isDiving) this.learnDiag.stateLearn.diving++;
          if (isLying) this.learnDiag.stateLearn.lying++;
          this.rollout.push({
            obs: tr.obs ?? null,
          action: tr.action ?? null,
          reward: Number(tr.reward ?? 0),
          done: !!tr.done,
          oldLogp: Number(info.logp ?? 0),
          value: Number(info.value ?? 0),
          playerIndex: this.learningPlayer,
        });
        pushedThisPoint++;
        this.learnDiag.pushedSteps++;
      }

      const rolloutLenAfterPoint = this.rollout.length;
      // Always log when pushed is unexpectedly small compared to info-ok transitions.
      // This should only happen if transitions themselves are tiny or filtered upstream.
      if (okInfoThisPoint > 0 && pushedThisPoint <= 5) {
        logDebug(
          `[POINT-PUSH] ep=${this.totalEpisodes} trans=${transitions.length} okInfo=${okInfoThisPoint} pushed=${pushedThisPoint} skippedNoInfo=${skippedNoInfo} skippedBadFields=${skippedBadFields} rolloutLen=${rolloutLenBeforePoint}->${rolloutLenAfterPoint}`
        );
      }

      // accumulate skip counters
      this.learnDiag.skippedNoInfo += skippedNoInfo;
      this.learnDiag.skippedBadFields += skippedBadFields;

      // update when enough rollout steps are collected
      while (this.rollout.length >= this.rolloutSteps) {
        const rolloutLenBeforeBatch = this.rollout.length;
        const batch = this.rollout.splice(0, this.rolloutSteps);
        this.learnDiag.consumedSteps += batch.length;
        const rolloutLenAfterBatch = this.rollout.length;
        logDebug(`[BATCH-CONSUME] batch=${batch.length} rolloutLen=${rolloutLenBeforeBatch}->${rolloutLenAfterBatch} pushedStepsSinceFlush=${this.learnDiag.pushedSteps} consumedStepsSinceFlush=${this.learnDiag.consumedSteps}`);

        // compute GAE + returns
        this.policy.computeGAE(batch);

        const stats = this.policy.ppoUpdate(batch, {
          epochs: this.ppoEpochs,
          minibatch: this.ppoMinibatch,
        });

        if (!stats) { logDebug('[PPO] flush stats missing'); continue; }
        const akl = (typeof stats.approxKl === 'number') ? stats.approxKl.toFixed(6) : 'NA';
        logDebug(`[PPO] flush steps=${stats.steps ?? 'NA'} updates=${stats.updates ?? 'NA'} approxKL=${akl}`);


        if (stats && typeof stats.policyLoss === 'number') {
        logDebug(`[PPO-LOSS] policyLoss=${stats.policyLoss.toFixed(6)} valueLoss=${stats.valueLoss.toFixed(6)} clipFrac=${stats.clipFrac.toFixed(4)} gradNorm=${stats.gradNorm.toFixed(6)} wNorm=${stats.wNorm.toFixed(3)}`);
        if (stats.vrCorr !== undefined) logDebug(`[VALUE-DIAG] corr=${stats.vrCorr.toFixed(4)}`);
        if (stats.advPos !== undefined) logDebug(`[ADV-SIGN] pos=${stats.advPos} neg=${stats.advNeg} zero=${stats.advZero}`);
        logDebug(`[PPO-ADV] advMean=${stats.advMean.toFixed(6)} advStd=${stats.advStd.toFixed(6)} retMean=${stats.retMean.toFixed(6)} retStd=${stats.retStd.toFixed(6)} rewMean=${stats.rewMean.toFixed(6)} rewStd=${stats.rewStd.toFixed(6)}`);
        }

        // ---- learnDiag logging + reset (한 번만) ----
        if (this.learnDiag) {
        if (this.learnDiag.stateLearn) {
            const s = this.learnDiag.stateLearn;
            logDebug(`[STATE-LEARN] canAct=${s.canAct} ground=${s.ground} air=${s.air} diving=${s.diving} lying=${s.lying}`);
            this.learnDiag.stateLearn = { canAct:0, air:0, ground:0, diving:0, lying:0 };
        }

        const transitionsN = Math.max(1, this.learnDiag.transitions);
        const rolloutLeft = this.rollout.length;
        const badRate = this.learnDiag.skippedBadFields / transitionsN;
        const noInfoRate = this.learnDiag.skippedNoInfo / transitionsN;

        logDebug(`[LEARN-DIAG] episodes=${this.learnDiag.episodes} transitions=${this.learnDiag.transitions} rolloutLen=${rolloutLeft} pushedSteps=${this.learnDiag.pushedSteps} consumedSteps=${this.learnDiag.consumedSteps} skippedNoInfo=${this.learnDiag.skippedNoInfo} skippedBadFields=${this.learnDiag.skippedBadFields}`);
        logDebug(`[LEARN-RATE] rolloutLenPerTransition=${(rolloutLeft / transitionsN).toFixed(4)} noInfoRate=${noInfoRate.toFixed(4)} badFieldRate=${badRate.toFixed(4)}`);

        if (this.learnDiag.pushedSteps > 0) {
            logDebug(`[LEARN-FLOW] pushedSteps=${this.learnDiag.pushedSteps} consumedSteps=${this.learnDiag.consumedSteps} leftoverRollout=${rolloutLeft} consumedRate=${(this.learnDiag.consumedSteps/this.learnDiag.pushedSteps).toFixed(4)}`);
        }

        const epFramesTotal = Math.max(1, this.learnDiag.ep_framesTotal);
        logDebug(`[EP-DIAG] framesTotal=${this.learnDiag.ep_framesTotal} canActFalse=${this.learnDiag.ep_framesCanActFalse} decisionSampled=${this.learnDiag.ep_framesDecisionSampled} legacyAction=${this.learnDiag.ep_framesLegacyAction}`);
        logDebug(`[EP-DIAG] stepsAdded=${this.learnDiag.ep_stepsAdded} skippedNoDecisionInfo=${this.learnDiag.ep_stepsSkippedNoDecisionInfo} skippedForcedIdle=${this.learnDiag.ep_stepsSkippedForcedIdle}`);
        logDebug(`[EP-RATE] canActFalseRate=${(this.learnDiag.ep_framesCanActFalse/epFramesTotal).toFixed(4)} decisionSampledRate=${(this.learnDiag.ep_framesDecisionSampled/epFramesTotal).toFixed(4)} stepsAddedPerFrame=${(this.learnDiag.ep_stepsAdded/epFramesTotal).toFixed(6)}`);

        // reset per flush (딱 1번만)
        this.learnDiag.episodes = 0;
        this.learnDiag.transitions = 0;
        this.learnDiag.pushedSteps = 0;
        this.learnDiag.consumedSteps = 0;
        this.learnDiag.skippedNoInfo = 0;
        this.learnDiag.skippedBadFields = 0;

        this.learnDiag.ep_framesTotal = 0;
        this.learnDiag.ep_framesCanActFalse = 0;
        this.learnDiag.ep_framesDecisionSampled = 0;
        this.learnDiag.ep_framesLegacyAction = 0;
        this.learnDiag.ep_stepsAdded = 0;
        this.learnDiag.ep_stepsSkippedNoDecisionInfo = 0;
        this.learnDiag.ep_stepsSkippedForcedIdle = 0;
        }

        // Diagnostics
        if (this.policy && this.policy.debug) {
          const pg = /** @type {any} */ (this.policy.debug.powerGate || {});
          const frames = Number(pg.frames ?? 0);
          const eligible = Number(pg.eligibleFrames ?? 0);
          const blocked = Number(pg.blockedFrames ?? (frames - eligible));
          const sumPPower = Number(pg.sumPPower ?? 0);
          const sumPPowerEl = Number(pg.sumPPowerEligible ?? 0);
          const expApplied = Number(pg.expectedApplied ?? 0);
          const sampledApplied = Number(pg.sampledApplied ?? 0);

          const div = (a, b, digits=4) => (b > 0 ? (a / b).toFixed(digits) : 'NA');
          const mean = (s, n, digits=4) => (n > 0 ? (s / n).toFixed(digits) : 'NA');

          logDebug(
            `[PPO-DIAG] nanFeatures=${this.policy.debug.nanFeatures} invalidSteps=${this.policy.debug.invalidFeatureSteps} ` +
            `forcedIdle=${this.policy.debug.forcedIdle} (noAct=${this.policy.debug.forcedIdleNoAct}, lying=${this.policy.debug.forcedIdleLying}, diving=${this.policy.debug.forcedIdleDiving}) ` +
            `powerHitReq=${this.policy.debug.powerHitSampled} powerMasked=${this.policy.debug.powerMasked} ` +
            `gateFrames=${frames} eligible=${eligible} blocked=${blocked} eligibleRate=${div(eligible, frames)} ` +
            `pPowerMean=${mean(sumPPower, frames)} pPowerMeanEligible=${mean(sumPPowerEl, eligible)} ` +
            `expApplied=${expApplied.toFixed(2)} expAppliedRateEligible=${div(expApplied, eligible)} ` +
            `sampledApplied=${sampledApplied} sampledAppliedRate=${div(sampledApplied, frames)}`
          );

          // --- P0-3 HEAD-DIAG (power head learning check) ---
          {
            const pol = /** @type {any} */ (this.policy);
            const Wp = pol.Wp;
            const bp = pol.bp;

            let wpSumSq = 0;
            if (Array.isArray(Wp)) {
              for (let k = 0; k < Wp.length; k++) {
                const row = Wp[k];
                if (row && row.length) {
                  for (let i = 0; i < row.length; i++) {
                    const v = Number(row[i] ?? 0);
                    wpSumSq += v * v;
                  }
                }
              }
            }

            const wpNorm = Math.sqrt(wpSumSq);
            const b0 = Number(bp?.[0] ?? 0);
            const b1 = Number(bp?.[1] ?? 0);
            const bDiff = b1 - b0;

            logDebug(`[HEAD-DIAG] WpNorm=${wpNorm.toFixed(6)} bp0=${b0.toFixed(6)} bp1=${b1.toFixed(6)} bpDiff=${bDiff.toFixed(6)}`);
          }


          // Observation sanity
          {
              const os = /** @type {any} */ (this.policy.debug.obsStats || {});
              const cnt = Number(os.count ?? 0);
              const avg2 = (s) => (cnt ? (s / cnt).toFixed(4) : 'NA');
              logDebug(`[OBS-DIAG] obsCount=${cnt} |absMeX|=${avg2(os.sumAbsMeX||0)} |absMeY|=${avg2(os.sumAbsMeY||0)} |absBallX|=${avg2(os.sumAbsBallX||0)} |absBallY|=${avg2(os.sumAbsBallY||0)} |dx|=${avg2(os.sumAbsDx||0)} |dy|=${avg2(os.sumAbsDy||0)}`);

              if (this.policy.debug.obsStats) {
              this.policy.debug.obsStats.count = 0;
              this.policy.debug.obsStats.sumAbsMeX = 0;
              this.policy.debug.obsStats.sumAbsMeY = 0;
              this.policy.debug.obsStats.sumAbsBallX = 0;
              this.policy.debug.obsStats.sumAbsBallY = 0;
              this.policy.debug.obsStats.sumAbsDx = 0;
              this.policy.debug.obsStats.sumAbsDy = 0;
              }
          }

          // Additional rolling diagnostics (action/mask/feat)  <-- 이 아래 전부 같은 블록 안
          const as = this.policy.debug.actionStats;
          if (as && as.n > 0) {
              const n = as.n;
              const ap1 = (as.apCounts?.[1] ?? 0);
              const phNear = (as.powerHitNearBall ?? 0);
              const phTot = Math.max(1, (as.powerHitTotal ?? ap1));
              logDebug(`[PPO-ACTION] n=${n} entX=${(as.entX/n).toFixed(4)} entY=${(as.entY/n).toFixed(4)} entP=${(as.entP/n).toFixed(4)} maxX=${(as.maxX/n).toFixed(4)} maxY=${(as.maxY/n).toFixed(4)} maxP=${(as.maxP/n).toFixed(4)} ap1=${ap1} phNear=${phNear}/${phTot} phGround=${(as.powerHitGround ?? 0)}/${Math.max(1, ap1)} phAir=${(as.powerHitAir ?? 0)}/${Math.max(1, ap1)} phAllowed=${(as.powerHitAllowed ?? 0)}/${Math.max(1, ap1)} ax=${JSON.stringify(as.axCounts)} ay=${JSON.stringify(as.ayCounts)} ap=${JSON.stringify(as.apCounts)}`);

              const ms = this.policy.debug.maskStats;
              if (ms && ms.n > 0) {
              const nms = Math.max(1, ms.n);
              logDebug(`[ACTION-MASK] n=${ms.n} groundN=${ms.groundN ?? 0} airN=${ms.airN ?? 0} yNegMaskedCount=${ms.yNegMaskedCount ?? 0} yNegMaskedMassAvg=${((ms.yNegMaskedMass ?? 0)/nms).toFixed(6)} xZeroMaskedCount=${ms.xZeroMaskedCount ?? 0} xZeroMaskedMassAvg=${((ms.xZeroMaskedMass ?? 0)/nms).toFixed(6)} illegalYPrevented=${ms.illegalYSampledPrevented ?? 0} illegalXPrevented=${ms.illegalXSampledPrevented ?? 0}`);
              }
          }

          const fs = this.policy.debug.featStats;
          if (fs && fs.n > 0) {
              const n = fs.n;
              const pick = (i) => {
              const mean = fs.sum[i] / n;
              const varr = Math.max(0, fs.sumsq[i] / n - mean * mean);
              const std = Math.sqrt(varr);
              return { i, min: fs.min[i], max: fs.max[i], mean, std };
              };
              const keys = [0,1,2,8,9,10,14,15,16,17,18,19];
              const rows = keys.filter(i => i < this.policy.featureLen).map(pick);
              logDebug(`[FEAT-DIAG] n=${n} ` + rows.map(r => `f${r.i}[min=${r.min.toFixed(2)},max=${r.max.toFixed(2)},mean=${r.mean.toFixed(2)},std=${r.std.toFixed(2)}]`).join(' '));
          }

          // reset rolling diagnostics per flush
          this.policy.debug.nanFeatures = 0;
          this.policy.debug.invalidFeatureSteps = 0;
          this.policy.debug.forcedIdle = 0;
          this.policy.debug.forcedIdleNoAct = 0;
          this.policy.debug.forcedIdleLying = 0;
          this.policy.debug.forcedIdleDiving = 0;
          this.policy.debug.powerHitSampled = 0;
          this.policy.debug.powerMasked = 0;
          if (this.policy.debug.powerGate) {
            const pg = this.policy.debug.powerGate;
            pg.frames = 0;
            pg.eligibleFrames = 0;
            pg.blockedFrames = 0;
            pg.sumPPower = 0;
            pg.sumPPowerEligible = 0;
            pg.expectedApplied = 0;
            pg.sampledApplied = 0;
            pg.blockedNoContactWindow = 0;
          }
          this.policy.debug.featStats = null;
          this.policy.debug.lastUpdate = null;
        }

        const gameAny = /** @type {any} */ (this.game);
        if (gameAny && gameAny.debugStats) {
          const ds = gameAny.debugStats;
          const req = ds.powerHitRequested | 0;
          const app = ds.powerHitApplied | 0;
          const contact = ds.powerHitContact | 0;
          const success = ds.powerHitSuccess | 0;

          logDebug(
            `[GAME-DIAG] decisions=${ds.decisions | 0} forcedIdle=${ds.forcedIdle | 0} ` +
            `powerHitReq=${req} powerHitApplied=${app} powerHitContact=${contact} powerHitSuccess=${success}`
          );

          // ✅ 추가: 이번 flush 구간의 "실제 엔진 발동" 수치를 저장
          this._lastPowerHitRequested = req;
          this._lastPowerHitApplied = app;

          if (req > 0) logDebug(`[ACTION-EFFECTIVE] powerHitAppliedRate=${(app / req).toFixed(4)}`);
          if (app > 0) logDebug(`[PH-GT] contactRate=${(contact / app).toFixed(4)} trueSuccessRate=${(success / app).toFixed(4)}`);
          if (contact > 0) logDebug(`[PH-GT2] successGivenContact=${(success / contact).toFixed(4)}`);

          // reset
          ds.decisions = 0;
          ds.forcedIdle = 0;
          ds.powerHitRequested = 0;
          ds.powerHitApplied = 0;
          ds.powerHitContact = 0;
          ds.powerHitSuccess = 0;
        }
      }
    } else {
      this.bufferSkipped++;
    }
  }

  // 4) global stats / recent1000
  if (cfg.updateGlobalStats) {
    this.totalEpisodes++;
    const isWin = (res.ok && res.scoredBy === this.learningPlayer);
    if (isWin) this.totalWins++;
    else if (res.ok) this.totalLosses++;

    if (cfg.updateRecent1000 && res.ok) {
      this._recordRecent1000(isWin);
      await this._maybePersistRecent1000(false);
    }

    logDebug(
      `[TRAIN] ep=${this.totalEpisodes} W=${this.totalWins} L=${this.totalLosses} ` +
      `WR=${(this.totalWins / Math.max(1, this.totalEpisodes)).toFixed(3)} ` +
      `last={scoredBy:${res.scoredBy}, loser:${res.loser}, frames:${res.frames}}`
    );
  }

  // 5) set 로직 + autosave (Eval에서는 OFF 권장/필수)
  if (cfg.updateSetAndAutosave) {
    // 여기 블록은 기존 start()의
    // - set score 업데이트
    // - 세트 종료 처리(P1 bestMargin 저장/졸업/flush/리셋)
    // - autosave
    // 를 그대로 옮기면 됨.
    // set score 업데이트
    if (res.ok && (res.scoredBy === 1 || res.scoredBy === 2)) {
      if (res.scoredBy === 1) this.currentSet.p1++;
      else this.currentSet.p2++;
    }

    // 세트 종료 처리
    if (this._isSetFinished()) {
      const p1Won = this._didP1WinSet();

      // Phase1: margin-based "current" snapshot rule
      // ✅ PBT 운영 중이면 혼선/간섭 방지: P1 bestMargin 저장 로직 OFF
      if (!this.pbtEnabled) {
        const margin = (this.currentSet.p1 | 0) - (this.currentSet.p2 | 0);
        if (margin > (this.phase1BestMargin | 0)) {
          this.phase1BestMargin = margin;
          await this.storage.setCheckpoint('model_state', this.policy.saveState());
          await this._saveStats();
          logDebug(`[P1] new bestMargin=${this.phase1BestMargin} (score ${this.currentSet.p1}-${this.currentSet.p2}) -> saved current`);
        }
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
        logDebug('[P1->P2] graduated via 3 consecutive set wins; mode set to PHASE2');
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

  return res;
}


  _setAgentMode(mode) {
  // prev 저장해두면, start() 밖에서 eval 돌릴 때도 안전
  const prev = { det: !!this.agent.deterministic, eps: Number(this.agent.epsilon ?? 0) };

  if (mode === 'eval') {
    this.agent.deterministic = true;
    this.agent.epsilon = 0;
  } else {
    this.agent.deterministic = false;
    this.agent.epsilon = 0.02;
  }

  return prev;
}

_restoreAgentMode(prev) {
  if (!prev) return;
  this.agent.deterministic = !!prev.det;
  this.agent.epsilon = Number(prev.eps ?? 0);
}

async runPoints(n, mode) {
  const points = Math.max(0, n | 0);

  // ---- 모드 토글 + 시작 로그(체크포인트) ----
  const prev = this._setAgentMode(mode);

  // ✅ PBT 태그(몇 번째 cycle의 로그인지)
  const tag = this.pbtEnabled ? ` cycle=${this.pbtCycle}` : '';

  // ✅ policy/genome 시그니처(Train/Eval 동일성 검증 핵심)
  const sig0 = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
  const gsig0 = (typeof this._genomeSig === 'function')
    ? this._genomeSig(this.pbtCurrentGenome)
    : JSON.stringify(this.pbtCurrentGenome ?? {});

  // ---- 학습 OFF 검증용 스냅샷 ----
  const ppoFlushBefore = this.ppoFlushCount || 0;
  const ppoStepsBefore = this.ppoFlushSteps || 0;
  const batchFlushBefore = this.flushCount || 0;
  const rolloutBefore = (this.rollout?.length ?? 0);

  // ✅ Eval 액션분포 로그를 위해: policy actionStats를 "runPoints 단위"로 리셋
  //   (PPO-DIAG는 flush 단위라서, Eval 30포인트 전체 분포를 보고 싶을 때 필요)
  if (mode === 'eval' && this.policy && this.policy.debug && this.policy.debug.actionStats) {
    const as = this.policy.debug.actionStats;
    as.n = 0;
    as.entX = 0; as.entY = 0; as.entP = 0;
    as.maxX = 0; as.maxY = 0; as.maxP = 0;
    as.axCounts = [0, 0, 0];
    as.ayCounts = [0, 0, 0];
    as.apCounts = [0, 0];
    // tie 카운터가 없을 수도 있으니 안전하게 초기화
    if (as.tieX !== undefined) as.tieX = 0;
    if (as.tieY !== undefined) as.tieY = 0;
    if (as.tieP !== undefined) as.tieP = 0;
  }

  if (mode === 'eval') {
    logDebug(
      `[EVAL] begin points=${points}${tag} learningOff=true eps=0 det=true ` +
      `sig=${sig0} genome=${gsig0} ` + // ✅ 추가
      `ppoFlush=${ppoFlushBefore} batchFlush=${batchFlushBefore} rollout=${rolloutBefore}`
    );
  } else {
    logDebug(
      `[TRAIN] begin points=${points}${tag} learningOn=true eps=${this.agent.epsilon} det=${this.agent.deterministic} ` +
      `sig=${sig0} genome=${gsig0}` // ✅ 추가
    );
  }

  let wins = 0;
  let losses = 0;

  // Eval은 “절대 학습/저장/세트/autosave”가 돌면 안 됨
  const cfg = (mode === 'eval')
    ? {
        tag: 'EVAL',
        learn: false,
        saveEpisode: false,
        saveReplay: false,
        updateGlobalStats: false,
        updateRecent1000: false,
        updateSetAndAutosave: false,
      }
    : {
        tag: 'TRAIN',
        learn: true,
        saveEpisode: true,
        saveReplay: true,
        updateGlobalStats: true,
        updateRecent1000: true,
        updateSetAndAutosave: true,
      };

  for (let i = 0; i < points; i++) {
    const res = await this._runOnePointWithMode(cfg);
    if (!res || !res.ok) continue;

    // ✅ C1: 승패 라벨 검증 (처음 3포인트만)
    if (i < 3) {
      logDebug(
        `[${mode.toUpperCase()}-LABEL] ` +
        `i=${i} LP=${this.learningPlayer} ` +
        `scoredBy=${res.scoredBy} loser=${res.loser} ` +
        `isWin=${res.scoredBy === this.learningPlayer}`
      );
    }

    const isWin = (res.scoredBy === this.learningPlayer);
    if (isWin) wins++;
    else losses++;
  }

  const winrate = (wins + losses) > 0 ? (wins / (wins + losses)) : 0;

  const ppoFlushAfter = this.ppoFlushCount || 0;
  const ppoStepsAfter = this.ppoFlushSteps || 0;
  const batchFlushAfter = this.flushCount || 0;
  const rolloutAfter = (this.rollout?.length ?? 0);

  const dPpoFlush = ppoFlushAfter - ppoFlushBefore;
  const dPpoSteps = ppoStepsAfter - ppoStepsBefore;
  const dBatchFlush = batchFlushAfter - batchFlushBefore;
  const dRollout = rolloutAfter - rolloutBefore;

  // ✅ end에서도 sig/genome 다시 찍기(롤백/변이 후 같은 runPoints 안에서도 변할 수 있음)
  const sig1 = (typeof this._policySig === 'function') ? this._policySig() : 'n/a';
  const gsig1 = (typeof this._genomeSig === 'function')
    ? this._genomeSig(this.pbtCurrentGenome)
    : JSON.stringify(this.pbtCurrentGenome ?? {});

  if (mode === 'eval') {
    logDebug(
      `[EVAL] end points=${points}${tag} wins=${wins} losses=${losses} winrate=${winrate.toFixed(4)} ` +
      `learningOff=true eps=0 det=true ` +
      `dPpoFlush=${dPpoFlush} dPpoSteps=${dPpoSteps} dBatchFlush=${dBatchFlush} dRollout=${dRollout} ` +
      `sig=${sig1} genome=${gsig1}` // ✅ 추가
    );

    // ✅ Eval 액션분포 요약(결정론 고정/동률 타이브레이크 문제 확정용)
    if (this.policy && this.policy.debug && this.policy.debug.actionStats) {
      const as = this.policy.debug.actionStats;
      const denom = Math.max(1, as.n | 0);

      const tieX = (as.tieX !== undefined) ? (as.tieX | 0) : 0;
      const tieY = (as.tieY !== undefined) ? (as.tieY | 0) : 0;
      const tieP = (as.tieP !== undefined) ? (as.tieP | 0) : 0;

      const nearTieX = (as.nearTieX | 0);
      const nearTieY = (as.nearTieY | 0);
      const nearTieP = (as.nearTieP | 0);

      const mX = (Number(as.marginXSum) / denom).toFixed(6);
      const mY = (Number(as.marginYSum) / denom).toFixed(6);
      const mP = (Number(as.marginPSum) / denom).toFixed(6);

      const gAllow = (as.gateAllowN | 0);
      const gBlock = (as.gateBlockN | 0);

      logDebug(
        `[EVAL-ACTS] n=${as.n | 0} ` +
        `ax=${(as.axCounts || [0,0,0]).join(',')} ` +
        `ay=${(as.ayCounts || [0,0,0]).join(',')} ` +
        `ap=${(as.apCounts || [0,0]).join(',')} ` +
        `entX=${(Number(as.entX) / denom).toFixed(4)} entY=${(Number(as.entY) / denom).toFixed(4)} entP=${(Number(as.entP) / denom).toFixed(4)} ` +
        `maxX=${(Number(as.maxX) / denom).toFixed(4)} maxY=${(Number(as.maxY) / denom).toFixed(4)} maxP=${(Number(as.maxP) / denom).toFixed(4)} ` +
        `tieX=${tieX} tieY=${tieY} tieP=${tieP}` +
        `nearTieX=${nearTieX} nearTieY=${nearTieY} nearTieP=${nearTieP}` +
        `marginX=${mX} marginY=${mY} marginP=${mP}` +
        `gateAllow=${gAllow} gateBlock=${gBlock}`
      );
    } else {
      logDebug('[EVAL-ACTS] actionStats unavailable (policy.debug.actionStats missing)');
    }
  } else {
    logDebug(
      `[TRAIN] end points=${points}${tag} wins=${wins} losses=${losses} winrate=${winrate.toFixed(4)} ` +
      `sig=${sig1} genome=${gsig1}` // ✅ 추가
    );
  }

  this._restoreAgentMode(prev);
  return { points, wins, losses, winrate };
}

_flushBatch(allRemaining = false) {
  const n = allRemaining ? this.pointBuffer.length : this.batchPoints;
  if (!n || n <= 0) return;

  const batch = this.pointBuffer.splice(0, n);
  let updatedTotal = 0;
  for (const item of batch) {
    try {
      const policyAny = /** @type {any} */ (this.policy);
      const out = policyAny.learnFromEpisode(item.episode);
      if (out && typeof out.updated === 'number') updatedTotal += out.updated;
    } catch (e) {
      logDebug('[WARN]', '[BATCH] learnFromEpisode failed; skip item', e);
    }
  }

  this.flushCount++;
  logDebug(`[BATCH] flush=${this.flushCount} size=${batch.length} updated=${updatedTotal} skipped=${this.bufferSkipped}`);
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

    // reset PBT state
    this.pbtCycle = 0;
    this.pbtBestScore = 0.0;   // ✅ 초기 BEST는 0으로 통일
    this.pbtBestGenome = null;
    this.pbtCurrentGenome = this._makeGenomeFromPolicy();

    this.consecutiveSetWins = 0;
    this.currentSet = { p1: 0, p2: 0, index: 1 };

    await this.storage.setCheckpoint('model_state', this.policy.saveState());

    // clearAll 후에도 checkpoint로 이어갈 수 있도록 초기화 저장
    await this.storage.setCheckpoint('pbt_best_score', this.pbtBestScore);
    await this.storage.setCheckpoint('pbt_best_genome', null);
    await this.storage.setCheckpoint('pbt_best_model_state', null);
    await this.storage.setCheckpoint('pbt_current_genome', this.pbtCurrentGenome);

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

        // compute GAE + returns
        this.policy.computeGAE(batch);

        const stats = this.policy.ppoUpdate(batch, {
          epochs: this.ppoEpochs,
          minibatch: this.ppoMinibatch,
        });

        // ---- PPO flush 카운터 ----
        this.ppoFlushCount++;
        this.ppoFlushSteps += (stats?.steps ?? batch.length);

        // ---- PPO logs ----
        logDebug(`[PPO] flush steps=${stats.steps} updates=${stats.updates} approxKL=${stats.approxKl.toFixed(6)}`);
        if (stats && typeof stats.policyLoss === 'number') {
        logDebug(
            `[PPO-LOSS] policyLoss=${stats.policyLoss.toFixed(6)} ` +
            `valueLoss=${stats.valueLoss.toFixed(6)} clipFrac=${stats.clipFrac.toFixed(4)} ` +
            `gradNorm=${stats.gradNorm.toFixed(6)} wNorm=${stats.wNorm.toFixed(3)}`
        );
        logDebug(
            `[PPO-ADV] advMean=${stats.advMean.toFixed(6)} advStd=${stats.advStd.toFixed(6)} ` +
            `retMean=${stats.retMean.toFixed(6)} retStd=${stats.retStd.toFixed(6)} ` +
            `rewMean=${stats.rewMean.toFixed(6)} rewStd=${stats.rewStd.toFixed(6)}`
        );
        }

        // ---- optional learnDiag logs + reset (flush에서만) ----
        if (this.learnDiag) {
        const transitionsN = Math.max(1, this.learnDiag.transitions);

        const pushedRate = (batch.length / transitionsN);
        const badRate = (this.learnDiag.skippedBadFields / transitionsN);
        const noInfoRate = (this.learnDiag.skippedNoInfo / transitionsN);

        logDebug(
            `[LEARN-DIAG] episodes=${this.learnDiag.episodes} transitions=${this.learnDiag.transitions} ` +
            `flushedBatch=${batch.length} skippedNoInfo=${this.learnDiag.skippedNoInfo} skippedBadFields=${this.learnDiag.skippedBadFields}`
        );
        logDebug(
            `[LEARN-RATE] batchPerTransition=${pushedRate.toFixed(4)} noInfoRate=${noInfoRate.toFixed(4)} badFieldRate=${badRate.toFixed(4)}`
        );

        // Episode-runner side diagnostics
        const epFramesTotal = Math.max(1, this.learnDiag.ep_framesTotal);
        logDebug(
            `[EP-DIAG] framesTotal=${this.learnDiag.ep_framesTotal} canActFalse=${this.learnDiag.ep_framesCanActFalse} ` +
            `decisionSampled=${this.learnDiag.ep_framesDecisionSampled} legacyAction=${this.learnDiag.ep_framesLegacyAction}`
        );
        logDebug(
            `[EP-DIAG] stepsAdded=${this.learnDiag.ep_stepsAdded} skippedNoDecisionInfo=${this.learnDiag.ep_stepsSkippedNoDecisionInfo} ` +
            `skippedForcedIdle=${this.learnDiag.ep_stepsSkippedForcedIdle}`
        );
        logDebug(
            `[EP-RATE] canActFalseRate=${(this.learnDiag.ep_framesCanActFalse / epFramesTotal).toFixed(4)} ` +
            `decisionSampledRate=${(this.learnDiag.ep_framesDecisionSampled / epFramesTotal).toFixed(4)} ` +
            `stepsAddedPerFrame=${(this.learnDiag.ep_stepsAdded / epFramesTotal).toFixed(6)}`
        );

        // reset per flush
        this.learnDiag.episodes = 0;
        this.learnDiag.transitions = 0;
        this.learnDiag.skippedNoInfo = 0;
        this.learnDiag.skippedBadFields = 0;

        this.learnDiag.ep_framesTotal = 0;
        this.learnDiag.ep_framesCanActFalse = 0;
        this.learnDiag.ep_framesDecisionSampled = 0;
        this.learnDiag.ep_framesLegacyAction = 0;
        this.learnDiag.ep_stepsAdded = 0;
        this.learnDiag.ep_stepsSkippedNoDecisionInfo = 0;
        this.learnDiag.ep_stepsSkippedForcedIdle = 0;
        }

        // ---- policy debug logs + reset ----
        if (this.policy && this.policy.debug) {
          const dbg = this.policy.debug;

          // ✅ flush 시점의 엔진 applied를 직접 읽는다 (이번 flush 구간 값)
          const gameAny = /** @type {any} */ (this.game);
          const ds = (gameAny && gameAny.debugStats) ? gameAny.debugStats : null;
          const appliedEngine = ds ? (ds.powerHitApplied | 0) : Number(this._lastPowerHitApplied ?? 0);

          const pg = /** @type {any} */ (dbg.powerGate || {});
          const frames = Number(pg.frames ?? 0);
          const eligible = Number(pg.eligibleFrames ?? 0);
          const expApplied = Number(pg.expectedApplied ?? 0);
          const sampledApplied = Number(pg.sampledApplied ?? 0);

          const div = (a, b) => (b > 0 ? (a / b) : 0);

          logDebug(
            `[PPO-DIAG] nanFeatures=${dbg.nanFeatures} invalidSteps=${dbg.invalidFeatureSteps} ` +
            `forcedIdle=${dbg.forcedIdle} (noAct=${dbg.forcedIdleNoAct}, lying=${dbg.forcedIdleLying}, diving=${dbg.forcedIdleDiving}) ` +
            `gateFrames=${frames} eligible=${eligible} eligibleRate=${div(eligible, frames).toFixed(4)} ` +
            `[POWER-EXPECT] expApplied=${expApplied.toFixed(2)} expAppliedRateEligible=${div(expApplied, eligible).toFixed(4)} ` +
            `[POWER-SAMPLE] sampledApplied=${sampledApplied} sampledRate=${div(sampledApplied, frames).toFixed(4)} ` +
            `[POWER-ENGINE] applied=${appliedEngine} appliedRateVsExpected=${(expApplied > 0 ? (appliedEngine / expApplied).toFixed(4) : 'NA')}`
          );

        // reset core counters
        dbg.nanFeatures = 0;
        dbg.invalidFeatureSteps = 0;
        dbg.forcedIdle = 0;
        dbg.forcedIdleNoAct = 0;
        dbg.forcedIdleLying = 0;
        dbg.forcedIdleDiving = 0;
        dbg.powerHitSampled = 0;
        dbg.powerMasked = 0;


        // reset actionStats (있을 때만)
        if (dbg.actionStats) {
            dbg.actionStats.n = 0;
            dbg.actionStats.entX = 0;
            dbg.actionStats.entY = 0;
            dbg.actionStats.entP = 0;
            dbg.actionStats.maxX = 0;
            dbg.actionStats.maxY = 0;
            dbg.actionStats.maxP = 0;
            dbg.actionStats.axCounts = [0, 0, 0];
            dbg.actionStats.ayCounts = [0, 0, 0];
            dbg.actionStats.apCounts = [0, 0];
            dbg.actionStats.powerHitTotal = 0;
            dbg.actionStats.powerHitNearBall = 0;
            dbg.actionStats.powerHitGround = 0;
            dbg.actionStats.powerHitAir = 0;
            dbg.actionStats.powerHitAllowed = 0;
        }

        // reset maskStats (있을 때만)
        if (dbg.maskStats) {
            dbg.maskStats.n = 0;
            dbg.maskStats.groundN = 0;
            dbg.maskStats.airN = 0;
            dbg.maskStats.yChosenCounts = [0, 0, 0];
            dbg.maskStats.yNegMaskedMass = 0;
            dbg.maskStats.yNegMaskedCount = 0;
            dbg.maskStats.xZeroMaskedMass = 0;
            dbg.maskStats.xZeroMaskedCount = 0;
            dbg.maskStats.illegalYSampledPrevented = 0;
            dbg.maskStats.illegalXSampledPrevented = 0;
        }

        // featStats는 누적이 커서 flush마다 끄는 게 맞다면 null로
        dbg.featStats = null;
        dbg.lastUpdate = null;
        }

        // ---- game debugStats logs + reset (한 번만) ----
        const gameAny2 = /** @type {any} */ (this.game);
        if (gameAny2 && gameAny2.debugStats) {
          const ds = gameAny2.debugStats;

          const req = ds.powerHitRequested | 0;
          const app = ds.powerHitApplied | 0;
          const contact = ds.powerHitContact | 0;
          const success = ds.powerHitSuccess | 0;

          logDebug(
            `[GAME-DIAG] decisions=${ds.decisions | 0} forcedIdle=${ds.forcedIdle | 0} ` +
            `powerHitReq=${req} powerHitApplied=${app} powerHitContact=${contact} powerHitSuccess=${success}`
          );

          // ✅ (선택) 이번 flush 값을 last로 보관 (다른 로그에서 쓰면 일관성 상승)
          this._lastPowerHitRequested = req;
          this._lastPowerHitApplied = app;

          // reset
          ds.decisions = 0;
          ds.forcedIdle = 0;
          ds.powerHitRequested = 0;
          ds.powerHitApplied = 0;
          ds.powerHitContact = 0;
          ds.powerHitSuccess = 0;
        }

        // loop exit conditions
        if (!force) break;
        if (this.rollout.length < this.minRolloutToUpdate) break;
    }
  }
}
