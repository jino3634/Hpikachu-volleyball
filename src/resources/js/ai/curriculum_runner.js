// curriculum_runner.js
'use strict';

const DEFAULT_WARMUP_FRAMES = 4000;
const DEFAULT_TARGET_STREAK = 3;

export class CurriculumRunner {
  constructor(game, agentP1, opts = {}) {
    this.game = game;
    this.agentP1 = agentP1;
    this.warmupFrames = opts.warmupFrames ?? DEFAULT_WARMUP_FRAMES;
    this.targetWinStreak = opts.targetWinStreak ?? DEFAULT_TARGET_STREAK;
    this.onSetEnd = opts.onSetEnd ?? null;
    this.onGraduate = opts.onGraduate ?? null;
  }

  _configureTeacherMatch() {
    // P1 external vs P2 builtin teacher
    this.game.setExternalVsBuiltin(true, false);
    this.game.setAgents(this.agentP1, null);
    this.game.physics.player1.isComputer = false;
    this.game.physics.player2.isComputer = true;
  }

  _warmupToRound() {
    for (let i = 0; i < this.warmupFrames; i++) {
      this.game.stepLogic();
      if (this.game.state === this.game.round) return true;
    }
    return false;
  }
}
