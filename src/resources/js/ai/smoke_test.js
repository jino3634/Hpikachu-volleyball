// ai/smoke_test.js
'use strict';

import { PikaEnv } from './env.js';

function randomAction3bit() {
  // A-only 정책: (left/right/up) 조합 중 하나.
  // 여기서는 단순 랜덤. 나중에 정책 네트워크가 출력하면 됨.
  const a = Math.floor(Math.random() * 8); // 0..7
  // left+right 동시에(3) 같은 건 그냥 허용(Env가 neutral 처리)
  return a;
}

function runOneGame() {
  const env = new PikaEnv({
    winningScore: 3,       // smoke test는 짧게
    maxFramesPerGame: 5000,
    keepReplays: 10,
  });

  env.reset(123);

  let done = false;
  let totalReward = 0;

  while (!done) {
    const a1 = randomAction3bit();
    const a2 = randomAction3bit();

    const { obs, reward, done: d, info } = env.step(a1, a2);
    totalReward += reward;
    done = d;

    // 가끔 진행 로그
    if (obs.frame % 300 === 0) {
      console.log(`[frame ${obs.frame}] score=${info.score.p1}-${info.score.p2} reward=${reward}`);
    }
  }

  const rep = env.getLatestReplay();
  console.log('Game finished:', rep);
  console.log('Total reward (P1):', totalReward);

  // 리플레이 프레임 수 확인
  console.log('Recorded frames:', rep.frames.length);

env.saveLatestReplayToFile('ai/replay_latest.json');
env.saveRecentReplaysToFile('ai/replays.json');
console.log('Saved replays:', { latest: 'ai/replay_latest.json', recent: 'ai/replays.json' });


}

runOneGame();
