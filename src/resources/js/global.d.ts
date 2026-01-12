export {};

declare global {
  interface Window {
    __PV_TRAINING_MUTE__?: boolean;

    // train은 dev/debug에서만 달릴 수도 있으니 optional로
    train?: {
      // 기존에 쓰던 콘솔 제어 함수들
      enable?: (on: boolean) => void;
      speed?: (n: number) => void;
      status?: () => any;
      curriculumStart?: () => Promise<void>;
      curriculumStop?: () => void;

      // ✅ 추가: 디버깅/리플레이 접근용
      trainer?: any;
      storage?: any;
      replay?: any;
    };
  }
}
