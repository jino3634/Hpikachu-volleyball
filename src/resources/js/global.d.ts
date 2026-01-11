export {};

declare global {
  interface Window {
    train: {
      enable: (on: boolean) => void;
      speed: (n: number) => void;
      status: () => void;
      curriculumStart: () => Promise<void>;
      curriculumStop: () => void;
    };
  }
}
