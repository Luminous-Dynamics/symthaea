import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    globals: false,
    testTimeout: 120_000,
    // Run test files sequentially — each spawns Holochain conductors
    // and parallel execution causes lair-keystore resource contention
    fileParallelism: false,
  },
});
