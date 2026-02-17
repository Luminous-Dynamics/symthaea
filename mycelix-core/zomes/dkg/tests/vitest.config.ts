import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 300_000, // 5 minutes - Holochain tests can be slow
    hookTimeout: 120_000,
    include: ['src/**/*.test.ts'],
    globals: true,
    reporters: ['verbose'],
    pool: 'forks', // Use forks for Holochain conductor isolation
  },
});
