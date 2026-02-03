import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    globals: true,
    include: ['tests/**/*.test.ts', 'src/**/__tests__/*.test.ts'],
    // Setup file to handle libsodium cleanup errors
    setupFiles: ['./vitest.setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.d.ts'],
    },
    // Handle problematic ESM dependencies
    server: {
      deps: {
        inline: [
          'libsodium-wrappers',
          'libsodium',
          '@holochain/client',
        ],
      },
    },
    // Use forks pool to avoid module state issues with libsodium
    pool: 'forks',
    // Vitest 4: poolOptions moved to top-level options
    // isolate: true ensures each test file gets fresh module state
    // This prevents libsodium and crypto state from bleeding between test files
    isolate: true,
    // Suppress unhandled errors from libsodium module initialization during teardown
    dangerouslyIgnoreUnhandledErrors: true,
  },
  bench: {
    include: ['benchmarks/**/*.bench.ts'],
    reporters: ['default'],
  },
  resolve: {
    alias: {
      // Force libsodium to use CJS builds to avoid ESM "Cannot set property default" errors
      // The ESM builds (libsodium.mjs) try to assign module.exports which fails in vitest's module evaluator
      'libsodium-wrappers': resolve(__dirname, 'node_modules/libsodium-wrappers/dist/modules/libsodium-wrappers.js'),
      'libsodium': resolve(__dirname, 'node_modules/libsodium/dist/modules/libsodium.js'),
    },
  },
});
