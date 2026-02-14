import { defineConfig } from 'vitest/config';

/**
 * Vitest configuration for Conductor Integration Tests
 *
 * These tests require a running Holochain conductor and are excluded from
 * the main test suite. Run with:
 *   CONDUCTOR_AVAILABLE=true npm run test:conductor
 *
 * Or use the full test script:
 *   ./scripts/conductor-tests.sh
 */
export default defineConfig({
  test: {
    globals: true,
    include: ['tests/conductor/**/*.test.ts'],
    // No exclude - we want ALL conductor tests to run
    exclude: ['node_modules'],
    testTimeout: 30000, // Longer timeout for network operations
    hookTimeout: 60000, // Allow time for conductor setup/teardown
    // Handle libsodium-wrappers ESM compatibility issues
    deps: {
      optimizer: {
        web: {
          include: [
            'libsodium-wrappers',
            'libsodium-wrappers-sumo',
            '@holochain/client',
          ],
        },
      },
    },
  },
});
