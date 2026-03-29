// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 180000, // 3 minutes for Holochain operations
    hookTimeout: 60000, // 1 minute for setup/teardown
    pool: 'forks', // Use forks for Holochain conductor isolation
    poolOptions: {
      forks: {
        singleFork: true, // Run tests sequentially in a single fork
      },
    },
    include: ['src/**/*.test.ts'],
    reporters: ['verbose'],
  },
});
