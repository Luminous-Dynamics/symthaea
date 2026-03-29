// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte({ hot: !process.env.VITEST })],
  test: {
    // Test file patterns
    include: ['src/**/*.{test,spec}.{js,ts}'],

    // Environment for Tauri tests
    environment: 'jsdom',

    // Global test setup
    globals: true,

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/lib/**/*.ts'],
      exclude: ['src/lib/tests/**'],
    },

    // Timeout for async tests (Tauri calls can be slow)
    testTimeout: 30000,

    // Reporter
    reporters: ['verbose'],

    // Setup files
    setupFiles: ['./src/lib/tests/setup.ts'],
  },
  resolve: {
    alias: {
      $lib: '/src/lib',
      $services: '/src/lib/services',
      $stores: '/src/lib/stores',
      $components: '/src/lib/components',
    },
  },
});
