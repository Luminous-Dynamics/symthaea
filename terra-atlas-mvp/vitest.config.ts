import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  // Disable CSS processing for tests
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
  },

  test: {
    // Test environment
    environment: 'node',

    // Skip CSS file processing
    css: false,

    // Global test APIs (describe, it, expect, etc.)
    globals: true,

    // Test file patterns
    include: ['tests/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['**/node_modules/**', '**/dist/**'],

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
        '**/*.test.ts',
        '**/*.spec.ts',
        '.next/',
        'coverage/',
      ],
      // Thresholds to enforce as we build coverage
      // thresholds: {
      //   lines: 80,
      //   functions: 80,
      //   branches: 80,
      //   statements: 80
      // }
    },

    // Timeout for tests
    testTimeout: 10000,

    // Run tests in parallel
    pool: 'threads',

    // Reporter
    reporters: ['verbose'],
  },

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
      '@lib': path.resolve(__dirname, './lib'),
      '@utils': path.resolve(__dirname, './utils'),
      '@components': path.resolve(__dirname, './components'),
    },
  },
})
