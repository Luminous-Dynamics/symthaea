// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Setup
 *
 * Configuration and setup for Vitest tests.
 */

import { beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from './mocks';

// MSW Server for API mocking
export const server = setupServer(...handlers);

/**
 * Setup MSW server for all tests
 */
export function setupMSW() {
  beforeAll(() => {
    server.listen({ onUnhandledRequest: 'warn' });
  });

  afterEach(() => {
    server.resetHandlers();
  });

  afterAll(() => {
    server.close();
  });
}

/**
 * Setup browser environment mocks
 */
export function setupBrowserMocks() {
  // Mock window.matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });

  // Mock ResizeObserver
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  // Mock IntersectionObserver
  global.IntersectionObserver = class IntersectionObserver {
    root = null;
    rootMargin = '';
    thresholds = [];
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() { return []; }
  } as any;

  // Mock Audio
  global.Audio = class MockAudio {
    src = '';
    currentTime = 0;
    duration = 180;
    volume = 1;
    muted = false;
    paused = true;
    play() { this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; }
    addEventListener() {}
    removeEventListener() {}
  } as any;

  // Mock localStorage
  const localStorageMock = {
    store: new Map<string, string>(),
    getItem(key: string) { return this.store.get(key) ?? null; },
    setItem(key: string, value: string) { this.store.set(key, value); },
    removeItem(key: string) { this.store.delete(key); },
    clear() { this.store.clear(); },
  };
  Object.defineProperty(window, 'localStorage', { value: localStorageMock });

  // Mock fetch if not available
  if (typeof fetch === 'undefined') {
    global.fetch = async () => new Response();
  }
}

/**
 * Vitest configuration for browser tests
 */
export const vitestBrowserConfig = {
  test: {
    environment: 'happy-dom',
    setupFiles: ['./src/setup.ts'],
    globals: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'dist/**',
        '**/*.d.ts',
        '**/*.test.ts',
        '**/*.spec.ts',
      ],
    },
  },
};

/**
 * Vitest configuration for Node tests
 */
export const vitestNodeConfig = {
  test: {
    environment: 'node',
    globals: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
};
