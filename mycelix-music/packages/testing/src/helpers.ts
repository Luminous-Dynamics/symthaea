// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Helpers
 *
 * Utility functions for testing.
 */

import { expect } from 'vitest';

// ==================== Async Helpers ====================

/**
 * Wait for a condition to be true
 */
export async function waitFor(
  condition: () => boolean | Promise<boolean>,
  options: { timeout?: number; interval?: number } = {}
): Promise<void> {
  const { timeout = 5000, interval = 50 } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) return;
    await sleep(interval);
  }

  throw new Error(`waitFor timed out after ${timeout}ms`);
}

/**
 * Sleep for a given duration
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry an async function until it succeeds
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: { attempts?: number; delay?: number } = {}
): Promise<T> {
  const { attempts = 3, delay = 1000 } = options;
  let lastError: Error | undefined;

  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < attempts - 1) await sleep(delay);
    }
  }

  throw lastError;
}

// ==================== API Helpers ====================

/**
 * Create an authenticated request helper
 */
export function createAuthenticatedFetch(token: string) {
  return async (url: string, options: RequestInit = {}) => {
    const headers = new Headers(options.headers);
    headers.set('Authorization', `Bearer ${token}`);
    headers.set('Content-Type', 'application/json');

    return fetch(url, { ...options, headers });
  };
}

/**
 * Parse API response and extract data
 */
export async function parseApiResponse<T>(response: Response): Promise<T> {
  const json = await response.json();

  if (!response.ok) {
    throw new Error(json.error?.message || `Request failed with status ${response.status}`);
  }

  return json.data;
}

// ==================== Assertion Helpers ====================

/**
 * Assert that an object has all required fields
 */
export function assertHasFields<T extends Record<string, unknown>>(
  obj: T,
  fields: (keyof T)[]
): void {
  for (const field of fields) {
    expect(obj).toHaveProperty(field as string);
    expect(obj[field]).not.toBeUndefined();
  }
}

/**
 * Assert that a song object is valid
 */
export function assertValidSong(song: unknown): void {
  expect(song).toBeDefined();
  assertHasFields(song as Record<string, unknown>, [
    'id',
    'title',
    'artist',
    'artistAddress',
    'duration',
    'audioUrl',
  ]);
}

/**
 * Assert that a playlist object is valid
 */
export function assertValidPlaylist(playlist: unknown): void {
  expect(playlist).toBeDefined();
  assertHasFields(playlist as Record<string, unknown>, [
    'id',
    'name',
    'ownerAddress',
    'songCount',
  ]);
}

/**
 * Assert that an artist object is valid
 */
export function assertValidArtist(artist: unknown): void {
  expect(artist).toBeDefined();
  assertHasFields(artist as Record<string, unknown>, [
    'address',
    'name',
  ]);
}

// ==================== Mock Helpers ====================

/**
 * Create a mock audio element
 */
export function createMockAudioElement(): Partial<HTMLAudioElement> {
  let _src = '';
  let _currentTime = 0;
  let _volume = 1;
  let _muted = false;
  let _paused = true;

  return {
    get src() { return _src; },
    set src(value: string) { _src = value; },
    get currentTime() { return _currentTime; },
    set currentTime(value: number) { _currentTime = value; },
    get volume() { return _volume; },
    set volume(value: number) { _volume = value; },
    get muted() { return _muted; },
    set muted(value: boolean) { _muted = value; },
    get paused() { return _paused; },
    duration: 180,
    play: async () => { _paused = false; },
    pause: () => { _paused = true; },
    addEventListener: () => {},
    removeEventListener: () => {},
  };
}

/**
 * Create mock localStorage
 */
export function createMockStorage(): Storage {
  const store = new Map<string, string>();

  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => { store.set(key, value); },
    removeItem: (key: string) => { store.delete(key); },
    clear: () => { store.clear(); },
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    get length() { return store.size; },
  };
}

// ==================== Ethereum Helpers ====================

/**
 * Generate a random Ethereum address
 */
export function randomAddress(): string {
  const chars = '0123456789abcdef';
  let address = '0x';
  for (let i = 0; i < 40; i++) {
    address += chars[Math.floor(Math.random() * chars.length)];
  }
  return address;
}

/**
 * Generate a random transaction hash
 */
export function randomTxHash(): string {
  const chars = '0123456789abcdef';
  let hash = '0x';
  for (let i = 0; i < 64; i++) {
    hash += chars[Math.floor(Math.random() * chars.length)];
  }
  return hash;
}

// ==================== Time Helpers ====================

/**
 * Create a date relative to now
 */
export function relativeDate(days: number): Date {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return date;
}

/**
 * Format date to ISO string without time
 */
export function toDateString(date: Date): string {
  return date.toISOString().split('T')[0];
}
