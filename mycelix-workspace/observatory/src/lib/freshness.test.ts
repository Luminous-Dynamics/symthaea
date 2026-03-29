// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import { createFreshness, timeAgo } from './freshness';

// ============================================================================
// timeAgo
// ============================================================================

describe('timeAgo', () => {
  it('returns "never" for 0', () => {
    expect(timeAgo(0)).toBe('never');
  });

  it('returns "just now" for a recent timestamp', () => {
    expect(timeAgo(Date.now())).toBe('just now');
  });

  it('returns seconds ago for 10-59s', () => {
    expect(timeAgo(Date.now() - 30_000)).toBe('30s ago');
  });

  it('returns minutes ago for 60s+', () => {
    expect(timeAgo(Date.now() - 120_000)).toBe('2m ago');
  });

  it('returns hours ago for 3600s+', () => {
    expect(timeAgo(Date.now() - 3_600_000)).toBe('1h ago');
  });

  it('returns multiple hours ago', () => {
    expect(timeAgo(Date.now() - 7_200_000)).toBe('2h ago');
  });
});

// ============================================================================
// createFreshness
// ============================================================================

describe('createFreshness', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('sets lastUpdated on successful fetch', async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { lastUpdated, refresh } = createFreshness(fetchFn, 0);

    expect(get(lastUpdated)).toBe(0);
    await refresh();
    expect(get(lastUpdated)).toBeGreaterThan(0);
  });

  it('clears loadError on successful fetch', async () => {
    const fetchFn = vi.fn()
      .mockRejectedValueOnce(new Error('network fail'))
      .mockResolvedValueOnce(undefined);
    const { loadError, refresh } = createFreshness(fetchFn, 0);

    await refresh();
    expect(get(loadError)).toBe('network fail');

    await refresh();
    expect(get(loadError)).toBe('');
  });

  it('sets loadError on failed fetch', async () => {
    const fetchFn = vi.fn().mockRejectedValue(new Error('connection refused'));
    const { loadError, refresh } = createFreshness(fetchFn, 0);

    await refresh();
    expect(get(loadError)).toBe('connection refused');
  });

  it('sets generic error message for non-Error throws', async () => {
    const fetchFn = vi.fn().mockRejectedValue('string error');
    const { loadError, refresh } = createFreshness(fetchFn, 0);

    await refresh();
    expect(get(loadError)).toBe('Failed to load data');
  });

  it('tracks refreshing state', async () => {
    let resolvePromise: () => void;
    const fetchFn = vi.fn().mockImplementation(
      () => new Promise<void>((resolve) => { resolvePromise = resolve; })
    );
    const { refreshing, refresh } = createFreshness(fetchFn, 0);

    expect(get(refreshing)).toBe(false);
    const p = refresh();
    expect(get(refreshing)).toBe(true);
    resolvePromise!();
    await p;
    expect(get(refreshing)).toBe(false);
  });

  it('startPolling triggers periodic refresh', async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { startPolling, stopPolling } = createFreshness(fetchFn, 1000);

    startPolling();
    expect(fetchFn).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchFn).toHaveBeenCalledTimes(2);

    stopPolling();
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it('startPolling does nothing with intervalMs=0', () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { startPolling } = createFreshness(fetchFn, 0);
    startPolling();
    vi.advanceTimersByTime(5000);
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it('startPolling is idempotent (no duplicate timers)', async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { startPolling, stopPolling } = createFreshness(fetchFn, 1000);

    startPolling();
    startPolling(); // second call should be no-op

    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchFn).toHaveBeenCalledTimes(1); // not 2

    stopPolling();
  });

  it('stopPolling is safe to call when not polling', () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { stopPolling } = createFreshness(fetchFn, 1000);
    expect(() => stopPolling()).not.toThrow();
  });
});
