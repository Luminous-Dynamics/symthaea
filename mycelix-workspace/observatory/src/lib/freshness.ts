/**
 * Data freshness utilities — polling, stale indicators, error tracking.
 *
 * Usage in a route:
 *   const { lastUpdated, loadError, startPolling, stopPolling, refresh } = createFreshness(fetchData, 30_000);
 *   onMount(() => { refresh(); startPolling(); });
 *   onDestroy(() => stopPolling());
 */

import { writable, type Writable } from 'svelte/store';

export interface Freshness {
  /** Timestamp of last successful data load (ms since epoch). */
  lastUpdated: Writable<number>;
  /** Error message from most recent failed load, or empty string. */
  loadError: Writable<string>;
  /** Whether a refresh is currently in progress. */
  refreshing: Writable<boolean>;
  /** Start periodic polling. */
  startPolling: () => void;
  /** Stop periodic polling (call in onDestroy). */
  stopPolling: () => void;
  /** Trigger a single refresh now. */
  refresh: () => Promise<void>;
}

/**
 * Create freshness tracking for a route.
 *
 * @param fetchFn — async function that fetches and updates stores. Should throw on failure.
 * @param intervalMs — polling interval in ms (0 = no polling, manual refresh only)
 */
export function createFreshness(fetchFn: () => Promise<void>, intervalMs: number): Freshness {
  const lastUpdated = writable<number>(0);
  const loadError = writable<string>('');
  const refreshing = writable<boolean>(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  async function refresh(): Promise<void> {
    refreshing.set(true);
    try {
      await fetchFn();
      lastUpdated.set(Date.now());
      loadError.set('');
    } catch (e) {
      loadError.set(e instanceof Error ? e.message : 'Failed to load data');
    } finally {
      refreshing.set(false);
    }
  }

  function startPolling(): void {
    if (intervalMs > 0 && !timer) {
      timer = setInterval(refresh, intervalMs);
    }
  }

  function stopPolling(): void {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  return { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh };
}

/**
 * Format a timestamp as relative time (e.g., "just now", "2m ago", "1h ago").
 */
export function timeAgo(ts: number): string {
  if (ts === 0) return 'never';
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}
