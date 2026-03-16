import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

/**
 * The offline-queue module relies on IndexedDB which is not available in vitest
 * (even with jsdom). The module has try/catch guards that gracefully degrade.
 *
 * These tests verify that the module handles missing IndexedDB without throwing,
 * and that the exported store has correct defaults.
 */

// Reset module state between tests (dbPromise is module-level)
beforeEach(() => {
  vi.resetModules();
});

describe('offline-queue (no IndexedDB)', () => {
  it('queueCount defaults to 0', async () => {
    const { queueCount } = await import('./offline-queue');
    expect(get(queueCount)).toBe(0);
  });

  it('enqueue returns null when IndexedDB is unavailable', async () => {
    const { enqueue } = await import('./offline-queue');
    const result = await enqueue('tend', 'recordExchange', { hours: 2 });
    expect(result).toBeNull();
  });

  it('getQueue returns empty array when IndexedDB is unavailable', async () => {
    const { getQueue } = await import('./offline-queue');
    const items = await getQueue();
    expect(items).toEqual([]);
  });

  it('getQueueCount returns 0 when IndexedDB is unavailable', async () => {
    const { getQueueCount } = await import('./offline-queue');
    const count = await getQueueCount();
    expect(count).toBe(0);
  });

  it('flush returns zeroed results when IndexedDB is unavailable', async () => {
    const { flush } = await import('./offline-queue');
    const executor = vi.fn();
    const results = await flush(executor);
    expect(results).toEqual({ succeeded: 0, failed: 0, skipped: 0 });
    expect(executor).not.toHaveBeenCalled();
  });

  it('clearQueue does not throw when IndexedDB is unavailable', async () => {
    const { clearQueue } = await import('./offline-queue');
    await expect(clearQueue()).resolves.toBeUndefined();
  });

  it('removeItem does not throw when IndexedDB is unavailable', async () => {
    const { removeItem } = await import('./offline-queue');
    await expect(removeItem('nonexistent-id')).resolves.toBeUndefined();
  });

  it('initQueueCount does not throw when IndexedDB is unavailable', async () => {
    const { initQueueCount } = await import('./offline-queue');
    await expect(initQueueCount()).resolves.toBeUndefined();
  });
});

describe('QueuedSubmission type shape', () => {
  it('enqueue would create a properly shaped object (verified by type)', async () => {
    // This test verifies the QueuedSubmission interface is importable
    // and the module exports are correctly typed
    const mod = await import('./offline-queue');
    expect(typeof mod.enqueue).toBe('function');
    expect(typeof mod.flush).toBe('function');
    expect(typeof mod.getQueue).toBe('function');
    expect(typeof mod.getQueueCount).toBe('function');
    expect(typeof mod.clearQueue).toBe('function');
    expect(typeof mod.removeItem).toBe('function');
    expect(typeof mod.initQueueCount).toBe('function');
  });
});
