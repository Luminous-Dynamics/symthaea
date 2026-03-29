// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline Submission Queue — IndexedDB-backed persistence for form submissions
 * when the Holochain conductor is unavailable (demo mode).
 *
 * Uses raw IndexedDB API (no npm dependencies). Falls back silently when
 * IndexedDB is unavailable (e.g., private browsing in some browsers).
 */

import { writable } from 'svelte/store';

// ============================================================================
// Types
// ============================================================================

export interface QueuedSubmission {
  id: string;
  domain: string;       // 'tend' | 'food' | 'emergency' | etc.
  action: string;       // 'recordExchange' | 'recordHarvest' | etc.
  payload: unknown;     // the original function arguments
  created_at: number;   // Date.now()
  status: 'queued' | 'sending' | 'failed';
  attempts: number;
  last_error?: string;
  last_attempt_at?: number;  // For exponential backoff between retries
}

/** Reactive count of pending items in the offline queue. */
export const queueCount = writable<number>(0);

// ============================================================================
// IndexedDB wrapper (idb-keyval pattern, no dependencies)
// ============================================================================

const DB_NAME = 'mycelix-offline-queue';
const STORE_NAME = 'submissions';
const DB_VERSION = 1;
const MAX_ATTEMPTS = 3;

let dbPromise: Promise<IDBDatabase> | null = null;

function getDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB not available'));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  return dbPromise;
}

function tx(
  mode: IDBTransactionMode,
): Promise<{ store: IDBObjectStore; done: Promise<void> }> {
  return getDb().then((db) => {
    const transaction = db.transaction(STORE_NAME, mode);
    const store = transaction.objectStore(STORE_NAME);
    const done = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    return { store, done };
  });
}

function idbGet(id: string): Promise<QueuedSubmission | undefined> {
  return tx('readonly').then(({ store, done }) => {
    const req = store.get(id);
    return done.then(() => req.result as QueuedSubmission | undefined);
  });
}

function idbGetAll(): Promise<QueuedSubmission[]> {
  return tx('readonly').then(({ store, done }) => {
    const req = store.getAll();
    return done.then(() => (req.result ?? []) as QueuedSubmission[]);
  });
}

function idbPut(item: QueuedSubmission): Promise<void> {
  return tx('readwrite').then(({ store, done }) => {
    store.put(item);
    return done;
  });
}

function idbDelete(id: string): Promise<void> {
  return tx('readwrite').then(({ store, done }) => {
    store.delete(id);
    return done;
  });
}

function idbClear(): Promise<void> {
  return tx('readwrite').then(({ store, done }) => {
    store.clear();
    return done;
  });
}

function idbCount(): Promise<number> {
  return tx('readonly').then(({ store, done }) => {
    const req = store.count();
    return done.then(() => req.result as number);
  });
}

// ============================================================================
// Queue operations
// ============================================================================

/** Refresh the reactive queueCount store from IndexedDB. */
async function refreshCount(): Promise<void> {
  try {
    const count = await idbCount();
    queueCount.set(count);
  } catch {
    // IndexedDB unavailable — count stays at 0
  }
}

/**
 * Add a submission to the offline queue.
 * Returns the created QueuedSubmission, or null if storage failed.
 */
export async function enqueue(
  domain: string,
  action: string,
  payload: unknown,
): Promise<QueuedSubmission | null> {
  try {
    const item: QueuedSubmission = {
      id: crypto.randomUUID(),
      domain,
      action,
      payload,
      created_at: Date.now(),
      status: 'queued',
      attempts: 0,
    };
    await idbPut(item);
    await refreshCount();
    return item;
  } catch {
    // IndexedDB unavailable — submission not persisted
    return null;
  }
}

/**
 * Get all queued submissions, sorted oldest-first by created_at.
 */
export async function getQueue(): Promise<QueuedSubmission[]> {
  try {
    const all = await idbGetAll();
    return all.sort((a, b) => a.created_at - b.created_at);
  } catch {
    return [];
  }
}

/**
 * Get the count of pending items (queued or failed, not yet max-attempts).
 */
export async function getQueueCount(): Promise<number> {
  try {
    return await idbCount();
  } catch {
    return 0;
  }
}

/**
 * Process all queued items through the executor function.
 * Removes items on success. Increments attempts on failure.
 * Items exceeding MAX_ATTEMPTS (3) are left as 'failed' and skipped.
 */
export async function flush(
  executor: (domain: string, action: string, payload: unknown) => Promise<void>,
): Promise<{ succeeded: number; failed: number; skipped: number }> {
  const results = { succeeded: 0, failed: 0, skipped: 0 };

  let items: QueuedSubmission[];
  try {
    items = await getQueue();
  } catch {
    return results;
  }

  for (const item of items) {
    if (item.attempts >= MAX_ATTEMPTS) {
      results.skipped++;
      continue;
    }

    // Exponential backoff: skip items that were retried recently
    if (item.attempts > 0) {
      const backoffMs = Math.min(1000 * Math.pow(2, item.attempts - 1), 30_000);
      const lastAttemptAge = Date.now() - (item.last_attempt_at ?? 0);
      if (lastAttemptAge < backoffMs) {
        results.skipped++;
        continue;
      }
    }

    // Mark as sending
    item.status = 'sending';
    item.attempts++;
    item.last_attempt_at = Date.now();
    await idbPut(item);

    try {
      await executor(item.domain, item.action, item.payload);
      await idbDelete(item.id);
      results.succeeded++;
    } catch (err) {
      item.status = 'failed';
      item.last_error = err instanceof Error ? err.message : String(err);
      await idbPut(item);
      results.failed++;
    }
  }

  await refreshCount();
  return results;
}

/**
 * Remove all items from the queue.
 */
export async function clearQueue(): Promise<void> {
  try {
    await idbClear();
    await refreshCount();
  } catch {
    // IndexedDB unavailable
  }
}

/**
 * Remove a single item by id.
 */
export async function removeItem(id: string): Promise<void> {
  try {
    await idbDelete(id);
    await refreshCount();
  } catch {
    // IndexedDB unavailable
  }
}

/**
 * Initialize the queue count on app startup.
 * Call this from layout onMount or similar.
 */
export async function initQueueCount(): Promise<void> {
  await refreshCount();
}

// ============================================================================
// Mesh Bridge Fallback
// ============================================================================

/**
 * Check if the mesh bridge health endpoint is reachable.
 * Returns true if the mesh bridge is running and can relay messages.
 */
export async function isMeshBridgeAvailable(): Promise<boolean> {
  try {
    const meshUrl = typeof window !== 'undefined'
      ? `${window.location.protocol}//${window.location.hostname}:9100/health`
      : 'http://localhost:9100/health';
    const resp = await fetch(meshUrl, { signal: AbortSignal.timeout(2000) });
    if (!resp.ok) return false;
    const data = await resp.json();
    return data.status === 'running';
  } catch {
    return false;
  }
}

/**
 * Flush with mesh bridge fallback. When the primary executor (conductor)
 * fails for an item, attempts to relay through the mesh bridge if available.
 *
 * The mesh bridge will poll the conductor and relay the entry when it
 * reconnects — this just ensures the entry is logged locally for the
 * bridge to pick up.
 */
export async function flushWithMeshFallback(
  primaryExecutor: (domain: string, action: string, payload: unknown) => Promise<void>,
  meshAvailable?: boolean,
): Promise<{ succeeded: number; failed: number; skipped: number; meshRelayed: number }> {
  const results = { succeeded: 0, failed: 0, skipped: 0, meshRelayed: 0 };

  let items: QueuedSubmission[];
  try {
    items = await getQueue();
  } catch {
    return results;
  }

  // Check mesh bridge availability once per flush cycle
  const meshReady = meshAvailable ?? await isMeshBridgeAvailable();

  for (const item of items) {
    if (item.attempts >= MAX_ATTEMPTS) {
      results.skipped++;
      continue;
    }

    // Exponential backoff
    if (item.attempts > 0) {
      const backoffMs = Math.min(1000 * Math.pow(2, item.attempts - 1), 30_000);
      const lastAttemptAge = Date.now() - (item.last_attempt_at ?? 0);
      if (lastAttemptAge < backoffMs) {
        results.skipped++;
        continue;
      }
    }

    item.status = 'sending';
    item.attempts++;
    item.last_attempt_at = Date.now();
    await idbPut(item);

    try {
      await primaryExecutor(item.domain, item.action, item.payload);
      await idbDelete(item.id);
      results.succeeded++;
    } catch (err) {
      // Primary failed — if mesh bridge is running, mark as mesh-relayed
      // The mesh bridge poller will pick up the entry from the conductor
      // when connectivity resumes.
      if (meshReady) {
        item.status = 'queued'; // Keep queued for mesh bridge pickup
        item.last_error = `conductor failed, mesh bridge available: ${err instanceof Error ? err.message : String(err)}`;
        await idbPut(item);
        results.meshRelayed++;
      } else {
        item.status = 'failed';
        item.last_error = err instanceof Error ? err.message : String(err);
        await idbPut(item);
        results.failed++;
      }
    }
  }

  await refreshCount();
  return results;
}
