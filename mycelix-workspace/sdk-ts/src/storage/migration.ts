/**
 * UESS Cross-Backend Migration Tools
 *
 * Utilities for migrating data between storage backends:
 * - local → DHT (promote to network persistence)
 * - DHT → IPFS (promote to immutable archival)
 * - Any backend → Any backend (generic migration)
 * - Bulk export/import with progress tracking
 *
 * @see docs/architecture/uess/UESS-01-CORE.md
 */

import type { StorageBackendAdapter } from './backends/memory.js';
import type { StorageMetadata } from './types.js';

// =============================================================================
// Types
// =============================================================================

/** Migration direction */
export type MigrationDirection =
  | 'local→dht'
  | 'dht→ipfs'
  | 'local→ipfs'
  | 'memory→local'
  | 'custom';

/** Progress callback for long-running migrations */
export type MigrationProgressCallback = (progress: MigrationProgress) => void;

/** Migration progress info */
export interface MigrationProgress {
  /** Total keys to migrate */
  total: number;
  /** Keys completed so far */
  completed: number;
  /** Keys that failed */
  failed: number;
  /** Keys skipped (already exist in target) */
  skipped: number;
  /** Current key being processed */
  currentKey?: string;
  /** Elapsed time in ms */
  elapsedMs: number;
}

/** Result of a single key migration */
export interface MigrationItemResult {
  key: string;
  status: 'migrated' | 'skipped' | 'failed';
  error?: string;
}

/** Overall migration result */
export interface MigrationResult {
  direction: MigrationDirection;
  total: number;
  migrated: number;
  skipped: number;
  failed: number;
  elapsedMs: number;
  items: MigrationItemResult[];
}

/** Migration options */
export interface MigrationOptions {
  /** Only migrate keys matching this pattern (glob-like, uses simple prefix/suffix matching) */
  keyPattern?: string;
  /** Skip keys that already exist in the target backend */
  skipExisting?: boolean;
  /** Delete from source after successful migration */
  deleteFromSource?: boolean;
  /** Maximum number of concurrent migrations */
  concurrency?: number;
  /** Progress callback */
  onProgress?: MigrationProgressCallback;
  /** Abort signal */
  signal?: AbortSignal;
}

/** Export format */
export interface ExportBundle {
  version: '1.0.0';
  exportedAt: number;
  agentId?: string;
  itemCount: number;
  items: ExportItem[];
}

/** Single exported item */
export interface ExportItem {
  key: string;
  data: unknown;
  metadata: StorageMetadata;
}

// =============================================================================
// Key Pattern Matching
// =============================================================================

function matchesPattern(key: string, pattern: string): boolean {
  if (pattern === '*') return true;
  if (pattern.endsWith('*')) {
    return key.startsWith(pattern.slice(0, -1));
  }
  if (pattern.startsWith('*')) {
    return key.endsWith(pattern.slice(1));
  }
  return key === pattern;
}

// =============================================================================
// Migration Engine
// =============================================================================

/**
 * Migrate data between two storage backends.
 *
 * ```typescript
 * import { migrate, createLocalBackend, createDHTBackend } from '@mycelix/sdk/storage';
 *
 * const result = await migrate(localBackend, dhtBackend, {
 *   keyPattern: 'profile:*',
 *   skipExisting: true,
 *   onProgress: (p) => console.log(`${p.completed}/${p.total}`),
 * });
 * ```
 */
export async function migrate(
  source: StorageBackendAdapter,
  target: StorageBackendAdapter,
  options: MigrationOptions = {}
): Promise<MigrationResult> {
  const {
    keyPattern = '*',
    skipExisting = true,
    deleteFromSource = false,
    concurrency = 4,
    onProgress,
    signal,
  } = options;

  const startTime = Date.now();
  const allKeys = await source.keys();
  const keys = allKeys.filter((k) => matchesPattern(k, keyPattern));
  const items: MigrationItemResult[] = [];
  let migrated = 0;
  let skipped = 0;
  let failed = 0;

  const reportProgress = () => {
    onProgress?.({
      total: keys.length,
      completed: migrated + skipped + failed,
      failed,
      skipped,
      elapsedMs: Date.now() - startTime,
    });
  };

  // Process in batches for concurrency control
  for (let i = 0; i < keys.length; i += concurrency) {
    if (signal?.aborted) break;

    const batch = keys.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      batch.map(async (key) => {
        if (signal?.aborted) {
          return { key, status: 'skipped' as const };
        }

        // Check if target already has this key
        if (skipExisting && (await target.has(key))) {
          return { key, status: 'skipped' as const };
        }

        // Read from source
        const entry = await source.get(key);
        if (!entry) {
          return { key, status: 'skipped' as const };
        }

        // Write to target
        const { cid: _cid, sizeBytes: _size, ...metadataWithout } = entry.metadata;
        await target.set(key, entry.data, metadataWithout);

        // Optionally delete from source
        if (deleteFromSource) {
          await source.delete(key);
        }

        return { key, status: 'migrated' as const };
      })
    );

    for (const result of results) {
      if (result.status === 'fulfilled') {
        items.push(result.value);
        if (result.value.status === 'migrated') migrated++;
        else if (result.value.status === 'skipped') skipped++;
      } else {
        const key = batch[results.indexOf(result)] ?? 'unknown';
        items.push({ key, status: 'failed', error: String(result.reason) });
        failed++;
      }
    }

    reportProgress();
  }

  return {
    direction: 'custom',
    total: keys.length,
    migrated,
    skipped,
    failed,
    elapsedMs: Date.now() - startTime,
    items,
  };
}

// =============================================================================
// Export / Import
// =============================================================================

/**
 * Export all data from a backend to a portable bundle.
 *
 * ```typescript
 * const bundle = await exportBackend(localBackend, { keyPattern: 'profile:*' });
 * const json = JSON.stringify(bundle);
 * ```
 */
export async function exportBackend(
  source: StorageBackendAdapter,
  options: {
    keyPattern?: string;
    agentId?: string;
    onProgress?: MigrationProgressCallback;
  } = {}
): Promise<ExportBundle> {
  const { keyPattern = '*', agentId, onProgress } = options;
  const startTime = Date.now();

  const allKeys = await source.keys();
  const keys = allKeys.filter((k) => matchesPattern(k, keyPattern));
  const items: ExportItem[] = [];

  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const entry = await source.get(key);
    if (entry) {
      items.push({ key, data: entry.data, metadata: entry.metadata });
    }

    onProgress?.({
      total: keys.length,
      completed: i + 1,
      failed: 0,
      skipped: 0,
      elapsedMs: Date.now() - startTime,
    });
  }

  return {
    version: '1.0.0',
    exportedAt: Date.now(),
    agentId,
    itemCount: items.length,
    items,
  };
}

/**
 * Import data from an export bundle into a backend.
 *
 * ```typescript
 * const bundle = JSON.parse(jsonStr) as ExportBundle;
 * const result = await importBundle(dhtBackend, bundle, { skipExisting: true });
 * ```
 */
export async function importBundle(
  target: StorageBackendAdapter,
  bundle: ExportBundle,
  options: {
    skipExisting?: boolean;
    onProgress?: MigrationProgressCallback;
  } = {}
): Promise<MigrationResult> {
  const { skipExisting = true, onProgress } = options;
  const startTime = Date.now();
  const items: MigrationItemResult[] = [];
  let migrated = 0;
  let skipped = 0;
  let failed = 0;

  for (let i = 0; i < bundle.items.length; i++) {
    const item = bundle.items[i];

    try {
      if (skipExisting && (await target.has(item.key))) {
        items.push({ key: item.key, status: 'skipped' });
        skipped++;
      } else {
        const { cid: _cid, sizeBytes: _size, ...metadataWithout } = item.metadata;
        await target.set(item.key, item.data, metadataWithout);
        items.push({ key: item.key, status: 'migrated' });
        migrated++;
      }
    } catch (err) {
      items.push({ key: item.key, status: 'failed', error: String(err) });
      failed++;
    }

    onProgress?.({
      total: bundle.items.length,
      completed: i + 1,
      failed,
      skipped,
      elapsedMs: Date.now() - startTime,
    });
  }

  return {
    direction: 'custom',
    total: bundle.items.length,
    migrated,
    skipped,
    failed,
    elapsedMs: Date.now() - startTime,
    items,
  };
}

// =============================================================================
// Sync Utility
// =============================================================================

/** Sync result */
export interface SyncResult {
  /** Keys only in source (will be copied to target if direction includes 'push') */
  sourceOnly: string[];
  /** Keys only in target (will be copied to source if direction includes 'pull') */
  targetOnly: string[];
  /** Keys in both backends */
  shared: string[];
  /** Migration result if sync was executed */
  migration?: MigrationResult;
}

/**
 * Compare two backends and optionally sync them.
 *
 * ```typescript
 * // Dry run - just compare
 * const diff = await sync(localBackend, dhtBackend, { dryRun: true });
 * console.log(`${diff.sourceOnly.length} keys need migration`);
 *
 * // Push local → DHT
 * const result = await sync(localBackend, dhtBackend, { direction: 'push' });
 * ```
 */
export async function sync(
  source: StorageBackendAdapter,
  target: StorageBackendAdapter,
  options: {
    direction?: 'push' | 'pull' | 'bidirectional';
    dryRun?: boolean;
    keyPattern?: string;
    onProgress?: MigrationProgressCallback;
  } = {}
): Promise<SyncResult> {
  const { direction = 'push', dryRun = false, keyPattern = '*', onProgress } = options;

  const sourceKeys = (await source.keys()).filter((k) => matchesPattern(k, keyPattern));
  const targetKeys = (await target.keys()).filter((k) => matchesPattern(k, keyPattern));

  const sourceSet = new Set(sourceKeys);
  const targetSet = new Set(targetKeys);

  const sourceOnly = sourceKeys.filter((k) => !targetSet.has(k));
  const targetOnly = targetKeys.filter((k) => !sourceSet.has(k));
  const shared = sourceKeys.filter((k) => targetSet.has(k));

  const result: SyncResult = { sourceOnly, targetOnly, shared };

  if (dryRun) return result;

  // Push: source → target
  if (direction === 'push' || direction === 'bidirectional') {
    if (sourceOnly.length > 0) {
      result.migration = await migrate(source, target, {
        keyPattern,
        skipExisting: true,
        onProgress,
      });
    }
  }

  // Pull: target → source
  if (direction === 'pull' || direction === 'bidirectional') {
    if (targetOnly.length > 0) {
      const pullResult = await migrate(target, source, {
        keyPattern,
        skipExisting: true,
        onProgress,
      });
      // Merge results
      if (result.migration) {
        result.migration.migrated += pullResult.migrated;
        result.migration.skipped += pullResult.skipped;
        result.migration.failed += pullResult.failed;
        result.migration.items.push(...pullResult.items);
      } else {
        result.migration = pullResult;
      }
    }
  }

  return result;
}
