// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Batch Operations
 *
 * Bulk storage operations for performance optimization.
 * @see docs/architecture/uess/UESS-12-BULK-OPERATIONS.md
 */

import type { StorageReceipt, StoreOptions, StorageMetadata } from './types.js';
import type { EpistemicClassification } from '../epistemic/types.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Batch operation item
 */
export interface BatchItem<T = unknown> {
  /** Unique key for the item */
  key: string;

  /** Data to store */
  data: T;

  /** Classification for the data */
  classification: EpistemicClassification;

  /** Storage options */
  options?: StoreOptions;
}

/**
 * Batch operation result
 */
export interface BatchResult<T = unknown> {
  /** Key of the item */
  key: string;

  /** Whether the operation succeeded */
  success: boolean;

  /** Storage receipt if successful */
  receipt?: StorageReceipt;

  /** Error message if failed */
  error?: string;

  /** Data if retrieved */
  data?: T;

  /** Metadata if retrieved */
  metadata?: StorageMetadata;
}

/**
 * Batch operation configuration
 */
export interface BatchConfig {
  /** Maximum items per batch (default: 100) */
  maxBatchSize: number;

  /** Whether to continue on errors (default: true) */
  continueOnError: boolean;

  /** Maximum concurrent operations (default: 10) */
  maxConcurrency: number;

  /** Timeout per item in ms (default: 30000) */
  itemTimeoutMs: number;

  /** Whether to use transactions where supported (default: true) */
  useTransactions: boolean;

  /** Retry failed items (default: 3) */
  retryCount: number;

  /** Retry delay in ms (default: 100) */
  retryDelayMs: number;
}

const DEFAULT_BATCH_CONFIG: BatchConfig = {
  maxBatchSize: 100,
  continueOnError: true,
  maxConcurrency: 10,
  itemTimeoutMs: 30000,
  useTransactions: true,
  retryCount: 3,
  retryDelayMs: 100,
};

/**
 * Batch operation statistics
 */
export interface BatchStats {
  totalItems: number;
  successCount: number;
  errorCount: number;
  retriedCount: number;
  startTime: number;
  endTime?: number;
  durationMs?: number;
  itemsPerSecond?: number;
}

/**
 * Batch operation progress callback
 */
export type BatchProgressCallback = (progress: {
  completed: number;
  total: number;
  currentKey: string;
  percentComplete: number;
}) => void;

// =============================================================================
// Batch Executor
// =============================================================================

/**
 * Storage operation function type
 */
export type StoreFn<T> = (
  key: string,
  data: T,
  classification: EpistemicClassification,
  options?: StoreOptions
) => Promise<StorageReceipt>;

export type RetrieveFn<T> = (
  key: string
) => Promise<{ data: T; metadata: StorageMetadata } | null>;

export type DeleteFn = (key: string) => Promise<boolean>;

/**
 * Batch Executor - Handles bulk storage operations
 */
export class BatchExecutor {
  private readonly config: BatchConfig;

  constructor(config?: Partial<BatchConfig>) {
    this.config = { ...DEFAULT_BATCH_CONFIG, ...config };
  }

  // ===========================================================================
  // Batch Store
  // ===========================================================================

  /**
   * Store multiple items in batch
   */
  async batchStore<T>(
    items: BatchItem<T>[],
    storeFn: StoreFn<T>,
    onProgress?: BatchProgressCallback
  ): Promise<{
    results: BatchResult<T>[];
    stats: BatchStats;
  }> {
    const stats: BatchStats = {
      totalItems: items.length,
      successCount: 0,
      errorCount: 0,
      retriedCount: 0,
      startTime: Date.now(),
    };

    const results: BatchResult<T>[] = [];

    // Process in batches
    for (let i = 0; i < items.length; i += this.config.maxBatchSize) {
      const batch = items.slice(i, i + this.config.maxBatchSize);
      const batchResults = await this.processBatch(batch, async (item) => {
        const receipt = await storeFn(
          item.key,
          item.data,
          item.classification,
          item.options
        );
        return { success: true, receipt, key: item.key };
      }, stats, onProgress, i);

      results.push(...batchResults);
    }

    stats.endTime = Date.now();
    stats.durationMs = stats.endTime - stats.startTime;
    stats.itemsPerSecond = stats.durationMs > 0
      ? (stats.totalItems / stats.durationMs) * 1000
      : 0;

    return { results, stats };
  }

  // ===========================================================================
  // Batch Retrieve
  // ===========================================================================

  /**
   * Retrieve multiple items in batch
   */
  async batchRetrieve<T>(
    keys: string[],
    retrieveFn: RetrieveFn<T>,
    onProgress?: BatchProgressCallback
  ): Promise<{
    results: BatchResult<T>[];
    stats: BatchStats;
  }> {
    const stats: BatchStats = {
      totalItems: keys.length,
      successCount: 0,
      errorCount: 0,
      retriedCount: 0,
      startTime: Date.now(),
    };

    const results: BatchResult<T>[] = [];

    // Process in batches
    for (let i = 0; i < keys.length; i += this.config.maxBatchSize) {
      const batch = keys.slice(i, i + this.config.maxBatchSize);
      const batchResults = await this.processBatch(
        batch.map(key => ({ key })),
        async (item) => {
          const result = await retrieveFn(item.key);
          if (result) {
            return {
              success: true,
              key: item.key,
              data: result.data,
              metadata: result.metadata,
            };
          }
          return { success: false, key: item.key, error: 'Not found' };
        },
        stats,
        onProgress,
        i
      );

      results.push(...batchResults);
    }

    stats.endTime = Date.now();
    stats.durationMs = stats.endTime - stats.startTime;
    stats.itemsPerSecond = stats.durationMs > 0
      ? (stats.totalItems / stats.durationMs) * 1000
      : 0;

    return { results, stats };
  }

  // ===========================================================================
  // Batch Delete
  // ===========================================================================

  /**
   * Delete multiple items in batch
   */
  async batchDelete(
    keys: string[],
    deleteFn: DeleteFn,
    onProgress?: BatchProgressCallback
  ): Promise<{
    results: BatchResult[];
    stats: BatchStats;
  }> {
    const stats: BatchStats = {
      totalItems: keys.length,
      successCount: 0,
      errorCount: 0,
      retriedCount: 0,
      startTime: Date.now(),
    };

    const results: BatchResult[] = [];

    // Process in batches
    for (let i = 0; i < keys.length; i += this.config.maxBatchSize) {
      const batch = keys.slice(i, i + this.config.maxBatchSize);
      const batchResults = await this.processBatch(
        batch.map(key => ({ key })),
        async (item) => {
          const deleted = await deleteFn(item.key);
          return {
            success: deleted,
            key: item.key,
            error: deleted ? undefined : 'Delete failed',
          };
        },
        stats,
        onProgress,
        i
      );

      results.push(...batchResults);
    }

    stats.endTime = Date.now();
    stats.durationMs = stats.endTime - stats.startTime;
    stats.itemsPerSecond = stats.durationMs > 0
      ? (stats.totalItems / stats.durationMs) * 1000
      : 0;

    return { results, stats };
  }

  // ===========================================================================
  // Batch Reclassify
  // ===========================================================================

  /**
   * Reclassify multiple items in batch
   */
  async batchReclassify(
    items: Array<{ key: string; newClassification: EpistemicClassification }>,
    reclassifyFn: (key: string, newClassification: EpistemicClassification) => Promise<StorageReceipt>,
    onProgress?: BatchProgressCallback
  ): Promise<{
    results: BatchResult[];
    stats: BatchStats;
  }> {
    const stats: BatchStats = {
      totalItems: items.length,
      successCount: 0,
      errorCount: 0,
      retriedCount: 0,
      startTime: Date.now(),
    };

    const results: BatchResult[] = [];

    for (let i = 0; i < items.length; i += this.config.maxBatchSize) {
      const batch = items.slice(i, i + this.config.maxBatchSize);
      const batchResults = await this.processBatch(
        batch,
        async (item) => {
          const receipt = await reclassifyFn(item.key, item.newClassification);
          return { success: true, key: item.key, receipt };
        },
        stats,
        onProgress,
        i
      );

      results.push(...batchResults);
    }

    stats.endTime = Date.now();
    stats.durationMs = stats.endTime - stats.startTime;
    stats.itemsPerSecond = stats.durationMs > 0
      ? (stats.totalItems / stats.durationMs) * 1000
      : 0;

    return { results, stats };
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private async processBatch<I extends { key: string }, R extends BatchResult>(
    items: I[],
    processFn: (item: I) => Promise<R>,
    stats: BatchStats,
    onProgress?: BatchProgressCallback,
    offset: number = 0
  ): Promise<R[]> {
    const results: R[] = [];
    const semaphore = new Semaphore(this.config.maxConcurrency);

    const promises = items.map(async (item, index) => {
      await semaphore.acquire();

      try {
        const result = await this.processWithRetry(item, processFn, stats);
        results[index] = result;

        if (result.success) {
          stats.successCount++;
        } else {
          stats.errorCount++;
        }

        if (onProgress) {
          onProgress({
            completed: offset + index + 1,
            total: stats.totalItems,
            currentKey: item.key,
            percentComplete: ((offset + index + 1) / stats.totalItems) * 100,
          });
        }
      } finally {
        semaphore.release();
      }
    });

    await Promise.all(promises);
    return results;
  }

  private async processWithRetry<I extends { key: string }, R extends BatchResult>(
    item: I,
    processFn: (item: I) => Promise<R>,
    stats: BatchStats
  ): Promise<R> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.config.retryCount; attempt++) {
      try {
        return await this.withTimeout(
          processFn(item),
          this.config.itemTimeoutMs
        );
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt > 0) {
          stats.retriedCount++;
        }

        if (!this.config.continueOnError && attempt === this.config.retryCount - 1) {
          throw lastError;
        }

        if (attempt < this.config.retryCount - 1) {
          await this.sleep(this.config.retryDelayMs * Math.pow(2, attempt));
        }
      }
    }

    return {
      success: false,
      key: item.key,
      error: lastError?.message ?? 'Unknown error',
    } as R;
  }

  private withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) =>
        setTimeout(() => reject(new Error('Operation timed out')), timeoutMs)
      ),
    ]);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Semaphore for Concurrency Control
// =============================================================================

class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise(resolve => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    const next = this.waiting.shift();
    if (next) {
      next();
    } else {
      this.permits++;
    }
  }
}

// =============================================================================
// Streaming Batch Operations
// =============================================================================

/**
 * Batch item source for streaming operations
 */
export interface BatchItemSource<T> {
  /** Get next batch of items */
  next(): Promise<BatchItem<T>[] | null>;

  /** Total items if known */
  totalCount?: number;
}

/**
 * Streaming batch executor for large datasets
 */
export class StreamingBatchExecutor {
  private readonly config: BatchConfig;

  constructor(config?: Partial<BatchConfig>) {
    this.config = { ...DEFAULT_BATCH_CONFIG, ...config };
  }

  /**
   * Stream store items from a source
   */
  async *streamStore<T>(
    source: BatchItemSource<T>,
    storeFn: StoreFn<T>
  ): AsyncGenerator<BatchResult<T>[], void, unknown> {
    const executor = new BatchExecutor(this.config);

    let batch = await source.next();
    while (batch !== null) {
      const { results } = await executor.batchStore(batch, storeFn);
      yield results;
      batch = await source.next();
    }
  }

  /**
   * Stream retrieve items
   */
  async *streamRetrieve<T>(
    keys: AsyncIterable<string>,
    retrieveFn: RetrieveFn<T>,
    batchSize: number = this.config.maxBatchSize
  ): AsyncGenerator<BatchResult<T>[], void, unknown> {
    const executor = new BatchExecutor(this.config);
    let batch: string[] = [];

    for await (const key of keys) {
      batch.push(key);

      if (batch.length >= batchSize) {
        const { results } = await executor.batchRetrieve(batch, retrieveFn);
        yield results;
        batch = [];
      }
    }

    // Process remaining items
    if (batch.length > 0) {
      const { results } = await executor.batchRetrieve(batch, retrieveFn);
      yield results;
    }
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a batch executor
 */
export function createBatchExecutor(config?: Partial<BatchConfig>): BatchExecutor {
  return new BatchExecutor(config);
}

/**
 * Create a streaming batch executor
 */
export function createStreamingBatchExecutor(config?: Partial<BatchConfig>): StreamingBatchExecutor {
  return new StreamingBatchExecutor(config);
}

/**
 * Helper to create an array source for streaming
 */
export function createArraySource<T>(
  items: BatchItem<T>[],
  batchSize: number = 100
): BatchItemSource<T> {
  let offset = 0;

  return {
    totalCount: items.length,
    async next(): Promise<BatchItem<T>[] | null> {
      if (offset >= items.length) {
        return null;
      }

      const batch = items.slice(offset, offset + batchSize);
      offset += batchSize;
      return batch;
    },
  };
}

/**
 * Batch store helper function
 */
export async function batchStore<T>(
  items: BatchItem<T>[],
  storeFn: StoreFn<T>,
  config?: Partial<BatchConfig>
): Promise<{ results: BatchResult<T>[]; stats: BatchStats }> {
  const executor = new BatchExecutor(config);
  return executor.batchStore(items, storeFn);
}

/**
 * Batch retrieve helper function
 */
export async function batchRetrieve<T>(
  keys: string[],
  retrieveFn: RetrieveFn<T>,
  config?: Partial<BatchConfig>
): Promise<{ results: BatchResult<T>[]; stats: BatchStats }> {
  const executor = new BatchExecutor(config);
  return executor.batchRetrieve(keys, retrieveFn);
}

/**
 * Batch delete helper function
 */
export async function batchDelete(
  keys: string[],
  deleteFn: DeleteFn,
  config?: Partial<BatchConfig>
): Promise<{ results: BatchResult[]; stats: BatchStats }> {
  const executor = new BatchExecutor(config);
  return executor.batchDelete(keys, deleteFn);
}
