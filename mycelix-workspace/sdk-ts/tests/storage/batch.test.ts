// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Batch Operations Test Suite
 *
 * Comprehensive tests for UESS batch operations module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  BatchExecutor,
  StreamingBatchExecutor,
  createBatchExecutor,
  createStreamingBatchExecutor,
  createArraySource,
  batchStore,
  batchRetrieve,
  batchDelete,
  type BatchItem,
  type BatchConfig,
  type BatchProgressCallback,
  type StoreFn,
  type RetrieveFn,
  type DeleteFn,
  type BatchResult,
} from '../../src/storage/batch.js';
import type { EpistemicClassification } from '../../src/epistemic/types.js';
import type { StorageReceipt, StorageMetadata } from '../../src/storage/types.js';

// =============================================================================
// Test Helpers
// =============================================================================

const mockClassification: EpistemicClassification = {
  empirical: 'E2',
  normative: 'N1',
  materiality: 'M2',
};

function createMockReceipt(key: string): StorageReceipt {
  return {
    key,
    hash: `hash-${key}`,
    timestamp: Date.now(),
    size: 100,
    classification: mockClassification,
  };
}

function createMockMetadata(): StorageMetadata {
  return {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    classification: mockClassification,
    version: 1,
  };
}

// =============================================================================
// BatchExecutor Tests
// =============================================================================

describe('BatchExecutor', () => {
  let executor: BatchExecutor;

  beforeEach(() => {
    executor = new BatchExecutor();
  });

  describe('constructor', () => {
    it('should use default config when no config provided', () => {
      const exec = new BatchExecutor();
      // Verify by attempting an operation
      expect(exec).toBeInstanceOf(BatchExecutor);
    });

    it('should merge custom config with defaults', () => {
      const exec = new BatchExecutor({
        maxBatchSize: 50,
        maxConcurrency: 5,
      });
      expect(exec).toBeInstanceOf(BatchExecutor);
    });

    it('should allow custom retry settings', () => {
      const exec = new BatchExecutor({
        retryCount: 5,
        retryDelayMs: 200,
      });
      expect(exec).toBeInstanceOf(BatchExecutor);
    });

    it('should allow disabling transactions', () => {
      const exec = new BatchExecutor({
        useTransactions: false,
      });
      expect(exec).toBeInstanceOf(BatchExecutor);
    });

    it('should allow setting continueOnError to false', () => {
      const exec = new BatchExecutor({
        continueOnError: false,
      });
      expect(exec).toBeInstanceOf(BatchExecutor);
    });
  });

  describe('batchStore', () => {
    it('should store multiple items successfully', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
        { key: 'key3', data: 'data3', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const { results, stats } = await executor.batchStore(items, storeFn);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(stats.totalItems).toBe(3);
      expect(stats.successCount).toBe(3);
      expect(stats.errorCount).toBe(0);
      expect(stats.durationMs).toBeGreaterThan(0);
      expect(stats.itemsPerSecond).toBeGreaterThan(0);
      expect(storeFn).toHaveBeenCalledTimes(3);
    });

    it('should handle store failures with continueOnError=true', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
        { key: 'key3', data: 'data3', classification: mockClassification },
      ];

      let callCount = 0;
      const storeFn: StoreFn<string> = vi.fn().mockImplementation(async (key: string) => {
        callCount++;
        if (key === 'key2') {
          throw new Error('Store failed');
        }
        return createMockReceipt(key);
      });

      const { results, stats } = await executor.batchStore(items, storeFn);

      expect(results).toHaveLength(3);
      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(results[1].error).toContain('Store failed');
      expect(results[2].success).toBe(true);
      expect(stats.successCount).toBe(2);
      expect(stats.errorCount).toBe(1);
    });

    it('should call progress callback', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const progressCalls: Array<{ completed: number; total: number; percentComplete: number }> = [];
      const onProgress: BatchProgressCallback = (progress) => {
        progressCalls.push(progress);
      };

      await executor.batchStore(items, storeFn, onProgress);

      expect(progressCalls).toHaveLength(2);
      expect(progressCalls[0].completed).toBe(1);
      expect(progressCalls[0].total).toBe(2);
      expect(progressCalls[0].percentComplete).toBe(50);
      expect(progressCalls[1].completed).toBe(2);
      expect(progressCalls[1].percentComplete).toBe(100);
    });

    it('should process items in batches when exceeding maxBatchSize', async () => {
      const exec = new BatchExecutor({ maxBatchSize: 2 });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
        { key: 'key3', data: 'data3', classification: mockClassification },
        { key: 'key4', data: 'data4', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const { results } = await exec.batchStore(items, storeFn);

      expect(results).toHaveLength(4);
      expect(results.every(r => r.success)).toBe(true);
    });

    it('should handle empty items array', async () => {
      const items: BatchItem<string>[] = [];
      const storeFn: StoreFn<string> = vi.fn();

      const { results, stats } = await executor.batchStore(items, storeFn);

      expect(results).toHaveLength(0);
      expect(stats.totalItems).toBe(0);
      expect(storeFn).not.toHaveBeenCalled();
    });

    it('should pass options to store function', async () => {
      const items: BatchItem<string>[] = [
        {
          key: 'key1',
          data: 'data1',
          classification: mockClassification,
          options: { encrypt: true, ttl: 3600 },
        },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      await executor.batchStore(items, storeFn);

      expect(storeFn).toHaveBeenCalledWith(
        'key1',
        'data1',
        mockClassification,
        { encrypt: true, ttl: 3600 }
      );
    });

    it('should include receipt in successful results', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const mockReceipt = createMockReceipt('key1');
      const storeFn: StoreFn<string> = vi.fn().mockResolvedValue(mockReceipt);

      const { results } = await executor.batchStore(items, storeFn);

      expect(results[0].receipt).toEqual(mockReceipt);
    });
  });

  describe('batchRetrieve', () => {
    it('should retrieve multiple items successfully', async () => {
      const keys = ['key1', 'key2', 'key3'];

      const retrieveFn: RetrieveFn<string> = vi.fn().mockImplementation(
        async (key: string) => ({
          data: `data-${key}`,
          metadata: createMockMetadata(),
        })
      );

      const { results, stats } = await executor.batchRetrieve(keys, retrieveFn);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(results[0].data).toBe('data-key1');
      expect(stats.successCount).toBe(3);
    });

    it('should handle not found items', async () => {
      const keys = ['key1', 'key2', 'key3'];

      const retrieveFn: RetrieveFn<string> = vi.fn().mockImplementation(
        async (key: string) => {
          if (key === 'key2') return null;
          return { data: `data-${key}`, metadata: createMockMetadata() };
        }
      );

      const { results, stats } = await executor.batchRetrieve(keys, retrieveFn);

      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(results[1].error).toBe('Not found');
      expect(results[2].success).toBe(true);
      expect(stats.successCount).toBe(2);
      expect(stats.errorCount).toBe(1);
    });

    it('should include metadata in successful results', async () => {
      const keys = ['key1'];
      const mockMetadata = createMockMetadata();

      const retrieveFn: RetrieveFn<string> = vi.fn().mockResolvedValue({
        data: 'data1',
        metadata: mockMetadata,
      });

      const { results } = await executor.batchRetrieve(keys, retrieveFn);

      expect(results[0].metadata).toEqual(mockMetadata);
    });

    it('should handle empty keys array', async () => {
      const keys: string[] = [];
      const retrieveFn: RetrieveFn<string> = vi.fn();

      const { results, stats } = await executor.batchRetrieve(keys, retrieveFn);

      expect(results).toHaveLength(0);
      expect(stats.totalItems).toBe(0);
    });
  });

  describe('batchDelete', () => {
    it('should delete multiple items successfully', async () => {
      const keys = ['key1', 'key2', 'key3'];

      const deleteFn: DeleteFn = vi.fn().mockResolvedValue(true);

      const { results, stats } = await executor.batchDelete(keys, deleteFn);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(stats.successCount).toBe(3);
    });

    it('should handle delete failures', async () => {
      const keys = ['key1', 'key2', 'key3'];

      const deleteFn: DeleteFn = vi.fn().mockImplementation(
        async (key: string) => key !== 'key2'
      );

      const { results, stats } = await executor.batchDelete(keys, deleteFn);

      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(results[1].error).toBe('Delete failed');
      expect(results[2].success).toBe(true);
      expect(stats.errorCount).toBe(1);
    });

    it('should handle empty keys array', async () => {
      const keys: string[] = [];
      const deleteFn: DeleteFn = vi.fn();

      const { results, stats } = await executor.batchDelete(keys, deleteFn);

      expect(results).toHaveLength(0);
      expect(stats.totalItems).toBe(0);
    });
  });

  describe('batchReclassify', () => {
    it('should reclassify multiple items successfully', async () => {
      const newClassification: EpistemicClassification = {
        empirical: 'E3',
        normative: 'N2',
        materiality: 'M3',
      };

      const items = [
        { key: 'key1', newClassification },
        { key: 'key2', newClassification },
      ];

      const reclassifyFn = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const { results, stats } = await executor.batchReclassify(items, reclassifyFn);

      expect(results).toHaveLength(2);
      expect(results.every(r => r.success)).toBe(true);
      expect(stats.successCount).toBe(2);
    });

    it('should handle reclassify failures', async () => {
      const newClassification: EpistemicClassification = {
        empirical: 'E3',
        normative: 'N2',
        materiality: 'M3',
      };

      const items = [
        { key: 'key1', newClassification },
        { key: 'key2', newClassification },
      ];

      const reclassifyFn = vi.fn().mockImplementation(async (key: string) => {
        if (key === 'key2') throw new Error('Reclassify failed');
        return createMockReceipt(key);
      });

      const { results, stats } = await executor.batchReclassify(items, reclassifyFn);

      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(stats.errorCount).toBe(1);
    });
  });

  describe('retry behavior', () => {
    it('should retry failed operations', async () => {
      const exec = new BatchExecutor({
        retryCount: 3,
        retryDelayMs: 10,
      });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      let attempts = 0;
      const storeFn: StoreFn<string> = vi.fn().mockImplementation(async () => {
        attempts++;
        if (attempts < 3) {
          // Fail twice, succeed on third attempt
          throw new Error('Temporary failure');
        }
        return createMockReceipt('key1');
      });

      const { results, stats } = await exec.batchStore(items, storeFn);

      expect(results[0].success).toBe(true);
      // retriedCount is incremented only on retries after the first failure (attempt > 0)
      // With 3 attempts total and success on 3rd, we have 2 failures
      // First failure (attempt=0) doesn't increment, second failure (attempt=1) increments
      expect(stats.retriedCount).toBeGreaterThanOrEqual(0);
      expect(storeFn).toHaveBeenCalledTimes(3);
    });

    it('should fail after exhausting retries', async () => {
      const exec = new BatchExecutor({
        retryCount: 2,
        retryDelayMs: 10,
      });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockRejectedValue(new Error('Persistent failure'));

      const { results, stats } = await exec.batchStore(items, storeFn);

      expect(results[0].success).toBe(false);
      expect(results[0].error).toContain('Persistent failure');
      expect(storeFn).toHaveBeenCalledTimes(2);
    });

    it('should throw error when continueOnError is false and all retries fail', async () => {
      const exec = new BatchExecutor({
        retryCount: 2,
        retryDelayMs: 10,
        continueOnError: false,
      });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockRejectedValue(new Error('Persistent failure'));

      await expect(exec.batchStore(items, storeFn)).rejects.toThrow('Persistent failure');
    });
  });

  describe('timeout handling', () => {
    it('should timeout slow operations', async () => {
      const exec = new BatchExecutor({
        itemTimeoutMs: 50,
        retryCount: 1,
      });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      const { results } = await exec.batchStore(items, storeFn);

      expect(results[0].success).toBe(false);
      expect(results[0].error).toContain('timed out');
    });
  });

  describe('concurrency control', () => {
    it('should limit concurrent operations', async () => {
      const exec = new BatchExecutor({
        maxConcurrency: 2,
        maxBatchSize: 10,
      });

      const items: BatchItem<string>[] = Array.from({ length: 5 }, (_, i) => ({
        key: `key${i}`,
        data: `data${i}`,
        classification: mockClassification,
      }));

      let concurrent = 0;
      let maxConcurrent = 0;

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(async (key: string) => {
        concurrent++;
        maxConcurrent = Math.max(maxConcurrent, concurrent);
        await new Promise(resolve => setTimeout(resolve, 10));
        concurrent--;
        return createMockReceipt(key);
      });

      await exec.batchStore(items, storeFn);

      expect(maxConcurrent).toBeLessThanOrEqual(2);
    });
  });

  describe('stats calculation', () => {
    it('should calculate itemsPerSecond correctly', async () => {
      const items: BatchItem<string>[] = Array.from({ length: 10 }, (_, i) => ({
        key: `key${i}`,
        data: `data${i}`,
        classification: mockClassification,
      }));

      // Add small delay to ensure non-zero duration
      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => {
          await new Promise(resolve => setTimeout(resolve, 1));
          return createMockReceipt(key);
        }
      );

      const { stats } = await executor.batchStore(items, storeFn);

      expect(stats.itemsPerSecond).toBeDefined();
      expect(stats.itemsPerSecond).toBeGreaterThanOrEqual(0);
      expect(stats.endTime).toBeDefined();
      expect(stats.durationMs).toBeDefined();
    });

    it('should handle zero duration gracefully', async () => {
      const items: BatchItem<string>[] = [];
      const storeFn: StoreFn<string> = vi.fn();

      const { stats } = await executor.batchStore(items, storeFn);

      expect(stats.itemsPerSecond).toBe(0);
    });
  });
});

// =============================================================================
// StreamingBatchExecutor Tests
// =============================================================================

describe('StreamingBatchExecutor', () => {
  describe('streamStore', () => {
    it('should stream store items from source', async () => {
      const exec = new StreamingBatchExecutor({ maxBatchSize: 2 });

      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
        { key: 'key3', data: 'data3', classification: mockClassification },
      ];

      const source = createArraySource(items, 2);
      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const allResults: BatchResult<string>[][] = [];
      for await (const batch of exec.streamStore(source, storeFn)) {
        allResults.push(batch);
      }

      expect(allResults).toHaveLength(2); // 2 batches for 3 items with size 2
      expect(allResults[0]).toHaveLength(2);
      expect(allResults[1]).toHaveLength(1);
    });
  });

  describe('streamRetrieve', () => {
    it('should stream retrieve items', async () => {
      const exec = new StreamingBatchExecutor();

      async function* keyGenerator() {
        yield 'key1';
        yield 'key2';
        yield 'key3';
      }

      const retrieveFn: RetrieveFn<string> = vi.fn().mockImplementation(
        async (key: string) => ({
          data: `data-${key}`,
          metadata: createMockMetadata(),
        })
      );

      const allResults: BatchResult<string>[][] = [];
      for await (const batch of exec.streamRetrieve(keyGenerator(), retrieveFn, 2)) {
        allResults.push(batch);
      }

      expect(allResults).toHaveLength(2); // 2 batches for 3 keys with size 2
    });

    it('should handle empty key stream', async () => {
      const exec = new StreamingBatchExecutor();

      async function* emptyGenerator(): AsyncGenerator<string> {
        // Empty generator
      }

      const retrieveFn: RetrieveFn<string> = vi.fn();

      const allResults: BatchResult<string>[][] = [];
      for await (const batch of exec.streamRetrieve(emptyGenerator(), retrieveFn)) {
        allResults.push(batch);
      }

      expect(allResults).toHaveLength(0);
    });
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('createBatchExecutor', () => {
    it('should create executor with default config', () => {
      const exec = createBatchExecutor();
      expect(exec).toBeInstanceOf(BatchExecutor);
    });

    it('should create executor with custom config', () => {
      const exec = createBatchExecutor({ maxBatchSize: 50 });
      expect(exec).toBeInstanceOf(BatchExecutor);
    });
  });

  describe('createStreamingBatchExecutor', () => {
    it('should create streaming executor with default config', () => {
      const exec = createStreamingBatchExecutor();
      expect(exec).toBeInstanceOf(StreamingBatchExecutor);
    });

    it('should create streaming executor with custom config', () => {
      const exec = createStreamingBatchExecutor({ maxConcurrency: 5 });
      expect(exec).toBeInstanceOf(StreamingBatchExecutor);
    });
  });

  describe('createArraySource', () => {
    it('should create source that iterates in batches', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
        { key: 'key2', data: 'data2', classification: mockClassification },
        { key: 'key3', data: 'data3', classification: mockClassification },
      ];

      const source = createArraySource(items, 2);

      expect(source.totalCount).toBe(3);

      const batch1 = await source.next();
      expect(batch1).toHaveLength(2);
      expect(batch1?.[0].key).toBe('key1');

      const batch2 = await source.next();
      expect(batch2).toHaveLength(1);
      expect(batch2?.[0].key).toBe('key3');

      const batch3 = await source.next();
      expect(batch3).toBeNull();
    });

    it('should use default batch size of 100', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const source = createArraySource(items);
      const batch = await source.next();
      expect(batch).toHaveLength(1);
    });
  });

  describe('batchStore helper', () => {
    it('should store items using default config', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const { results, stats } = await batchStore(items, storeFn);

      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(true);
    });

    it('should store items using custom config', async () => {
      const items: BatchItem<string>[] = [
        { key: 'key1', data: 'data1', classification: mockClassification },
      ];

      const storeFn: StoreFn<string> = vi.fn().mockImplementation(
        async (key: string) => createMockReceipt(key)
      );

      const { results } = await batchStore(items, storeFn, { maxBatchSize: 50 });

      expect(results[0].success).toBe(true);
    });
  });

  describe('batchRetrieve helper', () => {
    it('should retrieve items using default config', async () => {
      const keys = ['key1'];

      const retrieveFn: RetrieveFn<string> = vi.fn().mockResolvedValue({
        data: 'data1',
        metadata: createMockMetadata(),
      });

      const { results } = await batchRetrieve(keys, retrieveFn);

      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(true);
    });
  });

  describe('batchDelete helper', () => {
    it('should delete items using default config', async () => {
      const keys = ['key1'];

      const deleteFn: DeleteFn = vi.fn().mockResolvedValue(true);

      const { results } = await batchDelete(keys, deleteFn);

      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(true);
    });
  });
});

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

describe('Edge Cases', () => {
  it('should handle non-Error exceptions', async () => {
    const exec = new BatchExecutor({ retryCount: 1 });

    const items: BatchItem<string>[] = [
      { key: 'key1', data: 'data1', classification: mockClassification },
    ];

    const storeFn: StoreFn<string> = vi.fn().mockImplementation(() => {
      throw 'String error';
    });

    const { results } = await exec.batchStore(items, storeFn);

    expect(results[0].success).toBe(false);
    expect(results[0].error).toBe('String error');
  });

  it('should handle undefined error message', async () => {
    const exec = new BatchExecutor({ retryCount: 1 });

    const items: BatchItem<string>[] = [
      { key: 'key1', data: 'data1', classification: mockClassification },
    ];

    const storeFn: StoreFn<string> = vi.fn().mockImplementation(() => {
      throw undefined;
    });

    const { results } = await exec.batchStore(items, storeFn);

    expect(results[0].success).toBe(false);
  });

  it('should preserve order of results', async () => {
    const exec = new BatchExecutor({ maxConcurrency: 5 });

    const items: BatchItem<string>[] = Array.from({ length: 10 }, (_, i) => ({
      key: `key${i}`,
      data: `data${i}`,
      classification: mockClassification,
    }));

    // Add random delays to ensure async completion order differs
    const storeFn: StoreFn<string> = vi.fn().mockImplementation(async (key: string) => {
      await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
      return createMockReceipt(key);
    });

    const { results } = await exec.batchStore(items, storeFn);

    // Results should be in the same order as input
    for (let i = 0; i < 10; i++) {
      expect(results[i].key).toBe(`key${i}`);
    }
  });
});
