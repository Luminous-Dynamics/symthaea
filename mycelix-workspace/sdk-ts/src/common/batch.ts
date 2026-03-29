// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common Batch Utilities
 *
 * Request batching and deduplication for SDK operations.
 *
 * @module @mycelix/sdk/common/batch
 */

/**
 * Batch configuration options
 */
export interface BatchOptions {
  /** Maximum batch size (default: 100) */
  maxBatchSize?: number;
  /** Delay before executing batch in milliseconds (default: 10) */
  batchDelayMs?: number;
  /** Whether to deduplicate requests with same key (default: true) */
  deduplicate?: boolean;
}

const DEFAULT_BATCH_OPTIONS: Required<BatchOptions> = {
  maxBatchSize: 100,
  batchDelayMs: 10,
  deduplicate: true,
};

/**
 * Pending request in the batch queue
 */
interface PendingRequest<K, V> {
  key: K;
  resolve: (value: V) => void;
  reject: (error: unknown) => void;
}

/**
 * Request batcher for combining multiple requests into batch operations
 *
 * Similar to DataLoader pattern for GraphQL, but for general SDK use.
 *
 * @example
 * ```typescript
 * const userBatcher = new RequestBatcher<string, User>(
 *   async (ids: string[]) => {
 *     // Fetch multiple users in one request
 *     const users = await api.getUsers(ids);
 *     return ids.map(id => users.find(u => u.id === id) ?? null);
 *   },
 *   { maxBatchSize: 50 }
 * );
 *
 * // Individual requests are automatically batched
 * const [user1, user2, user3] = await Promise.all([
 *   userBatcher.load('user-1'),
 *   userBatcher.load('user-2'),
 *   userBatcher.load('user-3'),
 * ]);
 * ```
 */
export class RequestBatcher<K, V> {
  private queue: PendingRequest<K, V>[] = [];
  private keyMap = new Map<K, Promise<V>>();
  private batchTimeout: ReturnType<typeof setTimeout> | null = null;
  private readonly options: Required<BatchOptions>;

  constructor(
    private readonly batchFn: (keys: K[]) => Promise<(V | Error)[]>,
    options: BatchOptions = {}
  ) {
    this.options = { ...DEFAULT_BATCH_OPTIONS, ...options };
  }

  /**
   * Load a single value by key
   */
  load(key: K): Promise<V> {
    // Check for duplicate request
    if (this.options.deduplicate) {
      const existing = this.keyMap.get(key);
      if (existing) return existing;
    }

    const promise = new Promise<V>((resolve, reject) => {
      this.queue.push({ key, resolve, reject });
      this.scheduleBatch();
    });

    if (this.options.deduplicate) {
      this.keyMap.set(key, promise);
    }

    return promise;
  }

  /**
   * Load multiple values by keys
   */
  loadMany(keys: K[]): Promise<(V | Error)[]> {
    return Promise.all(
      keys.map((key) =>
        this.load(key).catch((error) => error as Error)
      )
    );
  }

  /**
   * Clear the cache for a key (for deduplication)
   */
  clear(key: K): void {
    this.keyMap.delete(key);
  }

  /**
   * Clear all cached promises
   */
  clearAll(): void {
    this.keyMap.clear();
  }

  /**
   * Prime the cache with a known value
   */
  prime(key: K, value: V): void {
    if (this.options.deduplicate && !this.keyMap.has(key)) {
      this.keyMap.set(key, Promise.resolve(value));
    }
  }

  private scheduleBatch(): void {
    // Execute immediately if batch is full
    if (this.queue.length >= this.options.maxBatchSize) {
      this.executeBatch();
      return;
    }

    // Schedule batch execution
    if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.executeBatch();
      }, this.options.batchDelayMs);
    }
  }

  private executeBatch(): void {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }

    const batch = this.queue.splice(0, this.options.maxBatchSize);
    if (batch.length === 0) return;

    const keys = batch.map((req) => req.key);

    this.batchFn(keys)
      .then((results) => {
        if (results.length !== keys.length) {
          const error = new Error(
            `Batch function returned ${results.length} results for ${keys.length} keys`
          );
          batch.forEach((req) => req.reject(error));
          return;
        }

        batch.forEach((req, index) => {
          const result = results[index];
          if (result instanceof Error) {
            req.reject(result);
          } else {
            req.resolve(result);
          }
        });
      })
      .catch((error) => {
        batch.forEach((req) => req.reject(error));
      })
      .finally(() => {
        // Clear dedupe map for completed requests
        if (this.options.deduplicate) {
          keys.forEach((key) => this.keyMap.delete(key));
        }

        // Process any remaining items
        if (this.queue.length > 0) {
          this.scheduleBatch();
        }
      });
  }
}

/**
 * Batch multiple function calls with automatic grouping
 *
 * @example
 * ```typescript
 * const batchedFetch = createBatchedFunction<string, User>(
 *   async (ids) => {
 *     const response = await fetch(`/api/users?ids=${ids.join(',')}`);
 *     return response.json();
 *   },
 *   { maxBatchSize: 20, batchDelayMs: 5 }
 * );
 *
 * // Calls within the delay window are batched
 * const results = await Promise.all([
 *   batchedFetch('1'),
 *   batchedFetch('2'),
 *   batchedFetch('3'),
 * ]);
 * ```
 */
export function createBatchedFunction<K, V>(
  batchFn: (keys: K[]) => Promise<(V | Error)[]>,
  options: BatchOptions = {}
): (key: K) => Promise<V> {
  const batcher = new RequestBatcher<K, V>(batchFn, options);
  return (key: K) => batcher.load(key);
}

/**
 * Batch executor for arbitrary operations
 *
 * Groups operations that occur within a time window into a single batch.
 *
 * @example
 * ```typescript
 * const analytics = new BatchExecutor<AnalyticsEvent>(
 *   async (events) => {
 *     await api.sendAnalytics(events);
 *   },
 *   { maxBatchSize: 100, batchDelayMs: 1000 }
 * );
 *
 * // Events are automatically batched
 * analytics.add({ type: 'click', element: 'button' });
 * analytics.add({ type: 'view', page: '/home' });
 * ```
 */
export class BatchExecutor<T> {
  private batch: T[] = [];
  private batchTimeout: ReturnType<typeof setTimeout> | null = null;
  private readonly options: Required<BatchOptions>;

  constructor(
    private readonly executeFn: (items: T[]) => Promise<void>,
    options: BatchOptions = {}
  ) {
    this.options = { ...DEFAULT_BATCH_OPTIONS, ...options };
  }

  /**
   * Add an item to the batch
   */
  add(item: T): void {
    this.batch.push(item);

    if (this.batch.length >= this.options.maxBatchSize) {
      this.flush();
      return;
    }

    if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.flush();
      }, this.options.batchDelayMs);
    }
  }

  /**
   * Add multiple items to the batch
   */
  addMany(items: T[]): void {
    items.forEach((item) => this.add(item));
  }

  /**
   * Immediately execute pending batch
   */
  async flush(): Promise<void> {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }

    if (this.batch.length === 0) return;

    const items = this.batch.splice(0);
    await this.executeFn(items);
  }

  /**
   * Get number of pending items
   */
  get pending(): number {
    return this.batch.length;
  }
}

/**
 * Debounce multiple calls to a function, executing only the last one
 *
 * @example
 * ```typescript
 * const debouncedSearch = debounce(
 *   async (query: string) => await api.search(query),
 *   300
 * );
 *
 * // Only the last call executes
 * debouncedSearch('h');
 * debouncedSearch('he');
 * debouncedSearch('hel');
 * debouncedSearch('hell');
 * debouncedSearch('hello'); // Only this executes
 * ```
 */
export function debounce<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  delayMs: number
): (...args: TArgs) => Promise<TResult> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let pending: {
    resolve: (value: TResult) => void;
    reject: (error: unknown) => void;
  } | null = null;

  return (...args: TArgs): Promise<TResult> => {
    if (timeout) {
      clearTimeout(timeout);
    }

    return new Promise((resolve, reject) => {
      pending = { resolve, reject };

      timeout = setTimeout(async () => {
        timeout = null;
        try {
          const result = await fn(...args);
          pending?.resolve(result);
        } catch (error) {
          pending?.reject(error);
        }
        pending = null;
      }, delayMs);
    });
  };
}

/**
 * Throttle calls to a function, allowing at most one call per interval
 *
 * @example
 * ```typescript
 * const throttledSave = throttle(
 *   async (data: Data) => await api.save(data),
 *   1000
 * );
 *
 * // At most one save per second
 * throttledSave(data1); // Executes immediately
 * throttledSave(data2); // Ignored
 * throttledSave(data3); // Ignored
 * // After 1 second...
 * throttledSave(data4); // Executes
 * ```
 */
export function throttle<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  intervalMs: number
): (...args: TArgs) => Promise<TResult | undefined> {
  let lastCall = 0;
  let pending: Promise<TResult> | null = null;

  return async (...args: TArgs): Promise<TResult | undefined> => {
    const now = Date.now();

    if (now - lastCall >= intervalMs) {
      lastCall = now;
      pending = fn(...args);
      return pending;
    }

    return pending ?? undefined;
  };
}
