// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Request Coalescing
 *
 * Deduplicates identical concurrent requests to reduce load.
 * If 100 users request the same resource simultaneously, only one
 * fetch is performed and the result is shared.
 */

import { createHash } from 'crypto';
import { getMetrics } from '../metrics';
import { getLogger } from '../logging';

/**
 * Pending request state
 */
interface PendingRequest<T> {
  promise: Promise<T>;
  startTime: number;
  waiters: number;
}

/**
 * Coalescing configuration
 */
export interface CoalescingConfig {
  /** Maximum time to keep requests pending in ms */
  maxPendingTime: number;
  /** Maximum number of waiters per request */
  maxWaiters: number;
  /** Generate cache key from request params */
  keyGenerator?: (...args: unknown[]) => string;
}

/**
 * Default configuration
 */
const defaultConfig: CoalescingConfig = {
  maxPendingTime: 30000,
  maxWaiters: 1000,
};

/**
 * Request Coalescer
 */
export class RequestCoalescer<T = unknown> {
  private pending: Map<string, PendingRequest<T>> = new Map();
  private config: Required<CoalescingConfig>;
  private name: string;
  private logger = getLogger();

  constructor(name: string, config: Partial<CoalescingConfig> = {}) {
    this.name = name;
    this.config = {
      maxPendingTime: config.maxPendingTime ?? defaultConfig.maxPendingTime,
      maxWaiters: config.maxWaiters ?? defaultConfig.maxWaiters,
      keyGenerator: config.keyGenerator ?? this.defaultKeyGenerator,
    };

    // Register metrics
    const metrics = getMetrics();
    metrics.createCounter('request_coalescing_hits_total', 'Coalesced request hits', ['coalescer']);
    metrics.createCounter('request_coalescing_misses_total', 'Coalesced request misses', ['coalescer']);
    metrics.createGauge('request_coalescing_pending', 'Pending coalesced requests', ['coalescer']);
  }

  /**
   * Execute with coalescing
   */
  async execute(fn: () => Promise<T>, ...keyArgs: unknown[]): Promise<T> {
    const key = this.config.keyGenerator(...keyArgs);
    const metrics = getMetrics();

    // Check for existing pending request
    const existing = this.pending.get(key);

    if (existing) {
      // Check if too many waiters
      if (existing.waiters >= this.config.maxWaiters) {
        this.logger.warn(`Coalescer ${this.name} hit max waiters`, {
          key,
          waiters: existing.waiters,
        });
        // Execute independently
        return fn();
      }

      // Check if pending too long
      if (Date.now() - existing.startTime > this.config.maxPendingTime) {
        this.logger.warn(`Coalescer ${this.name} pending too long`, {
          key,
          pendingTime: Date.now() - existing.startTime,
        });
        // Clean up and execute independently
        this.pending.delete(key);
        return fn();
      }

      // Join the existing request
      existing.waiters++;
      metrics.incCounter('request_coalescing_hits_total', { coalescer: this.name });

      this.logger.debug(`Coalescing request for ${this.name}`, {
        key,
        waiters: existing.waiters,
      });

      return existing.promise;
    }

    // Create new pending request
    metrics.incCounter('request_coalescing_misses_total', { coalescer: this.name });

    const pendingRequest: PendingRequest<T> = {
      promise: this.executeWithCleanup(key, fn),
      startTime: Date.now(),
      waiters: 1,
    };

    this.pending.set(key, pendingRequest);
    metrics.setGauge('request_coalescing_pending', this.pending.size, { coalescer: this.name });

    return pendingRequest.promise;
  }

  /**
   * Execute and clean up when done
   */
  private async executeWithCleanup(key: string, fn: () => Promise<T>): Promise<T> {
    try {
      const result = await fn();
      return result;
    } finally {
      this.pending.delete(key);
      getMetrics().setGauge('request_coalescing_pending', this.pending.size, { coalescer: this.name });
    }
  }

  /**
   * Default key generator
   */
  private defaultKeyGenerator = (...args: unknown[]): string => {
    const content = JSON.stringify(args);
    return createHash('sha256').update(content).digest('hex').slice(0, 16);
  };

  /**
   * Get pending count
   */
  getPendingCount(): number {
    return this.pending.size;
  }

  /**
   * Clear all pending requests (for testing)
   */
  clear(): void {
    this.pending.clear();
  }
}

/**
 * Coalesced function wrapper
 */
export function coalesce<T, Args extends unknown[]>(
  name: string,
  fn: (...args: Args) => Promise<T>,
  config?: Partial<CoalescingConfig>
): (...args: Args) => Promise<T> {
  const coalescer = new RequestCoalescer<T>(name, config);

  return (...args: Args): Promise<T> => {
    return coalescer.execute(() => fn(...args), ...args);
  };
}

/**
 * Decorator for coalescing class methods
 */
export function Coalesce(name?: string, config?: Partial<CoalescingConfig>) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;
    const coalescerName = name || `${target.constructor.name}.${propertyKey}`;
    const coalescer = new RequestCoalescer(coalescerName, config);

    descriptor.value = async function (...args: unknown[]) {
      return coalescer.execute(() => originalMethod.apply(this, args), ...args);
    };

    return descriptor;
  };
}

/**
 * Pre-built coalescers for common operations
 */
export class Coalescers {
  private static instances: Map<string, RequestCoalescer<unknown>> = new Map();

  /**
   * Get or create a coalescer
   */
  static get<T>(name: string, config?: Partial<CoalescingConfig>): RequestCoalescer<T> {
    if (!this.instances.has(name)) {
      this.instances.set(name, new RequestCoalescer<T>(name, config));
    }
    return this.instances.get(name) as RequestCoalescer<T>;
  }

  /**
   * Clear all coalescers
   */
  static clearAll(): void {
    for (const coalescer of this.instances.values()) {
      coalescer.clear();
    }
    this.instances.clear();
  }
}

/**
 * Batch request coalescer
 *
 * Collects multiple individual requests and executes them as a batch.
 */
export class BatchCoalescer<K, V> {
  private pending: Map<K, {
    resolve: (value: V) => void;
    reject: (error: Error) => void;
  }[]> = new Map();
  private timer: NodeJS.Timeout | null = null;
  private logger = getLogger();

  constructor(
    private name: string,
    private batchFn: (keys: K[]) => Promise<Map<K, V>>,
    private options: {
      maxBatchSize?: number;
      maxWaitTime?: number;
    } = {}
  ) {
    this.options = {
      maxBatchSize: options.maxBatchSize ?? 100,
      maxWaitTime: options.maxWaitTime ?? 10, // 10ms default
    };
  }

  /**
   * Load a single item (will be batched)
   */
  async load(key: K): Promise<V> {
    return new Promise((resolve, reject) => {
      if (!this.pending.has(key)) {
        this.pending.set(key, []);
      }
      this.pending.get(key)!.push({ resolve, reject });

      // Check if we should execute immediately
      if (this.pending.size >= this.options.maxBatchSize!) {
        this.execute();
      } else if (!this.timer) {
        // Schedule batch execution
        this.timer = setTimeout(() => this.execute(), this.options.maxWaitTime);
      }
    });
  }

  /**
   * Load multiple items
   */
  async loadMany(keys: K[]): Promise<Map<K, V>> {
    const results = await Promise.all(keys.map(k => this.load(k)));
    const resultMap = new Map<K, V>();
    keys.forEach((key, index) => {
      resultMap.set(key, results[index]);
    });
    return resultMap;
  }

  /**
   * Execute the batch
   */
  private async execute(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const batch = new Map(this.pending);
    this.pending.clear();

    if (batch.size === 0) return;

    const keys = Array.from(batch.keys());

    this.logger.debug(`Executing batch for ${this.name}`, { size: keys.length });

    try {
      const results = await this.batchFn(keys);

      for (const [key, callbacks] of batch) {
        const value = results.get(key);
        if (value !== undefined) {
          for (const { resolve } of callbacks) {
            resolve(value);
          }
        } else {
          for (const { reject } of callbacks) {
            reject(new Error(`Key not found in batch result: ${key}`));
          }
        }
      }
    } catch (error) {
      for (const callbacks of batch.values()) {
        for (const { reject } of callbacks) {
          reject(error as Error);
        }
      }
    }
  }

  /**
   * Clear pending requests
   */
  clear(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.pending.clear();
  }
}

/**
 * Create a DataLoader-style batcher
 */
export function createBatchLoader<K, V>(
  name: string,
  batchFn: (keys: K[]) => Promise<Map<K, V>>,
  options?: { maxBatchSize?: number; maxWaitTime?: number }
): BatchCoalescer<K, V> {
  return new BatchCoalescer(name, batchFn, options);
}

export default {
  RequestCoalescer,
  coalesce,
  Coalesce,
  Coalescers,
  BatchCoalescer,
  createBatchLoader,
};
