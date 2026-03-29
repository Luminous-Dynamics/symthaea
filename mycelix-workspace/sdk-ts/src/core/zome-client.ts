// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Base Zome Client
 *
 * Abstract base class for all zome-specific clients, providing standardized
 * zome call patterns with error handling and retry support.
 *
 * @module @mycelix/sdk/core/zome-client
 */

import { SdkError, SdkErrorCode, ZomeCallError } from './errors.js';
import { withRetryAndTimeout, type RetryConfig } from './retry.js';
import { withGateRetry } from './consciousness-gate.js';

import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for a zome client
 */
export interface ZomeClientConfig {
  /** Role name (DNA alias) in the hApp */
  roleName: string;

  /** Zome name */
  zomeName: string;

  /** Timeout for zome calls in milliseconds (default: 30000) */
  timeout?: number;

  /** Retry configuration for failed calls */
  retry?: Partial<RetryConfig>;

  /**
   * Optional callback to refresh the consciousness credential.
   * When provided, write calls (callZomeOnce) automatically retry once
   * on expired credential errors via withGateRetry.
   */
  gateRefreshFn?: () => Promise<void>;
}

/**
 * Default zome client configuration
 */
const DEFAULT_ZOME_CONFIG: Pick<ZomeClientConfig, 'timeout' | 'retry'> = {
  timeout: 30000,
  retry: {
    maxRetries: 2,
    baseDelay: 500,
    maxDelay: 5000,
  },
};

/**
 * Base class for zome-specific clients
 *
 * Provides standardized patterns for calling zome functions with
 * error handling, retry logic, and response processing.
 *
 * @example
 * ```typescript
 * class MyZomeClient extends ZomeClient {
 *   protected readonly zomeName = 'my_zome';
 *
 *   async getItem(id: string): Promise<Item> {
 *     return this.callZome('get_item', id);
 *   }
 *
 *   async createItem(item: Item): Promise<ActionHash> {
 *     const record = await this.callZome<HolochainRecord>('create_item', item);
 *     return record.signed_action.hashed.hash;
 *   }
 * }
 * ```
 */
export abstract class ZomeClient {
  /** Holochain app client */
  protected readonly client: AppClient;

  /** Role name (DNA alias) */
  protected readonly roleName: string;

  /** Zome name - must be set by subclass */
  protected abstract readonly zomeName: string;

  /** Timeout for zome calls */
  protected readonly timeout: number;

  /** Retry configuration */
  protected readonly retryConfig: Partial<RetryConfig>;

  /** Optional consciousness gate credential refresh callback */
  protected readonly gateRefreshFn?: () => Promise<void>;

  constructor(client: AppClient, config: Partial<ZomeClientConfig>) {
    this.client = client;
    this.roleName = config.roleName ?? 'mycelix';
    this.timeout = config.timeout ?? DEFAULT_ZOME_CONFIG.timeout!;
    this.retryConfig = config.retry ?? DEFAULT_ZOME_CONFIG.retry!;
    this.gateRefreshFn = config.gateRefreshFn;
  }

  /**
   * Call a zome function with automatic retry and error handling
   *
   * @param fnName - Name of the zome function to call
   * @param payload - Payload to pass to the function
   * @returns Promise resolving to the function result
   * @throws ZomeCallError if the call fails
   *
   * @example
   * ```typescript
   * const result = await this.callZome<Item>('get_item', { id });
   * ```
   */
  protected async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await withRetryAndTimeout(
        async () => {
          return this.client.callZome({
            role_name: this.roleName,
            zome_name: this.zomeName,
            fn_name: fnName,
            payload,
          });
        },
        this.retryConfig,
        this.timeout
      );

      return result as T;
    } catch (error) {
      throw this.wrapZomeError(fnName, payload, error);
    }
  }

  /**
   * Call a zome function without retry (use for non-idempotent operations)
   *
   * @param fnName - Name of the zome function to call
   * @param payload - Payload to pass to the function
   * @returns Promise resolving to the function result
   * @throws ZomeCallError if the call fails
   */
  protected async callZomeOnce<T>(fnName: string, payload: unknown): Promise<T> {
    const doCall = async (): Promise<T> => {
      try {
        const result = await this.client.callZome({
          role_name: this.roleName,
          zome_name: this.zomeName,
          fn_name: fnName,
          payload,
        });

        return result as T;
      } catch (error) {
        throw this.wrapZomeError(fnName, payload, error);
      }
    };

    // When gateRefreshFn is configured, wrap writes with automatic
    // retry on expired consciousness credentials
    if (this.gateRefreshFn) {
      return withGateRetry(doCall, this.gateRefreshFn);
    }
    return doCall();
  }

  /**
   * Call a zome function and return null if not found
   *
   * Convenience method for optional/nullable results.
   *
   * @param fnName - Name of the zome function to call
   * @param payload - Payload to pass to the function
   * @returns Promise resolving to the result or null
   */
  protected async callZomeOrNull<T>(
    fnName: string,
    payload: unknown
  ): Promise<T | null> {
    try {
      const result = await this.callZome<T | null>(fnName, payload);
      return result;
    } catch (error) {
      if (error instanceof SdkError && error.code === SdkErrorCode.NOT_FOUND) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Extract entry from a Holochain record
   *
   * @param record - Holochain record
   * @returns Extracted entry data
   * @throws SdkError if record doesn't contain an entry
   */
  protected extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new SdkError(
        SdkErrorCode.INVALID_RESPONSE,
        'Record does not contain app entry',
        { record: { hash: record.signed_action.hashed.hash } }
      );
    }

    // Entry is nested in the Present variant
    return (record.entry as { Present: T }).Present;
  }

  /**
   * Extract entry from a Holochain record, returning null if not present
   */
  protected extractEntryOrNull<T>(record: HolochainRecord | null): T | null {
    if (!record) return null;

    if (!record.entry || !('Present' in record.entry)) {
      return null;
    }

    return (record.entry as { Present: T }).Present;
  }

  /**
   * Get the action hash from a record
   */
  protected getActionHash(record: HolochainRecord): Uint8Array {
    return record.signed_action.hashed.hash;
  }

  /**
   * Wrap an error as a ZomeCallError
   */
  private wrapZomeError(fnName: string, payload: unknown, error: unknown): SdkError {
    if (error instanceof SdkError) {
      // Enrich existing error with zome context
      return new ZomeCallError(
        this.zomeName,
        fnName,
        error.message,
        { ...error.details, payload }
      );
    }

    const message = error instanceof Error ? error.message : String(error);

    // Detect specific error types from message
    if (message.includes('entry not found') || message.includes('not found')) {
      return new SdkError(SdkErrorCode.NOT_FOUND, message, {
        zomeName: this.zomeName,
        fnName,
        payload,
        cause: error,
      });
    }

    return new ZomeCallError(this.zomeName, fnName, message, {
      cause: error,
      payload,
    });
  }
}

/**
 * Create a simple zome client for ad-hoc zome calls
 *
 * Useful for one-off calls or prototyping without creating a full client class.
 *
 * @param client - Holochain app client
 * @param roleName - Role name (DNA alias)
 * @param zomeName - Zome name
 * @param config - Optional configuration
 * @returns Object with callZome method
 *
 * @example
 * ```typescript
 * const zome = createZomeClient(client, 'myapp', 'my_zome');
 * const result = await zome.call<Item>('get_item', { id });
 * ```
 */
export function createZomeClient(
  client: AppClient,
  roleName: string,
  zomeName: string,
  config: Pick<ZomeClientConfig, 'timeout' | 'retry'> = {}
): {
  call: <T>(fnName: string, payload: unknown) => Promise<T>;
  callOnce: <T>(fnName: string, payload: unknown) => Promise<T>;
} {
  const timeout = config.timeout ?? DEFAULT_ZOME_CONFIG.timeout!;
  const retryConfig = config.retry ?? DEFAULT_ZOME_CONFIG.retry!;

  const wrapError = (fnName: string, error: unknown): SdkError => {
    if (error instanceof SdkError) return error;
    const message = error instanceof Error ? error.message : String(error);
    return new ZomeCallError(zomeName, fnName, message, { cause: error });
  };

  return {
    call: async <T>(fnName: string, payload: unknown): Promise<T> => {
      try {
        const result = await withRetryAndTimeout(
          async () => {
            return client.callZome({
              role_name: roleName,
              zome_name: zomeName,
              fn_name: fnName,
              payload,
            });
          },
          retryConfig,
          timeout
        );
        return result as T;
      } catch (error) {
        throw wrapError(fnName, error);
      }
    },

    callOnce: async <T>(fnName: string, payload: unknown): Promise<T> => {
      try {
        const result = await client.callZome({
          role_name: roleName,
          zome_name: zomeName,
          fn_name: fnName,
          payload,
        });
        return result as T;
      } catch (error) {
        throw wrapError(fnName, error);
      }
    },
  };
}
