/**
 * Base Client Utilities
 *
 * Common patterns for all Mycelix SDK clients.
 *
 * @module @mycelix/sdk/common/base-client
 */

import { SdkError } from './errors';

import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Base configuration for all clients
 */
export interface BaseClientConfig {
  /** Role ID for the DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

/**
 * Connection options for creating clients
 */
export interface ConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
}

/**
 * Base class for zome clients
 *
 * Provides common functionality for calling zome functions
 * and extracting entries from Holochain records.
 */
export abstract class BaseZomeClient {
  protected readonly roleId: string;
  protected readonly zomeName: string;

  constructor(
    protected readonly client: AppClient,
    config: BaseClientConfig
  ) {
    this.roleId = config.roleId;
    this.zomeName = config.zomeName;
  }

  /**
   * Call a zome function with error handling
   */
  protected async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.roleId,
        zome_name: this.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new SdkError(
        'ZOME_ERROR',
        `Failed to call ${this.zomeName}.${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  protected extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new SdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  /**
   * Extract entries from multiple Holochain records
   */
  protected extractEntries<T>(records: HolochainRecord[]): T[] {
    return records.map(r => this.extractEntry<T>(r));
  }

  /**
   * Call a zome function and extract entry from result
   */
  protected async callAndExtract<T>(fnName: string, payload: unknown): Promise<T> {
    const record = await this.call<HolochainRecord>(fnName, payload);
    return this.extractEntry<T>(record);
  }

  /**
   * Call a zome function and extract entries from result
   */
  protected async callAndExtractMany<T>(fnName: string, payload: unknown): Promise<T[]> {
    const records = await this.call<HolochainRecord[]>(fnName, payload);
    return this.extractEntries<T>(records);
  }

  /**
   * Call a zome function that may return null
   */
  protected async callOptional<T>(fnName: string, payload: unknown): Promise<T | null> {
    const record = await this.call<HolochainRecord | null>(fnName, payload);
    if (!record) return null;
    return this.extractEntry<T>(record);
  }
}

/**
 * Type for a unified client that composes multiple zome clients
 */
export interface UnifiedClientBase {
  /** Get the underlying Holochain client */
  getClient(): AppClient;
}

/**
 * Helper to create a timestamp in microseconds (Holochain format)
 */
export function createTimestamp(): number {
  return Date.now() * 1000;
}

/**
 * Helper to generate a unique ID
 */
export function generateId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}
