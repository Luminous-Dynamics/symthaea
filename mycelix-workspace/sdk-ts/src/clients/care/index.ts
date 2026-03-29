// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Care hApp Client (Phase 3)
 *
 * Complete TypeScript client for the Mycelix Care hApp providing access
 * to all 3 care zomes:
 *
 * - **timebank** - Service offers, requests, time exchanges, balances
 * - **circles** - Care circle creation, membership, management
 * - **matching** - Request-to-offer matching and suggestions
 *
 * @module @mycelix/sdk/clients/care
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { CirclesClient } from './circles';
import { MatchingClient } from './matching';
import { TimebankClient } from './timebank';
import { CareError } from './types';

// ============================================================================
// Client Configuration
// ============================================================================

export interface CareClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface CareConnectionOptions {
  url: string;
  timeout?: number;
  config?: CareClientConfig;
}

const DEFAULT_CONFIG: Required<CareClientConfig> = {
  roleName: 'commons',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Care Client
// ============================================================================

/**
 * Unified Care hApp Client
 *
 * Provides access to all 3 care zomes through a single interface.
 *
 * @example
 * ```typescript
 * const care = await CareClient.connect({ url: 'ws://localhost:8888' });
 *
 * // Create a service offer
 * const offer = await care.timebank.createServiceOffer({
 *   title: 'Garden Help',
 *   description: 'Help with community garden',
 *   category: 'Gardening',
 *   estimatedMinutes: 120,
 * });
 *
 * // Create a care circle
 * const circle = await care.circles.createCircle({
 *   name: 'Neighborhood Support',
 *   description: 'Local mutual aid',
 * });
 * ```
 */
export class CareClient {
  public readonly timebank: TimebankClient;
  public readonly circles: CirclesClient;
  public readonly matching: MatchingClient;

  private readonly _client: AppClient;
  private readonly _config: Required<CareClientConfig>;

  private constructor(client: AppClient, config: CareClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.timebank = new TimebankClient(client, baseConfig);
    this.circles = new CirclesClient(client, baseConfig);
    this.matching = new MatchingClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[care-sdk] Client initialized with 3 zome clients');
    }
  }

  static async connect(options: CareConnectionOptions): Promise<CareClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new CareClient(client, options.config);
    } catch (error) {
      throw new CareError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  static fromClient(client: AppClient, config: CareClientConfig = {}): CareClient {
    return new CareClient(client, config);
  }

  getClient(): AppClient {
    return this._client;
  }

  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a CareClient from an existing AppClient
 */
export function createCareClient(client: AppClient, config?: CareClientConfig): CareClient {
  return CareClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { TimebankClient, type TimebankClientConfig } from './timebank';
export { CirclesClient, type CirclesClientConfig } from './circles';
export { MatchingClient, type MatchingClientConfig } from './matching';
export * from './types';
export default CareClient;
