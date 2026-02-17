/**
 * Water hApp Client (Phase 3)
 *
 * Complete TypeScript client for the Mycelix Water hApp providing access
 * to all 3 water zomes:
 *
 * - **flow** - Source registration, share allocation, credit transfers, usage
 * - **purity** - Quality readings, potability checks, alerts
 * - **steward** - Watershed definition, water rights, disputes
 *
 * @module @mycelix/sdk/clients/water
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { FlowClient } from './flow';
import { PurityClient } from './purity';
import { StewardClient } from './steward';
import { WaterError } from './types';

import type { ActionHash } from '../../generated/common';

// ============================================================================
// Client Configuration
// ============================================================================

export interface WaterClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface WaterConnectionOptions {
  url: string;
  timeout?: number;
  config?: WaterClientConfig;
}

const DEFAULT_CONFIG: Required<WaterClientConfig> = {
  roleName: 'commons',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Water Client
// ============================================================================

/**
 * Unified Water hApp Client
 *
 * @example
 * ```typescript
 * const water = await WaterClient.connect({ url: 'ws://localhost:8888' });
 *
 * const source = await water.flow.registerSource({
 *   name: 'Community Well #3',
 *   description: 'Shared groundwater well',
 *   sourceType: 'Well',
 *   latitude: 32.95,
 *   longitude: -96.73,
 *   estimatedYieldLitersPerDay: 50000,
 * });
 *
 * const reading = await water.purity.submitReading({
 *   sourceId: source.id,
 *   ph: 7.2,
 *   turbidity: 0.5,
 *   tds: 250,
 * });
 * ```
 */
export class WaterClient {
  public readonly flow: FlowClient;
  public readonly purity: PurityClient;
  public readonly steward: StewardClient;

  private readonly _client: AppClient;
  private readonly _config: Required<WaterClientConfig>;

  private constructor(client: AppClient, config: WaterClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.flow = new FlowClient(client, baseConfig);
    this.purity = new PurityClient(client, baseConfig);
    this.steward = new StewardClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[water-sdk] Client initialized with 3 zome clients');
    }
  }

  static async connect(options: WaterConnectionOptions): Promise<WaterClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new WaterClient(client, options.config);
    } catch (error) {
      throw new WaterError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  static fromClient(client: AppClient, config: WaterClientConfig = {}): WaterClient {
    return new WaterClient(client, config);
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

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get comprehensive source status including latest readings and alerts
   */
  async getSourceStatus(sourceId: ActionHash): Promise<{
    source: Awaited<ReturnType<FlowClient['getSource']>>;
    shares: Awaited<ReturnType<FlowClient['getSharesForSource']>>;
    readings: Awaited<ReturnType<PurityClient['getReadingsForSource']>>;
    alerts: Awaited<ReturnType<PurityClient['getActiveAlerts']>>;
  }> {
    const [source, shares, readings, alerts] = await Promise.all([
      this.flow.getSource(sourceId),
      this.flow.getSharesForSource(sourceId),
      this.purity.getReadingsForSource(sourceId, 10),
      this.purity.getActiveAlerts(sourceId),
    ]);

    return { source, shares, readings, alerts };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createWaterClient(client: AppClient, config?: WaterClientConfig): WaterClient {
  return WaterClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { FlowClient, type FlowClientConfig } from './flow';
export { PurityClient, type PurityClientConfig } from './purity';
export { StewardClient, type StewardClientConfig } from './steward';
export * from './types';
export default WaterClient;
