// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Housing hApp Client (Phase 3)
 *
 * Complete TypeScript client for the Mycelix Housing hApp providing access
 * to all 4 housing zomes:
 *
 * - **units** - Building and unit registration, availability, occupancy
 * - **membership** - Applications, approvals, rent-to-own agreements
 * - **finances** - Monthly charges, payments, financial summaries
 * - **governance** - Meetings, resolutions, voting, elections
 *
 * @module @mycelix/sdk/clients/housing
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { FinancesClient } from './finances';
import { HousingGovernanceClient } from './governance';
import { MembershipClient } from './membership';
import { HousingError } from './types';
import { UnitsClient } from './units';

import type { ActionHash } from '../../generated/common';

// ============================================================================
// Client Configuration
// ============================================================================

export interface HousingClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface HousingConnectionOptions {
  url: string;
  timeout?: number;
  config?: HousingClientConfig;
}

const DEFAULT_CONFIG: Required<HousingClientConfig> = {
  roleName: 'commons',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Housing Client
// ============================================================================

/**
 * Unified Housing hApp Client
 *
 * @example
 * ```typescript
 * const housing = await HousingClient.connect({ url: 'ws://localhost:8888' });
 *
 * const building = await housing.units.registerBuilding({
 *   name: 'Harmony Co-Op',
 *   address: '123 Community Lane',
 *   buildingType: 'CoOp',
 *   totalUnits: 24,
 * });
 *
 * const member = await housing.membership.submitApplication({
 *   buildingId: building.id,
 *   name: 'Alice',
 * });
 *
 * const summary = await housing.finances.getFinancialSummary(building.id, '2026-02');
 * ```
 */
export class HousingClient {
  public readonly units: UnitsClient;
  public readonly membership: MembershipClient;
  public readonly finances: FinancesClient;
  public readonly governance: HousingGovernanceClient;

  private readonly _client: AppClient;
  private readonly _config: Required<HousingClientConfig>;

  private constructor(client: AppClient, config: HousingClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.units = new UnitsClient(client, baseConfig);
    this.membership = new MembershipClient(client, baseConfig);
    this.finances = new FinancesClient(client, baseConfig);
    this.governance = new HousingGovernanceClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[housing-sdk] Client initialized with 4 zome clients');
    }
  }

  static async connect(options: HousingConnectionOptions): Promise<HousingClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new HousingClient(client, options.config);
    } catch (error) {
      throw new HousingError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  static fromClient(client: AppClient, config: HousingClientConfig = {}): HousingClient {
    return new HousingClient(client, config);
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
   * Get comprehensive building overview
   */
  async getBuildingOverview(buildingId: ActionHash): Promise<{
    building: Awaited<ReturnType<UnitsClient['getBuilding']>>;
    units: Awaited<ReturnType<UnitsClient['getUnitsForBuilding']>>;
    members: Awaited<ReturnType<MembershipClient['getMembersForBuilding']>>;
    resolutions: Awaited<ReturnType<HousingGovernanceClient['getResolutionsForBuilding']>>;
  }> {
    const [building, units, members, resolutions] = await Promise.all([
      this.units.getBuilding(buildingId),
      this.units.getUnitsForBuilding(buildingId),
      this.membership.getMembersForBuilding(buildingId),
      this.governance.getResolutionsForBuilding(buildingId),
    ]);

    return { building, units, members, resolutions };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createHousingClient(client: AppClient, config?: HousingClientConfig): HousingClient {
  return HousingClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { UnitsClient, type UnitsClientConfig } from './units';
export { MembershipClient, type MembershipClientConfig } from './membership';
export { FinancesClient, type FinancesClientConfig } from './finances';
export { HousingGovernanceClient, type HousingGovernanceClientConfig } from './governance';
export * from './types';
export default HousingClient;
