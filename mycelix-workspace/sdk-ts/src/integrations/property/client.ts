// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Property Client
 *
 * Unified client for the Property hApp SDK, providing access to
 * asset registry, transfers, liens, and commons management.
 *
 * @module @mycelix/sdk/integrations/property
 */

import {
  type AppClient,
  AppWebsocket,

} from '@holochain/client';

import { PropertySdkError } from './types';
import { CommonsClient } from './zomes/commons';
import { LienClient } from './zomes/lien';
import { RegistryClient } from './zomes/registry';
import { TransferClient } from './zomes/transfer';

import type {
  Asset,
  Transfer,
  Lien,

  OwnershipShare,
  Valuation,
  CollateralPledge,
  PledgeCollateralInput,
  OwnershipVerification,
  PropertyEvent,
  PropertyEventType,
} from './types';


/**
 * Configuration for the Property client
 */
export interface PropertyClientConfig {
  /** Role ID for the property DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: PropertyClientConfig = {
  roleId: 'property',
  zomeName: 'property',
};

/**
 * Connection options for creating a new client
 */
export interface PropertyConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Optional app client constructor args */
  appClientArgs?: Record<string, unknown>;
}

/**
 * Unified Mycelix Property Client
 *
 * Provides access to all property functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixPropertyClient } from '@mycelix/sdk/integrations/property';
 *
 * // Connect to Holochain
 * const property = await MycelixPropertyClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Register an asset
 * const asset = await property.registry.registerAsset({
 *   id: 'property-001',
 *   asset_type: 'Building',
 *   name: 'Community Center',
 *   description: 'Multi-purpose building',
 *   owner_did: 'did:mycelix:community-dao',
 *   valuation: 500000,
 *   valuation_currency: 'USD',
 * });
 *
 * // Initiate a transfer
 * const transfer = await property.transfers.initiateTransfer({
 *   asset_id: 'property-001',
 *   to_owner: 'did:mycelix:new-owner',
 *   percentage: 100,
 *   consideration: 600000,
 *   currency: 'USD',
 *   use_escrow: true,
 * });
 *
 * // Create a commons
 * const commons = await property.commons.createCommons({
 *   id: 'community-garden',
 *   name: 'Community Garden',
 *   description: 'Shared garden space',
 *   commons_type: 'Land',
 *   dao_id: 'garden-dao',
 *   governance_rules: '{"maxPlots": 50}',
 *   created_by: 'did:mycelix:founder',
 * });
 * ```
 */
export class MycelixPropertyClient {
  /** Asset registry operations */
  public readonly registry: RegistryClient;

  /** Ownership transfer operations */
  public readonly transfers: TransferClient;

  /** Lien and encumbrance operations */
  public readonly liens: LienClient;

  /** Commons and shared resource operations */
  public readonly commons: CommonsClient;

  private readonly config: PropertyClientConfig;

  /**
   * Create a property client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<PropertyClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize sub-clients
    this.registry = new RegistryClient(client, this.config);
    this.transfers = new TransferClient(client, this.config);
    this.liens = new LienClient(client, this.config);
    this.commons = new CommonsClient(client, this.config);
  }

  /**
   * Connect to Holochain and create a property client
   *
   * @param options - Connection options
   * @returns Connected property client
   *
   * @example
   * ```typescript
   * const property = await MycelixPropertyClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(
    options: PropertyConnectionOptions
  ): Promise<MycelixPropertyClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
        ...options.appClientArgs,
      });

      return new MycelixPropertyClient(client);
    } catch (error) {
      throw new PropertySdkError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Create a property client from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Property client
   */
  static fromClient(
    client: AppClient,
    config: Partial<PropertyClientConfig> = {}
  ): MycelixPropertyClient {
    return new MycelixPropertyClient(client, config);
  }

  // ============================================================================
  // Cross-hApp Integration
  // ============================================================================

  /**
   * Verify ownership for cross-hApp coordination
   *
   * @param assetId - Asset identifier
   * @param ownerDid - Claimed owner DID
   * @returns Ownership verification result
   */
  async verifyOwnership(assetId: string, ownerDid: string): Promise<OwnershipVerification> {
    const [verification, activeLiens] = await Promise.all([
      this.registry.verifyOwnership(assetId, ownerDid),
      this.liens.getActiveLiens(assetId),
    ]);

    const totalLienAmount = activeLiens.reduce((sum, l) => sum + l.amount, 0);

    return {
      asset_id: assetId,
      owner_did: ownerDid,
      verified: verification.verified,
      percentage: verification.percentage,
      active_liens: activeLiens.length,
      total_lien_amount: totalLienAmount,
      verified_at: Date.now() * 1000,
    };
  }

  /**
   * Pledge collateral for finance integration
   *
   * @param input - Collateral pledge parameters
   * @returns The collateral pledge
   */
  async pledgeCollateral(input: PledgeCollateralInput): Promise<CollateralPledge> {
    return this.callZome<CollateralPledge>('pledge_collateral', input);
  }

  /**
   * Release collateral after loan repayment
   *
   * @param pledgeId - Pledge identifier
   * @returns Released pledge
   */
  async releaseCollateral(pledgeId: string): Promise<CollateralPledge> {
    return this.callZome<CollateralPledge>('release_collateral', pledgeId);
  }

  /**
   * Get active collateral for a hApp
   *
   * @param happId - hApp identifier
   * @returns Array of active pledges
   */
  async getActiveCollateral(happId: string): Promise<CollateralPledge[]> {
    return this.callZome<CollateralPledge[]>('get_active_collateral', happId);
  }

  /**
   * Broadcast a property event
   *
   * @param eventType - Event type
   * @param assetId - Asset ID (optional)
   * @param did - DID involved (optional)
   * @param payload - Event payload (JSON string)
   * @returns The event
   */
  async broadcastEvent(
    eventType: PropertyEventType,
    assetId?: string,
    did?: string,
    payload: string = '{}'
  ): Promise<PropertyEvent> {
    return this.callZome<PropertyEvent>('broadcast_property_event', {
      event_type: eventType,
      asset_id: assetId,
      did,
      payload,
    });
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get comprehensive asset status
   *
   * Returns asset details, ownership, valuation, liens, and transfers.
   *
   * @param assetId - Asset identifier
   * @returns Asset status summary
   */
  async getAssetStatus(assetId: string): Promise<{
    asset: Asset | null;
    owners: OwnershipShare[];
    valuation: Valuation | null;
    liens: Lien[];
    pendingTransfers: Transfer[];
  }> {
    const [asset, owners, valuation, liens, transfers] = await Promise.all([
      this.registry.getAsset(assetId),
      this.registry.getOwnershipShares(assetId),
      this.registry.getCurrentValuation(assetId),
      this.liens.getActiveLiens(assetId),
      this.transfers.getTransfersForAsset(assetId),
    ]);

    const pendingTransfers = transfers.filter(
      t => t.status === 'Proposed' || t.status === 'Pending' || t.status === 'Accepted'
    );

    return {
      asset,
      owners,
      valuation,
      liens,
      pendingTransfers,
    };
  }

  /**
   * Get portfolio summary for a DID
   *
   * @param did - DID to query
   * @returns Portfolio summary
   */
  async getPortfolio(did: string): Promise<{
    assets: Asset[];
    totalValue: number;
    commonsCount: number;
    pendingTransfers: Transfer[];
  }> {
    const assets = await this.registry.getAssetsByOwner(did);
    const pendingTransfers = await this.transfers.getPendingTransfers(did);

    let totalValue = 0;
    for (const asset of assets) {
      const valuation = await this.registry.getCurrentValuation(asset.id);
      if (valuation) {
        totalValue += valuation.amount;
      }
    }

    // Count commons memberships
    // This would need a separate query in production
    const commonsCount = 0;

    return {
      assets,
      totalValue,
      commonsCount,
      pendingTransfers,
    };
  }

  /**
   * Check if asset can be transferred
   *
   * @param assetId - Asset identifier
   * @returns Whether asset can be transferred and any blockers
   */
  async canTransfer(assetId: string): Promise<{
    canTransfer: boolean;
    blockers: string[];
  }> {
    const blockers: string[] = [];

    const [asset, liens, transfers] = await Promise.all([
      this.registry.getAsset(assetId),
      this.liens.getActiveLiens(assetId),
      this.transfers.getTransfersForAsset(assetId),
    ]);

    if (!asset) {
      return { canTransfer: false, blockers: ['Asset not found'] };
    }

    if (asset.status === 'Disputed') {
      blockers.push('Asset is under dispute');
    }

    if (asset.status === 'Archived') {
      blockers.push('Asset is archived');
    }

    if (liens.length > 0) {
      blockers.push(`Asset has ${liens.length} active lien(s)`);
    }

    const activePending = transfers.filter(
      t => t.status === 'Proposed' || t.status === 'Pending' || t.status === 'Accepted'
    );
    if (activePending.length > 0) {
      blockers.push('Asset has pending transfer(s)');
    }

    return {
      canTransfer: blockers.length === 0,
      blockers,
    };
  }

  /**
   * Calculate equity in an asset
   *
   * @param assetId - Asset identifier
   * @returns Equity (valuation - liens)
   */
  async calculateEquity(assetId: string): Promise<number> {
    const [valuation, liens] = await Promise.all([
      this.registry.getCurrentValuation(assetId),
      this.liens.getActiveLiens(assetId),
    ]);

    if (!valuation) return 0;

    const totalLiens = liens.reduce((sum, l) => sum + l.amount, 0);
    return Math.max(0, valuation.amount - totalLiens);
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Helper to call zome functions
   */
  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new PropertySdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }
}
