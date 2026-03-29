// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Registry Zome Client
 *
 * Handles asset registration and management.
 *
 * @module @mycelix/sdk/integrations/property/zomes/registry
 */

import { PropertySdkError } from '../types';

import type {
  Asset,
  AssetType,
  AssetStatus,
  RegisterAssetInput,
  Ownership,
  OwnershipShare,
  Valuation,
  RecordValuationInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Registry client
 */
export interface RegistryClientConfig {
  /** Role ID for the property DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: RegistryClientConfig = {
  roleId: 'property',
  zomeName: 'property',
};

/**
 * Client for asset registry operations
 *
 * @example
 * ```typescript
 * const registry = new RegistryClient(holochainClient);
 *
 * // Register a real estate asset
 * const asset = await registry.registerAsset({
 *   id: 'property-001',
 *   asset_type: 'Building',
 *   name: 'Community Center',
 *   description: 'Multi-purpose community building',
 *   owner_did: 'did:mycelix:community-dao',
 *   location: {
 *     latitude: 37.7749,
 *     longitude: -122.4194,
 *     address: '123 Main St',
 *     city: 'San Francisco',
 *     country: 'USA',
 *   },
 *   valuation: 500000,
 *   valuation_currency: 'USD',
 * });
 * ```
 */
export class RegistryClient {
  private readonly config: RegistryClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<RegistryClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
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

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new PropertySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Asset Operations
  // ============================================================================

  /**
   * Register a new asset
   *
   * @param input - Asset registration parameters
   * @returns The registered asset
   */
  async registerAsset(input: RegisterAssetInput): Promise<Asset> {
    const record = await this.call<HolochainRecord>('register_asset', input);
    return this.extractEntry<Asset>(record);
  }

  /**
   * Get an asset by ID
   *
   * @param assetId - Asset identifier
   * @returns The asset or null
   */
  async getAsset(assetId: string): Promise<Asset | null> {
    const record = await this.call<HolochainRecord | null>('get_asset', assetId);
    if (!record) return null;
    return this.extractEntry<Asset>(record);
  }

  /**
   * Get assets by owner
   *
   * @param ownerDid - Owner's DID
   * @returns Array of assets
   */
  async getAssetsByOwner(ownerDid: string): Promise<Asset[]> {
    const records = await this.call<HolochainRecord[]>('get_assets_by_owner', ownerDid);
    return records.map(r => this.extractEntry<Asset>(r));
  }

  /**
   * Get assets by type
   *
   * @param assetType - Asset type
   * @returns Array of assets
   */
  async getAssetsByType(assetType: AssetType): Promise<Asset[]> {
    const records = await this.call<HolochainRecord[]>('get_assets_by_type', assetType);
    return records.map(r => this.extractEntry<Asset>(r));
  }

  /**
   * Update asset metadata
   *
   * @param assetId - Asset identifier
   * @param metadata - New metadata (JSON string)
   * @returns Updated asset
   */
  async updateMetadata(assetId: string, metadata: string): Promise<Asset> {
    const record = await this.call<HolochainRecord>('update_asset_metadata', {
      asset_id: assetId,
      metadata,
    });
    return this.extractEntry<Asset>(record);
  }

  /**
   * Update asset status
   *
   * @param assetId - Asset identifier
   * @param status - New status
   * @returns Updated asset
   */
  async updateStatus(assetId: string, status: AssetStatus): Promise<Asset> {
    const record = await this.call<HolochainRecord>('update_asset_status', {
      asset_id: assetId,
      status,
    });
    return this.extractEntry<Asset>(record);
  }

  /**
   * Get asset history (all versions)
   *
   * @param assetId - Asset identifier
   * @returns Array of asset versions
   */
  async getAssetHistory(assetId: string): Promise<Asset[]> {
    const records = await this.call<HolochainRecord[]>('get_asset_history', assetId);
    return records.map(r => this.extractEntry<Asset>(r));
  }

  // ============================================================================
  // Ownership Operations
  // ============================================================================

  /**
   * Get ownership record for an asset
   *
   * @param assetId - Asset identifier
   * @returns Ownership records
   */
  async getOwnership(assetId: string): Promise<Ownership[]> {
    const records = await this.call<HolochainRecord[]>('get_ownership', assetId);
    return records.map(r => this.extractEntry<Ownership>(r));
  }

  /**
   * Get ownership shares summary
   *
   * @param assetId - Asset identifier
   * @returns Array of ownership shares
   */
  async getOwnershipShares(assetId: string): Promise<OwnershipShare[]> {
    return this.call<OwnershipShare[]>('get_ownership_shares', assetId);
  }

  /**
   * Verify ownership
   *
   * @param assetId - Asset identifier
   * @param ownerDid - Claimed owner DID
   * @returns Verification result
   */
  async verifyOwnership(
    assetId: string,
    ownerDid: string
  ): Promise<{ verified: boolean; percentage: number }> {
    return this.call<{ verified: boolean; percentage: number }>('verify_ownership', {
      asset_id: assetId,
      owner_did: ownerDid,
    });
  }

  // ============================================================================
  // Valuation Operations
  // ============================================================================

  /**
   * Record a valuation for an asset
   *
   * @param input - Valuation parameters
   * @returns The valuation record
   */
  async recordValuation(input: RecordValuationInput): Promise<Valuation> {
    const record = await this.call<HolochainRecord>('record_valuation', input);
    return this.extractEntry<Valuation>(record);
  }

  /**
   * Get current valuation
   *
   * @param assetId - Asset identifier
   * @returns Current valuation or null
   */
  async getCurrentValuation(assetId: string): Promise<Valuation | null> {
    const record = await this.call<HolochainRecord | null>('get_current_valuation', assetId);
    if (!record) return null;
    return this.extractEntry<Valuation>(record);
  }

  /**
   * Get valuation history
   *
   * @param assetId - Asset identifier
   * @returns Array of valuations
   */
  async getValuationHistory(assetId: string): Promise<Valuation[]> {
    const records = await this.call<HolochainRecord[]>('get_valuation_history', assetId);
    return records.map(r => this.extractEntry<Valuation>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick asset registration with auto-generated ID
   *
   * @param assetType - Asset type
   * @param name - Asset name
   * @param description - Description
   * @param ownerDid - Owner's DID
   * @returns The registered asset
   */
  async quickRegister(
    assetType: AssetType,
    name: string,
    description: string,
    ownerDid: string
  ): Promise<Asset> {
    const id = `asset-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.registerAsset({
      id,
      asset_type: assetType,
      name,
      description,
      owner_did: ownerDid,
    });
  }

  /**
   * Get total value of assets owned by a DID
   *
   * @param ownerDid - Owner's DID
   * @param currency - Currency to filter by (optional)
   * @returns Total valuation
   */
  async getTotalValue(ownerDid: string, currency?: string): Promise<number> {
    const assets = await this.getAssetsByOwner(ownerDid);
    let total = 0;

    for (const asset of assets) {
      const valuation = await this.getCurrentValuation(asset.id);
      if (valuation && (!currency || valuation.currency === currency)) {
        total += valuation.amount;
      }
    }

    return total;
  }

  /**
   * Check if asset has any liens
   *
   * @param assetId - Asset identifier
   * @returns True if asset has liens
   */
  async hasLiens(assetId: string): Promise<boolean> {
    const asset = await this.getAsset(assetId);
    return asset?.status === 'Encumbered';
  }
}
