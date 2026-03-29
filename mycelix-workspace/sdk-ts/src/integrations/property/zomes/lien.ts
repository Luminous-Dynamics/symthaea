// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Lien Zome Client
 *
 * Handles lien and encumbrance management on assets.
 *
 * @module @mycelix/sdk/integrations/property/zomes/lien
 */

import { PropertySdkError } from '../types';

import type {
  Lien,
  LienType,
  LienStatus,
  PlaceLienInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Lien client
 */
export interface LienClientConfig {
  /** Role ID for the property DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: LienClientConfig = {
  roleId: 'property',
  zomeName: 'property',
};

/**
 * Client for lien and encumbrance operations
 *
 * @example
 * ```typescript
 * const liens = new LienClient(holochainClient);
 *
 * // Place a mortgage lien
 * const lien = await liens.placeLien({
 *   asset_id: 'property-001',
 *   lien_type: 'Mortgage',
 *   amount: 400000,
 *   currency: 'USD',
 *   loan_id: 'loan-001',
 *   description: 'Primary mortgage',
 * });
 *
 * // Check liens on asset
 * const assetLiens = await liens.getLiensForAsset('property-001');
 *
 * // Release lien after payoff
 * await liens.releaseLien(lien.id);
 * ```
 */
export class LienClient {
  private readonly config: LienClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<LienClientConfig> = {}
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
  // Lien Operations
  // ============================================================================

  /**
   * Place a lien on an asset
   *
   * @param input - Lien parameters
   * @returns The created lien
   */
  async placeLien(input: PlaceLienInput): Promise<Lien> {
    const record = await this.call<HolochainRecord>('place_lien', input);
    return this.extractEntry<Lien>(record);
  }

  /**
   * Get a lien by ID
   *
   * @param lienId - Lien identifier
   * @returns The lien or null
   */
  async getLien(lienId: string): Promise<Lien | null> {
    const record = await this.call<HolochainRecord | null>('get_lien', lienId);
    if (!record) return null;
    return this.extractEntry<Lien>(record);
  }

  /**
   * Get liens for an asset
   *
   * @param assetId - Asset identifier
   * @returns Array of liens
   */
  async getLiensForAsset(assetId: string): Promise<Lien[]> {
    const records = await this.call<HolochainRecord[]>('get_liens_for_asset', assetId);
    return records.map(r => this.extractEntry<Lien>(r));
  }

  /**
   * Get active liens for an asset
   *
   * @param assetId - Asset identifier
   * @returns Array of active liens
   */
  async getActiveLiens(assetId: string): Promise<Lien[]> {
    const liens = await this.getLiensForAsset(assetId);
    return liens.filter(l => l.status === 'Active');
  }

  /**
   * Get liens by holder
   *
   * @param holderDid - Lien holder's DID
   * @returns Array of liens
   */
  async getLiensByHolder(holderDid: string): Promise<Lien[]> {
    const records = await this.call<HolochainRecord[]>('get_liens_by_holder', holderDid);
    return records.map(r => this.extractEntry<Lien>(r));
  }

  /**
   * Mark a lien as satisfied (debt paid)
   *
   * @param lienId - Lien identifier
   * @returns Updated lien
   */
  async satisfyLien(lienId: string): Promise<Lien> {
    const record = await this.call<HolochainRecord>('satisfy_lien', lienId);
    return this.extractEntry<Lien>(record);
  }

  /**
   * Release a lien (remove from asset)
   *
   * @param lienId - Lien identifier
   * @returns Released lien
   */
  async releaseLien(lienId: string): Promise<Lien> {
    const record = await this.call<HolochainRecord>('release_lien', lienId);
    return this.extractEntry<Lien>(record);
  }

  /**
   * Foreclose on a lien
   *
   * @param lienId - Lien identifier
   * @param reason - Foreclosure reason
   * @returns Updated lien
   */
  async forecloseLien(lienId: string, reason: string): Promise<Lien> {
    const record = await this.call<HolochainRecord>('foreclose_lien', {
      lien_id: lienId,
      reason,
    });
    return this.extractEntry<Lien>(record);
  }

  /**
   * Update lien priority
   *
   * @param lienId - Lien identifier
   * @param newPriority - New priority value
   * @returns Updated lien
   */
  async updatePriority(lienId: string, newPriority: number): Promise<Lien> {
    const record = await this.call<HolochainRecord>('update_lien_priority', {
      lien_id: lienId,
      priority: newPriority,
    });
    return this.extractEntry<Lien>(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Calculate total lien amount for an asset
   *
   * @param assetId - Asset identifier
   * @param currency - Currency to filter by (optional)
   * @returns Total lien amount
   */
  async getTotalLienAmount(assetId: string, currency?: string): Promise<number> {
    const liens = await this.getActiveLiens(assetId);
    return liens
      .filter(l => !currency || l.currency === currency)
      .reduce((sum, l) => sum + l.amount, 0);
  }

  /**
   * Check if asset is clear of liens
   *
   * @param assetId - Asset identifier
   * @returns True if no active liens
   */
  async isAssetClear(assetId: string): Promise<boolean> {
    const activeLiens = await this.getActiveLiens(assetId);
    return activeLiens.length === 0;
  }

  /**
   * Get liens sorted by priority
   *
   * @param assetId - Asset identifier
   * @returns Liens sorted by priority (lowest first)
   */
  async getLiensByPriority(assetId: string): Promise<Lien[]> {
    const liens = await this.getActiveLiens(assetId);
    return liens.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Get lien type description
   *
   * @param lienType - Lien type
   * @returns Human-readable description
   */
  getLienTypeDescription(lienType: LienType): string {
    const descriptions: Record<LienType, string> = {
      Mortgage: 'Secured loan against property',
      Tax: 'Government tax obligation',
      Mechanic: 'Unpaid contractor/labor claim',
      Judgment: 'Court-ordered debt collection',
      SecurityInterest: 'Secured creditor interest',
      Other: 'Other encumbrance',
    };
    return descriptions[lienType];
  }

  /**
   * Get lien status description
   *
   * @param status - Lien status
   * @returns Human-readable description
   */
  getStatusDescription(status: LienStatus): string {
    const descriptions: Record<LienStatus, string> = {
      Active: 'Lien is active and enforceable',
      Satisfied: 'Underlying debt has been paid',
      Released: 'Lien has been formally released',
      Foreclosed: 'Lien holder has initiated foreclosure',
    };
    return descriptions[status];
  }

  /**
   * Check if lien can be released
   *
   * @param lien - The lien
   * @returns True if lien can be released
   */
  canRelease(lien: Lien): boolean {
    return lien.status === 'Active' || lien.status === 'Satisfied';
  }
}
