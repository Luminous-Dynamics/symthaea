// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Steward Zome Client
 *
 * Handles watershed definition, water rights, transfers, and disputes.
 *
 * @module @mycelix/sdk/clients/water/steward
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Watershed,
  DefineWatershedInput,
  WaterRight,
  RegisterWaterRightInput,
  TransferWaterRightInput,
  WaterDispute,
  FileDisputeInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface StewardClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: StewardClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Water Stewardship operations
 */
export class StewardClient extends ZomeClient {
  protected readonly zomeName = 'water_steward';

  constructor(client: AppClient, config: StewardClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Watersheds
  // ============================================================================

  async defineWatershed(input: DefineWatershedInput): Promise<Watershed> {
    const record = await this.callZomeOnce<HolochainRecord>('define_watershed', {
      name: input.name,
      description: input.description,
      boundary_geo_json: input.boundaryGeoJson,
      annual_yield_megaliters: input.annualYieldMegaliters,
    });
    return this.mapWatershed(record);
  }

  async getWatershed(watershedId: ActionHash): Promise<Watershed | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_watershed', watershedId);
    if (!record) return null;
    return this.mapWatershed(record);
  }

  async listWatersheds(): Promise<Watershed[]> {
    const records = await this.callZome<HolochainRecord[]>('list_watersheds', null);
    return records.map(r => this.mapWatershed(r));
  }

  // ============================================================================
  // Water Rights
  // ============================================================================

  async registerWaterRight(input: RegisterWaterRightInput): Promise<WaterRight> {
    const record = await this.callZomeOnce<HolochainRecord>('register_water_right', {
      watershed_id: input.watershedId,
      holder_did: input.holderDid,
      annual_allocation_megaliters: input.annualAllocationMegaliters,
      right_type: input.rightType,
      priority: input.priority ?? 5,
      purpose: input.purpose,
      conditions: input.conditions ?? '',
      expires_at: input.expiresAt,
    });
    return this.mapWaterRight(record);
  }

  async getWaterRight(rightId: ActionHash): Promise<WaterRight | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_water_right', rightId);
    if (!record) return null;
    return this.mapWaterRight(record);
  }

  async getRightsForWatershed(watershedId: ActionHash): Promise<WaterRight[]> {
    const records = await this.callZome<HolochainRecord[]>('get_rights_for_watershed', watershedId);
    return records.map(r => this.mapWaterRight(r));
  }

  async transferRight(input: TransferWaterRightInput): Promise<WaterRight> {
    const record = await this.callZomeOnce<HolochainRecord>('transfer_right', {
      right_id: input.rightId,
      to_did: input.toDid,
      reason: input.reason,
    });
    return this.mapWaterRight(record);
  }

  // ============================================================================
  // Disputes
  // ============================================================================

  async fileDispute(input: FileDisputeInput): Promise<WaterDispute> {
    const record = await this.callZomeOnce<HolochainRecord>('file_dispute', {
      watershed_id: input.watershedId,
      against_did: input.againstDid,
      right_ids: input.rightIds,
      description: input.description,
    });
    return this.mapDispute(record);
  }

  async getDispute(disputeId: ActionHash): Promise<WaterDispute | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_dispute', disputeId);
    if (!record) return null;
    return this.mapDispute(record);
  }

  async getDisputesForWatershed(watershedId: ActionHash): Promise<WaterDispute[]> {
    const records = await this.callZome<HolochainRecord[]>('get_disputes_for_watershed', watershedId);
    return records.map(r => this.mapDispute(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapWatershed(record: HolochainRecord): Watershed {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      name: entry.name,
      description: entry.description,
      stewardDid: entry.steward_did,
      boundaryGeoJson: entry.boundary_geo_json,
      annualYieldMegaliters: entry.annual_yield_megaliters,
      sourceCount: entry.source_count,
      activeRightsCount: entry.active_rights_count,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapWaterRight(record: HolochainRecord): WaterRight {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      watershedId: entry.watershed_id,
      holderDid: entry.holder_did,
      annualAllocationMegaliters: entry.annual_allocation_megaliters,
      rightType: entry.right_type,
      priority: entry.priority,
      purpose: entry.purpose,
      conditions: entry.conditions,
      active: entry.active,
      grantedAt: entry.granted_at,
      expiresAt: entry.expires_at,
    };
  }

  private mapDispute(record: HolochainRecord): WaterDispute {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      watershedId: entry.watershed_id,
      filedByDid: entry.filed_by_did,
      againstDid: entry.against_did,
      rightIds: entry.right_ids,
      description: entry.description,
      status: entry.status,
      resolution: entry.resolution,
      filedAt: entry.filed_at,
      resolvedAt: entry.resolved_at,
    };
  }
}
