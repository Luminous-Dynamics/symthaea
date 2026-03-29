// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Flow Zome Client
 *
 * Handles water source registration, share allocation, credit transfers, and usage.
 *
 * @module @mycelix/sdk/clients/water/flow
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  WaterSource,
  RegisterWaterSourceInput,
  WaterShare,
  AllocateSharesInput,
  TransferCreditsInput,
  WaterBalance,
  WaterUsageRecord,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface FlowClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: FlowClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Water Flow operations
 */
export class FlowClient extends ZomeClient {
  protected readonly zomeName = 'water_flow';

  constructor(client: AppClient, config: FlowClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Sources
  // ============================================================================

  async registerSource(input: RegisterWaterSourceInput): Promise<WaterSource> {
    const record = await this.callZomeOnce<HolochainRecord>('register_source', {
      name: input.name,
      description: input.description,
      source_type: input.sourceType,
      latitude: input.latitude,
      longitude: input.longitude,
      watershed_id: input.watershedId,
      estimated_yield_liters_per_day: input.estimatedYieldLitersPerDay,
    });
    return this.mapSource(record);
  }

  async getSource(sourceId: ActionHash): Promise<WaterSource | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_source', sourceId);
    if (!record) return null;
    return this.mapSource(record);
  }

  async getMySources(): Promise<WaterSource[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_sources', null);
    return records.map(r => this.mapSource(r));
  }

  // ============================================================================
  // Shares & Allocation
  // ============================================================================

  async allocateShares(input: AllocateSharesInput): Promise<WaterShare> {
    const record = await this.callZomeOnce<HolochainRecord>('allocate_shares', {
      source_id: input.sourceId,
      holder_did: input.holderDid,
      daily_allocation_liters: input.dailyAllocationLiters,
      priority: input.priority ?? 5,
      purpose: input.purpose,
      expires_at: input.expiresAt,
    });
    return this.mapShare(record);
  }

  async getSharesForSource(sourceId: ActionHash): Promise<WaterShare[]> {
    const records = await this.callZome<HolochainRecord[]>('get_shares_for_source', sourceId);
    return records.map(r => this.mapShare(r));
  }

  // ============================================================================
  // Credits & Usage
  // ============================================================================

  async transferCredits(input: TransferCreditsInput): Promise<void> {
    await this.callZomeOnce<void>('transfer_credits', {
      from_did: input.fromDid,
      to_did: input.toDid,
      source_id: input.sourceId,
      liters: input.liters,
      reason: input.reason,
    });
  }

  async getMyBalance(): Promise<WaterBalance> {
    const result = await this.callZome<any>('get_my_balance', null);
    return {
      did: result.did,
      totalDailyAllocation: result.total_daily_allocation,
      usedToday: result.used_today,
      remainingToday: result.remaining_today,
      transferableCredits: result.transferable_credits,
    };
  }

  async recordUsage(sourceId: ActionHash, litersUsed: number, purpose: WaterShare['purpose']): Promise<WaterUsageRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('record_usage', {
      source_id: sourceId,
      liters_used: litersUsed,
      purpose,
    });
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      sourceId: entry.source_id,
      userDid: entry.user_did,
      litersUsed: entry.liters_used,
      purpose: entry.purpose,
      recordedAt: entry.recorded_at,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapSource(record: HolochainRecord): WaterSource {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      name: entry.name,
      description: entry.description,
      sourceType: entry.source_type,
      ownerDid: entry.owner_did,
      latitude: entry.latitude,
      longitude: entry.longitude,
      watershedId: entry.watershed_id,
      estimatedYieldLitersPerDay: entry.estimated_yield_liters_per_day,
      availableAllocation: entry.available_allocation,
      status: entry.status,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapShare(record: HolochainRecord): WaterShare {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      sourceId: entry.source_id,
      holderDid: entry.holder_did,
      dailyAllocationLiters: entry.daily_allocation_liters,
      priority: entry.priority,
      purpose: entry.purpose,
      active: entry.active,
      grantedAt: entry.granted_at,
      expiresAt: entry.expires_at,
    };
  }
}
