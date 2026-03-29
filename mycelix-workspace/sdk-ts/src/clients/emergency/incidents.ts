// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Incidents Zome Client
 *
 * Handles disaster declaration, status tracking, and lifecycle management.
 *
 * @module @mycelix/sdk/clients/emergency/incidents
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Disaster,
  DeclareDisasterInput,
  UpdateDisasterStatusInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface IncidentsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: IncidentsClientConfig = {
  roleName: 'civic',
};

/**
 * Client for Incident/Disaster operations
 */
export class IncidentsClient extends ZomeClient {
  protected readonly zomeName = 'emergency_incidents';

  constructor(client: AppClient, config: IncidentsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async declareDisaster(input: DeclareDisasterInput): Promise<Disaster> {
    const record = await this.callZomeOnce<HolochainRecord>('declare_disaster', {
      title: input.title,
      description: input.description,
      disaster_type: input.disasterType,
      severity: input.severity,
      latitude: input.latitude,
      longitude: input.longitude,
      radius_km: input.radiusKm,
      estimated_affected: input.estimatedAffected,
    });
    return this.mapDisaster(record);
  }

  async getDisaster(disasterId: ActionHash): Promise<Disaster | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_disaster', disasterId);
    if (!record) return null;
    return this.mapDisaster(record);
  }

  async getActiveDisasters(): Promise<Disaster[]> {
    const records = await this.callZome<HolochainRecord[]>('get_active_disasters', null);
    return records.map(r => this.mapDisaster(r));
  }

  async updateDisasterStatus(input: UpdateDisasterStatusInput): Promise<Disaster> {
    const record = await this.callZomeOnce<HolochainRecord>('update_disaster_status', {
      disaster_id: input.disasterId,
      status: input.status,
      severity: input.severity,
      description: input.description,
      radius_km: input.radiusKm,
      estimated_affected: input.estimatedAffected,
    });
    return this.mapDisaster(record);
  }

  async endDisaster(disasterId: ActionHash): Promise<Disaster> {
    const record = await this.callZomeOnce<HolochainRecord>('end_disaster', disasterId);
    return this.mapDisaster(record);
  }

  async getDisasterHistory(): Promise<Disaster[]> {
    const records = await this.callZome<HolochainRecord[]>('get_disaster_history', null);
    return records.map(r => this.mapDisaster(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapDisaster(record: HolochainRecord): Disaster {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      title: entry.title,
      description: entry.description,
      disasterType: entry.disaster_type,
      severity: entry.severity,
      declaredByDid: entry.declared_by_did,
      latitude: entry.latitude,
      longitude: entry.longitude,
      radiusKm: entry.radius_km,
      estimatedAffected: entry.estimated_affected,
      status: entry.status,
      declaredAt: entry.declared_at,
      updatedAt: entry.updated_at,
      endedAt: entry.ended_at,
    };
  }
}
