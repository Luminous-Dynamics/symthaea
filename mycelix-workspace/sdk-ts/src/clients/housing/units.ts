// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Units Zome Client
 *
 * Handles building and unit registration, availability, and occupancy.
 *
 * @module @mycelix/sdk/clients/housing/units
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Building,
  RegisterBuildingInput,
  Unit,
  RegisterUnitInput,
  AssignOccupantInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface UnitsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: UnitsClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Building and Unit operations
 */
export class UnitsClient extends ZomeClient {
  protected readonly zomeName = 'housing_units';

  constructor(client: AppClient, config: UnitsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Buildings
  // ============================================================================

  async registerBuilding(input: RegisterBuildingInput): Promise<Building> {
    const record = await this.callZomeOnce<HolochainRecord>('register_building', {
      name: input.name,
      address: input.address,
      latitude: input.latitude,
      longitude: input.longitude,
      building_type: input.buildingType,
      total_units: input.totalUnits,
      year_built: input.yearBuilt,
      amenities: input.amenities ?? [],
    });
    return this.mapBuilding(record);
  }

  async getBuilding(buildingId: ActionHash): Promise<Building | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_building', buildingId);
    if (!record) return null;
    return this.mapBuilding(record);
  }

  async listBuildings(): Promise<Building[]> {
    const records = await this.callZome<HolochainRecord[]>('list_buildings', null);
    return records.map(r => this.mapBuilding(r));
  }

  // ============================================================================
  // Units
  // ============================================================================

  async registerUnit(input: RegisterUnitInput): Promise<Unit> {
    const record = await this.callZomeOnce<HolochainRecord>('register_unit', {
      building_id: input.buildingId,
      unit_number: input.unitNumber,
      floor_plan: input.floorPlan,
      square_meters: input.squareMeters,
      bedrooms: input.bedrooms,
      bathrooms: input.bathrooms,
      monthly_charge: input.monthlyCharge,
    });
    return this.mapUnit(record);
  }

  async getUnit(unitId: ActionHash): Promise<Unit | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_unit', unitId);
    if (!record) return null;
    return this.mapUnit(record);
  }

  async getAvailableUnits(buildingId: ActionHash): Promise<Unit[]> {
    const records = await this.callZome<HolochainRecord[]>('get_available_units', buildingId);
    return records.map(r => this.mapUnit(r));
  }

  async getUnitsForBuilding(buildingId: ActionHash): Promise<Unit[]> {
    const records = await this.callZome<HolochainRecord[]>('get_units_for_building', buildingId);
    return records.map(r => this.mapUnit(r));
  }

  async assignOccupant(input: AssignOccupantInput): Promise<Unit> {
    const record = await this.callZomeOnce<HolochainRecord>('assign_occupant', {
      unit_id: input.unitId,
      occupant_did: input.occupantDid,
      move_in_date: input.moveInDate,
    });
    return this.mapUnit(record);
  }

  async vacateUnit(unitId: ActionHash): Promise<Unit> {
    const record = await this.callZomeOnce<HolochainRecord>('vacate_unit', unitId);
    return this.mapUnit(record);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapBuilding(record: HolochainRecord): Building {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      name: entry.name,
      address: entry.address,
      latitude: entry.latitude,
      longitude: entry.longitude,
      ownerDid: entry.owner_did,
      buildingType: entry.building_type,
      totalUnits: entry.total_units,
      occupiedUnits: entry.occupied_units,
      yearBuilt: entry.year_built,
      amenities: entry.amenities,
      active: entry.active,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapUnit(record: HolochainRecord): Unit {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      unitNumber: entry.unit_number,
      floorPlan: entry.floor_plan,
      squareMeters: entry.square_meters,
      bedrooms: entry.bedrooms,
      bathrooms: entry.bathrooms,
      monthlyCharge: entry.monthly_charge,
      occupantDid: entry.occupant_did,
      status: entry.status,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }
}
