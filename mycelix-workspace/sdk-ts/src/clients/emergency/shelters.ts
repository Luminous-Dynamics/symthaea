/**
 * Shelters Zome Client
 *
 * Handles shelter registration, occupancy tracking, and search.
 *
 * @module @mycelix/sdk/clients/emergency/shelters
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { Shelter, RegisterShelterInput } from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface SheltersClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: SheltersClientConfig = {
  roleName: 'civic',
};

/**
 * Client for Shelter operations
 */
export class SheltersClient extends ZomeClient {
  protected readonly zomeName = 'emergency_shelters';

  constructor(client: AppClient, config: SheltersClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async registerShelter(input: RegisterShelterInput): Promise<Shelter> {
    const record = await this.callZomeOnce<HolochainRecord>('register_shelter', {
      disaster_id: input.disasterId,
      name: input.name,
      address: input.address,
      latitude: input.latitude,
      longitude: input.longitude,
      capacity: input.capacity,
      amenities: input.amenities ?? [],
      pet_friendly: input.petFriendly ?? false,
      accessible: input.accessible ?? false,
    });
    return this.mapShelter(record);
  }

  async getShelter(shelterId: ActionHash): Promise<Shelter | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_shelter', shelterId);
    if (!record) return null;
    return this.mapShelter(record);
  }

  async checkInPerson(shelterId: ActionHash, personIdentifier: string): Promise<Shelter> {
    const record = await this.callZomeOnce<HolochainRecord>('check_in_person', {
      shelter_id: shelterId,
      person_identifier: personIdentifier,
    });
    return this.mapShelter(record);
  }

  async checkOutPerson(shelterId: ActionHash, personIdentifier: string): Promise<Shelter> {
    const record = await this.callZomeOnce<HolochainRecord>('check_out_person', {
      shelter_id: shelterId,
      person_identifier: personIdentifier,
    });
    return this.mapShelter(record);
  }

  async findNearbyShelters(latitude: number, longitude: number, radiusKm: number): Promise<Shelter[]> {
    const records = await this.callZome<HolochainRecord[]>('find_nearby_shelters', {
      latitude,
      longitude,
      radius_km: radiusKm,
    });
    return records.map(r => this.mapShelter(r));
  }

  async getSheltersForDisaster(disasterId: ActionHash): Promise<Shelter[]> {
    const records = await this.callZome<HolochainRecord[]>('get_shelters_for_disaster', disasterId);
    return records.map(r => this.mapShelter(r));
  }

  async updateShelterStatus(shelterId: ActionHash, status: Shelter['status']): Promise<Shelter> {
    const record = await this.callZomeOnce<HolochainRecord>('update_shelter_status', {
      shelter_id: shelterId,
      status,
    });
    return this.mapShelter(record);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapShelter(record: HolochainRecord): Shelter {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      disasterId: entry.disaster_id,
      name: entry.name,
      address: entry.address,
      latitude: entry.latitude,
      longitude: entry.longitude,
      capacity: entry.capacity,
      currentOccupancy: entry.current_occupancy,
      managerDid: entry.manager_did,
      amenities: entry.amenities,
      petFriendly: entry.pet_friendly,
      accessible: entry.accessible,
      status: entry.status,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }
}
