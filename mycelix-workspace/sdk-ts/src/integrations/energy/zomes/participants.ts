// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Participants Zome Client
 *
 * Handles energy participant registration and management.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/participants
 */

import { EnergySdkError } from '../types';

import type {
  EnergyParticipant,
  RegisterParticipantInput,
  UpdateParticipantInput,
  ParticipantSearchParams,
  ParticipantStatistics,
  ProductionReading,
  ReportProductionInput,
  EnergySource,
  ParticipantType,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Participants client
 */
export interface ParticipantsClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: ParticipantsClientConfig = {
  roleId: 'energy',
  zomeName: 'participants',
};

/**
 * Client for energy participant operations
 *
 * @example
 * ```typescript
 * const participants = new ParticipantsClient(holochainClient);
 *
 * // Register as a prosumer
 * const participant = await participants.register({
 *   participant_type: 'Prosumer',
 *   sources: ['Solar'],
 *   capacity_kwh: 50,
 *   location: { lat: 32.95, lng: -96.73 },
 * });
 *
 * // Report production
 * await participants.reportProduction({
 *   participant_id: participant.id,
 *   production_kwh: 25,
 *   consumption_kwh: 18,
 *   source: 'Solar',
 * });
 * ```
 */
export class ParticipantsClient {
  private readonly config: ParticipantsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<ParticipantsClientConfig> = {}
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
      throw new EnergySdkError(
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
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Participant Operations
  // ============================================================================

  /**
   * Register a new energy participant
   *
   * @param input - Registration parameters
   * @returns The created participant
   */
  async register(input: RegisterParticipantInput): Promise<EnergyParticipant> {
    const record = await this.call<HolochainRecord>('register', {
      ...input,
      reputation_score: 0.5,
      total_produced_kwh: 0,
      total_consumed_kwh: 0,
      carbon_offset_kg: 0,
      active: true,
      registered_at: Date.now() * 1000,
      updated_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyParticipant>(record);
  }

  /**
   * Get a participant by ID
   *
   * @param participantId - Participant identifier
   * @returns The participant or null if not found
   */
  async getParticipant(participantId: string): Promise<EnergyParticipant | null> {
    const record = await this.call<HolochainRecord | null>('get_participant', participantId);
    if (!record) return null;
    return this.extractEntry<EnergyParticipant>(record);
  }

  /**
   * Get participant by DID
   *
   * @param did - Participant's DID
   * @returns The participant or null if not found
   */
  async getParticipantByDid(did: string): Promise<EnergyParticipant | null> {
    const record = await this.call<HolochainRecord | null>('get_participant_by_did', did);
    if (!record) return null;
    return this.extractEntry<EnergyParticipant>(record);
  }

  /**
   * Update participant information
   *
   * @param input - Update parameters
   * @returns The updated participant
   */
  async updateParticipant(input: UpdateParticipantInput): Promise<EnergyParticipant> {
    const record = await this.call<HolochainRecord>('update_participant', {
      ...input,
      updated_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyParticipant>(record);
  }

  /**
   * Get participants by type
   *
   * @param type - Participant type
   * @returns Array of participants
   */
  async getParticipantsByType(type: ParticipantType): Promise<EnergyParticipant[]> {
    const records = await this.call<HolochainRecord[]>('get_by_type', type);
    return records.map(r => this.extractEntry<EnergyParticipant>(r));
  }

  /**
   * Get participants by energy source
   *
   * @param source - Energy source
   * @returns Array of participants
   */
  async getParticipantsBySource(source: EnergySource): Promise<EnergyParticipant[]> {
    const records = await this.call<HolochainRecord[]>('get_by_source', source);
    return records.map(r => this.extractEntry<EnergyParticipant>(r));
  }

  /**
   * Search participants
   *
   * @param params - Search parameters
   * @returns Array of matching participants
   */
  async searchParticipants(params: ParticipantSearchParams): Promise<EnergyParticipant[]> {
    const records = await this.call<HolochainRecord[]>('search_participants', params);
    return records.map(r => this.extractEntry<EnergyParticipant>(r));
  }

  /**
   * Update capacity
   *
   * @param participantId - Participant identifier
   * @param capacityKwh - New capacity in kWh
   * @returns The updated participant
   */
  async updateCapacity(participantId: string, capacityKwh: number): Promise<EnergyParticipant> {
    const record = await this.call<HolochainRecord>('update_capacity', {
      participant_id: participantId,
      capacity_kwh: capacityKwh,
    });
    return this.extractEntry<EnergyParticipant>(record);
  }

  /**
   * Deactivate a participant
   *
   * @param participantId - Participant identifier
   * @returns The updated participant
   */
  async deactivate(participantId: string): Promise<EnergyParticipant> {
    return this.updateParticipant({
      participant_id: participantId,
      active: false,
    });
  }

  /**
   * Reactivate a participant
   *
   * @param participantId - Participant identifier
   * @returns The updated participant
   */
  async reactivate(participantId: string): Promise<EnergyParticipant> {
    return this.updateParticipant({
      participant_id: participantId,
      active: true,
    });
  }

  // ============================================================================
  // Production & Consumption
  // ============================================================================

  /**
   * Report energy production
   *
   * @param input - Production report parameters
   * @returns The production reading
   */
  async reportProduction(input: ReportProductionInput): Promise<ProductionReading> {
    const record = await this.call<HolochainRecord>('report_production', {
      ...input,
      timestamp: Date.now() * 1000,
      net_export_kwh: input.production_kwh - input.consumption_kwh,
      verified: false,
    });
    return this.extractEntry<ProductionReading>(record);
  }

  /**
   * Get production readings for a participant
   *
   * @param participantId - Participant identifier
   * @param limit - Maximum number of readings
   * @returns Array of production readings
   */
  async getProductionReadings(participantId: string, limit: number = 100): Promise<ProductionReading[]> {
    const records = await this.call<HolochainRecord[]>('get_production_readings', {
      participant_id: participantId,
      limit,
    });
    return records.map(r => this.extractEntry<ProductionReading>(r));
  }

  /**
   * Get participant statistics
   *
   * @param participantId - Participant identifier
   * @returns Participant statistics
   */
  async getStatistics(participantId: string): Promise<ParticipantStatistics> {
    return this.call<ParticipantStatistics>('get_participant_statistics', participantId);
  }

  /**
   * Get participants near a location
   *
   * @param lat - Latitude
   * @param lng - Longitude
   * @param radiusKm - Search radius in kilometers
   * @returns Array of nearby participants
   */
  async getParticipantsNearLocation(
    lat: number,
    lng: number,
    radiusKm: number = 25
  ): Promise<EnergyParticipant[]> {
    const records = await this.call<HolochainRecord[]>('get_participants_near_location', {
      lat,
      lng,
      radius_km: radiusKm,
    });
    return records.map(r => this.extractEntry<EnergyParticipant>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Register as a producer
   *
   * @param sources - Energy sources
   * @param capacityKwh - Capacity in kWh
   * @param location - Optional location
   * @returns The created participant
   */
  async registerProducer(
    sources: EnergySource[],
    capacityKwh: number,
    location?: { lat: number; lng: number }
  ): Promise<EnergyParticipant> {
    return this.register({
      participant_type: 'Producer',
      sources,
      capacity_kwh: capacityKwh,
      location,
    });
  }

  /**
   * Register as a consumer
   *
   * @param capacityKwh - Capacity in kWh
   * @param location - Optional location
   * @returns The created participant
   */
  async registerConsumer(
    capacityKwh: number,
    location?: { lat: number; lng: number }
  ): Promise<EnergyParticipant> {
    return this.register({
      participant_type: 'Consumer',
      sources: ['Grid'],
      capacity_kwh: capacityKwh,
      location,
    });
  }

  /**
   * Register as a prosumer
   *
   * @param sources - Energy sources
   * @param capacityKwh - Capacity in kWh
   * @param location - Optional location
   * @returns The created participant
   */
  async registerProsumer(
    sources: EnergySource[],
    capacityKwh: number,
    location?: { lat: number; lng: number }
  ): Promise<EnergyParticipant> {
    return this.register({
      participant_type: 'Prosumer',
      sources,
      capacity_kwh: capacityKwh,
      location,
    });
  }

  /**
   * Get net energy balance for a participant
   *
   * @param participant - The participant
   * @returns Net balance (positive = net producer)
   */
  getNetBalance(participant: EnergyParticipant): number {
    return participant.total_produced_kwh - participant.total_consumed_kwh;
  }

  /**
   * Check if participant is a net producer
   *
   * @param participant - The participant
   * @returns True if net producer
   */
  isNetProducer(participant: EnergyParticipant): boolean {
    return this.getNetBalance(participant) > 0;
  }

  /**
   * Get participant type description
   *
   * @param type - Participant type
   * @returns Human-readable description
   */
  getTypeDescription(type: ParticipantType): string {
    const descriptions: Record<ParticipantType, string> = {
      Producer: 'Generates energy for the grid',
      Consumer: 'Consumes energy from the grid',
      Prosumer: 'Both produces and consumes energy',
      Operator: 'Operates energy infrastructure',
      Investor: 'Invests in energy projects',
      Storage: 'Provides energy storage services',
    };
    return descriptions[type];
  }

  /**
   * Calculate carbon offset for renewable production
   *
   * @param productionKwh - Energy produced in kWh
   * @param source - Energy source
   * @returns Carbon offset in kg CO2
   */
  calculateCarbonOffset(productionKwh: number, source: EnergySource): number {
    if (source === 'Grid') return 0;
    const gridEmissionsFactor = 0.4; // kg CO2 per kWh (US average)
    return productionKwh * gridEmissionsFactor;
  }
}
