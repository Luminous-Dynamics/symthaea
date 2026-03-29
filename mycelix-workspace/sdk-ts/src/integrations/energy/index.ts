// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Energy Integration
 *
 * Unified Energy SDK providing:
 * - Project registration and management
 * - Participant operations (producers, consumers, prosumers)
 * - P2P energy trading
 * - Energy credits and carbon certificates
 * - Project investments and dividends
 * - Grid management and community energy
 * - Cross-hApp energy settlement via Bridge
 *
 * @packageDocumentation
 * @module integrations/energy
 */

// ============================================================================
// New Holochain Zome Clients (Unified SDK)
// ============================================================================

// Main unified client
export { MycelixEnergyClient } from './client.js';
export type { EnergyClientConfig, EnergyConnectionOptions } from './client.js';

// Individual zome clients
export { ProjectsClient } from './zomes/projects.js';
export type { ProjectsClientConfig } from './zomes/projects.js';

export { ParticipantsClient } from './zomes/participants.js';
export type { ParticipantsClientConfig } from './zomes/participants.js';

export { TradingClient } from './zomes/trading.js';
export type { TradingClientConfig } from './zomes/trading.js';

export { CreditsClient } from './zomes/credits.js';
export type { CreditsClientConfig } from './zomes/credits.js';

export { InvestmentsClient } from './zomes/investments.js';
export type { InvestmentsClientConfig } from './zomes/investments.js';

export { GridClient } from './zomes/grid.js';
export type { GridClientConfig } from './zomes/grid.js';

// Types
export * from './types.js';

// ============================================================================
// Legacy Mock Service (for backward compatibility)
// Re-exported from ./legacy.ts for backward compatibility
// ============================================================================

export {
  // Legacy types
  type LegacyEnergySource,
  type ParticipantRole,
  type EnergyTransactionType,
  type LegacyEnergyParticipant,
  type LegacyEnergyReading,
  type LegacyEnergyTransaction,
  type LegacyGridBalanceRequest,
  type LegacyEnergyCredit,
  // Legacy service
  EnergyService,
  getEnergyService,
  resetEnergyService,
} from './legacy.js';

// ============================================================================
// Bridge Client (Holochain Zome Calls)
// ============================================================================

import { type MycelixClient } from '../../client/index.js';

import type { LegacyEnergySource } from './legacy.js';

const ENERGY_ROLE = 'energy';
const BRIDGE_ZOME = 'energy_bridge';

/** Energy availability query */
export interface EnergyAvailabilityQuery {
  id: string;
  source_happ: string;
  location?: { lat: number; lng: number; radius_km: number };
  energy_sources?: LegacyEnergySource[];
  min_amount?: number;
  queried_at: number;
}

/** Energy availability result */
export interface EnergyAvailability {
  participant_id: string;
  did: string;
  available_kwh: number;
  source: LegacyEnergySource;
  price_per_kwh?: number;
  location?: { lat: number; lng: number };
  reputation_score: number;
}

/** Cross-hApp energy settlement */
export interface EnergySettlementLegacy {
  id: string;
  source_happ: string;
  seller_did: string;
  buyer_did: string;
  amount_kwh: number;
  price_total?: number;
  carbon_credits: number;
  energy_source: LegacyEnergySource;
  status: 'Pending' | 'Confirmed' | 'Settled' | 'Disputed';
  created_at: number;
  settled_at?: number;
}

/** Carbon credit certificate */
export interface CarbonCreditCertificate {
  id: string;
  holder_did: string;
  amount_kg_co2: number;
  energy_source: LegacyEnergySource;
  energy_amount_kwh: number;
  issued_by: string;
  issued_at: number;
  expires_at?: number;
  transferred: boolean;
}

/** Energy bridge event types */
export type EnergyBridgeEventType =
  | 'EnergyListed'
  | 'EnergyPurchased'
  | 'SettlementCompleted'
  | 'CarbonCreditIssued'
  | 'GridBalanceRequest'
  | 'GridBalanceFulfilled';

/** Energy bridge event */
export interface EnergyBridgeEvent {
  id: string;
  event_type: EnergyBridgeEventType;
  participant_did?: string;
  settlement_id?: string;
  amount_kwh?: number;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Query available energy input */
export interface QueryAvailableEnergyInput {
  source_happ: string;
  location?: { lat: number; lng: number; radius_km: number };
  energy_sources?: LegacyEnergySource[];
  min_amount?: number;
}

/** Request energy purchase input */
export interface RequestEnergyPurchaseInput {
  seller_did: string;
  amount_kwh: number;
  max_price_per_kwh?: number;
  source_happ: string;
}

/** Report production input */
export interface ReportProductionInput {
  participant_id: string;
  production_kwh: number;
  consumption_kwh: number;
  energy_source: LegacyEnergySource;
  timestamp: number;
}

/**
 * Energy Bridge Client - Direct Holochain zome calls for cross-hApp energy
 */
export class EnergyBridgeClient {
  constructor(private client: MycelixClient) {}

  async queryAvailableEnergy(input: QueryAvailableEnergyInput): Promise<EnergyAvailability[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_available_energy',
      payload: input,
    });
  }

  async requestEnergyPurchase(input: RequestEnergyPurchaseInput): Promise<EnergySettlementLegacy> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'request_energy_purchase',
      payload: input,
    });
  }

  async confirmSettlement(settlementId: string): Promise<EnergySettlementLegacy> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'confirm_settlement',
      payload: settlementId,
    });
  }

  async getPendingSettlements(did: string): Promise<EnergySettlementLegacy[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_pending_settlements',
      payload: did,
    });
  }

  async reportProduction(input: ReportProductionInput): Promise<void> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'report_production',
      payload: input,
    });
  }

  async getCarbonCredits(holderDid: string): Promise<CarbonCreditCertificate[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_carbon_credits',
      payload: holderDid,
    });
  }

  async transferCarbonCredits(certificateId: string, recipientDid: string): Promise<CarbonCreditCertificate> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'transfer_carbon_credits',
      payload: { certificate_id: certificateId, recipient_did: recipientDid },
    });
  }

  async broadcastEnergyEvent(
    eventType: EnergyBridgeEventType,
    participantDid?: string,
    settlementId?: string,
    amountKwh?: number,
    payload: string = '{}'
  ): Promise<EnergyBridgeEvent> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_energy_event',
      payload: { event_type: eventType, participant_did: participantDid, settlement_id: settlementId, amount_kwh: amountKwh, payload },
    });
  }

  async getRecentEvents(limit?: number): Promise<EnergyBridgeEvent[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }

  async getGridStatistics(): Promise<{
    total_production_kwh: number;
    total_consumption_kwh: number;
    net_balance_kwh: number;
    participant_count: number;
    total_carbon_offset_kg: number;
  }> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_grid_statistics',
      payload: null,
    });
  }
}

// Bridge client singleton
let bridgeInstance: EnergyBridgeClient | null = null;

export function getEnergyBridgeClient(client: MycelixClient): EnergyBridgeClient {
  if (!bridgeInstance) bridgeInstance = new EnergyBridgeClient(client);
  return bridgeInstance;
}

export function resetEnergyBridgeClient(): void {
  bridgeInstance = null;
}
