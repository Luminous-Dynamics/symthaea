/**
 * Mycelix Energy Module
 *
 * TypeScript client for the mycelix-energy hApp.
 * Provides energy project registry, trading, and community ownership (Terra Atlas integration).
 *
 * @module @mycelix/sdk/energy
 */

export interface HolochainRecord<T = unknown> {
  signed_action: { hashed: { hash: string; content: unknown }; signature: string };
  entry: { Present: T };
}

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Energy Types
// ============================================================================

export type EnergySource =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Nuclear'
  | 'Geothermal'
  | 'Storage'
  | 'Other';
export type ProjectStatus =
  | 'Proposed'
  | 'Planning'
  | 'Development'
  | 'Construction'
  | 'Operational'
  | 'Decommissioned';
export type ParticipantType = 'Producer' | 'Consumer' | 'Prosumer' | 'Operator' | 'Investor';

export interface EnergyProject {
  id: string;
  name: string;
  description: string;
  source: EnergySource;
  capacity_kw: number;
  location: { lat: number; lng: number; address?: string };
  status: ProjectStatus;
  owner_dao?: string;
  investment_goal?: number;
  investment_raised: number;
  operating_since?: number;
  created_at: number;
  updated_at: number;
}

export interface RegisterProjectInput {
  name: string;
  description: string;
  source: EnergySource;
  capacity_kw: number;
  location: { lat: number; lng: number; address?: string };
  investment_goal?: number;
}

export interface EnergyParticipant {
  id: string;
  did: string;
  type_: ParticipantType;
  sources: EnergySource[];
  capacity_kwh: number;
  location?: { lat: number; lng: number };
  reputation_score: number;
  active: boolean;
  registered_at: number;
}

export interface RegisterParticipantInput {
  type_: ParticipantType;
  sources: EnergySource[];
  capacity_kwh: number;
  location?: { lat: number; lng: number };
}

export interface EnergyTrade {
  id: string;
  seller: string;
  buyer: string;
  amount_kwh: number;
  source: EnergySource;
  price_per_kwh: number;
  currency: string;
  status: 'Pending' | 'Confirmed' | 'Delivered' | 'Disputed';
  created_at: number;
  delivered_at?: number;
}

export interface TradeInput {
  buyer: string;
  amount_kwh: number;
  source: EnergySource;
  price_per_kwh: number;
  currency: string;
}

export interface EnergyCredit {
  id: string;
  participant: string;
  amount_kwh: number;
  source: EnergySource;
  issued_at: number;
  expires_at?: number;
  used: boolean;
}

export interface Investment {
  id: string;
  project_id: string;
  investor: string;
  amount: number;
  currency: string;
  ownership_percentage: number;
  invested_at: number;
}

export interface InvestInput {
  project_id: string;
  amount: number;
  currency: string;
}

// ============================================================================
// Clients
// ============================================================================

const ENERGY_ROLE = 'energy';

export class ProjectsClient {
  constructor(private readonly client: ZomeCallable) {}

  async registerProject(input: RegisterProjectInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'projects',
      fn_name: 'register_project',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async getProject(projectId: string): Promise<HolochainRecord<EnergyProject> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'projects',
      fn_name: 'get_project',
      payload: projectId,
    }) as Promise<HolochainRecord<EnergyProject> | null>;
  }

  async getProjectsBySource(source: EnergySource): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'projects',
      fn_name: 'get_projects_by_source',
      payload: source,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async getProjectsByStatus(status: ProjectStatus): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'projects',
      fn_name: 'get_projects_by_status',
      payload: status,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async searchProjects(query: {
    source?: EnergySource;
    status?: ProjectStatus;
    minCapacity?: number;
  }): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'projects',
      fn_name: 'search_projects',
      payload: query,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }
}

export class ParticipantsClient {
  constructor(private readonly client: ZomeCallable) {}

  async register(input: RegisterParticipantInput): Promise<HolochainRecord<EnergyParticipant>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'participants',
      fn_name: 'register',
      payload: input,
    }) as Promise<HolochainRecord<EnergyParticipant>>;
  }

  async getParticipant(did: string): Promise<HolochainRecord<EnergyParticipant> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'participants',
      fn_name: 'get_participant',
      payload: did,
    }) as Promise<HolochainRecord<EnergyParticipant> | null>;
  }

  async getParticipantsByType(
    type_: ParticipantType
  ): Promise<HolochainRecord<EnergyParticipant>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'participants',
      fn_name: 'get_by_type',
      payload: type_,
    }) as Promise<HolochainRecord<EnergyParticipant>[]>;
  }

  async updateCapacity(capacityKwh: number): Promise<HolochainRecord<EnergyParticipant>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'participants',
      fn_name: 'update_capacity',
      payload: capacityKwh,
    }) as Promise<HolochainRecord<EnergyParticipant>>;
  }
}

export class TradingClient {
  constructor(private readonly client: ZomeCallable) {}

  async createTrade(input: TradeInput): Promise<HolochainRecord<EnergyTrade>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'trading',
      fn_name: 'create_trade',
      payload: input,
    }) as Promise<HolochainRecord<EnergyTrade>>;
  }

  async acceptTrade(tradeId: string): Promise<HolochainRecord<EnergyTrade>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'trading',
      fn_name: 'accept_trade',
      payload: tradeId,
    }) as Promise<HolochainRecord<EnergyTrade>>;
  }

  async confirmDelivery(tradeId: string): Promise<HolochainRecord<EnergyTrade>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'trading',
      fn_name: 'confirm_delivery',
      payload: tradeId,
    }) as Promise<HolochainRecord<EnergyTrade>>;
  }

  async getTradesByParticipant(did: string): Promise<HolochainRecord<EnergyTrade>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'trading',
      fn_name: 'get_trades_by_participant',
      payload: did,
    }) as Promise<HolochainRecord<EnergyTrade>[]>;
  }

  async getOpenTrades(source?: EnergySource): Promise<HolochainRecord<EnergyTrade>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'trading',
      fn_name: 'get_open_trades',
      payload: source ?? null,
    }) as Promise<HolochainRecord<EnergyTrade>[]>;
  }
}

export class CreditsClient {
  constructor(private readonly client: ZomeCallable) {}

  async issueCredit(
    participantDid: string,
    amountKwh: number,
    source: EnergySource
  ): Promise<HolochainRecord<EnergyCredit>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'credits',
      fn_name: 'issue_credit',
      payload: { participant: participantDid, amount_kwh: amountKwh, source },
    }) as Promise<HolochainRecord<EnergyCredit>>;
  }

  async getCredits(participantDid: string): Promise<HolochainRecord<EnergyCredit>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'credits',
      fn_name: 'get_credits',
      payload: participantDid,
    }) as Promise<HolochainRecord<EnergyCredit>[]>;
  }

  async useCredit(creditId: string): Promise<HolochainRecord<EnergyCredit>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'credits',
      fn_name: 'use_credit',
      payload: creditId,
    }) as Promise<HolochainRecord<EnergyCredit>>;
  }
}

export class InvestmentsClient {
  constructor(private readonly client: ZomeCallable) {}

  async invest(input: InvestInput): Promise<HolochainRecord<Investment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'investments',
      fn_name: 'invest',
      payload: input,
    }) as Promise<HolochainRecord<Investment>>;
  }

  async getInvestmentsByProject(projectId: string): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'investments',
      fn_name: 'get_by_project',
      payload: projectId,
    }) as Promise<HolochainRecord<Investment>[]>;
  }

  async getInvestmentsByInvestor(investorDid: string): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'investments',
      fn_name: 'get_by_investor',
      payload: investorDid,
    }) as Promise<HolochainRecord<Investment>[]>;
  }
}

export function createEnergyClients(client: ZomeCallable) {
  return {
    projects: new ProjectsClient(client),
    participants: new ParticipantsClient(client),
    trading: new TradingClient(client),
    credits: new CreditsClient(client),
    investments: new InvestmentsClient(client),
  };
}

// Re-export sovereignty module for consumer-facing features
export * from './sovereignty.js';

// Re-export API client for real data integration
export { NRELSolarClient, ERCOTGridClient, WeatherClient, TerraAtlasBridge, EnergyAPIClient, energyAPI, type NRELSolarResource, type SolarProjectAnalysis, type GridCondition, type GridAlert, type WeatherData, type WeatherForecast, type NearbyProject, type TerraAtlasStats } from './api-client.js';
export type { GridStatus } from './sovereignty.js';

export default {
  ProjectsClient,
  ParticipantsClient,
  TradingClient,
  CreditsClient,
  InvestmentsClient,
  createEnergyClients,
};
