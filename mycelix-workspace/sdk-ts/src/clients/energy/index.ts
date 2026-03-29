// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Energy Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-energy hApp.
 * Provides energy project registry, P2P grid trading, investments,
 * regenerative exit contracts, and Terra Atlas integration.
 *
 * @module @mycelix/sdk/clients/energy
 */

import type { AppClient } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
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
// Project Types
// ============================================================================

export type ProjectType =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Nuclear'
  | 'Geothermal'
  | 'BatteryStorage'
  | 'PumpedHydro'
  | 'Hydrogen'
  | 'Biomass';

export type ProjectStatus =
  | 'Proposed'
  | 'Planning'
  | 'Permitting'
  | 'Financing'
  | 'Construction'
  | 'Operational'
  | 'Decommissioned';

export interface ProjectLocation {
  latitude: number;
  longitude: number;
  country: string;
  region: string;
  address?: string;
}

export interface ProjectFinancials {
  total_cost: number;
  funded_amount: number;
  currency: string;
  target_irr: number;
  payback_years: number;
  annual_revenue_estimate: number;
}

export interface EnergyProject {
  id: string;
  terra_atlas_id?: string;
  name: string;
  description: string;
  project_type: ProjectType;
  location: ProjectLocation;
  capacity_mw: number;
  status: ProjectStatus;
  developer_did: string;
  community_did?: string;
  financials: ProjectFinancials;
  created: number;
  updated: number;
}

export interface RegisterProjectInput {
  terra_atlas_id?: string;
  name: string;
  description: string;
  project_type: ProjectType;
  location: ProjectLocation;
  capacity_mw: number;
  developer_did: string;
  community_did?: string;
  financials: ProjectFinancials;
}

export interface UpdateStatusInput {
  project_id: string;
  new_status: ProjectStatus;
}

export interface ProjectMilestone {
  id: string;
  project_id: string;
  name: string;
  description: string;
  target_date: number;
  completed_date?: number;
  verification_evidence?: string;
}

export interface AddMilestoneInput {
  project_id: string;
  name: string;
  description: string;
  target_date: number;
}

export interface CompleteMilestoneInput {
  milestone_id: string;
  evidence: string;
}

export interface LocationSearchInput {
  latitude: number;
  longitude: number;
  radius_km: number;
}

export interface UpdateFinancialsInput {
  project_id: string;
  requester_did: string;
  financials: ProjectFinancials;
}

export interface AssignCommunityInput {
  project_id: string;
  requester_did: string;
  community_did: string;
}

export interface LinkTerraAtlasInput {
  project_id: string;
  terra_atlas_id: string;
}

export interface UpdateCapacityInput {
  project_id: string;
  requester_did: string;
  capacity_mw: number;
}

// ============================================================================
// Grid Trading Types
// ============================================================================

export type OfferStatus = 'Active' | 'PartiallyFilled' | 'Filled' | 'Cancelled' | 'Expired';

export interface EnergyProduction {
  id: string;
  producer_did: string;
  project_id: string;
  amount_kwh: number;
  timestamp: number;
  period_hours: number;
  meter_reading?: number;
  verified: boolean;
}

export interface RecordProductionInput {
  producer_did: string;
  project_id: string;
  amount_kwh: number;
  period_hours: number;
  meter_reading?: number;
}

export interface TradeOffer {
  id: string;
  seller_did: string;
  project_id?: string;
  amount_kwh: number;
  price_per_kwh: number;
  currency: string;
  available_from: number;
  available_until: number;
  status: OfferStatus;
  created: number;
}

export interface CreateOfferInput {
  seller_did: string;
  project_id?: string;
  amount_kwh: number;
  price_per_kwh: number;
  currency: string;
  available_from: number;
  available_until: number;
}

export interface Trade {
  id: string;
  offer_id: string;
  seller_did: string;
  buyer_did: string;
  amount_kwh: number;
  price_per_kwh: number;
  total_price: number;
  currency: string;
  executed: number;
  settled: boolean;
  payment_reference?: string;
}

export interface ExecuteTradeInput {
  offer_id: string;
  buyer_did: string;
  amount_kwh: number;
}

export interface SettleTradeInput {
  trade_id: string;
  payment_reference: string;
}

export interface VerifyProductionInput {
  production_id: string;
  verifier_did: string;
}

export interface CancelOfferInput {
  offer_id: string;
  requester_did: string;
}

export interface UpdateOfferPriceInput {
  offer_id: string;
  requester_did: string;
  new_price_per_kwh: number;
}

export interface ProducerStats {
  producer_did: string;
  total_kwh: number;
  verified_kwh: number;
  record_count: number;
}

export interface GridSummary {
  active_offers: number;
  total_kwh_available: number;
  total_trades: number;
  total_kwh_traded: number;
  total_value_traded: number;
}

// ============================================================================
// Investment Types
// ============================================================================

export type InvestmentType =
  | 'Equity'
  | 'Debt'
  | 'ConvertibleNote'
  | 'RevenueShare'
  | 'CommunityShare';

export type InvestmentStatus =
  | 'Pledged'
  | 'PendingPayment'
  | 'Confirmed'
  | 'Cancelled';

export interface Investment {
  id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  shares: number;
  share_percentage: number;
  investment_type: InvestmentType;
  status: InvestmentStatus;
  pledged: number;
  confirmed?: number;
}

export interface PledgeInput {
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  shares: number;
  share_percentage: number;
  investment_type: InvestmentType;
}

export interface Dividend {
  id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  period_start: number;
  period_end: number;
  distributed: number;
  payment_reference?: string;
}

export interface DividendInput {
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  period_start: number;
  period_end: number;
  payment_reference?: string;
}

export interface CancelInvestmentInput {
  investment_id: string;
  requester_did: string;
}

export interface UpdateInvestmentAmountInput {
  investment_id: string;
  requester_did: string;
  new_amount: number;
  new_shares: number;
  new_share_percentage: number;
}

export interface DistributeProjectDividendsInput {
  project_id: string;
  total_amount: number;
  currency: string;
  period_start: number;
  period_end: number;
  payment_reference?: string;
}

export interface ProjectTotalInvestment {
  project_id: string;
  total_amount: number;
  total_shares: number;
  investor_count: number;
}

export interface PortfolioSummary {
  investor_did: string;
  total_invested: number;
  total_shares: number;
  project_count: number;
  unique_projects: number;
  total_dividends_received: number;
}

// ============================================================================
// Regenerative Exit Types
// ============================================================================

export type ContractStatus =
  | 'Active'
  | 'TransitionInProgress'
  | 'TransitionComplete'
  | 'Paused'
  | 'Terminated';

export type ConditionType =
  | 'CommunityReadiness'
  | 'FinancialSustainability'
  | 'OperationalCompetence'
  | 'GovernanceMaturity'
  | 'InvestorReturns'
  | 'ReserveAccountFunded'
  | 'MinimumOperatingHistory';

export interface TransitionCondition {
  condition_type: ConditionType;
  threshold: number;
  current_value: number;
  weight: number;
  satisfied: boolean;
}

export interface RegenerativeContract {
  id: string;
  project_id: string;
  community_did: string;
  conditions: TransitionCondition[];
  current_ownership_percentage: number;
  target_ownership_percentage: number;
  reserve_account_balance: number;
  currency: string;
  status: ContractStatus;
  created: number;
  last_assessment: number;
}

export interface CreateContractInput {
  project_id: string;
  community_did: string;
  conditions: TransitionCondition[];
  target_ownership_percentage: number;
  currency: string;
}

export interface ConditionScore {
  condition_type: ConditionType;
  score: number;
  evidence: string;
}

export interface ReadinessAssessment {
  id: string;
  contract_id: string;
  assessor_did: string;
  scores: ConditionScore[];
  overall_readiness: number;
  recommendations: string[];
  assessed: number;
}

export interface AssessmentInput {
  contract_id: string;
  assessor_did: string;
  scores: ConditionScore[];
  recommendations: string[];
}

export interface OwnershipTransfer {
  id: string;
  contract_id: string;
  from_percentage: number;
  to_percentage: number;
  shares_transferred: number;
  transfer_price: number;
  currency: string;
  executed: number;
}

export interface TransferInput {
  contract_id: string;
  new_percentage: number;
  shares_transferred: number;
  transfer_price: number;
}

export interface UpdateContractStatusInput {
  contract_id: string;
  new_status: ContractStatus;
}

export interface AddToReserveInput {
  contract_id: string;
  amount: number;
}

export interface UpdateConditionsInput {
  contract_id: string;
  conditions: TransitionCondition[];
}

export interface ReadinessProgress {
  contract_id: string;
  current_ownership_percentage: number;
  target_ownership_percentage: number;
  reserve_balance: number;
  overall_readiness: number;
  conditions_met: number;
  total_conditions: number;
  status: ContractStatus;
}

export interface RegenerativeSummary {
  active_contracts: number;
  completed_contracts: number;
  total_reserve_balance: number;
  total_ownership_transferred: number;
  total_transfers: number;
}

// ============================================================================
// Bridge Types (Terra Atlas Integration)
// ============================================================================

export type EnergyType = 'Solar' | 'Wind' | 'Hydro' | 'Geothermal' | 'Nuclear' | 'Storage' | 'Mixed';

export type BridgeProjectStatus =
  | 'Discovery'
  | 'Funding'
  | 'Development'
  | 'Operational'
  | 'Transitioning'
  | 'CommunityOwned';

export type EnergyEventType =
  | 'ProjectDiscovered'
  | 'InvestmentReceived'
  | 'MilestoneAchieved'
  | 'TransitionInitiated'
  | 'CommunityOwnershipComplete'
  | 'ProductionUpdate'
  | 'StatusChanged'
  | 'SyncPending';

export type SyncType =
  | 'InvestmentUpdate'
  | 'ProductionMetrics'
  | 'MilestoneProgress'
  | 'StatusChange'
  | 'TransitionProgress';

export type MilestoneType =
  | 'CommunityFormation'
  | 'OperatorTraining'
  | 'FinancialIndependence'
  | 'GovernanceEstablished'
  | 'FullTransition';

export type BridgeInvestmentType = 'Equity' | 'Loan' | 'Grant' | 'CommunityShare';

export interface GeoLocation {
  latitude: number;
  longitude: number;
  region: string;
  country: string;
}

export interface TerraAtlasProject {
  id: string;
  terra_atlas_id: string;
  name: string;
  project_type: EnergyType;
  location: GeoLocation;
  capacity_mw: number;
  total_investment: number;
  current_investment: number;
  status: BridgeProjectStatus;
  regenerative_progress: number;
  synced_at: number;
}

export interface SyncProjectInput {
  terra_atlas_id: string;
  name: string;
  project_type: EnergyType;
  location: GeoLocation;
  capacity_mw: number;
  total_investment: number;
  current_investment: number;
  status: BridgeProjectStatus;
}

export interface InvestmentRecord {
  id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  source_happ: string;
  investment_type: BridgeInvestmentType;
  created_at: number;
}

export interface RecordInvestmentInput {
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  source_happ: string;
  investment_type: BridgeInvestmentType;
}

export interface RegenerativeMilestone {
  id: string;
  project_id: string;
  milestone_type: MilestoneType;
  community_readiness: number;
  operator_certification: boolean;
  financial_sustainability: number;
  achieved_at?: number;
  verified_by?: string;
}

export interface RecordMilestoneInput {
  project_id: string;
  milestone_type: MilestoneType;
  community_readiness: number;
  operator_certification: boolean;
  financial_sustainability: number;
  verified_by?: string;
}

export interface ProductionRecord {
  id: string;
  project_id: string;
  terra_atlas_id: string;
  period_start: number;
  period_end: number;
  energy_generated_mwh: number;
  capacity_factor: number;
  revenue_generated: number;
  currency: string;
  grid_injection_mwh: number;
  self_consumption_mwh: number;
  recorded_at: number;
  verified_by?: string;
}

export interface RecordProductionUpdateInput {
  project_id: string;
  terra_atlas_id: string;
  period_start: number;
  period_end: number;
  energy_generated_mwh: number;
  capacity_factor: number;
  revenue_generated: number;
  currency: string;
  grid_injection_mwh: number;
  self_consumption_mwh: number;
  verified_by?: string;
}

export interface EnergyBridgeEvent {
  id: string;
  event_type: EnergyEventType;
  project_id: string;
  payload: string;
  source: string;
  timestamp: number;
}

export interface BroadcastEnergyEventInput {
  event_type: EnergyEventType;
  project_id: string;
  payload: string;
}

export interface PendingSyncRecord {
  id: string;
  sync_type: SyncType;
  target_system: string;
  payload: string;
  created_at: number;
  synced_at?: number;
  retry_count: number;
  last_error?: string;
}

export interface QueueSyncInput {
  sync_type: SyncType;
  project_id: string;
  payload: string;
}

export interface MarkSyncCompleteInput {
  sync_id: string;
  success: boolean;
  error?: string;
}

export interface UpdateProjectStatusInput {
  project_id: string;
  terra_atlas_id: string;
  new_status: BridgeProjectStatus;
  reason?: string;
}

export interface InvestmentSummary {
  project_id: string;
  total_amount: number;
  investor_count: number;
  by_type: Array<[string, number]>;
}

export interface TransitionRecord {
  id: string;
  project_id: string;
  terra_atlas_id: string;
  from_ownership_pct: number;
  to_ownership_pct: number;
  community_did: string;
  reserve_account_balance: number;
  conditions_met: string[];
  conditions_pending: string[];
  status: 'Proposed' | 'InProgress' | 'Completed' | 'Cancelled';
  initiated_at: number;
  completed_at?: number;
}

export interface InitiateTransitionInput {
  project_id: string;
  terra_atlas_id: string;
  community_did: string;
  current_community_ownership: number;
  target_ownership_pct: number;
  reserve_account_balance: number;
  conditions_met: string[];
  conditions_pending: string[];
}

export interface CompleteTransitionInput {
  transition_id: string;
  project_id: string;
  terra_atlas_id: string;
  community_did: string;
  final_ownership_pct: number;
}

// ============================================================================
// Projects Client
// ============================================================================

const ENERGY_ROLE = 'energy';
const PROJECTS_ZOME = 'projects';

/**
 * Projects Client - Energy project registration and management
 *
 * Covers 15 zome functions from projects coordinator
 */
export class ProjectsClient {
  constructor(private readonly client: ZomeCallable) {}

  async registerProject(input: RegisterProjectInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'register_project',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async getProject(projectId: string): Promise<HolochainRecord<EnergyProject> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_project',
      payload: projectId,
    }) as Promise<HolochainRecord<EnergyProject> | null>;
  }

  async updateProjectStatus(input: UpdateStatusInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'update_project_status',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async addMilestone(input: AddMilestoneInput): Promise<HolochainRecord<ProjectMilestone>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'add_milestone',
      payload: input,
    }) as Promise<HolochainRecord<ProjectMilestone>>;
  }

  async completeMilestone(input: CompleteMilestoneInput): Promise<HolochainRecord<ProjectMilestone>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'complete_milestone',
      payload: input,
    }) as Promise<HolochainRecord<ProjectMilestone>>;
  }

  async searchProjectsByLocation(input: LocationSearchInput): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'search_projects_by_location',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async getDeveloperProjects(developerDid: string): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_developer_projects',
      payload: developerDid,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async getCommunityProjects(communityDid: string): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_community_projects',
      payload: communityDid,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async getProjectMilestones(projectId: string): Promise<HolochainRecord<ProjectMilestone>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_project_milestones',
      payload: projectId,
    }) as Promise<HolochainRecord<ProjectMilestone>[]>;
  }

  async updateProjectFinancials(input: UpdateFinancialsInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'update_project_financials',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async getProjectsByStatus(status: ProjectStatus): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_projects_by_status',
      payload: status,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async getProjectsByType(projectType: ProjectType): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'get_projects_by_type',
      payload: projectType,
    }) as Promise<HolochainRecord<EnergyProject>[]>;
  }

  async assignCommunity(input: AssignCommunityInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'assign_community',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async linkToTerraAtlas(input: LinkTerraAtlasInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'link_to_terra_atlas',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }

  async updateCapacity(input: UpdateCapacityInput): Promise<HolochainRecord<EnergyProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: PROJECTS_ZOME,
      fn_name: 'update_capacity',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProject>>;
  }
}

// ============================================================================
// Grid Trading Client
// ============================================================================

const GRID_ZOME = 'grid';

/**
 * Grid Client - P2P energy trading
 *
 * Covers 17 zome functions from grid coordinator
 */
export class GridClient {
  constructor(private readonly client: ZomeCallable) {}

  async recordProduction(input: RecordProductionInput): Promise<HolochainRecord<EnergyProduction>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'record_production',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProduction>>;
  }

  async createTradeOffer(input: CreateOfferInput): Promise<HolochainRecord<TradeOffer>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'create_trade_offer',
      payload: input,
    }) as Promise<HolochainRecord<TradeOffer>>;
  }

  async executeTrade(input: ExecuteTradeInput): Promise<HolochainRecord<Trade>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'execute_trade',
      payload: input,
    }) as Promise<HolochainRecord<Trade>>;
  }

  async getActiveOffers(): Promise<HolochainRecord<TradeOffer>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_active_offers',
      payload: null,
    }) as Promise<HolochainRecord<TradeOffer>[]>;
  }

  async settleTrade(input: SettleTradeInput): Promise<HolochainRecord<Trade>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'settle_trade',
      payload: input,
    }) as Promise<HolochainRecord<Trade>>;
  }

  async getProducerProduction(producerDid: string): Promise<HolochainRecord<EnergyProduction>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_producer_production',
      payload: producerDid,
    }) as Promise<HolochainRecord<EnergyProduction>[]>;
  }

  async getSellerOffers(sellerDid: string): Promise<HolochainRecord<TradeOffer>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_seller_offers',
      payload: sellerDid,
    }) as Promise<HolochainRecord<TradeOffer>[]>;
  }

  async getBuyerTrades(buyerDid: string): Promise<HolochainRecord<Trade>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_buyer_trades',
      payload: buyerDid,
    }) as Promise<HolochainRecord<Trade>[]>;
  }

  async getOfferTrades(offerId: string): Promise<HolochainRecord<Trade>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_offer_trades',
      payload: offerId,
    }) as Promise<HolochainRecord<Trade>[]>;
  }

  async verifyProduction(input: VerifyProductionInput): Promise<HolochainRecord<EnergyProduction>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'verify_production',
      payload: input,
    }) as Promise<HolochainRecord<EnergyProduction>>;
  }

  async cancelOffer(input: CancelOfferInput): Promise<HolochainRecord<TradeOffer>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'cancel_offer',
      payload: input,
    }) as Promise<HolochainRecord<TradeOffer>>;
  }

  async getTrade(tradeId: string): Promise<HolochainRecord<Trade> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_trade',
      payload: tradeId,
    }) as Promise<HolochainRecord<Trade> | null>;
  }

  async getOffer(offerId: string): Promise<HolochainRecord<TradeOffer> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_offer',
      payload: offerId,
    }) as Promise<HolochainRecord<TradeOffer> | null>;
  }

  async getUnsettledTrades(): Promise<HolochainRecord<Trade>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_unsettled_trades',
      payload: null,
    }) as Promise<HolochainRecord<Trade>[]>;
  }

  async getProducerTotalProduction(producerDid: string): Promise<ProducerStats> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_producer_total_production',
      payload: producerDid,
    }) as Promise<ProducerStats>;
  }

  async updateOfferPrice(input: UpdateOfferPriceInput): Promise<HolochainRecord<TradeOffer>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'update_offer_price',
      payload: input,
    }) as Promise<HolochainRecord<TradeOffer>>;
  }

  async getGridSummary(): Promise<GridSummary> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: GRID_ZOME,
      fn_name: 'get_grid_summary',
      payload: null,
    }) as Promise<GridSummary>;
  }
}

// ============================================================================
// Investments Client
// ============================================================================

const INVESTMENTS_ZOME = 'investments';

/**
 * Investments Client - Community investment in energy projects
 *
 * Covers 14 zome functions from investments coordinator
 */
export class InvestmentsClient {
  constructor(private readonly client: ZomeCallable) {}

  async pledgeInvestment(input: PledgeInput): Promise<HolochainRecord<Investment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'pledge_investment',
      payload: input,
    }) as Promise<HolochainRecord<Investment>>;
  }

  async confirmInvestment(investmentId: string): Promise<HolochainRecord<Investment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'confirm_investment',
      payload: investmentId,
    }) as Promise<HolochainRecord<Investment>>;
  }

  async distributeDividend(input: DividendInput): Promise<HolochainRecord<Dividend>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'distribute_dividend',
      payload: input,
    }) as Promise<HolochainRecord<Dividend>>;
  }

  async getInvestorPortfolio(did: string): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_investor_portfolio',
      payload: did,
    }) as Promise<HolochainRecord<Investment>[]>;
  }

  async getProjectInvestments(projectId: string): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_project_investments',
      payload: projectId,
    }) as Promise<HolochainRecord<Investment>[]>;
  }

  async getInvestment(investmentId: string): Promise<HolochainRecord<Investment> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_investment',
      payload: investmentId,
    }) as Promise<HolochainRecord<Investment> | null>;
  }

  async cancelInvestment(input: CancelInvestmentInput): Promise<HolochainRecord<Investment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'cancel_investment',
      payload: input,
    }) as Promise<HolochainRecord<Investment>>;
  }

  async getInvestorDividends(did: string): Promise<HolochainRecord<Dividend>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_investor_dividends',
      payload: did,
    }) as Promise<HolochainRecord<Dividend>[]>;
  }

  async getInvestmentsByStatus(status: InvestmentStatus): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_investments_by_status',
      payload: status,
    }) as Promise<HolochainRecord<Investment>[]>;
  }

  async getInvestmentsByType(investmentType: InvestmentType): Promise<HolochainRecord<Investment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_investments_by_type',
      payload: investmentType,
    }) as Promise<HolochainRecord<Investment>[]>;
  }

  async getProjectTotalInvestment(projectId: string): Promise<ProjectTotalInvestment> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_project_total_investment',
      payload: projectId,
    }) as Promise<ProjectTotalInvestment>;
  }

  async distributeProjectDividends(input: DistributeProjectDividendsInput): Promise<HolochainRecord<Dividend>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'distribute_project_dividends',
      payload: input,
    }) as Promise<HolochainRecord<Dividend>[]>;
  }

  async updateInvestmentAmount(input: UpdateInvestmentAmountInput): Promise<HolochainRecord<Investment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'update_investment_amount',
      payload: input,
    }) as Promise<HolochainRecord<Investment>>;
  }

  async getPortfolioSummary(did: string): Promise<PortfolioSummary> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: INVESTMENTS_ZOME,
      fn_name: 'get_portfolio_summary',
      payload: did,
    }) as Promise<PortfolioSummary>;
  }
}

// ============================================================================
// Regenerative Exit Client
// ============================================================================

const REGENERATIVE_ZOME = 'regenerative';

/**
 * Regenerative Client - Community ownership transition contracts
 *
 * Covers 17 zome functions from regenerative coordinator
 */
export class RegenerativeClient {
  constructor(private readonly client: ZomeCallable) {}

  async createRegenerativeContract(input: CreateContractInput): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'create_regenerative_contract',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async submitReadinessAssessment(input: AssessmentInput): Promise<HolochainRecord<ReadinessAssessment>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'submit_readiness_assessment',
      payload: input,
    }) as Promise<HolochainRecord<ReadinessAssessment>>;
  }

  async executeOwnershipTransfer(input: TransferInput): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'execute_ownership_transfer',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async getContract(contractId: string): Promise<HolochainRecord<RegenerativeContract> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_contract',
      payload: contractId,
    }) as Promise<HolochainRecord<RegenerativeContract> | null>;
  }

  async getProjectContracts(projectId: string): Promise<HolochainRecord<RegenerativeContract>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_project_contracts',
      payload: projectId,
    }) as Promise<HolochainRecord<RegenerativeContract>[]>;
  }

  async getCommunityContracts(communityDid: string): Promise<HolochainRecord<RegenerativeContract>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_community_contracts',
      payload: communityDid,
    }) as Promise<HolochainRecord<RegenerativeContract>[]>;
  }

  async getContractAssessments(contractId: string): Promise<HolochainRecord<ReadinessAssessment>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_contract_assessments',
      payload: contractId,
    }) as Promise<HolochainRecord<ReadinessAssessment>[]>;
  }

  async getContractTransfers(contractId: string): Promise<HolochainRecord<OwnershipTransfer>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_contract_transfers',
      payload: contractId,
    }) as Promise<HolochainRecord<OwnershipTransfer>[]>;
  }

  async updateContractStatus(input: UpdateContractStatusInput): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'update_contract_status',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async getContractsByStatus(status: ContractStatus): Promise<HolochainRecord<RegenerativeContract>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_contracts_by_status',
      payload: status,
    }) as Promise<HolochainRecord<RegenerativeContract>[]>;
  }

  async addToReserve(input: AddToReserveInput): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'add_to_reserve',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async updateConditions(input: UpdateConditionsInput): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'update_conditions',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async getLatestAssessment(contractId: string): Promise<HolochainRecord<ReadinessAssessment> | null> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_latest_assessment',
      payload: contractId,
    }) as Promise<HolochainRecord<ReadinessAssessment> | null>;
  }

  async getReadinessProgress(contractId: string): Promise<ReadinessProgress> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_readiness_progress',
      payload: contractId,
    }) as Promise<ReadinessProgress>;
  }

  async getRegenerativeSummary(): Promise<RegenerativeSummary> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'get_regenerative_summary',
      payload: null,
    }) as Promise<RegenerativeSummary>;
  }

  async pauseContract(contractId: string): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'pause_contract',
      payload: contractId,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }

  async resumeContract(contractId: string): Promise<HolochainRecord<RegenerativeContract>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: REGENERATIVE_ZOME,
      fn_name: 'resume_contract',
      payload: contractId,
    }) as Promise<HolochainRecord<RegenerativeContract>>;
  }
}

// ============================================================================
// Bridge Client (Terra Atlas Integration)
// ============================================================================

const BRIDGE_ZOME = 'bridge';

/**
 * Bridge Client - Terra Atlas integration and cross-hApp communication
 *
 * Covers 21 zome functions from bridge coordinator
 */
export class BridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  async syncTerraAtlasProject(input: SyncProjectInput): Promise<HolochainRecord<TerraAtlasProject>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'sync_terra_atlas_project',
      payload: input,
    }) as Promise<HolochainRecord<TerraAtlasProject>>;
  }

  async recordInvestment(input: RecordInvestmentInput): Promise<HolochainRecord<InvestmentRecord>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'record_investment',
      payload: input,
    }) as Promise<HolochainRecord<InvestmentRecord>>;
  }

  async recordMilestone(input: RecordMilestoneInput): Promise<HolochainRecord<RegenerativeMilestone>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'record_milestone',
      payload: input,
    }) as Promise<HolochainRecord<RegenerativeMilestone>>;
  }

  async getProjectInvestments(projectId: string): Promise<HolochainRecord<InvestmentRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_project_investments',
      payload: projectId,
    }) as Promise<HolochainRecord<InvestmentRecord>[]>;
  }

  async getInvestmentsByDid(did: string): Promise<HolochainRecord<InvestmentRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_investments_by_did',
      payload: did,
    }) as Promise<HolochainRecord<InvestmentRecord>[]>;
  }

  async broadcastEnergyEvent(input: BroadcastEnergyEventInput): Promise<HolochainRecord<EnergyBridgeEvent>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_energy_event',
      payload: input,
    }) as Promise<HolochainRecord<EnergyBridgeEvent>>;
  }

  async recordProductionUpdate(input: RecordProductionUpdateInput): Promise<HolochainRecord<ProductionRecord>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'record_production_update',
      payload: input,
    }) as Promise<HolochainRecord<ProductionRecord>>;
  }

  async getProjectProductionHistory(projectId: string): Promise<HolochainRecord<ProductionRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_project_production_history',
      payload: projectId,
    }) as Promise<HolochainRecord<ProductionRecord>[]>;
  }

  async getProjectInvestmentTotal(projectId: string): Promise<InvestmentSummary> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_project_investment_total',
      payload: projectId,
    }) as Promise<InvestmentSummary>;
  }

  async queueSyncToTerraAtlas(input: QueueSyncInput): Promise<HolochainRecord<PendingSyncRecord>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'queue_sync_to_terra_atlas',
      payload: input,
    }) as Promise<HolochainRecord<PendingSyncRecord>>;
  }

  async getPendingSyncs(): Promise<HolochainRecord<PendingSyncRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_pending_syncs',
      payload: null,
    }) as Promise<HolochainRecord<PendingSyncRecord>[]>;
  }

  async markSyncComplete(input: MarkSyncCompleteInput): Promise<boolean> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'mark_sync_complete',
      payload: input,
    }) as Promise<boolean>;
  }

  async updateProjectStatus(input: UpdateProjectStatusInput): Promise<HolochainRecord<EnergyBridgeEvent>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'update_project_status',
      payload: input,
    }) as Promise<HolochainRecord<EnergyBridgeEvent>>;
  }

  async getAllProjects(): Promise<HolochainRecord<TerraAtlasProject>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_all_projects',
      payload: null,
    }) as Promise<HolochainRecord<TerraAtlasProject>[]>;
  }

  async getProjectMilestones(projectId: string): Promise<HolochainRecord<RegenerativeMilestone>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_project_milestones',
      payload: projectId,
    }) as Promise<HolochainRecord<RegenerativeMilestone>[]>;
  }

  async getRecentEvents(limit: number): Promise<HolochainRecord<EnergyBridgeEvent>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit,
    }) as Promise<HolochainRecord<EnergyBridgeEvent>[]>;
  }

  async initiateTransition(input: InitiateTransitionInput): Promise<HolochainRecord<TransitionRecord>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'initiate_transition',
      payload: input,
    }) as Promise<HolochainRecord<TransitionRecord>>;
  }

  async completeTransition(input: CompleteTransitionInput): Promise<HolochainRecord<EnergyBridgeEvent>> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'complete_transition',
      payload: input,
    }) as Promise<HolochainRecord<EnergyBridgeEvent>>;
  }

  async getActiveTransitions(): Promise<HolochainRecord<TransitionRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_active_transitions',
      payload: null,
    }) as Promise<HolochainRecord<TransitionRecord>[]>;
  }

  async getProjectTransitions(projectId: string): Promise<HolochainRecord<TransitionRecord>[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_project_transitions',
      payload: projectId,
    }) as Promise<HolochainRecord<TransitionRecord>[]>;
  }
}

// ============================================================================
// Unified Energy Client
// ============================================================================

/**
 * Unified Energy Client for mycelix-energy hApp
 *
 * Provides access to all energy functionality through a single interface:
 * - projects: Project registration and management (15 functions)
 * - grid: P2P energy trading (17 functions)
 * - investments: Community investment (14 functions)
 * - regenerative: Ownership transition contracts (17 functions)
 * - bridge: Terra Atlas integration (21 functions)
 *
 * Total: 84 functions covered
 *
 * @example
 * ```typescript
 * import { EnergyClient } from '@mycelix/sdk/clients/energy';
 *
 * const energy = new EnergyClient(appClient, 'mycelix-energy');
 *
 * // Register a solar project
 * const project = await energy.projects.registerProject({
 *   name: 'Community Solar Farm',
 *   description: 'A 50MW solar installation',
 *   project_type: 'Solar',
 *   location: { latitude: 37.77, longitude: -122.41, country: 'USA', region: 'California' },
 *   capacity_mw: 50,
 *   developer_did: 'did:mycelix:dev1',
 *   financials: { total_cost: 10000000, funded_amount: 0, currency: 'USD', target_irr: 12, payback_years: 7, annual_revenue_estimate: 1500000 },
 * });
 *
 * // Create a trade offer
 * const offer = await energy.grid.createTradeOffer({
 *   seller_did: 'did:mycelix:producer1',
 *   amount_kwh: 1000,
 *   price_per_kwh: 0.12,
 *   currency: 'USD',
 *   available_from: Date.now() * 1000,
 *   available_until: (Date.now() + 86400000) * 1000,
 * });
 *
 * // Pledge an investment
 * const investment = await energy.investments.pledgeInvestment({
 *   project_id: project.entry.Present.id,
 *   investor_did: 'did:mycelix:investor1',
 *   amount: 50000,
 *   currency: 'USD',
 *   shares: 100,
 *   share_percentage: 5,
 *   investment_type: 'Equity',
 * });
 *
 * // Create a regenerative exit contract
 * const contract = await energy.regenerative.createRegenerativeContract({
 *   project_id: project.entry.Present.id,
 *   community_did: 'did:mycelix:community1',
 *   conditions: [{ condition_type: 'CommunityReadiness', threshold: 0.8, current_value: 0, weight: 0.25, satisfied: false }],
 *   target_ownership_percentage: 100,
 *   currency: 'USD',
 * });
 * ```
 */
export class EnergyClient {
  /** Energy project registration and management */
  readonly projects: ProjectsClient;

  /** P2P energy trading */
  readonly grid: GridClient;

  /** Community investment */
  readonly investments: InvestmentsClient;

  /** Regenerative ownership transition */
  readonly regenerative: RegenerativeClient;

  /** Terra Atlas integration and cross-hApp communication */
  readonly bridge: BridgeClient;

  constructor(
    client: AppClient | ZomeCallable,
    _appId: string = 'energy'
  ) {
    const callable = client as ZomeCallable;
    this.projects = new ProjectsClient(callable);
    this.grid = new GridClient(callable);
    this.investments = new InvestmentsClient(callable);
    this.regenerative = new RegenerativeClient(callable);
    this.bridge = new BridgeClient(callable);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Energy hApp clients
 */
export function createEnergyClients(client: ZomeCallable) {
  return {
    projects: new ProjectsClient(client),
    grid: new GridClient(client),
    investments: new InvestmentsClient(client),
    regenerative: new RegenerativeClient(client),
    bridge: new BridgeClient(client),
  };
}

export default EnergyClient;
