/**
 * Energy SDK Types
 *
 * Comprehensive type definitions for the Mycelix Energy hApp SDK.
 *
 * @module @mycelix/sdk/integrations/energy/types
 */

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Error codes for energy operations
 */
export type EnergySdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'NOT_FOUND'
  | 'INSUFFICIENT_CREDITS'
  | 'CAPACITY_EXCEEDED'
  | 'UNAUTHORIZED'
  | 'GRID_ERROR'
  | 'TRADE_FAILED'
  | 'SETTLEMENT_FAILED';

/**
 * Custom error class for Energy SDK operations
 */
export class EnergySdkError extends Error {
  constructor(
    public readonly code: EnergySdkErrorCode,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'EnergySdkError';
  }
}

// ============================================================================
// Energy Sources & Project Types
// ============================================================================

/**
 * Types of energy sources
 */
export type EnergySource =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Nuclear'
  | 'Geothermal'
  | 'Storage'
  | 'Biomass'
  | 'Grid'
  | 'Other';

/**
 * Extended energy source types for detailed asset tracking
 */
export type DetailedEnergySource =
  | 'solar_pv'
  | 'solar_thermal'
  | 'wind_turbine'
  | 'micro_hydro'
  | 'generator_gas'
  | 'generator_diesel'
  | 'generator_dual'
  | 'fuel_cell'
  | 'grid'
  | 'community_solar'
  | 'geothermal';

/**
 * Storage types
 */
export type StorageType =
  | 'battery_lithium'
  | 'battery_lead_acid'
  | 'battery_flow'
  | 'thermal_water'
  | 'thermal_ice'
  | 'pumped_hydro'
  | 'flywheel'
  | 'hydrogen';

/**
 * Project status
 */
export type ProjectStatus =
  | 'Proposed'
  | 'Planning'
  | 'Development'
  | 'Construction'
  | 'Operational'
  | 'Decommissioned';

/**
 * Participant type
 */
export type ParticipantType =
  | 'Producer'
  | 'Consumer'
  | 'Prosumer'
  | 'Operator'
  | 'Investor'
  | 'Storage';

/**
 * Trade status
 */
export type TradeStatus =
  | 'Pending'
  | 'Confirmed'
  | 'Delivered'
  | 'Disputed'
  | 'Cancelled';

/**
 * Settlement status
 */
export type SettlementStatus =
  | 'Pending'
  | 'Confirmed'
  | 'Settled'
  | 'Disputed';

/**
 * Grid condition status
 */
export type GridCondition =
  | 'normal'
  | 'watch'
  | 'advisory'
  | 'emergency'
  | 'critical'
  | 'blackout';

/**
 * Energy priority levels for emergency allocation
 */
export type EnergyPriority =
  | 'critical_medical'
  | 'essential_safety'
  | 'food_preservation'
  | 'water_systems'
  | 'climate_extreme'
  | 'standard'
  | 'deferrable'
  | 'luxury';

// ============================================================================
// Core Entities
// ============================================================================

/**
 * Geographic location
 */
export interface GeoLocation {
  lat: number;
  lng: number;
  address?: string;
}

/**
 * Energy project
 */
export interface EnergyProject {
  id: string;
  name: string;
  description: string;
  source: EnergySource;
  capacity_kw: number;
  location: GeoLocation;
  status: ProjectStatus;
  owner_dao?: string;
  owner_did?: string;
  investment_goal?: number;
  investment_raised: number;
  operating_since?: number;
  annual_generation_mwh?: number;
  carbon_offset_tons?: number;
  created_at: number;
  updated_at: number;
}

/**
 * Input for registering a new project
 */
export interface RegisterProjectInput {
  name: string;
  description: string;
  source: EnergySource;
  capacity_kw: number;
  location: GeoLocation;
  investment_goal?: number;
  owner_dao?: string;
}

/**
 * Input for updating a project
 */
export interface UpdateProjectInput {
  project_id: string;
  name?: string;
  description?: string;
  status?: ProjectStatus;
  capacity_kw?: number;
  investment_goal?: number;
}

/**
 * Energy participant
 */
export interface EnergyParticipant {
  id: string;
  did: string;
  participant_type: ParticipantType;
  sources: EnergySource[];
  capacity_kwh: number;
  location?: GeoLocation;
  reputation_score: number;
  total_produced_kwh: number;
  total_consumed_kwh: number;
  carbon_offset_kg: number;
  active: boolean;
  registered_at: number;
  updated_at: number;
}

/**
 * Input for registering a participant
 */
export interface RegisterParticipantInput {
  participant_type: ParticipantType;
  sources: EnergySource[];
  capacity_kwh: number;
  location?: GeoLocation;
}

/**
 * Input for updating a participant
 */
export interface UpdateParticipantInput {
  participant_id: string;
  sources?: EnergySource[];
  capacity_kwh?: number;
  location?: GeoLocation;
  active?: boolean;
}

// ============================================================================
// Trading & Transactions
// ============================================================================

/**
 * Energy trade
 */
export interface EnergyTrade {
  id: string;
  seller_did: string;
  buyer_did: string;
  amount_kwh: number;
  source: EnergySource;
  price_per_kwh: number;
  currency: string;
  status: TradeStatus;
  carbon_credits_kg?: number;
  settlement_id?: string;
  created_at: number;
  delivered_at?: number;
}

/**
 * Input for creating a trade offer
 */
export interface CreateTradeInput {
  buyer_did: string;
  amount_kwh: number;
  source: EnergySource;
  price_per_kwh: number;
  currency: string;
}

/**
 * Energy settlement for cross-hApp transactions
 */
export interface EnergySettlement {
  id: string;
  source_happ: string;
  seller_did: string;
  buyer_did: string;
  amount_kwh: number;
  price_total?: number;
  carbon_credits_kg: number;
  energy_source: EnergySource;
  status: SettlementStatus;
  created_at: number;
  settled_at?: number;
}

/**
 * Input for requesting energy purchase
 */
export interface RequestPurchaseInput {
  seller_did: string;
  amount_kwh: number;
  max_price_per_kwh?: number;
  source_happ: string;
}

// ============================================================================
// Energy Credits & Carbon
// ============================================================================

/**
 * Energy credit
 */
export interface EnergyCredit {
  id: string;
  holder_did: string;
  amount_kwh: number;
  source: EnergySource;
  carbon_offset_kg: number;
  project_id?: string;
  issued_at: number;
  expires_at?: number;
  used: boolean;
  transferable: boolean;
}

/**
 * Input for issuing energy credits
 */
export interface IssueCreditInput {
  holder_did: string;
  amount_kwh: number;
  source: EnergySource;
  project_id?: string;
  expires_at?: number;
  transferable?: boolean;
}

/**
 * Input for transferring credits
 */
export interface TransferCreditInput {
  credit_id: string;
  recipient_did: string;
}

/**
 * Carbon credit certificate
 */
export interface CarbonCertificate {
  id: string;
  holder_did: string;
  amount_kg_co2: number;
  energy_source: EnergySource;
  energy_amount_kwh: number;
  project_id?: string;
  issued_by: string;
  issued_at: number;
  expires_at?: number;
  retired: boolean;
  retirement_reason?: string;
}

/**
 * Input for retiring carbon credits
 */
export interface RetireCarbonInput {
  certificate_id: string;
  reason: string;
}

// ============================================================================
// Investments
// ============================================================================

/**
 * Project investment
 */
export interface Investment {
  id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  ownership_percentage: number;
  dividend_rate?: number;
  invested_at: number;
  status: 'Active' | 'Exited' | 'Pending';
}

/**
 * Input for investing in a project
 */
export interface InvestInput {
  project_id: string;
  amount: number;
  currency: string;
}

/**
 * Investment dividend
 */
export interface Dividend {
  id: string;
  investment_id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  period_start: number;
  period_end: number;
  energy_generated_kwh: number;
  distributed_at: number;
}

// ============================================================================
// Grid & Production
// ============================================================================

/**
 * Energy production reading
 */
export interface ProductionReading {
  id: string;
  participant_id: string;
  project_id?: string;
  timestamp: number;
  production_kwh: number;
  consumption_kwh: number;
  net_export_kwh: number;
  source: EnergySource;
  verified: boolean;
  verifier_did?: string;
}

/**
 * Input for reporting production
 */
export interface ReportProductionInput {
  participant_id: string;
  project_id?: string;
  production_kwh: number;
  consumption_kwh: number;
  source: EnergySource;
}

/**
 * Grid status information
 */
export interface GridStatus {
  region: string;
  timestamp: number;
  condition: GridCondition;
  current_load_mw: number;
  capacity_mw: number;
  reserve_margin_percent: number;
  frequency_hz: number;
  renewable_percent: number;
  current_price_mwh?: number;
  alerts: GridAlert[];
}

/**
 * Grid alert
 */
export interface GridAlert {
  id: string;
  type: 'conservation' | 'emergency' | 'outage' | 'price_spike' | 'weather';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  region: string;
  start_time: number;
  end_time?: number;
}

/**
 * Grid balance request
 */
export interface GridBalanceRequest {
  id: string;
  requester_did: string;
  type: 'supply' | 'demand';
  amount_kwh: number;
  max_price?: number;
  priority: EnergyPriority;
  expires_at: number;
  fulfilled: boolean;
  fulfilled_by?: string[];
}

/**
 * Input for requesting grid balance
 */
export interface RequestGridBalanceInput {
  type: 'supply' | 'demand';
  amount_kwh: number;
  max_price?: number;
  priority?: EnergyPriority;
  expires_in_hours?: number;
}

// ============================================================================
// Assets & Sovereignty
// ============================================================================

/**
 * Energy asset (generation or storage equipment)
 */
export interface EnergyAsset {
  id: string;
  owner_did: string;
  asset_type: DetailedEnergySource | StorageType;
  make?: string;
  model?: string;
  serial_number?: string;
  location: {
    address?: string;
    zip_code: string;
    coordinates?: GeoLocation;
    installation_type: 'rooftop' | 'ground_mount' | 'carport' | 'building_integrated' | 'portable';
  };
  capacity_kw: number;
  capacity_kwh?: number;
  efficiency_percent: number;
  degradation_per_year: number;
  current_age_years: number;
  purchase_cost: number;
  installation_cost: number;
  annual_maintenance: number;
  warranty_years: number;
  grid_connected: boolean;
  net_metering_enabled: boolean;
  status: 'active' | 'maintenance' | 'offline' | 'decommissioned';
  verified: boolean;
  verification_method?: 'utility_bill' | 'photo' | 'installer_cert' | 'inspection';
  created_at: number;
  updated_at: number;
}

/**
 * Storage asset with additional battery properties
 */
export interface StorageAsset extends EnergyAsset {
  storage_type: StorageType;
  usable_capacity_kwh: number;
  depth_of_discharge: number;
  round_trip_efficiency: number;
  max_charge_rate_kw: number;
  max_discharge_rate_kw: number;
  cycle_life: number;
  current_cycles: number;
  state_of_charge: number;
  state_of_health: number;
  can_provide_backup: boolean;
  islanding_capable: boolean;
}

/**
 * Input for registering an energy asset
 */
export interface RegisterAssetInput {
  asset_type: DetailedEnergySource | StorageType;
  make?: string;
  model?: string;
  serial_number?: string;
  location: {
    address?: string;
    zip_code: string;
    coordinates?: GeoLocation;
    installation_type: 'rooftop' | 'ground_mount' | 'carport' | 'building_integrated' | 'portable';
  };
  capacity_kw: number;
  capacity_kwh?: number;
  efficiency_percent?: number;
  purchase_cost?: number;
  installation_cost?: number;
  installation_date?: number;
}

/**
 * Household energy profile
 */
export interface HouseholdProfile {
  id: string;
  owner_did: string;
  name: string;
  zip_code: string;
  coordinates?: GeoLocation;
  utility_provider?: string;
  rate_plan?: string;
  building_type: 'single_family' | 'multi_family' | 'apartment' | 'condo' | 'mobile' | 'commercial';
  square_feet?: number;
  heating_type: 'electric' | 'gas' | 'oil' | 'propane' | 'heat_pump' | 'wood' | 'none';
  cooling_type: 'central_ac' | 'window_ac' | 'heat_pump' | 'evaporative' | 'none';
  water_heating_type: 'electric' | 'gas' | 'solar' | 'heat_pump' | 'tankless';
  occupants: number;
  medical_equipment: MedicalEquipment[];
  ev_charging: boolean;
  assets: EnergyAsset[];
  emergency_priority: EnergyPriority;
  created_at: number;
  updated_at: number;
}

/**
 * Medical equipment requiring power
 */
export interface MedicalEquipment {
  type: 'oxygen_concentrator' | 'cpap' | 'bipap' | 'ventilator' | 'dialysis' |
        'infusion_pump' | 'refrigerated_meds' | 'electric_wheelchair' | 'other';
  power_requirement_watts: number;
  battery_backup_hours?: number;
  criticality: 'life_sustaining' | 'important' | 'convenience';
}

/**
 * Input for creating household profile
 */
export interface CreateHouseholdInput {
  name: string;
  zip_code: string;
  coordinates?: GeoLocation;
  utility_provider?: string;
  building_type: HouseholdProfile['building_type'];
  square_feet?: number;
  heating_type: HouseholdProfile['heating_type'];
  cooling_type: HouseholdProfile['cooling_type'];
  water_heating_type: HouseholdProfile['water_heating_type'];
  occupants: number;
  medical_equipment?: MedicalEquipment[];
  ev_charging?: boolean;
}

// ============================================================================
// Community & Statistics
// ============================================================================

/**
 * Community energy summary
 */
export interface CommunityEnergySummary {
  community_id: string;
  timestamp: number;
  total_generation_capacity_kw: number;
  total_storage_capacity_kwh: number;
  households_with_solar: number;
  households_with_storage: number;
  total_households: number;
  current_generation_kw: number;
  current_consumption_kw: number;
  current_storage_kwh: number;
  grid_import_kw: number;
  grid_export_kw: number;
  today_generation_kwh: number;
  today_consumption_kwh: number;
  status: 'surplus' | 'balanced' | 'deficit' | 'emergency';
  active_offers: number;
  active_requests: number;
  today_trades: number;
  today_traded_kwh: number;
}

/**
 * Grid statistics
 */
export interface GridStatistics {
  total_production_kwh: number;
  total_consumption_kwh: number;
  net_balance_kwh: number;
  participant_count: number;
  project_count: number;
  total_capacity_kw: number;
  total_carbon_offset_kg: number;
  by_source: Record<EnergySource, {
    production_kwh: number;
    capacity_kw: number;
    project_count: number;
  }>;
}

/**
 * Participant statistics
 */
export interface ParticipantStatistics {
  participant_id: string;
  total_produced_kwh: number;
  total_consumed_kwh: number;
  total_traded_kwh: number;
  total_credits_earned: number;
  total_credits_used: number;
  carbon_offset_kg: number;
  trade_count: number;
  reputation_score: number;
  by_source: Record<EnergySource, {
    produced_kwh: number;
    consumed_kwh: number;
  }>;
}

/**
 * Project statistics
 */
export interface ProjectStatistics {
  project_id: string;
  total_generated_kwh: number;
  total_invested: number;
  investor_count: number;
  average_generation_kwh_per_day: number;
  capacity_factor: number;
  carbon_offset_kg: number;
  dividends_distributed: number;
  uptime_percent: number;
}

// ============================================================================
// Query & Search
// ============================================================================

/**
 * Project search parameters
 */
export interface ProjectSearchParams {
  source?: EnergySource;
  status?: ProjectStatus;
  min_capacity_kw?: number;
  max_capacity_kw?: number;
  location?: GeoLocation;
  radius_km?: number;
  owner_dao?: string;
  has_investment_open?: boolean;
  limit?: number;
  offset?: number;
}

/**
 * Participant search parameters
 */
export interface ParticipantSearchParams {
  participant_type?: ParticipantType;
  sources?: EnergySource[];
  min_capacity_kwh?: number;
  location?: GeoLocation;
  radius_km?: number;
  active_only?: boolean;
  limit?: number;
  offset?: number;
}

/**
 * Trade search parameters
 */
export interface TradeSearchParams {
  seller_did?: string;
  buyer_did?: string;
  source?: EnergySource;
  status?: TradeStatus;
  min_amount_kwh?: number;
  from_date?: number;
  to_date?: number;
  limit?: number;
  offset?: number;
}

/**
 * Available energy query
 */
export interface AvailableEnergyQuery {
  source_happ: string;
  location?: {
    lat: number;
    lng: number;
    radius_km: number;
  };
  energy_sources?: EnergySource[];
  min_amount_kwh?: number;
}

/**
 * Available energy result
 */
export interface AvailableEnergy {
  participant_id: string;
  participant_did: string;
  available_kwh: number;
  source: EnergySource;
  price_per_kwh?: number;
  location?: GeoLocation;
  reputation_score: number;
}

// ============================================================================
// Events
// ============================================================================

/**
 * Energy event types
 */
export type EnergyEventType =
  | 'ProjectRegistered'
  | 'ProjectUpdated'
  | 'ParticipantRegistered'
  | 'TradeCreated'
  | 'TradeConfirmed'
  | 'TradeDelivered'
  | 'CreditIssued'
  | 'CreditUsed'
  | 'CreditTransferred'
  | 'InvestmentMade'
  | 'DividendDistributed'
  | 'SettlementCompleted'
  | 'GridBalanceRequest'
  | 'GridBalanceFulfilled'
  | 'AlertIssued';

/**
 * Energy event
 */
export interface EnergyEvent {
  id: string;
  event_type: EnergyEventType;
  source_happ: string;
  participant_did?: string;
  project_id?: string;
  trade_id?: string;
  settlement_id?: string;
  amount_kwh?: number;
  payload: Record<string, unknown>;
  timestamp: number;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Energy client configuration
 */
export interface EnergyClientConfig {
  /** Role ID for the energy DNA */
  roleId: string;
  /** Projects zome name */
  projectsZome: string;
  /** Participants zome name */
  participantsZome: string;
  /** Trading zome name */
  tradingZome: string;
  /** Credits zome name */
  creditsZome: string;
  /** Investments zome name */
  investmentsZome: string;
  /** Bridge zome name */
  bridgeZome: string;
}

/**
 * Connection options for creating a new client
 */
export interface EnergyConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
}
