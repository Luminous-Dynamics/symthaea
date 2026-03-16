/**
 * Resilience Client — Typed wrappers for TEND, food, mutual-aid, and emergency zome calls.
 *
 * In demo mode (no conductor), returns mock data for UI development.
 * In live mode, proxies to callZome() with correct role/zome/fn routing.
 */

import { callZome, isConnected } from './conductor';
import { enqueue } from './offline-queue';

// ============================================================================
// Types — TEND (Time Exchange)
// ============================================================================

export interface BalanceInfo {
  member_did: string;
  dao_did: string;
  balance: number;
  total_earned: number;
  total_spent: number;
  exchange_count: number;
}

export interface ExchangeRecord {
  id: string;
  provider_did: string;
  receiver_did: string;
  hours: number;
  service_description: string;
  service_category: string;
  status: 'Proposed' | 'Confirmed' | 'Disputed' | 'Cancelled';
  timestamp: number;
}

export interface ServiceListing {
  id: string;
  provider_did: string;
  title: string;
  description: string;
  category: string;
  hours_estimate: number;
  dao_did: string;
  created_at: number;
  active: boolean;
}

export interface ServiceRequest {
  id: string;
  requester_did: string;
  title: string;
  description: string;
  category: string;
  hours_budget: number;
  urgency: string;
  dao_did: string;
  created_at: number;
  open: boolean;
}

export type OracleTier = 'Normal' | 'Elevated' | 'High' | 'Emergency';

export interface OracleState {
  vitality: number;
  tier: OracleTier;
  updated_at: number;
}

// ============================================================================
// Types — Price Oracle
// ============================================================================

export interface PriceReportInput {
  item: string;
  price_tend: number;
  evidence: string;
}

export interface ConsensusResult {
  item: string;
  median_price: number;
  reporter_count: number;
  std_dev: number;
  window_start: number;
  /** Average accuracy of reporters (0.0-1.0). Higher = more trustworthy consensus. */
  signal_integrity: number;
}

export interface ReporterAccuracyInfo {
  reporter_did: string;
  accuracy_score: number;
  report_count: number;
}

export interface BasketItemInput {
  item: string;
  weight: number;
}

export interface ItemPriceResult {
  item: string;
  price: number;
  weight: number;
  weighted_price: number;
}

export interface BasketIndexResult {
  basket_name: string;
  index: number;
  item_prices: ItemPriceResult[];
  computed_at: number;
}

export interface VolatilityResult {
  basket_name: string;
  current_index: number;
  previous_index: number;
  weekly_change: number;
  recommended_tier: string;
  escalated: boolean;
}

// ============================================================================
// Types — Food Production
// ============================================================================

export interface FoodPlot {
  id: string;
  owner_did: string;
  name: string;
  location: string;
  area_sqm: number;
  plot_type: string;
  created_at: number;
}

export interface HarvestRecord {
  id: string;
  plot_id: string;
  crop_name: string;
  quantity_kg: number;
  harvested_at: number;
  notes: string;
}

export interface PlantingRecord {
  id: string;
  plot_id: string;
  crop_name: string;
  planted_at: number;
  expected_harvest: number;
  area_sqm: number;
}

export type ResourceType =
  | 'KitchenWaste'
  | 'GreenWaste'
  | 'Biochar'
  | 'Vermicompost'
  | 'Digestate'
  | 'Manure'
  | 'Mulch'
  | 'CoverCrop'
  | 'Inoculant';

export interface NutrientProfile {
  nitrogen_pct: number;
  phosphorus_pct: number;
  potassium_pct: number;
}

export interface ResourceInput {
  id: string;
  plot_id: string | null;
  input_type: ResourceType;
  quantity_kg: number;
  nutrient_estimate: NutrientProfile;
  contributor_did: string;
  contributed_at: number;
  notes: string;
}

export interface NutrientSummary {
  total_kg_by_type: [string, number][];
  total_nitrogen_kg: number;
  total_phosphorus_kg: number;
  total_potassium_kg: number;
  total_contributions: number;
}

export const RESOURCE_TYPES: ResourceType[] = [
  'KitchenWaste',
  'GreenWaste',
  'Biochar',
  'Vermicompost',
  'Digestate',
  'Manure',
  'Mulch',
  'CoverCrop',
  'Inoculant',
];

export const RESOURCE_TYPE_LABELS: Record<ResourceType, string> = {
  KitchenWaste: 'Kitchen Waste',
  GreenWaste: 'Green Waste',
  Biochar: 'Biochar',
  Vermicompost: 'Vermicompost',
  Digestate: 'Digestate',
  Manure: 'Manure',
  Mulch: 'Mulch',
  CoverCrop: 'Cover Crop',
  Inoculant: 'Inoculant',
};

// ============================================================================
// Types — Mutual Aid / Timebank
// ============================================================================

export interface AidOffer {
  id: string;
  provider_did: string;
  title: string;
  description: string;
  category: string;
  hours_available: number;
  recurring: boolean;
  created_at: number;
}

export interface AidRequest {
  id: string;
  requester_did: string;
  title: string;
  description: string;
  category: string;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  hours_needed: number;
  created_at: number;
  fulfilled: boolean;
}

// ============================================================================
// Types — Emergency Comms
// ============================================================================

export type EmergencyPriority = 'Flash' | 'Immediate' | 'Priority' | 'Routine';

export interface EmergencyMessage {
  id: string;
  sender_did: string;
  channel_id: string;
  content: string;
  priority: EmergencyPriority;
  sent_at: number;
  synced: boolean;
}

export interface EmergencyChannel {
  id: string;
  name: string;
  description: string;
  created_by: string;
  created_at: number;
  member_count: number;
}

// ============================================================================
// TEND Client
// ============================================================================

// Read DAO identifier from community config (set in community-config.json)
import { getDefaultDao } from './community';
const DEFAULT_DAO = getDefaultDao();

export async function getBalance(memberDid: string, daoDid = DEFAULT_DAO): Promise<BalanceInfo> {
  if (!isConnected()) return mockBalance(memberDid, daoDid);
  return callZome<BalanceInfo>({
    role_name: 'finance',
    zome_name: 'tend',
    fn_name: 'get_balance',
    payload: { member_did: memberDid, dao_did: daoDid },
  });
}

export async function recordExchange(
  receiverDid: string,
  hours: number,
  serviceDescription: string,
  serviceCategory: string,
  daoDid = DEFAULT_DAO,
): Promise<ExchangeRecord> {
  if (!isConnected()) {
    // Queue for replay when conductor reconnects
    const queued = await enqueue('tend', 'recordExchange', {
      receiverDid,
      hours,
      serviceDescription,
      serviceCategory,
      daoDid,
    });
    const mock = mockExchange(receiverDid, hours, serviceDescription, serviceCategory);
    // Use the queue ID if available so the UI record can be correlated later
    if (queued) mock.id = `queued-${queued.id}`;
    return mock;
  }
  return callZome<ExchangeRecord>({
    role_name: 'finance',
    zome_name: 'tend',
    fn_name: 'record_exchange',
    payload: {
      receiver_did: receiverDid,
      hours,
      service_description: serviceDescription,
      service_category: serviceCategory,
      dao_did: daoDid,
    },
  });
}

export async function getDaoListings(daoDid = DEFAULT_DAO): Promise<ServiceListing[]> {
  if (!isConnected()) return mockListings();
  return callZome<ServiceListing[]>({
    role_name: 'finance',
    zome_name: 'tend',
    fn_name: 'get_dao_listings',
    payload: { dao_did: daoDid },
  });
}

export async function getDaoRequests(daoDid = DEFAULT_DAO): Promise<ServiceRequest[]> {
  if (!isConnected()) return mockRequests();
  return callZome<ServiceRequest[]>({
    role_name: 'finance',
    zome_name: 'tend',
    fn_name: 'get_dao_requests',
    payload: { dao_did: daoDid },
  });
}

export async function getOracleState(): Promise<OracleState> {
  if (!isConnected()) return { vitality: 72, tier: 'Normal', updated_at: Date.now() };
  return callZome<OracleState>({
    role_name: 'finance',
    zome_name: 'tend',
    fn_name: 'get_oracle_state',
    payload: null,
  });
}

// ============================================================================
// Food Production Client
// ============================================================================

export async function getAllPlots(daoDid = DEFAULT_DAO): Promise<FoodPlot[]> {
  if (!isConnected()) return mockPlots();
  return callZome<FoodPlot[]>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'get_all_plots',
    payload: { dao_did: daoDid },
  });
}

export async function registerPlot(
  name: string,
  location: string,
  areaSqm: number,
  plotType: string,
): Promise<FoodPlot> {
  if (!isConnected()) return mockPlot(name, location, areaSqm, plotType);
  // Zome expects a Plot entry struct directly
  return callZome<FoodPlot>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'register_plot',
    payload: { name, location, area_sqm: areaSqm, plot_type: plotType, steward: null },
  });
}

export async function recordHarvest(
  plotId: string,
  cropName: string,
  quantityKg: number,
  notes: string,
): Promise<HarvestRecord> {
  if (!isConnected()) return mockHarvest(plotId, cropName, quantityKg, notes);
  // Zome expects a YieldRecord entry struct
  return callZome<HarvestRecord>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'record_harvest',
    payload: { crop_hash: plotId, quantity_kg: quantityKg, quality: 'good', notes },
  });
}

// ============================================================================
// Resource Input Client
// ============================================================================

export async function logResourceInput(
  inputType: ResourceType,
  quantityKg: number,
  plotId: string | null,
  notes: string,
): Promise<ResourceInput> {
  if (!isConnected()) return mockResourceInput(inputType, quantityKg, plotId, notes);
  return callZome<ResourceInput>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'log_resource_input',
    payload: { plot_hash: plotId, input_type: inputType, quantity_kg: quantityKg, notes },
  });
}

export async function getCommunityInputs(limit = 50): Promise<ResourceInput[]> {
  if (!isConnected()) return mockResourceInputs();
  return callZome<ResourceInput[]>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'get_community_inputs',
    payload: { limit },
  });
}

export async function getNutrientSummary(): Promise<NutrientSummary> {
  if (!isConnected()) return mockNutrientSummary();
  return callZome<NutrientSummary>({
    role_name: 'commons_care',
    zome_name: 'food_production',
    fn_name: 'get_nutrient_summary',
    payload: null,
  });
}

// ============================================================================
// Mutual Aid Client
// ============================================================================

export async function getServiceOffers(daoDid = DEFAULT_DAO): Promise<AidOffer[]> {
  if (!isConnected()) return mockOffers();
  // Zome uses search_offers with SearchOffersInput
  return callZome<AidOffer[]>({
    role_name: 'commons_care',
    zome_name: 'mutualaid_timebank',
    fn_name: 'search_offers',
    payload: { category: null, keyword: null, limit: 100 },
  });
}

export async function createServiceOffer(
  title: string,
  description: string,
  category: string,
  hoursAvailable: number,
): Promise<AidOffer> {
  if (!isConnected()) return mockOffer(title, description, category, hoursAvailable);
  return callZome<AidOffer>({
    role_name: 'commons_care',
    zome_name: 'mutualaid_timebank',
    fn_name: 'create_service_offer',
    payload: { title, description, category, hours_available: hoursAvailable },
  });
}

export async function getServiceRequests(daoDid = DEFAULT_DAO): Promise<AidRequest[]> {
  if (!isConnected()) return mockAidRequests();
  // Zome uses search_requests with SearchRequestsInput
  return callZome<AidRequest[]>({
    role_name: 'commons_care',
    zome_name: 'mutualaid_timebank',
    fn_name: 'search_requests',
    payload: { category: null, urgency: null, keyword: null, limit: 100 },
  });
}

export async function createServiceRequest(
  title: string,
  description: string,
  category: string,
  urgency: AidRequest['urgency'],
  hoursNeeded: number,
): Promise<AidRequest> {
  if (!isConnected()) return mockAidRequest(title, description, category, urgency, hoursNeeded);
  return callZome<AidRequest>({
    role_name: 'commons_care',
    zome_name: 'mutualaid_timebank',
    fn_name: 'create_service_request',
    payload: { title, description, category, urgency, hours_needed: hoursNeeded },
  });
}

// ============================================================================
// Emergency Comms Client
// ============================================================================

export async function getChannels(): Promise<EmergencyChannel[]> {
  if (!isConnected()) return mockChannels();
  return callZome<EmergencyChannel[]>({
    role_name: 'civic',
    zome_name: 'emergency_comms',
    fn_name: 'get_channels',
    payload: null,
  });
}

export async function createChannel(name: string, description: string): Promise<EmergencyChannel> {
  if (!isConnected()) return mockChannel(name, description);
  return callZome<EmergencyChannel>({
    role_name: 'civic',
    zome_name: 'emergency_comms',
    fn_name: 'create_channel',
    payload: { name, description },
  });
}

export async function sendMessage(
  channelId: string,
  content: string,
  priority: EmergencyPriority,
): Promise<EmergencyMessage> {
  if (!isConnected()) return mockMessage(channelId, content, priority);
  return callZome<EmergencyMessage>({
    role_name: 'civic',
    zome_name: 'emergency_comms',
    fn_name: 'send_message',
    payload: { channel_id: channelId, content, priority },
  });
}

export async function getMessages(channelId: string): Promise<EmergencyMessage[]> {
  if (!isConnected()) return mockMessages(channelId);
  // Zome uses get_channel_messages with channel ActionHash
  return callZome<EmergencyMessage[]>({
    role_name: 'civic',
    zome_name: 'emergency_comms',
    fn_name: 'get_channel_messages',
    payload: channelId,
  });
}

// ============================================================================
// Price Oracle Client
// ============================================================================

export async function reportPrice(input: PriceReportInput): Promise<unknown> {
  if (!isConnected()) return mockPriceReport(input);
  return callZome({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'report_price',
    payload: input,
  });
}

export async function getConsensusPrice(item: string): Promise<ConsensusResult> {
  if (!isConnected()) return mockConsensus(item);
  return callZome<ConsensusResult>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_consensus_price',
    payload: { item },
  });
}

export async function defineBasket(
  name: string,
  items: BasketItemInput[],
): Promise<unknown> {
  if (!isConnected()) return { name, items };
  return callZome({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'define_basket',
    payload: { name, items },
  });
}

export async function getBasketIndex(basketName: string): Promise<BasketIndexResult> {
  if (!isConnected()) return mockBasketIndex(basketName);
  return callZome<BasketIndexResult>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_basket_index',
    payload: { basket_name: basketName },
  });
}

export async function computeVolatility(basketName: string): Promise<VolatilityResult> {
  if (!isConnected()) return mockVolatility(basketName);
  return callZome<VolatilityResult>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'compute_volatility',
    payload: { basket_name: basketName },
  });
}

export async function getItemReports(item: string): Promise<unknown[]> {
  if (!isConnected()) return [];
  return callZome<unknown[]>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_item_reports',
    payload: item,
  });
}

export async function getAllBaskets(): Promise<unknown[]> {
  if (!isConnected()) return [];
  return callZome<unknown[]>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_all_baskets',
    payload: null,
  });
}

export async function getReporterAccuracy(reporterDid: string): Promise<ReporterAccuracyInfo> {
  if (!isConnected()) return { reporter_did: reporterDid, accuracy_score: 0.92, report_count: 12 };
  return callZome<ReporterAccuracyInfo>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_reporter_accuracy',
    payload: reporterDid,
  });
}

export async function getTopReporters(item: string, limit = 10): Promise<ReporterAccuracyInfo[]> {
  if (!isConnected()) return mockTopReporters();
  return callZome<ReporterAccuracyInfo[]>({
    role_name: 'finance',
    zome_name: 'price_oracle',
    fn_name: 'get_top_reporters',
    payload: { item, limit },
  });
}

// ============================================================================
// Types — Care Circles
// ============================================================================

export interface CareCircle {
  id: string;
  name: string;
  description: string;
  circle_type: string;
  location: string;
  max_members: number;
  member_count?: number;
  created_at: number;
}

export interface CircleMembership {
  id: string;
  circle_id: string;
  member_did: string;
  role: MemberRole;
  joined_at: number;
  active: boolean;
}

export type MemberRole = 'Organizer' | 'Member' | 'Observer';

// ============================================================================
// Types — Emergency Shelter
// ============================================================================

export interface Building {
  id: string;
  name: string;
  address: string;
  total_units: number;
  available_units: number;
}

export interface HousingUnit {
  id: string;
  building_id: string;
  unit_number: string;
  unit_type: string;
  bedrooms: number;
  bathrooms: number;
  square_meters: number;
  floor: number;
  status: string;
  accessibility_features: string[];
}

// ============================================================================
// Types — Supply Tracking
// ============================================================================

export interface InventoryItem {
  id: string;
  name: string;
  description?: string;
  category: string;
  sku: string;
  unit: string;
  reorder_point: number;
  reorder_quantity: number;
}

export interface StockLevel {
  id: string;
  item_id: string;
  quantity: number;
  location: string;
  recorded_by: string;
  recorded_at: number;
  notes?: string;
}

export interface LowStockItem {
  item: InventoryItem;
  total_stock: number;
}

// ============================================================================
// Types — Water Safety
// ============================================================================

export interface WaterSystem {
  id: string;
  name: string;
  system_type: string;
  capacity_liters: number;
  catchment_area_sqm?: number;
  efficiency_percent: number;
  owner_did: string;
  location_lat: number;
  location_lon: number;
  installed_at: number;
}

export interface StorageTank {
  id: string;
  system_id: string;
  capacity_liters: number;
  current_level_liters: number;
  material: string;
}

export interface WaterReading {
  id: string;
  source_id: string;
  parameter: string;
  value: number;
  unit: string;
  location: string;
  recorded_at: number;
  recorder_did: string;
}

export interface ContaminationAlert {
  id: string;
  source_id: string;
  contaminant: string;
  measured_value: number;
  threshold_value: number;
  severity: string;
  reported_at: number;
  reported_by: string;
}

// ============================================================================
// Types — Household (Hearth)
// ============================================================================

export interface Hearth {
  id: string;
  name: string;
  member_count: number;
  created_at: number;
}

export interface EmergencyPlan {
  id: string;
  hearth_id: string;
  contacts: EmergencyContact[];
  meeting_points: string[];
  last_reviewed: number;
}

export interface EmergencyContact {
  name: string;
  phone: string;
  relationship: string;
  email?: string;
}

export interface HearthAlert {
  id: string;
  hearth_id: string;
  alert_type: HearthAlertType;
  severity: HearthAlertSeverity;
  message: string;
  created_at: number;
  resolved: boolean;
}

export type HearthAlertType = 'Fire' | 'Flood' | 'Medical' | 'Violence' | 'Other';
export type HearthAlertSeverity = 'Low' | 'Medium' | 'High' | 'Critical';
export type SafetyStatus = 'Safe' | 'NeedHelp' | 'Evacuating' | 'Unknown';
export type HearthResourceType = 'Tools' | 'Food' | 'Water' | 'Medical' | 'Shelter' | 'Communications';

export interface SafetyCheckIn {
  id: string;
  alert_id: string;
  member_did: string;
  status: SafetyStatus;
  location_hint: string | null;
  checked_in_at: number;
}

export interface SharedResource {
  id: string;
  hearth_id: string;
  name: string;
  description: string;
  resource_type: HearthResourceType;
  current_holder: string | null;
  condition: string;
  location: string;
}

export interface ResourceLoan {
  id: string;
  resource_id: string;
  borrower_did: string;
  lent_at: number;
  returned_at: number | null;
}

// ============================================================================
// Types — Community Knowledge
// ============================================================================

export interface KnowledgeClaim {
  id: string;
  author_did: string;
  content: string;
  tags: string[];
  confidence: number;
  e_score: number;
  n_score: number;
  m_score: number;
  created_at: number;
}

export interface GraphStats {
  total_claims: number;
  total_relationships: number;
  total_ontologies: number;
  total_concepts: number;
}

// ============================================================================
// Care Circles Client
// ============================================================================

export async function getAllCareCircles(): Promise<CareCircle[]> {
  if (!isConnected()) return mockCareCircles();
  return callZome<CareCircle[]>({
    role_name: 'commons_care',
    zome_name: 'care_circles',
    fn_name: 'get_all_circles',
    payload: null,
  });
}

export async function getCirclesByType(circleType: string): Promise<CareCircle[]> {
  if (!isConnected()) return mockCareCircles().filter((c) => c.circle_type === circleType);
  return callZome<CareCircle[]>({
    role_name: 'commons_care',
    zome_name: 'care_circles',
    fn_name: 'get_circles_by_type',
    payload: { circle_type: circleType },
  });
}

export async function getCircleMembers(circleHash: string): Promise<CircleMembership[]> {
  if (!isConnected()) return mockCircleMembers(circleHash);
  return callZome<CircleMembership[]>({
    role_name: 'commons_care',
    zome_name: 'care_circles',
    fn_name: 'get_circle_members',
    payload: circleHash,
  });
}

export async function joinCircle(circleHash: string, role: MemberRole = 'Member'): Promise<CircleMembership> {
  if (!isConnected()) return {
    id: `cm-${Date.now()}`,
    circle_id: circleHash,
    member_did: 'self.did',
    role,
    joined_at: Date.now(),
    active: true,
  };
  return callZome<CircleMembership>({
    role_name: 'commons_care',
    zome_name: 'care_circles',
    fn_name: 'join_circle',
    payload: { circle_hash: circleHash, role },
  });
}

// ============================================================================
// Emergency Shelter Client
// ============================================================================

export async function getAvailableUnits(): Promise<HousingUnit[]> {
  if (!isConnected()) return mockHousingUnits();
  return callZome<HousingUnit[]>({
    role_name: 'commons_care',
    zome_name: 'housing_units',
    fn_name: 'get_available_units',
    payload: null,
  });
}

export async function getBuildingUnits(buildingHash: string): Promise<HousingUnit[]> {
  if (!isConnected()) return mockHousingUnits().filter((u) => u.building_id === buildingHash);
  return callZome<HousingUnit[]>({
    role_name: 'commons_care',
    zome_name: 'housing_units',
    fn_name: 'get_building_units',
    payload: buildingHash,
  });
}

// ============================================================================
// Supply Tracking Client
// ============================================================================

export async function getAllInventoryItems(): Promise<InventoryItem[]> {
  if (!isConnected()) return mockInventoryItems();
  return callZome<InventoryItem[]>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'get_all_items',
    payload: null,
  });
}

export async function getItemsByCategory(category: string): Promise<InventoryItem[]> {
  if (!isConnected()) return mockInventoryItems().filter((i) => i.category === category);
  return callZome<InventoryItem[]>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'get_items_by_category',
    payload: { category },
  });
}

export async function getLowStockItems(): Promise<LowStockItem[]> {
  if (!isConnected()) return mockLowStockItems();
  return callZome<LowStockItem[]>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'get_low_stock_items',
    payload: null,
  });
}

export async function getStockLevels(itemHash: string): Promise<StockLevel[]> {
  if (!isConnected()) return mockStockLevels(itemHash);
  return callZome<StockLevel[]>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'get_stock_levels',
    payload: itemHash,
  });
}

export async function addInventoryItem(item: Omit<InventoryItem, 'id'>): Promise<InventoryItem> {
  if (!isConnected()) return { ...item, id: `inv-${Date.now()}` };
  return callZome<InventoryItem>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'add_item',
    payload: item,
  });
}

export async function updateStockLevel(
  itemHash: string,
  quantity: number,
  notes?: string,
): Promise<StockLevel> {
  if (!isConnected()) return {
    id: `sl-${Date.now()}`,
    item_id: itemHash,
    quantity,
    location: 'Roodepoort Community Store',
    recorded_by: 'self.did',
    recorded_at: Date.now(),
    notes,
  };
  return callZome<StockLevel>({
    role_name: 'supplychain',
    zome_name: 'inventory_coordinator',
    fn_name: 'record_stock_level',
    payload: { item_hash: itemHash, quantity, notes },
  });
}

// ============================================================================
// Water Safety Client
// ============================================================================

export async function getAllWaterSystems(): Promise<WaterSystem[]> {
  if (!isConnected()) return mockWaterSystems();
  return callZome<WaterSystem[]>({
    role_name: 'commons_care',
    zome_name: 'water_capture',
    fn_name: 'get_all_systems',
    payload: null,
  });
}

export async function getMyWaterSystems(): Promise<WaterSystem[]> {
  if (!isConnected()) return mockWaterSystems().filter((s) => s.owner_did === 'self.did');
  return callZome<WaterSystem[]>({
    role_name: 'commons_care',
    zome_name: 'water_capture',
    fn_name: 'get_my_systems',
    payload: null,
  });
}

export async function registerWaterSystem(system: Omit<WaterSystem, 'id'>): Promise<WaterSystem> {
  if (!isConnected()) return { ...system, id: `ws-${Date.now()}` };
  return callZome<WaterSystem>({
    role_name: 'commons_care',
    zome_name: 'water_capture',
    fn_name: 'register_harvest_system',
    payload: system,
  });
}

export async function updateTankLevel(tankHash: string, newLevelLiters: number): Promise<StorageTank> {
  if (!isConnected()) return {
    id: tankHash,
    system_id: 'ws-001',
    capacity_liters: 5000,
    current_level_liters: newLevelLiters,
    material: 'Polyethylene',
  };
  return callZome<StorageTank>({
    role_name: 'commons_care',
    zome_name: 'water_capture',
    fn_name: 'update_tank_level',
    payload: { tank_hash: tankHash, new_level_liters: newLevelLiters },
  });
}

export async function submitWaterReading(reading: Omit<WaterReading, 'id'>): Promise<WaterReading> {
  if (!isConnected()) return { ...reading, id: `wr-${Date.now()}` };
  return callZome<WaterReading>({
    role_name: 'commons_care',
    zome_name: 'water_purity',
    fn_name: 'submit_reading',
    payload: reading,
  });
}

export async function getActiveWaterAlerts(): Promise<ContaminationAlert[]> {
  if (!isConnected()) return mockWaterAlerts();
  return callZome<ContaminationAlert[]>({
    role_name: 'commons_care',
    zome_name: 'water_purity',
    fn_name: 'get_active_alerts',
    payload: null,
  });
}

// ============================================================================
// Household (Hearth) Client
// ============================================================================

export async function getMyHearths(): Promise<Hearth[]> {
  if (!isConnected()) return mockHearths();
  return callZome<Hearth[]>({
    role_name: 'hearth',
    zome_name: 'hearth_kinship',
    fn_name: 'get_my_hearths',
    payload: null,
  });
}

export async function getEmergencyPlan(hearthHash: string): Promise<EmergencyPlan> {
  if (!isConnected()) return mockEmergencyPlan(hearthHash);
  return callZome<EmergencyPlan>({
    role_name: 'hearth',
    zome_name: 'hearth_emergency',
    fn_name: 'get_emergency_plan',
    payload: hearthHash,
  });
}

export async function createEmergencyPlan(
  hearthHash: string,
  contacts: EmergencyContact[],
  meetingPoints: string[],
): Promise<EmergencyPlan> {
  if (!isConnected()) return {
    id: `ep-${Date.now()}`,
    hearth_id: hearthHash,
    contacts,
    meeting_points: meetingPoints,
    last_reviewed: Date.now(),
  };
  return callZome<EmergencyPlan>({
    role_name: 'hearth',
    zome_name: 'hearth_emergency',
    fn_name: 'create_emergency_plan',
    payload: { hearth_hash: hearthHash, contacts, meeting_points: meetingPoints },
  });
}

export async function raiseHearthAlert(
  hearthHash: string,
  alertType: HearthAlertType,
  severity: HearthAlertSeverity,
  message: string,
): Promise<HearthAlert> {
  if (!isConnected()) return {
    id: `ha-${Date.now()}`,
    hearth_id: hearthHash,
    alert_type: alertType,
    severity,
    message,
    created_at: Date.now(),
    resolved: false,
  };
  return callZome<HearthAlert>({
    role_name: 'hearth',
    zome_name: 'hearth_emergency',
    fn_name: 'raise_alert',
    payload: { hearth_hash: hearthHash, alert_type: alertType, severity, message },
  });
}

export async function getActiveHearthAlerts(hearthHash: string): Promise<HearthAlert[]> {
  if (!isConnected()) return mockHearthAlerts(hearthHash);
  return callZome<HearthAlert[]>({
    role_name: 'hearth',
    zome_name: 'hearth_emergency',
    fn_name: 'get_active_alerts',
    payload: hearthHash,
  });
}

export async function checkIn(
  alertHash: string,
  status: SafetyStatus,
): Promise<SafetyCheckIn> {
  if (!isConnected()) return {
    id: `sci-${Date.now()}`,
    alert_id: alertHash,
    member_did: 'self.did',
    status,
    location_hint: null,
    checked_in_at: Date.now(),
  };
  return callZome<SafetyCheckIn>({
    role_name: 'hearth',
    zome_name: 'hearth_emergency',
    fn_name: 'check_in',
    payload: { alert_hash: alertHash, status },
  });
}

export async function getHearthInventory(hearthHash: string): Promise<SharedResource[]> {
  if (!isConnected()) return mockSharedResources(hearthHash);
  return callZome<SharedResource[]>({
    role_name: 'hearth',
    zome_name: 'hearth_resources',
    fn_name: 'get_hearth_inventory',
    payload: hearthHash,
  });
}

export async function registerResource(
  hearthHash: string,
  name: string,
  description: string,
  resourceType: HearthResourceType,
  condition: string,
  location: string,
): Promise<SharedResource> {
  if (!isConnected()) return {
    id: `sr-${Date.now()}`,
    hearth_id: hearthHash,
    name,
    description,
    resource_type: resourceType,
    current_holder: null,
    condition,
    location,
  };
  return callZome<SharedResource>({
    role_name: 'hearth',
    zome_name: 'hearth_resources',
    fn_name: 'register_resource',
    payload: { hearth_hash: hearthHash, name, description, resource_type: resourceType, condition, location },
  });
}

// ============================================================================
// Community Knowledge Client
// ============================================================================

export async function searchClaimsByTag(tag: string): Promise<KnowledgeClaim[]> {
  if (!isConnected()) return mockKnowledgeClaims().filter((c) => c.tags.includes(tag));
  return callZome<KnowledgeClaim[]>({
    role_name: 'knowledge',
    zome_name: 'claims',
    fn_name: 'get_claims_by_tag',
    payload: { tag },
  });
}

export async function submitClaim(
  content: string,
  tags: string[],
  confidence: number,
): Promise<KnowledgeClaim> {
  if (!isConnected()) return {
    id: `kc-${Date.now()}`,
    author_did: 'self.did',
    content,
    tags,
    confidence,
    e_score: 2,
    n_score: 1,
    m_score: 1,
    created_at: Date.now(),
  };
  return callZome<KnowledgeClaim>({
    role_name: 'knowledge',
    zome_name: 'claims',
    fn_name: 'submit_claim',
    payload: { content, tags, confidence },
  });
}

export async function getGraphStats(): Promise<GraphStats> {
  if (!isConnected()) return mockGraphStats();
  return callZome<GraphStats>({
    role_name: 'knowledge',
    zome_name: 'graph',
    fn_name: 'get_graph_stats',
    payload: null,
  });
}

export async function findKnowledgePath(source: string, target: string): Promise<unknown> {
  if (!isConnected()) return { source, target, path: [source, 'community-resilience', target], hops: 2 };
  return callZome({
    role_name: 'knowledge',
    zome_name: 'graph',
    fn_name: 'find_path',
    payload: { source, target },
  });
}

// ============================================================================
// Mock Data (Demo Mode)
// ============================================================================

function mockBalance(memberDid: string, daoDid: string): BalanceInfo {
  return {
    member_did: memberDid,
    dao_did: daoDid,
    balance: 24,
    total_earned: 87,
    total_spent: 63,
    exchange_count: 34,
  };
}

function mockExchange(
  receiverDid: string,
  hours: number,
  desc: string,
  cat: string,
): ExchangeRecord {
  return {
    id: `ex-${Date.now()}`,
    provider_did: 'self.did',
    receiver_did: receiverDid,
    hours,
    service_description: desc,
    service_category: cat,
    status: 'Proposed',
    timestamp: Date.now(),
  };
}

function mockListings(): ServiceListing[] {
  return [
    { id: 'ls-001', provider_did: 'thandi.did', title: 'Plumbing repair', description: 'Basic plumbing fixes, pipe replacement', category: 'Maintenance', hours_estimate: 2, dao_did: DEFAULT_DAO, created_at: Date.now() - 86400000, active: true },
    { id: 'ls-002', provider_did: 'sipho.did', title: 'Vegetable gardening lessons', description: 'Teaching companion planting and permaculture basics', category: 'Education', hours_estimate: 1.5, dao_did: DEFAULT_DAO, created_at: Date.now() - 172800000, active: true },
    { id: 'ls-003', provider_did: 'fatima.did', title: 'Child minding (weekdays)', description: 'After-school care for primary school children', category: 'Childcare', hours_estimate: 3, dao_did: DEFAULT_DAO, created_at: Date.now() - 259200000, active: true },
    { id: 'ls-004', provider_did: 'james.did', title: 'Vehicle maintenance', description: 'Oil changes, brake pads, basic diagnostics', category: 'Transport', hours_estimate: 2, dao_did: DEFAULT_DAO, created_at: Date.now() - 345600000, active: true },
    { id: 'ls-005', provider_did: 'noma.did', title: 'Bread baking (bulk)', description: 'Fresh bread, 10 loaves per batch', category: 'Food', hours_estimate: 4, dao_did: DEFAULT_DAO, created_at: Date.now() - 432000000, active: true },
  ];
}

function mockRequests(): ServiceRequest[] {
  return [
    { id: 'rq-001', requester_did: 'lerato.did', title: 'Roof leak repair', description: 'Corrugated iron roof leaking in two places', category: 'Maintenance', hours_budget: 3, urgency: 'High', dao_did: DEFAULT_DAO, created_at: Date.now() - 43200000, open: true },
    { id: 'rq-002', requester_did: 'mandla.did', title: 'Maths tutoring (Grade 10)', description: 'Need help with algebra and trigonometry', category: 'Education', hours_budget: 2, urgency: 'Medium', dao_did: DEFAULT_DAO, created_at: Date.now() - 86400000, open: true },
    { id: 'rq-003', requester_did: 'busi.did', title: 'Transport to clinic', description: 'Weekly transport to Leratong Hospital for treatment', category: 'Transport', hours_budget: 1, urgency: 'High', dao_did: DEFAULT_DAO, created_at: Date.now() - 129600000, open: true },
  ];
}

function mockPlots(): FoodPlot[] {
  return [
    { id: 'plot-001', owner_did: 'sipho.did', name: 'Sipho\'s backyard', location: 'Florida, Roodepoort', area_sqm: 45, plot_type: 'Raised beds', created_at: Date.now() - 2592000000 },
    { id: 'plot-002', owner_did: 'community.did', name: 'Ontdekkers Park community garden', location: 'Ontdekkers Rd', area_sqm: 200, plot_type: 'Open field', created_at: Date.now() - 5184000000 },
    { id: 'plot-003', owner_did: 'fatima.did', name: 'Fatima\'s tunnel', location: 'Horison, Roodepoort', area_sqm: 30, plot_type: 'Greenhouse tunnel', created_at: Date.now() - 1296000000 },
  ];
}

function mockPlot(name: string, location: string, areaSqm: number, plotType: string): FoodPlot {
  return { id: `plot-${Date.now()}`, owner_did: 'self.did', name, location, area_sqm: areaSqm, plot_type: plotType, created_at: Date.now() };
}

function mockHarvest(plotId: string, cropName: string, quantityKg: number, notes: string): HarvestRecord {
  return { id: `har-${Date.now()}`, plot_id: plotId, crop_name: cropName, quantity_kg: quantityKg, harvested_at: Date.now(), notes };
}

const NUTRIENT_LOOKUP: Record<ResourceType, NutrientProfile> = {
  KitchenWaste: { nitrogen_pct: 1.5, phosphorus_pct: 0.5, potassium_pct: 1.0 },
  GreenWaste: { nitrogen_pct: 2.0, phosphorus_pct: 0.3, potassium_pct: 1.5 },
  Biochar: { nitrogen_pct: 0.5, phosphorus_pct: 0.2, potassium_pct: 1.0 },
  Vermicompost: { nitrogen_pct: 2.5, phosphorus_pct: 1.5, potassium_pct: 1.5 },
  Digestate: { nitrogen_pct: 3.0, phosphorus_pct: 0.8, potassium_pct: 2.0 },
  Manure: { nitrogen_pct: 2.0, phosphorus_pct: 1.0, potassium_pct: 2.0 },
  Mulch: { nitrogen_pct: 0.5, phosphorus_pct: 0.1, potassium_pct: 0.3 },
  CoverCrop: { nitrogen_pct: 3.0, phosphorus_pct: 0.4, potassium_pct: 1.0 },
  Inoculant: { nitrogen_pct: 0.1, phosphorus_pct: 0.1, potassium_pct: 0.1 },
};

function mockResourceInputs(): ResourceInput[] {
  return [
    { id: 'ri-001', plot_id: 'plot-001', input_type: 'KitchenWaste', quantity_kg: 12.5, nutrient_estimate: NUTRIENT_LOOKUP.KitchenWaste, contributor_did: 'sipho.did', contributed_at: Date.now() - 604800000, notes: 'Weekly collection' },
    { id: 'ri-002', plot_id: 'plot-002', input_type: 'Vermicompost', quantity_kg: 30.0, nutrient_estimate: NUTRIENT_LOOKUP.Vermicompost, contributor_did: 'fatima.did', contributed_at: Date.now() - 432000000, notes: 'Worm farm harvest' },
    { id: 'ri-003', plot_id: null, input_type: 'Biochar', quantity_kg: 8.0, nutrient_estimate: NUTRIENT_LOOKUP.Biochar, contributor_did: 'community.did', contributed_at: Date.now() - 172800000, notes: 'Pyrolysis kiln run' },
    { id: 'ri-004', plot_id: 'plot-001', input_type: 'GreenWaste', quantity_kg: 20.0, nutrient_estimate: NUTRIENT_LOOKUP.GreenWaste, contributor_did: 'sipho.did', contributed_at: Date.now() - 86400000, notes: 'Garden clippings' },
  ];
}

function mockResourceInput(inputType: ResourceType, quantityKg: number, plotId: string | null, notes: string): ResourceInput {
  return {
    id: `ri-${Date.now()}`,
    plot_id: plotId,
    input_type: inputType,
    quantity_kg: quantityKg,
    nutrient_estimate: NUTRIENT_LOOKUP[inputType],
    contributor_did: 'self.did',
    contributed_at: Date.now(),
    notes,
  };
}

function mockNutrientSummary(): NutrientSummary {
  const inputs = mockResourceInputs();
  const byType = new Map<string, number>();
  let n = 0, p = 0, k = 0;
  for (const ri of inputs) {
    byType.set(ri.input_type, (byType.get(ri.input_type) ?? 0) + ri.quantity_kg);
    n += ri.quantity_kg * ri.nutrient_estimate.nitrogen_pct / 100;
    p += ri.quantity_kg * ri.nutrient_estimate.phosphorus_pct / 100;
    k += ri.quantity_kg * ri.nutrient_estimate.potassium_pct / 100;
  }
  return {
    total_kg_by_type: [...byType.entries()].sort((a, b) => b[1] - a[1]),
    total_nitrogen_kg: n,
    total_phosphorus_kg: p,
    total_potassium_kg: k,
    total_contributions: inputs.length,
  };
}

function mockOffers(): AidOffer[] {
  return [
    { id: 'ao-001', provider_did: 'thandi.did', title: 'Elder home visit', description: 'Weekly check-in on elderly neighbors, medication reminders', category: 'Care', hours_available: 2, recurring: true, created_at: Date.now() - 604800000 },
    { id: 'ao-002', provider_did: 'james.did', title: 'Lift share (Roodepoort-JHB CBD)', description: 'Daily commute, can take 3 passengers', category: 'Transport', hours_available: 1, recurring: true, created_at: Date.now() - 432000000 },
    { id: 'ao-003', provider_did: 'noma.did', title: 'Bulk cooking for families in need', description: 'Prepare nutritious meals for 10+ people', category: 'Food', hours_available: 4, recurring: false, created_at: Date.now() - 259200000 },
  ];
}

function mockOffer(title: string, description: string, category: string, hoursAvailable: number): AidOffer {
  return { id: `ao-${Date.now()}`, provider_did: 'self.did', title, description, category, hours_available: hoursAvailable, recurring: false, created_at: Date.now() };
}

function mockAidRequests(): AidRequest[] {
  return [
    { id: 'ar-001', requester_did: 'grace.did', title: 'Help with load-shedding prep', description: 'Need help installing a gas stove safely', category: 'Maintenance', urgency: 'medium', hours_needed: 2, created_at: Date.now() - 172800000, fulfilled: false },
    { id: 'ar-002', requester_did: 'peter.did', title: 'Food parcel delivery', description: 'Elderly neighbor needs groceries delivered weekly', category: 'Food', urgency: 'high', hours_needed: 1, created_at: Date.now() - 86400000, fulfilled: false },
  ];
}

function mockAidRequest(title: string, description: string, category: string, urgency: AidRequest['urgency'], hoursNeeded: number): AidRequest {
  return { id: `ar-${Date.now()}`, requester_did: 'self.did', title, description, category, urgency, hours_needed: hoursNeeded, created_at: Date.now(), fulfilled: false };
}

function mockChannels(): EmergencyChannel[] {
  return [
    { id: 'ch-001', name: 'Roodepoort General', description: 'Community-wide emergency channel', created_by: 'admin.did', created_at: Date.now() - 2592000000, member_count: 156 },
    { id: 'ch-002', name: 'Load Shedding Alerts', description: 'Real-time Eskom schedule updates', created_by: 'admin.did', created_at: Date.now() - 2592000000, member_count: 89 },
    { id: 'ch-003', name: 'Water Supply', description: 'Rand Water and Joburg Water alerts', created_by: 'admin.did', created_at: Date.now() - 1296000000, member_count: 67 },
  ];
}

function mockChannel(name: string, description: string): EmergencyChannel {
  return { id: `ch-${Date.now()}`, name, description, created_by: 'self.did', created_at: Date.now(), member_count: 1 };
}

function mockMessage(channelId: string, content: string, priority: EmergencyPriority): EmergencyMessage {
  return { id: `msg-${Date.now()}`, sender_did: 'self.did', channel_id: channelId, content, priority, sent_at: Date.now(), synced: true };
}

function mockMessages(channelId: string): EmergencyMessage[] {
  if (channelId === 'ch-002') {
    return [
      { id: 'msg-001', sender_did: 'eskom.did', channel_id: channelId, content: 'Stage 4 load shedding from 16:00-20:30. Roodepoort affected areas: Florida, Horison, Constantia Kloof.', priority: 'Immediate', sent_at: Date.now() - 3600000, synced: true },
      { id: 'msg-002', sender_did: 'admin.did', channel_id: channelId, content: 'Community kitchen at Ontdekkers Park open during outage. Hot meals available.', priority: 'Priority', sent_at: Date.now() - 1800000, synced: true },
    ];
  }
  return [
    { id: 'msg-010', sender_did: 'admin.did', channel_id: channelId, content: 'Water pressure low in Florida area. Fill containers as precaution.', priority: 'Priority', sent_at: Date.now() - 7200000, synced: true },
    { id: 'msg-011', sender_did: 'thandi.did', channel_id: channelId, content: 'Confirmed: pressure restored in Horison. Florida still affected.', priority: 'Routine', sent_at: Date.now() - 3600000, synced: true },
  ];
}

// ============================================================================
// Mock Data — Price Oracle (Demo Mode)
// ============================================================================

/** Default prices for demo mode (TEND = hours of labor) */
const DEMO_PRICES: Record<string, number> = {
  bread_750g: 0.15,
  eggs_6: 0.25,
  diesel_1l: 0.50,
  milk_1l: 0.10,
  'mealie_meal_2.5kg': 0.20,
  solar_kwh: 0.40,
  taxi_trip: 0.30,
  chicken_whole: 0.75,
  cooking_oil_750ml: 0.20,
  'sugar_2.5kg': 0.15,
};

function mockPriceReport(input: PriceReportInput): unknown {
  return { item: input.item, price_tend: input.price_tend, evidence: input.evidence, reporter_did: 'self.did', reported_at: Date.now() };
}

function mockConsensus(item: string): ConsensusResult {
  return {
    item,
    median_price: DEMO_PRICES[item] ?? 0.25,
    reporter_count: 5,
    std_dev: 0.03,
    window_start: Date.now() - 7 * 24 * 60 * 60 * 1000,
    signal_integrity: 0.91,
  };
}

function mockTopReporters(): ReporterAccuracyInfo[] {
  return [
    { reporter_did: 'sipho.did', accuracy_score: 0.97, report_count: 42 },
    { reporter_did: 'thandi.did', accuracy_score: 0.94, report_count: 38 },
    { reporter_did: 'fatima.did', accuracy_score: 0.91, report_count: 25 },
    { reporter_did: 'james.did', accuracy_score: 0.88, report_count: 19 },
    { reporter_did: 'noma.did', accuracy_score: 0.85, report_count: 11 },
  ];
}

function mockBasketIndex(basketName: string): BasketIndexResult {
  const items: ItemPriceResult[] = Object.entries(DEMO_PRICES).map(([item, price]) => ({
    item,
    price,
    weight: 0.1,
    weighted_price: price * 0.1,
  }));
  const index = items.reduce((sum, i) => sum + i.weighted_price, 0);
  return { basket_name: basketName, index, item_prices: items, computed_at: Date.now() };
}

function mockVolatility(basketName: string): VolatilityResult {
  return {
    basket_name: basketName,
    current_index: 0.30,
    previous_index: 0.29,
    weekly_change: 0.034,
    recommended_tier: 'Normal',
    escalated: false,
  };
}

// ============================================================================
// Mock Data — Care Circles (Demo Mode)
// ============================================================================

function mockCareCircles(): CareCircle[] {
  return [
    { id: 'cc-001', name: 'Sector 7 Neighbourhood Watch', description: 'Community safety and mutual support for Sector 7 residents', circle_type: 'Neighbourhood', location: 'Sector 7, Roodepoort', max_members: 50, member_count: 34, created_at: Date.now() - 7776000000 },
    { id: 'cc-002', name: 'Florida Lake Gardeners', description: 'Shared gardening knowledge and seed exchange around Florida Lake', circle_type: 'Neighbourhood', location: 'Florida, Roodepoort', max_members: 30, member_count: 18, created_at: Date.now() - 5184000000 },
    { id: 'cc-003', name: 'Roodepoort First Responders', description: 'Workplace first-aid trained volunteers for emergency response', circle_type: 'Workplace', location: 'Roodepoort CBD', max_members: 20, member_count: 12, created_at: Date.now() - 2592000000 },
    { id: 'cc-004', name: "St. Mark's Care Network", description: 'Faith-based elder care and food distribution', circle_type: 'Faith', location: 'Horison, Roodepoort', max_members: 40, member_count: 27, created_at: Date.now() - 10368000000 },
  ];
}

function mockCircleMembers(circleId: string): CircleMembership[] {
  const now = Date.now();
  const members: Record<string, CircleMembership[]> = {
    'cc-001': [
      { id: 'cm-001', circle_id: 'cc-001', member_did: 'thandi.did', role: 'Organizer', joined_at: now - 7776000000, active: true },
      { id: 'cm-002', circle_id: 'cc-001', member_did: 'sipho.did', role: 'Member', joined_at: now - 6048000000, active: true },
      { id: 'cm-003', circle_id: 'cc-001', member_did: 'nomsa.did', role: 'Member', joined_at: now - 4320000000, active: true },
    ],
    'cc-002': [
      { id: 'cm-004', circle_id: 'cc-002', member_did: 'fatima.did', role: 'Organizer', joined_at: now - 5184000000, active: true },
      { id: 'cm-005', circle_id: 'cc-002', member_did: 'james.did', role: 'Member', joined_at: now - 3456000000, active: true },
    ],
    'cc-003': [
      { id: 'cm-006', circle_id: 'cc-003', member_did: 'sipho.did', role: 'Organizer', joined_at: now - 2592000000, active: true },
      { id: 'cm-007', circle_id: 'cc-003', member_did: 'lerato.did', role: 'Member', joined_at: now - 1728000000, active: true },
    ],
    'cc-004': [
      { id: 'cm-008', circle_id: 'cc-004', member_did: 'nomsa.did', role: 'Organizer', joined_at: now - 10368000000, active: true },
      { id: 'cm-009', circle_id: 'cc-004', member_did: 'grace.did', role: 'Member', joined_at: now - 8640000000, active: true },
      { id: 'cm-010', circle_id: 'cc-004', member_did: 'busi.did', role: 'Observer', joined_at: now - 6912000000, active: true },
    ],
  };
  return members[circleId] ?? [];
}

// ============================================================================
// Mock Data — Emergency Shelter (Demo Mode)
// ============================================================================

function mockHousingUnits(): HousingUnit[] {
  return [
    { id: 'hu-001', building_id: 'bld-001', unit_number: 'A1', unit_type: 'Studio', bedrooms: 0, bathrooms: 1, square_meters: 28, floor: 0, status: 'Available', accessibility_features: ['WheelchairAccessible', 'GrabBars'] },
    { id: 'hu-002', building_id: 'bld-001', unit_number: 'B3', unit_type: 'OneBedroom', bedrooms: 1, bathrooms: 1, square_meters: 42, floor: 1, status: 'Available', accessibility_features: [] },
    { id: 'hu-003', building_id: 'bld-002', unit_number: 'C2', unit_type: 'TwoBedroom', bedrooms: 2, bathrooms: 1, square_meters: 58, floor: 0, status: 'Available', accessibility_features: ['WheelchairAccessible', 'WideDoorways'] },
    { id: 'hu-004', building_id: 'bld-002', unit_number: 'D1', unit_type: 'ThreeBedroom', bedrooms: 3, bathrooms: 2, square_meters: 76, floor: 1, status: 'Available', accessibility_features: ['GrabBars'] },
  ];
}

// ============================================================================
// Mock Data — Supply Tracking (Demo Mode)
// ============================================================================

function mockInventoryItems(): InventoryItem[] {
  return [
    { id: 'inv-001', name: 'Rice 5kg bags', description: 'Long-grain white rice, sealed bags', category: 'Food', sku: 'FD-RICE-5KG', unit: 'bags', reorder_point: 20, reorder_quantity: 50 },
    { id: 'inv-002', name: 'Water purification tablets', description: 'Chlorine-based water treatment, 50 per box', category: 'Water', sku: 'WT-PURIFY-50', unit: 'boxes', reorder_point: 15, reorder_quantity: 40 },
    { id: 'inv-003', name: 'First aid kits', description: 'Standard community first aid kit with bandages, antiseptic, gloves', category: 'Medical', sku: 'MD-FAID-STD', unit: 'kits', reorder_point: 10, reorder_quantity: 25 },
    { id: 'inv-004', name: 'Diesel 20L', description: 'Diesel fuel for generator backup', category: 'Fuel', sku: 'FL-DIESEL-20L', unit: 'jerricans', reorder_point: 8, reorder_quantity: 20 },
    { id: 'inv-005', name: 'Soap bars', description: 'Antibacterial soap, individually wrapped', category: 'Hygiene', sku: 'HY-SOAP-BAR', unit: 'bars', reorder_point: 50, reorder_quantity: 200 },
    { id: 'inv-006', name: 'Emergency blankets', description: 'Mylar thermal blankets, single-use', category: 'Shelter', sku: 'SH-BLANKET-EM', unit: 'blankets', reorder_point: 30, reorder_quantity: 100 },
  ];
}

function mockLowStockItems(): LowStockItem[] {
  const items = mockInventoryItems();
  return [
    { item: items[3], total_stock: 5 },  // Diesel — below reorder_point of 8
    { item: items[2], total_stock: 7 },   // First aid kits — below reorder_point of 10
  ];
}

function mockStockLevels(itemId: string): StockLevel[] {
  return [
    { id: 'sl-001', item_id: itemId, quantity: 35, location: 'Roodepoort Community Store', recorded_by: 'thandi.did', recorded_at: Date.now() - 172800000, notes: 'Monthly count' },
    { id: 'sl-002', item_id: itemId, quantity: 30, location: 'Roodepoort Community Store', recorded_by: 'sipho.did', recorded_at: Date.now() - 604800000, notes: 'After distribution' },
  ];
}

// ============================================================================
// Mock Data — Water Safety (Demo Mode)
// ============================================================================

function mockWaterSystems(): WaterSystem[] {
  return [
    { id: 'ws-001', name: 'Sector 7 Community Roof Harvest', system_type: 'RoofRainwater', capacity_liters: 10000, catchment_area_sqm: 120, efficiency_percent: 85, owner_did: 'community.did', location_lat: -26.1496, location_lon: 27.8625, installed_at: Date.now() - 15552000000 },
    { id: 'ws-002', name: 'Florida Lake Ground Catchment', system_type: 'GroundCatchment', capacity_liters: 25000, catchment_area_sqm: 500, efficiency_percent: 60, owner_did: 'community.did', location_lat: -26.1700, location_lon: 27.9100, installed_at: Date.now() - 31104000000 },
    { id: 'ws-003', name: 'Hilltop Fog Collector', system_type: 'FogCollection', capacity_liters: 2000, catchment_area_sqm: 40, efficiency_percent: 30, owner_did: 'sipho.did', location_lat: -26.1350, location_lon: 27.8450, installed_at: Date.now() - 7776000000 },
  ];
}

function mockWaterAlerts(): ContaminationAlert[] {
  return [
    { id: 'wa-001', source_id: 'ws-002', contaminant: 'E. coli', measured_value: 12.0, threshold_value: 1.0, severity: 'Warning', reported_at: Date.now() - 86400000, reported_by: 'thandi.did' },
  ];
}

// ============================================================================
// Mock Data — Household / Hearth (Demo Mode)
// ============================================================================

function mockHearths(): Hearth[] {
  return [
    { id: 'hth-001', name: 'Stoltz Family', member_count: 4, created_at: Date.now() - 31104000000 },
    { id: 'hth-002', name: 'Sector 7 Neighbours', member_count: 8, created_at: Date.now() - 15552000000 },
  ];
}

function mockEmergencyPlan(hearthId: string): EmergencyPlan {
  return {
    id: 'ep-001',
    hearth_id: hearthId,
    contacts: [
      { name: 'Thandi Mokoena', phone: '+27 72 345 6789', relationship: 'Neighbour' },
      { name: 'Sipho Nkosi', phone: '+27 83 456 7890', relationship: 'Community Leader' },
      { name: 'Fatima Patel', phone: '+27 61 567 8901', relationship: 'First Responder', email: 'fatima@roodepoort-responders.org' },
    ],
    meeting_points: [
      'Ontdekkers Park main entrance',
      'Florida Lake parking area',
      'St. Mark\'s Church hall, Horison',
    ],
    last_reviewed: Date.now() - 2592000000,
  };
}

function mockHearthAlerts(hearthId: string): HearthAlert[] {
  return [
    { id: 'ha-001', hearth_id: hearthId, alert_type: 'Flood', severity: 'Medium', message: 'Heavy rainfall expected overnight. Secure low-lying items.', created_at: Date.now() - 43200000, resolved: false },
  ];
}

function mockSharedResources(hearthId: string): SharedResource[] {
  return [
    { id: 'sr-001', hearth_id: hearthId, name: 'Generator (petrol, 2kVA)', description: 'Portable generator for load-shedding backup', resource_type: 'Tools', current_holder: 'sipho.did', condition: 'Good', location: 'Sipho\'s garage' },
    { id: 'sr-002', hearth_id: hearthId, name: 'Community first-aid kit', description: 'Fully stocked trauma kit, restocked monthly', resource_type: 'Medical', current_holder: null, condition: 'Excellent', location: 'Sector 7 community hall' },
    { id: 'sr-003', hearth_id: hearthId, name: 'Emergency water drums (4x 25L)', description: 'Clean water storage drums, filled and rotated weekly', resource_type: 'Water', current_holder: null, condition: 'Good', location: 'Thandi\'s backyard' },
    { id: 'sr-004', hearth_id: hearthId, name: 'Two-way radios (set of 4)', description: 'UHF radios for comms during outages, charged weekly', resource_type: 'Communications', current_holder: 'thandi.did', condition: 'Good', location: 'Thandi\'s house' },
  ];
}

// ============================================================================
// Mock Data — Community Knowledge (Demo Mode)
// ============================================================================

function mockKnowledgeClaims(): KnowledgeClaim[] {
  return [
    { id: 'kc-001', author_did: 'sipho.did', content: 'Companion planting tomatoes with basil reduces aphid damage by approximately 40% in Roodepoort clay soils', tags: ['food-production', 'gardening', 'pest-control'], confidence: 0.85, e_score: 3, n_score: 1, m_score: 2, created_at: Date.now() - 2592000000 },
    { id: 'kc-002', author_did: 'thandi.did', content: 'Boiling water for 3 minutes at Roodepoort altitude is sufficient to eliminate E. coli and most waterborne pathogens', tags: ['water-safety', 'health', 'purification'], confidence: 0.95, e_score: 4, n_score: 1, m_score: 3, created_at: Date.now() - 5184000000 },
    { id: 'kc-003', author_did: 'fatima.did', content: 'Direct pressure with a clean cloth for 10 minutes stops most wound bleeding; tourniquets only for limb-threatening haemorrhage', tags: ['first-aid', 'emergency', 'health'], confidence: 0.92, e_score: 4, n_score: 2, m_score: 3, created_at: Date.now() - 7776000000 },
    { id: 'kc-004', author_did: 'james.did', content: 'Roodepoort loam soil pH averages 5.8-6.2; adding wood ash raises pH by ~0.5 per 1kg/m2 application', tags: ['food-production', 'soil', 'gardening'], confidence: 0.78, e_score: 2, n_score: 0, m_score: 2, created_at: Date.now() - 1296000000 },
    { id: 'kc-005', author_did: 'nomsa.did', content: 'Load-shedding schedules follow 2-hour blocks; keeping a charged power bank and LPG stove covers essential needs during Stage 4', tags: ['energy', 'load-shedding', 'preparedness'], confidence: 0.88, e_score: 3, n_score: 1, m_score: 2, created_at: Date.now() - 864000000 },
    { id: 'kc-006', author_did: 'grace.did', content: 'Florida Lake water is not potable without filtration; heavy metal levels exceed SANS 241 limits after summer rains', tags: ['water-safety', 'contamination', 'florida-lake'], confidence: 0.90, e_score: 3, n_score: 1, m_score: 3, created_at: Date.now() - 3456000000 },
  ];
}

function mockGraphStats(): GraphStats {
  return {
    total_claims: 47,
    total_relationships: 83,
    total_ontologies: 5,
    total_concepts: 31,
  };
}
