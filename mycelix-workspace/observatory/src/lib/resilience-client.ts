/**
 * Resilience Client — Typed wrappers for TEND, food, mutual-aid, and emergency zome calls.
 *
 * In demo mode (no conductor), returns mock data for UI development.
 * In live mode, proxies to callZome() with correct role/zome/fn routing.
 */

import { callZome, isConnected } from './conductor';

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

const DEFAULT_DAO = 'roodepoort-resilience';

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
  if (!isConnected()) return mockExchange(receiverDid, hours, serviceDescription, serviceCategory);
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
