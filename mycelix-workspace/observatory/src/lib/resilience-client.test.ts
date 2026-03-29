// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for resilience-client module — demo mode fallback and return type correctness.
 *
 * The conductor is mocked as disconnected so every exported function exercises
 * its demo/mock data path.  We verify the shape of each returned object matches
 * the declared TypeScript interface.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the conductor module — always disconnected (demo mode)
vi.mock('./conductor', () => ({
  isConnected: vi.fn(() => false),
  callZome: vi.fn(() => Promise.reject(new Error('not connected'))),
}));

// Mock the offline-queue module — enqueue returns null (no IndexedDB in jsdom)
vi.mock('./offline-queue', () => ({
  enqueue: vi.fn(() => Promise.resolve(null)),
}));

// ---------------------------------------------------------------------------
// Import the module under test AFTER mocks are registered
// ---------------------------------------------------------------------------
import {
  getBalance,
  recordExchange,
  getDaoListings,
  getDaoRequests,
  getOracleState,
  getAllPlots,
  registerPlot,
  recordHarvest,
  logResourceInput,
  getCommunityInputs,
  getNutrientSummary,
  getServiceOffers,
  createServiceOffer,
  getServiceRequests,
  createServiceRequest,
  getChannels,
  createChannel,
  sendMessage,
  getMessages,
  reportPrice,
  getConsensusPrice,
  defineBasket,
  getBasketIndex,
  computeVolatility,
  getItemReports,
  getAllBaskets,
  getReporterAccuracy,
  getTopReporters,
  getAllCareCircles,
  getCirclesByType,
  getCircleMembers,
  joinCircle,
  getAvailableUnits,
  getBuildingUnits,
  getAllInventoryItems,
  getItemsByCategory,
  getLowStockItems,
  getStockLevels,
  addInventoryItem,
  updateStockLevel,
  getAllWaterSystems,
  getMyWaterSystems,
  registerWaterSystem,
  updateTankLevel,
  submitWaterReading,
  getActiveWaterAlerts,
  getMyHearths,
  getEmergencyPlan,
  createEmergencyPlan,
  raiseHearthAlert,
  getActiveHearthAlerts,
  checkIn,
  getHearthInventory,
  registerResource,
  searchClaimsByTag,
  submitClaim,
  getGraphStats,
  findKnowledgePath,
  RESOURCE_TYPES,
  RESOURCE_TYPE_LABELS,
} from './resilience-client';

import type {
  BalanceInfo,
  ExchangeRecord,
  ServiceListing,
  ServiceRequest,
  OracleState,
  FoodPlot,
  HarvestRecord,
  ResourceInput,
  NutrientSummary,
  AidOffer,
  AidRequest,
  EmergencyChannel,
  EmergencyMessage,
  ConsensusResult,
  BasketIndexResult,
  VolatilityResult,
  ReporterAccuracyInfo,
  CareCircle,
  CircleMembership,
  HousingUnit,
  InventoryItem,
  LowStockItem,
  StockLevel,
  WaterSystem,
  StorageTank,
  ContaminationAlert,
  Hearth,
  EmergencyPlan,
  HearthAlert,
  SafetyCheckIn,
  SharedResource,
  KnowledgeClaim,
  GraphStats,
} from './resilience-client';

// ============================================================================
// TEND (Time Exchange) Client
// ============================================================================

describe('resilience-client: TEND domain (demo mode)', () => {
  it('getBalance returns a valid BalanceInfo', async () => {
    const b = await getBalance('alice.did', 'dao-001');
    expect(b.member_did).toBe('alice.did');
    expect(b.dao_did).toBe('dao-001');
    expect(typeof b.balance).toBe('number');
    expect(typeof b.total_earned).toBe('number');
    expect(typeof b.total_spent).toBe('number');
    expect(typeof b.exchange_count).toBe('number');
  });

  it('recordExchange returns a valid ExchangeRecord', async () => {
    const ex = await recordExchange('bob.did', 2, 'Plumbing', 'Maintenance');
    expect(ex.receiver_did).toBe('bob.did');
    expect(ex.hours).toBe(2);
    expect(ex.service_description).toBe('Plumbing');
    expect(ex.service_category).toBe('Maintenance');
    expect(ex.status).toBe('Proposed');
    expect(typeof ex.id).toBe('string');
    expect(typeof ex.timestamp).toBe('number');
  });

  it('getDaoListings returns an array of ServiceListings', async () => {
    const listings = await getDaoListings();
    expect(Array.isArray(listings)).toBe(true);
    expect(listings.length).toBeGreaterThan(0);
    for (const l of listings) {
      expect(typeof l.id).toBe('string');
      expect(typeof l.title).toBe('string');
      expect(typeof l.hours_estimate).toBe('number');
      expect(typeof l.active).toBe('boolean');
    }
  });

  it('getDaoRequests returns an array of ServiceRequests', async () => {
    const requests = await getDaoRequests();
    expect(Array.isArray(requests)).toBe(true);
    expect(requests.length).toBeGreaterThan(0);
    for (const r of requests) {
      expect(typeof r.id).toBe('string');
      expect(typeof r.title).toBe('string');
      expect(typeof r.hours_budget).toBe('number');
      expect(typeof r.open).toBe('boolean');
    }
  });

  it('getOracleState returns a valid OracleState', async () => {
    const state = await getOracleState();
    expect(typeof state.vitality).toBe('number');
    expect(['Normal', 'Elevated', 'High', 'Emergency']).toContain(state.tier);
    expect(typeof state.updated_at).toBe('number');
  });
});

// ============================================================================
// Food Production Client
// ============================================================================

describe('resilience-client: Food Production domain (demo mode)', () => {
  it('getAllPlots returns an array of FoodPlots', async () => {
    const plots = await getAllPlots();
    expect(Array.isArray(plots)).toBe(true);
    expect(plots.length).toBeGreaterThan(0);
    for (const p of plots) {
      expect(typeof p.id).toBe('string');
      expect(typeof p.name).toBe('string');
      expect(typeof p.area_sqm).toBe('number');
      expect(typeof p.plot_type).toBe('string');
    }
  });

  it('registerPlot returns a FoodPlot with given name', async () => {
    const plot = await registerPlot('Test Plot', 'Test Location', 50, 'Raised beds');
    expect(plot.name).toBe('Test Plot');
    expect(plot.location).toBe('Test Location');
    expect(plot.area_sqm).toBe(50);
    expect(plot.plot_type).toBe('Raised beds');
    expect(typeof plot.id).toBe('string');
  });

  it('recordHarvest returns a valid HarvestRecord', async () => {
    const h = await recordHarvest('plot-001', 'Tomatoes', 12.5, 'Good quality');
    expect(h.plot_id).toBe('plot-001');
    expect(h.crop_name).toBe('Tomatoes');
    expect(h.quantity_kg).toBe(12.5);
    expect(typeof h.id).toBe('string');
    expect(typeof h.harvested_at).toBe('number');
  });
});

// ============================================================================
// Resource Input Client
// ============================================================================

describe('resilience-client: Resource Input domain (demo mode)', () => {
  it('logResourceInput returns a valid ResourceInput', async () => {
    const ri = await logResourceInput('KitchenWaste', 5.0, 'plot-001', 'Test');
    expect(ri.input_type).toBe('KitchenWaste');
    expect(ri.quantity_kg).toBe(5.0);
    expect(ri.plot_id).toBe('plot-001');
    expect(typeof ri.nutrient_estimate.nitrogen_pct).toBe('number');
    expect(typeof ri.nutrient_estimate.phosphorus_pct).toBe('number');
    expect(typeof ri.nutrient_estimate.potassium_pct).toBe('number');
  });

  it('getCommunityInputs returns an array of ResourceInputs', async () => {
    const inputs = await getCommunityInputs();
    expect(Array.isArray(inputs)).toBe(true);
    expect(inputs.length).toBeGreaterThan(0);
    for (const ri of inputs) {
      expect(typeof ri.id).toBe('string');
      expect(typeof ri.input_type).toBe('string');
      expect(typeof ri.quantity_kg).toBe('number');
    }
  });

  it('getNutrientSummary returns a valid NutrientSummary', async () => {
    const ns = await getNutrientSummary();
    expect(Array.isArray(ns.total_kg_by_type)).toBe(true);
    expect(typeof ns.total_nitrogen_kg).toBe('number');
    expect(typeof ns.total_phosphorus_kg).toBe('number');
    expect(typeof ns.total_potassium_kg).toBe('number');
    expect(typeof ns.total_contributions).toBe('number');
    expect(ns.total_contributions).toBeGreaterThan(0);
  });

  it('RESOURCE_TYPES has 9 entries', () => {
    expect(RESOURCE_TYPES).toHaveLength(9);
    expect(RESOURCE_TYPES).toContain('KitchenWaste');
    expect(RESOURCE_TYPES).toContain('Biochar');
  });

  it('RESOURCE_TYPE_LABELS maps all types to labels', () => {
    for (const rt of RESOURCE_TYPES) {
      expect(typeof RESOURCE_TYPE_LABELS[rt]).toBe('string');
    }
  });
});

// ============================================================================
// Mutual Aid Client
// ============================================================================

describe('resilience-client: Mutual Aid domain (demo mode)', () => {
  it('getServiceOffers returns an array of AidOffers', async () => {
    const offers = await getServiceOffers();
    expect(Array.isArray(offers)).toBe(true);
    expect(offers.length).toBeGreaterThan(0);
    for (const o of offers) {
      expect(typeof o.id).toBe('string');
      expect(typeof o.title).toBe('string');
      expect(typeof o.hours_available).toBe('number');
      expect(typeof o.recurring).toBe('boolean');
    }
  });

  it('createServiceOffer returns a valid AidOffer', async () => {
    const offer = await createServiceOffer('Tutoring', 'Math help', 'Education', 2);
    expect(offer.title).toBe('Tutoring');
    expect(offer.description).toBe('Math help');
    expect(offer.category).toBe('Education');
    expect(offer.hours_available).toBe(2);
    expect(typeof offer.id).toBe('string');
  });

  it('getServiceRequests returns an array of AidRequests', async () => {
    const requests = await getServiceRequests();
    expect(Array.isArray(requests)).toBe(true);
    expect(requests.length).toBeGreaterThan(0);
    for (const r of requests) {
      expect(typeof r.id).toBe('string');
      expect(['low', 'medium', 'high', 'critical']).toContain(r.urgency);
      expect(typeof r.fulfilled).toBe('boolean');
    }
  });

  it('createServiceRequest returns a valid AidRequest', async () => {
    const req = await createServiceRequest('Help needed', 'Details', 'Care', 'high', 3);
    expect(req.title).toBe('Help needed');
    expect(req.urgency).toBe('high');
    expect(req.hours_needed).toBe(3);
    expect(req.fulfilled).toBe(false);
  });
});

// ============================================================================
// Emergency Comms Client
// ============================================================================

describe('resilience-client: Emergency Comms domain (demo mode)', () => {
  it('getChannels returns an array of EmergencyChannels', async () => {
    const channels = await getChannels();
    expect(Array.isArray(channels)).toBe(true);
    expect(channels.length).toBeGreaterThan(0);
    for (const ch of channels) {
      expect(typeof ch.id).toBe('string');
      expect(typeof ch.name).toBe('string');
      expect(typeof ch.member_count).toBe('number');
    }
  });

  it('createChannel returns a valid EmergencyChannel', async () => {
    const ch = await createChannel('Test Channel', 'A test channel');
    expect(ch.name).toBe('Test Channel');
    expect(ch.description).toBe('A test channel');
    expect(typeof ch.id).toBe('string');
    expect(typeof ch.member_count).toBe('number');
  });

  it('sendMessage returns a valid EmergencyMessage', async () => {
    const msg = await sendMessage('ch-001', 'Alert!', 'Flash');
    expect(msg.channel_id).toBe('ch-001');
    expect(msg.content).toBe('Alert!');
    expect(msg.priority).toBe('Flash');
    expect(typeof msg.id).toBe('string');
    expect(typeof msg.sent_at).toBe('number');
  });

  it('getMessages returns an array of EmergencyMessages', async () => {
    const msgs = await getMessages('ch-002');
    expect(Array.isArray(msgs)).toBe(true);
    expect(msgs.length).toBeGreaterThan(0);
    for (const m of msgs) {
      expect(typeof m.id).toBe('string');
      expect(typeof m.content).toBe('string');
      expect(['Flash', 'Immediate', 'Priority', 'Routine']).toContain(m.priority);
    }
  });
});

// ============================================================================
// Price Oracle Client
// ============================================================================

describe('resilience-client: Price Oracle domain (demo mode)', () => {
  it('reportPrice returns a mock report', async () => {
    const report = await reportPrice({ item: 'bread_750g', price_tend: 0.15, evidence: 'shoprite' });
    expect(report).toBeDefined();
    expect((report as Record<string, unknown>).item).toBe('bread_750g');
  });

  it('getConsensusPrice returns a valid ConsensusResult', async () => {
    const c = await getConsensusPrice('bread_750g');
    expect(c.item).toBe('bread_750g');
    expect(typeof c.median_price).toBe('number');
    expect(typeof c.reporter_count).toBe('number');
    expect(typeof c.std_dev).toBe('number');
    expect(typeof c.signal_integrity).toBe('number');
  });

  it('defineBasket returns the basket definition', async () => {
    const result = await defineBasket('test-basket', [{ item: 'bread_750g', weight: 0.5 }]);
    expect(result).toBeDefined();
    expect((result as Record<string, unknown>).name).toBe('test-basket');
  });

  it('getBasketIndex returns a valid BasketIndexResult', async () => {
    const bi = await getBasketIndex('community-basket');
    expect(bi.basket_name).toBe('community-basket');
    expect(typeof bi.index).toBe('number');
    expect(Array.isArray(bi.item_prices)).toBe(true);
    expect(typeof bi.computed_at).toBe('number');
  });

  it('computeVolatility returns a valid VolatilityResult', async () => {
    const v = await computeVolatility('community-basket');
    expect(v.basket_name).toBe('community-basket');
    expect(typeof v.current_index).toBe('number');
    expect(typeof v.previous_index).toBe('number');
    expect(typeof v.weekly_change).toBe('number');
    expect(typeof v.escalated).toBe('boolean');
  });

  it('getItemReports returns an empty array in demo mode', async () => {
    const reports = await getItemReports('bread_750g');
    expect(Array.isArray(reports)).toBe(true);
    expect(reports).toHaveLength(0);
  });

  it('getAllBaskets returns an empty array in demo mode', async () => {
    const baskets = await getAllBaskets();
    expect(Array.isArray(baskets)).toBe(true);
    expect(baskets).toHaveLength(0);
  });

  it('getReporterAccuracy returns a valid ReporterAccuracyInfo', async () => {
    const info = await getReporterAccuracy('sipho.did');
    expect(info.reporter_did).toBe('sipho.did');
    expect(typeof info.accuracy_score).toBe('number');
    expect(typeof info.report_count).toBe('number');
  });

  it('getTopReporters returns an array of ReporterAccuracyInfo', async () => {
    const reporters = await getTopReporters('bread_750g');
    expect(Array.isArray(reporters)).toBe(true);
    expect(reporters.length).toBeGreaterThan(0);
    for (const r of reporters) {
      expect(typeof r.reporter_did).toBe('string');
      expect(typeof r.accuracy_score).toBe('number');
      expect(typeof r.report_count).toBe('number');
    }
  });
});

// ============================================================================
// Care Circles Client
// ============================================================================

describe('resilience-client: Care Circles domain (demo mode)', () => {
  it('getAllCareCircles returns an array of CareCircles', async () => {
    const circles = await getAllCareCircles();
    expect(Array.isArray(circles)).toBe(true);
    expect(circles.length).toBeGreaterThan(0);
    for (const c of circles) {
      expect(typeof c.id).toBe('string');
      expect(typeof c.name).toBe('string');
      expect(typeof c.max_members).toBe('number');
    }
  });

  it('getCirclesByType filters by type', async () => {
    const circles = await getCirclesByType('Neighbourhood');
    expect(Array.isArray(circles)).toBe(true);
    for (const c of circles) {
      expect(c.circle_type).toBe('Neighbourhood');
    }
  });

  it('getCircleMembers returns members for a known circle', async () => {
    const members = await getCircleMembers('cc-001');
    expect(Array.isArray(members)).toBe(true);
    expect(members.length).toBeGreaterThan(0);
    for (const m of members) {
      expect(typeof m.id).toBe('string');
      expect(['Organizer', 'Member', 'Observer']).toContain(m.role);
      expect(typeof m.active).toBe('boolean');
    }
  });

  it('joinCircle returns a valid CircleMembership', async () => {
    const membership = await joinCircle('cc-002', 'Member');
    expect(membership.circle_id).toBe('cc-002');
    expect(membership.role).toBe('Member');
    expect(membership.active).toBe(true);
    expect(typeof membership.id).toBe('string');
  });
});

// ============================================================================
// Emergency Shelter Client
// ============================================================================

describe('resilience-client: Emergency Shelter domain (demo mode)', () => {
  it('getAvailableUnits returns an array of HousingUnits', async () => {
    const units = await getAvailableUnits();
    expect(Array.isArray(units)).toBe(true);
    expect(units.length).toBeGreaterThan(0);
    for (const u of units) {
      expect(typeof u.id).toBe('string');
      expect(typeof u.unit_number).toBe('string');
      expect(typeof u.bedrooms).toBe('number');
      expect(Array.isArray(u.accessibility_features)).toBe(true);
    }
  });

  it('getBuildingUnits filters by building', async () => {
    const units = await getBuildingUnits('bld-001');
    expect(Array.isArray(units)).toBe(true);
    for (const u of units) {
      expect(u.building_id).toBe('bld-001');
    }
  });
});

// ============================================================================
// Supply Tracking Client
// ============================================================================

describe('resilience-client: Supply Tracking domain (demo mode)', () => {
  it('getAllInventoryItems returns an array of InventoryItems', async () => {
    const items = await getAllInventoryItems();
    expect(Array.isArray(items)).toBe(true);
    expect(items.length).toBeGreaterThan(0);
    for (const item of items) {
      expect(typeof item.id).toBe('string');
      expect(typeof item.name).toBe('string');
      expect(typeof item.sku).toBe('string');
      expect(typeof item.reorder_point).toBe('number');
    }
  });

  it('getItemsByCategory filters by category', async () => {
    const items = await getItemsByCategory('Food');
    expect(Array.isArray(items)).toBe(true);
    for (const item of items) {
      expect(item.category).toBe('Food');
    }
  });

  it('getLowStockItems returns items below reorder point', async () => {
    const low = await getLowStockItems();
    expect(Array.isArray(low)).toBe(true);
    expect(low.length).toBeGreaterThan(0);
    for (const ls of low) {
      expect(ls.item).toBeDefined();
      expect(typeof ls.total_stock).toBe('number');
      expect(ls.total_stock).toBeLessThan(ls.item.reorder_point);
    }
  });

  it('getStockLevels returns an array of StockLevels', async () => {
    const levels = await getStockLevels('inv-001');
    expect(Array.isArray(levels)).toBe(true);
    expect(levels.length).toBeGreaterThan(0);
    for (const sl of levels) {
      expect(sl.item_id).toBe('inv-001');
      expect(typeof sl.quantity).toBe('number');
    }
  });

  it('addInventoryItem returns an InventoryItem with an id', async () => {
    const item = await addInventoryItem({
      name: 'Batteries',
      category: 'Electronics',
      sku: 'EL-BAT-AA',
      unit: 'packs',
      reorder_point: 10,
      reorder_quantity: 50,
    });
    expect(typeof item.id).toBe('string');
    expect(item.name).toBe('Batteries');
  });

  it('updateStockLevel returns a valid StockLevel', async () => {
    const sl = await updateStockLevel('inv-001', 42, 'Restock');
    expect(sl.item_id).toBe('inv-001');
    expect(sl.quantity).toBe(42);
    expect(typeof sl.id).toBe('string');
    expect(typeof sl.recorded_at).toBe('number');
  });
});

// ============================================================================
// Water Safety Client
// ============================================================================

describe('resilience-client: Water Safety domain (demo mode)', () => {
  it('getAllWaterSystems returns an array of WaterSystems', async () => {
    const systems = await getAllWaterSystems();
    expect(Array.isArray(systems)).toBe(true);
    expect(systems.length).toBeGreaterThan(0);
    for (const s of systems) {
      expect(typeof s.id).toBe('string');
      expect(typeof s.capacity_liters).toBe('number');
      expect(typeof s.efficiency_percent).toBe('number');
    }
  });

  it('getMyWaterSystems filters to self-owned systems', async () => {
    const systems = await getMyWaterSystems();
    expect(Array.isArray(systems)).toBe(true);
    // In demo mode, filters to owner_did === 'self.did'
    for (const s of systems) {
      expect(s.owner_did).toBe('self.did');
    }
  });

  it('registerWaterSystem returns a WaterSystem with id', async () => {
    const system = await registerWaterSystem({
      name: 'Test System',
      system_type: 'RoofRainwater',
      capacity_liters: 5000,
      catchment_area_sqm: 60,
      efficiency_percent: 80,
      owner_did: 'self.did',
      location_lat: -26.15,
      location_lon: 27.86,
      installed_at: Date.now(),
    });
    expect(typeof system.id).toBe('string');
    expect(system.name).toBe('Test System');
  });

  it('updateTankLevel returns a valid StorageTank', async () => {
    const tank = await updateTankLevel('tank-001', 3500);
    expect(tank.id).toBe('tank-001');
    expect(tank.current_level_liters).toBe(3500);
    expect(typeof tank.capacity_liters).toBe('number');
  });

  it('submitWaterReading returns a WaterReading with id', async () => {
    const reading = await submitWaterReading({
      source_id: 'ws-001',
      parameter: 'pH',
      value: 7.2,
      unit: 'pH',
      location: 'Test',
      recorded_at: Date.now(),
      recorder_did: 'self.did',
    });
    expect(typeof reading.id).toBe('string');
    expect(reading.parameter).toBe('pH');
    expect(reading.value).toBe(7.2);
  });

  it('getActiveWaterAlerts returns an array of ContaminationAlerts', async () => {
    const alerts = await getActiveWaterAlerts();
    expect(Array.isArray(alerts)).toBe(true);
    for (const a of alerts) {
      expect(typeof a.id).toBe('string');
      expect(typeof a.contaminant).toBe('string');
      expect(typeof a.measured_value).toBe('number');
      expect(typeof a.threshold_value).toBe('number');
    }
  });
});

// ============================================================================
// Household / Hearth Client
// ============================================================================

describe('resilience-client: Hearth domain (demo mode)', () => {
  it('getMyHearths returns an array of Hearths', async () => {
    const hearths = await getMyHearths();
    expect(Array.isArray(hearths)).toBe(true);
    expect(hearths.length).toBeGreaterThan(0);
    for (const h of hearths) {
      expect(typeof h.id).toBe('string');
      expect(typeof h.name).toBe('string');
      expect(typeof h.member_count).toBe('number');
    }
  });

  it('getEmergencyPlan returns a valid EmergencyPlan', async () => {
    const plan = await getEmergencyPlan('hth-001');
    expect(plan.hearth_id).toBe('hth-001');
    expect(Array.isArray(plan.contacts)).toBe(true);
    expect(plan.contacts.length).toBeGreaterThan(0);
    expect(Array.isArray(plan.meeting_points)).toBe(true);
    expect(typeof plan.last_reviewed).toBe('number');
  });

  it('createEmergencyPlan returns a plan with given contacts', async () => {
    const contacts = [{ name: 'Test', phone: '123', relationship: 'Neighbour' }];
    const points = ['Park entrance'];
    const plan = await createEmergencyPlan('hth-001', contacts, points);
    expect(plan.hearth_id).toBe('hth-001');
    expect(plan.contacts).toEqual(contacts);
    expect(plan.meeting_points).toEqual(points);
  });

  it('raiseHearthAlert returns a valid HearthAlert', async () => {
    const alert = await raiseHearthAlert('hth-001', 'Flood', 'High', 'Heavy rain');
    expect(alert.hearth_id).toBe('hth-001');
    expect(alert.alert_type).toBe('Flood');
    expect(alert.severity).toBe('High');
    expect(alert.resolved).toBe(false);
  });

  it('getActiveHearthAlerts returns an array of HearthAlerts', async () => {
    const alerts = await getActiveHearthAlerts('hth-001');
    expect(Array.isArray(alerts)).toBe(true);
    for (const a of alerts) {
      expect(typeof a.id).toBe('string');
      expect(typeof a.message).toBe('string');
    }
  });

  it('checkIn returns a valid SafetyCheckIn', async () => {
    const ci = await checkIn('ha-001', 'Safe');
    expect(ci.alert_id).toBe('ha-001');
    expect(ci.status).toBe('Safe');
    expect(typeof ci.id).toBe('string');
    expect(typeof ci.checked_in_at).toBe('number');
  });

  it('getHearthInventory returns an array of SharedResources', async () => {
    const resources = await getHearthInventory('hth-001');
    expect(Array.isArray(resources)).toBe(true);
    expect(resources.length).toBeGreaterThan(0);
    for (const r of resources) {
      expect(typeof r.id).toBe('string');
      expect(typeof r.name).toBe('string');
      expect(typeof r.resource_type).toBe('string');
    }
  });

  it('registerResource returns a valid SharedResource', async () => {
    const r = await registerResource('hth-001', 'Shovel', 'Garden shovel', 'Tools', 'Good', 'Shed');
    expect(r.hearth_id).toBe('hth-001');
    expect(r.name).toBe('Shovel');
    expect(r.resource_type).toBe('Tools');
    expect(r.current_holder).toBeNull();
  });
});

// ============================================================================
// Community Knowledge Client
// ============================================================================

describe('resilience-client: Knowledge domain (demo mode)', () => {
  it('searchClaimsByTag returns claims matching the tag', async () => {
    const claims = await searchClaimsByTag('water-safety');
    expect(Array.isArray(claims)).toBe(true);
    for (const c of claims) {
      expect(c.tags).toContain('water-safety');
      expect(typeof c.confidence).toBe('number');
      expect(typeof c.e_score).toBe('number');
    }
  });

  it('submitClaim returns a valid KnowledgeClaim', async () => {
    const claim = await submitClaim('Test claim', ['test'], 0.8);
    expect(claim.content).toBe('Test claim');
    expect(claim.tags).toEqual(['test']);
    expect(claim.confidence).toBe(0.8);
    expect(typeof claim.id).toBe('string');
  });

  it('getGraphStats returns a valid GraphStats', async () => {
    const stats = await getGraphStats();
    expect(typeof stats.total_claims).toBe('number');
    expect(typeof stats.total_relationships).toBe('number');
    expect(typeof stats.total_ontologies).toBe('number');
    expect(typeof stats.total_concepts).toBe('number');
  });

  it('findKnowledgePath returns a path object', async () => {
    const result = await findKnowledgePath('water', 'health') as Record<string, unknown>;
    expect(result.source).toBe('water');
    expect(result.target).toBe('health');
    expect(Array.isArray(result.path)).toBe(true);
    expect(typeof result.hops).toBe('number');
  });
});
