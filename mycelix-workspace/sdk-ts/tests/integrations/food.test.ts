// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Food Sovereignty Integration Tests
 *
 * Tests for FoodClient -- the domain-specific SDK client for the
 * food sovereignty zomes within the mycelix-commons cluster DNA.
 * Covers production (plots, crops, harvests, season plans),
 * distribution (markets, listings, orders), preservation (batches,
 * methods, storage), and knowledge (seeds, practices, recipes).
 *
 * All calls are dispatched through the commons role to the food_production,
 * food_distribution, food_preservation, and food_knowledge coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  FoodClient,
  createFoodClient,
  FOOD_ZOMES,
  type Plot,
  type PlotStatus,
  type Crop,
  type CropStatus,
  type YieldRecord,
  type QualityGrade,
  type SeasonPlan,
  type Market,
  type MarketType,
  type Listing,
  type Order,
  type OrderStatus,
  type PreservationBatch,
  type PreservationStatus,
  type PreservationMethod,
  type SkillLevel,
  type StorageUnit,
  type StorageType,
  type SeedVariety,
  type TraditionalPractice,
  type PracticeCategory,
  type Recipe,
  type SeedStock,
  type FoodSeedRequest,
  type SeedRequestStatus,
  type MatchSeedInput,
  type NutrientProfile,
  type AllergenSearchInput,
  type GardenRole,
  type GardenMembership,
  type AddMemberInput,
  type RemoveMemberInput,
} from '../../src/integrations/food/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

// ============================================================================
// Test helpers
// ============================================================================

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function makePlot(overrides: Partial<Plot> = {}): Plot {
  return {
    id: 'plot-001',
    name: 'Community Garden A',
    area_sqm: 500,
    soil_type: 'loam',
    location_lat: 32.95,
    location_lon: -96.75,
    steward: fakeAgentKey(),
    status: 'Active',
    ...overrides,
  };
}

function makeCrop(overrides: Partial<Crop> = {}): Crop {
  return {
    plot_hash: fakeActionHash(),
    name: 'Tomato',
    variety: 'Cherokee Purple',
    planted_at: Date.now(),
    expected_harvest: Date.now() + 90 * 24 * 3600_000,
    status: 'Planted',
    allergen_flags: [],
    organic_certified: false,
    ...overrides,
  };
}

function makeYieldRecord(overrides: Partial<YieldRecord> = {}): YieldRecord {
  return {
    crop_hash: fakeActionHash(),
    quantity_kg: 25.5,
    quality_grade: 'Premium',
    harvested_at: Date.now(),
    notes: 'Excellent harvest',
    ...overrides,
  };
}

function makeSeasonPlan(overrides: Partial<SeasonPlan> = {}): SeasonPlan {
  return {
    plot_hash: fakeActionHash(),
    year: 2026,
    season: 'Spring',
    planned_crops: ['Tomato', 'Basil', 'Pepper'],
    rotation_notes: 'Follow legumes from last season',
    ...overrides,
  };
}

function makeMarket(overrides: Partial<Market> = {}): Market {
  return {
    id: 'market-001',
    name: 'Downtown Farmers Market',
    location_lat: 32.96,
    location_lon: -96.76,
    market_type: 'Farmers',
    steward: fakeAgentKey(),
    schedule: 'Every Saturday 8am-1pm',
    ...overrides,
  };
}

function makeListing(overrides: Partial<Listing> = {}): Listing {
  return {
    market_hash: fakeActionHash(),
    producer: fakeAgentKey(),
    product_name: 'Heirloom Tomatoes',
    quantity_kg: 50,
    price_per_kg: 4.5,
    available_from: Date.now(),
    status: 'Available',
    allergen_flags: [],
    organic: false,
    cultural_markers: [],
    ...overrides,
  };
}

function makeOrder(overrides: Partial<Order> = {}): Order {
  return {
    listing_hash: fakeActionHash(),
    buyer: fakeAgentKey(),
    quantity_kg: 5,
    status: 'Pending',
    ...overrides,
  };
}

function makeBatch(overrides: Partial<PreservationBatch> = {}): PreservationBatch {
  return {
    id: 'batch-001',
    source_crop_hash: fakeActionHash(),
    method: 'Canning',
    quantity_kg: 10,
    started_at: Date.now(),
    expected_ready: Date.now() + 7 * 24 * 3600_000,
    status: 'InProgress',
    notes: 'First canning batch of the season',
    ...overrides,
  };
}

function makeMethod(overrides: Partial<PreservationMethod> = {}): PreservationMethod {
  return {
    name: 'Water Bath Canning',
    description: 'Preserve high-acid foods in sealed jars',
    shelf_life_days: 365,
    equipment_needed: ['Canning jars', 'Water bath canner', 'Jar lifter'],
    skill_level: 'Beginner',
    ...overrides,
  };
}

function makeStorageUnit(overrides: Partial<StorageUnit> = {}): StorageUnit {
  return {
    id: 'storage-001',
    name: 'Community Root Cellar',
    capacity_kg: 2000,
    storage_type: 'RootCellar',
    steward: fakeAgentKey(),
    ...overrides,
  };
}

function makeSeed(overrides: Partial<SeedVariety> = {}): SeedVariety {
  return {
    name: 'Cherokee Purple',
    species: 'Solanum lycopersicum',
    origin: 'Tennessee, USA',
    days_to_maturity: 80,
    companion_plants: ['Basil', 'Carrot', 'Marigold'],
    avoid_plants: ['Brassicas', 'Fennel'],
    seed_saving_notes: 'Allow fruits to fully ripen on vine before saving seeds',
    ...overrides,
  };
}

function makePractice(overrides: Partial<TraditionalPractice> = {}): TraditionalPractice {
  return {
    name: 'Three Sisters Planting',
    description: 'Companion planting of corn, beans, and squash',
    region: 'North America',
    season: 'Spring',
    category: 'Planting',
    ...overrides,
  };
}

function makeRecipe(overrides: Partial<Recipe> = {}): Recipe {
  return {
    name: 'Garden Salsa',
    ingredients: ['tomatoes', 'onion', 'jalapeno', 'cilantro', 'lime juice', 'salt'],
    instructions: 'Dice all ingredients, mix, and season to taste',
    servings: 8,
    prep_time_min: 15,
    tags: ['salsa', 'fresh', 'summer', 'no-cook'],
    source_attribution: 'Community Garden Cookbook',
    ...overrides,
  };
}

// ============================================================================
// Constants
// ============================================================================

describe('Food Constants', () => {
  it('should export all 4 food zome names', () => {
    expect(FOOD_ZOMES).toEqual([
      'food_production',
      'food_distribution',
      'food_preservation',
      'food_knowledge',
    ]);
    expect(FOOD_ZOMES).toHaveLength(4);
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Food Types', () => {
  describe('Plot', () => {
    it('should construct a valid Plot', () => {
      const plot = makePlot();
      expect(plot.id).toBe('plot-001');
      expect(plot.name).toBe('Community Garden A');
      expect(plot.area_sqm).toBe(500);
      expect(plot.soil_type).toBe('loam');
      expect(plot.status).toBe('Active');
      expect(plot.steward).toBeInstanceOf(Uint8Array);
    });

    it('should accept all PlotStatus variants', () => {
      const statuses: PlotStatus[] = ['Active', 'Fallow', 'Preparing', 'Retired'];
      statuses.forEach((s) => {
        const p = makePlot({ status: s });
        expect(p.status).toBe(s);
      });
    });
  });

  describe('Crop', () => {
    it('should construct a valid Crop', () => {
      const crop = makeCrop();
      expect(crop.name).toBe('Tomato');
      expect(crop.variety).toBe('Cherokee Purple');
      expect(crop.status).toBe('Planted');
      expect(crop.plot_hash).toBeInstanceOf(Uint8Array);
    });

    it('should accept all CropStatus variants', () => {
      const statuses: CropStatus[] = ['Planned', 'Planted', 'Growing', 'Ready', 'Harvested', 'Failed'];
      statuses.forEach((s) => {
        const c = makeCrop({ status: s });
        expect(c.status).toBe(s);
      });
    });
  });

  describe('YieldRecord', () => {
    it('should construct a valid YieldRecord', () => {
      const y = makeYieldRecord();
      expect(y.quantity_kg).toBe(25.5);
      expect(y.quality_grade).toBe('Premium');
      expect(y.crop_hash).toBeInstanceOf(Uint8Array);
    });

    it('should accept all QualityGrade variants', () => {
      const grades: QualityGrade[] = ['Premium', 'Standard', 'Processing', 'Compost'];
      grades.forEach((g) => {
        const y = makeYieldRecord({ quality_grade: g });
        expect(y.quality_grade).toBe(g);
      });
    });
  });

  describe('SeasonPlan', () => {
    it('should construct a valid SeasonPlan', () => {
      const plan = makeSeasonPlan();
      expect(plan.year).toBe(2026);
      expect(plan.season).toBe('Spring');
      expect(plan.planned_crops).toHaveLength(3);
    });
  });

  describe('Market', () => {
    it('should construct a valid Market', () => {
      const market = makeMarket();
      expect(market.name).toBe('Downtown Farmers Market');
      expect(market.market_type).toBe('Farmers');
      expect(market.steward).toBeInstanceOf(Uint8Array);
    });

    it('should accept all MarketType variants', () => {
      const types: MarketType[] = ['Farmers', 'CSA', 'FoodBank', 'CoOp'];
      types.forEach((t) => {
        const m = makeMarket({ market_type: t });
        expect(m.market_type).toBe(t);
      });
    });
  });

  describe('Order', () => {
    it('should construct a valid Order', () => {
      const order = makeOrder();
      expect(order.quantity_kg).toBe(5);
      expect(order.status).toBe('Pending');
      expect(order.buyer).toBeInstanceOf(Uint8Array);
    });

    it('should accept all OrderStatus variants', () => {
      const statuses: OrderStatus[] = ['Pending', 'Confirmed', 'Fulfilled', 'Cancelled'];
      statuses.forEach((s) => {
        const o = makeOrder({ status: s });
        expect(o.status).toBe(s);
      });
    });
  });

  describe('PreservationBatch', () => {
    it('should construct with nullable source_crop_hash', () => {
      const batch = makeBatch({ source_crop_hash: null });
      expect(batch.source_crop_hash).toBeNull();
      expect(batch.status).toBe('InProgress');
    });

    it('should accept all PreservationStatus variants', () => {
      const statuses: PreservationStatus[] = ['InProgress', 'Complete', 'Failed'];
      statuses.forEach((s) => {
        const b = makeBatch({ status: s });
        expect(b.status).toBe(s);
      });
    });
  });

  describe('PreservationMethod', () => {
    it('should construct a valid PreservationMethod', () => {
      const method = makeMethod();
      expect(method.name).toBe('Water Bath Canning');
      expect(method.shelf_life_days).toBe(365);
      expect(method.equipment_needed).toHaveLength(3);
    });

    it('should accept all SkillLevel variants', () => {
      const levels: SkillLevel[] = ['Beginner', 'Intermediate', 'Advanced'];
      levels.forEach((l) => {
        const m = makeMethod({ skill_level: l });
        expect(m.skill_level).toBe(l);
      });
    });
  });

  describe('StorageUnit', () => {
    it('should construct a valid StorageUnit', () => {
      const unit = makeStorageUnit();
      expect(unit.name).toBe('Community Root Cellar');
      expect(unit.capacity_kg).toBe(2000);
      expect(unit.storage_type).toBe('RootCellar');
    });

    it('should accept all StorageType variants', () => {
      const types: StorageType[] = ['RootCellar', 'Freezer', 'Dehydrator', 'Fermenter', 'Pantry'];
      types.forEach((t) => {
        const u = makeStorageUnit({ storage_type: t });
        expect(u.storage_type).toBe(t);
      });
    });
  });

  describe('SeedVariety', () => {
    it('should construct a valid SeedVariety', () => {
      const seed = makeSeed();
      expect(seed.name).toBe('Cherokee Purple');
      expect(seed.species).toBe('Solanum lycopersicum');
      expect(seed.days_to_maturity).toBe(80);
      expect(seed.companion_plants).toHaveLength(3);
      expect(seed.avoid_plants).toHaveLength(2);
    });
  });

  describe('TraditionalPractice', () => {
    it('should construct a valid TraditionalPractice', () => {
      const practice = makePractice();
      expect(practice.name).toBe('Three Sisters Planting');
      expect(practice.category).toBe('Planting');
    });

    it('should accept all PracticeCategory variants', () => {
      const categories: PracticeCategory[] = ['Planting', 'Harvest', 'Soil', 'Pest', 'Water'];
      categories.forEach((c) => {
        const p = makePractice({ category: c });
        expect(p.category).toBe(c);
      });
    });
  });

  describe('Recipe', () => {
    it('should construct a valid Recipe', () => {
      const recipe = makeRecipe();
      expect(recipe.name).toBe('Garden Salsa');
      expect(recipe.servings).toBe(8);
      expect(recipe.prep_time_min).toBe(15);
      expect(recipe.ingredients).toHaveLength(6);
      expect(recipe.tags).toHaveLength(4);
    });
  });
});

// ============================================================================
// FoodClient -- Factory
// ============================================================================

describe('FoodClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let food: FoodClient;

  beforeEach(() => {
    client = createMockClient();
    food = createFoodClient(client);
  });

  describe('factory', () => {
    it('should create via createFoodClient', () => {
      expect(food).toBeInstanceOf(FoodClient);
    });

    it('should also be constructable directly', () => {
      const direct = new FoodClient(client);
      expect(direct).toBeInstanceOf(FoodClient);
    });
  });

  // ==========================================================================
  // Production zome
  // ==========================================================================

  describe('Production (food_production)', () => {
    describe('registerPlot', () => {
      it('should call food_production.register_plot with correct params', async () => {
        const plot = makePlot();
        const mockHash = fakeActionHash();
        client.callZome.mockResolvedValue(mockHash);

        const result = await food.registerPlot(plot);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'register_plot',
          payload: plot,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass all plot statuses through to the zome', async () => {
        const statuses: PlotStatus[] = ['Active', 'Fallow', 'Preparing', 'Retired'];
        for (const s of statuses) {
          const plot = makePlot({ status: s });
          await food.registerPlot(plot);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.status).toBe(s);
        }
      });
    });

    describe('getPlot', () => {
      it('should call food_production.get_plot with action hash', async () => {
        const hash = fakeActionHash();
        const mockPlot = makePlot();
        client.callZome.mockResolvedValue(mockPlot);

        const result = await food.getPlot(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_plot',
          payload: hash,
        });
        expect(result).toEqual(mockPlot);
      });
    });

    describe('getAllPlots', () => {
      it('should call food_production.get_all_plots with null payload', async () => {
        const mockPlots = [makePlot({ id: 'p1' }), makePlot({ id: 'p2' })];
        client.callZome.mockResolvedValue(mockPlots);

        const result = await food.getAllPlots();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_all_plots',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('plantCrop', () => {
      it('should call food_production.plant_crop with crop data', async () => {
        const crop = makeCrop();
        const mockHash = fakeActionHash();
        client.callZome.mockResolvedValue(mockHash);

        const result = await food.plantCrop(crop);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'plant_crop',
          payload: crop,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass all crop statuses through', async () => {
        const statuses: CropStatus[] = ['Planned', 'Planted', 'Growing', 'Ready', 'Harvested', 'Failed'];
        for (const s of statuses) {
          const crop = makeCrop({ status: s });
          await food.plantCrop(crop);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.status).toBe(s);
        }
      });
    });

    describe('recordHarvest', () => {
      it('should call food_production.record_harvest with yield data', async () => {
        const yieldRecord = makeYieldRecord();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.recordHarvest(yieldRecord);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'record_harvest',
          payload: yieldRecord,
        });
      });

      it('should pass all quality grades through', async () => {
        const grades: QualityGrade[] = ['Premium', 'Standard', 'Processing', 'Compost'];
        for (const g of grades) {
          const y = makeYieldRecord({ quality_grade: g });
          await food.recordHarvest(y);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.quality_grade).toBe(g);
        }
      });
    });

    describe('getPlotCrops', () => {
      it('should call food_production.get_plot_crops with plot hash', async () => {
        const plotHash = fakeActionHash();
        const mockCrops = [makeCrop({ name: 'Tomato' }), makeCrop({ name: 'Basil' })];
        client.callZome.mockResolvedValue(mockCrops);

        const result = await food.getPlotCrops(plotHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_plot_crops',
          payload: plotHash,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getCropYields', () => {
      it('should call food_production.get_crop_yields with crop hash', async () => {
        const cropHash = fakeActionHash();
        const mockYields = [makeYieldRecord()];
        client.callZome.mockResolvedValue(mockYields);

        const result = await food.getCropYields(cropHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_crop_yields',
          payload: cropHash,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('createSeasonPlan', () => {
      it('should call food_production.create_season_plan with plan data', async () => {
        const plan = makeSeasonPlan();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.createSeasonPlan(plan);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'create_season_plan',
          payload: plan,
        });
      });

      it('should pass planned_crops array through', async () => {
        const plan = makeSeasonPlan({ planned_crops: ['Corn', 'Beans', 'Squash'] });
        await food.createSeasonPlan(plan);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.planned_crops).toHaveLength(3);
        expect(lastCall.payload.planned_crops[0]).toBe('Corn');
      });
    });

    describe('getSeasonPlans', () => {
      it('should call food_production.get_season_plans with plot hash', async () => {
        const plotHash = fakeActionHash();
        const mockPlans = [makeSeasonPlan()];
        client.callZome.mockResolvedValue(mockPlans);

        const result = await food.getSeasonPlans(plotHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_season_plans',
          payload: plotHash,
        });
        expect(result).toHaveLength(1);
      });
    });
  });

  // ==========================================================================
  // Distribution zome
  // ==========================================================================

  describe('Distribution (food_distribution)', () => {
    describe('createMarket', () => {
      it('should call food_distribution.create_market with market data', async () => {
        const market = makeMarket();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.createMarket(market);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'create_market',
          payload: market,
        });
      });

      it('should pass all market types through', async () => {
        const types: MarketType[] = ['Farmers', 'CSA', 'FoodBank', 'CoOp'];
        for (const t of types) {
          const market = makeMarket({ market_type: t });
          await food.createMarket(market);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.market_type).toBe(t);
        }
      });
    });

    describe('getAllMarkets', () => {
      it('should call food_distribution.get_all_markets with null payload', async () => {
        const mockMarkets = [makeMarket()];
        client.callZome.mockResolvedValue(mockMarkets);

        const result = await food.getAllMarkets();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'get_all_markets',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('listProduct', () => {
      it('should call food_distribution.list_product with listing data', async () => {
        const listing = makeListing();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.listProduct(listing);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'list_product',
          payload: listing,
        });
      });

      it('should pass price and quantity through correctly', async () => {
        const listing = makeListing({ quantity_kg: 100, price_per_kg: 6.0 });
        await food.listProduct(listing);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.quantity_kg).toBe(100);
        expect(lastCall.payload.price_per_kg).toBe(6.0);
      });
    });

    describe('getMarketListings', () => {
      it('should call food_distribution.get_market_listings with market hash', async () => {
        const marketHash = fakeActionHash();
        const mockListings = [makeListing(), makeListing({ product_name: 'Peppers' })];
        client.callZome.mockResolvedValue(mockListings);

        const result = await food.getMarketListings(marketHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'get_market_listings',
          payload: marketHash,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getProducerListings', () => {
      it('should call food_distribution.get_producer_listings with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await food.getProducerListings();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'get_producer_listings',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('placeOrder', () => {
      it('should call food_distribution.place_order with order data', async () => {
        const order = makeOrder();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.placeOrder(order);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'place_order',
          payload: order,
        });
      });
    });

    describe('fulfillOrder', () => {
      it('should call food_distribution.fulfill_order with order hash and Fulfilled status', async () => {
        const orderHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await food.fulfillOrder(orderHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'fulfill_order',
          payload: { order_hash: orderHash, new_status: 'Fulfilled' },
        });
      });
    });

    describe('cancelOrder', () => {
      it('should call food_distribution.cancel_order with order hash', async () => {
        const orderHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await food.cancelOrder(orderHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'cancel_order',
          payload: orderHash,
        });
      });
    });

    describe('getMyOrders', () => {
      it('should call food_distribution.get_my_orders with null payload', async () => {
        const mockOrders = [makeOrder(), makeOrder({ status: 'Confirmed' })];
        client.callZome.mockResolvedValue(mockOrders);

        const result = await food.getMyOrders();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'get_my_orders',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });
  });

  // ==========================================================================
  // Preservation zome
  // ==========================================================================

  describe('Preservation (food_preservation)', () => {
    describe('startBatch', () => {
      it('should call food_preservation.start_batch with batch data', async () => {
        const batch = makeBatch();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.startBatch(batch);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'start_batch',
          payload: batch,
        });
      });

      it('should handle batch with null source_crop_hash', async () => {
        const batch = makeBatch({ source_crop_hash: null });
        await food.startBatch(batch);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.source_crop_hash).toBeNull();
      });
    });

    describe('completeBatch', () => {
      it('should call food_preservation.complete_batch with batch hash', async () => {
        const batchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await food.completeBatch(batchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'complete_batch',
          payload: batchHash,
        });
      });
    });

    describe('getBatch', () => {
      it('should call food_preservation.get_batch with batch hash', async () => {
        const batchHash = fakeActionHash();
        const mockBatch = makeBatch();
        client.callZome.mockResolvedValue(mockBatch);

        const result = await food.getBatch(batchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'get_batch',
          payload: batchHash,
        });
        expect(result).toEqual(mockBatch);
      });
    });

    describe('registerMethod', () => {
      it('should call food_preservation.register_method with method data', async () => {
        const method = makeMethod();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.registerMethod(method);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'register_method',
          payload: method,
        });
      });

      it('should pass all skill levels through', async () => {
        const levels: SkillLevel[] = ['Beginner', 'Intermediate', 'Advanced'];
        for (const l of levels) {
          const method = makeMethod({ skill_level: l });
          await food.registerMethod(method);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.skill_level).toBe(l);
        }
      });
    });

    describe('getAllMethods', () => {
      it('should call food_preservation.get_all_methods with null payload', async () => {
        const mockMethods = [makeMethod()];
        client.callZome.mockResolvedValue(mockMethods);

        const result = await food.getAllMethods();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'get_all_methods',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('registerStorage', () => {
      it('should call food_preservation.register_storage with unit data', async () => {
        const unit = makeStorageUnit();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.registerStorage(unit);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'register_storage',
          payload: unit,
        });
      });

      it('should pass all storage types through', async () => {
        const types: StorageType[] = ['RootCellar', 'Freezer', 'Dehydrator', 'Fermenter', 'Pantry'];
        for (const t of types) {
          const unit = makeStorageUnit({ storage_type: t });
          await food.registerStorage(unit);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.storage_type).toBe(t);
        }
      });
    });

    describe('getStorageInventory', () => {
      it('should call food_preservation.get_storage_inventory with storage hash', async () => {
        const storageHash = fakeActionHash();
        client.callZome.mockResolvedValue([]);

        const result = await food.getStorageInventory(storageHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'get_storage_inventory',
          payload: storageHash,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getAgentBatches', () => {
      it('should call food_preservation.get_agent_batches with null payload', async () => {
        const mockBatches = [makeBatch(), makeBatch({ id: 'batch-002' })];
        client.callZome.mockResolvedValue(mockBatches);

        const result = await food.getAgentBatches();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_preservation',
          fn_name: 'get_agent_batches',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });
  });

  // ==========================================================================
  // Knowledge zome
  // ==========================================================================

  describe('Knowledge (food_knowledge)', () => {
    describe('catalogSeed', () => {
      it('should call food_knowledge.catalog_seed with seed data', async () => {
        const seed = makeSeed();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.catalogSeed(seed);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'catalog_seed',
          payload: seed,
        });
      });

      it('should pass companion and avoid plant arrays through', async () => {
        const seed = makeSeed({
          companion_plants: ['Lettuce'],
          avoid_plants: ['Walnut'],
        });
        await food.catalogSeed(seed);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.companion_plants).toEqual(['Lettuce']);
        expect(lastCall.payload.avoid_plants).toEqual(['Walnut']);
      });
    });

    describe('getSeed', () => {
      it('should call food_knowledge.get_seed with seed hash', async () => {
        const seedHash = fakeActionHash();
        const mockSeed = makeSeed();
        client.callZome.mockResolvedValue(mockSeed);

        const result = await food.getSeed(seedHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_seed',
          payload: seedHash,
        });
        expect(result).toEqual(mockSeed);
      });
    });

    describe('getSeedsBySpecies', () => {
      it('should call food_knowledge.get_seeds_by_species with species string', async () => {
        const species = 'Solanum lycopersicum';
        const mockSeeds = [makeSeed(), makeSeed({ name: 'Brandywine' })];
        client.callZome.mockResolvedValue(mockSeeds);

        const result = await food.getSeedsBySpecies(species);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_seeds_by_species',
          payload: species,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('sharePractice', () => {
      it('should call food_knowledge.share_practice with practice data', async () => {
        const practice = makePractice();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.sharePractice(practice);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'share_practice',
          payload: practice,
        });
      });

      it('should pass all practice categories through', async () => {
        const categories: PracticeCategory[] = ['Planting', 'Harvest', 'Soil', 'Pest', 'Water'];
        for (const c of categories) {
          const practice = makePractice({ category: c });
          await food.sharePractice(practice);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.category).toBe(c);
        }
      });
    });

    describe('getPracticesByCategory', () => {
      it('should call food_knowledge.get_practices_by_category with category', async () => {
        const mockPractices = [makePractice()];
        client.callZome.mockResolvedValue(mockPractices);

        const result = await food.getPracticesByCategory('Planting');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_practices_by_category',
          payload: 'Planting',
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('shareRecipe', () => {
      it('should call food_knowledge.share_recipe with recipe data', async () => {
        const recipe = makeRecipe();
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.shareRecipe(recipe);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'share_recipe',
          payload: recipe,
        });
      });

      it('should pass ingredients and tags arrays through', async () => {
        const recipe = makeRecipe({
          ingredients: ['flour', 'water', 'salt'],
          tags: ['bread', 'baking'],
        });
        await food.shareRecipe(recipe);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.ingredients).toHaveLength(3);
        expect(lastCall.payload.tags).toHaveLength(2);
      });
    });

    describe('getRecipesByTag', () => {
      it('should call food_knowledge.get_recipes_by_tag with tag string', async () => {
        const mockRecipes = [makeRecipe()];
        client.callZome.mockResolvedValue(mockRecipes);

        const result = await food.getRecipesByTag('salsa');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_recipes_by_tag',
          payload: 'salsa',
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('searchKnowledge', () => {
      it('should call food_knowledge.search_knowledge with query string', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await food.searchKnowledge('companion planting');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'search_knowledge',
          payload: 'companion planting',
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Production lifecycle (plot -> crop -> harvest -> season plan)
  // ==========================================================================

  describe('Production Lifecycle', () => {
    it('should support the full production lifecycle through mock calls', async () => {
      const plotHash = new Uint8Array(39).fill(0x01);
      const cropHash = new Uint8Array(39).fill(0x02);
      const yieldHash = new Uint8Array(39).fill(0x03);
      const planHash = new Uint8Array(39).fill(0x04);

      // Step 1: Register a plot
      client.callZome.mockResolvedValueOnce(plotHash);
      const plotResult = await food.registerPlot(makePlot());
      expect(plotResult).toEqual(plotHash);

      // Step 2: Plant a crop
      client.callZome.mockResolvedValueOnce(cropHash);
      const cropResult = await food.plantCrop(makeCrop({ plot_hash: plotHash }));
      expect(cropResult).toEqual(cropHash);

      // Step 3: Record a harvest
      client.callZome.mockResolvedValueOnce(yieldHash);
      await food.recordHarvest(makeYieldRecord({ crop_hash: cropHash }));

      // Step 4: Create a season plan
      client.callZome.mockResolvedValueOnce(planHash);
      await food.createSeasonPlan(makeSeasonPlan({ plot_hash: plotHash }));

      // Verify 4 zome calls were made
      expect(client.callZome).toHaveBeenCalledTimes(4);

      // Verify the correct sequence of zome functions
      const fnNames = client.callZome.mock.calls.map((c: unknown[]) => (c[0] as { fn_name: string }).fn_name);
      expect(fnNames).toEqual([
        'register_plot',
        'plant_crop',
        'record_harvest',
        'create_season_plan',
      ]);
    });
  });

  // ==========================================================================
  // Distribution lifecycle (market -> listing -> order -> fulfill)
  // ==========================================================================

  describe('Distribution Lifecycle', () => {
    it('should support the full order lifecycle through mock calls', async () => {
      const marketHash = new Uint8Array(39).fill(0x01);
      const listingHash = new Uint8Array(39).fill(0x02);
      const orderHash = new Uint8Array(39).fill(0x03);

      // Step 1: Create a market
      client.callZome.mockResolvedValueOnce(marketHash);
      await food.createMarket(makeMarket());

      // Step 2: List a product
      client.callZome.mockResolvedValueOnce(listingHash);
      await food.listProduct(makeListing({ market_hash: marketHash }));

      // Step 3: Place an order
      client.callZome.mockResolvedValueOnce(orderHash);
      await food.placeOrder(makeOrder({ listing_hash: listingHash }));

      // Step 4: Fulfill the order
      client.callZome.mockResolvedValueOnce({});
      await food.fulfillOrder(orderHash);

      expect(client.callZome).toHaveBeenCalledTimes(4);

      const fnNames = client.callZome.mock.calls.map((c: unknown[]) => (c[0] as { fn_name: string }).fn_name);
      expect(fnNames).toEqual([
        'create_market',
        'list_product',
        'place_order',
        'fulfill_order',
      ]);
    });

    it('should support cancellation flow', async () => {
      const orderHash = fakeActionHash();
      client.callZome.mockResolvedValue({});

      await food.cancelOrder(orderHash);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'food_distribution',
        fn_name: 'cancel_order',
        payload: orderHash,
      });
    });
  });

  // ==========================================================================
  // Cross-domain dispatch pattern
  // ==========================================================================

  describe('Cross-domain dispatch to food', () => {
    it('should use commons role for all food_production calls', async () => {
      await food.registerPlot(makePlot());
      await food.plantCrop(makeCrop());
      await food.getAllPlots();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_land');
        expect((call[0] as { zome_name: string }).zome_name).toBe('food_production');
      }
    });

    it('should use commons role for all food_distribution calls', async () => {
      await food.createMarket(makeMarket());
      await food.getAllMarkets();
      await food.getProducerListings();
      await food.getMyOrders();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_land');
        expect((call[0] as { zome_name: string }).zome_name).toBe('food_distribution');
      }
    });

    it('should use commons role for all food_preservation calls', async () => {
      await food.startBatch(makeBatch());
      await food.getAllMethods();
      await food.getAgentBatches();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_land');
        expect((call[0] as { zome_name: string }).zome_name).toBe('food_preservation');
      }
    });

    it('should use commons role for all food_knowledge calls', async () => {
      await food.catalogSeed(makeSeed());
      await food.sharePractice(makePractice());
      await food.shareRecipe(makeRecipe());
      await food.searchKnowledge('tomato');

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_land');
        expect((call[0] as { zome_name: string }).zome_name).toBe('food_knowledge');
      }
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate zome call errors for registerPlot', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(food.registerPlot(makePlot())).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate zome call errors for plantCrop', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(food.plantCrop(makeCrop())).rejects.toThrow('Validation failed');
    });

    it('should propagate zome call errors for createMarket', async () => {
      client.callZome.mockRejectedValue(new Error('Rate limited'));
      await expect(food.createMarket(makeMarket())).rejects.toThrow('Rate limited');
    });

    it('should propagate zome call errors for startBatch', async () => {
      client.callZome.mockRejectedValue(new Error('Invalid method'));
      await expect(food.startBatch(makeBatch())).rejects.toThrow('Invalid method');
    });

    it('should propagate zome call errors for searchKnowledge', async () => {
      client.callZome.mockRejectedValue(new Error('Network error'));
      await expect(food.searchKnowledge('query')).rejects.toThrow('Network error');
    });
  });

  // ==========================================================================
  // Seed Exchange (Phase 2)
  // ==========================================================================

  describe('Seed Exchange (food_knowledge)', () => {
    describe('offerSeeds', () => {
      it('should call food_knowledge.offer_seeds with stock data', async () => {
        const stock: SeedStock = {
          variety_hash: fakeActionHash(),
          grower: fakeAgentKey(),
          quantity_grams: 250,
          location: 'Richardson, TX',
          germination_rate_pct: 92.5,
          available_for_exchange: true,
          notes: 'Saved from 2025 harvest',
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.offerSeeds(stock);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'offer_seeds',
          payload: stock,
        });
      });

      it('should handle nullable germination_rate_pct and notes', async () => {
        const stock: SeedStock = {
          variety_hash: fakeActionHash(),
          grower: fakeAgentKey(),
          quantity_grams: 100,
          location: 'Dallas, TX',
          germination_rate_pct: null,
          available_for_exchange: true,
          notes: null,
        };
        await food.offerSeeds(stock);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.germination_rate_pct).toBeNull();
        expect(lastCall.payload.notes).toBeNull();
      });
    });

    describe('requestSeeds', () => {
      it('should call food_knowledge.request_seeds with request data', async () => {
        const request: FoodSeedRequest = {
          wanted_variety: 'Cherokee Purple',
          quantity_grams: 50,
          requester: fakeAgentKey(),
          status: 'Open',
          deadline: Date.now() + 30 * 24 * 3600_000,
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.requestSeeds(request);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'request_seeds',
          payload: request,
        });
      });

      it('should accept all SeedRequestStatus variants', async () => {
        const statuses: SeedRequestStatus[] = ['Open', 'Matched', 'Fulfilled'];
        for (const s of statuses) {
          const request: FoodSeedRequest = {
            wanted_variety: 'Brandywine',
            quantity_grams: 25,
            requester: fakeAgentKey(),
            status: s,
            deadline: null,
          };
          await food.requestSeeds(request);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.status).toBe(s);
        }
      });
    });

    describe('getAvailableSeeds', () => {
      it('should call food_knowledge.get_available_seeds with variety hash', async () => {
        const varietyHash = fakeActionHash();
        client.callZome.mockResolvedValue([]);

        const result = await food.getAvailableSeeds(varietyHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_available_seeds',
          payload: varietyHash,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getOpenSeedRequests', () => {
      it('should call food_knowledge.get_open_seed_requests with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await food.getOpenSeedRequests();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_open_seed_requests',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('matchSeedRequest', () => {
      it('should call food_knowledge.match_seed_request with input data', async () => {
        const input: MatchSeedInput = {
          request_hash: new Uint8Array(39).fill(0x01),
          stock_hash: new Uint8Array(39).fill(0x02),
        };
        client.callZome.mockResolvedValue({});

        await food.matchSeedRequest(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'match_seed_request',
          payload: input,
        });
      });
    });
  });

  // ==========================================================================
  // Nutrient Tracking (Phase 2)
  // ==========================================================================

  describe('Nutrient Tracking (food_knowledge)', () => {
    describe('addNutrientProfile', () => {
      it('should call food_knowledge.add_nutrient_profile with profile data', async () => {
        const profile: NutrientProfile = {
          crop_name: 'Cherokee Purple Tomato',
          calories_per_100g: 18,
          protein_g: 0.9,
          carbs_g: 3.9,
          fat_g: 0.2,
          fiber_g: 1.2,
          key_vitamins: ['C', 'K', 'A'],
          key_minerals: ['Potassium', 'Manganese'],
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.addNutrientProfile(profile);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'add_nutrient_profile',
          payload: profile,
        });
      });

      it('should pass vitamin and mineral arrays through', async () => {
        const profile: NutrientProfile = {
          crop_name: 'Kale',
          calories_per_100g: 49,
          protein_g: 4.3,
          carbs_g: 8.8,
          fat_g: 0.9,
          fiber_g: 3.6,
          key_vitamins: ['K', 'A', 'C', 'B6'],
          key_minerals: ['Calcium', 'Iron', 'Magnesium'],
        };
        await food.addNutrientProfile(profile);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.key_vitamins).toHaveLength(4);
        expect(lastCall.payload.key_minerals).toHaveLength(3);
      });
    });

    describe('getNutrientProfile', () => {
      it('should call food_knowledge.get_nutrient_profile with crop name', async () => {
        client.callZome.mockResolvedValue({});

        await food.getNutrientProfile('Cherokee Purple Tomato');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_knowledge',
          fn_name: 'get_nutrient_profile',
          payload: 'Cherokee Purple Tomato',
        });
      });
    });
  });

  // ==========================================================================
  // Allergen Search (Phase 2)
  // ==========================================================================

  describe('Allergen Search (food_distribution)', () => {
    describe('searchAllergenSafe', () => {
      it('should call food_distribution.search_allergen_safe with input', async () => {
        const input: AllergenSearchInput = {
          market_hash: fakeActionHash(),
          exclude_allergens: ['peanuts', 'gluten', 'dairy'],
        };
        client.callZome.mockResolvedValue([]);

        const result = await food.searchAllergenSafe(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_distribution',
          fn_name: 'search_allergen_safe',
          payload: input,
        });
        expect(result).toEqual([]);
      });

      it('should pass empty allergen list through', async () => {
        const input: AllergenSearchInput = {
          market_hash: fakeActionHash(),
          exclude_allergens: [],
        };
        await food.searchAllergenSafe(input);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.exclude_allergens).toHaveLength(0);
      });
    });
  });

  // ==========================================================================
  // Garden Membership (Phase 2)
  // ==========================================================================

  describe('Garden Membership (food_production)', () => {
    describe('addGardenMember', () => {
      it('should call food_production.add_garden_member with AddMemberInput', async () => {
        const input: AddMemberInput = {
          plot_hash: fakeActionHash(),
          member: fakeAgentKey(),
          role: 'Volunteer',
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await food.addGardenMember(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'add_garden_member',
          payload: input,
        });
      });

      it('should accept all GardenRole variants', async () => {
        const roles: GardenRole[] = ['Steward', 'Volunteer', 'Member'];
        for (const r of roles) {
          const input: AddMemberInput = {
            plot_hash: fakeActionHash(),
            member: fakeAgentKey(),
            role: r,
          };
          await food.addGardenMember(input);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.role).toBe(r);
        }
      });
    });

    describe('getPlotMembers', () => {
      it('should call food_production.get_plot_members with plot hash', async () => {
        const plotHash = fakeActionHash();
        client.callZome.mockResolvedValue([]);

        const result = await food.getPlotMembers(plotHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'get_plot_members',
          payload: plotHash,
        });
        expect(result).toEqual([]);
      });
    });

    describe('removeGardenMember', () => {
      it('should call food_production.remove_garden_member with input', async () => {
        const input: RemoveMemberInput = {
          membership_hash: fakeActionHash(),
        };
        client.callZome.mockResolvedValue({});

        await food.removeGardenMember(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'food_production',
          fn_name: 'remove_garden_member',
          payload: input,
        });
      });
    });
  });

  // ==========================================================================
  // Seed Exchange Lifecycle
  // ==========================================================================

  describe('Seed Exchange Lifecycle', () => {
    it('should support the full seed exchange flow through mock calls', async () => {
      const stockHash = new Uint8Array(39).fill(0x01);
      const requestHash = new Uint8Array(39).fill(0x02);

      // Step 1: Offer seeds
      client.callZome.mockResolvedValueOnce(stockHash);
      const stock: SeedStock = {
        variety_hash: fakeActionHash(),
        grower: fakeAgentKey(),
        quantity_grams: 200,
        location: 'Richardson, TX',
        germination_rate_pct: 95.0,
        available_for_exchange: true,
        notes: null,
      };
      await food.offerSeeds(stock);

      // Step 2: Request seeds
      client.callZome.mockResolvedValueOnce(requestHash);
      const request: FoodSeedRequest = {
        wanted_variety: 'Cherokee Purple',
        quantity_grams: 50,
        requester: fakeAgentKey(),
        status: 'Open',
        deadline: null,
      };
      await food.requestSeeds(request);

      // Step 3: Match request to stock
      client.callZome.mockResolvedValueOnce({});
      const matchInput: MatchSeedInput = {
        request_hash: requestHash,
        stock_hash: stockHash,
      };
      await food.matchSeedRequest(matchInput);

      expect(client.callZome).toHaveBeenCalledTimes(3);

      const fnNames = client.callZome.mock.calls.map(
        (c: unknown[]) => (c[0] as { fn_name: string }).fn_name,
      );
      expect(fnNames).toEqual(['offer_seeds', 'request_seeds', 'match_seed_request']);
    });
  });

  // ==========================================================================
  // Phase 2 error propagation
  // ==========================================================================

  describe('Phase 2 error propagation', () => {
    it('should propagate errors for offerSeeds', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      const stock: SeedStock = {
        variety_hash: fakeActionHash(),
        grower: fakeAgentKey(),
        quantity_grams: 100,
        location: 'test',
        germination_rate_pct: null,
        available_for_exchange: true,
        notes: null,
      };
      await expect(food.offerSeeds(stock)).rejects.toThrow('Validation failed');
    });

    it('should propagate errors for searchAllergenSafe', async () => {
      client.callZome.mockRejectedValue(new Error('Market not found'));
      await expect(
        food.searchAllergenSafe({ market_hash: fakeActionHash(), exclude_allergens: [] }),
      ).rejects.toThrow('Market not found');
    });

    it('should propagate errors for addGardenMember', async () => {
      client.callZome.mockRejectedValue(new Error('Not authorized'));
      const input: AddMemberInput = {
        plot_hash: fakeActionHash(),
        member: fakeAgentKey(),
        role: 'Member',
      };
      await expect(food.addGardenMember(input)).rejects.toThrow('Not authorized');
    });
  });
});
