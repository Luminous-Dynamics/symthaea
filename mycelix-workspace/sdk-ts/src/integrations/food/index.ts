/**
 * @mycelix/sdk Food Sovereignty Integration
 *
 * Domain-specific client for the food sovereignty zomes within the
 * mycelix-commons cluster DNA. Covers production (plots, crops, harvests),
 * distribution (markets, listings, orders), preservation (batches, methods,
 * storage), and knowledge (seeds, practices, recipes).
 *
 * All calls are dispatched through the commons_bridge coordinator zome.
 *
 * @packageDocumentation
 * @module integrations/food
 */

// ============================================================================
// Types — Food Production
// ============================================================================

export type PlotStatus = 'Active' | 'Fallow' | 'Preparing' | 'Retired';

export interface Plot {
  id: string;
  name: string;
  area_sqm: number;
  soil_type: string;
  location_lat: number;
  location_lon: number;
  steward: Uint8Array;
  status: PlotStatus;
}

export type CropStatus = 'Planned' | 'Planted' | 'Growing' | 'Ready' | 'Harvested' | 'Failed';

export interface Crop {
  plot_hash: Uint8Array;
  name: string;
  variety: string;
  planted_at: number;
  expected_harvest: number;
  status: CropStatus;
  allergen_flags: string[];
  organic_certified: boolean;
}

export type QualityGrade = 'Premium' | 'Standard' | 'Processing' | 'Compost';

export interface YieldRecord {
  crop_hash: Uint8Array;
  quantity_kg: number;
  quality_grade: QualityGrade;
  harvested_at: number;
  notes: string | null;
}

export interface SeasonPlan {
  plot_hash: Uint8Array;
  year: number;
  season: string;
  planned_crops: string[];
  rotation_notes: string;
}

// ============================================================================
// Types — Food Distribution
// ============================================================================

export type MarketType = 'Farmers' | 'CSA' | 'FoodBank' | 'CoOp';
export type ListingStatus = 'Available' | 'Reserved' | 'Sold' | 'Expired';
export type OrderStatus = 'Pending' | 'Confirmed' | 'Fulfilled' | 'Cancelled';

export interface Market {
  id: string;
  name: string;
  location_lat: number;
  location_lon: number;
  market_type: MarketType;
  steward: Uint8Array;
  schedule: string;
}

export interface Listing {
  market_hash: Uint8Array;
  producer: Uint8Array;
  product_name: string;
  quantity_kg: number;
  price_per_kg: number;
  available_from: number;
  status: ListingStatus;
  allergen_flags: string[];
  organic: boolean;
  cultural_markers: string[];
}

export interface Order {
  listing_hash: Uint8Array;
  buyer: Uint8Array;
  quantity_kg: number;
  status: OrderStatus;
}

// ============================================================================
// Types — Food Preservation
// ============================================================================

export type PreservationStatus = 'InProgress' | 'Complete' | 'Failed';
export type StorageType = 'RootCellar' | 'Freezer' | 'Dehydrator' | 'Fermenter' | 'Pantry';
export type SkillLevel = 'Beginner' | 'Intermediate' | 'Advanced';

export interface PreservationBatch {
  id: string;
  source_crop_hash: Uint8Array | null;
  method: string;
  quantity_kg: number;
  started_at: number;
  expected_ready: number;
  status: PreservationStatus;
  notes: string;
}

export interface PreservationMethod {
  name: string;
  description: string;
  shelf_life_days: number;
  equipment_needed: string[];
  skill_level: SkillLevel;
}

export interface StorageUnit {
  id: string;
  name: string;
  capacity_kg: number;
  storage_type: StorageType;
  steward: Uint8Array;
}

// ============================================================================
// Types — Food Knowledge
// ============================================================================

export type PracticeCategory = 'Planting' | 'Harvest' | 'Soil' | 'Pest' | 'Water';

export interface SeedVariety {
  name: string;
  species: string;
  origin: string | null;
  days_to_maturity: number;
  companion_plants: string[];
  avoid_plants: string[];
  seed_saving_notes: string | null;
}

export interface TraditionalPractice {
  name: string;
  description: string;
  region: string | null;
  season: string | null;
  category: PracticeCategory;
}

export interface Recipe {
  name: string;
  ingredients: string[];
  instructions: string;
  servings: number;
  prep_time_min: number;
  tags: string[];
  source_attribution: string | null;
}

// ============================================================================
// Types — Seed Quality Rating (Phase 2)
// ============================================================================

export interface SeedQualityRating {
  exchange_hash: Uint8Array;
  grower: Uint8Array;
  quality_score: number;
  germination_success: boolean;
  notes: string | null;
}

// ============================================================================
// Types — Seed Exchange (Phase 2)
// ============================================================================

export interface SeedStock {
  variety_hash: Uint8Array;
  grower: Uint8Array;
  quantity_grams: number;
  location: string;
  germination_rate_pct: number | null;
  available_for_exchange: boolean;
  notes: string | null;
}

export type SeedRequestStatus = 'Open' | 'Matched' | 'Fulfilled';

export interface FoodSeedRequest {
  wanted_variety: string;
  quantity_grams: number;
  requester: Uint8Array;
  status: SeedRequestStatus;
  deadline: number | null;
}

export interface MatchSeedInput {
  request_hash: Uint8Array;
  stock_hash: Uint8Array;
}

// ============================================================================
// Types — Nutrient Tracking (Phase 2)
// ============================================================================

export interface NutrientProfile {
  crop_name: string;
  calories_per_100g: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g: number;
  key_vitamins: string[];
  key_minerals: string[];
}

// ============================================================================
// Types — Allergen Search (Phase 2)
// ============================================================================

export interface AllergenSearchInput {
  market_hash: Uint8Array;
  exclude_allergens: string[];
}

// ============================================================================
// Types — Garden Membership (Phase 2)
// ============================================================================

export type GardenRole = 'Steward' | 'Volunteer' | 'Member';

export interface GardenMembership {
  plot_hash: Uint8Array;
  member: Uint8Array;
  role: GardenRole;
  joined_at: number;
}

export interface AddMemberInput {
  plot_hash: Uint8Array;
  member: Uint8Array;
  role: GardenRole;
}

export interface RemoveMemberInput {
  membership_hash: Uint8Array;
}

// ============================================================================
// Types — Update Inputs (Phase 2)
// ============================================================================

export interface UpdateSeedVarietyInput {
  original_action_hash: Uint8Array;
  updated_entry: SeedVariety;
}

export interface UpdateTraditionalPracticeInput {
  original_action_hash: Uint8Array;
  updated_entry: TraditionalPractice;
}

export interface UpdateRecipeInput {
  original_action_hash: Uint8Array;
  updated_entry: Recipe;
}

// ============================================================================
// Holochain ZomeCallable interface (minimal)
// ============================================================================

interface ZomeCallable {
  callZome<T>(params: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

// ============================================================================
// Constants
// ============================================================================

const COMMONS_ROLE = 'commons_land';

export const FOOD_ZOMES = [
  'food_production',
  'food_distribution',
  'food_preservation',
  'food_knowledge',
] as const;

// ============================================================================
// Food Client
// ============================================================================

/**
 * Client for the food sovereignty domain within the commons cluster.
 *
 * Provides typed access to all 4 food zomes: production, distribution,
 * preservation, and knowledge.
 */
export class FoodClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Production ---

  async registerPlot(plot: Plot): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'register_plot',
      payload: plot,
    });
  }

  async getPlot(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_plot',
      payload: actionHash,
    });
  }

  async getAllPlots(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_all_plots',
      payload: null,
    });
  }

  async plantCrop(crop: Crop): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'plant_crop',
      payload: crop,
    });
  }

  async recordHarvest(yieldRecord: YieldRecord): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'record_harvest',
      payload: yieldRecord,
    });
  }

  async getPlotCrops(plotHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_plot_crops',
      payload: plotHash,
    });
  }

  async getCropYields(cropHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_crop_yields',
      payload: cropHash,
    });
  }

  async createSeasonPlan(plan: SeasonPlan): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'create_season_plan',
      payload: plan,
    });
  }

  async getSeasonPlans(plotHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_season_plans',
      payload: plotHash,
    });
  }

  // --- Distribution ---

  async createMarket(market: Market): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'create_market',
      payload: market,
    });
  }

  async getAllMarkets(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'get_all_markets',
      payload: null,
    });
  }

  async listProduct(listing: Listing): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'list_product',
      payload: listing,
    });
  }

  async getMarketListings(marketHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'get_market_listings',
      payload: marketHash,
    });
  }

  async getProducerListings(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'get_producer_listings',
      payload: null,
    });
  }

  async placeOrder(order: Order): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'place_order',
      payload: order,
    });
  }

  async fulfillOrder(orderHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'fulfill_order',
      payload: { order_hash: orderHash, new_status: 'Fulfilled' },
    });
  }

  async cancelOrder(orderHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'cancel_order',
      payload: orderHash,
    });
  }

  async getMyOrders(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'get_my_orders',
      payload: null,
    });
  }

  // --- Preservation ---

  async startBatch(batch: PreservationBatch): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'start_batch',
      payload: batch,
    });
  }

  async completeBatch(batchHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'complete_batch',
      payload: batchHash,
    });
  }

  async getBatch(batchHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'get_batch',
      payload: batchHash,
    });
  }

  async registerMethod(method: PreservationMethod): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'register_method',
      payload: method,
    });
  }

  async getAllMethods(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'get_all_methods',
      payload: null,
    });
  }

  async registerStorage(unit: StorageUnit): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'register_storage',
      payload: unit,
    });
  }

  async getStorageInventory(storageHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'get_storage_inventory',
      payload: storageHash,
    });
  }

  async getAgentBatches(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_preservation',
      fn_name: 'get_agent_batches',
      payload: null,
    });
  }

  // --- Knowledge ---

  async catalogSeed(seed: SeedVariety): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'catalog_seed',
      payload: seed,
    });
  }

  async getSeed(seedHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_seed',
      payload: seedHash,
    });
  }

  async getSeedsBySpecies(species: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_seeds_by_species',
      payload: species,
    });
  }

  async sharePractice(practice: TraditionalPractice): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'share_practice',
      payload: practice,
    });
  }

  async getPracticesByCategory(category: PracticeCategory): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_practices_by_category',
      payload: category,
    });
  }

  async shareRecipe(recipe: Recipe): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'share_recipe',
      payload: recipe,
    });
  }

  async getRecipesByTag(tag: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_recipes_by_tag',
      payload: tag,
    });
  }

  async searchKnowledge(query: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'search_knowledge',
      payload: query,
    });
  }

  // --- Seed Exchange (Phase 2) ---

  async offerSeeds(stock: SeedStock): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'offer_seeds',
      payload: stock,
    });
  }

  async requestSeeds(request: FoodSeedRequest): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'request_seeds',
      payload: request,
    });
  }

  async getAvailableSeeds(varietyHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_available_seeds',
      payload: varietyHash,
    });
  }

  async getOpenSeedRequests(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_open_seed_requests',
      payload: null,
    });
  }

  async matchSeedRequest(input: MatchSeedInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'match_seed_request',
      payload: input,
    });
  }

  // --- Nutrient Tracking (Phase 2) ---

  async addNutrientProfile(profile: NutrientProfile): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'add_nutrient_profile',
      payload: profile,
    });
  }

  async getNutrientProfile(cropName: string): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_nutrient_profile',
      payload: cropName,
    });
  }

  // --- Allergen Search (Phase 2) ---

  async searchAllergenSafe(input: AllergenSearchInput): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_distribution',
      fn_name: 'search_allergen_safe',
      payload: input,
    });
  }

  // --- Garden Membership (Phase 2) ---

  async addGardenMember(input: AddMemberInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'add_garden_member',
      payload: input,
    });
  }

  async getPlotMembers(plotHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'get_plot_members',
      payload: plotHash,
    });
  }

  async removeGardenMember(input: RemoveMemberInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_production',
      fn_name: 'remove_garden_member',
      payload: input,
    });
  }

  // --- Seed Quality Ratings (Phase 2) ---

  async rateSeedExchange(rating: SeedQualityRating): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'rate_seed_exchange',
      payload: rating,
    });
  }

  async getExchangeRatings(exchangeHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_exchange_ratings',
      payload: exchangeHash,
    });
  }

  async getGrowerRatings(grower: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'get_grower_ratings',
      payload: grower,
    });
  }

  // --- Update Methods (Phase 2) ---

  async updateSeedVariety(input: UpdateSeedVarietyInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'update_seed_variety',
      payload: input,
    });
  }

  async updateTraditionalPractice(input: UpdateTraditionalPracticeInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'update_traditional_practice',
      payload: input,
    });
  }

  async updateRecipe(input: UpdateRecipeInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'food_knowledge',
      fn_name: 'update_recipe',
      payload: input,
    });
  }
}

// ============================================================================
// Factory
// ============================================================================

/** Create a FoodClient from an AppWebsocket or compatible client */
export function createFoodClient(client: ZomeCallable): FoodClient {
  return new FoodClient(client);
}
