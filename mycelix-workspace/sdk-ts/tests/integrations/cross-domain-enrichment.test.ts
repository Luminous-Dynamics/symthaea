// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-Domain Enrichment Integration Tests
 *
 * Tests for Phase 2-4 enrichment features spanning food, transport, and
 * support domains. Validates cross-domain workflows, new SDK methods,
 * and bridge dispatch routing for all newly added functions.
 *
 * Types aligned to Rust integrity structs (2026-02-22).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  FoodClient,
  type SeedStock,
  type SeedQualityRating,
  type FoodSeedRequest,
  type MatchSeedInput,
  type NutrientProfile,
  type AllergenSearchInput,
  type AddMemberInput,
  type RemoveMemberInput,
  type Plot,
  type Crop,
  type YieldRecord,
  type Market,
  type Listing,
  type Order,
} from '../../src/integrations/food/index.js';

import {
  TransportClient,
  type MaintenanceRecord,
  type VehicleFeatures,
  type RideReview,
  type FindNearbyInput,
  type RedeemInput,
  type CarbonBalance,
} from '../../src/integrations/transport/index.js';

import {
  SupportClient,
  type SupportTicket,
  type KnowledgeArticle,
  type PreemptiveAlert,
  type SatisfactionSurvey,
  type HelperProfile,
  type CognitiveUpdate,
  type PrivacyPreference,
  type EscalateInput,
  type LinkArticleInput,
} from '../../src/integrations/support/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function fakeEncoding(): Uint8Array {
  return new Uint8Array(2048).fill(0x55);
}

// ============================================================================
// Cross-Domain: Food → Transport (delivery + carbon credits)
// ============================================================================

describe('Food-Transport cross-domain workflows', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;
  let transport: TransportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
    transport = new TransportClient(mockClient);
  });

  it('should complete food delivery with carbon credit redemption', async () => {
    // 1. Search for allergen-safe products
    const allergenInput: AllergenSearchInput = {
      market_hash: fakeActionHash(),
      exclude_allergens: ['peanuts', 'gluten'],
    };
    await food.searchAllergenSafe(allergenInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'food_distribution',
        fn_name: 'search_allergen_safe',
        payload: allergenInput,
      }),
    );

    // 2. Find nearby rides for delivery
    mockClient.callZome.mockClear();
    const nearbyInput: FindNearbyInput = { origin_lat: 32.95, origin_lon: -96.75, radius_km: 10 };
    await transport.findNearbyRides(nearbyInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'transport_sharing',
        fn_name: 'find_nearby_rides',
        payload: nearbyInput,
      }),
    );

    // 3. Redeem carbon credits after delivery
    mockClient.callZome.mockClear();
    const redeemInput: RedeemInput = {
      credits_redeemed: 5.0,
      redeemed_for: 'Food delivery offset',
    };
    await transport.redeemCredits(redeemInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'transport_impact',
        fn_name: 'redeem_credits',
        payload: redeemInput,
      }),
    );

    // 4. Check remaining balance
    mockClient.callZome.mockClear();
    mockClient.callZome.mockResolvedValueOnce({
      total_earned: 20.0,
      total_redeemed: 5.0,
      balance: 15.0,
    } satisfies CarbonBalance);
    const balance = await transport.getAgentCarbonBalance(fakeAgentKey());
    expect(balance).toEqual(expect.objectContaining({ balance: 15.0 }));
  });

  it('should track nutrient profiles for delivered produce', async () => {
    const profile: NutrientProfile = {
      crop_name: 'Carrot',
      calories_per_100g: 41,
      protein_g: 0.9,
      fat_g: 0.2,
      carbs_g: 9.6,
      fiber_g: 2.8,
      key_vitamins: ['A', 'K', 'B6'],
      key_minerals: ['potassium', 'biotin'],
    };
    await food.addNutrientProfile(profile);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'food_knowledge',
        fn_name: 'add_nutrient_profile',
        payload: profile,
      }),
    );
  });
});

// ============================================================================
// Cross-Domain: Support → Food (quality ticket → knowledge)
// ============================================================================

describe('Support-Food cross-domain workflows', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;
  let support: SupportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
    support = new SupportClient(mockClient);
  });

  it('should create support ticket for food quality issue and link article', async () => {
    const ticket: SupportTicket = {
      title: 'Produce quality issue at market',
      description: 'Tomatoes received with signs of blight',
      category: 'General',
      priority: 'Medium',
      status: 'Open',
      requester: fakeAgentKey(),
      assignee: null,
      autonomy_level: 'Advisory',
      system_info: null,
      is_preemptive: false,
      prediction_confidence: null,
      created_at: Date.now() * 1000,
      updated_at: Date.now() * 1000,
    };
    await support.createTicket(ticket);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'create_ticket' }),
    );

    mockClient.callZome.mockClear();
    const article: KnowledgeArticle = {
      title: 'Preventing Tomato Blight',
      content: 'Early blight can be prevented by proper spacing...',
      category: 'General',
      tags: ['tomato', 'blight', 'prevention'],
      author: fakeAgentKey(),
      source: 'Community',
      difficulty_level: 'Beginner',
      upvotes: 0,
      verified: false,
      deprecated: false,
      deprecation_reason: null,
      version: 1,
    };
    await support.createArticle(article);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'create_article' }),
    );

    mockClient.callZome.mockClear();
    const linkInput: LinkArticleInput = {
      article_hash: fakeActionHash(),
      ticket_hash: fakeActionHash(),
      link_reason: 'SuggestedFAQ',
    };
    await support.linkArticleToTicket(linkInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'link_article_to_ticket' }),
    );
  });
});

// ============================================================================
// Cross-Domain: Transport → Support (ride issue → support ticket)
// ============================================================================

describe('Transport-Support cross-domain workflows', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let transport: TransportClient;
  let support: SupportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    transport = new TransportClient(mockClient);
    support = new SupportClient(mockClient);
  });

  it('should handle ride issue → review → escalation → maintenance', async () => {
    // 1. Review a problematic ride
    const review: RideReview = {
      match_hash: fakeActionHash(),
      reviewer: fakeAgentKey(),
      role: 'Passenger',
      rating: 1,
      comment: 'Vehicle had mechanical issues',
      safety_concern: true,
      created_at: Date.now() * 1000,
    };
    await transport.reviewRide(review);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        fn_name: 'review_ride',
        payload: review,
      }),
    );

    // 2. Escalate the resulting support ticket
    mockClient.callZome.mockClear();
    const escalation: EscalateInput = {
      ticket_hash: fakeActionHash(),
      from_level: 'Tier1',
      to_level: 'Tier2',
      reason: 'Safety-critical vehicle issue',
    };
    await support.escalateTicket(escalation);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'escalate_ticket' }),
    );

    // 3. Log maintenance
    mockClient.callZome.mockClear();
    const maint: MaintenanceRecord = {
      id: 'maint-001',
      vehicle_hash: fakeActionHash(),
      maintenance_type: 'Repair',
      description: 'Brake pad replacement',
      cost: 150.0,
      completed_at: Date.now() * 1000,
      next_due: (Date.now() + 90 * 86400000) * 1000,
      mechanic_notes: 'Front pads worn to 2mm, replaced with ceramic',
    };
    await transport.logMaintenance(maint);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        fn_name: 'log_maintenance',
        payload: maint,
      }),
    );
  });

  it('should create preemptive alert from vehicle diagnostics', async () => {
    await support.runDiagnostic({
      ticket_hash: null,
      diagnostic_type: 'ServiceStatus',
      findings: JSON.stringify({ vehicles_overdue: 3 }),
      severity: 'Warning',
      recommendations: ['Schedule maintenance for 3 vehicles'],
      agent: fakeAgentKey(),
      scrubbed: false,
      created_at: Date.now() * 1000,
    });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'run_diagnostic' }),
    );

    mockClient.callZome.mockClear();
    const alert: PreemptiveAlert = {
      predicted_failure: 'Vehicle fleet maintenance overdue',
      expected_time_to_failure: '7 days',
      free_energy: 0.85,
      recommended_action: 'Schedule fleet-wide inspection',
      auto_generated_ticket: null,
      created_at: Date.now() * 1000,
    };
    await support.createPreemptiveAlert(alert);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'create_preemptive_alert' }),
    );

    mockClient.callZome.mockClear();
    await transport.getVehiclesNeedingMaintenance(Date.now());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_vehicles_needing_maintenance' }),
    );
  });
});

// ============================================================================
// Seed Exchange lifecycle
// ============================================================================

describe('Seed Exchange lifecycle', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
  });

  it('should offer, request, and match seeds', async () => {
    const stock: SeedStock = {
      variety_hash: fakeActionHash(),
      grower: fakeAgentKey(),
      quantity_grams: 50.0,
      location: 'Richardson, TX',
      germination_rate_pct: 92.0,
      available_for_exchange: true,
      notes: 'Cherokee Purple heirloom, 2025 harvest',
    };
    await food.offerSeeds(stock);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'offer_seeds', payload: stock }),
    );

    mockClient.callZome.mockClear();
    const request: FoodSeedRequest = {
      wanted_variety: 'Cherokee Purple',
      quantity_grams: 25.0,
      requester: fakeAgentKey(),
      status: 'Open',
      deadline: null,
    };
    await food.requestSeeds(request);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'request_seeds', payload: request }),
    );

    mockClient.callZome.mockClear();
    await food.getAvailableSeeds(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_available_seeds', payload: fakeActionHash() }),
    );

    mockClient.callZome.mockClear();
    await food.getOpenSeedRequests();
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_open_seed_requests' }),
    );

    mockClient.callZome.mockClear();
    const matchInput: MatchSeedInput = {
      request_hash: fakeActionHash(),
      stock_hash: fakeActionHash(),
    };
    await food.matchSeedRequest(matchInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'match_seed_request', payload: matchInput }),
    );
  });
});

// ============================================================================
// Garden Membership lifecycle
// ============================================================================

describe('Garden Membership lifecycle', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
  });

  it('should add, list, and remove garden members', async () => {
    const membership: AddMemberInput = {
      plot_hash: fakeActionHash(),
      member: fakeAgentKey(),
      role: 'Volunteer',
    };
    await food.addGardenMember(membership);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'add_garden_member', payload: membership }),
    );

    mockClient.callZome.mockClear();
    await food.getPlotMembers(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_plot_members' }),
    );

    mockClient.callZome.mockClear();
    const removeInput: RemoveMemberInput = { membership_hash: fakeActionHash() };
    await food.removeGardenMember(removeInput);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'remove_garden_member', payload: removeInput }),
    );
  });
});

// ============================================================================
// Vehicle features + accessibility
// ============================================================================

describe('Vehicle features and accessibility', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let transport: TransportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    transport = new TransportClient(mockClient);
  });

  it('should set features and query accessible vehicles', async () => {
    const features: VehicleFeatures = {
      vehicle_hash: fakeActionHash(),
      wheelchair_accessible: true,
      child_seat: false,
      pet_friendly: false,
      air_conditioning: true,
      bike_rack: true,
      luggage_capacity_liters: 200,
    };
    await transport.setVehicleFeatures(features);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'set_vehicle_features', payload: features }),
    );

    mockClient.callZome.mockClear();
    await transport.getAccessibleVehicles();
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_accessible_vehicles' }),
    );
  });

  it('should track maintenance history and reviews', async () => {
    await transport.getVehicleMaintenance(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_vehicle_maintenance' }),
    );

    mockClient.callZome.mockClear();
    await transport.getRideReviews(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_ride_reviews' }),
    );

    mockClient.callZome.mockClear();
    await transport.getDriverRating(fakeAgentKey());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_driver_rating' }),
    );

    mockClient.callZome.mockClear();
    await transport.getMyRedemptions();
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_my_redemptions' }),
    );
  });
});

// ============================================================================
// Support helper + cognitive update lifecycle
// ============================================================================

describe('Support helper and cognitive lifecycle', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let support: SupportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    support = new SupportClient(mockClient);
  });

  it('should register helper and manage availability', async () => {
    const helper: HelperProfile = {
      agent: fakeAgentKey(),
      expertise_categories: ['Network', 'Holochain'],
      max_concurrent: 3,
      difficulty_preference: 'Advanced',
      available: true,
      created_at: Date.now() * 1000,
    };
    await support.registerHelper(helper);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'register_helper', payload: helper }),
    );

    mockClient.callZome.mockClear();
    await support.updateAvailability({ helper_hash: fakeActionHash(), available: false });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'update_availability' }),
    );

    mockClient.callZome.mockClear();
    await support.getAvailableHelpers('Network');
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_available_helpers', payload: 'Network' }),
    );
  });

  it('should publish and absorb cognitive updates', async () => {
    const update: CognitiveUpdate = {
      category: 'Holochain',
      encoding: fakeEncoding(),
      phi: 0.75,
      resolution_pattern: 'Restart conductor when DHT sync stalls',
      source_agent: fakeAgentKey(),
      created_at: Date.now() * 1000,
    };
    await support.publishCognitiveUpdate(update);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'publish_cognitive_update', payload: update }),
    );

    mockClient.callZome.mockClear();
    await support.absorbCognitiveUpdate(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'absorb_cognitive_update' }),
    );
  });

  it('should manage privacy preferences', async () => {
    const pref: PrivacyPreference = {
      agent: fakeAgentKey(),
      sharing_tier: 'Anonymized',
      allowed_categories: ['Network', 'Hardware'],
      share_system_info: true,
      share_resolution_patterns: true,
      share_cognitive_updates: false,
      updated_at: Date.now() * 1000,
    };
    await support.setPrivacyPreference(pref);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'set_privacy_preference' }),
    );

    mockClient.callZome.mockClear();
    await support.getShareableDiagnostics(fakeAgentKey());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_shareable_diagnostics' }),
    );
  });

  it('should handle satisfaction, escalation history, and undo', async () => {
    const survey: SatisfactionSurvey = {
      ticket_hash: fakeActionHash(),
      respondent: fakeAgentKey(),
      rating: 4,
      comment: 'Quick resolution',
      would_recommend: true,
      submitted_at: Date.now() * 1000,
    };
    await support.submitSatisfaction(survey);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'submit_satisfaction' }),
    );

    mockClient.callZome.mockClear();
    await support.getTicketSatisfaction(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_ticket_satisfaction' }),
    );

    mockClient.callZome.mockClear();
    await support.getEscalationHistory(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'get_escalation_history' }),
    );

    mockClient.callZome.mockClear();
    await support.createUndo({
      original_action_hash: fakeActionHash(),
      reason: 'Accidental cache clear',
      rollback_result: 'Cache restored',
      created_at: Date.now() * 1000,
    });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'create_undo' }),
    );

    mockClient.callZome.mockClear();
    await support.listPreemptiveAlerts();
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'list_preemptive_alerts' }),
    );

    mockClient.callZome.mockClear();
    await support.promoteAlertToTicket({
      alert_hash: fakeActionHash(),
      ticket: {
        title: 'Promoted alert',
        description: 'Auto-promoted from preemptive alert',
        category: 'General',
        priority: 'High',
        status: 'Open',
        requester: fakeAgentKey(),
        assignee: null,
        autonomy_level: 'Advisory',
        system_info: null,
        is_preemptive: true,
        prediction_confidence: null,
        created_at: Date.now() * 1000,
        updated_at: Date.now() * 1000,
      },
    });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'promote_alert_to_ticket' }),
    );
  });
});

// ============================================================================
// Seed Quality Rating lifecycle
// ============================================================================

describe('Seed Quality Rating lifecycle', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
  });

  it('should rate a seed exchange', async () => {
    const rating: SeedQualityRating = {
      exchange_hash: fakeActionHash(),
      grower: fakeAgentKey(),
      quality_score: 4,
      germination_success: true,
      notes: 'Excellent germination, strong seedlings',
    };
    await food.rateSeedExchange(rating);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'food_knowledge',
        fn_name: 'rate_seed_exchange',
        payload: rating,
      }),
    );
  });

  it('should get ratings for a specific exchange', async () => {
    mockClient.callZome.mockResolvedValueOnce([
      { quality_score: 5, germination_success: true },
      { quality_score: 3, germination_success: false },
    ]);
    const ratings = await food.getExchangeRatings(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'food_knowledge',
        fn_name: 'get_exchange_ratings',
        payload: fakeActionHash(),
      }),
    );
    expect(ratings).toHaveLength(2);
  });

  it('should get all ratings for a grower', async () => {
    mockClient.callZome.mockResolvedValueOnce([]);
    const ratings = await food.getGrowerRatings(fakeAgentKey());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'food_knowledge',
        fn_name: 'get_grower_ratings',
        payload: fakeAgentKey(),
      }),
    );
    expect(ratings).toHaveLength(0);
  });
});

// ============================================================================
// Full lifecycle: plot → crop → harvest → market → order → delivery → carbon
// ============================================================================

describe('Full Food-Transport lifecycle', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let food: FoodClient;
  let transport: TransportClient;

  beforeEach(() => {
    mockClient = createMockClient();
    food = new FoodClient(mockClient);
    transport = new TransportClient(mockClient);
  });

  it('should complete farm-to-table with carbon offset', async () => {
    // 1. Register a plot
    const plot: Plot = {
      id: 'plot-001',
      name: 'Community Garden A',
      area_sqm: 500,
      soil_type: 'Loam',
      location_lat: 32.95,
      location_lon: -96.75,
      steward: fakeAgentKey(),
      status: 'Active',
    };
    await food.registerPlot(plot);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'register_plot' }),
    );

    // 2. Plant a crop
    mockClient.callZome.mockClear();
    const crop: Crop = {
      plot_hash: fakeActionHash(),
      name: 'Tomato',
      variety: 'Cherokee Purple',
      planted_at: Date.now() * 1000,
      expected_harvest: (Date.now() + 75 * 86400000) * 1000,
      status: 'Planted',
      allergen_flags: [],
      organic_certified: true,
    };
    await food.plantCrop(crop);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'plant_crop' }),
    );

    // 3. Record harvest
    mockClient.callZome.mockClear();
    const harvest: YieldRecord = {
      crop_hash: fakeActionHash(),
      quantity_kg: 45.5,
      quality_grade: 'Premium',
      harvested_at: Date.now() * 1000,
      notes: 'Excellent yield, minimal pest damage',
    };
    await food.recordHarvest(harvest);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'record_harvest' }),
    );

    // 4. List at market
    mockClient.callZome.mockClear();
    const listing: Listing = {
      market_hash: fakeActionHash(),
      producer: fakeAgentKey(),
      product_name: 'Cherokee Purple Tomatoes',
      quantity_kg: 40,
      price_per_kg: 4.50,
      available_from: Date.now() * 1000,
      status: 'Available',
      allergen_flags: [],
      organic: true,
      cultural_markers: ['heirloom'],
    };
    await food.listProduct(listing);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'list_product' }),
    );

    // 5. Place order
    mockClient.callZome.mockClear();
    const order: Order = {
      listing_hash: fakeActionHash(),
      buyer: fakeAgentKey(),
      quantity_kg: 5,
      status: 'Pending',
    };
    await food.placeOrder(order);
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'place_order' }),
    );

    // 6. Find nearby ride for delivery
    mockClient.callZome.mockClear();
    await transport.findNearbyRides({ origin_lat: 32.95, origin_lon: -96.75, radius_km: 5 });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'find_nearby_rides' }),
    );

    // 7. Fulfill order
    mockClient.callZome.mockClear();
    await food.fulfillOrder(fakeActionHash());
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'fulfill_order' }),
    );

    // 8. Redeem carbon credits from the delivery trip
    mockClient.callZome.mockClear();
    await transport.redeemCredits({ credits_redeemed: 2.5, redeemed_for: 'Local produce delivery' });
    expect(mockClient.callZome).toHaveBeenCalledWith(
      expect.objectContaining({ fn_name: 'redeem_credits' }),
    );
  });
});
