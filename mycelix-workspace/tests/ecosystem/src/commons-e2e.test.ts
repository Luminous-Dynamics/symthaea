// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Commons Cluster E2E Tests
 *
 * Tests cross-domain workflows within the commons cluster via Tryorama.
 * Requires: mycelix-commons.happ bundle built via `just build-commons`
 *
 * The commons hApp has TWO roles (from happ.yaml):
 *   - commons_land: property, housing, water, food
 *   - commons_care: care, mutualaid, transport, support, space, bridge
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario, runScenario, dhtSync, type PlayerApp } from '@holochain/tryorama';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Commons uses the split sub-cluster DNAs in production,
// but the monolithic bundle works for cross-domain E2E.
const COMMONS_HAPP_PATH = join(__dirname, '../../../../mycelix-commons/mycelix-commons.happ');

function skipIfNoBuild() {
  if (!existsSync(COMMONS_HAPP_PATH)) {
    console.warn(`Skipping: ${COMMONS_HAPP_PATH} not found. Run 'just build-commons' first.`);
    return true;
  }
  return false;
}

// Helper to call a commons zome on the correct role
async function callZome<T>(
  player: PlayerApp,
  zome: string,
  fn_name: string,
  payload: unknown = null,
  role: string = 'commons_land',
): Promise<T> {
  return await player.appWs.callZome({
    role_name: role,
    zome_name: zome,
    fn_name,
    payload,
  }) as T;
}

// Zomes in commons_land: property_registry, housing_units, water_steward, water_purity,
//                         food_production, food_distribution
// Zomes in commons_care: care_circles, mutualaid_needs, transport_routes, transport_sharing,
//                         commons_bridge, support_*, space, etc.

describe('Commons Cluster E2E', () => {
  if (skipIfNoBuild()) return;

  describe('Property Registry + Housing Flow', () => {
    it('should register a property and register a building', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // Register a property in the property registry
        // Fn: register_property(RegisterPropertyInput) -> Record
        const property = await callZome(alice, 'property_registry', 'register_property', {
          property_type: 'Land',
          title: 'Community Land Trust Parcel A',
          description: 'Urban lot for cooperative housing',
          owner_did: 'did:key:z6MkTest001',
          co_owners: [],
          geolocation: { latitude: 32.9483, longitude: -96.7299 },
          address: null,
          metadata: {
            appraised_value: 250000.0,
            currency: 'USD',
            legal_description: null,
            parcel_number: null,
            attachments: [],
          },
        }, 'commons_land');
        expect(property).toBeDefined();

        // Register a building in housing_units
        // Fn: register_building(Building) -> Record
        const building = await callZome(alice, 'housing_units', 'register_building', {
          id: 'bldg-001',
          name: 'Oak Terrace',
          address: '123 Main St',
          location_lat: 32.9483,
          location_lon: -96.7299,
          total_units: 24,
          year_built: 2020,
          building_type: 'Apartment',
          cooperative_hash: null,
        }, 'commons_land');
        expect(building).toBeDefined();
      });
    });
  });

  describe('Water Stewardship Flow', () => {
    it('should define a watershed and submit a quality reading', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // Define a watershed
        // Fn: define_watershed(Watershed) -> Record
        const watershed = await callZome(alice, 'water_steward', 'define_watershed', {
          id: 'ws-001',
          name: 'Community Watershed',
          huc_code: null,
          boundary: [[32.95, -96.73], [32.96, -96.72], [32.94, -96.71]],
          area_sq_km: 10.0,
          stewardship_type: 'Commons',
          governing_body: null,
          primary_source_type: 'Well',
        }, 'commons_land');
        expect(watershed).toBeDefined();

        // Submit a water quality reading
        // Fn: submit_reading(QualityReading) -> Record
        // NOTE: source_hash and sampler need real hashes from the conductor;
        // in an E2E test they'd come from created records. Using watershed
        // action hash and alice's agent key.
        const watershedActionHash = (watershed as any).signed_action?.hashed?.hash;
        const reading = await callZome(alice, 'water_purity', 'submit_reading', {
          source_hash: watershedActionHash,
          sampler: alice.agentPubKey,
          timestamp: Date.now() * 1000, // microseconds
          temperature_celsius: 18.5,
          turbidity_ntu: 0.5,
          ph: 7.2,
          tds_ppm: 250.0,
          dissolved_oxygen_mg_l: 8.1,
          nitrates_mg_l: 5.0,
          arsenic_ug_l: null,
          lead_ug_l: null,
          total_coliform_cfu: null,
          e_coli_cfu: null,
          chlorine_mg_l: null,
          potability_score: 0.95,
          meets_who_standards: true,
          meets_epa_standards: true,
        }, 'commons_land');
        expect(reading).toBeDefined();
      });
    });
  });

  describe('Care Circle + Mutual Aid Flow', () => {
    it('should create care circle and post mutual aid need', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // Alice creates a care circle
        // Fn: create_circle(CareCircle) -> Record
        const circle = await callZome(alice, 'care_circles', 'create_circle', {
          name: 'Neighborhood Care Circle',
          description: 'Mutual support for our block',
          location: 'Downtown Richardson',
          max_members: 20,
          created_by: alice.agentPubKey,
          circle_type: 'Neighborhood',
          active: true,
          created_at: Date.now() * 1000, // Timestamp in microseconds
        }, 'commons_care');
        expect(circle).toBeDefined();

        // Bob posts a mutual aid need
        // Fn: create_need(CreateNeedInput) -> Record
        const need = await callZome(bob, 'mutualaid_needs', 'create_need', {
          title: 'Grocery delivery needed',
          description: 'Recovering from surgery, need help with weekly groceries',
          category: 'Food',
          urgency: 'Medium',
          emergency: false,
          quantity: null,
          location: 'Remote',
          needed_by: null,
          reciprocity_offers: [],
        }, 'commons_care');
        expect(need).toBeDefined();

        // Sync DHT between alice and bob
        await dhtSync(
          [alice, bob],
          alice.cells[0].cell_id[0],
        );
      });
    });
  });

  describe('Food Production + Distribution Flow', () => {
    it('should register a plot and create a market', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // Register a food production plot
        // Fn: register_plot(Plot) -> Record
        const plot = await callZome(alice, 'food_production', 'register_plot', {
          id: 'plot-001',
          name: 'Sunrise Community Garden',
          area_sqm: 250.5,
          soil_type: 'Loam',
          plot_type: 'Garden',
          location_lat: 32.95,
          location_lon: -96.73,
          steward: alice.agentPubKey,
          status: 'Active',
        }, 'commons_land');
        expect(plot).toBeDefined();

        // Create a distribution market
        // Fn: create_market(Market) -> Record
        const market = await callZome(alice, 'food_distribution', 'create_market', {
          id: 'mkt-001',
          name: 'Saturday Farm Share',
          location_lat: 32.95,
          location_lon: -96.73,
          market_type: 'Farmers',
          steward: alice.agentPubKey,
          schedule: 'Saturdays 8am-1pm',
        }, 'commons_land');
        expect(market).toBeDefined();
      });
    });
  });

  describe('Transport Flow', () => {
    it('should create a route and post a ride offer', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // First register a vehicle (needed for ride offer)
        // Fn: register_vehicle(Vehicle) -> Record
        const vehicle = await callZome(alice, 'transport_routes', 'register_vehicle', {
          id: 'v-001',
          owner: alice.agentPubKey,
          vehicle_type: 'Car',
          capacity_kg: 500.0,
          capacity_passengers: 4,
          status: 'Available',
        }, 'commons_care');
        expect(vehicle).toBeDefined();

        // Create a transport route
        // Fn: create_route(Route) -> Record
        const route = await callZome(alice, 'transport_routes', 'create_route', {
          id: 'rt-001',
          name: 'Downtown Shuttle',
          waypoints: [
            { lat: 32.95, lon: -96.73, label: 'Community Center' },
            { lat: 32.96, lon: -96.74, label: 'Main St' },
            { lat: 32.97, lon: -96.75, label: 'Downtown Station' },
          ],
          distance_km: 5.2,
          estimated_minutes: 20,
          mode: 'Driving',
        }, 'commons_care');
        expect(route).toBeDefined();

        // Offer ride sharing
        // Fn: post_ride_offer(RideOffer) -> Record
        const vehicleActionHash = (vehicle as any).signed_action?.hashed?.hash;
        const routeActionHash = (route as any).signed_action?.hashed?.hash;
        const offer = await callZome(alice, 'transport_sharing', 'post_ride_offer', {
          vehicle_hash: vehicleActionHash,
          route_hash: routeActionHash,
          driver: alice.agentPubKey,
          departure_time: Math.floor(Date.now() / 1000) + 3600,
          seats_available: 3,
          price_per_seat: 5.0,
          status: 'Open',
        }, 'commons_care');
        expect(offer).toBeDefined();
      });
    });
  });

  describe('Bridge Dispatch', () => {
    it('should successfully call commons bridge health check', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: COMMONS_HAPP_PATH } },
        ]);

        // Call the commons bridge for health check
        // Fn: health_check(()) -> BridgeHealth
        const health = await callZome(alice, 'commons_bridge', 'health_check', null, 'commons_care');
        expect(health).toBeDefined();
      });
    });
  });
});
