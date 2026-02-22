/**
 * Transport/Mobility Integration Tests
 *
 * Tests for TransportClient — the domain-specific SDK client for the
 * transport coordination zomes within the mycelix-commons cluster DNA.
 * Covers routes (vehicles, routes, stops), sharing (ride offers, requests,
 * matches, cargo), and impact (trip logs, emissions, carbon credits).
 *
 * All calls are dispatched through the commons role to the transport_routes,
 * transport_sharing, and transport_impact coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  TransportClient,
  createTransportClient,
  TRANSPORT_ZOMES,
  type Vehicle,
  type VehicleType,
  type VehicleStatus,
  type Route,
  type TransportMode,
  type Waypoint,
  type Stop,
  type StopType,
  type RideOffer,
  type RideOfferStatus,
  type RideRequest,
  type RideRequestStatus,
  type RideMatch,
  type RideMatchStatus,
  type CargoOffer,
  type TripLog,
  type TripMode,
  type CarbonCredit,
  type CreditSource,
  type EmissionsCalcInput,
  type EmissionsCalcResult,
  type CommunityImpactSummary,
  type MaintenanceType,
  type MaintenanceRecord,
  type VehicleFeatures,
  type ReviewerRole,
  type RideReview,
  type FindNearbyInput,
  type RedeemInput,
  type CreditRedemption,
  type CarbonBalance,
} from '../../src/integrations/transport/index.js';

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

function makeVehicle(overrides: Partial<Vehicle> = {}): Vehicle {
  return {
    id: 'vehicle-001',
    owner: fakeAgentKey(),
    vehicle_type: 'Car',
    capacity_kg: 500,
    capacity_passengers: 4,
    status: 'Available',
    ...overrides,
  };
}

function makeRoute(overrides: Partial<Route> = {}): Route {
  return {
    id: 'route-001',
    name: 'Downtown Loop',
    waypoints: [
      { lat: 32.95, lon: -96.75, label: 'Central Station' },
      { lat: 32.96, lon: -96.76, label: 'Market Square' },
      { lat: 32.97, lon: -96.74, label: 'City Park' },
    ],
    distance_km: 8.5,
    estimated_minutes: 25,
    mode: 'Driving',
    ...overrides,
  };
}

function makeStop(overrides: Partial<Stop> = {}): Stop {
  return {
    route_hash: fakeActionHash(),
    name: 'Main St & 5th Ave',
    location_lat: 32.955,
    location_lon: -96.755,
    scheduled_time: Date.now(),
    stop_type: 'Pickup',
    ...overrides,
  };
}

function makeRideOffer(overrides: Partial<RideOffer> = {}): RideOffer {
  return {
    vehicle_hash: fakeActionHash(),
    route_hash: null,
    driver: fakeAgentKey(),
    departure_time: Date.now() + 3600_000,
    seats_available: 3,
    price_per_seat: 5.0,
    status: 'Open',
    ...overrides,
  };
}

function makeRideRequest(overrides: Partial<RideRequest> = {}): RideRequest {
  return {
    requester: fakeAgentKey(),
    origin_lat: 32.95,
    origin_lon: -96.75,
    destination_lat: 32.97,
    destination_lon: -96.74,
    requested_time: Date.now() + 3600_000,
    passengers: 1,
    status: 'Open',
    ...overrides,
  };
}

function makeCargoOffer(overrides: Partial<CargoOffer> = {}): CargoOffer {
  return {
    vehicle_hash: fakeActionHash(),
    origin_lat: 32.95,
    origin_lon: -96.75,
    destination_lat: 33.10,
    destination_lon: -96.80,
    capacity_kg: 200,
    price_per_kg: 0.5,
    departure_time: Date.now() + 7200_000,
    ...overrides,
  };
}

function makeTripLog(overrides: Partial<TripLog> = {}): TripLog {
  return {
    vehicle_hash: fakeActionHash(),
    route_hash: fakeActionHash(),
    distance_km: 12.3,
    mode: 'Driving',
    passengers: 2,
    cargo_kg: 0,
    emissions_kg_co2: 2.46,
    logged_at: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Constants
// ============================================================================

describe('Transport Constants', () => {
  it('should export all 3 transport zome names', () => {
    expect(TRANSPORT_ZOMES).toEqual([
      'transport_routes',
      'transport_sharing',
      'transport_impact',
    ]);
    expect(TRANSPORT_ZOMES).toHaveLength(3);
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Transport Types', () => {
  describe('Vehicle', () => {
    it('should construct a valid Vehicle', () => {
      const vehicle = makeVehicle();
      expect(vehicle.id).toBe('vehicle-001');
      expect(vehicle.vehicle_type).toBe('Car');
      expect(vehicle.capacity_kg).toBe(500);
      expect(vehicle.capacity_passengers).toBe(4);
      expect(vehicle.status).toBe('Available');
      expect(vehicle.owner).toBeInstanceOf(Uint8Array);
    });

    it('should accept all VehicleType variants', () => {
      const types: VehicleType[] = [
        'Car', 'Van', 'Bike', 'Bus', 'Cargo', 'ElectricScooter', 'Helicopter',
        'EVTOL', 'AirTaxi', 'Ferry', 'Boat', 'Train', 'Tram', 'Skateboard',
        'Wheelchair', 'Segway', 'AutonomousVehicle', 'Drone',
      ];
      types.forEach((t) => {
        const v = makeVehicle({ vehicle_type: t });
        expect(v.vehicle_type).toBe(t);
      });
    });

    it('should accept all VehicleStatus variants', () => {
      const statuses: VehicleStatus[] = ['Available', 'InUse', 'Maintenance', 'Retired'];
      statuses.forEach((s) => {
        const v = makeVehicle({ status: s });
        expect(v.status).toBe(s);
      });
    });
  });

  describe('Route', () => {
    it('should construct a valid Route with waypoints', () => {
      const route = makeRoute();
      expect(route.name).toBe('Downtown Loop');
      expect(route.waypoints).toHaveLength(3);
      expect(route.distance_km).toBe(8.5);
      expect(route.estimated_minutes).toBe(25);
      expect(route.mode).toBe('Driving');
    });

    it('should accept all TransportMode variants', () => {
      const modes: TransportMode[] = [
        'Driving', 'Cycling', 'Walking', 'Transit', 'Mixed',
        'Flying', 'Water', 'Rail', 'Micromobility', 'Autonomous',
      ];
      modes.forEach((m) => {
        const r = makeRoute({ mode: m });
        expect(r.mode).toBe(m);
      });
    });
  });

  describe('Stop', () => {
    it('should construct a valid Stop', () => {
      const stop = makeStop();
      expect(stop.name).toBe('Main St & 5th Ave');
      expect(stop.route_hash).toBeInstanceOf(Uint8Array);
      expect(stop.location_lat).toBeCloseTo(32.955);
      expect(stop.location_lon).toBeCloseTo(-96.755);
    });

    it('should accept all StopType variants', () => {
      const types: StopType[] = ['Pickup', 'Dropoff', 'Transfer'];
      types.forEach((t) => {
        const s = makeStop({ stop_type: t });
        expect(s.stop_type).toBe(t);
      });
    });
  });

  describe('RideOffer', () => {
    it('should construct with nullable route_hash', () => {
      const offer = makeRideOffer();
      expect(offer.route_hash).toBeNull();
      expect(offer.seats_available).toBe(3);
      expect(offer.price_per_seat).toBe(5.0);
    });

    it('should accept all RideOfferStatus variants', () => {
      const statuses: RideOfferStatus[] = ['Open', 'Full', 'InProgress', 'Completed', 'Cancelled'];
      statuses.forEach((s) => {
        const o = makeRideOffer({ status: s });
        expect(o.status).toBe(s);
      });
    });
  });

  describe('RideRequest', () => {
    it('should construct a valid RideRequest', () => {
      const request = makeRideRequest();
      expect(request.origin_lat).toBeCloseTo(32.95);
      expect(request.passengers).toBe(1);
      expect(request.status).toBe('Open');
    });

    it('should accept all RideRequestStatus variants', () => {
      const statuses: RideRequestStatus[] = ['Open', 'Matched', 'Cancelled'];
      statuses.forEach((s) => {
        const r = makeRideRequest({ status: s });
        expect(r.status).toBe(s);
      });
    });
  });

  describe('RideMatch', () => {
    it('should construct with nullable confirmed_at', () => {
      const match: RideMatch = {
        offer_hash: fakeActionHash(),
        request_hash: fakeActionHash(),
        confirmed_at: null,
        status: 'Pending',
      };
      expect(match.confirmed_at).toBeNull();
      expect(match.status).toBe('Pending');
    });

    it('should accept all RideMatchStatus variants', () => {
      const statuses: RideMatchStatus[] = ['Pending', 'Confirmed', 'InProgress', 'Completed', 'Cancelled'];
      statuses.forEach((s) => {
        const m: RideMatch = {
          offer_hash: fakeActionHash(),
          request_hash: fakeActionHash(),
          confirmed_at: s === 'Pending' ? null : Date.now(),
          status: s,
        };
        expect(m.status).toBe(s);
      });
    });
  });

  describe('CargoOffer', () => {
    it('should construct a valid CargoOffer', () => {
      const cargo = makeCargoOffer();
      expect(cargo.capacity_kg).toBe(200);
      expect(cargo.price_per_kg).toBe(0.5);
      expect(cargo.origin_lat).toBeCloseTo(32.95);
      expect(cargo.destination_lat).toBeCloseTo(33.10);
    });
  });

  describe('TripLog', () => {
    it('should construct with nullable vehicle and route hashes', () => {
      const trip = makeTripLog({ vehicle_hash: null, route_hash: null });
      expect(trip.vehicle_hash).toBeNull();
      expect(trip.route_hash).toBeNull();
      expect(trip.distance_km).toBe(12.3);
    });

    it('should accept all TripMode variants', () => {
      const modes: TripMode[] = [
        'Driving', 'Cycling', 'Walking', 'Transit', 'Carpool',
        'ElectricVehicle', 'Flying', 'Water', 'Rail', 'Micromobility', 'Autonomous',
      ];
      modes.forEach((m) => {
        const t = makeTripLog({ mode: m });
        expect(t.mode).toBe(m);
      });
    });
  });

  describe('CarbonCredit', () => {
    it('should construct a valid CarbonCredit', () => {
      const credit: CarbonCredit = {
        holder: fakeAgentKey(),
        credits_kg_co2: 1.5,
        earned_from: 'Cycling',
        earned_at: Date.now(),
      };
      expect(credit.credits_kg_co2).toBe(1.5);
      expect(credit.earned_from).toBe('Cycling');
    });

    it('should accept all CreditSource variants', () => {
      const sources: CreditSource[] = ['Cycling', 'Walking', 'Transit', 'Carpool', 'ElectricVehicle'];
      sources.forEach((s) => {
        const c: CarbonCredit = {
          holder: fakeAgentKey(),
          credits_kg_co2: 1.0,
          earned_from: s,
          earned_at: Date.now(),
        };
        expect(c.earned_from).toBe(s);
      });
    });
  });

  describe('EmissionsCalcInput / EmissionsCalcResult', () => {
    it('should construct valid emissions calculation types', () => {
      const input: EmissionsCalcInput = {
        distance_km: 20,
        mode: 'Carpool',
        passengers: 4,
      };
      expect(input.distance_km).toBe(20);
      expect(input.mode).toBe('Carpool');
      expect(input.passengers).toBe(4);

      const result: EmissionsCalcResult = {
        emissions_kg_co2: 1.0,
        baseline_emissions: 4.0,
        savings_kg_co2: 3.0,
      };
      expect(result.savings_kg_co2).toBe(3.0);
      expect(result.emissions_kg_co2).toBeLessThan(result.baseline_emissions);
    });
  });

  describe('CommunityImpactSummary', () => {
    it('should construct a valid summary', () => {
      const summary: CommunityImpactSummary = {
        total_trips: 1200,
        total_distance_km: 15000,
        total_emissions_kg_co2: 3500,
        total_credits_earned: 800,
      };
      expect(summary.total_trips).toBe(1200);
      expect(summary.total_credits_earned).toBe(800);
    });
  });
});

// ============================================================================
// TransportClient — Factory
// ============================================================================

describe('TransportClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let transport: TransportClient;

  beforeEach(() => {
    client = createMockClient();
    transport = createTransportClient(client);
  });

  describe('factory', () => {
    it('should create via createTransportClient', () => {
      expect(transport).toBeInstanceOf(TransportClient);
    });

    it('should also be constructable directly', () => {
      const direct = new TransportClient(client);
      expect(direct).toBeInstanceOf(TransportClient);
    });
  });

  // ==========================================================================
  // Routes zome
  // ==========================================================================

  describe('Routes (transport_routes)', () => {
    describe('registerVehicle', () => {
      it('should call transport_routes.register_vehicle with correct params', async () => {
        const vehicle = makeVehicle();
        const mockHash = fakeActionHash();
        client.callZome.mockResolvedValue(mockHash);

        const result = await transport.registerVehicle(vehicle);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'register_vehicle',
          payload: vehicle,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass all vehicle types through to the zome', async () => {
        const types: VehicleType[] = [
          'Car', 'Van', 'Bike', 'Bus', 'Cargo', 'ElectricScooter', 'Helicopter',
          'EVTOL', 'AirTaxi', 'Ferry', 'Boat', 'Train', 'Tram', 'Skateboard',
          'Wheelchair', 'Segway', 'AutonomousVehicle', 'Drone',
        ];
        for (const vt of types) {
          const vehicle = makeVehicle({ vehicle_type: vt });
          await transport.registerVehicle(vehicle);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.vehicle_type).toBe(vt);
        }
      });
    });

    describe('getVehicle', () => {
      it('should call transport_routes.get_vehicle with action hash', async () => {
        const hash = fakeActionHash();
        const mockVehicle = makeVehicle();
        client.callZome.mockResolvedValue(mockVehicle);

        const result = await transport.getVehicle(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_vehicle',
          payload: hash,
        });
        expect(result).toEqual(mockVehicle);
      });
    });

    describe('getMyVehicles', () => {
      it('should call transport_routes.get_my_vehicles with null payload', async () => {
        const mockVehicles = [makeVehicle({ id: 'v1' }), makeVehicle({ id: 'v2' })];
        client.callZome.mockResolvedValue(mockVehicles);

        const result = await transport.getMyVehicles();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_my_vehicles',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('createRoute', () => {
      it('should call transport_routes.create_route with route data', async () => {
        const route = makeRoute();
        const mockHash = fakeActionHash();
        client.callZome.mockResolvedValue(mockHash);

        const result = await transport.createRoute(route);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'create_route',
          payload: route,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass waypoints array through', async () => {
        const waypoints: Waypoint[] = [
          { lat: 32.90, lon: -96.70, label: 'A' },
          { lat: 32.91, lon: -96.71, label: 'B' },
        ];
        const route = makeRoute({ waypoints });
        await transport.createRoute(route);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.waypoints).toHaveLength(2);
        expect(lastCall.payload.waypoints[0].label).toBe('A');
      });
    });

    describe('addStop', () => {
      it('should call transport_routes.add_stop with stop data', async () => {
        const stop = makeStop();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.addStop(stop);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'add_stop',
          payload: stop,
        });
      });

      it('should support all stop types', async () => {
        const types: StopType[] = ['Pickup', 'Dropoff', 'Transfer'];
        for (const st of types) {
          const stop = makeStop({ stop_type: st });
          await transport.addStop(stop);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.stop_type).toBe(st);
        }
      });
    });

    describe('getRouteStops', () => {
      it('should call transport_routes.get_route_stops with route hash', async () => {
        const routeHash = fakeActionHash();
        const mockStops = [makeStop({ name: 'Stop A' }), makeStop({ name: 'Stop B' })];
        client.callZome.mockResolvedValue(mockStops);

        const result = await transport.getRouteStops(routeHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_route_stops',
          payload: routeHash,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getAllRoutes', () => {
      it('should call transport_routes.get_all_routes with null payload', async () => {
        const mockRoutes = [makeRoute({ name: 'Route A' })];
        client.callZome.mockResolvedValue(mockRoutes);

        const result = await transport.getAllRoutes();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_all_routes',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('updateVehicleStatus', () => {
      it('should call transport_routes.update_vehicle_status with hash and status', async () => {
        const vehicleHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.updateVehicleStatus(vehicleHash, 'Maintenance');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'update_vehicle_status',
          payload: { vehicle_hash: vehicleHash, new_status: 'Maintenance' },
        });
      });

      it('should handle all vehicle status transitions', async () => {
        const statuses: VehicleStatus[] = ['Available', 'InUse', 'Maintenance', 'Retired'];
        for (const s of statuses) {
          await transport.updateVehicleStatus(fakeActionHash(), s);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.new_status).toBe(s);
        }
      });
    });
  });

  // ==========================================================================
  // Sharing zome
  // ==========================================================================

  describe('Sharing (transport_sharing)', () => {
    describe('postRideOffer', () => {
      it('should call transport_sharing.post_ride_offer with offer data', async () => {
        const offer = makeRideOffer();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.postRideOffer(offer);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'post_ride_offer',
          payload: offer,
        });
      });

      it('should allow route_hash to be non-null when a route is linked', async () => {
        const routeHash = fakeActionHash();
        const offer = makeRideOffer({ route_hash: routeHash });
        await transport.postRideOffer(offer);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.route_hash).toEqual(routeHash);
      });
    });

    describe('requestRide', () => {
      it('should call transport_sharing.request_ride with request data', async () => {
        const request = makeRideRequest();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.requestRide(request);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'request_ride',
          payload: request,
        });
      });

      it('should pass multi-passenger requests through', async () => {
        const request = makeRideRequest({ passengers: 3 });
        await transport.requestRide(request);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.passengers).toBe(3);
      });
    });

    describe('matchRide', () => {
      it('should call transport_sharing.match_ride with offer and request hashes', async () => {
        const offerHash = new Uint8Array(39).fill(0x01);
        const requestHash = new Uint8Array(39).fill(0x02);
        const mockMatch: RideMatch = {
          offer_hash: offerHash,
          request_hash: requestHash,
          confirmed_at: null,
          status: 'Pending',
        };
        client.callZome.mockResolvedValue(mockMatch);

        const result = await transport.matchRide(offerHash, requestHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'match_ride',
          payload: { offer_hash: offerHash, request_hash: requestHash },
        });
        expect(result).toEqual(mockMatch);
      });
    });

    describe('confirmMatch', () => {
      it('should call transport_sharing.confirm_match with UpdateMatchStatusInput', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.confirmMatch(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'confirm_match',
          payload: { match_hash: matchHash, new_status: 'Confirmed' },
        });
      });
    });

    describe('completeRide', () => {
      it('should call transport_sharing.complete_ride with match hash', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.completeRide(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'complete_ride',
          payload: matchHash,
        });
      });
    });

    describe('cancelRide', () => {
      it('should call transport_sharing.cancel_ride with match hash', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.cancelRide(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'cancel_ride',
          payload: matchHash,
        });
      });
    });

    describe('postCargoOffer', () => {
      it('should call transport_sharing.post_cargo_offer with cargo data', async () => {
        const cargo = makeCargoOffer();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.postCargoOffer(cargo);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'post_cargo_offer',
          payload: cargo,
        });
      });

      it('should pass capacity and price through correctly', async () => {
        const cargo = makeCargoOffer({ capacity_kg: 1500, price_per_kg: 1.25 });
        await transport.postCargoOffer(cargo);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.capacity_kg).toBe(1500);
        expect(lastCall.payload.price_per_kg).toBe(1.25);
      });
    });

    describe('getAvailableRides', () => {
      it('should call transport_sharing.get_available_rides with null payload', async () => {
        const mockRides = [makeRideOffer(), makeRideOffer({ seats_available: 1 })];
        client.callZome.mockResolvedValue(mockRides);

        const result = await transport.getAvailableRides();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'get_available_rides',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getMyRides', () => {
      it('should call transport_sharing.get_my_rides with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await transport.getMyRides();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'get_my_rides',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Impact zome
  // ==========================================================================

  describe('Impact (transport_impact)', () => {
    describe('logTrip', () => {
      it('should call transport_impact.log_trip with trip data', async () => {
        const trip = makeTripLog();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.logTrip(trip);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'log_trip',
          payload: trip,
        });
      });

      it('should pass all trip modes through', async () => {
        const modes: TripMode[] = [
          'Driving', 'Cycling', 'Walking', 'Transit', 'Carpool',
          'ElectricVehicle', 'Flying', 'Water', 'Rail', 'Micromobility', 'Autonomous',
        ];
        for (const m of modes) {
          const trip = makeTripLog({ mode: m });
          await transport.logTrip(trip);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.mode).toBe(m);
        }
      });

      it('should handle trip with null vehicle and route hashes', async () => {
        const trip = makeTripLog({ vehicle_hash: null, route_hash: null, mode: 'Walking' });
        await transport.logTrip(trip);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.vehicle_hash).toBeNull();
        expect(lastCall.payload.route_hash).toBeNull();
      });
    });

    describe('getMyTrips', () => {
      it('should call transport_impact.get_my_trips with null payload', async () => {
        const mockTrips = [makeTripLog(), makeTripLog({ distance_km: 5.0 })];
        client.callZome.mockResolvedValue(mockTrips);

        const result = await transport.getMyTrips();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_my_trips',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getMyCarbonCredits', () => {
      it('should call transport_impact.get_my_carbon_credits with null payload', async () => {
        const mockCredits: CarbonCredit[] = [
          { holder: fakeAgentKey(), credits_kg_co2: 2.5, earned_from: 'Cycling', earned_at: Date.now() },
          { holder: fakeAgentKey(), credits_kg_co2: 1.0, earned_from: 'Walking', earned_at: Date.now() },
        ];
        client.callZome.mockResolvedValue(mockCredits);

        const result = await transport.getMyCarbonCredits();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_my_carbon_credits',
          payload: null,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('calculateEmissions', () => {
      it('should call transport_impact.calculate_emissions with input', async () => {
        const input: EmissionsCalcInput = {
          distance_km: 20,
          mode: 'Carpool',
          passengers: 4,
        };
        const mockResult: EmissionsCalcResult = {
          emissions_kg_co2: 1.0,
          baseline_emissions: 4.0,
          savings_kg_co2: 3.0,
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await transport.calculateEmissions(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'calculate_emissions',
          payload: input,
        });
        expect(result.emissions_kg_co2).toBe(1.0);
        expect(result.savings_kg_co2).toBe(3.0);
        expect(result.baseline_emissions).toBe(4.0);
      });

      it('should calculate emissions for different modes', async () => {
        const modes: TripMode[] = ['Driving', 'ElectricVehicle', 'Transit', 'Cycling'];
        for (const m of modes) {
          const input: EmissionsCalcInput = { distance_km: 10, mode: m, passengers: 1 };
          await transport.calculateEmissions(input);

          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.mode).toBe(m);
        }
      });
    });

    describe('getVehicleTripHistory', () => {
      it('should call transport_impact.get_vehicle_trip_history with vehicle hash', async () => {
        const vehicleHash = fakeActionHash();
        const mockHistory = [makeTripLog(), makeTripLog()];
        client.callZome.mockResolvedValue(mockHistory);

        const result = await transport.getVehicleTripHistory(vehicleHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_vehicle_trip_history',
          payload: vehicleHash,
        });
        expect(result).toHaveLength(2);
      });
    });

    describe('getCommunityImpactSummary', () => {
      it('should call transport_impact.get_community_impact_summary with null payload', async () => {
        const mockSummary: CommunityImpactSummary = {
          total_trips: 500,
          total_distance_km: 6200,
          total_emissions_kg_co2: 1400,
          total_credits_earned: 320,
        };
        client.callZome.mockResolvedValue(mockSummary);

        const result = await transport.getCommunityImpactSummary();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_community_impact_summary',
          payload: null,
        });
        expect(result.total_trips).toBe(500);
        expect(result.total_distance_km).toBe(6200);
        expect(result.total_emissions_kg_co2).toBe(1400);
        expect(result.total_credits_earned).toBe(320);
      });
    });
  });

  // ==========================================================================
  // Ride lifecycle (offer -> request -> match -> confirm -> complete)
  // ==========================================================================

  describe('Ride Offer/Request/Match Lifecycle', () => {
    it('should support the full ride lifecycle through mock calls', async () => {
      const offerHash = new Uint8Array(39).fill(0x01);
      const requestHash = new Uint8Array(39).fill(0x02);
      const matchHash = new Uint8Array(39).fill(0x03);

      // Step 1: Post a ride offer
      client.callZome.mockResolvedValueOnce(offerHash);
      const offer = makeRideOffer();
      const offerResult = await transport.postRideOffer(offer);
      expect(offerResult).toEqual(offerHash);

      // Step 2: Request a ride
      client.callZome.mockResolvedValueOnce(requestHash);
      const request = makeRideRequest();
      const requestResult = await transport.requestRide(request);
      expect(requestResult).toEqual(requestHash);

      // Step 3: Match offer to request
      const matchData: RideMatch = {
        offer_hash: offerHash,
        request_hash: requestHash,
        confirmed_at: null,
        status: 'Pending',
      };
      client.callZome.mockResolvedValueOnce(matchData);
      const matchResult = await transport.matchRide(offerHash, requestHash);
      expect(matchResult).toEqual(matchData);

      // Step 4: Confirm the match
      client.callZome.mockResolvedValueOnce({ ...matchData, confirmed_at: Date.now(), status: 'Confirmed' });
      await transport.confirmMatch(matchHash);

      // Step 5: Complete the ride
      client.callZome.mockResolvedValueOnce({ ...matchData, status: 'Completed' });
      await transport.completeRide(matchHash);

      // Verify 5 zome calls were made
      expect(client.callZome).toHaveBeenCalledTimes(5);

      // Verify the correct sequence of zome functions
      const fnNames = client.callZome.mock.calls.map((c: unknown[]) => (c[0] as { fn_name: string }).fn_name);
      expect(fnNames).toEqual([
        'post_ride_offer',
        'request_ride',
        'match_ride',
        'confirm_match',
        'complete_ride',
      ]);
    });

    it('should support cancellation flow', async () => {
      const matchHash = fakeActionHash();
      client.callZome.mockResolvedValue({});

      await transport.cancelRide(matchHash);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_care',
        zome_name: 'transport_sharing',
        fn_name: 'cancel_ride',
        payload: matchHash,
      });
    });
  });

  // ==========================================================================
  // Cross-domain dispatch pattern
  // ==========================================================================

  describe('Cross-domain dispatch to transport', () => {
    it('should use commons role for all transport_routes calls', async () => {
      await transport.registerVehicle(makeVehicle());
      await transport.createRoute(makeRoute());
      await transport.getAllRoutes();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('transport_routes');
      }
    });

    it('should use commons role for all transport_sharing calls', async () => {
      await transport.postRideOffer(makeRideOffer());
      await transport.requestRide(makeRideRequest());
      await transport.getAvailableRides();
      await transport.getMyRides();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('transport_sharing');
      }
    });

    it('should use commons role for all transport_impact calls', async () => {
      await transport.logTrip(makeTripLog());
      await transport.getMyTrips();
      await transport.getMyCarbonCredits();
      await transport.calculateEmissions({ distance_km: 10, mode: 'Driving', passengers: 1 });
      await transport.getVehicleTripHistory(fakeActionHash());
      await transport.getCommunityImpactSummary();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('transport_impact');
      }
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate zome call errors for registerVehicle', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(transport.registerVehicle(makeVehicle())).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate zome call errors for createRoute', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(transport.createRoute(makeRoute())).rejects.toThrow('Validation failed');
    });

    it('should propagate zome call errors for postRideOffer', async () => {
      client.callZome.mockRejectedValue(new Error('Rate limited'));
      await expect(transport.postRideOffer(makeRideOffer())).rejects.toThrow('Rate limited');
    });

    it('should propagate zome call errors for calculateEmissions', async () => {
      client.callZome.mockRejectedValue(new Error('Invalid mode'));
      await expect(
        transport.calculateEmissions({ distance_km: 10, mode: 'Driving', passengers: 1 }),
      ).rejects.toThrow('Invalid mode');
    });

    it('should propagate zome call errors for getCommunityImpactSummary', async () => {
      client.callZome.mockRejectedValue(new Error('Network error'));
      await expect(transport.getCommunityImpactSummary()).rejects.toThrow('Network error');
    });
  });

  // ==========================================================================
  // Maintenance (Phase 3)
  // ==========================================================================

  describe('Maintenance (transport_routes)', () => {
    describe('logMaintenance', () => {
      it('should call transport_routes.log_maintenance with record data', async () => {
        const record: MaintenanceRecord = {
          id: 'maint-001',
          vehicle_hash: fakeActionHash(),
          maintenance_type: 'Scheduled',
          description: 'Oil change and tire rotation',
          cost: 85.0,
          completed_at: Date.now(),
          next_due: Date.now() + 90 * 24 * 3600_000,
          mechanic_notes: 'All fluids topped off',
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.logMaintenance(record);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'log_maintenance',
          payload: record,
        });
      });

      it('should accept all MaintenanceType variants', async () => {
        const types: MaintenanceType[] = ['Scheduled', 'Repair', 'Inspection'];
        for (const mt of types) {
          const record: MaintenanceRecord = {
            id: `maint-${mt}`,
            vehicle_hash: fakeActionHash(),
            maintenance_type: mt,
            description: `${mt} service`,
            cost: 50,
            completed_at: Date.now(),
            next_due: null,
            mechanic_notes: '',
          };
          await transport.logMaintenance(record);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.maintenance_type).toBe(mt);
        }
      });

      it('should handle nullable next_due', async () => {
        const record: MaintenanceRecord = {
          id: 'maint-002',
          vehicle_hash: fakeActionHash(),
          maintenance_type: 'Repair',
          description: 'Flat tire replacement',
          cost: 120,
          completed_at: Date.now(),
          next_due: null,
          mechanic_notes: 'Replaced front-left tire',
        };
        await transport.logMaintenance(record);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.next_due).toBeNull();
      });
    });

    describe('getVehicleMaintenance', () => {
      it('should call transport_routes.get_vehicle_maintenance with vehicle hash', async () => {
        const vehicleHash = fakeActionHash();
        client.callZome.mockResolvedValue([]);

        const result = await transport.getVehicleMaintenance(vehicleHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_vehicle_maintenance',
          payload: vehicleHash,
        });
        expect(result).toEqual([]);
      });
    });

    describe('setVehicleFeatures', () => {
      it('should call transport_routes.set_vehicle_features with features data', async () => {
        const features: VehicleFeatures = {
          vehicle_hash: fakeActionHash(),
          wheelchair_accessible: true,
          child_seat: false,
          pet_friendly: true,
          air_conditioning: true,
          bike_rack: true,
          luggage_capacity_liters: 400,
        };
        client.callZome.mockResolvedValue({});

        await transport.setVehicleFeatures(features);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'set_vehicle_features',
          payload: features,
        });
      });

      it('should pass all boolean feature flags through', async () => {
        const features: VehicleFeatures = {
          vehicle_hash: fakeActionHash(),
          wheelchair_accessible: false,
          child_seat: true,
          pet_friendly: false,
          air_conditioning: false,
          bike_rack: false,
          luggage_capacity_liters: 0,
        };
        await transport.setVehicleFeatures(features);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.wheelchair_accessible).toBe(false);
        expect(lastCall.payload.child_seat).toBe(true);
        expect(lastCall.payload.luggage_capacity_liters).toBe(0);
      });
    });

    describe('getAccessibleVehicles', () => {
      it('should call transport_routes.get_accessible_vehicles with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await transport.getAccessibleVehicles();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_accessible_vehicles',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getVehiclesNeedingMaintenance', () => {
      it('should call transport_routes.get_vehicles_needing_maintenance with currentTime', async () => {
        client.callZome.mockResolvedValue([]);
        const now = Date.now();

        const result = await transport.getVehiclesNeedingMaintenance(now);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_routes',
          fn_name: 'get_vehicles_needing_maintenance',
          payload: now,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Reviews (Phase 2)
  // ==========================================================================

  describe('Reviews (transport_sharing)', () => {
    describe('reviewRide', () => {
      it('should call transport_sharing.review_ride with review data', async () => {
        const review: RideReview = {
          match_hash: fakeActionHash(),
          reviewer: fakeAgentKey(),
          role: 'Passenger',
          rating: 5,
          comment: 'Great ride, very safe driver',
          safety_concern: false,
          created_at: Date.now(),
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.reviewRide(review);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'review_ride',
          payload: review,
        });
      });

      it('should accept all ReviewerRole variants', async () => {
        const roles: ReviewerRole[] = ['Driver', 'Passenger'];
        for (const role of roles) {
          const review: RideReview = {
            match_hash: fakeActionHash(),
            reviewer: fakeAgentKey(),
            role,
            rating: 4,
            comment: 'Good experience',
            safety_concern: false,
            created_at: Date.now(),
          };
          await transport.reviewRide(review);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.role).toBe(role);
        }
      });

      it('should flag safety concerns', async () => {
        const review: RideReview = {
          match_hash: fakeActionHash(),
          reviewer: fakeAgentKey(),
          role: 'Passenger',
          rating: 1,
          comment: 'Reckless driving',
          safety_concern: true,
          created_at: Date.now(),
        };
        await transport.reviewRide(review);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.safety_concern).toBe(true);
        expect(lastCall.payload.rating).toBe(1);
      });
    });

    describe('getRideReviews', () => {
      it('should call transport_sharing.get_ride_reviews with match hash', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue([]);

        const result = await transport.getRideReviews(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'get_ride_reviews',
          payload: matchHash,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getDriverRating', () => {
      it('should call transport_sharing.get_driver_rating with driver key', async () => {
        const driver = fakeAgentKey();
        client.callZome.mockResolvedValue({ average: 4.5, count: 12 });

        const result = await transport.getDriverRating(driver);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'get_driver_rating',
          payload: driver,
        });
        expect(result).toEqual({ average: 4.5, count: 12 });
      });
    });

    describe('findNearbyRides', () => {
      it('should call transport_sharing.find_nearby_rides with location input', async () => {
        const input: FindNearbyInput = {
          origin_lat: 32.95,
          origin_lon: -96.75,
          radius_km: 10,
        };
        client.callZome.mockResolvedValue([]);

        const result = await transport.findNearbyRides(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_sharing',
          fn_name: 'find_nearby_rides',
          payload: input,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Carbon Redemption (Phase 2)
  // ==========================================================================

  describe('Carbon Redemption (transport_impact)', () => {
    describe('redeemCredits', () => {
      it('should call transport_impact.redeem_credits with input data', async () => {
        const input: RedeemInput = {
          credits_redeemed: 5.0,
          redeemed_for: 'Community garden compost',
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.redeemCredits(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'redeem_credits',
          payload: input,
        });
      });
    });

    describe('getMyRedemptions', () => {
      it('should call transport_impact.get_my_redemptions with null payload', async () => {
        const mockRedemptions: CreditRedemption[] = [
          {
            holder: fakeAgentKey(),
            credits_redeemed: 3.0,
            redeemed_for: 'Tree planting',
            redeemed_at: Date.now(),
          },
        ];
        client.callZome.mockResolvedValue(mockRedemptions);

        const result = await transport.getMyRedemptions();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_my_redemptions',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('getAgentCarbonBalance', () => {
      it('should call transport_impact.get_agent_carbon_balance with agent key', async () => {
        const agent = fakeAgentKey();
        const mockBalance: CarbonBalance = {
          total_earned: 25.0,
          total_redeemed: 10.0,
          balance: 15.0,
        };
        client.callZome.mockResolvedValue(mockBalance);

        const result = await transport.getAgentCarbonBalance(agent);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'transport_impact',
          fn_name: 'get_agent_carbon_balance',
          payload: agent,
        });
        expect(result.total_earned).toBe(25.0);
        expect(result.total_redeemed).toBe(10.0);
        expect(result.balance).toBe(15.0);
      });
    });
  });

  // ==========================================================================
  // Phase 2-3 Type Construction
  // ==========================================================================

  describe('Phase 2-3 Types', () => {
    describe('MaintenanceRecord', () => {
      it('should construct a valid MaintenanceRecord', () => {
        const record: MaintenanceRecord = {
          id: 'maint-001',
          vehicle_hash: fakeActionHash(),
          maintenance_type: 'Inspection',
          description: 'Annual safety inspection',
          cost: 50,
          completed_at: Date.now(),
          next_due: Date.now() + 365 * 24 * 3600_000,
          mechanic_notes: 'All systems check out',
        };
        expect(record.id).toBe('maint-001');
        expect(record.maintenance_type).toBe('Inspection');
        expect(record.vehicle_hash).toBeInstanceOf(Uint8Array);
      });
    });

    describe('VehicleFeatures', () => {
      it('should construct with all boolean flags', () => {
        const features: VehicleFeatures = {
          vehicle_hash: fakeActionHash(),
          wheelchair_accessible: true,
          child_seat: true,
          pet_friendly: true,
          air_conditioning: true,
          bike_rack: true,
          luggage_capacity_liters: 500,
        };
        expect(features.wheelchair_accessible).toBe(true);
        expect(features.luggage_capacity_liters).toBe(500);
      });
    });

    describe('RideReview', () => {
      it('should construct with all fields including safety_concern', () => {
        const review: RideReview = {
          match_hash: fakeActionHash(),
          reviewer: fakeAgentKey(),
          role: 'Driver',
          rating: 5,
          comment: 'Polite passenger',
          safety_concern: false,
          created_at: Date.now(),
        };
        expect(review.role).toBe('Driver');
        expect(review.safety_concern).toBe(false);
        expect(review.rating).toBe(5);
      });
    });

    describe('CarbonBalance', () => {
      it('should construct with earned, redeemed, and balance', () => {
        const balance: CarbonBalance = {
          total_earned: 100,
          total_redeemed: 40,
          balance: 60,
        };
        expect(balance.balance).toBe(60);
        expect(balance.total_earned - balance.total_redeemed).toBe(balance.balance);
      });
    });

    describe('CreditRedemption', () => {
      it('should construct a valid CreditRedemption', () => {
        const redemption: CreditRedemption = {
          holder: fakeAgentKey(),
          credits_redeemed: 10,
          redeemed_for: 'Solar panel discount',
          redeemed_at: Date.now(),
        };
        expect(redemption.credits_redeemed).toBe(10);
        expect(redemption.holder).toBeInstanceOf(Uint8Array);
      });
    });
  });

  // ==========================================================================
  // Phase 2-3 error propagation
  // ==========================================================================

  describe('Phase 2-3 error propagation', () => {
    it('should propagate errors for logMaintenance', async () => {
      client.callZome.mockRejectedValue(new Error('Vehicle not found'));
      const record: MaintenanceRecord = {
        id: 'maint-err',
        vehicle_hash: fakeActionHash(),
        maintenance_type: 'Repair',
        description: 'test',
        cost: 0,
        completed_at: Date.now(),
        next_due: null,
        mechanic_notes: '',
      };
      await expect(transport.logMaintenance(record)).rejects.toThrow('Vehicle not found');
    });

    it('should propagate errors for reviewRide', async () => {
      client.callZome.mockRejectedValue(new Error('Match not found'));
      const review: RideReview = {
        match_hash: fakeActionHash(),
        reviewer: fakeAgentKey(),
        role: 'Passenger',
        rating: 3,
        comment: '',
        safety_concern: false,
        created_at: Date.now(),
      };
      await expect(transport.reviewRide(review)).rejects.toThrow('Match not found');
    });

    it('should propagate errors for redeemCredits', async () => {
      client.callZome.mockRejectedValue(new Error('Insufficient credits'));
      await expect(
        transport.redeemCredits({ credits_redeemed: 999, redeemed_for: 'test' }),
      ).rejects.toThrow('Insufficient credits');
    });

    it('should propagate errors for getAgentCarbonBalance', async () => {
      client.callZome.mockRejectedValue(new Error('Agent not found'));
      await expect(transport.getAgentCarbonBalance(fakeAgentKey())).rejects.toThrow(
        'Agent not found',
      );
    });
  });
});
