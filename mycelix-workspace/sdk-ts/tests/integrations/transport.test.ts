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
  type RouteMode,
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
      { lat: 32.95, lon: -96.75, name: 'Central Station' },
      { lat: 32.96, lon: -96.76, name: 'Market Square' },
      { lat: 32.97, lon: -96.74, name: 'City Park' },
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
      const types: VehicleType[] = ['Car', 'Van', 'Bike', 'Bus', 'Cargo'];
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

    it('should accept all RouteMode variants', () => {
      const modes: RouteMode[] = ['Driving', 'Transit', 'Cycling', 'Walking', 'Mixed'];
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
      const statuses: RideRequestStatus[] = ['Open', 'Matched', 'Completed', 'Cancelled'];
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
      const modes: TripMode[] = ['Driving', 'ElectricVehicle', 'Transit', 'Carpool', 'Cycling', 'Walking'];
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
          role_name: 'commons',
          zome_name: 'transport_routes',
          fn_name: 'register_vehicle',
          payload: vehicle,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass all vehicle types through to the zome', async () => {
        const types: VehicleType[] = ['Car', 'Van', 'Bike', 'Bus', 'Cargo'];
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
          zome_name: 'transport_routes',
          fn_name: 'create_route',
          payload: route,
        });
        expect(result).toEqual(mockHash);
      });

      it('should pass waypoints array through', async () => {
        const waypoints: Waypoint[] = [
          { lat: 32.90, lon: -96.70, name: 'A' },
          { lat: 32.91, lon: -96.71, name: 'B' },
        ];
        const route = makeRoute({ waypoints });
        await transport.createRoute(route);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.waypoints).toHaveLength(2);
        expect(lastCall.payload.waypoints[0].name).toBe('A');
      });
    });

    describe('addStop', () => {
      it('should call transport_routes.add_stop with stop data', async () => {
        const stop = makeStop();
        client.callZome.mockResolvedValue(fakeActionHash());

        await transport.addStop(stop);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
          zome_name: 'transport_sharing',
          fn_name: 'match_ride',
          payload: { offer_hash: offerHash, request_hash: requestHash },
        });
        expect(result).toEqual(mockMatch);
      });
    });

    describe('confirmMatch', () => {
      it('should call transport_sharing.confirm_match with match hash', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.confirmMatch(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons',
          zome_name: 'transport_sharing',
          fn_name: 'confirm_match',
          payload: matchHash,
        });
      });
    });

    describe('completeRide', () => {
      it('should call transport_sharing.complete_ride with match hash', async () => {
        const matchHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await transport.completeRide(matchHash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
          zome_name: 'transport_impact',
          fn_name: 'log_trip',
          payload: trip,
        });
      });

      it('should pass all trip modes through', async () => {
        const modes: TripMode[] = ['Driving', 'ElectricVehicle', 'Transit', 'Carpool', 'Cycling', 'Walking'];
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
          role_name: 'commons',
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
        role_name: 'commons',
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
        expect((call[0] as { role_name: string }).role_name).toBe('commons');
        expect((call[0] as { zome_name: string }).zome_name).toBe('transport_routes');
      }
    });

    it('should use commons role for all transport_sharing calls', async () => {
      await transport.postRideOffer(makeRideOffer());
      await transport.requestRide(makeRideRequest());
      await transport.getAvailableRides();
      await transport.getMyRides();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons');
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
        expect((call[0] as { role_name: string }).role_name).toBe('commons');
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
});
