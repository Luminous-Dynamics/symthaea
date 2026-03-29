// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Transport/Mobility Integration
 *
 * Domain-specific client for the transport coordination zomes within the
 * mycelix-commons cluster DNA. Covers routes (vehicles, routes, stops),
 * sharing (ride offers, requests, matches, cargo), and impact (trip logs,
 * emissions, carbon credits).
 *
 * All calls are dispatched through the commons_bridge coordinator zome.
 *
 * @packageDocumentation
 * @module integrations/transport
 */

// ============================================================================
// Types — Transport Routes
// ============================================================================

export type VehicleType =
  | 'Car'
  | 'Van'
  | 'Bike'
  | 'Bus'
  | 'Cargo'
  | 'ElectricScooter'
  | 'Helicopter'
  | 'EVTOL'
  | 'AirTaxi'
  | 'Ferry'
  | 'Boat'
  | 'Train'
  | 'Tram'
  | 'Skateboard'
  | 'Wheelchair'
  | 'Segway'
  | 'AutonomousVehicle'
  | 'Drone';

export type VehicleStatus = 'Available' | 'InUse' | 'Maintenance' | 'Retired';

export interface Vehicle {
  id: string;
  owner: Uint8Array;
  vehicle_type: VehicleType;
  capacity_kg: number;
  capacity_passengers: number;
  status: VehicleStatus;
}

export type TransportMode =
  | 'Driving'
  | 'Cycling'
  | 'Walking'
  | 'Transit'
  | 'Mixed'
  | 'Flying'
  | 'Water'
  | 'Rail'
  | 'Micromobility'
  | 'Autonomous';

export interface Waypoint {
  lat: number;
  lon: number;
  label: string | null;
}

export interface Route {
  id: string;
  name: string;
  waypoints: Waypoint[];
  distance_km: number;
  estimated_minutes: number;
  mode: TransportMode;
}

export type StopType = 'Pickup' | 'Dropoff' | 'Transfer';

export interface Stop {
  route_hash: Uint8Array;
  name: string;
  location_lat: number;
  location_lon: number;
  scheduled_time: number | null;
  stop_type: StopType;
}

// ============================================================================
// Types — Transport Sharing
// ============================================================================

export type RideOfferStatus = 'Open' | 'Full' | 'InProgress' | 'Completed' | 'Cancelled';
export type RideRequestStatus = 'Open' | 'Matched' | 'Cancelled';
export type RideMatchStatus = 'Pending' | 'Confirmed' | 'InProgress' | 'Completed' | 'Cancelled';
export type MatchStatus = 'Pending' | 'Confirmed' | 'InProgress' | 'Completed' | 'Cancelled';

export interface UpdateMatchStatusInput {
  match_hash: Uint8Array;
  new_status: MatchStatus;
}

export interface RideOffer {
  vehicle_hash: Uint8Array;
  route_hash: Uint8Array | null;
  driver: Uint8Array;
  departure_time: number;
  seats_available: number;
  price_per_seat: number;
  status: RideOfferStatus;
}

export interface RideRequest {
  requester: Uint8Array;
  origin_lat: number;
  origin_lon: number;
  destination_lat: number;
  destination_lon: number;
  requested_time: number;
  passengers: number;
  status: RideRequestStatus;
}

export interface RideMatch {
  offer_hash: Uint8Array;
  request_hash: Uint8Array;
  confirmed_at: number | null;
  status: RideMatchStatus;
}

export interface CargoOffer {
  vehicle_hash: Uint8Array;
  origin_lat: number;
  origin_lon: number;
  destination_lat: number;
  destination_lon: number;
  capacity_kg: number;
  price_per_kg: number;
  departure_time: number;
}

// ============================================================================
// Types — Transport Impact
// ============================================================================

export type TripMode =
  | 'Driving'
  | 'Cycling'
  | 'Walking'
  | 'Transit'
  | 'Carpool'
  | 'ElectricVehicle'
  | 'Flying'
  | 'Water'
  | 'Rail'
  | 'Micromobility'
  | 'Autonomous';

export type CreditSource = 'Cycling' | 'Walking' | 'Transit' | 'Carpool' | 'ElectricVehicle';

export interface TripLog {
  vehicle_hash: Uint8Array | null;
  route_hash: Uint8Array | null;
  distance_km: number;
  mode: TripMode;
  passengers: number;
  cargo_kg: number;
  emissions_kg_co2: number;
  logged_at: number;
}

export interface CarbonCredit {
  holder: Uint8Array;
  credits_kg_co2: number;
  earned_from: CreditSource;
  earned_at: number;
}

export interface EmissionsCalcInput {
  distance_km: number;
  mode: TripMode;
  passengers: number;
}

export interface EmissionsCalcResult {
  emissions_kg_co2: number;
  baseline_emissions: number;
  savings_kg_co2: number;
}

export interface CommunityImpactSummary {
  total_trips: number;
  total_distance_km: number;
  total_emissions_kg_co2: number;
  total_credits_earned: number;
}

// ============================================================================
// Types — Maintenance (Phase 3)
// ============================================================================

export type MaintenanceType = 'Scheduled' | 'Repair' | 'Inspection';

export interface MaintenanceRecord {
  id: string;
  vehicle_hash: Uint8Array;
  maintenance_type: MaintenanceType;
  description: string;
  cost: number;
  completed_at: number;
  next_due: number | null;
  mechanic_notes: string;
}

export interface VehicleFeatures {
  vehicle_hash: Uint8Array;
  wheelchair_accessible: boolean;
  child_seat: boolean;
  pet_friendly: boolean;
  air_conditioning: boolean;
  bike_rack: boolean;
  luggage_capacity_liters: number;
}

// ============================================================================
// Types — Reviews (Phase 2)
// ============================================================================

export type ReviewerRole = 'Driver' | 'Passenger';

export interface RideReview {
  match_hash: Uint8Array;
  reviewer: Uint8Array;
  role: ReviewerRole;
  rating: number;
  comment: string;
  safety_concern: boolean;
  created_at: number;
}

export interface FindNearbyInput {
  origin_lat: number;
  origin_lon: number;
  radius_km: number;
}

// ============================================================================
// Types — Carbon Redemption (Phase 2)
// ============================================================================

export interface RedeemInput {
  credits_redeemed: number;
  redeemed_for: string;
}

export interface CreditRedemption {
  holder: Uint8Array;
  credits_redeemed: number;
  redeemed_for: string;
  redeemed_at: number;
}

export interface CarbonBalance {
  total_earned: number;
  total_redeemed: number;
  balance: number;
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

const COMMONS_ROLE = 'commons_care';

export const TRANSPORT_ZOMES = [
  'transport_routes',
  'transport_sharing',
  'transport_impact',
] as const;

// ============================================================================
// Transport Client
// ============================================================================

/**
 * Client for the transport/mobility domain within the commons cluster.
 *
 * Provides typed access to all 3 transport zomes: routes, sharing, and impact.
 */
export class TransportClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Routes ---

  async registerVehicle(vehicle: Vehicle): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'register_vehicle',
      payload: vehicle,
    });
  }

  async getVehicle(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_vehicle',
      payload: actionHash,
    });
  }

  async getMyVehicles(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_my_vehicles',
      payload: null,
    });
  }

  async createRoute(route: Route): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'create_route',
      payload: route,
    });
  }

  async addStop(stop: Stop): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'add_stop',
      payload: stop,
    });
  }

  async getRouteStops(routeHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_route_stops',
      payload: routeHash,
    });
  }

  async getAllRoutes(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_all_routes',
      payload: null,
    });
  }

  async updateVehicleStatus(vehicleHash: Uint8Array, status: VehicleStatus): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'update_vehicle_status',
      payload: { vehicle_hash: vehicleHash, new_status: status },
    });
  }

  // --- Sharing ---

  async postRideOffer(offer: RideOffer): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'post_ride_offer',
      payload: offer,
    });
  }

  async requestRide(request: RideRequest): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'request_ride',
      payload: request,
    });
  }

  async matchRide(offerHash: Uint8Array, requestHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'match_ride',
      payload: { offer_hash: offerHash, request_hash: requestHash },
    });
  }

  async confirmMatch(matchHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'confirm_match',
      payload: { match_hash: matchHash, new_status: 'Confirmed' } as UpdateMatchStatusInput,
    });
  }

  async completeRide(matchHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'complete_ride',
      payload: matchHash,
    });
  }

  async cancelRide(matchHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'cancel_ride',
      payload: matchHash,
    });
  }

  async postCargoOffer(offer: CargoOffer): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'post_cargo_offer',
      payload: offer,
    });
  }

  async getAvailableRides(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'get_available_rides',
      payload: null,
    });
  }

  async getMyRides(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'get_my_rides',
      payload: null,
    });
  }

  // --- Impact ---

  async logTrip(trip: TripLog): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'log_trip',
      payload: trip,
    });
  }

  async getMyTrips(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_my_trips',
      payload: null,
    });
  }

  async getMyCarbonCredits(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_my_carbon_credits',
      payload: null,
    });
  }

  async calculateEmissions(input: EmissionsCalcInput): Promise<EmissionsCalcResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'calculate_emissions',
      payload: input,
    });
  }

  async getVehicleTripHistory(vehicleHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_vehicle_trip_history',
      payload: vehicleHash,
    });
  }

  async getCommunityImpactSummary(): Promise<CommunityImpactSummary> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_community_impact_summary',
      payload: null,
    });
  }

  // --- Maintenance (Phase 3) ---

  async logMaintenance(record: MaintenanceRecord): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'log_maintenance',
      payload: record,
    });
  }

  async getVehicleMaintenance(vehicleHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_vehicle_maintenance',
      payload: vehicleHash,
    });
  }

  async setVehicleFeatures(features: VehicleFeatures): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'set_vehicle_features',
      payload: features,
    });
  }

  async getAccessibleVehicles(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_accessible_vehicles',
      payload: null,
    });
  }

  async getVehiclesNeedingMaintenance(currentTime: number): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_routes',
      fn_name: 'get_vehicles_needing_maintenance',
      payload: currentTime,
    });
  }

  // --- Reviews (Phase 2) ---

  async reviewRide(review: RideReview): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'review_ride',
      payload: review,
    });
  }

  async getRideReviews(matchHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'get_ride_reviews',
      payload: matchHash,
    });
  }

  async getDriverRating(driver: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'get_driver_rating',
      payload: driver,
    });
  }

  async findNearbyRides(input: FindNearbyInput): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_sharing',
      fn_name: 'find_nearby_rides',
      payload: input,
    });
  }

  // --- Carbon Redemption (Phase 2) ---

  async redeemCredits(input: RedeemInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'redeem_credits',
      payload: input,
    });
  }

  async getMyRedemptions(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_my_redemptions',
      payload: null,
    });
  }

  async getAgentCarbonBalance(agent: Uint8Array): Promise<CarbonBalance> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'transport_impact',
      fn_name: 'get_agent_carbon_balance',
      payload: agent,
    });
  }
}

// ============================================================================
// Factory
// ============================================================================

/** Create a TransportClient from an AppWebsocket or compatible client */
export function createTransportClient(client: ZomeCallable): TransportClient {
  return new TransportClient(client);
}
