// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Mutual Aid Integration
 *
 * hApp-specific adapter for Mycelix-MutualAid providing:
 * - Gift circle coordination and tracking
 * - Timebanking with hour-based credits
 * - Community resource pooling
 * - Needs matching and fulfillment
 * - Cross-hApp reputation for mutual aid participants
 * - MATL-based trust scoring for circle members
 *
 * @packageDocumentation
 * @module integrations/mutualaid
 * @see {@link MutualAidService} - Main service class
 * @see {@link getMutualAidService} - Singleton accessor
 *
 * @example Basic gift circle workflow
 * ```typescript
 * import { getMutualAidService } from '@mycelix/sdk/integrations/mutualaid';
 *
 * const aid = getMutualAidService();
 *
 * // Create a gift circle
 * const circle = aid.createCircle({
 *   id: 'circle-001',
 *   name: 'Neighborhood Helpers',
 *   description: 'Mutual support for our block',
 * });
 *
 * // Post a need
 * const need = aid.postNeed({
 *   circleId: 'circle-001',
 *   requesterId: 'member-001',
 *   title: 'Help moving furniture',
 *   category: 'labor',
 *   estimatedHours: 3,
 * });
 *
 * // Fulfill and log timebank hours
 * aid.fulfillNeed(need.id, 'member-002', 3);
 * ```
 */

import { LocalBridge } from '../../bridge/index.js';
import {
  createReputation,
  recordPositive,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Mutual Aid Types
// ============================================================================

/** Need category */
export type NeedCategory =
  | 'labor'
  | 'transport'
  | 'childcare'
  | 'eldercare'
  | 'food'
  | 'housing'
  | 'education'
  | 'emotional_support'
  | 'technology'
  | 'medical'
  | 'legal'
  | 'other';

/** Gift circle */
export interface GiftCircle {
  id: string;
  name: string;
  description?: string;
  memberIds: string[];
  createdAt: number;
  active: boolean;
}

/** Community need request */
export interface NeedRequest {
  id: string;
  circleId: string;
  requesterId: string;
  title: string;
  description?: string;
  category: NeedCategory;
  estimatedHours: number;
  urgent: boolean;
  status: 'open' | 'claimed' | 'fulfilled' | 'cancelled';
  claimedBy?: string;
  fulfilledAt?: number;
  createdAt: number;
}

/** Timebank ledger entry */
export interface TimebankEntry {
  id: string;
  circleId: string;
  needId: string;
  giverId: string;
  receiverId: string;
  hours: number;
  category: NeedCategory;
  timestamp: number;
  note?: string;
}

/** Resource pool contribution */
export interface ResourceContribution {
  id: string;
  circleId: string;
  contributorId: string;
  resourceType: string;
  quantity: number;
  unit: string;
  availableFrom: number;
  availableUntil?: number;
  claimed: boolean;
}

/** Member summary within a circle */
export interface MemberSummary {
  memberId: string;
  hoursGiven: number;
  hoursReceived: number;
  needsFulfilled: number;
  needsRequested: number;
  reputation: number;
}

// ============================================================================
// Mutual Aid Service
// ============================================================================

/**
 * MutualAidService - Gift circle and timebanking coordination
 *
 * Integrates with the mycelix-mutualaid hApp zomes:
 * - `circles` - Gift circle creation and membership
 * - `needs` - Need posting and matching
 * - `requests` - Request lifecycle management
 * - `timebank` - Hour-based credit tracking
 * - `resources` - Community resource pooling
 * - `pools` - Shared resource pools
 * - `governance` - Circle governance decisions
 * - `bridge` - Cross-hApp reputation sharing
 */
export class MutualAidService {
  private circles: Map<string, GiftCircle> = new Map();
  private needs: Map<string, NeedRequest> = new Map();
  private timebankEntries: TimebankEntry[] = [];
  private resources: Map<string, ResourceContribution> = new Map();
  private memberReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('mutualaid');
  }

  /**
   * Create a new gift circle
   */
  createCircle(input: { id: string; name: string; description?: string }): GiftCircle {
    const circle: GiftCircle = {
      ...input,
      memberIds: [],
      createdAt: Date.now(),
      active: true,
    };

    this.circles.set(circle.id, circle);
    return circle;
  }

  /**
   * Join a gift circle
   */
  joinCircle(circleId: string, memberId: string): GiftCircle {
    const circle = this.circles.get(circleId);
    if (!circle) {
      throw new Error(`Circle not found: ${circleId}`);
    }
    if (circle.memberIds.includes(memberId)) {
      throw new Error(`Already a member of circle: ${circleId}`);
    }

    circle.memberIds.push(memberId);

    if (!this.memberReputations.has(memberId)) {
      this.memberReputations.set(memberId, createReputation(memberId));
    }

    return circle;
  }

  /**
   * Post a need to a circle
   */
  postNeed(input: {
    circleId: string;
    requesterId: string;
    title: string;
    description?: string;
    category: NeedCategory;
    estimatedHours: number;
    urgent?: boolean;
  }): NeedRequest {
    const circle = this.circles.get(input.circleId);
    if (!circle) {
      throw new Error(`Circle not found: ${input.circleId}`);
    }

    const need: NeedRequest = {
      id: `need-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      circleId: input.circleId,
      requesterId: input.requesterId,
      title: input.title,
      description: input.description,
      category: input.category,
      estimatedHours: input.estimatedHours,
      urgent: input.urgent || false,
      status: 'open',
      createdAt: Date.now(),
    };

    this.needs.set(need.id, need);
    return need;
  }

  /**
   * Claim a need (volunteer to fulfill it)
   */
  claimNeed(needId: string, volunteerId: string): NeedRequest {
    const need = this.needs.get(needId);
    if (!need) {
      throw new Error(`Need not found: ${needId}`);
    }
    if (need.status !== 'open') {
      throw new Error(`Need not open: ${needId} (status: ${need.status})`);
    }

    need.status = 'claimed';
    need.claimedBy = volunteerId;

    return need;
  }

  /**
   * Fulfill a need and log timebank hours
   */
  fulfillNeed(needId: string, fulfillerId: string, hoursSpent: number): TimebankEntry {
    const need = this.needs.get(needId);
    if (!need) {
      throw new Error(`Need not found: ${needId}`);
    }
    if (need.status !== 'claimed' && need.status !== 'open') {
      throw new Error(`Need cannot be fulfilled: ${needId} (status: ${need.status})`);
    }

    need.status = 'fulfilled';
    need.fulfilledAt = Date.now();
    need.claimedBy = need.claimedBy || fulfillerId;

    const entry: TimebankEntry = {
      id: `tb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      circleId: need.circleId,
      needId,
      giverId: fulfillerId,
      receiverId: need.requesterId,
      hours: hoursSpent,
      category: need.category,
      timestamp: Date.now(),
    };

    this.timebankEntries.push(entry);

    // Update reputations
    const giverRep = this.memberReputations.get(fulfillerId);
    if (giverRep) {
      this.memberReputations.set(fulfillerId, recordPositive(giverRep));
    }

    return entry;
  }

  /**
   * Contribute a resource to a circle pool
   */
  contributeResource(input: Omit<ResourceContribution, 'id' | 'claimed'>): ResourceContribution {
    const resource: ResourceContribution = {
      ...input,
      id: `res-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      claimed: false,
    };

    this.resources.set(resource.id, resource);
    return resource;
  }

  /**
   * Get circle by ID
   */
  getCircle(circleId: string): GiftCircle | undefined {
    return this.circles.get(circleId);
  }

  /**
   * Get open needs for a circle
   */
  getOpenNeeds(circleId: string): NeedRequest[] {
    return Array.from(this.needs.values()).filter(
      (n) => n.circleId === circleId && n.status === 'open'
    );
  }

  /**
   * Get member summary (hours given/received, reputation)
   */
  getMemberSummary(memberId: string): MemberSummary {
    const given = this.timebankEntries.filter((e) => e.giverId === memberId);
    const received = this.timebankEntries.filter((e) => e.receiverId === memberId);
    const rep = this.memberReputations.get(memberId);

    return {
      memberId,
      hoursGiven: given.reduce((sum, e) => sum + e.hours, 0),
      hoursReceived: received.reduce((sum, e) => sum + e.hours, 0),
      needsFulfilled: given.length,
      needsRequested: received.length,
      reputation: rep ? reputationValue(rep) : 0,
    };
  }

  /**
   * Get timebank balance (hours given minus hours received)
   */
  getTimebankBalance(memberId: string): number {
    const summary = this.getMemberSummary(memberId);
    return summary.hoursGiven - summary.hoursReceived;
  }
}

// ============================================================================
// Bridge Client (Holochain Zome Calls)
// ============================================================================

import { type MycelixClient } from '../../client/index.js';

const COMMONS_CARE_ROLE = 'commons_care';

/**
 * MutualAidBridgeClient - Direct Holochain zome calls for mutual aid operations
 *
 * Provides the same capabilities as MutualAidService but backed by the
 * Holochain conductor instead of in-memory Maps.
 */
export class MutualAidBridgeClient {
  constructor(private client: MycelixClient) {}

  // --- mutualaid_needs zome ---

  async createNeed(input: {
    circle_id: string;
    requester_id: string;
    title: string;
    description?: string;
    category: NeedCategory;
    estimated_hours: number;
    urgent?: boolean;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'create_need',
      payload: input,
    });
  }

  async getNeed(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'get_need',
      payload: actionHash,
    });
  }

  async getMyNeeds(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'get_my_needs',
      payload: null,
    });
  }

  async searchNeeds(input: {
    category?: NeedCategory;
    urgent_only?: boolean;
    circle_id?: string;
  }): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'search_needs',
      payload: input,
    });
  }

  async getEmergencyNeeds(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'get_emergency_needs',
      payload: null,
    });
  }

  async withdrawNeed(needHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'withdraw_need',
      payload: needHash,
    });
  }

  async createOffer(input: {
    need_hash: Uint8Array;
    description?: string;
    estimated_hours: number;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'create_offer',
      payload: input,
    });
  }

  async getOffer(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'get_offer',
      payload: actionHash,
    });
  }

  async proposeMatch(input: {
    need_hash: Uint8Array;
    offer_hash: Uint8Array;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'propose_match',
      payload: input,
    });
  }

  async acceptMatch(matchHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'accept_match',
      payload: matchHash,
    });
  }

  async fulfillMatch(input: {
    match_hash: Uint8Array;
    hours_spent: number;
    note?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'fulfill_match',
      payload: input,
    });
  }

  async confirmFulfillment(fulfillmentHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_needs',
      fn_name: 'confirm_fulfillment',
      payload: fulfillmentHash,
    });
  }

  // --- mutualaid_requests zome ---

  async createRequest(input: {
    request_type: string;
    title: string;
    description: string;
    urgency: string;
    requester_did: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_requests',
      fn_name: 'create_request',
      payload: input,
    });
  }

  async getRequest(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_requests',
      fn_name: 'get_request',
      payload: actionHash,
    });
  }

  async getOpenRequests(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_requests',
      fn_name: 'get_open_requests',
      payload: null,
    });
  }

  async updateRequestStatus(input: {
    request_hash: Uint8Array;
    new_status: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_requests',
      fn_name: 'update_request_status',
      payload: input,
    });
  }

  async cancelRequest(requestHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_requests',
      fn_name: 'cancel_request',
      payload: requestHash,
    });
  }

  // --- mutualaid_resources zome ---

  async createResource(input: {
    name: string;
    description: string;
    resource_type: string;
    owner: Uint8Array;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'create_resource',
      payload: input,
    });
  }

  async getResource(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'get_resource',
      payload: actionHash,
    });
  }

  async getMyResources(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'get_my_resources',
      payload: null,
    });
  }

  async searchResources(input: {
    resource_type?: string;
    available_only?: boolean;
  }): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'search_resources',
      payload: input,
    });
  }

  async createBooking(input: {
    resource_hash: Uint8Array;
    start_time: number;
    end_time: number;
    purpose?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'create_booking',
      payload: input,
    });
  }

  async confirmBooking(bookingHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'confirm_booking',
      payload: bookingHash,
    });
  }

  async cancelBooking(bookingHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: COMMONS_CARE_ROLE,
      zome_name: 'mutualaid_resources',
      fn_name: 'cancel_booking',
      payload: bookingHash,
    });
  }
}

// Bridge client singleton
let mutualAidBridgeInstance: MutualAidBridgeClient | null = null;

export function getMutualAidBridgeClient(client: MycelixClient): MutualAidBridgeClient {
  if (!mutualAidBridgeInstance) mutualAidBridgeInstance = new MutualAidBridgeClient(client);
  return mutualAidBridgeInstance;
}

export function resetMutualAidBridgeClient(): void {
  mutualAidBridgeInstance = null;
}

// ============================================================================
// Singleton
// ============================================================================

let defaultService: MutualAidService | null = null;

/**
 * Get the default MutualAidService instance
 */
export function getMutualAidService(): MutualAidService {
  if (!defaultService) {
    defaultService = new MutualAidService();
  }
  return defaultService;
}

/**
 * Reset the default MutualAidService instance (for testing)
 */
export function resetMutualAidService(): void {
  defaultService = null;
}
