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
