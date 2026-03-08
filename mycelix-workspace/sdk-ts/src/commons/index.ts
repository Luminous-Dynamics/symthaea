/**
 * @mycelix/sdk Universal Commons Framework
 *
 * Generic infrastructure for any commons resource:
 * Food, Water, Energy, Shelter, Medicine - all instances of ONE pattern.
 *
 * Core concepts:
 * - CommonsResource<T>: Universal container for shared resources
 * - StewardshipService: Role-based custodianship with reputation
 * - MetabolicEngine: Production/consumption/allocation tracking
 * - GovernanceEngine: Reputation-weighted decision making
 *
 * @packageDocumentation
 * @module commons
 */

import { LocalBridge } from '../bridge/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  type ReputationScore,
} from '../matl/index.js';


// ============================================================================
// Resource Types
// ============================================================================

/** Supported resource domains */
export type ResourceType = 'food' | 'water' | 'energy' | 'shelter' | 'medicine';

/** Base metadata interface for extension */
export interface ResourceMetadata {
  [key: string]: unknown;
}

/** Steward roles (universal across domains) */
export type StewardRole =
  | 'lead_steward'
  | 'steward'
  | 'volunteer'
  | 'apprentice'
  | 'beneficiary'
  | 'trustee'
  | 'custodian';

/** Domain-specific role name mappings */
export const ROLE_NAMES: Record<ResourceType, Partial<Record<StewardRole, string>>> = {
  food: {
    lead_steward: 'Head Farmer',
    steward: 'Farmer',
    volunteer: 'Garden Helper',
    beneficiary: 'CSA Member',
  },
  water: {
    lead_steward: 'Watershed Manager',
    steward: 'Water Keeper',
    volunteer: 'Monitor',
    beneficiary: 'Water User',
  },
  energy: {
    lead_steward: 'Grid Coordinator',
    steward: 'Prosumer',
    volunteer: 'Meter Reader',
    beneficiary: 'Consumer',
  },
  shelter: {
    lead_steward: 'Land Trust Steward',
    steward: 'Building Manager',
    volunteer: 'Maintenance',
    beneficiary: 'Resident',
  },
  medicine: {
    lead_steward: 'Health Collective Lead',
    steward: 'Provider',
    volunteer: 'Community Health Worker',
    beneficiary: 'Patient',
  },
};

// ============================================================================
// Core Interfaces
// ============================================================================

/**
 * Universal container for any commons resource.
 * Extend metadata type T for domain-specific data.
 */
export interface CommonsResource<T extends ResourceMetadata = ResourceMetadata> {
  // Identity
  id: string;
  type: ResourceType;
  name: string;
  description?: string;

  // Quantification
  quantity: number;
  unit: string;
  quality?: number; // 0-1 quality score

  // Ownership/Stewardship
  commonsId?: string;
  stewards: StewardProfile[];

  // Access Rights
  accessRights: AccessRights;
  allocations: Allocation[];

  // Metabolic Accounting
  production: MetabolicFlow[];
  consumption: MetabolicFlow[];
  circularCredits: CircularCredit[];

  // Governance
  governanceRules: GovernanceRule[];

  // Trust & Verification
  trustScore: number;
  epistemicClaim?: EpistemicClaim;

  // Cross-Domain
  bridges: BridgeRegistration[];

  // Domain-Specific
  metadata: T;

  // Temporal
  createdAt: number;
  updatedAt: number;
}

/** Steward profile (universal across domains) */
export interface StewardProfile {
  id: string;
  did: string;
  name: string;
  role: StewardRole;

  // Contributions
  contributions: ContributionRecord[];
  hoursContributed: number;

  // Earned Rights
  allocationRights: number; // percentage or absolute
  harvestRights?: number;   // For food commons

  // Reputation
  reputation: ReputationScore;
  trustScore: number;

  // Temporal
  joinedAt: number;
  lastActivity: number;
}

/** Contribution record (stewardship input) */
export interface ContributionRecord {
  id: string;
  stewardId: string;
  date: number;
  hours: number;
  activityType: string;
  description: string;
  witnesses: string[];
  verified: boolean;
}

/** Access rights for commons resources */
export interface AccessRights {
  canAccess: boolean;
  canAllocate: boolean;
  canGovern: boolean;
  canDelegate: boolean;
  canModify: boolean;
  allocationCeiling?: number; // Max allocation
  reciprocalObligation?: ReciprocityRule;
}

/** Reciprocity rules (give to receive) */
export interface ReciprocityRule {
  type: 'hours' | 'contribution' | 'payment' | 'stewardship';
  requirement: number;
  period: 'week' | 'month' | 'season' | 'year';
  description: string;
}

/** Resource allocation */
export interface Allocation {
  id: string;
  resourceId: string;
  recipientId: string;
  recipientDid: string;
  quantity: number;
  unit: string;
  allocationType: 'earned' | 'need' | 'lottery' | 'purchased' | 'gifted';
  effectiveFrom: number;
  effectiveUntil?: number;
  status: 'pending' | 'active' | 'consumed' | 'expired' | 'revoked';
}

// ============================================================================
// Metabolic Accounting
// ============================================================================

/** Production/consumption flow record */
export interface MetabolicFlow {
  id: string;
  resourceType: ResourceType;
  flowType: 'production' | 'consumption' | 'transfer' | 'loss' | 'regeneration';

  // What
  quantity: number;
  unit: string;

  // Who/Where
  actorId: string;
  actorDid: string;
  source?: string;
  destination?: string;
  commonsId?: string;

  // When
  timestamp: number;

  // Impact
  carbonImpact?: number; // kg CO2 (negative = sequestered)
  socialImpact?: number; // community benefit score
  ecologicalImpact?: number; // ecosystem health impact

  // Verification
  evidence: Evidence[];
  epistemicLevel: EmpiricalLevel;
  verified: boolean;
}

/** Evidence for metabolic flows */
export interface Evidence {
  type: 'witness' | 'sensor' | 'photo' | 'document' | 'certification';
  data: Record<string, unknown>;
  timestamp: number;
  verified: boolean;
  verifiedBy?: string;
}

/** Circular credit (earned through production/stewardship) */
export interface CircularCredit {
  id: string;
  resourceType: ResourceType;
  creditType: string; // 'TEND', 'H2O', 'kWh', etc.

  // Earned
  earnedBy: string;
  earnedByDid: string;
  earnedAmount: number;
  earnedThrough: 'production' | 'stewardship' | 'contribution' | 'conservation' | 'regeneration';
  earnedAt: number;

  // Spent/Transferred
  currentBalance: number;
  transfers: CreditTransfer[];

  // Temporal
  expiresAt?: number;

  // Verification
  epistemicClaim?: EpistemicClaim;
}

/** Credit transfer record */
export interface CreditTransfer {
  id: string;
  fromId: string;
  toId: string;
  amount: number;
  reason: string;
  timestamp: number;
}

// ============================================================================
// Governance
// ============================================================================

/** Governance rule for commons */
export interface GovernanceRule {
  id: string;
  commonsId: string;
  type: GovernanceRuleType;
  description: string;

  // Decision parameters
  votingThreshold?: number; // 0-1 for approval
  quorumRequirement?: number; // minimum participation
  decisionMethod?: 'majority' | 'supermajority' | 'consensus' | 'conviction';

  // Enforcement
  enforcement: 'automatic' | 'manual' | 'reputation';
  consequence?: string;

  // Metadata
  effectiveDate: number;
  proposalId?: string;
  createdBy: string;
}

export type GovernanceRuleType =
  | 'allocation'
  | 'membership'
  | 'stewardship'
  | 'contribution'
  | 'decision_making'
  | 'dispute_resolution'
  | 'resource_use'
  | 'environmental'
  | 'reciprocity';

// ============================================================================
// Cross-Domain Bridge
// ============================================================================

/** Bridge registration for cross-hApp coordination */
export interface BridgeRegistration {
  id: string;
  sourceHapp: string;
  targetHapp: string;
  resourceId: string;
  bridgeType: 'reputation' | 'credit' | 'governance' | 'data';
  status: 'active' | 'suspended' | 'terminated';
  registeredAt: number;
}

// ============================================================================
// Bioregional Context
// ============================================================================

/** Bioregion data for place-based sovereignty */
export interface Bioregion {
  id: string;
  name: string;
  ecoregionCode: string;

  // Geographic
  coordinates: { lat: number; lng: number };
  boundingBox?: { north: number; south: number; east: number; west: number };

  // Climate
  hardinessZone?: number;
  annualPrecipitation?: number; // mm
  growingDegreeDays?: number;
  aridity?: number;

  // Soil
  dominantSoilType?: string;
  soilPh?: number;
  soilFertility?: number;

  // Ecosystem
  nativeSpecies?: string[];
  watershedName?: string;

  // Traditional Knowledge
  ancestralPeoples?: string[];
  traditionalPractices?: TraditionalPractice[];

  // Sovereignty Scores
  foodSovereigntyScore?: number;
  waterSovereigntyScore?: number;
  energySovereigntyScore?: number;
}

/** Traditional practice record */
export interface TraditionalPractice {
  id: string;
  name: string;
  description: string;
  resourceType: ResourceType;
  culturalSource: string;
  consentGiven: boolean;
  attributionRequired: boolean;
  commercialUseAllowed: boolean;
}

// ============================================================================
// Wisdom Layer (Consciousness Integration)
// ============================================================================

/** Eight Harmonies for consciousness-gated decisions */
export enum Harmony {
  ResonantCoherence = 0,        // Integration-Knowing
  PanSentientFlourishing = 1,   // Care-Knowing
  IntegralWisdom = 2,           // Truth-Knowing
  InfinitePlay = 3,             // Creative-Knowing
  UniversalInterconnectedness = 4, // Relational-Knowing
  SacredReciprocity = 5,        // Exchange-Knowing
  EvolutionaryProgression = 6,  // Developmental-Knowing
}

/** Wisdom guidance for commons decisions */
export interface CommonsWisdom {
  resourceId: string;
  commonsId?: string;

  // Epistemic classification with Harmonic dimension
  epistemicClaim: EpistemicClaim;
  affectedHarmonies: Harmony[];

  // Moral uncertainty (consciousness-aware)
  moralUncertainty: {
    epistemic: number;   // Uncertainty about FACTS (0-1)
    axiological: number; // Uncertainty about VALUES (0-1)
    deontic: number;     // Uncertainty about RIGHT ACTION (0-1)
  };

  // Guidance
  actionGuidance: 'proceed_confidently' | 'proceed_with_monitoring' | 'pause_for_reflection' | 'seek_consultation' | 'defer_action';

  // Traditional knowledge connection
  traditionalKnowledge?: TraditionalPractice[];

  // Calibration (self-correcting confidence)
  calibrationScore?: number;
}

// ============================================================================
// Universal Commons Service
// ============================================================================

/**
 * UniversalCommonsService - Generic service for any commons resource type.
 *
 * @example
 * ```typescript
 * // Create a food commons service
 * const foodCommons = new UniversalCommonsService<FoodMetadata>('food');
 *
 * // Create a water commons service
 * const waterCommons = new UniversalCommonsService<WaterMetadata>('water');
 *
 * // Same operations work for both
 * foodCommons.addSteward(commonsId, steward);
 * waterCommons.recordProduction(flow);
 * ```
 */
export class UniversalCommonsService<T extends ResourceMetadata = ResourceMetadata> {
  protected resourceType: ResourceType;
  protected bridge: LocalBridge;

  // Storage (in production, these would be Holochain)
  protected resources: Map<string, CommonsResource<T>> = new Map();
  protected stewards: Map<string, StewardProfile> = new Map();
  protected allocations: Map<string, Allocation> = new Map();
  protected credits: Map<string, CircularCredit> = new Map();
  protected flows: Map<string, MetabolicFlow> = new Map();
  protected commons: Map<string, CommonsDefinition> = new Map();

  constructor(resourceType: ResourceType) {
    this.resourceType = resourceType;
    this.bridge = new LocalBridge();
    this.bridge.registerHapp(resourceType);
  }

  // ===========================================================================
  // Commons Management
  // ===========================================================================

  /**
   * Create a new commons for shared resource management
   */
  createCommons(params: {
    name: string;
    description: string;
    location?: string;
    bioregion?: Bioregion;
    initialStewards: Array<{
      did: string;
      name: string;
      role: StewardRole;
      allocationRights: number;
    }>;
    governanceRules?: GovernanceRule[];
  }): CommonsDefinition {
    const commons: CommonsDefinition = {
      id: `commons-${this.resourceType}-${Date.now()}`,
      resourceType: this.resourceType,
      name: params.name,
      description: params.description,
      location: params.location,
      bioregion: params.bioregion,
      stewards: params.initialStewards.map((s, i) => ({
        id: `steward-${i}-${Date.now()}`,
        did: s.did,
        name: s.name,
        role: s.role,
        contributions: [],
        hoursContributed: 0,
        allocationRights: s.allocationRights,
        reputation: createReputation(s.did),
        trustScore: 0.5,
        joinedAt: Date.now(),
        lastActivity: Date.now(),
      })),
      governanceRules: params.governanceRules || this.getDefaultGovernanceRules(),
      resources: [],
      createdAt: Date.now(),
    };

    this.commons.set(commons.id, commons);

    // Register stewards
    for (const steward of commons.stewards) {
      this.stewards.set(steward.id, steward);
    }

    return commons;
  }

  /**
   * Add a steward to a commons
   */
  addSteward(commonsId: string, steward: Omit<StewardProfile, 'id' | 'contributions' | 'hoursContributed' | 'reputation' | 'trustScore' | 'joinedAt' | 'lastActivity'>): StewardProfile {
    const commons = this.commons.get(commonsId);
    if (!commons) throw new Error('Commons not found');

    const newSteward: StewardProfile = {
      ...steward,
      id: `steward-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      contributions: [],
      hoursContributed: 0,
      reputation: createReputation(steward.did),
      trustScore: 0.5,
      joinedAt: Date.now(),
      lastActivity: Date.now(),
    };

    commons.stewards.push(newSteward);
    this.stewards.set(newSteward.id, newSteward);

    return newSteward;
  }

  /**
   * Record a stewardship contribution
   */
  recordContribution(params: {
    stewardId: string;
    commonsId: string;
    hours: number;
    activityType: string;
    description: string;
    witnesses?: string[];
  }): ContributionRecord {
    const steward = this.stewards.get(params.stewardId);
    if (!steward) throw new Error('Steward not found');

    const contribution: ContributionRecord = {
      id: `contrib-${Date.now()}`,
      stewardId: params.stewardId,
      date: Date.now(),
      hours: params.hours,
      activityType: params.activityType,
      description: params.description,
      witnesses: params.witnesses || [],
      verified: (params.witnesses?.length || 0) > 0,
    };

    steward.contributions.push(contribution);
    steward.hoursContributed += params.hours;
    steward.lastActivity = Date.now();
    steward.reputation = recordPositive(steward.reputation);
    steward.trustScore = reputationValue(steward.reputation);

    return contribution;
  }

  // ===========================================================================
  // Resource Management
  // ===========================================================================

  /**
   * Register a resource in the commons
   */
  registerResource(params: {
    name: string;
    description?: string;
    quantity: number;
    unit: string;
    quality?: number;
    commonsId?: string;
    metadata: T;
    accessRights?: Partial<AccessRights>;
  }): CommonsResource<T> {
    const resource: CommonsResource<T> = {
      id: `resource-${this.resourceType}-${Date.now()}`,
      type: this.resourceType,
      name: params.name,
      description: params.description,
      quantity: params.quantity,
      unit: params.unit,
      quality: params.quality ?? 1.0,
      commonsId: params.commonsId,
      stewards: params.commonsId ? (this.commons.get(params.commonsId)?.stewards || []) : [],
      accessRights: {
        canAccess: true,
        canAllocate: false,
        canGovern: false,
        canDelegate: false,
        canModify: false,
        ...params.accessRights,
      },
      allocations: [],
      production: [],
      consumption: [],
      circularCredits: [],
      governanceRules: [],
      trustScore: 0.5,
      bridges: [],
      metadata: params.metadata,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.resources.set(resource.id, resource);

    // Add to commons if specified
    if (params.commonsId) {
      const commons = this.commons.get(params.commonsId);
      if (commons) {
        commons.resources.push(resource.id);
      }
    }

    return resource;
  }

  /**
   * Get a resource by ID
   */
  getResource(resourceId: string): CommonsResource<T> | undefined {
    return this.resources.get(resourceId);
  }

  /**
   * List resources, optionally filtered by commons
   */
  listResources(commonsId?: string): CommonsResource<T>[] {
    const all = Array.from(this.resources.values());
    if (commonsId) {
      return all.filter(r => r.commonsId === commonsId);
    }
    return all;
  }

  // ===========================================================================
  // Metabolic Accounting
  // ===========================================================================

  /**
   * Record production of a resource
   */
  recordProduction(params: {
    resourceId: string;
    quantity: number;
    actorId: string;
    actorDid: string;
    source?: string;
    evidence?: Evidence[];
    carbonImpact?: number;
  }): MetabolicFlow {
    const resource = this.resources.get(params.resourceId);
    if (!resource) throw new Error('Resource not found');

    const flow: MetabolicFlow = {
      id: `flow-${Date.now()}`,
      resourceType: this.resourceType,
      flowType: 'production',
      quantity: params.quantity,
      unit: resource.unit,
      actorId: params.actorId,
      actorDid: params.actorDid,
      source: params.source,
      commonsId: resource.commonsId,
      timestamp: Date.now(),
      carbonImpact: params.carbonImpact,
      evidence: params.evidence || [],
      epistemicLevel: params.evidence?.length ? EmpiricalLevel.E2_PrivateVerify : EmpiricalLevel.E1_Testimonial,
      verified: (params.evidence?.length || 0) > 0,
    };

    this.flows.set(flow.id, flow);
    resource.production.push(flow);
    resource.quantity += params.quantity;
    resource.updatedAt = Date.now();

    // Issue circular credits for production
    this.issueCredits({
      earnedBy: params.actorId,
      earnedByDid: params.actorDid,
      amount: params.quantity,
      earnedThrough: 'production',
    });

    return flow;
  }

  /**
   * Record consumption of a resource
   */
  recordConsumption(params: {
    resourceId: string;
    quantity: number;
    actorId: string;
    actorDid: string;
    destination?: string;
    evidence?: Evidence[];
  }): MetabolicFlow {
    const resource = this.resources.get(params.resourceId);
    if (!resource) throw new Error('Resource not found');

    if (resource.quantity < params.quantity) {
      throw new Error('Insufficient resource quantity');
    }

    const flow: MetabolicFlow = {
      id: `flow-${Date.now()}`,
      resourceType: this.resourceType,
      flowType: 'consumption',
      quantity: params.quantity,
      unit: resource.unit,
      actorId: params.actorId,
      actorDid: params.actorDid,
      destination: params.destination,
      commonsId: resource.commonsId,
      timestamp: Date.now(),
      evidence: params.evidence || [],
      epistemicLevel: EmpiricalLevel.E1_Testimonial,
      verified: true,
    };

    this.flows.set(flow.id, flow);
    resource.consumption.push(flow);
    resource.quantity -= params.quantity;
    resource.updatedAt = Date.now();

    return flow;
  }

  // ===========================================================================
  // Circular Credits
  // ===========================================================================

  /**
   * Issue circular credits for production/stewardship
   */
  issueCredits(params: {
    earnedBy: string;
    earnedByDid: string;
    amount: number;
    earnedThrough: CircularCredit['earnedThrough'];
  }): CircularCredit {
    const creditType = this.getCreditType();

    const credit: CircularCredit = {
      id: `credit-${Date.now()}`,
      resourceType: this.resourceType,
      creditType,
      earnedBy: params.earnedBy,
      earnedByDid: params.earnedByDid,
      earnedAmount: params.amount,
      earnedThrough: params.earnedThrough,
      earnedAt: Date.now(),
      currentBalance: params.amount,
      transfers: [],
      epistemicClaim: claim(`${creditType} credits earned through ${params.earnedThrough}`)
        .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N2_Network)
        .withMateriality(MaterialityLevel.M1_Temporal)
        .build(),
    };

    this.credits.set(credit.id, credit);
    return credit;
  }

  /**
   * Transfer credits between actors
   */
  transferCredits(params: {
    creditId: string;
    fromId: string;
    toId: string;
    amount: number;
    reason: string;
  }): CreditTransfer {
    const credit = this.credits.get(params.creditId);
    if (!credit) throw new Error('Credit not found');

    if (credit.currentBalance < params.amount) {
      throw new Error('Insufficient credit balance');
    }

    const transfer: CreditTransfer = {
      id: `transfer-${Date.now()}`,
      fromId: params.fromId,
      toId: params.toId,
      amount: params.amount,
      reason: params.reason,
      timestamp: Date.now(),
    };

    credit.transfers.push(transfer);
    credit.currentBalance -= params.amount;

    // Create new credit for recipient
    this.issueCredits({
      earnedBy: params.toId,
      earnedByDid: `did:mycelix:${params.toId}`,
      amount: params.amount,
      earnedThrough: 'contribution', // Transfer counts as contribution
    });

    return transfer;
  }

  /**
   * Get credit balance for an actor
   */
  getCreditBalance(actorId: string): number {
    return Array.from(this.credits.values())
      .filter(c => c.earnedBy === actorId)
      .reduce((sum, c) => sum + c.currentBalance, 0);
  }

  // ===========================================================================
  // Allocations
  // ===========================================================================

  /**
   * Create an allocation for a resource
   */
  createAllocation(params: {
    resourceId: string;
    recipientId: string;
    recipientDid: string;
    quantity: number;
    allocationType: Allocation['allocationType'];
    effectiveFrom?: number;
    effectiveUntil?: number;
  }): Allocation {
    const resource = this.resources.get(params.resourceId);
    if (!resource) throw new Error('Resource not found');

    const allocation: Allocation = {
      id: `alloc-${Date.now()}`,
      resourceId: params.resourceId,
      recipientId: params.recipientId,
      recipientDid: params.recipientDid,
      quantity: params.quantity,
      unit: resource.unit,
      allocationType: params.allocationType,
      effectiveFrom: params.effectiveFrom ?? Date.now(),
      effectiveUntil: params.effectiveUntil,
      status: 'active',
    };

    this.allocations.set(allocation.id, allocation);
    resource.allocations.push(allocation);

    return allocation;
  }

  /**
   * Get allocations for a recipient
   */
  getAllocations(recipientId: string): Allocation[] {
    return Array.from(this.allocations.values())
      .filter(a => a.recipientId === recipientId && a.status === 'active');
  }

  // ===========================================================================
  // Sovereignty Metrics
  // ===========================================================================

  /**
   * Calculate sovereignty score for a commons
   */
  calculateSovereigntyScore(commonsId: string): number {
    const commons = this.commons.get(commonsId);
    if (!commons) return 0;

    const resources = this.listResources(commonsId);
    const stewards = commons.stewards;

    // Factors:
    // 1. Resource availability (quantity vs demand)
    // 2. Stewardship engagement (active stewards, hours)
    // 3. Governance health (rules, participation)
    // 4. Circular economy strength (credits in circulation)
    // 5. Trust scores

    const resourceScore = resources.length > 0 ? Math.min(resources.reduce((sum, r) => sum + r.quantity, 0) / 100, 1) : 0;
    const stewardScore = stewards.length > 0 ? Math.min(stewards.length / 10, 1) : 0;
    const engagementScore = stewards.reduce((sum, s) => sum + s.hoursContributed, 0) / (stewards.length * 100) || 0;
    const governanceScore = commons.governanceRules.length > 0 ? Math.min(commons.governanceRules.length / 5, 1) : 0;
    const trustScore = stewards.reduce((sum, s) => sum + s.trustScore, 0) / stewards.length || 0.5;

    const sovereigntyScore = (
      resourceScore * 0.25 +
      stewardScore * 0.2 +
      engagementScore * 0.2 +
      governanceScore * 0.15 +
      trustScore * 0.2
    );

    return Math.round(sovereigntyScore * 100);
  }

  // ===========================================================================
  // Wisdom Layer
  // ===========================================================================

  /**
   * Get wisdom guidance for a resource decision
   */
  getWisdomGuidance(params: {
    resourceId: string;
    decision: string;
    context?: Record<string, unknown>;
  }): CommonsWisdom {
    const resource = this.resources.get(params.resourceId);

    // Identify affected harmonies based on decision type
    const affectedHarmonies: Harmony[] = [];

    if (params.decision.includes('allocat') || params.decision.includes('share')) {
      affectedHarmonies.push(Harmony.SacredReciprocity);
    }
    if (params.decision.includes('sustain') || params.decision.includes('regenerat')) {
      affectedHarmonies.push(Harmony.EvolutionaryProgression);
      affectedHarmonies.push(Harmony.PanSentientFlourishing);
    }
    if (params.decision.includes('communit') || params.decision.includes('collective')) {
      affectedHarmonies.push(Harmony.UniversalInterconnectedness);
    }

    // Calculate moral uncertainty
    const moralUncertainty = {
      epistemic: 0.3,   // Generally low - we know facts about resources
      axiological: 0.5, // Medium - values vary by community
      deontic: 0.4,     // Medium - right action depends on context
    };

    // Determine action guidance
    const totalUncertainty = (moralUncertainty.epistemic + moralUncertainty.axiological + moralUncertainty.deontic) / 3;
    let actionGuidance: CommonsWisdom['actionGuidance'];

    if (totalUncertainty < 0.3) {
      actionGuidance = 'proceed_confidently';
    } else if (totalUncertainty < 0.5) {
      actionGuidance = 'proceed_with_monitoring';
    } else if (totalUncertainty < 0.7) {
      actionGuidance = 'pause_for_reflection';
    } else {
      actionGuidance = 'seek_consultation';
    }

    return {
      resourceId: params.resourceId,
      commonsId: resource?.commonsId,
      epistemicClaim: claim(`Wisdom guidance for ${params.decision}`)
        .withEmpirical(EmpiricalLevel.E1_Testimonial)
        .withNormative(NormativeLevel.N1_Communal)
        .withMateriality(MaterialityLevel.M1_Temporal)
        .build(),
      affectedHarmonies,
      moralUncertainty,
      actionGuidance,
    };
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  protected getCreditType(): string {
    const creditTypes: Record<ResourceType, string> = {
      food: 'TEND',
      water: 'H2O',
      energy: 'kWh',
      shelter: 'HOME',
      medicine: 'CARE',
    };
    return creditTypes[this.resourceType];
  }

  protected getDefaultGovernanceRules(): GovernanceRule[] {
    return [
      {
        id: 'default-allocation',
        commonsId: '',
        type: 'allocation',
        description: 'Allocations proportional to stewardship contributions',
        enforcement: 'automatic',
        effectiveDate: Date.now(),
        createdBy: 'system',
      },
      {
        id: 'default-decision',
        commonsId: '',
        type: 'decision_making',
        description: 'Major decisions require 2/3 majority of stewards',
        votingThreshold: 0.67,
        quorumRequirement: 0.5,
        decisionMethod: 'supermajority',
        enforcement: 'manual',
        effectiveDate: Date.now(),
        createdBy: 'system',
      },
      {
        id: 'default-reciprocity',
        commonsId: '',
        type: 'reciprocity',
        description: 'Stewards contribute minimum 4 hours/month to maintain allocation rights',
        enforcement: 'reputation',
        effectiveDate: Date.now(),
        createdBy: 'system',
      },
    ];
  }
}

// ============================================================================
// Commons Definition
// ============================================================================

/** Commons definition (container for resources and stewards) */
export interface CommonsDefinition {
  id: string;
  resourceType: ResourceType;
  name: string;
  description: string;
  location?: string;
  bioregion?: Bioregion;
  stewards: StewardProfile[];
  governanceRules: GovernanceRule[];
  resources: string[]; // Resource IDs
  createdAt: number;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a commons service for a specific resource type
 */
export function createCommonsService<T extends ResourceMetadata = ResourceMetadata>(
  resourceType: ResourceType
): UniversalCommonsService<T> {
  return new UniversalCommonsService<T>(resourceType);
}

// ============================================================================
// Re-exports
// ============================================================================

export {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
};
