/**
 * @mycelix/sdk Bridge Routing Types
 *
 * Type-safe domain and zome routing types that mirror the Rust enums
 * in `mycelix-bridge-common/src/routing.rs`. These ensure compile-time
 * safety when dispatching cross-domain and cross-cluster calls.
 *
 * @packageDocumentation
 * @module integrations/bridge-routing
 */

// ============================================================================
// Bridge Domains
// ============================================================================

/** All 12 bridge domains across Commons and Civic clusters. */
export type BridgeDomain =
  | 'property'
  | 'housing'
  | 'care'
  | 'mutualaid'
  | 'water'
  | 'food'
  | 'transport'
  | 'support'
  | 'space'
  | 'justice'
  | 'emergency'
  | 'media';

/** Domains served by the Commons cluster (9 domains). */
export type CommonsDomain =
  | 'property'
  | 'housing'
  | 'care'
  | 'mutualaid'
  | 'water'
  | 'food'
  | 'transport'
  | 'support'
  | 'space';

/** Domains served by the Civic cluster (3 domains). */
export type CivicDomain = 'justice' | 'emergency' | 'media';

/** All commons domains as a readonly array. */
export const COMMONS_DOMAINS: readonly CommonsDomain[] = [
  'property',
  'housing',
  'care',
  'mutualaid',
  'water',
  'food',
  'transport',
  'support',
  'space',
] as const;

/** All civic domains as a readonly array. */
export const CIVIC_DOMAINS: readonly CivicDomain[] = [
  'justice',
  'emergency',
  'media',
] as const;

// ============================================================================
// Commons Zomes (38 total)
// ============================================================================

/** All 38 zome names in the Commons cluster. */
export type CommonsZome =
  // Property domain (4)
  | 'property_registry'
  | 'property_transfer'
  | 'property_disputes'
  | 'property_commons'
  // Housing domain (6)
  | 'housing_units'
  | 'housing_membership'
  | 'housing_finances'
  | 'housing_maintenance'
  | 'housing_clt'
  | 'housing_governance'
  // Care domain (5)
  | 'care_timebank'
  | 'care_circles'
  | 'care_matching'
  | 'care_plans'
  | 'care_credentials'
  // Mutual aid domain (7)
  | 'mutualaid_needs'
  | 'mutualaid_circles'
  | 'mutualaid_governance'
  | 'mutualaid_pools'
  | 'mutualaid_requests'
  | 'mutualaid_resources'
  | 'mutualaid_timebank'
  // Water domain (5)
  | 'water_flow'
  | 'water_purity'
  | 'water_capture'
  | 'water_steward'
  | 'water_wisdom'
  // Food domain (4)
  | 'food_production'
  | 'food_distribution'
  | 'food_preservation'
  | 'food_knowledge'
  // Transport domain (3)
  | 'transport_routes'
  | 'transport_sharing'
  | 'transport_impact'
  // Support domain (3)
  | 'support_knowledge'
  | 'support_tickets'
  | 'support_diagnostics'
  // Space (1)
  | 'space';

// ============================================================================
// Civic Zomes (15 total)
// ============================================================================

/** All 15 zome names in the Civic cluster. */
export type CivicZome =
  // Justice domain (5)
  | 'justice_cases'
  | 'justice_evidence'
  | 'justice_arbitration'
  | 'justice_restorative'
  | 'justice_enforcement'
  // Emergency domain (6)
  | 'emergency_incidents'
  | 'emergency_triage'
  | 'emergency_resources'
  | 'emergency_coordination'
  | 'emergency_shelters'
  | 'emergency_comms'
  // Media domain (4)
  | 'media_publication'
  | 'media_attribution'
  | 'media_factcheck'
  | 'media_curation';

// ============================================================================
// Cross-Cluster Roles
// ============================================================================

/** Cluster roles for cross-cluster dispatch. */
export type CrossClusterRole = 'commons' | 'civic' | 'identity';

// ============================================================================
// Default Zome Lookups
// ============================================================================

/** Default zome for each commons domain (used when no keyword match). */
export const COMMONS_DEFAULT_ZOME: Record<CommonsDomain, CommonsZome> = {
  property: 'property_registry',
  housing: 'housing_units',
  care: 'care_timebank',
  mutualaid: 'mutualaid_needs',
  water: 'water_flow',
  food: 'food_production',
  transport: 'transport_routes',
  support: 'support_knowledge',
  space: 'space',
};

/** Default zome for each civic domain (used when no keyword match). */
export const CIVIC_DEFAULT_ZOME: Record<CivicDomain, CivicZome> = {
  justice: 'justice_cases',
  emergency: 'emergency_incidents',
  media: 'media_publication',
};

// ============================================================================
// Consciousness Profile Types
// ============================================================================

/** 4-dimensional consciousness profile. */
export interface ConsciousnessProfile {
  /** Identity verification strength (0.0-1.0) */
  identity: number;
  /** Cross-hApp reputation (0.0-1.0) */
  reputation: number;
  /** Community trust attestations (0.0-1.0) */
  community: number;
  /** Domain-specific engagement (0.0-1.0) */
  engagement: number;
}

/** Consciousness tiers — mirrors Rust ConsciousnessTier enum. */
export type ConsciousnessTier =
  | 'Observer'
  | 'Participant'
  | 'Citizen'
  | 'Steward'
  | 'Guardian';

/** Time-limited credential containing a ConsciousnessProfile. */
export interface ConsciousnessCredential {
  did: string;
  profile: ConsciousnessProfile;
  tier: ConsciousnessTier;
  issued_at: number;
  expires_at: number;
  issuer: string;
}

/** Result of evaluating a profile against a governance requirement. */
export interface GovernanceEligibility {
  eligible: boolean;
  weight_bp: number;
  tier: ConsciousnessTier;
  profile: ConsciousnessProfile;
  reasons: string[];
}

// ============================================================================
// Type Guards
// ============================================================================

/** Check if a string is a valid BridgeDomain. */
export function isBridgeDomain(value: string): value is BridgeDomain {
  return [...COMMONS_DOMAINS, ...CIVIC_DOMAINS].includes(value as BridgeDomain);
}

/** Check if a string is a valid CommonsDomain. */
export function isCommonsDomain(value: string): value is CommonsDomain {
  return COMMONS_DOMAINS.includes(value as CommonsDomain);
}

/** Check if a string is a valid CivicDomain. */
export function isCivicDomain(value: string): value is CivicDomain {
  return CIVIC_DOMAINS.includes(value as CivicDomain);
}

/** Tier score thresholds (matches Rust ConsciousnessTier::min_score). */
export const TIER_THRESHOLDS: Record<ConsciousnessTier, number> = {
  Observer: 0.0,
  Participant: 0.3,
  Citizen: 0.4,
  Steward: 0.6,
  Guardian: 0.8,
};

/** Progressive vote weight in basis points per tier. */
export const TIER_VOTE_WEIGHT_BP: Record<ConsciousnessTier, number> = {
  Observer: 0,
  Participant: 5000,
  Citizen: 7500,
  Steward: 10000,
  Guardian: 10000,
};
