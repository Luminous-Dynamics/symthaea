// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge SDK
 *
 * Decentralized Knowledge Graph infrastructure featuring:
 * - Epistemic Classification: 3D E-N-M cube positioning
 * - Belief Graphs: Propagate belief through connected claims
 * - MATL-Enhanced Credibility: Multi-dimensional trust assessment
 * - Bidirectional Markets Integration: Claims ↔ Verification Markets
 * - Fact-Checking API: Cross-hApp knowledge verification
 *
 * @packageDocumentation
 */

import {
  AppClient,
  AgentPubKey,
  EntryHash,
  ActionHash,
  Record as HolochainRecord,
} from "@holochain/client";

// ============================================================================
// TYPES - Epistemic Classification
// ============================================================================

/**
 * 3D Epistemic Position in the E-N-M cube
 * All values are normalized 0.0 to 1.0
 */
export interface EpistemicPosition {
  /** Empirical: How verifiable through observation? (0.0 = subjective, 1.0 = measurable) */
  empirical: number;
  /** Normative: Ethical/value alignment (0.0 = personal, 1.0 = universal) */
  normative: number;
  /** Mythic: Narrative/meaning significance (0.0 = transient, 1.0 = foundational) */
  mythic: number;
}

/** Empirical axis discrete levels */
export type EmpiricalLevel =
  | "E0_Subjective"    // No external verification
  | "E1_Testimonial"   // Witness testimony
  | "E2_PrivateVerify" // ZK proofs, credentials
  | "E3_Cryptographic" // On-chain verification
  | "E4_Measurable";   // Scientific reproducibility

/** Normative axis discrete levels */
export type NormativeLevel =
  | "N0_Personal"   // Individual perspective
  | "N1_Communal"   // Community consensus
  | "N2_Network"    // Network-wide agreement
  | "N3_Universal"; // Objective truth

/** Mythic axis discrete levels */
export type MythicLevel =
  | "M0_Transient"    // Doesn't persist
  | "M1_Temporal"     // Time-limited
  | "M2_Persistent"   // Permanent record
  | "M3_Foundational"; // Core understanding

// ============================================================================
// TYPES - Claims
// ============================================================================

/** Evidence types for claims */
export type EvidenceType =
  | "Empirical"
  | "Testimonial"
  | "Cryptographic"
  | "CrossReference";

/** Evidence supporting a claim */
export interface Evidence {
  id: string;
  evidenceType: EvidenceType;
  source: string;
  content: string;
  strength: number;
  verifiedAt?: number;
  verifier?: AgentPubKey;
}

/** Source reference for claims */
export interface Source {
  uri: string;
  title: string;
  author?: string;
  publishedAt?: number;
  reliability: number;
}

/** Core claim structure */
export interface Claim {
  id: string;
  content: string;
  author: AgentPubKey;
  classification: EpistemicPosition;
  evidence: Evidence[];
  sources: Source[];
  domain: string;
  topics: string[];
  timestamp: number;
  updatedAt: number;
  version: number;
  status: ClaimStatus;
  verificationLevel?: number;
}

export type ClaimStatus =
  | "Draft"
  | "Published"
  | "Disputed"
  | "Verified"
  | "Retracted";

/** Input for creating a claim */
export interface CreateClaimInput {
  content: string;
  classification: EpistemicPosition;
  evidence?: Evidence[];
  sources?: Source[];
  domain: string;
  topics?: string[];
}

/** Input for updating a claim */
export interface UpdateClaimInput {
  claimHash: ActionHash;
  content?: string;
  classification?: EpistemicPosition;
  additionalEvidence?: Evidence[];
  additionalSources?: Source[];
}

// ============================================================================
// TYPES - Market Integration
// ============================================================================

/** Types of links between claims and markets */
export type MarketLinkType =
  | "VerificationMarket"      // Market to verify this claim
  | "ClaimAsEvidence"         // Claim used as evidence in market
  | "PredictionAboutClaim"    // Market predicting claim outcome
  | "ResolutionSource";       // Market resolved based on claim

/** Status of market verification */
export type MarketVerificationStatus =
  | "Pending"
  | "MarketCreated"
  | "Voting"
  | "Resolved"
  | "Disputed"
  | "Cancelled";

/** Target epistemic level for verification */
export interface EpistemicTarget {
  targetE: number;
  minConfidence: number;
}

/** Resolution from a verification market */
export interface MarketResolution {
  outcome: "Verified" | "Refuted" | "Inconclusive";
  confidence: number;
  resolvedAt: number;
  oracleCount: number;
  marketFinalPrice?: number;
}

/** Link between a claim and verification market */
export interface ClaimMarketLink {
  id: string;
  claimId: string;
  marketId: string;
  linkType: MarketLinkType;
  status: MarketVerificationStatus;
  targetEpistemic: EpistemicTarget;
  requestedBy: AgentPubKey;
  createdAt: number;
  resolvedAt?: number;
  resolution?: MarketResolution;
}

/** Input for spawning a verification market */
export interface SpawnVerificationMarketInput {
  claimId: string;
  targetE: number;
  minConfidence: number;
  closesAt: number;
  initialSubsidy?: number;
  tags?: string[];
}

/** Summary of a market for display */
export interface MarketSummary {
  marketId: string;
  question: string;
  currentPrice: number;
  status: MarketVerificationStatus;
  participantCount: number;
  totalStake: number;
}

// ============================================================================
// TYPES - Belief Graph
// ============================================================================

/** Types of dependencies between claims */
export type DependencyType =
  | "Supports"       // Source supports target
  | "Contradicts"    // Source contradicts target
  | "DerivedFrom"    // Source derived from target
  | "Generalizes"    // Source is generalization of target
  | "Specializes"    // Source is specialization of target
  | "PartOf"         // Source is part of target
  | "Equivalent"     // Claims are equivalent
  | "Causes"         // Source causes target
  | "Temporal";      // Temporal relationship

/** Direction of influence between claims */
export type InfluenceDirection =
  | "Positive"   // Increases belief in target
  | "Negative"   // Decreases belief in target
  | "Neutral";   // No direct influence

/** Dependency between claims */
export interface ClaimDependency {
  id: string;
  dependentClaimId: string;
  dependencyClaimId: string;
  dependencyType: DependencyType;
  weight: number;
  influence: InfluenceDirection;
  establishedBy: AgentPubKey;
  createdAt: number;
  active: boolean;
  justification?: string;
}

/** Type of influence on belief */
export type InfluenceType =
  | "Support"
  | "Contradiction"
  | "Evidential"
  | "Entailment"
  | "MarketVerification";

/** An influence on a belief node */
export interface BeliefInfluence {
  sourceClaimId: string;
  influenceType: InfluenceType;
  weight: number;
  sourceBelief: number;
}

/** A node in the belief graph */
export interface BeliefNode {
  id: string;
  claimId: string;
  beliefStrength: number;
  priorBelief: number;
  confidence: number;
  supportCount: number;
  contradictionCount: number;
  lastUpdated: number;
  propagationIterations: number;
  converged: boolean;
  influences: BeliefInfluence[];
}

/** Result of belief propagation */
export interface PropagationResult {
  sourceClaimId: string;
  nodesAffected: number;
  iterations: number;
  converged: boolean;
  maxDelta: number;
  processingTimeMs: number;
  updatedNodes: string[];
}

/** Result of cascade update */
export interface CascadeResult {
  claimId: string;
  claimsUpdated: number;
  claimsNotified: number;
  errors: string[];
  processingTimeMs: number;
  updatedClaims: CascadeUpdateInfo[];
  marketNotifications: MarketNotification[];
}

export interface CascadeUpdateInfo {
  claimId: string;
  oldE: number;
  newE: number;
  changeReason: string;
}

export interface MarketNotification {
  marketId: string;
  notificationType: string;
  processed: boolean;
}

// ============================================================================
// TYPES - Information Value
// ============================================================================

/** Information value assessment for prioritizing verification */
export interface InformationValue {
  id: string;
  claimId: string;
  expectedValue: number;
  dependentCount: number;
  averageDependencyWeight: number;
  uncertainty: number;
  impactScore: number;
  recommendedForVerification: boolean;
  assessedAt: number;
  reasoning: string;
}

/** Node in a dependency tree */
export interface DependencyTreeNode {
  claimId: string;
  depth: number;
  weight: number;
  children: string[];
  isLeaf: boolean;
}

/** Dependency tree structure */
export interface DependencyTree {
  rootClaimId: string;
  nodes: DependencyTreeNode[];
  depth: number;
  totalDependencies: number;
  aggregateWeight: number;
}

/** Cascade impact assessment */
export interface CascadeImpact {
  claimId: string;
  totalAffected: number;
  affectedByDepth: number[];
  maxDepth: number;
  impactScore: number;
  highImpactClaims: string[];
  assessedAt: number;
}

// ============================================================================
// TYPES - Graph Relationships
// ============================================================================

/** Types of relationships between claims */
export type RelationshipType =
  | "Supports"
  | "Contradicts"
  | "DerivedFrom"
  | "ExampleOf"
  | "Generalizes"
  | "PartOf"
  | "Causes"
  | "RelatedTo"
  | "Equivalent"
  | "SpecializedFrom"
  | { Custom: string };

/** Relationship/edge between claims */
export interface Relationship {
  id: string;
  source: string;
  target: string;
  relationshipType: RelationshipType;
  weight: number;
  properties?: Record<string, unknown>;
  creator: string;
  created: number;
}

/** Ontology for knowledge organization */
export interface Ontology {
  id: string;
  name: string;
  description: string;
  namespace: string;
  schema: string;
  version: string;
  creator: string;
  created: number;
  updated: number;
}

/** Concept within an ontology */
export interface Concept {
  id: string;
  ontologyId: string;
  name: string;
  definition: string;
  parent?: string;
  synonyms: string[];
  created: number;
}

// ============================================================================
// TYPES - Credibility & MATL
// ============================================================================

/** Subject types for credibility assessment */
export type CredibilitySubjectType = "Claim" | "Source" | "Author";

/** Basic credibility components */
export interface CredibilityComponents {
  accuracy: number;
  consistency: number;
  transparency: number;
  trackRecord: number;
  corroboration: number;
}

/** MATL (Multi-dimensional Adaptive Trust Layer) components */
export interface MatlCredibilityComponents {
  matlComposite: number;
  matlQuality: number;
  matlConsistency: number;
  matlReputation: number;
  matlStakeWeighted: number;
  oracleAgreement?: number;
  oracleCount: number;
}

/** Evidence strength analysis */
export interface EvidenceStrengthComponents {
  empiricalEvidenceCount: number;
  testimonialEvidenceCount: number;
  cryptoEvidenceCount: number;
  crossReferenceCount: number;
  marketResolutionCount: number;
  empiricalAverageStrength: number;
  testimonialAverageStrength: number;
  totalEvidenceCount: number;
  evidenceDiversity: number;
}

/** Domain-specific reputation */
export interface DomainReputation {
  domain: string;
  score: number;
  claimCount: number;
  accuracyRate: number;
}

/** Author reputation profile */
export interface AuthorReputation {
  id: string;
  authorDid: string;
  overallScore: number;
  domainScores: DomainReputation[];
  historicalAccuracy: number;
  claimsAuthored: number;
  claimsVerifiedTrue: number;
  claimsVerifiedFalse: number;
  claimsPending: number;
  averageEpistemicE: number;
  matlTrust: number;
  updatedAt: number;
  reputationAgeDays: number;
}

/** Factor contributing to credibility */
export interface CredibilityFactor {
  name: string;
  value: number;
  explanation: string;
}

/** Enhanced credibility score with MATL integration */
export interface EnhancedCredibilityScore {
  id: string;
  subject: string;
  subjectType: CredibilitySubjectType;
  overallScore: number;
  components: CredibilityComponents;
  matl: MatlCredibilityComponents;
  evidenceStrength: EvidenceStrengthComponents;
  authorReputation?: AuthorReputation;
  factors: CredibilityFactor[];
  assessedAt: number;
  expiresAt?: number;
  assessmentModel: string;
  assessmentConfidence: number;
}

/** Summary of enhanced credibility (for batch results) */
export interface EnhancedCredibilitySummary {
  subject: string;
  overallScore: number;
  matlComposite: number;
  evidenceCount: number;
  assessmentConfidence: number;
}

/** Batch credibility result */
export interface BatchCredibilityResult {
  results: EnhancedCredibilitySummary[];
  totalAssessed: number;
  averageScore: number;
  highCredibilityCount: number;
  lowCredibilityCount: number;
  processingTimeMs: number;
}

// ============================================================================
// TYPES - Inference
// ============================================================================

/** Types of inferences */
export type InferenceType =
  | "ImpliedRelation"
  | "Contradiction"
  | "Pattern"
  | "Prediction"
  | "Synthesis"
  | "Similarity"
  | "Anomaly"
  | "CredibilityAssessment";

/** An inference drawn from claims */
export interface Inference {
  id: string;
  inferenceType: InferenceType;
  sourceClaims: string[];
  conclusion: string;
  confidence: number;
  reasoning: string;
  model: string;
  created: number;
  verified: boolean;
}

/** Types of patterns */
export type PatternType =
  | "Temporal"
  | "Cluster"
  | "Causal"
  | "Correlation"
  | "ContradictionCluster"
  | "EchoChamber";

/** Pattern detected in knowledge graph */
export interface Pattern {
  id: string;
  patternType: PatternType;
  description: string;
  claims: string[];
  strength: number;
  detectedAt: number;
}

// ============================================================================
// TYPES - Fact Check
// ============================================================================

/** Verdict from fact checking */
export type FactCheckVerdict =
  | "True"
  | "MostlyTrue"
  | "Mixed"
  | "MostlyFalse"
  | "False"
  | "Unverifiable"
  | "InsufficientEvidence";

/** Supporting claim in fact check */
export interface SupportingClaim {
  claimId: string;
  relevance: number;
  relationship: "Supports" | "Contradicts" | "Contextual";
  epistemicPosition: EpistemicPosition;
  credibilityScore: number;
}

/** Result of a fact check */
export interface FactCheckResult {
  id: string;
  statement: string;
  verdict: FactCheckVerdict;
  confidence: number;
  supportingClaims: SupportingClaim[];
  contradictingClaims: SupportingClaim[];
  reasoning: string;
  epistemicAssessment: EpistemicPosition;
  checkedAt: number;
  expiresAt?: number;
  sourceHapp?: string;
}

/** Input for fact checking */
export interface FactCheckInput {
  statement: string;
  context?: string;
  minE?: number;
  minN?: number;
  sourceHapp?: string;
  requestedBy?: AgentPubKey;
}

/** Options for claim queries */
export interface ClaimQueryOptions {
  minE?: number;
  maxE?: number;
  minN?: number;
  maxN?: number;
  minM?: number;
  maxM?: number;
  status?: ClaimStatus;
  domain?: string;
  limit?: number;
  offset?: number;
}

// ============================================================================
// TYPES - Query
// ============================================================================

/** Query operators for filtering */
export type QueryOperator =
  | { Equals: unknown }
  | { GreaterThan: number }
  | { LessThan: number }
  | { Contains: string }
  | { In: unknown[] }
  | { Between: [number, number] };

/** Field to query on */
export interface QueryField {
  field: string;
  operator: QueryOperator;
}

/** Sort direction */
export type SortDirection = "Asc" | "Desc";

/** Sort specification */
export interface QuerySort {
  field: string;
  direction: SortDirection;
}

/** Full query specification */
export interface Query {
  filters: QueryField[];
  sort?: QuerySort[];
  limit?: number;
  offset?: number;
}

/** Result of a query */
export interface QueryResult<T> {
  items: T[];
  total: number;
  hasMore: boolean;
}

// ============================================================================
// CLIENTS
// ============================================================================

/**
 * Client for claims management
 */
export class ClaimsClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "claims"
  ) {}

  async createClaim(input: CreateClaimInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_claim",
      payload: input,
    });
  }

  async getClaim(claimHash: ActionHash): Promise<Claim | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claim",
      payload: claimHash,
    });
  }

  async updateClaim(input: UpdateClaimInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "update_claim",
      payload: input,
    });
  }

  async addEvidence(claimId: string, evidence: Evidence): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "add_evidence",
      payload: { claim_id: claimId, evidence },
    });
  }

  async listClaimsByAuthor(author: AgentPubKey): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claims_by_author",
      payload: author,
    });
  }

  async listClaimsByDomain(domain: string): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claims_by_domain",
      payload: domain,
    });
  }

  async searchClaimsByTopic(topic: string): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "search_claims_by_topic",
      payload: topic,
    });
  }

  async getClaimsPendingVerification(minE: number): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claims_pending_verification",
      payload: minE,
    });
  }

  // Market integration methods

  async spawnVerificationMarket(
    input: SpawnVerificationMarketInput
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "spawn_verification_market",
      payload: input,
    });
  }

  async getClaimMarkets(claimId: string): Promise<MarketSummary[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claim_markets",
      payload: claimId,
    });
  }

  // Dependency methods

  async registerDependency(
    dependentClaimId: string,
    dependencyClaimId: string,
    dependencyType: DependencyType,
    weight: number,
    justification?: string
  ): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "register_dependency",
      payload: {
        dependent_claim_id: dependentClaimId,
        dependency_claim_id: dependencyClaimId,
        dependency_type: dependencyType,
        weight,
        justification,
      },
    });
  }

  async cascadeUpdate(claimId: string): Promise<CascadeResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "cascade_update",
      payload: claimId,
    });
  }
}

/**
 * Client for knowledge graph operations
 */
export class GraphClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "graph"
  ) {}

  async createRelationship(
    source: string,
    target: string,
    relationshipType: RelationshipType,
    weight: number,
    properties?: Record<string, unknown>
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_relationship",
      payload: {
        source,
        target,
        relationship_type: relationshipType,
        weight,
        properties,
      },
    });
  }

  async getRelationship(relationshipHash: ActionHash): Promise<Relationship | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_relationship",
      payload: relationshipHash,
    });
  }

  async getOutgoingRelationships(claimId: string): Promise<Relationship[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_outgoing_relationships",
      payload: claimId,
    });
  }

  async getIncomingRelationships(claimId: string): Promise<Relationship[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_incoming_relationships",
      payload: claimId,
    });
  }

  // Belief graph methods

  async propagateBelief(claimId: string): Promise<PropagationResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "propagate_belief",
      payload: claimId,
    });
  }

  async getBeliefNode(claimId: string): Promise<BeliefNode | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_belief_node",
      payload: claimId,
    });
  }

  async getDependencyTree(
    claimId: string,
    maxDepth: number = 5
  ): Promise<DependencyTree> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_dependency_tree",
      payload: { claim_id: claimId, max_depth: maxDepth },
    });
  }

  async rankByInformationValue(limit: number = 10): Promise<InformationValue[]> {
    const records = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "rank_by_information_value",
      payload: limit,
    });
    return records;
  }

  async detectCircularDependencies(claimId: string): Promise<string[][]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "detect_circular_dependencies",
      payload: claimId,
    });
  }

  async calculateCascadeImpact(claimId: string): Promise<CascadeImpact> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "calculate_cascade_impact",
      payload: claimId,
    });
  }

  // Ontology methods

  async createOntology(
    name: string,
    description: string,
    namespace: string,
    schema: Record<string, unknown>
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_ontology",
      payload: {
        name,
        description,
        namespace,
        schema: JSON.stringify(schema),
      },
    });
  }

  async createConcept(
    ontologyId: string,
    name: string,
    definition: string,
    parent?: string,
    synonyms?: string[]
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_concept",
      payload: {
        ontology_id: ontologyId,
        name,
        definition,
        parent,
        synonyms: synonyms || [],
      },
    });
  }
}

/**
 * Client for query operations
 */
export class QueryClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "query"
  ) {}

  async search(queryText: string, options?: ClaimQueryOptions): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "search",
      payload: { query: queryText, options },
    });
  }

  async queryByEpistemic(
    minE?: number,
    maxE?: number,
    minN?: number,
    maxN?: number,
    minM?: number,
    maxM?: number,
    limit?: number
  ): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "query_by_epistemic",
      payload: { min_e: minE, max_e: maxE, min_n: minN, max_n: maxN, min_m: minM, max_m: maxM, limit },
    });
  }

  async findRelated(
    claimId: string,
    maxDepth: number = 2,
    limit: number = 20
  ): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "find_related",
      payload: { claim_id: claimId, max_depth: maxDepth, limit },
    });
  }

  async findContradictions(claimId: string): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "find_contradictions",
      payload: claimId,
    });
  }

  async advancedQuery(query: Query): Promise<QueryResult<Claim>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "advanced_query",
      payload: query,
    });
  }
}

/**
 * Client for inference and credibility operations
 */
export class InferenceClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "inference"
  ) {}

  async createInference(
    sourceClaims: string[],
    conclusion: string,
    inferenceType: InferenceType,
    reasoning: string,
    model: string
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_inference",
      payload: {
        source_claims: sourceClaims,
        conclusion,
        inference_type: inferenceType,
        reasoning,
        model,
      },
    });
  }

  async getInferencesForClaim(claimId: string): Promise<Inference[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_inferences_for_claim",
      payload: claimId,
    });
  }

  // Credibility methods

  async calculateEnhancedCredibility(
    subject: string,
    subjectType: CredibilitySubjectType,
    includeMatl: boolean = true
  ): Promise<EnhancedCredibilityScore> {
    const record = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "calculate_enhanced_credibility",
      payload: { subject, subject_type: subjectType, include_matl: includeMatl },
    });
    return record;
  }

  async assessEvidenceStrength(claimId: string): Promise<EvidenceStrengthComponents> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "assess_evidence_strength",
      payload: claimId,
    });
  }

  async getAuthorReputation(authorDid: string): Promise<AuthorReputation> {
    const record = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_author_reputation",
      payload: authorDid,
    });
    return record;
  }

  async batchCredibilityAssessment(claimIds: string[]): Promise<BatchCredibilityResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "batch_credibility_assessment",
      payload: claimIds,
    });
  }

  // Pattern detection

  async detectPatterns(claimIds: string[], patternType?: PatternType): Promise<Pattern[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "detect_patterns",
      payload: { claim_ids: claimIds, pattern_type: patternType },
    });
  }
}

/**
 * Client for markets integration operations
 */
export class MarketsIntegrationClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "markets_integration"
  ) {}

  async requestVerificationMarket(
    claimId: string,
    targetE: number,
    minConfidence: number,
    closesAt: number,
    tags?: string[]
  ): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "request_verification_market",
      payload: {
        claim_id: claimId,
        target_epistemic: { target_e: targetE, min_confidence: minConfidence },
        closes_at: closesAt,
        tags,
      },
    });
  }

  async registerClaimAsEvidence(
    claimId: string,
    marketId: string,
    usage: string
  ): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "register_claim_in_prediction",
      payload: { claim_id: claimId, market_id: marketId, usage },
    });
  }

  async getClaimMarkets(claimId: string): Promise<MarketSummary[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claim_markets",
      payload: claimId,
    });
  }

  async getClaimsUsedInMarket(marketId: string): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_claims_for_market",
      payload: marketId,
    });
  }

  async calculateMarketValue(claimId: string): Promise<{
    expectedValue: number;
    recommendedSubsidy: number;
    reasoning: string;
  }> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "calculate_market_value",
      payload: claimId,
    });
  }
}

/**
 * Client for fact-checking operations
 */
export class FactCheckClient {
  constructor(
    private client: AppClient,
    private roleName: string = "knowledge",
    private zomeName: string = "factcheck"
  ) {}

  async factCheck(input: FactCheckInput): Promise<FactCheckResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "fact_check",
      payload: input,
    });
  }

  async batchFactCheck(inputs: FactCheckInput[]): Promise<FactCheckResult[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "batch_fact_check",
      payload: inputs,
    });
  }

  async queryClaimsBySubject(
    subject: string,
    options?: ClaimQueryOptions
  ): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "query_claims_by_subject",
      payload: { subject, options },
    });
  }

  async queryClaimsByTopic(
    topics: string[],
    options?: ClaimQueryOptions
  ): Promise<Claim[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "query_claims_by_topic",
      payload: { topics, options },
    });
  }

  async getFactCheckHistory(statement: string): Promise<FactCheckResult[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_fact_check_history",
      payload: statement,
    });
  }
}

// ============================================================================
// UNIFIED CLIENT
// ============================================================================

/**
 * Unified client for all knowledge graph operations
 *
 * @example
 * ```typescript
 * import { AppWebsocket } from '@holochain/client';
 * import { KnowledgeClient } from '@mycelix/knowledge-sdk';
 *
 * const client = await AppWebsocket.connect('ws://localhost:8888');
 * const knowledge = new KnowledgeClient(client);
 *
 * // Create a claim
 * const claimHash = await knowledge.claims.createClaim({
 *   content: "Global average temperature rose 1.1°C since pre-industrial",
 *   classification: { empirical: 0.9, normative: 0.1, mythic: 0.1 },
 *   domain: "climate",
 *   topics: ["temperature", "climate-change"]
 * });
 *
 * // Get credibility assessment
 * const credibility = await knowledge.inference.calculateEnhancedCredibility(
 *   claimHash,
 *   "Claim"
 * );
 *
 * // Fact-check a statement
 * const result = await knowledge.factcheck.factCheck({
 *   statement: "The Earth is warming due to human activity",
 *   minE: 0.7
 * });
 * ```
 */
export class KnowledgeClient {
  public readonly claims: ClaimsClient;
  public readonly graph: GraphClient;
  public readonly query: QueryClient;
  public readonly inference: InferenceClient;
  public readonly marketsIntegration: MarketsIntegrationClient;
  public readonly factcheck: FactCheckClient;

  constructor(client: AppClient, roleName: string = "knowledge") {
    this.claims = new ClaimsClient(client, roleName, "claims");
    this.graph = new GraphClient(client, roleName, "graph");
    this.query = new QueryClient(client, roleName, "query");
    this.inference = new InferenceClient(client, roleName, "inference");
    this.marketsIntegration = new MarketsIntegrationClient(
      client,
      roleName,
      "markets_integration"
    );
    this.factcheck = new FactCheckClient(client, roleName, "factcheck");
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Convert numeric epistemic position to discrete levels
 */
export function toDiscreteEpistemic(position: EpistemicPosition): {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  mythic: MythicLevel;
} {
  const toEmpirical = (e: number): EmpiricalLevel => {
    if (e < 0.2) return "E0_Subjective";
    if (e < 0.4) return "E1_Testimonial";
    if (e < 0.6) return "E2_PrivateVerify";
    if (e < 0.8) return "E3_Cryptographic";
    return "E4_Measurable";
  };

  const toNormative = (n: number): NormativeLevel => {
    if (n < 0.25) return "N0_Personal";
    if (n < 0.5) return "N1_Communal";
    if (n < 0.75) return "N2_Network";
    return "N3_Universal";
  };

  const toMythic = (m: number): MythicLevel => {
    if (m < 0.25) return "M0_Transient";
    if (m < 0.5) return "M1_Temporal";
    if (m < 0.75) return "M2_Persistent";
    return "M3_Foundational";
  };

  return {
    empirical: toEmpirical(position.empirical),
    normative: toNormative(position.normative),
    mythic: toMythic(position.mythic),
  };
}

/**
 * Calculate information value for a claim based on uncertainty and impact
 */
export function calculateInformationValue(
  uncertainty: number,
  dependentCount: number,
  averageWeight: number
): number {
  // Information value = Uncertainty * Impact
  // Impact = dependentCount * averageWeight
  const impact = dependentCount * averageWeight;
  return uncertainty * impact;
}

/**
 * Recommend whether a claim should be sent to verification market
 */
export function recommendVerification(
  claim: Claim,
  informationValue: InformationValue
): {
  recommend: boolean;
  reason: string;
  suggestedTargetE: number;
} {
  const currentE = claim.classification.empirical;

  // High information value + low empirical = good candidate
  if (informationValue.expectedValue > 0.5 && currentE < 0.5) {
    return {
      recommend: true,
      reason: "High information value with low verifiability - verification would be valuable",
      suggestedTargetE: Math.min(currentE + 0.2, 0.8),
    };
  }

  // Many dependents but uncertain = important to verify
  if (informationValue.dependentCount > 10 && informationValue.uncertainty > 0.3) {
    return {
      recommend: true,
      reason: "Many claims depend on this uncertain claim",
      suggestedTargetE: Math.min(currentE + 0.15, 0.9),
    };
  }

  // Already highly verifiable
  if (currentE > 0.8) {
    return {
      recommend: false,
      reason: "Claim is already highly verifiable",
      suggestedTargetE: currentE,
    };
  }

  // Default: only if significant uncertainty
  if (informationValue.uncertainty > 0.5) {
    return {
      recommend: true,
      reason: "Significant uncertainty warrants verification",
      suggestedTargetE: currentE + 0.1,
    };
  }

  return {
    recommend: false,
    reason: "Verification not cost-effective at current state",
    suggestedTargetE: currentE,
  };
}

/**
 * Calculate composite credibility score from components
 */
export function calculateCompositeCredibility(
  components: CredibilityComponents,
  matl?: MatlCredibilityComponents
): number {
  // Base credibility from components (equal weights)
  const baseScore =
    (components.accuracy +
      components.consistency +
      components.transparency +
      components.trackRecord +
      components.corroboration) /
    5;

  if (!matl) {
    return baseScore;
  }

  // MATL-weighted score uses quadratic weighting for Byzantine resilience
  // Higher MATL scores have proportionally more influence
  const matlWeight = matl.matlComposite * matl.matlComposite; // Quadratic

  // Blend base score with MATL score
  // MATL contributes more as it increases (due to quadratic)
  return baseScore * (1 - matlWeight * 0.3) + matl.matlComposite * matlWeight * 0.3 + baseScore * 0.7;
}

/**
 * Determine fact-check verdict from supporting and contradicting claims
 */
export function determineVerdict(
  supportingClaims: SupportingClaim[],
  contradictingClaims: SupportingClaim[]
): { verdict: FactCheckVerdict; confidence: number } {
  if (supportingClaims.length === 0 && contradictingClaims.length === 0) {
    return { verdict: "InsufficientEvidence", confidence: 0.1 };
  }

  // Calculate weighted support
  const supportScore = supportingClaims.reduce(
    (acc, c) => acc + c.relevance * c.credibilityScore,
    0
  );
  const contradictScore = contradictingClaims.reduce(
    (acc, c) => acc + c.relevance * c.credibilityScore,
    0
  );

  const total = supportScore + contradictScore;
  if (total === 0) {
    return { verdict: "InsufficientEvidence", confidence: 0.1 };
  }

  const supportRatio = supportScore / total;
  const confidence = Math.min(0.95, total / (supportingClaims.length + contradictingClaims.length));

  if (supportRatio > 0.9) {
    return { verdict: "True", confidence };
  } else if (supportRatio > 0.7) {
    return { verdict: "MostlyTrue", confidence };
  } else if (supportRatio > 0.45 && supportRatio < 0.55) {
    return { verdict: "Mixed", confidence: confidence * 0.8 };
  } else if (supportRatio > 0.3) {
    return { verdict: "MostlyFalse", confidence };
  } else {
    return { verdict: "False", confidence };
  }
}

/**
 * Format epistemic position for display
 */
export function formatEpistemicPosition(position: EpistemicPosition): string {
  const discrete = toDiscreteEpistemic(position);
  return `E:${discrete.empirical} N:${discrete.normative} M:${discrete.mythic}`;
}

/**
 * Check if two claims potentially contradict each other
 * based on their epistemic positions and content similarity
 */
export function mayContradict(
  claim1: EpistemicPosition,
  claim2: EpistemicPosition
): boolean {
  // Claims at opposite ends of normative spectrum may contradict
  const normativeDiff = Math.abs(claim1.normative - claim2.normative);

  // High empirical claims that differ significantly may contradict
  if (claim1.empirical > 0.7 && claim2.empirical > 0.7 && normativeDiff > 0.5) {
    return true;
  }

  return false;
}

// ============================================================================
// SERVICES (High-level abstractions)
// ============================================================================

export {
  KnowledgeService,
  BeliefGraphService,
} from "./services";

export type {
  ClaimSubmissionResult,
  ComprehensiveFactCheckResult,
  EnrichedClaim,
  GraphNode,
  GraphEdge,
  VisualizationGraph,
  ConsistencyAnalysis,
  BeliefPath,
} from "./services";

// ============================================================================
// EXPORTS
// ============================================================================

export default KnowledgeClient;
