// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Module
 *
 * TypeScript clients for the Mycelix-Knowledge hApp providing:
 * - Epistemic claims with E/N/M classification
 * - Knowledge graph relationships and traversal
 * - Query language for claim discovery
 * - ML-powered inference and credibility scoring
 * - Cross-hApp knowledge bridge
 *
 * @packageDocumentation
 * @module knowledge
 */

// ============================================================================
// Type Definitions
// ============================================================================

/** Holochain record wrapper */
export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: {
      hash: string;
      content: unknown;
    };
    signature: string;
  };
  entry: {
    Present: T;
  };
}

/** Generic zome call interface */
export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Claims Types
// ============================================================================

/**
 * Epistemic position on the 3D cube (E/N/M axes)
 * Each axis represents a different dimension of knowledge validity
 */
export interface EpistemicPosition {
  /** Empirical validity (0.0 to 1.0) - How verifiable through observation/experiment? */
  empirical: number;
  /** Normative coherence (0.0 to 1.0) - How aligned with ethical frameworks? */
  normative: number;
  /** Mythic resonance (0.0 to 1.0) - What narrative/meaning significance? */
  mythic: number;
}

/** Types of claims in the knowledge graph */
export type ClaimType =
  | 'Fact'
  | 'Opinion'
  | 'Prediction'
  | 'Hypothesis'
  | 'Definition'
  | 'Historical'
  | 'Normative'
  | 'Narrative';

/**
 * Source types for claim verification and calibration
 * Used by the Calibration Engine to track reliability by source
 */
export type SourceType =
  | 'AcademicPaper'      // Peer-reviewed research
  | 'DataSet'            // Government or institutional data
  | 'PredictionMarket'   // Market-based forecasts
  | 'ExpertTestimony'    // Domain expert opinion
  | 'CrowdWisdom'        // Aggregated crowd opinion
  | 'Sensor'             // IoT/automated measurement
  | 'Simulation'         // Computational model output
  | 'Historical'         // Historical records
  | 'Attestation'        // Signed attestation
  | 'Other';             // Unclassified source

/** A claim in the knowledge graph */
export interface Claim {
  /** Unique claim identifier */
  id: string;
  /** The claim content (statement) */
  content: string;
  /** Epistemic classification on E/N/M axes */
  classification: EpistemicPosition;
  /** Author's DID */
  author: string;
  /** Source references (URIs to evidence) */
  sources: string[];
  /** Tags for categorization */
  tags: string[];
  /** Claim type */
  claim_type: ClaimType;
  /** Confidence level (0.0 to 1.0) */
  confidence: number;
  /** Optional expiration timestamp */
  expires?: number;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
  /** Version for updates */
  version: number;
}

/** Input for updating a claim */
export interface UpdateClaimInput {
  claim_id: string;
  content?: string;
  classification?: EpistemicPosition;
  sources?: string[];
  tags?: string[];
  confidence?: number;
  expires?: number;
}

/** Types of evidence */
export type EvidenceType =
  | 'Observation'
  | 'Experiment'
  | 'Statistical'
  | 'Expert'
  | 'Document'
  | 'CrossReference'
  | 'Logical';

/** Evidence supporting a claim */
export interface Evidence {
  /** Evidence identifier */
  id: string;
  /** Claim this evidence supports */
  claim_id: string;
  /** Evidence type */
  evidence_type: EvidenceType;
  /** URI to evidence source */
  source_uri: string;
  /** Evidence content/summary */
  content: string;
  /** Strength of evidence (0.0 to 1.0) */
  strength: number;
  /** Who submitted this evidence */
  submitted_by: string;
  /** Submission timestamp */
  submitted_at: number;
}

/** Status of a challenge */
export type ChallengeStatus = 'Pending' | 'UnderReview' | 'Accepted' | 'Rejected' | 'Withdrawn';

/** Claim challenge/dispute */
export interface ClaimChallenge {
  /** Challenge identifier */
  id: string;
  /** Claim being challenged */
  claim_id: string;
  /** Challenger's DID */
  challenger: string;
  /** Reason for challenge */
  reason: string;
  /** Counter-evidence */
  counter_evidence: string[];
  /** Challenge status */
  status: ChallengeStatus;
  /** Creation timestamp */
  created: number;
}

/** Input for challenging a claim */
export interface ChallengeClaimInput {
  claim_id: string;
  challenger_did: string;
  reason: string;
  counter_evidence: string[];
}

/** Input for epistemic range search */
export interface EpistemicRangeSearch {
  e_min: number;
  e_max: number;
  n_min: number;
  n_max: number;
  m_min: number;
  m_max: number;
}

// ============================================================================
// Graph Types
// ============================================================================

/** Types of relationships between claims */
export type RelationshipType =
  | 'Supports'
  | 'Contradicts'
  | 'DerivedFrom'
  | 'ExampleOf'
  | 'Generalizes'
  | 'PartOf'
  | 'Causes'
  | 'RelatedTo'
  | 'Equivalent'
  | 'SpecializedFrom'
  | { Custom: string };

/** A relationship/edge between two claims */
export interface Relationship {
  /** Relationship identifier */
  id: string;
  /** Source claim ID */
  source: string;
  /** Target claim ID */
  target: string;
  /** Relationship type */
  relationship_type: RelationshipType;
  /** Relationship strength/weight (0.0 to 1.0) */
  weight: number;
  /** Additional properties (JSON) */
  properties?: string;
  /** Creator's DID */
  creator: string;
  /** Creation timestamp */
  created: number;
}

/** Ontology definition for knowledge organization */
export interface Ontology {
  /** Ontology identifier */
  id: string;
  /** Ontology name */
  name: string;
  /** Description */
  description: string;
  /** Namespace URI */
  namespace: string;
  /** Schema (JSON-LD or similar) */
  schema: string;
  /** Version */
  version: string;
  /** Creator's DID */
  creator: string;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/** Concept within an ontology */
export interface Concept {
  /** Concept identifier */
  id: string;
  /** Ontology this concept belongs to */
  ontology_id: string;
  /** Concept name */
  name: string;
  /** Definition */
  definition: string;
  /** Parent concept (if any) */
  parent?: string;
  /** Synonyms */
  synonyms: string[];
  /** Creation timestamp */
  created: number;
}

/** Input for path finding */
export interface FindPathInput {
  source: string;
  target: string;
  max_depth: number;
}

/** Graph statistics */
export interface GraphStats {
  relationship_count: number;
}

// ============================================================================
// Query Types
// ============================================================================

/** A saved query for reuse */
export interface SavedQuery {
  /** Query identifier */
  id: string;
  /** Query name */
  name: string;
  /** Query description */
  description: string;
  /** Query in our query language */
  query: string;
  /** Query parameters (JSON) */
  parameters?: string;
  /** Creator's DID */
  creator: string;
  /** Whether query is public */
  public: boolean;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/** Types of query steps */
export type QueryStepType =
  | 'FullScan'
  | 'TagLookup'
  | 'AuthorLookup'
  | 'GraphTraversal'
  | 'EpistemicFilter'
  | 'Join';

/** Filter operators */
export type FilterOperator =
  | 'Equals'
  | 'NotEquals'
  | 'GreaterThan'
  | 'LessThan'
  | 'GreaterOrEqual'
  | 'LessOrEqual'
  | 'Contains'
  | 'StartsWith'
  | 'EndsWith'
  | 'In'
  | 'Between';

/** Filter conditions */
export interface QueryFilter {
  field: string;
  operator: FilterOperator;
  value: string;
}

/** A step in the query execution plan */
export interface QueryStep {
  step_type: QueryStepType;
  target: string;
  filters: QueryFilter[];
}

/** Query execution plan */
export interface QueryPlan {
  steps: QueryStep[];
  estimated_cost: number;
  use_cache: boolean;
}

/** Input for executing a query */
export interface ExecuteQueryInput {
  query: string;
  parameters?: string;
  use_cache: boolean;
  limit?: number;
  offset?: number;
}

/** Query execution result */
export interface QueryExecutionResult {
  results: string[];
  count: number;
  execution_time_ms: number;
  plan?: QueryPlan;
}

// ============================================================================
// Inference Types
// ============================================================================

/** Types of inferences */
export type InferenceType =
  | 'Implication'
  | 'Contradiction'
  | 'Causation'
  | 'Correlation'
  | 'Generalization'
  | 'Specialization';

/** An inference generated from claims */
export interface Inference {
  /** Inference identifier */
  id: string;
  /** Inference type */
  inference_type: InferenceType;
  /** Source claims used for inference */
  source_claims: string[];
  /** Conclusion drawn */
  conclusion: string;
  /** Confidence in the inference (0.0 to 1.0) */
  confidence: number;
  /** Reasoning explanation */
  reasoning: string;
  /** Model used for inference */
  model: string;
  /** Creation timestamp */
  created: number;
  /** Whether inference has been human-verified */
  verified: boolean;
}

/** Types of credibility subjects */
export type CredibilitySubjectType = 'Claim' | 'Author' | 'Source' | 'Evidence';

/** Components of a credibility score */
export interface CredibilityComponents {
  /** How accurate is the subject? */
  accuracy: number;
  /** How consistent is the subject? */
  consistency: number;
  /** How transparent is the methodology? */
  transparency: number;
  /** Historical track record */
  track_record: number;
  /** Level of corroboration from other sources */
  corroboration: number;
}

/** A credibility score for a subject */
export interface CredibilityScore {
  /** Score identifier */
  id: string;
  /** Subject being scored (claim ID, author DID, etc.) */
  subject: string;
  /** Type of subject */
  subject_type: CredibilitySubjectType;
  /** Overall score (0.0 to 1.0) */
  score: number;
  /** Component scores */
  components: CredibilityComponents;
  /** Factors affecting the score */
  factors: string[];
  /** When assessed */
  assessed_at: number;
  /** When score expires */
  expires_at?: number;
}

/** Input for credibility assessment */
export interface AssessCredibilityInput {
  subject: string;
  subject_type: CredibilitySubjectType;
  expires_at?: number;
}

/** Types of patterns */
export type PatternType = 'Cluster' | 'Trend' | 'CausalChain' | 'Contradiction' | 'Emergence';

/** A detected pattern in the knowledge graph */
export interface Pattern {
  /** Pattern identifier */
  id: string;
  /** Pattern type */
  pattern_type: PatternType;
  /** Description of the pattern */
  description: string;
  /** Claims involved in the pattern */
  claims: string[];
  /** Pattern strength (0.0 to 1.0) */
  strength: number;
  /** When detected */
  detected_at: number;
}

/** Input for pattern detection */
export interface DetectPatternsInput {
  claims: string[];
  pattern_types?: PatternType[];
}

/** Input for verifying an inference */
export interface VerifyInferenceInput {
  inference_id: string;
  is_correct: boolean;
  verifier_did: string;
  comment?: string;
}

// ============================================================================
// Bridge Types
// ============================================================================

/** Knowledge query types */
export type KnowledgeQueryType =
  | 'VerifyClaim'
  | 'ClaimsBySubject'
  | 'EpistemicScore'
  | 'GraphTraversal'
  | 'FactCheck';

/** Knowledge event types */
export type KnowledgeEventType =
  | 'ClaimCreated'
  | 'ClaimUpdated'
  | 'ClaimDisputed'
  | 'ClaimVerified'
  | 'GraphUpdated'
  | 'InferenceCompleted';

/** Input for querying knowledge */
export interface QueryKnowledgeInput {
  source_happ: string;
  query_type: KnowledgeQueryType;
  parameters: Record<string, unknown>;
}

/** Result of a knowledge query */
export interface QueryKnowledgeResult {
  success: boolean;
  data?: Record<string, unknown>;
  error?: string;
}

/** Fact-check verdict */
export type FactCheckVerdict = 'True' | 'False' | 'PartiallyTrue' | 'Misleading' | 'Unverified';

/** Input for fact-checking */
export interface FactCheckInput {
  source_happ: string;
  statement: string;
  context?: string;
}

/** Result of a fact-check */
export interface FactCheckResult {
  statement: string;
  verdict: FactCheckVerdict;
  confidence: number;
  supporting_claims: string[];
  contradicting_claims: string[];
}

/** Input for registering an external claim */
export interface RegisterClaimInput {
  source_happ: string;
  subject: string;
  predicate: string;
  object: string;
  epistemic_e?: number;
  epistemic_n?: number;
  epistemic_m?: number;
}

/** Input for broadcasting a knowledge event */
export interface BroadcastKnowledgeEventInput {
  event_type: KnowledgeEventType;
  claim_id?: string;
  subject: string;
  payload: string;
}

/** Knowledge bridge event */
export interface KnowledgeBridgeEvent {
  id: string;
  event_type: KnowledgeEventType;
  claim_id?: string;
  subject: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

// ============================================================================
// Claims Client
// ============================================================================

const KNOWLEDGE_ROLE = 'knowledge';
const CLAIMS_ZOME = 'claims';

/**
 * Claims Client - Manage epistemic claims in the knowledge graph
 *
 * @example
 * ```typescript
 * import { ClaimsClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const claims = new ClaimsClient(client);
 *
 * // Submit a claim with epistemic classification
 * const claim = await claims.submitClaim({
 *   id: 'claim-1',
 *   content: 'The Earth is approximately 4.5 billion years old',
 *   classification: { empirical: 0.95, normative: 0.1, mythic: 0.2 },
 *   author: 'did:mycelix:abc123',
 *   sources: ['https://science.nasa.gov/earth'],
 *   tags: ['geology', 'earth-science'],
 *   claim_type: 'Fact',
 *   confidence: 0.99,
 *   created: Date.now() * 1000,
 *   updated: Date.now() * 1000,
 *   version: 1,
 * });
 * ```
 */
export class ClaimsClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Submit a new claim to the knowledge graph
   */
  async submitClaim(claim: Claim): Promise<HolochainRecord<Claim>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'submit_claim',
      payload: claim,
    }) as Promise<HolochainRecord<Claim>>;
  }

  /**
   * Get a claim by ID
   */
  async getClaim(claimId: string): Promise<HolochainRecord<Claim> | null> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'get_claim',
      payload: claimId,
    }) as Promise<HolochainRecord<Claim> | null>;
  }

  /**
   * Get claims by author DID
   */
  async getClaimsByAuthor(authorDid: string): Promise<HolochainRecord<Claim>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'get_claims_by_author',
      payload: authorDid,
    }) as Promise<HolochainRecord<Claim>[]>;
  }

  /**
   * Get claims by tag
   */
  async getClaimsByTag(tag: string): Promise<HolochainRecord<Claim>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'get_claims_by_tag',
      payload: tag,
    }) as Promise<HolochainRecord<Claim>[]>;
  }

  /**
   * Update a claim
   */
  async updateClaim(input: UpdateClaimInput): Promise<HolochainRecord<Claim>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'update_claim',
      payload: input,
    }) as Promise<HolochainRecord<Claim>>;
  }

  /**
   * Add evidence to a claim
   */
  async addEvidence(evidence: Evidence): Promise<HolochainRecord<Evidence>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'add_evidence',
      payload: evidence,
    }) as Promise<HolochainRecord<Evidence>>;
  }

  /**
   * Get evidence for a claim
   */
  async getClaimEvidence(claimId: string): Promise<HolochainRecord<Evidence>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'get_claim_evidence',
      payload: claimId,
    }) as Promise<HolochainRecord<Evidence>[]>;
  }

  /**
   * Challenge a claim
   */
  async challengeClaim(input: ChallengeClaimInput): Promise<HolochainRecord<ClaimChallenge>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'challenge_claim',
      payload: input,
    }) as Promise<HolochainRecord<ClaimChallenge>>;
  }

  /**
   * Get challenges for a claim
   */
  async getClaimChallenges(claimId: string): Promise<HolochainRecord<ClaimChallenge>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'get_claim_challenges',
      payload: claimId,
    }) as Promise<HolochainRecord<ClaimChallenge>[]>;
  }

  /**
   * Search claims by epistemic position range
   */
  async searchByEpistemicRange(input: EpistemicRangeSearch): Promise<HolochainRecord<Claim>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: CLAIMS_ZOME,
      fn_name: 'search_by_epistemic_range',
      payload: input,
    }) as Promise<HolochainRecord<Claim>[]>;
  }
}

// ============================================================================
// Graph Client
// ============================================================================

const GRAPH_ZOME = 'graph';

/**
 * Graph Client - Manage relationships and ontologies in the knowledge graph
 *
 * @example
 * ```typescript
 * import { GraphClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const graph = new GraphClient(client);
 *
 * // Create a relationship between claims
 * const rel = await graph.createRelationship({
 *   id: 'rel-1',
 *   source: 'claim-1',
 *   target: 'claim-2',
 *   relationship_type: 'Supports',
 *   weight: 0.8,
 *   creator: 'did:mycelix:abc123',
 *   created: Date.now() * 1000,
 * });
 *
 * // Find path between claims
 * const path = await graph.findPath({
 *   source: 'claim-1',
 *   target: 'claim-5',
 *   max_depth: 5,
 * });
 * ```
 */
export class GraphClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Create a relationship between two claims
   */
  async createRelationship(relationship: Relationship): Promise<HolochainRecord<Relationship>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'create_relationship',
      payload: relationship,
    }) as Promise<HolochainRecord<Relationship>>;
  }

  /**
   * Get outgoing relationships from a claim
   */
  async getOutgoingRelationships(claimId: string): Promise<HolochainRecord<Relationship>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'get_outgoing_relationships',
      payload: claimId,
    }) as Promise<HolochainRecord<Relationship>[]>;
  }

  /**
   * Get incoming relationships to a claim
   */
  async getIncomingRelationships(claimId: string): Promise<HolochainRecord<Relationship>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'get_incoming_relationships',
      payload: claimId,
    }) as Promise<HolochainRecord<Relationship>[]>;
  }

  /**
   * Find path between two claims
   */
  async findPath(input: FindPathInput): Promise<string[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'find_path',
      payload: input,
    }) as Promise<string[]>;
  }

  /**
   * Create an ontology
   */
  async createOntology(ontology: Ontology): Promise<HolochainRecord<Ontology>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'create_ontology',
      payload: ontology,
    }) as Promise<HolochainRecord<Ontology>>;
  }

  /**
   * Create a concept in an ontology
   */
  async createConcept(concept: Concept): Promise<HolochainRecord<Concept>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'create_concept',
      payload: concept,
    }) as Promise<HolochainRecord<Concept>>;
  }

  /**
   * Get concepts in an ontology
   */
  async getOntologyConcepts(ontologyId: string): Promise<HolochainRecord<Concept>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'get_ontology_concepts',
      payload: ontologyId,
    }) as Promise<HolochainRecord<Concept>[]>;
  }

  /**
   * Get child concepts
   */
  async getChildConcepts(conceptId: string): Promise<HolochainRecord<Concept>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'get_child_concepts',
      payload: conceptId,
    }) as Promise<HolochainRecord<Concept>[]>;
  }

  /**
   * Get graph statistics
   */
  async getGraphStats(): Promise<GraphStats> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: GRAPH_ZOME,
      fn_name: 'get_graph_stats',
      payload: null,
    }) as Promise<GraphStats>;
  }
}

// ============================================================================
// Query Client
// ============================================================================

const QUERY_ZOME = 'query';

/**
 * Query Client - Execute and manage knowledge graph queries
 *
 * @example
 * ```typescript
 * import { QueryClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const query = new QueryClient(client);
 *
 * // Execute a query
 * const results = await query.executeQuery({
 *   query: "SELECT * FROM claims WHERE tag = 'climate-science'",
 *   use_cache: true,
 *   limit: 100,
 * });
 *
 * // Save a query for reuse
 * const saved = await query.saveQuery({
 *   id: 'q-1',
 *   name: 'High Empirical Claims',
 *   description: 'Find claims with high empirical validity',
 *   query: "SELECT * FROM claims WHERE e > 0.8",
 *   creator: 'did:mycelix:abc123',
 *   public: true,
 *   created: Date.now() * 1000,
 *   updated: Date.now() * 1000,
 * });
 * ```
 */
export class QueryClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Execute a query on the knowledge graph
   */
  async executeQuery(input: ExecuteQueryInput): Promise<QueryExecutionResult> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: QUERY_ZOME,
      fn_name: 'execute_query',
      payload: input,
    }) as Promise<QueryExecutionResult>;
  }

  /**
   * Save a query for reuse
   */
  async saveQuery(query: SavedQuery): Promise<HolochainRecord<SavedQuery>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: QUERY_ZOME,
      fn_name: 'save_query',
      payload: query,
    }) as Promise<HolochainRecord<SavedQuery>>;
  }

  /**
   * Get saved queries by creator
   */
  async getMyQueries(creatorDid: string): Promise<HolochainRecord<SavedQuery>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: QUERY_ZOME,
      fn_name: 'get_my_queries',
      payload: creatorDid,
    }) as Promise<HolochainRecord<SavedQuery>[]>;
  }

  /**
   * Get public queries
   */
  async getPublicQueries(): Promise<HolochainRecord<SavedQuery>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: QUERY_ZOME,
      fn_name: 'get_public_queries',
      payload: null,
    }) as Promise<HolochainRecord<SavedQuery>[]>;
  }
}

// ============================================================================
// Inference Client
// ============================================================================

const INFERENCE_ZOME = 'inference';

/**
 * Inference Client - ML-powered reasoning and credibility scoring
 *
 * @example
 * ```typescript
 * import { InferenceClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const inference = new InferenceClient(client);
 *
 * // Assess credibility of a claim
 * const score = await inference.assessCredibility({
 *   subject: 'claim-123',
 *   subject_type: 'Claim',
 * });
 *
 * // Find contradictions
 * const contradictions = await inference.findContradictions([
 *   'claim-1', 'claim-2', 'claim-3'
 * ]);
 *
 * // Detect patterns
 * const patterns = await inference.detectPatterns({
 *   claims: ['claim-1', 'claim-2', 'claim-3'],
 *   pattern_types: ['Cluster', 'Trend'],
 * });
 * ```
 */
export class InferenceClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Create an inference
   */
  async createInference(inference: Inference): Promise<HolochainRecord<Inference>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'create_inference',
      payload: inference,
    }) as Promise<HolochainRecord<Inference>>;
  }

  /**
   * Get inferences for a claim
   */
  async getClaimInferences(claimId: string): Promise<HolochainRecord<Inference>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'get_claim_inferences',
      payload: claimId,
    }) as Promise<HolochainRecord<Inference>[]>;
  }

  /**
   * Assess credibility of a subject
   */
  async assessCredibility(
    input: AssessCredibilityInput
  ): Promise<HolochainRecord<CredibilityScore>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'assess_credibility',
      payload: input,
    }) as Promise<HolochainRecord<CredibilityScore>>;
  }

  /**
   * Get credibility score for a subject
   */
  async getCredibility(subject: string): Promise<HolochainRecord<CredibilityScore> | null> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'get_credibility',
      payload: subject,
    }) as Promise<HolochainRecord<CredibilityScore> | null>;
  }

  /**
   * Detect patterns in the knowledge graph
   */
  async detectPatterns(input: DetectPatternsInput): Promise<HolochainRecord<Pattern>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'detect_patterns',
      payload: input,
    }) as Promise<HolochainRecord<Pattern>[]>;
  }

  /**
   * Find contradictions between claims
   */
  async findContradictions(claimIds: string[]): Promise<HolochainRecord<Inference>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'find_contradictions',
      payload: claimIds,
    }) as Promise<HolochainRecord<Inference>[]>;
  }

  /**
   * Verify an inference (human review)
   */
  async verifyInference(input: VerifyInferenceInput): Promise<HolochainRecord<Inference>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: INFERENCE_ZOME,
      fn_name: 'verify_inference',
      payload: input,
    }) as Promise<HolochainRecord<Inference>>;
  }
}

// ============================================================================
// Knowledge Bridge Client
// ============================================================================

const BRIDGE_ZOME = 'knowledge_bridge';

/**
 * Knowledge Bridge Client - Cross-hApp knowledge queries and fact-checking
 *
 * @example
 * ```typescript
 * import { KnowledgeBridgeClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const bridge = new KnowledgeBridgeClient(client);
 *
 * // Query knowledge from your hApp
 * const result = await bridge.queryKnowledge({
 *   source_happ: 'my-marketplace',
 *   query_type: 'VerifyClaim',
 *   parameters: { claim_id: 'claim-123' },
 * });
 *
 * // Fact-check a statement
 * const check = await bridge.factCheck({
 *   source_happ: 'my-news-app',
 *   statement: 'The economy grew by 3% last quarter',
 * });
 * ```
 */
export class KnowledgeBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Query knowledge graph from another hApp
   */
  async queryKnowledge(input: QueryKnowledgeInput): Promise<QueryKnowledgeResult> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_knowledge',
      payload: input,
    }) as Promise<QueryKnowledgeResult>;
  }

  /**
   * Fact-check a statement via the knowledge graph
   */
  async factCheck(input: FactCheckInput): Promise<FactCheckResult> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'fact_check',
      payload: input,
    }) as Promise<FactCheckResult>;
  }

  /**
   * Register a claim from another hApp
   */
  async registerExternalClaim(input: RegisterClaimInput): Promise<HolochainRecord<unknown>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'register_external_claim',
      payload: input,
    }) as Promise<HolochainRecord<unknown>>;
  }

  /**
   * Broadcast a knowledge event
   */
  async broadcastKnowledgeEvent(
    input: BroadcastKnowledgeEventInput
  ): Promise<HolochainRecord<KnowledgeBridgeEvent>> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_knowledge_event',
      payload: input,
    }) as Promise<HolochainRecord<KnowledgeBridgeEvent>>;
  }

  /**
   * Get recent knowledge events
   */
  async getRecentKnowledgeEvents(since?: number): Promise<HolochainRecord<KnowledgeBridgeEvent>[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_knowledge_events',
      payload: since ?? null,
    }) as Promise<HolochainRecord<KnowledgeBridgeEvent>[]>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Knowledge hApp clients from a connected MycelixClient
 *
 * @example
 * ```typescript
 * import { createKnowledgeClients, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-knowledge' });
 * await client.connect();
 *
 * const { claims, graph, query, inference, bridge } = createKnowledgeClients(client);
 *
 * // Use individual clients
 * const claim = await claims.submitClaim({ ... });
 * const path = await graph.findPath({ ... });
 * const results = await query.executeQuery({ ... });
 * const score = await inference.assessCredibility({ ... });
 * const check = await bridge.factCheck({ ... });
 * ```
 */
export function createKnowledgeClients(client: ZomeCallable) {
  return {
    claims: new ClaimsClient(client),
    graph: new GraphClient(client),
    query: new QueryClient(client),
    inference: new InferenceClient(client),
    bridge: new KnowledgeBridgeClient(client),
  };
}

// Default export for convenient importing
export default {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  createKnowledgeClients,
};
