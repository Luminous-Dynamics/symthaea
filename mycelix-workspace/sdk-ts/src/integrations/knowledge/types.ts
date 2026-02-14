/**
 * Mycelix Knowledge SDK Types
 *
 * Type definitions for the Knowledge hApp SDK, providing epistemic claims,
 * knowledge graph operations, fact-checking, and semantic search.
 *
 * @module @mycelix/sdk/integrations/knowledge
 */

import type { Record as _HolochainRecord } from '@holochain/client';

// ============================================================================
// Claim Types
// ============================================================================

/**
 * Epistemic claim with E/N/M classification
 */
export interface Claim {
  /** Unique claim identifier */
  id: string;
  /** Author's DID */
  author_did: string;
  /** Claim title */
  title: string;
  /** Full claim content */
  content: string;
  /** Empirical score (0-1) - E dimension */
  empirical: number;
  /** Normative score (0-1) - N dimension */
  normative: number;
  /** Metaphysical score (0-1) - M dimension */
  metaphysical: number;
  /** Tags for categorization */
  tags: string[];
  /** Creation timestamp (microseconds) */
  created_at: number;
  /** Last update timestamp (microseconds) */
  updated_at: number;
  /** Number of endorsements received */
  endorsement_count: number;
}

/**
 * Input for submitting a new claim
 */
export interface SubmitClaimInput {
  /** Unique claim identifier */
  id: string;
  /** Author's DID (must match caller) */
  author_did: string;
  /** Claim title */
  title: string;
  /** Full claim content */
  content: string;
  /** Empirical score (0-1) */
  empirical: number;
  /** Normative score (0-1) */
  normative: number;
  /** Metaphysical score (0-1) */
  metaphysical: number;
  /** Tags for categorization */
  tags: string[];
}

/**
 * Input for updating a claim
 */
export interface ClaimUpdate {
  /** New title (optional) */
  title?: string;
  /** New content (optional) */
  content?: string;
  /** New empirical score (optional) */
  empirical?: number;
  /** New normative score (optional) */
  normative?: number;
  /** New metaphysical score (optional) */
  metaphysical?: number;
  /** New tags (optional) */
  tags?: string[];
}

// ============================================================================
// Relation Types
// ============================================================================

/**
 * Relationship between claims
 */
export interface ClaimRelation {
  /** Unique relation identifier */
  id: string;
  /** Source claim ID */
  source_claim_id: string;
  /** Target claim ID */
  target_claim_id: string;
  /** Type of relationship */
  relation_type: RelationType;
  /** Relationship strength (0-1) */
  strength: number;
  /** Creator's DID */
  created_by: string;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

/**
 * Types of relationships between claims
 */
export type RelationType =
  | 'Supports'
  | 'Contradicts'
  | 'Extends'
  | 'Cites'
  | 'Supersedes';

// ============================================================================
// Ontology Types
// ============================================================================

/**
 * Ontology/schema for claims
 */
export interface Ontology {
  /** Unique ontology identifier */
  id: string;
  /** Ontology name */
  name: string;
  /** Version string */
  version: string;
  /** Description */
  description: string;
  /** JSON schema definition */
  schema: string;
  /** Author's DID */
  author_did: string;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

// ============================================================================
// Endorsement Types
// ============================================================================

/**
 * Claim endorsement
 */
export interface ClaimEndorsement {
  /** Unique endorsement identifier */
  id: string;
  /** Claim being endorsed */
  claim_id: string;
  /** Endorser's DID */
  endorser_did: string;
  /** Endorsement weight (MATL-based) */
  weight: number;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

// ============================================================================
// Source and Citation Types
// ============================================================================

/**
 * External source reference
 */
export interface Source {
  /** Unique source identifier */
  id: string;
  /** Type of source */
  source_type: SourceType;
  /** Source title */
  title: string;
  /** Author names */
  authors: string[];
  /** URL (optional) */
  url?: string;
  /** DOI (optional) */
  doi?: string;
  /** Publication date (optional, microseconds) */
  publication_date?: number;
  /** Publisher name (optional) */
  publisher?: string;
  /** Credibility score (0-1) */
  credibility_score: number;
  /** Who added this source */
  added_by: string;
  /** When added (microseconds) */
  added_at: number;
}

/**
 * Types of sources
 */
export type SourceType =
  | 'AcademicPaper'
  | 'Book'
  | 'NewsArticle'
  | 'WebPage'
  | 'Dataset'
  | 'PrimarySource'
  | 'Expert'
  | 'Official'
  | 'Other';

/**
 * Citation linking claim to source
 */
export interface Citation {
  /** Unique citation identifier */
  id: string;
  /** Claim ID */
  claim_id: string;
  /** Source ID */
  source_id: string;
  /** Type of citation */
  citation_type: CitationType;
  /** Specific location (page, section, timestamp) */
  specific_location?: string;
  /** Direct quote (optional) */
  quote?: string;
  /** Who created this citation */
  created_by: string;
  /** Creation timestamp (microseconds) */
  created_at: number;
  /** Whether citation has been verified */
  verified: boolean;
}

/**
 * Types of citations
 */
export type CitationType =
  | 'DirectQuote'
  | 'Paraphrase'
  | 'Reference'
  | 'Derived'
  | 'Contradicts';

/**
 * Input for adding a source
 */
export interface AddSourceInput {
  /** Unique source identifier */
  id: string;
  /** Type of source */
  source_type: SourceType;
  /** Source title */
  title: string;
  /** Author names */
  authors: string[];
  /** URL (optional) */
  url?: string;
  /** DOI (optional) */
  doi?: string;
  /** Publication date (optional, microseconds) */
  publication_date?: number;
  /** Publisher name (optional) */
  publisher?: string;
  /** Credibility score (optional, defaults to 0.5) */
  credibility_score?: number;
  /** Who added this source */
  added_by: string;
}

/**
 * Input for citing a source
 */
export interface CiteSourceInput {
  /** Unique citation identifier */
  id: string;
  /** Claim ID */
  claim_id: string;
  /** Source ID */
  source_id: string;
  /** Type of citation */
  citation_type: CitationType;
  /** Specific location (optional) */
  specific_location?: string;
  /** Direct quote (optional) */
  quote?: string;
  /** Who created this citation */
  created_by: string;
}

// ============================================================================
// Inference Types
// ============================================================================

/**
 * Inference request
 */
export interface InferenceRequest {
  /** Unique request identifier */
  id: string;
  /** Source claim IDs */
  source_claims: string[];
  /** Type of inference */
  inference_type: InferenceType;
  /** Current status */
  status: InferenceStatus;
  /** Inference result (if completed) */
  result?: string;
  /** Confidence in result (if completed) */
  confidence?: number;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

/**
 * Types of inference
 */
export type InferenceType = 'Deductive' | 'Inductive' | 'Abductive';

/**
 * Inference status
 */
export type InferenceStatus = 'Pending' | 'Processing' | 'Completed' | 'Failed';

/**
 * Consistency check result
 */
export interface ConsistencyResult {
  /** Whether claims are consistent */
  consistent: boolean;
  /** List of conflicts found */
  conflicts: ClaimConflict[];
}

/**
 * Conflict between claims
 */
export interface ClaimConflict {
  /** First claim ID */
  claim_a: string;
  /** Second claim ID */
  claim_b: string;
  /** Reason for conflict */
  reason: string;
}

/**
 * Reasoning input
 */
export interface ReasoningInput {
  /** Premise claim IDs */
  premise_claim_ids: string[];
  /** Inference rule to apply */
  inference_rule: InferenceRule;
  /** Additional context (optional) */
  context?: string;
}

/**
 * Inference rules
 */
export type InferenceRule =
  | 'ModusPonens'
  | 'ModusTollens'
  | 'Generalization'
  | 'Analogy';

/**
 * Reasoning result
 */
export interface ReasoningResult {
  /** Premise claim IDs */
  premises: string[];
  /** Applied rule */
  rule: InferenceRule;
  /** Derived conclusion (if any) */
  conclusion?: string;
  /** Confidence in conclusion */
  confidence: number;
  /** Whether reasoning is valid */
  valid: boolean;
  /** Explanation of reasoning */
  explanation: string;
}

// ============================================================================
// Graph Traversal Types
// ============================================================================

/**
 * Graph traversal input
 */
export interface TraversalInput {
  /** Starting claim ID */
  start_claim_id: string;
  /** Traversal direction */
  direction: TraversalDirection;
  /** Relation types to follow (optional, all if not specified) */
  relation_types?: RelationType[];
  /** Maximum depth to traverse */
  max_depth: number;
  /** Maximum claims to visit */
  limit: number;
}

/**
 * Traversal direction
 */
export type TraversalDirection = 'Outgoing' | 'Incoming' | 'Both';

/**
 * Graph traversal result
 */
export interface TraversalResult {
  /** Starting claim ID */
  start_claim: string;
  /** All visited claim IDs */
  visited_claims: string[];
  /** Paths discovered */
  paths: ClaimPath[];
  /** Maximum depth reached */
  depth_reached: number;
  /** Total relations traversed */
  total_relations: number;
}

/**
 * Path through claims
 */
export interface ClaimPath {
  /** Ordered claim IDs in path */
  claims: string[];
  /** Relation types between claims */
  relations: RelationType[];
  /** Product of relation strengths */
  total_strength: number;
}

/**
 * Path finding input
 */
export interface PathInput {
  /** Source claim ID */
  from_claim: string;
  /** Target claim ID */
  to_claim: string;
  /** Maximum path length */
  max_length: number;
}

/**
 * Claim cluster (connected component)
 */
export interface ClaimCluster {
  /** Center claim ID */
  center_claim: string;
  /** All claim IDs in cluster */
  claims: string[];
  /** Total endorsements in cluster */
  total_endorsements: number;
  /** Average empirical score */
  avg_empirical: number;
  /** Average normative score */
  avg_normative: number;
  /** Average metaphysical score */
  avg_metaphysical: number;
  /** Cluster coherence score (0-1) */
  coherence_score: number;
}

// ============================================================================
// Search Types
// ============================================================================

/**
 * Graph query input
 */
export interface GraphQuery {
  /** Query string */
  query: string;
  /** Ontology filter (optional) */
  ontology_id?: string;
  /** Maximum results */
  limit?: number;
}

/**
 * Semantic search input
 */
export interface SemanticSearchInput {
  /** Search query */
  query: string;
  /** Search filters (optional) */
  filters?: SearchFilters;
  /** Maximum results */
  limit: number;
  /** Results offset for pagination */
  offset: number;
}

/**
 * Search filters
 */
export interface SearchFilters {
  /** Minimum empirical score */
  min_empirical?: number;
  /** Maximum empirical score */
  max_empirical?: number;
  /** Minimum normative score */
  min_normative?: number;
  /** Maximum normative score */
  max_normative?: number;
  /** Minimum metaphysical score */
  min_metaphysical?: number;
  /** Maximum metaphysical score */
  max_metaphysical?: number;
  /** Filter by author DID */
  author_did?: string;
  /** Required tags (all must match) */
  tags?: string[];
  /** Minimum endorsement count */
  min_endorsements?: number;
  /** Filter by ontology */
  ontology_id?: string;
  /** Created after timestamp (microseconds) */
  created_after?: number;
  /** Created before timestamp (microseconds) */
  created_before?: number;
}

/**
 * Search result
 */
export interface SearchResult {
  /** Claim ID */
  claim_id: string;
  /** Relevance score (0-1) */
  score: number;
  /** Claim title */
  title: string;
  /** Content snippet */
  snippet: string;
  /** Author DID */
  author_did: string;
  /** Endorsement count */
  endorsement_count: number;
}

// ============================================================================
// Fact-Checking Types
// ============================================================================

/**
 * Fact-check request
 */
export interface FactCheckRequest {
  /** Unique request identifier */
  id: string;
  /** Media content ID (from Media hApp) */
  media_content_id: string;
  /** Claim text to verify */
  claim_text: string;
  /** Additional context (optional) */
  context?: string;
  /** Requester's DID */
  requested_by: string;
  /** Current status */
  status: FactCheckStatus;
  /** Result (if completed) */
  result?: FactCheckResult;
  /** Creation timestamp (microseconds) */
  created_at: number;
  /** Completion timestamp (optional, microseconds) */
  completed_at?: number;
}

/**
 * Fact-check status
 */
export type FactCheckStatus =
  | 'Pending'
  | 'InProgress'
  | 'Completed'
  | 'Inconclusive';

/**
 * Fact-check result
 */
export interface FactCheckResult {
  /** Verdict */
  verdict: FactCheckVerdict;
  /** Confidence in verdict (0-1) */
  confidence: number;
  /** Supporting claim IDs */
  supporting_claims: string[];
  /** Contradicting claim IDs */
  contradicting_claims: string[];
  /** Explanation of verdict */
  explanation: string;
  /** Source IDs used */
  sources: string[];
}

/**
 * Fact-check verdicts
 */
export type FactCheckVerdict =
  | 'True'
  | 'MostlyTrue'
  | 'Mixed'
  | 'MostlyFalse'
  | 'False'
  | 'Unverifiable'
  | 'OutOfContext'
  | 'Satire';

/**
 * Submit fact-check input
 */
export interface SubmitFactCheckInput {
  /** Unique request identifier */
  id: string;
  /** Media content ID */
  media_content_id: string;
  /** Claim text to verify */
  claim_text: string;
  /** Additional context (optional) */
  context?: string;
  /** Requester's DID */
  requested_by: string;
}

/**
 * Complete fact-check input
 */
export interface CompleteFactCheckInput {
  /** Request ID */
  request_id: string;
  /** Verdict */
  verdict: FactCheckVerdict;
  /** Confidence (0-1) */
  confidence: number;
  /** Supporting claim IDs */
  supporting_claims: string[];
  /** Contradicting claim IDs */
  contradicting_claims: string[];
  /** Explanation */
  explanation: string;
  /** Source IDs */
  sources: string[];
  /** Completer's DID */
  completed_by: string;
}

// ============================================================================
// Cross-hApp Types
// ============================================================================

/**
 * Cross-hApp link
 */
export interface CrossHappLink {
  /** Unique link identifier */
  id: string;
  /** Claim ID */
  claim_id: string;
  /** Target hApp name */
  target_happ: string;
  /** Target entry type */
  target_entry_type: string;
  /** Target entry ID */
  target_entry_id: string;
  /** Type of link */
  link_type: CrossHappLinkType;
  /** Creator's DID */
  created_by: string;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

/**
 * Cross-hApp link types
 */
export type CrossHappLinkType =
  | 'Substantiates'
  | 'References'
  | 'Describes'
  | 'Disputes'
  | 'Metadata';

/**
 * Link external input
 */
export interface LinkExternalInput {
  /** Unique link identifier */
  id: string;
  /** Claim ID */
  claim_id: string;
  /** Target hApp name */
  target_happ: string;
  /** Target entry type */
  target_entry_type: string;
  /** Target entry ID */
  target_entry_id: string;
  /** Type of link */
  link_type: CrossHappLinkType;
  /** Creator's DID */
  created_by: string;
}

/**
 * Get linked claims input
 */
export interface GetLinkedClaimsInput {
  /** Target hApp name */
  target_happ: string;
  /** Target entry ID */
  target_entry_id: string;
}

// ============================================================================
// Authority Types
// ============================================================================

/**
 * Knowledge authority score
 */
export interface KnowledgeAuthority {
  /** DID */
  did: string;
  /** Number of claims authored */
  claims_authored: number;
  /** Total endorsements received */
  total_endorsements_received: number;
  /** Average endorsement weight */
  avg_endorsement_weight: number;
  /** Number of citations */
  citation_count: number;
  /** H-index equivalent */
  h_index: number;
  /** Domain expertise areas */
  domains: DomainExpertise[];
  /** Overall authority score (0-1) */
  authority_score: number;
  /** When calculated (microseconds) */
  calculated_at: number;
}

/**
 * Domain expertise
 */
export interface DomainExpertise {
  /** Domain name */
  domain: string;
  /** Number of claims in domain */
  claim_count: number;
  /** Endorsement ratio */
  endorsement_ratio: number;
  /** Expertise level */
  expertise_level: ExpertiseLevel;
}

/**
 * Expertise levels
 */
export type ExpertiseLevel =
  | 'Novice'
  | 'Intermediate'
  | 'Advanced'
  | 'Expert'
  | 'Authority';

/**
 * Contradiction report
 */
export interface ContradictionReport {
  /** Number of claims analyzed */
  claims_analyzed: number;
  /** Potential contradictions found */
  potential_contradictions: PotentialContradiction[];
  /** Overall consistency score (0-1) */
  consistency_score: number;
  /** When analyzed (microseconds) */
  analyzed_at: number;
}

/**
 * Potential contradiction
 */
export interface PotentialContradiction {
  /** First claim ID */
  claim_a: string;
  /** Second claim ID */
  claim_b: string;
  /** Type of contradiction */
  contradiction_type: ContradictionType;
  /** Confidence in contradiction (0-1) */
  confidence: number;
  /** Explanation */
  explanation: string;
}

/**
 * Contradiction types
 */
export type ContradictionType =
  | 'Direct'
  | 'Implicit'
  | 'Contextual'
  | 'Temporal';

// ============================================================================
// Error Types
// ============================================================================

/**
 * Knowledge SDK error codes
 */
export type KnowledgeSdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_ENM_SCORES'
  | 'CLAIM_NOT_FOUND'
  | 'INVALID_RELATION'
  | 'CYCLE_DETECTED';

/**
 * Knowledge SDK error
 */
export class KnowledgeSdkError extends Error {
  constructor(
    public readonly code: KnowledgeSdkErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'KnowledgeSdkError';
  }
}
