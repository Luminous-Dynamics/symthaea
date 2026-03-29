// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID hApp Client Types
 *
 * Type definitions for the LUCID hApp SDK client,
 * providing coverage of all 8 LUCID zomes:
 * lucid, bridge, collective, reasoning, temporal, temporal-consciousness, sources, privacy
 *
 * @module @mycelix/sdk/clients/lucid/types
 */

// ============================================================================
// Common Types
// ============================================================================

export type Timestamp = number;

// ============================================================================
// Thought Types (lucid zome)
// ============================================================================

export type ThoughtType =
  | 'Belief'
  | 'Question'
  | 'Observation'
  | 'Hypothesis'
  | 'Argument'
  | 'Decision'
  | 'Reflection'
  | 'Intuition';

export type ConfidenceLevel = 'VeryLow' | 'Low' | 'Medium' | 'High' | 'VeryHigh';

export type RelationshipType =
  | 'Supports'
  | 'Contradicts'
  | 'Elaborates'
  | 'Questions'
  | 'Depends'
  | 'Supersedes'
  | 'RelatedTo';

export interface CreateThoughtInput {
  content: string;
  thought_type: ThoughtType;
  confidence: ConfidenceLevel;
  domain?: string;
  tags?: string[];
  parent_thought_id?: string;
  metadata?: Record<string, string>;
}

export interface UpdateThoughtInput {
  thought_id: string;
  content?: string;
  thought_type?: ThoughtType;
  confidence?: ConfidenceLevel;
  domain?: string;
  tags?: string[];
  metadata?: Record<string, string>;
}

export interface SearchThoughtsInput {
  query?: string;
  thought_type?: ThoughtType;
  domain?: string;
  tag?: string;
  confidence?: ConfidenceLevel;
  limit?: number;
}

export interface CreateTagInput {
  name: string;
  description?: string;
  color?: string;
}

export interface CreateDomainInput {
  name: string;
  description?: string;
  parent_domain?: string;
}

export interface CreateRelationshipInput {
  source_thought_id: string;
  target_thought_id: string;
  relationship_type: RelationshipType;
  description?: string;
  strength?: number;
}

export interface KnowledgeGraphStats {
  total_thoughts: number;
  total_relationships: number;
  total_tags: number;
  total_domains: number;
  thoughts_by_type: Record<string, number>;
  thoughts_by_domain: Record<string, number>;
}

export interface UpdateEmbeddingInput {
  thought_id: string;
  embedding: number[];
  embedding_version: number;
}

export interface UpdateCoherenceInput {
  thought_id: string;
  coherence_score: number;
  contradictions: string[];
}

export interface SemanticSearchInput {
  query_embedding: number[];
  limit?: number;
  min_similarity?: number;
  domain?: string;
}

export interface SemanticSearchResult {
  thought_id: string;
  similarity: number;
  record: unknown;
}

export interface ExploreGardenInput {
  similarity_threshold?: number;
  max_clusters?: number;
  domain?: string;
}

export interface ThoughtCluster {
  centroid_thought_id: string;
  thoughts: string[];
  average_similarity: number;
  dominant_domain?: string;
  dominant_type?: ThoughtType;
}

export interface SuggestConnectionsInput {
  thought_id: string;
  limit?: number;
  min_similarity?: number;
}

export interface ConnectionSuggestion {
  target_thought_id: string;
  similarity: number;
  suggested_relationship: RelationshipType;
  reason: string;
}

export interface KnowledgeGap {
  domain: string;
  gap_type: string;
  description: string;
  confidence: number;
}

export interface DiscoveredPattern {
  pattern_type: string;
  description: string;
  thought_ids: string[];
  confidence: number;
}

export interface CoherenceRangeInput {
  min_coherence: number;
  max_coherence: number;
}

// ============================================================================
// Bridge Types (bridge zome)
// ============================================================================

export interface StreamPhiInput {
  /** Phi (integrated information) from Symthaea (0.0-1.0) */
  phi: number;
  /** Coherence score at time of measurement (0.0-1.0) */
  coherence: number;
  /** Number of active thoughts in cognitive loop */
  active_thought_count: number;
  /** Symthaea engine version */
  engine_version?: string;
}

export interface PhiStreamEntry {
  agent: Uint8Array;
  phi: number;
  coherence: number;
  active_thought_count: number;
  measured_at: Timestamp;
  engine_version?: string;
}

export interface RecordFederationInput {
  thought_id: string;
  target_happ: string;
  target_dna: string;
  federation_type: string;
}

export interface CacheReputationInput {
  agent: Uint8Array;
  source_happ: string;
  score: number;
  evidence_count: number;
}

export interface StoreCoherenceInput {
  analysis_id: string;
  thought_ids: string[];
  coherence_score: number;
  contradictions: string[];
  suggestions: string[];
}

export interface GetCoherenceCandidatesInput {
  thought_id: string;
  limit?: number;
}

export interface CoherenceCandidate {
  thought_id: string;
  content: string;
  domain?: string;
  similarity?: number;
}

export interface CoherenceCheckResult {
  thought_id: string;
  coherence_score: number;
  contradictions: string[];
  checked_at: Timestamp;
}

// ============================================================================
// Collective Types (collective zome)
// ============================================================================

export interface ShareBeliefInput {
  thought_id: string;
  content: string;
  domain?: string;
  tags?: string[];
  confidence: ConfidenceLevel;
}

export interface CastVoteInput {
  belief_hash: Uint8Array;
  agreement: number;
  confidence: number;
  reasoning?: string;
}

export interface DetectPatternsInput {
  min_cluster_size?: number;
  similarity_threshold?: number;
  time_window_hours?: number;
}

export interface PatternCluster {
  pattern_id: string;
  beliefs: string[];
  average_agreement: number;
  description: string;
}

export interface PhiWeightedConsensusInput {
  belief_hash: Uint8Array;
  phi_scores: Record<string, number>;
}

export interface RecordPatternInput {
  pattern_type: string;
  description: string;
  belief_hashes: Uint8Array[];
  confidence: number;
}

export interface UpdateReputationInput {
  domain: string;
  accuracy_delta: number;
  participation_count: number;
}

export interface DomainExpertiseInput {
  agent: Uint8Array;
  domain: string;
}

export interface DomainExpertiseResult {
  domain: string;
  expertise_score: number;
  accuracy: number;
  participation_count: number;
  last_active: Timestamp;
}

export interface ConsensusExplanation {
  belief_hash: Uint8Array;
  total_votes: number;
  weighted_agreement: number;
  top_supporters: string[];
  top_dissenters: string[];
  phi_impact: string;
}

export interface CollectiveStats {
  total_beliefs: number;
  total_votes: number;
  total_patterns: number;
  total_agents: number;
  average_agreement: number;
}

export interface UpdateRelationshipInput {
  other_agent: Uint8Array;
  trust_delta: number;
  interaction_type: string;
}

export interface AgentRelationship {
  agent: Uint8Array;
  other_agent: Uint8Array;
  trust_score: number;
  interaction_count: number;
  last_interaction: Timestamp;
}

export interface WeightedConsensusResult {
  belief_hash: Uint8Array;
  weighted_agreement: number;
  confidence: number;
  voter_count: number;
  relationship_factor: number;
}

export interface TrendAnalysisInput {
  domain?: string;
  time_window_hours?: number;
  min_votes?: number;
}

export interface TrendAnalysisResult {
  trends: Array<{
    topic: string;
    direction: string;
    velocity: number;
    belief_count: number;
  }>;
  herding_warnings: string[];
}

export interface BeliefLifecycle {
  tag: string;
  phases: Array<{
    phase: string;
    start_time: Timestamp;
    belief_count: number;
    average_agreement: number;
  }>;
}

// ============================================================================
// Reasoning Types (reasoning zome)
// ============================================================================

export interface RecordContradictionInput {
  thought_a_id: string;
  thought_b_id: string;
  description: string;
  severity: number;
}

export interface RecordContradictionAnalyzedInput {
  thought_a_id: string;
  thought_b_id: string;
  description: string;
  severity: number;
  semantic_similarity: number;
  resolution_suggestion?: string;
}

export interface CreateReportInput {
  title: string;
  description: string;
  thought_ids: string[];
  overall_coherence: number;
  recommendations: string[];
}

export interface RecordInferenceInput {
  premise_thought_ids: string[];
  conclusion: string;
  inference_type: string;
  confidence: number;
}

// ============================================================================
// Temporal Types (temporal zome)
// ============================================================================

export interface RecordVersionInput {
  thought_id: string;
  content: string;
  change_reason?: string;
}

export interface BeliefAtTimeInput {
  thought_id: string;
  timestamp: Timestamp;
}

export interface CreateSnapshotInput {
  label?: string;
  description?: string;
  thought_ids: string[];
}

// ============================================================================
// Temporal Consciousness Types (temporal-consciousness zome)
// ============================================================================

export interface RecordSnapshotInput {
  thought_id: string;
  content: string;
  confidence: ConfidenceLevel;
  coherence_score?: number;
  phi_score?: number;
}

export interface BeliefTrajectory {
  thought_id: string;
  snapshots: Array<{
    content: string;
    confidence: ConfidenceLevel;
    coherence_score?: number;
    phi_score?: number;
    recorded_at: Timestamp;
  }>;
}

export interface TrajectoryAnalysis {
  thought_id: string;
  stability: number;
  direction: string;
  confidence_trend: string;
  coherence_trend: string;
  phase: string;
}

export interface RecordEvolutionInput {
  period_start: Timestamp;
  period_end: Timestamp;
  total_thoughts: number;
  new_thoughts: number;
  revised_thoughts: number;
  contradictions_resolved: number;
  average_coherence: number;
  average_phi: number;
  dominant_domains: string[];
}

export interface ConsciousnessEvolution {
  period_start: Timestamp;
  period_end: Timestamp;
  total_thoughts: number;
  new_thoughts: number;
  revised_thoughts: number;
  contradictions_resolved: number;
  average_coherence: number;
  average_phi: number;
  dominant_domains: string[];
}

// ============================================================================
// Sources Types (sources zome)
// ============================================================================

export type SourceType = 'Article' | 'Book' | 'Paper' | 'Website' | 'Video' | 'Podcast' | 'Person' | 'Experience' | 'Other';

export interface CreateSourceInput {
  title: string;
  url?: string;
  source_type: SourceType;
  author?: string;
  published_date?: string;
  description?: string;
  credibility_score?: number;
}

export interface CreateCitationInput {
  thought_id: string;
  source_id: string;
  excerpt?: string;
  page_number?: string;
  relevance?: number;
}

// ============================================================================
// Privacy Types (privacy zome)
// ============================================================================

export type SharingLevel = 'Private' | 'Trusted' | 'Community' | 'Public';

export interface SetPolicyInput {
  thought_id: string;
  sharing_level: SharingLevel;
  allowed_agents?: Uint8Array[];
}

export interface GrantAccessInput {
  thought_id: string;
  grantee: Uint8Array;
  expires_at?: Timestamp;
}

export interface CheckAccessInput {
  thought_id: string;
  agent: Uint8Array;
}

export interface LogAccessInput {
  thought_id: string;
  agent: Uint8Array;
  access_type: string;
}

export interface SubmitAttestationInput {
  proof_cid: string;
  proof_type: string;
  subject_hash: string;
  verifier: Uint8Array;
  verified: boolean;
  expires_at?: Timestamp;
  metadata?: Record<string, string>;
}

export interface ValidAttestationInput {
  subject_hash: string;
  proof_type?: string;
}

// ============================================================================
// Error Types
// ============================================================================

export const LucidErrorCode = {
  NOT_FOUND: 'LUCID_NOT_FOUND',
  VALIDATION_FAILED: 'LUCID_VALIDATION_FAILED',
  PERMISSION_DENIED: 'LUCID_PERMISSION_DENIED',
  CONNECTION_ERROR: 'LUCID_CONNECTION_ERROR',
} as const;

export class LucidError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'LucidError';
  }
}
