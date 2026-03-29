// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Type Definitions
 *
 * TypeScript types matching the Rust entry types in the LUCID hApp.
 */

import type { ActionHash, AgentPubKey, Timestamp } from '@holochain/client';

// ============================================================================
// EPISTEMIC CLASSIFICATION (E/N/M/H)
// ============================================================================

/** Empirical Level (E0-E4) - Degree of empirical validation */
export enum EmpiricalLevel {
  /** E0: Unverified - Personal intuition */
  E0 = 'E0',
  /** E1: Anecdotal - Single observation */
  E1 = 'E1',
  /** E2: Tested - Multiple observations */
  E2 = 'E2',
  /** E3: Verified - Systematic verification */
  E3 = 'E3',
  /** E4: Established - Consensus or proof */
  E4 = 'E4',
}

/** Normative Level (N0-N3) - Normative/ethical coherence */
export enum NormativeLevel {
  /** N0: Personal - Meaningful to self only */
  N0 = 'N0',
  /** N1: Contested - Recognized but disputed */
  N1 = 'N1',
  /** N2: Emerging - Growing acceptance */
  N2 = 'N2',
  /** N3: Endorsed - Broad agreement */
  N3 = 'N3',
}

/** Materiality Level (M0-M3) - Practical/material significance */
export enum MaterialityLevel {
  /** M0: Abstract - Purely theoretical */
  M0 = 'M0',
  /** M1: Potential - May have implications */
  M1 = 'M1',
  /** M2: Applicable - Practical applications */
  M2 = 'M2',
  /** M3: Transformative - Significant impact */
  M3 = 'M3',
}

/** Harmonic Level (H0-H4) - Alignment with higher purpose */
export enum HarmonicLevel {
  /** H0: Discordant - Potentially harmful */
  H0 = 'H0',
  /** H1: Neutral - No particular alignment */
  H1 = 'H1',
  /** H2: Resonant - Aligns with one harmony */
  H2 = 'H2',
  /** H3: Harmonic - Aligns with multiple harmonies */
  H3 = 'H3',
  /** H4: Transcendent - Serves universal flourishing */
  H4 = 'H4',
}

/** Full E/N/M/H epistemic classification */
export interface EpistemicClassification {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
  harmonic: HarmonicLevel;
}

// ============================================================================
// THOUGHT TYPES
// ============================================================================

/** Type of thought/knowledge unit */
export enum ThoughtType {
  Claim = 'Claim',
  Note = 'Note',
  Question = 'Question',
  Insight = 'Insight',
  Definition = 'Definition',
  Prediction = 'Prediction',
  Hypothesis = 'Hypothesis',
  Reflection = 'Reflection',
  Quote = 'Quote',
  Task = 'Task',
}

/** A thought - the primary knowledge unit in LUCID */
export interface Thought {
  id: string;
  content: string;
  thought_type: ThoughtType;
  epistemic: EpistemicClassification;
  confidence: number;
  tags: string[];
  domain: string | null;
  related_thoughts: string[];
  source_hashes: ActionHash[];
  parent_thought: string | null;
  created_at: Timestamp;
  updated_at: Timestamp;
  version: number;
  /** 16,384D HDC embedding from Symthaea */
  embedding: number[] | null;
  /** Version of thought when embedding was computed */
  embedding_version: number | null;
  /** Coherence score from Symthaea analysis (0.0-1.0) */
  coherence_score: number | null;
  /** Phi integration score from Symthaea analysis */
  phi_score: number | null;
}

/** Input for creating a thought */
export interface CreateThoughtInput {
  content: string;
  thought_type?: ThoughtType;
  epistemic?: EpistemicClassification;
  confidence?: number;
  tags?: string[];
  domain?: string;
  related_thoughts?: string[];
  parent_thought?: string;
  /** Optional pre-computed embedding from Symthaea (16,384D) */
  embedding?: number[];
}

/** Input for updating a thought */
export interface UpdateThoughtInput {
  thought_id: string;
  content?: string;
  thought_type?: ThoughtType;
  epistemic?: EpistemicClassification;
  confidence?: number;
  tags?: string[];
  domain?: string;
  related_thoughts?: string[];
}

/** Search filters for thoughts */
export interface SearchThoughtsInput {
  tags?: string[];
  domain?: string;
  thought_type?: ThoughtType;
  min_empirical?: EmpiricalLevel;
  min_confidence?: number;
  content_contains?: string;
  limit?: number;
  offset?: number;
}

// ============================================================================
// TAGS & DOMAINS
// ============================================================================

/** A tag for organizing thoughts */
export interface Tag {
  name: string;
  description: string | null;
  color: string | null;
  parent_tag: string | null;
  created_at: Timestamp;
}

/** Input for creating a tag */
export interface CreateTagInput {
  name: string;
  description?: string;
  color?: string;
  parent_tag?: string;
}

/** A knowledge domain */
export interface Domain {
  name: string;
  description: string | null;
  parent_domain: string | null;
  related_domains: string[];
  created_at: Timestamp;
}

/** Input for creating a domain */
export interface CreateDomainInput {
  name: string;
  description?: string;
  parent_domain?: string;
  related_domains?: string[];
}

// ============================================================================
// RELATIONSHIPS
// ============================================================================

/** Type of relationship between thoughts */
export enum RelationshipType {
  Supports = 'Supports',
  Contradicts = 'Contradicts',
  Implies = 'Implies',
  ImpliedBy = 'ImpliedBy',
  Refines = 'Refines',
  ExampleOf = 'ExampleOf',
  RelatedTo = 'RelatedTo',
  DependsOn = 'DependsOn',
  Supersedes = 'Supersedes',
  RespondsTo = 'RespondsTo',
}

/** A relationship between two thoughts */
export interface Relationship {
  id: string;
  from_thought_id: string;
  to_thought_id: string;
  relationship_type: RelationshipType;
  strength: number;
  description: string | null;
  bidirectional: boolean;
  created_at: Timestamp;
  active: boolean;
}

/** Input for creating a relationship */
export interface CreateRelationshipInput {
  from_thought_id: string;
  to_thought_id: string;
  relationship_type: RelationshipType;
  strength?: number;
  description?: string;
  bidirectional?: boolean;
}

// ============================================================================
// SOURCES
// ============================================================================

/** Type of source */
export enum SourceType {
  AcademicPaper = 'AcademicPaper',
  Book = 'Book',
  NewsArticle = 'NewsArticle',
  BlogPost = 'BlogPost',
  WebPage = 'WebPage',
  Video = 'Video',
  Audio = 'Audio',
  SocialMedia = 'SocialMedia',
  Conversation = 'Conversation',
  PersonalExperience = 'PersonalExperience',
  OfficialDocument = 'OfficialDocument',
  Dataset = 'Dataset',
  LucidThought = 'LucidThought',
  KnowledgeBase = 'KnowledgeBase',
  Other = 'Other',
}

/** A source of knowledge */
export interface Source {
  id: string;
  source_type: SourceType;
  title: string;
  url: string | null;
  authors: string[];
  publication_date: string | null;
  publisher: string | null;
  doi: string | null;
  isbn: string | null;
  description: string | null;
  content_hash: string | null;
  credibility: number;
  notes: string | null;
  tags: string[];
  added_at: Timestamp;
  updated_at: Timestamp;
}

/** Input for creating a source */
export interface CreateSourceInput {
  source_type: SourceType;
  title: string;
  url?: string;
  authors?: string[];
  publication_date?: string;
  publisher?: string;
  doi?: string;
  isbn?: string;
  description?: string;
  credibility?: number;
  notes?: string;
  tags?: string[];
}

/** How a source relates to a thought */
export enum CitationRelationship {
  Supports = 'Supports',
  Refutes = 'Refutes',
  Context = 'Context',
  Origin = 'Origin',
  Alternative = 'Alternative',
  Methodology = 'Methodology',
  Definition = 'Definition',
  Reference = 'Reference',
}

/** A citation linking a thought to a source */
export interface Citation {
  id: string;
  thought_id: string;
  source_id: string;
  location: string | null;
  quote: string | null;
  relationship: CitationRelationship;
  strength: number;
  notes: string | null;
  created_at: Timestamp;
}

/** Input for creating a citation */
export interface CreateCitationInput {
  thought_id: string;
  source_id: string;
  location?: string;
  quote?: string;
  relationship?: CitationRelationship;
  strength?: number;
  notes?: string;
}

// ============================================================================
// TEMPORAL CONSCIOUSNESS (Trajectory + Evolution)
// ============================================================================

/** What triggered the creation of a belief snapshot */
export enum SnapshotTrigger {
  Created = 'Created',
  Edited = 'Edited',
  Scheduled = 'Scheduled',
  ContradictionDetected = 'ContradictionDetected',
  ReviewedConfirmed = 'ReviewedConfirmed',
  ReviewedModified = 'ReviewedModified',
  EvidenceUpdated = 'EvidenceUpdated',
  CoherenceChanged = 'CoherenceChanged',
}

/** Classification of how a belief has evolved */
export enum TrajectoryType {
  Stable = 'Stable',
  Growing = 'Growing',
  Weakening = 'Weakening',
  Oscillating = 'Oscillating',
  Entrenched = 'Entrenched',
  Volatile = 'Volatile',
  Insufficient = 'Insufficient',
}

/** Level of entrenchment (resistance to change despite evidence) */
export enum EntrenchmentLevel {
  None = 'None',
  Mild = 'Mild',
  Moderate = 'Moderate',
  High = 'High',
  Severe = 'Severe',
}

/** A snapshot of belief state at a point in time */
export interface BeliefSnapshot {
  epistemic_code: string;
  confidence: number;
  phi: number;
  coherence: number;
  timestamp: Timestamp;
  trigger: SnapshotTrigger;
}

/** A trajectory tracking the evolution of a belief over time */
export interface BeliefTrajectoryRecord {
  thought_id: string;
  snapshots: BeliefSnapshot[];
  trajectory_type: TrajectoryType;
  entrenchment: EntrenchmentLevel | null;
  started_at: Timestamp;
  last_updated: Timestamp;
}

/** Analysis of a belief trajectory */
export interface TrajectoryAnalysis {
  id: string;
  thought_id: string;
  trajectory_type: TrajectoryType;
  confidence_trend: number;
  phi_trend: number;
  coherence_trend: number;
  entrenchment: EntrenchmentLevel | null;
  contradiction_count: number;
  recommendation: string;
  analyzed_at: Timestamp;
}

/** Consciousness evolution summary over a time period */
export interface ConsciousnessEvolution {
  id: string;
  period_start: Timestamp;
  period_end: Timestamp;
  avg_phi: number;
  phi_trend: number;
  avg_coherence: number;
  coherence_trend: number;
  stable_belief_count: number;
  growing_belief_count: number;
  weakening_belief_count: number;
  entrenched_belief_count: number;
  insights: string[];
  created_at: Timestamp;
}

/** Input for recording a belief snapshot */
export interface RecordSnapshotInput {
  thought_id: string;
  epistemic_code: string;
  confidence: number;
  phi: number;
  coherence: number;
  trigger: SnapshotTrigger;
}

/** Input for recording consciousness evolution */
export interface RecordEvolutionInput {
  period_start: Timestamp;
  period_end: Timestamp;
  avg_phi: number;
  phi_trend: number;
  avg_coherence: number;
  coherence_trend: number;
  stable_belief_count: number;
  growing_belief_count: number;
  weakening_belief_count: number;
  entrenched_belief_count: number;
  insights: string[];
}

// ============================================================================
// TEMPORAL (Belief Versioning)
// ============================================================================

/** A snapshot of a thought at a point in time */
export interface BeliefVersion {
  thought_id: string;
  version: number;
  content: string;
  confidence: number;
  epistemic_code: string;
  change_reason: string | null;
  previous_version: ActionHash | null;
  timestamp: Timestamp;
}

/** Input for recording a version */
export interface RecordVersionInput {
  thought_id: string;
  version: number;
  content: string;
  confidence: number;
  epistemic_code: string;
  change_reason?: string;
  previous_version?: ActionHash;
}

/** A knowledge graph snapshot */
export interface GraphSnapshot {
  id: string;
  name: string;
  description: string | null;
  timestamp: Timestamp;
  thought_count: number;
  avg_confidence: number;
  tags: string[];
}

/** Input for creating a snapshot */
export interface CreateSnapshotInput {
  name: string;
  description?: string;
  thought_count: number;
  avg_confidence: number;
  tags?: string[];
}

/** Input for getting belief at a specific time */
export interface BeliefAtTimeInput {
  thought_id: string;
  timestamp: Timestamp;
}

// ============================================================================
// PRIVACY
// ============================================================================

/** Visibility level for a thought */
export type Visibility =
  | { type: 'Private' }
  | { type: 'SharedWith'; agents: AgentPubKey[] }
  | { type: 'GroupMembers'; group_id: string }
  | { type: 'Federated'; happs: string[] }
  | { type: 'Public' };

/** Sharing policy for a thought */
export interface SharingPolicy {
  thought_id: string;
  visibility: Visibility;
  expires_at: Timestamp | null;
  delegatable: boolean;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/** Input for setting a sharing policy */
export interface SetPolicyInput {
  thought_id: string;
  visibility: Visibility;
  expires_at?: Timestamp;
  delegatable?: boolean;
}

/** An access grant */
export interface AccessGrant {
  id: string;
  thought_id: string;
  grantee: AgentPubKey;
  grantor: AgentPubKey;
  permissions: string[];
  expires_at: Timestamp | null;
  can_delegate: boolean;
  revoked: boolean;
  created_at: Timestamp;
}

/** Input for granting access */
export interface GrantAccessInput {
  thought_id: string;
  grantee: AgentPubKey;
  permissions: string[];
  expires_at?: Timestamp;
  can_delegate?: boolean;
}

/** Input for checking access */
export interface CheckAccessInput {
  thought_id: string;
  agent: AgentPubKey;
}

// ============================================================================
// REASONING
// ============================================================================

/** Type of contradiction */
export enum ContradictionType {
  Logical = 'Logical',
  Factual = 'Factual',
  Temporal = 'Temporal',
  SourceConflict = 'SourceConflict',
  ConfidenceInconsistency = 'ConfidenceInconsistency',
}

/** A detected contradiction */
export interface Contradiction {
  id: string;
  thought_a_id: string;
  thought_b_id: string;
  contradiction_type: ContradictionType;
  severity: number;
  explanation: string;
  suggested_resolution: string | null;
  resolved: boolean;
  resolution_notes: string | null;
  detected_at: Timestamp;
  resolved_at: Timestamp | null;
}

/** Input for recording a contradiction */
export interface RecordContradictionInput {
  thought_a_id: string;
  thought_b_id: string;
  contradiction_type: ContradictionType;
  severity: number;
  explanation: string;
  suggested_resolution?: string;
}

/** A coherence report */
export interface CoherenceReport {
  id: string;
  thought_ids: string[];
  coherence_score: number;
  phi_estimate: number | null;
  contradiction_count: number;
  knowledge_gaps: string[];
  suggestions: string[];
  created_at: Timestamp;
}

/** Input for creating a coherence report */
export interface CreateReportInput {
  thought_ids: string[];
  coherence_score: number;
  phi_estimate?: number;
  contradiction_count: number;
  knowledge_gaps: string[];
  suggestions: string[];
}

/** Type of inference */
export enum InferenceType {
  Deduction = 'Deduction',
  Induction = 'Induction',
  Abduction = 'Abduction',
  Analogy = 'Analogy',
}

/** An inference derived from existing thoughts */
export interface Inference {
  id: string;
  content: string;
  premise_ids: string[];
  inference_type: InferenceType;
  confidence: number;
  accepted: boolean;
  created_at: Timestamp;
}

/** Input for recording an inference */
export interface RecordInferenceInput {
  content: string;
  premise_ids: string[];
  inference_type: InferenceType;
  confidence: number;
}

// ============================================================================
// STATISTICS
// ============================================================================

/** Statistics about the knowledge graph */
export interface KnowledgeGraphStats {
  total_thoughts: number;
  by_type: [string, number][];
  by_domain: [string, number][];
  average_confidence: number;
  average_epistemic_strength: number;
  top_tags: [string, number][];
}

// ============================================================================
// BRIDGE
// ============================================================================

/** Federation status */
export enum FederationStatus {
  Pending = 'Pending',
  Active = 'Active',
  Revoked = 'Revoked',
  Failed = 'Failed',
}

/** Federation record */
export interface FederationRecord {
  thought_id: string;
  target_happ: string;
  external_id: string | null;
  status: FederationStatus;
  federated_at: Timestamp;
}

/** External reputation score */
export interface ExternalReputation {
  agent: AgentPubKey;
  k_vector: [number, number, number, number, number, number, number, number];
  trust_score: number;
  source_happ: string;
  fetched_at: Timestamp;
}

/** Coherence check result from Symthaea */
export interface CoherenceResult {
  coherent: boolean;
  phi_estimate: number;
  suggestions: string[];
}

// ============================================================================
// COLLECTIVE SENSEMAKING
// ============================================================================

/** Stance on a belief */
export enum BeliefStance {
  StronglyAgree = 'StronglyAgree',
  Agree = 'Agree',
  Neutral = 'Neutral',
  Disagree = 'Disagree',
  StronglyDisagree = 'StronglyDisagree',
}

/** A shared belief in the collective */
export interface BeliefShare {
  content_hash: string;
  content: string;
  belief_type: string;
  epistemic_code: string;
  confidence: number;
  tags: string[];
  shared_at: Timestamp;
  evidence_hashes: string[];
  embedding: number[];
  stance: BeliefStance | null;
}

/** Input for sharing a belief */
export interface ShareBeliefInput {
  content_hash: string;
  content: string;
  belief_type: string;
  epistemic_code: string;
  confidence: number;
  tags: string[];
  evidence_hashes: string[];
  embedding?: number[];
  stance?: BeliefStance;
}

/** Vote types for belief validation */
export enum ValidationVoteType {
  Corroborate = 'Corroborate',
  Contradict = 'Contradict',
  Plausible = 'Plausible',
  Implausible = 'Implausible',
  Abstain = 'Abstain',
}

/** A validation vote on a shared belief */
export interface ValidationVote {
  belief_share_hash: ActionHash;
  vote_type: ValidationVoteType;
  evidence: string | null;
  voter_weight: number;
  voted_at: Timestamp;
}

/** Input for casting a vote */
export interface CastVoteInput {
  belief_share_hash: ActionHash;
  vote_type: ValidationVoteType;
  evidence?: string;
}

/** Types of consensus */
export enum ConsensusType {
  StrongConsensus = 'StrongConsensus',
  ModerateConsensus = 'ModerateConsensus',
  WeakConsensus = 'WeakConsensus',
  Contested = 'Contested',
  Insufficient = 'Insufficient',
}

/** A consensus record for a belief */
export interface ConsensusRecord {
  belief_share_hash: ActionHash;
  consensus_type: ConsensusType;
  validator_count: number;
  agreement_score: number;
  summary: string;
  reached_at: Timestamp;
}

/** Types of emergent patterns */
export enum PatternType {
  Convergence = 'Convergence',
  Divergence = 'Divergence',
  Trend = 'Trend',
  Cluster = 'Cluster',
  ContradictionCluster = 'ContradictionCluster',
}

/** An emergent pattern detected in the collective */
export interface EmergentPattern {
  pattern_id: string;
  description: string;
  belief_hashes: ActionHash[];
  pattern_type: PatternType;
  confidence: number;
  detected_at: Timestamp;
}

/** Input for recording a pattern */
export interface RecordPatternInput {
  pattern_id: string;
  description: string;
  belief_hashes: ActionHash[];
  pattern_type: PatternType;
  confidence: number;
}

/** Input for detecting patterns */
export interface DetectPatternsInput {
  similarity_threshold?: number;
  min_cluster_size?: number;
}

/** A pattern cluster from detection */
export interface PatternCluster {
  pattern_id: string;
  representative_content: string;
  representative_hash: ActionHash;
  member_hashes: ActionHash[];
  member_count: number;
  pattern_type: PatternType;
  coherence: number;
  tags: string[];
}

/** Epistemic reputation for a validator */
export interface EpistemicReputation {
  agent: AgentPubKey;
  domains: string[];
  accuracy_score: number;
  calibration_score: number;
  validation_count: number;
  updated_at: Timestamp;
}

/** Input for updating reputation */
export interface UpdateReputationInput {
  agent: AgentPubKey;
  domains: string[];
  accuracy_score: number;
  calibration_score: number;
  validation_count: number;
}

/** Relationship stages */
export enum RelationshipStage {
  NoRelation = 'NoRelation',
  Acquaintance = 'Acquaintance',
  Collaborator = 'Collaborator',
  TrustedPeer = 'TrustedPeer',
  PartnerInTruth = 'PartnerInTruth',
}

/** Relationship between agents */
export interface AgentRelationship {
  other_agent: AgentPubKey;
  trust_score: number;
  interaction_count: number;
  last_interaction: Timestamp;
  relationship_stage: RelationshipStage;
  shared_domains: string[];
  agreement_ratio: number;
}

/** Input for updating a relationship */
export interface UpdateRelationshipInput {
  other_agent: AgentPubKey;
  trust_delta?: number;
  relationship_stage?: RelationshipStage;
  new_domains?: string[];
  agreement_ratio?: number;
}

/** Weighted consensus result */
export interface WeightedConsensusResult {
  weighted_value: number;
  confidence: number;
  total_weight: number;
  voter_count: number;
  breakdown: ConsensusBreakdown;
}

/** Consensus breakdown by vote type */
export interface ConsensusBreakdown {
  corroborate: number;
  plausible: number;
  abstain: number;
  implausible: number;
  contradict: number;
}

/** Collective statistics */
export interface CollectiveStats {
  total_belief_shares: number;
  total_patterns: number;
  active_validators: number;
}

// ============================================================================
// SYMTHAEA INTEGRATION - EMBEDDINGS & COHERENCE
// ============================================================================

/** Input for updating a thought's embedding */
export interface UpdateEmbeddingInput {
  thought_id: string;
  /** 16,384-dimensional HDC embedding from Symthaea */
  embedding: number[];
}

/** Input for updating coherence/phi scores */
export interface UpdateCoherenceInput {
  thought_id: string;
  coherence_score: number;
  phi_score?: number;
}

/** Input for semantic search */
export interface SemanticSearchInput {
  /** Query embedding (16,384 dimensions) */
  query_embedding: number[];
  /** Minimum similarity threshold (0.0-1.0) */
  threshold: number;
  /** Maximum number of results */
  limit?: number;
}

/** Result of semantic search */
export interface SemanticSearchResult {
  thought_id: string;
  action_hash: ActionHash;
  similarity: number;
  content_preview: string;
}

/** Input for coherence range query */
export interface CoherenceRangeInput {
  min_coherence: number;
  max_coherence: number;
}

// ============================================================================
// EPISTEMIC GARDEN
// ============================================================================

/** Input for exploring the epistemic garden */
export interface ExploreGardenInput {
  max_clusters: number;
  min_cluster_size: number;
  domain_filter?: string;
}

/** A cluster of semantically related thoughts */
export interface ThoughtCluster {
  cluster_id: number;
  centroid_thought_id: string;
  thought_ids: string[];
  avg_confidence: number;
  avg_coherence: number | null;
  dominant_domain: string | null;
  dominant_tags: string[];
}

/** Input for suggesting connections */
export interface SuggestConnectionsInput {
  thought_id: string;
  max_suggestions: number;
  min_similarity: number;
}

/** A suggestion for connecting two unlinked thoughts */
export interface ConnectionSuggestion {
  thought_a_id: string;
  thought_b_id: string;
  similarity: number;
  shared_tags: string[];
}

/** A knowledge gap in a domain */
export interface KnowledgeGap {
  domain: string;
  thought_count: number;
  avg_confidence: number;
  low_confidence_count: number;
  missing_embeddings: number;
  sparse: boolean;
}

/** A discovered pattern in the knowledge graph */
export interface DiscoveredPattern {
  pattern_type: string;
  description: string;
  involved_thought_ids: string[];
  confidence: number;
}
