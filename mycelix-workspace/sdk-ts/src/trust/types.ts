/**
 * @mycelix/sdk Trust Types
 *
 * TypeScript type definitions for the trust pipeline, K-Vector, and domain translation.
 * These types mirror the Rust implementations in sdk/src/agentic/.
 *
 * @packageDocumentation
 * @module trust/types
 */

import type { HarmonicLevel } from '../epistemic/gis.js';
import type { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../epistemic/types.js';

// ============================================================================
// K-Vector Types
// ============================================================================

/**
 * K-Vector dimension enumeration
 */
export enum KVectorDimension {
  Reputation = 0,
  Activity = 1,
  Integrity = 2,
  Performance = 3,
  Membership = 4,
  Stake = 5,
  Historical = 6,
  Topology = 7,
  Verification = 8,
}

/**
 * Dimension names for display
 */
export const KVECTOR_DIMENSION_NAMES: Record<KVectorDimension, string> = {
  [KVectorDimension.Reputation]: 'Reputation',
  [KVectorDimension.Activity]: 'Activity',
  [KVectorDimension.Integrity]: 'Integrity',
  [KVectorDimension.Performance]: 'Performance',
  [KVectorDimension.Membership]: 'Membership',
  [KVectorDimension.Stake]: 'Stake',
  [KVectorDimension.Historical]: 'Historical',
  [KVectorDimension.Topology]: 'Topology',
  [KVectorDimension.Verification]: 'Verification',
};

/**
 * Default K-Vector dimension weights
 */
export const KVECTOR_WEIGHTS: Record<KVectorDimension, number> = {
  [KVectorDimension.Reputation]: 0.15,
  [KVectorDimension.Activity]: 0.10,
  [KVectorDimension.Integrity]: 0.15,
  [KVectorDimension.Performance]: 0.15,
  [KVectorDimension.Membership]: 0.10,
  [KVectorDimension.Stake]: 0.10,
  [KVectorDimension.Historical]: 0.10,
  [KVectorDimension.Topology]: 0.10,
  [KVectorDimension.Verification]: 0.05,
};

/**
 * 9-dimensional trust vector
 *
 * Each dimension represents a different aspect of trustworthiness:
 * - k_r: Reputation - community standing
 * - k_a: Activity - recent participation
 * - k_i: Integrity - honest behavior
 * - k_p: Performance - task completion quality
 * - k_m: Membership - duration in network
 * - k_s: Stake - economic commitment
 * - k_h: Historical - consistency over time
 * - k_topo: Topology - network position
 * - k_v: Verification - identity verification level
 */
export interface KVector {
  /** Reputation dimension (0.0-1.0) */
  k_r: number;
  /** Activity dimension (0.0-1.0) */
  k_a: number;
  /** Integrity dimension (0.0-1.0) */
  k_i: number;
  /** Performance dimension (0.0-1.0) */
  k_p: number;
  /** Membership dimension (0.0-1.0) */
  k_m: number;
  /** Stake dimension (0.0-1.0) */
  k_s: number;
  /** Historical dimension (0.0-1.0) */
  k_h: number;
  /** Topology dimension (0.0-1.0) */
  k_topo: number;
  /** Verification dimension (0.0-1.0) */
  k_v: number;
}

/**
 * Create a new K-Vector with all dimensions set to the same value
 */
export function createUniformKVector(value: number): KVector {
  const clamped = Math.max(0, Math.min(1, value));
  return {
    k_r: clamped,
    k_a: clamped,
    k_i: clamped,
    k_p: clamped,
    k_m: clamped,
    k_s: clamped,
    k_h: clamped,
    k_topo: clamped,
    k_v: clamped,
  };
}

/**
 * Create a K-Vector from an array of 9 values
 */
export function kvectorFromArray(arr: [number, number, number, number, number, number, number, number, number]): KVector {
  return {
    k_r: Math.max(0, Math.min(1, arr[0])),
    k_a: Math.max(0, Math.min(1, arr[1])),
    k_i: Math.max(0, Math.min(1, arr[2])),
    k_p: Math.max(0, Math.min(1, arr[3])),
    k_m: Math.max(0, Math.min(1, arr[4])),
    k_s: Math.max(0, Math.min(1, arr[5])),
    k_h: Math.max(0, Math.min(1, arr[6])),
    k_topo: Math.max(0, Math.min(1, arr[7])),
    k_v: Math.max(0, Math.min(1, arr[8])),
  };
}

/**
 * Convert K-Vector to array
 */
export function kvectorToArray(kv: KVector): [number, number, number, number, number, number, number, number, number] {
  return [kv.k_r, kv.k_a, kv.k_i, kv.k_p, kv.k_m, kv.k_s, kv.k_h, kv.k_topo, kv.k_v];
}

/**
 * Compute weighted trust score from K-Vector
 */
export function computeTrustScore(kv: KVector, weights: Record<KVectorDimension, number> = KVECTOR_WEIGHTS): number {
  const arr = kvectorToArray(kv);
  let weightedSum = 0;
  let weightSum = 0;

  for (let i = 0; i < 9; i++) {
    const weight = weights[i as KVectorDimension];
    weightedSum += arr[i] * weight;
    weightSum += weight;
  }

  return weightSum > 0 ? weightedSum / weightSum : 0;
}

/**
 * Check if K-Vector indicates verification
 */
export function isVerified(kv: KVector, threshold = 0.5): boolean {
  return kv.k_v >= threshold;
}

// ============================================================================
// Trust Direction
// ============================================================================

/**
 * Direction of trust change
 */
export enum TrustDirection {
  Increase = 'Increase',
  Decrease = 'Decrease',
  Unchanged = 'Unchanged',
}

// ============================================================================
// Trust Delta
// ============================================================================

/**
 * Change in trust applied to K-Vector
 */
export interface TrustDelta {
  /** Change to reputation (k_r) */
  reputationDelta: number;
  /** Change to performance (k_p) */
  performanceDelta: number;
  /** Change to integrity (k_i) */
  integrityDelta: number;
  /** Change to historical (k_h) */
  historicalDelta: number;
  /** Overall direction */
  direction: TrustDirection;
  /** Reason for the change */
  reason: string;
}

/**
 * Apply a trust delta to a K-Vector
 */
export function applyTrustDelta(kv: KVector, delta: TrustDelta): KVector {
  return {
    k_r: Math.max(0, Math.min(1, kv.k_r + delta.reputationDelta)),
    k_a: kv.k_a, // Activity unchanged by deltas
    k_i: Math.max(0, Math.min(1, kv.k_i + delta.integrityDelta)),
    k_p: Math.max(0, Math.min(1, kv.k_p + delta.performanceDelta)),
    k_m: kv.k_m, // Membership unchanged
    k_s: kv.k_s, // Stake unchanged
    k_h: Math.max(0, Math.min(1, kv.k_h + delta.historicalDelta)),
    k_topo: kv.k_topo, // Topology unchanged
    k_v: kv.k_v, // Verification unchanged
  };
}

// ============================================================================
// Epistemic Classification Extended
// ============================================================================

/**
 * Extended epistemic classification (E-N-M-H)
 */
export interface EpistemicClassificationExtended {
  /** Empirical level (E0-E4) */
  empirical: EmpiricalLevel;
  /** Normative level (N0-N3) */
  normative: NormativeLevel;
  /** Materiality level (M0-M3) */
  materiality: MaterialityLevel;
  /** Harmonic level (H0-H4) */
  harmonic: HarmonicLevel;
}

/**
 * Generate classification code string
 */
export function classificationCode(c: EpistemicClassificationExtended): string {
  return `E${c.empirical}-N${c.normative}-M${c.materiality}-H${c.harmonic}`;
}

// ============================================================================
// Agent Output
// ============================================================================

/**
 * Content types for agent outputs
 */
export type OutputContent =
  | { type: 'text'; value: string }
  | { type: 'json'; value: Record<string, unknown> }
  | { type: 'binary'; hash: string; size: number }
  | { type: 'reference'; uri: string }
  | { type: 'composite'; parts: OutputContent[] };

/**
 * Output from an AI agent
 */
export interface AgentOutput {
  /** Unique output identifier */
  outputId: string;
  /** Agent that produced this output */
  agentId: string;
  /** Content of the output */
  content: OutputContent;
  /** Epistemic classification (E-N-M-H) */
  classification: EpistemicClassificationExtended;
  /** Confidence in the classification (0.0-1.0) */
  classificationConfidence: number;
  /** When this output was created */
  timestamp: number;
  /** Whether this output includes a proof */
  hasProof: boolean;
  /** Optional proof data (hex-encoded) */
  proofData?: string;
  /** References to context used */
  contextReferences: string[];
}

// ============================================================================
// Provenance Types
// ============================================================================

/**
 * How knowledge was derived
 */
export enum DerivationType {
  /** Original knowledge (not derived) */
  Original = 'Original',
  /** Derived through inference */
  Inference = 'Inference',
  /** Derived through transformation */
  Transformation = 'Transformation',
  /** Aggregated from multiple sources */
  Aggregation = 'Aggregation',
  /** Filtered from larger set */
  Filtering = 'Filtering',
  /** Summary of longer content */
  Summarization = 'Summarization',
  /** Verified by external source */
  Verification = 'Verification',
}

/**
 * A node in the provenance graph
 */
export interface ProvenanceNode {
  /** Unique identifier */
  nodeId: string;
  /** SHA3-256 hash of content */
  contentHash: string;
  /** Agent who created this */
  creator: string;
  /** When this was created */
  timestamp: number;
  /** Epistemic level (E0-E4) */
  epistemicLevel: number;
  /** Minimum epistemic level in ancestry */
  epistemicFloor: number;
  /** Confidence (0.0-1.0) */
  confidence: number;
  /** How this was derived */
  derivationType: DerivationType;
  /** Parent node IDs */
  parentIds: string[];
  /** Cryptographic signature */
  signature?: string;
}

/**
 * A chain of provenance nodes
 */
export interface ProvenanceChain {
  /** Root node ID */
  rootId: string;
  /** All nodes in the chain */
  nodes: ProvenanceNode[];
  /** Chain depth */
  depth: number;
  /** Minimum epistemic level in chain */
  minEpistemicLevel: number;
  /** Chain confidence */
  confidence: number;
  /** Whether chain has verified roots */
  hasVerifiedRoots: boolean;
  /** Total derivation steps */
  derivationSteps: number;
}

// ============================================================================
// Consensus Types
// ============================================================================

/**
 * Status of consensus
 */
export enum PhiConsensusStatus {
  /** Consensus reached with high coherence */
  Reached = 'Reached',
  /** Consensus reached but with low coherence */
  ReachedLowCoherence = 'ReachedLowCoherence',
  /** No consensus - too much disagreement */
  Deadlock = 'Deadlock',
  /** Insufficient votes */
  InsufficientVotes = 'InsufficientVotes',
  /** Consensus pending */
  Pending = 'Pending',
}

/**
 * Vote from an agent
 */
export interface AgentVote {
  /** Agent ID */
  agentId: string;
  /** Vote value (0.0-1.0) */
  value: number;
  /** Confidence in vote (0.0-1.0) */
  confidence: number;
  /** Epistemic level of voter */
  epistemicLevel: number;
  /** Optional reasoning */
  reasoning?: string;
  /** Vote timestamp */
  timestamp: number;
}

/**
 * Result of phi-weighted consensus
 */
export interface PhiConsensusResult {
  /** Consensus value (0.0-1.0) */
  consensusValue: number;
  /** Confidence in consensus */
  confidence: number;
  /** Number of participants */
  participantCount: number;
  /** Total trust weight */
  totalTrustWeight: number;
  /** Dissent level */
  dissent: number;
  /** Whether consensus was reached */
  consensusReached: boolean;
  /** Status */
  status: PhiConsensusStatus;
  /** Population phi (coherence) */
  populationPhi: number;
  /** Harmonic confidence */
  harmonicConfidence: number;
}

// ============================================================================
// Pipeline Types
// ============================================================================

/**
 * Configuration for the trust pipeline
 */
export interface PipelineConfig {
  /** Minimum consensus value for positive trust delta */
  minConsensusForTrust: number;
  /** Maximum trust delta per run */
  maxTrustDelta: number;
  /** Whether to require ZK attestation */
  requireAttestation: boolean;
  /** Default domain ID */
  defaultDomainId?: string;
  /** Enable provenance tracking */
  trackProvenance: boolean;
  /** Phi consensus threshold */
  phiConsensusThreshold: number;
  /** Minimum participants for valid consensus */
  minParticipants: number;
}

/**
 * Default pipeline configuration
 */
export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  minConsensusForTrust: 0.5,
  maxTrustDelta: 0.1,
  requireAttestation: false,
  trackProvenance: true,
  phiConsensusThreshold: 0.6,
  minParticipants: 2,
};

/**
 * Stage 1: Registered output with provenance
 */
export interface RegisteredOutput {
  /** The agent output */
  output: AgentOutput;
  /** Provenance node */
  provenance: ProvenanceNode;
  /** Parent provenance nodes */
  parents: ProvenanceNode[];
  /** Registration timestamp */
  registeredAt: number;
  /** Agent ID */
  agentId: string;
}

/**
 * Stage 2: Consensus outcome
 */
export interface ConsensusOutcome {
  /** Registered output */
  output: RegisteredOutput;
  /** Consensus result */
  consensus: PhiConsensusResult;
  /** Individual votes */
  votes: AgentVote[];
  /** Whether consensus reached */
  consensusReached: boolean;
  /** Effective consensus value */
  effectiveValue: number;
}

/**
 * Stage 3: Trust update
 */
export interface TrustUpdate {
  /** Consensus outcome */
  outcome: ConsensusOutcome;
  /** Agent ID */
  agentId: string;
  /** Previous K-Vector */
  previousKVector: KVector;
  /** Updated K-Vector */
  updatedKVector: KVector;
  /** Trust delta applied */
  trustDelta: TrustDelta;
  /** New trust score */
  newTrustScore: number;
}

/**
 * Stage 4: Trust attestation
 */
export interface TrustAttestation {
  /** Trust update */
  update: TrustUpdate;
  /** Proof ID */
  proofId: string;
  /** K-Vector commitment (hash) */
  kvectorCommitment: string;
  /** Provenance chain */
  provenanceChain?: ProvenanceChain;
  /** Attestation timestamp */
  attestedAt: number;
  /** Whether proof is verified */
  isVerified: boolean;
}

/**
 * Pipeline result
 */
export interface PipelineResult {
  /** Success status */
  success: boolean;
  /** Error message if failed */
  error?: string;
  /** Attestation if successful */
  attestation?: TrustAttestation;
  /** Translated attestation if domain specified */
  translatedAttestation?: TranslatedAttestation;
  /** Pipeline execution time (ms) */
  executionTimeMs: number;
}

// ============================================================================
// Cross-Domain Types (forward declaration)
// ============================================================================

/**
 * Translated attestation (defined in domains.ts)
 */
export interface TranslatedAttestation {
  /** Original attestation */
  original: TrustAttestation;
  /** Target domain ID */
  targetDomainId: string;
  /** Translated K-Vector */
  translatedKVector: KVector;
  /** Translation confidence */
  translationConfidence: number;
  /** Whether requirements met */
  meetsRequirements: boolean;
  /** Warnings */
  warnings: string[];
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * Trust pipeline error codes
 */
export enum TrustPipelineErrorCode {
  AgentNotFound = 'AGENT_NOT_FOUND',
  InvalidOutput = 'INVALID_OUTPUT',
  ConsensusFailure = 'CONSENSUS_FAILURE',
  ProvenanceError = 'PROVENANCE_ERROR',
  ProofError = 'PROOF_ERROR',
  DomainNotFound = 'DOMAIN_NOT_FOUND',
  TranslationError = 'TRANSLATION_ERROR',
  InsufficientVotes = 'INSUFFICIENT_VOTES',
  ValidationError = 'VALIDATION_ERROR',
}

/**
 * Trust pipeline error
 */
export class TrustPipelineError extends Error {
  constructor(
    public readonly code: TrustPipelineErrorCode,
    message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'TrustPipelineError';
  }
}
