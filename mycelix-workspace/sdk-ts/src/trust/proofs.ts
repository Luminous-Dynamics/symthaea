/**
 * @mycelix/sdk Trust Proofs
 *
 * Zero-knowledge proof interfaces for trust attestations.
 * These mirror the Rust ZK proof implementations in sdk/src/agentic/zk_trust.rs
 *
 * @packageDocumentation
 * @module trust/proofs
 */

import type { KVector, ProvenanceChain } from './types.js';

// ============================================================================
// Proof Statement Types
// ============================================================================

/**
 * Types of proof statements that can be proven
 */
export enum ProofStatementType {
  /** Trust exceeds a threshold */
  TrustExceedsThreshold = 'TrustExceedsThreshold',
  /** Trust is in a specific range */
  TrustInRange = 'TrustInRange',
  /** A specific dimension exceeds threshold */
  DimensionExceedsThreshold = 'DimensionExceedsThreshold',
  /** Multiple dimensions meet criteria */
  MultipleDimensionsExceed = 'MultipleDimensionsExceed',
  /** K-Vector matches a commitment */
  KVectorMatchesCommitment = 'KVectorMatchesCommitment',
  /** Provenance epistemic floor */
  ProvenanceEpistemicFloor = 'ProvenanceEpistemicFloor',
  /** Provenance chain depth */
  ProvenanceChainDepth = 'ProvenanceChainDepth',
  /** Provenance confidence */
  ProvenanceConfidence = 'ProvenanceConfidence',
  /** Provenance derivation types */
  ProvenanceDerivationTypes = 'ProvenanceDerivationTypes',
  /** Provenance has verified roots */
  ProvenanceHasVerifiedRoots = 'ProvenanceHasVerifiedRoots',
  /** Provenance chain is valid */
  ProvenanceChainValid = 'ProvenanceChainValid',
  /** Combined provenance quality */
  ProvenanceQuality = 'ProvenanceQuality',
}

/**
 * Base proof statement
 */
export interface BaseProofStatement {
  type: ProofStatementType;
}

/**
 * Statement: Trust exceeds threshold
 */
export interface TrustExceedsThresholdStatement extends BaseProofStatement {
  type: ProofStatementType.TrustExceedsThreshold;
  threshold: number;
}

/**
 * Statement: Trust in range
 */
export interface TrustInRangeStatement extends BaseProofStatement {
  type: ProofStatementType.TrustInRange;
  minTrust: number;
  maxTrust: number;
}

/**
 * Statement: Dimension exceeds threshold
 */
export interface DimensionExceedsThresholdStatement extends BaseProofStatement {
  type: ProofStatementType.DimensionExceedsThreshold;
  dimension: number;
  threshold: number;
}

/**
 * Statement: Multiple dimensions exceed threshold
 */
export interface MultipleDimensionsExceedStatement extends BaseProofStatement {
  type: ProofStatementType.MultipleDimensionsExceed;
  dimensions: number[];
  threshold: number;
}

/**
 * Statement: K-Vector matches commitment
 */
export interface KVectorMatchesCommitmentStatement extends BaseProofStatement {
  type: ProofStatementType.KVectorMatchesCommitment;
  commitmentHash: string;
}

/**
 * Statement: Provenance epistemic floor
 */
export interface ProvenanceEpistemicFloorStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceEpistemicFloor;
  minLevel: number;
}

/**
 * Statement: Provenance chain depth
 */
export interface ProvenanceChainDepthStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceChainDepth;
  maxDepth: number;
}

/**
 * Statement: Provenance confidence
 */
export interface ProvenanceConfidenceStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceConfidence;
  minConfidence: number;
}

/**
 * Statement: Provenance derivation types
 */
export interface ProvenanceDerivationTypesStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceDerivationTypes;
  allowedTypes: number[];
}

/**
 * Statement: Provenance has verified roots
 */
export interface ProvenanceHasVerifiedRootsStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceHasVerifiedRoots;
}

/**
 * Statement: Provenance chain valid
 */
export interface ProvenanceChainValidStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceChainValid;
}

/**
 * Statement: Provenance quality (combined)
 */
export interface ProvenanceQualityStatement extends BaseProofStatement {
  type: ProofStatementType.ProvenanceQuality;
  minEpistemicLevel: number;
  maxChainDepth: number;
  minConfidence: number;
}

/**
 * Union of all proof statements
 */
export type ProofStatement =
  | TrustExceedsThresholdStatement
  | TrustInRangeStatement
  | DimensionExceedsThresholdStatement
  | MultipleDimensionsExceedStatement
  | KVectorMatchesCommitmentStatement
  | ProvenanceEpistemicFloorStatement
  | ProvenanceChainDepthStatement
  | ProvenanceConfidenceStatement
  | ProvenanceDerivationTypesStatement
  | ProvenanceHasVerifiedRootsStatement
  | ProvenanceChainValidStatement
  | ProvenanceQualityStatement;

// ============================================================================
// Proof Statement Builders
// ============================================================================

/**
 * Build a trust exceeds threshold statement
 */
export function trustAbove(threshold: number): TrustExceedsThresholdStatement {
  return {
    type: ProofStatementType.TrustExceedsThreshold,
    threshold: Math.max(0, Math.min(1, threshold)),
  };
}

/**
 * Build a trust in range statement
 */
export function trustInRange(minTrust: number, maxTrust: number): TrustInRangeStatement {
  return {
    type: ProofStatementType.TrustInRange,
    minTrust: Math.max(0, Math.min(1, minTrust)),
    maxTrust: Math.max(0, Math.min(1, maxTrust)),
  };
}

/**
 * Build a dimension exceeds threshold statement
 */
export function dimensionAbove(dimension: number, threshold: number): DimensionExceedsThresholdStatement {
  return {
    type: ProofStatementType.DimensionExceedsThreshold,
    dimension: Math.max(0, Math.min(8, dimension)),
    threshold: Math.max(0, Math.min(1, threshold)),
  };
}

/**
 * Build a multiple dimensions exceed statement
 */
export function multipleDimensionsAbove(dimensions: number[], threshold: number): MultipleDimensionsExceedStatement {
  return {
    type: ProofStatementType.MultipleDimensionsExceed,
    dimensions: dimensions.filter(d => d >= 0 && d <= 8),
    threshold: Math.max(0, Math.min(1, threshold)),
  };
}

/**
 * Build a provenance quality statement
 */
export function provenanceQuality(
  minEpistemicLevel: number,
  maxChainDepth: number,
  minConfidence: number
): ProvenanceQualityStatement {
  return {
    type: ProofStatementType.ProvenanceQuality,
    minEpistemicLevel: Math.max(0, Math.min(4, minEpistemicLevel)),
    maxChainDepth: Math.max(1, maxChainDepth),
    minConfidence: Math.max(0, Math.min(1, minConfidence)),
  };
}

// ============================================================================
// K-Vector Commitment
// ============================================================================

/**
 * Commitment to a K-Vector (hides actual values)
 */
export interface KVectorCommitment {
  /** Commitment hash */
  hash: string;
  /** Agent ID */
  agentId: string;
  /** Timestamp */
  timestamp: number;
  /** Salt/blinding factor hash (not the actual blinding) */
  blindingHash: string;
}

// ============================================================================
// Trust Proof
// ============================================================================

/**
 * A zero-knowledge proof of trust properties
 */
export interface TrustProof {
  /** Unique proof ID */
  proofId: string;
  /** Agent ID */
  agentId: string;
  /** Statement being proven */
  statement: ProofStatement;
  /** K-Vector commitment */
  commitment: KVectorCommitment;
  /** Proof data (hex-encoded) */
  proofData: string;
  /** Whether this is a simulation (no actual ZK) */
  isSimulation: boolean;
  /** Timestamp */
  timestamp: number;
  /** Optional provenance chain */
  provenanceChain?: ProvenanceChain;
}

/**
 * Result of proof verification
 */
export interface VerificationResult {
  /** Whether proof is valid */
  isValid: boolean;
  /** Error message if invalid */
  error?: string;
  /** Verified statement */
  verifiedStatement?: ProofStatement;
  /** Verification timestamp */
  verifiedAt: number;
}

// ============================================================================
// Proof Evaluator (Client-Side)
// ============================================================================

/**
 * Evaluate if a K-Vector satisfies a proof statement
 * (Used for client-side validation before proof generation)
 */
export function evaluateStatement(kvector: KVector, statement: ProofStatement): boolean {
  const arr = [kvector.k_r, kvector.k_a, kvector.k_i, kvector.k_p, kvector.k_m, kvector.k_s, kvector.k_h, kvector.k_topo, kvector.k_v];

  switch (statement.type) {
    case ProofStatementType.TrustExceedsThreshold: {
      const trustScore = computeSimpleTrustScore(arr);
      return trustScore > statement.threshold;
    }

    case ProofStatementType.TrustInRange: {
      const trustScore = computeSimpleTrustScore(arr);
      return trustScore >= statement.minTrust && trustScore <= statement.maxTrust;
    }

    case ProofStatementType.DimensionExceedsThreshold: {
      return arr[statement.dimension] > statement.threshold;
    }

    case ProofStatementType.MultipleDimensionsExceed: {
      return statement.dimensions.every(dim => arr[dim] > statement.threshold);
    }

    case ProofStatementType.KVectorMatchesCommitment: {
      // Cannot verify without actual commitment - always return false
      return false;
    }

    default:
      return false;
  }
}

/**
 * Evaluate if a provenance chain satisfies a proof statement
 */
export function evaluateProvenanceStatement(
  chain: ProvenanceChain,
  statement: ProofStatement
): boolean {
  switch (statement.type) {
    case ProofStatementType.ProvenanceEpistemicFloor:
      return chain.minEpistemicLevel >= statement.minLevel;

    case ProofStatementType.ProvenanceChainDepth:
      return chain.depth <= statement.maxDepth;

    case ProofStatementType.ProvenanceConfidence:
      return chain.confidence >= statement.minConfidence;

    case ProofStatementType.ProvenanceHasVerifiedRoots:
      return chain.hasVerifiedRoots;

    case ProofStatementType.ProvenanceChainValid:
      // Check basic validity
      return chain.nodes.length > 0 && chain.depth >= 1;

    case ProofStatementType.ProvenanceQuality:
      return (
        chain.minEpistemicLevel >= statement.minEpistemicLevel &&
        chain.depth <= statement.maxChainDepth &&
        chain.confidence >= statement.minConfidence
      );

    default:
      return false;
  }
}

/**
 * Simple trust score computation (equal weights)
 */
function computeSimpleTrustScore(arr: number[]): number {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

// ============================================================================
// Proof Generator (Simulation Mode)
// ============================================================================

/**
 * Configuration for proof generation
 */
export interface ProverConfig {
  /** Agent ID */
  agentId: string;
  /** Timestamp */
  timestamp: number;
  /** Whether to use simulation mode */
  simulationMode: boolean;
}

/**
 * Generate a simulated proof (for testing/development)
 *
 * In production, this would call into a RISC-0 prover or similar ZK system.
 */
export function generateSimulatedProof(
  kvector: KVector,
  statement: ProofStatement,
  config: ProverConfig
): TrustProof {
  const arr = [kvector.k_r, kvector.k_a, kvector.k_i, kvector.k_p, kvector.k_m, kvector.k_s, kvector.k_h, kvector.k_topo, kvector.k_v];

  // Create commitment
  const commitment: KVectorCommitment = {
    hash: hashKVector(arr),
    agentId: config.agentId,
    timestamp: config.timestamp,
    blindingHash: generateRandomHex(32),
  };

  // Create simulated proof
  return {
    proofId: generateRandomHex(16),
    agentId: config.agentId,
    statement,
    commitment,
    proofData: generateSimulatedProofData(statement),
    isSimulation: true,
    timestamp: config.timestamp,
  };
}

/**
 * Verify a simulated proof
 */
export function verifySimulatedProof(proof: TrustProof): VerificationResult {
  // In simulation mode, just check the proof format
  if (!proof.isSimulation) {
    return {
      isValid: false,
      error: 'Real proofs require a proper ZK verifier',
      verifiedAt: Date.now(),
    };
  }

  // Validate basic structure
  if (!proof.proofId || !proof.agentId || !proof.statement || !proof.commitment) {
    return {
      isValid: false,
      error: 'Invalid proof structure',
      verifiedAt: Date.now(),
    };
  }

  return {
    isValid: true,
    verifiedStatement: proof.statement,
    verifiedAt: Date.now(),
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Hash a K-Vector to create a commitment
 */
function hashKVector(arr: number[]): string {
  // Simple hash for simulation - in production use SHA3-256
  let hash = 0;
  for (const val of arr) {
    hash = ((hash << 5) - hash) + Math.floor(val * 1000000);
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(16, '0');
}

/**
 * Generate random hex string
 */
function generateRandomHex(bytes: number): string {
  const array = new Array(bytes);
  for (let i = 0; i < bytes; i++) {
    array[i] = Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
  }
  return array.join('');
}

/**
 * Generate simulated proof data
 */
function generateSimulatedProofData(statement: ProofStatement): string {
  // Create a deterministic but fake proof
  const prefix = statement.type.slice(0, 8);
  return `SIM_PROOF_${prefix}_${generateRandomHex(32)}`;
}

// ============================================================================
// Proof Aggregation
// ============================================================================

/**
 * Aggregated proof combining multiple statements
 */
export interface AggregatedProof {
  /** Component proofs */
  proofs: TrustProof[];
  /** Combined validity */
  allValid: boolean;
  /** Aggregation timestamp */
  aggregatedAt: number;
}

/**
 * Aggregate multiple proofs
 */
export function aggregateProofs(proofs: TrustProof[]): AggregatedProof {
  const results = proofs.map(p => verifySimulatedProof(p));
  const allValid = results.every(r => r.isValid);

  return {
    proofs,
    allValid,
    aggregatedAt: Date.now(),
  };
}
