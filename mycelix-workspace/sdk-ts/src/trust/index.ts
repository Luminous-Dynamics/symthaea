// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Trust Module
 *
 * Comprehensive trust infrastructure for AI agents including:
 * - K-Vector (9-dimensional trust profiles)
 * - Trust Pipeline (provenance -> consensus -> trust update -> attestation)
 * - Cross-Domain Translation
 * - ZK Proof Interfaces
 *
 * @packageDocumentation
 * @module trust
 */

// Re-export types
export {
  // K-Vector
  KVectorDimension,
  KVECTOR_DIMENSION_NAMES,
  KVECTOR_WEIGHTS,
  type KVector,
  createUniformKVector,
  kvectorFromArray,
  kvectorToArray,
  computeTrustScore,
  isVerified,

  // Trust Delta
  TrustDirection,
  type TrustDelta,
  applyTrustDelta,

  // Epistemic Classification
  type EpistemicClassificationExtended,
  classificationCode,

  // Agent Output
  type OutputContent,
  type AgentOutput,

  // Provenance
  DerivationType,
  type ProvenanceNode,
  type ProvenanceChain,

  // Consensus
  PhiConsensusStatus,
  type AgentVote,
  type PhiConsensusResult,

  // Pipeline Types
  type PipelineConfig,
  DEFAULT_PIPELINE_CONFIG,
  type RegisteredOutput,
  type ConsensusOutcome,
  type TrustUpdate,
  type TrustAttestation,
  type PipelineResult,
  type TranslatedAttestation,

  // Errors
  TrustPipelineErrorCode,
  TrustPipelineError,
} from './types.js';

// Re-export domains
export {
  // Domain Relevance
  type DomainRelevance,
  balancedRelevance,
  relevanceToArray,
  relevanceSimilarity,

  // Trust Domain
  type TrustDomain,

  // Predefined Domains
  DOMAIN_FINANCIAL,
  DOMAIN_CODE_REVIEW,
  DOMAIN_SOCIAL,
  DOMAIN_GOVERNANCE,
  DOMAIN_RESEARCH,
  DOMAIN_INFRASTRUCTURE,
  ALL_DOMAINS,

  // Translation
  type DimensionTranslation,
  type TranslationResult,
  computeDomainTrust,
  translateTrust,
  translateAttestation,

  // Registry
  DomainRegistry,
  defaultDomainRegistry,

  // Compatibility
  type DomainCompatibility,
  analyzeDomainCompatibility,
} from './domains.js';

// Re-export proofs
export {
  // Statement Types
  ProofStatementType,
  type BaseProofStatement,
  type TrustExceedsThresholdStatement,
  type TrustInRangeStatement,
  type DimensionExceedsThresholdStatement,
  type MultipleDimensionsExceedStatement,
  type KVectorMatchesCommitmentStatement,
  type ProvenanceEpistemicFloorStatement,
  type ProvenanceChainDepthStatement,
  type ProvenanceConfidenceStatement,
  type ProvenanceDerivationTypesStatement,
  type ProvenanceHasVerifiedRootsStatement,
  type ProvenanceChainValidStatement,
  type ProvenanceQualityStatement,
  type ProofStatement,

  // Statement Builders
  trustAbove,
  trustInRange,
  dimensionAbove,
  multipleDimensionsAbove,
  provenanceQuality,

  // Commitment & Proof
  type KVectorCommitment,
  type TrustProof,
  type VerificationResult,

  // Evaluation
  evaluateStatement,
  evaluateProvenanceStatement,

  // Generation & Verification
  type ProverConfig,
  generateSimulatedProof,
  verifySimulatedProof,

  // Aggregation
  type AggregatedProof,
  aggregateProofs,
} from './proofs.js';

// Re-export pipeline
export {
  type RegisteredAgent,
  TrustPipeline,
  createTrustPipeline,
  createAgentOutput,
  createAgentVote,
} from './pipeline.js';
