//! zkSTARK Proof System
//!
//! Provides zero-knowledge proof generation and verification for federated learning
//! using the Winterfell STARK library.
//!
//! ## Overview
//!
//! This module implements 5 proof circuits for verifiable federated learning:
//!
//! | Circuit | Purpose | Use Case |
//! |---------|---------|----------|
//! | [`RangeProof`] | Prove value within bounds | Parameter validation |
//! | [`MembershipProof`] | Prove Merkle tree membership | Participant verification |
//! | [`GradientIntegrityProof`] | Prove valid gradient | FL contribution validation |
//! | [`IdentityAssuranceProof`] | Prove assurance level | Identity verification |
//! | [`VoteEligibilityProof`] | Prove voter qualification | Governance voting |
//!
//! ## Security Levels
//!
//! Three security levels are supported:
//!
//! | Level | Bits | Queries | Use Case |
//! |-------|------|---------|----------|
//! | `Standard96` | 96 | 27 | Fast proofs, lower security |
//! | `Standard128` | 128 | 36 | Default, balanced |
//! | `High256` | 256 | 72 | Maximum security |
//!
//! ## Important Note
//!
//! Winterfell does NOT provide perfect zero-knowledge. Proofs may leak some
//! information about secret inputs. This is acceptable for verification use cases
//! where we're proving validity/integrity rather than hiding participation.
//!
//! ## Quick Start
//!
//! ### Range Proof
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{RangeProof, ProofConfig};
//!
//! // Prove value 42 is in [0, 100]
//! let proof = RangeProof::generate(42, 0, 100, ProofConfig::default())?;
//! assert!(proof.verify()?.valid);
//! ```
//!
//! ### Gradient Integrity Proof
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{GradientIntegrityProof, ProofConfig};
//!
//! let gradients = vec![0.1, 0.2, -0.1, 0.05];
//! let max_norm = 1.0;
//!
//! let proof = GradientIntegrityProof::generate(&gradients, max_norm, ProofConfig::default())?;
//! assert!(proof.verify()?.valid);
//! ```
//!
//! ### Identity Assurance Proof
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     IdentityAssuranceProof, ProofAssuranceLevel, ProofIdentityFactor, ProofConfig
//! };
//!
//! let factors = vec![
//!     ProofIdentityFactor::new(0.5, 0, true),  // CryptoKey factor
//!     ProofIdentityFactor::new(0.3, 1, true),  // Social factor
//! ];
//!
//! let proof = IdentityAssuranceProof::generate(
//!     "did:mycelix:alice",
//!     &factors,
//!     ProofAssuranceLevel::E2,
//!     ProofConfig::default(),
//! )?;
//! assert!(proof.meets_threshold());
//! ```
//!
//! ### Vote Eligibility Proof
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     VoteEligibilityProof, ProofProposalType, ProofVoterProfile, ProofConfig
//! };
//!
//! let voter = ProofVoterProfile {
//!     did: "did:mycelix:voter".to_string(),
//!     assurance_level: 2,
//!     matl_score: 0.7,
//!     stake: 500.0,
//!     account_age_days: 90,
//!     participation_rate: 0.4,
//!     has_humanity_proof: true,
//!     fl_contributions: 15,
//! };
//!
//! let proof = VoteEligibilityProof::generate(
//!     &voter,
//!     ProofProposalType::Constitutional,
//!     ProofConfig::default(),
//! )?;
//! assert!(proof.is_eligible());
//! ```
//!
//! ## Integration Layer
//!
//! The [`integration`] module provides higher-level abstractions:
//!
//! - [`VerifiedGradientSubmission`] - Gradient with attached proof
//! - [`IdentityProofGenerator`] - Trait for identity proof generation
//! - [`EligibilityProofGenerator`] - Trait for vote eligibility proofs
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{ProofableVoter, EligibilityProofGenerator};
//!
//! let voter = ProofableVoter::new("did:mycelix:alice".to_string())
//!     .with_assurance_level(2)
//!     .with_matl_score(0.7)
//!     .with_stake(500.0)
//!     .with_humanity_proof(true);
//!
//! let bundle = voter.generate_eligibility_proof(
//!     ProofProposalType::Standard,
//!     ProofConfig::default(),
//! )?;
//! assert!(bundle.is_eligible());
//! ```
//!
//! ## Performance Characteristics
//!
//! Typical performance on modern hardware (Standard128 security):
//!
//! | Operation | Time | Proof Size |
//! |-----------|------|------------|
//! | RangeProof generation | ~50ms | ~15KB |
//! | MembershipProof generation | ~80ms | ~20KB |
//! | GradientIntegrityProof (1K elements) | ~200ms | ~25KB |
//! | IdentityAssuranceProof | ~60ms | ~18KB |
//! | VoteEligibilityProof | ~50ms | ~16KB |
//! | All proof verification | <50ms | - |
//!
//! ---
//!
//! # Integration Guide
//!
//! ## FL Aggregator Integration
//!
//! ### Verified Gradient Submissions
//!
//! Use [`VerifiedGradientSubmission`] to attach proofs to FL gradient contributions:
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     VerifiedGradientSubmission, GradientProofBundle, ProofConfig,
//! };
//! use fl_aggregator::Gradient;
//! use ndarray::Array1;
//!
//! // 1. Create gradient data
//! let gradient_data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
//! let max_norm = 5.0;
//!
//! // 2. Generate proof bundle
//! let bundle = GradientProofBundle::generate(
//!     &gradient_data,
//!     max_norm,
//!     ProofConfig::default(),
//! )?;
//!
//! // 3. Wrap in verified submission
//! let gradient: Gradient = Array1::from_vec(gradient_data);
//! let mut submission = VerifiedGradientSubmission::with_proofs(
//!     "node_id".to_string(),
//!     gradient,
//!     bundle,
//! );
//!
//! // 4. Verify before aggregation
//! let result = submission.verify()?;
//! if result.valid && submission.is_verified() {
//!     // Safe to include in aggregation
//! }
//! ```
//!
//! ### Server-Side Verification
//!
//! Implement [`ProofVerifyingSubmitter`] for custom submission handling:
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     ProofVerifyingSubmitter, VerifiedGradientSubmission, ProofResult,
//! };
//!
//! struct MyAggregator {
//!     verified_submissions: Vec<VerifiedGradientSubmission>,
//! }
//!
//! impl ProofVerifyingSubmitter for MyAggregator {
//!     fn submit_with_verification(
//!         &mut self,
//!         mut submission: VerifiedGradientSubmission,
//!     ) -> ProofResult<bool> {
//!         // Verify proofs
//!         let result = submission.verify()?;
//!         if result.valid {
//!             self.verified_submissions.push(submission);
//!             Ok(true)
//!         } else {
//!             Ok(false)
//!         }
//!     }
//! }
//! ```
//!
//! ## Identity System Integration
//!
//! ### Custom Identity Types
//!
//! Implement [`IdentityProofGenerator`] for your identity type:
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     IdentityProofGenerator, IdentityProofBundle, ProofIdentityFactor,
//!     ProofAssuranceLevel, ProofConfig, ProofResult,
//! };
//!
//! struct MyIdentity {
//!     did: String,
//!     crypto_key_verified: bool,
//!     social_accounts: Vec<String>,
//!     biometric_hash: Option<[u8; 32]>,
//! }
//!
//! impl IdentityProofGenerator for MyIdentity {
//!     fn did(&self) -> &str {
//!         &self.did
//!     }
//!
//!     fn get_proof_factors(&self) -> Vec<ProofIdentityFactor> {
//!         let mut factors = Vec::new();
//!
//!         // Category 0: Cryptographic key
//!         if self.crypto_key_verified {
//!             factors.push(ProofIdentityFactor::new(0.4, 0, true));
//!         }
//!
//!         // Category 1: Social proof (0.1 per account, max 0.3)
//!         let social_contrib = (self.social_accounts.len() as f32 * 0.1).min(0.3);
//!         if social_contrib > 0.0 {
//!             factors.push(ProofIdentityFactor::new(social_contrib, 1, true));
//!         }
//!
//!         // Category 2: Biometric
//!         if self.biometric_hash.is_some() {
//!             factors.push(ProofIdentityFactor::new(0.25, 2, true));
//!         }
//!
//!         factors
//!     }
//!
//!     fn generate_assurance_proof(
//!         &self,
//!         min_level: ProofAssuranceLevel,
//!         config: ProofConfig,
//!     ) -> ProofResult<IdentityProofBundle> {
//!         let factors = self.get_proof_factors();
//!         IdentityProofBundle::generate(&self.did, &factors, min_level, config)
//!     }
//! }
//! ```
//!
//! ### Assurance Level Thresholds
//!
//! | Level | Threshold | Typical Requirements |
//! |-------|-----------|---------------------|
//! | E0 | 0 | Any factor |
//! | E1 | 300 | 1 verified factor |
//! | E2 | 500 | 2+ factors, 1+ category |
//! | E3 | 700 | 3+ factors, 2+ categories |
//! | E4 | 900 | 4+ factors, 3+ categories, biometric |
//!
//! Score calculation: `(sum of contributions) + (categories * 0.05)` scaled to 1000.
//!
//! ## Governance Integration
//!
//! ### Vote Eligibility
//!
//! Use [`EligibilityProofGenerator`] for voter qualification:
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{
//!     EligibilityProofGenerator, ProofVoterProfile, ProofProposalType,
//!     ProofConfig, EligibilityProofBundle,
//! };
//!
//! // Build voter profile from your user data
//! let voter = ProofVoterProfile {
//!     did: user.did.clone(),
//!     assurance_level: user.assurance_level as u8,
//!     matl_score: user.reputation_score,
//!     stake: user.staked_tokens,
//!     account_age_days: user.days_since_registration(),
//!     participation_rate: user.governance_participation_rate(),
//!     has_humanity_proof: user.humanity_verified,
//!     fl_contributions: user.total_fl_contributions,
//! };
//!
//! // Generate eligibility proof
//! let bundle = EligibilityProofBundle::generate(
//!     &voter,
//!     ProofProposalType::Constitutional,
//!     ProofConfig::default(),
//! )?;
//!
//! if bundle.is_eligible() {
//!     // User can vote on this proposal type
//!     let proof_bytes = bundle.serialize()?;
//!     submit_vote_with_proof(proposal_id, vote_choice, proof_bytes);
//! }
//! ```
//!
//! ### Proposal Type Requirements
//!
//! | Proposal Type | Key Requirements |
//! |---------------|------------------|
//! | Standard | Basic account (minimal) |
//! | Constitutional | Humanity proof, stake >= 500, assurance >= 2 |
//! | ModelGovernance | FL contributions >= 10, MATL >= 0.5 |
//! | Emergency | High assurance (3+), stake >= 1000 |
//! | Treasury | Stake >= 500, participation >= 0.3 |
//! | Membership | Account age >= 30 days |
//!
//! ## Error Handling
//!
//! All proof operations return [`ProofResult<T>`]:
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{ProofError, RangeProof, ProofConfig};
//!
//! match RangeProof::generate(value, min, max, ProofConfig::default()) {
//!     Ok(proof) => {
//!         match proof.verify() {
//!             Ok(result) if result.valid => println!("Valid proof"),
//!             Ok(result) => println!("Invalid: {:?}", result.details),
//!             Err(ProofError::VerificationFailed(msg)) => {
//!                 eprintln!("Verification error: {}", msg);
//!             }
//!             Err(e) => eprintln!("Error: {:?}", e),
//!         }
//!     }
//!     Err(ProofError::InvalidInput(msg)) => {
//!         eprintln!("Bad input: {}", msg);
//!     }
//!     Err(ProofError::ConstraintViolation(msg)) => {
//!         eprintln!("Constraint violated: {}", msg);
//!     }
//!     Err(e) => eprintln!("Generation failed: {:?}", e),
//! }
//! ```
//!
//! ## Configuration
//!
//! ### ProofConfig Options
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{ProofConfig, SecurityLevel};
//!
//! // Fast proofs for testing
//! let fast_config = ProofConfig {
//!     security_level: SecurityLevel::Standard96,
//!     parallel: false,
//!     max_proof_size: 0, // No limit
//! };
//!
//! // Production config
//! let prod_config = ProofConfig {
//!     security_level: SecurityLevel::Standard128,
//!     parallel: true,  // Use multiple cores
//!     max_proof_size: 50_000, // 50KB limit
//! };
//!
//! // High security for critical operations
//! let high_security = ProofConfig {
//!     security_level: SecurityLevel::High256,
//!     parallel: true,
//!     max_proof_size: 0,
//! };
//! ```
//!
//! ## CLI Tool
//!
//! The `proof-cli` binary provides command-line proof operations:
//!
//! ```bash
//! # Generate range proof
//! proof-cli range --value 42 --min 0 --max 100 --security 128
//!
//! # Generate gradient proof
//! proof-cli gradient --values "0.1,-0.2,0.3,-0.4" --max-norm 5.0
//!
//! # Generate identity proof
//! proof-cli identity --did "did:mycelix:alice" --level E2 \
//!     --factors "0.5:0:true,0.3:1:true"
//!
//! # Generate vote eligibility proof
//! proof-cli vote --did "did:mycelix:voter" --proposal constitutional \
//!     --assurance 3 --matl 0.8 --stake 750 --humanity
//!
//! # Run benchmarks
//! proof-cli bench --iterations 10
//! ```
//!
//! ## Testing
//!
//! ```bash
//! # Run all proof tests
//! cargo test --features proofs
//!
//! # Run end-to-end tests
//! cargo test --features proofs --test e2e_proofs
//!
//! # Run benchmarks
//! cargo bench --features proofs --bench proof_bench
//! ```

pub mod error;
pub mod types;
pub mod field;
pub mod circuits;
pub mod integration;
pub mod gpu;
pub mod timestamped;
pub mod recursive;
#[cfg(feature = "prometheus")]
pub mod metrics;
pub mod simulation;
pub mod gradient_circuit;

#[cfg(feature = "http-api")]
pub mod api;

#[cfg(feature = "proofs-grpc")]
pub mod grpc;

pub mod distributed;
pub mod realtime;
pub mod expiry;
pub mod distributed_cache;
pub mod benchmark;
pub mod audit;
pub mod optimizer;
pub mod ratelimit;
pub mod visualization;
pub mod hsm;
pub mod mpc;

#[cfg(test)]
mod proptest_support;

pub use error::{ProofError, ProofResult};
pub use types::{
    Commitment, ProofConfig, ProofStats, ProofType, SecurityLevel, VerificationResult,
};
pub use field::{
    bool_to_field, decompose_to_bits, f32_to_field, field_to_f32, field_to_u64,
    hash_to_field_elements, recompose_from_bits, u32_to_field, u64_to_field,
    ToFieldElements, F32_SCALE,
};
pub use circuits::{
    RangeProof, RangeProofAir, RangePublicInputs,
    MembershipProof, MembershipProofAir, MembershipPublicInputs,
    MerklePathNode, build_merkle_tree, compute_leaf_hash,
    GradientIntegrityProof, GradientIntegrityAir, GradientPublicInputs,
    compute_gradient_commitment,
    IdentityAssuranceProof, IdentityAssuranceAir, IdentityPublicInputs,
    ProofAssuranceLevel, ProofIdentityFactor, compute_identity_commitment,
    VoteEligibilityProof, VoteEligibilityAir, VotePublicInputs,
    ProofProposalType, ProofEligibilityRequirements, ProofVoterProfile,
    compute_voter_commitment,
};
pub use integration::{
    VerifiedGradientSubmission, GradientProofBundle, ProofVerifyingSubmitter,
    IdentityProofGenerator, IdentityProofBundle,
    EligibilityProofGenerator, EligibilityProofBundle,
    BatchVerifier, BatchVerificationResult, ProofVerificationEntry,
    AnyProof, BatchTypeStats,
    GradientAggregateProof, AggregateBuilder, AggregateSubmissionMeta, AggregateStats,
    ParallelProofGenerator, ParallelGenerationResult, ParallelProofResult, ProofTask,
    ProofEnvelope, ProofBatch, SERIALIZATION_VERSION, MAGIC_BYTES,
    ProofCache, CacheConfig, CacheStats, CacheKeyBuilder,
    ComposedProof, CompositionBuilder, CompositionAttestation,
    CompositionMetadata, CompositionVerificationResult,
};
pub use gpu::{
    GpuBackend, GpuConfig, GpuDeviceInfo, GpuStatus, GpuOperation, GpuBenchmark,
    check_gpu_availability, list_devices, run_benchmarks as gpu_benchmarks,
    estimate_gpu_memory,
};
pub use timestamped::{
    TimestampConfig, TimestampedProof, TimestampableProof, ProofTimestamp,
    TimestampedVerificationResult, NonceRegistry,
    DEFAULT_VALIDITY_SECS, MAX_CLOCK_SKEW_SECS,
};
pub use recursive::{
    ProofBatch as RecursiveBatch, RecursiveProof, ProofCommitment, ProofEntry,
    BatchVerificationResult as RecursiveBatchResult, RecursiveVerificationResult,
    MAX_BATCH_SIZE as MAX_RECURSIVE_BATCH_SIZE,
};
#[cfg(feature = "prometheus")]
pub use metrics::{
    ProofMetrics, MetricsSnapshot, GenerationTimer, VerificationTimer, METRICS,
};
pub use simulation::{
    ProofSimulator, SimulationMode, SimulationResult, CostEstimates,
    ProofRequest, BatchSimulationResult, SimulationStats,
    MockProofVerifier, MockVerificationResult,
};
pub use gradient_circuit::{
    ProductionGradientAir, ProductionGradientProver, ProductionGradientInputs,
    SerializedProductionProof, compute_production_commitment,
    generate_production_proof, verify_production_proof,
    MIN_TRACE_LEN as PRODUCTION_MIN_TRACE_LEN,
    PRODUCTION_TRACE_WIDTH, PRODUCTION_SCALE, MAX_GRADIENT_ELEMENT,
};

#[cfg(feature = "http-api")]
pub use api::{
    ProofApiRouter, ApiConfig, ApiState,
    RangeProofRequest, GradientProofRequest, IdentityProofRequest, VoteProofRequest,
    VerifyRequest, BatchVerifyRequest, SimulationRequest,
    ProofResponse, VerifyResponse, BatchVerifyResponse, SimulationResponse, HealthResponse,
    // PoGQ types (production-ready gradient proofs)
    ProofOfGradientQuality, GradientStatistics, GradientQualityVerificationResult,
    ProofMetadata, BatchPoGQVerifier,
};

pub use distributed::{
    // Note: ProofCoordinator, CoordinatorConfig, CoordinatorStats are hidden
    // (incomplete implementation - local processing only)
    ProofWorker, WorkerConfig, WorkerCapabilities, WorkerState, WorkerStatus, WorkerStats,
    DistributedProofRequest, DistributedProofData, DistributedProofResult,
    WorkChunk, ChunkData, ChunkResult,
};
pub use realtime::{
    ProofEventBus, EventBusConfig, ProofEvent, Subscriber, EventFilter,
    ProgressTracker, BatchProgressTracker,
};
pub use expiry::{
    ExpiringProof, RevocationRegistry, RevocationConfig, RevocationEntry, RevocationReason,
    RevocationStats, ValidityChecker, ValidityStatus,
};
pub use distributed_cache::{
    DistributedCache, DistributedCacheConfig,
    CacheStats as DistCacheStats, CacheKey,
};
pub use benchmark::{
    BenchmarkConfig, BenchmarkRunner, BenchmarkResults, ProofBenchmarkResult,
    TimingStats, MemoryStats, SystemInfo,
    ThroughputConfig, ThroughputResult, run_throughput_test,
    compare_proof_types, compare_security_levels,
};
// PoGQ benchmarks require http-api feature (for ProofOfGradientQuality)
#[cfg(feature = "http-api")]
pub use benchmark::{
    PoGQBenchmarkConfig, PoGQBenchmarkResult,
    run_pogq_benchmarks, format_pogq_results, quick_pogq_benchmark,
};
pub use audit::{
    AuditLog, AuditConfig, AuditEvent, AuditEventType, AuditStats,
    AuditFilter, AuditLogger, ChainVerificationResult,
};
pub use optimizer::{
    CircuitOptimizer, OptimizerConfig, OptimizedConfig, OptimizationHints,
    CircuitAnalysis, OptimizationOpportunity, OptimizationType,
    AutoTuner, Workload, WorkloadRecommendation, Recommendation, Impact,
    analyze_circuit,
};
pub use ratelimit::{
    RateLimiter, RateLimitConfig, RateLimitResult, RateLimitStats,
    Quota, TokenBucket, SlidingWindow, RemainingQuota,
    CostParams, estimate_cost,
};
pub use visualization::{
    VisualizationConfig, ColorScheme, CircuitInfo, ProofSizeBreakdown,
    visualize_circuit, visualize_proof_size, visualize_verification_flow,
    ascii_circuit,
};
pub use hsm::{
    Hsm, SoftwareHsm, HsmConfig, HsmProvider, HsmCredentials, HsmError, HsmResult,
    KeyHandle, KeyMetadata, KeyPurpose, KeyAlgorithm,
    Signature as HsmSignature, KeyAttestation, HsmStats,
    ProofSigner, SignedProof,
};
pub use mpc::{
    MpcSession, MpcConfig, MpcCoordinator, MpcStats, MpcError,
    MpcPhase, MpcMessage, MpcMessageType, MpcProofResult,
    PartyId, PartyInfo, PartyStatus,
    Share, PartialResult, SharingScheme, SessionState,
};

#[cfg(feature = "proofs-grpc")]
pub use grpc::{
    ProofServer, ProofService, ProofServiceImpl, GrpcServiceConfig, ServiceStats,
    RangeRequest as GrpcRangeRequest, GradientRequest as GrpcGradientRequest,
    IdentityRequest as GrpcIdentityRequest, VoteRequest as GrpcVoteRequest,
    VerifyRequest as GrpcVerifyRequest, GenerateRequest as GrpcGenerateRequest,
    ProofResponse as GrpcProofResponse, VerifyResponse as GrpcVerifyResponse,
    HealthResponse as GrpcHealthResponse, ProgressUpdate as GrpcProgressUpdate,
    GrpcSecurityLevel, GrpcProofType, GrpcProposalType,
    GrpcProofConfig, GrpcIdentityFactor, GrpcVoterProfile,
    ClientRequestBuilder,
};

use winterfell::{FieldExtension, ProofOptions};

/// Build standard proof options from security level
pub fn build_proof_options(security: SecurityLevel) -> ProofOptions {
    ProofOptions::new(
        security.num_queries(),
        security.blowup_factor(),
        security.grinding_factor(),
        FieldExtension::None,
        8,   // FRI folding factor
        31,  // FRI max remainder polynomial degree
    )
}

/// Trait for types that can generate proofs
pub trait ProofGenerator {
    /// The proof type produced
    type Proof: Clone + Send + Sync;

    /// The public inputs type
    type PublicInputs: Clone + Send + Sync;

    /// Generate a proof
    fn generate_proof(&self, config: &ProofConfig) -> ProofResult<Self::Proof>;

    /// Get the public inputs
    fn public_inputs(&self) -> Self::PublicInputs;
}

/// Trait for types that can verify proofs
pub trait ProofVerifier {
    /// The proof type to verify
    type Proof;

    /// The public inputs type
    type PublicInputs;

    /// Verify a proof against public inputs
    fn verify_proof(
        proof: &Self::Proof,
        public_inputs: &Self::PublicInputs,
    ) -> ProofResult<VerificationResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_proof_options() {
        let options = build_proof_options(SecurityLevel::Standard128);
        // Just verify it doesn't panic
        assert!(options.num_queries() > 0);
    }

    #[test]
    fn test_security_level_default() {
        let level = SecurityLevel::default();
        assert_eq!(level, SecurityLevel::Standard128);
    }
}
