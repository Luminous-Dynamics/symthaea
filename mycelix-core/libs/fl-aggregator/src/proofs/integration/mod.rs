//! Proof Integration Layer
//!
//! Integrates zkSTARK proofs with the FL aggregator, identity, and governance systems.
//!
//! ## Modules
//!
//! - [`aggregator`] - Verified gradient submission with proofs
//! - [`identity`] - Identity proof generation traits
//! - [`governance`] - Vote eligibility proof generation traits
//! - [`batch`] - Batch verification for multiple proofs
//! - [`aggregate`] - Aggregate proofs for multiple submissions
//! - [`parallel`] - Parallel proof generation
//! - [`serialization`] - Proof serialization for network transport
//! - [`cache`] - Proof caching layer
//! - [`composition`] - Proof composition (lightweight recursion alternative)
//! - [`kyber`] - Post-quantum key encapsulation (requires `proofs-kyber`)
//! - [`threshold`] - Threshold signatures with FROST (requires `proofs-threshold`)
//! - [`wasm`] - WebAssembly bindings (requires `proofs-wasm`)
//! - [`telemetry`] - OpenTelemetry metrics (requires `proofs-telemetry`)
//! - [`streaming`] - Async proof streaming (requires `proofs-streaming`)
//! - [`anchoring`] - Cross-chain proof anchoring (requires `proofs-anchoring`)
//! - [`hsm`] - Hardware Security Module integration (requires `proofs-hsm`)

pub mod aggregator;
pub mod identity;
pub mod governance;
pub mod batch;
pub mod aggregate;
pub mod parallel;
pub mod serialization;
pub mod cache;
pub mod composition;
pub mod zome_bridge;

// Feature-gated modules
#[cfg(feature = "proofs-kyber")]
pub mod kyber;

#[cfg(feature = "proofs-threshold")]
pub mod threshold;

#[cfg(feature = "proofs-wasm")]
pub mod wasm;

#[cfg(feature = "proofs-telemetry")]
pub mod telemetry;

// Streaming has core sync functions always available
pub mod streaming;

#[cfg(feature = "proofs-anchoring")]
pub mod anchoring;

#[cfg(feature = "proofs-hsm")]
pub mod hsm;

// Core exports
pub use aggregator::{
    VerifiedGradientSubmission, GradientProofBundle, ProofVerifyingSubmitter,
};
pub use identity::{IdentityProofGenerator, IdentityProofBundle};
pub use governance::{EligibilityProofGenerator, EligibilityProofBundle};
pub use batch::{
    BatchVerifier, BatchVerificationResult, ProofVerificationEntry,
    AnyProof, BatchTypeStats,
};
pub use aggregate::{
    GradientAggregateProof, AggregateBuilder, AggregateSubmissionMeta, AggregateStats,
};
pub use parallel::{
    ParallelProofGenerator, ParallelGenerationResult, ParallelProofResult, ProofTask,
};
pub use serialization::{
    ProofEnvelope, ProofBatch, SERIALIZATION_VERSION, MAGIC_BYTES,
};
pub use cache::{
    ProofCache, CacheConfig, CacheStats, CacheKeyBuilder,
};
pub use composition::{
    ComposedProof, CompositionBuilder, CompositionAttestation,
    CompositionMetadata, CompositionVerificationResult,
};
pub use zome_bridge::{
    ZomeBridgeClient, VoterProfileBuilder, SerializableProof,
    EligibilityCheck, ProofAttestation, VerifierService, VerifierConfig,
    VerificationResult, StoreProofInput, CastVerifiedVoteInput,
};

// Post-quantum attestation types (requires proofs-pq feature)
#[cfg(feature = "proofs-pq")]
pub use composition::{PqAttestation, PqAlgorithm};

// Kyber KEM exports (requires proofs-kyber feature)
#[cfg(feature = "proofs-kyber")]
pub use kyber::{
    KyberKeyPair, EncryptedProofEnvelope,
    encrypt_envelope, decrypt_envelope, kyber1024_info,
};

// Threshold signature exports (requires proofs-threshold feature)
#[cfg(feature = "proofs-threshold")]
pub use threshold::{
    ThresholdConfig, KeyShare, GroupPublicKey, ThresholdSignature,
    ThresholdAttestation, generate_shares, verify_threshold_signature,
    SigningCoordinator, ThresholdParticipant,
};

// WASM exports (requires proofs-wasm feature)
#[cfg(feature = "proofs-wasm")]
pub use wasm::{
    WasmVerificationResult, WasmEnvelopeInfo, WasmBatchResult, WasmLibraryInfo,
    WasmProofVerifier,
    verify_range_proof, verify_gradient_proof, verify_identity_proof,
    verify_vote_proof, verify_membership_proof,
    parse_proof_envelope, verify_proof_envelope, verify_batch, verify_recursive_batch,
    supported_proof_types, supported_security_levels, library_version, library_info,
    blake3_hash, generate_commitment,
};

// Telemetry exports (requires proofs-telemetry feature)
#[cfg(feature = "proofs-telemetry")]
pub use telemetry::{ProofTelemetry, TelemetryStats, init_telemetry, shutdown_telemetry};

// Streaming exports (requires proofs-streaming feature)
#[cfg(feature = "proofs-streaming")]
pub use streaming::{
    StreamVerificationResult, StreamStats,
    stream_verify_batch, stream_verify_chunked, stream_verify_with_progress,
    StreamingProofGenerator,
};

// Always export sync verification
pub use streaming::{verify_batch_sync};

// Anchoring exports (requires proofs-anchoring feature)
#[cfg(feature = "proofs-anchoring")]
pub use anchoring::{
    Chain, AnchorConfig, AnchorReceipt, AnchorMetadata, AnchorStatus,
    Anchor, BitcoinAnchor, EthereumAnchor,
    anchor_composition, verify_composition_anchor, create_anchor_merkle_root,
};

// HSM exports (requires proofs-hsm feature)
#[cfg(feature = "proofs-hsm")]
pub use hsm::{
    HsmProvider, HsmConfig, HsmProviderType, HsmAuth,
    KeyType, KeyMetadata, HsmSignature, HsmResult,
    VaultHsm, SoftHsm, create_provider,
};
