// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix SDK
//!
//! Shared library for the Mycelix ecosystem providing:
//! - **MATL**: Mycelix Adaptive Trust Layer (45% Byzantine tolerance)
//!   - K-Vector: 8-dimensional trust scoring (k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo)
//!   - Governance Tiers: Observer, Basic (Φ≥0.3), Major (Φ≥0.4), Constitutional (Φ≥0.6)
//!   - RB-BFT: Reputation-Based Byzantine Fault Tolerance with reputation² voting
//! - **DKG**: Distributed Knowledge Graph with confidence scoring (Truth Engine)
//!   - 5-factor confidence: attestation, reputation², source quality, time decay, consistency
//!   - Genesis Simulation: Social consensus defeats misinformation
//!   - Epistemic classification: E/N/M (Empirical, Normative, Metaphysical)
//! - **FL**: Federated Learning with Byzantine-resistant aggregation
//! - **HyperFeel**: 2000x gradient compression using hyperdimensional computing
//! - **zkProof**: zkSTARK gradient provenance verification
//! - **Epistemic**: Epistemic Charter v2.0 with GIS v4.0 integration
//!   - 3D Classification: E/N/M (Empirical, Normative, Materiality)
//!   - 4D Extended: E/N/M/H (adds Harmonic axis for consciousness metrics)
//! - **Credentials**: W3C Verifiable Credentials
//! - **Bridge**: Inter-hApp communication protocol
//! - **Identity**: WebAuthn/FIDO2 hardware key authentication
//! - **Error**: Error sanitization to prevent PII leakage in production
//! - **Economics**: Metabolic Bridge economic primitives (MIP-E-002)
//! - **PoG**: Proof of Grounding for physical infrastructure (MIP-E-003)
//! - **Agentic**: Instrumental Actor framework for AI agents (MIP-E-004)
//! - **Temporal**: Temporal Economics with patient capital (MIP-E-005)
//! - **Intentions**: Civilizational Intention Framework (MIP-E-007)
//!
//! ## Features
//!
//! - `holochain` (default): Enable Holochain zome integration
//! - `standalone`: Pure Rust library without Holochain dependencies
//!
//! ## Example
//!
//! ```rust
//! use mycelix_sdk::matl::ProofOfGradientQuality;
//! use mycelix_sdk::epistemic::{EpistemicClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel};
//!
//! // Create a trust measurement
//! let pogq = ProofOfGradientQuality::new(0.95, 0.88, 0.12);
//! let score = pogq.composite_score(0.75);
//!
//! // Classify a claim
//! let claim = EpistemicClaim::new(
//!     "User completed verification",
//!     EmpiricalLevel::E3Cryptographic,
//!     NormativeLevel::N2Network,
//!     MaterialityLevel::M2Persistent,
//! );
//! ```

#![warn(missing_docs)]
#![deny(clippy::unwrap_used)]

pub mod matl;
pub mod dkg;
pub mod fl;
pub mod crypto;

// TypeScript binding generation (ts-export feature only)
#[cfg(feature = "ts-export")]
mod export_bindings;
pub mod hyperfeel;
pub mod zkproof;
pub mod epistemic;
pub mod credentials;
pub mod bridge;
pub mod economics;
pub mod pog;
pub mod agentic;
pub mod temporal;
pub mod intentions;
pub mod pagination;
pub mod storage;
pub mod identity;
pub mod error;

// WASM bindings (wasm feature only)
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience
pub use matl::{ProofOfGradientQuality, ReputationScore, CompositeScore, AdaptiveThreshold};
pub use matl::{KVector, KVectorWeights, KVectorDimension, GovernanceTier, KVECTOR_WEIGHTS};
pub use matl::{RbBftConsensus, RbBftConfig, RoundState, ConsensusResult, Vote, VoteType};
pub use dkg::{
    VerifiableTriple, TripleValue, EpistemicType as DkgEpistemicType, URI, StoredTriple, DKGConfig,
    calculate_confidence, ConfidenceScore, ConfidenceFactors, ConfidenceThresholds, ConfidenceInput, meets_threshold,
    Attestation, AttestationType, AttestationSet, AttestationCounts,
    QueryFilter, ObjectFilter, query_by_subject, query_by_predicate, query_triples,
};
pub use fl::{FLCoordinator, FLConfig, AggregationMethod, GradientUpdate, AggregatedGradient};
pub use hyperfeel::{HyperFeelEncoder, EncodingConfig, HyperGradient};
pub use zkproof::{GradientProofCircuit, GradientProof, PublicInputs};
pub use epistemic::{EpistemicClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel};
pub use epistemic::EpistemicClassificationExtended;
pub use credentials::{VerifiableCredential, CredentialBuilder};
pub use bridge::{BridgeMessage, CrossHappReputation};
pub use pagination::{PaginationRequest, PaginatedResponse, paginate_vec};
pub use storage::{EpistemicStorage, StorageConfig, StorageRouter, StorageReceipt, StorageError};
pub use identity::{
    WebAuthnCredential, WebAuthnService, WebAuthnError,
    HardwareKeyBridge, HardwareKeyBridgeConfig, BridgeError as HardwareKeyBridgeError,
};
pub use error::{SanitizedError, SanitizedResult, SanitizeExt, sanitize_message, contains_sensitive_data};

// Agentic Economy exports (MIP-E-004: Epistemic-Aware AI Agency)
pub use agentic::{
    // Core types
    AgentId, InstrumentalActor, AgentClass, AgentStatus, AgentConstraints,
    BehaviorLogEntry, ActionOutcome,
    // K-Vector integration
    KVectorBridgeConfig, compute_trust_score, calculate_kredit_from_trust,
    // GIS integration (Graceful Ignorance System)
    MoralUncertainty, MoralUncertaintyType, MoralActionGuidance,
    EscalationRequest, UncertaintyCalibration,
    // Epistemic classification
    AgentOutput, EpistemicStats,
    // Coherence (Phi) measurement
    CoherenceState, CoherenceCheckResult, PhiMeasurementConfig,
    // API service
    AgentApiService, ApiError, ApiResult,
    CreateAgentRequest, UpdateAgentRequest,
    // Gaming detection
    GamingDetector, GamingDetectionConfig, GamingDetectionResult, GamingResponse,
    // Quarantine
    QuarantineManager, QuarantineReason,
};

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the SDK (placeholder for future initialization logic)
pub fn init() {
    // Future: Initialize logging, metrics, etc.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_matl_basic() {
        let pogq = ProofOfGradientQuality::new(0.9, 0.85, 0.1);
        assert!(pogq.quality >= 0.0 && pogq.quality <= 1.0);

        let composite = pogq.composite_score(0.8);
        assert!(composite >= 0.0 && composite <= 1.0);
    }

    #[test]
    fn test_epistemic_claim() {
        let claim = EpistemicClaim::new(
            "Test claim",
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        );

        assert_eq!(claim.empirical, EmpiricalLevel::E2PrivateVerify);
        assert!(claim.meets_standard(EmpiricalLevel::E1Testimonial, NormativeLevel::N0Personal));
    }
}
