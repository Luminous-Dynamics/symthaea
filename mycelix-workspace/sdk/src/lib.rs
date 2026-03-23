// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Clippy configuration - allow certain lints that are too noisy
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::inconsistent_digit_grouping)]
// Allow unwrap in certain contexts (we use expect() for better messages where appropriate)
#![allow(clippy::unwrap_used)]
// Allow result unwrap (similar reasoning)
#![allow(clippy::unwrap_in_result)]
// Allow manual clamp patterns (often clearer for simple cases)
#![allow(clippy::manual_clamp)]
// Allow unused results in test contexts
#![cfg_attr(test, allow(unused_must_use))]
// Allow unknown cfg conditions for feature flags used in conditional compilation
#![allow(unexpected_cfgs)]

//! # Mycelix SDK
//!
//! Shared library for the Mycelix ecosystem providing:
//! - **MATL**: Mycelix Adaptive Trust Layer (34% validated Byzantine tolerance)
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
//! - **Finance**: Finance bridge client for cross-cluster currency queries
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

pub mod crypto;
pub mod dkg;
pub mod fl;
pub mod matl;

// TypeScript binding generation (ts-export feature only)
pub mod agentic;
pub mod bridge;
pub mod credentials;
pub mod economics;
pub mod epistemic;
pub mod error;
#[cfg(feature = "ts-export")]
mod export_bindings;
pub mod finance;
pub mod hyperfeel;
pub mod space;
pub mod identity;
pub mod intentions;
pub mod pagination;
pub mod pog;
pub mod storage;
pub mod temporal;
pub mod zkproof;

// WASM bindings (wasm feature only)
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience
pub use bridge::{BridgeMessage, CrossHappReputation};
pub use credentials::{CredentialBuilder, VerifiableCredential};
pub use dkg::{
    calculate_confidence, meets_threshold, query_by_predicate, query_by_subject, query_triples,
    Attestation, AttestationCounts, AttestationSet, AttestationType, ConfidenceFactors,
    ConfidenceInput, ConfidenceScore, ConfidenceThresholds, DKGConfig,
    EpistemicType as DkgEpistemicType, ObjectFilter, QueryFilter, StoredTriple, TripleValue,
    VerifiableTriple, URI,
};
pub use epistemic::EpistemicClassificationExtended;
pub use epistemic::{
    EmpiricalLevel, EpistemicClaim, HarmonicLevel, MaterialityLevel, NormativeLevel,
};
pub use error::{
    contains_sensitive_data, sanitize_message, SanitizeExt, SanitizedError, SanitizedResult,
};
pub use fl::{AggregatedGradient, AggregationMethod, FLConfig, FLCoordinator, GradientUpdate};
pub use hyperfeel::{EncodingConfig, HyperFeelEncoder, HyperGradient};
pub use identity::{
    BridgeError as HardwareKeyBridgeError, HardwareKeyBridge, HardwareKeyBridgeConfig,
    WebAuthnCredential, WebAuthnError, WebAuthnService,
};
pub use matl::{AdaptiveThreshold, CompositeScore, ProofOfGradientQuality, ReputationScore};
pub use matl::{ConsensusResult, RbBftConfig, RbBftConsensus, RoundState, Vote, VoteType};
pub use matl::{GovernanceTier, KVector, KVectorDimension, KVectorWeights, KVECTOR_WEIGHTS};
pub use pagination::{paginate_vec, PaginatedResponse, PaginationRequest};
pub use storage::{EpistemicStorage, StorageConfig, StorageError, StorageReceipt, StorageRouter};

// Finance bridge exports (cross-cluster finance queries)
pub use finance::{
    BalanceResponse, FeeTierResponse, FinanceBridgeClient, FinanceBridgeHealth,
    FinanceSummaryResponse, TendBalanceResponse, TendLimitResponse,
};
pub use zkproof::{GradientProof, GradientProofCircuit, PublicInputs};

// Space cluster exports (SSA: conjunction screening, negotiation, debris bounties)
pub use space::{
    ConjunctionAssessment, CreateBountyRequest, CreateProposalRequest, ManeuverOption,
    OptionTally, ProposalTally, ScreenConjunctionRequest, SpaceClient, VoteRequest,
};

// Agentic Economy exports (MIP-E-004: Epistemic-Aware AI Agency)
pub use agentic::{
    calculate_kredit_from_trust,
    compute_trust_score,
    ActionOutcome,
    // API service
    AgentApiService,
    AgentClass,
    AgentConstraints,
    // Core types
    AgentId,
    // Epistemic classification
    AgentOutput,
    AgentStatus,
    ApiError,
    ApiResult,
    BehaviorLogEntry,
    CoherenceCheckResult,
    CoherenceMeasurementConfig,
    // Coherence (Phi) measurement
    CoherenceState,
    CreateAgentRequest,
    EpistemicStats,
    EscalationRequest,
    GamingDetectionConfig,
    GamingDetectionResult,
    // Gaming detection
    GamingDetector,
    GamingResponse,
    InstrumentalActor,
    // K-Vector integration
    KVectorBridgeConfig,
    MoralActionGuidance,
    // GIS integration (Graceful Ignorance System)
    MoralUncertainty,
    MoralUncertaintyType,
    // Quarantine
    QuarantineManager,
    QuarantineReason,
    UncertaintyCalibration,
    UpdateAgentRequest,
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
