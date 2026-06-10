// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated Learning Module
//!
//! Byzantine-resistant distributed machine learning with MATL integration.
//! Supports multiple aggregation algorithms with trust-weighted contributions.
//!
//! # Key Features
//!
//! - **FedAvg**: Standard Federated Averaging
//! - **TrimmedMean**: Outlier-resistant averaging
//! - **CoordinateMedian**: Robust to 50% Byzantine participants
//! - **Krum**: Selects gradients closest to neighbors
//! - **TrustWeighted**: MATL-integrated trust-based aggregation
//!
//! # Example
//!
//! ```rust
//! use mycelix_sdk::fl::{FLCoordinator, FLConfig, AggregationMethod, GradientUpdate};
//!
//! // Create coordinator with default config
//! let mut coordinator = FLCoordinator::new(FLConfig::default());
//!
//! // Register participants
//! coordinator.register_participant("participant-1".to_string());
//! coordinator.register_participant("participant-2".to_string());
//! coordinator.register_participant("participant-3".to_string());
//!
//! // Start a round
//! coordinator.start_round().expect("Failed to start round");
//!
//! // Submit gradient updates
//! let update = GradientUpdate::new(
//!     "participant-1".to_string(),
//!     1,
//!     vec![0.1, 0.2, 0.3],
//!     100,
//!     0.5,
//! );
//! coordinator.submit_update(update);
//! ```

mod advanced_integrations;
mod aggregation;
mod coordinator;
mod epistemic_fl_bridge;
mod hyperfeel_bridge;
pub mod matl_feedback;
pub mod plugin_adapters;
pub mod prover_integration;
mod rbbft_bridge;
mod serialization;
mod types;
pub mod unified_pipeline;
mod unified_zkrbbft_bridge;
mod zkproof_bridge;

#[cfg(test)]
mod byzantine_stress_tests;

pub use aggregation::{
    aggregate_with_early_termination,
    aggregate_with_multi_signal_detection,
    coordinate_median,
    euclidean_distance,
    fedavg,
    fedavg_optimized,
    krum,
    trimmed_mean,
    trust_weighted_aggregation,
    validate_gradient_consistency,
    AdaptiveAggregationResult,
    // Adaptive aggregation with automatic method selection
    AdaptiveAggregator,
    AdaptiveAggregatorConfig,
    AdaptiveAggregatorStats,
    AggregationError,
    ByzantineDetectionResult,
    DetectionStats,
    EarlyByzantineDetector,
    // Performance optimizations
    GradientAccumulator,
    GradientStats,
    // Multi-signal Byzantine detection (addresses adaptive adversary simulation finding)
    MultiSignalByzantineDetector,
    MultiSignalDetectionResult,
    ParticipantBehavior,
    SignalBreakdown,
    SignalWeights,
};
pub use coordinator::FLCoordinator;
pub use hyperfeel_bridge::{
    AggregatedHyperGradient, ByzantineAnalysis, CompressedSubmission, CompressionStats,
    HVAggregationMethod, HyperFeelFLBridge, HyperFeelFLConfig, HyperFeelFLCoordinator,
};
pub use serialization::{
    deserialize_gradients, serialize_gradients, SerializedAggregatedGradient,
    SerializedGradientUpdate,
};
pub use types::{
    AggregatedGradient, AggregationMethod, FLConfig, FLRound, GradientMetadata, GradientUpdate,
    Participant, RoundStatus,
};
// ZKProofFLBridge requires simulation or risc0 feature
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use zkproof_bridge::ZKProofFLBridge;

// Types that don't require prover features
pub use zkproof_bridge::{
    VerifiedAggregationMethod, VerifiedAggregationResult, ZKFLError, ZKFLStats,
};

// ProvenGradientUpdate contains GradientProofReceipt, so requires feature gate
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use zkproof_bridge::ProvenGradientUpdate;

// RB-BFT integration for 34% validated Byzantine tolerance
pub use rbbft_bridge::{
    RbbftFLBridge, RbbftFLConfig, RbbftFLError, RbbftFLStats, RbbftParticipant, RbbftVote,
    RoundInfo as RbbftRoundInfo, RoundState as RbbftRoundState,
};

// Unified ZK + RB-BFT integration (ZK proofs + 34% validated Byzantine tolerance)
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use unified_zkrbbft_bridge::UnifiedZkRbbftBridge;

pub use unified_zkrbbft_bridge::{
    ProvenSubmission, UnifiedAggregationResult, UnifiedParticipant, UnifiedRoundInfo,
    UnifiedRoundState, UnifiedZkRbbftConfig, UnifiedZkRbbftError, UnifiedZkRbbftStats,
};

// Epistemic-weighted FL integration (E-N-M-H classification for gradients)
pub use epistemic_fl_bridge::{
    analyze_byzantine_with_phi,
    check_coherence_participation,
    classify_byzantine_attack,
    classify_byzantine_batch,
    classify_gradient,
    compute_round_feedback,
    epistemic_weighted_aggregation,
    phi_filter_updates,
    // Phi-gated Byzantine detection
    phi_thresholds,
    // Byzantine attack classification with epistemic dimensions
    ByzantineAttackType,
    ByzantineScope,
    EpistemicAggregationError,
    EpistemicAggregationResult,
    EpistemicByzantineResult,
    EpistemicGradientUpdate,
    EpistemicQualityTier,
    // Agent behavioral feedback loop (FL → K-Vector)
    FLRoundFeedback,
    GradientClassificationHints,
    GradientEpistemicClassification,
    KVectorPenalties,
    KVectorUpdates,
    ParticipantFeedback,
    PhiAwareByzantineAnalysis,
    PhiByzantineAction,
    PhiByzantineClassification,
    PhiParticipationResult,
    PhiRecommendation,
    RoundEpistemicStats,
};

// Advanced integrations (#175-180)
pub use advanced_integrations::{
    build_gradient_merkle_tree,
    check_pog_participation,
    compress_agent_state,
    compute_compressed_consensus,
    compute_state_similarity,
    create_round_archive,
    generate_merkle_proof,
    // #177: PoG-weighted FL participation
    pog_thresholds,
    pog_weighted_aggregation,
    verify_cross_domain_proof,
    verify_merkle_proof,
    AgentCompressionConfig,
    CommitmentStatus,
    CommitmentTier,
    // #175: HyperFeel for agent communication
    CompressedAgentState,
    CompressedConsensusResult,
    CrossDomainCredits,
    CrossDomainVerification,
    FLMerkleLeaf,
    // #180: Storage-backed FL immutability
    FLMerkleNode,
    // #176: DKG quality claims
    FLQualityClaim,
    FLRoundArchive,
    // #179: Temporal commitment-backed FL
    FLTemporalCommitment,
    // #178: Cross-domain ZK proofs
    GradientContributionProof,
    PogParticipant,
    PogParticipationResult,
    PogVerificationLevel,
    PogWeightedResult,
    QualityAttestation,
    QualityEvidence,
    StorageBackend,
    StorageReceipt,
    ZkProofData,
    ZkProofType,
};

// Unified FL Pipeline (shared with Symthaea via mycelix-fl-core)
pub use unified_pipeline::{PipelineConfigF64, PipelineResultF64, UnifiedPipelineF64};

// Re-export core pipeline types for direct access
pub use mycelix_fl_core::{
    byzantine::MultiSignalByzantineDetector as CoreMultiSignalDetector,
    hybrid_bft::{HybridAggregationResult, HybridBftConfig},
    pipeline::{
        ExternalWeightMap, ParticipantWeightAdjustment, PipelineConfig, PipelineResult,
        PipelineStats, UnifiedPipeline,
    },
    privacy::{DifferentialPrivacyConfig, PrivacyReport, RdpBudgetTracker},
    types::MAX_BYZANTINE_TOLERANCE,
};

// Plugin adapters (wrap SDK bridges as core plugin traits)
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use plugin_adapters::ZkVerificationPlugin;
pub use plugin_adapters::{
    ChecksumVerificationPlugin, EpistemicByzantinePlugin, HyperFeelByzantinePlugin,
    HyperFeelCompressionPlugin, MatlByzantinePlugin,
};

// Re-export core plugin traits and types for ergonomic use
pub use mycelix_fl_core::pipeline::PluginPipelineResult;
pub use mycelix_fl_core::plugins::{
    ByzantinePlugin, CompressedGradient, CompressionPlugin, PipelinePlugins, VerificationPlugin,
    VerificationResult as PluginVerificationResult,
};

// MATL Feedback Loop (#184-185)
pub use matl_feedback::{
    FLMatlFeedback,
    FeedbackStats,
    GradientQualityConfig,
    // Gradient quality signals
    GradientQualitySignals,
    // K-Vector delta application
    KVectorDelta,
    // MATL feedback computation
    MatlFeedbackComputer,
    MatlFeedbackConfig,
    QualityTier,
};

/// Default FL configuration
pub const DEFAULT_MIN_PARTICIPANTS: usize = 3;
/// Maximum number of FL participants.
pub const DEFAULT_MAX_PARTICIPANTS: usize = 100;
/// Default round timeout in milliseconds.
pub const DEFAULT_ROUND_TIMEOUT_MS: u64 = 60_000;
/// Default Byzantine fault tolerance ratio.
pub const DEFAULT_BYZANTINE_TOLERANCE: f64 = 0.33;
/// Default minimum trust threshold for participation.
pub const DEFAULT_TRUST_THRESHOLD: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_fl_round() {
        // Create coordinator
        let config = FLConfig {
            min_participants: 2,
            max_participants: 10,
            round_timeout_ms: 60_000,
            byzantine_tolerance: 0.33,
            aggregation_method: AggregationMethod::FedAvg,
            trust_threshold: 0.5,
        };

        let mut coordinator = FLCoordinator::new(config);

        // Register participants
        coordinator.register_participant("p1".to_string());
        coordinator.register_participant("p2".to_string());

        // Start round
        assert!(coordinator.start_round().is_ok());

        // Submit updates
        let update1 = GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2], 100, 0.5);
        let update2 = GradientUpdate::new("p2".to_string(), 1, vec![0.2, 0.3], 100, 0.4);

        assert!(coordinator.submit_update(update1));
        assert!(coordinator.submit_update(update2));

        // Aggregate
        let result = coordinator.aggregate_round();
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.participant_count, 2);
        assert_eq!(aggregated.gradients.len(), 2);
    }

    #[test]
    fn test_aggregation_methods() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.2, 0.3, 0.4], 100, 0.4),
            GradientUpdate::new("p3".to_string(), 1, vec![0.15, 0.25, 0.35], 100, 0.45),
        ];

        // Test FedAvg
        let avg = fedavg(&updates).expect("FedAvg failed");
        assert_eq!(avg.len(), 3);

        // Test Median
        let median = coordinate_median(&updates).expect("Median failed");
        assert_eq!(median.len(), 3);

        // Test Trimmed Mean
        let trimmed = trimmed_mean(&updates, 0.1).expect("Trimmed mean failed");
        assert_eq!(trimmed.len(), 3);
    }
}
