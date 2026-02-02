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

mod types;
mod aggregation;
mod coordinator;
mod serialization;
mod hyperfeel_bridge;
mod zkproof_bridge;
mod rbbft_bridge;
mod unified_zkrbbft_bridge;
mod epistemic_fl_bridge;
mod advanced_integrations;
pub mod matl_feedback;
pub mod prover_integration;

#[cfg(test)]
mod byzantine_stress_tests;

pub use types::{
    GradientUpdate, GradientMetadata, AggregatedGradient,
    Participant, FLRound, RoundStatus, AggregationMethod, FLConfig,
};
pub use aggregation::{
    fedavg, trimmed_mean, coordinate_median, krum, trust_weighted_aggregation,
    euclidean_distance, validate_gradient_consistency, AggregationError,
    // Performance optimizations
    GradientAccumulator, ByzantineDetectionResult, EarlyByzantineDetector,
    fedavg_optimized, aggregate_with_early_termination, GradientStats,
    // Multi-signal Byzantine detection (addresses adaptive adversary simulation finding)
    MultiSignalByzantineDetector, MultiSignalDetectionResult, SignalBreakdown,
    SignalWeights, DetectionStats, aggregate_with_multi_signal_detection,
    // Adaptive aggregation with automatic method selection
    AdaptiveAggregator, AdaptiveAggregatorConfig, AdaptiveAggregationResult,
    AdaptiveAggregatorStats, ParticipantBehavior,
};
pub use coordinator::FLCoordinator;
pub use serialization::{
    serialize_gradients, deserialize_gradients,
    SerializedGradientUpdate, SerializedAggregatedGradient,
};
pub use hyperfeel_bridge::{
    HyperFeelFLBridge, HyperFeelFLConfig, HyperFeelFLCoordinator,
    CompressedSubmission, ByzantineAnalysis, CompressionStats,
    AggregatedHyperGradient, HVAggregationMethod,
};
// ZKProofFLBridge requires simulation or risc0 feature
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use zkproof_bridge::ZKProofFLBridge;

// Types that don't require prover features
pub use zkproof_bridge::{
    ProvenGradientUpdate, ZKFLStats, ZKFLError,
    VerifiedAggregationMethod, VerifiedAggregationResult,
};

// RB-BFT integration for 45% Byzantine tolerance
pub use rbbft_bridge::{
    RbbftFLBridge, RbbftFLConfig, RbbftFLError, RbbftFLStats,
    RbbftParticipant, RbbftVote, RoundState as RbbftRoundState, RoundInfo as RbbftRoundInfo,
};

// Unified ZK + RB-BFT integration (ZK proofs + 45% Byzantine tolerance)
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use unified_zkrbbft_bridge::UnifiedZkRbbftBridge;

pub use unified_zkrbbft_bridge::{
    UnifiedZkRbbftConfig, UnifiedZkRbbftError, UnifiedZkRbbftStats,
    UnifiedParticipant, ProvenSubmission, UnifiedRoundState, UnifiedRoundInfo,
    UnifiedAggregationResult,
};

// Epistemic-weighted FL integration (E-N-M-H classification for gradients)
pub use epistemic_fl_bridge::{
    GradientEpistemicClassification, EpistemicGradientUpdate,
    GradientClassificationHints, classify_gradient,
    EpistemicAggregationResult, epistemic_weighted_aggregation,
    EpistemicAggregationError, RoundEpistemicStats, EpistemicQualityTier,
    // Byzantine attack classification with epistemic dimensions
    ByzantineAttackType, ByzantineScope, EpistemicByzantineResult,
    KVectorPenalties, classify_byzantine_attack, classify_byzantine_batch,
    // Agent behavioral feedback loop (FL → K-Vector)
    FLRoundFeedback, ParticipantFeedback, KVectorUpdates, compute_round_feedback,
    // Phi-gated Byzantine detection
    phi_thresholds, PhiParticipationResult, PhiRecommendation,
    check_phi_participation, PhiAwareByzantineAnalysis,
    PhiByzantineClassification, PhiByzantineAction,
    analyze_byzantine_with_phi, phi_filter_updates,
};

// Advanced integrations (#175-180)
pub use advanced_integrations::{
    // #175: HyperFeel for agent communication
    CompressedAgentState, AgentCompressionConfig, compress_agent_state,
    compute_state_similarity, CompressedConsensusResult, compute_compressed_consensus,
    // #176: DKG quality claims
    FLQualityClaim, QualityEvidence, QualityAttestation,
    // #177: PoG-weighted FL participation
    pog_thresholds, PogParticipant, PogVerificationLevel, PogParticipationResult,
    check_pog_participation, PogWeightedResult, pog_weighted_aggregation,
    // #178: Cross-domain ZK proofs
    GradientContributionProof, ZkProofData, ZkProofType,
    CrossDomainVerification, CrossDomainCredits, verify_cross_domain_proof,
    // #179: Temporal commitment-backed FL
    FLTemporalCommitment, CommitmentTier, CommitmentStatus,
    // #180: Storage-backed FL immutability
    FLMerkleNode, FLMerkleLeaf, FLRoundArchive, StorageReceipt, StorageBackend,
    build_gradient_merkle_tree, create_round_archive,
    generate_merkle_proof, verify_merkle_proof,
};

// MATL Feedback Loop (#184-185)
pub use matl_feedback::{
    // Gradient quality signals
    GradientQualitySignals, GradientQualityConfig, QualityTier,
    // K-Vector delta application
    KVectorDelta,
    // MATL feedback computation
    MatlFeedbackComputer, MatlFeedbackConfig, FLMatlFeedback, FeedbackStats,
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
