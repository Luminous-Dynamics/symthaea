// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! MATL - Mycelix Adaptive Trust Layer
//!
//! Core trust mechanisms enabling 45% Byzantine fault tolerance through:
//! - Proof of Gradient Quality (PoGQ) with v4.1 Enhanced support
//! - Reputation-weighted validation
//! - Adaptive per-node thresholds
//! - Hierarchical + cartel Byzantine detection
//! - Network-level adaptive Byzantine tolerance
//! - Warm-up quota and hysteresis for stable detection
//!
//! # Security Assumptions
//!
//! This module assumes:
//! - **Byzantine Fraction**: Less than 45% of network nodes are Byzantine
//! - **Network Connectivity**: Messages are eventually delivered (async model)
//! - **Bounded Delay**: Network delays are bounded for liveness
//! - **Correct Validators**: Honest nodes follow the protocol correctly
//!
//! ## Threat Model
//!
//! - Byzantine nodes may send arbitrary messages
//! - Byzantine nodes may collude to manipulate consensus
//! - Byzantine nodes may attempt to game reputation scores
//! - Adversary cannot control more than 45% of weighted voting power
//!
//! ## Limitations
//!
//! - 45% Byzantine tolerance is a theoretical maximum under ideal conditions
//! - Floating-point reputation calculations may have edge cases (NaN, overflow)
//! - Adaptive thresholds use EMA smoothing which can be slowly manipulated
//! - Cartel detection has false positive/negative rates at boundaries
//!
//! ## Security Best Practices
//!
//! 1. Monitor Byzantine detection metrics and set alerts for anomalies
//! 2. Validate all floating-point inputs (check for NaN/Infinity)
//! 3. Use per-node adaptive thresholds for Byzantine detection
//! 4. Implement proper key management for validator identities
//! 5. Consider formal verification for critical consensus paths

mod adaptive;
mod adaptive_byzantine;
mod cartel;
mod composite;
mod engine;
mod hierarchical;
mod kvector;
mod pogq;
mod pogq_enhanced;
mod rbbft;
mod reputation;

pub use adaptive::{AdaptiveThreshold, AdaptiveThresholdManager};
pub use adaptive_byzantine::{
    AdaptiveByzantineThreshold, MIN_BYZANTINE_TOLERANCE, NetworkStatus, ThresholdRecommendation,
};
pub use cartel::CartelDetector;
pub use composite::CompositeScore;
pub use engine::{MatlEngine, NetworkEvaluation, NodeEvaluation};
pub use hierarchical::HierarchicalDetector;
pub use kvector::{
    CachedKVector, GovernanceTier, KVECTOR_WEIGHTS, KVECTOR_WEIGHTS_ARRAY, KVector, KVectorBatch,
    KVectorDimension, KVectorWeights,
};
pub use pogq::ProofOfGradientQuality;
pub use pogq_enhanced::{
    ClientState, DetectionStatistics, PoGQEvaluation, PoGQv41Config, PoGQv41Enhanced,
};
pub use rbbft::{
    BlockProposal, ChallengeEvidence, ConsensusResult, ConsensusStats, MIN_VALIDATOR_REPUTATION,
    QUORUM_THRESHOLD, RBBFT_BYZANTINE_THRESHOLD, RbBftConfig, RbBftConsensus, RoundState,
    ValidatorNode, ViolationType, Vote, VoteType,
};
pub use reputation::{ReputationHistory, ReputationScore};

/// Default weights for composite score calculation.
/// Default weight for gradient quality in composite scoring.
pub const DEFAULT_QUALITY_WEIGHT: f64 = 0.4;
/// Default weight for temporal consistency in composite scoring.
pub const DEFAULT_CONSISTENCY_WEIGHT: f64 = 0.3;
/// Default weight for reputation in composite scoring.
pub const DEFAULT_REPUTATION_WEIGHT: f64 = 0.3;

/// Byzantine detection threshold (below this = suspicious).
pub const DEFAULT_BYZANTINE_THRESHOLD: f64 = 0.5;

/// Maximum Byzantine tolerance (revolutionary 45% vs classical 33%).
pub const MAX_BYZANTINE_TOLERANCE: f64 = 0.45;
