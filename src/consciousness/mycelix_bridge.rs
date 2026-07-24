#![allow(deprecated)]
// Uses legacy ConsciousnessCredential for backward-compat bridge
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Bridge - Consciousness-Gated Governance
//!
//! This module bridges Symthaea consciousness measurement to Mycelix governance,
//! enabling:
//! - **Consciousness-gated proposals**: Only submit when Φ > threshold
//! - **Value-aligned voting**: Evaluate proposals against Eight Harmonies
//! - **Federated value learning**: Share value insights via Mycelix network
//! - **Cross-hApp reputation**: Aggregate consciousness metrics across hApps
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    SYMTHAEA ↔ MYCELIX BRIDGE                          │
//! │                                                                       │
//! │  ┌────────────────────┐                ┌────────────────────┐        │
//! │  │    Symthaea        │                │      Mycelix       │        │
//! │  │                    │                │                    │        │
//! │  │ • Consciousness Φ  │◄──────────────►│ • Proposals (Gov)  │        │
//! │  │ • Eight Harmonies  │   Bridge       │ • MATL (Trust)     │        │
//! │  │ • Affective State  │   Protocol     │ • HyperFeel (FL)   │        │
//! │  │ • Unified Evaluator│                │ • Epistemic Charter│        │
//! │  └────────────────────┘                └────────────────────┘        │
//! │                                                                       │
//! │  ┌─────────────────────────────────────────────────────────────────┐ │
//! │  │                   Consciousness Gate                             │ │
//! │  │  • Proposal: Φ > 0.3 + value alignment + authenticity check     │ │
//! │  │  • Voting: Φ > 0.4 + harmony evaluation                         │ │
//! │  │  • Constitutional: Φ > 0.6 + full evaluation                    │ │
//! │  └─────────────────────────────────────────────────────────────────┘ │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration with Mycelix SDK
//!
//! When the `mycelix` feature is enabled, this module uses actual SDK types:
//! - `bridge::BridgeMessage` for inter-hApp communication
//! - `bridge::LocalBridge` for testing and local governance
//! - `matl::ProofOfGradientQuality` for value learning verification
//! - `hyperfeel::HyperFeelEncoder` for gradient compression (2000x)
//! - `hyperfeel::HyperGradient` for compressed value gradients
//! - `epistemic::EpistemicClaim` for truth classification
//!
//! Without the feature, fallback implementations maintain API compatibility.

// Canonical Phi threshold constants — must match mycelix_bridge_common::phi_thresholds::PhiThresholds::default().
// Source of truth: crates/mycelix-bridge-common/src/phi_thresholds.rs
const GOV_BASIC: f64 = 0.2;
const GOV_PROPOSAL: f64 = 0.3;
const GOV_VOTING: f64 = 0.4;
const GOV_CONSTITUTIONAL: f64 = 0.6;

// Reputation thresholds — intentionally higher than consciousness thresholds
// because they combine consciousness (60%) + hApp reputation (40%).
const REP_BASIC: f64 = 0.3;
const REP_GOVERNANCE: f64 = 0.5;
const REP_VOTING: f64 = 0.6;
const REP_CONSTITUTIONAL: f64 = 0.8;

use super::affective_consciousness::CoreAffect;
use super::eight_harmonies::Harmony;
use super::unified_value_evaluator::{
    ActionType, AffectiveSystemsState, Decision, EvaluationContext, EvaluationResult,
    UnifiedValueEvaluator,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// MYCELIX SDK INTEGRATION (when feature enabled)
// ============================================================================

// mycelix_sdk is currently unavailable (removed for Nix sandbox compatibility).
// Re-enable when mycelix-sdk is published to crates.io.
#[cfg(feature = "mycelix_sdk")]
use mycelix_sdk::{
    bridge::{BridgeEvent, BridgeMessage, CrossHappReputation, HappReputationScore, LocalBridge},
    epistemic::{EmpiricalLevel, EpistemicClaim, MaterialityLevel, NormativeLevel},
    hyperfeel::{EncodingConfig, HyperFeelEncoder, HyperGradient},
    matl::ProofOfGradientQuality,
};

#[cfg(feature = "mycelix_sdk")]
use sha3::{Digest, Sha3_256};

// ============================================================================
// CONSCIOUSNESS METADATA
// ============================================================================

/// Consciousness state snapshot for governance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    /// Integrated information (Φ)
    pub phi: f64,
    /// Meta-awareness level
    pub meta_awareness: f64,
    /// Self-model accuracy
    pub self_model_accuracy: f64,
    /// Narrative coherence
    pub coherence: f64,
    /// Affective state summary
    pub affective_valence: f64,
    /// CARE system activation
    pub care_activation: f64,
    /// Timestamp
    pub timestamp_secs: u64,
}

impl ConsciousnessSnapshot {
    /// Create from current consciousness state
    pub fn new(
        phi: f64,
        meta_awareness: f64,
        self_model_accuracy: f64,
        coherence: f64,
        affective_valence: f64,
        care_activation: f64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            phi,
            meta_awareness,
            self_model_accuracy,
            coherence,
            affective_valence,
            care_activation,
            timestamp_secs: now,
        }
    }

    /// Maximum age (seconds) before a consciousness snapshot is considered stale.
    /// Prevents governance actions based on outdated consciousness state.
    const MAX_AGE_SECS: u64 = 30;

    /// Check if consciousness is adequate for action type.
    /// Returns false if the snapshot is stale (older than MAX_AGE_SECS).
    pub fn is_adequate_for(&self, action_type: ActionType) -> bool {
        // Reject stale snapshots — consciousness can decay significantly in 30s
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now > 0
            && self.timestamp_secs > 0
            && now.saturating_sub(self.timestamp_secs) > Self::MAX_AGE_SECS
        {
            return false;
        }

        let required = match action_type {
            ActionType::Basic => GOV_BASIC,
            ActionType::Governance => GOV_PROPOSAL,
            ActionType::Voting => GOV_VOTING,
            ActionType::Constitutional => GOV_CONSTITUTIONAL,
        };
        self.phi >= required
    }

    /// Overall consciousness quality score
    pub fn quality_score(&self) -> f64 {
        (self.phi * 0.4
            + self.meta_awareness * 0.2
            + self.self_model_accuracy * 0.2
            + self.coherence * 0.2)
            .clamp(0.0, 1.0)
    }
}

// ============================================================================
// FACTCHECK → EPISTEMIC CUBE FEEDBACK (Phase 4)
// ============================================================================

/// Result from a Mycelix factcheck query, mapped to epistemic cube coordinates.
///
/// This is the inward direction of the Mycelix-Symthaea loop:
/// Mycelix factcheck → EpistemicCubeFromFactcheck → inject_epistemic_cube()
///
/// The outward direction (Symthaea → Mycelix) is `create_epistemic_claim()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactcheckEpistemicFeedback {
    /// E-tier (0-4): empirical verifiability derived from evidence quality
    pub e_tier: u8,
    /// N-tier (0-3): normative authority derived from source consensus
    pub n_tier: u8,
    /// M-tier (0-3): materiality/permanence derived from evidence persistence
    pub m_tier: u8,
    /// H-value (0.0-1.0): harmonic coherence from verdict confidence
    pub h_value: f32,
    /// Composite quality score
    pub quality: f32,
    /// Original verdict for logging
    pub verdict: String,
    /// Source statement
    pub statement: String,
}

impl FactcheckEpistemicFeedback {
    /// Convert a Mycelix factcheck result into epistemic cube coordinates.
    ///
    /// Maps the 3D EpistemicPosition (empirical, normative, mythic) from the
    /// knowledge graph directly to the consciousness cube axes:
    /// - `empirical` (0.0-1.0) → E-tier (0-4): evidence quality
    /// - `normative` (0.0-1.0) → N-tier (0-3): source consensus
    /// - `mythic` (0.0-1.0) → M-tier (0-3): persistence/foundationality
    /// - `verdict_confidence` → H-value: how coherently the evidence supports
    ///
    /// Science: Goldman (1986) — reliabilist epistemology; knowledge justified
    /// by the reliability of the process that produced it.
    pub fn from_factcheck(
        statement: &str,
        verdict: &str,
        verdict_confidence: f64,
        empirical: f64,
        normative: f64,
        mythic: f64,
        credibility: f64,
    ) -> Self {
        // Map continuous 0.0-1.0 to discrete tiers
        let e_tier = match empirical {
            e if e >= 0.8 => 4, // reproducible
            e if e >= 0.6 => 3, // proven
            e if e >= 0.4 => 2, // verifiable
            e if e >= 0.2 => 1, // testimonial
            _ => 0,             // opinion
        };

        let n_tier = match normative {
            n if n >= 0.75 => 3, // axiomatic
            n if n >= 0.5 => 2,  // network consensus
            n if n >= 0.25 => 1, // communal
            _ => 0,              // personal
        };

        let m_tier = match mythic {
            m if m >= 0.75 => 3, // foundational
            m if m >= 0.5 => 2,  // persistent
            m if m >= 0.25 => 1, // temporal
            _ => 0,              // ephemeral
        };

        // H-value: blend verdict confidence with credibility
        let h_value = (verdict_confidence * 0.7 + credibility * 0.3).clamp(0.0, 1.0) as f32;

        // Quality: same formula as the cube encoder
        let quality = (e_tier as f32 / 4.0) * 0.40
            + (n_tier as f32 / 3.0) * 0.35
            + (m_tier as f32 / 3.0) * 0.25;

        Self {
            e_tier,
            n_tier,
            m_tier,
            h_value,
            quality,
            verdict: verdict.to_string(),
            statement: statement.to_string(),
        }
    }

    /// Create feedback from a verdict string (convenience for non-Holochain callers).
    ///
    /// Maps verdict names to rough epistemic positions:
    /// - True/MostlyTrue → high empirical, high confidence
    /// - Mixed → moderate empirical, low confidence
    /// - MostlyFalse/False → low empirical (contradicted)
    /// - Unverifiable → E0 (opinion domain)
    pub fn from_verdict_string(statement: &str, verdict: &str, confidence: f64) -> Self {
        let (empirical, normative, mythic) = match verdict {
            "True" => (0.9, 0.7, 0.6),
            "MostlyTrue" => (0.7, 0.5, 0.5),
            "Mixed" => (0.4, 0.3, 0.4),
            "MostlyFalse" => (0.2, 0.3, 0.3),
            "False" => (0.1, 0.2, 0.2),
            "Unverifiable" => (0.0, 0.1, 0.1),
            _ => (0.3, 0.3, 0.3), // unknown verdict → moderate
        };

        Self::from_factcheck(
            statement, verdict, confidence, empirical, normative, mythic, confidence,
        )
    }
}

/// Value alignment result for governance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueAlignmentResult {
    /// Overall alignment score (-1 to +1)
    pub overall_score: f64,
    /// Individual harmony scores
    pub harmony_scores: HashMap<String, f64>,
    /// Violations detected
    pub violations: Vec<String>,
    /// Authenticity score (genuine caring check)
    pub authenticity: f64,
    /// Recommendation
    pub recommendation: GovernanceRecommendation,
}

/// Governance recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernanceRecommendation {
    /// Strongly support this proposal
    StrongSupport,
    /// Support with minor concerns
    Support,
    /// Neutral - needs more consideration
    Neutral,
    /// Oppose due to misalignment
    Oppose,
    /// Strongly oppose due to value violations
    StrongOppose,
    /// Cannot evaluate (insufficient consciousness)
    CannotEvaluate,
}

// ============================================================================
// GOVERNANCE TYPES
// ============================================================================

/// A governance proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique identifier
    pub id: String,
    /// Proposal title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Proposer agent ID
    pub proposer: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Proposal type
    pub proposal_type: ProposalType,
    /// Required consciousness level
    pub required_phi: f64,
}

/// Types of proposals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalType {
    /// Standard governance proposal
    Standard,
    /// Constitutional amendment
    Constitutional,
    /// Emergency action
    Emergency,
    /// Community grant
    Grant,
    /// Parameter change
    Parameter,
}

impl ProposalType {
    /// Get the action type for evaluation
    pub fn to_action_type(&self) -> ActionType {
        match self {
            ProposalType::Standard => ActionType::Governance,
            ProposalType::Constitutional => ActionType::Constitutional,
            ProposalType::Emergency => ActionType::Governance,
            ProposalType::Grant => ActionType::Governance,
            ProposalType::Parameter => ActionType::Governance,
        }
    }
}

/// A vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Proposal ID
    pub proposal_id: String,
    /// Voter agent ID
    pub voter: String,
    /// Vote value
    pub value: VoteValue,
    /// Consciousness snapshot at time of vote
    pub consciousness: ConsciousnessSnapshot,
    /// Value alignment evaluation
    pub alignment: ValueAlignmentResult,
    /// Timestamp
    pub timestamp: u64,
}

/// Vote values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteValue {
    /// Strong support
    StrongYes,
    /// Support
    Yes,
    /// Abstain
    Abstain,
    /// Oppose
    No,
    /// Strong opposition
    StrongNo,
}

// ============================================================================
// FEDERATED VALUE LEARNING
// ============================================================================

/// Value learning update for federated sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueLearningUpdate {
    /// Agent ID
    pub agent_id: String,
    /// Harmony being updated
    pub harmony: String,
    /// Importance delta (+/- adjustment)
    pub importance_delta: f64,
    /// Affirmation count change
    pub affirmation_delta: i64,
    /// Context description (for verification)
    pub context: String,
    /// Consciousness level when learning occurred
    pub phi_at_learning: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Compressed value gradient for efficient transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedValueGradient {
    /// Harmony encoded as BinaryHV
    pub harmony_encoding: Vec<u8>,
    /// Importance gradient (compressed)
    pub importance_gradient: Vec<u8>,
    /// Metadata
    pub round: u64,
    pub agent_id: String,
    pub compression_ratio: f64,
}

// ============================================================================
// MYCELIX BRIDGE
// ============================================================================

/// Configuration for the Mycelix bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Minimum Φ for proposal submission
    pub min_phi_proposal: f64,
    /// Minimum Φ for voting
    pub min_phi_voting: f64,
    /// Minimum Φ for constitutional changes
    pub min_phi_constitutional: f64,
    /// Require CARE activation for proposals affecting others
    pub require_care_for_others: bool,
    /// Minimum CARE activation
    pub min_care_activation: f64,
    /// Enable federated value learning
    pub enable_federated_learning: bool,
    /// Value learning batch size
    pub fl_batch_size: usize,
    /// Value learning sync interval (seconds)
    pub fl_sync_interval_secs: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            min_phi_proposal: GOV_PROPOSAL,
            min_phi_voting: GOV_VOTING,
            min_phi_constitutional: GOV_CONSTITUTIONAL,
            require_care_for_others: true,
            min_care_activation: GOV_PROPOSAL,
            enable_federated_learning: true,
            fl_batch_size: 10,
            fl_sync_interval_secs: 300, // 5 minutes
        }
    }
}

/// The Mycelix Bridge - connects consciousness to governance
pub struct MycelixBridge {
    /// Unified value evaluator
    evaluator: UnifiedValueEvaluator,
    /// Configuration
    config: BridgeConfig,
    /// Pending value learning updates
    pending_updates: Vec<ValueLearningUpdate>,
    /// Last sync time
    last_sync: Instant,
    /// Local agent ID
    agent_id: String,
    /// Proposal evaluation cache
    proposal_cache: HashMap<String, (ValueAlignmentResult, Instant)>,
    /// Cache TTL
    cache_ttl: Duration,
    /// Pending factcheck epistemic feedback for cognitive loop injection.
    /// Populated by `submit_factcheck_feedback()`, drained by the cognitive loop.
    pending_factcheck_feedback: Vec<FactcheckEpistemicFeedback>,
    /// Pending waste/circular economy events for CLS injection.
    #[cfg(feature = "circular")]
    pending_waste_events: Vec<WasteBridgeEvent>,
    /// Pending governance events for CLS injection
    #[cfg(feature = "mycelix")]
    pending_gov_events: Vec<crate::cognitive_loop::managers::governance_manager::GovernanceEvent>,
    /// Pending governance outcomes for CLS injection
    #[cfg(feature = "mycelix")]
    pending_gov_outcomes:
        Vec<crate::cognitive_loop::managers::governance_manager::GovernanceOutcome>,
    /// Channel sender for dispatching governance actions to an external
    /// Holochain conductor process. The conductor bridge (`symthaea-mycelix-holochain`)
    /// runs separately to avoid serde version conflicts.
    /// Set via `set_governance_dispatch_tx()`.
    #[cfg(feature = "mycelix")]
    governance_dispatch_tx: Option<std::sync::mpsc::SyncSender<GovernanceDispatchCommand>>,
    /// Pending dispatch confirmations: correlation_id -> dispatch time.
    /// Tracked so we can detect when the conductor has not acknowledged a command.
    #[cfg(feature = "mycelix")]
    pending_confirmations: HashMap<u64, Instant>,
    /// Monotonically increasing correlation ID counter.
    #[cfg(feature = "mycelix")]
    next_correlation_id: u64,
    /// On-chain asset/order binding for robotics telemetry (None = not
    /// registered → telemetry is not emitted).
    #[cfg(feature = "mycelix")]
    robotics_binding: Option<RoboticsDispatchBinding>,
    /// Host-supplied mission status (None = no position/progress source →
    /// telemetry is not emitted).
    #[cfg(feature = "mycelix")]
    robotics_status: Option<RoboticsMissionStatus>,
    /// Last robotics telemetry dispatch time (rate limiting — the loop runs
    /// ~31 Hz; one DHT entry per cycle would be abusive).
    #[cfg(feature = "mycelix")]
    last_robotics_dispatch: Option<Instant>,
    /// HyperFeel encoder for gradient compression (2000x via JL projection).
    /// Feature-gated: only available when the mycelix SDK is linked.
    #[cfg(feature = "mycelix_sdk")]
    hyperfeel_encoder: HyperFeelEncoder,
}

/// Commands dispatched to an external Holochain conductor process.
///
/// The conductor bridge (`symthaea-mycelix-holochain` crate) receives these
/// via `std::sync::mpsc::Receiver<GovernanceDispatchCommand>` and translates
/// them into real zome calls via `AppWebsocket`.
#[cfg(feature = "mycelix")]
#[derive(Debug, Clone)]
pub enum GovernanceDispatchCommand {
    /// Submit a proposal to the governance cluster.
    SubmitProposal {
        /// Unique correlation ID for matching dispatch to confirmation.
        correlation_id: u64,
        description: String,
        proposer_did: String,
        /// Enriched consciousness signals from Symthaea snapshot.
        /// Used by the bridge adapter to call `ConsciousnessCredential::from_symthaea()`.
        consciousness_phi: f64,
        meta_awareness: f64,
        coherence: f64,
        care_activation: f64,
        alignment_score: f64,
    },
    /// Cast a vote on an existing proposal.
    CastVote {
        /// Unique correlation ID for matching dispatch to confirmation.
        correlation_id: u64,
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
        /// Enriched consciousness signals for vote weighting.
        consciousness_phi: f64,
        meta_awareness: f64,
        coherence: f64,
        care_activation: f64,
    },
    /// Query active proposals (response arrives via governance event channel).
    QueryActiveProposals,
    /// Evaluate an asset and record the consciousness assessment on-chain.
    EvaluateAsset {
        correlation_id: u64,
        project_id: String,
        description: String,
        project_type: String,
        capacity_mw: f64,
        community_did: Option<String>,
        impact_claims: Vec<String>,
        phi_score: f64,
        harmony_alignment: f64,
        per_harmony_scores: String,
        care_activation: f64,
        meta_awareness: f64,
    },
    /// Declare a civic crisis to the Mycelix emergency-incidents zome.
    DeclareCrisis {
        correlation_id: u64,
        /// FEMA-aligned severity (1-5).
        severity: u8,
        /// Maps to Mycelix DisasterType.
        crisis_type: String,
        /// Human-readable description of the detected anomaly.
        description: String,
        /// Detection confidence (0.0-1.0).
        confidence: f64,
        /// Cycle at which the crisis was detected.
        detected_at_cycle: u64,
    },
    /// Submit a robotics telemetry report to the robotics-dispatch zome.
    ///
    /// Field-for-field mirror of `symthaea-mycelix-conductor`'s
    /// `DispatchCommand::SubmitRoboticsTelemetry` (the crate with the built
    /// `submit_telemetry` zome-call handler). Only emitted when a
    /// `RoboticsDispatchBinding` (asset + order hashes from on-chain
    /// registration) AND a `RoboticsMissionStatus` (position/progress/fuel —
    /// signals the cognitive loop itself has no source for) are both set;
    /// never emitted with zeroed placeholders.
    SubmitRoboticsTelemetry {
        correlation_id: u64,
        /// ActionHash of the registered RoboticAsset (raw 39-byte hash).
        asset_hash: Vec<u8>,
        /// ActionHash of the active DispatchOrder (raw 39-byte hash).
        order_hash: Vec<u8>,
        /// Current position (WGS84 lat/lon, meters altitude).
        lat: f64,
        lon: f64,
        alt: f64,
        /// Current Phi / consciousness level.
        consciousness_level: f64,
        /// Safety tier string — "Green"/"Yellow"/"Orange"/"Red".
        safety_level: String,
        /// Mission progress 0.0–1.0.
        mission_progress: f64,
        /// Fuel/battery level 0.0–1.0.
        fuel_level: f64,
        /// Platform name (e.g., "helicopter").
        platform: String,
        /// Platform-specific serialized telemetry bytes (opaque to the zome).
        platform_specific: Vec<u8>,
    },
}

/// On-chain identity binding for robotics telemetry: the ActionHashes minted
/// by `register_asset` + `dispatch_mission` on the robotics-dispatch zome.
/// The cognitive loop cannot invent these — the host application sets them
/// after registration.
#[cfg(feature = "mycelix")]
#[derive(Debug, Clone)]
pub struct RoboticsDispatchBinding {
    pub asset_hash: Vec<u8>,
    pub order_hash: Vec<u8>,
}

/// Mission-level status signals a telemetry report requires but the
/// cognitive loop has no internal source for (GPS fix, mission driver,
/// battery monitor). Updated by the host application; telemetry emission is
/// gated on this being present rather than shipping zeros.
#[cfg(feature = "mycelix")]
#[derive(Debug, Clone, Copy)]
pub struct RoboticsMissionStatus {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    /// Mission progress 0.0–1.0.
    pub mission_progress: f64,
    /// Fuel/battery level 0.0–1.0.
    pub fuel_level: f64,
}

/// Outcome received from the conductor confirming or rejecting a dispatched command.
///
/// The conductor bridge should send these back via the governance event channel
/// after processing each `GovernanceDispatchCommand`.
#[cfg(feature = "mycelix")]
#[derive(Debug, Clone)]
pub enum GovernanceDispatchOutcome {
    /// Proposal was accepted by the conductor and written to the DHT.
    ProposalAccepted {
        correlation_id: u64,
        /// The proposal action hash from Holochain, if available.
        action_hash: Option<String>,
    },
    /// Proposal was rejected by the conductor.
    ProposalRejected { correlation_id: u64, reason: String },
    /// Vote was accepted by the conductor and written to the DHT.
    VoteAccepted {
        correlation_id: u64,
        action_hash: Option<String>,
    },
    /// Vote was rejected by the conductor.
    VoteRejected { correlation_id: u64, reason: String },
}

/// Waste/circular economy events dispatched to or received from Mycelix DHT.
///
/// These bridge symthaea-circular AI classifications to Holochain waste-registry
/// entries, closing the AI → DHT → feedback loop.
#[cfg(feature = "circular")]
#[derive(Debug, Clone)]
pub enum WasteBridgeEvent {
    /// AI classified a waste stream — publish as WasteClassification entry.
    ClassificationResult {
        /// Waste stream hash on DHT (if known).
        stream_hash: Option<String>,
        /// Determined category (e.g. "Organic", "PlasticPET").
        category: String,
        /// Classification confidence [0.0, 1.0].
        confidence: f32,
        /// Method used (always "VisionAI" for Symthaea).
        method: String,
    },
    /// Contamination detected — emit alert to source and facility.
    ContaminationAlert {
        /// Facility identifier.
        facility_id: String,
        /// Contaminant type detected.
        contaminant: String,
        /// Severity [0.0, 1.0].
        severity: f32,
    },
    /// Decomposition prediction for a compost batch.
    DecompositionPrediction {
        /// Batch identifier.
        batch_id: String,
        /// Predicted completion timestamp (Unix microseconds).
        predicted_completion_us: u64,
        /// Decomposition percentage at prediction time.
        decomposition_pct: f32,
    },
}

impl MycelixBridge {
    /// Create a new Mycelix bridge
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            evaluator: UnifiedValueEvaluator::new(),
            config: BridgeConfig::default(),
            pending_updates: Vec::new(),
            last_sync: Instant::now(),
            agent_id: agent_id.into(),
            proposal_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(60),
            pending_factcheck_feedback: Vec::new(),
            #[cfg(feature = "circular")]
            pending_waste_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_outcomes: Vec::new(),
            #[cfg(feature = "mycelix")]
            governance_dispatch_tx: None,
            #[cfg(feature = "mycelix")]
            pending_confirmations: HashMap::new(),
            #[cfg(feature = "mycelix")]
            next_correlation_id: 1,
            #[cfg(feature = "mycelix")]
            robotics_binding: None,
            #[cfg(feature = "mycelix")]
            robotics_status: None,
            #[cfg(feature = "mycelix")]
            last_robotics_dispatch: None,
            #[cfg(feature = "mycelix_sdk")]
            hyperfeel_encoder: HyperFeelEncoder::new(EncodingConfig::default()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(agent_id: impl Into<String>, config: BridgeConfig) -> Self {
        Self {
            evaluator: UnifiedValueEvaluator::new(),
            config,
            pending_updates: Vec::new(),
            last_sync: Instant::now(),
            agent_id: agent_id.into(),
            proposal_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(60),
            pending_factcheck_feedback: Vec::new(),
            #[cfg(feature = "circular")]
            pending_waste_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_outcomes: Vec::new(),
            #[cfg(feature = "mycelix")]
            governance_dispatch_tx: None,
            #[cfg(feature = "mycelix")]
            pending_confirmations: HashMap::new(),
            #[cfg(feature = "mycelix")]
            next_correlation_id: 1,
            #[cfg(feature = "mycelix")]
            robotics_binding: None,
            #[cfg(feature = "mycelix")]
            robotics_status: None,
            #[cfg(feature = "mycelix")]
            last_robotics_dispatch: None,
            #[cfg(feature = "mycelix_sdk")]
            hyperfeel_encoder: HyperFeelEncoder::new(EncodingConfig::default()),
        }
    }

    /// Set the governance dispatch channel for real Holochain connectivity.
    ///
    /// The receiver end should be owned by the `symthaea-mycelix-holochain`
    /// conductor bridge, which translates commands into real zome calls.
    /// Without this channel, proposals and votes are evaluated locally
    /// but not submitted on-chain.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (tx, rx) = MycelixBridge::create_governance_channel();
    /// bridge.set_governance_dispatch_tx(tx);
    /// // In a separate async task:
    /// // conductor_bridge::run_dispatch_loop(rx, conductor).await;
    /// ```
    #[cfg(feature = "mycelix")]
    pub fn set_governance_dispatch_tx(
        &mut self,
        tx: std::sync::mpsc::SyncSender<GovernanceDispatchCommand>,
    ) {
        self.governance_dispatch_tx = Some(tx);
    }

    /// Create the bounded dispatch channel pair for Holochain governance connectivity.
    ///
    /// Returns `(SyncSender, Receiver)` — caller gives the sender to this bridge
    /// via `set_governance_dispatch_tx()` and passes the receiver to the
    /// conductor bridge process. Bounded to 64 commands to provide backpressure
    /// and prevent unbounded memory growth under load.
    #[cfg(feature = "mycelix")]
    pub fn create_governance_channel() -> (
        std::sync::mpsc::SyncSender<GovernanceDispatchCommand>,
        std::sync::mpsc::Receiver<GovernanceDispatchCommand>,
    ) {
        std::sync::mpsc::sync_channel(64)
    }

    // ========================================================================
    // FACTCHECK EPISTEMIC FEEDBACK (Phase 4)
    // ========================================================================

    /// Submit a factcheck result as epistemic feedback for the cognitive loop.
    ///
    /// The feedback is queued and will be injected into the consciousness cube
    /// on the next cognitive cycle via `drain_factcheck_feedback()`.
    ///
    /// This is the inward direction of the Mycelix factcheck loop:
    /// `fact_check()` → `FactcheckEpistemicFeedback` → `inject_epistemic_cube()`
    pub fn submit_factcheck_feedback(&mut self, feedback: FactcheckEpistemicFeedback) {
        self.pending_factcheck_feedback.push(feedback);
    }

    /// Submit a factcheck by verdict string (convenience method).
    pub fn submit_factcheck_verdict(&mut self, statement: &str, verdict: &str, confidence: f64) {
        let feedback =
            FactcheckEpistemicFeedback::from_verdict_string(statement, verdict, confidence);
        self.submit_factcheck_feedback(feedback);
    }

    /// Drain pending factcheck feedback for injection into the cognitive loop.
    ///
    /// Called by the cognitive loop each cycle to pick up any queued factcheck
    /// results and update the epistemic cube accordingly.
    pub fn drain_factcheck_feedback(&mut self) -> Vec<FactcheckEpistemicFeedback> {
        std::mem::take(&mut self.pending_factcheck_feedback)
    }

    /// Check if there are pending factcheck results.
    pub fn has_pending_factcheck(&self) -> bool {
        !self.pending_factcheck_feedback.is_empty()
    }

    // ========================================================================
    // WASTE BRIDGE EVENTS
    // ========================================================================

    /// Submit a waste classification result for dispatch to Mycelix DHT.
    #[cfg(feature = "circular")]
    pub fn submit_waste_event(&mut self, event: WasteBridgeEvent) {
        self.pending_waste_events.push(event);
    }

    /// Drain pending waste events for dispatch to the Holochain conductor.
    #[cfg(feature = "circular")]
    pub fn drain_waste_events(&mut self) -> Vec<WasteBridgeEvent> {
        std::mem::take(&mut self.pending_waste_events)
    }

    /// Check if there are pending waste events.
    #[cfg(feature = "circular")]
    pub fn has_pending_waste_events(&self) -> bool {
        !self.pending_waste_events.is_empty()
    }

    // ========================================================================
    // PROPOSAL SUBMISSION
    // ========================================================================

    /// Submit a proposal with consciousness validation
    pub fn submit_proposal(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
    ) -> Result<SubmissionResult, BridgeError> {
        // 1. Check consciousness level
        let action_type = proposal.proposal_type.to_action_type();
        if !consciousness.is_adequate_for(action_type) {
            return Err(BridgeError::InsufficientConsciousness {
                current: consciousness.phi,
                required: proposal.required_phi,
                action: format!("submit {:?} proposal", proposal.proposal_type),
            });
        }

        // 2. Build evaluation context
        let context = EvaluationContext {
            consciousness_level: consciousness.phi,
            affective_state: CoreAffect::neutral(), // Use snapshot valence
            affective_systems: affective_state.clone(),
            action_type,
            action_domain: None,   // Auto-detect from proposal description
            involves_others: true, // Proposals always affect others
        };

        // 3. Evaluate proposal against values
        let eval_result = self.evaluator.evaluate(&proposal.description, context);

        // 4. Check for veto
        match &eval_result.decision {
            Decision::Veto(reason) => {
                return Err(BridgeError::ValueViolation {
                    reason: format!("{reason:?}"),
                });
            }
            Decision::Warn(_warnings) => {
                // Log warnings but allow
                // In production, might require confirmation
            }
            Decision::Allow => {}
        }

        // 5. Create submission result
        let alignment = self.create_alignment_result(&eval_result);

        // 6. Dispatch to Holochain conductor if channel is connected
        #[cfg(feature = "mycelix")]
        {
            let mut disconnected = false;
            if let Some(ref tx) = self.governance_dispatch_tx {
                let cid = self.next_correlation_id;
                self.next_correlation_id += 1;
                match tx.try_send(GovernanceDispatchCommand::SubmitProposal {
                    correlation_id: cid,
                    description: proposal.description.clone(),
                    proposer_did: self.agent_id.clone(),
                    consciousness_phi: consciousness.phi,
                    meta_awareness: consciousness.meta_awareness,
                    coherence: consciousness.coherence,
                    care_activation: consciousness.care_activation,
                    alignment_score: alignment.overall_score,
                }) {
                    Ok(()) => {
                        self.pending_confirmations.insert(cid, Instant::now());
                    }
                    Err(std::sync::mpsc::TrySendError::Full(_)) => {
                        tracing::warn!(
                            "Governance dispatch channel full (64) — proposal queued locally only"
                        );
                    }
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                        tracing::warn!(
                            "Governance dispatch channel disconnected — proposal queued locally only"
                        );
                        disconnected = true;
                    }
                }
            }
            if disconnected {
                self.governance_dispatch_tx = None;
            }
        }

        Ok(SubmissionResult {
            proposal_id: proposal.id.clone(),
            consciousness: consciousness.clone(),
            alignment,
            submitted_at: now_secs(),
            success: true,
        })
    }

    // ========================================================================
    // VOTING
    // ========================================================================

    /// Evaluate a proposal for voting
    pub fn evaluate_proposal(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
    ) -> Result<ValueAlignmentResult, BridgeError> {
        // Check cache first
        if let Some((cached, time)) = self.proposal_cache.get(&proposal.id) {
            if time.elapsed() < self.cache_ttl {
                return Ok(cached.clone());
            }
        }

        // Check consciousness level
        if consciousness.phi < self.config.min_phi_voting {
            return Err(BridgeError::InsufficientConsciousness {
                current: consciousness.phi,
                required: self.config.min_phi_voting,
                action: "evaluate proposal".to_string(),
            });
        }

        // Evaluate
        let context = EvaluationContext {
            consciousness_level: consciousness.phi,
            affective_state: CoreAffect::neutral(),
            affective_systems: affective_state,
            action_type: ActionType::Voting,
            action_domain: None, // Auto-detect from proposal
            involves_others: true,
        };

        let eval = self.evaluator.evaluate(&proposal.description, context);
        let alignment = self.create_alignment_result(&eval);

        // Cache result
        self.proposal_cache
            .insert(proposal.id.clone(), (alignment.clone(), Instant::now()));

        Ok(alignment)
    }

    /// Cast a vote on a proposal
    pub fn cast_vote(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
    ) -> Result<Vote, BridgeError> {
        // Evaluate proposal
        let alignment = self.evaluate_proposal(proposal, consciousness.clone(), affective_state)?;

        // Determine vote value from recommendation
        let value = match alignment.recommendation {
            GovernanceRecommendation::StrongSupport => VoteValue::StrongYes,
            GovernanceRecommendation::Support => VoteValue::Yes,
            GovernanceRecommendation::Neutral => VoteValue::Abstain,
            GovernanceRecommendation::Oppose => VoteValue::No,
            GovernanceRecommendation::StrongOppose => VoteValue::StrongNo,
            GovernanceRecommendation::CannotEvaluate => {
                return Err(BridgeError::CannotEvaluate);
            }
        };

        // Dispatch to Holochain conductor if channel is connected
        #[cfg(feature = "mycelix")]
        {
            let mut disconnected = false;
            if let Some(ref tx) = self.governance_dispatch_tx {
                let approve = matches!(value, VoteValue::Yes | VoteValue::StrongYes);
                let cid = self.next_correlation_id;
                self.next_correlation_id += 1;
                match tx.try_send(GovernanceDispatchCommand::CastVote {
                    correlation_id: cid,
                    proposal_id: proposal.id.clone(),
                    voter_did: self.agent_id.clone(),
                    approve,
                    rationale: format!(
                        "{:?}: score {:.2}",
                        alignment.recommendation, alignment.overall_score
                    ),
                    consciousness_phi: consciousness.phi,
                    meta_awareness: consciousness.meta_awareness,
                    coherence: consciousness.coherence,
                    care_activation: consciousness.care_activation,
                }) {
                    Ok(()) => {
                        self.pending_confirmations.insert(cid, Instant::now());
                    }
                    Err(std::sync::mpsc::TrySendError::Full(_)) => {
                        tracing::warn!(
                            "Governance dispatch channel full (64) — vote queued locally only"
                        );
                    }
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                        tracing::warn!(
                            "Governance dispatch channel disconnected — vote queued locally only"
                        );
                        disconnected = true;
                    }
                }
            }
            if disconnected {
                self.governance_dispatch_tx = None;
            }
        }

        Ok(Vote {
            proposal_id: proposal.id.clone(),
            voter: self.agent_id.clone(),
            value,
            consciousness,
            alignment,
            timestamp: now_secs(),
        })
    }

    // ========================================================================
    // FEDERATED VALUE LEARNING
    // ========================================================================

    /// Record a value learning event
    pub fn record_value_learning(
        &mut self,
        harmony: Harmony,
        importance_delta: f64,
        affirmation: bool,
        context: &str,
        phi: f64,
    ) {
        if !self.config.enable_federated_learning {
            return;
        }

        let update = ValueLearningUpdate {
            agent_id: self.agent_id.clone(),
            harmony: harmony.name().to_string(),
            importance_delta,
            affirmation_delta: if affirmation { 1 } else { 0 },
            context: context.to_string(),
            phi_at_learning: phi,
            timestamp: now_secs(),
        };

        self.pending_updates.push(update);

        // Check if we should sync
        if self.pending_updates.len() >= self.config.fl_batch_size
            || self.last_sync.elapsed().as_secs() >= self.config.fl_sync_interval_secs
        {
            // In real implementation, would call Mycelix network
            self.flush_learning_updates();
        }
    }

    /// Flush pending learning updates to network
    pub fn flush_learning_updates(&mut self) -> Vec<ValueLearningUpdate> {
        let updates = std::mem::take(&mut self.pending_updates);
        self.last_sync = Instant::now();

        // In real implementation:
        // 1. Compress updates using HyperFeel
        // 2. Sign with zkProof for provenance
        // 3. Submit to Mycelix network via Bridge
        // 4. MATL validates gradient quality

        updates
    }

    /// Apply value learning from network
    pub fn apply_network_learning(&mut self, updates: Vec<ValueLearningUpdate>) {
        // In real implementation:
        // 1. Verify updates with MATL (45% Byzantine tolerance)
        // 2. Weight by sender reputation
        // 3. Apply to local value system

        for update in updates {
            // Only apply if phi was sufficient when learning occurred
            if update.phi_at_learning >= self.config.min_phi_voting {
                // Apply update to local harmonies
                // (Would call self.evaluator.harmonies.adjust_importance(...)
            }
        }
    }

    /// Create compressed gradient for efficient transmission.
    ///
    /// When the `mycelix_sdk` feature is enabled, this encodes pending value
    /// updates into a 2KB HyperGradient via Johnson-Lindenstrauss projection.
    /// Without the SDK, returns a zero placeholder (backward compatible).
    pub fn create_compressed_gradient(&mut self, round: u64) -> CompressedValueGradient {
        #[cfg(feature = "mycelix_sdk")]
        {
            use super::eight_harmonies::Harmony;

            // Build a gradient vector from pending updates: 8 harmonies × 10 dimensions.
            let harmony_names: &[&str] = &[
                "CommunalVoice",
                "EcologicalWisdom",
                "IntergenerationalStewardship",
                "RadicalCompassion",
                "SovereignInterdependence",
                "TransparentAccountability",
                "EvolutionaryProgression",
                "SacredStillness",
            ];
            let mut gradient = vec![0.0f32; harmony_names.len() * 10];

            for update in &self.pending_updates {
                if let Some(idx) = harmony_names.iter().position(|&h| h == update.harmony) {
                    let base_idx = idx * 10;
                    gradient[base_idx] += update.importance_delta as f32;
                    if base_idx + 1 < gradient.len() {
                        gradient[base_idx + 1] += update.affirmation_delta as f32 * 0.1;
                    }
                    if base_idx + 2 < gradient.len() {
                        gradient[base_idx + 2] += update.phi_at_learning as f32 * 0.5;
                    }
                }
            }

            let hyper_gradient =
                self.hyperfeel_encoder
                    .encode_gradient(&gradient, round as u32, &self.agent_id);
            CompressedValueGradient {
                harmony_encoding: hyper_gradient.hypervector.clone(),
                importance_gradient: hyper_gradient.gradient_hash.to_vec(),
                round,
                agent_id: self.agent_id.clone(),
                compression_ratio: hyper_gradient.compression_ratio as f64,
            }
        }
        #[cfg(not(feature = "mycelix_sdk"))]
        {
            CompressedValueGradient {
                harmony_encoding: vec![0u8; 256],
                importance_gradient: vec![0u8; 256],
                round,
                agent_id: self.agent_id.clone(),
                compression_ratio: 2000.0,
            }
        }
    }

    // ========================================================================
    // ASSET EVALUATION
    // ========================================================================

    /// Evaluate a regenerative asset against the Eight Harmonies.
    ///
    /// This is the bridge method that connects Symthaea consciousness scoring
    /// to the Mycelix energy cluster. It wraps `AssetEvaluator::evaluate()`
    /// for convenience when you already have a `MycelixBridge` instance.
    /// Evaluate a regenerative asset against the Eight Harmonies.
    pub fn evaluate_asset(
        &self,
        metadata: &super::asset_evaluator::AssetMetadata,
        consciousness: &ConsciousnessSnapshot,
    ) -> super::asset_evaluator::AssetConsciousnessScore {
        let mut evaluator = super::asset_evaluator::AssetEvaluator::new();
        evaluator.evaluate(metadata, consciousness)
    }

    /// Evaluate an asset AND dispatch the result to the Holochain conductor.
    ///
    /// This is the full pipeline: evaluate → serialize → dispatch.
    /// The conductor bridge will call `record_consciousness_assessment()`
    /// on the energy bridge zome to store the result on-chain.
    #[cfg(feature = "mycelix")]
    pub fn evaluate_and_dispatch_asset(
        &mut self,
        project_id: &str,
        metadata: &super::asset_evaluator::AssetMetadata,
        consciousness: &ConsciousnessSnapshot,
    ) -> Result<super::asset_evaluator::AssetConsciousnessScore, BridgeError> {
        let score = self.evaluate_asset(metadata, consciousness);

        // Serialize per-harmony scores for on-chain storage
        let per_harmony_json =
            serde_json::to_string(&score.per_harmony).unwrap_or_else(|_| "{}".to_string());

        // Dispatch to conductor
        let mut disconnected = false;
        if let Some(ref tx) = self.governance_dispatch_tx {
            let cid = self.next_correlation_id;
            self.next_correlation_id += 1;

            match tx.try_send(GovernanceDispatchCommand::EvaluateAsset {
                correlation_id: cid,
                project_id: project_id.to_string(),
                description: metadata.description.clone(),
                project_type: metadata.project_type.clone(),
                capacity_mw: metadata.capacity_mw,
                community_did: metadata.community_did.clone(),
                impact_claims: metadata.impact_claims.clone(),
                phi_score: score.phi_score,
                harmony_alignment: score.harmony_alignment,
                per_harmony_scores: per_harmony_json,
                care_activation: score.care_activation,
                meta_awareness: score.meta_awareness,
            }) {
                Ok(()) => {
                    self.pending_confirmations.insert(cid, Instant::now());
                    tracing::info!(
                        project_id,
                        phi = score.phi_score,
                        harmony = score.harmony_alignment,
                        "Asset evaluation dispatched to conductor"
                    );
                }
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    tracing::warn!("Governance dispatch channel full — evaluation not dispatched");
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    tracing::warn!("Governance dispatch channel disconnected");
                    disconnected = true;
                }
            }
        }
        if disconnected {
            self.governance_dispatch_tx = None;
        }

        Ok(score)
    }

    // ========================================================================
    // CRISIS DISPATCH
    // ========================================================================

    /// Forward a civic crisis event to the Mycelix emergency-incidents zome.
    ///
    /// Uses the same `governance_dispatch_tx` channel as proposals and votes.
    /// The conductor bridge maps `DeclareCrisis` to an
    /// `emergency_incidents::declare_disaster` zome call on the civic role.
    /// Note: until Symthaea produces geospatial fields, the conductor adapter
    /// publishes a transparent placeholder affected_area ("global/unknown").
    #[cfg(feature = "mycelix")]
    pub fn dispatch_crisis(
        &mut self,
        event: &super::super::cognitive_loop::civic_crisis_detector::CivicCrisisEvent,
    ) {
        let mut disconnected = false;
        if let Some(ref tx) = self.governance_dispatch_tx {
            let cid = self.next_correlation_id;
            self.next_correlation_id += 1;
            match tx.try_send(GovernanceDispatchCommand::DeclareCrisis {
                correlation_id: cid,
                severity: event.severity,
                crisis_type: event.crisis_type.to_string(),
                description: event.description.clone(),
                confidence: event.confidence,
                detected_at_cycle: event.detected_at_cycle,
            }) {
                Ok(()) => {
                    self.pending_confirmations.insert(cid, Instant::now());
                    tracing::info!(
                        severity = event.severity,
                        crisis_type = %event.crisis_type,
                        confidence = event.confidence,
                        "Crisis event dispatched to Mycelix civic bridge"
                    );
                }
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    tracing::warn!(
                        "Governance dispatch channel full — crisis event not dispatched"
                    );
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    tracing::warn!(
                        "Governance dispatch channel disconnected — crisis event not dispatched"
                    );
                    disconnected = true;
                }
            }
        }
        if disconnected {
            self.governance_dispatch_tx = None;
        }
    }

    /// Bind this agent to an on-chain robotics asset + dispatch order.
    /// Telemetry is only emitted while a binding is present.
    #[cfg(feature = "mycelix")]
    pub fn set_robotics_binding(&mut self, binding: RoboticsDispatchBinding) {
        self.robotics_binding = Some(binding);
    }

    /// Clear the robotics binding (mission complete / asset recalled).
    #[cfg(feature = "mycelix")]
    pub fn clear_robotics_binding(&mut self) {
        self.robotics_binding = None;
    }

    /// Update the host-supplied mission status (position, progress, fuel).
    /// Telemetry is only emitted while a status is present — the cognitive
    /// loop has no GPS/mission/battery source of its own, and shipping
    /// zeroed placeholders on-chain would be worse than silence.
    #[cfg(feature = "mycelix")]
    pub fn update_robotics_mission_status(&mut self, status: RoboticsMissionStatus) {
        self.robotics_status = Some(status);
    }

    /// Minimum interval between robotics telemetry dispatches. The cognitive
    /// loop runs ~31 Hz; one DHT entry per cycle would be abusive.
    #[cfg(feature = "mycelix")]
    const ROBOTICS_TELEMETRY_MIN_INTERVAL: Duration = Duration::from_secs(5);

    /// Dispatch an embodiment telemetry report to the robotics-dispatch zome.
    ///
    /// This is the drain the 2026-07-06 robotics review found missing: the
    /// loop populated `sensorimotor.embodiment_telemetry` every cycle while
    /// `SubmitRoboticsTelemetry` was constructed only in conductor tests.
    /// Call from the host between cycles (see
    /// `CognitiveLoopService::poll_bridge_robotics_telemetry`).
    ///
    /// Returns `true` if a report was dispatched; `false` when unbound,
    /// missing mission status, rate-limited, or the channel is unavailable.
    #[cfg(feature = "mycelix")]
    pub fn dispatch_robotics_telemetry(
        &mut self,
        telemetry: &symthaea_core::embodiment::EmbodimentTelemetry,
        consciousness_level: f64,
    ) -> bool {
        let Some(binding) = self.robotics_binding.clone() else {
            return false;
        };
        let Some(status) = self.robotics_status else {
            return false;
        };
        if let Some(last) = self.last_robotics_dispatch
            && last.elapsed() < Self::ROBOTICS_TELEMETRY_MIN_INTERVAL
        {
            return false;
        }
        let mut dispatched = false;
        let mut disconnected = false;
        if let Some(ref tx) = self.governance_dispatch_tx {
            let cid = self.next_correlation_id;
            self.next_correlation_id += 1;
            match tx.try_send(GovernanceDispatchCommand::SubmitRoboticsTelemetry {
                correlation_id: cid,
                asset_hash: binding.asset_hash,
                order_hash: binding.order_hash,
                lat: status.lat,
                lon: status.lon,
                alt: status.alt,
                consciousness_level,
                // SubmitRoboticsTelemetry's wire contract is String
                // ("Green"/"Yellow"/"Orange"/"Red"); EmbodimentTelemetry's
                // safety_level became a real MotorSafetyLevel 2026-07-12 —
                // Display renders the identical strings.
                safety_level: telemetry.safety_level.to_string(),
                mission_progress: status.mission_progress,
                fuel_level: status.fuel_level,
                platform: telemetry.platform.clone(),
                platform_specific: telemetry.platform_specific.clone(),
            }) {
                Ok(()) => {
                    self.pending_confirmations.insert(cid, Instant::now());
                    self.last_robotics_dispatch = Some(Instant::now());
                    dispatched = true;
                    tracing::debug!(
                        platform = %telemetry.platform,
                        safety = %telemetry.safety_level,
                        phi = consciousness_level,
                        "Robotics telemetry dispatched to Mycelix robotics-dispatch"
                    );
                }
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    tracing::warn!(
                        "Governance dispatch channel full — robotics telemetry not dispatched"
                    );
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    tracing::warn!(
                        "Governance dispatch channel disconnected — robotics telemetry not dispatched"
                    );
                    disconnected = true;
                }
            }
        }
        if disconnected {
            self.governance_dispatch_tx = None;
        }
        dispatched
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    /// Create alignment result from evaluation
    fn create_alignment_result(&self, eval: &EvaluationResult) -> ValueAlignmentResult {
        let harmony_scores: HashMap<String, f64> =
            eval.breakdown.harmony_scores.iter().cloned().collect();

        let violations: Vec<String> = eval
            .harmony_alignment
            .alignments
            .iter()
            .filter(|(_, a)| a.score < -0.2)
            .map(|(h, _)| h.name().to_string())
            .collect();

        let recommendation = self.score_to_recommendation(
            eval.harmony_alignment.overall_score,
            eval.authenticity,
            !violations.is_empty(),
        );

        ValueAlignmentResult {
            overall_score: eval.harmony_alignment.overall_score,
            harmony_scores,
            violations,
            authenticity: eval.authenticity,
            recommendation,
        }
    }

    /// Convert scores to recommendation
    fn score_to_recommendation(
        &self,
        alignment: f64,
        authenticity: f64,
        has_violations: bool,
    ) -> GovernanceRecommendation {
        if has_violations {
            return GovernanceRecommendation::StrongOppose;
        }

        let combined = alignment * 0.6 + authenticity * 0.4;

        if combined > 0.7 {
            GovernanceRecommendation::StrongSupport
        } else if combined > 0.3 {
            GovernanceRecommendation::Support
        } else if combined > -0.3 {
            GovernanceRecommendation::Neutral
        } else if combined > -0.7 {
            GovernanceRecommendation::Oppose
        } else {
            GovernanceRecommendation::StrongOppose
        }
    }

    // ========================================================================
    // GOVERNANCE EVENT GENERATION (Phase 1-2 bridge wiring)
    // ========================================================================

    /// Record a governance outcome from a completed tally.
    ///
    /// Creates a [`GovernanceOutcome`] and a [`GovernanceEvent::TallyCompleted`],
    /// ready for injection into the CognitiveLoopService's GovernanceManager.
    /// The caller (Symthaea facade or network handler) is responsible for calling
    /// `cls.inject_governance_event()` and `cls.inject_governance_outcome()`.
    #[cfg(feature = "mycelix")]
    pub fn record_governance_outcome(
        &mut self,
        proposal_id: &str,
        passed: bool,
        collective_phi: f64,
        my_vote: Option<&Vote>,
    ) -> (
        crate::cognitive_loop::managers::governance_manager::GovernanceEvent,
        crate::cognitive_loop::managers::governance_manager::GovernanceOutcome,
    ) {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind, GovernanceOutcome,
        };

        let event = GovernanceEvent {
            kind: GovernanceEventKind::TallyCompleted {
                passed,
                collective_phi,
            },
            proposal_id: Some(proposal_id.to_string()),
            timestamp_secs: now_secs(),
        };

        // Determine vote alignment
        let my_vote_aligned = my_vote.map(|v| {
            let voted_yes = matches!(v.value, VoteValue::StrongYes | VoteValue::Yes);
            (voted_yes && passed) || (!voted_yes && !passed)
        });

        let value_alignment_score = my_vote.map(|v| v.alignment.overall_score).unwrap_or(0.5);

        let harmonic_resonance = my_vote.map(|v| v.alignment.authenticity).unwrap_or(0.5);

        let outcome = GovernanceOutcome {
            proposal_id: proposal_id.to_string(),
            passed,
            my_vote_aligned,
            value_alignment_score,
            harmonic_resonance,
        };

        // Queue for CLS injection
        self.pending_gov_events.push(event.clone());
        self.pending_gov_outcomes.push(outcome.clone());

        (event, outcome)
    }

    /// Create a governance event from a vote cast.
    #[cfg(feature = "mycelix")]
    pub fn create_vote_event(
        &mut self,
        vote: &Vote,
        voter_phi: f64,
    ) -> crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind,
        };

        let vote_value = match vote.value {
            VoteValue::StrongYes => 1.0,
            VoteValue::Yes => 0.5,
            VoteValue::Abstain => 0.0,
            VoteValue::No => -0.5,
            VoteValue::StrongNo => -1.0,
        };

        let event = GovernanceEvent {
            kind: GovernanceEventKind::VoteCast {
                voter_phi,
                vote_value,
            },
            proposal_id: Some(vote.proposal_id.clone()),
            timestamp_secs: now_secs(),
        };
        self.pending_gov_events.push(event.clone());
        event
    }

    /// Create a governance event for an emergency declaration.
    #[cfg(feature = "mycelix")]
    pub fn create_emergency_event(
        &mut self,
    ) -> crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind,
        };
        let event = GovernanceEvent {
            kind: GovernanceEventKind::EmergencyDeclared,
            proposal_id: None,
            timestamp_secs: now_secs(),
        };
        self.pending_gov_events.push(event.clone());
        event
    }

    /// Create a governance event for a reciprocity pledge.
    #[cfg(feature = "mycelix")]
    pub fn create_reciprocity_event(
        &mut self,
        amount: f64,
    ) -> crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind,
        };
        let event = GovernanceEvent {
            kind: GovernanceEventKind::ReciprocityPledge { amount },
            proposal_id: None,
            timestamp_secs: now_secs(),
        };
        self.pending_gov_events.push(event.clone());
        event
    }

    /// Create a governance event for a reputation change.
    #[cfg(feature = "mycelix")]
    pub fn create_reputation_event(
        &mut self,
        delta: f64,
    ) -> crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind,
        };
        let event = GovernanceEvent {
            kind: GovernanceEventKind::ReputationChanged { delta },
            proposal_id: None,
            timestamp_secs: now_secs(),
        };
        self.pending_gov_events.push(event.clone());
        event
    }

    /// Create a governance event for a justice dispute.
    #[cfg(feature = "mycelix")]
    pub fn create_dispute_event(
        &mut self,
        involves_self: bool,
    ) -> crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
        use crate::cognitive_loop::managers::governance_manager::{
            GovernanceEvent, GovernanceEventKind,
        };
        let event = GovernanceEvent {
            kind: GovernanceEventKind::JusticeDispute { involves_self },
            proposal_id: None,
            timestamp_secs: now_secs(),
        };
        self.pending_gov_events.push(event.clone());
        event
    }

    // ========================================================================
    // EVENT QUEUE (Bridge → CLS injection)
    // ========================================================================

    /// Push an arbitrary governance event into the bridge queue.
    #[cfg(feature = "mycelix")]
    pub fn push_governance_event(
        &mut self,
        event: crate::cognitive_loop::managers::governance_manager::GovernanceEvent,
    ) {
        self.pending_gov_events.push(event);
    }

    /// Drain all pending governance events and outcomes.
    /// Returns `(events, outcomes)` — caller injects them into CLS.
    #[cfg(feature = "mycelix")]
    pub fn drain_pending_governance(
        &mut self,
    ) -> (
        Vec<crate::cognitive_loop::managers::governance_manager::GovernanceEvent>,
        Vec<crate::cognitive_loop::managers::governance_manager::GovernanceOutcome>,
    ) {
        (
            std::mem::take(&mut self.pending_gov_events),
            std::mem::take(&mut self.pending_gov_outcomes),
        )
    }

    // ========================================================================
    // DISPATCH CONFIRMATION TRACKING
    // ========================================================================

    /// Process a dispatch outcome from the conductor, removing the matching
    /// correlation ID from pending confirmations.
    ///
    /// Should be called when the conductor bridge sends back confirmation
    /// of a dispatched command. Logs a warning if the oldest unconfirmed
    /// dispatch exceeds 60 seconds.
    #[cfg(feature = "mycelix")]
    pub fn confirm_dispatch(&mut self, outcome: &GovernanceDispatchOutcome) {
        let cid = match outcome {
            GovernanceDispatchOutcome::ProposalAccepted { correlation_id, .. }
            | GovernanceDispatchOutcome::ProposalRejected { correlation_id, .. }
            | GovernanceDispatchOutcome::VoteAccepted { correlation_id, .. }
            | GovernanceDispatchOutcome::VoteRejected { correlation_id, .. } => *correlation_id,
        };
        self.pending_confirmations.remove(&cid);

        // Check for stale unconfirmed dispatches
        if let Some(age) = self.oldest_unconfirmed_age() {
            if age > Duration::from_secs(60) {
                tracing::warn!(
                    unconfirmed = self.pending_confirmations.len(),
                    oldest_secs = age.as_secs(),
                    "Oldest unconfirmed governance dispatch exceeds 60s — conductor may be unresponsive"
                );
            }
        }
    }

    /// Number of dispatched commands awaiting conductor confirmation.
    #[cfg(feature = "mycelix")]
    pub fn unconfirmed_dispatches(&self) -> usize {
        self.pending_confirmations.len()
    }

    /// Age of the oldest unconfirmed dispatch, or `None` if all are confirmed.
    #[cfg(feature = "mycelix")]
    pub fn oldest_unconfirmed_age(&self) -> Option<Duration> {
        self.pending_confirmations
            .values()
            .map(|t| t.elapsed())
            .max()
    }

    // ========================================================================
    // EPISTEMIC GATING (Phase 4)
    // ========================================================================

    /// Check if a proposal should be escalated based on epistemic blind spots.
    ///
    /// If the proposal touches domains where the community has collective
    /// blind spots, the required consciousness tier is escalated.
    /// Returns `None` if no escalation is needed, or `Some(tier)` with the
    /// minimum required escalation tier.
    #[cfg(feature = "mycelix")]
    pub fn epistemic_escalation_check(
        &self,
        proposal_domains: &[String],
        mesh: &crate::mycelix::epistemic_mesh::EpistemicMesh,
    ) -> Option<crate::mycelix::epistemic_mesh::EscalationTier> {
        mesh.proposal_escalation_required(proposal_domains)
    }

    /// Submit a proposal with epistemic gating.
    ///
    /// Like `submit_proposal()`, but additionally checks the EpistemicMesh
    /// for blind-spot domains. If the proposal touches collective blind spots,
    /// a higher consciousness tier is required.
    #[cfg(feature = "mycelix")]
    pub fn submit_proposal_with_epistemic_gate(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
        proposal_domains: &[String],
        mesh: Option<&crate::mycelix::epistemic_mesh::EpistemicMesh>,
    ) -> Result<SubmissionResult, BridgeError> {
        // Check epistemic escalation first
        if let Some(mesh) = mesh {
            if let Some(tier) = mesh.proposal_escalation_required(proposal_domains) {
                use crate::mycelix::epistemic_mesh::EscalationTier;
                let required_phi = match tier {
                    EscalationTier::Citizen => GOV_VOTING,         // 0.4
                    EscalationTier::Steward => GOV_CONSTITUTIONAL, // 0.6
                    EscalationTier::Guardian => 0.8,               // Emergency-level
                };
                if consciousness.phi < required_phi {
                    return Err(BridgeError::InsufficientConsciousness {
                        current: consciousness.phi,
                        required: required_phi,
                        action: format!(
                            "submit proposal touching blind-spot domains (escalated to {:?})",
                            tier
                        ),
                    });
                }
            }
        }

        // Proceed with normal submission
        self.submit_proposal(proposal, consciousness, affective_state)
    }

    // ========================================================================
    // COLLECTIVE PHI (Phase 3) — extraction helpers
    // ========================================================================

    /// Extract the 6D consciousness vector components from a snapshot.
    ///
    /// Returns `(consciousness_level, meta_awareness, coherence, care_activation,
    ///           harmonic_alignment, epistemic_confidence)` — the six dimensions
    /// expected by `mycelix_bridge_common::collective_phi::AgentConsciousnessVector`.
    ///
    /// The caller (e.g., the Mycelix coordinator zome or a bridge adapter) constructs
    /// the `AgentConsciousnessVector` from these values. This avoids a direct dependency
    /// on `mycelix-bridge-common` from the symthaea crate.
    #[cfg(feature = "mycelix")]
    pub fn extract_consciousness_vector(
        snapshot: &ConsciousnessSnapshot,
    ) -> (f64, f64, f64, f64, f64, f64) {
        (
            snapshot.phi,
            snapshot.meta_awareness,
            snapshot.coherence,
            snapshot.care_activation,
            snapshot.quality_score(),
            snapshot.self_model_accuracy,
        )
    }

    /// Get bridge statistics
    pub fn stats(&self) -> BridgeStats {
        BridgeStats {
            pending_updates: self.pending_updates.len(),
            cached_proposals: self.proposal_cache.len(),
            time_since_sync: self.last_sync.elapsed().as_secs(),
            evaluator_stats: self.evaluator.stats(),
        }
    }
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// Result of proposal submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionResult {
    /// Proposal ID
    pub proposal_id: String,
    /// Consciousness at submission
    pub consciousness: ConsciousnessSnapshot,
    /// Value alignment
    pub alignment: ValueAlignmentResult,
    /// Submission timestamp
    pub submitted_at: u64,
    /// Whether submission succeeded
    pub success: bool,
}

/// Bridge error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeError {
    /// Consciousness level too low
    InsufficientConsciousness {
        current: f64,
        required: f64,
        action: String,
    },
    /// Value violation detected
    ValueViolation { reason: String },
    /// Cannot evaluate (insufficient information)
    CannotEvaluate,
    /// Network error
    NetworkError { message: String },
    /// Invalid proposal
    InvalidProposal { reason: String },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientConsciousness {
                current,
                required,
                action,
            } => {
                write!(
                    f,
                    "Insufficient consciousness for {action}: {current} < {required}"
                )
            }
            Self::ValueViolation { reason } => write!(f, "Value violation: {reason}"),
            Self::CannotEvaluate => write!(f, "Cannot evaluate proposal"),
            Self::NetworkError { message } => write!(f, "Network error: {message}"),
            Self::InvalidProposal { reason } => write!(f, "Invalid proposal: {reason}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Bridge statistics
#[derive(Debug, Clone)]
pub struct BridgeStats {
    pub pending_updates: usize,
    pub cached_proposals: usize,
    pub time_since_sync: u64,
    pub evaluator_stats: super::unified_value_evaluator::EvaluatorStats,
}

// ============================================================================
// UTILITIES
// ============================================================================

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// ENHANCED SDK INTEGRATION (requires `mycelix` feature)
// ============================================================================

/// Enhanced Mycelix Bridge with full SDK integration
///
/// When the `mycelix` feature is enabled, this provides:
/// - Real HyperFeel gradient compression (2000x compression)
/// - LocalBridge for inter-hApp communication
/// - Cross-hApp reputation tracking
/// - MATL trust verification
/// - Epistemic claim classification
#[cfg(feature = "mycelix_sdk")]
pub struct EnhancedMycelixBridge {
    /// Base bridge with value evaluation
    base: MycelixBridge,
    /// Local bridge for inter-hApp communication
    local_bridge: LocalBridge,
    /// HyperFeel encoder for gradient compression
    hyperfeel: HyperFeelEncoder,
    /// Federation round counter
    round: u32,
}

#[cfg(feature = "mycelix_sdk")]
impl EnhancedMycelixBridge {
    /// Create a new enhanced bridge with full SDK integration
    pub fn new(agent_id: impl Into<String>) -> Self {
        let agent = agent_id.into();
        Self {
            base: MycelixBridge::new(&agent),
            local_bridge: LocalBridge::new(),
            hyperfeel: HyperFeelEncoder::new(EncodingConfig::default()),
            round: 0,
        }
    }

    /// Submit a proposal with consciousness validation and network broadcast
    pub fn submit_proposal_networked(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
    ) -> Result<(SubmissionResult, BridgeEvent), BridgeError> {
        // Validate and submit through base bridge
        let result = self
            .base
            .submit_proposal(proposal, consciousness.clone(), affective_state)?;

        // Broadcast to network
        let event_payload = serde_json::to_vec(&result).unwrap_or_default();
        let event = BridgeEvent::new("proposal_submitted", "symthaea", event_payload);
        self.local_bridge.broadcast(event.clone());

        info!(
            "📢 Proposal '{}' submitted and broadcast to network (Φ={:.3})",
            proposal.id, consciousness.phi
        );

        Ok((result, event))
    }

    /// Cast a vote and broadcast to network
    pub fn cast_vote_networked(
        &mut self,
        proposal: &Proposal,
        consciousness: ConsciousnessSnapshot,
        affective_state: AffectiveSystemsState,
    ) -> Result<(Vote, BridgeEvent), BridgeError> {
        let vote = self
            .base
            .cast_vote(proposal, consciousness.clone(), affective_state)?;

        // Broadcast vote event
        let event_payload = serde_json::to_vec(&vote).unwrap_or_default();
        let event = BridgeEvent::new("vote_cast", "symthaea", event_payload);
        self.local_bridge.broadcast(event.clone());

        info!(
            "🗳️ Vote {:?} cast on proposal '{}' (Φ={:.3})",
            vote.value, proposal.id, consciousness.phi
        );

        Ok((vote, event))
    }

    /// Encode value learning updates using HyperFeel compression
    ///
    /// This compresses the value gradient to ~2KB for efficient transmission
    pub fn encode_value_learning(&mut self, phi: f64) -> HyperGradient {
        // Get pending updates and convert to gradient
        let updates = self.base.flush_learning_updates();

        // Convert updates to a gradient vector
        // Each harmony gets one dimension in the gradient
        let harmony_names = [
            "ResonantCoherence",
            "PanSentientFlourishing",
            "IntegralWisdom",
            "InfinitePlay",
            "UniversalInterconnectedness",
            "SacredReciprocity",
            "EvolutionaryProgression",
        ];

        let mut gradient = vec![0.0f32; harmony_names.len() * 10]; // Expand for detail

        for update in &updates {
            // Find harmony index
            if let Some(idx) = harmony_names.iter().position(|&h| h == update.harmony) {
                let base_idx = idx * 10;
                // Distribute update across gradient dimensions
                gradient[base_idx] += update.importance_delta as f32;
                gradient[base_idx + 1] += update.affirmation_delta as f32 * 0.1;
                gradient[base_idx + 2] += update.phi_at_learning as f32 * 0.5;
            }
        }

        // Encode using HyperFeel (2000x compression)
        self.round += 1;
        let hyper_gradient =
            self.hyperfeel
                .encode_gradient(&gradient, self.round, &self.base.agent_id);

        info!(
            "🧠 Encoded {} value updates to HyperGradient (round {})",
            updates.len(),
            self.round,
        );

        hyper_gradient
    }

    /// Create a MATL ProofOfGradientQuality for trust verification
    pub fn create_gradient_proof(&self, hyper_gradient: &HyperGradient) -> ProofOfGradientQuality {
        ProofOfGradientQuality::new(
            hyper_gradient.quality_score as f64,
            0.9,  // Agreement threshold
            0.05, // Noise estimate
        )
    }

    /// Record reputation score for an agent in Symthaea
    pub fn record_agent_reputation(&mut self, agent: &str, phi: f64, interaction_count: u64) {
        let score = HappReputationScore {
            happ_id: "symthaea".to_string(),
            happ_name: "Symthaea Consciousness System".to_string(),
            score: (phi * 0.8 + 0.2).clamp(0.0, 1.0), // Consciousness-weighted reputation
            interactions: interaction_count,
            last_updated: now_secs(),
        };

        self.local_bridge.record_reputation(agent, score);

        debug!(
            "📊 Recorded reputation for agent '{}': score={:.3}, interactions={}",
            agent, phi, interaction_count
        );
    }

    /// Query cross-hApp reputation for an agent
    pub fn query_reputation(&self, agent: &str) -> CrossHappReputation {
        self.local_bridge.query_reputation(agent)
    }

    /// Check if an agent is trustworthy for a given action type
    pub fn is_agent_trustworthy(&self, agent: &str, action_type: ActionType) -> bool {
        let threshold = match action_type {
            ActionType::Basic => REP_BASIC,
            ActionType::Governance => REP_GOVERNANCE,
            ActionType::Voting => REP_VOTING,
            ActionType::Constitutional => REP_CONSTITUTIONAL,
        };

        let rep = self.query_reputation(agent);
        rep.is_trustworthy(threshold)
    }

    /// Create an epistemic claim for a value evaluation
    ///
    /// Note: This requires the mycelix-sdk to be available at the configured path.
    /// The SDK defines EpistemicClaim, EmpiricalLevel, NormativeLevel, and MaterialityLevel.
    /// If compilation fails, ensure the mycelix-workspace is cloned alongside this repo.
    pub fn create_epistemic_claim(
        &self,
        content: &str,
        eval: &EvaluationResult,
        action_type: ActionType,
    ) -> EpistemicClaim {
        // Determine empirical level based on evaluation method
        // consciousness_adequacy is f64 (0-1), use threshold of 0.5
        let empirical = if eval.authenticity > 0.8 {
            EmpiricalLevel::E3Cryptographic
        } else if eval.consciousness_adequacy > 0.5 {
            EmpiricalLevel::E2PrivateVerify
        } else {
            EmpiricalLevel::E1Testimonial
        };

        // Determine normative level based on action type
        // SDK variants: N0Personal, N1Communal, N2Network, N3Axiomatic
        let normative = match action_type {
            ActionType::Constitutional => NormativeLevel::N3Axiomatic, // Constitutional law level
            ActionType::Governance | ActionType::Voting => NormativeLevel::N2Network,
            ActionType::Basic => NormativeLevel::N1Communal,
        };

        // Determine materiality based on persistence
        let materiality = MaterialityLevel::M2Persistent;

        EpistemicClaim::new(content, empirical, normative, materiality)
    }

    /// Get events from the local bridge
    pub fn get_governance_events(&self, event_type: &str, since: u64) -> Vec<&BridgeEvent> {
        self.local_bridge.get_events(event_type, since)
    }

    /// Get base bridge for direct access to value evaluation
    pub fn base(&self) -> &MycelixBridge {
        &self.base
    }

    /// Get mutable base bridge
    pub fn base_mut(&mut self) -> &mut MycelixBridge {
        &mut self.base
    }
}

/// Consciousness-weighted reputation score
///
/// Integrates Symthaea's Φ measurement with Mycelix reputation
#[cfg(feature = "mycelix_sdk")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessReputation {
    /// Agent identifier
    pub agent: String,
    /// Cross-hApp reputation from Mycelix
    pub happ_reputation: f64,
    /// Consciousness quality score from Symthaea
    pub consciousness_quality: f64,
    /// Combined weighted score
    pub combined_score: f64,
    /// Timestamp
    pub timestamp: u64,
}

#[cfg(feature = "mycelix_sdk")]
impl ConsciousnessReputation {
    /// Create from consciousness snapshot and cross-hApp reputation
    pub fn new(
        agent: impl Into<String>,
        consciousness: &ConsciousnessSnapshot,
        happ_rep: &CrossHappReputation,
    ) -> Self {
        let consciousness_quality = consciousness.quality_score();
        let happ_reputation = happ_rep.aggregate;

        // Combined score: 60% consciousness, 40% reputation
        let combined = consciousness_quality * 0.6 + happ_reputation * 0.4;

        Self {
            agent: agent.into(),
            happ_reputation,
            consciousness_quality,
            combined_score: combined,
            timestamp: now_secs(),
        }
    }

    /// Check if agent meets threshold for action type
    pub fn meets_threshold(&self, action_type: ActionType) -> bool {
        let threshold = match action_type {
            ActionType::Basic => REP_BASIC,
            ActionType::Governance => REP_GOVERNANCE,
            ActionType::Voting => REP_VOTING,
            ActionType::Constitutional => REP_CONSTITUTIONAL,
        };
        self.combined_score >= threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FINANCE HEALTH SIGNALS — received from Mycelix finance cluster
// ═══════════════════════════════════════════════════════════════════════════════

/// Financial health signals received from the Mycelix finance cluster.
///
/// These are polled periodically (not every cycle) and cached. Financial
/// stress affects the engagement dimension of the consciousness profile,
/// ensuring communities under financial duress have reduced governance
/// weight to prevent stress-driven bad decisions.
///
/// # Science
///
/// Borio (2014) — financial stress as systemic risk indicator.
/// Kahneman & Tversky (1979) — loss aversion amplifies under stress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FinanceHealthSignals {
    /// Total active collateral positions.
    pub active_positions: u32,
    /// Positions in Warning/MarginCall/Liquidation.
    pub stressed_positions: u32,
    /// Positions in MarginCall or Liquidation (critical subset).
    pub critical_positions: u32,
    /// Average LTV ratio across active positions.
    pub avg_ltv: f32,
    /// Financial stress index (stressed/total, capped at 1.0).
    pub stress_index: f32,
    /// Oracle consensus confidence (0.0-1.0).
    pub oracle_confidence: f32,
    /// Circuit breaker open count.
    pub open_breakers: u32,
    /// Total SAP in circulation (micro-SAP).
    pub sap_circulation: u64,
    /// Total compost collected this period (micro-SAP).
    pub compost_collected: u64,
    /// Number of active covenants.
    pub active_covenants: u32,
    /// When these signals were last updated (cycle number).
    pub last_updated_cycle: u64,
}

impl FinanceHealthSignals {
    /// Compute the financial stress index.
    ///
    /// `stress_index = stressed_positions / max(active_positions, 1)`
    /// Capped at 1.0. Higher = more financial stress in the community.
    pub fn compute_stress_index(&self) -> f32 {
        if self.active_positions == 0 {
            return 0.0;
        }
        (self.stressed_positions as f32 / self.active_positions as f32).min(1.0)
    }

    /// Compute engagement modulation for **individual** financial operations.
    ///
    /// Dampens engagement for deposits, collateral, minting — individual
    /// decisions degrade under stress (Kahneman & Tversky 1979, Mani et al. 2013).
    ///
    /// - `stress < 0.1`: no effect
    /// - `stress 0.1–0.5`: −5% to −15%
    /// - `stress > 0.5`: −15% to −30%
    pub fn individual_engagement_modulation(&self) -> f32 {
        let stress = self.compute_stress_index();
        if stress < 0.1 {
            0.0
        } else if stress < 0.5 {
            -0.05 - (stress - 0.1) * 0.25
        } else {
            -0.15 - (stress - 0.5) * 0.30
        }
    }

    /// Compute engagement modulation for **cooperative governance** operations.
    ///
    /// Cooperatives rally during moderate adversity — communities need MORE
    /// governance capacity during crises, not less. Only extreme shock (>0.7)
    /// triggers dampening (collective paralysis).
    ///
    /// - `stress < 0.1`: no effect (healthy)
    /// - `stress 0.1–0.5`: +5% to +10% (mobilize collective response)
    /// - `stress 0.5–0.7`: 0% (stress cancels boost)
    /// - `stress > 0.7`: −10% to −20% (extreme crisis — circuit breaker)
    ///
    /// # Science
    ///
    /// Ostrom (1990) — commons governance strengthens under moderate scarcity.
    /// Aldrich (2012) — social capital mobilizes during community crises.
    /// Solnit (2009) — communities self-organize in disasters ("A Paradise Built in Hell").
    pub fn cooperative_engagement_modulation(&self) -> f32 {
        let stress = self.compute_stress_index();
        if stress < 0.1 {
            0.0
        } else if stress < 0.5 {
            // Mobilize: +5% to +10% (linear)
            0.05 + (stress - 0.1) * 0.125
        } else if stress < 0.7 {
            // Neutral: boost fades as stress becomes extreme
            // Linear from +10% at 0.5 to 0% at 0.7
            0.10 - (stress - 0.5) * 0.50
        } else {
            // Extreme crisis: −10% to −20% (collective paralysis)
            -0.10 - (stress - 0.7) * 0.333
        }
    }

    /// Legacy engagement_modulation — delegates to individual modulation.
    ///
    /// Callers that need cooperative modulation should use
    /// `cooperative_engagement_modulation()` instead.
    pub fn engagement_modulation(&self) -> f32 {
        self.individual_engagement_modulation()
    }

    /// Whether the financial system is in crisis (stress > 0.7 or breakers open).
    ///
    /// Threshold raised from 0.5 to 0.7 to match the cooperative governance
    /// model — moderate stress (0.5) triggers mobilization, not crisis.
    pub fn is_crisis(&self) -> bool {
        self.compute_stress_index() > 0.7 || self.open_breakers > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = MycelixBridge::new("test-agent");
        let stats = bridge.stats();
        assert_eq!(stats.pending_updates, 0);
    }

    #[test]
    fn test_consciousness_snapshot() {
        let snapshot = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);
        assert!(snapshot.is_adequate_for(ActionType::Basic));
        assert!(snapshot.is_adequate_for(ActionType::Governance));
        assert!(snapshot.is_adequate_for(ActionType::Voting));
        assert!(!snapshot.is_adequate_for(ActionType::Constitutional));
    }

    #[test]
    fn test_proposal_evaluation() {
        let mut bridge = MycelixBridge::new("test-agent");

        let proposal = Proposal {
            id: "prop-1".to_string(),
            title: "Help community members".to_string(),
            description: "Create a mutual aid fund to help community members in need".to_string(),
            proposer: "proposer-1".to_string(),
            created_at: now_secs(),
            proposal_type: ProposalType::Standard,
            required_phi: 0.3,
        };

        let consciousness = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);
        let affective = AffectiveSystemsState {
            care: 0.7,
            play: 0.3,
            seeking: 0.5,
            ..Default::default()
        };

        let result = bridge.evaluate_proposal(&proposal, consciousness, affective);
        assert!(result.is_ok());
    }

    #[test]
    fn test_value_learning_recording() {
        let mut bridge = MycelixBridge::new("test-agent");

        bridge.record_value_learning(
            Harmony::PanSentientFlourishing,
            0.01,
            true,
            "Helped user with compassion",
            0.6,
        );

        assert_eq!(bridge.pending_updates.len(), 1);
    }

    #[test]
    fn test_insufficient_consciousness_rejected() {
        let mut bridge = MycelixBridge::new("test-agent");

        let proposal = Proposal {
            id: "prop-1".to_string(),
            title: "Test".to_string(),
            description: "Test proposal".to_string(),
            proposer: "proposer-1".to_string(),
            created_at: now_secs(),
            proposal_type: ProposalType::Constitutional,
            required_phi: 0.6,
        };

        // Low consciousness for constitutional change
        let consciousness = ConsciousnessSnapshot::new(0.3, 0.4, 0.5, 0.6, 0.5, 0.6);
        let affective = AffectiveSystemsState::default();

        let result = bridge.submit_proposal(&proposal, consciousness, affective);
        assert!(matches!(
            result,
            Err(BridgeError::InsufficientConsciousness { .. })
        ));
    }

    #[cfg(feature = "mycelix")]
    #[test]
    fn test_record_governance_outcome_aligned_pass() {
        let mut bridge = MycelixBridge::new("test-agent");
        let vote = Vote {
            proposal_id: "p1".into(),
            voter: "test-agent".into(),
            value: VoteValue::Yes,
            consciousness: ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6),
            alignment: ValueAlignmentResult {
                overall_score: 0.8,
                harmony_scores: HashMap::new(),
                violations: vec![],
                authenticity: 0.9,
                recommendation: GovernanceRecommendation::Support,
            },
            timestamp: 0,
        };

        let (event, outcome) = bridge.record_governance_outcome("p1", true, 0.7, Some(&vote));
        assert!(matches!(
            event.kind,
            crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::TallyCompleted {
                passed: true,
                ..
            }
        ));
        assert_eq!(outcome.my_vote_aligned, Some(true)); // voted yes, passed
        assert!((outcome.value_alignment_score - 0.8).abs() < 1e-6);
    }

    #[cfg(feature = "mycelix")]
    #[test]
    fn test_epistemic_escalation_blocks_low_phi() {
        use crate::mycelix::epistemic_mesh::{EpistemicMesh, EpistemicSummary};
        use crate::mycelix::gis::IgnoranceType;

        // Create mesh where "climate" is a blind spot (4/5 agents uncertain)
        let summaries = vec![
            EpistemicSummary {
                agent_id: "a1".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["climate".into()],
            },
            EpistemicSummary {
                agent_id: "a2".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["climate".into()],
            },
            EpistemicSummary {
                agent_id: "a3".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["climate".into()],
            },
            EpistemicSummary {
                agent_id: "a4".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["climate".into()],
            },
            EpistemicSummary {
                agent_id: "a5".into(),
                dominant_ignorance: IgnoranceType::Known,
                domain_expertise: vec![],
                blind_spots: vec![],
            },
        ];
        let mesh = EpistemicMesh::new(summaries);

        let mut bridge = MycelixBridge::new("test-agent");
        let proposal = Proposal {
            id: "p1".into(),
            title: "Climate action".into(),
            description: "Climate proposal".into(),
            proposer: "agent-1".into(),
            created_at: 0,
            proposal_type: ProposalType::Standard,
            required_phi: 0.3,
        };

        // Phi 0.5 — normally enough for Standard, but climate is blind spot
        // with severity 0.8 → Guardian tier → needs 0.8
        let consciousness = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);
        let affective = AffectiveSystemsState::default();

        let result = bridge.submit_proposal_with_epistemic_gate(
            &proposal,
            consciousness,
            affective,
            &["climate".into()],
            Some(&mesh),
        );
        assert!(
            matches!(result, Err(BridgeError::InsufficientConsciousness { .. })),
            "Blind-spot domain should escalate consciousness requirement"
        );
    }

    #[cfg(feature = "mycelix")]
    #[test]
    fn test_extract_consciousness_vector() {
        let snapshot = ConsciousnessSnapshot::new(0.5, 0.6, 0.7, 0.8, 0.5, 0.6);
        let (phi, meta, coherence, care, _quality, _epistemic) =
            MycelixBridge::extract_consciousness_vector(&snapshot);
        assert!((phi - 0.5).abs() < 1e-6);
        assert!((meta - 0.6).abs() < 1e-6);
        assert!((coherence - 0.8).abs() < 1e-6);
        assert!((care - 0.6).abs() < 1e-6);
    }

    #[cfg(feature = "mycelix")]
    #[test]
    fn test_create_event_helpers() {
        let mut bridge = MycelixBridge::new("test-agent");

        let emergency = bridge.create_emergency_event();
        assert!(matches!(
            emergency.kind,
            crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::EmergencyDeclared
        ));

        let reciprocity = bridge.create_reciprocity_event(10.0);
        assert!(matches!(
            reciprocity.kind,
            crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::ReciprocityPledge { amount }
            if (amount - 10.0).abs() < 1e-6
        ));

        let dispute = bridge.create_dispute_event(true);
        assert!(matches!(
            dispute.kind,
            crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::JusticeDispute { involves_self: true }
        ));

        let rep = bridge.create_reputation_event(-0.5);
        assert!(matches!(
            rep.kind,
            crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::ReputationChanged { delta }
            if (delta - (-0.5)).abs() < 1e-6
        ));

        // Verify drain collects all 4 events
        let (events, outcomes) = bridge.drain_pending_governance();
        assert_eq!(events.len(), 4, "4 events should be queued");
        assert_eq!(outcomes.len(), 0, "no outcomes from create_* methods");

        // After drain, queue is empty
        let (events2, outcomes2) = bridge.drain_pending_governance();
        assert_eq!(events2.len(), 0);
        assert_eq!(outcomes2.len(), 0);
    }

    // ── FinanceHealthSignals tests ──────────────────────────────────────

    #[test]
    fn test_finance_stress_index_zero_positions() {
        let signals = FinanceHealthSignals::default();
        assert_eq!(signals.compute_stress_index(), 0.0);
        assert_eq!(signals.engagement_modulation(), 0.0);
    }

    #[test]
    fn test_finance_stress_index_some_stressed() {
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 20,
            ..Default::default()
        };
        let stress = signals.compute_stress_index();
        assert!((stress - 0.2).abs() < 1e-6, "20/100 = 0.2, got {stress}");
    }

    #[test]
    fn test_finance_stress_index_all_stressed() {
        let signals = FinanceHealthSignals {
            active_positions: 50,
            stressed_positions: 80, // more stressed than active (edge case)
            ..Default::default()
        };
        let stress = signals.compute_stress_index();
        assert!((stress - 1.0).abs() < 1e-6, "capped at 1.0, got {stress}");
    }

    #[test]
    fn test_finance_individual_modulation_healthy() {
        // stress < 0.1 → no modulation
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 5,
            ..Default::default()
        };
        assert_eq!(signals.individual_engagement_modulation(), 0.0);
    }

    #[test]
    fn test_finance_individual_modulation_moderate() {
        // stress = 0.3 → between -5% and -15%
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 30,
            ..Default::default()
        };
        let mod_val = signals.individual_engagement_modulation();
        assert!(
            mod_val < -0.05 && mod_val > -0.15,
            "moderate stress should dampen individual engagement -5% to -15%, got {mod_val}"
        );
    }

    #[test]
    fn test_finance_individual_modulation_crisis() {
        // stress = 0.8 → between -15% and -30%
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 80,
            ..Default::default()
        };
        let mod_val = signals.individual_engagement_modulation();
        assert!(
            mod_val < -0.15 && mod_val > -0.31,
            "crisis should dampen individual engagement -15% to -30%, got {mod_val}"
        );
    }

    #[test]
    fn test_finance_cooperative_modulation_mobilize() {
        // stress = 0.3 → cooperatives RALLY: +5% to +10%
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 30,
            ..Default::default()
        };
        let mod_val = signals.cooperative_engagement_modulation();
        assert!(
            mod_val > 0.05 && mod_val < 0.10,
            "moderate stress should BOOST cooperative governance, got {mod_val}"
        );
    }

    #[test]
    fn test_finance_cooperative_modulation_neutral() {
        // stress = 0.6 → boost fading, near neutral
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 60,
            ..Default::default()
        };
        let mod_val = signals.cooperative_engagement_modulation();
        assert!(
            mod_val.abs() < 0.06,
            "high stress should be near-neutral for cooperatives, got {mod_val}"
        );
    }

    #[test]
    fn test_finance_cooperative_modulation_extreme_crisis() {
        // stress = 0.9 → even cooperatives freeze: -10% to -20%
        let signals = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 90,
            ..Default::default()
        };
        let mod_val = signals.cooperative_engagement_modulation();
        assert!(
            mod_val < -0.05 && mod_val > -0.25,
            "extreme crisis should dampen even cooperative governance, got {mod_val}"
        );
    }

    #[test]
    fn test_finance_default_no_stress() {
        let signals = FinanceHealthSignals::default();
        assert_eq!(signals.active_positions, 0);
        assert_eq!(signals.stressed_positions, 0);
        assert_eq!(signals.stress_index, 0.0);
        assert_eq!(signals.compute_stress_index(), 0.0);
        assert_eq!(signals.engagement_modulation(), 0.0);
        assert!(!signals.is_crisis());
    }

    #[test]
    fn test_finance_is_crisis_stress() {
        // stress = 0.8 > 0.7 threshold → crisis
        let signals = FinanceHealthSignals {
            active_positions: 10,
            stressed_positions: 8,
            ..Default::default()
        };
        assert!(signals.is_crisis());

        // stress = 0.6 < 0.7 threshold → NOT crisis (cooperatives mobilize)
        let moderate = FinanceHealthSignals {
            active_positions: 10,
            stressed_positions: 6,
            ..Default::default()
        };
        assert!(
            !moderate.is_crisis(),
            "moderate stress should NOT be crisis — cooperatives rally"
        );
    }

    #[test]
    fn test_finance_is_crisis_breakers() {
        let signals = FinanceHealthSignals {
            active_positions: 10,
            stressed_positions: 0,
            open_breakers: 1,
            ..Default::default()
        };
        assert!(signals.is_crisis());
    }

    // ── Serialization round-trip tests ─────────────────────────────────

    #[test]
    fn consciousness_snapshot_serde_roundtrip() {
        let original = ConsciousnessSnapshot::new(0.75, 0.8, 0.9, 0.85, 0.3, 0.6);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ConsciousnessSnapshot = serde_json::from_str(&json).unwrap();
        assert!((deserialized.phi - original.phi).abs() < f64::EPSILON);
        assert!((deserialized.meta_awareness - original.meta_awareness).abs() < f64::EPSILON);
        assert!(
            (deserialized.self_model_accuracy - original.self_model_accuracy).abs() < f64::EPSILON
        );
        assert!((deserialized.coherence - original.coherence).abs() < f64::EPSILON);
        assert!((deserialized.affective_valence - original.affective_valence).abs() < f64::EPSILON);
        assert!((deserialized.care_activation - original.care_activation).abs() < f64::EPSILON);
        assert_eq!(deserialized.timestamp_secs, original.timestamp_secs);
    }

    #[test]
    fn consciousness_snapshot_serde_zero_timestamp() {
        let original = ConsciousnessSnapshot {
            phi: 0.5,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            timestamp_secs: 0,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ConsciousnessSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.timestamp_secs, 0);
        assert!((deserialized.phi - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn consciousness_snapshot_serde_extreme_phi() {
        // Maximum phi
        let high = ConsciousnessSnapshot {
            phi: 1.0,
            meta_awareness: 1.0,
            self_model_accuracy: 1.0,
            coherence: 1.0,
            affective_valence: 1.0,
            care_activation: 1.0,
            timestamp_secs: u64::MAX,
        };
        let json = serde_json::to_string(&high).unwrap();
        let rt: ConsciousnessSnapshot = serde_json::from_str(&json).unwrap();
        assert!((rt.phi - 1.0).abs() < f64::EPSILON);
        assert_eq!(rt.timestamp_secs, u64::MAX);

        // Zero phi
        let zero = ConsciousnessSnapshot {
            phi: 0.0,
            meta_awareness: 0.0,
            self_model_accuracy: 0.0,
            coherence: 0.0,
            affective_valence: -1.0,
            care_activation: 0.0,
            timestamp_secs: 0,
        };
        let json = serde_json::to_string(&zero).unwrap();
        let rt: ConsciousnessSnapshot = serde_json::from_str(&json).unwrap();
        assert!((rt.phi - 0.0).abs() < f64::EPSILON);
        assert!((rt.affective_valence - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn consciousness_snapshot_serde_nan_inf() {
        // serde_json serializes NaN/Inf to JSON null/Infinity, which then
        // fails to deserialize back into f64. This means a full round-trip
        // is impossible for these values — the bridge must sanitize inputs.
        let nan_snap = ConsciousnessSnapshot {
            phi: f64::NAN,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            timestamp_secs: 0,
        };
        // serde_json may serialize NaN to null; either serialization or
        // deserialization should fail — a full round-trip must not succeed.
        let nan_roundtrip = serde_json::to_string(&nan_snap)
            .ok()
            .and_then(|json| serde_json::from_str::<ConsciousnessSnapshot>(&json).ok());
        assert!(
            nan_roundtrip.is_none(),
            "NaN should not survive a full serde_json round-trip"
        );

        let inf_snap = ConsciousnessSnapshot {
            phi: f64::INFINITY,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            timestamp_secs: 0,
        };
        let inf_roundtrip = serde_json::to_string(&inf_snap)
            .ok()
            .and_then(|json| serde_json::from_str::<ConsciousnessSnapshot>(&json).ok());
        assert!(
            inf_roundtrip.is_none(),
            "Infinity should not survive a full serde_json round-trip"
        );

        let neg_inf_snap = ConsciousnessSnapshot {
            phi: f64::NEG_INFINITY,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            timestamp_secs: 0,
        };
        let neg_inf_roundtrip = serde_json::to_string(&neg_inf_snap)
            .ok()
            .and_then(|json| serde_json::from_str::<ConsciousnessSnapshot>(&json).ok());
        assert!(
            neg_inf_roundtrip.is_none(),
            "Negative infinity should not survive a full serde_json round-trip"
        );
    }

    #[test]
    fn factcheck_epistemic_feedback_serde_roundtrip() {
        let original = FactcheckEpistemicFeedback::from_factcheck(
            "Water boils at 100C",
            "confirmed",
            0.95,
            0.9,
            0.8,
            0.3,
            0.85,
        );
        let json = serde_json::to_string(&original).unwrap();
        let rt: FactcheckEpistemicFeedback = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.e_tier, original.e_tier);
        assert_eq!(rt.n_tier, original.n_tier);
        assert_eq!(rt.m_tier, original.m_tier);
        assert!((rt.h_value - original.h_value).abs() < f32::EPSILON);
        assert!((rt.quality - original.quality).abs() < f32::EPSILON);
        assert_eq!(rt.verdict, original.verdict);
        assert_eq!(rt.statement, original.statement);
    }

    #[test]
    fn value_alignment_result_serde_roundtrip() {
        let mut harmony_scores = HashMap::new();
        harmony_scores.insert("care".to_string(), 0.9);
        harmony_scores.insert("justice".to_string(), 0.7);
        let original = ValueAlignmentResult {
            overall_score: 0.8,
            harmony_scores,
            violations: vec!["minor concern".to_string()],
            authenticity: 0.95,
            recommendation: GovernanceRecommendation::Support,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: ValueAlignmentResult = serde_json::from_str(&json).unwrap();
        assert!((rt.overall_score - original.overall_score).abs() < f64::EPSILON);
        assert_eq!(rt.harmony_scores.len(), 2);
        assert!((rt.harmony_scores["care"] - 0.9).abs() < f64::EPSILON);
        assert_eq!(rt.violations, original.violations);
        assert!((rt.authenticity - original.authenticity).abs() < f64::EPSILON);
        assert_eq!(rt.recommendation, GovernanceRecommendation::Support);
    }

    #[test]
    fn governance_recommendation_all_variants_serde_roundtrip() {
        let variants = [
            GovernanceRecommendation::StrongSupport,
            GovernanceRecommendation::Support,
            GovernanceRecommendation::Neutral,
            GovernanceRecommendation::Oppose,
            GovernanceRecommendation::StrongOppose,
            GovernanceRecommendation::CannotEvaluate,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let rt: GovernanceRecommendation = serde_json::from_str(&json).unwrap();
            assert_eq!(&rt, variant);
        }
    }

    #[test]
    fn proposal_serde_roundtrip() {
        let original = Proposal {
            id: "prop-42".to_string(),
            title: "Community garden".to_string(),
            description: "Establish a shared garden space".to_string(),
            proposer: "agent-7".to_string(),
            created_at: 1_700_000_000,
            proposal_type: ProposalType::Grant,
            required_phi: 0.3,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: Proposal = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.id, original.id);
        assert_eq!(rt.title, original.title);
        assert_eq!(rt.description, original.description);
        assert_eq!(rt.proposer, original.proposer);
        assert_eq!(rt.created_at, original.created_at);
        assert_eq!(rt.proposal_type, ProposalType::Grant);
        assert!((rt.required_phi - original.required_phi).abs() < f64::EPSILON);
    }

    #[test]
    fn proposal_type_all_variants_serde_roundtrip() {
        let variants = [
            ProposalType::Standard,
            ProposalType::Constitutional,
            ProposalType::Emergency,
            ProposalType::Grant,
            ProposalType::Parameter,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let rt: ProposalType = serde_json::from_str(&json).unwrap();
            assert_eq!(&rt, variant);
        }
    }

    #[test]
    fn vote_value_all_variants_serde_roundtrip() {
        let variants = [
            VoteValue::StrongYes,
            VoteValue::Yes,
            VoteValue::Abstain,
            VoteValue::No,
            VoteValue::StrongNo,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let rt: VoteValue = serde_json::from_str(&json).unwrap();
            assert_eq!(&rt, variant);
        }
    }

    #[test]
    fn vote_serde_roundtrip() {
        let original = Vote {
            proposal_id: "prop-1".to_string(),
            voter: "agent-3".to_string(),
            value: VoteValue::Yes,
            consciousness: ConsciousnessSnapshot::new(0.6, 0.7, 0.8, 0.9, 0.4, 0.5),
            alignment: ValueAlignmentResult {
                overall_score: 0.75,
                harmony_scores: HashMap::new(),
                violations: vec![],
                authenticity: 0.85,
                recommendation: GovernanceRecommendation::Support,
            },
            timestamp: 1_700_000_000,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: Vote = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.proposal_id, original.proposal_id);
        assert_eq!(rt.voter, original.voter);
        assert_eq!(rt.value, VoteValue::Yes);
        assert!((rt.consciousness.phi - 0.6).abs() < f64::EPSILON);
        assert!((rt.alignment.overall_score - 0.75).abs() < f64::EPSILON);
        assert_eq!(rt.timestamp, 1_700_000_000);
    }

    #[test]
    fn value_learning_update_serde_roundtrip() {
        let original = ValueLearningUpdate {
            agent_id: "agent-1".to_string(),
            harmony: "PanSentientFlourishing".to_string(),
            importance_delta: 0.01,
            affirmation_delta: 1,
            context: "Helped user with compassion".to_string(),
            phi_at_learning: 0.6,
            timestamp: 1_700_000_000,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: ValueLearningUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.agent_id, original.agent_id);
        assert_eq!(rt.harmony, original.harmony);
        assert!((rt.importance_delta - original.importance_delta).abs() < f64::EPSILON);
        assert_eq!(rt.affirmation_delta, original.affirmation_delta);
        assert_eq!(rt.context, original.context);
        assert!((rt.phi_at_learning - original.phi_at_learning).abs() < f64::EPSILON);
        assert_eq!(rt.timestamp, original.timestamp);
    }

    #[test]
    fn compressed_value_gradient_serde_roundtrip() {
        let original = CompressedValueGradient {
            harmony_encoding: vec![0xFF, 0x00, 0xAB, 0xCD],
            importance_gradient: vec![0x01, 0x02, 0x03],
            round: 42,
            agent_id: "agent-5".to_string(),
            compression_ratio: 2000.0,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: CompressedValueGradient = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.harmony_encoding, original.harmony_encoding);
        assert_eq!(rt.importance_gradient, original.importance_gradient);
        assert_eq!(rt.round, original.round);
        assert_eq!(rt.agent_id, original.agent_id);
        assert!((rt.compression_ratio - original.compression_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn submission_result_serde_roundtrip() {
        let original = SubmissionResult {
            proposal_id: "prop-99".to_string(),
            consciousness: ConsciousnessSnapshot::new(0.7, 0.8, 0.85, 0.9, 0.5, 0.6),
            alignment: ValueAlignmentResult {
                overall_score: 0.85,
                harmony_scores: {
                    let mut m = HashMap::new();
                    m.insert("care".to_string(), 0.95);
                    m
                },
                violations: vec![],
                authenticity: 0.9,
                recommendation: GovernanceRecommendation::StrongSupport,
            },
            submitted_at: 1_700_000_000,
            success: true,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: SubmissionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.proposal_id, original.proposal_id);
        assert!((rt.consciousness.phi - 0.7).abs() < f64::EPSILON);
        assert!((rt.alignment.overall_score - 0.85).abs() < f64::EPSILON);
        assert_eq!(
            rt.alignment.recommendation,
            GovernanceRecommendation::StrongSupport
        );
        assert_eq!(rt.submitted_at, 1_700_000_000);
        assert!(rt.success);
    }

    #[test]
    fn bridge_error_all_variants_serde_roundtrip() {
        let variants = vec![
            BridgeError::InsufficientConsciousness {
                current: 0.3,
                required: 0.6,
                action: "constitutional".to_string(),
            },
            BridgeError::ValueViolation {
                reason: "harm detected".to_string(),
            },
            BridgeError::CannotEvaluate,
            BridgeError::NetworkError {
                message: "timeout".to_string(),
            },
            BridgeError::InvalidProposal {
                reason: "empty title".to_string(),
            },
        ];
        for original in &variants {
            let json = serde_json::to_string(original).unwrap();
            let rt: BridgeError = serde_json::from_str(&json).unwrap();
            // Compare via Debug representation since BridgeError doesn't derive PartialEq
            assert_eq!(format!("{rt:?}"), format!("{original:?}"));
        }
    }

    #[test]
    fn finance_health_signals_serde_roundtrip() {
        let original = FinanceHealthSignals {
            active_positions: 100,
            stressed_positions: 20,
            critical_positions: 5,
            avg_ltv: 0.65,
            stress_index: 0.2,
            oracle_confidence: 0.95,
            open_breakers: 0,
            sap_circulation: 1_000_000,
            compost_collected: 50_000,
            active_covenants: 12,
            last_updated_cycle: 42,
        };
        let json = serde_json::to_string(&original).unwrap();
        let rt: FinanceHealthSignals = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.active_positions, original.active_positions);
        assert_eq!(rt.stressed_positions, original.stressed_positions);
        assert_eq!(rt.critical_positions, original.critical_positions);
        assert!((rt.avg_ltv - original.avg_ltv).abs() < f32::EPSILON);
        assert!((rt.stress_index - original.stress_index).abs() < f32::EPSILON);
        assert!((rt.oracle_confidence - original.oracle_confidence).abs() < f32::EPSILON);
        assert_eq!(rt.open_breakers, original.open_breakers);
        assert_eq!(rt.sap_circulation, original.sap_circulation);
        assert_eq!(rt.compost_collected, original.compost_collected);
        assert_eq!(rt.active_covenants, original.active_covenants);
        assert_eq!(rt.last_updated_cycle, original.last_updated_cycle);
    }

    #[test]
    fn finance_health_signals_default_serde_roundtrip() {
        let original = FinanceHealthSignals::default();
        let json = serde_json::to_string(&original).unwrap();
        let rt: FinanceHealthSignals = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.active_positions, 0);
        assert_eq!(rt.sap_circulation, 0);
        assert!((rt.stress_index - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cross_type_nested_serde_roundtrip() {
        // Test that deeply nested types (Vote contains ConsciousnessSnapshot
        // and ValueAlignmentResult) survive a double round-trip
        let vote = Vote {
            proposal_id: "p-nested".to_string(),
            voter: "agent-nested".to_string(),
            value: VoteValue::StrongNo,
            consciousness: ConsciousnessSnapshot::new(0.45, 0.55, 0.65, 0.75, -0.2, 0.3),
            alignment: ValueAlignmentResult {
                overall_score: -0.5,
                harmony_scores: {
                    let mut m = HashMap::new();
                    m.insert("justice".to_string(), -0.8);
                    m.insert("care".to_string(), 0.1);
                    m
                },
                violations: vec!["exploitation".to_string(), "coercion".to_string()],
                authenticity: 0.4,
                recommendation: GovernanceRecommendation::StrongOppose,
            },
            timestamp: 0, // zero timestamp edge case
        };

        // First round-trip
        let json1 = serde_json::to_string(&vote).unwrap();
        let rt1: Vote = serde_json::from_str(&json1).unwrap();

        // Second round-trip (ensures no drift)
        let json2 = serde_json::to_string(&rt1).unwrap();
        let rt2: Vote = serde_json::from_str(&json2).unwrap();

        // Compare via serde_json::Value (not raw strings) because HashMap
        // key ordering is non-deterministic across serializations.
        let val1: serde_json::Value = serde_json::from_str(&json1).unwrap();
        let val2: serde_json::Value = serde_json::from_str(&json2).unwrap();
        assert_eq!(
            val1, val2,
            "double round-trip should produce semantically identical JSON"
        );
        assert_eq!(rt2.value, VoteValue::StrongNo);
        assert_eq!(rt2.alignment.violations.len(), 2);
        assert_eq!(rt2.timestamp, 0);
        assert!((rt2.consciousness.affective_valence - (-0.2)).abs() < f64::EPSILON);
    }

    #[cfg(feature = "mycelix")]
    #[test]
    fn test_robotics_telemetry_dispatch_gating() {
        use symthaea_core::embodiment::{EmbodimentTelemetry, MotorSafetyLevel};
        let mut bridge = MycelixBridge::new("robotics-test");
        let (tx, rx) = MycelixBridge::create_governance_channel();
        bridge.set_governance_dispatch_tx(tx);

        let telemetry = EmbodimentTelemetry {
            total_steps: 100,
            control_effort: 0.4,
            prediction_error: 0.05,
            safety_level: MotorSafetyLevel::Green,
            platform: "quadruped".to_string(),
            num_actuators: 12,
            epistemic_grounding: "sensorimotor".to_string(),
            observation_confidence: 0.9,
            platform_specific: vec![1, 2, 3],
        };

        // Unbound: never emits (no zeroed placeholders on-chain).
        assert!(!bridge.dispatch_robotics_telemetry(&telemetry, 0.7));

        // Binding without mission status: still gated off.
        bridge.set_robotics_binding(RoboticsDispatchBinding {
            asset_hash: vec![1; 39],
            order_hash: vec![2; 39],
        });
        assert!(!bridge.dispatch_robotics_telemetry(&telemetry, 0.7));

        // Fully bound: dispatches, and the command carries the telemetry.
        bridge.update_robotics_mission_status(RoboticsMissionStatus {
            lat: 32.95,
            lon: -96.73,
            alt: 120.0,
            mission_progress: 0.5,
            fuel_level: 0.8,
        });
        assert!(bridge.dispatch_robotics_telemetry(&telemetry, 0.7));
        match rx.try_recv().expect("command must be on the channel") {
            GovernanceDispatchCommand::SubmitRoboticsTelemetry {
                platform,
                safety_level,
                consciousness_level,
                asset_hash,
                mission_progress,
                ..
            } => {
                assert_eq!(platform, "quadruped");
                assert_eq!(safety_level, "Green");
                assert!((consciousness_level - 0.7).abs() < 1e-9);
                assert_eq!(asset_hash, vec![1; 39]);
                assert!((mission_progress - 0.5).abs() < 1e-9);
            }
            other => panic!("wrong command dispatched: {other:?}"),
        }

        // Rate limit: an immediate second call is suppressed.
        assert!(!bridge.dispatch_robotics_telemetry(&telemetry, 0.7));
        assert!(rx.try_recv().is_err());

        // Clearing the binding gates it off again.
        bridge.clear_robotics_binding();
        assert!(!bridge.dispatch_robotics_telemetry(&telemetry, 0.7));
    }
}
