//! Mycelix Bridge - Consciousness-Gated Governance
//!
//! This module bridges Symthaea consciousness measurement to Mycelix governance,
//! enabling:
//! - **Consciousness-gated proposals**: Only submit when Φ > threshold
//! - **Value-aligned voting**: Evaluate proposals against Seven Harmonies
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
//! │  │ • Consciousness Φ  │◄──────────────►│ • Agora (Proposals)│        │
//! │  │ • Seven Harmonies  │   Bridge       │ • MATL (Trust)     │        │
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

use super::unified_value_evaluator::{
    UnifiedValueEvaluator, EvaluationContext, EvaluationResult,
    ActionType, Decision, VetoReason, AffectiveSystemsState,
};
use super::seven_harmonies::{SevenHarmonies, Harmony, AlignmentResult};
use super::affective_consciousness::CoreAffect;
use crate::hdc::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ============================================================================
// MYCELIX SDK INTEGRATION (when feature enabled)
// ============================================================================

#[cfg(feature = "mycelix")]
use mycelix_sdk::{
    bridge::{LocalBridge, BridgeMessage, CrossHappReputation, HappReputationScore, BridgeEvent},
    hyperfeel::{HyperFeelEncoder, EncodingConfig, HyperGradient},
    matl::ProofOfGradientQuality,
    epistemic::{EpistemicClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel},
};

#[cfg(feature = "mycelix")]
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

    /// Check if consciousness is adequate for action type
    pub fn is_adequate_for(&self, action_type: ActionType) -> bool {
        let required = match action_type {
            ActionType::Basic => 0.2,
            ActionType::Governance => 0.3,
            ActionType::Voting => 0.4,
            ActionType::Constitutional => 0.6,
        };
        self.phi >= required
    }

    /// Overall consciousness quality score
    pub fn quality_score(&self) -> f64 {
        (self.phi * 0.4
            + self.meta_awareness * 0.2
            + self.self_model_accuracy * 0.2
            + self.coherence * 0.2).clamp(0.0, 1.0)
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
    /// Harmony encoded as HV16
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
            min_phi_proposal: 0.3,
            min_phi_voting: 0.4,
            min_phi_constitutional: 0.6,
            require_care_for_others: true,
            min_care_activation: 0.3,
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
        }
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
            involves_others: true, // Proposals always affect others
        };

        // 3. Evaluate proposal against values
        let eval_result = self.evaluator.evaluate(&proposal.description, context);

        // 4. Check for veto
        match &eval_result.decision {
            Decision::Veto(reason) => {
                return Err(BridgeError::ValueViolation {
                    reason: format!("{:?}", reason),
                });
            }
            Decision::Warn(warnings) => {
                // Log warnings but allow
                // In production, might require confirmation
            }
            Decision::Allow => {}
        }

        // 5. Create submission result
        let alignment = self.create_alignment_result(&eval_result);

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
            involves_others: true,
        };

        let eval = self.evaluator.evaluate(&proposal.description, context);
        let alignment = self.create_alignment_result(&eval);

        // Cache result
        self.proposal_cache.insert(
            proposal.id.clone(),
            (alignment.clone(), Instant::now()),
        );

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

    /// Create compressed gradient for efficient transmission
    pub fn create_compressed_gradient(&self, round: u64) -> CompressedValueGradient {
        // In real implementation:
        // 1. Encode harmony importance deltas as gradient
        // 2. Compress using HyperFeel (2000x compression)
        // 3. Return compressed form

        CompressedValueGradient {
            harmony_encoding: vec![0u8; 256], // Placeholder
            importance_gradient: vec![0u8; 256], // Placeholder
            round,
            agent_id: self.agent_id.clone(),
            compression_ratio: 2000.0,
        }
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    /// Create alignment result from evaluation
    fn create_alignment_result(&self, eval: &EvaluationResult) -> ValueAlignmentResult {
        let harmony_scores: HashMap<String, f64> = eval.breakdown.harmony_scores
            .iter()
            .cloned()
            .collect();

        let violations: Vec<String> = eval.harmony_alignment.violations
            .iter()
            .map(|h| h.name().to_string())
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
            Self::InsufficientConsciousness { current, required, action } => {
                write!(f, "Insufficient consciousness for {}: {} < {}", action, current, required)
            }
            Self::ValueViolation { reason } => write!(f, "Value violation: {}", reason),
            Self::CannotEvaluate => write!(f, "Cannot evaluate proposal"),
            Self::NetworkError { message } => write!(f, "Network error: {}", message),
            Self::InvalidProposal { reason } => write!(f, "Invalid proposal: {}", reason),
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
#[cfg(feature = "mycelix")]
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

#[cfg(feature = "mycelix")]
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
        let result = self.base.submit_proposal(proposal, consciousness.clone(), affective_state)?;

        // Broadcast to network
        let event_payload = serde_json::to_vec(&result).unwrap_or_default();
        let event = BridgeEvent::new(
            "proposal_submitted",
            "symthaea",
            event_payload,
        );
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
        let vote = self.base.cast_vote(proposal, consciousness.clone(), affective_state)?;

        // Broadcast vote event
        let event_payload = serde_json::to_vec(&vote).unwrap_or_default();
        let event = BridgeEvent::new(
            "vote_cast",
            "symthaea",
            event_payload,
        );
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
        let hyper_gradient = self.hyperfeel.encode_gradient(
            &gradient,
            self.round,
            &self.base.agent_id,
        );

        info!(
            "🧠 Encoded {} value updates to HyperGradient (round {}, {} bytes compressed)",
            updates.len(),
            self.round,
            hyper_gradient.compressed_size(),
        );

        hyper_gradient
    }

    /// Create a MATL ProofOfGradientQuality for trust verification
    pub fn create_gradient_proof(&self, hyper_gradient: &HyperGradient) -> ProofOfGradientQuality {
        ProofOfGradientQuality::new(
            hyper_gradient.quality_score as f64,
            0.9, // Agreement threshold
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
            ActionType::Basic => 0.3,
            ActionType::Governance => 0.5,
            ActionType::Voting => 0.6,
            ActionType::Constitutional => 0.8,
        };

        let rep = self.query_reputation(agent);
        rep.is_trustworthy(threshold)
    }

    /// Create an epistemic claim for a value evaluation
    pub fn create_epistemic_claim(
        &self,
        content: &str,
        eval: &EvaluationResult,
    ) -> EpistemicClaim {
        // Determine empirical level based on evaluation method
        let empirical = if eval.authenticity > 0.8 {
            EmpiricalLevel::E3Cryptographic
        } else if eval.consciousness_adequacy {
            EmpiricalLevel::E2PrivateVerify
        } else {
            EmpiricalLevel::E1Testimonial
        };

        // Determine normative level based on action type
        let normative = match eval.breakdown.action_type {
            ActionType::Constitutional => NormativeLevel::N3Universal,
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
#[cfg(feature = "mycelix")]
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

#[cfg(feature = "mycelix")]
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
            ActionType::Basic => 0.3,
            ActionType::Governance => 0.5,
            ActionType::Voting => 0.6,
            ActionType::Constitutional => 0.8,
        };
        self.combined_score >= threshold
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
        assert!(matches!(result, Err(BridgeError::InsufficientConsciousness { .. })));
    }
}
