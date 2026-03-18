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
//! │  │ • Consciousness Φ  │◄──────────────►│ • Agora (Proposals)│        │
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
#[allow(dead_code)] // Used by EnhancedMycelixBridge (currently unwired)
const REP_BASIC: f64 = 0.3;
#[allow(dead_code)]
const REP_GOVERNANCE: f64 = 0.5;
#[allow(dead_code)]
const REP_VOTING: f64 = 0.6;
#[allow(dead_code)]
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

    /// Check if consciousness is adequate for action type
    pub fn is_adequate_for(&self, action_type: ActionType) -> bool {
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
    governance_dispatch_tx: Option<std::sync::mpsc::Sender<GovernanceDispatchCommand>>,
    /// Channel receiver for governance outcome events from the Holochain
    /// conductor bridge (dispatch loop confirmations and poller tally results).
    /// Set via `set_governance_outcome_rx()`.
    #[cfg(feature = "mycelix")]
    governance_outcome_rx: Option<std::sync::mpsc::Receiver<GovernanceDispatchOutcome>>,
}

/// Outcome events received from the Holochain conductor bridge.
///
/// The conductor's dispatch loop and governance poller produce these after
/// successful zome calls or completed tallies. The `MycelixBridge` drains
/// them and converts to `GovernanceEvent`/`GovernanceOutcome` for the
/// `GovernanceManager`.
#[cfg(feature = "mycelix")]
#[derive(Debug, Clone)]
pub enum GovernanceDispatchOutcome {
    /// A proposal was successfully submitted on-chain.
    ProposalSubmitted { proposal_id: String },
    /// A vote was successfully recorded on-chain.
    VoteRecorded { proposal_id: String },
    /// A proposal tally completed (passed or failed).
    TallyCompleted {
        proposal_id: String,
        passed: bool,
        tally_yes: u32,
        tally_no: u32,
    },
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
        description: String,
        proposer_did: String,
        consciousness_phi: f64,
        alignment_score: f64,
    },
    /// Cast a vote on an existing proposal.
    CastVote {
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
    },
    /// Query active proposals (response arrives via governance event channel).
    QueryActiveProposals,
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
            #[cfg(feature = "mycelix")]
            pending_gov_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_outcomes: Vec::new(),
            #[cfg(feature = "mycelix")]
            governance_dispatch_tx: None,
            #[cfg(feature = "mycelix")]
            governance_outcome_rx: None,
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
            #[cfg(feature = "mycelix")]
            pending_gov_events: Vec::new(),
            #[cfg(feature = "mycelix")]
            pending_gov_outcomes: Vec::new(),
            #[cfg(feature = "mycelix")]
            governance_dispatch_tx: None,
            #[cfg(feature = "mycelix")]
            governance_outcome_rx: None,
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
    /// let (tx, rx) = std::sync::mpsc::channel();
    /// bridge.set_governance_dispatch_tx(tx);
    /// // In a separate async task:
    /// // conductor_bridge::run_dispatch_loop(rx, conductor).await;
    /// ```
    #[cfg(feature = "mycelix")]
    pub fn set_governance_dispatch_tx(
        &mut self,
        tx: std::sync::mpsc::Sender<GovernanceDispatchCommand>,
    ) {
        self.governance_dispatch_tx = Some(tx);
    }

    /// Set the governance outcome channel receiver.
    ///
    /// The sender end should be shared between the dispatch loop (for
    /// submission/vote confirmations) and the governance poller (for
    /// completed tally results). Use `create_governance_outcome_channel()`
    /// for convenience.
    #[cfg(feature = "mycelix")]
    pub fn set_governance_outcome_rx(
        &mut self,
        rx: std::sync::mpsc::Receiver<GovernanceDispatchOutcome>,
    ) {
        self.governance_outcome_rx = Some(rx);
    }

    /// Create a governance outcome channel pair.
    ///
    /// Returns `(Sender, Receiver)`. Pass the sender to the dispatch loop
    /// and governance poller; pass the receiver to `set_governance_outcome_rx()`.
    #[cfg(feature = "mycelix")]
    pub fn create_governance_outcome_channel() -> (
        std::sync::mpsc::Sender<GovernanceDispatchOutcome>,
        std::sync::mpsc::Receiver<GovernanceDispatchOutcome>,
    ) {
        std::sync::mpsc::channel()
    }

    /// Create the dispatch channel pair for Holochain governance connectivity.
    ///
    /// Returns `(Sender, Receiver)` — caller gives the sender to this bridge
    /// via `set_governance_dispatch_tx()` and passes the receiver to the
    /// conductor bridge process.
    #[cfg(feature = "mycelix")]
    pub fn create_governance_channel() -> (
        std::sync::mpsc::Sender<GovernanceDispatchCommand>,
        std::sync::mpsc::Receiver<GovernanceDispatchCommand>,
    ) {
        std::sync::mpsc::channel()
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
        if let Some(ref tx) = self.governance_dispatch_tx {
            let _ = tx.send(GovernanceDispatchCommand::SubmitProposal {
                description: proposal.description.clone(),
                proposer_did: self.agent_id.clone(),
                consciousness_phi: consciousness.phi,
                alignment_score: alignment.overall_score,
            });
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
        if let Some(ref tx) = self.governance_dispatch_tx {
            let approve = matches!(value, VoteValue::Yes | VoteValue::StrongYes);
            let _ = tx.send(GovernanceDispatchCommand::CastVote {
                proposal_id: proposal.id.clone(),
                voter_did: self.agent_id.clone(),
                approve,
                rationale: format!("{:?}: score {:.2}", alignment.recommendation, alignment.overall_score),
            });
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

    /// Create compressed gradient for efficient transmission
    pub fn create_compressed_gradient(&self, round: u64) -> CompressedValueGradient {
        // In real implementation:
        // 1. Encode harmony importance deltas as gradient
        // 2. Compress using HyperFeel (2000x compression)
        // 3. Return compressed form

        CompressedValueGradient {
            harmony_encoding: vec![0u8; 256],    // Placeholder
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
    ///
    /// Also drains the outcome channel (if set) and converts
    /// `GovernanceDispatchOutcome` into `GovernanceEvent`/`GovernanceOutcome`
    /// for the `GovernanceManager`.
    ///
    /// Returns `(events, outcomes)` — caller injects them into CLS.
    #[cfg(feature = "mycelix")]
    pub fn drain_pending_governance(
        &mut self,
    ) -> (
        Vec<crate::cognitive_loop::managers::governance_manager::GovernanceEvent>,
        Vec<crate::cognitive_loop::managers::governance_manager::GovernanceOutcome>,
    ) {
        // Drain the outcome channel into pending_gov_events / pending_gov_outcomes
        if let Some(ref rx) = self.governance_outcome_rx {
            while let Ok(outcome) = rx.try_recv() {
                match outcome {
                    GovernanceDispatchOutcome::ProposalSubmitted { ref proposal_id } => {
                        self.pending_gov_events.push(
                            crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
                                kind: crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::ProposalCreated,
                                proposal_id: Some(proposal_id.clone()),
                                timestamp_secs: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            },
                        );
                    }
                    GovernanceDispatchOutcome::VoteRecorded { ref proposal_id } => {
                        self.pending_gov_events.push(
                            crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
                                kind: crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::VoteCast {
                                    voter_phi: 0.0,
                                    vote_value: 0.0,
                                },
                                proposal_id: Some(proposal_id.clone()),
                                timestamp_secs: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            },
                        );
                    }
                    GovernanceDispatchOutcome::TallyCompleted {
                        ref proposal_id,
                        passed,
                        ..
                    } => {
                        // Emit as a GovernanceEvent (TallyCompleted)
                        self.pending_gov_events.push(
                            crate::cognitive_loop::managers::governance_manager::GovernanceEvent {
                                kind: crate::cognitive_loop::managers::governance_manager::GovernanceEventKind::TallyCompleted {
                                    passed,
                                    collective_phi: 0.0,
                                },
                                proposal_id: Some(proposal_id.clone()),
                                timestamp_secs: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            },
                        );
                        // Also emit as a GovernanceOutcome for learning feedback
                        self.pending_gov_outcomes.push(
                            crate::cognitive_loop::managers::governance_manager::GovernanceOutcome {
                                proposal_id: proposal_id.clone(),
                                passed,
                                my_vote_aligned: None,
                                value_alignment_score: 0.0,
                                harmonic_resonance: 0.0,
                            },
                        );
                    }
                }
            }
        }

        (
            std::mem::take(&mut self.pending_gov_events),
            std::mem::take(&mut self.pending_gov_outcomes),
        )
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
}
