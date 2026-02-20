#![allow(clippy::manual_range_contains, clippy::useless_vec)]
//! Governance Bridge Integrity Zome
//!
//! Entry types for cross-hApp governance queries and voting verification.
//! Enhanced with Symthaea Consciousness Metrics integration for:
//! - Consciousness gate verification for governance actions
//! - Consciousness-gated proposal submission and voting
//! - Agent consciousness history tracking
//! - Value alignment assessment
//!
//! ## Consciousness Gates
//!
//! - Basic actions: consciousness >= 0.2
//! - Proposal submission: consciousness >= 0.3
//! - Voting: consciousness >= 0.4
//! - Constitutional changes: consciousness >= 0.6
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

// Canonical consciousness gate constants — must match mycelix_bridge_common::consciousness_gates defaults.
// Source of truth: crates/mycelix-bridge-common/src/consciousness_gates.rs
const GOV_BASIC: f64 = 0.2;
const GOV_PROPOSAL: f64 = 0.3;
const GOV_VOTING: f64 = 0.4;
const GOV_CONSTITUTIONAL: f64 = 0.6;
/// Voter consciousness level for constitutional proposals (stricter than gov_voting).
/// Not in canonical set — governance-specific.
const GOV_VOTER_CONSTITUTIONAL: f64 = 0.5;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Governance query from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceQuery {
    pub id: String,
    pub query_type: GovernanceQueryType,
    pub source_happ: String,
    pub parameters: String,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum GovernanceQueryType {
    /// Get active proposals
    ActiveProposals,
    /// Get proposal by ID
    ProposalById,
    /// Get voting status
    VotingStatus,
    /// Check voting eligibility
    VotingEligibility,
    /// Get constitutional rules
    ConstitutionalRules,
}

/// Proposal reference for cross-hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProposalReference {
    pub proposal_id: String,
    pub title: String,
    pub proposal_type: ProposalType,
    pub status: ProposalStatus,
    pub votes_for: u64,
    pub votes_against: u64,
    pub ends_at: Timestamp,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalType {
    Standard,
    Emergency,
    Constitutional,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalStatus {
    Draft,
    Active,
    Passed,
    Failed,
    Executed,
    Vetoed,
}

/// Governance event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceBridgeEvent {
    pub id: String,
    pub event_type: GovernanceEventType,
    pub proposal_id: Option<String>,
    pub subject: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum GovernanceEventType {
    ProposalCreated,
    VotingStarted,
    VoteReceived,
    VotingEnded,
    ProposalPassed,
    ProposalFailed,
    ProposalExecuted,
    ConstitutionAmended,
}

/// Cross-hApp execution request (from governance to other hApps)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ExecutionRequest {
    pub id: String,
    pub proposal_id: String,
    pub target_happ: String,
    pub action: String,
    pub parameters: String,
    pub status: ExecutionStatus,
    pub requested_at: Timestamp,
    pub executed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Reverted,
}

// =============================================================================
// CONSCIOUSNESS METRICS INTEGRATION (Symthaea Bridge)
// =============================================================================
//
// ## Threshold Hierarchy
//
// Three enums manage consciousness thresholds at different governance layers:
//
// 1. `GovernanceActionType` (this zome) — ACTION-level gate.
//    Controls what consciousness level is needed to perform an action:
//    Basic(0.2), ProposalSubmission(0.3), Voting(0.4), Constitutional(0.6)
//
// 2. `ProposalType` (this zome) — PROPOSAL-level adaptive thresholds.
//    Controls per-proposal-type voting parameters via `AdaptiveThreshold`:
//    Standard(min_voter_consciousness=0.2), Emergency(0.3), Constitutional(0.5)
//
// 3. `ProposalTier` (voting integrity zome) — VALIDATION-level tier.
//    Used by the voting integrity zome for entry validation:
//    Basic(0.3), Major(0.4), Constitutional(0.6)
//
// The voter consciousness requirement is the HIGHER of: GovernanceActionType::Voting (0.4)
// and AdaptiveThreshold.min_voter_consciousness for the proposal type. For Constitutional
// proposals: max(0.4, 0.5) = 0.5 (voter bar), while the proposer bar is 0.6.
//

/// Consciousness snapshot from Symthaea system
/// Records Φ (integrated information) and related metrics at a point in time
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessSnapshot {
    /// Unique snapshot ID
    pub id: String,
    /// Agent DID
    pub agent_did: String,
    /// Consciousness level - primary consciousness measure (0.0 - 1.0)
    pub consciousness_level: f64,
    /// Meta-awareness level (0.0 - 1.0)
    pub meta_awareness: f64,
    /// Self-model accuracy (0.0 - 1.0)
    pub self_model_accuracy: f64,
    /// Narrative coherence (0.0 - 1.0)
    pub coherence: f64,
    /// Affective valence (-1.0 to +1.0)
    pub affective_valence: f64,
    /// CARE system activation (0.0 - 1.0)
    pub care_activation: f64,
    /// Snapshot timestamp
    pub captured_at: Timestamp,
    /// Source system (e.g., "symthaea", "mycelix-local")
    pub source: String,
    /// Optional C-Vector (v2 snapshots).
    #[serde(default)]
    pub consciousness_vector: Option<ConsciousnessVectorEntry>,
}

impl ConsciousnessSnapshot {
    /// Overall consciousness quality score
    pub fn quality_score(&self) -> f64 {
        (self.consciousness_level * QUALITY_CONSCIOUSNESS_WEIGHT
            + self.meta_awareness * QUALITY_META_WEIGHT
            + self.self_model_accuracy * QUALITY_SELF_WEIGHT
            + self.coherence * QUALITY_COHERENCE_WEIGHT)
            .clamp(0.0, 1.0)
    }

    /// Check if consciousness meets threshold for action type
    pub fn meets_threshold(&self, action_type: &GovernanceActionType) -> bool {
        self.consciousness_level >= action_type.consciousness_gate()
    }
}

/// Type of governance action with associated consciousness gate thresholds
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernanceActionType {
    /// Basic participation: consciousness >= 0.2
    Basic,
    /// Proposal submission: consciousness >= 0.3
    ProposalSubmission,
    /// Voting: consciousness >= 0.4
    Voting,
    /// Constitutional changes: consciousness >= 0.6
    Constitutional,
}

impl GovernanceActionType {
    pub fn consciousness_gate(&self) -> f64 {
        match self {
            Self::Basic => GOV_BASIC,
            Self::ProposalSubmission => GOV_PROPOSAL,
            Self::Voting => GOV_VOTING,
            Self::Constitutional => GOV_CONSTITUTIONAL,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Basic => "Basic participation",
            Self::ProposalSubmission => "Proposal submission",
            Self::Voting => "Voting on proposals",
            Self::Constitutional => "Constitutional changes",
        }
    }
}

/// Multi-dimensional consciousness vector stored on DHT (v2 attestations).
///
/// Parallel struct to `symthaea_mycelix_bridge::ConsciousnessVector` — governance
/// zomes cannot depend on the bridge crate, so this is defined independently.
/// Each dimension is `Option<f64>` for incremental population.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConsciousnessVectorEntry {
    /// Spectral connectivity (Fiedler value) [0,1]. NOT IIT Phi.
    pub spectral_connectivity: Option<f64>,
    /// True IIT Phi [0,1]. Expensive: only computed for small component counts.
    pub true_phi: Option<f64>,
    /// Fast Phi approximation via effective information [0,1].
    pub phi_fast: Option<f64>,
    /// Shannon entropy of state space [0,1] (normalized).
    pub entropy: Option<f64>,
    /// Internal coherence [0,1].
    pub coherence: Option<f64>,
    /// Epistemic confidence from validation metrics [0,1].
    pub epistemic_confidence: Option<f64>,
}

impl ConsciousnessVectorEntry {
    /// Best available phi: true_phi > phi_fast > spectral_connectivity.
    pub fn best_phi(&self) -> f64 {
        self.true_phi
            .or(self.phi_fast)
            .or(self.spectral_connectivity)
            .unwrap_or(0.0)
    }

    /// Weighted composite matching bridge crate weights.
    /// Weights: phi=0.35, coherence=0.20, entropy=0.15, epistemic=0.15, spectral=0.15
    pub fn composite(&self) -> f64 {
        let mut total = 0.0;
        let mut weight_sum = 0.0;

        if let Some(v) = self.true_phi.or(self.phi_fast) {
            total += 0.35 * v;
            weight_sum += 0.35;
        }
        if let Some(v) = self.coherence {
            total += 0.20 * v;
            weight_sum += 0.20;
        }
        if let Some(v) = self.entropy {
            total += 0.15 * v;
            weight_sum += 0.15;
        }
        if let Some(v) = self.epistemic_confidence {
            total += 0.15 * v;
            weight_sum += 0.15;
        }
        if let Some(v) = self.spectral_connectivity {
            total += 0.15 * v;
            weight_sum += 0.15;
        }

        if weight_sum > 0.0 {
            (total / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Authenticated consciousness attestation with agent signature.
///
/// Unlike `ConsciousnessSnapshot`, this proves the consciousness level came from
/// the agent's own Symthaea instance via a signed attestation.
/// Governance prefers this over unsigned snapshots.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessAttestation {
    /// Agent DID
    pub agent_did: String,
    /// Consciousness level [0.0, 1.0] — scalar for backward compatibility
    pub consciousness_level: f64,
    /// Symthaea cognitive cycle number
    pub cycle_id: u64,
    /// Capture timestamp
    pub captured_at: Timestamp,
    /// Agent-signed hash of (agent_did, consciousness_level, cycle_id, captured_at)
    pub signature: Vec<u8>,
    /// Source system — validated to be "symthaea"
    pub source: String,
    /// Optional C-Vector (v2 attestations). When present, `consciousness_level`
    /// equals `consciousness_vector.composite()`. Signature binds to the scalar
    /// for backward-compatible verification.
    #[serde(default)]
    pub consciousness_vector: Option<ConsciousnessVectorEntry>,
}

/// How the consciousness value in a governance decision was obtained.
///
/// Governance tracks provenance so UIs can show transparency about
/// which votes had verified consciousness data vs. reputation-only.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessProvenance {
    /// Signed attestation from agent's Symthaea instance
    Attested,
    /// Legacy unsigned ConsciousnessSnapshot
    Snapshot,
    /// No consciousness data available — reputation-only voting
    Unavailable,
}

/// Consciousness gate verification result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessGate {
    /// Unique gate verification ID
    pub id: String,
    /// Agent DID
    pub agent_did: String,
    /// Action being gated
    pub action_type: GovernanceActionType,
    /// Reference to consciousness snapshot used
    pub snapshot_id: String,
    /// Φ value at verification
    pub consciousness_at_verification: f64,
    /// Required Φ threshold
    pub required_consciousness: f64,
    /// Whether gate passed
    pub passed: bool,
    /// Failure reason (if not passed)
    pub failure_reason: Option<String>,
    /// Related proposal/action ID
    pub action_id: Option<String>,
    /// Verification timestamp
    pub verified_at: Timestamp,
}

/// Agent consciousness history record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessHistory {
    /// Agent DID
    pub agent_did: String,
    /// Average Φ over recording period
    pub average_consciousness: f64,
    /// Peak Φ achieved
    pub peak_consciousness: f64,
    /// Minimum Φ recorded
    pub min_consciousness: f64,
    /// Standard deviation of Φ
    pub consciousness_std_dev: f64,
    /// Total snapshots recorded
    pub snapshot_count: u64,
    /// Period start
    pub period_start: Timestamp,
    /// Period end
    pub period_end: Timestamp,
    /// Trend direction: positive, negative, or stable
    pub trend: ConsciousnessTrend,
    /// Last updated
    pub updated_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessTrend {
    /// Φ increasing over time
    Increasing,
    /// Φ stable
    Stable,
    /// Φ decreasing over time
    Decreasing,
    /// Insufficient data
    Unknown,
}

/// Value alignment assessment linking consciousness to voting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ValueAlignmentAssessment {
    /// Unique assessment ID
    pub id: String,
    /// Agent DID
    pub agent_did: String,
    /// Proposal ID being assessed
    pub proposal_id: String,
    /// Overall alignment score (-1.0 to +1.0)
    pub overall_alignment: f64,
    /// Individual harmony scores
    pub harmony_scores: Vec<HarmonyScore>,
    /// Authenticity score (genuine caring check)
    pub authenticity: f64,
    /// Violations detected
    pub violations: Vec<String>,
    /// Recommendation
    pub recommendation: GovernanceRecommendation,
    /// Consciousness snapshot used
    pub snapshot_id: String,
    /// Assessment timestamp
    pub assessed_at: Timestamp,
}

/// Score for a single harmony dimension
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonyScore {
    pub harmony: String,
    pub score: f64,
}

/// Governance recommendation from value alignment
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
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

/// Consciousness-gated governance event
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConsciousnessEventType {
    /// Snapshot recorded
    SnapshotRecorded,
    /// Gate verification performed
    GateVerified,
    /// Gate failed (insufficient Φ)
    GateFailed,
    /// Value alignment assessed
    AlignmentAssessed,
    /// Consciousness history updated
    HistoryUpdated,
    /// Threshold adjusted (by constitutional process)
    ThresholdAdjusted,
}

// =============================================================================
// RB-BFT CONSENSUS INTEGRATION (Reputation-Based Byzantine Fault Tolerance)
// =============================================================================

/// K-Vector: 8-dimensional trust scoring model
/// Each component ranges from 0.0 to 1.0
/// Combined with consciousness Φ for holistic agent assessment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KVector {
    /// Agent DID
    pub agent_did: String,
    /// k_r: Reputation score (weight: 0.25) - historical behavior
    pub k_r: f64,
    /// k_a: Activity score (weight: 0.15) - recent participation
    pub k_a: f64,
    /// k_i: Integrity score (weight: 0.20) - consistency and honesty
    pub k_i: f64,
    /// k_p: Performance score (weight: 0.15) - quality of contributions
    pub k_p: f64,
    /// k_m: Membership duration (weight: 0.05) - time in network
    pub k_m: f64,
    /// k_s: Stake weight (weight: 0.10) - economic commitment
    pub k_s: f64,
    /// k_h: Historical consistency (weight: 0.05) - long-term patterns
    pub k_h: f64,
    /// k_topo: Network topology contribution (weight: 0.05) - connectivity
    pub k_topo: f64,
    /// Last updated timestamp
    pub updated_at: Timestamp,
}

impl KVector {
    /// Calculate trust score: weighted sum of all components
    /// T = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p + 0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo
    pub fn trust_score(&self) -> f64 {
        (0.25 * self.k_r
            + 0.15 * self.k_a
            + 0.20 * self.k_i
            + 0.15 * self.k_p
            + 0.05 * self.k_m
            + 0.10 * self.k_s
            + 0.05 * self.k_h
            + 0.05 * self.k_topo)
            .clamp(0.0, 1.0)
    }

    /// Get reputation component (primary for consensus weight)
    pub fn reputation(&self) -> f64 {
        self.k_r
    }

    /// Calculate voting weight using reputation² (enables 45% Byzantine tolerance)
    pub fn voting_weight(&self) -> f64 {
        self.k_r.powi(2)
    }

    /// Create a new K-Vector with default values (0.5 for new participants)
    pub fn new_participant(agent_did: String, now: Timestamp) -> Self {
        Self {
            agent_did,
            k_r: 0.5,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.1, // New member starts low
            k_s: 0.0, // No stake yet
            k_h: 0.5,
            k_topo: 0.3,
            updated_at: now,
        }
    }

    /// Check if agent meets minimum participation threshold (0.1)
    pub fn can_participate(&self) -> bool {
        self.k_r >= 0.1
    }
}

/// MATL Bridge: Multi-factor trust calculation
/// T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MatlTrustScore {
    /// Agent DID
    pub agent_did: String,
    /// PoGQ score: Proof of Gradient Quality (federated learning contribution)
    pub pogq_score: f64,
    /// TCDM score: Trust-Confidence-Decay Model (time-weighted trust)
    pub tcdm_score: f64,
    /// Entropy score: Diversity and unpredictability of contributions
    pub entropy_score: f64,
    /// Combined MATL trust score
    pub matl_score: f64,
    /// K-Vector trust score for comparison
    pub k_vector_score: f64,
    /// Consciousness Φ for holistic assessment
    pub phi: f64,
    /// Calculated at timestamp
    pub calculated_at: Timestamp,
}

impl MatlTrustScore {
    /// MATL formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
    pub fn calculate(pogq: f64, tcdm: f64, entropy: f64) -> f64 {
        (0.4 * pogq + 0.3 * tcdm + 0.3 * entropy).clamp(0.0, 1.0)
    }

    /// Combined holistic score incorporating consciousness
    /// Holistic = 0.5×MATL + 0.3×K-Vector + 0.2×Φ
    pub fn holistic_score(&self) -> f64 {
        (0.5 * self.matl_score + 0.3 * self.k_vector_score + 0.2 * self.phi).clamp(0.0, 1.0)
    }
}

// =============================================================================
// HOLISTIC VOTING WEIGHT SYSTEM
// =============================================================================

/// Maximum voting weight cap to prevent plutocratic dominance
/// Even with perfect reputation (1.0), max Φ (1.0), and max alignment (1.0),
/// no single vote can exceed this weight
pub const MAX_VOTING_WEIGHT: f64 = 1.5;

/// Consciousness multiplier coefficients for HolisticVotingWeight.
/// Formula: consciousness_multiplier = BASE + PHI_FACTOR × Φ
/// Range: BASE (Φ=0) to BASE + PHI_FACTOR (Φ=1)
pub const CONSCIOUSNESS_BASE: f64 = 0.7;
pub const CONSCIOUSNESS_PHI_FACTOR: f64 = 0.3;

/// Harmonic alignment bonus coefficient for HolisticVotingWeight.
/// Formula: harmonic_bonus = 1.0 + HARMONIC_FACTOR × max(0, alignment)
/// Range: 1.0 (no alignment) to 1.0 + HARMONIC_FACTOR (perfect alignment)
pub const HARMONIC_ALIGNMENT_FACTOR: f64 = 0.2;

/// Quality score weights for ConsciousnessSnapshot.
/// Formula: quality = PHI_WEIGHT×Φ + META_WEIGHT×meta + SELF_WEIGHT×self + COHERENCE_WEIGHT×coherence
pub const QUALITY_CONSCIOUSNESS_WEIGHT: f64 = 0.4;
pub const QUALITY_META_WEIGHT: f64 = 0.2;
pub const QUALITY_SELF_WEIGHT: f64 = 0.2;
pub const QUALITY_COHERENCE_WEIGHT: f64 = 0.2;

/// Holistic voting weight combining consciousness, reputation, and harmonic alignment
/// Formula: HolisticWeight = Reputation² × (0.7 + 0.3 × Φ) × (1 + 0.2 × HarmonicAlignment)
///
/// ## Weight Cap Enforcement
///
/// The final weight is capped at MAX_VOTING_WEIGHT (1.5) to prevent any single
/// voter from having disproportionate influence, regardless of reputation or
/// consciousness level. This ensures democratic balance per the Governance Charter.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HolisticVotingWeight {
    /// Base reputation (k_r from K-Vector)
    pub reputation: f64,
    /// Reputation squared component
    pub reputation_squared: f64,
    /// Consciousness level (0.0 - 1.0)
    pub consciousness_level: f64,
    /// Consciousness multiplier: 0.7 + 0.3 x consciousness_level (range: 0.7 - 1.0)
    pub consciousness_multiplier: f64,
    /// Harmonic alignment with proposal (-1.0 to +1.0)
    pub harmonic_alignment: f64,
    /// Harmonic bonus: 1 + 0.2 × max(0, alignment) (range: 1.0 - 1.2)
    pub harmonic_bonus: f64,
    /// Final holistic weight (capped at MAX_VOTING_WEIGHT)
    pub final_weight: f64,
    /// Whether the weight was capped
    pub was_capped: bool,
    /// Uncapped weight (for transparency)
    pub uncapped_weight: f64,
    /// Breakdown explanation
    pub calculation_breakdown: String,
}

impl HolisticVotingWeight {
    /// Calculate holistic voting weight from components
    /// Formula: Reputation² × (0.7 + 0.3 × Φ) × (1 + 0.2 × max(0, HarmonicAlignment))
    ///
    /// ## Weight Cap
    ///
    /// The final weight is capped at MAX_VOTING_WEIGHT (1.5) to ensure no single
    /// voter can have disproportionate influence. This cap is enforced regardless
    /// of input values.
    pub fn calculate(reputation: f64, consciousness_level: f64, harmonic_alignment: f64) -> Self {
        let reputation = reputation.clamp(0.0, 1.0);
        let consciousness_level = consciousness_level.clamp(0.0, 1.0);
        let harmonic_alignment = harmonic_alignment.clamp(-1.0, 1.0);

        let reputation_squared = reputation.powi(2);

        // Consciousness multiplier: ranges from CONSCIOUSNESS_BASE (level=0) to
        // CONSCIOUSNESS_BASE + CONSCIOUSNESS_PHI_FACTOR (level=1)
        // This gives high-consciousness voters up to 43% more influence
        let consciousness_multiplier = CONSCIOUSNESS_BASE + CONSCIOUSNESS_PHI_FACTOR * consciousness_level;

        // Harmonic bonus: only positive alignment gives bonus
        // Negative alignment gives no penalty here (handled in threshold)
        let harmonic_bonus = 1.0 + HARMONIC_ALIGNMENT_FACTOR * harmonic_alignment.max(0.0);

        let uncapped_weight = reputation_squared * consciousness_multiplier * harmonic_bonus;

        // Enforce maximum voting weight cap
        let was_capped = uncapped_weight > MAX_VOTING_WEIGHT;
        let final_weight = uncapped_weight.min(MAX_VOTING_WEIGHT);

        let cap_note = if was_capped {
            format!(" [CAPPED from {:.4}]", uncapped_weight)
        } else {
            String::new()
        };

        let calculation_breakdown = format!(
            "{:.3}² × ({} + {}×{:.3}) × (1 + {}×{:.3}) = {:.4} × {:.4} × {:.4} = {:.4}{}",
            reputation,
            CONSCIOUSNESS_BASE,
            CONSCIOUSNESS_PHI_FACTOR,
            consciousness_level,
            HARMONIC_ALIGNMENT_FACTOR,
            harmonic_alignment.max(0.0),
            reputation_squared,
            consciousness_multiplier,
            harmonic_bonus,
            final_weight,
            cap_note
        );

        Self {
            reputation,
            reputation_squared,
            consciousness_level,
            consciousness_multiplier,
            harmonic_alignment,
            harmonic_bonus,
            final_weight,
            was_capped,
            uncapped_weight,
            calculation_breakdown,
        }
    }

    /// Calculate weight without harmonic alignment (for general participation)
    pub fn calculate_base(reputation: f64, consciousness_level: f64) -> Self {
        Self::calculate(reputation, consciousness_level, 0.0)
    }

    /// Calculate weight from a C-Vector, using its composite as consciousness_level.
    pub fn calculate_from_vector(
        reputation: f64,
        vector: &ConsciousnessVectorEntry,
        harmonic_alignment: f64,
    ) -> Self {
        Self::calculate(reputation, vector.composite(), harmonic_alignment)
    }

    /// Check if a weight was capped
    pub fn is_capped(&self) -> bool {
        self.was_capped
    }

    /// Get the maximum allowed voting weight
    pub fn max_weight() -> f64 {
        MAX_VOTING_WEIGHT
    }
}

/// Adaptive consensus thresholds based on proposal type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveThreshold {
    /// Base threshold percentage (0.0 - 1.0)
    pub base_threshold: f64,
    /// Minimum Φ required for voters
    pub min_voter_consciousness: f64,
    /// Minimum participation (number of voters)
    pub min_participation: u64,
    /// Quorum percentage (minimum % of eligible voters)
    pub quorum: f64,
    /// Time extension allowed (in seconds)
    pub max_extension_secs: u64,
}

impl AdaptiveThreshold {
    /// Get threshold configuration for proposal type
    pub fn for_proposal_type(proposal_type: &ProposalType) -> Self {
        match proposal_type {
            ProposalType::Standard => Self {
                base_threshold: 0.51,      // Simple majority
                min_voter_consciousness: GOV_BASIC,  // Basic consciousness
                min_participation: 5,      // At least 5 voters
                quorum: 0.20,              // 20% of eligible must vote
                max_extension_secs: 86400, // 1 day extension max
            },
            ProposalType::Emergency => Self {
                base_threshold: 0.60,      // Higher bar for emergency
                min_voter_consciousness: GOV_PROPOSAL, // Proposal-level consciousness
                min_participation: 3,      // Faster with fewer
                quorum: 0.10,              // Lower quorum for speed
                max_extension_secs: 3600,  // 1 hour max
            },
            ProposalType::Constitutional => Self {
                base_threshold: 0.67,      // Supermajority required
                min_voter_consciousness: GOV_VOTER_CONSTITUTIONAL, // High consciousness required
                min_participation: 10,     // Significant participation
                quorum: 0.40,              // 40% must participate
                max_extension_secs: 604800, // 1 week extension allowed
            },
        }
    }

    /// Calculate actual threshold for a given total weight
    pub fn calculate_threshold(&self, total_weight: f64) -> f64 {
        total_weight * self.base_threshold
    }

    /// Check if voter meets consciousness requirement
    pub fn voter_meets_consciousness_requirement(&self, voter_consciousness: f64) -> bool {
        voter_consciousness >= self.min_voter_consciousness
    }

    /// Check if quorum is met
    pub fn quorum_met(&self, voters: u64, eligible: u64) -> bool {
        if eligible == 0 {
            return false;
        }
        let participation_rate = voters as f64 / eligible as f64;
        participation_rate >= self.quorum && voters >= self.min_participation
    }
}

/// Cross-module federated reputation aggregating signals from multiple sources
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederatedReputation {
    /// Agent DID
    pub agent_did: String,

    // Identity Module Signals
    /// DID verification status (0.0 = unverified, 1.0 = fully verified)
    pub identity_verification: f64,
    /// Number of verifiable credentials held
    pub credential_count: u64,
    /// Credential quality score (weighted by issuer reputation)
    pub credential_quality: f64,

    // Knowledge Module Signals
    /// Epistemic contribution score (claims submitted and verified)
    pub epistemic_contributions: f64,
    /// Fact-check accuracy rate
    pub factcheck_accuracy: f64,
    /// Dark spots resolved
    pub dark_spots_resolved: u64,

    // Finance Module Signals
    /// Stake amount (normalized to 0-1)
    pub stake_weight: f64,
    /// Payment reliability score
    pub payment_reliability: f64,
    /// Escrow completion rate
    pub escrow_completion_rate: f64,

    // Federated Learning Signals (from Mycelix-Core)
    /// PoGQ score from gradient contributions
    pub pogq_score: f64,
    /// Model contribution count
    pub fl_contributions: u64,
    /// Byzantine detection clean rate
    pub byzantine_clean_rate: f64,

    // Governance History
    /// Voting participation rate
    pub voting_participation: f64,
    /// Proposal success rate (for submitted proposals)
    pub proposal_success_rate: f64,
    /// Consensus alignment rate
    pub consensus_alignment: f64,

    /// Aggregated federated reputation score
    pub aggregated_score: f64,
    /// Last aggregation timestamp
    pub aggregated_at: Timestamp,
}

// =============================================================================
// DOMAIN BOUNDARY CONSTANTS (Per Commons Charter v1.0)
// =============================================================================

/// Maximum influence from economic/finance domain (Commons Charter Article II)
/// TEND activity can contribute at most 5% to any cross-domain reputation
pub const DOMAIN_CAP_FINANCE: f64 = 0.05;

/// Maximum influence from FL domain
/// Technical contributions important but bounded
pub const DOMAIN_CAP_FL: f64 = 0.25;

/// Identity domain contribution
pub const DOMAIN_CAP_IDENTITY: f64 = 0.15;

/// Knowledge domain contribution (epistemic contributions)
pub const DOMAIN_CAP_KNOWLEDGE: f64 = 0.25;

/// Governance history contribution
pub const DOMAIN_CAP_GOVERNANCE: f64 = 0.30;

impl FederatedReputation {
    /// Calculate aggregated federated reputation from all signals
    ///
    /// ## Domain Boundary Enforcement (Commons Charter v1.0)
    ///
    /// Per Commons Charter Article II, economic activity (finance/TEND) is capped
    /// at 5% maximum contribution to prevent plutocratic influence on governance.
    ///
    /// Domain weights (sum to 100%):
    /// - Identity: 15% (verification, credentials)
    /// - Knowledge: 25% (epistemic contributions, fact-checking)
    /// - Finance: 5% (stake, payments) - CAPPED per Commons Charter
    /// - FL: 25% (PoGQ, contributions, Byzantine detection)
    /// - Governance: 30% (voting history, proposal success)
    pub fn calculate_aggregated(&self) -> f64 {
        // Identity score (15% max contribution)
        let identity_score = (
            self.identity_verification * 0.5 +
            (self.credential_count.min(10) as f64 / 10.0) * 0.25 +
            self.credential_quality * 0.25
        ).clamp(0.0, 1.0);

        // Knowledge score (25% max contribution)
        let knowledge_score = (
            self.epistemic_contributions * 0.4 +
            self.factcheck_accuracy * 0.4 +
            (self.dark_spots_resolved.min(20) as f64 / 20.0) * 0.2
        ).clamp(0.0, 1.0);

        // Finance score (5% max - DOMAIN BOUNDARY ENFORCED)
        // Per Commons Charter, economic activity cannot dominate governance
        let finance_score = (
            self.stake_weight * 0.4 +
            self.payment_reliability * 0.3 +
            self.escrow_completion_rate * 0.3
        ).clamp(0.0, 1.0);

        // Federated Learning score (25% max contribution)
        let fl_score = (
            self.pogq_score * 0.5 +
            (self.fl_contributions.min(100) as f64 / 100.0) * 0.25 +
            self.byzantine_clean_rate * 0.25
        ).clamp(0.0, 1.0);

        // Governance score (30% max contribution)
        let governance_score = (
            self.voting_participation * 0.3 +
            self.proposal_success_rate * 0.3 +
            self.consensus_alignment * 0.4
        ).clamp(0.0, 1.0);

        // Weighted aggregation with domain boundary caps enforced
        (
            identity_score * DOMAIN_CAP_IDENTITY +
            knowledge_score * DOMAIN_CAP_KNOWLEDGE +
            finance_score * DOMAIN_CAP_FINANCE +      // 5% cap enforced
            fl_score * DOMAIN_CAP_FL +
            governance_score * DOMAIN_CAP_GOVERNANCE
        ).clamp(0.0, 1.0)
    }

    /// Create initial federated reputation for new participant
    pub fn new_participant(agent_did: String, now: Timestamp) -> Self {
        Self {
            agent_did,
            identity_verification: 0.0,
            credential_count: 0,
            credential_quality: 0.0,
            epistemic_contributions: 0.0,
            factcheck_accuracy: 0.5, // Neutral starting point
            dark_spots_resolved: 0,
            stake_weight: 0.0,
            payment_reliability: 0.5,
            escrow_completion_rate: 0.5,
            pogq_score: 0.5,
            fl_contributions: 0,
            byzantine_clean_rate: 1.0, // Clean until proven otherwise
            voting_participation: 0.0,
            proposal_success_rate: 0.5,
            consensus_alignment: 0.5,
            aggregated_score: 0.25, // Conservative starting score
            aggregated_at: now,
        }
    }

    /// Update aggregated score
    pub fn refresh_aggregation(&mut self, now: Timestamp) {
        self.aggregated_score = self.calculate_aggregated();
        self.aggregated_at = now;
    }
}

/// Consensus participant (validator) with reputation metrics
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsensusParticipant {
    /// Agent DID
    pub agent_did: String,
    /// K-Vector ID reference
    pub k_vector_id: String,
    /// Federated reputation ID reference
    pub federated_rep_id: Option<String>,
    /// Current reputation (k_r from K-Vector)
    pub reputation: f64,
    /// Voting weight (reputation²)
    pub voting_weight: f64,
    /// MATL trust score
    pub matl_score: f64,
    /// Consciousness Φ at registration
    pub phi: f64,
    /// Federated reputation score
    pub federated_score: f64,
    /// Active status
    pub is_active: bool,
    /// Rounds participated
    pub rounds_participated: u64,
    /// Successful votes (aligned with consensus)
    pub successful_votes: u64,
    /// Slashing events count
    pub slashing_events: u64,
    /// Last slashing timestamp (for cool-down)
    pub last_slashing_at: Option<Timestamp>,
    /// Consecutive good rounds (for streak bonus)
    pub streak_count: u64,
    /// Registered at
    pub registered_at: Timestamp,
    /// Last active at
    pub last_active_at: Timestamp,
}

impl ConsensusParticipant {
    /// Success rate: successful_votes / rounds_participated
    pub fn success_rate(&self) -> f64 {
        if self.rounds_participated == 0 {
            0.5 // Default for new participants
        } else {
            self.successful_votes as f64 / self.rounds_participated as f64
        }
    }

    /// Check if participant can vote (active + reputation >= 0.1)
    pub fn can_vote(&self) -> bool {
        self.is_active && self.reputation >= 0.1
    }

    /// Check if participant can vote on proposal type (meets Φ threshold)
    pub fn can_vote_on(&self, proposal_type: &ProposalType) -> bool {
        if !self.can_vote() {
            return false;
        }
        let threshold = AdaptiveThreshold::for_proposal_type(proposal_type);
        threshold.voter_meets_consciousness_requirement(self.phi)
    }

    /// Calculate holistic voting weight for a specific proposal
    pub fn calculate_holistic_weight(&self, harmonic_alignment: f64) -> HolisticVotingWeight {
        HolisticVotingWeight::calculate(self.reputation, self.phi, harmonic_alignment)
    }

    /// Calculate streak bonus (up to 10% bonus for 10+ consecutive good rounds)
    pub fn streak_bonus(&self) -> f64 {
        let capped_streak = self.streak_count.min(10) as f64;
        1.0 + (capped_streak * 0.01) // 1% per streak, max 10%
    }

    /// Check if in slashing cool-down period (24 hours after last slash)
    pub fn in_cooldown(&self, now: Timestamp) -> bool {
        if let Some(last_slash) = self.last_slashing_at {
            let cooldown_micros: i64 = 24 * 60 * 60 * 1_000_000; // 24 hours
            let elapsed = now.as_micros() - last_slash.as_micros();
            elapsed < cooldown_micros
        } else {
            false
        }
    }

    /// Get effective reputation (with streak bonus, reduced if in cooldown)
    pub fn effective_reputation(&self, now: Timestamp) -> f64 {
        let base = self.reputation;
        if self.in_cooldown(now) {
            base * 0.8 // 20% penalty during cooldown
        } else {
            base * self.streak_bonus()
        }
    }
}

/// Weighted vote in RB-BFT consensus
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WeightedVote {
    /// Vote ID
    pub id: String,
    /// Proposal ID being voted on
    pub proposal_id: String,
    /// Consensus round number
    pub round: u64,
    /// Voter agent DID
    pub voter_did: String,
    /// Vote decision
    pub decision: VoteDecision,
    /// Voter's reputation at time of vote
    pub reputation: f64,
    /// Vote weight (reputation²)
    pub weight: f64,
    /// Voter's Φ at time of vote
    pub phi: f64,
    /// Optional reason for vote
    pub reason: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
    /// Signature (placeholder for cryptographic signature)
    pub signature: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteDecision {
    /// Approve the proposal
    Approve,
    /// Reject the proposal
    Reject,
    /// Abstain from voting
    Abstain,
}

/// Consensus round state
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsensusRound {
    /// Round number
    pub round: u64,
    /// Proposal ID
    pub proposal_id: String,
    /// Current state
    pub state: RoundState,
    /// Leader for this round (proposer)
    pub leader_did: String,
    /// Total voting weight of all active participants
    pub total_weight: f64,
    /// Consensus threshold (55% of total weight)
    pub threshold: f64,
    /// Weighted approvals accumulated
    pub weighted_approvals: f64,
    /// Weighted rejections accumulated
    pub weighted_rejections: f64,
    /// Number of votes cast
    pub vote_count: u64,
    /// Round start time
    pub started_at: Timestamp,
    /// Round end time (if finished)
    pub ended_at: Option<Timestamp>,
    /// Timeout in milliseconds (default: 30000)
    pub timeout_ms: u64,
    /// Result hash (if committed)
    pub result_hash: Option<String>,
}

impl ConsensusRound {
    /// Check if consensus has been reached (approvals > threshold)
    pub fn consensus_reached(&self) -> bool {
        self.weighted_approvals > self.threshold
    }

    /// Check if proposal was rejected (rejections > threshold)
    pub fn rejected(&self) -> bool {
        self.weighted_rejections > self.threshold
    }

    /// Get approval percentage
    pub fn approval_percentage(&self) -> f64 {
        if self.total_weight == 0.0 {
            0.0
        } else {
            (self.weighted_approvals / self.total_weight) * 100.0
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundState {
    /// Round not yet started
    NotStarted,
    /// Waiting for proposal from leader
    WaitingForProposal,
    /// Voting in progress
    Voting,
    /// Consensus reached, committed
    Committed,
    /// Round failed (timeout or rejection)
    Failed,
    /// Round skipped (leader unavailable)
    Skipped,
}

/// Slashing record for Byzantine behavior
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SlashingRecord {
    /// Record ID
    pub id: String,
    /// Offender agent DID
    pub offender_did: String,
    /// Type of offense
    pub offense: SlashingOffense,
    /// Severity level
    pub severity: SlashingSeverity,
    /// Reputation penalty applied (0.0 - 1.0)
    pub penalty: f64,
    /// Reputation before slashing
    pub reputation_before: f64,
    /// Reputation after slashing
    pub reputation_after: f64,
    /// Evidence (proposal IDs, vote hashes, etc.)
    pub evidence: Vec<String>,
    /// Round number where offense occurred
    pub round: u64,
    /// Reporter agent DID
    pub reporter_did: String,
    /// Slashing timestamp
    pub slashed_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashingOffense {
    /// Voted differently in same round
    DoubleVoting,
    /// Submitted multiple proposals in same round
    DoubleProposing,
    /// Conflicting commit signatures
    ConflictingCommit,
    /// Consistent unjustified rejections
    MaliciousRejection,
    /// Failed to participate in required rounds
    Inactivity,
    /// Submitted invalid proposals repeatedly
    InvalidProposals,
    /// Attempted to manipulate reputation scores
    ReputationManipulation,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashingSeverity {
    /// Minor: -10% reputation
    Minor,
    /// Moderate: -30% reputation
    Moderate,
    /// Severe: -50% reputation
    Severe,
    /// Critical: -80% reputation
    Critical,
}

impl SlashingSeverity {
    /// Get penalty multiplier (reputation *= 1 - penalty)
    pub fn penalty(&self) -> f64 {
        match self {
            Self::Minor => 0.1,
            Self::Moderate => 0.3,
            Self::Severe => 0.5,
            Self::Critical => 0.8,
        }
    }
}

/// Consensus event for audit trail
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConsensusEventType {
    /// New participant registered
    ParticipantRegistered,
    /// Participant deactivated (low reputation)
    ParticipantDeactivated,
    /// Round started
    RoundStarted,
    /// Vote cast
    VoteCast,
    /// Consensus reached
    ConsensusReached,
    /// Round failed
    RoundFailed,
    /// Slashing applied
    SlashingApplied,
    /// K-Vector updated
    KVectorUpdated,
    /// MATL score recalculated
    MatlRecalculated,
    /// Consciousness configuration updated
    ConsciousnessConfigUpdated,
}

// =============================================================================
// CONFIGURABLE PHI PARAMETERS
// =============================================================================

/// Configurable consciousness gate thresholds and voting parameters.
///
/// Stored as a DHT entry so governance proposals can adjust thresholds
/// without code deployments. The hardcoded defaults in `GovernanceActionType`
/// and `AdaptiveThreshold` serve as compile-time documentation; at runtime,
/// the coordinator reads this config with fallback to those defaults.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceConsciousnessConfig {
    // --- GovernanceActionType thresholds ---
    /// Consciousness gate for Basic participation (default 0.2)
    pub consciousness_gate_basic: f64,
    /// Consciousness gate for Proposal submission (default 0.3)
    pub consciousness_gate_proposal: f64,
    /// Consciousness gate for Voting (default 0.4)
    pub consciousness_gate_voting: f64,
    /// Consciousness gate for Constitutional changes (default 0.6)
    pub consciousness_gate_constitutional: f64,

    // --- AdaptiveThreshold min_voter_consciousness per proposal type ---
    /// Min voter consciousness for Standard proposals (default 0.2)
    pub min_voter_consciousness_standard: f64,
    /// Min voter consciousness for Emergency proposals (default 0.3)
    pub min_voter_consciousness_emergency: f64,
    /// Min voter consciousness for Constitutional proposals (default 0.5)
    pub min_voter_consciousness_constitutional: f64,

    // --- Voting weight parameters ---
    /// Maximum voting weight cap (default 1.5) — per Governance Charter,
    /// no single voter can exceed this weight to prevent disproportionate influence.
    #[serde(default = "default_max_voting_weight")]
    pub max_voting_weight: f64,

    // --- Per-dimension C-Vector gates (optional) ---
    /// Minimum true_phi for constitutional changes. If `None`, only composite gate applies.
    #[serde(default)]
    pub min_true_phi_constitutional: Option<f64>,
    /// Minimum coherence for voting. If `None`, only composite gate applies.
    #[serde(default)]
    pub min_coherence_voting: Option<f64>,

    // --- Metadata ---
    /// Timestamp of last update
    pub updated_at: Timestamp,
    /// Proposal ID that authorized this change (None = bootstrap)
    pub changed_by_proposal: Option<String>,
}

fn default_max_voting_weight() -> f64 { 1.5 }

impl GovernanceConsciousnessConfig {
    /// Default configuration matching hardcoded values
    pub fn defaults(now: Timestamp) -> Self {
        Self {
            consciousness_gate_basic: GOV_BASIC,
            consciousness_gate_proposal: GOV_PROPOSAL,
            consciousness_gate_voting: GOV_VOTING,
            consciousness_gate_constitutional: GOV_CONSTITUTIONAL,
            min_voter_consciousness_standard: GOV_BASIC,
            min_voter_consciousness_emergency: GOV_PROPOSAL,
            min_voter_consciousness_constitutional: GOV_VOTER_CONSTITUTIONAL,
            max_voting_weight: 1.5,
            min_true_phi_constitutional: None,
            min_coherence_voting: None,
            updated_at: now,
            changed_by_proposal: None,
        }
    }

    /// Get the consciousness gate threshold for an action type from this config
    pub fn consciousness_gate_for(&self, action_type: &GovernanceActionType) -> f64 {
        match action_type {
            GovernanceActionType::Basic => self.consciousness_gate_basic,
            GovernanceActionType::ProposalSubmission => self.consciousness_gate_proposal,
            GovernanceActionType::Voting => self.consciousness_gate_voting,
            GovernanceActionType::Constitutional => self.consciousness_gate_constitutional,
        }
    }

    /// Get the min voter Phi for a proposal type from this config
    pub fn min_voter_consciousness_for(&self, proposal_type: &ProposalType) -> f64 {
        match proposal_type {
            ProposalType::Standard => self.min_voter_consciousness_standard,
            ProposalType::Emergency => self.min_voter_consciousness_emergency,
            ProposalType::Constitutional => self.min_voter_consciousness_constitutional,
        }
    }
}

/// Validate GovernanceConsciousnessConfig — pure function for testing
pub fn check_consciousness_config(config: &GovernanceConsciousnessConfig) -> Result<(), String> {
    let fields = [
        ("consciousness_gate_basic", config.consciousness_gate_basic),
        ("consciousness_gate_proposal", config.consciousness_gate_proposal),
        ("consciousness_gate_voting", config.consciousness_gate_voting),
        ("consciousness_gate_constitutional", config.consciousness_gate_constitutional),
        ("min_voter_consciousness_standard", config.min_voter_consciousness_standard),
        ("min_voter_consciousness_emergency", config.min_voter_consciousness_emergency),
        ("min_voter_consciousness_constitutional", config.min_voter_consciousness_constitutional),
    ];
    for (name, value) in &fields {
        if *value < 0.0 || *value > 1.0 {
            return Err(format!("{} must be between 0.0 and 1.0", name));
        }
    }
    // Consciousness gate thresholds must be monotonically non-decreasing
    if config.consciousness_gate_basic > config.consciousness_gate_proposal {
        return Err("consciousness_gate_basic must be <= consciousness_gate_proposal".into());
    }
    if config.consciousness_gate_proposal > config.consciousness_gate_voting {
        return Err("consciousness_gate_proposal must be <= consciousness_gate_voting".into());
    }
    if config.consciousness_gate_voting > config.consciousness_gate_constitutional {
        return Err("consciousness_gate_voting must be <= consciousness_gate_constitutional".into());
    }
    // Voter consciousness thresholds must be non-decreasing
    if config.min_voter_consciousness_standard > config.min_voter_consciousness_emergency {
        return Err("min_voter_consciousness_standard must be <= min_voter_consciousness_emergency".into());
    }
    if config.min_voter_consciousness_emergency > config.min_voter_consciousness_constitutional {
        return Err("min_voter_consciousness_emergency must be <= min_voter_consciousness_constitutional".into());
    }
    // Max voting weight must be positive and reasonable (0.5 to 5.0)
    if config.max_voting_weight < 0.5 || config.max_voting_weight > 5.0 {
        return Err("max_voting_weight must be between 0.5 and 5.0".into());
    }
    // Per-dimension C-Vector gates (optional)
    if let Some(v) = config.min_true_phi_constitutional {
        if v < 0.0 || v > 1.0 {
            return Err("min_true_phi_constitutional must be between 0.0 and 1.0".into());
        }
    }
    if let Some(v) = config.min_coherence_voting {
        if v < 0.0 || v > 1.0 {
            return Err("min_coherence_voting must be between 0.0 and 1.0".into());
        }
    }
    Ok(())
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    GovernanceQuery(GovernanceQuery),
    ProposalReference(ProposalReference),
    GovernanceBridgeEvent(GovernanceBridgeEvent),
    ExecutionRequest(ExecutionRequest),
    // Consciousness metrics entry types
    ConsciousnessSnapshot(ConsciousnessSnapshot),
    ConsciousnessGate(ConsciousnessGate),
    ConsciousnessHistory(ConsciousnessHistory),
    ValueAlignmentAssessment(ValueAlignmentAssessment),
    /// Authenticated consciousness attestation from Symthaea
    ConsciousnessAttestation(ConsciousnessAttestation),
    // RB-BFT Consensus entry types
    KVector(KVector),
    MatlTrustScore(MatlTrustScore),
    FederatedReputation(FederatedReputation),
    ConsensusParticipant(ConsensusParticipant),
    WeightedVote(WeightedVote),
    ConsensusRound(ConsensusRound),
    SlashingRecord(SlashingRecord),
    GovernanceConsciousnessConfig(GovernanceConsciousnessConfig),
}

#[hdk_link_types]
pub enum LinkTypes {
    ActiveProposals,
    RecentEvents,
    HappToExecutions,
    ProposalToExecutions,
    // Consciousness metrics link types
    AgentToSnapshots,
    AgentToGates,
    AgentToHistory,
    ProposalToAlignments,
    RecentSnapshots,
    /// Agent → their consciousness attestations
    AgentToAttestations,
    // RB-BFT Consensus link types
    AgentToKVector,
    AgentToMatlScore,
    AgentToFederatedRep,
    AgentToParticipant,
    ProposalToRound,
    RoundToVotes,
    AgentToVotes,
    AgentToSlashing,
    ActiveParticipants,
    ActiveRounds,
    /// O(1) lookup: execution ID anchor → execution request record
    ExecutionById,
    /// Consciousness config anchor → config record
    ConsciousnessConfigIndex,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::GovernanceQuery(query) => validate_create_query(action, query),
                EntryTypes::ProposalReference(proposal) => validate_create_proposal_ref(action, proposal),
                EntryTypes::GovernanceBridgeEvent(event) => validate_create_event(action, event),
                EntryTypes::ExecutionRequest(req) => validate_create_execution_req(action, req),
                // Consciousness metrics validations
                EntryTypes::ConsciousnessSnapshot(snapshot) => {
                    validate_create_consciousness_snapshot(action, snapshot)
                }
                EntryTypes::ConsciousnessGate(gate) => {
                    validate_create_consciousness_gate(action, gate)
                }
                EntryTypes::ConsciousnessHistory(history) => {
                    validate_create_consciousness_history(action, history)
                }
                EntryTypes::ValueAlignmentAssessment(assessment) => {
                    validate_create_value_alignment(action, assessment)
                }
                EntryTypes::ConsciousnessAttestation(attestation) => {
                    validate_create_consciousness_attestation(action, attestation)
                }
                // RB-BFT Consensus validations
                EntryTypes::KVector(k_vector) => validate_create_k_vector(action, k_vector),
                EntryTypes::MatlTrustScore(matl) => validate_create_matl_score(action, matl),
                EntryTypes::FederatedReputation(fed_rep) => {
                    validate_create_federated_reputation(action, fed_rep)
                }
                EntryTypes::ConsensusParticipant(participant) => {
                    validate_create_consensus_participant(action, participant)
                }
                EntryTypes::WeightedVote(vote) => validate_create_weighted_vote(action, vote),
                EntryTypes::ConsensusRound(round) => validate_create_consensus_round(action, round),
                EntryTypes::SlashingRecord(record) => validate_create_slashing_record(action, record),
                EntryTypes::GovernanceConsciousnessConfig(config) => {
                    match check_consciousness_config(&config) {
                        Ok(()) => Ok(ValidateCallbackResult::Valid),
                        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
                    }
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::GovernanceQuery(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ProposalReference(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::GovernanceBridgeEvent(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ExecutionRequest(req) => validate_update_execution_req(action, req),
                // Consciousness snapshots are immutable
                EntryTypes::ConsciousnessSnapshot(_) => Ok(ValidateCallbackResult::Invalid(
                    "Consciousness snapshots are immutable".into(),
                )),
                // Gates are immutable (verification is point-in-time)
                EntryTypes::ConsciousnessGate(_) => Ok(ValidateCallbackResult::Invalid(
                    "Consciousness gates are immutable".into(),
                )),
                // History can be updated (aggregated over time)
                EntryTypes::ConsciousnessHistory(_) => Ok(ValidateCallbackResult::Valid),
                // Assessments are immutable
                EntryTypes::ValueAlignmentAssessment(_) => Ok(ValidateCallbackResult::Invalid(
                    "Value alignment assessments are immutable".into(),
                )),
                // Phi attestations are immutable (point-in-time signed proof)
                EntryTypes::ConsciousnessAttestation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Consciousness attestations are immutable".into(),
                )),
                // RB-BFT Consensus update rules
                // K-Vectors can be updated (reputation changes over time)
                EntryTypes::KVector(_) => Ok(ValidateCallbackResult::Valid),
                // MATL scores can be updated (recalculated periodically)
                EntryTypes::MatlTrustScore(_) => Ok(ValidateCallbackResult::Valid),
                // Federated reputation can be updated (aggregated periodically)
                EntryTypes::FederatedReputation(_) => Ok(ValidateCallbackResult::Valid),
                // Participants can be updated (status, metrics)
                EntryTypes::ConsensusParticipant(_) => Ok(ValidateCallbackResult::Valid),
                // Votes are immutable (point-in-time decision)
                EntryTypes::WeightedVote(_) => Ok(ValidateCallbackResult::Invalid(
                    "Votes are immutable once cast".into(),
                )),
                // Rounds can be updated (state transitions)
                EntryTypes::ConsensusRound(_) => Ok(ValidateCallbackResult::Valid),
                // Slashing records are immutable (audit trail)
                EntryTypes::SlashingRecord(_) => Ok(ValidateCallbackResult::Invalid(
                    "Slashing records are immutable".into(),
                )),
                // Consciousness config can be updated (via governance proposal)
                EntryTypes::GovernanceConsciousnessConfig(config) => {
                    match check_consciousness_config(&config) {
                        Ok(()) => Ok(ValidateCallbackResult::Valid),
                        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
                    }
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ActiveProposals => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToExecutions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToExecutions => Ok(ValidateCallbackResult::Valid),
            // Consciousness metrics link types
            LinkTypes::AgentToSnapshots => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToGates => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToHistory => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToAlignments => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentSnapshots => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToAttestations => Ok(ValidateCallbackResult::Valid),
            // RB-BFT Consensus link types
            LinkTypes::AgentToKVector => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToMatlScore => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToFederatedRep => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToParticipant => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToRound => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RoundToVotes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToVotes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToSlashing => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveParticipants => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveRounds => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ExecutionById => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ConsciousnessConfigIndex => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ActiveProposals => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate governance query creation
fn validate_create_query(
    _action: Create,
    query: GovernanceQuery,
) -> ExternResult<ValidateCallbackResult> {
    if query.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Source hApp required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate proposal reference creation
fn validate_create_proposal_ref(
    _action: Create,
    _proposal: ProposalReference,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate governance event creation
fn validate_create_event(
    _action: Create,
    event: GovernanceBridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Source hApp required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate execution request creation
fn validate_create_execution_req(
    _action: Create,
    req: ExecutionRequest,
) -> ExternResult<ValidateCallbackResult> {
    if req.target_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Target hApp required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate execution request update
fn validate_update_execution_req(
    _action: Update,
    _req: ExecutionRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Status updates are allowed
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// CONSCIOUSNESS METRICS VALIDATION FUNCTIONS
// =============================================================================

/// Validate consciousness snapshot creation
fn validate_create_consciousness_snapshot(
    _action: Create,
    snapshot: ConsciousnessSnapshot,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID is present
    if snapshot.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Snapshot ID required".into(),
        ));
    }

    // Validate agent DID format
    if !snapshot.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate consciousness level is in range [0, 1]
    if snapshot.consciousness_level < 0.0 || snapshot.consciousness_level > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness level must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate meta_awareness is in range [0, 1]
    if snapshot.meta_awareness < 0.0 || snapshot.meta_awareness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meta-awareness must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate self_model_accuracy is in range [0, 1]
    if snapshot.self_model_accuracy < 0.0 || snapshot.self_model_accuracy > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Self-model accuracy must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate coherence is in range [0, 1]
    if snapshot.coherence < 0.0 || snapshot.coherence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Coherence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate affective_valence is in range [-1, 1]
    if snapshot.affective_valence < -1.0 || snapshot.affective_valence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affective valence must be between -1.0 and 1.0".into(),
        ));
    }

    // Validate care_activation is in range [0, 1]
    if snapshot.care_activation < 0.0 || snapshot.care_activation > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "CARE activation must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate source is present
    if snapshot.source.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source system required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate consciousness attestation creation
fn validate_create_consciousness_attestation(
    _action: Create,
    attestation: ConsciousnessAttestation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID format
    if !attestation.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate consciousness level is in range [0, 1]
    if attestation.consciousness_level < 0.0 || attestation.consciousness_level > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness level must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate signature is non-empty
    if attestation.signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation signature must be non-empty".into(),
        ));
    }

    // Validate source must be "symthaea"
    if attestation.source != "symthaea" {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation source must be \"symthaea\"".into(),
        ));
    }

    // Validate cycle_id is positive
    if attestation.cycle_id == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID must be > 0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate consciousness gate creation
fn validate_create_consciousness_gate(
    _action: Create,
    gate: ConsciousnessGate,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID is present
    if gate.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Gate ID required".into(),
        ));
    }

    // Validate agent DID format
    if !gate.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate snapshot ID is present
    if gate.snapshot_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Snapshot ID required for gate verification".into(),
        ));
    }

    // Validate Φ values are in range
    if gate.consciousness_at_verification < 0.0 || gate.consciousness_at_verification > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness at verification must be between 0.0 and 1.0".into(),
        ));
    }

    if gate.required_consciousness < 0.0 || gate.required_consciousness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Required consciousness must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate pass/fail consistency
    let should_pass = gate.consciousness_at_verification >= gate.required_consciousness;
    if gate.passed != should_pass {
        return Ok(ValidateCallbackResult::Invalid(
            "Gate pass/fail inconsistent with consciousness values".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate consciousness history creation
fn validate_create_consciousness_history(
    _action: Create,
    history: ConsciousnessHistory,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID format
    if !history.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate Φ statistics are in range
    if history.average_consciousness < 0.0 || history.average_consciousness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Average phi must be between 0.0 and 1.0".into(),
        ));
    }

    if history.peak_consciousness < 0.0 || history.peak_consciousness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Peak phi must be between 0.0 and 1.0".into(),
        ));
    }

    if history.min_consciousness < 0.0 || history.min_consciousness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum phi must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate min <= avg <= peak
    if history.min_consciousness > history.average_consciousness || history.average_consciousness > history.peak_consciousness {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness statistics must satisfy: min <= average <= peak".into(),
        ));
    }

    // Validate snapshot count is positive
    if history.snapshot_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Snapshot count must be positive".into(),
        ));
    }

    // Validate period order
    if history.period_end < history.period_start {
        return Ok(ValidateCallbackResult::Invalid(
            "Period end must be after period start".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate value alignment assessment creation
fn validate_create_value_alignment(
    _action: Create,
    assessment: ValueAlignmentAssessment,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID is present
    if assessment.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Assessment ID required".into(),
        ));
    }

    // Validate agent DID format
    if !assessment.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate proposal ID is present
    if assessment.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID required for alignment assessment".into(),
        ));
    }

    // Validate overall alignment is in range [-1, 1]
    if assessment.overall_alignment < -1.0 || assessment.overall_alignment > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Overall alignment must be between -1.0 and 1.0".into(),
        ));
    }

    // Validate harmony scores are in range
    for hs in &assessment.harmony_scores {
        if hs.score < -1.0 || hs.score > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Harmony score for {} must be between -1.0 and 1.0", hs.harmony),
            ));
        }
    }

    // Validate authenticity is in range [0, 1]
    if assessment.authenticity < 0.0 || assessment.authenticity > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Authenticity must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate snapshot ID is present
    if assessment.snapshot_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Snapshot ID required for alignment assessment".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// RB-BFT CONSENSUS VALIDATION FUNCTIONS
// =============================================================================

/// Validate K-Vector creation
fn validate_create_k_vector(
    _action: Create,
    k_vector: KVector,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID
    if !k_vector.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate all components are in range [0, 1]
    let components = [
        ("k_r", k_vector.k_r),
        ("k_a", k_vector.k_a),
        ("k_i", k_vector.k_i),
        ("k_p", k_vector.k_p),
        ("k_m", k_vector.k_m),
        ("k_s", k_vector.k_s),
        ("k_h", k_vector.k_h),
        ("k_topo", k_vector.k_topo),
    ];

    for (name, value) in components {
        if value < 0.0 || value > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between 0.0 and 1.0", name),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate MATL trust score creation
fn validate_create_matl_score(
    _action: Create,
    matl: MatlTrustScore,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID
    if !matl.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate scores are in range [0, 1]
    let scores = [
        ("pogq_score", matl.pogq_score),
        ("tcdm_score", matl.tcdm_score),
        ("entropy_score", matl.entropy_score),
        ("matl_score", matl.matl_score),
        ("k_vector_score", matl.k_vector_score),
        ("phi", matl.phi),
    ];

    for (name, value) in scores {
        if value < 0.0 || value > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between 0.0 and 1.0", name),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate federated reputation creation
fn validate_create_federated_reputation(
    _action: Create,
    fed_rep: FederatedReputation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID
    if !fed_rep.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate all score fields are in range [0, 1]
    let scores = [
        ("identity_verification", fed_rep.identity_verification),
        ("credential_quality", fed_rep.credential_quality),
        ("epistemic_contributions", fed_rep.epistemic_contributions),
        ("factcheck_accuracy", fed_rep.factcheck_accuracy),
        ("stake_weight", fed_rep.stake_weight),
        ("payment_reliability", fed_rep.payment_reliability),
        ("escrow_completion_rate", fed_rep.escrow_completion_rate),
        ("pogq_score", fed_rep.pogq_score),
        ("byzantine_clean_rate", fed_rep.byzantine_clean_rate),
        ("voting_participation", fed_rep.voting_participation),
        ("proposal_success_rate", fed_rep.proposal_success_rate),
        ("consensus_alignment", fed_rep.consensus_alignment),
        ("aggregated_score", fed_rep.aggregated_score),
    ];

    for (name, value) in scores {
        if value < 0.0 || value > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between 0.0 and 1.0", name),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate consensus participant creation
fn validate_create_consensus_participant(
    _action: Create,
    participant: ConsensusParticipant,
) -> ExternResult<ValidateCallbackResult> {
    // Validate agent DID
    if !participant.agent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent must have valid DID".into(),
        ));
    }

    // Validate K-Vector ID is present
    if participant.k_vector_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector ID required for participant".into(),
        ));
    }

    // Validate reputation is in range [0, 1]
    if participant.reputation < 0.0 || participant.reputation > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate voting weight is non-negative
    if participant.voting_weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Voting weight must be non-negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate weighted vote creation
fn validate_create_weighted_vote(
    _action: Create,
    vote: WeightedVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate vote ID
    if vote.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Vote ID required".into()));
    }

    // Validate proposal ID
    if vote.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID required for vote".into(),
        ));
    }

    // Validate voter DID
    if !vote.voter_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must have valid DID".into(),
        ));
    }

    // Validate reputation is in range [0, 1]
    if vote.reputation < 0.0 || vote.reputation > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate weight is non-negative
    if vote.weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote weight must be non-negative".into(),
        ));
    }

    // Validate Φ is in range [0, 1]
    if vote.phi < 0.0 || vote.phi > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Φ must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate consensus round creation
fn validate_create_consensus_round(
    _action: Create,
    round: ConsensusRound,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proposal ID
    if round.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID required for consensus round".into(),
        ));
    }

    // Validate leader DID
    if !round.leader_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Leader must have valid DID".into(),
        ));
    }

    // Validate total weight is non-negative
    if round.total_weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total weight must be non-negative".into(),
        ));
    }

    // Validate threshold is reasonable (between 0 and total_weight)
    if round.threshold < 0.0 || round.threshold > round.total_weight {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold must be between 0 and total weight".into(),
        ));
    }

    // Validate timeout is reasonable (minimum 1 second)
    if round.timeout_ms < 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Timeout must be at least 1000ms".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate slashing record creation
fn validate_create_slashing_record(
    _action: Create,
    record: SlashingRecord,
) -> ExternResult<ValidateCallbackResult> {
    // Validate record ID
    if record.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Slashing record ID required".into(),
        ));
    }

    // Validate offender DID
    if !record.offender_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Offender must have valid DID".into(),
        ));
    }

    // Validate reporter DID
    if !record.reporter_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Reporter must have valid DID".into(),
        ));
    }

    // Validate penalty is in range [0, 1]
    if record.penalty < 0.0 || record.penalty > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Penalty must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate reputation values are in range [0, 1]
    if record.reputation_before < 0.0 || record.reputation_before > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation before must be between 0.0 and 1.0".into(),
        ));
    }

    if record.reputation_after < 0.0 || record.reputation_after > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation after must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate reputation after is less than or equal to before (slashing reduces reputation)
    if record.reputation_after > record.reputation_before {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation after slashing cannot be greater than before".into(),
        ));
    }

    // Validate evidence is present
    if record.evidence.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence required for slashing".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// UNIT TESTS FOR CONSCIOUSNESS METRICS
// =============================================================================

#[cfg(test)]
mod consciousness_snapshot_tests {
    use super::*;

    fn create_test_snapshot(phi: f64) -> ConsciousnessSnapshot {
        ConsciousnessSnapshot {
            id: "test-snapshot-1".to_string(),
            agent_did: "did:mycelix:test123".to_string(),
            consciousness_level: phi,
            meta_awareness: 0.6,
            self_model_accuracy: 0.7,
            coherence: 0.8,
            affective_valence: 0.5,
            care_activation: 0.6,
            captured_at: Timestamp::from_micros(1000000),
            source: "symthaea".to_string(),
            consciousness_vector: None,
        }
    }

    #[test]
    fn test_quality_score_calculation() {
        let snapshot = create_test_snapshot(0.5);
        let quality = snapshot.quality_score();
        assert!((quality - 0.62).abs() < 0.001, "Quality score should be 0.62, got {}", quality);
    }

    #[test]
    fn test_quality_score_clamping() {
        let mut snapshot = create_test_snapshot(1.0);
        snapshot.meta_awareness = 1.0;
        snapshot.self_model_accuracy = 1.0;
        snapshot.coherence = 1.0;

        let quality = snapshot.quality_score();
        assert!(quality <= 1.0, "Quality score should be <= 1.0");
    }

    #[test]
    fn test_meets_threshold_basic() {
        let snapshot = create_test_snapshot(0.25);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(!snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
    }

    #[test]
    fn test_meets_threshold_proposal() {
        let snapshot = create_test_snapshot(0.35);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Voting));
    }

    #[test]
    fn test_meets_threshold_voting() {
        let snapshot = create_test_snapshot(0.45);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Voting));
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Constitutional));
    }

    #[test]
    fn test_meets_threshold_constitutional() {
        let snapshot = create_test_snapshot(0.65);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Voting));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Constitutional));
    }
}

#[cfg(test)]
mod action_type_tests {
    use super::*;

    #[test]
    fn test_consciousness_gates() {
        assert_eq!(GovernanceActionType::Basic.consciousness_gate(), 0.2);
        assert_eq!(GovernanceActionType::ProposalSubmission.consciousness_gate(), 0.3);
        assert_eq!(GovernanceActionType::Voting.consciousness_gate(), 0.4);
        assert_eq!(GovernanceActionType::Constitutional.consciousness_gate(), 0.6);
    }

    #[test]
    fn test_descriptions() {
        assert_eq!(GovernanceActionType::Basic.description(), "Basic participation");
        assert_eq!(GovernanceActionType::ProposalSubmission.description(), "Proposal submission");
        assert_eq!(GovernanceActionType::Voting.description(), "Voting on proposals");
        assert_eq!(GovernanceActionType::Constitutional.description(), "Constitutional changes");
    }

    #[test]
    fn test_threshold_ordering() {
        assert!(GovernanceActionType::Basic.consciousness_gate() <
                GovernanceActionType::ProposalSubmission.consciousness_gate());
        assert!(GovernanceActionType::ProposalSubmission.consciousness_gate() <
                GovernanceActionType::Voting.consciousness_gate());
        assert!(GovernanceActionType::Voting.consciousness_gate() <
                GovernanceActionType::Constitutional.consciousness_gate());
    }
}

#[cfg(test)]
mod recommendation_tests {
    use super::*;

    fn determine_recommendation(
        alignment: f64,
        authenticity: f64,
        has_violations: bool,
    ) -> GovernanceRecommendation {
        if has_violations {
            return GovernanceRecommendation::StrongOppose;
        }

        if authenticity < 0.2 {
            return GovernanceRecommendation::CannotEvaluate;
        }

        let combined = alignment * 0.6 + authenticity * 0.4;

        match combined {
            c if c > 0.7 => GovernanceRecommendation::StrongSupport,
            c if c > 0.3 => GovernanceRecommendation::Support,
            c if c > -0.3 => GovernanceRecommendation::Neutral,
            c if c > -0.7 => GovernanceRecommendation::Oppose,
            _ => GovernanceRecommendation::StrongOppose,
        }
    }

    #[test]
    fn test_strong_support() {
        let rec = determine_recommendation(0.9, 0.8, false);
        assert_eq!(rec, GovernanceRecommendation::StrongSupport);
    }

    #[test]
    fn test_support() {
        let rec = determine_recommendation(0.5, 0.5, false);
        assert_eq!(rec, GovernanceRecommendation::Support);
    }

    #[test]
    fn test_neutral() {
        let rec = determine_recommendation(0.0, 0.5, false);
        assert_eq!(rec, GovernanceRecommendation::Neutral);
    }

    #[test]
    fn test_strong_oppose_from_violations() {
        let rec = determine_recommendation(0.9, 0.9, true);
        assert_eq!(rec, GovernanceRecommendation::StrongOppose);
    }

    #[test]
    fn test_cannot_evaluate_low_authenticity() {
        let rec = determine_recommendation(0.9, 0.1, false);
        assert_eq!(rec, GovernanceRecommendation::CannotEvaluate);
    }
}

#[cfg(test)]
mod consciousness_workflow_tests {
    use super::*;

    #[test]
    fn test_gate_voting_workflow() {
        let phi = 0.45;
        let action_type = GovernanceActionType::Voting;
        let required = action_type.consciousness_gate();

        assert!(phi >= required, "Φ {} should meet voting threshold {}", phi, required);
    }

    #[test]
    fn test_gate_rejection_workflow() {
        let phi = 0.35;
        let action_type = GovernanceActionType::Voting;
        let required = action_type.consciousness_gate();

        assert!(phi < required, "Φ {} should NOT meet voting threshold {}", phi, required);
    }

    #[test]
    fn test_constitutional_gate() {
        let low_phi = 0.55;
        let high_phi = 0.65;

        assert!(low_phi < GovernanceActionType::Constitutional.consciousness_gate());
        assert!(high_phi >= GovernanceActionType::Constitutional.consciousness_gate());
    }
}

#[cfg(test)]
mod consciousness_edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_phi() {
        let snapshot = ConsciousnessSnapshot {
            id: "test".to_string(),
            agent_did: "did:mycelix:test".to_string(),
            consciousness_level: 0.0,
            meta_awareness: 0.0,
            self_model_accuracy: 0.0,
            coherence: 0.0,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".to_string(),
            consciousness_vector: None,
        };

        assert_eq!(snapshot.quality_score(), 0.0);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Basic));
    }

    #[test]
    fn test_max_phi() {
        let snapshot = ConsciousnessSnapshot {
            id: "test".to_string(),
            agent_did: "did:mycelix:test".to_string(),
            consciousness_level: 1.0,
            meta_awareness: 1.0,
            self_model_accuracy: 1.0,
            coherence: 1.0,
            affective_valence: 1.0,
            care_activation: 1.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".to_string(),
            consciousness_vector: None,
        };

        assert_eq!(snapshot.quality_score(), 1.0);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Constitutional));
    }
}

// =============================================================================
// UNIT TESTS FOR RB-BFT CONSENSUS
// =============================================================================

#[cfg(test)]
mod k_vector_tests {
    use super::*;

    fn create_test_k_vector() -> KVector {
        KVector {
            agent_did: "did:mycelix:test123".to_string(),
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.7,
            k_topo: 0.3,
            updated_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn test_trust_score_calculation() {
        let k = create_test_k_vector();
        // T = 0.25×0.8 + 0.15×0.7 + 0.20×0.9 + 0.15×0.6 + 0.05×0.5 + 0.10×0.4 + 0.05×0.7 + 0.05×0.3
        //   = 0.2 + 0.105 + 0.18 + 0.09 + 0.025 + 0.04 + 0.035 + 0.015 = 0.69
        let score = k.trust_score();
        assert!((score - 0.69).abs() < 0.001, "Trust score should be 0.69, got {}", score);
    }

    #[test]
    fn test_voting_weight_squared() {
        let k = create_test_k_vector();
        // Weight = 0.8² = 0.64
        let weight = k.voting_weight();
        assert!((weight - 0.64).abs() < 0.001, "Voting weight should be 0.64, got {}", weight);
    }

    #[test]
    fn test_reputation_returns_k_r() {
        let k = create_test_k_vector();
        assert_eq!(k.reputation(), 0.8);
    }

    #[test]
    fn test_can_participate_threshold() {
        let mut k = create_test_k_vector();
        assert!(k.can_participate()); // 0.8 >= 0.1

        k.k_r = 0.05;
        assert!(!k.can_participate()); // 0.05 < 0.1

        k.k_r = 0.1;
        assert!(k.can_participate()); // 0.1 >= 0.1 (boundary)
    }

    #[test]
    fn test_new_participant_defaults() {
        let k = KVector::new_participant(
            "did:mycelix:newuser".to_string(),
            Timestamp::from_micros(0),
        );
        assert_eq!(k.k_r, 0.5);
        assert_eq!(k.k_a, 0.5);
        assert_eq!(k.k_i, 0.5);
        assert_eq!(k.k_p, 0.5);
        assert_eq!(k.k_m, 0.1); // New member starts low
        assert_eq!(k.k_s, 0.0); // No stake yet
        assert!(k.can_participate());
    }
}

#[cfg(test)]
mod matl_tests {
    use super::*;

    #[test]
    fn test_matl_formula() {
        // T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
        let score = MatlTrustScore::calculate(0.8, 0.6, 0.7);
        // = 0.4×0.8 + 0.3×0.6 + 0.3×0.7 = 0.32 + 0.18 + 0.21 = 0.71
        assert!((score - 0.71).abs() < 0.001, "MATL score should be 0.71, got {}", score);
    }

    #[test]
    fn test_matl_clamping() {
        let score = MatlTrustScore::calculate(1.5, 1.5, 1.5);
        assert!(score <= 1.0, "MATL score should be clamped to 1.0");
    }

    #[test]
    fn test_holistic_score() {
        let matl = MatlTrustScore {
            agent_did: "did:mycelix:test".to_string(),
            pogq_score: 0.8,
            tcdm_score: 0.6,
            entropy_score: 0.7,
            matl_score: 0.71,
            k_vector_score: 0.69,
            phi: 0.5,
            calculated_at: Timestamp::from_micros(0),
        };
        // Holistic = 0.5×MATL + 0.3×K-Vector + 0.2×Φ
        // = 0.5×0.71 + 0.3×0.69 + 0.2×0.5 = 0.355 + 0.207 + 0.1 = 0.662
        let holistic = matl.holistic_score();
        assert!((holistic - 0.662).abs() < 0.001, "Holistic score should be 0.662, got {}", holistic);
    }
}

#[cfg(test)]
mod consensus_participant_tests {
    use super::*;

    fn create_test_participant() -> ConsensusParticipant {
        ConsensusParticipant {
            agent_did: "did:mycelix:validator1".to_string(),
            k_vector_id: "kvec-123".to_string(),
            federated_rep_id: None,
            reputation: 0.8,
            voting_weight: 0.64,
            matl_score: 0.7,
            phi: 0.5,
            federated_score: 0.5,
            is_active: true,
            rounds_participated: 100,
            successful_votes: 95,
            slashing_events: 0,
            last_slashing_at: None,
            streak_count: 0,
            registered_at: Timestamp::from_micros(0),
            last_active_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn test_success_rate() {
        let p = create_test_participant();
        let rate = p.success_rate();
        assert!((rate - 0.95).abs() < 0.001, "Success rate should be 0.95, got {}", rate);
    }

    #[test]
    fn test_success_rate_new_participant() {
        let mut p = create_test_participant();
        p.rounds_participated = 0;
        p.successful_votes = 0;
        assert_eq!(p.success_rate(), 0.5); // Default for new participants
    }

    #[test]
    fn test_can_vote() {
        let mut p = create_test_participant();
        assert!(p.can_vote()); // Active + reputation >= 0.1

        p.is_active = false;
        assert!(!p.can_vote()); // Inactive

        p.is_active = true;
        p.reputation = 0.05;
        assert!(!p.can_vote()); // Low reputation
    }
}

#[cfg(test)]
mod consensus_round_tests {
    use super::*;

    fn create_test_round() -> ConsensusRound {
        ConsensusRound {
            round: 1,
            proposal_id: "prop-123".to_string(),
            state: RoundState::Voting,
            leader_did: "did:mycelix:leader".to_string(),
            total_weight: 10.0,
            threshold: 5.5, // 55% of 10
            weighted_approvals: 0.0,
            weighted_rejections: 0.0,
            vote_count: 0,
            started_at: Timestamp::from_micros(0),
            ended_at: None,
            timeout_ms: 30000,
            result_hash: None,
        }
    }

    #[test]
    fn test_consensus_reached() {
        let mut round = create_test_round();
        assert!(!round.consensus_reached()); // 0 > 5.5 is false

        round.weighted_approvals = 5.6;
        assert!(round.consensus_reached()); // 5.6 > 5.5 is true
    }

    #[test]
    fn test_rejected() {
        let mut round = create_test_round();
        assert!(!round.rejected()); // 0 > 5.5 is false

        round.weighted_rejections = 5.6;
        assert!(round.rejected()); // 5.6 > 5.5 is true
    }

    #[test]
    fn test_approval_percentage() {
        let mut round = create_test_round();
        round.weighted_approvals = 6.0;
        let pct = round.approval_percentage();
        assert!((pct - 60.0).abs() < 0.001, "Approval should be 60%, got {}", pct);
    }

    #[test]
    fn test_approval_percentage_zero_weight() {
        let mut round = create_test_round();
        round.total_weight = 0.0;
        assert_eq!(round.approval_percentage(), 0.0);
    }
}

#[cfg(test)]
mod slashing_tests {
    use super::*;

    #[test]
    fn test_slashing_severity_penalties() {
        assert_eq!(SlashingSeverity::Minor.penalty(), 0.1);
        assert_eq!(SlashingSeverity::Moderate.penalty(), 0.3);
        assert_eq!(SlashingSeverity::Severe.penalty(), 0.5);
        assert_eq!(SlashingSeverity::Critical.penalty(), 0.8);
    }

    #[test]
    fn test_apply_slashing() {
        let reputation_before = 0.8;
        let severity = SlashingSeverity::Moderate;
        let reputation_after = reputation_before * (1.0 - severity.penalty());
        // 0.8 * (1.0 - 0.3) = 0.8 * 0.7 = 0.56
        assert!((reputation_after - 0.56).abs() < 0.001);
    }
}

#[cfg(test)]
mod consensus_workflow_tests {
    use super::*;

    #[test]
    fn test_weighted_voting_workflow() {
        // Simulate 5 validators voting with reputation² weighting
        let validators = vec![
            (0.9_f64, VoteDecision::Approve),   // weight: 0.81
            (0.8, VoteDecision::Approve),       // weight: 0.64
            (0.6, VoteDecision::Reject),        // weight: 0.36
            (0.4, VoteDecision::Reject),        // weight: 0.16
            (0.2, VoteDecision::Abstain),       // weight: 0.04
        ];

        let total_weight: f64 = validators.iter().map(|(r, _)| r.powi(2)).sum();
        let threshold = total_weight * 0.55;

        let weighted_approvals: f64 = validators
            .iter()
            .filter(|(_, d)| matches!(d, VoteDecision::Approve))
            .map(|(r, _)| r.powi(2))
            .sum();

        let weighted_rejections: f64 = validators
            .iter()
            .filter(|(_, d)| matches!(d, VoteDecision::Reject))
            .map(|(r, _)| r.powi(2))
            .sum();

        // total_weight = 0.81 + 0.64 + 0.36 + 0.16 + 0.04 = 2.01
        // threshold = 2.01 * 0.55 = 1.1055
        // approvals = 0.81 + 0.64 = 1.45
        // rejections = 0.36 + 0.16 = 0.52

        assert!((total_weight - 2.01).abs() < 0.001);
        assert!((threshold - 1.1055).abs() < 0.001);
        assert!((weighted_approvals - 1.45).abs() < 0.001);
        assert!((weighted_rejections - 0.52).abs() < 0.001);

        // Consensus reached: 1.45 > 1.1055
        assert!(weighted_approvals > threshold);
    }

    #[test]
    fn test_byzantine_resistance() {
        // Test that low-reputation Byzantine validators can't influence consensus
        // 3 honest (high rep) vs 2 Byzantine (low rep)
        let validators = vec![
            (0.9_f64, true),   // Honest: weight 0.81
            (0.85, true),      // Honest: weight 0.7225
            (0.8, true),       // Honest: weight 0.64
            (0.3, false),      // Byzantine: weight 0.09
            (0.2, false),      // Byzantine: weight 0.04
        ];

        let total_weight: f64 = validators.iter().map(|(r, _)| r.powi(2)).sum();
        let honest_weight: f64 = validators
            .iter()
            .filter(|(_, honest)| *honest)
            .map(|(r, _)| r.powi(2))
            .sum();
        let byzantine_weight: f64 = validators
            .iter()
            .filter(|(_, honest)| !*honest)
            .map(|(r, _)| r.powi(2))
            .sum();

        // total = 0.81 + 0.7225 + 0.64 + 0.09 + 0.04 = 2.3025
        // honest = 0.81 + 0.7225 + 0.64 = 2.1725
        // byzantine = 0.09 + 0.04 = 0.13

        let byzantine_percentage = (byzantine_weight / total_weight) * 100.0;
        // 0.13 / 2.3025 * 100 = 5.64%

        assert!(byzantine_percentage < 10.0, "Byzantine should have < 10% voting power");
        assert!(honest_weight > total_weight * 0.55, "Honest validators should reach consensus alone");
    }
}

// =============================================================================
// HOLISTIC VOTING WEIGHT TESTS
// =============================================================================

#[cfg(test)]
mod holistic_voting_weight_tests {
    use super::*;

    #[test]
    fn test_basic_calculation() {
        // reputation=0.8, phi=0.5, alignment=0.0
        let weight = HolisticVotingWeight::calculate(0.8, 0.5, 0.0);

        // reputation² = 0.64
        assert!((weight.reputation_squared - 0.64).abs() < 0.001);

        // consciousness_multiplier = 0.7 + 0.3 × 0.5 = 0.85
        assert!((weight.consciousness_multiplier - 0.85).abs() < 0.001);

        // harmonic_bonus = 1.0 + 0.2 × 0 = 1.0
        assert!((weight.harmonic_bonus - 1.0).abs() < 0.001);

        // final = 0.64 × 0.85 × 1.0 = 0.544
        assert!((weight.final_weight - 0.544).abs() < 0.001);
    }

    #[test]
    fn test_high_consciousness_bonus() {
        // High Φ (0.9) should give significant boost
        let weight = HolisticVotingWeight::calculate(0.8, 0.9, 0.0);

        // consciousness_multiplier = 0.7 + 0.3 × 0.9 = 0.97
        assert!((weight.consciousness_multiplier - 0.97).abs() < 0.001);

        // final = 0.64 × 0.97 × 1.0 = 0.6208
        assert!((weight.final_weight - 0.6208).abs() < 0.001);
    }

    #[test]
    fn test_low_consciousness_penalty() {
        // Low Φ (0.1) should reduce influence
        let weight = HolisticVotingWeight::calculate(0.8, 0.1, 0.0);

        // consciousness_multiplier = 0.7 + 0.3 × 0.1 = 0.73
        assert!((weight.consciousness_multiplier - 0.73).abs() < 0.001);

        // final = 0.64 × 0.73 × 1.0 = 0.4672
        assert!((weight.final_weight - 0.4672).abs() < 0.001);
    }

    #[test]
    fn test_harmonic_alignment_bonus() {
        // Positive harmonic alignment (0.8) should give bonus
        let weight = HolisticVotingWeight::calculate(0.8, 0.5, 0.8);

        // harmonic_bonus = 1.0 + 0.2 × 0.8 = 1.16
        assert!((weight.harmonic_bonus - 1.16).abs() < 0.001);

        // final = 0.64 × 0.85 × 1.16 = 0.63104
        assert!((weight.final_weight - 0.63104).abs() < 0.001);
    }

    #[test]
    fn test_negative_alignment_no_bonus() {
        // Negative harmonic alignment should NOT give penalty (handled elsewhere)
        let weight = HolisticVotingWeight::calculate(0.8, 0.5, -0.5);

        // harmonic_bonus = 1.0 + 0.2 × max(0, -0.5) = 1.0
        assert!((weight.harmonic_bonus - 1.0).abs() < 0.001);

        // Same as zero alignment
        let weight_zero = HolisticVotingWeight::calculate(0.8, 0.5, 0.0);
        assert!((weight.final_weight - weight_zero.final_weight).abs() < 0.001);
    }

    #[test]
    fn test_max_alignment_bonus() {
        // Maximum alignment (1.0) gives 20% bonus
        let weight = HolisticVotingWeight::calculate(0.8, 0.5, 1.0);

        // harmonic_bonus = 1.0 + 0.2 × 1.0 = 1.2
        assert!((weight.harmonic_bonus - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_combined_high_consciousness_and_alignment() {
        // High Φ (0.9) + perfect alignment (1.0) = maximum influence
        let weight = HolisticVotingWeight::calculate(0.9, 0.9, 1.0);

        // reputation² = 0.81
        // consciousness_multiplier = 0.97
        // harmonic_bonus = 1.2
        // final = 0.81 × 0.97 × 1.2 = 0.94284
        assert!((weight.final_weight - 0.94284).abs() < 0.001);
    }

    #[test]
    fn test_clamping_input_values() {
        // Values outside range should be clamped
        let weight = HolisticVotingWeight::calculate(1.5, 2.0, 3.0);

        assert_eq!(weight.reputation, 1.0);
        assert_eq!(weight.consciousness_level, 1.0);
        assert_eq!(weight.harmonic_alignment, 1.0);
    }

    #[test]
    fn test_calculate_base_convenience() {
        // calculate_base should be equivalent to calculate with alignment=0
        let base = HolisticVotingWeight::calculate_base(0.8, 0.5);
        let full = HolisticVotingWeight::calculate(0.8, 0.5, 0.0);

        assert!((base.final_weight - full.final_weight).abs() < 0.001);
    }

    #[test]
    fn test_breakdown_string_present() {
        let weight = HolisticVotingWeight::calculate(0.8, 0.5, 0.3);

        assert!(!weight.calculation_breakdown.is_empty());
        assert!(weight.calculation_breakdown.contains("0.800"));
    }
}

// =============================================================================
// ADAPTIVE THRESHOLD TESTS
// =============================================================================

#[cfg(test)]
mod adaptive_threshold_tests {
    use super::*;

    #[test]
    fn test_standard_proposal_thresholds() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);

        assert!((threshold.base_threshold - 0.51).abs() < 0.001);
        assert!((threshold.min_voter_consciousness - 0.2).abs() < 0.001);
        assert_eq!(threshold.min_participation, 5);
        assert!((threshold.quorum - 0.20).abs() < 0.001);
        assert_eq!(threshold.max_extension_secs, 86400);
    }

    #[test]
    fn test_emergency_proposal_thresholds() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency);

        assert!((threshold.base_threshold - 0.60).abs() < 0.001);
        assert!((threshold.min_voter_consciousness - 0.3).abs() < 0.001);
        assert_eq!(threshold.min_participation, 3);
        assert!((threshold.quorum - 0.10).abs() < 0.001);
        assert_eq!(threshold.max_extension_secs, 3600);
    }

    #[test]
    fn test_constitutional_proposal_thresholds() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);

        assert!((threshold.base_threshold - 0.67).abs() < 0.001);
        assert!((threshold.min_voter_consciousness - 0.5).abs() < 0.001);
        assert_eq!(threshold.min_participation, 10);
        assert!((threshold.quorum - 0.40).abs() < 0.001);
        assert_eq!(threshold.max_extension_secs, 604800);
    }

    #[test]
    fn test_calculate_threshold() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        let required = threshold.calculate_threshold(100.0);

        // 100 × 0.51 = 51
        assert!((required - 51.0).abs() < 0.001);
    }

    #[test]
    fn test_voter_meets_consciousness_requirement_standard() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);

        assert!(threshold.voter_meets_consciousness_requirement(0.25)); // 0.25 >= 0.2
        assert!(threshold.voter_meets_consciousness_requirement(0.20)); // 0.20 >= 0.2 (boundary)
        assert!(!threshold.voter_meets_consciousness_requirement(0.15)); // 0.15 < 0.2
    }

    #[test]
    fn test_voter_meets_consciousness_requirement_constitutional() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);

        assert!(threshold.voter_meets_consciousness_requirement(0.6)); // 0.6 >= 0.5
        assert!(threshold.voter_meets_consciousness_requirement(0.5)); // 0.5 >= 0.5 (boundary)
        assert!(!threshold.voter_meets_consciousness_requirement(0.45)); // 0.45 < 0.5
    }

    #[test]
    fn test_quorum_met_standard() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);

        // 20% of 100 = 20 voters, min 5
        assert!(threshold.quorum_met(25, 100)); // 25% > 20%, and 25 >= 5
        assert!(threshold.quorum_met(20, 100)); // 20% >= 20%, and 20 >= 5
        assert!(!threshold.quorum_met(15, 100)); // 15% < 20%

        // Even with high participation rate, need min_participation
        assert!(!threshold.quorum_met(4, 10)); // 40% but only 4 voters
        assert!(threshold.quorum_met(5, 10)); // 50% and 5 voters (boundary)
    }

    #[test]
    fn test_quorum_met_emergency() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency);

        // 10% quorum, min 3 voters
        assert!(threshold.quorum_met(3, 20)); // 15% > 10%, and 3 >= 3
        assert!(!threshold.quorum_met(2, 50)); // 4% < 10%
    }

    #[test]
    fn test_quorum_met_zero_eligible() {
        let threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);

        assert!(!threshold.quorum_met(10, 0)); // Division by zero protection
    }

    #[test]
    fn test_threshold_ordering() {
        let standard = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        let emergency = AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency);
        let constitutional = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);

        // Emergency and Constitutional require higher thresholds than Standard
        assert!(emergency.base_threshold > standard.base_threshold);
        assert!(constitutional.base_threshold > emergency.base_threshold);

        // Constitutional requires highest Φ
        assert!(constitutional.min_voter_consciousness > emergency.min_voter_consciousness);
        assert!(emergency.min_voter_consciousness > standard.min_voter_consciousness);
    }
}

// =============================================================================
// FEDERATED REPUTATION TESTS
// =============================================================================

#[cfg(test)]
mod federated_reputation_tests {
    use super::*;

    fn create_test_fed_rep() -> FederatedReputation {
        FederatedReputation {
            agent_did: "did:mycelix:test123".to_string(),
            identity_verification: 1.0,
            credential_count: 5,
            credential_quality: 0.8,
            epistemic_contributions: 0.7,
            factcheck_accuracy: 0.9,
            dark_spots_resolved: 10,
            stake_weight: 0.6,
            payment_reliability: 0.95,
            escrow_completion_rate: 1.0,
            pogq_score: 0.85,
            fl_contributions: 50,
            byzantine_clean_rate: 1.0,
            voting_participation: 0.8,
            proposal_success_rate: 0.7,
            consensus_alignment: 0.9,
            aggregated_score: 0.0, // Will be calculated
            aggregated_at: Timestamp::from_micros(0),
        }
    }

    #[test]
    fn test_calculate_aggregated() {
        let fed_rep = create_test_fed_rep();
        let score = fed_rep.calculate_aggregated();

        // Score should be in valid range
        assert!(score >= 0.0 && score <= 1.0);

        // With good scores across the board, should be relatively high
        assert!(score > 0.7, "Score should be > 0.7 with good metrics, got {}", score);
    }

    #[test]
    fn test_identity_score_calculation() {
        let mut fed_rep = create_test_fed_rep();

        // Set only identity signals, zero out others
        fed_rep.epistemic_contributions = 0.0;
        fed_rep.factcheck_accuracy = 0.0;
        fed_rep.dark_spots_resolved = 0;
        fed_rep.stake_weight = 0.0;
        fed_rep.payment_reliability = 0.0;
        fed_rep.escrow_completion_rate = 0.0;
        fed_rep.pogq_score = 0.0;
        fed_rep.fl_contributions = 0;
        fed_rep.byzantine_clean_rate = 0.0;
        fed_rep.voting_participation = 0.0;
        fed_rep.proposal_success_rate = 0.0;
        fed_rep.consensus_alignment = 0.0;

        let score = fed_rep.calculate_aggregated();

        // Identity is 15% of total weight
        // Identity score = 1.0×0.5 + (5/10)×0.25 + 0.8×0.25 = 0.5 + 0.125 + 0.2 = 0.825
        // Total contribution = 0.825 × 0.15 = 0.12375
        assert!(score > 0.1 && score < 0.2, "Identity-only score should be ~0.12, got {}", score);
    }

    #[test]
    fn test_knowledge_score_calculation() {
        let mut fed_rep = create_test_fed_rep();

        // Set only knowledge signals high, zero out others
        fed_rep.identity_verification = 0.0;
        fed_rep.credential_count = 0;
        fed_rep.credential_quality = 0.0;
        fed_rep.stake_weight = 0.0;
        fed_rep.payment_reliability = 0.0;
        fed_rep.escrow_completion_rate = 0.0;
        fed_rep.pogq_score = 0.0;
        fed_rep.fl_contributions = 0;
        fed_rep.byzantine_clean_rate = 0.0;
        fed_rep.voting_participation = 0.0;
        fed_rep.proposal_success_rate = 0.0;
        fed_rep.consensus_alignment = 0.0;

        // Keep: epistemic_contributions=0.7, factcheck_accuracy=0.9, dark_spots_resolved=10

        let score = fed_rep.calculate_aggregated();

        // Knowledge is 20% of total weight
        assert!(score > 0.1 && score < 0.25, "Knowledge-only score should be reasonable, got {}", score);
    }

    #[test]
    fn test_fl_score_emphasizes_pogq() {
        let mut fed_rep = create_test_fed_rep();

        // Zero out all except FL signals
        fed_rep.identity_verification = 0.0;
        fed_rep.credential_count = 0;
        fed_rep.credential_quality = 0.0;
        fed_rep.epistemic_contributions = 0.0;
        fed_rep.factcheck_accuracy = 0.0;
        fed_rep.dark_spots_resolved = 0;
        fed_rep.stake_weight = 0.0;
        fed_rep.payment_reliability = 0.0;
        fed_rep.escrow_completion_rate = 0.0;
        fed_rep.voting_participation = 0.0;
        fed_rep.proposal_success_rate = 0.0;
        fed_rep.consensus_alignment = 0.0;

        // Keep: pogq_score=0.85, fl_contributions=50, byzantine_clean_rate=1.0

        let score = fed_rep.calculate_aggregated();

        // FL is 25% of total weight, PoGQ is 50% of FL
        assert!(score > 0.15 && score < 0.30, "FL-only score should be ~0.2, got {}", score);
    }

    #[test]
    fn test_governance_score_emphasizes_alignment() {
        let mut fed_rep = create_test_fed_rep();

        // Zero out all except governance signals
        fed_rep.identity_verification = 0.0;
        fed_rep.credential_count = 0;
        fed_rep.credential_quality = 0.0;
        fed_rep.epistemic_contributions = 0.0;
        fed_rep.factcheck_accuracy = 0.0;
        fed_rep.dark_spots_resolved = 0;
        fed_rep.stake_weight = 0.0;
        fed_rep.payment_reliability = 0.0;
        fed_rep.escrow_completion_rate = 0.0;
        fed_rep.pogq_score = 0.0;
        fed_rep.fl_contributions = 0;
        fed_rep.byzantine_clean_rate = 0.0;

        // Keep: voting_participation=0.8, proposal_success_rate=0.7, consensus_alignment=0.9

        let score = fed_rep.calculate_aggregated();

        // Governance is 25% of total weight
        assert!(score > 0.15 && score < 0.25, "Governance-only score should be ~0.2, got {}", score);
    }

    #[test]
    fn test_new_participant_defaults() {
        let fed_rep = FederatedReputation::new_participant(
            "did:mycelix:newuser".to_string(),
            Timestamp::from_micros(0),
        );

        // New participants start with conservative defaults
        assert_eq!(fed_rep.identity_verification, 0.0);
        assert_eq!(fed_rep.credential_count, 0);
        assert_eq!(fed_rep.factcheck_accuracy, 0.5); // Neutral
        assert_eq!(fed_rep.byzantine_clean_rate, 1.0); // Clean until proven otherwise
        assert_eq!(fed_rep.aggregated_score, 0.25); // Conservative starting score
    }

    #[test]
    fn test_refresh_aggregation() {
        let mut fed_rep = create_test_fed_rep();
        fed_rep.aggregated_score = 0.0;

        let new_timestamp = Timestamp::from_micros(1000000);
        fed_rep.refresh_aggregation(new_timestamp);

        assert!(fed_rep.aggregated_score > 0.0);
        assert_eq!(fed_rep.aggregated_at, new_timestamp);
    }

    #[test]
    fn test_credential_count_capping() {
        let mut fed_rep = create_test_fed_rep();
        fed_rep.credential_count = 100; // Way over cap

        let score1 = fed_rep.calculate_aggregated();

        fed_rep.credential_count = 10; // At cap
        let score2 = fed_rep.calculate_aggregated();

        // Should be the same (capped at 10)
        assert!((score1 - score2).abs() < 0.001);
    }

    #[test]
    fn test_fl_contributions_capping() {
        let mut fed_rep = create_test_fed_rep();
        fed_rep.fl_contributions = 1000; // Way over cap

        let score1 = fed_rep.calculate_aggregated();

        fed_rep.fl_contributions = 100; // At cap
        let score2 = fed_rep.calculate_aggregated();

        // Should be the same (capped at 100)
        assert!((score1 - score2).abs() < 0.001);
    }
}

// =============================================================================
// ENHANCED CONSENSUS PARTICIPANT TESTS
// =============================================================================

#[cfg(test)]
mod enhanced_consensus_participant_tests {
    use super::*;

    fn create_enhanced_participant() -> ConsensusParticipant {
        ConsensusParticipant {
            agent_did: "did:mycelix:validator1".to_string(),
            k_vector_id: "kvec-123".to_string(),
            federated_rep_id: Some("fed-456".to_string()),
            reputation: 0.8,
            voting_weight: 0.64,
            matl_score: 0.7,
            phi: 0.5,
            federated_score: 0.75,
            is_active: true,
            rounds_participated: 100,
            successful_votes: 95,
            slashing_events: 0,
            last_slashing_at: None,
            streak_count: 10,
            registered_at: Timestamp::from_micros(0),
            last_active_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn test_streak_bonus_calculation() {
        let mut p = create_enhanced_participant();

        p.streak_count = 0;
        assert!((p.streak_bonus() - 1.0).abs() < 0.001); // No bonus

        p.streak_count = 5;
        assert!((p.streak_bonus() - 1.05).abs() < 0.001); // 5% bonus

        p.streak_count = 10;
        assert!((p.streak_bonus() - 1.10).abs() < 0.001); // 10% bonus (max)

        p.streak_count = 20;
        assert!((p.streak_bonus() - 1.10).abs() < 0.001); // Still 10% (capped)
    }

    #[test]
    fn test_in_cooldown() {
        let mut p = create_enhanced_participant();
        let now = Timestamp::from_micros(86_400_000_000); // 1 day in micros

        // No slashing - not in cooldown
        assert!(!p.in_cooldown(now));

        // Slashed 12 hours ago - in cooldown
        p.last_slashing_at = Some(Timestamp::from_micros(43_200_000_000)); // 12 hours ago
        assert!(p.in_cooldown(now));

        // Slashed 25 hours ago - not in cooldown
        p.last_slashing_at = Some(Timestamp::from_micros(0)); // Way before
        let now_late = Timestamp::from_micros(90_000_000_000); // 25+ hours
        assert!(!p.in_cooldown(now_late));
    }

    #[test]
    fn test_effective_reputation_with_streak() {
        let mut p = create_enhanced_participant();
        p.streak_count = 10;
        let now = Timestamp::from_micros(1000000);

        // No cooldown, 10% streak bonus
        let effective = p.effective_reputation(now);
        // 0.8 × 1.10 = 0.88
        assert!((effective - 0.88).abs() < 0.001);
    }

    #[test]
    fn test_effective_reputation_in_cooldown() {
        let mut p = create_enhanced_participant();
        p.streak_count = 10;
        let now = Timestamp::from_micros(86_400_000_000); // 1 day
        p.last_slashing_at = Some(Timestamp::from_micros(43_200_000_000)); // 12 hours ago

        // In cooldown - 20% penalty, no streak bonus
        let effective = p.effective_reputation(now);
        // 0.8 × 0.8 = 0.64
        assert!((effective - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_can_vote_on_standard() {
        let mut p = create_enhanced_participant();
        p.phi = 0.25; // Above 0.2 for standard

        assert!(p.can_vote_on(&ProposalType::Standard));
    }

    #[test]
    fn test_can_vote_on_constitutional() {
        let mut p = create_enhanced_participant();

        p.phi = 0.4; // Below 0.5 for constitutional
        assert!(!p.can_vote_on(&ProposalType::Constitutional));

        p.phi = 0.6; // At or above 0.5
        assert!(p.can_vote_on(&ProposalType::Constitutional));
    }

    #[test]
    fn test_calculate_holistic_weight() {
        let p = create_enhanced_participant();
        let weight = p.calculate_holistic_weight(0.5);

        // Should use participant's reputation and consciousness level
        assert!((weight.reputation - 0.8).abs() < 0.001);
        assert!((weight.consciousness_level - 0.5).abs() < 0.001);
        assert!((weight.harmonic_alignment - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_federated_rep_integration() {
        let p = create_enhanced_participant();

        // Participant should have federated rep ID and score
        assert!(p.federated_rep_id.is_some());
        assert!((p.federated_score - 0.75).abs() < 0.001);
    }
}

// =============================================================================
// CONSCIOUSNESS-WEIGHTED CONSENSUS INTEGRATION TESTS
// =============================================================================

#[cfg(test)]
mod consciousness_weighted_consensus_tests {
    use super::*;

    #[test]
    fn test_holistic_weighted_voting_workflow() {
        // Simulate 5 validators with different consciousness and reputation
        struct Validator {
            reputation: f64,
            phi: f64,
            alignment: f64,
            decision: VoteDecision,
        }

        let validators = vec![
            Validator { reputation: 0.9, phi: 0.8, alignment: 0.9, decision: VoteDecision::Approve },
            Validator { reputation: 0.8, phi: 0.6, alignment: 0.7, decision: VoteDecision::Approve },
            Validator { reputation: 0.7, phi: 0.4, alignment: -0.2, decision: VoteDecision::Reject },
            Validator { reputation: 0.5, phi: 0.3, alignment: -0.5, decision: VoteDecision::Reject },
            Validator { reputation: 0.3, phi: 0.2, alignment: 0.0, decision: VoteDecision::Abstain },
        ];

        let weights: Vec<(f64, VoteDecision)> = validators
            .iter()
            .map(|v| {
                let weight = HolisticVotingWeight::calculate(v.reputation, v.phi, v.alignment);
                (weight.final_weight, v.decision)
            })
            .collect();

        let total_weight: f64 = weights.iter().map(|(w, _)| w).sum();

        let weighted_approvals: f64 = weights
            .iter()
            .filter(|(_, d)| matches!(d, VoteDecision::Approve))
            .map(|(w, _)| w)
            .sum();

        let _weighted_rejections: f64 = weights
            .iter()
            .filter(|(_, d)| matches!(d, VoteDecision::Reject))
            .map(|(w, _)| w)
            .sum();

        // High-consciousness, aligned approvers should dominate
        let approval_ratio = weighted_approvals / total_weight;
        assert!(approval_ratio > 0.5, "Approvals should be > 50%, got {}%", approval_ratio * 100.0);
    }

    #[test]
    fn test_adaptive_threshold_applied() {
        let proposal_type = ProposalType::Constitutional;
        let threshold = AdaptiveThreshold::for_proposal_type(&proposal_type);

        // Test that high-Φ requirement filters low-consciousness voters
        let low_phi_voter = 0.4;
        let high_phi_voter = 0.6;

        assert!(!threshold.voter_meets_consciousness_requirement(low_phi_voter));
        assert!(threshold.voter_meets_consciousness_requirement(high_phi_voter));
    }

    #[test]
    fn test_federated_reputation_influences_weight() {
        // Create two participants with same base reputation but different federated scores
        let high_fed = FederatedReputation {
            agent_did: "did:mycelix:high".to_string(),
            identity_verification: 1.0,
            credential_count: 10,
            credential_quality: 0.9,
            epistemic_contributions: 0.9,
            factcheck_accuracy: 0.95,
            dark_spots_resolved: 20,
            stake_weight: 0.8,
            payment_reliability: 1.0,
            escrow_completion_rate: 1.0,
            pogq_score: 0.95,
            fl_contributions: 100,
            byzantine_clean_rate: 1.0,
            voting_participation: 0.9,
            proposal_success_rate: 0.85,
            consensus_alignment: 0.95,
            aggregated_score: 0.0,
            aggregated_at: Timestamp::from_micros(0),
        };

        let low_fed = FederatedReputation::new_participant(
            "did:mycelix:low".to_string(),
            Timestamp::from_micros(0),
        );

        let high_score = high_fed.calculate_aggregated();
        let low_score = low_fed.calculate_aggregated();

        // High federated score should be significantly higher
        assert!(high_score > low_score * 2.0, "High fed ({}) should be much higher than low fed ({})", high_score, low_score);
    }

    #[test]
    fn test_consciousness_gates_protect_constitutional_votes() {
        // Only high-Φ agents should be able to vote on constitutional changes
        let agents = vec![
            ("did:mycelix:sage", 0.7),    // Can vote on constitutional
            ("did:mycelix:voter", 0.45),   // Can vote on standard/emergency, not constitutional
            ("did:mycelix:observer", 0.15), // Cannot vote at all
        ];

        let constitutional_threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);
        let standard_threshold = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);

        let can_vote_constitutional: Vec<&str> = agents
            .iter()
            .filter(|(_, phi)| constitutional_threshold.voter_meets_consciousness_requirement(*phi))
            .map(|(did, _)| *did)
            .collect();

        let can_vote_standard: Vec<&str> = agents
            .iter()
            .filter(|(_, phi)| standard_threshold.voter_meets_consciousness_requirement(*phi))
            .map(|(did, _)| *did)
            .collect();

        assert_eq!(can_vote_constitutional.len(), 1);
        assert!(can_vote_constitutional.contains(&"did:mycelix:sage"));

        assert_eq!(can_vote_standard.len(), 2);
        assert!(can_vote_standard.contains(&"did:mycelix:sage"));
        assert!(can_vote_standard.contains(&"did:mycelix:voter"));
    }

    #[test]
    fn test_streak_bonus_rewards_consistent_participation() {
        let base_reputation = 0.7;

        // New participant (no streak)
        let streak_0_bonus = 1.0;
        let effective_0 = base_reputation * streak_0_bonus;

        // Participant with 10 consecutive good rounds
        let streak_10_bonus = 1.10;
        let effective_10 = base_reputation * streak_10_bonus;

        // 10% advantage for consistent participation
        let advantage: f64 = (effective_10 - effective_0) / effective_0 * 100.0;
        assert!((advantage - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_cooldown_penalty_after_slashing() {
        let reputation = 0.8;

        // Normal effective reputation with streak
        let streak_bonus = 1.05;
        let normal_effective = reputation * streak_bonus;

        // Effective reputation during cooldown (20% penalty)
        let cooldown_effective = reputation * 0.8;

        // Cooldown should significantly reduce influence
        assert!(cooldown_effective < normal_effective);
        let reduction = (normal_effective - cooldown_effective) / normal_effective * 100.0;
        assert!(reduction > 20.0, "Cooldown should reduce influence by > 20%, got {}%", reduction);
    }

    // --- GovernanceConsciousnessConfig ---

    fn default_config() -> GovernanceConsciousnessConfig {
        GovernanceConsciousnessConfig::defaults(Timestamp::from_micros(1000000))
    }

    #[test]
    fn test_consciousness_config_defaults_valid() {
        assert!(check_consciousness_config(&default_config()).is_ok());
    }

    #[test]
    fn test_consciousness_config_defaults_match_hardcoded() {
        let config = default_config();
        assert!((config.consciousness_gate_basic - GovernanceActionType::Basic.consciousness_gate()).abs() < f64::EPSILON);
        assert!((config.consciousness_gate_proposal - GovernanceActionType::ProposalSubmission.consciousness_gate()).abs() < f64::EPSILON);
        assert!((config.consciousness_gate_voting - GovernanceActionType::Voting.consciousness_gate()).abs() < f64::EPSILON);
        assert!((config.consciousness_gate_constitutional - GovernanceActionType::Constitutional.consciousness_gate()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consciousness_config_threshold_lookup() {
        let config = default_config();
        assert!((config.consciousness_gate_for(&GovernanceActionType::Basic) - 0.2).abs() < f64::EPSILON);
        assert!((config.consciousness_gate_for(&GovernanceActionType::Constitutional) - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consciousness_config_voter_consciousness_lookup() {
        let config = default_config();
        assert!((config.min_voter_consciousness_for(&ProposalType::Standard) - 0.2).abs() < f64::EPSILON);
        assert!((config.min_voter_consciousness_for(&ProposalType::Emergency) - 0.3).abs() < f64::EPSILON);
        assert!((config.min_voter_consciousness_for(&ProposalType::Constitutional) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consciousness_config_out_of_range() {
        let mut config = default_config();
        config.consciousness_gate_basic = -0.1;
        assert!(check_consciousness_config(&config).is_err());

        config = default_config();
        config.consciousness_gate_constitutional = 1.1;
        assert!(check_consciousness_config(&config).is_err());
    }

    #[test]
    fn test_consciousness_config_non_monotonic_rejected() {
        let mut config = default_config();
        // Basic > ProposalSubmission violates ordering
        config.consciousness_gate_basic = 0.5;
        config.consciousness_gate_proposal = 0.3;
        assert!(check_consciousness_config(&config).unwrap_err().contains("consciousness_gate_basic"));
    }

    #[test]
    fn test_consciousness_config_voter_consciousness_non_monotonic_rejected() {
        let mut config = default_config();
        config.min_voter_consciousness_standard = 0.4;
        config.min_voter_consciousness_emergency = 0.3;
        assert!(check_consciousness_config(&config).unwrap_err().contains("min_voter_consciousness_standard"));
    }

    #[test]
    fn test_consciousness_config_equal_thresholds_ok() {
        // Equal thresholds should be fine (non-decreasing, not strictly increasing)
        let mut config = default_config();
        config.consciousness_gate_basic = 0.3;
        config.consciousness_gate_proposal = 0.3;
        assert!(check_consciousness_config(&config).is_ok());
    }

    #[test]
    fn test_consciousness_config_custom_values() {
        // Governance could lower basic and raise constitutional
        let mut config = default_config();
        config.consciousness_gate_basic = 0.1;
        config.consciousness_gate_constitutional = 0.8;
        config.min_voter_consciousness_constitutional = 0.7;
        assert!(check_consciousness_config(&config).is_ok());
    }

    #[test]
    fn test_consciousness_config_max_voting_weight_default() {
        let config = default_config();
        assert!((config.max_voting_weight - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consciousness_config_max_voting_weight_range() {
        let mut config = default_config();

        // Too low
        config.max_voting_weight = 0.3;
        assert!(check_consciousness_config(&config).unwrap_err().contains("max_voting_weight"));

        // Too high
        config.max_voting_weight = 6.0;
        assert!(check_consciousness_config(&config).unwrap_err().contains("max_voting_weight"));

        // Valid custom value
        config.max_voting_weight = 2.0;
        assert!(check_consciousness_config(&config).is_ok());
    }
}
