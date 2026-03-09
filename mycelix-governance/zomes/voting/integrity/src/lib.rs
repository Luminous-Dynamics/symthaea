//! Voting Integrity Zome
//! Defines entry types and validation for governance voting
//!
//! Enhanced with Φ-weighted voting, quadratic voting, and delegation decay
//! Per GIS v4.0 integration plan
//!
//! ## ZK-STARK Vote Eligibility Integration
//!
//! This module now supports privacy-preserving vote eligibility verification
//! using ZK-STARK proofs. Voters can prove they meet proposal requirements
//! without revealing their exact scores (assurance, MATL, stake, etc.).
//!
//! See `EligibilityProof` and `cast_verified_vote` for details.

#![allow(
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::unnecessary_cast
)]

use hdi::prelude::*;

/// How the Phi value in a governance decision was obtained.
///
/// Mirrors `bridge_integrity::PhiProvenance` — defined here to avoid
/// cross-zome integrity dependencies (which cause duplicate HDI symbols in WASM).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiProvenance {
    /// Signed attestation from agent's Symthaea instance
    Attested,
    /// Legacy unsigned ConsciousnessSnapshot
    Snapshot,
    /// No Phi data available — reputation-only voting
    Unavailable,
}

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// Φ (PHI) WEIGHTED VOTING SYSTEM
// ============================================================================

// Canonical governance Phi thresholds — must match mycelix_bridge_common::phi_thresholds.
// Source of truth: crates/mycelix-bridge-common/src/phi_thresholds.rs
// Note: Voting uses a 3-tier system; gov_basic (0.2) maps to the bridge, not voting.
const GOV_PROPOSAL: f64 = 0.3;
const GOV_VOTING: f64 = 0.4;
const GOV_CONSTITUTIONAL: f64 = 0.6;

/// Proposal tier based on governance impact
/// Different tiers require different Φ thresholds for approval
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalTier {
    /// Basic operational decisions (Φ ≥ 0.3)
    Basic,
    /// Significant policy changes (Φ ≥ 0.4)
    Major,
    /// Fundamental governance changes (Φ ≥ 0.6)
    Constitutional,
}

impl ProposalTier {
    /// Get the required Φ threshold for this tier
    pub fn phi_threshold(&self) -> f64 {
        match self {
            ProposalTier::Basic => GOV_PROPOSAL,
            ProposalTier::Major => GOV_VOTING,
            ProposalTier::Constitutional => GOV_CONSTITUTIONAL,
        }
    }

    /// Get the required quorum percentage for this tier
    pub fn quorum_requirement(&self) -> f64 {
        match self {
            ProposalTier::Basic => 0.15,          // 15% participation
            ProposalTier::Major => 0.25,          // 25% participation
            ProposalTier::Constitutional => 0.40, // 40% participation
        }
    }

    /// Get the approval threshold for this tier
    pub fn approval_threshold(&self) -> f64 {
        match self {
            ProposalTier::Basic => 0.50,          // Simple majority
            ProposalTier::Major => 0.60,          // 60% supermajority
            ProposalTier::Constitutional => 0.67, // 2/3 supermajority
        }
    }

    /// Default timelock duration in hours for this tier
    ///
    /// Higher-impact decisions get longer cooling-off periods.
    pub fn timelock_duration_hours(&self) -> u32 {
        match self {
            ProposalTier::Basic => 24,           // 1 day
            ProposalTier::Major => 72,           // 3 days
            ProposalTier::Constitutional => 168, // 7 days
        }
    }
}

/// Φ (Phi) weight components for consciousness-integrated governance
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PhiWeight {
    /// Base Φ score from consciousness metrics (0.0-1.0)
    pub phi_score: f64,
    /// How the phi_score was obtained (Attested, Snapshot, or Unavailable)
    #[serde(default = "default_provenance")]
    pub phi_provenance: PhiProvenance,
    /// K-vector derived trust score (0.0-1.0)
    pub k_trust: f64,
    /// Stake-based weight (from staking position)
    pub stake_weight: f64,
    /// Historical participation score (0.0-1.0)
    pub participation_score: f64,
    /// Reputation in relevant domain (0.0-1.0)
    pub domain_reputation: f64,
}

fn default_provenance() -> PhiProvenance {
    PhiProvenance::Unavailable
}

impl PhiWeight {
    /// Calculate composite voting weight using Φ-integration
    /// Formula: W = (0.3×Φ + 0.25×K + 0.2×S + 0.15×P + 0.1×D)
    pub fn composite_weight(&self) -> f64 {
        0.30 * self.phi_score +
        0.25 * self.k_trust +
        0.20 * self.stake_weight.min(1.0) +  // Cap stake influence
        0.15 * self.participation_score +
        0.10 * self.domain_reputation
    }

    /// Check if voter meets minimum Φ threshold for a tier
    pub fn meets_threshold(&self, tier: &ProposalTier) -> bool {
        self.phi_score >= tier.phi_threshold()
    }

    /// Create a default weight for new participants
    pub fn default_participant() -> Self {
        Self {
            phi_score: 0.1,
            phi_provenance: PhiProvenance::Unavailable,
            k_trust: 0.1,
            stake_weight: 0.0,
            participation_score: 0.0,
            domain_reputation: 0.0,
        }
    }
}

/// A vote on a proposal with Φ-weighted voting support
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// Vote identifier
    pub id: String,
    /// Proposal being voted on
    pub proposal_id: String,
    /// Voter's DID
    pub voter: String,
    /// Vote choice
    pub choice: VoteChoice,
    /// Vote weight (MATL-adjusted) - legacy field
    pub weight: f64,
    /// Optional reason/comment
    pub reason: Option<String>,
    /// Whether vote is delegated
    pub delegated: bool,
    /// If delegated, original voter's DID
    pub delegator: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

/// Φ-weighted vote with full consciousness integration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PhiWeightedVote {
    /// Vote identifier
    pub id: String,
    /// Proposal being voted on
    pub proposal_id: String,
    /// Proposal tier (determines Φ threshold)
    pub proposal_tier: ProposalTier,
    /// Voter's DID
    pub voter: String,
    /// Vote choice
    pub choice: VoteChoice,
    /// Full Φ weight breakdown
    pub phi_weight: PhiWeight,
    /// Final calculated vote weight
    pub effective_weight: f64,
    /// Optional reason/comment
    pub reason: Option<String>,
    /// Whether vote is delegated
    pub delegated: bool,
    /// If delegated, original voter's DID
    pub delegator: Option<String>,
    /// Delegation chain (for transitive delegation tracking)
    pub delegation_chain: Vec<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

/// Vote choices
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

// ============================================================================
// QUADRATIC VOTING
// ============================================================================

/// Quadratic vote - weight = √(credits spent)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QuadraticVote {
    /// Vote identifier
    pub id: String,
    /// Proposal being voted on
    pub proposal_id: String,
    /// Voter's DID
    pub voter: String,
    /// Vote choice
    pub choice: VoteChoice,
    /// Voice credits spent (actual cost)
    pub credits_spent: u64,
    /// Effective vote weight (√credits_spent)
    pub effective_weight: f64,
    /// Optional reason/comment
    pub reason: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

impl QuadraticVote {
    /// Calculate quadratic vote weight from credits
    pub fn calculate_weight(credits: u64) -> f64 {
        (credits as f64).sqrt()
    }
}

/// Voice credit balance for quadratic voting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VoiceCredits {
    /// Owner's DID
    pub owner: String,
    /// Total credits allocated this period
    pub allocated: u64,
    /// Credits spent this period
    pub spent: u64,
    /// Credits remaining
    pub remaining: u64,
    /// Period start timestamp
    pub period_start: Timestamp,
    /// Period end timestamp (credits reset)
    pub period_end: Timestamp,
}

// ============================================================================
// DELEGATION WITH DECAY
// ============================================================================

/// Delegation decay model
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DelegationDecay {
    /// No decay - constant delegation
    None,
    /// Linear decay over time
    Linear {
        /// Days until full decay
        decay_days: u32,
    },
    /// Exponential decay (half-life model)
    Exponential {
        /// Half-life in days
        half_life_days: u32,
    },
    /// Step decay - drops at intervals
    Step {
        /// Days between steps
        step_interval_days: u32,
        /// Percentage drop per step (0.0-1.0)
        drop_per_step: f64,
    },
}

impl DelegationDecay {
    /// Calculate decay multiplier given elapsed days
    pub fn calculate_multiplier(&self, elapsed_days: f64) -> f64 {
        match self {
            DelegationDecay::None => 1.0,
            DelegationDecay::Linear { decay_days } => {
                let ratio = elapsed_days / *decay_days as f64;
                (1.0 - ratio).max(0.0)
            }
            DelegationDecay::Exponential { half_life_days } => {
                let half_lives = elapsed_days / *half_life_days as f64;
                0.5_f64.powf(half_lives)
            }
            DelegationDecay::Step {
                step_interval_days,
                drop_per_step,
            } => {
                let steps = (elapsed_days / *step_interval_days as f64).floor() as u32;
                let remaining = 1.0 - (*drop_per_step * steps as f64);
                remaining.max(0.0)
            }
        }
    }
}

/// Vote delegation with decay support
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Delegation {
    /// Delegation identifier
    pub id: String,
    /// Delegator's DID (person delegating)
    pub delegator: String,
    /// Delegate's DID (person receiving delegation)
    pub delegate: String,
    /// Delegation percentage (0.0 to 1.0)
    pub percentage: f64,
    /// Topic filter (None = all topics)
    pub topics: Option<Vec<String>>,
    /// Proposal tier filter (None = all tiers)
    pub tier_filter: Option<Vec<ProposalTier>>,
    /// Whether delegation is active
    pub active: bool,
    /// Decay model for this delegation
    pub decay: DelegationDecay,
    /// Whether delegation is transitive (can be re-delegated)
    pub transitive: bool,
    /// Maximum chain depth for transitive delegation
    pub max_chain_depth: u8,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last renewed timestamp (resets decay)
    pub renewed: Timestamp,
    /// Expiration (None = no expiry, subject to decay)
    pub expires: Option<Timestamp>,
}

impl Delegation {
    /// Calculate current effective percentage with decay applied
    pub fn effective_percentage(&self, current_time: Timestamp) -> f64 {
        // Calculate days since creation or last renewal
        let elapsed_micros = current_time.as_micros() - self.renewed.as_micros();
        let elapsed_days = elapsed_micros as f64 / (24.0 * 60.0 * 60.0 * 1_000_000.0);

        let decay_multiplier = self.decay.calculate_multiplier(elapsed_days);
        self.percentage * decay_multiplier
    }

    /// Check if delegation is effectively expired (below threshold)
    pub fn is_effectively_expired(&self, current_time: Timestamp, min_threshold: f64) -> bool {
        self.effective_percentage(current_time) < min_threshold
    }
}

// ============================================================================
// Φ-WEIGHTED VOTE TALLY
// ============================================================================

/// Vote tally for a proposal (legacy)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VoteTally {
    /// Proposal ID
    pub proposal_id: String,
    /// Total votes for
    pub votes_for: f64,
    /// Total votes against
    pub votes_against: f64,
    /// Total abstentions
    pub abstentions: f64,
    /// Total vote weight cast
    pub total_weight: f64,
    /// Quorum reached
    pub quorum_reached: bool,
    /// Approval threshold met
    pub approved: bool,
    /// Tally timestamp
    pub tallied_at: Timestamp,
    /// Whether tally is final
    pub final_tally: bool,
}

/// Φ-weighted vote tally with full breakdown
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PhiWeightedTally {
    /// Proposal ID
    pub proposal_id: String,
    /// Proposal tier
    pub tier: ProposalTier,
    /// Φ-weighted votes for
    pub phi_votes_for: f64,
    /// Φ-weighted votes against
    pub phi_votes_against: f64,
    /// Φ-weighted abstentions
    pub phi_abstentions: f64,
    /// Raw unweighted votes for (for transparency)
    pub raw_votes_for: u64,
    /// Raw unweighted votes against
    pub raw_votes_against: u64,
    /// Raw abstentions
    pub raw_abstentions: u64,
    /// Average Φ score of voters
    pub average_phi: f64,
    /// Total Φ-weighted participation
    pub total_phi_weight: f64,
    /// Eligible voter count
    pub eligible_voters: u64,
    /// Quorum requirement for this tier
    pub quorum_requirement: f64,
    /// Quorum reached
    pub quorum_reached: bool,
    /// Approval threshold for this tier
    pub approval_threshold: f64,
    /// Approval reached
    pub approved: bool,
    /// Tally timestamp
    pub tallied_at: Timestamp,
    /// Whether tally is final
    pub final_tally: bool,
    /// Breakdown by Φ tier (for analysis)
    pub phi_tier_breakdown: PhiTierBreakdown,
    /// Votes with real Phi data (Attested or Snapshot)
    #[serde(default)]
    pub phi_enhanced_count: u64,
    /// Votes without Phi data (reputation-only)
    #[serde(default)]
    pub reputation_only_count: u64,
    /// Fraction of votes with verified consciousness data (phi_enhanced / total)
    #[serde(default)]
    pub phi_coverage: f64,
}

/// Breakdown of votes by voter Φ tier
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PhiTierBreakdown {
    /// Votes from high-Φ voters (≥0.6)
    pub high_phi_votes: TallySegment,
    /// Votes from medium-Φ voters (0.4-0.6)
    pub medium_phi_votes: TallySegment,
    /// Votes from low-Φ voters (<0.4)
    pub low_phi_votes: TallySegment,
}

/// Segment of tally for analysis
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TallySegment {
    pub votes_for: f64,
    pub votes_against: f64,
    pub abstentions: f64,
    pub voter_count: u64,
}

/// Quadratic vote tally
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QuadraticTally {
    /// Proposal ID
    pub proposal_id: String,
    /// Total quadratic weight for
    pub qv_for: f64,
    /// Total quadratic weight against
    pub qv_against: f64,
    /// Total credits spent
    pub total_credits_spent: u64,
    /// Average credits per voter
    pub avg_credits_per_voter: f64,
    /// Voter count
    pub voter_count: u64,
    /// Quorum reached
    pub quorum_reached: bool,
    /// Approved
    pub approved: bool,
    /// Tally timestamp
    pub tallied_at: Timestamp,
    /// Whether tally is final
    pub final_tally: bool,
}

// ============================================================================
// ZK-STARK VOTE ELIGIBILITY PROOFS
// ============================================================================

/// Proposal type for ZK eligibility proofs
///
/// Maps to ProofProposalType in fl-aggregator/proofs/circuits/vote.rs
/// Each type has different eligibility requirements (assurance, MATL, stake, etc.)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ZkProposalType {
    /// Standard operational proposals (lowest requirements)
    Standard = 0,
    /// Constitutional changes (highest requirements)
    Constitutional = 1,
    /// Model governance (requires FL participation)
    ModelGovernance = 2,
    /// Emergency proposals (high requirements, expedited)
    Emergency = 3,
    /// Treasury proposals (financial stake required)
    Treasury = 4,
    /// Membership changes
    Membership = 5,
}

impl ZkProposalType {
    /// Convert from numeric value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ZkProposalType::Standard),
            1 => Some(ZkProposalType::Constitutional),
            2 => Some(ZkProposalType::ModelGovernance),
            3 => Some(ZkProposalType::Emergency),
            4 => Some(ZkProposalType::Treasury),
            5 => Some(ZkProposalType::Membership),
            _ => None,
        }
    }

    /// Map ProposalTier to ZkProposalType
    pub fn from_tier(tier: &ProposalTier) -> Self {
        match tier {
            ProposalTier::Basic => ZkProposalType::Standard,
            ProposalTier::Major => ZkProposalType::Treasury, // Major decisions require stake
            ProposalTier::Constitutional => ZkProposalType::Constitutional,
        }
    }
}

/// ZK-STARK eligibility proof stored on-chain
///
/// This entry stores the serialized STARK proof that a voter meets
/// the requirements for a specific proposal type, without revealing
/// their actual scores.
///
/// ## Privacy Guarantees
///
/// - **Hidden**: Voter's exact assurance level, MATL score, stake amount,
///   account age, participation rate, humanity proof status, FL contributions
/// - **Public**: Voter commitment (hash), proposal type, eligibility result,
///   number of requirements met
///
/// ## Security Properties
///
/// - **Soundness**: Cannot forge proof for requirements not met (~96-bit security)
/// - **Binding**: Voter commitment prevents proof reuse with different identity
/// - **Non-transferable**: Proof is bound to specific voter commitment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EligibilityProof {
    /// Unique identifier
    pub id: String,
    /// Voter's DID
    pub voter_did: String,
    /// Commitment to voter profile (Blake3 hash)
    /// Binds the proof to the voter's actual credentials
    pub voter_commitment: Vec<u8>,
    /// Proposal type this proof is valid for
    pub proposal_type: ZkProposalType,
    /// Whether voter meets all requirements
    pub eligible: bool,
    /// Number of requirements satisfied
    pub requirements_met: u8,
    /// Total active requirements for this proposal type
    pub active_requirements: u8,
    /// Serialized STARK proof bytes
    pub proof_bytes: Vec<u8>,
    /// Proof generation timestamp
    pub generated_at: Timestamp,
    /// Optional expiration (proofs become stale as voter profile changes)
    pub expires_at: Option<Timestamp>,
}

impl EligibilityProof {
    /// Default proof validity period (24 hours)
    pub const DEFAULT_VALIDITY_HOURS: i64 = 24;

    /// Maximum proof size (prevent DoS)
    pub const MAX_PROOF_SIZE: usize = 500_000; // 500KB

    /// Check if the proof has expired
    pub fn is_expired(&self, current_time: Timestamp) -> bool {
        if let Some(expires) = self.expires_at {
            current_time.as_micros() > expires.as_micros()
        } else {
            false
        }
    }

    /// Check if this proof is valid for a given proposal tier
    pub fn is_valid_for_tier(&self, tier: &ProposalTier) -> bool {
        let required_type = ZkProposalType::from_tier(tier);

        // Higher requirement proofs are valid for lower tiers
        match (self.proposal_type, required_type) {
            // Constitutional proofs work for anything
            (ZkProposalType::Constitutional, _) => self.eligible,
            (ZkProposalType::Emergency, _) => self.eligible,
            // Treasury/Major proofs work for Basic
            (ZkProposalType::Treasury, ZkProposalType::Standard) => self.eligible,
            (ZkProposalType::Treasury, ZkProposalType::Treasury) => self.eligible,
            // Exact match
            (a, b) if a == b => self.eligible,
            // Standard only works for Standard
            (ZkProposalType::Standard, ZkProposalType::Standard) => self.eligible,
            _ => false,
        }
    }
}

/// Vote with ZK eligibility proof
///
/// This vote type requires a valid ZK-STARK proof of eligibility.
/// The voter proves they meet requirements without revealing exact scores.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiedVote {
    /// Vote identifier
    pub id: String,
    /// Proposal being voted on
    pub proposal_id: String,
    /// Proposal tier
    pub proposal_tier: ProposalTier,
    /// Voter's DID
    pub voter: String,
    /// Vote choice
    pub choice: VoteChoice,
    /// Reference to the eligibility proof entry
    pub eligibility_proof_hash: ActionHash,
    /// Voter commitment (must match proof)
    pub voter_commitment: Vec<u8>,
    /// Effective weight (capped, computed from proof validation)
    pub effective_weight: f64,
    /// Optional reason/comment
    pub reason: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

// ============================================================================
// PROOF ATTESTATION (External Verifier Integration)
// ============================================================================

/// Attestation from an external verifier that a proof has been validated
///
/// Since Holochain WASM cannot run full STARK verification (Winterfell is too
/// heavy), proofs are verified by external oracle services that produce signed
/// attestations. This entry stores those attestations.
///
/// ## Verification Flow
///
/// 1. Voter generates eligibility proof using fl-aggregator
/// 2. Voter stores proof on-chain via `store_eligibility_proof`
/// 3. External verifier fetches proof, runs STARK verification
/// 4. Verifier produces signed attestation, stores via `store_proof_attestation`
/// 5. Vote can now be cast with reference to verified proof
///
/// ## Security Model
///
/// - Multiple verifiers can attest to the same proof
/// - Attestations are signed with Ed25519
/// - Attestations expire (default 24 hours)
/// - Verifier public keys should be registered/trusted by governance
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProofAttestation {
    /// Unique identifier
    pub id: String,
    /// Hash of the proof being attested (Blake3)
    pub proof_hash: Vec<u8>,
    /// Reference to the eligibility proof entry
    pub proof_action_hash: ActionHash,
    /// Voter commitment from the proof (for cross-check)
    pub voter_commitment: Vec<u8>,
    /// Proposal type the proof is valid for
    pub proposal_type: ZkProposalType,
    /// Whether the STARK proof verified successfully
    pub verified: bool,
    /// Verification timestamp
    pub verified_at: Timestamp,
    /// Attestation expiration
    pub expires_at: Timestamp,
    /// Verifier's Ed25519 public key (32 bytes)
    pub verifier_pubkey: Vec<u8>,
    /// Ed25519 signature over attestation data (64 bytes)
    pub signature: Vec<u8>,
    /// Security level used in verification
    pub security_level: String,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
}

impl ProofAttestation {
    /// Check if attestation has expired
    pub fn is_expired(&self, current_time: Timestamp) -> bool {
        current_time.as_micros() > self.expires_at.as_micros()
    }

    /// Check if attestation is valid (verified and not expired)
    pub fn is_valid(&self, current_time: Timestamp) -> bool {
        self.verified && !self.is_expired(current_time)
    }
}

// ============================================================================
// COLLECTIVE MIRROR: SWARM WISDOM SENSING
// ============================================================================

/// Risk level for echo chamber detection
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EchoChamberRiskLevel {
    Low,
    Moderate,
    High,
    Critical,
}

/// Topology pattern for agreement structure
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TopologyPattern {
    /// Many independent voices (healthy)
    Mesh,
    /// A few central influencers
    HubAndSpoke,
    /// Two opposing camps
    Polarized,
    /// Single dominant voice
    Monopole,
    /// Unknown/insufficient data
    Unknown,
}

/// Trend direction for temporal analysis
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Rising,
    Stable,
    Falling,
    Unknown,
}

/// Stored reflection of collective state during proposal voting
///
/// This is a lightweight on-chain representation of the full CollectiveMirror
/// analysis. The coordinator generates this from the heavy analysis.
///
/// ## The Mirror Philosophy
///
/// This entry reflects what IS, not what SHOULD BE. It doesn't claim
/// wisdom - it shows the group how they look so they can decide what
/// to do with that information.
///
/// ## Key Metrics
///
/// - **Topology**: Is agreement distributed (mesh) or centralized (hub-and-spoke)?
/// - **Shadow**: Which values/perspectives are absent from the conversation?
/// - **Signal Integrity**: Is consensus verified or just an echo chamber?
/// - **Trajectory**: Is the group converging, diverging, or stable?
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProposalReflection {
    /// Unique identifier
    pub id: String,
    /// Proposal being reflected on
    pub proposal_id: String,
    /// Timestamp of this reflection
    pub timestamp: Timestamp,
    /// Number of voters analyzed
    pub voter_count: u64,

    // === TOPOLOGY ===
    /// Agreement structure pattern
    pub topology_pattern: TopologyPattern,
    /// Agreement centralization (0=distributed, 1=single voice)
    pub centralization: f64,
    /// Number of distinct "camps" or clusters
    pub cluster_count: u8,

    // === SHADOW ===
    /// Harmonies absent from the conversation (by name)
    pub absent_harmonies: Vec<String>,
    /// Percentage of harmonies represented (0-1)
    pub harmony_coverage: f64,

    // === SIGNAL INTEGRITY ===
    /// Average epistemic level of voters (0-1)
    pub average_epistemic_level: f64,
    /// Echo chamber risk assessment
    pub echo_chamber_risk: EchoChamberRiskLevel,
    /// Whether high agreement is backed by verification
    pub agreement_verified: bool,

    // === TRAJECTORY ===
    /// Direction of agreement trend
    pub agreement_trend: TrendDirection,
    /// Direction of centralization trend
    pub centralization_trend: TrendDirection,
    /// Warning flags
    pub rapid_convergence_warning: bool,
    pub fragmentation_warning: bool,

    // === VOTE SUMMARY ===
    /// Votes for (raw count)
    pub votes_for: u64,
    /// Votes against (raw count)
    pub votes_against: u64,
    /// Abstentions
    pub abstentions: u64,
    /// Weighted approval ratio
    pub approval_ratio: f64,
    /// Polarization score (0=consensus, 1=split)
    pub polarization: f64,

    // === PROMPTS & INTERVENTIONS ===
    /// Suggested interventions (human-readable)
    pub suggested_interventions: Vec<String>,
    /// Reflection prompts for the group
    pub reflection_prompts: Vec<String>,

    /// Whether this reflection suggests review before finalizing
    pub needs_review: bool,
    /// Human-readable summary
    pub summary: String,
}

impl ProposalReflection {
    /// Check if this reflection flags any concerns
    pub fn has_concerns(&self) -> bool {
        self.needs_review
            || matches!(
                self.echo_chamber_risk,
                EchoChamberRiskLevel::High | EchoChamberRiskLevel::Critical
            )
            || self.rapid_convergence_warning
            || self.fragmentation_warning
            || self.harmony_coverage < 0.3
    }

    /// Get a severity score (0-10) based on concerns
    pub fn concern_severity(&self) -> u8 {
        let mut severity = 0u8;

        match self.echo_chamber_risk {
            EchoChamberRiskLevel::Critical => severity += 4,
            EchoChamberRiskLevel::High => severity += 3,
            EchoChamberRiskLevel::Moderate => severity += 1,
            EchoChamberRiskLevel::Low => {}
        }

        if self.rapid_convergence_warning {
            severity += 2;
        }
        if self.fragmentation_warning {
            severity += 2;
        }
        if self.harmony_coverage < 0.3 {
            severity += 2;
        }
        if self.centralization > 0.8 {
            severity += 1;
        }

        severity.min(10)
    }
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Vote(Vote),
    PhiWeightedVote(PhiWeightedVote),
    QuadraticVote(QuadraticVote),
    VoiceCredits(VoiceCredits),
    Delegation(Delegation),
    VoteTally(VoteTally),
    PhiWeightedTally(PhiWeightedTally),
    QuadraticTally(QuadraticTally),
    /// ZK-STARK eligibility proof
    EligibilityProof(EligibilityProof),
    /// Vote with ZK eligibility verification
    VerifiedVote(VerifiedVote),
    /// External verifier attestation for proof validity
    ProofAttestation(ProofAttestation),
    /// Collective mirror reflection on proposal voting
    ProposalReflection(ProposalReflection),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Proposal to votes
    ProposalToVote,
    /// Proposal to Φ-weighted votes
    ProposalToPhiVote,
    /// Proposal to quadratic votes
    ProposalToQuadraticVote,
    /// Voter to their votes
    VoterToVote,
    /// Voter to their voice credits
    VoterToVoiceCredits,
    /// Delegator to delegations
    DelegatorToDelegation,
    /// Delegate to delegations
    DelegateToDelegation,
    /// Proposal to tally
    ProposalToTally,
    /// Proposal to Φ-weighted tally
    ProposalToPhiTally,
    /// Proposal to quadratic tally
    ProposalToQuadraticTally,
    /// Voter to their eligibility proofs
    VoterToEligibilityProof,
    /// Proposal to verified votes
    ProposalToVerifiedVote,
    /// Eligibility proof to attestations
    ProofToAttestation,
    /// Verifier to their attestations
    VerifierToAttestation,
    /// Proposal to collective mirror reflections
    ProposalToReflection,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Vote(vote) => validate_create_vote(action, vote),
                EntryTypes::PhiWeightedVote(vote) => validate_create_phi_vote(action, vote),
                EntryTypes::QuadraticVote(vote) => validate_create_quadratic_vote(action, vote),
                EntryTypes::VoiceCredits(credits) => validate_create_voice_credits(action, credits),
                EntryTypes::Delegation(delegation) => {
                    validate_create_delegation(action, delegation)
                }
                EntryTypes::VoteTally(tally) => validate_create_tally(action, tally),
                EntryTypes::PhiWeightedTally(tally) => validate_create_phi_tally(action, tally),
                EntryTypes::QuadraticTally(tally) => validate_create_quadratic_tally(action, tally),
                EntryTypes::EligibilityProof(proof) => {
                    validate_create_eligibility_proof(action, proof)
                }
                EntryTypes::VerifiedVote(vote) => validate_create_verified_vote(action, vote),
                EntryTypes::ProofAttestation(attestation) => {
                    validate_create_proof_attestation(action, attestation)
                }
                EntryTypes::ProposalReflection(reflection) => {
                    validate_create_proposal_reflection(action, reflection)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Vote(_)
                | EntryTypes::PhiWeightedVote(_)
                | EntryTypes::QuadraticVote(_)
                | EntryTypes::VerifiedVote(_) => {
                    // Votes cannot be updated once cast
                    Ok(ValidateCallbackResult::Invalid(
                        "Votes cannot be modified after casting".into(),
                    ))
                }
                EntryTypes::EligibilityProof(_) => {
                    // Proofs are immutable - generate a new one instead
                    Ok(ValidateCallbackResult::Invalid(
                        "Eligibility proofs cannot be updated - generate a new proof".into(),
                    ))
                }
                EntryTypes::ProofAttestation(_) => {
                    // Attestations are immutable - verifier must issue a new one
                    Ok(ValidateCallbackResult::Invalid(
                        "Proof attestations cannot be updated - issue a new attestation".into(),
                    ))
                }
                EntryTypes::VoiceCredits(credits) => validate_update_voice_credits(action, credits),
                EntryTypes::Delegation(delegation) => {
                    validate_update_delegation(action, delegation)
                }
                EntryTypes::VoteTally(tally) => validate_update_tally(action, tally),
                EntryTypes::PhiWeightedTally(tally) => validate_update_phi_tally(action, tally),
                EntryTypes::QuadraticTally(tally) => validate_update_quadratic_tally(action, tally),
                EntryTypes::ProposalReflection(reflection) => {
                    validate_update_proposal_reflection(action, reflection)
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
            LinkTypes::ProposalToVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToPhiVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToQuadraticVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VoterToVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VoterToVoiceCredits => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DelegatorToDelegation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DelegateToDelegation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToTally => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToPhiTally => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToQuadraticTally => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VoterToEligibilityProof => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToVerifiedVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProofToAttestation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VerifierToAttestation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToReflection => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            // Allow removing delegation links when delegation is revoked
            LinkTypes::DelegatorToDelegation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DelegateToDelegation => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate vote creation
fn validate_create_vote(_action: Create, vote: Vote) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !vote.voter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate weight is positive
    if vote.weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote weight cannot be negative".into(),
        ));
    }

    // Validate delegator if delegated
    if vote.delegated && vote.delegator.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegated vote must have delegator".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate delegation creation
fn validate_create_delegation(
    _action: Create,
    delegation: Delegation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate delegator is a DID
    if !delegation.delegator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegator must be a valid DID".into(),
        ));
    }

    // Validate delegate is a DID
    if !delegation.delegate.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegate must be a valid DID".into(),
        ));
    }

    // Cannot delegate to self
    if delegation.delegator == delegation.delegate {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot delegate to yourself".into(),
        ));
    }

    // Validate percentage range
    if delegation.percentage <= 0.0 || delegation.percentage > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegation percentage must be between 0 and 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate delegation update
fn validate_update_delegation(
    _action: Update,
    delegation: Delegation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate percentage range on update
    if delegation.percentage <= 0.0 || delegation.percentage > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegation percentage must be between 0 and 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate vote tally creation
fn validate_create_tally(
    _action: Create,
    tally: VoteTally,
) -> ExternResult<ValidateCallbackResult> {
    // Validate votes are non-negative
    if tally.votes_for < 0.0 || tally.votes_against < 0.0 || tally.abstentions < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote counts cannot be negative".into(),
        ));
    }

    // Validate total weight matches
    let expected_total = tally.votes_for + tally.votes_against + tally.abstentions;
    if (tally.total_weight - expected_total).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total weight must equal sum of votes".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate vote tally update
fn validate_update_tally(
    _action: Update,
    tally: VoteTally,
) -> ExternResult<ValidateCallbackResult> {
    // Same validation as create - validate the tally data directly
    // Validate votes are non-negative
    if tally.votes_for < 0.0 || tally.votes_against < 0.0 || tally.abstentions < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote counts cannot be negative".into(),
        ));
    }

    // Validate total weight matches
    let expected_total = tally.votes_for + tally.votes_against + tally.abstentions;
    if (tally.total_weight - expected_total).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total weight must equal sum of votes".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Φ-WEIGHTED VOTING VALIDATION
// ============================================================================

/// Validate Φ-weighted vote creation
fn validate_create_phi_vote(
    _action: Create,
    vote: PhiWeightedVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !vote.voter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate Φ weight components are in valid range
    let pw = &vote.phi_weight;
    if pw.phi_score < 0.0 || pw.phi_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Φ score must be between 0 and 1".into(),
        ));
    }
    if pw.k_trust < 0.0 || pw.k_trust > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-trust must be between 0 and 1".into(),
        ));
    }
    if pw.stake_weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Stake weight cannot be negative".into(),
        ));
    }
    if pw.participation_score < 0.0 || pw.participation_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Participation score must be between 0 and 1".into(),
        ));
    }
    if pw.domain_reputation < 0.0 || pw.domain_reputation > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain reputation must be between 0 and 1".into(),
        ));
    }

    // Validate voter meets Φ threshold for this tier
    if !pw.meets_threshold(&vote.proposal_tier) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Voter Φ score ({:.2}) does not meet threshold ({:.2}) for {:?} tier",
            pw.phi_score,
            vote.proposal_tier.phi_threshold(),
            vote.proposal_tier
        )));
    }

    // Validate delegator if delegated
    if vote.delegated && vote.delegator.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegated vote must have delegator".into(),
        ));
    }

    // Validate delegation chain depth
    if vote.delegation_chain.len() > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegation chain too deep (max 5)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate quadratic vote creation
fn validate_create_quadratic_vote(
    _action: Create,
    vote: QuadraticVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !vote.voter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate credits spent is reasonable
    if vote.credits_spent == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must spend at least 1 credit to vote".into(),
        ));
    }

    // Validate effective weight calculation
    let expected_weight = (vote.credits_spent as f64).sqrt();
    if (vote.effective_weight - expected_weight).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Effective weight must equal √(credits_spent)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate voice credits creation
fn validate_create_voice_credits(
    _action: Create,
    credits: VoiceCredits,
) -> ExternResult<ValidateCallbackResult> {
    // Validate owner is a DID
    if !credits.owner.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }

    // Validate remaining = allocated - spent
    if credits.remaining != credits.allocated - credits.spent {
        return Ok(ValidateCallbackResult::Invalid(
            "Remaining credits must equal allocated minus spent".into(),
        ));
    }

    // Validate period
    if credits.period_end <= credits.period_start {
        return Ok(ValidateCallbackResult::Invalid(
            "Period end must be after period start".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate voice credits update
fn validate_update_voice_credits(
    _action: Update,
    credits: VoiceCredits,
) -> ExternResult<ValidateCallbackResult> {
    // Validate remaining = allocated - spent
    if credits.remaining != credits.allocated - credits.spent {
        return Ok(ValidateCallbackResult::Invalid(
            "Remaining credits must equal allocated minus spent".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Φ-weighted tally creation
fn validate_create_phi_tally(
    _action: Create,
    tally: PhiWeightedTally,
) -> ExternResult<ValidateCallbackResult> {
    // Validate votes are non-negative
    if tally.phi_votes_for < 0.0 || tally.phi_votes_against < 0.0 || tally.phi_abstentions < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote counts cannot be negative".into(),
        ));
    }

    // Validate total weight matches
    let expected_total = tally.phi_votes_for + tally.phi_votes_against + tally.phi_abstentions;
    if (tally.total_phi_weight - expected_total).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total Φ weight must equal sum of votes".into(),
        ));
    }

    // Validate average Φ is in range
    if tally.average_phi < 0.0 || tally.average_phi > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Average Φ must be between 0 and 1".into(),
        ));
    }

    // Validate thresholds match tier
    if (tally.quorum_requirement - tally.tier.quorum_requirement()).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quorum requirement must match tier".into(),
        ));
    }
    if (tally.approval_threshold - tally.tier.approval_threshold()).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Approval threshold must match tier".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Φ-weighted tally update
fn validate_update_phi_tally(
    _action: Update,
    tally: PhiWeightedTally,
) -> ExternResult<ValidateCallbackResult> {
    // Same validation as create
    if tally.phi_votes_for < 0.0 || tally.phi_votes_against < 0.0 || tally.phi_abstentions < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote counts cannot be negative".into(),
        ));
    }

    let expected_total = tally.phi_votes_for + tally.phi_votes_against + tally.phi_abstentions;
    if (tally.total_phi_weight - expected_total).abs() > 0.001 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total Φ weight must equal sum of votes".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate quadratic tally creation
fn validate_create_quadratic_tally(
    _action: Create,
    tally: QuadraticTally,
) -> ExternResult<ValidateCallbackResult> {
    // Validate votes are non-negative
    if tally.qv_for < 0.0 || tally.qv_against < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quadratic vote counts cannot be negative".into(),
        ));
    }

    // Validate voter count matches average
    if tally.voter_count > 0 {
        let expected_avg = tally.total_credits_spent as f64 / tally.voter_count as f64;
        if (tally.avg_credits_per_voter - expected_avg).abs() > 0.001 {
            return Ok(ValidateCallbackResult::Invalid(
                "Average credits must equal total / voter count".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate quadratic tally update
fn validate_update_quadratic_tally(
    _action: Update,
    tally: QuadraticTally,
) -> ExternResult<ValidateCallbackResult> {
    // Validate votes are non-negative
    if tally.qv_for < 0.0 || tally.qv_against < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quadratic vote counts cannot be negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// ZK-STARK ELIGIBILITY PROOF VALIDATION
// ============================================================================

/// Validate eligibility proof creation
///
/// ## Security Note
///
/// This validation checks structural properties of the proof entry.
/// The actual STARK proof verification is done in the coordinator zome
/// using the fl-aggregator proof verification library.
///
/// Key validations:
/// 1. Voter DID is valid
/// 2. Commitment is correct length (32 bytes)
/// 3. Proof bytes don't exceed size limit (DoS prevention)
/// 4. Timestamps are reasonable
fn validate_create_eligibility_proof(
    _action: Create,
    proof: EligibilityProof,
) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !proof.voter_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate commitment length (Blake3 hash = 32 bytes)
    if proof.voter_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter commitment must be exactly 32 bytes".into(),
        ));
    }

    // Validate proof size (prevent DoS attacks)
    if proof.proof_bytes.len() > EligibilityProof::MAX_PROOF_SIZE {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Proof size {} exceeds maximum {}",
            proof.proof_bytes.len(),
            EligibilityProof::MAX_PROOF_SIZE
        )));
    }

    // Validate proof bytes are not empty
    if proof.proof_bytes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof bytes cannot be empty".into(),
        ));
    }

    // Validate requirements counts
    if proof.requirements_met > proof.active_requirements {
        return Ok(ValidateCallbackResult::Invalid(
            "Requirements met cannot exceed active requirements".into(),
        ));
    }

    // If eligible, all requirements must be met
    if proof.eligible && proof.requirements_met != proof.active_requirements {
        return Ok(ValidateCallbackResult::Invalid(
            "Eligible proof must have all requirements met".into(),
        ));
    }

    // Validate expiration is after generation
    if let Some(expires) = proof.expires_at {
        if expires.as_micros() <= proof.generated_at.as_micros() {
            return Ok(ValidateCallbackResult::Invalid(
                "Expiration must be after generation time".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate verified vote creation
///
/// ## Security Note
///
/// This validates the structural integrity of the vote.
/// The coordinator zome is responsible for:
/// 1. Fetching the referenced eligibility proof
/// 2. Verifying the STARK proof
/// 3. Checking the proof is not expired
/// 4. Ensuring voter commitment matches
fn validate_create_verified_vote(
    _action: Create,
    vote: VerifiedVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !vote.voter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate commitment length
    if vote.voter_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter commitment must be exactly 32 bytes".into(),
        ));
    }

    // Validate effective weight is positive
    if vote.effective_weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Effective weight cannot be negative".into(),
        ));
    }

    // Cap effective weight (prevents inflation attacks)
    const MAX_VERIFIED_VOTE_WEIGHT: f64 = 2.0;
    if vote.effective_weight > MAX_VERIFIED_VOTE_WEIGHT {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Effective weight {} exceeds maximum {}",
            vote.effective_weight, MAX_VERIFIED_VOTE_WEIGHT
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// PROOF ATTESTATION VALIDATION
// ============================================================================

/// Validate proof attestation creation
///
/// ## Security Checks
///
/// 1. Proof hash is correct length (32 bytes)
/// 2. Voter commitment is correct length (32 bytes)
/// 3. Verifier public key is correct length (32 bytes Ed25519)
/// 4. Signature is correct length (64 bytes Ed25519)
/// 5. Expiration is after verification time
/// 6. Proposal type is valid
///
/// Note: The actual signature verification should be done in the coordinator
/// zome before calling this, as HDI validation cannot perform cryptographic
/// operations reliably.
fn validate_create_proof_attestation(
    _action: Create,
    attestation: ProofAttestation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proof hash length (Blake3 = 32 bytes)
    if attestation.proof_hash.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof hash must be exactly 32 bytes".into(),
        ));
    }

    // Validate voter commitment length (Blake3 = 32 bytes)
    if attestation.voter_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter commitment must be exactly 32 bytes".into(),
        ));
    }

    // Validate verifier public key length (Ed25519 = 32 bytes)
    if attestation.verifier_pubkey.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier public key must be exactly 32 bytes".into(),
        ));
    }

    // Validate signature length (Ed25519 = 64 bytes)
    if attestation.signature.len() != 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature must be exactly 64 bytes".into(),
        ));
    }

    // Validate expiration is after verification time
    if attestation.expires_at.as_micros() <= attestation.verified_at.as_micros() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation expiration must be after verification time".into(),
        ));
    }

    // Validate security level is not empty
    if attestation.security_level.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Security level must not be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// PROPOSAL REFLECTION VALIDATION
// ============================================================================

/// Validate proposal reflection creation
///
/// ## Validation Checks
///
/// 1. Proposal ID is not empty
/// 2. Centralization is in valid range (0-1)
/// 3. Harmony coverage is in valid range (0-1)
/// 4. Epistemic level is in valid range (0-1)
/// 5. Approval ratio is in valid range (0-1)
/// 6. Polarization is in valid range (0-1)
fn validate_create_proposal_reflection(
    _action: Create,
    reflection: ProposalReflection,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proposal ID is not empty
    if reflection.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID cannot be empty".into(),
        ));
    }

    // Validate centralization range
    if reflection.centralization < 0.0 || reflection.centralization > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Centralization must be between 0 and 1".into(),
        ));
    }

    // Validate harmony coverage range
    if reflection.harmony_coverage < 0.0 || reflection.harmony_coverage > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harmony coverage must be between 0 and 1".into(),
        ));
    }

    // Validate epistemic level range
    if reflection.average_epistemic_level < 0.0 || reflection.average_epistemic_level > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Average epistemic level must be between 0 and 1".into(),
        ));
    }

    // Validate approval ratio range
    if reflection.approval_ratio < 0.0 || reflection.approval_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Approval ratio must be between 0 and 1".into(),
        ));
    }

    // Validate polarization range
    if reflection.polarization < 0.0 || reflection.polarization > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Polarization must be between 0 and 1".into(),
        ));
    }

    // Validate absent harmonies count (max 8 - the Eight Harmonies)
    if reflection.absent_harmonies.len() > 8 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 8 absent harmonies".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate proposal reflection update
fn validate_update_proposal_reflection(
    _action: Update,
    reflection: ProposalReflection,
) -> ExternResult<ValidateCallbackResult> {
    // Same validation as create
    if reflection.centralization < 0.0 || reflection.centralization > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Centralization must be between 0 and 1".into(),
        ));
    }

    if reflection.harmony_coverage < 0.0 || reflection.harmony_coverage > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harmony coverage must be between 0 and 1".into(),
        ));
    }

    if reflection.average_epistemic_level < 0.0 || reflection.average_epistemic_level > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Average epistemic level must be between 0 and 1".into(),
        ));
    }

    if reflection.approval_ratio < 0.0 || reflection.approval_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Approval ratio must be between 0 and 1".into(),
        ));
    }

    if reflection.polarization < 0.0 || reflection.polarization > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Polarization must be between 0 and 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ProposalTier tests
    // ========================================================================

    #[test]
    fn test_proposal_tier_phi_thresholds() {
        assert!((ProposalTier::Basic.phi_threshold() - 0.3).abs() < f64::EPSILON);
        assert!((ProposalTier::Major.phi_threshold() - 0.4).abs() < f64::EPSILON);
        assert!((ProposalTier::Constitutional.phi_threshold() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proposal_tier_quorum_requirements() {
        assert!((ProposalTier::Basic.quorum_requirement() - 0.15).abs() < f64::EPSILON);
        assert!((ProposalTier::Major.quorum_requirement() - 0.25).abs() < f64::EPSILON);
        assert!((ProposalTier::Constitutional.quorum_requirement() - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proposal_tier_approval_thresholds() {
        assert!((ProposalTier::Basic.approval_threshold() - 0.50).abs() < f64::EPSILON);
        assert!((ProposalTier::Major.approval_threshold() - 0.60).abs() < f64::EPSILON);
        assert!((ProposalTier::Constitutional.approval_threshold() - 0.67).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proposal_tier_timelock_durations() {
        assert_eq!(ProposalTier::Basic.timelock_duration_hours(), 24);
        assert_eq!(ProposalTier::Major.timelock_duration_hours(), 72);
        assert_eq!(ProposalTier::Constitutional.timelock_duration_hours(), 168);
    }

    // ========================================================================
    // PhiWeight tests
    // ========================================================================

    #[test]
    fn test_composite_weight_all_ones() {
        let pw = PhiWeight {
            phi_score: 1.0,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 1.0,
            stake_weight: 1.0,
            participation_score: 1.0,
            domain_reputation: 1.0,
        };
        // 0.3 + 0.25 + 0.2 + 0.15 + 0.1 = 1.0
        assert!((pw.composite_weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composite_weight_all_zeros() {
        let pw = PhiWeight {
            phi_score: 0.0,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 0.0,
            stake_weight: 0.0,
            participation_score: 0.0,
            domain_reputation: 0.0,
        };
        assert!((pw.composite_weight() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composite_weight_mixed() {
        let pw = PhiWeight {
            phi_score: 0.8,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 0.6,
            stake_weight: 0.5,
            participation_score: 0.4,
            domain_reputation: 0.3,
        };
        // 0.3*0.8 + 0.25*0.6 + 0.2*0.5 + 0.15*0.4 + 0.1*0.3
        // = 0.24 + 0.15 + 0.10 + 0.06 + 0.03 = 0.58
        assert!((pw.composite_weight() - 0.58).abs() < 0.001);
    }

    #[test]
    fn test_composite_weight_stake_capped_at_one() {
        let pw = PhiWeight {
            phi_score: 0.5,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 0.5,
            stake_weight: 5.0, // should be capped to 1.0
            participation_score: 0.5,
            domain_reputation: 0.5,
        };
        let pw_capped = PhiWeight {
            stake_weight: 1.0,
            ..pw.clone()
        };
        assert!((pw.composite_weight() - pw_capped.composite_weight()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meets_threshold_boundary() {
        // Exactly at threshold
        let pw = PhiWeight {
            phi_score: 0.3,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 0.0,
            stake_weight: 0.0,
            participation_score: 0.0,
            domain_reputation: 0.0,
        };
        assert!(pw.meets_threshold(&ProposalTier::Basic));
        assert!(!pw.meets_threshold(&ProposalTier::Major));
        assert!(!pw.meets_threshold(&ProposalTier::Constitutional));
    }

    #[test]
    fn test_meets_threshold_just_below() {
        let pw = PhiWeight {
            phi_score: 0.299,
            phi_provenance: PhiProvenance::Attested,
            k_trust: 1.0,
            stake_weight: 1.0,
            participation_score: 1.0,
            domain_reputation: 1.0,
        };
        // High everything else, but phi_score below Basic threshold
        assert!(!pw.meets_threshold(&ProposalTier::Basic));
    }

    #[test]
    fn test_default_participant() {
        let pw = PhiWeight::default_participant();
        assert!((pw.phi_score - 0.1).abs() < f64::EPSILON);
        assert!((pw.k_trust - 0.1).abs() < f64::EPSILON);
        assert!((pw.stake_weight - 0.0).abs() < f64::EPSILON);
        assert!((pw.participation_score - 0.0).abs() < f64::EPSILON);
        assert!((pw.domain_reputation - 0.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // DelegationDecay tests
    // ========================================================================

    #[test]
    fn test_decay_none() {
        let decay = DelegationDecay::None;
        assert!((decay.calculate_multiplier(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((decay.calculate_multiplier(100.0) - 1.0).abs() < f64::EPSILON);
        assert!((decay.calculate_multiplier(999.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_linear() {
        let decay = DelegationDecay::Linear { decay_days: 100 };
        assert!((decay.calculate_multiplier(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((decay.calculate_multiplier(50.0) - 0.5).abs() < f64::EPSILON);
        assert!((decay.calculate_multiplier(100.0) - 0.0).abs() < f64::EPSILON);
        // Past full decay — clamped to 0
        assert!((decay.calculate_multiplier(150.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_exponential() {
        let decay = DelegationDecay::Exponential { half_life_days: 30 };
        assert!((decay.calculate_multiplier(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((decay.calculate_multiplier(30.0) - 0.5).abs() < 0.001);
        assert!((decay.calculate_multiplier(60.0) - 0.25).abs() < 0.001);
        assert!((decay.calculate_multiplier(90.0) - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_decay_step() {
        let decay = DelegationDecay::Step {
            step_interval_days: 30,
            drop_per_step: 0.2,
        };
        // Day 0: no steps yet
        assert!((decay.calculate_multiplier(0.0) - 1.0).abs() < f64::EPSILON);
        // Day 15: still 0 steps
        assert!((decay.calculate_multiplier(15.0) - 1.0).abs() < f64::EPSILON);
        // Day 30: 1 step → 1 - 0.2 = 0.8
        assert!((decay.calculate_multiplier(30.0) - 0.8).abs() < 0.001);
        // Day 60: 2 steps → 1 - 0.4 = 0.6
        assert!((decay.calculate_multiplier(60.0) - 0.6).abs() < 0.001);
        // Day 150: 5 steps → 1 - 1.0 = 0.0
        assert!((decay.calculate_multiplier(150.0) - 0.0).abs() < 0.001);
        // Day 180: 6 steps → clamped to 0.0
        assert!((decay.calculate_multiplier(180.0) - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Delegation tests
    // ========================================================================

    fn make_delegation(percentage: f64, decay: DelegationDecay) -> Delegation {
        Delegation {
            id: "del-1".to_string(),
            delegator: "did:test:alice".to_string(),
            delegate: "did:test:bob".to_string(),
            percentage,
            topics: None,
            tier_filter: None,
            active: true,
            decay,
            transitive: false,
            max_chain_depth: 1,
            created: Timestamp::from_micros(0),
            renewed: Timestamp::from_micros(0),
            expires: None,
        }
    }

    #[test]
    fn test_effective_percentage_no_decay() {
        let d = make_delegation(0.75, DelegationDecay::None);
        let later = Timestamp::from_micros(86_400_000_000 * 100); // 100 days
        assert!((d.effective_percentage(later) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_effective_percentage_linear_decay() {
        let d = make_delegation(0.80, DelegationDecay::Linear { decay_days: 100 });
        // 50 days later: multiplier = 0.5, effective = 0.80 * 0.5 = 0.40
        let t50 = Timestamp::from_micros(86_400_000_000 * 50);
        assert!((d.effective_percentage(t50) - 0.40).abs() < 0.01);
    }

    #[test]
    fn test_is_effectively_expired() {
        let d = make_delegation(0.80, DelegationDecay::Linear { decay_days: 100 });
        let min_threshold = 0.05;
        // At day 0: effective = 0.80 → not expired
        assert!(!d.is_effectively_expired(Timestamp::from_micros(0), min_threshold));
        // At day 95: effective = 0.80 * 0.05 = 0.04 → expired (below 0.05)
        let t95 = Timestamp::from_micros(86_400_000_000 * 95);
        assert!(d.is_effectively_expired(t95, min_threshold));
    }

    // ========================================================================
    // QuadraticVote tests
    // ========================================================================

    #[test]
    fn test_quadratic_weight() {
        assert!((QuadraticVote::calculate_weight(0) - 0.0).abs() < f64::EPSILON);
        assert!((QuadraticVote::calculate_weight(1) - 1.0).abs() < f64::EPSILON);
        assert!((QuadraticVote::calculate_weight(4) - 2.0).abs() < f64::EPSILON);
        assert!((QuadraticVote::calculate_weight(9) - 3.0).abs() < f64::EPSILON);
        assert!((QuadraticVote::calculate_weight(100) - 10.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // ZkProposalType tests
    // ========================================================================

    #[test]
    fn test_zk_proposal_type_from_u8() {
        assert_eq!(ZkProposalType::from_u8(0), Some(ZkProposalType::Standard));
        assert_eq!(
            ZkProposalType::from_u8(1),
            Some(ZkProposalType::Constitutional)
        );
        assert_eq!(
            ZkProposalType::from_u8(2),
            Some(ZkProposalType::ModelGovernance)
        );
        assert_eq!(ZkProposalType::from_u8(3), Some(ZkProposalType::Emergency));
        assert_eq!(ZkProposalType::from_u8(4), Some(ZkProposalType::Treasury));
        assert_eq!(ZkProposalType::from_u8(5), Some(ZkProposalType::Membership));
        assert_eq!(ZkProposalType::from_u8(6), None);
        assert_eq!(ZkProposalType::from_u8(255), None);
    }

    #[test]
    fn test_zk_proposal_type_from_tier() {
        assert_eq!(
            ZkProposalType::from_tier(&ProposalTier::Basic),
            ZkProposalType::Standard
        );
        assert_eq!(
            ZkProposalType::from_tier(&ProposalTier::Major),
            ZkProposalType::Treasury
        );
        assert_eq!(
            ZkProposalType::from_tier(&ProposalTier::Constitutional),
            ZkProposalType::Constitutional
        );
    }

    // ========================================================================
    // EligibilityProof tests
    // ========================================================================

    fn make_proof(
        eligible: bool,
        proposal_type: ZkProposalType,
        expires_at: Option<Timestamp>,
    ) -> EligibilityProof {
        EligibilityProof {
            id: "proof-1".to_string(),
            voter_did: "did:test:voter".to_string(),
            voter_commitment: vec![0u8; 32],
            proposal_type,
            eligible,
            requirements_met: if eligible { 5 } else { 3 },
            active_requirements: 5,
            proof_bytes: vec![1, 2, 3],
            generated_at: Timestamp::from_micros(0),
            expires_at,
        }
    }

    #[test]
    fn test_eligibility_proof_not_expired_no_expiry() {
        let proof = make_proof(true, ZkProposalType::Standard, None);
        assert!(!proof.is_expired(Timestamp::from_micros(999_999_999_999)));
    }

    #[test]
    fn test_eligibility_proof_not_expired_future() {
        let proof = make_proof(
            true,
            ZkProposalType::Standard,
            Some(Timestamp::from_micros(1_000_000)),
        );
        assert!(!proof.is_expired(Timestamp::from_micros(500_000)));
    }

    #[test]
    fn test_eligibility_proof_expired() {
        let proof = make_proof(
            true,
            ZkProposalType::Standard,
            Some(Timestamp::from_micros(1_000_000)),
        );
        assert!(proof.is_expired(Timestamp::from_micros(1_000_001)));
    }

    #[test]
    fn test_is_valid_for_tier_constitutional_works_for_all() {
        let proof = make_proof(true, ZkProposalType::Constitutional, None);
        assert!(proof.is_valid_for_tier(&ProposalTier::Basic));
        assert!(proof.is_valid_for_tier(&ProposalTier::Major));
        assert!(proof.is_valid_for_tier(&ProposalTier::Constitutional));
    }

    #[test]
    fn test_is_valid_for_tier_standard_only_for_basic() {
        let proof = make_proof(true, ZkProposalType::Standard, None);
        assert!(proof.is_valid_for_tier(&ProposalTier::Basic));
        assert!(!proof.is_valid_for_tier(&ProposalTier::Major));
        assert!(!proof.is_valid_for_tier(&ProposalTier::Constitutional));
    }

    #[test]
    fn test_is_valid_for_tier_treasury_for_basic_and_major() {
        let proof = make_proof(true, ZkProposalType::Treasury, None);
        assert!(proof.is_valid_for_tier(&ProposalTier::Basic));
        assert!(proof.is_valid_for_tier(&ProposalTier::Major));
        assert!(!proof.is_valid_for_tier(&ProposalTier::Constitutional));
    }

    #[test]
    fn test_is_valid_for_tier_ineligible_proof_always_false() {
        let proof = make_proof(false, ZkProposalType::Constitutional, None);
        assert!(!proof.is_valid_for_tier(&ProposalTier::Basic));
        assert!(!proof.is_valid_for_tier(&ProposalTier::Major));
        assert!(!proof.is_valid_for_tier(&ProposalTier::Constitutional));
    }

    // ========================================================================
    // ProofAttestation tests
    // ========================================================================

    fn make_attestation(verified: bool, verified_at: i64, expires_at: i64) -> ProofAttestation {
        ProofAttestation {
            id: "att-1".to_string(),
            proof_hash: vec![0u8; 32],
            proof_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            voter_commitment: vec![0u8; 32],
            proposal_type: ZkProposalType::Standard,
            verified,
            verified_at: Timestamp::from_micros(verified_at),
            expires_at: Timestamp::from_micros(expires_at),
            verifier_pubkey: vec![0u8; 32],
            signature: vec![0u8; 64],
            security_level: "96-bit".to_string(),
            verification_time_ms: 100,
        }
    }

    #[test]
    fn test_attestation_not_expired() {
        let att = make_attestation(true, 0, 1_000_000);
        assert!(!att.is_expired(Timestamp::from_micros(500_000)));
    }

    #[test]
    fn test_attestation_expired() {
        let att = make_attestation(true, 0, 1_000_000);
        assert!(att.is_expired(Timestamp::from_micros(1_000_001)));
    }

    #[test]
    fn test_attestation_is_valid_verified_not_expired() {
        let att = make_attestation(true, 0, 1_000_000);
        assert!(att.is_valid(Timestamp::from_micros(500_000)));
    }

    #[test]
    fn test_attestation_is_invalid_when_unverified() {
        let att = make_attestation(false, 0, 1_000_000);
        assert!(!att.is_valid(Timestamp::from_micros(500_000)));
    }

    #[test]
    fn test_attestation_is_invalid_when_expired() {
        let att = make_attestation(true, 0, 1_000_000);
        assert!(!att.is_valid(Timestamp::from_micros(2_000_000)));
    }

    // ========================================================================
    // ProposalReflection tests
    // ========================================================================

    fn make_reflection(
        echo_risk: EchoChamberRiskLevel,
        rapid_convergence: bool,
        fragmentation: bool,
        harmony_coverage: f64,
        centralization: f64,
        needs_review: bool,
    ) -> ProposalReflection {
        ProposalReflection {
            id: "ref-1".to_string(),
            proposal_id: "MIP-001".to_string(),
            timestamp: Timestamp::from_micros(0),
            voter_count: 10,
            topology_pattern: TopologyPattern::Mesh,
            centralization,
            cluster_count: 3,
            absent_harmonies: vec![],
            harmony_coverage,
            average_epistemic_level: 0.7,
            echo_chamber_risk: echo_risk,
            agreement_verified: true,
            agreement_trend: TrendDirection::Stable,
            centralization_trend: TrendDirection::Stable,
            rapid_convergence_warning: rapid_convergence,
            fragmentation_warning: fragmentation,
            votes_for: 7,
            votes_against: 2,
            abstentions: 1,
            approval_ratio: 0.7,
            polarization: 0.3,
            suggested_interventions: vec![],
            reflection_prompts: vec![],
            needs_review,
            summary: "Test reflection".to_string(),
        }
    }

    #[test]
    fn test_has_concerns_none() {
        let r = make_reflection(EchoChamberRiskLevel::Low, false, false, 0.8, 0.3, false);
        assert!(!r.has_concerns());
    }

    #[test]
    fn test_has_concerns_needs_review() {
        let r = make_reflection(EchoChamberRiskLevel::Low, false, false, 0.8, 0.3, true);
        assert!(r.has_concerns());
    }

    #[test]
    fn test_has_concerns_high_echo_chamber() {
        let r = make_reflection(EchoChamberRiskLevel::High, false, false, 0.8, 0.3, false);
        assert!(r.has_concerns());
    }

    #[test]
    fn test_has_concerns_critical_echo_chamber() {
        let r = make_reflection(
            EchoChamberRiskLevel::Critical,
            false,
            false,
            0.8,
            0.3,
            false,
        );
        assert!(r.has_concerns());
    }

    #[test]
    fn test_has_concerns_rapid_convergence() {
        let r = make_reflection(EchoChamberRiskLevel::Low, true, false, 0.8, 0.3, false);
        assert!(r.has_concerns());
    }

    #[test]
    fn test_has_concerns_fragmentation() {
        let r = make_reflection(EchoChamberRiskLevel::Low, false, true, 0.8, 0.3, false);
        assert!(r.has_concerns());
    }

    #[test]
    fn test_has_concerns_low_harmony_coverage() {
        let r = make_reflection(EchoChamberRiskLevel::Low, false, false, 0.2, 0.3, false);
        assert!(r.has_concerns());
    }

    #[test]
    fn test_concern_severity_zero() {
        let r = make_reflection(EchoChamberRiskLevel::Low, false, false, 0.8, 0.3, false);
        assert_eq!(r.concern_severity(), 0);
    }

    #[test]
    fn test_concern_severity_moderate_echo() {
        let r = make_reflection(
            EchoChamberRiskLevel::Moderate,
            false,
            false,
            0.8,
            0.3,
            false,
        );
        assert_eq!(r.concern_severity(), 1);
    }

    #[test]
    fn test_concern_severity_max_all_flags() {
        // Critical(4) + rapid(2) + fragment(2) + low_harmony(2) + high_central(1) = 11 → capped at 10
        let r = make_reflection(EchoChamberRiskLevel::Critical, true, true, 0.1, 0.9, false);
        assert_eq!(r.concern_severity(), 10);
    }

    #[test]
    fn test_concern_severity_high_echo_plus_convergence() {
        // High(3) + rapid(2) = 5
        let r = make_reflection(EchoChamberRiskLevel::High, true, false, 0.8, 0.3, false);
        assert_eq!(r.concern_severity(), 5);
    }
}
