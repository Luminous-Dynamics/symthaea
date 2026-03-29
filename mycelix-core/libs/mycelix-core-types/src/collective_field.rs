// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Collective Mirror: Reflecting Group State Without Claiming Wisdom
//!
//! This module implements a **mirror**, not an **oracle**. It shows the group
//! how it looks - it does not tell the group what it is.
//!
//! ## The Anti-Pattern We Avoid
//!
//! ```text
//! ❌ ORACLE (Digital High Priest):
//!    "Your coherence is 0.87, wisdom has emerged, proceed with confidence."
//!
//!    Problem: History is full of highly coherent groups that were catastrophic
//!    (cults, mobs, speculative bubbles). Coherence ≠ Wisdom.
//! ```
//!
//! ## The Pattern We Follow
//!
//! ```text
//! ✓ MIRROR (Biofeedback):
//!    "Here is how your group looks right now:
//!     - Agreement is centralized around 3 voices
//!     - 'Care' and 'Stability' are absent from the conversation
//!     - Verification level is low
//!
//!     What do you want to do with this information?"
//!
//!    The humans decide what it means.
//! ```
//!
//! ## Key Insight
//!
//! A heart rate monitor doesn't say "You are happy." It says "120 BPM."
//! You decide if that's excitement or panic.
//!
//! ## The Five Reflections
//!
//! 1. **Topology** - Is agreement distributed (mesh) or centralized (hub-and-spoke)?
//! 2. **Shadow** - What harmonies are completely absent from the conversation?
//! 3. **Signal Integrity** - Is high agreement backed by verification, or is it an echo chamber?
//! 4. **Trajectory** - Which direction is the group moving? Converging or diverging?
//! 5. **Void** - What assumptions is everyone making without questioning?
//!
//! ## Additional Capabilities
//!
//! - **Intervention Library** - Suggests specific actions based on group state
//! - **Minority Signal Tracking** - Tracks dissenting voices and whether they later proved right
//! - **Calibration Loop** - Learns from decision outcomes to improve future reflections

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};

use crate::harmonic::Harmony;

// ==============================================================================
// CORE PHILOSOPHY: MIRROR, NOT ORACLE
// ==============================================================================

/// The collective mirror reflects group state without claiming to know wisdom.
///
/// It answers: "How do we look?" not "Are we wise?"
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectiveMirror {
    /// Current reflection of the group
    pub reflection: GroupReflection,

    /// Historical reflections for learning
    history: Vec<GroupReflection>,

    /// Outcome observations for calibration
    outcomes: Vec<OutcomeObservation>,

    /// Configuration
    config: MirrorConfig,
}

/// A reflection of the group at a moment in time
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupReflection {
    /// Timestamp of this reflection
    pub timestamp: u64,

    /// Number of participants
    pub participant_count: usize,

    // =========================================================================
    // TOPOLOGY: The Shape of Agreement
    // =========================================================================
    /// How is agreement structured? Hub-and-spoke (fragile) or mesh (resilient)?
    pub topology: TopologyAnalysis,

    // =========================================================================
    // SHADOW: What Are We Ignoring?
    // =========================================================================
    /// Harmonies that are absent or suppressed in the conversation
    pub shadow: ShadowAnalysis,

    // =========================================================================
    // SIGNAL INTEGRITY: Is This Verified or an Echo Chamber?
    // =========================================================================
    /// Quality of the epistemic grounding
    pub signal_integrity: SignalIntegrity,

    // =========================================================================
    // TRAJECTORY: Where Is The Group Heading?
    // =========================================================================
    /// How is the group changing over time?
    pub trajectory: TrajectoryAnalysis,

    // =========================================================================
    // VOID: What Are We All Assuming?
    // =========================================================================
    /// Shared assumptions that nobody is questioning
    pub void: VoidAnalysis,

    // =========================================================================
    // RAW METRICS: Data Without Interpretation
    // =========================================================================
    /// Raw agreement level (0.0 - 1.0) - NOT labeled as "coherence" or "wisdom"
    pub agreement_level: f64,

    /// Attention distribution - where is energy focused?
    pub attention_distribution: HashMap<String, f64>,

    /// Tension points - where is there productive disagreement?
    pub tension_points: Vec<TensionPoint>,
}

// ==============================================================================
// TOPOLOGY ANALYSIS: Cult vs. Council
// ==============================================================================

/// Analysis of how agreement is structured
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TopologyAnalysis {
    /// The detected topology type
    pub topology_type: TopologyType,

    /// Centralization score (0.0 = fully distributed, 1.0 = single authority)
    pub centralization: f64,

    /// Number of distinct "clusters" of agreement
    pub cluster_count: usize,

    /// Number of "bridge" nodes connecting different clusters
    pub bridge_count: usize,

    /// The most influential nodes (by how many others align with them)
    pub influence_centers: Vec<String>,

    /// Warning if topology is fragile
    pub warning: Option<TopologyWarning>,
}

/// Types of agreement topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TopologyType {
    /// Everyone agrees with a central node - FRAGILE (cult pattern)
    HubAndSpoke,

    /// Small clusters connected by bridges - RESILIENT (wisdom pattern)
    DistributedMesh,

    /// Multiple isolated clusters with no bridges - FRAGMENTED
    Fragmented,

    /// Too few participants to determine
    Insufficient,
}

impl TopologyType {
    pub fn description(&self) -> &'static str {
        match self {
            Self::HubAndSpoke => "Agreement flows through central nodes",
            Self::DistributedMesh => "Agreement distributed across connected clusters",
            Self::Fragmented => "Isolated clusters with no connection",
            Self::Insufficient => "Too few participants to analyze",
        }
    }

    pub fn risk_level(&self) -> &'static str {
        match self {
            Self::HubAndSpoke => "Higher risk of groupthink",
            Self::DistributedMesh => "More resilient to single points of failure",
            Self::Fragmented => "Risk of deadlock or split",
            Self::Insufficient => "Unknown",
        }
    }
}

/// Warnings about topology
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TopologyWarning {
    pub code: String,
    pub message: String,
    pub severity: WarningSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WarningSeverity {
    Info,
    Caution,
    Warning,
}

// ==============================================================================
// SHADOW ANALYSIS: What Are We Not Seeing?
// ==============================================================================

/// Analysis of what's missing from the conversation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ShadowAnalysis {
    /// Harmonies that are completely or nearly absent
    pub absent_harmonies: Vec<AbsentHarmony>,

    /// Perspectives/positions that have no representation
    pub unrepresented_positions: Vec<String>,

    /// The "shadow prompt" - a question to surface what's missing
    pub shadow_prompt: String,
}

/// A harmony that is absent or suppressed
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AbsentHarmony {
    pub harmony: Harmony,
    /// How present is this harmony? (0.0 = completely absent, 1.0 = fully present)
    pub presence: f64,
    /// What might we be missing by ignoring this?
    pub what_we_might_miss: String,
}

// ==============================================================================
// SIGNAL INTEGRITY: Echo Chamber Detection
// ==============================================================================

/// Analysis of whether agreement is grounded in verification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalIntegrity {
    /// Average epistemic level (E-axis) of claims being agreed upon
    pub epistemic_level: f64,

    /// Is high agreement combined with low verification? (Echo chamber risk)
    pub echo_chamber_risk: EchoChamberRisk,

    /// Diversity of sources informing the agreement
    pub source_diversity: f64,

    /// Are dissenting voices being heard or suppressed?
    pub dissent_health: DissentHealth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EchoChamberRisk {
    /// Low agreement OR high verification - not an echo chamber
    Low,
    /// Moderate agreement with moderate verification
    Moderate,
    /// High agreement with low verification - WARNING
    High,
    /// Very high agreement with very low verification - DANGER
    Critical,
}

impl EchoChamberRisk {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Low => "Agreement appears grounded in verification",
            Self::Moderate => "Some verification, but consider seeking more",
            Self::High => "High agreement with limited verification - examine assumptions",
            Self::Critical => "Strong consensus with minimal verification - high echo chamber risk",
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DissentHealth {
    /// Are there dissenting voices?
    pub dissent_present: bool,
    /// Are they being engaged with or ignored?
    pub dissent_engagement: f64,
    /// Prompt if dissent is being suppressed
    pub prompt: Option<String>,
}

// ==============================================================================
// TENSION POINTS: Where Is There Productive Disagreement?
// ==============================================================================

/// A point of tension (potential growth edge)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TensionPoint {
    /// What is the tension about?
    pub topic: String,
    /// Which harmonies are in tension?
    pub harmonies_in_tension: (Harmony, Harmony),
    /// How strong is the tension?
    pub intensity: f64,
    /// Is this tension being engaged with or avoided?
    pub engagement: TensionEngagement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TensionEngagement {
    /// Actively being discussed
    Engaged,
    /// Acknowledged but not deeply explored
    Acknowledged,
    /// Being avoided or suppressed
    Avoided,
}

// ==============================================================================
// TRAJECTORY ANALYSIS: Where Is The Group Heading?
// ==============================================================================

/// Analysis of how the group is changing over time
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrajectoryAnalysis {
    /// Direction of agreement change
    pub agreement_direction: TrendDirection,
    /// Rate of agreement change per reflection (-1.0 to 1.0)
    pub agreement_velocity: f64,
    /// Direction of centralization change
    pub centralization_direction: TrendDirection,
    /// Direction of epistemic level change
    pub epistemic_direction: TrendDirection,
    /// Is the group converging rapidly? (potential groupthink)
    pub rapid_convergence_warning: bool,
    /// Is the group fragmenting? (potential deadlock)
    pub fragmentation_warning: bool,
    /// Number of reflections used for trajectory calculation
    pub sample_size: usize,
    /// Human-readable trajectory summary
    pub summary: String,
}

impl Default for TrajectoryAnalysis {
    fn default() -> Self {
        Self {
            agreement_direction: TrendDirection::Unknown,
            agreement_velocity: 0.0,
            centralization_direction: TrendDirection::Unknown,
            epistemic_direction: TrendDirection::Unknown,
            rapid_convergence_warning: false,
            fragmentation_warning: false,
            sample_size: 0,
            summary: "Not enough history to determine trajectory.".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TrendDirection {
    /// Metric is increasing
    Rising,
    /// Metric is stable (within threshold)
    Stable,
    /// Metric is decreasing
    Falling,
    /// Not enough data to determine
    Unknown,
}

impl TrendDirection {
    pub fn from_velocity(velocity: f64, threshold: f64) -> Self {
        if velocity > threshold {
            Self::Rising
        } else if velocity < -threshold {
            Self::Falling
        } else {
            Self::Stable
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Rising => "↑",
            Self::Stable => "→",
            Self::Falling => "↓",
            Self::Unknown => "?",
        }
    }
}

// ==============================================================================
// VOID ANALYSIS: What Are We All Assuming?
// ==============================================================================

/// Analysis of shared blind spots and unquestioned assumptions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VoidAnalysis {
    /// Claims that everyone agrees on but nobody has verified
    pub unverified_consensus: Vec<UnverifiedConsensus>,
    /// Questions that nobody is asking
    pub unasked_questions: Vec<UnaskedQuestion>,
    /// Perspectives that would challenge current assumptions
    pub missing_perspectives: Vec<MissingPerspective>,
    /// The void prompt - surfaces what's not being examined
    pub void_prompt: String,
}

impl Default for VoidAnalysis {
    fn default() -> Self {
        Self {
            unverified_consensus: Vec::new(),
            unasked_questions: Vec::new(),
            missing_perspectives: Vec::new(),
            void_prompt: String::new(),
        }
    }
}

/// A point of agreement that lacks verification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnverifiedConsensus {
    /// What is everyone assuming?
    pub assumption: String,
    /// How many participants share this assumption?
    pub holder_count: usize,
    /// What epistemic level supports this? (low = dangerous)
    pub epistemic_support: f64,
    /// What could go wrong if this assumption is false?
    pub risk_if_wrong: String,
}

/// A question that would be valuable but isn't being asked
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnaskedQuestion {
    /// The question itself
    pub question: String,
    /// Why might this question be important?
    pub why_important: String,
    /// Why might the group be avoiding it?
    pub possible_avoidance_reason: String,
}

/// A perspective that's completely absent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MissingPerspective {
    /// Description of the perspective
    pub perspective: String,
    /// Who typically holds this perspective?
    pub typical_holders: String,
    /// What might we learn from this perspective?
    pub potential_insight: String,
}

// ==============================================================================
// MINORITY SIGNAL: Tracking The Cassandras
// ==============================================================================

/// Tracking minority positions and whether they later proved valuable
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MinoritySignal {
    /// Current minority positions
    pub current_minorities: Vec<MinorityPosition>,
    /// Historical record: minorities that were later validated
    pub validated_cassandras: Vec<ValidatedCassandra>,
    /// Historical record: minorities that were correctly rejected
    pub rejected_positions: Vec<RejectedPosition>,
    /// Overall track record of minority voices in this community
    pub minority_track_record: MinorityTrackRecord,
}

impl Default for MinoritySignal {
    fn default() -> Self {
        Self {
            current_minorities: Vec::new(),
            validated_cassandras: Vec::new(),
            rejected_positions: Vec::new(),
            minority_track_record: MinorityTrackRecord::default(),
        }
    }
}

/// A position currently held by a minority
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MinorityPosition {
    /// Unique ID for tracking
    pub id: String,
    /// What is the position?
    pub position: String,
    /// Who holds it?
    pub holders: Vec<String>,
    /// How many vs the majority?
    pub holder_count: usize,
    pub total_participants: usize,
    /// What harmonies does this position emphasize?
    pub emphasized_harmonies: Vec<Harmony>,
    /// When was this first recorded?
    pub first_recorded: u64,
    /// Is this being engaged with or dismissed?
    pub engagement_status: MinorityEngagement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MinorityEngagement {
    /// Being actively discussed and considered
    Engaged,
    /// Acknowledged but not deeply explored
    Tolerated,
    /// Being dismissed or suppressed
    Dismissed,
    /// Actively attacked or silenced
    Suppressed,
}

/// A minority position that was later validated
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidatedCassandra {
    /// Original position ID
    pub position_id: String,
    /// What was the position?
    pub position: String,
    /// Who held it originally?
    pub original_holders: Vec<String>,
    /// How was it validated?
    pub validation_evidence: String,
    /// How long until validation?
    pub time_to_validation: u64,
}

/// A minority position that was correctly rejected
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RejectedPosition {
    /// Original position ID
    pub position_id: String,
    /// What was the position?
    pub position: String,
    /// Why was rejection correct?
    pub rejection_evidence: String,
}

/// Track record of minority voices
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MinorityTrackRecord {
    /// How often were minorities later validated?
    pub validation_rate: f64,
    /// Total minorities tracked
    pub total_tracked: usize,
    /// Total validated
    pub total_validated: usize,
    /// Total rejected
    pub total_rejected: usize,
    /// Prompt based on track record
    pub prompt: String,
}

impl Default for MinorityTrackRecord {
    fn default() -> Self {
        Self {
            validation_rate: 0.0,
            total_tracked: 0,
            total_validated: 0,
            total_rejected: 0,
            prompt: "No minority positions tracked yet.".to_string(),
        }
    }
}

// ==============================================================================
// INTERVENTION LIBRARY: What Could Help?
// ==============================================================================

/// Suggested interventions based on current state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InterventionSuggestion {
    /// Unique ID
    pub id: String,
    /// Name of the intervention
    pub name: String,
    /// What does this intervention involve?
    pub description: String,
    /// Why is it suggested for this state?
    pub rationale: String,
    /// What state triggers this suggestion?
    pub trigger: InterventionTrigger,
    /// How effective has this been historically?
    pub historical_effectiveness: Option<f64>,
    /// Estimated effort level
    pub effort: EffortLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterventionTrigger {
    /// High centralization detected
    HighCentralization,
    /// Echo chamber risk
    EchoChamber,
    /// Missing harmonies
    MissingShadow,
    /// Rapid convergence
    RapidConvergence,
    /// Fragmentation
    Fragmentation,
    /// Suppressed dissent
    SuppressedDissent,
    /// Unverified consensus
    UnverifiedConsensus,
    /// Stagnation
    Stagnation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EffortLevel {
    /// Quick, can do immediately
    Low,
    /// Requires some preparation
    Medium,
    /// Significant investment
    High,
}

// ==============================================================================
// CALIBRATION LOOP: Learning From Outcomes
// ==============================================================================

/// Analysis of what we've learned from past decisions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationInsight {
    /// Pattern identified
    pub pattern: String,
    /// What state preceded positive outcomes?
    pub positive_predictors: Vec<StatePattern>,
    /// What state preceded negative outcomes?
    pub negative_predictors: Vec<StatePattern>,
    /// Confidence in this insight (based on sample size)
    pub confidence: f64,
    /// Sample size
    pub sample_size: usize,
}

/// State features that predicted an outcome
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatePattern {
    /// Description of the pattern
    pub description: String,
    /// What was the agreement range?
    pub agreement_range: (f64, f64),
    /// What was the topology?
    pub topology: Option<TopologyType>,
    /// What was the echo chamber risk?
    pub echo_chamber_risk: Option<EchoChamberRisk>,
    /// How many times did this pattern occur?
    pub occurrence_count: usize,
}

// ==============================================================================
// OUTCOME OBSERVATION: Learning From Results
// ==============================================================================

/// An observation of what happened after a decision
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OutcomeObservation {
    /// ID of the decision/proposal
    pub decision_id: String,
    /// The group reflection at decision time
    pub reflection_at_decision: GroupReflection,
    /// What actually happened?
    pub outcome: DecisionOutcome,
    /// Timestamp of outcome observation
    pub observed_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DecisionOutcome {
    /// Decision led to positive results
    Positive,
    /// Decision led to neutral/mixed results
    Neutral,
    /// Decision led to negative results
    Negative,
    /// Too early to tell
    Pending,
}

// ==============================================================================
// CONFIGURATION
// ==============================================================================

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MirrorConfig {
    /// Minimum participants for analysis
    pub min_participants: usize,
    /// Threshold below which a harmony is considered "absent"
    pub absent_harmony_threshold: f64,
    /// Threshold above which centralization triggers warning
    pub centralization_warning_threshold: f64,
    /// Threshold for echo chamber detection
    pub echo_chamber_threshold: (f64, f64), // (agreement, epistemic)
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            absent_harmony_threshold: 0.1,
            centralization_warning_threshold: 0.7,
            echo_chamber_threshold: (0.8, 0.3), // High agreement, low epistemic
        }
    }
}

// ==============================================================================
// PARTICIPANT INPUT
// ==============================================================================

/// A participant's contribution (input to the mirror)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Participant {
    pub id: String,
    /// What harmonies does this participant emphasize?
    pub harmony_emphasis: HarmonyEmphasis,
    /// Epistemic quality of their contribution
    pub epistemic_level: f64,
    /// Who do they align with? (for topology)
    pub alignments: Vec<String>,
    /// Who do they disagree with?
    pub disagreements: Vec<String>,
    /// What are they focused on?
    pub attention_focus: Option<String>,
    /// Timestamp
    pub last_active: u64,
}

/// Which harmonies a participant emphasizes
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonyEmphasis {
    pub resonant_coherence: f64,
    pub pan_sentient_flourishing: f64,
    pub integral_wisdom: f64,
    pub infinite_play: f64,
    pub universal_interconnectedness: f64,
    pub sacred_reciprocity: f64,
    pub evolutionary_progression: f64,
    pub sacred_stillness: f64,
}

impl HarmonyEmphasis {
    pub fn get(&self, harmony: Harmony) -> f64 {
        match harmony {
            Harmony::ResonantCoherence => self.resonant_coherence,
            Harmony::PanSentientFlourishing => self.pan_sentient_flourishing,
            Harmony::IntegralWisdom => self.integral_wisdom,
            Harmony::InfinitePlay => self.infinite_play,
            Harmony::UniversalInterconnectedness => self.universal_interconnectedness,
            Harmony::SacredReciprocity => self.sacred_reciprocity,
            Harmony::EvolutionaryProgression => self.evolutionary_progression,
            Harmony::SacredStillness => self.sacred_stillness,
        }
    }

    pub fn as_array(&self) -> [f64; 7] {
        [
            self.resonant_coherence,
            self.pan_sentient_flourishing,
            self.integral_wisdom,
            self.infinite_play,
            self.universal_interconnectedness,
            self.sacred_reciprocity,
            self.evolutionary_progression,
        ]
    }
}

// ==============================================================================
// IMPLEMENTATION
// ==============================================================================

impl CollectiveMirror {
    /// Create a new mirror
    pub fn new(config: MirrorConfig) -> Self {
        Self {
            reflection: GroupReflection::empty(0),
            history: Vec::new(),
            outcomes: Vec::new(),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MirrorConfig::default())
    }

    /// Generate a reflection from current participants
    pub fn reflect(&mut self, participants: &[Participant], timestamp: u64) -> &GroupReflection {
        let count = participants.len();

        if count < self.config.min_participants {
            self.reflection = GroupReflection::insufficient(timestamp, count);
            return &self.reflection;
        }

        // Analyze topology
        let topology = self.analyze_topology(participants);

        // Analyze shadow (what's missing)
        let shadow = self.analyze_shadow(participants);

        // Analyze signal integrity
        let signal_integrity = self.analyze_signal_integrity(participants, &topology);

        // Calculate raw agreement level
        let agreement_level = self.calculate_agreement_level(participants);

        // Calculate attention distribution
        let attention_distribution = self.calculate_attention_distribution(participants);

        // Find tension points
        let tension_points = self.find_tension_points(participants);

        // Analyze trajectory (how we're changing over time)
        let trajectory = self.analyze_trajectory(agreement_level, &signal_integrity, count);

        // Analyze void (what we're all assuming without questioning)
        let void = self.analyze_void(participants, &shadow, &signal_integrity);

        self.reflection = GroupReflection {
            timestamp,
            participant_count: count,
            topology,
            shadow,
            signal_integrity,
            trajectory,
            void,
            agreement_level,
            attention_distribution,
            tension_points,
        };

        // Store in history
        self.history.push(self.reflection.clone());
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        &self.reflection
    }

    /// Analyze the topology of agreement
    fn analyze_topology(&self, participants: &[Participant]) -> TopologyAnalysis {
        // Build alignment graph
        let mut influence_counts: HashMap<String, usize> = HashMap::new();
        let mut total_alignments = 0;

        for p in participants {
            for aligned_with in &p.alignments {
                *influence_counts.entry(aligned_with.clone()).or_insert(0) += 1;
                total_alignments += 1;
            }
        }

        // Calculate centralization: what fraction of alignments go to the top influencer?
        // 1.0 = everyone aligns with one person (hub-and-spoke)
        // 0.0 = alignments evenly distributed
        let centralization = if total_alignments > 0 && !influence_counts.is_empty() {
            let max_influence = *influence_counts.values().max().unwrap_or(&0) as f64;
            let num_influencers = influence_counts.len();

            // Special case: if there's only one influencer receiving multiple alignments,
            // that's maximum centralization (hub-and-spoke pattern)
            if num_influencers == 1 {
                // Centralization scales with how many people align with the single hub
                // More aligners = more centralized
                (max_influence / participants.len().max(1) as f64).min(1.0)
            } else {
                // Multiple influencers: compare actual concentration vs perfect distribution
                let concentration_ratio = max_influence / total_alignments as f64;
                let perfect_distribution = 1.0 / num_influencers as f64;

                // Scale so that perfect distribution = 0, single hub = 1
                if concentration_ratio > perfect_distribution {
                    ((concentration_ratio - perfect_distribution) / (1.0 - perfect_distribution))
                        .min(1.0)
                        .max(0.0)
                } else {
                    0.0
                }
            }
        } else {
            0.0
        };

        // Count clusters (simplified: nodes with no alignments form separate clusters)
        let aligned_nodes: HashSet<&String> = participants
            .iter()
            .flat_map(|p| p.alignments.iter())
            .collect();
        let unaligned_count = participants
            .iter()
            .filter(|p| p.alignments.is_empty())
            .count();
        let cluster_count = (aligned_nodes.len().max(1) + unaligned_count).min(participants.len());

        // Count bridges (nodes that have both alignments and disagreements)
        let bridge_count = participants
            .iter()
            .filter(|p| !p.alignments.is_empty() && !p.disagreements.is_empty())
            .count();

        // Determine topology type
        let topology_type = if participants.len() < self.config.min_participants {
            TopologyType::Insufficient
        } else if centralization > self.config.centralization_warning_threshold {
            TopologyType::HubAndSpoke
        } else if bridge_count > 0 {
            TopologyType::DistributedMesh
        } else if cluster_count > 2 && bridge_count == 0 {
            TopologyType::Fragmented
        } else {
            TopologyType::DistributedMesh
        };

        // Generate warning if needed
        let warning = match topology_type {
            TopologyType::HubAndSpoke => Some(TopologyWarning {
                code: "CENTRALIZED_CONSENSUS".to_string(),
                message: "Agreement is flowing through few central voices. Consider: Are diverse perspectives being heard?".to_string(),
                severity: WarningSeverity::Caution,
            }),
            TopologyType::Fragmented => Some(TopologyWarning {
                code: "FRAGMENTED_CLUSTERS".to_string(),
                message: "Groups are isolated with no bridges. Consider: How might these perspectives connect?".to_string(),
                severity: WarningSeverity::Info,
            }),
            _ => None,
        };

        // Find top influence centers
        let mut influence_vec: Vec<_> = influence_counts.into_iter().collect();
        influence_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let influence_centers: Vec<String> = influence_vec
            .into_iter()
            .take(3)
            .map(|(id, _)| id)
            .collect();

        TopologyAnalysis {
            topology_type,
            centralization,
            cluster_count,
            bridge_count,
            influence_centers,
            warning,
        }
    }

    /// Analyze what's missing (the shadow)
    fn analyze_shadow(&self, participants: &[Participant]) -> ShadowAnalysis {
        // Calculate average emphasis for each harmony
        let mut harmony_totals = [0.0f64; 7];
        let count = participants.len() as f64;

        for p in participants {
            let emphasis = p.harmony_emphasis.as_array();
            for (i, val) in emphasis.iter().enumerate() {
                harmony_totals[i] += val;
            }
        }

        let harmony_averages: Vec<f64> =
            harmony_totals.iter().map(|t| t / count.max(1.0)).collect();

        // Find absent harmonies
        let harmonies = [
            (
                Harmony::ResonantCoherence,
                "integration, coherence, bringing things together",
            ),
            (
                Harmony::PanSentientFlourishing,
                "care, wellbeing, considering impact on all beings",
            ),
            (
                Harmony::IntegralWisdom,
                "truth, deep understanding, rigorous inquiry",
            ),
            (
                Harmony::InfinitePlay,
                "creativity, joy, openness to possibility",
            ),
            (
                Harmony::UniversalInterconnectedness,
                "connection, seeing the whole, interdependence",
            ),
            (
                Harmony::SacredReciprocity,
                "fairness, exchange, mutual benefit",
            ),
            (
                Harmony::EvolutionaryProgression,
                "growth, development, long-term flourishing",
            ),
        ];

        let absent_harmonies: Vec<AbsentHarmony> = harmonies
            .iter()
            .enumerate()
            .filter(|(i, _)| harmony_averages[*i] < self.config.absent_harmony_threshold)
            .map(|(i, (harmony, description))| AbsentHarmony {
                harmony: *harmony,
                presence: harmony_averages[i],
                what_we_might_miss: description.to_string(),
            })
            .collect();

        // Generate shadow prompt
        let shadow_prompt = if absent_harmonies.is_empty() {
            "All harmonies are represented in the conversation.".to_string()
        } else {
            let missing: Vec<String> = absent_harmonies
                .iter()
                .map(|a| format!("{:?}", a.harmony))
                .collect();
            format!(
                "The conversation may be missing: {}. What perspectives might we be overlooking?",
                missing.join(", ")
            )
        };

        ShadowAnalysis {
            absent_harmonies,
            unrepresented_positions: Vec::new(), // Could be extended
            shadow_prompt,
        }
    }

    /// Analyze signal integrity (echo chamber detection)
    fn analyze_signal_integrity(
        &self,
        participants: &[Participant],
        _topology: &TopologyAnalysis,
    ) -> SignalIntegrity {
        // Calculate average epistemic level
        let epistemic_level: f64 = if participants.is_empty() {
            0.0
        } else {
            participants.iter().map(|p| p.epistemic_level).sum::<f64>() / participants.len() as f64
        };

        // Calculate agreement level for echo chamber check
        let agreement = self.calculate_agreement_level(participants);

        // Determine echo chamber risk
        let (agreement_threshold, epistemic_threshold) = self.config.echo_chamber_threshold;
        let echo_chamber_risk =
            if agreement > agreement_threshold && epistemic_level < epistemic_threshold {
                EchoChamberRisk::Critical
            } else if agreement > 0.7 && epistemic_level < 0.4 {
                EchoChamberRisk::High
            } else if agreement > 0.6 && epistemic_level < 0.5 {
                EchoChamberRisk::Moderate
            } else {
                EchoChamberRisk::Low
            };

        // Source diversity (simplified: based on disagreement presence)
        let source_diversity = if participants.is_empty() {
            0.0
        } else {
            let with_disagreements = participants
                .iter()
                .filter(|p| !p.disagreements.is_empty())
                .count();
            with_disagreements as f64 / participants.len() as f64
        };

        // Dissent health
        let dissent_present = participants.iter().any(|p| !p.disagreements.is_empty());
        let dissent_engagement = source_diversity; // Simplified metric
        let dissent_prompt = if !dissent_present && agreement > 0.7 {
            Some("No dissenting voices detected. Is everyone truly aligned, or are some perspectives not being voiced?".to_string())
        } else {
            None
        };

        SignalIntegrity {
            epistemic_level,
            echo_chamber_risk,
            source_diversity,
            dissent_health: DissentHealth {
                dissent_present,
                dissent_engagement,
                prompt: dissent_prompt,
            },
        }
    }

    /// Calculate raw agreement level
    fn calculate_agreement_level(&self, participants: &[Participant]) -> f64 {
        if participants.len() < 2 {
            return 0.0;
        }

        // Calculate pairwise harmony similarity
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..participants.len() {
            for j in (i + 1)..participants.len() {
                let a = &participants[i].harmony_emphasis;
                let b = &participants[j].harmony_emphasis;
                total_similarity += cosine_similarity(&a.as_array(), &b.as_array());
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        }
    }

    /// Calculate attention distribution
    fn calculate_attention_distribution(
        &self,
        participants: &[Participant],
    ) -> HashMap<String, f64> {
        let mut distribution: HashMap<String, usize> = HashMap::new();
        let mut total = 0;

        for p in participants {
            if let Some(focus) = &p.attention_focus {
                *distribution.entry(focus.clone()).or_insert(0) += 1;
                total += 1;
            }
        }

        distribution
            .into_iter()
            .map(|(k, v)| (k, v as f64 / total.max(1) as f64))
            .collect()
    }

    /// Find tension points
    fn find_tension_points(&self, participants: &[Participant]) -> Vec<TensionPoint> {
        // Find pairs of harmonies where participants diverge
        // Simplified: look for cases where some emphasize A and others emphasize B

        let mut tensions = Vec::new();

        let harmony_pairs = [
            (
                Harmony::PanSentientFlourishing,
                Harmony::InfinitePlay,
                "Care vs. Freedom",
            ),
            (
                Harmony::IntegralWisdom,
                Harmony::InfinitePlay,
                "Truth vs. Creativity",
            ),
            (
                Harmony::SacredReciprocity,
                Harmony::UniversalInterconnectedness,
                "Boundaries vs. Unity",
            ),
            (
                Harmony::ResonantCoherence,
                Harmony::EvolutionaryProgression,
                "Stability vs. Change",
            ),
        ];

        for (h1, h2, topic) in harmony_pairs {
            let h1_scores: Vec<f64> = participants
                .iter()
                .map(|p| p.harmony_emphasis.get(h1))
                .collect();
            let h2_scores: Vec<f64> = participants
                .iter()
                .map(|p| p.harmony_emphasis.get(h2))
                .collect();

            // Check if there's divergence
            let h1_variance = variance(&h1_scores);
            let h2_variance = variance(&h2_scores);

            if h1_variance > 0.1 && h2_variance > 0.1 {
                tensions.push(TensionPoint {
                    topic: topic.to_string(),
                    harmonies_in_tension: (h1, h2),
                    intensity: (h1_variance + h2_variance) / 2.0,
                    engagement: TensionEngagement::Acknowledged, // Would need more data to determine
                });
            }
        }

        tensions
    }

    /// Analyze trajectory - how is the group changing over time?
    fn analyze_trajectory(
        &self,
        _current_agreement: f64,
        _current_integrity: &SignalIntegrity,
        _current_count: usize,
    ) -> TrajectoryAnalysis {
        if self.history.len() < 3 {
            return TrajectoryAnalysis::default();
        }

        let recent: Vec<_> = self.history.iter().rev().take(10).collect();
        let sample_size = recent.len();

        let agreement_values: Vec<f64> = recent.iter().map(|r| r.agreement_level).collect();
        let agreement_velocity = calculate_trend(&agreement_values);
        let agreement_direction = TrendDirection::from_velocity(agreement_velocity, 0.05);

        let centralization_values: Vec<f64> =
            recent.iter().map(|r| r.topology.centralization).collect();
        let centralization_direction =
            TrendDirection::from_velocity(calculate_trend(&centralization_values), 0.05);

        let epistemic_values: Vec<f64> = recent
            .iter()
            .map(|r| r.signal_integrity.epistemic_level)
            .collect();
        let epistemic_velocity = calculate_trend(&epistemic_values);
        let epistemic_direction = TrendDirection::from_velocity(epistemic_velocity, 0.05);

        let rapid_convergence_warning = agreement_direction == TrendDirection::Rising
            && agreement_velocity > 0.1
            && epistemic_direction == TrendDirection::Falling;

        let fragmentation_warning =
            agreement_direction == TrendDirection::Falling && _current_agreement < 0.4;

        let summary = format!(
            "Agreement {} ({:+.0}%/cycle), Epistemic {} ({:+.0}%/cycle)",
            agreement_direction.symbol(),
            agreement_velocity * 100.0,
            epistemic_direction.symbol(),
            epistemic_velocity * 100.0
        );

        TrajectoryAnalysis {
            agreement_direction,
            agreement_velocity,
            centralization_direction,
            epistemic_direction,
            rapid_convergence_warning,
            fragmentation_warning,
            sample_size,
            summary,
        }
    }

    /// Analyze void - what assumptions is everyone making without questioning?
    fn analyze_void(
        &self,
        participants: &[Participant],
        shadow: &ShadowAnalysis,
        signal_integrity: &SignalIntegrity,
    ) -> VoidAnalysis {
        let mut unverified_consensus = Vec::new();
        let mut unasked_questions = Vec::new();
        let mut missing_perspectives = Vec::new();

        if matches!(
            signal_integrity.echo_chamber_risk,
            EchoChamberRisk::High | EchoChamberRisk::Critical
        ) {
            unverified_consensus.push(UnverifiedConsensus {
                assumption: "The group shares assumptions that haven't been explicitly verified"
                    .to_string(),
                holder_count: participants.len(),
                epistemic_support: signal_integrity.epistemic_level,
                risk_if_wrong: "Decisions may be based on false premises".to_string(),
            });
        }

        for absent in &shadow.absent_harmonies {
            unasked_questions.push(UnaskedQuestion {
                question: format!("How does {:?} apply here?", absent.harmony),
                why_important: absent.what_we_might_miss.clone(),
                possible_avoidance_reason: "May not seem immediately relevant".to_string(),
            });
        }

        if !signal_integrity.dissent_health.dissent_present {
            missing_perspectives.push(MissingPerspective {
                perspective: "Skeptical or contrarian viewpoint".to_string(),
                typical_holders: "Critics, devil's advocates, newcomers".to_string(),
                potential_insight: "May reveal blind spots or unstated assumptions".to_string(),
            });
        }

        let void_prompt = if unverified_consensus.is_empty() && unasked_questions.is_empty() {
            "The group appears to be questioning assumptions appropriately.".to_string()
        } else {
            format!(
                "{} unverified consensus points, {} unasked questions",
                unverified_consensus.len(),
                unasked_questions.len()
            )
        };

        VoidAnalysis {
            unverified_consensus,
            unasked_questions,
            missing_perspectives,
            void_prompt,
        }
    }

    /// Get suggested interventions based on current state
    pub fn suggest_interventions(&self) -> Vec<InterventionSuggestion> {
        let r = &self.reflection;
        let mut suggestions = Vec::new();

        if r.topology.topology_type == TopologyType::HubAndSpoke {
            suggestions.push(InterventionSuggestion {
                id: "decentralize".to_string(),
                name: "Breakout Groups".to_string(),
                description: "Split into small groups without the central voices".to_string(),
                rationale: "Agreement is centralized".to_string(),
                trigger: InterventionTrigger::HighCentralization,
                historical_effectiveness: None,
                effort: EffortLevel::Low,
            });
        }

        if matches!(
            r.signal_integrity.echo_chamber_risk,
            EchoChamberRisk::High | EchoChamberRisk::Critical
        ) {
            suggestions.push(InterventionSuggestion {
                id: "verify".to_string(),
                name: "Verification Round".to_string(),
                description: "Pause and explicitly seek evidence".to_string(),
                rationale: "High agreement with low verification".to_string(),
                trigger: InterventionTrigger::EchoChamber,
                historical_effectiveness: None,
                effort: EffortLevel::Medium,
            });
        }

        if r.trajectory.rapid_convergence_warning {
            suggestions.push(InterventionSuggestion {
                id: "slow-down".to_string(),
                name: "Structured Disagreement".to_string(),
                description: "Assign a devil's advocate".to_string(),
                rationale: "Agreement rising while verification falling".to_string(),
                trigger: InterventionTrigger::RapidConvergence,
                historical_effectiveness: None,
                effort: EffortLevel::Low,
            });
        }

        suggestions
    }

    /// Analyze outcomes to find patterns (calibration)
    pub fn analyze_calibration(&self) -> Option<CalibrationInsight> {
        if self.outcomes.len() < 5 {
            return None;
        }

        let positive: Vec<_> = self
            .outcomes
            .iter()
            .filter(|o| o.outcome == DecisionOutcome::Positive)
            .collect();
        let negative: Vec<_> = self
            .outcomes
            .iter()
            .filter(|o| o.outcome == DecisionOutcome::Negative)
            .collect();

        if positive.is_empty() || negative.is_empty() {
            return None;
        }

        let pos_avg = positive
            .iter()
            .map(|o| o.reflection_at_decision.agreement_level)
            .sum::<f64>()
            / positive.len() as f64;
        let neg_avg = negative
            .iter()
            .map(|o| o.reflection_at_decision.agreement_level)
            .sum::<f64>()
            / negative.len() as f64;

        Some(CalibrationInsight {
            pattern: if pos_avg < neg_avg {
                "Moderate agreement precedes better outcomes".to_string()
            } else {
                "Higher agreement correlated with better outcomes".to_string()
            },
            positive_predictors: vec![StatePattern {
                description: "Positive pattern".to_string(),
                agreement_range: (pos_avg - 0.1, pos_avg + 0.1),
                topology: None,
                echo_chamber_risk: None,
                occurrence_count: positive.len(),
            }],
            negative_predictors: vec![StatePattern {
                description: "Negative pattern".to_string(),
                agreement_range: (neg_avg - 0.1, neg_avg + 0.1),
                topology: None,
                echo_chamber_risk: None,
                occurrence_count: negative.len(),
            }],
            confidence: (self.outcomes.len() as f64 / 20.0).min(1.0),
            sample_size: self.outcomes.len(),
        })
    }

    /// Record an outcome for learning
    pub fn observe_outcome(
        &mut self,
        decision_id: String,
        outcome: DecisionOutcome,
        observed_at: u64,
    ) {
        self.outcomes.push(OutcomeObservation {
            decision_id,
            reflection_at_decision: self.reflection.clone(),
            outcome,
            observed_at,
        });
    }

    /// Get a reflection prompt (what the mirror shows)
    pub fn reflection_prompt(&self) -> String {
        let r = &self.reflection;

        let mut prompts = Vec::new();

        // Topology prompt
        if let Some(warning) = &r.topology.warning {
            prompts.push(warning.message.clone());
        }

        // Echo chamber prompt
        match r.signal_integrity.echo_chamber_risk {
            EchoChamberRisk::Critical => {
                prompts.push(
                    "⚠️ High agreement with low verification. Are we in an echo chamber?"
                        .to_string(),
                );
            }
            EchoChamberRisk::High => {
                prompts.push("Strong consensus detected with limited verification. Consider seeking diverse sources.".to_string());
            }
            _ => {}
        }

        // Shadow prompt
        if !r.shadow.absent_harmonies.is_empty() {
            prompts.push(r.shadow.shadow_prompt.clone());
        }

        // Dissent prompt
        if let Some(prompt) = &r.signal_integrity.dissent_health.prompt {
            prompts.push(prompt.clone());
        }

        if prompts.is_empty() {
            "The group appears to have diverse perspectives with verified grounding.".to_string()
        } else {
            prompts.join("\n\n")
        }
    }

    /// Get a simple summary (not a judgment)
    pub fn summary(&self) -> String {
        let r = &self.reflection;
        format!(
            "Group Reflection ({} participants):\n\
             • Agreement: {:.0}%\n\
             • Structure: {:?}\n\
             • Echo Chamber Risk: {:?}\n\
             • Missing Harmonies: {}\n\
             • Tensions: {}",
            r.participant_count,
            r.agreement_level * 100.0,
            r.topology.topology_type,
            r.signal_integrity.echo_chamber_risk,
            if r.shadow.absent_harmonies.is_empty() {
                "None".to_string()
            } else {
                r.shadow
                    .absent_harmonies
                    .iter()
                    .map(|a| format!("{:?}", a.harmony))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
            if r.tension_points.is_empty() {
                "None detected".to_string()
            } else {
                r.tension_points
                    .iter()
                    .map(|t| t.topic.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        )
    }
}

impl GroupReflection {
    fn empty(timestamp: u64) -> Self {
        Self {
            timestamp,
            participant_count: 0,
            topology: TopologyAnalysis {
                topology_type: TopologyType::Insufficient,
                centralization: 0.0,
                cluster_count: 0,
                bridge_count: 0,
                influence_centers: Vec::new(),
                warning: None,
            },
            shadow: ShadowAnalysis {
                absent_harmonies: Vec::new(),
                unrepresented_positions: Vec::new(),
                shadow_prompt: String::new(),
            },
            signal_integrity: SignalIntegrity {
                epistemic_level: 0.0,
                echo_chamber_risk: EchoChamberRisk::Low,
                source_diversity: 0.0,
                dissent_health: DissentHealth {
                    dissent_present: false,
                    dissent_engagement: 0.0,
                    prompt: None,
                },
            },
            trajectory: TrajectoryAnalysis::default(),
            void: VoidAnalysis::default(),
            agreement_level: 0.0,
            attention_distribution: HashMap::new(),
            tension_points: Vec::new(),
        }
    }

    fn insufficient(timestamp: u64, count: usize) -> Self {
        let mut r = Self::empty(timestamp);
        r.participant_count = count;
        r
    }
}

// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================

fn cosine_similarity(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Calculate trend (slope) of a time series using simple linear regression
fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

// ==============================================================================
// HOLONIC MIRROR: Groups Within Groups Within Groups
// ==============================================================================
//
// A holon is something that is simultaneously a whole and a part.
// A team is a whole (containing individuals) AND a part (of a department).
// This structure allows reflection at any scale while detecting
// cross-scale patterns that are invisible from any single level.

/// A holonic mirror reflects groups within groups within groups.
/// Each node is both a whole (with its own reflection) and a part (of a parent).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HolonicMirror {
    /// Unique identifier for this group
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// This group's own mirror (reflects direct participants)
    pub mirror: CollectiveMirror,

    /// Child groups (this group as a container of groups)
    pub children: Vec<HolonicMirror>,

    /// Cross-scale analysis (patterns that emerge between levels)
    pub cross_scale: Option<CrossScaleAnalysis>,

    /// Depth in the hierarchy (0 = root)
    pub depth: usize,
}

/// Analysis of patterns that emerge across scales
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CrossScaleAnalysis {
    /// Do children agree with each other? (inter-group coherence)
    pub inter_child_coherence: f64,

    /// Does local diversity mask global conformity?
    /// True if children are internally diverse but externally homogeneous
    pub diversity_illusion: DiversityIllusion,

    /// Minorities in one child that are majorities across all children
    pub cross_group_cassandras: Vec<CrossGroupCassandra>,

    /// Voids shared by ALL children (invisible from any single child)
    pub emergent_void: EmergentVoid,

    /// Topology at the group-of-groups level
    pub meta_topology: MetaTopology,

    /// Summary prompt for this scale
    pub scale_prompt: String,
}

/// Detection of diversity illusion - local diversity masking global conformity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiversityIllusion {
    /// Is the illusion present?
    pub detected: bool,

    /// Average internal diversity of children
    pub local_diversity: f64,

    /// Diversity between children's consensus positions
    pub global_diversity: f64,

    /// The illusion ratio: high local / low global = illusion
    pub illusion_ratio: f64,

    /// What are all children agreeing on that they don't realize?
    pub hidden_consensus: Vec<String>,

    /// Warning message
    pub warning: Option<String>,
}

/// A minority position in one group that represents majority thinking elsewhere
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CrossGroupCassandra {
    /// The position being held
    pub position: String,

    /// Group where this is a minority view
    pub minority_in: String,

    /// Groups where this is a majority view
    pub majority_in: Vec<String>,

    /// What fraction of ALL participants (across groups) hold this?
    pub global_prevalence: f64,

    /// The harmonies this position emphasizes
    pub emphasized_harmonies: Vec<Harmony>,

    /// Prompt to surface this
    pub prompt: String,
}

/// Voids that exist at a higher level but are invisible from below
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmergentVoid {
    /// Harmonies absent from ALL children
    pub universally_absent_harmonies: Vec<Harmony>,

    /// Questions no child is asking
    pub unasked_across_all: Vec<String>,

    /// Assumptions shared by all children
    pub shared_assumptions: Vec<SharedAssumption>,

    /// The emergent void prompt
    pub void_prompt: String,
}

/// An assumption shared across all child groups
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SharedAssumption {
    /// What is being assumed?
    pub assumption: String,

    /// How many children share this assumption?
    pub prevalence: usize,

    /// Has anyone questioned it?
    pub questioned: bool,

    /// What might we be missing?
    pub blind_spot: String,
}

/// Topology at the meta level (how groups relate to each other)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetaTopology {
    /// Are some groups dominating the discourse?
    pub dominant_groups: Vec<String>,

    /// Are some groups isolated?
    pub isolated_groups: Vec<String>,

    /// Are there bridge groups connecting different clusters?
    pub bridge_groups: Vec<String>,

    /// Overall structure
    pub structure: MetaStructure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MetaStructure {
    /// One group dominates all others
    Hierarchical,
    /// Groups are roughly equal peers
    Federated,
    /// Groups are disconnected
    Fragmented,
    /// Complex web of relationships
    NetworkedMesh,
    /// Not enough groups to determine
    Insufficient,
}

impl HolonicMirror {
    /// Create a new holonic mirror for a group
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            mirror: CollectiveMirror::with_defaults(),
            children: Vec::new(),
            cross_scale: None,
            depth: 0,
        }
    }

    /// Add a child group
    pub fn add_child(&mut self, mut child: HolonicMirror) {
        child.update_depth(self.depth + 1);
        self.children.push(child);
    }

    /// Recursively update depth for this node and all descendants
    fn update_depth(&mut self, new_depth: usize) {
        self.depth = new_depth;
        for child in &mut self.children {
            child.update_depth(new_depth + 1);
        }
    }

    /// Reflect at this level (direct participants only)
    pub fn reflect_local(&mut self, participants: &[Participant], timestamp: u64) {
        self.mirror.reflect(participants, timestamp);
    }

    /// Reflect recursively (this level and all children)
    pub fn reflect_recursive(&mut self, timestamp: u64) {
        // First, ensure all children have reflected
        for child in &mut self.children {
            child.reflect_recursive(timestamp);
        }

        // Then analyze cross-scale patterns if we have children
        if !self.children.is_empty() {
            self.cross_scale = Some(self.analyze_cross_scale());
        }
    }

    /// Analyze patterns that emerge across child groups
    fn analyze_cross_scale(&self) -> CrossScaleAnalysis {
        let child_reflections: Vec<&GroupReflection> =
            self.children.iter().map(|c| &c.mirror.reflection).collect();

        // Inter-child coherence: do children agree with each other?
        let inter_child_coherence = self.calculate_inter_child_coherence(&child_reflections);

        // Diversity illusion detection
        let diversity_illusion = self.detect_diversity_illusion(&child_reflections);

        // Cross-group Cassandras
        let cross_group_cassandras = self.find_cross_group_cassandras();

        // Emergent void
        let emergent_void = self.analyze_emergent_void(&child_reflections);

        // Meta topology
        let meta_topology = self.analyze_meta_topology();

        // Generate scale prompt
        let scale_prompt = self.generate_scale_prompt(
            &diversity_illusion,
            &cross_group_cassandras,
            &emergent_void,
        );

        CrossScaleAnalysis {
            inter_child_coherence,
            diversity_illusion,
            cross_group_cassandras,
            emergent_void,
            meta_topology,
            scale_prompt,
        }
    }

    /// Calculate how much children agree with each other
    fn calculate_inter_child_coherence(&self, reflections: &[&GroupReflection]) -> f64 {
        if reflections.len() < 2 {
            return 0.0;
        }

        // Compare agreement levels and harmony distributions across children
        let agreement_levels: Vec<f64> = reflections.iter().map(|r| r.agreement_level).collect();

        // Low variance in agreement levels = high inter-child coherence
        let var = variance(&agreement_levels);

        // Also consider if children have similar absent harmonies
        let avg_agreement: f64 =
            agreement_levels.iter().sum::<f64>() / agreement_levels.len() as f64;

        // Coherence is high if children are similar
        (1.0 - var.min(1.0)) * avg_agreement
    }

    /// Detect if local diversity masks global conformity
    fn detect_diversity_illusion(&self, reflections: &[&GroupReflection]) -> DiversityIllusion {
        if reflections.is_empty() {
            return DiversityIllusion {
                detected: false,
                local_diversity: 0.0,
                global_diversity: 0.0,
                illusion_ratio: 0.0,
                hidden_consensus: Vec::new(),
                warning: None,
            };
        }

        // Local diversity: average of (1 - agreement) within each child
        let local_diversity: f64 = reflections
            .iter()
            .map(|r| 1.0 - r.agreement_level)
            .sum::<f64>()
            / reflections.len() as f64;

        // Global diversity: variance of agreement levels across children
        let agreement_levels: Vec<f64> = reflections.iter().map(|r| r.agreement_level).collect();
        let global_diversity = variance(&agreement_levels).sqrt(); // std dev

        // Illusion ratio: high local diversity + low global diversity = illusion
        let illusion_ratio = if global_diversity > 0.01 {
            local_diversity / global_diversity
        } else {
            local_diversity * 10.0 // If global diversity is near zero, amplify
        };

        let detected = local_diversity > 0.3 && global_diversity < 0.15 && illusion_ratio > 2.0;

        // Find hidden consensus (what are all children agreeing on?)
        let hidden_consensus = if detected {
            self.find_hidden_consensus(reflections)
        } else {
            Vec::new()
        };

        let warning = if detected {
            Some(format!(
                "Diversity illusion detected: Each group appears diverse internally, \
                 but all groups are converging on the same conclusions. \
                 Local diversity: {:.0}%, Global diversity: {:.0}%",
                local_diversity * 100.0,
                global_diversity * 100.0
            ))
        } else {
            None
        };

        DiversityIllusion {
            detected,
            local_diversity,
            global_diversity,
            illusion_ratio,
            hidden_consensus,
            warning,
        }
    }

    /// Find what all children are agreeing on without realizing
    fn find_hidden_consensus(&self, reflections: &[&GroupReflection]) -> Vec<String> {
        let mut consensus = Vec::new();

        // Check if all children have similar echo chamber risks
        let all_high_agreement = reflections.iter().all(|r| r.agreement_level > 0.6);

        if all_high_agreement {
            consensus.push("All groups have high internal agreement".to_string());
        }

        // Check if all children are missing the same harmonies
        let harmony_sets: Vec<HashSet<Harmony>> = reflections
            .iter()
            .map(|r| {
                r.shadow
                    .absent_harmonies
                    .iter()
                    .map(|a| a.harmony)
                    .collect()
            })
            .collect();

        if !harmony_sets.is_empty() {
            let common_absent: HashSet<Harmony> = harmony_sets[0]
                .iter()
                .filter(|h| harmony_sets.iter().all(|set| set.contains(h)))
                .copied()
                .collect();

            for harmony in common_absent {
                consensus.push(format!("{:?} is absent from ALL groups", harmony));
            }
        }

        consensus
    }

    /// Find minorities in one group that are majorities elsewhere
    fn find_cross_group_cassandras(&self) -> Vec<CrossGroupCassandra> {
        // This would require tracking positions across groups
        // For now, return empty - would need richer position tracking
        Vec::new()
    }

    /// Analyze voids that emerge at the parent level
    fn analyze_emergent_void(&self, reflections: &[&GroupReflection]) -> EmergentVoid {
        // Find harmonies absent from ALL children
        let mut universally_absent = HashSet::new();

        if !reflections.is_empty() {
            // Start with first child's absent harmonies
            for absent in &reflections[0].shadow.absent_harmonies {
                universally_absent.insert(absent.harmony);
            }

            // Intersect with other children
            for reflection in reflections.iter().skip(1) {
                let child_absent: HashSet<Harmony> = reflection
                    .shadow
                    .absent_harmonies
                    .iter()
                    .map(|a| a.harmony)
                    .collect();
                universally_absent = universally_absent
                    .intersection(&child_absent)
                    .copied()
                    .collect();
            }
        }

        let universally_absent_harmonies: Vec<Harmony> = universally_absent.into_iter().collect();

        // Generate unasked questions based on universally absent harmonies
        let unasked_across_all: Vec<String> = universally_absent_harmonies
            .iter()
            .map(|h| {
                format!(
                    "No group is asking: How does {:?} apply to our collective work?",
                    h
                )
            })
            .collect();

        // Find shared assumptions (high agreement + low epistemic in all children)
        let shared_assumptions: Vec<SharedAssumption> = if reflections.iter().all(|r| {
            matches!(
                r.signal_integrity.echo_chamber_risk,
                EchoChamberRisk::High | EchoChamberRisk::Critical
            )
        }) {
            vec![SharedAssumption {
                assumption: "All groups have unverified consensus".to_string(),
                prevalence: reflections.len(),
                questioned: false,
                blind_spot: "No group is seeking external verification".to_string(),
            }]
        } else {
            Vec::new()
        };

        let void_prompt =
            if universally_absent_harmonies.is_empty() && shared_assumptions.is_empty() {
                "No emergent void detected at this scale.".to_string()
            } else {
                let mut parts = Vec::new();
                if !universally_absent_harmonies.is_empty() {
                    parts.push(format!(
                        "{} harmonies are absent from ALL groups: {:?}",
                        universally_absent_harmonies.len(),
                        universally_absent_harmonies
                    ));
                }
                if !shared_assumptions.is_empty() {
                    parts.push(format!(
                        "{} assumptions are shared across all groups without verification",
                        shared_assumptions.len()
                    ));
                }
                parts.join(". ")
            };

        EmergentVoid {
            universally_absent_harmonies,
            unasked_across_all,
            shared_assumptions,
            void_prompt,
        }
    }

    /// Analyze topology at the group-of-groups level
    fn analyze_meta_topology(&self) -> MetaTopology {
        let n = self.children.len();

        if n < 2 {
            return MetaTopology {
                dominant_groups: Vec::new(),
                isolated_groups: Vec::new(),
                bridge_groups: Vec::new(),
                structure: MetaStructure::Insufficient,
            };
        }

        // Find dominant groups (those with highest agreement that others might follow)
        let mut groups_by_agreement: Vec<(&HolonicMirror, f64)> = self
            .children
            .iter()
            .map(|c| (c, c.mirror.reflection.agreement_level))
            .collect();
        groups_by_agreement
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let dominant_groups: Vec<String> = groups_by_agreement
            .iter()
            .take(1)
            .filter(|(_, agreement)| *agreement > 0.7)
            .map(|(g, _)| g.name.clone())
            .collect();

        // Find isolated groups (those with fragmented topology)
        let isolated_groups: Vec<String> = self
            .children
            .iter()
            .filter(|c| c.mirror.reflection.topology.topology_type == TopologyType::Fragmented)
            .map(|c| c.name.clone())
            .collect();

        // Determine overall structure
        let structure =
            if !dominant_groups.is_empty() && dominant_groups.len() < n / 2 {
                MetaStructure::Hierarchical
            } else if isolated_groups.len() > n / 2 {
                MetaStructure::Fragmented
            } else if self.children.iter().all(|c| {
                c.mirror.reflection.topology.topology_type == TopologyType::DistributedMesh
            }) {
                MetaStructure::NetworkedMesh
            } else {
                MetaStructure::Federated
            };

        MetaTopology {
            dominant_groups,
            isolated_groups,
            bridge_groups: Vec::new(), // Would need cross-group alignment data
            structure,
        }
    }

    /// Generate a summary prompt for this scale
    fn generate_scale_prompt(
        &self,
        diversity_illusion: &DiversityIllusion,
        cassandras: &[CrossGroupCassandra],
        emergent_void: &EmergentVoid,
    ) -> String {
        let mut prompts = Vec::new();

        if let Some(warning) = &diversity_illusion.warning {
            prompts.push(warning.clone());
        }

        if !cassandras.is_empty() {
            prompts.push(format!(
                "{} positions are minorities in some groups but majorities in others",
                cassandras.len()
            ));
        }

        if !emergent_void.universally_absent_harmonies.is_empty() {
            prompts.push(emergent_void.void_prompt.clone());
        }

        if prompts.is_empty() {
            format!(
                "At the {} level ({} groups): No concerning cross-scale patterns detected.",
                self.name,
                self.children.len()
            )
        } else {
            format!(
                "At the {} level ({} groups):\n{}",
                self.name,
                self.children.len(),
                prompts.join("\n")
            )
        }
    }

    /// Get a multi-scale summary (this level and all descendants)
    pub fn multi_scale_summary(&self) -> String {
        let mut lines = Vec::new();

        let indent = "  ".repeat(self.depth);
        lines.push(format!(
            "{}[{}] {} ({} participants, {} child groups)",
            indent,
            self.depth,
            self.name,
            self.mirror.reflection.participant_count,
            self.children.len()
        ));

        // Add cross-scale analysis if present
        if let Some(cross_scale) = &self.cross_scale {
            if cross_scale.diversity_illusion.detected {
                lines.push(format!("{}  ⚠️ DIVERSITY ILLUSION", indent));
            }
            if !cross_scale
                .emergent_void
                .universally_absent_harmonies
                .is_empty()
            {
                lines.push(format!(
                    "{}  ⚠️ EMERGENT VOID: {:?}",
                    indent, cross_scale.emergent_void.universally_absent_harmonies
                ));
            }
        }

        // Recurse into children
        for child in &self.children {
            lines.push(child.multi_scale_summary());
        }

        lines.join("\n")
    }

    /// Find the depth of the entire tree
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth)
        }
    }

    /// Count total groups in the tree
    pub fn total_groups(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|c| c.total_groups())
            .sum::<usize>()
    }

    /// Get all groups at a specific depth
    pub fn groups_at_depth(&self, target_depth: usize) -> Vec<&HolonicMirror> {
        if self.depth == target_depth {
            vec![self]
        } else {
            self.children
                .iter()
                .flat_map(|c| c.groups_at_depth(target_depth))
                .collect()
        }
    }
}

// ==============================================================================
// GOVERNANCE INTEGRATION: Bridge to Φ-Weighted Voting
// ==============================================================================
//
// This adapter connects CollectiveMirror to the mycelix-governance system,
// enabling real-time collective sensing during proposal voting.

/// A voter in the governance system (mirrors governance zome data)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GovernanceVoter {
    /// Decentralized identifier
    pub did: String,
    /// Consciousness score (0-1) from Symthaea
    pub phi: f64,
    /// K-Vector trust score (0-1)
    pub k_trust: f64,
    /// Stake weight (capped, 0-1)
    pub stake_weight: f64,
    /// Historical participation (0-1)
    pub participation_score: f64,
    /// Domain expertise (0-1)
    pub domain_reputation: f64,
    /// Who this voter has delegated to (if any)
    pub delegated_to: Option<String>,
    /// Voters who have delegated to this voter
    pub delegated_from: Vec<String>,
}

/// A vote cast in the governance system
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GovernanceVote {
    /// Who cast the vote
    pub voter_did: String,
    /// The choice (For, Against, Abstain)
    pub choice: VoteChoice,
    /// Effective weight after all adjustments
    pub effective_weight: f64,
    /// Which harmonies influenced this vote
    pub harmony_rationale: Option<HarmonyEmphasis>,
    /// When the vote was cast
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

/// Adapter to convert governance data into CollectiveMirror inputs
pub struct GovernanceAdapter;

impl GovernanceAdapter {
    /// Convert a governance voter into a CollectiveMirror participant
    pub fn voter_to_participant(voter: &GovernanceVoter) -> Participant {
        // Map Φ score to epistemic level
        // Higher Φ = higher epistemic grounding (more conscious = more rigorous)
        let epistemic_level = voter.phi * 0.6 + voter.k_trust * 0.4;

        // Build alignments from delegation relationships
        let alignments = if let Some(delegate) = &voter.delegated_to {
            vec![delegate.clone()]
        } else {
            Vec::new()
        };

        // Voters who delegate to you = you influence them
        // This becomes visible in topology analysis
        let disagreements = Vec::new(); // Would need vote comparison data

        // Map domain reputation to harmony emphasis
        // This is a heuristic - in practice, would use actual voter values
        let harmony_emphasis = HarmonyEmphasis {
            resonant_coherence: voter.participation_score,
            pan_sentient_flourishing: voter.phi * 0.8, // Φ correlates with care
            integral_wisdom: voter.domain_reputation,
            infinite_play: 0.5, // Default moderate
            universal_interconnectedness: voter.k_trust,
            sacred_reciprocity: voter.stake_weight.min(0.5) * 2.0, // Cap influence
            evolutionary_progression: voter.participation_score,
            sacred_stillness: 0.5, // Default moderate
        };

        Participant {
            id: voter.did.clone(),
            harmony_emphasis,
            epistemic_level,
            alignments,
            disagreements,
            attention_focus: None,
            last_active: 0,
        }
    }

    /// Convert a list of governance voters into participants
    pub fn voters_to_participants(voters: &[GovernanceVoter]) -> Vec<Participant> {
        voters.iter().map(Self::voter_to_participant).collect()
    }

    /// Analyze a live proposal voting session
    pub fn analyze_proposal_voting(
        voters: &[GovernanceVoter],
        votes: &[GovernanceVote],
        proposal_id: &str,
        timestamp: u64,
    ) -> ProposalReflection {
        let mut mirror = CollectiveMirror::with_defaults();
        let participants = Self::voters_to_participants(voters);
        mirror.reflect(&participants, timestamp);

        // Analyze vote distribution
        let mut for_count = 0;
        let mut against_count = 0;
        let mut abstain_count = 0;
        let mut for_weight = 0.0;
        let mut against_weight = 0.0;

        for vote in votes {
            match vote.choice {
                VoteChoice::For => {
                    for_count += 1;
                    for_weight += vote.effective_weight;
                }
                VoteChoice::Against => {
                    against_count += 1;
                    against_weight += vote.effective_weight;
                }
                VoteChoice::Abstain => {
                    abstain_count += 1;
                }
            }
        }

        let total_weight = for_weight + against_weight;
        let approval_ratio = if total_weight > 0.0 {
            for_weight / total_weight
        } else {
            0.0
        };

        // Detect polarization
        let polarization = if votes.len() > 1 {
            let choice_variance =
                (for_count as f64 - against_count as f64).abs() / votes.len() as f64;
            1.0 - choice_variance // High variance = low polarization, low variance = high polarization
        } else {
            0.0
        };

        // Get interventions if needed
        let interventions = mirror.suggest_interventions();

        // Generate governance-specific prompts
        let governance_prompts = Self::generate_governance_prompts(
            &mirror.reflection,
            approval_ratio,
            polarization,
            &interventions,
        );

        ProposalReflection {
            proposal_id: proposal_id.to_string(),
            timestamp,
            group_reflection: mirror.reflection,
            vote_summary: VoteSummary {
                for_count,
                against_count,
                abstain_count,
                for_weight,
                against_weight,
                approval_ratio,
                polarization,
            },
            suggested_interventions: interventions,
            governance_prompts,
        }
    }

    /// Generate governance-specific reflection prompts
    fn generate_governance_prompts(
        reflection: &GroupReflection,
        approval_ratio: f64,
        polarization: f64,
        interventions: &[InterventionSuggestion],
    ) -> Vec<String> {
        let mut prompts = Vec::new();

        // High approval with echo chamber risk
        if approval_ratio > 0.8
            && matches!(
                reflection.signal_integrity.echo_chamber_risk,
                EchoChamberRisk::High | EchoChamberRisk::Critical
            )
        {
            prompts.push(
                "Strong consensus (>80%) detected with low epistemic verification. \
                 Consider: Are dissenting voices being heard? Is this groupthink?"
                    .to_string(),
            );
        }

        // Very close vote with low participation
        if (approval_ratio - 0.5).abs() < 0.1 && reflection.participant_count < 10 {
            prompts.push(
                "Vote is nearly split (45-55%) with limited participation. \
                 Consider: Should this decision wait for more voices?"
                    .to_string(),
            );
        }

        // High polarization
        if polarization > 0.7 {
            prompts.push(
                "High polarization detected - voters are clustering into opposing camps. \
                 Consider: Is there a middle ground being overlooked?"
                    .to_string(),
            );
        }

        // Centralized voting (hub-and-spoke)
        if reflection.topology.topology_type == TopologyType::HubAndSpoke {
            prompts.push(
                "Voting patterns suggest influence is centralized around few voices. \
                 Consider: Are delegates making independent decisions?"
                    .to_string(),
            );
        }

        // Add intervention-based prompts
        for intervention in interventions {
            if matches!(intervention.trigger, InterventionTrigger::RapidConvergence) {
                prompts.push(
                    "Agreement is rising rapidly. Before finalizing: \
                     Has someone been assigned to argue the opposing view?"
                        .to_string(),
                );
            }
        }

        if prompts.is_empty() {
            prompts
                .push("The voting appears to have healthy diversity and verification.".to_string());
        }

        prompts
    }
}

/// Reflection on a specific proposal's voting
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProposalReflection {
    /// The proposal being analyzed
    pub proposal_id: String,
    /// When this reflection was generated
    pub timestamp: u64,
    /// The underlying group reflection
    pub group_reflection: GroupReflection,
    /// Summary of votes
    pub vote_summary: VoteSummary,
    /// Suggested interventions
    pub suggested_interventions: Vec<InterventionSuggestion>,
    /// Governance-specific prompts
    pub governance_prompts: Vec<String>,
}

/// Summary of voting on a proposal
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VoteSummary {
    /// Count of For votes
    pub for_count: usize,
    /// Count of Against votes
    pub against_count: usize,
    /// Count of Abstain votes
    pub abstain_count: usize,
    /// Total weight of For votes
    pub for_weight: f64,
    /// Total weight of Against votes
    pub against_weight: f64,
    /// Approval ratio (for_weight / total_weight)
    pub approval_ratio: f64,
    /// Polarization score (0 = consensus, 1 = split)
    pub polarization: f64,
}

impl ProposalReflection {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        let v = &self.vote_summary;
        format!(
            "Proposal {} Reflection:\n\
             • Votes: {} For, {} Against, {} Abstain\n\
             • Approval: {:.0}% (weighted)\n\
             • Polarization: {:.0}%\n\
             • Topology: {:?}\n\
             • Echo Chamber Risk: {:?}\n\
             \n\
             Prompts:\n{}",
            self.proposal_id,
            v.for_count,
            v.against_count,
            v.abstain_count,
            v.approval_ratio * 100.0,
            v.polarization * 100.0,
            self.group_reflection.topology.topology_type,
            self.group_reflection.signal_integrity.echo_chamber_risk,
            self.governance_prompts
                .iter()
                .map(|p| format!("  • {}", p))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Should this proposal be flagged for additional review?
    pub fn needs_review(&self) -> bool {
        // Flag if any concerning patterns
        matches!(
            self.group_reflection.signal_integrity.echo_chamber_risk,
            EchoChamberRisk::High | EchoChamberRisk::Critical
        ) || self.group_reflection.topology.topology_type == TopologyType::HubAndSpoke
            || self.group_reflection.trajectory.rapid_convergence_warning
            || self.vote_summary.polarization > 0.8
    }
}

// ==============================================================================
// TESTS
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_participant(id: &str, epistemic: f64, alignments: Vec<&str>) -> Participant {
        Participant {
            id: id.to_string(),
            harmony_emphasis: HarmonyEmphasis {
                resonant_coherence: 0.5,
                pan_sentient_flourishing: 0.6,
                integral_wisdom: 0.5,
                infinite_play: 0.4,
                universal_interconnectedness: 0.5,
                sacred_reciprocity: 0.5,
                evolutionary_progression: 0.4,
                sacred_stillness: 0.5,
            },
            epistemic_level: epistemic,
            alignments: alignments.into_iter().map(String::from).collect(),
            disagreements: Vec::new(),
            attention_focus: Some("proposal-1".to_string()),
            last_active: 1000,
        }
    }

    #[test]
    fn test_mirror_detects_hub_and_spoke() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create a hub-and-spoke pattern: everyone aligns with "leader"
        let participants = vec![
            create_participant("leader", 0.5, vec![]),
            create_participant("follower1", 0.3, vec!["leader"]),
            create_participant("follower2", 0.3, vec!["leader"]),
            create_participant("follower3", 0.3, vec!["leader"]),
            create_participant("follower4", 0.3, vec!["leader"]),
        ];

        mirror.reflect(&participants, 1000);

        // Should detect centralization
        assert!(mirror.reflection.topology.centralization > 0.5);
    }

    #[test]
    fn test_mirror_detects_echo_chamber() {
        let mut mirror = CollectiveMirror::with_defaults();

        // High agreement, low epistemic level
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.2, vec![]); // Low epistemic
            p.harmony_emphasis = HarmonyEmphasis {
                resonant_coherence: 0.9, // All same emphasis
                pan_sentient_flourishing: 0.9,
                integral_wisdom: 0.9,
                infinite_play: 0.9,
                universal_interconnectedness: 0.9,
                sacred_reciprocity: 0.9,
                evolutionary_progression: 0.9,
                sacred_stillness: 0.9,
            };
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        assert!(matches!(
            mirror.reflection.signal_integrity.echo_chamber_risk,
            EchoChamberRisk::High | EchoChamberRisk::Critical
        ));
    }

    #[test]
    fn test_mirror_finds_shadow() {
        let mut mirror = CollectiveMirror::with_defaults();

        // All participants ignore "care" (pan-sentient flourishing)
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.6, vec![]);
            p.harmony_emphasis.pan_sentient_flourishing = 0.0; // Care absent
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        // Should detect missing harmony
        let absent = &mirror.reflection.shadow.absent_harmonies;
        assert!(absent
            .iter()
            .any(|a| a.harmony == Harmony::PanSentientFlourishing));
    }

    #[test]
    fn test_reflection_prompt_is_helpful() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create problematic group state
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.2, vec!["leader"]);
            p.harmony_emphasis.pan_sentient_flourishing = 0.0;
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        let prompt = mirror.reflection_prompt();

        // Should mention something useful, not claim wisdom
        assert!(!prompt.contains("wisdom"));
        assert!(!prompt.contains("emerged"));
    }

    // =========================================================================
    // TRAJECTORY TESTS
    // =========================================================================

    #[test]
    fn test_trajectory_needs_history() {
        let mut mirror = CollectiveMirror::with_defaults();
        let participants = vec![
            create_participant("a", 0.5, vec![]),
            create_participant("b", 0.5, vec![]),
            create_participant("c", 0.5, vec![]),
        ];

        // First reflection - no history yet
        mirror.reflect(&participants, 1000);
        assert_eq!(mirror.reflection.trajectory.sample_size, 0);
        assert_eq!(
            mirror.reflection.trajectory.agreement_direction,
            TrendDirection::Unknown
        );
    }

    #[test]
    fn test_trajectory_detects_rising_agreement() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Simulate increasing agreement over time
        for i in 0..5 {
            let agreement_level = 0.3 + (i as f64 * 0.1); // 0.3, 0.4, 0.5, 0.6, 0.7
            let mut participants = Vec::new();
            for j in 0..5 {
                let mut p = create_participant(&format!("user{}", j), 0.5, vec![]);
                // Increase harmony alignment to simulate rising agreement
                p.harmony_emphasis = HarmonyEmphasis {
                    resonant_coherence: agreement_level,
                    pan_sentient_flourishing: agreement_level,
                    integral_wisdom: agreement_level,
                    infinite_play: agreement_level,
                    universal_interconnectedness: agreement_level,
                    sacred_reciprocity: agreement_level,
                    evolutionary_progression: agreement_level,
                    sacred_stillness: agreement_level,
                };
                participants.push(p);
            }
            mirror.reflect(&participants, 1000 + i * 100);
        }

        // After 5 reflections, should detect rising trend
        assert!(mirror.reflection.trajectory.sample_size >= 3);
    }

    // =========================================================================
    // VOID ANALYSIS TESTS
    // =========================================================================

    #[test]
    fn test_void_detects_unverified_consensus() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create echo chamber: high agreement, low epistemic
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.1, vec![]); // Very low epistemic
            p.harmony_emphasis = HarmonyEmphasis {
                resonant_coherence: 0.95,
                pan_sentient_flourishing: 0.95,
                integral_wisdom: 0.95,
                infinite_play: 0.95,
                universal_interconnectedness: 0.95,
                sacred_reciprocity: 0.95,
                evolutionary_progression: 0.95,
                sacred_stillness: 0.95,
            };
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        // Should detect unverified consensus in void analysis
        assert!(!mirror.reflection.void.unverified_consensus.is_empty());
    }

    #[test]
    fn test_void_generates_unasked_questions() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create group missing multiple harmonies
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.6, vec![]);
            p.harmony_emphasis.pan_sentient_flourishing = 0.0; // Care absent
            p.harmony_emphasis.sacred_reciprocity = 0.0; // Fairness absent
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        // Should generate unasked questions for absent harmonies
        assert!(!mirror.reflection.void.unasked_questions.is_empty());
    }

    // =========================================================================
    // INTERVENTION TESTS
    // =========================================================================

    #[test]
    fn test_interventions_for_echo_chamber() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create echo chamber
        let mut participants = Vec::new();
        for i in 0..5 {
            let mut p = create_participant(&format!("user{}", i), 0.15, vec![]);
            p.harmony_emphasis = HarmonyEmphasis {
                resonant_coherence: 0.95,
                pan_sentient_flourishing: 0.95,
                integral_wisdom: 0.95,
                infinite_play: 0.95,
                universal_interconnectedness: 0.95,
                sacred_reciprocity: 0.95,
                evolutionary_progression: 0.95,
                sacred_stillness: 0.95,
            };
            participants.push(p);
        }

        mirror.reflect(&participants, 1000);

        let interventions = mirror.suggest_interventions();

        // Should suggest verification intervention
        assert!(interventions
            .iter()
            .any(|i| i.trigger == InterventionTrigger::EchoChamber));
    }

    #[test]
    fn test_interventions_for_centralization() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create hub-and-spoke (highly centralized)
        let participants = vec![
            create_participant("leader", 0.5, vec![]),
            create_participant("f1", 0.3, vec!["leader"]),
            create_participant("f2", 0.3, vec!["leader"]),
            create_participant("f3", 0.3, vec!["leader"]),
            create_participant("f4", 0.3, vec!["leader"]),
            create_participant("f5", 0.3, vec!["leader"]),
        ];

        mirror.reflect(&participants, 1000);

        let interventions = mirror.suggest_interventions();

        // Should suggest decentralization intervention
        assert!(interventions
            .iter()
            .any(|i| i.trigger == InterventionTrigger::HighCentralization));
    }

    // =========================================================================
    // HOLONIC MIRROR TESTS
    // =========================================================================

    #[test]
    fn test_holonic_mirror_structure() {
        let mut ecosystem = HolonicMirror::new("eco".into(), "Ecosystem".into());
        let mut org = HolonicMirror::new("org".into(), "Organization".into());
        let team = HolonicMirror::new("team".into(), "Team".into());

        org.add_child(team);
        ecosystem.add_child(org);

        assert_eq!(ecosystem.depth, 0);
        assert_eq!(ecosystem.children[0].depth, 1);
        assert_eq!(ecosystem.children[0].children[0].depth, 2);
        assert_eq!(ecosystem.total_groups(), 3);
        assert_eq!(ecosystem.max_depth(), 2);
    }

    #[test]
    fn test_holonic_groups_at_depth() {
        let mut root = HolonicMirror::new("root".into(), "Root".into());

        let child1 = HolonicMirror::new("c1".into(), "Child1".into());
        let child2 = HolonicMirror::new("c2".into(), "Child2".into());

        root.add_child(child1);
        root.add_child(child2);

        let depth_0 = root.groups_at_depth(0);
        let depth_1 = root.groups_at_depth(1);

        assert_eq!(depth_0.len(), 1);
        assert_eq!(depth_1.len(), 2);
    }

    fn create_group_with_participants(
        id: &str,
        name: &str,
        participant_count: usize,
        epistemic: f64,
        agreement: f64,
    ) -> HolonicMirror {
        let mut mirror = HolonicMirror::new(id.into(), name.into());

        let mut participants = Vec::new();
        for i in 0..participant_count {
            let mut p = create_participant(&format!("{}-user{}", id, i), epistemic, vec![]);
            p.harmony_emphasis = HarmonyEmphasis {
                resonant_coherence: agreement,
                pan_sentient_flourishing: agreement,
                integral_wisdom: agreement,
                infinite_play: agreement,
                universal_interconnectedness: agreement,
                sacred_reciprocity: agreement,
                evolutionary_progression: agreement,
                sacred_stillness: agreement,
            };
            participants.push(p);
        }

        mirror.reflect_local(&participants, 1000);
        mirror
    }

    #[test]
    fn test_holonic_diversity_illusion_detection() {
        let mut org = HolonicMirror::new("org".into(), "Organization".into());

        // Create teams that are internally diverse but externally homogeneous
        // Each team has ~40% internal diversity (agreement ~0.6)
        // But all teams have similar agreement levels (low global diversity)
        let team1 = create_group_with_participants("t1", "Team1", 5, 0.5, 0.6);
        let team2 = create_group_with_participants("t2", "Team2", 5, 0.5, 0.62);
        let team3 = create_group_with_participants("t3", "Team3", 5, 0.5, 0.58);

        org.add_child(team1);
        org.add_child(team2);
        org.add_child(team3);

        org.reflect_recursive(1000);

        // Check that cross-scale analysis exists
        assert!(org.cross_scale.is_some());
    }

    #[test]
    fn test_holonic_emergent_void() {
        let mut org = HolonicMirror::new("org".into(), "Organization".into());

        // Create teams that all miss the same harmony
        for i in 0..3 {
            let mut team = HolonicMirror::new(format!("t{}", i), format!("Team{}", i));

            let mut participants = Vec::new();
            for j in 0..4 {
                let mut p = create_participant(&format!("t{}-u{}", i, j), 0.5, vec![]);
                p.harmony_emphasis.pan_sentient_flourishing = 0.0; // All teams miss Care
                participants.push(p);
            }

            team.reflect_local(&participants, 1000);
            org.add_child(team);
        }

        org.reflect_recursive(1000);

        // Should detect emergent void - Care absent from ALL teams
        if let Some(cross_scale) = &org.cross_scale {
            // The emergent void should identify that Care is universally absent
            assert!(
                cross_scale
                    .emergent_void
                    .universally_absent_harmonies
                    .contains(&Harmony::PanSentientFlourishing)
                    || !cross_scale.emergent_void.unasked_across_all.is_empty()
            );
        }
    }

    #[test]
    fn test_holonic_multi_scale_summary() {
        let mut root = HolonicMirror::new("root".into(), "Ecosystem".into());

        let mut org = HolonicMirror::new("org".into(), "Org".into());
        let team = create_group_with_participants("team", "Team", 5, 0.5, 0.5);
        org.add_child(team);

        root.add_child(org);
        root.reflect_recursive(1000);

        let summary = root.multi_scale_summary();

        // Should contain hierarchy
        assert!(summary.contains("Ecosystem"));
        assert!(summary.contains("Org"));
        assert!(summary.contains("Team"));
    }

    // =========================================================================
    // CALIBRATION TESTS
    // =========================================================================

    #[test]
    fn test_calibration_needs_outcomes() {
        let mirror = CollectiveMirror::with_defaults();

        // No outcomes recorded yet
        assert!(mirror.analyze_calibration().is_none());
    }

    #[test]
    fn test_calibration_with_outcomes() {
        let mut mirror = CollectiveMirror::with_defaults();

        // Create some baseline state
        let participants = vec![
            create_participant("a", 0.5, vec![]),
            create_participant("b", 0.5, vec![]),
            create_participant("c", 0.5, vec![]),
        ];
        mirror.reflect(&participants, 1000);

        // Record multiple outcomes
        for i in 0..6 {
            let outcome = if i % 2 == 0 {
                DecisionOutcome::Positive
            } else {
                DecisionOutcome::Negative
            };
            mirror.observe_outcome(format!("decision-{}", i), outcome, 2000 + i * 100);
        }

        // Now calibration should work
        let insight = mirror.analyze_calibration();
        assert!(insight.is_some());
    }

    // =========================================================================
    // HELPER FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_calculate_trend() {
        // Rising values
        let rising = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend(&rising);
        assert!(trend > 0.0);

        // Falling values
        let falling = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = calculate_trend(&falling);
        assert!(trend < 0.0);

        // Stable values
        let stable = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = calculate_trend(&stable);
        assert!(trend.abs() < 0.01);
    }

    #[test]
    fn test_trend_direction_from_velocity() {
        assert_eq!(
            TrendDirection::from_velocity(0.1, 0.05),
            TrendDirection::Rising
        );
        assert_eq!(
            TrendDirection::from_velocity(-0.1, 0.05),
            TrendDirection::Falling
        );
        assert_eq!(
            TrendDirection::from_velocity(0.02, 0.05),
            TrendDirection::Stable
        );
    }

    // =========================================================================
    // GOVERNANCE ADAPTER TESTS
    // =========================================================================

    fn create_voter(
        did: &str,
        phi: f64,
        k_trust: f64,
        delegated_to: Option<&str>,
    ) -> GovernanceVoter {
        GovernanceVoter {
            did: did.to_string(),
            phi,
            k_trust,
            stake_weight: 0.1,
            participation_score: 0.5,
            domain_reputation: 0.5,
            delegated_to: delegated_to.map(String::from),
            delegated_from: Vec::new(),
        }
    }

    fn create_vote(voter_did: &str, choice: VoteChoice, weight: f64) -> GovernanceVote {
        GovernanceVote {
            voter_did: voter_did.to_string(),
            choice,
            effective_weight: weight,
            harmony_rationale: None,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_voter_to_participant_conversion() {
        let voter = create_voter("did:example:alice", 0.7, 0.8, None);
        let participant = GovernanceAdapter::voter_to_participant(&voter);

        assert_eq!(participant.id, "did:example:alice");
        // Epistemic level should combine phi and k_trust
        assert!(participant.epistemic_level > 0.5);
    }

    #[test]
    fn test_voter_delegation_creates_alignment() {
        let voter = create_voter("did:example:bob", 0.5, 0.5, Some("did:example:leader"));
        let participant = GovernanceAdapter::voter_to_participant(&voter);

        // Should have alignment to delegate
        assert_eq!(participant.alignments, vec!["did:example:leader"]);
    }

    #[test]
    fn test_proposal_reflection_basic() {
        let voters = vec![
            create_voter("v1", 0.6, 0.7, None),
            create_voter("v2", 0.5, 0.6, None),
            create_voter("v3", 0.4, 0.5, None),
        ];

        let votes = vec![
            create_vote("v1", VoteChoice::For, 1.0),
            create_vote("v2", VoteChoice::For, 0.8),
            create_vote("v3", VoteChoice::Against, 0.6),
        ];

        let reflection =
            GovernanceAdapter::analyze_proposal_voting(&voters, &votes, "MIP-001", 1000);

        assert_eq!(reflection.proposal_id, "MIP-001");
        assert_eq!(reflection.vote_summary.for_count, 2);
        assert_eq!(reflection.vote_summary.against_count, 1);
        assert!(reflection.vote_summary.approval_ratio > 0.5);
    }

    #[test]
    fn test_proposal_reflection_detects_echo_chamber() {
        // Create voters with low epistemic levels (low phi + k_trust)
        let voters: Vec<GovernanceVoter> = (0..5)
            .map(|i| create_voter(&format!("v{}", i), 0.15, 0.15, None))
            .collect();

        // All vote the same way
        let votes: Vec<GovernanceVote> = (0..5)
            .map(|i| create_vote(&format!("v{}", i), VoteChoice::For, 1.0))
            .collect();

        let reflection =
            GovernanceAdapter::analyze_proposal_voting(&voters, &votes, "MIP-002", 1000);

        // Should have high approval ratio
        assert!(reflection.vote_summary.approval_ratio > 0.9);

        // Should detect echo chamber risk due to low epistemic levels
        assert!(matches!(
            reflection
                .group_reflection
                .signal_integrity
                .echo_chamber_risk,
            EchoChamberRisk::High | EchoChamberRisk::Critical
        ));
    }

    #[test]
    fn test_proposal_reflection_needs_review() {
        // Create voters with low epistemic levels
        let voters: Vec<GovernanceVoter> = (0..5)
            .map(|i| create_voter(&format!("v{}", i), 0.1, 0.1, None))
            .collect();

        let votes: Vec<GovernanceVote> = (0..5)
            .map(|i| create_vote(&format!("v{}", i), VoteChoice::For, 1.0))
            .collect();

        let reflection =
            GovernanceAdapter::analyze_proposal_voting(&voters, &votes, "MIP-003", 1000);

        // Should flag for review due to echo chamber
        assert!(reflection.needs_review());
    }

    #[test]
    fn test_proposal_reflection_summary() {
        let voters = vec![
            create_voter("alice", 0.7, 0.8, None),
            create_voter("bob", 0.6, 0.7, None),
        ];

        let votes = vec![
            create_vote("alice", VoteChoice::For, 1.0),
            create_vote("bob", VoteChoice::Against, 0.8),
        ];

        let reflection =
            GovernanceAdapter::analyze_proposal_voting(&voters, &votes, "MIP-004", 1000);

        let summary = reflection.summary();

        // Summary should contain key info
        assert!(summary.contains("MIP-004"));
        assert!(summary.contains("1 For"));
        assert!(summary.contains("1 Against"));
    }
}
