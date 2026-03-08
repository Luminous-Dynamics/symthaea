//! Core WisdomEngine types and the Four Components
//!
//! 1. **The Lens** (HarmonicWeights) - The Eight Harmonies as epistemic lenses
//! 2. **The Setting** (CommunityProfile) - Per-community harmonic weight profiles
//! 3. **The Mirror** (DiversityAuditor) - Bias detection and diversity metrics
//! 4. **The Hand** (ReparationsManager) - Power corrections for marginalized voices
//! 5. **Emergent Weights** - Adaptive learning from outcomes
//! 6. **Multi-Epistemology** - Support for diverse knowledge systems
//! 7. **Causal Graph** - Prediction tracking and reality-checking

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::epistemic::{
    EpistemicClassification, EpistemicContext, NormativeLevel, TestimonialQuality,
};
use crate::harmonic::{Harmony, HarmonicImpact};

use std::collections::HashMap;

use super::{CommunityId, PredictionId, CausalNodeId};

// ==============================================================================
// COMPONENT 1: THE LENS - Harmonic Weights Model
// ==============================================================================

/// Custom harmonic weights that can override the default base weights
///
/// Allows communities to prioritize different harmonies based on their values.
/// All weights should sum to 1.0 for normalized scoring.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonicWeights {
    /// Resonant Coherence weight (default: 0.17)
    pub rc: f32,
    /// Pan-Sentient Flourishing weight (default: 0.17)
    pub psf: f32,
    /// Integral Wisdom weight (default: 0.13)
    pub iw: f32,
    /// Infinite Play weight (default: 0.09)
    pub ip: f32,
    /// Universal Interconnectedness weight (default: 0.13)
    pub ui: f32,
    /// Sacred Reciprocity weight (default: 0.09)
    pub sr: f32,
    /// Evolutionary Progression weight (default: 0.09)
    pub ep: f32,
    /// Sacred Stillness weight (default: 0.13)
    pub ss: f32,
}

impl HarmonicWeights {
    /// Create with default (base) weights from Harmony enum
    pub fn default_weights() -> Self {
        Self {
            rc: 0.17,
            psf: 0.17,
            iw: 0.13,
            ip: 0.09,
            ui: 0.13,
            sr: 0.09,
            ep: 0.09,
            ss: 0.13,
        }
    }

    /// Create balanced weights (all equal: 1/8)
    pub fn balanced() -> Self {
        let w = 1.0 / 8.0;
        Self {
            rc: w,
            psf: w,
            iw: w,
            ip: w,
            ui: w,
            sr: w,
            ep: w,
            ss: w,
        }
    }

    /// Create weights for Indigenous/Cultural communities
    /// Prioritizes: UI (interconnectedness) and SR (reciprocity)
    pub fn indigenous_profile() -> Self {
        Self {
            rc: 0.10,
            psf: 0.13,
            iw: 0.08,
            ip: 0.11,
            ui: 0.23,  // Elevated - interconnectedness
            sr: 0.18,  // Elevated - reciprocity
            ep: 0.05,
            ss: 0.12,
        }
    }

    /// Create weights for Scientific communities
    /// Prioritizes: IW (truth/wisdom) and RC (coherence)
    pub fn scientific_profile() -> Self {
        Self {
            rc: 0.22,  // Elevated - coherence
            psf: 0.09,
            iw: 0.27,  // Elevated - truth/wisdom
            ip: 0.09,
            ui: 0.09,
            sr: 0.07,
            ep: 0.07,
            ss: 0.10,
        }
    }

    /// Create weights for Artistic/Creative communities
    /// Prioritizes: IP (play/creativity) and PSF (flourishing)
    pub fn artistic_profile() -> Self {
        Self {
            rc: 0.08,
            psf: 0.22,  // Elevated - flourishing
            iw: 0.07,
            ip: 0.27,  // Elevated - infinite play
            ui: 0.10,
            sr: 0.08,
            ep: 0.05,
            ss: 0.13,
        }
    }

    /// Create weights for Governance/Political communities
    /// Prioritizes: PSF (flourishing) and SR (reciprocity)
    pub fn governance_profile() -> Self {
        Self {
            rc: 0.13,
            psf: 0.22,  // Elevated - care for all
            iw: 0.10,
            ip: 0.05,
            ui: 0.12,
            sr: 0.18,  // Elevated - fair exchange
            ep: 0.08,
            ss: 0.12,
        }
    }

    /// Create weights for Contemplative/Spiritual communities
    /// Prioritizes: RC (coherence), UI (interconnectedness), and SS (stillness)
    pub fn contemplative_profile() -> Self {
        Self {
            rc: 0.20,  // Elevated - integration
            psf: 0.12,
            iw: 0.12,
            ip: 0.08,
            ui: 0.20,  // Elevated - unity
            sr: 0.05,
            ep: 0.05,
            ss: 0.18,  // Elevated - sacred stillness
        }
    }

    /// Get weight for a specific harmony
    pub fn get(&self, harmony: Harmony) -> f32 {
        match harmony {
            Harmony::ResonantCoherence => self.rc,
            Harmony::PanSentientFlourishing => self.psf,
            Harmony::IntegralWisdom => self.iw,
            Harmony::InfinitePlay => self.ip,
            Harmony::UniversalInterconnectedness => self.ui,
            Harmony::SacredReciprocity => self.sr,
            Harmony::EvolutionaryProgression => self.ep,
            Harmony::SacredStillness => self.ss,
        }
    }

    /// Set weight for a specific harmony
    pub fn set(&mut self, harmony: Harmony, value: f32) {
        let target = match harmony {
            Harmony::ResonantCoherence => &mut self.rc,
            Harmony::PanSentientFlourishing => &mut self.psf,
            Harmony::IntegralWisdom => &mut self.iw,
            Harmony::InfinitePlay => &mut self.ip,
            Harmony::UniversalInterconnectedness => &mut self.ui,
            Harmony::SacredReciprocity => &mut self.sr,
            Harmony::EvolutionaryProgression => &mut self.ep,
            Harmony::SacredStillness => &mut self.ss,
        };
        *target = value;
    }

    /// Check if weights sum to approximately 1.0
    pub fn is_normalized(&self) -> bool {
        let sum = self.rc + self.psf + self.iw + self.ip + self.ui + self.sr + self.ep + self.ss;
        (sum - 1.0).abs() < 0.01
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.rc + self.psf + self.iw + self.ip + self.ui + self.sr + self.ep + self.ss;
        if sum > 0.0 {
            self.rc /= sum;
            self.psf /= sum;
            self.iw /= sum;
            self.ip /= sum;
            self.ui /= sum;
            self.sr /= sum;
            self.ep /= sum;
            self.ss /= sum;
        }
    }

    /// Calculate weighted harmonic score from impact values
    pub fn score(&self, impact: &HarmonicImpact) -> f32 {
        self.rc * impact.resonant_coherence
            + self.psf * impact.pan_sentient_flourishing
            + self.iw * impact.integral_wisdom
            + self.ip * impact.infinite_play
            + self.ui * impact.universal_interconnectedness
            + self.sr * impact.sacred_reciprocity
            + self.ep * impact.evolutionary_progression
            + self.ss * impact.sacred_stillness
    }

    /// Get as array (ordered by Harmony::ALL)
    pub fn to_array(&self) -> [f32; 8] {
        [self.rc, self.psf, self.iw, self.ip, self.ui, self.sr, self.ep, self.ss]
    }

    /// Create from array
    pub fn from_array(values: [f32; 8]) -> Self {
        Self {
            rc: values[0],
            psf: values[1],
            iw: values[2],
            ip: values[3],
            ui: values[4],
            sr: values[5],
            ep: values[6],
            ss: values[7],
        }
    }
}

impl Default for HarmonicWeights {
    fn default() -> Self {
        Self::default_weights()
    }
}

// ==============================================================================
// COMPONENT 2: THE SETTING - Community Profiles
// ==============================================================================

/// Unique identifier for a community

/// A community's epistemic profile - their customized approach to knowledge
///
/// Each community can define their own harmonic weights, E/N/M context preferences,
/// and epistemological framework.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CommunityProfile {
    /// Unique community identifier
    pub id: CommunityId,

    /// Human-readable community name
    pub name: String,

    /// Custom harmonic weights for this community
    pub harmonic_weights: HarmonicWeights,

    /// Preferred epistemic context for E/N/M scoring
    pub default_context: EpistemicContext,

    /// Primary epistemological framework
    pub epistemology: Epistemology,

    /// Normative level at which this profile was decided
    pub consensus_level: NormativeLevel,

    /// How this profile was created
    pub rationale: String,

    /// When this profile should be reviewed
    pub review_timestamp: Option<u64>,

    /// Whether testimonial traditions are elevated in this community
    pub honors_oral_tradition: bool,

    /// Minimum testimonial quality for acceptance (if oral tradition honored)
    pub min_testimonial_quality: Option<TestimonialQuality>,
}

impl CommunityProfile {
    /// Create a new community profile with default settings
    pub fn new(id: CommunityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            harmonic_weights: HarmonicWeights::default_weights(),
            default_context: EpistemicContext::Standard,
            epistemology: Epistemology::WesternScientific,
            consensus_level: NormativeLevel::Group,
            rationale: String::new(),
            review_timestamp: None,
            honors_oral_tradition: false,
            min_testimonial_quality: None,
        }
    }

    /// Create an Indigenous community profile with appropriate defaults
    pub fn indigenous(id: CommunityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            harmonic_weights: HarmonicWeights::indigenous_profile(),
            default_context: EpistemicContext::Indigenous,
            epistemology: Epistemology::IndigenousRelational,
            consensus_level: NormativeLevel::Group,
            rationale: "Indigenous knowledge prioritizes interconnectedness and reciprocity".into(),
            review_timestamp: None,
            honors_oral_tradition: true,
            min_testimonial_quality: Some(TestimonialQuality::CorroboratedOral),
        }
    }

    /// Create a Scientific community profile with appropriate defaults
    pub fn scientific(id: CommunityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            harmonic_weights: HarmonicWeights::scientific_profile(),
            default_context: EpistemicContext::Scientific,
            epistemology: Epistemology::WesternScientific,
            consensus_level: NormativeLevel::Network,
            rationale: "Scientific knowledge prioritizes reproducibility and coherence".into(),
            review_timestamp: None,
            honors_oral_tradition: false,
            min_testimonial_quality: None,
        }
    }

    /// Create a Contemplative/Spiritual community profile
    pub fn contemplative(id: CommunityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            harmonic_weights: HarmonicWeights::contemplative_profile(),
            default_context: EpistemicContext::Contemplative,
            epistemology: Epistemology::BuddhistContemplative,
            consensus_level: NormativeLevel::Group,
            rationale: "Contemplative knowledge values deep integration and experiential knowing".into(),
            review_timestamp: None,
            honors_oral_tradition: true,
            min_testimonial_quality: Some(TestimonialQuality::ElderCouncil),
        }
    }
}

// ==============================================================================
// COMPONENT 3: THE MIRROR - Diversity Auditor
// ==============================================================================

/// Structural position in power hierarchies
///
/// Used for epistemic reparations and diversity tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StructuralPosition {
    /// Dominant/mainstream position with full access
    Dominant,
    /// Mainstream position with standard access
    Mainstream,
    /// Marginalized position with reduced access
    Marginalized,
    /// Historically silenced - voices suppressed over time
    HistoricallySilenced,
}

impl StructuralPosition {
    /// Get reparation adjustment factor for this position
    ///
    /// Higher values for historically silenced voices.
    pub fn reparation_factor(&self) -> f32 {
        match self {
            Self::Dominant => 0.0,
            Self::Mainstream => 0.0,
            Self::Marginalized => 0.05,
            Self::HistoricallySilenced => 0.10,
        }
    }
}

impl Default for StructuralPosition {
    fn default() -> Self {
        Self::Mainstream
    }
}

/// Record of a single evaluation for audit purposes
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluationRecord {
    /// Timestamp of evaluation (Unix epoch)
    pub timestamp: u64,
    /// Community that evaluated
    pub community_id: CommunityId,
    /// Author's structural position
    pub author_position: StructuralPosition,
    /// Epistemology used
    pub epistemology: Epistemology,
    /// Raw score before adjustments
    pub raw_score: f32,
    /// Final score after all adjustments
    pub final_score: f32,
    /// Whether reparations were applied
    pub reparations_applied: bool,
    /// The E/N/M classification code
    pub classification_code: String,
}

/// Aggregated diversity metrics for a time period
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiversityMetrics {
    /// Total evaluations in period
    pub total_evaluations: u64,
    /// Evaluations by structural position
    pub by_position: PositionCounts,
    /// Evaluations by epistemology
    pub by_epistemology: EpistemologyCounts,
    /// Average scores by position (for bias detection)
    pub avg_score_by_position: PositionScores,
    /// Reparations applied count
    pub reparations_count: u64,
    /// High-score claims from marginalized voices
    pub marginalized_high_scores: u64,
    /// Diversity index (0.0-1.0, higher = more diverse)
    pub diversity_index: f32,
}

/// Counts of evaluations by structural position
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PositionCounts {
    pub dominant: u64,
    pub mainstream: u64,
    pub marginalized: u64,
    pub historically_silenced: u64,
}

/// Counts of evaluations by epistemology
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EpistemologyCounts {
    pub western_scientific: u64,
    pub indigenous_relational: u64,
    pub buddhist_contemplative: u64,
    pub islamic_scholarly: u64,
    pub african_ubuntu: u64,
    pub other: u64,
}

/// Average scores by structural position
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PositionScores {
    pub dominant: f32,
    pub mainstream: f32,
    pub marginalized: f32,
    pub historically_silenced: f32,
}

/// The Mirror: Tracks epistemic fairness and diversity metrics
///
/// Records all evaluations to detect systemic bias and ensure
/// the epistemic system is serving all communities fairly.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiversityAuditor {
    /// Rolling buffer of recent evaluations
    pub recent_records: Vec<EvaluationRecord>,
    /// Maximum records to keep
    pub max_records: usize,
    /// Aggregated metrics for current period
    pub current_metrics: DiversityMetrics,
    /// Bias alert threshold (score gap between positions)
    pub bias_threshold: f32,
}

impl DiversityAuditor {
    /// Create new auditor with default settings
    pub fn new() -> Self {
        Self {
            recent_records: Vec::new(),
            max_records: 10_000,
            current_metrics: DiversityMetrics::default(),
            bias_threshold: 0.15, // Alert if >15% gap between positions
        }
    }

    /// Record an evaluation for auditing
    pub fn record(&mut self, evaluation: EvaluationRecord) {
        // Update position counts
        match evaluation.author_position {
            StructuralPosition::Dominant => self.current_metrics.by_position.dominant += 1,
            StructuralPosition::Mainstream => self.current_metrics.by_position.mainstream += 1,
            StructuralPosition::Marginalized => self.current_metrics.by_position.marginalized += 1,
            StructuralPosition::HistoricallySilenced => {
                self.current_metrics.by_position.historically_silenced += 1
            }
        }

        // Update epistemology counts
        match evaluation.epistemology {
            Epistemology::WesternScientific => {
                self.current_metrics.by_epistemology.western_scientific += 1
            }
            Epistemology::IndigenousRelational => {
                self.current_metrics.by_epistemology.indigenous_relational += 1
            }
            Epistemology::BuddhistContemplative => {
                self.current_metrics.by_epistemology.buddhist_contemplative += 1
            }
            Epistemology::IslamicScholarly => {
                self.current_metrics.by_epistemology.islamic_scholarly += 1
            }
            Epistemology::AfricanUbuntu => self.current_metrics.by_epistemology.african_ubuntu += 1,
            _ => self.current_metrics.by_epistemology.other += 1,
        }

        // Track reparations
        if evaluation.reparations_applied {
            self.current_metrics.reparations_count += 1;
        }

        // Track high scores from marginalized voices
        if matches!(
            evaluation.author_position,
            StructuralPosition::Marginalized | StructuralPosition::HistoricallySilenced
        ) && evaluation.final_score > 0.7
        {
            self.current_metrics.marginalized_high_scores += 1;
        }

        self.current_metrics.total_evaluations += 1;

        // Add to recent records (with rotation)
        if self.recent_records.len() >= self.max_records {
            self.recent_records.remove(0);
        }
        self.recent_records.push(evaluation);

        // Recalculate diversity index
        self.recalculate_diversity_index();
    }

    /// Recalculate the diversity index
    fn recalculate_diversity_index(&mut self) {
        let total = self.current_metrics.total_evaluations as f32;
        if total == 0.0 {
            self.current_metrics.diversity_index = 0.0;
            return;
        }

        // Shannon entropy-based diversity index for positions
        let position_probs = [
            self.current_metrics.by_position.dominant as f32 / total,
            self.current_metrics.by_position.mainstream as f32 / total,
            self.current_metrics.by_position.marginalized as f32 / total,
            self.current_metrics.by_position.historically_silenced as f32 / total,
        ];

        let mut entropy = 0.0;
        for p in position_probs {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize to 0-1 (max entropy for 4 categories is ln(4))
        self.current_metrics.diversity_index = entropy / 4.0_f32.ln();
    }

    /// Check if there's potential bias detected
    pub fn detect_bias(&self) -> Option<BiasAlert> {
        // Calculate average scores by position from recent records
        let mut dominant_scores: Vec<f32> = Vec::new();
        let mut marginalized_scores: Vec<f32> = Vec::new();

        for record in &self.recent_records {
            match record.author_position {
                StructuralPosition::Dominant => dominant_scores.push(record.final_score),
                StructuralPosition::Marginalized | StructuralPosition::HistoricallySilenced => {
                    marginalized_scores.push(record.final_score)
                }
                _ => {}
            }
        }

        if dominant_scores.is_empty() || marginalized_scores.is_empty() {
            return None;
        }

        let dominant_avg = dominant_scores.iter().sum::<f32>() / dominant_scores.len() as f32;
        let marginalized_avg =
            marginalized_scores.iter().sum::<f32>() / marginalized_scores.len() as f32;

        let gap = dominant_avg - marginalized_avg;

        if gap > self.bias_threshold {
            Some(BiasAlert {
                gap,
                dominant_avg,
                marginalized_avg,
                samples_dominant: dominant_scores.len(),
                samples_marginalized: marginalized_scores.len(),
                recommendation: format!(
                    "Score gap of {:.1}% detected between dominant and marginalized voices. \
                     Consider reviewing weight configurations and reparation factors.",
                    gap * 100.0
                ),
            })
        } else {
            None
        }
    }

    /// Reset metrics for a new period
    pub fn reset_period(&mut self) {
        self.current_metrics = DiversityMetrics::default();
        // Note: recent_records are kept for rolling analysis
    }
}

/// Alert for potential epistemic bias
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BiasAlert {
    /// Score gap between groups
    pub gap: f32,
    /// Average score for dominant group
    pub dominant_avg: f32,
    /// Average score for marginalized groups
    pub marginalized_avg: f32,
    /// Number of dominant samples
    pub samples_dominant: usize,
    /// Number of marginalized samples
    pub samples_marginalized: usize,
    /// Recommended action
    pub recommendation: String,
}

// ==============================================================================
// COMPONENT 4: THE HAND - Reparations Manager
// ==============================================================================

/// Epistemic Reparations Manager
///
/// Applies power differential corrections to epistemic scores,
/// boosting marginalized and historically silenced voices.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReparationsManager {
    /// Whether reparations are enabled
    pub enabled: bool,
    /// Base adjustment for marginalized voices
    pub marginalized_boost: f32,
    /// Base adjustment for historically silenced voices
    pub silenced_boost: f32,
    /// Maximum total adjustment allowed
    pub max_adjustment: f32,
    /// Decay factor (reparations decrease as diversity improves)
    pub diversity_decay_enabled: bool,
    /// Target diversity index at which reparations reach zero
    pub target_diversity: f32,
}

impl ReparationsManager {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            enabled: true,
            marginalized_boost: 0.05,
            silenced_boost: 0.10,
            max_adjustment: 0.15,
            diversity_decay_enabled: true,
            target_diversity: 0.90, // Reparations phase out at 90% diversity index
        }
    }

    /// Create disabled reparations manager
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new()
        }
    }

    /// Calculate adjusted score based on author's structural position
    pub fn adjust(
        &self,
        raw_score: f32,
        position: StructuralPosition,
        current_diversity: f32,
    ) -> (f32, bool) {
        if !self.enabled {
            return (raw_score, false);
        }

        let base_boost = match position {
            StructuralPosition::Dominant | StructuralPosition::Mainstream => 0.0,
            StructuralPosition::Marginalized => self.marginalized_boost,
            StructuralPosition::HistoricallySilenced => self.silenced_boost,
        };

        if base_boost == 0.0 {
            return (raw_score, false);
        }

        // Apply diversity decay if enabled
        let decay_factor = if self.diversity_decay_enabled {
            // As diversity approaches target, reparations decrease
            let diversity_ratio = (current_diversity / self.target_diversity).min(1.0);
            1.0 - diversity_ratio
        } else {
            1.0
        };

        let adjustment = (base_boost * decay_factor).min(self.max_adjustment);
        let adjusted_score = (raw_score + adjustment).min(1.0);

        (adjusted_score, adjustment > 0.001)
    }
}

impl Default for ReparationsManager {
    fn default() -> Self {
        Self::new()
    }
}

// ==============================================================================
// COMPONENT 5: EMERGENT WEIGHTS - Adaptive Learning
// ==============================================================================

/// Record of an epistemic outcome for learning
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EpistemicOutcome {
    /// Original evaluation timestamp
    pub evaluation_timestamp: u64,
    /// Original score given
    pub original_score: f32,
    /// Outcome observed (did the claim prove true/useful?)
    pub outcome_type: OutcomeType,
    /// Verification attempts if applicable
    pub verification_count: u32,
    /// Time to outcome observation
    pub time_to_outcome: u64,
    /// The context used for evaluation
    pub context_used: EpistemicContext,
    /// The community that evaluated
    pub community_id: CommunityId,
}

/// Types of epistemic outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutcomeType {
    /// Claim was verified/proven true
    Verified,
    /// Claim was falsified/disproven
    Falsified,
    /// Claim remains uncertain
    Uncertain,
    /// Claim proved useful regardless of truth value
    PracticallyUseful,
    /// Claim proved harmful
    Harmful,
    /// No outcome observable
    NotApplicable,
}

/// Framework for adaptive weight learning from outcomes
///
/// Tracks how well current weights predict valuable knowledge
/// and suggests adjustments based on empirical outcomes.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmergentWeightLearner {
    /// Recorded outcomes for learning
    pub outcomes: Vec<EpistemicOutcome>,
    /// Maximum outcomes to track
    pub max_outcomes: usize,
    /// Learning rate for weight adjustments
    pub learning_rate: f32,
    /// Minimum outcomes before suggesting adjustments
    pub min_samples: usize,
}

impl EmergentWeightLearner {
    /// Create new learner
    pub fn new() -> Self {
        Self {
            outcomes: Vec::new(),
            max_outcomes: 1000,
            learning_rate: 0.01,
            min_samples: 100,
        }
    }

    /// Record an outcome
    pub fn record_outcome(&mut self, outcome: EpistemicOutcome) {
        if self.outcomes.len() >= self.max_outcomes {
            self.outcomes.remove(0);
        }
        self.outcomes.push(outcome);
    }

    /// Calculate success rate for a given context
    pub fn context_success_rate(&self, context: EpistemicContext) -> Option<f32> {
        let relevant: Vec<&EpistemicOutcome> = self
            .outcomes
            .iter()
            .filter(|o| o.context_used == context)
            .collect();

        if relevant.len() < self.min_samples {
            return None;
        }

        let successes = relevant
            .iter()
            .filter(|o| {
                matches!(
                    o.outcome_type,
                    OutcomeType::Verified | OutcomeType::PracticallyUseful
                )
            })
            .count();

        Some(successes as f32 / relevant.len() as f32)
    }

    /// Suggest weight adjustments based on outcomes
    ///
    /// Returns suggested new weights if enough data and significant pattern detected.
    pub fn suggest_adjustments(
        &self,
        current_weights: &HarmonicWeights,
    ) -> Option<WeightAdjustmentSuggestion> {
        if self.outcomes.len() < self.min_samples {
            return None;
        }

        // Calculate success rates for high-score vs low-score claims
        let high_score_outcomes: Vec<&EpistemicOutcome> =
            self.outcomes.iter().filter(|o| o.original_score > 0.7).collect();

        let low_score_outcomes: Vec<&EpistemicOutcome> =
            self.outcomes.iter().filter(|o| o.original_score < 0.3).collect();

        if high_score_outcomes.is_empty() || low_score_outcomes.is_empty() {
            return None;
        }

        let high_success = high_score_outcomes
            .iter()
            .filter(|o| {
                matches!(
                    o.outcome_type,
                    OutcomeType::Verified | OutcomeType::PracticallyUseful
                )
            })
            .count() as f32
            / high_score_outcomes.len() as f32;

        let low_success = low_score_outcomes
            .iter()
            .filter(|o| {
                matches!(
                    o.outcome_type,
                    OutcomeType::Verified | OutcomeType::PracticallyUseful
                )
            })
            .count() as f32
            / low_score_outcomes.len() as f32;

        // If high-scored claims have lower success than low-scored, weights may be inverted
        let calibration_score = high_success - low_success;

        if calibration_score < 0.1 {
            // Poorly calibrated - high scores should have much higher success rate
            Some(WeightAdjustmentSuggestion {
                reason: "High-scored claims are not performing significantly better than low-scored claims.".into(),
                calibration_score,
                high_score_success_rate: high_success,
                low_score_success_rate: low_success,
                suggested_weights: current_weights.clone(), // For now, just return current
                confidence: 0.0, // Low confidence without more sophisticated analysis
            })
        } else {
            None // System is well-calibrated
        }
    }
}

/// Suggestion for weight adjustments
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WeightAdjustmentSuggestion {
    /// Reason for suggestion
    pub reason: String,
    /// Calibration score (high_success - low_success)
    pub calibration_score: f32,
    /// Success rate for high-scored claims
    pub high_score_success_rate: f32,
    /// Success rate for low-scored claims
    pub low_score_success_rate: f32,
    /// Suggested new weights
    pub suggested_weights: HarmonicWeights,
    /// Confidence in suggestion (0.0-1.0)
    pub confidence: f32,
}

// ==============================================================================
// COMPONENT 6: MULTI-EPISTEMOLOGY SUPPORT
// ==============================================================================

/// Different epistemological frameworks supported
///
/// Each framework has different approaches to knowledge validation,
/// truth, and wisdom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Epistemology {
    /// Western Scientific - Empiricism, reproducibility, falsifiability
    WesternScientific,

    /// Indigenous Relational - Interconnectedness, place-based, oral tradition
    IndigenousRelational,

    /// Buddhist Contemplative - Direct experience, meditation, interdependence
    BuddhistContemplative,

    /// Islamic Scholarly - Revealed knowledge, scholarly consensus (ijma), analogy (qiyas)
    IslamicScholarly,

    /// African Ubuntu - Communal knowledge, "I am because we are"
    AfricanUbuntu,

    /// Hindu Darshana - Multiple valid perspectives, experiential realization
    HinduDarshana,

    /// Taoist - Harmony with nature, paradox, non-action (wu wei)
    Taoist,

    /// Pragmatist - Practical consequences, what works
    Pragmatist,

    /// Feminist Standpoint - Situated knowledge, marginalized perspectives
    FeministStandpoint,

    /// Constructivist - Knowledge as social construction
    Constructivist,

    /// Hybrid/Custom - Community-defined epistemology
    Custom,
}

impl Epistemology {
    /// Get a description of this epistemology
    pub fn description(&self) -> &'static str {
        match self {
            Self::WesternScientific => {
                "Knowledge through empirical observation, reproducibility, and falsifiability"
            }
            Self::IndigenousRelational => {
                "Knowledge through relationship with land, community, and oral tradition"
            }
            Self::BuddhistContemplative => {
                "Knowledge through direct experience, meditation, and understanding interdependence"
            }
            Self::IslamicScholarly => {
                "Knowledge through revealed text, scholarly consensus (ijma), and analogical reasoning (qiyas)"
            }
            Self::AfricanUbuntu => {
                "Knowledge through community - 'I am because we are'"
            }
            Self::HinduDarshana => {
                "Knowledge through multiple valid perspectives (darshanas) and experiential realization"
            }
            Self::Taoist => {
                "Knowledge through harmony with nature, embracing paradox, and non-action (wu wei)"
            }
            Self::Pragmatist => {
                "Knowledge validated by practical consequences - 'what works'"
            }
            Self::FeministStandpoint => {
                "Knowledge as situated, valuing marginalized perspectives"
            }
            Self::Constructivist => "Knowledge as social construction, contextual truth",
            Self::Custom => "Community-defined epistemological framework",
        }
    }

    /// Get default harmonic weights for this epistemology
    pub fn default_weights(&self) -> HarmonicWeights {
        match self {
            Self::WesternScientific => HarmonicWeights::scientific_profile(),
            Self::IndigenousRelational => HarmonicWeights::indigenous_profile(),
            Self::BuddhistContemplative => HarmonicWeights::contemplative_profile(),
            Self::AfricanUbuntu => {
                // Ubuntu: Community and reciprocity focused
                HarmonicWeights {
                    rc: 0.13,
                    psf: 0.18,
                    iw: 0.09,
                    ip: 0.09,
                    ui: 0.22, // "I am because we are"
                    sr: 0.13,
                    ep: 0.04,
                    ss: 0.12,
                }
            }
            Self::Pragmatist => {
                // Pragmatism: What works and flourishing
                HarmonicWeights {
                    rc: 0.13,
                    psf: 0.22,
                    iw: 0.18,
                    ip: 0.13,
                    ui: 0.09,
                    sr: 0.09,
                    ep: 0.04,
                    ss: 0.12,
                }
            }
            Self::FeministStandpoint => {
                // Feminist: Care and reciprocity
                HarmonicWeights {
                    rc: 0.09,
                    psf: 0.26,
                    iw: 0.09,
                    ip: 0.09,
                    ui: 0.13,
                    sr: 0.18,
                    ep: 0.04,
                    ss: 0.12,
                }
            }
            _ => HarmonicWeights::balanced(),
        }
    }

    /// Get recommended E/N/M context for this epistemology
    pub fn recommended_context(&self) -> EpistemicContext {
        match self {
            Self::WesternScientific => EpistemicContext::Scientific,
            Self::IndigenousRelational => EpistemicContext::Indigenous,
            Self::BuddhistContemplative => EpistemicContext::Contemplative,
            Self::Pragmatist => EpistemicContext::Standard,
            Self::FeministStandpoint => EpistemicContext::Governance, // Values consensus
            Self::AfricanUbuntu => EpistemicContext::Indigenous,
            _ => EpistemicContext::Personal,
        }
    }
}

impl Default for Epistemology {
    fn default() -> Self {
        Self::WesternScientific
    }
}

// ==============================================================================
// THE WISDOM ENGINE - Central Orchestrator
// ==============================================================================

/// Simplified claim structure for evaluation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Claim {
    /// Unique claim identifier
    pub id: u64,
    /// The epistemic classification
    pub classification: EpistemicClassification,
    /// Author's structural position
    pub author_position: StructuralPosition,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of evaluating a claim through the WisdomEngine
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluation {
    /// The claim that was evaluated
    pub claim_id: u64,
    /// Community that evaluated
    pub community_id: CommunityId,
    /// Raw E/N/M quality score
    pub enm_score: f32,
    /// Harmonic-weighted score (through the Lens)
    pub harmonic_score: f32,
    /// Score after reparations (through the Hand)
    pub final_score: f32,
    /// Whether reparations were applied
    pub reparations_applied: bool,
    /// The epistemology used
    pub epistemology: Epistemology,
    /// Timestamp
    pub timestamp: u64,
}

/// The WisdomEngine: Full Stack Wisdom Orchestrator
///
/// Combines all four components:
/// 1. The Lens (harmonic weights)
/// 2. The Setting (community profiles)
/// 3. The Mirror (diversity auditor)
/// 4. The Hand (reparations manager)
///
/// Plus emergent weight learning and multi-epistemology support.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WisdomEngine {
    // ===== The Lens =====
    /// Default harmonic weights (used when community not specified)
    pub default_weights: HarmonicWeights,

    // ===== The Setting =====
    /// Community profiles indexed by ID
    pub communities: HashMap<CommunityId, CommunityProfile>,

    // ===== The Mirror =====
    /// Diversity auditor for bias tracking
    pub auditor: DiversityAuditor,

    // ===== The Hand =====
    /// Reparations manager for power corrections
    pub reparations: ReparationsManager,

    // ===== Emergent Weights =====
    /// Adaptive weight learning system
    pub learner: EmergentWeightLearner,
}

impl WisdomEngine {
    /// Create a new WisdomEngine with default settings
    pub fn new() -> Self {
        Self {
            default_weights: HarmonicWeights::default_weights(),
            communities: HashMap::new(),
            auditor: DiversityAuditor::new(),
            reparations: ReparationsManager::new(),
            learner: EmergentWeightLearner::new(),
        }
    }

    /// Create with reparations disabled
    pub fn without_reparations() -> Self {
        Self {
            reparations: ReparationsManager::disabled(),
            ..Self::new()
        }
    }

    /// Register a community profile
    pub fn register_community(&mut self, profile: CommunityProfile) {
        self.communities.insert(profile.id, profile);
    }

    /// Get a community profile
    pub fn get_community(&self, id: CommunityId) -> Option<&CommunityProfile> {
        self.communities.get(&id)
    }

    /// Evaluate a claim through the full wisdom pipeline
    pub fn evaluate(&mut self, claim: &Claim, community_id: CommunityId) -> Evaluation {
        let timestamp = claim.timestamp;

        // A. Load Profile (The Setting)
        let (harmonic_weights, context, epistemology): (HarmonicWeights, EpistemicContext, Epistemology) =
            if let Some(p) = self.communities.get(&community_id) {
                (
                    p.harmonic_weights.clone(),
                    p.default_context,
                    p.epistemology,
                )
            } else {
                (
                    self.default_weights.clone(),
                    EpistemicContext::Standard,
                    Epistemology::default(),
                )
            };

        // Calculate base E/N/M score with context-adaptive weights
        let enm_score = claim.classification.quality_score_contextual(context);

        // B. Apply Harmonic Weights (The Lens)
        let harmonic_score: f32 = if let Some(ref h) = claim.classification.harmonic {
            // If claim has harmonic impact, weight it
            let h_score = harmonic_weights.score(h);
            // Blend E/N/M score with harmonic score
            enm_score * 0.7 + h_score * 0.3
        } else {
            enm_score
        };

        // C. Apply Reparations (The Hand)
        let (final_score, reparations_applied) = self.reparations.adjust(
            harmonic_score,
            claim.author_position,
            self.auditor.current_metrics.diversity_index,
        );

        // D. Log for Audit (The Mirror)
        let classification_code = claim.classification.code();
        let record = EvaluationRecord {
            timestamp,
            community_id,
            author_position: claim.author_position,
            epistemology,
            raw_score: enm_score,
            final_score,
            reparations_applied,
            classification_code: classification_code.clone(),
        };
        self.auditor.record(record);

        Evaluation {
            claim_id: claim.id,
            community_id,
            enm_score,
            harmonic_score,
            final_score,
            reparations_applied,
            epistemology,
            timestamp,
        }
    }

    /// Check for bias in recent evaluations
    pub fn check_bias(&self) -> Option<BiasAlert> {
        self.auditor.detect_bias()
    }

    /// Get current diversity metrics
    pub fn diversity_metrics(&self) -> &DiversityMetrics {
        &self.auditor.current_metrics
    }

    /// Record an outcome for learning
    pub fn record_outcome(&mut self, outcome: EpistemicOutcome) {
        self.learner.record_outcome(outcome);
    }

    /// Check if weight adjustments are suggested
    pub fn check_for_weight_adjustments(&self) -> Option<WeightAdjustmentSuggestion> {
        self.learner.suggest_adjustments(&self.default_weights)
    }

    /// Get the number of registered communities
    pub fn community_count(&self) -> usize {
        self.communities.len()
    }
}

// ==============================================================================
// COMPONENT 7: CAUSAL GRAPH - Reality Check / Causal Feedback Loops
// ==============================================================================
//
// The CausalGraph transforms Mycelix from a "Philosophy" into a "Scientific Instrument"
// that tests its own hypotheses against reality.
//
// ## How It Works
//
// 1. **Prediction Formation**: When a governance decision is made, the system records
//    what outcomes it expects based on current epistemic weights.
//
// 2. **Oracle Integration**: External oracles (sensors, audits, community reports)
//    feed back actual outcomes over time.
//
// 3. **Causal Backpropagation**: Prediction errors flow back through the causal graph
//    to adjust harmonic weights, context weights, and reparation factors.
//
// ## The Result
//
// A self-correcting civilizational operating system that:
// - Tests its own theories against reality
// - Improves its predictions over time
// - Provides empirical evidence for epistemic choices
// - Enables communities to see which knowledge frameworks actually work

/// Unique identifier for a prediction

/// Unique identifier for a causal node

/// A prediction made by the system about a future outcome
///
/// The fundamental unit of the "Reality Check" - we record what we expect
/// to happen so we can learn from what actually happens.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Prediction {
    /// Unique prediction identifier
    pub id: PredictionId,

    /// The governance decision or claim that generated this prediction
    pub source_claim_id: u64,

    /// Community that made the prediction
    pub community_id: CommunityId,

    /// What variable we're predicting (e.g., "community_health", "trust_index")
    pub target_variable: String,

    /// The predicted value (normalized 0.0-1.0)
    pub predicted_value: f32,

    /// Confidence in this prediction (0.0-1.0)
    pub confidence: f32,

    /// When the prediction was made
    pub timestamp: u64,

    /// When we expect to observe the outcome
    pub expected_observation_time: u64,

    /// The epistemic context used when making this prediction
    pub context_used: EpistemicContext,

    /// The harmonic weights active when prediction was made
    pub weights_used: HarmonicWeights,

    /// The actual observed value (filled in later by oracle)
    pub observed_value: Option<f32>,

    /// When the observation was made
    pub observation_timestamp: Option<u64>,

    /// The prediction error (predicted - observed), calculated after observation
    pub error: Option<f32>,

    /// Whether this prediction has been resolved
    pub resolved: bool,

    /// Causal chain: what variables influenced this prediction
    pub causal_parents: Vec<CausalNodeId>,
}

impl Prediction {
    /// Create a new unresolved prediction
    pub fn new(
        id: PredictionId,
        source_claim_id: u64,
        community_id: CommunityId,
        target_variable: impl Into<String>,
        predicted_value: f32,
        confidence: f32,
        timestamp: u64,
        expected_observation_time: u64,
        context_used: EpistemicContext,
        weights_used: HarmonicWeights,
    ) -> Self {
        Self {
            id,
            source_claim_id,
            community_id,
            target_variable: target_variable.into(),
            predicted_value: predicted_value.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            expected_observation_time,
            context_used,
            weights_used,
            observed_value: None,
            observation_timestamp: None,
            error: None,
            resolved: false,
            causal_parents: Vec::new(),
        }
    }

    /// Resolve this prediction with an observed value
    pub fn resolve(&mut self, observed_value: f32, observation_timestamp: u64) {
        self.observed_value = Some(observed_value.clamp(0.0, 1.0));
        self.observation_timestamp = Some(observation_timestamp);
        self.error = Some(self.predicted_value - observed_value);
        self.resolved = true;
    }

    /// Get the absolute error magnitude
    pub fn error_magnitude(&self) -> Option<f32> {
        self.error.map(|e| e.abs())
    }

    /// Check if prediction was successful (error below threshold)
    pub fn is_successful(&self, threshold: f32) -> Option<bool> {
        self.error_magnitude().map(|e| e < threshold)
    }

    /// Get the signed error (positive = over-predicted, negative = under-predicted)
    pub fn signed_error(&self) -> Option<f32> {
        self.error
    }
}

/// A node in the causal graph representing a measurable variable
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalNode {
    /// Unique node identifier
    pub id: CausalNodeId,

    /// Human-readable name
    pub name: String,

    /// Description of what this variable measures
    pub description: String,

    /// Current observed value (normalized 0.0-1.0)
    pub current_value: f32,

    /// Historical values for trend analysis
    pub history: Vec<(u64, f32)>, // (timestamp, value)

    /// Maximum history to retain
    pub max_history: usize,

    /// Parent nodes that causally influence this one
    pub parents: Vec<CausalNodeId>,

    /// Child nodes that this one causally influences
    pub children: Vec<CausalNodeId>,

    /// Influence weights from each parent (sum to 1.0)
    pub parent_weights: HashMap<CausalNodeId, f32>,

    /// Which oracle provides observations for this node
    pub oracle_source: Option<String>,

    /// Last update timestamp
    pub last_updated: u64,
}

impl CausalNode {
    /// Create a new causal node
    pub fn new(id: CausalNodeId, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: description.into(),
            current_value: 0.5, // Neutral starting point
            history: Vec::new(),
            max_history: 1000,
            parents: Vec::new(),
            children: Vec::new(),
            parent_weights: HashMap::new(),
            oracle_source: None,
            last_updated: 0,
        }
    }

    /// Update the current value and record history
    pub fn update(&mut self, value: f32, timestamp: u64) {
        // Record history
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push((timestamp, self.current_value));

        // Update current
        self.current_value = value.clamp(0.0, 1.0);
        self.last_updated = timestamp;
    }

    /// Add a parent with influence weight
    pub fn add_parent(&mut self, parent_id: CausalNodeId, weight: f32) {
        if !self.parents.contains(&parent_id) {
            self.parents.push(parent_id);
        }
        self.parent_weights.insert(parent_id, weight);
    }

    /// Get trend direction over recent history
    pub fn trend(&self, lookback: usize) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let start_idx = self.history.len().saturating_sub(lookback);
        let recent = &self.history[start_idx..];

        if recent.is_empty() {
            return 0.0;
        }

        let first_value = recent.first().map(|(_, v)| *v).unwrap_or(self.current_value);
        self.current_value - first_value
    }
}

/// Trait for oracles that provide outcome observations
///
/// Oracles are the "eyes" of the system - they observe reality and
/// report back actual outcomes so the system can learn.
pub trait Oracle: Send + Sync {
    /// Get the oracle's identifier
    fn id(&self) -> &str;

    /// Get the oracle's description
    fn description(&self) -> &str;

    /// Observe the current value of a variable
    fn observe(&self, variable: &str, timestamp: u64) -> Option<OracleObservation>;

    /// Check if this oracle can provide observations for a given variable
    fn can_observe(&self, variable: &str) -> bool;

    /// Get the oracle's trust level (0.0-1.0)
    fn trust_level(&self) -> f32;

    /// Get the oracle's verification level (maps to E-axis)
    fn verification_level(&self) -> OracleVerificationLevel;
}

/// Observation from an oracle
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OracleObservation {
    /// The variable observed
    pub variable: String,

    /// The observed value (normalized 0.0-1.0)
    pub value: f32,

    /// Confidence in this observation
    pub confidence: f32,

    /// Timestamp of observation
    pub timestamp: u64,

    /// Oracle that made the observation
    pub oracle_id: String,

    /// Any metadata about how the observation was made
    pub metadata: Option<String>,
}

/// Verification level of an oracle (maps to E-axis)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OracleVerificationLevel {
    /// Human testimony (E1)
    Testimonial,
    /// Guild audit (E2)
    Audited,
    /// Cryptographic/sensor proof (E3)
    Cryptographic,
    /// Publicly reproducible (E4)
    PubliclyReproducible,
}

impl OracleVerificationLevel {
    /// Get trust multiplier for this verification level
    pub fn trust_multiplier(&self) -> f32 {
        match self {
            Self::Testimonial => 0.6,
            Self::Audited => 0.8,
            Self::Cryptographic => 0.95,
            Self::PubliclyReproducible => 1.0,
        }
    }
}

/// A simple in-memory oracle for testing
#[derive(Debug, Clone)]
pub struct SimpleOracle {
    pub id: String,
    pub description: String,
    pub trust: f32,
    pub verification: OracleVerificationLevel,
    pub observations: HashMap<String, f32>,
}

impl SimpleOracle {
    /// Create a new simple oracle
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            trust: 0.8,
            verification: OracleVerificationLevel::Audited,
            observations: HashMap::new(),
        }
    }

    /// Set an observation value
    pub fn set_observation(&mut self, variable: impl Into<String>, value: f32) {
        self.observations.insert(variable.into(), value.clamp(0.0, 1.0));
    }
}

impl Oracle for SimpleOracle {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn observe(&self, variable: &str, timestamp: u64) -> Option<OracleObservation> {
        self.observations.get(variable).map(|&value| OracleObservation {
            variable: variable.to_string(),
            value,
            confidence: self.trust,
            timestamp,
            oracle_id: self.id.clone(),
            metadata: None,
        })
    }

    fn can_observe(&self, variable: &str) -> bool {
        self.observations.contains_key(variable)
    }

    fn trust_level(&self) -> f32 {
        self.trust
    }

    fn verification_level(&self) -> OracleVerificationLevel {
        self.verification
    }
}

/// Result of causal backpropagation - suggested weight adjustments
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalAdjustment {
    /// Which community's weights to adjust
    pub community_id: CommunityId,

    /// The prediction that drove this adjustment
    pub prediction_id: PredictionId,

    /// Suggested new harmonic weights
    pub suggested_weights: HarmonicWeights,

    /// The delta from current weights
    pub weight_deltas: [f32; 8],

    /// Confidence in this adjustment (based on prediction confidence and oracle trust)
    pub confidence: f32,

    /// Explanation of the adjustment
    pub explanation: String,

    /// The error that triggered this adjustment
    pub error_magnitude: f32,
}

/// The CausalGraph: A living model of cause and effect
///
/// This is the "Reality Check" - it tracks predictions, observes outcomes,
/// and learns what actually works.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalGraph {
    /// All causal nodes in the graph
    pub nodes: HashMap<CausalNodeId, CausalNode>,

    /// Next node ID
    next_node_id: CausalNodeId,

    /// All predictions (pending and resolved)
    pub predictions: Vec<Prediction>,

    /// Next prediction ID
    next_prediction_id: PredictionId,

    /// Learning rate for weight adjustments
    pub learning_rate: f32,

    /// Minimum confidence for acting on a prediction
    pub min_confidence: f32,

    /// Minimum error magnitude to trigger adjustment
    pub error_threshold: f32,

    /// Maximum predictions to retain
    pub max_predictions: usize,

    /// Track prediction accuracy over time
    pub accuracy_history: Vec<(u64, f32)>, // (timestamp, accuracy)

    /// Total predictions made
    pub total_predictions: u64,

    /// Successful predictions (error below threshold)
    pub successful_predictions: u64,
}

impl CausalGraph {
    /// Create a new causal graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_node_id: 1,
            predictions: Vec::new(),
            next_prediction_id: 1,
            learning_rate: 0.05,
            min_confidence: 0.5,
            error_threshold: 0.15,
            max_predictions: 10000,
            accuracy_history: Vec::new(),
            total_predictions: 0,
            successful_predictions: 0,
        }
    }

    /// Add a causal node
    pub fn add_node(&mut self, name: impl Into<String>, description: impl Into<String>) -> CausalNodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        let node = CausalNode::new(id, name, description);
        self.nodes.insert(id, node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: CausalNodeId) -> Option<&CausalNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: CausalNodeId) -> Option<&mut CausalNode> {
        self.nodes.get_mut(&id)
    }

    /// Add a causal link between nodes
    pub fn add_causal_link(&mut self, parent: CausalNodeId, child: CausalNodeId, weight: f32) {
        // Add parent to child
        if let Some(child_node) = self.nodes.get_mut(&child) {
            child_node.add_parent(parent, weight);
        }

        // Add child to parent's children list
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            if !parent_node.children.contains(&child) {
                parent_node.children.push(child);
            }
        }
    }

    /// Create a prediction
    pub fn predict(
        &mut self,
        source_claim_id: u64,
        community_id: CommunityId,
        target_variable: impl Into<String>,
        predicted_value: f32,
        confidence: f32,
        timestamp: u64,
        expected_observation_time: u64,
        context_used: EpistemicContext,
        weights_used: HarmonicWeights,
    ) -> PredictionId {
        let id = self.next_prediction_id;
        self.next_prediction_id += 1;

        let prediction = Prediction::new(
            id,
            source_claim_id,
            community_id,
            target_variable,
            predicted_value,
            confidence,
            timestamp,
            expected_observation_time,
            context_used,
            weights_used,
        );

        // Rotate old predictions
        if self.predictions.len() >= self.max_predictions {
            // Remove oldest resolved prediction
            if let Some(idx) = self.predictions.iter().position(|p| p.resolved) {
                self.predictions.remove(idx);
            } else {
                self.predictions.remove(0);
            }
        }

        self.predictions.push(prediction);
        self.total_predictions += 1;
        id
    }

    /// Resolve a prediction with an observation
    pub fn resolve_prediction(&mut self, prediction_id: PredictionId, observed_value: f32, timestamp: u64) -> Option<f32> {
        if let Some(prediction) = self.predictions.iter_mut().find(|p| p.id == prediction_id) {
            prediction.resolve(observed_value, timestamp);
            let error = prediction.error_magnitude();

            // Track accuracy
            if let Some(success) = prediction.is_successful(self.error_threshold) {
                if success {
                    self.successful_predictions += 1;
                }
            }

            // Record accuracy history
            let accuracy = self.current_accuracy();
            if self.accuracy_history.len() >= 1000 {
                self.accuracy_history.remove(0);
            }
            self.accuracy_history.push((timestamp, accuracy));

            return error;
        }
        None
    }

    /// Resolve predictions from an oracle
    pub fn resolve_from_oracle(&mut self, oracle: &dyn Oracle, timestamp: u64) -> Vec<(PredictionId, f32)> {
        let mut resolved = Vec::new();

        for prediction in &mut self.predictions {
            if prediction.resolved {
                continue;
            }

            // Check if oracle can observe this variable
            if oracle.can_observe(&prediction.target_variable) {
                if let Some(observation) = oracle.observe(&prediction.target_variable, timestamp) {
                    // Weight the observation by oracle trust and verification level
                    let trust_factor = oracle.trust_level() * oracle.verification_level().trust_multiplier();
                    let weighted_value = observation.value * trust_factor + prediction.predicted_value * (1.0 - trust_factor);

                    // Use the weighted value for high-trust oracles, raw value for lower trust
                    let final_value = if oracle.trust_level() > 0.9 {
                        observation.value
                    } else {
                        weighted_value
                    };

                    prediction.resolve(final_value, timestamp);

                    if let Some(error) = prediction.error_magnitude() {
                        resolved.push((prediction.id, error));

                        // Track accuracy
                        if error < self.error_threshold {
                            self.successful_predictions += 1;
                        }
                    }
                }
            }
        }

        // Record accuracy history
        if !resolved.is_empty() {
            let accuracy = self.current_accuracy();
            if self.accuracy_history.len() >= 1000 {
                self.accuracy_history.remove(0);
            }
            self.accuracy_history.push((timestamp, accuracy));
        }

        resolved
    }

    /// Get current prediction accuracy
    pub fn current_accuracy(&self) -> f32 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.successful_predictions as f32 / self.total_predictions as f32
    }

    /// Get pending predictions (not yet resolved)
    pub fn pending_predictions(&self) -> Vec<&Prediction> {
        self.predictions.iter().filter(|p| !p.resolved).collect()
    }

    /// Get resolved predictions
    pub fn resolved_predictions(&self) -> Vec<&Prediction> {
        self.predictions.iter().filter(|p| p.resolved).collect()
    }

    /// Calculate suggested weight adjustment based on prediction error
    ///
    /// This is the "backpropagation" - errors flow back to adjust weights.
    pub fn calculate_adjustment(&self, prediction: &Prediction) -> Option<CausalAdjustment> {
        // Only adjust for resolved predictions with significant error
        if !prediction.resolved {
            return None;
        }

        let error = prediction.error?;
        let error_magnitude = error.abs();

        if error_magnitude < self.error_threshold {
            return None; // Error too small to warrant adjustment
        }

        if prediction.confidence < self.min_confidence {
            return None; // Not confident enough in original prediction
        }

        // Calculate weight adjustments
        // Direction: if we over-predicted, reduce weights; if under-predicted, increase
        let direction = if error > 0.0 { -1.0 } else { 1.0 };
        let adjustment_magnitude = self.learning_rate * error_magnitude * prediction.confidence;

        let mut suggested = prediction.weights_used.clone();
        let mut deltas = [0.0f32; 8];

        // Adjust weights proportionally to their current values
        // Higher weights get larger adjustments
        let weights = suggested.to_array();
        for (i, w) in weights.iter().enumerate() {
            let delta = direction * adjustment_magnitude * w;
            deltas[i] = delta;
        }

        // Apply deltas
        suggested.rc += deltas[0];
        suggested.psf += deltas[1];
        suggested.iw += deltas[2];
        suggested.ip += deltas[3];
        suggested.ui += deltas[4];
        suggested.sr += deltas[5];
        suggested.ep += deltas[6];
        suggested.ss += deltas[7];

        // Clamp to valid range and normalize
        let clamp_weight = |w: f32| w.max(0.01); // Minimum 1% weight
        suggested.rc = clamp_weight(suggested.rc);
        suggested.psf = clamp_weight(suggested.psf);
        suggested.iw = clamp_weight(suggested.iw);
        suggested.ip = clamp_weight(suggested.ip);
        suggested.ui = clamp_weight(suggested.ui);
        suggested.sr = clamp_weight(suggested.sr);
        suggested.ep = clamp_weight(suggested.ep);
        suggested.ss = clamp_weight(suggested.ss);
        suggested.normalize();

        let explanation = if error > 0.0 {
            format!(
                "Over-predicted {} by {:.1}%. Reducing weights proportionally.",
                prediction.target_variable,
                error_magnitude * 100.0
            )
        } else {
            format!(
                "Under-predicted {} by {:.1}%. Increasing weights proportionally.",
                prediction.target_variable,
                error_magnitude * 100.0
            )
        };

        Some(CausalAdjustment {
            community_id: prediction.community_id,
            prediction_id: prediction.id,
            suggested_weights: suggested,
            weight_deltas: deltas,
            confidence: prediction.confidence * (1.0 - error_magnitude), // Reduce confidence on large errors
            explanation,
            error_magnitude,
        })
    }

    /// Get all suggested adjustments from resolved predictions
    pub fn get_all_adjustments(&self) -> Vec<CausalAdjustment> {
        self.predictions
            .iter()
            .filter(|p| p.resolved)
            .filter_map(|p| self.calculate_adjustment(p))
            .collect()
    }

    /// Get accuracy trend over time
    pub fn accuracy_trend(&self, lookback: usize) -> f32 {
        if self.accuracy_history.len() < 2 {
            return 0.0;
        }

        let start_idx = self.accuracy_history.len().saturating_sub(lookback);
        let recent = &self.accuracy_history[start_idx..];

        if recent.is_empty() {
            return 0.0;
        }

        let first = recent.first().map(|(_, a)| *a).unwrap_or(0.0);
        let last = recent.last().map(|(_, a)| *a).unwrap_or(0.0);
        last - first
    }
}

// ==============================================================================
// LIVING WISDOM ENGINE - WisdomEngine + CausalGraph
// ==============================================================================

/// The LivingWisdomEngine: WisdomEngine enhanced with causal feedback
///
/// This combines all components:
/// 1. The Lens (harmonic weights)
/// 2. The Setting (community profiles)
/// 3. The Mirror (diversity auditor)
/// 4. The Hand (reparations manager)
/// 5. Emergent Weights (adaptive learning)
/// 6. Multi-Epistemology Support
/// 7. **Causal Graph (Reality Check)** - NEW
///
/// The result is a self-correcting civilizational operating system.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LivingWisdomEngine {
    /// The base WisdomEngine
    pub wisdom: WisdomEngine,

    /// The causal feedback graph
    pub causal: CausalGraph,

    /// Whether to auto-generate predictions on evaluation
    pub auto_predict: bool,

    /// Whether to auto-apply adjustments when confidence is high
    pub auto_adjust: bool,

    /// Minimum confidence for auto-adjustment
    pub auto_adjust_threshold: f32,
}

impl LivingWisdomEngine {
    /// Create a new LivingWisdomEngine
    pub fn new() -> Self {
        Self {
            wisdom: WisdomEngine::new(),
            causal: CausalGraph::new(),
            auto_predict: true,
            auto_adjust: false, // Conservative default
            auto_adjust_threshold: 0.8,
        }
    }

    /// Create with auto-adjustment enabled
    pub fn with_auto_adjust() -> Self {
        Self {
            auto_adjust: true,
            ..Self::new()
        }
    }

    /// Register a community profile (delegates to WisdomEngine)
    pub fn register_community(&mut self, profile: CommunityProfile) {
        self.wisdom.register_community(profile);
    }

    /// Evaluate a claim and optionally generate a prediction
    pub fn evaluate(&mut self, claim: &Claim, community_id: CommunityId) -> EvaluationWithPrediction {
        let evaluation = self.wisdom.evaluate(claim, community_id);

        // Generate prediction if enabled
        let prediction_id = if self.auto_predict {
            // Predict that high scores correlate with positive outcomes
            let predicted_outcome = evaluation.final_score;

            // Get weights and context from community or defaults
            let (weights, context) = if let Some(p) = self.wisdom.get_community(community_id) {
                (p.harmonic_weights.clone(), p.default_context)
            } else {
                (self.wisdom.default_weights.clone(), EpistemicContext::Standard)
            };

            let prediction_id = self.causal.predict(
                claim.id,
                community_id,
                format!("claim_{}_outcome", claim.id),
                predicted_outcome,
                evaluation.final_score, // Higher scores = higher confidence
                claim.timestamp,
                claim.timestamp + 30 * 24 * 60 * 60, // 30 days out
                context,
                weights,
            );

            Some(prediction_id)
        } else {
            None
        };

        EvaluationWithPrediction {
            evaluation,
            prediction_id,
        }
    }

    /// Provide an outcome observation for a claim
    pub fn observe_outcome(&mut self, claim_id: u64, outcome_value: f32, timestamp: u64) {
        // Find predictions for this claim
        let variable = format!("claim_{}_outcome", claim_id);

        for prediction in &mut self.causal.predictions {
            if prediction.target_variable == variable && !prediction.resolved {
                prediction.resolve(outcome_value, timestamp);

                // Track accuracy
                if let Some(success) = prediction.is_successful(self.causal.error_threshold) {
                    if success {
                        self.causal.successful_predictions += 1;
                    }
                }
            }
        }

        // Check for auto-adjustment
        if self.auto_adjust {
            self.apply_high_confidence_adjustments();
        }
    }

    /// Apply adjustments with confidence above threshold
    pub fn apply_high_confidence_adjustments(&mut self) {
        let adjustments: Vec<CausalAdjustment> = self.causal.get_all_adjustments()
            .into_iter()
            .filter(|a| a.confidence >= self.auto_adjust_threshold)
            .collect();

        for adjustment in adjustments {
            // Apply to community if it exists
            if let Some(profile) = self.wisdom.communities.get_mut(&adjustment.community_id) {
                profile.harmonic_weights = adjustment.suggested_weights;
            }
        }
    }

    /// Get all pending adjustments for manual review
    pub fn pending_adjustments(&self) -> Vec<CausalAdjustment> {
        self.causal.get_all_adjustments()
    }

    /// Manually apply an adjustment
    pub fn apply_adjustment(&mut self, adjustment: &CausalAdjustment) {
        if let Some(profile) = self.wisdom.communities.get_mut(&adjustment.community_id) {
            profile.harmonic_weights = adjustment.suggested_weights.clone();
        }
    }

    /// Get current prediction accuracy
    pub fn prediction_accuracy(&self) -> f32 {
        self.causal.current_accuracy()
    }

    /// Check if the system is well-calibrated
    pub fn is_well_calibrated(&self) -> bool {
        self.causal.current_accuracy() > 0.7 && self.causal.accuracy_trend(100) >= 0.0
    }

    /// Get system health report
    pub fn health_report(&self) -> SystemHealthReport {
        SystemHealthReport {
            prediction_accuracy: self.causal.current_accuracy(),
            accuracy_trend: self.causal.accuracy_trend(100),
            total_predictions: self.causal.total_predictions,
            pending_predictions: self.causal.pending_predictions().len(),
            diversity_index: self.wisdom.auditor.current_metrics.diversity_index,
            bias_detected: self.wisdom.check_bias().is_some(),
            communities_registered: self.wisdom.community_count(),
            adjustments_pending: self.causal.get_all_adjustments().len(),
        }
    }
}

/// Evaluation result with optional prediction ID
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluationWithPrediction {
    /// The standard evaluation
    pub evaluation: Evaluation,
    /// Prediction ID if one was generated
    pub prediction_id: Option<PredictionId>,
}

/// System health report
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SystemHealthReport {
    /// Current prediction accuracy (0.0-1.0)
    pub prediction_accuracy: f32,
    /// Accuracy trend (positive = improving)
    pub accuracy_trend: f32,
    /// Total predictions made
    pub total_predictions: u64,
    /// Predictions awaiting resolution
    pub pending_predictions: usize,
    /// Diversity index (0.0-1.0)
    pub diversity_index: f32,
    /// Whether bias has been detected
    pub bias_detected: bool,
    /// Number of registered communities
    pub communities_registered: usize,
    /// Number of adjustments pending review
    pub adjustments_pending: usize,
}

impl SystemHealthReport {
    /// Get overall health status
    pub fn status(&self) -> SystemHealth {
        if self.prediction_accuracy > 0.8 && self.accuracy_trend >= 0.0 && !self.bias_detected && self.diversity_index > 0.6 {
            SystemHealth::Excellent
        } else if self.prediction_accuracy > 0.6 && self.accuracy_trend > -0.1 && self.diversity_index > 0.4 {
            SystemHealth::Good
        } else if self.prediction_accuracy > 0.4 {
            SystemHealth::NeedsAttention
        } else {
            SystemHealth::Critical
        }
    }
}

/// Overall system health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SystemHealth {
    /// System performing excellently
    Excellent,
    /// System performing well
    Good,
    /// System needs attention
    NeedsAttention,
    /// System in critical state
    Critical,
}

