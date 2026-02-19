//! FL-to-MATL Feedback Loop
//!
//! Comprehensive integration between Federated Learning outcomes and MATL K-Vector updates.
//! Maps gradient quality signals, Byzantine detection results, and round outcomes to
//! trust profile modifications.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                     FL → MATL Feedback Pipeline                                  │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │                                                                                  │
//! │  Gradient Quality     Byzantine Detection      Round Aggregation                │
//! │  Signals              Results                  Results                          │
//! │  ┌──────────────┐    ┌──────────────┐         ┌──────────────┐                 │
//! │  │ Norm valid   │    │ Attack type  │         │ Weight used  │                 │
//! │  │ Convergence  │ +  │ Penalties    │    +    │ Contribution │                 │
//! │  │ Magnitude    │    │ Scope        │         │ Included?    │                 │
//! │  └──────────────┘    └──────────────┘         └──────────────┘                 │
//! │         │                   │                        │                          │
//! │         └───────────────────┴────────────────────────┘                          │
//! │                            │                                                     │
//! │                            ▼                                                     │
//! │                   ┌──────────────────┐                                          │
//! │                   │ K-Vector Updates │                                          │
//! │                   │ k_r, k_a, k_i,   │                                          │
//! │                   │ k_p, k_coherence       │                                          │
//! │                   └──────────────────┘                                          │
//! │                            │                                                     │
//! │                            ▼                                                     │
//! │                   ┌──────────────────┐                                          │
//! │                   │ MATL Engine      │                                          │
//! │                   │ Trust Score      │                                          │
//! │                   │ Update           │                                          │
//! │                   └──────────────────┘                                          │
//! │                                                                                  │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::fl::matl_feedback::{
//!     GradientQualitySignals, MatlFeedbackComputer, FLMatlFeedback
//! };
//! use mycelix_sdk::matl::KVector;
//!
//! let mut computer = MatlFeedbackComputer::new(MatlFeedbackConfig::default());
//!
//! // Analyze gradient quality
//! let signals = GradientQualitySignals::analyze(&gradient, &config);
//!
//! // Compute feedback after round
//! let feedback = computer.compute_feedback(&round_info, &participants);
//!
//! // Apply to K-Vectors
//! for (id, delta) in feedback.kvector_deltas {
//!     participant.kvector = delta.apply(&participant.kvector);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::epistemic_fl_bridge::{
    ByzantineAttackType, ByzantineScope, EpistemicByzantineResult, EpistemicGradientUpdate,
    KVectorUpdates,
};
use crate::matl::KVector;

// ============================================================================
// Gradient Quality Signals (#185)
// ============================================================================

/// Comprehensive gradient quality signals for Byzantine detection and K-Vector updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientQualitySignals {
    /// Whether gradient norm is within valid range
    pub norm_valid: bool,
    /// Actual gradient norm
    pub gradient_norm: f64,
    /// Whether all element magnitudes are valid
    pub element_magnitude_valid: bool,
    /// Maximum element magnitude found
    pub max_element_magnitude: f64,
    /// Whether gradient dimension matches expected
    pub dimension_valid: bool,
    /// Actual dimension
    pub dimension: usize,
    /// Estimated loss improvement contribution
    pub loss_improvement_estimate: f64,
    /// Convergence quality (0.0-1.0)
    pub convergence_quality: f64,
    /// Gradient variance (sparsity indicator)
    pub gradient_variance: f64,
    /// Proportion of near-zero elements
    pub sparsity: f64,
    /// Cosine similarity to global average (if available)
    pub similarity_to_global: Option<f64>,
    /// Historical consistency with past submissions
    pub historical_consistency: f64,
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f64,
}

/// Configuration for gradient quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientQualityConfig {
    /// Minimum valid gradient norm
    pub min_gradient_norm: f64,
    /// Maximum valid gradient norm
    pub max_gradient_norm: f64,
    /// Maximum allowed element magnitude
    pub max_element_magnitude: f64,
    /// Expected gradient dimension
    pub expected_dimension: usize,
    /// Threshold for "near-zero" element
    pub sparsity_threshold: f64,
    /// Minimum similarity to global average to be considered consistent
    pub min_global_similarity: f64,
    /// Weight for norm validation in overall score
    pub norm_weight: f64,
    /// Weight for magnitude validation in overall score
    pub magnitude_weight: f64,
    /// Weight for convergence quality in overall score
    pub convergence_weight: f64,
    /// Weight for similarity in overall score
    pub similarity_weight: f64,
}

impl Default for GradientQualityConfig {
    fn default() -> Self {
        Self {
            min_gradient_norm: 0.0001,
            max_gradient_norm: 100.0,
            max_element_magnitude: 10.0,
            expected_dimension: 1000,
            sparsity_threshold: 1e-6,
            min_global_similarity: 0.3,
            norm_weight: 0.25,
            magnitude_weight: 0.25,
            convergence_weight: 0.25,
            similarity_weight: 0.25,
        }
    }
}

impl GradientQualitySignals {
    /// Analyze a gradient and compute quality signals
    pub fn analyze(gradient: &[f64], config: &GradientQualityConfig) -> Self {
        let dimension = gradient.len();

        // Compute norm
        let gradient_norm: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_valid = gradient_norm >= config.min_gradient_norm
            && gradient_norm <= config.max_gradient_norm
            && gradient_norm.is_finite();

        // Compute max element magnitude
        let max_element_magnitude = gradient
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));
        let element_magnitude_valid = max_element_magnitude <= config.max_element_magnitude
            && max_element_magnitude.is_finite();

        // Check dimension
        let dimension_valid =
            dimension == config.expected_dimension || config.expected_dimension == 0;

        // Compute variance and sparsity
        let mean = if dimension > 0 {
            gradient.iter().sum::<f64>() / dimension as f64
        } else {
            0.0
        };
        let gradient_variance = if dimension > 0 {
            gradient.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dimension as f64
        } else {
            0.0
        };

        let near_zero_count = gradient
            .iter()
            .filter(|x| x.abs() < config.sparsity_threshold)
            .count();
        let sparsity = if dimension > 0 {
            near_zero_count as f64 / dimension as f64
        } else {
            1.0
        };

        // Convergence quality estimation (based on gradient properties)
        // A good gradient should have moderate norm, low variance, and not be too sparse
        let norm_quality = if norm_valid {
            1.0 - (gradient_norm / config.max_gradient_norm).min(1.0) * 0.5
        } else {
            0.0
        };
        let variance_quality = (1.0 - gradient_variance.min(1.0)).max(0.0);
        let sparsity_quality = 1.0 - sparsity;
        let convergence_quality = (norm_quality + variance_quality + sparsity_quality) / 3.0;

        // Overall quality score
        let overall_quality = Self::compute_overall_quality(
            norm_valid,
            element_magnitude_valid,
            dimension_valid,
            convergence_quality,
            None,
            config,
        );

        Self {
            norm_valid,
            gradient_norm,
            element_magnitude_valid,
            max_element_magnitude,
            dimension_valid,
            dimension,
            loss_improvement_estimate: 0.0, // Computed externally when loss data available
            convergence_quality,
            gradient_variance,
            sparsity,
            similarity_to_global: None,
            historical_consistency: 1.0, // Default to consistent
            overall_quality,
        }
    }

    /// Analyze with comparison to global average
    pub fn analyze_with_global(
        gradient: &[f64],
        global_average: &[f64],
        config: &GradientQualityConfig,
    ) -> Self {
        let mut signals = Self::analyze(gradient, config);

        // Compute cosine similarity to global average
        if gradient.len() == global_average.len() && !gradient.is_empty() {
            let dot_product: f64 = gradient
                .iter()
                .zip(global_average.iter())
                .map(|(a, b)| a * b)
                .sum();

            let norm_a: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = global_average.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm_a > 0.0 && norm_b > 0.0 {
                let similarity = dot_product / (norm_a * norm_b);
                signals.similarity_to_global = Some(similarity.clamp(-1.0, 1.0));

                // Recompute overall quality with similarity
                signals.overall_quality = Self::compute_overall_quality(
                    signals.norm_valid,
                    signals.element_magnitude_valid,
                    signals.dimension_valid,
                    signals.convergence_quality,
                    signals.similarity_to_global,
                    config,
                );
            }
        }

        signals
    }

    fn compute_overall_quality(
        norm_valid: bool,
        magnitude_valid: bool,
        dimension_valid: bool,
        convergence_quality: f64,
        similarity_to_global: Option<f64>,
        config: &GradientQualityConfig,
    ) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        // Norm validation
        if norm_valid {
            score += config.norm_weight;
        }
        total_weight += config.norm_weight;

        // Magnitude validation
        if magnitude_valid {
            score += config.magnitude_weight;
        }
        total_weight += config.magnitude_weight;

        // Convergence quality
        score += convergence_quality * config.convergence_weight;
        total_weight += config.convergence_weight;

        // Similarity (if available)
        if let Some(sim) = similarity_to_global {
            // Convert similarity from [-1, 1] to [0, 1]
            let normalized_sim = (sim + 1.0) / 2.0;
            score += normalized_sim * config.similarity_weight;
            total_weight += config.similarity_weight;
        }

        // Dimension validity is a hard requirement
        if !dimension_valid {
            return 0.0;
        }

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }

    /// Check if gradient passes all quality checks
    pub fn passes_quality_checks(&self) -> bool {
        self.norm_valid && self.element_magnitude_valid && self.dimension_valid
    }

    /// Get quality tier for epistemic classification
    pub fn quality_tier(&self) -> QualityTier {
        if !self.passes_quality_checks() {
            return QualityTier::Invalid;
        }

        if self.overall_quality >= 0.9 {
            QualityTier::Excellent
        } else if self.overall_quality >= 0.7 {
            QualityTier::Good
        } else if self.overall_quality >= 0.5 {
            QualityTier::Acceptable
        } else if self.overall_quality >= 0.3 {
            QualityTier::Poor
        } else {
            QualityTier::Invalid
        }
    }
}

/// Quality tier for gradient classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTier {
    /// Excellent quality (>= 0.9)
    Excellent,
    /// Good quality (>= 0.7)
    Good,
    /// Acceptable quality (>= 0.5)
    Acceptable,
    /// Poor quality (>= 0.3)
    Poor,
    /// Invalid/failed quality checks
    Invalid,
}

impl QualityTier {
    /// Get K-Vector performance delta for this tier
    pub fn performance_delta(&self) -> f32 {
        match self {
            Self::Excellent => 0.02,
            Self::Good => 0.01,
            Self::Acceptable => 0.0,
            Self::Poor => -0.01,
            Self::Invalid => -0.03,
        }
    }

    /// Get K-Vector reputation delta for this tier
    pub fn reputation_delta(&self) -> f32 {
        match self {
            Self::Excellent => 0.01,
            Self::Good => 0.005,
            Self::Acceptable => 0.0,
            Self::Poor => -0.005,
            Self::Invalid => -0.02,
        }
    }
}

// ============================================================================
// K-Vector Delta Application
// ============================================================================

/// Delta to apply to a K-Vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVectorDelta {
    /// Participant ID
    pub participant_id: String,
    /// Delta for reputation (k_r)
    pub reputation_delta: f32,
    /// Delta for activity (k_a)
    pub activity_delta: f32,
    /// Delta for integrity (k_i)
    pub integrity_delta: f32,
    /// Delta for performance (k_p)
    pub performance_delta: f32,
    /// Delta for historical (k_h)
    pub historical_delta: f32,
    /// Delta for coherence/phi (k_coherence)
    pub coherence_delta: f32,
    /// Reason for this update
    pub reason: String,
    /// Whether this is a penalty (vs reward)
    pub is_penalty: bool,
}

impl KVectorDelta {
    /// Create from KVectorUpdates
    pub fn from_updates(participant_id: &str, updates: &KVectorUpdates, reason: &str) -> Self {
        let is_penalty = updates.reputation_delta < 0.0
            || updates.performance_delta < 0.0
            || updates.integrity_delta < 0.0;

        Self {
            participant_id: participant_id.to_string(),
            reputation_delta: updates.reputation_delta as f32,
            activity_delta: updates.activity_delta as f32,
            integrity_delta: updates.integrity_delta as f32,
            performance_delta: updates.performance_delta as f32,
            historical_delta: updates.historical_delta as f32,
            coherence_delta: 0.0, // Computed separately from Phi measurements
            reason: reason.to_string(),
            is_penalty,
        }
    }

    /// Create from gradient quality signals
    pub fn from_quality_signals(
        participant_id: &str,
        signals: &GradientQualitySignals,
        was_included: bool,
    ) -> Self {
        let tier = signals.quality_tier();

        Self {
            participant_id: participant_id.to_string(),
            reputation_delta: tier.reputation_delta(),
            activity_delta: if was_included { 0.01 } else { 0.005 }, // Credit for participation
            integrity_delta: if signals.passes_quality_checks() {
                0.001
            } else {
                -0.02
            },
            performance_delta: tier.performance_delta(),
            historical_delta: if signals.historical_consistency > 0.8 {
                0.002
            } else {
                0.0
            },
            coherence_delta: 0.0,
            reason: format!("Gradient quality: {:?}", tier),
            is_penalty: !signals.passes_quality_checks(),
        }
    }

    /// Create from Byzantine attack classification
    pub fn from_byzantine_attack(
        participant_id: &str,
        attack_type: &ByzantineAttackType,
        scope: &ByzantineScope,
    ) -> Self {
        // Scale penalties by attack severity and scope
        let base_reputation = match attack_type {
            ByzantineAttackType::UnverifiedAnomaly => -0.01,
            ByzantineAttackType::PeerReportedMalice => -0.03,
            ByzantineAttackType::VerifiedMalfunction => -0.05,
            ByzantineAttackType::CryptoProofFailure => -0.08,
            ByzantineAttackType::ManifestByzantine => -0.15,
        };

        let scope_multiplier = match scope {
            ByzantineScope::Individual => 1.0,
            ByzantineScope::SmallGroup => 1.5,
            ByzantineScope::NetworkCoordinated => 2.5,
            ByzantineScope::SystemicAttack => 5.0,
        };

        let reputation_delta = (base_reputation * scope_multiplier) as f32;

        Self {
            participant_id: participant_id.to_string(),
            reputation_delta,
            activity_delta: 0.0, // Still participated, just badly
            integrity_delta: (base_reputation * 1.5 * scope_multiplier) as f32,
            performance_delta: (base_reputation * 0.8 * scope_multiplier) as f32,
            historical_delta: (base_reputation * 0.5 * scope_multiplier) as f32,
            coherence_delta: match attack_type {
                ByzantineAttackType::ManifestByzantine => -0.1,
                ByzantineAttackType::CryptoProofFailure => -0.05,
                _ => 0.0,
            },
            reason: format!("Byzantine: {:?} ({:?})", attack_type, scope),
            is_penalty: true,
        }
    }

    /// Create a reward delta for successful contribution
    pub fn reward(participant_id: &str, contribution_weight: f64, reason: &str) -> Self {
        // Scale reward by contribution weight (higher weight = higher reward)
        let scaled = (contribution_weight * 0.5).min(1.0) as f32;

        Self {
            participant_id: participant_id.to_string(),
            reputation_delta: 0.005 + 0.01 * scaled,
            activity_delta: 0.01,
            integrity_delta: 0.002,
            performance_delta: 0.01 * scaled,
            historical_delta: 0.002,
            coherence_delta: 0.0,
            reason: reason.to_string(),
            is_penalty: false,
        }
    }

    /// Apply this delta to a K-Vector
    pub fn apply(&self, kvector: &KVector) -> KVector {
        KVector::new(
            (kvector.k_r + self.reputation_delta).clamp(0.0, 1.0),
            (kvector.k_a + self.activity_delta).clamp(0.0, 1.0),
            (kvector.k_i + self.integrity_delta).clamp(0.0, 1.0),
            (kvector.k_p + self.performance_delta).clamp(0.0, 1.0),
            kvector.k_m, // Membership doesn't change from FL feedback
            kvector.k_s, // Stake doesn't change from FL feedback
            (kvector.k_h + self.historical_delta).clamp(0.0, 1.0),
            kvector.k_topo, // Topology doesn't change from FL feedback
            kvector.k_v,    // Verification doesn't change from FL feedback
            (kvector.k_coherence + self.coherence_delta).clamp(0.0, 1.0),
        )
    }

    /// Merge multiple deltas for the same participant
    pub fn merge(deltas: &[KVectorDelta]) -> Option<KVectorDelta> {
        if deltas.is_empty() {
            return None;
        }

        let participant_id = deltas[0].participant_id.clone();

        // Sum all deltas
        let mut merged = KVectorDelta {
            participant_id,
            reputation_delta: 0.0,
            activity_delta: 0.0,
            integrity_delta: 0.0,
            performance_delta: 0.0,
            historical_delta: 0.0,
            coherence_delta: 0.0,
            reason: "Merged deltas".to_string(),
            is_penalty: false,
        };

        let mut reasons = Vec::new();
        for delta in deltas {
            merged.reputation_delta += delta.reputation_delta;
            merged.activity_delta += delta.activity_delta;
            merged.integrity_delta += delta.integrity_delta;
            merged.performance_delta += delta.performance_delta;
            merged.historical_delta += delta.historical_delta;
            merged.coherence_delta += delta.coherence_delta;
            if delta.is_penalty {
                merged.is_penalty = true;
            }
            reasons.push(delta.reason.clone());
        }

        merged.reason = reasons.join("; ");
        Some(merged)
    }
}

// ============================================================================
// MATL Feedback Computer
// ============================================================================

/// Configuration for MATL feedback computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlFeedbackConfig {
    /// Gradient quality analysis config
    pub quality_config: GradientQualityConfig,
    /// Maximum reputation penalty per round
    pub max_reputation_penalty: f32,
    /// Maximum reputation reward per round
    pub max_reputation_reward: f32,
    /// Whether to apply Phi-based coherence updates
    pub apply_phi_updates: bool,
    /// Minimum contribution weight to receive rewards
    pub min_reward_threshold: f64,
    /// Decay factor for historical consistency tracking
    pub historical_decay: f64,
}

impl Default for MatlFeedbackConfig {
    fn default() -> Self {
        Self {
            quality_config: GradientQualityConfig::default(),
            max_reputation_penalty: 0.2,
            max_reputation_reward: 0.05,
            apply_phi_updates: true,
            min_reward_threshold: 0.01,
            historical_decay: 0.95,
        }
    }
}

/// Result of MATL feedback computation for an FL round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLMatlFeedback {
    /// Round ID
    pub round_id: u64,
    /// K-Vector deltas for each participant
    pub kvector_deltas: HashMap<String, KVectorDelta>,
    /// Quality signals for each participant
    pub quality_signals: HashMap<String, GradientQualitySignals>,
    /// Summary statistics
    pub stats: FeedbackStats,
}

/// Summary statistics for feedback
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackStats {
    /// Total participants processed
    pub total_participants: usize,
    /// Participants receiving rewards
    pub rewarded_count: usize,
    /// Participants receiving penalties
    pub penalized_count: usize,
    /// Participants with neutral feedback
    pub neutral_count: usize,
    /// Average quality score
    pub avg_quality_score: f64,
    /// Byzantine fraction detected
    pub byzantine_fraction: f64,
}

/// Computer for FL-to-MATL feedback
pub struct MatlFeedbackComputer {
    config: MatlFeedbackConfig,
    /// Historical gradients for consistency tracking
    historical_gradients: HashMap<String, Vec<Vec<f64>>>,
    /// Maximum history to keep per participant
    max_history: usize,
}

impl MatlFeedbackComputer {
    /// Create a new feedback computer
    pub fn new(config: MatlFeedbackConfig) -> Self {
        Self {
            config,
            historical_gradients: HashMap::new(),
            max_history: 10,
        }
    }

    /// Analyze gradient quality for a participant
    pub fn analyze_gradient(
        &mut self,
        participant_id: &str,
        gradient: &[f64],
        global_average: Option<&[f64]>,
    ) -> GradientQualitySignals {
        let mut signals = if let Some(avg) = global_average {
            GradientQualitySignals::analyze_with_global(gradient, avg, &self.config.quality_config)
        } else {
            GradientQualitySignals::analyze(gradient, &self.config.quality_config)
        };

        // Check historical consistency
        if let Some(history) = self.historical_gradients.get(participant_id) {
            if !history.is_empty() {
                let mut total_similarity = 0.0;
                for past in history.iter() {
                    if past.len() == gradient.len() {
                        let dot: f64 = gradient.iter().zip(past.iter()).map(|(a, b)| a * b).sum();
                        let norm_a: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let norm_b: f64 = past.iter().map(|x| x * x).sum::<f64>().sqrt();
                        if norm_a > 0.0 && norm_b > 0.0 {
                            total_similarity += (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
                        }
                    }
                }
                signals.historical_consistency =
                    (total_similarity / history.len() as f64 + 1.0) / 2.0;
            }
        }

        // Update history
        let history = self
            .historical_gradients
            .entry(participant_id.to_string())
            .or_default();
        history.push(gradient.to_vec());
        if history.len() > self.max_history {
            history.remove(0);
        }

        signals
    }

    /// Compute feedback for an FL round
    pub fn compute_round_feedback(
        &mut self,
        round_id: u64,
        updates: &[EpistemicGradientUpdate],
        byzantine_results: &[EpistemicByzantineResult],
        participant_weights: &[(String, f64)],
        global_average: Option<&[f64]>,
    ) -> FLMatlFeedback {
        let mut kvector_deltas = HashMap::new();
        let mut quality_signals = HashMap::new();

        // Build lookup maps
        let weight_map: HashMap<_, _> = participant_weights.iter().cloned().collect();
        let byzantine_map: HashMap<_, _> = byzantine_results
            .iter()
            .map(|r| (r.participant_id.clone(), r))
            .collect();

        let mut stats = FeedbackStats::default();
        let mut total_quality = 0.0;
        let mut byzantine_count = 0;

        for update in updates {
            let pid = &update.gradient.participant_id;
            stats.total_participants += 1;

            // Analyze gradient quality
            let signals = self.analyze_gradient(pid, &update.gradient.gradients, global_average);
            total_quality += signals.overall_quality;
            quality_signals.insert(pid.clone(), signals.clone());

            // Start with quality-based delta
            let was_included = weight_map.contains_key(pid);
            let mut deltas = vec![KVectorDelta::from_quality_signals(
                pid,
                &signals,
                was_included,
            )];

            // Add Byzantine penalty if applicable
            if let Some(byzantine) = byzantine_map.get(pid) {
                byzantine_count += 1;
                deltas.push(KVectorDelta::from_byzantine_attack(
                    pid,
                    &byzantine.attack_type,
                    &byzantine.scope,
                ));
            }

            // Add contribution reward if included
            if let Some(&weight) = weight_map.get(pid) {
                if weight >= self.config.min_reward_threshold {
                    deltas.push(KVectorDelta::reward(
                        pid,
                        weight,
                        &format!("Contribution weight: {:.4}", weight),
                    ));
                }
            }

            // Merge all deltas for this participant
            if let Some(mut merged) = KVectorDelta::merge(&deltas) {
                // Apply caps
                merged.reputation_delta = merged.reputation_delta.clamp(
                    -self.config.max_reputation_penalty,
                    self.config.max_reputation_reward,
                );

                // Count by type
                if merged.is_penalty {
                    stats.penalized_count += 1;
                } else if merged.reputation_delta > 0.0 {
                    stats.rewarded_count += 1;
                } else {
                    stats.neutral_count += 1;
                }

                kvector_deltas.insert(pid.clone(), merged);
            }
        }

        stats.avg_quality_score = if stats.total_participants > 0 {
            total_quality / stats.total_participants as f64
        } else {
            0.0
        };
        stats.byzantine_fraction = if stats.total_participants > 0 {
            byzantine_count as f64 / stats.total_participants as f64
        } else {
            0.0
        };

        FLMatlFeedback {
            round_id,
            kvector_deltas,
            quality_signals,
            stats,
        }
    }

    /// Apply feedback to a map of K-Vectors
    pub fn apply_feedback(feedback: &FLMatlFeedback, kvectors: &mut HashMap<String, KVector>) {
        for (pid, delta) in &feedback.kvector_deltas {
            if let Some(kv) = kvectors.get_mut(pid) {
                *kv = delta.apply(kv);
            }
        }
    }

    /// Clear historical data for a participant (e.g., after major penalty)
    pub fn clear_history(&mut self, participant_id: &str) {
        self.historical_gradients.remove(participant_id);
    }

    /// Get current configuration
    pub fn config(&self) -> &MatlFeedbackConfig {
        &self.config
    }
}

impl Default for MatlFeedbackComputer {
    fn default() -> Self {
        Self::new(MatlFeedbackConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(size: usize, scale: f64) -> Vec<f64> {
        (0..size).map(|i| (i as f64 * 0.01).sin() * scale).collect()
    }

    #[test]
    fn test_gradient_quality_analysis() {
        let gradient = make_gradient(100, 0.5);
        let config = GradientQualityConfig {
            expected_dimension: 100,
            ..Default::default()
        };

        let signals = GradientQualitySignals::analyze(&gradient, &config);

        assert!(signals.norm_valid);
        assert!(signals.element_magnitude_valid);
        assert!(signals.dimension_valid);
        assert!(signals.passes_quality_checks());
        assert!(signals.overall_quality > 0.5);
    }

    #[test]
    fn test_gradient_quality_with_global() {
        let gradient = make_gradient(100, 0.5);
        let global_avg = make_gradient(100, 0.6);
        let config = GradientQualityConfig {
            expected_dimension: 100,
            ..Default::default()
        };

        let signals = GradientQualitySignals::analyze_with_global(&gradient, &global_avg, &config);

        assert!(signals.similarity_to_global.is_some());
        let sim = signals.similarity_to_global.unwrap();
        assert!(
            sim > 0.9,
            "Similar gradients should have high similarity: {}",
            sim
        );
    }

    #[test]
    fn test_invalid_gradient_detection() {
        let gradient = vec![0.0; 100]; // Zero gradient
        let config = GradientQualityConfig {
            expected_dimension: 100,
            ..Default::default()
        };

        let signals = GradientQualitySignals::analyze(&gradient, &config);

        assert!(!signals.norm_valid, "Zero gradient should fail norm check");
        assert!(!signals.passes_quality_checks());
        assert_eq!(signals.quality_tier(), QualityTier::Invalid);
    }

    #[test]
    fn test_kvector_delta_application() {
        let kv = KVector::new_participant();
        let delta = KVectorDelta {
            participant_id: "test".into(),
            reputation_delta: 0.05,
            activity_delta: 0.01,
            integrity_delta: 0.0,
            performance_delta: 0.02,
            historical_delta: 0.0,
            coherence_delta: 0.0,
            reason: "Test".into(),
            is_penalty: false,
        };

        let updated = delta.apply(&kv);

        assert!((updated.k_r - (kv.k_r + 0.05)).abs() < 0.001);
        assert!((updated.k_a - (kv.k_a + 0.01)).abs() < 0.001);
        assert!((updated.k_p - (kv.k_p + 0.02)).abs() < 0.001);
    }

    #[test]
    fn test_kvector_delta_clamping() {
        let kv = KVector::new(0.98, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let delta = KVectorDelta {
            participant_id: "test".into(),
            reputation_delta: 0.1, // Would exceed 1.0
            activity_delta: 0.0,
            integrity_delta: 0.0,
            performance_delta: 0.0,
            historical_delta: 0.0,
            coherence_delta: 0.0,
            reason: "Test".into(),
            is_penalty: false,
        };

        let updated = delta.apply(&kv);

        assert_eq!(updated.k_r, 1.0, "Should clamp to 1.0");
    }

    #[test]
    fn test_byzantine_attack_penalty() {
        let delta = KVectorDelta::from_byzantine_attack(
            "malicious",
            &ByzantineAttackType::ManifestByzantine,
            &ByzantineScope::NetworkCoordinated,
        );

        assert!(delta.is_penalty);
        assert!(
            delta.reputation_delta < -0.1,
            "Should have significant penalty: {}",
            delta.reputation_delta
        );
        // Integrity penalty should be more severe (more negative) than reputation
        assert!(
            delta.integrity_delta < delta.reputation_delta,
            "Integrity penalty ({}) should be more severe than reputation ({})",
            delta.integrity_delta,
            delta.reputation_delta
        );
    }

    #[test]
    fn test_delta_merge() {
        let deltas = vec![
            KVectorDelta {
                participant_id: "test".into(),
                reputation_delta: 0.05,
                activity_delta: 0.01,
                integrity_delta: 0.0,
                performance_delta: 0.02,
                historical_delta: 0.0,
                coherence_delta: 0.0,
                reason: "Quality".into(),
                is_penalty: false,
            },
            KVectorDelta {
                participant_id: "test".into(),
                reputation_delta: -0.02,
                activity_delta: 0.0,
                integrity_delta: -0.01,
                performance_delta: -0.01,
                historical_delta: 0.0,
                coherence_delta: 0.0,
                reason: "Byzantine".into(),
                is_penalty: true,
            },
        ];

        let merged = KVectorDelta::merge(&deltas).unwrap();

        assert!((merged.reputation_delta - 0.03).abs() < 0.001);
        assert!(merged.is_penalty, "Should be penalty if any penalty");
        assert!(merged.reason.contains("Quality"));
        assert!(merged.reason.contains("Byzantine"));
    }

    #[test]
    fn test_matl_feedback_computer() {
        use super::super::epistemic_fl_bridge::GradientEpistemicClassification;
        use super::super::types::GradientUpdate;
        use crate::epistemic::{EmpiricalLevel, HarmonicLevel, MaterialityLevel, NormativeLevel};

        let mut computer = MatlFeedbackComputer::default();

        // Create test updates
        let updates = vec![
            EpistemicGradientUpdate {
                gradient: GradientUpdate::new("p1".into(), 1, make_gradient(100, 0.5), 32, 0.5),
                classification: GradientEpistemicClassification {
                    empirical: EmpiricalLevel::E2PrivateVerify,
                    normative: NormativeLevel::N1Communal,
                    materiality: MaterialityLevel::M2Persistent,
                    harmonic: HarmonicLevel::H2Network,
                    confidence: 0.8,
                },
                proof_data: None,
                agent_phi: Some(0.7),
            },
            EpistemicGradientUpdate {
                gradient: GradientUpdate::new("p2".into(), 1, make_gradient(100, 0.4), 32, 0.4),
                classification: GradientEpistemicClassification {
                    empirical: EmpiricalLevel::E1Testimonial,
                    normative: NormativeLevel::N0Personal,
                    materiality: MaterialityLevel::M1Temporal,
                    harmonic: HarmonicLevel::H1Local,
                    confidence: 0.6,
                },
                proof_data: None,
                agent_phi: Some(0.5),
            },
        ];

        let participant_weights = vec![("p1".to_string(), 0.6), ("p2".to_string(), 0.4)];

        let feedback = computer.compute_round_feedback(
            1,
            &updates,
            &[], // No Byzantine
            &participant_weights,
            None,
        );

        assert_eq!(feedback.round_id, 1);
        assert_eq!(feedback.stats.total_participants, 2);
        assert_eq!(feedback.stats.byzantine_fraction, 0.0);
        assert!(feedback.kvector_deltas.contains_key("p1"));
        assert!(feedback.kvector_deltas.contains_key("p2"));
    }

    #[test]
    fn test_historical_consistency_tracking() {
        let mut computer = MatlFeedbackComputer::default();
        let pid = "consistent_participant";

        // Submit similar gradients
        for i in 0..5 {
            let gradient = make_gradient(100, 0.5 + i as f64 * 0.01);
            let signals = computer.analyze_gradient(pid, &gradient, None);

            if i > 0 {
                assert!(
                    signals.historical_consistency > 0.8,
                    "Consistent gradients should have high historical consistency"
                );
            }
        }
    }
}
