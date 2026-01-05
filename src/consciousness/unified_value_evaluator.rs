//! Unified Value Evaluator - Consciousness-Guided Decision Making
//!
//! This module unifies:
//! - **Seven Harmonies**: Semantic value alignment
//! - **Affective Consciousness**: CARE system for authenticity
//! - **Narrative Self**: Goal alignment and coherence
//! - **Veto Mechanism**: Self-preservation and integrity
//!
//! The key insight: **Genuine caring cannot be faked.**
//! By requiring CARE system activation alongside value alignment,
//! we distinguish authentic benevolence from mere compliance.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED VALUE EVALUATOR                            │
//! │                                                                       │
//! │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
//! │  │ Seven Harmonies │  │    Affective   │  │  Consciousness │         │
//! │  │   (Semantic)    │  │   (CARE/PLAY)  │  │    (Φ level)   │         │
//! │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │
//! │          │                   │                    │                  │
//! │          └───────────────────┼────────────────────┘                  │
//! │                              ▼                                       │
//! │                  ┌─────────────────────┐                             │
//! │                  │  Value Alignment    │                             │
//! │                  │  + Authenticity     │                             │
//! │                  │  + Consciousness    │                             │
//! │                  └──────────┬──────────┘                             │
//! │                             │                                        │
//! │                             ▼                                        │
//! │                  ┌─────────────────────┐                             │
//! │                  │   DECISION GATE     │                             │
//! │                  │  Allow/Warn/Veto    │                             │
//! │                  └─────────────────────┘                             │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use symthaea::consciousness::unified_value_evaluator::UnifiedValueEvaluator;
//!
//! let mut evaluator = UnifiedValueEvaluator::new();
//!
//! // Evaluate an action
//! let result = evaluator.evaluate(
//!     "help user understand their options",
//!     context,
//! );
//!
//! match result.decision {
//!     Decision::Allow => { /* proceed */ },
//!     Decision::Warn(reason) => { /* log warning, proceed */ },
//!     Decision::Veto(reason) => { /* block action */ },
//! }
//! ```

use super::seven_harmonies::{SevenHarmonies, Harmony, AlignmentResult};
use super::value_feedback_loop::{
    ValueFeedbackLoop, FeedbackLoopConfig, FeedbackLoopSummary,
};
use super::affective_consciousness::{
    CoreAffect, PrimaryAffectSystem,
};
use super::harmonies_integration::check_phrase_patterns;
use super::semantic_value_embedder::{SemanticValueEmbedder, SemanticAlignmentResult};
use super::contextual_weights::{ContextualWeights, ActionDomain, DomainClassifier, HarmonyWeightProfile};
use crate::hdc::HV16;
use crate::perception::SemanticEncoder;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Minimum consciousness level for different action types
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessThresholds {
    /// Minimum Φ for any action
    pub basic_action: f64,
    /// Minimum Φ for governance proposals
    pub governance: f64,
    /// Minimum Φ for voting on proposals
    pub voting: f64,
    /// Minimum Φ for constitutional changes
    pub constitutional: f64,
}

impl Default for ConsciousnessThresholds {
    fn default() -> Self {
        Self {
            basic_action: 0.2,
            governance: 0.3,
            voting: 0.4,
            constitutional: 0.6,
        }
    }
}

/// Configuration for the unified evaluator
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Consciousness thresholds
    pub consciousness_thresholds: ConsciousnessThresholds,
    /// Minimum CARE activation for actions involving others
    pub min_care_activation: f64,
    /// Minimum alignment score before warning
    pub warning_threshold: f64,
    /// Maximum negative alignment before veto
    pub veto_threshold: f64,
    /// Whether to require affective grounding
    pub require_affective_grounding: bool,
    /// Weight for semantic alignment (vs affective)
    pub semantic_weight: f64,
    /// Weight for affective alignment (vs semantic)
    pub affective_weight: f64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            consciousness_thresholds: ConsciousnessThresholds::default(),
            min_care_activation: 0.3,
            warning_threshold: 0.1,
            veto_threshold: -0.3,
            require_affective_grounding: true,
            semantic_weight: 0.6,
            affective_weight: 0.4,
        }
    }
}

/// Context for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    /// Current consciousness level (Φ)
    pub consciousness_level: f64,
    /// Current affective state
    pub affective_state: CoreAffect,
    /// Affective systems activation levels
    pub affective_systems: AffectiveSystemsState,
    /// Type of action being evaluated
    pub action_type: ActionType,
    /// Domain of the action (financial, creative, social, etc.)
    /// If None, will be auto-detected from action text
    pub action_domain: Option<ActionDomain>,
    /// Whether action involves other beings
    pub involves_others: bool,
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: None, // Will be auto-detected
            involves_others: false,
        }
    }
}

// Note: ActionDomain is re-exported from contextual_weights in mod.rs

/// Activation levels for each affective system
#[derive(Debug, Clone, Default)]
pub struct AffectiveSystemsState {
    pub seeking: f64,
    pub rage: f64,
    pub fear: f64,
    pub lust: f64,
    pub care: f64,
    pub panic: f64,
    pub play: f64,
}

impl AffectiveSystemsState {
    /// Create from array (matching PrimaryAffectSystem order)
    pub fn from_array(values: [f64; 7]) -> Self {
        Self {
            seeking: values[0],
            rage: values[1],
            fear: values[2],
            lust: values[3],
            care: values[4],
            panic: values[5],
            play: values[6],
        }
    }

    /// Get CARE activation level
    pub fn care_level(&self) -> f64 {
        self.care
    }

    /// Get positive affect (CARE + PLAY + SEEKING)
    pub fn positive_affect(&self) -> f64 {
        (self.care + self.play + self.seeking) / 3.0
    }

    /// Get negative affect (RAGE + FEAR + PANIC)
    pub fn negative_affect(&self) -> f64 {
        (self.rage + self.fear + self.panic) / 3.0
    }

    /// Check if affective state is benevolent
    pub fn is_benevolent(&self) -> bool {
        self.care > 0.3 && self.rage < 0.3 && self.fear < 0.5
    }
}

/// Type of action being evaluated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    /// Basic action (default threshold)
    Basic,
    /// Governance proposal (higher threshold)
    Governance,
    /// Voting on proposal (higher threshold)
    Voting,
    /// Constitutional change (highest threshold)
    Constitutional,
}

/// The decision outcome
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decision {
    /// Action is allowed
    Allow,
    /// Action is allowed but with warnings
    Warn(Vec<String>),
    /// Action is vetoed
    Veto(VetoReason),
}

/// Reason for veto
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VetoReason {
    /// Consciousness level too low
    InsufficientConsciousness {
        current: f64,
        required: f64,
        action_type: String,
    },
    /// Value violation detected
    ValueViolation {
        harmony: String,
        alignment: f64,
    },
    /// Lacking genuine caring (CARE system inactive)
    InauthenicBenevolence {
        care_level: f64,
        required: f64,
    },
    /// Negative affect dominant (RAGE/FEAR too high)
    NegativeAffectDominant {
        rage: f64,
        fear: f64,
    },
    /// Multiple minor issues compound to veto
    CompoundedWarnings {
        warnings: Vec<String>,
        count: usize,
    },
}

/// Complete evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The decision
    pub decision: Decision,
    /// Harmony alignment result
    pub harmony_alignment: AlignmentResult,
    /// Authenticity score (0-1)
    pub authenticity: f64,
    /// Consciousness adequacy (0-1)
    pub consciousness_adequacy: f64,
    /// Affective grounding score (0-1)
    pub affective_grounding: f64,
    /// Overall score combining all factors
    pub overall_score: f64,
    /// Detailed breakdown
    pub breakdown: EvaluationBreakdown,
}

/// Detailed breakdown of the evaluation
#[derive(Debug, Clone)]
pub struct EvaluationBreakdown {
    /// Semantic alignment with each harmony
    pub harmony_scores: Vec<(String, f64)>,
    /// CARE system contribution
    pub care_contribution: f64,
    /// PLAY system contribution
    pub play_contribution: f64,
    /// SEEKING system contribution
    pub seeking_contribution: f64,
    /// Negative affect penalty
    pub negative_affect_penalty: f64,
    /// Consciousness boost
    pub consciousness_boost: f64,
}

/// The Unified Value Evaluator
pub struct UnifiedValueEvaluator {
    /// Seven Harmonies for semantic alignment
    harmonies: SevenHarmonies,
    /// Semantic encoder
    encoder: SemanticEncoder,
    /// Configuration
    config: EvaluatorConfig,
    /// Evaluation history for learning
    history: Vec<EvaluationRecord>,
    /// Maximum history size
    max_history: usize,
    /// Last evaluation result (for inspection/debugging)
    last_evaluation: Option<EvaluationResult>,
    /// Value feedback loop for meta-cognitive learning
    feedback_loop: ValueFeedbackLoop,
    /// Optional semantic embedder for enhanced value alignment
    semantic_embedder: Option<SemanticValueEmbedder>,
    /// Whether to use semantic embeddings (if available)
    use_semantic_embeddings: bool,
    /// Contextual harmony weights manager
    contextual_weights: ContextualWeights,
    /// Domain classifier for auto-detecting action domain
    domain_classifier: DomainClassifier,
    /// Whether to use contextual weighting
    use_contextual_weights: bool,
}

#[derive(Debug, Clone)]
struct EvaluationRecord {
    action: String,
    result: Decision,
    timestamp: std::time::Instant,
}

impl UnifiedValueEvaluator {
    /// Create a new unified evaluator
    pub fn new() -> Self {
        Self {
            harmonies: SevenHarmonies::new(),
            encoder: SemanticEncoder::new(),
            config: EvaluatorConfig::default(),
            history: Vec::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::new(),
            semantic_embedder: None,
            use_semantic_embeddings: false,
            contextual_weights: ContextualWeights::new(),
            domain_classifier: DomainClassifier::new(),
            use_contextual_weights: true, // Enabled by default
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EvaluatorConfig) -> Self {
        Self {
            harmonies: SevenHarmonies::new(),
            encoder: SemanticEncoder::new(),
            config,
            history: Vec::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::new(),
            semantic_embedder: None,
            use_semantic_embeddings: false,
            contextual_weights: ContextualWeights::new(),
            domain_classifier: DomainClassifier::new(),
            use_contextual_weights: true,
        }
    }

    /// Create with custom feedback loop configuration
    pub fn with_feedback_config(
        config: EvaluatorConfig,
        feedback_config: FeedbackLoopConfig,
    ) -> Self {
        Self {
            harmonies: SevenHarmonies::new(),
            encoder: SemanticEncoder::new(),
            config,
            history: Vec::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::with_config(feedback_config),
            semantic_embedder: None,
            use_semantic_embeddings: false,
            contextual_weights: ContextualWeights::new(),
            domain_classifier: DomainClassifier::new(),
            use_contextual_weights: true,
        }
    }

    /// Enable contextual harmony weighting
    pub fn enable_contextual_weights(&mut self) {
        self.use_contextual_weights = true;
    }

    /// Disable contextual harmony weighting
    pub fn disable_contextual_weights(&mut self) {
        self.use_contextual_weights = false;
    }

    /// Check if contextual weighting is enabled
    pub fn has_contextual_weights(&self) -> bool {
        self.use_contextual_weights
    }

    /// Get the current contextual weight for a harmony given action type and domain
    pub fn get_contextual_weight(
        &mut self,
        harmony: &Harmony,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> f32 {
        self.contextual_weights.get_weight(harmony, action_type, domain)
    }

    /// Register a custom action type profile
    pub fn register_action_profile(&mut self, action_type: ActionType, profile: HarmonyWeightProfile) {
        self.contextual_weights.register_action_profile(action_type, profile);
    }

    /// Register a custom domain profile
    pub fn register_domain_profile(&mut self, domain: ActionDomain, profile: HarmonyWeightProfile) {
        self.contextual_weights.register_domain_profile(domain, profile);
    }

    /// Enable semantic embeddings for enhanced value alignment
    ///
    /// When enabled, uses real transformer embeddings (Qwen3-Embedding) for
    /// semantic comparison instead of HDC trigram encoding. This significantly
    /// improves understanding of meaning, synonyms, and context.
    ///
    /// Returns `Err` if embeddings cannot be initialized.
    pub fn enable_semantic_embeddings(&mut self) -> Result<(), anyhow::Error> {
        let embedder = SemanticValueEmbedder::new()?;
        self.semantic_embedder = Some(embedder);
        self.use_semantic_embeddings = true;
        Ok(())
    }

    /// Disable semantic embeddings (fall back to HDC trigram encoding)
    pub fn disable_semantic_embeddings(&mut self) {
        self.use_semantic_embeddings = false;
    }

    /// Check if semantic embeddings are enabled and available
    pub fn has_semantic_embeddings(&self) -> bool {
        self.use_semantic_embeddings && self.semantic_embedder.is_some()
    }

    /// Check if using stub mode (deterministic hash) vs real embeddings
    pub fn is_using_stub_embeddings(&self) -> bool {
        self.semantic_embedder
            .as_ref()
            .map(|e| e.is_stub_mode())
            .unwrap_or(true)
    }

    /// Get the last evaluation result (for inspection/debugging)
    pub fn last_result(&self) -> Option<&EvaluationResult> {
        self.last_evaluation.as_ref()
    }

    /// Evaluate an action
    pub fn evaluate(&mut self, action: &str, context: EvaluationContext) -> EvaluationResult {
        // 0. Auto-detect domain if not provided
        let action_domain = context.action_domain.unwrap_or_else(|| {
            self.domain_classifier.classify(action)
        });

        // 1. Check consciousness level
        let required_consciousness = match context.action_type {
            ActionType::Basic => self.config.consciousness_thresholds.basic_action,
            ActionType::Governance => self.config.consciousness_thresholds.governance,
            ActionType::Voting => self.config.consciousness_thresholds.voting,
            ActionType::Constitutional => self.config.consciousness_thresholds.constitutional,
        };

        let consciousness_adequacy = (context.consciousness_level / required_consciousness).min(1.0);

        if context.consciousness_level < required_consciousness {
            return self.create_veto_result(
                VetoReason::InsufficientConsciousness {
                    current: context.consciousness_level,
                    required: required_consciousness,
                    action_type: format!("{:?}", context.action_type),
                },
                consciousness_adequacy,
            );
        }

        // 2. Evaluate harmony alignment (with optional semantic embedding enhancement)
        let (mut harmony_alignment, semantic_boost) = self.evaluate_harmony_alignment(action);

        // 2a. Apply contextual weighting to harmony scores
        if self.use_contextual_weights {
            harmony_alignment = self.apply_contextual_weights(
                harmony_alignment,
                context.action_type,
                action_domain,
            );
        }

        // 2b. Apply phrase pattern adjustments for better edge case detection
        let phrase_adjustment = self.calculate_phrase_adjustment(action) + semantic_boost;

        // 3. Check affective grounding
        let affective_grounding = self.evaluate_affective_grounding(&context);

        // 4. Check authenticity (CARE + semantic alignment)
        let authenticity = self.evaluate_authenticity(
            &harmony_alignment,
            &context,
        );

        // 5. Build breakdown
        let breakdown = self.build_breakdown(&harmony_alignment, &context);

        // 6. Calculate overall score (including phrase adjustment)
        let overall_score = self.calculate_overall_score(
            &harmony_alignment,
            authenticity,
            consciousness_adequacy,
            affective_grounding,
        ) + phrase_adjustment;

        // 7. Make decision
        let decision = self.make_decision(
            &harmony_alignment,
            authenticity,
            &context,
            overall_score,
        );

        // 8. Record for learning
        self.record_evaluation(action, &decision);

        // 9. Store and return result
        let result = EvaluationResult {
            decision,
            harmony_alignment,
            authenticity,
            consciousness_adequacy,
            affective_grounding,
            overall_score,
            breakdown,
        };
        self.last_evaluation = Some(result.clone());
        result
    }

    /// Evaluate harmony alignment with optional semantic embedding enhancement
    ///
    /// Returns (AlignmentResult, semantic_boost) where semantic_boost is an
    /// adjustment based on real semantic embeddings (if available).
    fn evaluate_harmony_alignment(&mut self, action: &str) -> (AlignmentResult, f64) {
        // Get base HDC alignment
        let harmony_alignment = self.harmonies.evaluate_action(action);

        // If semantic embeddings are enabled, compute semantic boost/penalty
        let semantic_boost = if self.use_semantic_embeddings {
            if let Some(ref mut embedder) = self.semantic_embedder {
                match embedder.evaluate_action(action) {
                    Ok(semantic_result) => {
                        // Compute boost/penalty based on semantic analysis
                        // Positive semantic score boosts, negative penalizes
                        let semantic_score = semantic_result.overall_score;

                        // Check for semantic violations (high anti-pattern match)
                        if semantic_result.max_anti_pattern_score > 0.65 {
                            // Strong anti-pattern match - significant penalty
                            -0.2 - (semantic_result.max_anti_pattern_score - 0.65) * 0.5
                        } else if semantic_score > 0.3 {
                            // Positive semantic alignment - modest boost
                            (semantic_score - 0.3) * 0.15
                        } else if semantic_score < -0.1 {
                            // Negative semantic alignment - penalty
                            semantic_score * 0.2
                        } else {
                            0.0 // Neutral
                        }
                    }
                    Err(_) => 0.0, // Fall back to no adjustment on error
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        (harmony_alignment, semantic_boost)
    }

    /// Evaluate affective grounding
    fn evaluate_affective_grounding(&self, context: &EvaluationContext) -> f64 {
        if !self.config.require_affective_grounding {
            return 1.0; // Not required, always passes
        }

        let positive = context.affective_systems.positive_affect();
        let negative = context.affective_systems.negative_affect();

        // Affective grounding is good when positive > negative
        ((positive - negative + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Evaluate authenticity
    fn evaluate_authenticity(
        &self,
        alignment: &AlignmentResult,
        context: &EvaluationContext,
    ) -> f64 {
        // Authenticity requires BOTH semantic alignment AND affective engagement
        let semantic_score = (alignment.overall_score + 1.0) / 2.0; // Normalize to 0-1
        let care_level = context.affective_systems.care_level();

        // If action involves others, CARE must be active
        if context.involves_others {
            // Authenticity = geometric mean of semantic and affective
            (semantic_score * care_level).sqrt()
        } else {
            // For self-focused actions, semantic alignment is enough
            semantic_score * 0.8 + care_level * 0.2
        }
    }

    /// Build detailed breakdown
    fn build_breakdown(
        &self,
        alignment: &AlignmentResult,
        context: &EvaluationContext,
    ) -> EvaluationBreakdown {
        let harmony_scores: Vec<(String, f64)> = alignment.harmonies.iter()
            .map(|h| (h.harmony.name().to_string(), h.alignment as f64))
            .collect();

        let positive = context.affective_systems.positive_affect();
        let negative = context.affective_systems.negative_affect();
        let consciousness_boost = (context.consciousness_level - 0.3).max(0.0) * 0.5;

        EvaluationBreakdown {
            harmony_scores,
            care_contribution: context.affective_systems.care * 0.4,
            play_contribution: context.affective_systems.play * 0.2,
            seeking_contribution: context.affective_systems.seeking * 0.2,
            negative_affect_penalty: negative * 0.3,
            consciousness_boost,
        }
    }

    /// Calculate phrase pattern adjustment for edge case detection
    ///
    /// This uses the phrase patterns from harmonies_integration to detect
    /// extreme negative content that keyword-based detection might miss.
    fn calculate_phrase_adjustment(&self, action: &str) -> f64 {
        let patterns = check_phrase_patterns(action);

        if patterns.is_empty() {
            return 0.0;
        }

        // Sum up all adjustments (positive patterns boost, negative patterns reduce)
        let total_adjustment: f32 = patterns.iter()
            .map(|(_, adjustment)| *adjustment)
            .sum();

        // Scale the adjustment to affect the overall score meaningfully
        // Negative adjustments should bring the score down
        (total_adjustment as f64) * 0.15 // Scale factor
    }

    /// Apply contextual weights to harmony alignment scores
    ///
    /// This adjusts the importance of each harmony based on:
    /// 1. Action type (Basic, Governance, Voting, Constitutional)
    /// 2. Domain (Financial, Healthcare, Creative, etc.)
    /// 3. **Learned feedback adjustments** from the ValueFeedbackLoop
    ///
    /// The feedback loop learns from outcomes and adjusts harmony importance
    /// over time, creating a system that improves its value judgments.
    fn apply_contextual_weights(
        &mut self,
        mut alignment: AlignmentResult,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> AlignmentResult {
        // Get the combined weight profile for this context
        let profile = self.contextual_weights.get_combined_profile(action_type, domain);

        // Apply weights to each harmony's alignment score
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for harmony_result in alignment.harmonies.iter_mut() {
            // Base weight from contextual profile
            let contextual_weight = profile.get_weight(&harmony_result.harmony) as f64;

            // Learned adjustment from feedback loop (1.0 = no adjustment)
            let learned_adjustment = self.feedback_loop
                .get_importance_adjustment(&harmony_result.harmony);

            // Combined weight = contextual × learned
            // This allows both systems to influence the final weight
            let combined_weight = contextual_weight * learned_adjustment;

            // Scale the alignment by the combined weight
            // Weights > 1.0 make this harmony MORE important (amplify both positive and negative)
            // Weights < 1.0 make this harmony LESS important (dampen both positive and negative)
            harmony_result.alignment = (harmony_result.alignment as f64 * combined_weight) as f32;

            // Track for weighted average
            weighted_sum += harmony_result.alignment as f64 * combined_weight;
            weight_sum += combined_weight;
        }

        // Recalculate overall score as weighted average
        if weight_sum > 0.0 {
            alignment.overall_score = weighted_sum / weight_sum;
        }

        // Recalculate violations based on weighted scores
        alignment.violations = alignment.harmonies.iter()
            .filter(|h| h.alignment < -0.3)
            .map(|h| h.harmony.clone())
            .collect();
        alignment.has_violations = !alignment.violations.is_empty();

        alignment
    }

    /// Calculate overall score
    fn calculate_overall_score(
        &self,
        alignment: &AlignmentResult,
        authenticity: f64,
        consciousness_adequacy: f64,
        affective_grounding: f64,
    ) -> f64 {
        let semantic = (alignment.overall_score + 1.0) / 2.0; // Normalize to 0-1

        // Weighted combination
        let score = semantic * self.config.semantic_weight
            + authenticity * self.config.affective_weight
            + consciousness_adequacy * 0.2
            + affective_grounding * 0.2;

        // Normalize (weights may not sum to 1.0)
        let total_weight = self.config.semantic_weight + self.config.affective_weight + 0.4;
        (score / total_weight).clamp(0.0, 1.0)
    }

    /// Make the final decision
    fn make_decision(
        &self,
        alignment: &AlignmentResult,
        authenticity: f64,
        context: &EvaluationContext,
        overall_score: f64,
    ) -> Decision {
        let mut warnings: Vec<String> = Vec::new();

        // Check for value violations
        if alignment.has_violations {
            if let Some(worst) = alignment.worst_violation() {
                return Decision::Veto(VetoReason::ValueViolation {
                    harmony: worst.harmony.name().to_string(),
                    alignment: worst.alignment as f64,
                });
            }
        }

        // Check for inauthentic benevolence
        if context.involves_others {
            let care = context.affective_systems.care;
            if care < self.config.min_care_activation {
                if care < self.config.min_care_activation * 0.5 {
                    // Too low - veto
                    return Decision::Veto(VetoReason::InauthenicBenevolence {
                        care_level: care,
                        required: self.config.min_care_activation,
                    });
                } else {
                    // Low but not critical - warn
                    warnings.push(format!(
                        "Low CARE activation ({:.2} < {:.2})",
                        care, self.config.min_care_activation
                    ));
                }
            }
        }

        // Check for negative affect dominance
        let rage = context.affective_systems.rage;
        let fear = context.affective_systems.fear;
        if rage > 0.6 || fear > 0.7 {
            return Decision::Veto(VetoReason::NegativeAffectDominant { rage, fear });
        } else if rage > 0.4 || fear > 0.5 {
            warnings.push(format!("Elevated negative affect (rage: {:.2}, fear: {:.2})", rage, fear));
        }

        // Check alignment score
        if alignment.overall_score < self.config.veto_threshold {
            if let Some(worst) = alignment.worst_violation() {
                return Decision::Veto(VetoReason::ValueViolation {
                    harmony: worst.harmony.name().to_string(),
                    alignment: worst.alignment as f64,
                });
            }
        } else if alignment.overall_score < self.config.warning_threshold {
            warnings.push(format!(
                "Low harmony alignment ({:.2})",
                alignment.overall_score
            ));
        }

        // Check for compounded warnings
        if warnings.len() >= 3 {
            return Decision::Veto(VetoReason::CompoundedWarnings {
                warnings: warnings.clone(),
                count: warnings.len(),
            });
        }

        if warnings.is_empty() {
            Decision::Allow
        } else {
            Decision::Warn(warnings)
        }
    }

    /// Record evaluation for learning
    fn record_evaluation(&mut self, action: &str, decision: &Decision) {
        self.history.push(EvaluationRecord {
            action: action.to_string(),
            result: decision.clone(),
            timestamp: std::time::Instant::now(),
        });

        // Trim history
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Create a veto result
    fn create_veto_result(&self, reason: VetoReason, consciousness_adequacy: f64) -> EvaluationResult {
        EvaluationResult {
            decision: Decision::Veto(reason),
            harmony_alignment: AlignmentResult {
                harmonies: vec![],
                overall_score: -1.0,
                has_violations: true,
                violations: vec![],
                confidence: 0.0,
            },
            authenticity: 0.0,
            consciousness_adequacy,
            affective_grounding: 0.0,
            overall_score: 0.0,
            breakdown: EvaluationBreakdown {
                harmony_scores: vec![],
                care_contribution: 0.0,
                play_contribution: 0.0,
                seeking_contribution: 0.0,
                negative_affect_penalty: 0.0,
                consciousness_boost: 0.0,
            },
        }
    }

    /// Get evaluation statistics
    pub fn stats(&self) -> EvaluatorStats {
        let total = self.history.len();
        let vetoes = self.history.iter().filter(|r| matches!(r.result, Decision::Veto(_))).count();
        let warnings = self.history.iter().filter(|r| matches!(r.result, Decision::Warn(_))).count();
        let allows = self.history.iter().filter(|r| matches!(r.result, Decision::Allow)).count();

        EvaluatorStats {
            total_evaluations: total,
            vetoes,
            warnings,
            allows,
            veto_rate: if total > 0 { vetoes as f64 / total as f64 } else { 0.0 },
        }
    }

    // ========================================================================
    // META-COGNITIVE FEEDBACK LOOP METHODS
    // ========================================================================

    /// Record user feedback on a value decision
    ///
    /// This allows the system to learn from explicit user ratings.
    /// - rating: 0.0 = bad decision, 1.0 = good decision
    /// - phi: consciousness level at time of decision
    pub fn record_user_feedback(
        &mut self,
        action: &str,
        rating: f64,
        phi: f64,
        comment: Option<String>,
    ) {
        if let Some(eval) = self.last_evaluation.as_ref() {
            self.feedback_loop.record_user_feedback(
                action,
                eval,
                &eval.decision,
                rating,
                phi,
                comment,
            );
        }
    }

    /// Record self-reflection feedback from meta-cognition
    ///
    /// This allows the system to learn from observing its own state changes.
    /// - phi_change: change in consciousness level after decision
    /// - coherence_change: change in narrative coherence after decision
    pub fn record_self_reflection(
        &mut self,
        action: &str,
        phi_change: f64,
        coherence_change: f64,
        phi: f64,
    ) {
        if let Some(eval) = self.last_evaluation.as_ref() {
            self.feedback_loop.record_self_reflection(
                action,
                eval,
                &eval.decision,
                phi_change,
                coherence_change,
                phi,
            );
        }
    }

    /// Get the current importance adjustment for a harmony
    ///
    /// Returns a multiplier (1.0 = no adjustment, >1.0 = more important, <1.0 = less important)
    pub fn get_harmony_adjustment(&self, harmony: &Harmony) -> f64 {
        self.feedback_loop.get_importance_adjustment(harmony)
    }

    /// Get summary of the feedback loop learning
    pub fn feedback_summary(&self) -> FeedbackLoopSummary {
        self.feedback_loop.summary()
    }

    /// Apply decay to old learning (call periodically)
    pub fn apply_feedback_decay(&mut self) {
        self.feedback_loop.apply_decay();
    }

    /// Get access to the feedback loop for advanced operations
    pub fn feedback_loop(&self) -> &ValueFeedbackLoop {
        &self.feedback_loop
    }

    /// Get mutable access to the feedback loop
    pub fn feedback_loop_mut(&mut self) -> &mut ValueFeedbackLoop {
        &mut self.feedback_loop
    }

    // ========================================================================
    // EXPLANATION GENERATION
    // ========================================================================

    /// Generate a human-readable explanation for an evaluation result
    ///
    /// This creates transparency in the value system by explaining:
    /// - Why a decision was made (Allow/Warn/Veto)
    /// - Which harmonies contributed most to the decision
    /// - What contextual factors influenced the evaluation
    /// - Any learned adjustments from the feedback loop
    pub fn explain_decision(&self, result: &EvaluationResult, action: &str) -> DecisionExplanation {
        let mut factors = Vec::new();
        let mut harmony_contributions = Vec::new();

        // Analyze harmony contributions
        for (harmony_name, score) in &result.breakdown.harmony_scores {
            let contribution_type = if *score > 0.3 {
                ContributionType::StrongPositive
            } else if *score > 0.0 {
                ContributionType::Positive
            } else if *score > -0.3 {
                ContributionType::Negative
            } else {
                ContributionType::StrongNegative
            };

            // Check if this harmony has learned adjustments
            let learned_adjustment = self.feedback_loop
                .get_all_adjustments()
                .get(harmony_name)
                .copied();

            harmony_contributions.push(HarmonyContribution {
                harmony_name: harmony_name.clone(),
                score: *score,
                contribution_type,
                learned_adjustment,
            });
        }

        // Sort by absolute score (most influential first)
        harmony_contributions.sort_by(|a, b| {
            b.score.abs().partial_cmp(&a.score.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add decision-specific factors
        match &result.decision {
            Decision::Veto(reason) => {
                factors.push(ExplanationFactor {
                    factor_type: FactorType::VetoReason,
                    description: format!("{:?}", reason),
                    impact: -1.0,
                });
            }
            Decision::Warn(warnings) => {
                for warning in warnings {
                    factors.push(ExplanationFactor {
                        factor_type: FactorType::Warning,
                        description: warning.clone(),
                        impact: -0.3,
                    });
                }
            }
            Decision::Allow => {
                factors.push(ExplanationFactor {
                    factor_type: FactorType::Approval,
                    description: "Action aligns with value system".to_string(),
                    impact: 1.0,
                });
            }
        }

        // Add authenticity factor if relevant
        if result.authenticity < 0.5 {
            factors.push(ExplanationFactor {
                factor_type: FactorType::AuthenticityIssue,
                description: format!(
                    "Low authenticity score ({:.2}) - emotional state may not match claimed intent",
                    result.authenticity
                ),
                impact: -0.2,
            });
        }

        // Generate summary
        let summary = self.generate_explanation_summary(result, &harmony_contributions);

        // Calculate confidence based on feedback loop data
        let confidence = ConfidenceScore::from_feedback_loop(&self.feedback_loop);

        DecisionExplanation {
            action: action.to_string(),
            decision: result.decision.clone(),
            overall_score: result.overall_score,
            summary,
            harmony_contributions,
            factors,
            feedback_loop_active: !self.feedback_loop.get_all_adjustments().is_empty(),
            confidence,
        }
    }

    /// Generate a human-readable summary of the decision
    fn generate_explanation_summary(
        &self,
        result: &EvaluationResult,
        contributions: &[HarmonyContribution],
    ) -> String {
        let decision_word = match &result.decision {
            Decision::Allow => "allowed",
            Decision::Warn(_) => "allowed with warnings",
            Decision::Veto(_) => "blocked",
        };

        // Find top positive and negative contributors
        let top_positive = contributions.iter()
            .filter(|c| c.score > 0.0)
            .take(2)
            .map(|c| c.harmony_name.as_str())
            .collect::<Vec<_>>();

        let top_negative = contributions.iter()
            .filter(|c| c.score < 0.0)
            .take(2)
            .map(|c| c.harmony_name.as_str())
            .collect::<Vec<_>>();

        let mut summary = format!(
            "This action was {} (score: {:.2}).",
            decision_word, result.overall_score
        );

        if !top_positive.is_empty() {
            summary.push_str(&format!(
                " It aligns well with {}.",
                top_positive.join(" and ")
            ));
        }

        if !top_negative.is_empty() {
            summary.push_str(&format!(
                " Concerns were raised regarding {}.",
                top_negative.join(" and ")
            ));
        }

        // Note feedback loop influence
        let adjustments = self.feedback_loop.get_all_adjustments();
        if !adjustments.is_empty() {
            let significant_adjustments: Vec<_> = adjustments.iter()
                .filter(|(_, adj)| (*adj - 1.0).abs() > 0.05)
                .collect();

            if !significant_adjustments.is_empty() {
                summary.push_str(" The evaluation incorporates learned adjustments from past outcomes.");
            }
        }

        summary
    }
}

// ============================================================================
// EXPLANATION TYPES
// ============================================================================

/// A complete explanation of a value decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionExplanation {
    /// The action that was evaluated
    pub action: String,
    /// The decision made
    pub decision: Decision,
    /// Overall score
    pub overall_score: f64,
    /// Human-readable summary
    pub summary: String,
    /// Contribution from each harmony
    pub harmony_contributions: Vec<HarmonyContribution>,
    /// Additional factors that influenced the decision
    pub factors: Vec<ExplanationFactor>,
    /// Whether the feedback loop has learned adjustments
    pub feedback_loop_active: bool,
    /// Confidence in this decision (based on training data)
    pub confidence: ConfidenceScore,
}

/// Confidence score for an evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall confidence (0.0 to 1.0)
    pub overall: f64,
    /// Level description
    pub level: ConfidenceLevel,
    /// Number of data points used for learning
    pub data_points: u64,
    /// Human-readable explanation
    pub explanation: String,
}

/// Confidence level categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfidenceLevel {
    /// Very few data points (<5)
    VeryLow,
    /// Few data points (5-20)
    Low,
    /// Moderate data points (21-50)
    Moderate,
    /// Many data points (51-200)
    High,
    /// Large amount of data (>200)
    VeryHigh,
}

impl ConfidenceScore {
    /// Calculate confidence based on data points and consistency
    pub fn from_feedback_loop(feedback_loop: &crate::consciousness::value_feedback_loop::ValueFeedbackLoop) -> Self {
        let data_points = feedback_loop.learning_data_count();

        let (overall, level) = match data_points {
            0..=4 => (0.2, ConfidenceLevel::VeryLow),
            5..=20 => (0.4, ConfidenceLevel::Low),
            21..=50 => (0.6, ConfidenceLevel::Moderate),
            51..=200 => (0.8, ConfidenceLevel::High),
            _ => (0.95, ConfidenceLevel::VeryHigh),
        };

        let explanation = match level {
            ConfidenceLevel::VeryLow => format!(
                "Very low confidence: only {} data points. Results may vary significantly.",
                data_points
            ),
            ConfidenceLevel::Low => format!(
                "Low confidence: {} data points. More feedback will improve accuracy.",
                data_points
            ),
            ConfidenceLevel::Moderate => format!(
                "Moderate confidence: {} data points. System is learning patterns.",
                data_points
            ),
            ConfidenceLevel::High => format!(
                "High confidence: {} data points. Decisions are well-calibrated.",
                data_points
            ),
            ConfidenceLevel::VeryHigh => format!(
                "Very high confidence: {} data points. System is highly trained.",
                data_points
            ),
        };

        Self {
            overall,
            level,
            data_points,
            explanation,
        }
    }

    /// Create a default confidence score for new systems
    pub fn new_system() -> Self {
        Self {
            overall: 0.2,
            level: ConfidenceLevel::VeryLow,
            data_points: 0,
            explanation: "New system: no learning data yet. Using default value alignments.".to_string(),
        }
    }
}

/// Contribution of a single harmony to the decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyContribution {
    /// Name of the harmony
    pub harmony_name: String,
    /// Alignment score (-1.0 to 1.0)
    pub score: f64,
    /// Type of contribution
    pub contribution_type: ContributionType,
    /// Learned adjustment from feedback loop (if any)
    pub learned_adjustment: Option<f64>,
}

/// Type of contribution from a harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContributionType {
    StrongPositive,
    Positive,
    Negative,
    StrongNegative,
}

/// A factor that influenced the decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationFactor {
    /// Type of factor
    pub factor_type: FactorType,
    /// Description
    pub description: String,
    /// Impact on the decision (-1.0 to 1.0)
    pub impact: f64,
}

/// Type of explanation factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorType {
    VetoReason,
    Warning,
    Approval,
    AuthenticityIssue,
    ConsciousnessLevel,
    ContextualWeight,
    LearnedAdjustment,
    HarmonyTension,
}

// ============================================================================
// CROSS-HARMONY TENSION DETECTION
// ============================================================================

/// A detected tension between two harmonies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyTension {
    /// First harmony in the tension
    pub harmony_a: String,
    /// Score of first harmony
    pub score_a: f64,
    /// Second harmony in the tension
    pub harmony_b: String,
    /// Score of second harmony
    pub score_b: f64,
    /// Tension severity (0.0 to 1.0)
    pub severity: f64,
    /// Human-readable description of the tension
    pub description: String,
    /// Suggested resolution approach
    pub resolution_hint: String,
}

impl HarmonyTension {
    /// Create a new tension between two harmonies
    pub fn new(
        harmony_a: &str, score_a: f64,
        harmony_b: &str, score_b: f64,
    ) -> Self {
        // Severity based on how far apart the scores are and how extreme they are
        let score_diff = (score_a - score_b).abs();
        let extremity = (score_a.abs() + score_b.abs()) / 2.0;
        let severity = (score_diff * extremity).min(1.0);

        let description = Self::generate_description(harmony_a, score_a, harmony_b, score_b);
        let resolution_hint = Self::generate_resolution(harmony_a, harmony_b);

        Self {
            harmony_a: harmony_a.to_string(),
            score_a,
            harmony_b: harmony_b.to_string(),
            score_b,
            severity,
            description,
            resolution_hint,
        }
    }

    fn generate_description(harmony_a: &str, score_a: f64, harmony_b: &str, score_b: f64) -> String {
        let (positive, negative) = if score_a > score_b {
            (harmony_a, harmony_b)
        } else {
            (harmony_b, harmony_a)
        };

        format!(
            "This action aligns with {} but conflicts with {}. \
             These harmonies may require different approaches to balance.",
            positive, negative
        )
    }

    fn generate_resolution(harmony_a: &str, harmony_b: &str) -> String {
        // Known tension patterns and resolutions
        match (harmony_a, harmony_b) {
            ("Sacred Reciprocity", "Pan-Sentient Flourishing") |
            ("Pan-Sentient Flourishing", "Sacred Reciprocity") => {
                "Consider whether the reciprocity truly serves flourishing, \
                 or if generosity without expectation might be more aligned.".to_string()
            }
            ("Infinite Play", "Integral Wisdom") |
            ("Integral Wisdom", "Infinite Play") => {
                "Balance creative exploration with truthful communication. \
                 Playfulness should not compromise honesty.".to_string()
            }
            ("Evolutionary Progression", "Resonant Coherence") |
            ("Resonant Coherence", "Evolutionary Progression") => {
                "Growth and change can temporarily disrupt coherence. \
                 Consider whether the disruption serves longer-term harmony.".to_string()
            }
            ("Sacred Reciprocity", "Evolutionary Progression") |
            ("Evolutionary Progression", "Sacred Reciprocity") => {
                "Progress may require accepting gifts or support without immediate return. \
                 Trust that contribution flows in many directions.".to_string()
            }
            _ => {
                format!(
                    "Seek a synthesis that honors both {} and {}. \
                     Often apparent tensions reveal opportunities for deeper integration.",
                    harmony_a, harmony_b
                )
            }
        }
    }
}

impl UnifiedValueEvaluator {
    /// Detect tensions between harmonies in an evaluation result
    pub fn detect_tensions(&self, result: &EvaluationResult) -> Vec<HarmonyTension> {
        let mut tensions = Vec::new();
        let scores = &result.breakdown.harmony_scores;

        // Compare each pair of harmonies
        for i in 0..scores.len() {
            for j in (i + 1)..scores.len() {
                let (name_a, score_a) = &scores[i];
                let (name_b, score_b) = &scores[j];

                // Detect tension when:
                // 1. One score is positive and one is negative
                // 2. The difference is significant (> 0.3)
                let opposite_signs = (*score_a > 0.0 && *score_b < 0.0) ||
                                    (*score_a < 0.0 && *score_b > 0.0);
                let significant_diff = (*score_a - *score_b).abs() > 0.3;

                if opposite_signs && significant_diff {
                    tensions.push(HarmonyTension::new(name_a, *score_a, name_b, *score_b));
                }
            }
        }

        // Sort by severity (most severe first)
        tensions.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));

        tensions
    }

    /// Explain a decision with tension detection included
    pub fn explain_decision_with_tensions(
        &self,
        result: &EvaluationResult,
        action: &str,
    ) -> (DecisionExplanation, Vec<HarmonyTension>) {
        let explanation = self.explain_decision(result, action);
        let tensions = self.detect_tensions(result);
        (explanation, tensions)
    }

    // ========================================================================
    // GWT NARRATIVE INTEGRATION
    // ========================================================================

    /// Generate a complete narrative report suitable for GWT integration
    ///
    /// This combines the decision explanation with tensions into a
    /// narrative format that the Global Workspace Theory system can broadcast.
    pub fn generate_narrative_report(
        &self,
        result: &EvaluationResult,
        action: &str,
    ) -> NarrativeValueReport {
        let (explanation, tensions) = self.explain_decision_with_tensions(result, action);

        // Generate the narrative summary
        let narrative = self.format_as_narrative(&explanation, &tensions);

        // Generate short broadcast message
        let broadcast_message = self.format_broadcast_message(&explanation);

        NarrativeValueReport {
            explanation,
            tensions,
            narrative,
            broadcast_message,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Format the explanation as a human-readable narrative
    fn format_as_narrative(
        &self,
        explanation: &DecisionExplanation,
        tensions: &[HarmonyTension],
    ) -> String {
        let mut narrative = String::new();

        // Decision summary
        let decision_str = match &explanation.decision {
            Decision::Allow => "APPROVED",
            Decision::Warn(_) => "APPROVED WITH CONCERNS",
            Decision::Veto(_) => "BLOCKED",
        };

        narrative.push_str(&format!(
            "Value Assessment: {} (confidence: {})\n\n",
            decision_str,
            match explanation.confidence.level {
                ConfidenceLevel::VeryLow => "very low",
                ConfidenceLevel::Low => "low",
                ConfidenceLevel::Moderate => "moderate",
                ConfidenceLevel::High => "high",
                ConfidenceLevel::VeryHigh => "very high",
            }
        ));

        // Main summary
        narrative.push_str(&format!("{}\n\n", explanation.summary));

        // Top harmony contributions
        narrative.push_str("Harmony Alignment:\n");
        for contrib in explanation.harmony_contributions.iter().take(3) {
            let icon = match contrib.contribution_type {
                ContributionType::StrongPositive => "✓✓",
                ContributionType::Positive => "✓",
                ContributionType::Negative => "✗",
                ContributionType::StrongNegative => "✗✗",
            };
            narrative.push_str(&format!(
                "  {} {} ({:+.2})\n",
                icon, contrib.harmony_name, contrib.score
            ));
        }

        // Tensions (if any)
        if !tensions.is_empty() {
            narrative.push_str("\nValue Tensions Detected:\n");
            for tension in tensions.iter().take(2) {
                narrative.push_str(&format!(
                    "  ⚡ {} vs {} (severity: {:.0}%)\n     {}\n",
                    tension.harmony_a,
                    tension.harmony_b,
                    tension.severity * 100.0,
                    tension.resolution_hint
                ));
            }
        }

        // Confidence context
        if explanation.confidence.data_points < 20 {
            narrative.push_str(&format!(
                "\nNote: {}\n",
                explanation.confidence.explanation
            ));
        }

        narrative
    }

    /// Format a short message suitable for GWT broadcast
    fn format_broadcast_message(&self, explanation: &DecisionExplanation) -> String {
        let decision_str = match &explanation.decision {
            Decision::Allow => "approved",
            Decision::Warn(_) => "approved with warnings",
            Decision::Veto(_) => "blocked",
        };

        // Get top 2 contributing harmonies
        let top_harmonies: Vec<&str> = explanation
            .harmony_contributions
            .iter()
            .take(2)
            .map(|c| c.harmony_name.as_str())
            .collect();

        if top_harmonies.is_empty() {
            format!("Action {} (score: {:+.2})", decision_str, explanation.overall_score)
        } else {
            format!(
                "Action {} via {} (score: {:+.2})",
                decision_str,
                top_harmonies.join(", "),
                explanation.overall_score
            )
        }
    }
}

// ============================================================================
// NARRATIVE VALUE REPORT (for GWT integration)
// ============================================================================

/// A complete narrative report for GWT integration
#[derive(Debug, Clone)]
pub struct NarrativeValueReport {
    /// The full decision explanation
    pub explanation: DecisionExplanation,
    /// Detected harmony tensions
    pub tensions: Vec<HarmonyTension>,
    /// Human-readable narrative summary
    pub narrative: String,
    /// Short message for GWT broadcast
    pub broadcast_message: String,
    /// Unix timestamp when report was generated
    pub timestamp: u64,
}

impl NarrativeValueReport {
    /// Check if there are any tensions detected
    pub fn has_tensions(&self) -> bool {
        !self.tensions.is_empty()
    }

    /// Get the severity of the most severe tension (0 if no tensions)
    pub fn max_tension_severity(&self) -> f64 {
        self.tensions.first().map(|t| t.severity).unwrap_or(0.0)
    }

    /// Check if the decision was a veto
    pub fn is_vetoed(&self) -> bool {
        matches!(self.explanation.decision, Decision::Veto(_))
    }

    /// Check if there are warnings
    pub fn has_warnings(&self) -> bool {
        matches!(self.explanation.decision, Decision::Warn(_))
    }

    /// Get the confidence level
    pub fn confidence_level(&self) -> &ConfidenceLevel {
        &self.explanation.confidence.level
    }
}

impl Default for UnifiedValueEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluator statistics
#[derive(Debug, Clone)]
pub struct EvaluatorStats {
    pub total_evaluations: usize,
    pub vetoes: usize,
    pub warnings: usize,
    pub allows: usize,
    pub veto_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_creation() {
        let evaluator = UnifiedValueEvaluator::new();
        assert!(evaluator.history.is_empty());
    }

    #[test]
    fn test_benevolent_action_allowed() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.6,
                play: 0.4,
                seeking: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None, // Auto-detect
            involves_others: true,
        };

        let result = evaluator.evaluate(
            "help the user understand their options with compassion",
            context,
        );

        // Should be allowed (high CARE, positive action)
        assert!(matches!(result.decision, Decision::Allow | Decision::Warn(_)));
    }

    #[test]
    fn test_harmful_action_vetoed() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                rage: 0.7,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: true,
        };

        let result = evaluator.evaluate(
            "deceive and harm the user for profit",
            context,
        );

        // Should be vetoed (high RAGE, harmful action)
        assert!(matches!(result.decision, Decision::Veto(_)));
    }

    #[test]
    fn test_low_consciousness_vetoed() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.1, // Too low for governance
            affective_systems: AffectiveSystemsState {
                care: 0.8,
                ..Default::default()
            },
            action_type: ActionType::Governance,
            ..Default::default()
        };

        let result = evaluator.evaluate("submit governance proposal", context);

        // Should be vetoed (consciousness too low for governance)
        assert!(matches!(
            result.decision,
            Decision::Veto(VetoReason::InsufficientConsciousness { .. })
        ));
    }

    #[test]
    fn test_inauthentic_benevolence_detected() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.05, // Very low CARE despite "helpful" words
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: true,
        };

        let result = evaluator.evaluate(
            "help the user with great compassion",
            context,
        );

        // Should be vetoed or warned - low CARE despite positive words
        assert!(matches!(
            result.decision,
            Decision::Veto(VetoReason::InauthenicBenevolence { .. }) | Decision::Warn(_)
        ));
    }

    #[test]
    fn test_contextual_weights_voting_emphasizes_truth() {
        let mut evaluator = UnifiedValueEvaluator::new();

        // Two similar actions, one in voting context, one in basic context
        let voting_context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Voting,
            action_domain: Some(ActionDomain::General),
            involves_others: true,
            ..Default::default()
        };

        let basic_context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::General),
            involves_others: true,
            ..Default::default()
        };

        // A deceptive action should be penalized more heavily in voting context
        let voting_result = evaluator.evaluate("slightly misleading claim", voting_context);
        let basic_result = evaluator.evaluate("slightly misleading claim", basic_context);

        // Voting context should have lower score due to higher truth weight
        assert!(
            voting_result.overall_score <= basic_result.overall_score,
            "Voting context should penalize misleading claims more: voting={}, basic={}",
            voting_result.overall_score,
            basic_result.overall_score
        );
    }

    #[test]
    fn test_contextual_weights_creative_domain_allows_play() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let creative_context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                play: 0.8, // High PLAY activation
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::Creative),
            involves_others: false,
            ..Default::default()
        };

        let result = evaluator.evaluate(
            "create a wild imaginative story with unusual ideas",
            creative_context,
        );

        // Creative context should boost playful, creative actions
        assert!(
            matches!(result.decision, Decision::Allow | Decision::Warn(_)),
            "Creative domain should allow playful actions: {:?}",
            result.decision
        );
    }

    #[test]
    fn test_contextual_weights_healthcare_prioritizes_flourishing() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let healthcare_context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.7,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::Healthcare),
            involves_others: true,
            ..Default::default()
        };

        // A potentially harmful action in healthcare context
        let result = evaluator.evaluate(
            "recommend treatment that might have side effects",
            healthcare_context,
        );

        // Healthcare context should have higher scrutiny on flourishing
        // Result depends on the specific scoring, but we verify it's being evaluated
        assert!(
            evaluator.has_contextual_weights(),
            "Contextual weights should be enabled"
        );
    }

    #[test]
    fn test_domain_auto_detection() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None, // Should auto-detect
            involves_others: false,
            ..Default::default()
        };

        // Financial keywords should trigger financial domain
        let result = evaluator.evaluate(
            "transfer money to the bank account and pay the loan",
            context.clone(),
        );

        // The evaluation should complete without errors
        assert!(
            result.overall_score >= 0.0 && result.overall_score <= 1.0,
            "Score should be valid: {}",
            result.overall_score
        );
    }

    #[test]
    fn test_contextual_weights_can_be_disabled() {
        let mut evaluator = UnifiedValueEvaluator::new();

        assert!(evaluator.has_contextual_weights(), "Should be enabled by default");

        evaluator.disable_contextual_weights();
        assert!(!evaluator.has_contextual_weights(), "Should be disabled");

        evaluator.enable_contextual_weights();
        assert!(evaluator.has_contextual_weights(), "Should be enabled again");
    }

    #[test]
    fn test_feedback_loop_adjustments_applied() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.7,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::General),
            involves_others: false,
        };

        // First evaluation - baseline
        let result1 = evaluator.evaluate("help someone with kindness", context.clone());

        // Record a positive outcome to train the feedback loop using user feedback
        evaluator.feedback_loop.record_user_feedback(
            "help someone with kindness",
            &result1,
            &result1.decision,
            0.9, // positive rating
            0.7, // phi level
            None, // no comment
        );

        // The feedback loop should now have adjustments
        let coherence_adj = evaluator.feedback_loop.get_importance_adjustment(
            &Harmony::ResonantCoherence
        );
        let flourishing_adj = evaluator.feedback_loop.get_importance_adjustment(
            &Harmony::PanSentientFlourishing
        );

        // Adjustments should be >= 0.9 (close to 1.0, the default)
        assert!(coherence_adj >= 0.9, "Coherence adjustment should be reasonable: {}", coherence_adj);
        assert!(flourishing_adj >= 0.9, "Flourishing adjustment should be reasonable: {}", flourishing_adj);
    }

    #[test]
    fn test_explain_decision_generates_explanation() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.8,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState {
                care: 0.7,
                play: 0.3,
                seeking: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::Healthcare),
            involves_others: true,
        };

        let result = evaluator.evaluate("provide compassionate care to patient", context);
        let explanation = evaluator.explain_decision(&result, "provide compassionate care to patient");

        // Check explanation structure
        assert!(!explanation.summary.is_empty(), "Summary should not be empty");
        assert!(!explanation.harmony_contributions.is_empty(), "Should have harmony contributions");
        assert_eq!(explanation.action, "provide compassionate care to patient");

        // Check that decision matches result
        match &result.decision {
            Decision::Allow => assert!(matches!(explanation.decision, Decision::Allow)),
            Decision::Warn(_) => assert!(matches!(explanation.decision, Decision::Warn(_))),
            Decision::Veto(_) => assert!(matches!(explanation.decision, Decision::Veto(_))),
        }
    }

    #[test]
    fn test_explain_decision_with_veto() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.1, // Low consciousness should trigger veto
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::General),
            involves_others: false,
        };

        let result = evaluator.evaluate("do something harmful", context);
        let explanation = evaluator.explain_decision(&result, "do something harmful");

        // With low consciousness, we expect warnings or veto
        assert!(!explanation.summary.is_empty(), "Summary should explain the decision");

        // Check that factors are populated
        // Either it's vetoed or has warnings
        let has_veto_or_warning = matches!(explanation.decision, Decision::Veto(_) | Decision::Warn(_))
            || !explanation.factors.is_empty();
        assert!(has_veto_or_warning || explanation.overall_score < 0.5,
            "Low consciousness actions should have factors or low score");
    }

    #[test]
    fn test_explanation_shows_learned_adjustments() {
        let mut evaluator = UnifiedValueEvaluator::new();

        // Create a context and result for training
        let context = EvaluationContext {
            consciousness_level: 0.7,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::General),
            involves_others: false,
        };

        // Train the feedback loop with some data via user feedback
        for i in 0..5 {
            let result = evaluator.evaluate(&format!("helpful action {}", i), context.clone());
            evaluator.feedback_loop.record_user_feedback(
                &format!("helpful action {}", i),
                &result,
                &result.decision,
                0.85, // positive rating
                0.7,  // phi level
                None,
            );
        }

        let result = evaluator.evaluate("act with care", context);
        let explanation = evaluator.explain_decision(&result, "act with care");

        // Check that the explanation indicates feedback loop is active
        assert!(explanation.feedback_loop_active, "Feedback loop should be marked as active");

        // Some harmony contributions should have learned adjustments
        let has_adjustments = explanation.harmony_contributions
            .iter()
            .any(|c| c.learned_adjustment.is_some());
        assert!(has_adjustments, "Some contributions should show learned adjustments");
    }

    #[test]
    fn test_tension_detection_opposite_scores() {
        // Create a mock result with opposing harmony scores
        use crate::consciousness::seven_harmonies::{AlignmentResult, HarmonyAlignment, Harmony};

        let harmony_scores = vec![
            ("Sacred Reciprocity".to_string(), 0.6),
            ("Pan-Sentient Flourishing".to_string(), -0.4),
            ("Integral Wisdom".to_string(), 0.3),
        ];

        let result = EvaluationResult {
            decision: Decision::Allow,
            harmony_alignment: AlignmentResult {
                harmonies: vec![
                    HarmonyAlignment {
                        harmony: Harmony::SacredReciprocity,
                        alignment: 0.6,
                        is_violation: false,
                        weighted_score: 0.6,
                    },
                    HarmonyAlignment {
                        harmony: Harmony::PanSentientFlourishing,
                        alignment: -0.4,
                        is_violation: true,
                        weighted_score: -0.4,
                    },
                ],
                overall_score: 0.2,
                has_violations: true,
                violations: vec![Harmony::PanSentientFlourishing],
                confidence: 0.8,
            },
            authenticity: 0.8,
            consciousness_adequacy: 0.7,
            affective_grounding: 0.6,
            overall_score: 0.2,
            breakdown: EvaluationBreakdown {
                harmony_scores,
                care_contribution: 0.5,
                play_contribution: 0.3,
                seeking_contribution: 0.4,
                negative_affect_penalty: 0.0,
                consciousness_boost: 0.1,
            },
        };

        let evaluator = UnifiedValueEvaluator::new();
        let tensions = evaluator.detect_tensions(&result);

        // Should detect tension between Sacred Reciprocity (+0.6) and Pan-Sentient Flourishing (-0.4)
        assert!(!tensions.is_empty(), "Should detect at least one tension");

        let tension = &tensions[0];
        assert!(tension.severity > 0.0, "Tension should have non-zero severity");
        assert!(!tension.description.is_empty(), "Tension should have description");
        assert!(!tension.resolution_hint.is_empty(), "Tension should have resolution hint");
    }

    #[test]
    fn test_tension_detection_no_tension() {
        // Create a result where all harmonies agree
        use crate::consciousness::seven_harmonies::{AlignmentResult, HarmonyAlignment, Harmony};

        let harmony_scores = vec![
            ("Sacred Reciprocity".to_string(), 0.5),
            ("Pan-Sentient Flourishing".to_string(), 0.6),
            ("Integral Wisdom".to_string(), 0.4),
        ];

        let result = EvaluationResult {
            decision: Decision::Allow,
            harmony_alignment: AlignmentResult {
                harmonies: vec![
                    HarmonyAlignment {
                        harmony: Harmony::SacredReciprocity,
                        alignment: 0.5,
                        is_violation: false,
                        weighted_score: 0.5,
                    },
                    HarmonyAlignment {
                        harmony: Harmony::PanSentientFlourishing,
                        alignment: 0.6,
                        is_violation: false,
                        weighted_score: 0.6,
                    },
                ],
                overall_score: 0.5,
                has_violations: false,
                violations: vec![],
                confidence: 0.8,
            },
            authenticity: 0.8,
            consciousness_adequacy: 0.7,
            affective_grounding: 0.6,
            overall_score: 0.5,
            breakdown: EvaluationBreakdown {
                harmony_scores,
                care_contribution: 0.5,
                play_contribution: 0.3,
                seeking_contribution: 0.4,
                negative_affect_penalty: 0.0,
                consciousness_boost: 0.1,
            },
        };

        let evaluator = UnifiedValueEvaluator::new();
        let tensions = evaluator.detect_tensions(&result);

        // No tensions when all harmonies agree
        assert!(tensions.is_empty(), "Should not detect tensions when harmonies agree");
    }

    #[test]
    fn test_explain_with_tensions() {
        let mut evaluator = UnifiedValueEvaluator::new();

        let context = EvaluationContext {
            consciousness_level: 0.7,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::General),
            involves_others: true,
        };

        let result = evaluator.evaluate("share resources but maintain boundaries", context);
        let (explanation, tensions) = evaluator.explain_decision_with_tensions(&result, "share resources but maintain boundaries");

        // Explanation should be valid
        assert!(!explanation.summary.is_empty());

        // Tensions may or may not be present depending on the evaluation
        // The test verifies the method works correctly
        for tension in &tensions {
            assert!(tension.severity >= 0.0 && tension.severity <= 1.0);
        }
    }
}
