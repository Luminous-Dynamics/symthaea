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
            involves_others: false,
        }
    }
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq)]
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
        }
    }

    /// Get the last evaluation result (for inspection/debugging)
    pub fn last_result(&self) -> Option<&EvaluationResult> {
        self.last_evaluation.as_ref()
    }

    /// Evaluate an action
    pub fn evaluate(&mut self, action: &str, context: EvaluationContext) -> EvaluationResult {
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

        // 2. Evaluate harmony alignment
        let harmony_alignment = self.harmonies.evaluate_action(action);

        // 3. Check affective grounding
        let affective_grounding = self.evaluate_affective_grounding(&context);

        // 4. Check authenticity (CARE + semantic alignment)
        let authenticity = self.evaluate_authenticity(
            &harmony_alignment,
            &context,
        );

        // 5. Build breakdown
        let breakdown = self.build_breakdown(&harmony_alignment, &context);

        // 6. Calculate overall score
        let overall_score = self.calculate_overall_score(
            &harmony_alignment,
            authenticity,
            consciousness_adequacy,
            affective_grounding,
        );

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
}
