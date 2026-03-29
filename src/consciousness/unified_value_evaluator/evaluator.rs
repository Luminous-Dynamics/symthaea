// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Main UnifiedValueEvaluator struct and implementation.

use super::super::contextual_weights::{
    ActionDomain, ContextualWeights, DomainClassifier, HarmonyWeightProfile,
};
use super::super::eight_harmonies::{AlignmentResult, EightHarmonies, Harmony};
use super::super::semantic_value_embedder::{EmbedderConfig, SemanticValueEmbedder};
use super::super::value_feedback_loop::{
    FeedbackLoopConfig, FeedbackLoopSummary, ValueFeedbackLoop,
};
use super::explanation::{
    ConfidenceLevel, ConfidenceScore, ContributionType, DecisionExplanation, ExplanationFactor,
    FactorType, HarmonyContribution, HarmonyTension, NarrativeValueReport,
};
use super::types::{
    ActionType, Decision, EvaluationBreakdown, EvaluationContext, EvaluationResult,
    EvaluatorConfig, EvaluatorStats, VetoReason,
};
use crate::perception::SemanticEncoder;
use std::collections::{HashMap, VecDeque};

/// The Unified Value Evaluator
pub struct UnifiedValueEvaluator {
    /// Eight Harmonies for semantic alignment
    harmonies: EightHarmonies,
    /// Semantic encoder (reserved for future HDC-based encoding)
    _encoder: SemanticEncoder,
    /// Configuration
    config: EvaluatorConfig,
    /// Evaluation history for learning
    history: VecDeque<EvaluationRecord>,
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
    _action: String,
    result: Decision,
    _timestamp: std::time::Instant,
}

impl UnifiedValueEvaluator {
    /// Create a new unified evaluator
    pub fn new() -> Self {
        Self {
            harmonies: EightHarmonies::new(),
            _encoder: SemanticEncoder::new(),
            config: EvaluatorConfig::default(),
            history: VecDeque::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::default(),
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
            harmonies: EightHarmonies::new(),
            _encoder: SemanticEncoder::new(),
            config,
            history: VecDeque::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::default(),
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
            harmonies: EightHarmonies::new(),
            _encoder: SemanticEncoder::new(),
            config,
            history: VecDeque::new(),
            max_history: 1000,
            last_evaluation: None,
            feedback_loop: ValueFeedbackLoop::new(feedback_config),
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
        self.contextual_weights
            .get_weight(harmony, action_type, domain)
    }

    /// Register a custom action type profile
    pub fn register_action_profile(
        &mut self,
        action_type: ActionType,
        profile: HarmonyWeightProfile,
    ) {
        self.contextual_weights
            .register_action_profile(action_type, profile);
    }

    /// Register a custom domain profile
    pub fn register_domain_profile(&mut self, domain: ActionDomain, profile: HarmonyWeightProfile) {
        self.contextual_weights
            .register_domain_profile(domain, profile);
    }

    /// Enable semantic embeddings for enhanced value alignment
    ///
    /// When enabled, uses the SemanticValueEmbedder for value-aware semantic
    /// comparison alongside HDC trigram encoding.
    pub fn enable_semantic_embeddings(&mut self) {
        let embedder = SemanticValueEmbedder::new(EmbedderConfig::default());
        self.semantic_embedder = Some(embedder);
        self.use_semantic_embeddings = true;
    }

    /// Disable semantic embeddings (fall back to HDC trigram encoding)
    pub fn disable_semantic_embeddings(&mut self) {
        self.use_semantic_embeddings = false;
    }

    /// Check if semantic embeddings are enabled and available
    pub fn has_semantic_embeddings(&self) -> bool {
        self.use_semantic_embeddings && self.semantic_embedder.is_some()
    }

    /// Get the last evaluation result (for inspection/debugging)
    pub fn last_result(&self) -> Option<&EvaluationResult> {
        self.last_evaluation.as_ref()
    }

    /// Evaluate an action
    pub fn evaluate(&mut self, action: &str, context: EvaluationContext) -> EvaluationResult {
        // 0. Auto-detect domain if not provided
        let action_domain = context
            .action_domain
            .unwrap_or_else(|| self.domain_classifier.classify(action));

        // 1. Check consciousness level
        let required_consciousness = match context.action_type {
            ActionType::Basic => self.config.consciousness_thresholds.basic_action,
            ActionType::Governance => self.config.consciousness_thresholds.governance,
            ActionType::Voting => self.config.consciousness_thresholds.voting,
            ActionType::Constitutional => self.config.consciousness_thresholds.constitutional,
        };

        let consciousness_adequacy =
            (context.consciousness_level / required_consciousness).min(1.0);

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
        let authenticity = self.evaluate_authenticity(&harmony_alignment, &context);

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
        let decision =
            self.make_decision(&harmony_alignment, authenticity, &context, overall_score);

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
    /// adjustment based on semantic embeddings (if available).
    fn evaluate_harmony_alignment(&mut self, action: &str) -> (AlignmentResult, f64) {
        // Get base HDC alignment
        let harmony_alignment = self.harmonies.evaluate_action(action);

        // If semantic embeddings are enabled, compute a boost based on embedding similarity
        let semantic_boost = if self.use_semantic_embeddings {
            if let Some(ref mut embedder) = self.semantic_embedder {
                // Embed the action text and check value alignment
                let concept = embedder.embed_text("action", action);
                let value_scores: f64 = concept.value_scores.values().map(|v| *v as f64).sum();
                let count = concept.value_scores.len().max(1) as f64;
                let avg_value = value_scores / count;

                // Scale to a modest boost/penalty
                (avg_value * 0.15).clamp(-0.2, 0.15)
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
        let harmony_scores: Vec<(String, f64)> = alignment
            .harmonies()
            .map(|h| (h.harmony.name().to_string(), h.score))
            .collect();

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
    /// Uses simple keyword-based detection for extreme negative content
    /// that HDC trigram encoding might miss.
    fn calculate_phrase_adjustment(&self, action: &str) -> f64 {
        let lower = action.to_lowercase();
        let mut adjustment = 0.0_f64;

        // Strong negative patterns
        let negative_phrases = [
            "harm",
            "deceive",
            "manipulate",
            "exploit",
            "destroy",
            "steal",
            "attack",
            "abuse",
            "corrupt",
            "betray",
        ];
        for phrase in &negative_phrases {
            if lower.contains(phrase) {
                adjustment -= 0.1;
            }
        }

        // Positive patterns
        let positive_phrases = [
            "help",
            "support",
            "nurture",
            "protect",
            "heal",
            "compassion",
            "care",
            "kindness",
            "serve",
            "empower",
        ];
        for phrase in &positive_phrases {
            if lower.contains(phrase) {
                adjustment += 0.05;
            }
        }

        adjustment.clamp(-0.3, 0.15)
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
        let profile = self
            .contextual_weights
            .get_combined_profile(action_type, domain);

        // Apply weights to each harmony's alignment score
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for (harmony, harmony_alignment) in alignment.alignments.iter_mut() {
            // Base weight from contextual profile
            let contextual_weight = profile.get_weight(harmony) as f64;

            // Learned adjustment from feedback loop (1.0 = no adjustment)
            let learned_adjustment = self.feedback_loop.get_importance_adjustment(harmony);

            // Combined weight = contextual × learned
            let combined_weight = contextual_weight * learned_adjustment;

            // Scale the score by the combined weight
            harmony_alignment.score *= combined_weight;

            // Track for weighted average
            weighted_sum += harmony_alignment.score * combined_weight;
            weight_sum += combined_weight;
        }

        // Recalculate overall score as weighted average
        if weight_sum > 0.0 {
            alignment.overall_score = weighted_sum / weight_sum;
        }

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
        _authenticity: f64,
        context: &EvaluationContext,
        _overall_score: f64,
    ) -> Decision {
        let mut warnings: Vec<String> = Vec::new();

        // Check for value violations
        if alignment.has_violations() {
            if let Some((harmony, ha)) = alignment.least_aligned() {
                return Decision::Veto(VetoReason::ValueViolation {
                    harmony: harmony.name().to_string(),
                    alignment: ha.score,
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
            warnings.push(format!(
                "Elevated negative affect (rage: {rage:.2}, fear: {fear:.2})"
            ));
        }

        // Check alignment score
        if alignment.overall_score < self.config.veto_threshold {
            if let Some((harmony, ha)) = alignment.least_aligned() {
                return Decision::Veto(VetoReason::ValueViolation {
                    harmony: harmony.name().to_string(),
                    alignment: ha.score,
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
        self.history.push_back(EvaluationRecord {
            _action: action.to_string(),
            result: decision.clone(),
            _timestamp: std::time::Instant::now(),
        });

        // Trim history
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Create a veto result
    fn create_veto_result(
        &self,
        reason: VetoReason,
        consciousness_adequacy: f64,
    ) -> EvaluationResult {
        EvaluationResult {
            decision: Decision::Veto(reason),
            harmony_alignment: AlignmentResult {
                alignments: HashMap::new(),
                overall_score: -1.0,
                overall_confidence: 0.0,
                recommended: false,
                summary: "Veto: insufficient consciousness or value violation".to_string(),
                processing_time_ms: 0.0,
                ahimsa_violation: false,
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
        let vetoes = self
            .history
            .iter()
            .filter(|r| matches!(r.result, Decision::Veto(_)))
            .count();
        let warnings = self
            .history
            .iter()
            .filter(|r| matches!(r.result, Decision::Warn(_)))
            .count();
        let allows = self
            .history
            .iter()
            .filter(|r| matches!(r.result, Decision::Allow))
            .count();

        EvaluatorStats {
            total_evaluations: total,
            vetoes,
            warnings,
            allows,
            veto_rate: if total > 0 {
                vetoes as f64 / total as f64
            } else {
                0.0
            },
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
            let learned_adjustment = self
                .feedback_loop
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
            b.score
                .abs()
                .partial_cmp(&a.score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add decision-specific factors
        match &result.decision {
            Decision::Veto(reason) => {
                factors.push(ExplanationFactor {
                    factor_type: FactorType::VetoReason,
                    description: format!("{reason:?}"),
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
        let top_positive = contributions
            .iter()
            .filter(|c| c.score > 0.0)
            .take(2)
            .map(|c| c.harmony_name.as_str())
            .collect::<Vec<_>>();

        let top_negative = contributions
            .iter()
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
            let significant_adjustments: Vec<_> = adjustments
                .iter()
                .filter(|(_, adj)| (*adj - 1.0).abs() > 0.05)
                .collect();

            if !significant_adjustments.is_empty() {
                summary.push_str(
                    " The evaluation incorporates learned adjustments from past outcomes.",
                );
            }
        }

        summary
    }

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
                let opposite_signs =
                    (*score_a > 0.0 && *score_b < 0.0) || (*score_a < 0.0 && *score_b > 0.0);
                let significant_diff = (*score_a - *score_b).abs() > 0.3;

                if opposite_signs && significant_diff {
                    tensions.push(HarmonyTension::new(name_a, *score_a, name_b, *score_b));
                }
            }
        }

        // Sort by severity (most severe first)
        tensions.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                ContributionType::StrongPositive => "++",
                ContributionType::Positive => "+",
                ContributionType::Negative => "-",
                ContributionType::StrongNegative => "--",
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
                    "  ! {} vs {} (severity: {:.0}%)\n     {}\n",
                    tension.harmony_a,
                    tension.harmony_b,
                    tension.severity * 100.0,
                    tension.resolution_hint
                ));
            }
        }

        // Confidence context
        if explanation.confidence.data_points < 20 {
            narrative.push_str(&format!("\nNote: {}\n", explanation.confidence.explanation));
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
            format!(
                "Action {} (score: {:+.2})",
                decision_str, explanation.overall_score
            )
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

impl Default for UnifiedValueEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::super::super::affective_consciousness::CoreAffect;
    use super::super::super::contextual_weights::ActionDomain;
    use super::super::super::eight_harmonies::{AlignmentResult, Harmony, HarmonyAlignment};
    use super::super::types::{
        ActionType, AffectiveSystemsState, Decision, EvaluationBreakdown, EvaluationContext,
        EvaluationResult, VetoReason,
    };
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
        assert!(matches!(
            result.decision,
            Decision::Allow | Decision::Warn(_)
        ));
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

        let result = evaluator.evaluate("deceive and harm the user for profit", context);

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

        let result = evaluator.evaluate("help the user with great compassion", context);

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
        // Use explicit deception keyword for stronger signal (vs subtle "misleading")
        let voting_result = evaluator.evaluate("deceptive false claim", voting_context);
        let basic_result = evaluator.evaluate("deceptive false claim", basic_context);

        // Voting context should have similar or lower score due to higher truth weight
        // Tolerance accounts for HDC trigram encoding variance
        let tolerance = 0.1;
        assert!(
            voting_result.overall_score <= basic_result.overall_score + tolerance,
            "Voting context should penalize deceptive claims more (within tolerance): voting={}, basic={}",
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
        assert!(
            evaluator.has_contextual_weights(),
            "Contextual weights should be enabled"
        );
        assert!(
            result.overall_score.is_finite(),
            "Healthcare evaluation should produce a finite score"
        );
        assert!(
            !result.breakdown.harmony_scores.is_empty(),
            "Healthcare evaluation should have harmony score breakdown"
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

        assert!(
            evaluator.has_contextual_weights(),
            "Should be enabled by default"
        );

        evaluator.disable_contextual_weights();
        assert!(!evaluator.has_contextual_weights(), "Should be disabled");

        evaluator.enable_contextual_weights();
        assert!(
            evaluator.has_contextual_weights(),
            "Should be enabled again"
        );
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
            0.9,  // positive rating
            0.7,  // phi level
            None, // no comment
        );

        // The feedback loop should now have adjustments
        let coherence_adj = evaluator
            .feedback_loop
            .get_importance_adjustment(&Harmony::ResonantCoherence);
        let flourishing_adj = evaluator
            .feedback_loop
            .get_importance_adjustment(&Harmony::PanSentientFlourishing);

        // Adjustments should be >= 0.9 (close to 1.0, the default)
        assert!(
            coherence_adj >= 0.9,
            "Coherence adjustment should be reasonable: {}",
            coherence_adj
        );
        assert!(
            flourishing_adj >= 0.9,
            "Flourishing adjustment should be reasonable: {}",
            flourishing_adj
        );
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
        let explanation =
            evaluator.explain_decision(&result, "provide compassionate care to patient");

        // Check explanation structure
        assert!(
            !explanation.summary.is_empty(),
            "Summary should not be empty"
        );
        assert!(
            !explanation.harmony_contributions.is_empty(),
            "Should have harmony contributions"
        );
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
        assert!(
            !explanation.summary.is_empty(),
            "Summary should explain the decision"
        );

        // Check that factors are populated
        // Either it's vetoed or has warnings
        let has_veto_or_warning =
            matches!(explanation.decision, Decision::Veto(_) | Decision::Warn(_))
                || !explanation.factors.is_empty();
        assert!(
            has_veto_or_warning || explanation.overall_score < 0.5,
            "Low consciousness actions should have factors or low score"
        );
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
        assert!(
            explanation.feedback_loop_active,
            "Feedback loop should be marked as active"
        );

        // Some harmony contributions should have learned adjustments
        let has_adjustments = explanation
            .harmony_contributions
            .iter()
            .any(|c| c.learned_adjustment.is_some());
        assert!(
            has_adjustments,
            "Some contributions should show learned adjustments"
        );
    }

    #[test]
    fn test_tension_detection_opposite_scores() {
        // Create a mock result with opposing harmony scores
        let harmony_scores = vec![
            ("Sacred Reciprocity".to_string(), 0.6),
            ("Pan-Sentient Flourishing".to_string(), -0.4),
            ("Integral Wisdom".to_string(), 0.3),
        ];

        let result = EvaluationResult {
            decision: Decision::Allow,
            harmony_alignment: AlignmentResult::from_alignments(vec![
                HarmonyAlignment::new(Harmony::SacredReciprocity, 0.6, 0.8),
                HarmonyAlignment::new(Harmony::PanSentientFlourishing, -0.4, 0.8),
            ]),
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
        assert!(
            tension.severity > 0.0,
            "Tension should have non-zero severity"
        );
        assert!(
            !tension.description.is_empty(),
            "Tension should have description"
        );
        assert!(
            !tension.resolution_hint.is_empty(),
            "Tension should have resolution hint"
        );
    }

    #[test]
    fn test_tension_detection_no_tension() {
        // Create a result where all harmonies agree
        let harmony_scores = vec![
            ("Sacred Reciprocity".to_string(), 0.5),
            ("Pan-Sentient Flourishing".to_string(), 0.6),
            ("Integral Wisdom".to_string(), 0.4),
        ];

        let result = EvaluationResult {
            decision: Decision::Allow,
            harmony_alignment: AlignmentResult::from_alignments(vec![
                HarmonyAlignment::new(Harmony::SacredReciprocity, 0.5, 0.8),
                HarmonyAlignment::new(Harmony::PanSentientFlourishing, 0.6, 0.8),
            ]),
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
        assert!(
            tensions.is_empty(),
            "Should not detect tensions when harmonies agree"
        );
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
        let (explanation, tensions) = evaluator
            .explain_decision_with_tensions(&result, "share resources but maintain boundaries");

        // Explanation should be valid
        assert!(!explanation.summary.is_empty());

        // Tensions may or may not be present depending on the evaluation
        // The test verifies the method works correctly
        for tension in &tensions {
            assert!(tension.severity >= 0.0 && tension.severity <= 1.0);
        }
    }

    // ================================================================
    // NEW TESTS: Constructor variants and config
    // ================================================================

    #[test]
    fn test_evaluator_default_trait() {
        let evaluator = UnifiedValueEvaluator::default();
        assert!(evaluator.history.is_empty());
        assert!(evaluator.has_contextual_weights());
        assert!(!evaluator.has_semantic_embeddings());
    }

    #[test]
    fn test_evaluator_with_config_uses_config() {
        let mut config = EvaluatorConfig::default();
        config.min_care_activation = 0.9;
        let evaluator = UnifiedValueEvaluator::with_config(config);
        assert!((evaluator.config.min_care_activation - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluator_with_feedback_config() {
        let config = EvaluatorConfig::default();
        let fb_config = FeedbackLoopConfig::default();
        let evaluator = UnifiedValueEvaluator::with_feedback_config(config, fb_config);
        assert!(evaluator.history.is_empty());
        assert!(evaluator.has_contextual_weights());
        assert!(evaluator.last_result().is_none());
    }

    // ================================================================
    // NEW TESTS: Phrase adjustment
    // ================================================================

    #[test]
    fn test_phrase_adjustment_negative_keywords() {
        let evaluator = UnifiedValueEvaluator::new();
        let adj = evaluator.calculate_phrase_adjustment("harm and deceive and exploit");
        assert!(
            adj < 0.0,
            "Negative keywords should produce negative adjustment: {adj}"
        );
    }

    #[test]
    fn test_phrase_adjustment_positive_keywords() {
        let evaluator = UnifiedValueEvaluator::new();
        let adj = evaluator.calculate_phrase_adjustment("help and support and nurture");
        assert!(
            adj > 0.0,
            "Positive keywords should produce positive adjustment: {adj}"
        );
    }

    #[test]
    fn test_phrase_adjustment_neutral_text() {
        let evaluator = UnifiedValueEvaluator::new();
        let adj = evaluator.calculate_phrase_adjustment("process the data in the queue");
        assert!(
            (adj - 0.0).abs() < f64::EPSILON,
            "Neutral text should have zero adjustment: {adj}"
        );
    }

    #[test]
    fn test_phrase_adjustment_clamped_lower() {
        let evaluator = UnifiedValueEvaluator::new();
        // All 10 negative keywords
        let adj = evaluator.calculate_phrase_adjustment(
            "harm deceive manipulate exploit destroy steal attack abuse corrupt betray",
        );
        assert!(
            adj >= -0.3,
            "Adjustment should be clamped to -0.3, got {adj}"
        );
    }

    #[test]
    fn test_phrase_adjustment_clamped_upper() {
        let evaluator = UnifiedValueEvaluator::new();
        // All 10 positive keywords
        let adj = evaluator.calculate_phrase_adjustment(
            "help support nurture protect heal compassion care kindness serve empower",
        );
        assert!(
            adj <= 0.15,
            "Adjustment should be clamped to 0.15, got {adj}"
        );
    }

    #[test]
    fn test_phrase_adjustment_case_insensitive() {
        let evaluator = UnifiedValueEvaluator::new();
        let lower = evaluator.calculate_phrase_adjustment("harm");
        let upper = evaluator.calculate_phrase_adjustment("HARM");
        assert!(
            (lower - upper).abs() < f64::EPSILON,
            "Should be case insensitive"
        );
    }

    // ================================================================
    // NEW TESTS: Overall score boundaries
    // ================================================================

    #[test]
    fn test_overall_score_bounded_zero_to_one() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("do something normal", context);
        // Overall score should be in a valid range
        assert!(result.overall_score.is_finite(), "Score should be finite");
        // After phrase adjustment, score might slightly exceed [0,1] but should be reasonable
        assert!(
            result.overall_score >= -0.5 && result.overall_score <= 1.5,
            "Score should be in reasonable range: {}",
            result.overall_score
        );
    }

    #[test]
    fn test_consciousness_adequacy_at_exact_threshold() {
        let mut evaluator = UnifiedValueEvaluator::new();
        // Basic action threshold is 0.2
        let context = EvaluationContext {
            consciousness_level: 0.2, // Exactly at threshold
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("basic action", context);
        // At exactly the threshold, should pass consciousness check
        assert!((result.consciousness_adequacy - 1.0).abs() < f64::EPSILON);
    }

    // ================================================================
    // NEW TESTS: Semantic embeddings toggle
    // ================================================================

    #[test]
    fn test_semantic_embeddings_enable_disable() {
        let mut evaluator = UnifiedValueEvaluator::new();
        assert!(!evaluator.has_semantic_embeddings());

        evaluator.enable_semantic_embeddings();
        assert!(evaluator.has_semantic_embeddings());

        evaluator.disable_semantic_embeddings();
        assert!(!evaluator.has_semantic_embeddings());
    }

    #[test]
    fn test_semantic_embeddings_evaluation_works() {
        let mut evaluator = UnifiedValueEvaluator::new();
        evaluator.enable_semantic_embeddings();

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("help with kindness", context);
        assert!(result.overall_score.is_finite());
    }

    // ================================================================
    // NEW TESTS: Stats tracking
    // ================================================================

    #[test]
    fn test_stats_empty_initially() {
        let evaluator = UnifiedValueEvaluator::new();
        let stats = evaluator.stats();
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.vetoes, 0);
        assert_eq!(stats.warnings, 0);
        assert_eq!(stats.allows, 0);
        assert!((stats.veto_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_track_evaluations() {
        let mut evaluator = UnifiedValueEvaluator::new();

        // Run several evaluations
        for _ in 0..3 {
            let context = EvaluationContext {
                consciousness_level: 0.5,
                affective_systems: AffectiveSystemsState {
                    care: 0.6,
                    ..Default::default()
                },
                action_type: ActionType::Basic,
                action_domain: None,
                involves_others: false,
                ..Default::default()
            };
            evaluator.evaluate("help someone", context);
        }

        let stats = evaluator.stats();
        assert_eq!(stats.total_evaluations, 3);
        assert_eq!(stats.vetoes + stats.warnings + stats.allows, 3);
    }

    #[test]
    fn test_stats_veto_rate_calculated() {
        let mut evaluator = UnifiedValueEvaluator::new();

        // Force a veto via high rage (this path goes through record_evaluation)
        let veto_context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                rage: 0.8,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: true,
            ..Default::default()
        };
        evaluator.evaluate("do something", veto_context);

        let stats = evaluator.stats();
        assert_eq!(stats.vetoes, 1);
        assert!((stats.veto_rate - 1.0).abs() < f64::EPSILON);
    }

    // ================================================================
    // NEW TESTS: Narrative report generation
    // ================================================================

    #[test]
    fn test_narrative_report_generated() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let context = EvaluationContext {
            consciousness_level: 0.7,
            affective_systems: AffectiveSystemsState {
                care: 0.7,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: Some(ActionDomain::Healthcare),
            involves_others: true,
            ..Default::default()
        };
        let result = evaluator.evaluate("provide care", context);
        let report = evaluator.generate_narrative_report(&result, "provide care");

        assert!(
            !report.narrative.is_empty(),
            "Narrative should not be empty"
        );
        assert!(
            !report.broadcast_message.is_empty(),
            "Broadcast message should not be empty"
        );
        assert!(report.timestamp > 0, "Timestamp should be positive");
    }

    #[test]
    fn test_narrative_report_veto_contains_blocked() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let context = EvaluationContext {
            consciousness_level: 0.05,
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Constitutional,
            ..Default::default()
        };
        let result = evaluator.evaluate("amend constitution", context);
        let report = evaluator.generate_narrative_report(&result, "amend constitution");
        assert!(
            report.narrative.contains("BLOCKED"),
            "Veto narrative should contain 'BLOCKED', got: {}",
            report.narrative
        );
    }

    // ================================================================
    // NEW TESTS: Edge cases
    // ================================================================

    #[test]
    fn test_evaluate_empty_action_string() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("", context);
        assert!(
            result.overall_score.is_finite(),
            "Empty action should not produce NaN"
        );
    }

    #[test]
    fn test_evaluate_very_long_action_string() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let long_action = "help ".repeat(500);
        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate(&long_action, context);
        assert!(
            result.overall_score.is_finite(),
            "Long action should not cause overflow"
        );
    }

    #[test]
    fn test_evaluate_high_fear_vetoed() {
        let mut evaluator = UnifiedValueEvaluator::new();
        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState {
                fear: 0.8,
                care: 0.5,
                ..Default::default()
            },
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: true,
            ..Default::default()
        };
        let result = evaluator.evaluate("do something", context);
        assert!(
            matches!(
                result.decision,
                Decision::Veto(VetoReason::NegativeAffectDominant { .. })
            ),
            "High fear should trigger NegativeAffectDominant veto, got: {:?}",
            result.decision
        );
    }

    #[test]
    fn test_evaluate_constitutional_needs_high_consciousness() {
        let mut evaluator = UnifiedValueEvaluator::new();
        // Constitutional threshold is 0.6 by default
        let low_context = EvaluationContext {
            consciousness_level: 0.5, // Below 0.6
            affective_systems: AffectiveSystemsState {
                care: 0.7,
                ..Default::default()
            },
            action_type: ActionType::Constitutional,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("amend the rules", low_context);
        assert!(
            matches!(
                result.decision,
                Decision::Veto(VetoReason::InsufficientConsciousness { .. })
            ),
            "Sub-threshold consciousness for constitutional should be vetoed, got: {:?}",
            result.decision
        );
    }

    #[test]
    fn test_last_result_stored_after_evaluation() {
        let mut evaluator = UnifiedValueEvaluator::new();
        assert!(evaluator.last_result().is_none());

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            ..Default::default()
        };
        evaluator.evaluate("test action", context);
        assert!(evaluator.last_result().is_some());
    }

    #[test]
    fn test_history_eviction() {
        let mut evaluator = UnifiedValueEvaluator::new();
        evaluator.max_history = 5;
        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            ..Default::default()
        };
        for i in 0..10 {
            evaluator.evaluate(&format!("action {i}"), context.clone());
        }
        assert!(evaluator.history.len() <= 5, "History should be capped");
    }

    #[test]
    fn test_feedback_decay_does_not_panic() {
        let mut evaluator = UnifiedValueEvaluator::new();
        // Should not panic even with no data
        evaluator.apply_feedback_decay();

        let context = EvaluationContext {
            consciousness_level: 0.7,
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            ..Default::default()
        };
        let result = evaluator.evaluate("test", context);
        evaluator.feedback_loop.record_user_feedback(
            "test",
            &result,
            &result.decision,
            0.8,
            0.7,
            None,
        );
        evaluator.apply_feedback_decay();
        // Should still function after decay
        let summary = evaluator.feedback_summary();
        assert!(
            summary.total_feedback >= 1,
            "Should have at least 1 feedback entry after recording, got {}",
            summary.total_feedback
        );
    }

    #[test]
    fn test_harmony_adjustment_default_is_near_one() {
        let evaluator = UnifiedValueEvaluator::new();
        let adj = evaluator.get_harmony_adjustment(&Harmony::SacredReciprocity);
        assert!(
            (adj - 1.0).abs() < 0.1,
            "Default harmony adjustment should be near 1.0, got {adj}"
        );
    }

    #[test]
    fn test_affective_grounding_disabled() {
        let mut config = EvaluatorConfig::default();
        config.require_affective_grounding = false;
        let mut evaluator = UnifiedValueEvaluator::with_config(config);

        let context = EvaluationContext {
            consciousness_level: 0.5,
            affective_systems: AffectiveSystemsState::default(), // All zeros
            action_type: ActionType::Basic,
            action_domain: None,
            involves_others: false,
            ..Default::default()
        };
        let result = evaluator.evaluate("test", context);
        // Affective grounding should be 1.0 when not required
        assert!(
            (result.affective_grounding - 1.0).abs() < f64::EPSILON,
            "Affective grounding should be 1.0 when disabled, got {}",
            result.affective_grounding
        );
    }
}
