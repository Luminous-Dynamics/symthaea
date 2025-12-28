// Revolutionary Enhancement #4 - Phase 4: Causal Explanations
//
// Implements Natural Language Explanation Generation for Causal Reasoning
//
// Key Innovation: Transform causal analysis into understandable narratives
//
// Core Questions:
//   "Why did Y happen?"
//   "How does X affect Y?"
//   "What would happen if X?"
//   "Why should we do X?"
//
// Explanation Types:
//   1. Causal Attribution: "X caused Y because..."
//   2. Contrastive: "X rather than Y because..."
//   3. Counterfactual: "If X had been different, Y would..."
//   4. Mechanistic: "X affects Y through Z"
//
// This enables:
// - User-friendly explanations
// - Debugging and transparency
// - Decision support
// - Educational value

use super::{
    causal_graph::{CausalGraph, EdgeType},
    probabilistic_inference::{ProbabilisticCausalGraph, ProbabilisticPrediction},
    causal_intervention::{InterventionResult, InterventionSpec},
    counterfactual_reasoning::CounterfactualResult,
    action_planning::{ActionPlan, Goal},
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Level of detail for explanations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationLevel {
    /// Brief one-liner
    Brief,

    /// Standard explanation with key details
    Standard,

    /// Detailed explanation with full reasoning
    Detailed,

    /// Expert-level with mathematical details
    Expert,
}

/// Type of causal explanation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationType {
    /// "X caused Y"
    Attribution,

    /// "X rather than Y"
    Contrastive,

    /// "If X had been different..."
    Counterfactual,

    /// "X affects Y through Z"
    Mechanistic,

    /// "Do X to achieve Y"
    Recommendation,
}

/// A natural language explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalExplanation {
    /// Type of explanation
    pub explanation_type: ExplanationType,

    /// Brief summary (one sentence)
    pub summary: String,

    /// Detailed narrative
    pub narrative: String,

    /// Supporting evidence
    pub evidence: Vec<String>,

    /// Confidence in explanation (0.0 - 1.0)
    pub confidence: f64,

    /// Optional: visual representation hints
    pub visual_hints: Option<VisualHints>,
}

/// Hints for visual representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualHints {
    /// Causal chain to display
    pub causal_chain: Vec<String>,

    /// Key probabilities
    pub probabilities: HashMap<String, f64>,

    /// Highlight nodes
    pub highlights: Vec<String>,
}

/// Explanation generator
pub struct ExplanationGenerator {
    /// Detail level
    level: ExplanationLevel,
}

impl ExplanationGenerator {
    /// Create new generator with standard detail level
    pub fn new() -> Self {
        Self {
            level: ExplanationLevel::Standard,
        }
    }

    /// Set explanation detail level
    pub fn with_level(mut self, level: ExplanationLevel) -> Self {
        self.level = level;
        self
    }

    /// Explain an intervention result
    pub fn explain_intervention(&self, result: &InterventionResult) -> CausalExplanation {
        let summary = match self.level {
            ExplanationLevel::Brief => {
                format!("Predicted effect: {:.0}%", result.predicted_value * 100.0)
            }
            _ => {
                format!(
                    "Intervening will change {} to {:.0}% (baseline: {:.0}%)",
                    result.target,
                    result.predicted_value * 100.0,
                    result.baseline_value.unwrap_or(0.5) * 100.0
                )
            }
        };

        let narrative = match self.level {
            ExplanationLevel::Brief => summary.clone(),
            ExplanationLevel::Standard => {
                format!(
                    "{}\n\nCausal effect: {:+.0}%\nConfidence: {:.0}% - {:.0}%",
                    result.explanation,
                    (result.predicted_value - result.baseline_value.unwrap_or(0.5)) * 100.0,
                    result.confidence_interval.0 * 100.0,
                    result.confidence_interval.1 * 100.0
                )
            }
            ExplanationLevel::Detailed | ExplanationLevel::Expert => {
                format!(
                    "INTERVENTION ANALYSIS\n\n\
                     Target: {}\n\
                     Predicted Value: {:.2} ({:.0}%)\n\
                     Baseline (no intervention): {:.2} ({:.0}%)\n\
                     Causal Effect: {:+.2} ({:+.0}%)\n\n\
                     Causal Path: {}\n\n\
                     Uncertainty: {:.2}\n\
                     95% Confidence Interval: [{:.2}, {:.2}]\n\
                     Source: {}\n\n\
                     {}",
                    result.target,
                    result.predicted_value, result.predicted_value * 100.0,
                    result.baseline_value.unwrap_or(0.5),
                    result.baseline_value.unwrap_or(0.5) * 100.0,
                    result.predicted_value - result.baseline_value.unwrap_or(0.5),
                    (result.predicted_value - result.baseline_value.unwrap_or(0.5)) * 100.0,
                    result.causal_path.join(" → "),
                    result.uncertainty,
                    result.confidence_interval.0,
                    result.confidence_interval.1,
                    result.uncertainty_source,
                    result.explanation
                )
            }
        };

        let evidence = vec![
            format!("Causal path: {}", result.causal_path.join(" → ")),
            format!("Confidence interval: {:.0}% - {:.0}%",
                result.confidence_interval.0 * 100.0,
                result.confidence_interval.1 * 100.0),
        ];

        CausalExplanation {
            explanation_type: ExplanationType::Recommendation,
            summary,
            narrative,
            evidence,
            confidence: 1.0 - result.uncertainty,
            visual_hints: Some(VisualHints {
                causal_chain: result.causal_path.clone(),
                probabilities: [(result.target.clone(), result.predicted_value)].iter().cloned().collect(),
                highlights: result.causal_path.clone(),
            }),
        }
    }

    /// Explain a counterfactual result
    pub fn explain_counterfactual(&self, result: &CounterfactualResult) -> CausalExplanation {
        let summary = match self.level {
            ExplanationLevel::Brief => {
                format!("Would be {:.0}% (actual: {:.0}%)",
                    result.counterfactual_value * 100.0,
                    result.actual_value * 100.0)
            }
            _ => {
                format!(
                    "If different, {} would be {:.0}% instead of {:.0}%",
                    result.target,
                    result.counterfactual_value * 100.0,
                    result.actual_value * 100.0
                )
            }
        };

        let narrative = match self.level {
            ExplanationLevel::Brief => summary.clone(),
            ExplanationLevel::Standard => {
                format!(
                    "{}\n\nCausal effect: {:+.0}%",
                    result.explanation,
                    result.causal_effect * 100.0
                )
            }
            ExplanationLevel::Detailed | ExplanationLevel::Expert => {
                format!(
                    "COUNTERFACTUAL ANALYSIS\n\n\
                     Target: {}\n\
                     Actual Value: {:.2} ({:.0}%)\n\
                     Counterfactual Value: {:.2} ({:.0}%)\n\
                     Causal Effect: {:+.2} ({:+.0}%)\n\n\
                     Hidden State Confidence: {:.0}%\n\
                     Inference Method: {}\n\n\
                     {}",
                    result.target,
                    result.actual_value, result.actual_value * 100.0,
                    result.counterfactual_value, result.counterfactual_value * 100.0,
                    result.causal_effect, result.causal_effect * 100.0,
                    result.hidden_state.confidence * 100.0,
                    result.hidden_state.inference_method,
                    result.explanation
                )
            }
        };

        let evidence = vec![
            format!("Actual: {:.0}%", result.actual_value * 100.0),
            format!("Counterfactual: {:.0}%", result.counterfactual_value * 100.0),
            format!("Effect: {:+.0}%", result.causal_effect * 100.0),
        ];

        CausalExplanation {
            explanation_type: ExplanationType::Counterfactual,
            summary,
            narrative,
            evidence,
            confidence: result.hidden_state.confidence,
            visual_hints: None,
        }
    }

    /// Explain an action plan
    pub fn explain_plan(&self, plan: &ActionPlan) -> CausalExplanation {
        let summary = match self.level {
            ExplanationLevel::Brief => {
                format!("{} steps → {:.0}%",
                    plan.interventions.len(),
                    plan.predicted_value * 100.0)
            }
            _ => {
                format!(
                    "{}-step plan to achieve {} = {:.0}%",
                    plan.interventions.len(),
                    plan.goal.target,
                    plan.predicted_value * 100.0
                )
            }
        };

        let narrative = match self.level {
            ExplanationLevel::Brief => plan.explanation.clone(),
            ExplanationLevel::Standard => {
                format!(
                    "{}\n\nTotal cost: {:.2}\nConfidence: {:.0}%",
                    plan.explanation,
                    plan.total_cost,
                    plan.confidence * 100.0
                )
            }
            ExplanationLevel::Detailed | ExplanationLevel::Expert => {
                let steps = plan.interventions.iter()
                    .map(|i| format!(
                        "  Step {}: {}\n    Effect: {:+.2} | Cost: {:.2}",
                        i.step, i.rationale, i.effect, i.cost
                    ))
                    .collect::<Vec<_>>()
                    .join("\n");

                format!(
                    "ACTION PLAN\n\n\
                     Goal: {} = {:.2}\n\
                     Steps: {}\n\n\
                     {}\n\n\
                     Final Value: {:.2}\n\
                     Total Cost: {:.2}\n\
                     Confidence: {:.0}%\n\n\
                     {}",
                    plan.goal.target,
                    plan.goal.desired_value,
                    plan.interventions.len(),
                    steps,
                    plan.predicted_value,
                    plan.total_cost,
                    plan.confidence * 100.0,
                    if plan.goal.is_satisfied(plan.predicted_value) {
                        "✓ Goal will be achieved"
                    } else {
                        "⚠ Goal partially achieved (best effort)"
                    }
                )
            }
        };

        let evidence = plan.interventions.iter()
            .map(|i| format!("Step {}: {}", i.step, i.node))
            .collect();

        CausalExplanation {
            explanation_type: ExplanationType::Recommendation,
            summary,
            narrative,
            evidence,
            confidence: plan.confidence,
            visual_hints: Some(VisualHints {
                causal_chain: plan.interventions.iter().map(|i| i.node.clone()).collect(),
                probabilities: [(plan.goal.target.clone(), plan.predicted_value)].iter().cloned().collect(),
                highlights: plan.interventions.iter().map(|i| i.node.clone()).collect(),
            }),
        }
    }

    /// Explain a probabilistic prediction
    pub fn explain_prediction(&self, pred: &ProbabilisticPrediction) -> CausalExplanation {
        let summary = format!(
            "{}: {:.0}% likely",
            pred.event_type,
            pred.probability * 100.0
        );

        let narrative = match self.level {
            ExplanationLevel::Brief => summary.clone(),
            ExplanationLevel::Standard => {
                format!(
                    "{} will occur with {:.0}% probability\nConfidence: {:.0}%",
                    pred.event_type,
                    pred.probability * 100.0,
                    pred.confidence_level * 100.0
                )
            }
            ExplanationLevel::Detailed | ExplanationLevel::Expert => {
                format!(
                    "PROBABILISTIC PREDICTION\n\n\
                     Event: {}\n\
                     Probability: {:.2} ({:.0}%)\n\
                     Confidence Level: {:.0}%\n\
                     95% CI: [{:.0}%, {:.0}%]\n\n\
                     Causal Chain: {}\n\
                     Observations: {}\n\
                     Uncertainty Source: {}",
                    pred.event_type,
                    pred.probability, pred.probability * 100.0,
                    pred.confidence_level * 100.0,
                    pred.confidence_interval.0 * 100.0,
                    pred.confidence_interval.1 * 100.0,
                    pred.causal_chain.join(" → "),
                    pred.observations,
                    pred.uncertainty_source
                )
            }
        };

        CausalExplanation {
            explanation_type: ExplanationType::Attribution,
            summary,
            narrative,
            evidence: vec![
                format!("Probability: {:.0}%", pred.probability * 100.0),
                format!("Based on {} observations", pred.observations),
            ],
            confidence: pred.confidence_level,
            visual_hints: Some(VisualHints {
                causal_chain: pred.causal_chain.clone(),
                probabilities: [(pred.event_type.clone(), pred.probability)].iter().cloned().collect(),
                highlights: vec![pred.event_type.clone()],
            }),
        }
    }

    /// Generate contrastive explanation: "X rather than Y because..."
    pub fn explain_contrastive(
        &self,
        chosen: &str,
        chosen_value: f64,
        alternative: &str,
        alternative_value: f64,
    ) -> CausalExplanation {
        let difference = chosen_value - alternative_value;
        let summary = format!(
            "{} chosen over {} ({:+.0}% better)",
            chosen, alternative, difference * 100.0
        );

        let narrative = match self.level {
            ExplanationLevel::Brief => summary.clone(),
            _ => {
                format!(
                    "CONTRASTIVE EXPLANATION\n\n\
                     Chosen: {} ({:.0}%)\n\
                     Alternative: {} ({:.0}%)\n\
                     Difference: {:+.0}%\n\n\
                     Reason: {} provides {:.0}% higher expected value",
                    chosen, chosen_value * 100.0,
                    alternative, alternative_value * 100.0,
                    difference * 100.0,
                    chosen, difference.abs() * 100.0
                )
            }
        };

        CausalExplanation {
            explanation_type: ExplanationType::Contrastive,
            summary,
            narrative,
            evidence: vec![
                format!("{}: {:.0}%", chosen, chosen_value * 100.0),
                format!("{}: {:.0}%", alternative, alternative_value * 100.0),
            ],
            confidence: 0.9,
            visual_hints: None,
        }
    }
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::probabilistic_inference::UncertaintySource;

    #[test]
    fn test_generator_creation() {
        let gen = ExplanationGenerator::new();
        assert_eq!(gen.level, ExplanationLevel::Standard);

        let brief_gen = ExplanationGenerator::new().with_level(ExplanationLevel::Brief);
        assert_eq!(brief_gen.level, ExplanationLevel::Brief);
    }

    #[test]
    fn test_explain_intervention() {
        let result = InterventionResult {
            target: "phi_value".to_string(),
            predicted_value: 0.85,
            baseline_value: Some(0.5),
            uncertainty: 0.1,
            confidence_interval: (0.75, 0.92),
            causal_path: vec!["security_check".to_string(), "phi_value".to_string()],
            uncertainty_source: UncertaintySource::WellEstimated,
            explanation: "Enabling security_check will increase phi_value".to_string(),
        };

        let gen = ExplanationGenerator::new();
        let explanation = gen.explain_intervention(&result);

        assert_eq!(explanation.explanation_type, ExplanationType::Recommendation);
        assert!(!explanation.summary.is_empty());
        assert!(!explanation.narrative.is_empty());
        assert!(explanation.confidence > 0.0);
    }

    #[test]
    fn test_explanation_levels() {
        let result = InterventionResult {
            target: "test".to_string(),
            predicted_value: 0.8,
            baseline_value: Some(0.5),
            uncertainty: 0.1,
            confidence_interval: (0.7, 0.9),
            causal_path: vec!["A".to_string(), "test".to_string()],
            uncertainty_source: UncertaintySource::WellEstimated,
            explanation: "Test explanation".to_string(),
        };

        let brief = ExplanationGenerator::new()
            .with_level(ExplanationLevel::Brief)
            .explain_intervention(&result);

        let detailed = ExplanationGenerator::new()
            .with_level(ExplanationLevel::Detailed)
            .explain_intervention(&result);

        // Detailed should be longer
        assert!(detailed.narrative.len() > brief.narrative.len());
    }

    #[test]
    fn test_contrastive_explanation() {
        let gen = ExplanationGenerator::new();
        let explanation = gen.explain_contrastive("option_a", 0.8, "option_b", 0.6);

        assert_eq!(explanation.explanation_type, ExplanationType::Contrastive);
        assert!(explanation.summary.contains("option_a"));
        assert!(explanation.summary.contains("option_b"));
    }
}
