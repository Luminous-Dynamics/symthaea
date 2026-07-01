// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explanation and confidence types for value decisions.

use super::super::value_feedback_loop::ValueFeedbackLoop;
use super::types::Decision;
use serde::{Deserialize, Serialize};

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
    pub fn from_feedback_loop(feedback_loop: &ValueFeedbackLoop) -> Self {
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
                "Very low confidence: only {data_points} data points. Results may vary significantly."
            ),
            ConfidenceLevel::Low => format!(
                "Low confidence: {data_points} data points. More feedback will improve accuracy."
            ),
            ConfidenceLevel::Moderate => format!(
                "Moderate confidence: {data_points} data points. System is learning patterns."
            ),
            ConfidenceLevel::High => format!(
                "High confidence: {data_points} data points. Decisions are well-calibrated."
            ),
            ConfidenceLevel::VeryHigh => format!(
                "Very high confidence: {data_points} data points. System is highly trained."
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
            explanation: "New system: no learning data yet. Using default value alignments."
                .to_string(),
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
    pub fn new(harmony_a: &str, score_a: f64, harmony_b: &str, score_b: f64) -> Self {
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

    fn generate_description(
        harmony_a: &str,
        score_a: f64,
        harmony_b: &str,
        score_b: f64,
    ) -> String {
        let (positive, negative) = if score_a > score_b {
            (harmony_a, harmony_b)
        } else {
            (harmony_b, harmony_a)
        };

        format!(
            "This action aligns with {positive} but conflicts with {negative}. \
             These harmonies may require different approaches to balance."
        )
    }

    fn generate_resolution(harmony_a: &str, harmony_b: &str) -> String {
        // Known tension patterns and resolutions
        match (harmony_a, harmony_b) {
            ("Sacred Reciprocity", "Pan-Sentient Flourishing")
            | ("Pan-Sentient Flourishing", "Sacred Reciprocity") => {
                "Consider whether the reciprocity truly serves flourishing, \
                 or if generosity without expectation might be more aligned."
                    .to_string()
            }
            ("Infinite Play", "Integral Wisdom") | ("Integral Wisdom", "Infinite Play") => {
                "Balance creative exploration with truthful communication. \
                 Playfulness should not compromise honesty."
                    .to_string()
            }
            ("Evolutionary Progression", "Resonant Coherence")
            | ("Resonant Coherence", "Evolutionary Progression") => {
                "Growth and change can temporarily disrupt coherence. \
                 Consider whether the disruption serves longer-term harmony."
                    .to_string()
            }
            ("Sacred Reciprocity", "Evolutionary Progression")
            | ("Evolutionary Progression", "Sacred Reciprocity") => {
                "Progress may require accepting gifts or support without immediate return. \
                 Trust that contribution flows in many directions."
                    .to_string()
            }
            _ => {
                format!(
                    "Seek a synthesis that honors both {harmony_a} and {harmony_b}. \
                     Often apparent tensions reveal opportunities for deeper integration."
                )
            }
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
