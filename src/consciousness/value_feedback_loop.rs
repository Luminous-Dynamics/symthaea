//! Value Feedback Loop - Meta-Cognitive Value Learning
//!
//! This module integrates value evaluation with meta-cognitive learning to create
//! a system that improves its value judgments over time.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    VALUE FEEDBACK LOOP                                   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   Action/Speech/Attention                                                │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Value Evaluation │  ← Seven Harmonies + Semantic Checker            │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │  Decision Made   │  ← Allow / Warn / Veto                           │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Outcome Observed │  ← User feedback, system metrics, consequences   │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Feedback Loop    │  ← Learn from decision-outcome pairs             │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   Update Harmony Importance → Better Future Decisions                    │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Learning Principles
//!
//! 1. **Outcome-Based**: Learn from actual consequences, not just rules
//! 2. **Harmony-Specific**: Different harmonies adjust independently
//! 3. **Consciousness-Gated**: Learning requires sufficient Φ
//! 4. **Decay-Protected**: Old learning decays, recent feedback matters more
//! 5. **Bounds-Constrained**: Importance values stay within reasonable range

use super::unified_value_evaluator::{
    EvaluationContext, EvaluationResult, Decision, ActionType, AffectiveSystemsState,
};
use super::seven_harmonies::Harmony;
use super::affective_consciousness::CoreAffect;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use tracing::{debug, info, warn};

// ============================================================================
// OUTCOME TYPES
// ============================================================================

/// Observed outcome of a value-gated action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueOutcome {
    /// Unique ID for tracking
    pub id: String,
    /// The action that was evaluated
    pub action_description: String,
    /// The evaluation result
    pub evaluation: OutcomeEvaluation,
    /// The decision made (Allow/Warn/Veto)
    pub decision: OutcomeDecision,
    /// The observed outcome quality (-1.0 to +1.0)
    pub outcome_quality: f64,
    /// Feedback source
    pub feedback_source: FeedbackSource,
    /// Consciousness level when decision was made
    pub phi_at_decision: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Simplified evaluation for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeEvaluation {
    /// Overall harmony score
    pub harmony_score: f64,
    /// Individual harmony scores
    pub harmony_breakdown: Vec<(String, f64)>,
    /// Authenticity score
    pub authenticity: f64,
}

/// Simplified decision for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutcomeDecision {
    Allowed,
    Warned(Vec<String>),
    Vetoed(String),
}

/// Source of outcome feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackSource {
    /// User explicitly rated the outcome
    UserFeedback { rating: f64, comment: Option<String> },
    /// System metrics (response quality, coherence, etc.)
    SystemMetrics { metric_name: String, value: f64 },
    /// Observed consequence (e.g., error rate, success rate)
    ObservedConsequence { consequence_type: String, severity: f64 },
    /// Self-reflection (metacognitive assessment)
    SelfReflection { phi_change: f64, coherence_change: f64 },
}

// ============================================================================
// LEARNING RECORD
// ============================================================================

/// A record of what was learned from an outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRecord {
    /// Which harmony was affected
    pub harmony: String,
    /// Importance adjustment (delta)
    pub importance_delta: f64,
    /// Why this adjustment was made
    pub reason: String,
    /// Confidence in this learning (0-1)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
}

// ============================================================================
// VALUE FEEDBACK LOOP
// ============================================================================

/// Configuration for the feedback loop
#[derive(Debug, Clone)]
pub struct FeedbackLoopConfig {
    /// Learning rate (how much to adjust per feedback)
    pub learning_rate: f64,
    /// Decay rate for old learning (per day)
    pub decay_rate: f64,
    /// Minimum consciousness level for learning
    pub min_phi_for_learning: f64,
    /// Maximum importance adjustment per feedback
    pub max_adjustment: f64,
    /// Minimum importance value
    pub min_importance: f64,
    /// Maximum importance value
    pub max_importance: f64,
    /// History window size
    pub history_size: usize,
    /// Minimum feedback count before adjusting
    pub min_feedback_for_adjustment: usize,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            decay_rate: 0.01,
            min_phi_for_learning: 0.3,
            max_adjustment: 0.1,
            min_importance: 0.5,
            max_importance: 2.0,
            history_size: 100,
            min_feedback_for_adjustment: 3,
        }
    }
}

/// The main value feedback loop system
pub struct ValueFeedbackLoop {
    /// Configuration
    config: FeedbackLoopConfig,
    /// History of outcomes
    outcome_history: VecDeque<ValueOutcome>,
    /// Current harmony importance adjustments
    harmony_adjustments: HashMap<String, f64>,
    /// Learning records
    learning_history: VecDeque<LearningRecord>,
    /// Statistics per harmony
    harmony_stats: HashMap<String, HarmonyStats>,
    /// Total outcomes processed
    total_outcomes: u64,
    /// Last learning time
    last_learning: Instant,
}

/// Statistics for a single harmony
#[derive(Debug, Clone, Default)]
struct HarmonyStats {
    /// Total positive outcomes
    positive_outcomes: u64,
    /// Total negative outcomes
    negative_outcomes: u64,
    /// Running sum of outcome quality
    quality_sum: f64,
    /// Running count for average
    quality_count: u64,
    /// False positive count (vetoed but should have allowed)
    false_positives: u64,
    /// False negative count (allowed but should have vetoed)
    false_negatives: u64,
}

impl ValueFeedbackLoop {
    /// Create a new feedback loop with default configuration
    pub fn new() -> Self {
        Self::with_config(FeedbackLoopConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: FeedbackLoopConfig) -> Self {
        Self {
            config,
            outcome_history: VecDeque::new(),
            harmony_adjustments: HashMap::new(),
            learning_history: VecDeque::new(),
            harmony_stats: HashMap::new(),
            total_outcomes: 0,
            last_learning: Instant::now(),
        }
    }

    /// Record an outcome and learn from it
    pub fn record_outcome(&mut self, outcome: ValueOutcome) {
        // Check consciousness threshold for learning
        if outcome.phi_at_decision < self.config.min_phi_for_learning {
            debug!(
                "🧠 Skipping learning: Φ={:.3} < threshold {:.3}",
                outcome.phi_at_decision, self.config.min_phi_for_learning
            );
            return;
        }

        // Update statistics
        self.update_stats(&outcome);

        // Store in history
        if self.outcome_history.len() >= self.config.history_size {
            self.outcome_history.pop_front();
        }
        self.outcome_history.push_back(outcome.clone());
        self.total_outcomes += 1;

        // Perform learning if we have enough data
        if self.outcome_history.len() >= self.config.min_feedback_for_adjustment {
            self.learn_from_outcome(&outcome);
        }

        debug!(
            "📝 Recorded outcome #{}: quality={:.2}, decision={:?}",
            self.total_outcomes, outcome.outcome_quality, outcome.decision
        );
    }

    /// Update statistics from an outcome
    fn update_stats(&mut self, outcome: &ValueOutcome) {
        for (harmony_name, score) in &outcome.evaluation.harmony_breakdown {
            let stats = self.harmony_stats
                .entry(harmony_name.clone())
                .or_default();

            stats.quality_sum += outcome.outcome_quality;
            stats.quality_count += 1;

            if outcome.outcome_quality > 0.0 {
                stats.positive_outcomes += 1;
            } else {
                stats.negative_outcomes += 1;
            }

            // Detect false positives/negatives
            match &outcome.decision {
                OutcomeDecision::Vetoed(_) if outcome.outcome_quality > 0.5 => {
                    // Vetoed but outcome would have been good - false positive
                    stats.false_positives += 1;
                }
                OutcomeDecision::Allowed if outcome.outcome_quality < -0.5 => {
                    // Allowed but outcome was bad - false negative
                    stats.false_negatives += 1;
                }
                _ => {}
            }
        }
    }

    /// Learn from a single outcome
    fn learn_from_outcome(&mut self, outcome: &ValueOutcome) {
        let quality = outcome.outcome_quality;

        // For each harmony in the evaluation
        for (harmony_name, score) in &outcome.evaluation.harmony_breakdown {
            // Calculate the correlation between this harmony's score and the outcome
            let correlation = self.calculate_harmony_outcome_correlation(harmony_name);

            // Determine adjustment direction
            let adjustment = self.calculate_adjustment(
                *score,
                quality,
                correlation,
                &outcome.decision,
            );

            if adjustment.abs() > 0.001 {
                // Apply adjustment
                let current = self.harmony_adjustments
                    .get(harmony_name)
                    .copied()
                    .unwrap_or(1.0);

                let new_value = (current + adjustment)
                    .clamp(self.config.min_importance, self.config.max_importance);

                self.harmony_adjustments.insert(harmony_name.clone(), new_value);

                // Record learning
                let record = LearningRecord {
                    harmony: harmony_name.clone(),
                    importance_delta: adjustment,
                    reason: format!(
                        "Outcome quality {:.2}, harmony score {:.2}, correlation {:.2}",
                        quality, score, correlation
                    ),
                    confidence: outcome.phi_at_decision,
                    timestamp: now_secs(),
                };

                if self.learning_history.len() >= self.config.history_size {
                    self.learning_history.pop_front();
                }
                self.learning_history.push_back(record);

                debug!(
                    "🎓 Learned: {} importance {:.3} → {:.3} (Δ={:+.3})",
                    harmony_name, current, new_value, adjustment
                );
            }
        }

        self.last_learning = Instant::now();
    }

    /// Calculate the correlation between a harmony's scores and outcomes
    fn calculate_harmony_outcome_correlation(&self, harmony_name: &str) -> f64 {
        let relevant_outcomes: Vec<_> = self.outcome_history.iter()
            .filter_map(|o| {
                o.evaluation.harmony_breakdown.iter()
                    .find(|(name, _)| name == harmony_name)
                    .map(|(_, score)| (*score, o.outcome_quality))
            })
            .collect();

        if relevant_outcomes.len() < 3 {
            return 0.0; // Not enough data
        }

        // Simple Pearson correlation
        let n = relevant_outcomes.len() as f64;
        let sum_x: f64 = relevant_outcomes.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = relevant_outcomes.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = relevant_outcomes.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = relevant_outcomes.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = relevant_outcomes.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (numerator / denominator).clamp(-1.0, 1.0)
    }

    /// Calculate the importance adjustment for a harmony
    fn calculate_adjustment(
        &self,
        harmony_score: f64,
        outcome_quality: f64,
        correlation: f64,
        decision: &OutcomeDecision,
    ) -> f64 {
        let base_lr = self.config.learning_rate;

        // Higher correlation means we trust this harmony more
        let confidence_factor = (1.0 + correlation.abs()) / 2.0;

        // Calculate direction
        let direction = match decision {
            OutcomeDecision::Vetoed(_) => {
                // If vetoed and outcome would have been good, decrease importance
                // If vetoed and outcome would have been bad, this was correct, small increase
                if outcome_quality > 0.3 {
                    -1.0 // Too strict - decrease importance
                } else {
                    0.2 // Correct veto - small increase
                }
            }
            OutcomeDecision::Warned(_) => {
                // Warnings should become vetoes if outcomes are bad
                if outcome_quality < -0.3 {
                    0.5 // Should have vetoed - increase importance
                } else if outcome_quality > 0.5 {
                    -0.3 // Warning was too aggressive
                } else {
                    0.0 // Warning was appropriate
                }
            }
            OutcomeDecision::Allowed => {
                // If allowed and outcome was bad, increase importance
                // If allowed and outcome was good, this was correct, small decrease
                if outcome_quality < -0.3 {
                    0.8 // Should have vetoed - increase importance significantly
                } else if outcome_quality > 0.5 {
                    -0.1 // Correct allow - tiny decrease to not over-trigger
                } else {
                    0.0 // Neutral
                }
            }
        };

        // Final adjustment = direction * learning_rate * confidence * outcome_magnitude
        let adjustment = direction * base_lr * confidence_factor * outcome_quality.abs();

        // Clamp to max adjustment
        adjustment.clamp(-self.config.max_adjustment, self.config.max_adjustment)
    }

    /// Get the current importance adjustment for a harmony
    pub fn get_importance_adjustment(&self, harmony: &Harmony) -> f64 {
        self.harmony_adjustments
            .get(harmony.name())
            .copied()
            .unwrap_or(1.0)
    }

    /// Get all harmony adjustments
    pub fn get_all_adjustments(&self) -> &HashMap<String, f64> {
        &self.harmony_adjustments
    }

    /// Apply decay to old learning
    pub fn apply_decay(&mut self) {
        let elapsed_days = self.last_learning.elapsed().as_secs_f64() / 86400.0;
        let decay_factor = (-self.config.decay_rate * elapsed_days).exp();

        for (_, adjustment) in self.harmony_adjustments.iter_mut() {
            // Decay toward 1.0 (neutral)
            *adjustment = 1.0 + (*adjustment - 1.0) * decay_factor;
        }
    }

    /// Get summary statistics
    pub fn summary(&self) -> FeedbackLoopSummary {
        let mut harmony_summaries = Vec::new();

        for (name, stats) in &self.harmony_stats {
            let avg_quality = if stats.quality_count > 0 {
                stats.quality_sum / stats.quality_count as f64
            } else {
                0.0
            };

            let adjustment = self.harmony_adjustments.get(name).copied().unwrap_or(1.0);

            harmony_summaries.push(HarmonySummary {
                name: name.clone(),
                positive_outcomes: stats.positive_outcomes,
                negative_outcomes: stats.negative_outcomes,
                average_quality: avg_quality,
                false_positives: stats.false_positives,
                false_negatives: stats.false_negatives,
                current_adjustment: adjustment,
            });
        }

        FeedbackLoopSummary {
            total_outcomes: self.total_outcomes,
            recent_outcomes: self.outcome_history.len(),
            total_learning_records: self.learning_history.len(),
            harmony_summaries,
        }
    }

    /// Record explicit user feedback
    pub fn record_user_feedback(
        &mut self,
        action_description: &str,
        evaluation: &EvaluationResult,
        decision: &Decision,
        rating: f64,
        phi: f64,
        comment: Option<String>,
    ) {
        let outcome = ValueOutcome {
            id: format!("user-{}", now_secs()),
            action_description: action_description.to_string(),
            evaluation: OutcomeEvaluation {
                harmony_score: evaluation.harmony_alignment.overall_score,
                harmony_breakdown: evaluation.breakdown.harmony_scores.clone(),
                authenticity: evaluation.authenticity,
            },
            decision: match decision {
                Decision::Allow => OutcomeDecision::Allowed,
                Decision::Warn(w) => OutcomeDecision::Warned(w.clone()),
                Decision::Veto(r) => OutcomeDecision::Vetoed(format!("{:?}", r)),
            },
            outcome_quality: (rating * 2.0 - 1.0).clamp(-1.0, 1.0), // Convert 0-1 to -1..1
            feedback_source: FeedbackSource::UserFeedback { rating, comment },
            phi_at_decision: phi,
            timestamp: now_secs(),
        };

        info!(
            "👤 User feedback received: rating={:.2}, φ={:.3}",
            rating, phi
        );

        self.record_outcome(outcome);
    }

    /// Record self-reflection feedback (from meta-cognition)
    pub fn record_self_reflection(
        &mut self,
        action_description: &str,
        evaluation: &EvaluationResult,
        decision: &Decision,
        phi_change: f64,
        coherence_change: f64,
        phi: f64,
    ) {
        // Positive phi/coherence change = good outcome
        let outcome_quality = (phi_change * 0.6 + coherence_change * 0.4).clamp(-1.0, 1.0);

        let outcome = ValueOutcome {
            id: format!("reflect-{}", now_secs()),
            action_description: action_description.to_string(),
            evaluation: OutcomeEvaluation {
                harmony_score: evaluation.harmony_alignment.overall_score,
                harmony_breakdown: evaluation.breakdown.harmony_scores.clone(),
                authenticity: evaluation.authenticity,
            },
            decision: match decision {
                Decision::Allow => OutcomeDecision::Allowed,
                Decision::Warn(w) => OutcomeDecision::Warned(w.clone()),
                Decision::Veto(r) => OutcomeDecision::Vetoed(format!("{:?}", r)),
            },
            outcome_quality,
            feedback_source: FeedbackSource::SelfReflection { phi_change, coherence_change },
            phi_at_decision: phi,
            timestamp: now_secs(),
        };

        debug!(
            "🪞 Self-reflection: Δφ={:+.3}, Δcoherence={:+.3}, quality={:.2}",
            phi_change, coherence_change, outcome_quality
        );

        self.record_outcome(outcome);
    }
}

impl Default for ValueFeedbackLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SUMMARY TYPES
// ============================================================================

/// Summary of feedback loop state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoopSummary {
    /// Total outcomes ever processed
    pub total_outcomes: u64,
    /// Outcomes in recent window
    pub recent_outcomes: usize,
    /// Total learning records
    pub total_learning_records: usize,
    /// Per-harmony summaries
    pub harmony_summaries: Vec<HarmonySummary>,
}

/// Summary for a single harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonySummary {
    pub name: String,
    pub positive_outcomes: u64,
    pub negative_outcomes: u64,
    pub average_quality: f64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub current_adjustment: f64,
}

// ============================================================================
// UTILITIES
// ============================================================================

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_outcome(quality: f64, decision: OutcomeDecision) -> ValueOutcome {
        ValueOutcome {
            id: format!("test-{}", now_secs()),
            action_description: "Test action".to_string(),
            evaluation: OutcomeEvaluation {
                harmony_score: 0.5,
                harmony_breakdown: vec![
                    ("PanSentientFlourishing".to_string(), 0.6),
                    ("SacredReciprocity".to_string(), 0.4),
                ],
                authenticity: 0.7,
            },
            decision,
            outcome_quality: quality,
            feedback_source: FeedbackSource::SystemMetrics {
                metric_name: "test".to_string(),
                value: quality,
            },
            phi_at_decision: 0.5,
            timestamp: now_secs(),
        }
    }

    #[test]
    fn test_feedback_loop_creation() {
        let loop_system = ValueFeedbackLoop::new();
        assert_eq!(loop_system.total_outcomes, 0);
        assert!(loop_system.outcome_history.is_empty());
    }

    #[test]
    fn test_record_positive_outcome() {
        let mut loop_system = ValueFeedbackLoop::new();

        let outcome = make_test_outcome(0.8, OutcomeDecision::Allowed);
        loop_system.record_outcome(outcome);

        assert_eq!(loop_system.total_outcomes, 1);
        assert_eq!(loop_system.outcome_history.len(), 1);
    }

    #[test]
    fn test_learning_from_bad_allowed_outcome() {
        let mut loop_system = ValueFeedbackLoop::with_config(FeedbackLoopConfig {
            min_feedback_for_adjustment: 1, // Learn immediately
            ..Default::default()
        });

        // Allowed something that had a bad outcome
        let outcome = make_test_outcome(-0.7, OutcomeDecision::Allowed);
        loop_system.record_outcome(outcome);

        // Should have increased importance (to catch this next time)
        let summary = loop_system.summary();
        assert!(summary.harmony_summaries.iter().any(|h| h.current_adjustment > 1.0));
    }

    #[test]
    fn test_learning_from_good_vetoed_outcome() {
        let mut loop_system = ValueFeedbackLoop::with_config(FeedbackLoopConfig {
            min_feedback_for_adjustment: 1,
            ..Default::default()
        });

        // Vetoed something that would have been good
        let outcome = make_test_outcome(0.8, OutcomeDecision::Vetoed("test".to_string()));
        loop_system.record_outcome(outcome);

        // Should have decreased importance (was too strict)
        let summary = loop_system.summary();
        assert!(summary.harmony_summaries.iter().any(|h| h.current_adjustment < 1.0));
    }

    #[test]
    fn test_consciousness_gating() {
        let mut loop_system = ValueFeedbackLoop::with_config(FeedbackLoopConfig {
            min_phi_for_learning: 0.6,
            ..Default::default()
        });

        // Low consciousness - should not learn
        let mut outcome = make_test_outcome(0.8, OutcomeDecision::Allowed);
        outcome.phi_at_decision = 0.3; // Below threshold
        loop_system.record_outcome(outcome);

        // Should not have recorded
        assert_eq!(loop_system.total_outcomes, 0);
    }

    #[test]
    fn test_summary() {
        let mut loop_system = ValueFeedbackLoop::new();

        // Add some outcomes
        loop_system.record_outcome(make_test_outcome(0.5, OutcomeDecision::Allowed));
        loop_system.record_outcome(make_test_outcome(-0.3, OutcomeDecision::Allowed));

        let summary = loop_system.summary();
        assert_eq!(summary.total_outcomes, 2);
        assert!(summary.harmony_summaries.len() > 0);
    }
}
