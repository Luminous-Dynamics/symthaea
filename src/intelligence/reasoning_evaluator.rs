// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, benchmark-agnostic evaluation for reasoning episodes.
//!
//! Benchmark adapters decide task semantics and produce an [`EpisodeJudgment`]. This module
//! turns that judgment plus the immutable subject episode into common qualification metrics.
//! It intentionally preserves the metric vector rather than collapsing reasoning into one
//! opaque "intelligence score".

use super::reasoning_qualification::{
    QualificationMetric, QualificationValidationError, ReasoningEpisode, ReasoningOutcome,
    ReasoningQualificationReceipt,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Initial evaluator contract. Increment whenever metric semantics change.
pub const REASONING_EVALUATOR_VERSION: &str = "rq-evaluator-v1";

const LOG_LOSS_EPSILON: f64 = 1.0e-15;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

/// Optional benchmark-native score supplied by an adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskScore {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

/// Ground-truth judgment supplied by a benchmark-specific adapter.
///
/// `exact_correct` is independent of confidence. The evaluator derives calibration metrics
/// from the episode's asserted confidence when exact correctness is known.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodeJudgment {
    pub exact_correct: Option<bool>,
    pub task_score: Option<TaskScore>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningEvaluatorError {
    EmptyEvaluationLineage,
    EmptyTaskScoreName,
    NonFiniteTaskScore,
    Validation(QualificationValidationError),
}

impl fmt::Display for ReasoningEvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEvaluationLineage => write!(f, "evaluation lineage hash must not be empty"),
            Self::EmptyTaskScoreName => write!(f, "task score name must not be empty"),
            Self::NonFiniteTaskScore => write!(f, "task score must be finite"),
            Self::Validation(err) => write!(f, "qualification validation failed: {err}"),
        }
    }
}

impl std::error::Error for ReasoningEvaluatorError {}

impl From<QualificationValidationError> for ReasoningEvaluatorError {
    fn from(value: QualificationValidationError) -> Self {
        Self::Validation(value)
    }
}

/// Deterministically evaluate one episode without mutating it.
pub fn evaluate_episode(
    episode: &ReasoningEpisode,
    evaluation_lineage_hash: &str,
    judgment: &EpisodeJudgment,
) -> Result<ReasoningQualificationReceipt, ReasoningEvaluatorError> {
    if evaluation_lineage_hash.trim().is_empty() {
        return Err(ReasoningEvaluatorError::EmptyEvaluationLineage);
    }
    episode.validate()?;

    if let Some(score) = &judgment.task_score {
        if score.name.trim().is_empty() {
            return Err(ReasoningEvaluatorError::EmptyTaskScoreName);
        }
        if !score.value.is_finite() {
            return Err(ReasoningEvaluatorError::NonFiniteTaskScore);
        }
    }

    let asserted = matches!(episode.outcome, ReasoningOutcome::Asserted { .. });
    let mut metrics = vec![QualificationMetric {
        name: "coverage".into(),
        value: if asserted { 1.0 } else { 0.0 },
        unit: "fraction".into(),
    }];

    if let Some(correct) = judgment.exact_correct {
        // Overall exact accuracy counts abstention as unsolved. Selective accuracy is emitted
        // only for answered episodes and therefore measures correctness conditional on coverage.
        metrics.push(QualificationMetric {
            name: "exact_accuracy".into(),
            value: if asserted && correct { 1.0 } else { 0.0 },
            unit: "fraction".into(),
        });

        if let ReasoningOutcome::Asserted { confidence, .. } = &episode.outcome {
            let y = if correct { 1.0 } else { 0.0 };
            let brier = (confidence - y).powi(2);
            let probability_correct = if correct {
                *confidence
            } else {
                1.0 - *confidence
            };
            let clipped = probability_correct.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON);

            metrics.push(QualificationMetric {
                name: "selective_accuracy".into(),
                value: if correct { 1.0 } else { 0.0 },
                unit: "fraction".into(),
            });
            metrics.push(QualificationMetric {
                name: "brier_score".into(),
                value: brier,
                unit: "score".into(),
            });
            metrics.push(QualificationMetric {
                name: "log_loss".into(),
                value: -clipped.ln(),
                unit: "nats".into(),
            });
        }
    }

    if let Some(score) = &judgment.task_score {
        metrics.push(QualificationMetric {
            name: format!("task.{}", score.name),
            value: score.value,
            unit: score.unit.clone(),
        });
    }

    Ok(ReasoningQualificationReceipt::new(
        episode,
        "symthaea-reasoning-evaluator",
        REASONING_EVALUATOR_VERSION,
        evaluation_lineage_hash,
        metrics,
        judgment.exact_correct.map(|correct| asserted && correct),
    )?)
}

/// Confidence interval for a binomial proportion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProportionInterval {
    pub lower: f64,
    pub upper: f64,
}

/// Aggregate capability vector for a homogeneous benchmark/domain slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySlice {
    pub episodes: usize,
    pub covered: usize,
    pub exact_judged: usize,
    pub exact_correct: usize,
    /// Overall exact accuracy; abstentions are unsolved.
    pub exact_accuracy: Option<f64>,
    pub exact_accuracy_95: Option<ProportionInterval>,
    /// Fraction of tasks on which an answer was asserted.
    pub coverage: f64,
    /// Accuracy conditional on asserting an answer.
    pub selective_accuracy: Option<f64>,
    /// Mean Brier score over answered tasks with exact ground truth.
    pub mean_brier_score: Option<f64>,
    /// Mean log loss over answered tasks with exact ground truth.
    pub mean_log_loss: Option<f64>,
}

/// Aggregate common metrics from evaluator receipts.
///
/// Missing metrics remain missing; they are never silently replaced with zero. Receipts from
/// benchmark adapters that lack exact ground truth can therefore coexist with exact-task lanes
/// without fabricating accuracy/calibration claims.
pub fn aggregate_receipts(receipts: &[ReasoningQualificationReceipt]) -> CapabilitySlice {
    let episodes = receipts.len();
    let mut covered = 0usize;
    let mut exact_judged = 0usize;
    let mut exact_correct = 0usize;
    let mut selective_sum = 0.0;
    let mut selective_n = 0usize;
    let mut brier_sum = 0.0;
    let mut brier_n = 0usize;
    let mut log_loss_sum = 0.0;
    let mut log_loss_n = 0usize;

    for receipt in receipts {
        if metric_value(receipt, "coverage") == Some(1.0) {
            covered += 1;
        }
        if let Some(correct) = receipt.exact_correct {
            exact_judged += 1;
            if correct {
                exact_correct += 1;
            }
        }
        if let Some(value) = metric_value(receipt, "selective_accuracy") {
            selective_sum += value;
            selective_n += 1;
        }
        if let Some(value) = metric_value(receipt, "brier_score") {
            brier_sum += value;
            brier_n += 1;
        }
        if let Some(value) = metric_value(receipt, "log_loss") {
            log_loss_sum += value;
            log_loss_n += 1;
        }
    }

    let exact_accuracy = ratio(exact_correct, exact_judged);
    CapabilitySlice {
        episodes,
        covered,
        exact_judged,
        exact_correct,
        exact_accuracy,
        exact_accuracy_95: if exact_judged == 0 {
            None
        } else {
            Some(wilson_interval_95(exact_correct, exact_judged))
        },
        coverage: if episodes == 0 {
            0.0
        } else {
            covered as f64 / episodes as f64
        },
        selective_accuracy: mean(selective_sum, selective_n),
        mean_brier_score: mean(brier_sum, brier_n),
        mean_log_loss: mean(log_loss_sum, log_loss_n),
    }
}

/// Wilson 95% interval, preferable to a normal approximation near 0/1 or for small n.
pub fn wilson_interval_95(successes: usize, total: usize) -> ProportionInterval {
    if total == 0 {
        return ProportionInterval {
            lower: 0.0,
            upper: 1.0,
        };
    }
    let n = total as f64;
    let p = successes.min(total) as f64 / n;
    let z2 = WILSON_Z_95 * WILSON_Z_95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = WILSON_Z_95
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    ProportionInterval {
        lower: (center - margin).clamp(0.0, 1.0),
        upper: (center + margin).clamp(0.0, 1.0),
    }
}

fn metric_value(receipt: &ReasoningQualificationReceipt, name: &str) -> Option<f64> {
    receipt
        .metrics
        .iter()
        .find(|metric| metric.name == name)
        .map(|metric| metric.value)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn mean(sum: f64, count: usize) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::reasoning_qualification::{
        AbstentionReason, ReasoningDomain, ReasoningProblemRef, ResourceUsage,
    };

    fn episode(outcome: ReasoningOutcome) -> ReasoningEpisode {
        match ReasoningEpisode::new(
            "subject",
            "config",
            ReasoningDomain::Logic,
            ReasoningProblemRef {
                benchmark: "logic".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: "p1".into(),
                problem_hash: "sha256:p1".into(),
            },
            vec![],
            vec![],
            vec![],
            outcome,
            ResourceUsage::default(),
        ) {
            Ok(value) => value,
            Err(err) => panic!("test episode must validate: {err}"),
        }
    }

    fn metric(receipt: &ReasoningQualificationReceipt, name: &str) -> f64 {
        match receipt.metrics.iter().find(|m| m.name == name) {
            Some(value) => value.value,
            None => panic!("missing metric {name}"),
        }
    }

    #[test]
    fn correct_assertion_scores_accuracy_and_calibration() {
        let subject = episode(ReasoningOutcome::Asserted {
            value: "A".into(),
            confidence: 0.8,
        });
        let receipt = evaluate_episode(
            &subject,
            "eval-lineage",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
        );
        let receipt = match receipt {
            Ok(value) => value,
            Err(err) => panic!("evaluation must succeed: {err}"),
        };
        assert_eq!(metric(&receipt, "coverage"), 1.0);
        assert_eq!(metric(&receipt, "exact_accuracy"), 1.0);
        assert_eq!(metric(&receipt, "selective_accuracy"), 1.0);
        assert!((metric(&receipt, "brier_score") - 0.04).abs() < 1.0e-12);
        assert_eq!(receipt.exact_correct, Some(true));
    }

    #[test]
    fn wrong_confident_assertion_is_penalized() {
        let subject = episode(ReasoningOutcome::Asserted {
            value: "A".into(),
            confidence: 0.9,
        });
        let receipt = evaluate_episode(
            &subject,
            "eval-lineage",
            &EpisodeJudgment {
                exact_correct: Some(false),
                task_score: None,
            },
        );
        let receipt = match receipt {
            Ok(value) => value,
            Err(err) => panic!("evaluation must succeed: {err}"),
        };
        assert_eq!(metric(&receipt, "exact_accuracy"), 0.0);
        assert!((metric(&receipt, "brier_score") - 0.81).abs() < 1.0e-12);
        assert!(metric(&receipt, "log_loss") > 2.0);
    }

    #[test]
    fn abstention_reduces_coverage_without_fabricating_calibration() {
        let subject = episode(ReasoningOutcome::Abstained {
            reason: AbstentionReason::InsufficientEvidence,
            answerability: 0.1,
        });
        let receipt = evaluate_episode(
            &subject,
            "eval-lineage",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
        );
        let receipt = match receipt {
            Ok(value) => value,
            Err(err) => panic!("evaluation must succeed: {err}"),
        };
        assert_eq!(metric(&receipt, "coverage"), 0.0);
        assert_eq!(metric(&receipt, "exact_accuracy"), 0.0);
        assert!(receipt.metrics.iter().all(|m| m.name != "brier_score"));
        assert_eq!(receipt.exact_correct, Some(false));
    }

    #[test]
    fn aggregate_keeps_overall_and_selective_accuracy_distinct() {
        let correct = episode(ReasoningOutcome::Asserted {
            value: "A".into(),
            confidence: 0.9,
        });
        let wrong = episode(ReasoningOutcome::Asserted {
            value: "B".into(),
            confidence: 0.6,
        });
        let abstained = episode(ReasoningOutcome::Abstained {
            reason: AbstentionReason::Unidentified,
            answerability: 0.2,
        });

        let judgments = [
            (&correct, Some(true)),
            (&wrong, Some(false)),
            (&abstained, Some(true)),
        ];
        let mut receipts = Vec::new();
        for (subject, exact_correct) in judgments {
            match evaluate_episode(
                subject,
                "eval-lineage",
                &EpisodeJudgment {
                    exact_correct,
                    task_score: None,
                },
            ) {
                Ok(value) => receipts.push(value),
                Err(err) => panic!("evaluation must succeed: {err}"),
            }
        }

        let aggregate = aggregate_receipts(&receipts);
        assert_eq!(aggregate.episodes, 3);
        assert_eq!(aggregate.covered, 2);
        assert_eq!(aggregate.exact_correct, 1);
        assert_eq!(aggregate.exact_accuracy, Some(1.0 / 3.0));
        assert_eq!(aggregate.selective_accuracy, Some(0.5));
        assert!((aggregate.coverage - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn task_native_score_is_preserved_without_becoming_a_composite_index() {
        let subject = episode(ReasoningOutcome::Asserted {
            value: "candidate".into(),
            confidence: 0.5,
        });
        let receipt = evaluate_episode(
            &subject,
            "eval-lineage",
            &EpisodeJudgment {
                exact_correct: None,
                task_score: Some(TaskScore {
                    name: "partial_credit".into(),
                    value: 0.75,
                    unit: "fraction".into(),
                }),
            },
        );
        let receipt = match receipt {
            Ok(value) => value,
            Err(err) => panic!("evaluation must succeed: {err}"),
        };
        assert_eq!(metric(&receipt, "task.partial_credit"), 0.75);
        assert_eq!(receipt.exact_correct, None);
    }

    #[test]
    fn wilson_interval_is_bounded_and_contains_observed_rate() {
        let interval = wilson_interval_95(7, 10);
        assert!((0.0..=1.0).contains(&interval.lower));
        assert!((0.0..=1.0).contains(&interval.upper));
        assert!(interval.lower < 0.7);
        assert!(interval.upper > 0.7);
    }
}
