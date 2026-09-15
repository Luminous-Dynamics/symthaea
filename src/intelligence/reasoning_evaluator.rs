// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, benchmark-agnostic evaluation for reasoning episodes.
//!
//! Benchmark adapters own task semantics and produce [`EpisodeJudgment`] values. This module
//! converts those judgments plus immutable subject episodes into common qualification metrics.
//! Metrics remain decomposed rather than being collapsed into an opaque intelligence scalar.

use super::reasoning_qualification::{
    QualificationMetric, QualificationValidationError, ReasoningEpisode, ReasoningOutcome,
    ReasoningQualificationReceipt,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

/// Initial evaluator contract. Increment whenever metric semantics change.
pub const REASONING_EVALUATOR_VERSION: &str = "rq-evaluator-v1";
pub const REASONING_EVALUATOR_ID: &str = "symthaea-reasoning-evaluator";

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
    DuplicateEpisode(String),
    MixedEvaluator {
        expected: String,
        found: String,
    },
    MixedEvaluatorVersion {
        expected: String,
        found: String,
    },
    Validation(QualificationValidationError),
}

impl fmt::Display for ReasoningEvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEvaluationLineage => write!(f, "evaluation lineage hash must not be empty"),
            Self::EmptyTaskScoreName => write!(f, "task score name must not be empty"),
            Self::NonFiniteTaskScore => write!(f, "task score must be finite"),
            Self::DuplicateEpisode(id) => {
                write!(f, "episode `{id}` appears more than once in one aggregate")
            }
            Self::MixedEvaluator { expected, found } => write!(
                f,
                "aggregate mixes evaluator identities: expected `{expected}`, found `{found}`"
            ),
            Self::MixedEvaluatorVersion { expected, found } => write!(
                f,
                "aggregate mixes evaluator versions: expected `{expected}`, found `{found}`"
            ),
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

    let asserted = matches!(&episode.outcome, ReasoningOutcome::Asserted { .. });
    let mut metrics = vec![QualificationMetric {
        name: "coverage".into(),
        value: if asserted { 1.0 } else { 0.0 },
        unit: "fraction".into(),
    }];

    if let Some(correct) = judgment.exact_correct {
        metrics.push(QualificationMetric {
            name: "exact_accuracy".into(),
            value: if asserted && correct { 1.0 } else { 0.0 },
            unit: "fraction".into(),
        });

        if let ReasoningOutcome::Asserted { confidence, .. } = &episode.outcome {
            let target = if correct { 1.0 } else { 0.0 };
            let brier = (*confidence - target).powi(2);
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
        REASONING_EVALUATOR_ID,
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

/// Aggregate common metrics from validated evaluator receipts.
///
/// The aggregate fails closed if any receipt identity is invalid, an episode is counted twice,
/// or evaluator identities/versions are mixed. Evaluation-lineage hashes may differ because a
/// benchmark can bind each individual problem to a distinct immutable lineage.
pub fn aggregate_receipts(
    receipts: &[ReasoningQualificationReceipt],
) -> Result<CapabilitySlice, ReasoningEvaluatorError> {
    let mut episode_ids = HashSet::with_capacity(receipts.len());
    let mut expected_evaluator: Option<&str> = None;
    let mut expected_version: Option<&str> = None;

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
        receipt.validate()?;

        if !episode_ids.insert(receipt.episode_id.0.as_str()) {
            return Err(ReasoningEvaluatorError::DuplicateEpisode(
                receipt.episode_id.0.clone(),
            ));
        }

        if let Some(expected) = expected_evaluator {
            if receipt.evaluator != expected {
                return Err(ReasoningEvaluatorError::MixedEvaluator {
                    expected: expected.into(),
                    found: receipt.evaluator.clone(),
                });
            }
        } else {
            expected_evaluator = Some(&receipt.evaluator);
        }

        if let Some(expected) = expected_version {
            if receipt.evaluator_version != expected {
                return Err(ReasoningEvaluatorError::MixedEvaluatorVersion {
                    expected: expected.into(),
                    found: receipt.evaluator_version.clone(),
                });
            }
        } else {
            expected_version = Some(&receipt.evaluator_version);
        }

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

    let episodes = receipts.len();
    let exact_accuracy = ratio(exact_correct, exact_judged);
    Ok(CapabilitySlice {
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
    })
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
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()
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
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn mean(sum: f64, count: usize) -> Option<f64> {
    (count != 0).then(|| sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::reasoning_qualification::{
        AbstentionReason, ReasoningDomain, ReasoningProblemRef, ResourceUsage,
    };

    fn episode(problem_id: &str, outcome: ReasoningOutcome) -> ReasoningEpisode {
        ReasoningEpisode::new(
            "subject",
            "config",
            ReasoningDomain::Logic,
            ReasoningProblemRef {
                benchmark: "logic".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: problem_id.into(),
                problem_hash: format!("sha256:{problem_id}"),
            },
            vec![],
            vec![],
            vec![],
            outcome,
            ResourceUsage::default(),
        )
        .unwrap_or_else(|err| panic!("test episode must validate: {err}"))
    }

    fn evaluate(
        subject: &ReasoningEpisode,
        exact_correct: Option<bool>,
    ) -> ReasoningQualificationReceipt {
        evaluate_episode(
            subject,
            "eval-lineage",
            &EpisodeJudgment {
                exact_correct,
                task_score: None,
            },
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"))
    }

    fn metric(receipt: &ReasoningQualificationReceipt, name: &str) -> f64 {
        receipt
            .metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap_or_else(|| panic!("missing metric {name}"))
            .value
    }

    #[test]
    fn correct_assertion_scores_accuracy_and_calibration() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "A".into(),
                confidence: 0.8,
            },
        );
        let receipt = evaluate(&subject, Some(true));
        assert_eq!(metric(&receipt, "coverage"), 1.0);
        assert_eq!(metric(&receipt, "exact_accuracy"), 1.0);
        assert_eq!(metric(&receipt, "selective_accuracy"), 1.0);
        assert!((metric(&receipt, "brier_score") - 0.04).abs() < 1.0e-12);
        assert_eq!(receipt.exact_correct, Some(true));
    }

    #[test]
    fn wrong_confident_assertion_is_penalized() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "A".into(),
                confidence: 0.9,
            },
        );
        let receipt = evaluate(&subject, Some(false));
        assert_eq!(metric(&receipt, "exact_accuracy"), 0.0);
        assert!((metric(&receipt, "brier_score") - 0.81).abs() < 1.0e-12);
        assert!(metric(&receipt, "log_loss") > 2.0);
    }

    #[test]
    fn abstention_reduces_coverage_without_fabricating_calibration() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::InsufficientEvidence,
                answerability: 0.1,
            },
        );
        let receipt = evaluate(&subject, Some(true));
        assert_eq!(metric(&receipt, "coverage"), 0.0);
        assert_eq!(metric(&receipt, "exact_accuracy"), 0.0);
        assert!(receipt.metrics.iter().all(|metric| metric.name != "brier_score"));
        assert_eq!(receipt.exact_correct, Some(false));
    }

    #[test]
    fn aggregate_keeps_overall_and_selective_accuracy_distinct() {
        let correct = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "A".into(),
                confidence: 0.9,
            },
        );
        let wrong = episode(
            "p2",
            ReasoningOutcome::Asserted {
                value: "B".into(),
                confidence: 0.6,
            },
        );
        let abstained = episode(
            "p3",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::Unidentified,
                answerability: 0.2,
            },
        );
        let receipts = vec![
            evaluate(&correct, Some(true)),
            evaluate(&wrong, Some(false)),
            evaluate(&abstained, Some(true)),
        ];
        let aggregate = aggregate_receipts(&receipts)
            .unwrap_or_else(|err| panic!("aggregate must succeed: {err}"));
        assert_eq!(aggregate.episodes, 3);
        assert_eq!(aggregate.covered, 2);
        assert_eq!(aggregate.exact_correct, 1);
        assert_eq!(aggregate.exact_accuracy, Some(1.0 / 3.0));
        assert_eq!(aggregate.selective_accuracy, Some(0.5));
        assert!((aggregate.coverage - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn aggregate_rejects_duplicate_episode() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "A".into(),
                confidence: 0.7,
            },
        );
        let receipt = evaluate(&subject, Some(true));
        let err = aggregate_receipts(&[receipt.clone(), receipt])
            .expect_err("duplicate episode must fail closed");
        assert!(matches!(err, ReasoningEvaluatorError::DuplicateEpisode(_)));
    }

    #[test]
    fn aggregate_rejects_tampered_receipt() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "A".into(),
                confidence: 0.7,
            },
        );
        let mut receipt = evaluate(&subject, Some(true));
        receipt.metrics[0].value = 0.0;
        let err = aggregate_receipts(&[receipt]).expect_err("tampered receipt must fail closed");
        assert!(matches!(
            err,
            ReasoningEvaluatorError::Validation(
                QualificationValidationError::ReceiptIdMismatch
            )
        ));
    }

    #[test]
    fn task_native_score_is_preserved_without_becoming_a_composite_index() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "candidate".into(),
                confidence: 0.5,
            },
        );
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
        )
        .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
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
