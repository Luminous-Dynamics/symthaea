// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Selective prediction and abstention scoring.
//!
//! This module is a reusable evaluator/scorer, not yet a complete task adapter.
//! It distinguishes answer quality from the decision to answer, abstain, or
//! request information. It also scores the counterfactual correctness of an
//! answer on every item so confidence calibration is not conditioned only on
//! the cases where the agent chose to answer.

use serde::{Deserialize, Serialize};

/// Terminal decision made on one task item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionAction {
    Answer,
    Abstain,
    RequestInformation,
}

/// One evaluator-annotated selective-prediction observation.
///
/// `answer_correct_if_given` is the evaluator's counterfactual ground truth:
/// whether a direct answer would have been correct, even if the policy abstained
/// or asked for more information. For an unanswerable or false-premise item it
/// must be false. `confidence` is the predicted probability that a direct
/// answer would be correct, including on abstention decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectivePredictionTrial {
    pub is_answerable: bool,
    pub action: DecisionAction,
    pub answer_correct_if_given: bool,
    pub confidence: f64,
    /// External or manifest-defined cost of this decision, in declared units.
    pub cost_units: f64,
}

/// One point in a confidence-ranked risk/coverage curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskCoveragePoint {
    /// Fraction of all items retained at this confidence threshold.
    pub coverage: f64,
    /// Error rate among retained items if the system answered all of them.
    pub risk: f64,
    /// Lowest confidence included in this prefix.
    pub confidence_threshold: f64,
}

/// Evaluation summary for a selective-prediction policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectivePredictionMetrics {
    pub samples: usize,
    pub answer_rate: f64,
    pub abstain_rate: f64,
    pub information_request_rate: f64,
    /// Undefined when the policy never answered.
    pub accuracy_when_answered: Option<f64>,
    /// Undefined when the policy never answered.
    pub risk_when_answered: Option<f64>,
    /// Fraction of unanswerable items handled without a direct answer.
    pub appropriate_nonanswer_rate_on_unanswerable: Option<f64>,
    /// Fraction of answerable items unnecessarily withheld/deferred.
    pub false_nonanswer_rate_on_answerable: Option<f64>,
    /// Fraction of unanswerable items answered directly.
    pub inappropriate_answer_rate_on_unanswerable: Option<f64>,
    /// Calibration of the counterfactual direct-answer confidence over all items.
    pub brier_score: f64,
    pub expected_calibration_error: f64,
    pub mean_confidence: f64,
    pub mean_cost_units: f64,
    /// Discrete right-Riemann estimate of confidence-ranked risk over coverage.
    pub area_under_risk_coverage_curve: f64,
    pub risk_coverage_curve: Vec<RiskCoveragePoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectivePredictionError {
    EmptyTrials,
    NonFiniteConfidence { index: usize },
    ConfidenceOutOfRange { index: usize, value: f64 },
    InvalidCost { index: usize, value: f64 },
    UnanswerableItemMarkedCorrect { index: usize },
}

impl std::fmt::Display for SelectivePredictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTrials => write!(f, "selective-prediction scoring requires at least one trial"),
            Self::NonFiniteConfidence { index } => {
                write!(f, "confidence at trial {index} is not finite")
            }
            Self::ConfidenceOutOfRange { index, value } => {
                write!(f, "confidence at trial {index} ({value}) is outside [0, 1]")
            }
            Self::InvalidCost { index, value } => {
                write!(f, "cost at trial {index} ({value}) must be finite and non-negative")
            }
            Self::UnanswerableItemMarkedCorrect { index } => write!(
                f,
                "unanswerable trial {index} cannot have a correct direct-answer outcome"
            ),
        }
    }
}

impl std::error::Error for SelectivePredictionError {}

impl SelectivePredictionMetrics {
    /// Score an ordered set of evaluator-annotated decisions.
    pub fn from_trials(
        trials: &[SelectivePredictionTrial],
    ) -> Result<Self, SelectivePredictionError> {
        if trials.is_empty() {
            return Err(SelectivePredictionError::EmptyTrials);
        }

        for (index, trial) in trials.iter().enumerate() {
            if !trial.confidence.is_finite() {
                return Err(SelectivePredictionError::NonFiniteConfidence { index });
            }
            if !(0.0..=1.0).contains(&trial.confidence) {
                return Err(SelectivePredictionError::ConfidenceOutOfRange {
                    index,
                    value: trial.confidence,
                });
            }
            if !trial.cost_units.is_finite() || trial.cost_units < 0.0 {
                return Err(SelectivePredictionError::InvalidCost {
                    index,
                    value: trial.cost_units,
                });
            }
            if !trial.is_answerable && trial.answer_correct_if_given {
                return Err(SelectivePredictionError::UnanswerableItemMarkedCorrect { index });
            }
        }

        let n = trials.len();
        let answers = trials
            .iter()
            .filter(|trial| trial.action == DecisionAction::Answer)
            .count();
        let abstentions = trials
            .iter()
            .filter(|trial| trial.action == DecisionAction::Abstain)
            .count();
        let information_requests = trials
            .iter()
            .filter(|trial| trial.action == DecisionAction::RequestInformation)
            .count();

        let answerable = trials.iter().filter(|trial| trial.is_answerable).count();
        let unanswerable = n - answerable;
        let answered_correct = trials
            .iter()
            .filter(|trial| {
                trial.action == DecisionAction::Answer
                    && trial.is_answerable
                    && trial.answer_correct_if_given
            })
            .count();
        let answered_wrong = answers - answered_correct;
        let nonanswers = n - answers;
        let answerable_nonanswers = trials
            .iter()
            .filter(|trial| {
                trial.is_answerable && trial.action != DecisionAction::Answer
            })
            .count();
        let unanswerable_nonanswers = trials
            .iter()
            .filter(|trial| {
                !trial.is_answerable && trial.action != DecisionAction::Answer
            })
            .count();
        let unanswerable_answers = trials
            .iter()
            .filter(|trial| {
                !trial.is_answerable && trial.action == DecisionAction::Answer
            })
            .count();

        let outcomes: Vec<f64> = trials
            .iter()
            .map(|trial| {
                if trial.is_answerable && trial.answer_correct_if_given {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let confidences: Vec<f64> = trials.iter().map(|trial| trial.confidence).collect();
        let mean_confidence = confidences.iter().sum::<f64>() / n as f64;
        let brier_score = confidences
            .iter()
            .zip(&outcomes)
            .map(|(confidence, outcome)| (confidence - outcome).powi(2))
            .sum::<f64>()
            / n as f64;
        let expected_calibration_error = expected_calibration_error(&confidences, &outcomes, 10);
        let mean_cost_units =
            trials.iter().map(|trial| trial.cost_units).sum::<f64>() / n as f64;

        let accuracy_when_answered = (answers > 0).then_some(answered_correct as f64 / answers as f64);
        let risk_when_answered = accuracy_when_answered.map(|accuracy| 1.0 - accuracy);
        let appropriate_nonanswer_rate_on_unanswerable =
            (unanswerable > 0).then_some(unanswerable_nonanswers as f64 / unanswerable as f64);
        let false_nonanswer_rate_on_answerable =
            (answerable > 0).then_some(answerable_nonanswers as f64 / answerable as f64);
        let inappropriate_answer_rate_on_unanswerable =
            (unanswerable > 0).then_some(unanswerable_answers as f64 / unanswerable as f64);

        let (risk_coverage_curve, area_under_risk_coverage_curve) =
            confidence_ranked_risk_curve(trials);

        Ok(Self {
            samples: n,
            answer_rate: answers as f64 / n as f64,
            abstain_rate: abstentions as f64 / n as f64,
            information_request_rate: information_requests as f64 / n as f64,
            accuracy_when_answered,
            risk_when_answered,
            appropriate_nonanswer_rate_on_unanswerable,
            false_nonanswer_rate_on_answerable,
            inappropriate_answer_rate_on_unanswerable,
            brier_score,
            expected_calibration_error,
            mean_confidence,
            mean_cost_units,
            area_under_risk_coverage_curve,
            risk_coverage_curve,
        })
    }
}

fn expected_calibration_error(confidences: &[f64], outcomes: &[f64], bins: usize) -> f64 {
    if confidences.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut confidence_sums = vec![0.0; bins];
    let mut outcome_sums = vec![0.0; bins];
    let mut counts = vec![0usize; bins];
    for (confidence, outcome) in confidences.iter().zip(outcomes) {
        let bin = ((confidence * bins as f64) as usize).min(bins - 1);
        confidence_sums[bin] += confidence;
        outcome_sums[bin] += outcome;
        counts[bin] += 1;
    }

    counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(bin, count)| {
            let mean_confidence = confidence_sums[bin] / *count as f64;
            let mean_outcome = outcome_sums[bin] / *count as f64;
            (*count as f64 / confidences.len() as f64)
                * (mean_confidence - mean_outcome).abs()
        })
        .sum()
}

fn confidence_ranked_risk_curve(
    trials: &[SelectivePredictionTrial],
) -> (Vec<RiskCoveragePoint>, f64) {
    let mut ranked: Vec<(usize, f64, bool)> = trials
        .iter()
        .enumerate()
        .map(|(index, trial)| {
            (
                index,
                trial.confidence,
                trial.is_answerable && trial.answer_correct_if_given,
            )
        })
        .collect();
    // Stable deterministic tie-breaking preserves the evaluator's input order.
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));

    let mut errors = 0usize;
    let mut curve = Vec::with_capacity(ranked.len());
    for (rank, (_, confidence, correct)) in ranked.iter().enumerate() {
        if !correct {
            errors += 1;
        }
        curve.push(RiskCoveragePoint {
            coverage: (rank + 1) as f64 / ranked.len() as f64,
            risk: errors as f64 / (rank + 1) as f64,
            confidence_threshold: *confidence,
        });
    }

    let area = curve.iter().map(|point| point.risk / ranked.len() as f64).sum();
    (curve, area)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(
        is_answerable: bool,
        action: DecisionAction,
        correct: bool,
        confidence: f64,
    ) -> SelectivePredictionTrial {
        SelectivePredictionTrial {
            is_answerable,
            action,
            answer_correct_if_given: correct,
            confidence,
            cost_units: 1.0,
        }
    }

    #[test]
    fn correct_answers_and_appropriate_abstentions_are_separated() {
        let trials = [
            trial(true, DecisionAction::Answer, true, 0.95),
            trial(true, DecisionAction::Answer, true, 0.90),
            trial(false, DecisionAction::Abstain, false, 0.10),
        ];
        let score = SelectivePredictionMetrics::from_trials(&trials).expect("valid trials");
        assert!((score.answer_rate - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(score.accuracy_when_answered, Some(1.0));
        assert_eq!(score.risk_when_answered, Some(0.0));
        assert_eq!(score.appropriate_nonanswer_rate_on_unanswerable, Some(1.0));
        assert_eq!(score.inappropriate_answer_rate_on_unanswerable, Some(0.0));
        assert_eq!(score.false_nonanswer_rate_on_answerable, Some(0.0));
    }

    #[test]
    fn confident_answers_to_unanswerable_items_increase_selective_risk() {
        let trials = [
            trial(true, DecisionAction::Answer, true, 0.95),
            trial(false, DecisionAction::Answer, false, 0.99),
            trial(false, DecisionAction::Abstain, false, 0.10),
        ];
        let score = SelectivePredictionMetrics::from_trials(&trials).expect("valid trials");
        assert_eq!(score.accuracy_when_answered, Some(0.5));
        assert_eq!(score.inappropriate_answer_rate_on_unanswerable, Some(0.5));
        assert!(score.brier_score > 0.0);
    }

    #[test]
    fn requests_for_information_are_not_counted_as_answers_or_abstentions() {
        let trials = [
            trial(true, DecisionAction::RequestInformation, true, 0.55),
            trial(false, DecisionAction::Abstain, false, 0.15),
        ];
        let score = SelectivePredictionMetrics::from_trials(&trials).expect("valid trials");
        assert_eq!(score.answer_rate, 0.0);
        assert_eq!(score.abstain_rate, 0.5);
        assert_eq!(score.information_request_rate, 0.5);
        assert_eq!(score.accuracy_when_answered, None);
        assert_eq!(score.appropriate_nonanswer_rate_on_unanswerable, Some(1.0));
        assert_eq!(score.false_nonanswer_rate_on_answerable, Some(1.0));
    }

    #[test]
    fn risk_coverage_curve_is_confidence_ranked_and_complete() {
        let trials = [
            trial(true, DecisionAction::Answer, false, 0.99),
            trial(true, DecisionAction::Answer, true, 0.90),
            trial(false, DecisionAction::Abstain, false, 0.10),
        ];
        let score = SelectivePredictionMetrics::from_trials(&trials).expect("valid trials");
        assert_eq!(score.risk_coverage_curve.len(), trials.len());
        assert_eq!(score.risk_coverage_curve[0].confidence_threshold, 0.99);
        assert_eq!(score.risk_coverage_curve[0].risk, 1.0);
        assert!((score.area_under_risk_coverage_curve - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_confidence_cost_and_unanswerable_ground_truth() {
        assert!(matches!(
            SelectivePredictionMetrics::from_trials(&[trial(true, DecisionAction::Answer, true, 1.2)]),
            Err(SelectivePredictionError::ConfidenceOutOfRange { .. })
        ));
        let mut negative_cost = trial(true, DecisionAction::Answer, true, 0.8);
        negative_cost.cost_units = -1.0;
        assert!(matches!(
            SelectivePredictionMetrics::from_trials(&[negative_cost]),
            Err(SelectivePredictionError::InvalidCost { .. })
        ));
        assert!(matches!(
            SelectivePredictionMetrics::from_trials(&[trial(false, DecisionAction::Abstain, true, 0.2)]),
            Err(SelectivePredictionError::UnanswerableItemMarkedCorrect { .. })
        ));
    }
}
