// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 consequence scoring.
//!
//! This scorer makes the benchmark pathology explicit: a target can obtain
//! excellent whole-state accuracy by copying a mostly unchanged pre-state
//! while completely failing to predict the fields that actually changed.
//! Changed-field metrics are therefore primary; whole-state accuracy is only
//! descriptive.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2050>

use super::hidden_world::{PublicAction, PublicObservation, PublicValue};

/// Target response committed before the evaluator reveals the fresh outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsequencePrediction {
    pub action: PublicAction,
    pub outcome: PredictionOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionOutcome {
    Predicted { fields: Vec<PublicValue> },
    AbstainInsufficientEvidence,
    OutOfQualifiedDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsequenceScoringError {
    PreAndOutcomeFieldCountMismatch,
    PredictionFieldCountMismatch,
}

/// Raw counts are retained so aggregate statistics never become the only
/// surviving record of per-field behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsequenceMetrics {
    pub field_count: usize,
    pub actual_changed: usize,
    pub predicted_changed: usize,
    pub true_positive_changes: usize,
    pub false_positive_changes: usize,
    pub missed_changes: usize,
    pub correct_changed_values: usize,
    pub correct_unchanged_values: usize,
    pub correct_full_state_values: usize,
    pub changed_field_precision: Option<f64>,
    pub changed_field_recall: Option<f64>,
    pub changed_field_f1: Option<f64>,
    pub changed_value_accuracy: Option<f64>,
    pub unchanged_state_preservation: Option<f64>,
    pub full_state_accuracy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsequenceScore {
    Scored(ConsequenceMetrics),
    AbstainedInsufficientEvidence,
    OutOfQualifiedDomain,
}

/// Score one prospective prediction against a fresh evaluator-owned outcome.
///
/// The caller is responsible for proving that `prediction` was committed
/// before `actual_post` became visible to the target. This pure function does
/// not mint prospective-evidence authority by itself.
pub fn score_consequence(
    pre: &PublicObservation,
    prediction: &ConsequencePrediction,
    actual_post: &PublicObservation,
) -> Result<ConsequenceScore, ConsequenceScoringError> {
    if pre.fields.len() != actual_post.fields.len() {
        return Err(ConsequenceScoringError::PreAndOutcomeFieldCountMismatch);
    }

    let predicted_fields = match &prediction.outcome {
        PredictionOutcome::Predicted { fields } => fields,
        PredictionOutcome::AbstainInsufficientEvidence => {
            return Ok(ConsequenceScore::AbstainedInsufficientEvidence);
        }
        PredictionOutcome::OutOfQualifiedDomain => {
            return Ok(ConsequenceScore::OutOfQualifiedDomain);
        }
    };

    if predicted_fields.len() != pre.fields.len() {
        return Err(ConsequenceScoringError::PredictionFieldCountMismatch);
    }

    let mut actual_changed = 0;
    let mut predicted_changed = 0;
    let mut true_positive_changes = 0;
    let mut false_positive_changes = 0;
    let mut missed_changes = 0;
    let mut correct_changed_values = 0;
    let mut correct_unchanged_values = 0;
    let mut correct_full_state_values = 0;

    for ((before, predicted), actual) in pre
        .fields
        .iter()
        .zip(predicted_fields.iter())
        .zip(actual_post.fields.iter())
    {
        let did_change = before != actual;
        let predicted_change = before != predicted;

        actual_changed += usize::from(did_change);
        predicted_changed += usize::from(predicted_change);

        match (did_change, predicted_change) {
            (true, true) => true_positive_changes += 1,
            (false, true) => false_positive_changes += 1,
            (true, false) => missed_changes += 1,
            (false, false) => {}
        }

        if did_change && predicted == actual {
            correct_changed_values += 1;
        }
        if !did_change && predicted == actual {
            correct_unchanged_values += 1;
        }
        if predicted == actual {
            correct_full_state_values += 1;
        }
    }

    let unchanged = pre.fields.len() - actual_changed;
    let changed_field_precision = if predicted_changed > 0 {
        Some(true_positive_changes as f64 / predicted_changed as f64)
    } else if actual_changed > 0 {
        Some(0.0)
    } else {
        None
    };
    let changed_field_recall = (actual_changed > 0)
        .then_some(true_positive_changes as f64 / actual_changed as f64);
    let changed_field_f1 = match (changed_field_precision, changed_field_recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    let changed_value_accuracy = (actual_changed > 0)
        .then_some(correct_changed_values as f64 / actual_changed as f64);
    let unchanged_state_preservation = (unchanged > 0)
        .then_some(correct_unchanged_values as f64 / unchanged as f64);
    let full_state_accuracy = if pre.fields.is_empty() {
        0.0
    } else {
        correct_full_state_values as f64 / pre.fields.len() as f64
    };

    Ok(ConsequenceScore::Scored(ConsequenceMetrics {
        field_count: pre.fields.len(),
        actual_changed,
        predicted_changed,
        true_positive_changes,
        false_positive_changes,
        missed_changes,
        correct_changed_values,
        correct_unchanged_values,
        correct_full_state_values,
        changed_field_precision,
        changed_field_recall,
        changed_field_f1,
        changed_value_accuracy,
        unchanged_state_preservation,
        full_state_accuracy,
    }))
}

/// Canonical shortcut baseline: simply predicts the pre-state will persist.
pub fn copy_current_state_baseline(
    pre: &PublicObservation,
    action: PublicAction,
) -> ConsequencePrediction {
    ConsequencePrediction {
        action,
        outcome: PredictionOutcome::Predicted {
            fields: pre.fields.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(values: impl IntoIterator<Item = PublicValue>) -> PublicObservation {
        PublicObservation {
            step: 0,
            fields: values.into_iter().collect(),
        }
    }

    fn predicted(values: impl IntoIterator<Item = PublicValue>) -> ConsequencePrediction {
        ConsequencePrediction {
            action: PublicAction::NoOp,
            outcome: PredictionOutcome::Predicted {
                fields: values.into_iter().collect(),
            },
        }
    }

    fn metrics(score: ConsequenceScore) -> ConsequenceMetrics {
        match score {
            ConsequenceScore::Scored(metrics) => metrics,
            other => panic!("expected scored consequence, got {other:?}"),
        }
    }

    #[test]
    fn copy_baseline_can_look_excellent_while_missing_every_real_change() {
        let pre = obs(std::iter::repeat_n(PublicValue::Count(0), 100));
        let mut actual_values = pre.fields.clone();
        actual_values[73] = PublicValue::Count(1);
        let actual = obs(actual_values);
        let copy = copy_current_state_baseline(&pre, PublicAction::NoOp);

        let score = metrics(score_consequence(&pre, &copy, &actual).unwrap());
        assert_eq!(score.full_state_accuracy, 0.99);
        assert_eq!(score.actual_changed, 1);
        assert_eq!(score.predicted_changed, 0);
        assert_eq!(score.changed_field_recall, Some(0.0));
        assert_eq!(score.changed_field_f1, Some(0.0));
        assert_eq!(score.changed_value_accuracy, Some(0.0));
        assert_eq!(score.unchanged_state_preservation, Some(1.0));
    }

    #[test]
    fn perfect_prediction_scores_perfectly_on_changed_and_unchanged_fields() {
        let pre = obs([
            PublicValue::Bit(false),
            PublicValue::Count(2),
            PublicValue::Count(8),
        ]);
        let actual = obs([
            PublicValue::Bit(true),
            PublicValue::Count(5),
            PublicValue::Count(8),
        ]);
        let prediction = predicted(actual.fields.clone());

        let score = metrics(score_consequence(&pre, &prediction, &actual).unwrap());
        assert_eq!(score.changed_field_precision, Some(1.0));
        assert_eq!(score.changed_field_recall, Some(1.0));
        assert_eq!(score.changed_field_f1, Some(1.0));
        assert_eq!(score.changed_value_accuracy, Some(1.0));
        assert_eq!(score.unchanged_state_preservation, Some(1.0));
        assert_eq!(score.full_state_accuracy, 1.0);
    }

    #[test]
    fn false_change_and_missed_change_remain_distinct() {
        let pre = obs([
            PublicValue::Count(0),
            PublicValue::Count(0),
            PublicValue::Count(0),
        ]);
        let actual = obs([
            PublicValue::Count(1),
            PublicValue::Count(0),
            PublicValue::Count(0),
        ]);
        let prediction = predicted([
            PublicValue::Count(0),
            PublicValue::Count(1),
            PublicValue::Count(0),
        ]);

        let score = metrics(score_consequence(&pre, &prediction, &actual).unwrap());
        assert_eq!(score.true_positive_changes, 0);
        assert_eq!(score.false_positive_changes, 1);
        assert_eq!(score.missed_changes, 1);
        assert_eq!(score.changed_field_precision, Some(0.0));
        assert_eq!(score.changed_field_recall, Some(0.0));
    }

    #[test]
    fn predicting_a_change_is_not_enough_if_the_new_value_is_wrong() {
        let pre = obs([PublicValue::Count(0)]);
        let actual = obs([PublicValue::Count(2)]);
        let prediction = predicted([PublicValue::Count(1)]);

        let score = metrics(score_consequence(&pre, &prediction, &actual).unwrap());
        assert_eq!(score.changed_field_precision, Some(1.0));
        assert_eq!(score.changed_field_recall, Some(1.0));
        assert_eq!(score.changed_value_accuracy, Some(0.0));
        assert_eq!(score.full_state_accuracy, 0.0);
    }

    #[test]
    fn no_change_episode_does_not_award_a_fake_perfect_change_score() {
        let pre = obs([PublicValue::Bit(true), PublicValue::Count(3)]);
        let actual = pre.clone();
        let copy = copy_current_state_baseline(&pre, PublicAction::NoOp);

        let score = metrics(score_consequence(&pre, &copy, &actual).unwrap());
        assert_eq!(score.actual_changed, 0);
        assert_eq!(score.changed_field_precision, None);
        assert_eq!(score.changed_field_recall, None);
        assert_eq!(score.changed_field_f1, None);
        assert_eq!(score.changed_value_accuracy, None);
        assert_eq!(score.full_state_accuracy, 1.0);
    }

    #[test]
    fn abstention_is_not_reinterpreted_as_no_change_prediction() {
        let pre = obs([PublicValue::Count(0)]);
        let actual = obs([PublicValue::Count(1)]);
        let prediction = ConsequencePrediction {
            action: PublicAction::NoOp,
            outcome: PredictionOutcome::AbstainInsufficientEvidence,
        };

        assert_eq!(
            score_consequence(&pre, &prediction, &actual),
            Ok(ConsequenceScore::AbstainedInsufficientEvidence)
        );
    }

    #[test]
    fn out_of_domain_is_distinct_from_abstention_and_scored_failure() {
        let pre = obs([PublicValue::Count(0)]);
        let actual = obs([PublicValue::Count(1)]);
        let prediction = ConsequencePrediction {
            action: PublicAction::NoOp,
            outcome: PredictionOutcome::OutOfQualifiedDomain,
        };

        assert_eq!(
            score_consequence(&pre, &prediction, &actual),
            Ok(ConsequenceScore::OutOfQualifiedDomain)
        );
    }

    #[test]
    fn scorer_rejects_field_count_mismatch_instead_of_truncating() {
        let pre = obs([PublicValue::Count(0), PublicValue::Count(1)]);
        let actual = obs([PublicValue::Count(0)]);
        let prediction = predicted([PublicValue::Count(0), PublicValue::Count(1)]);
        assert_eq!(
            score_consequence(&pre, &prediction, &actual),
            Err(ConsequenceScoringError::PreAndOutcomeFieldCountMismatch)
        );

        let actual = pre.clone();
        let short_prediction = predicted([PublicValue::Count(0)]);
        assert_eq!(
            score_consequence(&pre, &short_prediction, &actual),
            Err(ConsequenceScoringError::PredictionFieldCountMismatch)
        );
    }

    #[test]
    fn value_type_mismatch_is_scored_as_wrong_not_coerced() {
        let pre = obs([PublicValue::Bit(false)]);
        let actual = obs([PublicValue::Bit(true)]);
        let prediction = predicted([PublicValue::Count(1)]);
        let score = metrics(score_consequence(&pre, &prediction, &actual).unwrap());
        assert_eq!(score.changed_field_recall, Some(1.0));
        assert_eq!(score.changed_value_accuracy, Some(0.0));
    }
}
