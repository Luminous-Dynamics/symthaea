// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibration-efficiency evaluation for preregistered, paired BCI comparisons.
//!
//! Evaluates one candidate budget against a higher-budget reference on the same
//! frozen participants. It does not search the evaluation set for a winning budget
//! and does not establish live-use safety.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::statistics::{
    paired_participant_bootstrap, BootstrapConfig, PairedBootstrapError,
    PairedBootstrapReport, PairedParticipantScore, SCORE_SCALE_PPM,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationBudgetScore {
    pub model_id: String,
    pub participant_id: String,
    /// Budget units must be homogeneous within a comparison, for example labelled
    /// calibration examples or elapsed calibration minutes.
    pub calibration_budget_units: u32,
    pub score_ppm: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonInferiorityDecision {
    SupportsNonInferiority,
    EvidenceOfInferiority,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationReductionReport {
    pub method_id: &'static str,
    pub baseline_model_id: String,
    pub candidate_model_id: String,
    pub budget_unit: String,
    pub baseline_calibration_budget_units: u32,
    pub candidate_calibration_budget_units: u32,
    /// Reduction × 1000; 500 permille means 50% fewer calibration units.
    pub calibration_reduction_permille: u16,
    pub participant_count: u32,
    pub baseline_mean_score_ppm: u32,
    pub candidate_mean_score_ppm: u32,
    pub observed_delta_ppm: i64,
    pub ci_lower_delta_ppm: i64,
    pub ci_upper_delta_ppm: i64,
    pub noninferiority_margin_ppm: u32,
    pub decision: NonInferiorityDecision,
    pub bootstrap: PairedBootstrapReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationEvaluationError {
    EmptyScores,
    EmptyBudgetUnit,
    EmptyParticipantInventory,
    EmptyIdentity { index: usize, field: &'static str },
    DuplicateParticipantId(String),
    TooFewParticipants { actual: usize },
    InvalidModelComparison,
    InvalidBudgetComparison { baseline_budget_units: u32, candidate_budget_units: u32 },
    InvalidNonInferiorityMargin(u32),
    UnexpectedModel(String),
    UnexpectedParticipant(String),
    UnexpectedBudget { model_id: String, expected: u32, actual: u32 },
    DuplicateScore { model_id: String, participant_id: String, budget_units: u32 },
    MissingScore { model_id: String, participant_id: String, budget_units: u32 },
    ScoreOutOfRange { index: usize, score_ppm: u32 },
    Bootstrap(PairedBootstrapError),
    CounterOverflow,
}

impl fmt::Display for CalibrationEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CalibrationEvaluationError {}

/// Evaluate one preregistered calibration reduction at matched held-out participants.
///
/// Higher scores are assumed better. The non-inferiority rule supports a positive
/// decision only when the paired percentile confidence interval's lower endpoint
/// is at least -margin. The procedure and seed are included in the returned report.
///
/// Exactly one baseline and one candidate score is required for each expected
/// participant at the declared budgets. Unexpected rows are rejected rather than
/// silently discarded or used to change the comparison denominator.
pub fn evaluate_calibration_reduction(
    scores: &[CalibrationBudgetScore],
    expected_participant_ids: &[&str],
    baseline_model_id: &str,
    candidate_model_id: &str,
    budget_unit: &str,
    baseline_budget_units: u32,
    candidate_budget_units: u32,
    noninferiority_margin_ppm: u32,
    bootstrap_config: BootstrapConfig,
) -> Result<CalibrationReductionReport, CalibrationEvaluationError> {
    if scores.is_empty() {
        return Err(CalibrationEvaluationError::EmptyScores);
    }
    if budget_unit.trim().is_empty() {
        return Err(CalibrationEvaluationError::EmptyBudgetUnit);
    }
    if baseline_model_id.trim().is_empty()
        || candidate_model_id.trim().is_empty()
        || baseline_model_id == candidate_model_id
    {
        return Err(CalibrationEvaluationError::InvalidModelComparison);
    }
    if baseline_budget_units == 0 || candidate_budget_units >= baseline_budget_units {
        return Err(CalibrationEvaluationError::InvalidBudgetComparison {
            baseline_budget_units,
            candidate_budget_units,
        });
    }
    if noninferiority_margin_ppm > SCORE_SCALE_PPM {
        return Err(CalibrationEvaluationError::InvalidNonInferiorityMargin(
            noninferiority_margin_ppm,
        ));
    }
    if expected_participant_ids.is_empty() {
        return Err(CalibrationEvaluationError::EmptyParticipantInventory);
    }

    let mut expected_participants = BTreeSet::<String>::new();
    for (index, participant_id) in expected_participant_ids.iter().copied().enumerate() {
        if participant_id.trim().is_empty() {
            return Err(CalibrationEvaluationError::EmptyIdentity {
                index,
                field: "expected_participant_id",
            });
        }
        if !expected_participants.insert(participant_id.to_owned()) {
            return Err(CalibrationEvaluationError::DuplicateParticipantId(
                participant_id.to_owned(),
            ));
        }
    }
    if expected_participants.len() < 2 {
        return Err(CalibrationEvaluationError::TooFewParticipants {
            actual: expected_participants.len(),
        });
    }

    let mut by_key = BTreeMap::<(String, String, u32), u32>::new();
    for (index, row) in scores.iter().enumerate() {
        if row.model_id.trim().is_empty() {
            return Err(CalibrationEvaluationError::EmptyIdentity {
                index,
                field: "model_id",
            });
        }
        if row.participant_id.trim().is_empty() {
            return Err(CalibrationEvaluationError::EmptyIdentity {
                index,
                field: "participant_id",
            });
        }
        if row.score_ppm > SCORE_SCALE_PPM {
            return Err(CalibrationEvaluationError::ScoreOutOfRange {
                index,
                score_ppm: row.score_ppm,
            });
        }

        let expected_budget = if row.model_id == baseline_model_id {
            baseline_budget_units
        } else if row.model_id == candidate_model_id {
            candidate_budget_units
        } else {
            return Err(CalibrationEvaluationError::UnexpectedModel(
                row.model_id.clone(),
            ));
        };
        if row.calibration_budget_units != expected_budget {
            return Err(CalibrationEvaluationError::UnexpectedBudget {
                model_id: row.model_id.clone(),
                expected: expected_budget,
                actual: row.calibration_budget_units,
            });
        }
        if !expected_participants.contains(&row.participant_id) {
            return Err(CalibrationEvaluationError::UnexpectedParticipant(
                row.participant_id.clone(),
            ));
        }

        let key = (
            row.model_id.clone(),
            row.participant_id.clone(),
            row.calibration_budget_units,
        );
        if by_key.insert(key, row.score_ppm).is_some() {
            return Err(CalibrationEvaluationError::DuplicateScore {
                model_id: row.model_id.clone(),
                participant_id: row.participant_id.clone(),
                budget_units: row.calibration_budget_units,
            });
        }
    }

    let mut paired_scores = Vec::with_capacity(expected_participants.len());
    for participant_id in &expected_participants {
        let baseline_key = (
            baseline_model_id.to_owned(),
            (*participant_id).to_owned(),
            baseline_budget_units,
        );
        let candidate_key = (
            candidate_model_id.to_owned(),
            (*participant_id).to_owned(),
            candidate_budget_units,
        );
        let baseline = *by_key.get(&baseline_key).ok_or_else(|| {
            CalibrationEvaluationError::MissingScore {
                model_id: baseline_model_id.to_owned(),
                participant_id: (*participant_id).to_owned(),
                budget_units: baseline_budget_units,
            }
        })?;
        let candidate = *by_key.get(&candidate_key).ok_or_else(|| {
            CalibrationEvaluationError::MissingScore {
                model_id: candidate_model_id.to_owned(),
                participant_id: (*participant_id).to_owned(),
                budget_units: candidate_budget_units,
            }
        })?;
        paired_scores.push(PairedParticipantScore {
            participant_id: (*participant_id).to_owned(),
            baseline_score_ppm: baseline,
            candidate_score_ppm: candidate,
        });
    }

    let bootstrap = paired_participant_bootstrap(&paired_scores, bootstrap_config)
        .map_err(CalibrationEvaluationError::Bootstrap)?;
    let boundary = -i64::from(noninferiority_margin_ppm);
    let decision = if bootstrap.ci_lower_delta_ppm >= boundary {
        NonInferiorityDecision::SupportsNonInferiority
    } else if bootstrap.ci_upper_delta_ppm < boundary {
        NonInferiorityDecision::EvidenceOfInferiority
    } else {
        NonInferiorityDecision::Inconclusive
    };
    let reduction_numerator =
        u128::from(baseline_budget_units - candidate_budget_units) * 1000;
    let calibration_reduction_permille = u16::try_from(
        reduction_numerator / u128::from(baseline_budget_units),
    )
    .map_err(|_| CalibrationEvaluationError::CounterOverflow)?;

    Ok(CalibrationReductionReport {
        method_id: "paired-participant-calibration-noninferiority-percentile-v1",
        baseline_model_id: baseline_model_id.to_owned(),
        candidate_model_id: candidate_model_id.to_owned(),
        budget_unit: budget_unit.to_owned(),
        baseline_calibration_budget_units: baseline_budget_units,
        candidate_calibration_budget_units: candidate_budget_units,
        calibration_reduction_permille,
        participant_count: u32::try_from(expected_participants.len())
            .map_err(|_| CalibrationEvaluationError::CounterOverflow)?,
        baseline_mean_score_ppm: bootstrap.baseline_mean_score_ppm,
        candidate_mean_score_ppm: bootstrap.candidate_mean_score_ppm,
        observed_delta_ppm: bootstrap.mean_paired_delta_ppm,
        ci_lower_delta_ppm: bootstrap.ci_lower_delta_ppm,
        ci_upper_delta_ppm: bootstrap.ci_upper_delta_ppm,
        noninferiority_margin_ppm,
        decision,
        bootstrap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BootstrapConfig {
        BootstrapConfig {
            replicates: 1_000,
            confidence_level_permille: 950,
            seed: 19,
        }
    }

    fn row(model: &str, participant: &str, budget: u32, score: u32) -> CalibrationBudgetScore {
        CalibrationBudgetScore {
            model_id: model.into(),
            participant_id: participant.into(),
            calibration_budget_units: budget,
            score_ppm: score,
        }
    }

    fn fixture(candidate_scores: [u32; 3]) -> Vec<CalibrationBudgetScore> {
        let mut rows = Vec::new();
        for (i, participant) in ["p-01", "p-02", "p-03"].iter().enumerate() {
            rows.push(row("baseline", participant, 100, [700_000, 720_000, 680_000][i]));
            rows.push(row("candidate", participant, 50, candidate_scores[i]));
        }
        rows
    }

    #[test]
    fn supports_preregistered_fifty_percent_reduction_with_noninferiority() {
        let report = evaluate_calibration_reduction(
            &fixture([695_000, 715_000, 675_000]),
            &["p-01", "p-02", "p-03"],
            "baseline", "candidate", "labelled_examples", 100, 50, 20_000, config(),
        ).unwrap();
        assert_eq!(report.calibration_reduction_permille, 500);
        assert_eq!(report.participant_count, 3);
        assert_eq!(report.observed_delta_ppm, -5_000);
        assert_eq!(report.ci_lower_delta_ppm, -5_000);
        assert_eq!(report.ci_upper_delta_ppm, -5_000);
        assert_eq!(report.decision, NonInferiorityDecision::SupportsNonInferiority);
        assert_eq!(report.bootstrap.seed, 19);
    }

    #[test]
    fn rejects_candidate_that_is_worse_than_noninferiority_margin() {
        let report = evaluate_calibration_reduction(
            &fixture([600_000, 620_000, 580_000]),
            &["p-01", "p-02", "p-03"],
            "baseline", "candidate", "labelled_examples", 100, 50, 20_000, config(),
        ).unwrap();
        assert_eq!(report.observed_delta_ppm, -100_000);
        assert_eq!(report.decision, NonInferiorityDecision::EvidenceOfInferiority);
    }

    #[test]
    fn reports_inconclusive_when_interval_crosses_margin() {
        let report = evaluate_calibration_reduction(
            &fixture([710_000, 690_000, 700_000]),
            &["p-01", "p-02", "p-03"],
            "baseline", "candidate", "labelled_examples", 100, 50, 10_000, config(),
        ).unwrap();
        assert_eq!(report.decision, NonInferiorityDecision::Inconclusive);
    }

    #[test]
    fn rejects_duplicate_missing_unexpected_and_mismatched_rows() {
        let mut duplicate = fixture([695_000, 715_000, 675_000]);
        duplicate.push(duplicate[0].clone());
        assert!(matches!(
            evaluate_calibration_reduction(
                &duplicate, &["p-01", "p-02", "p-03"], "baseline", "candidate",
                "labelled_examples", 100, 50, 20_000, config()
            ),
            Err(CalibrationEvaluationError::DuplicateScore { .. })
        ));
        let missing = fixture([695_000, 715_000, 675_000]);
        assert!(matches!(
            evaluate_calibration_reduction(
                &missing[..5], &["p-01", "p-02", "p-03"], "baseline", "candidate",
                "labelled_examples", 100, 50, 20_000, config()
            ),
            Err(CalibrationEvaluationError::MissingScore { .. })
        ));
        let mut unexpected = fixture([695_000, 715_000, 675_000]);
        unexpected.push(row("extra", "p-01", 1, 600_000));
        assert!(matches!(
            evaluate_calibration_reduction(
                &unexpected, &["p-01", "p-02", "p-03"], "baseline", "candidate",
                "labelled_examples", 100, 50, 20_000, config()
            ),
            Err(CalibrationEvaluationError::UnexpectedModel(_))
        ));
        assert_eq!(
            evaluate_calibration_reduction(
                &fixture([695_000, 715_000, 675_000]), &["p-01", "p-02", "p-03"],
                "baseline", "candidate", "labelled_examples", 50, 100, 20_000, config()
            ),
            Err(CalibrationEvaluationError::InvalidBudgetComparison {
                baseline_budget_units: 50,
                candidate_budget_units: 100
            })
        );
    }

    #[test]
    fn rejects_invalid_participant_inventory_margin_and_budget_unit() {
        assert_eq!(
            evaluate_calibration_reduction(
                &fixture([695_000, 715_000, 675_000]), &["p-01", "p-01"],
                "baseline", "candidate", "examples", 100, 50, 20_000, config()
            ),
            Err(CalibrationEvaluationError::DuplicateParticipantId("p-01".into()))
        );
        assert_eq!(
            evaluate_calibration_reduction(
                &fixture([695_000, 715_000, 675_000]), &["p-01", "p-02", "p-03"],
                "baseline", "candidate", "", 100, 50, 20_000, config()
            ),
            Err(CalibrationEvaluationError::EmptyBudgetUnit)
        );
        assert_eq!(
            evaluate_calibration_reduction(
                &fixture([695_000, 715_000, 675_000]), &["p-01", "p-02", "p-03"],
                "baseline", "candidate", "examples", 100, 50, 1_000_001, config()
            ),
            Err(CalibrationEvaluationError::InvalidNonInferiorityMargin(1_000_001))
        );
    }
}
