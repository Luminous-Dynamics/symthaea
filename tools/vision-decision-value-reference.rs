// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline evaluator arithmetic for decision value of perception compute (DVPC).
//!
//! This is a diagnostic reference, not an online allocation policy or oracle API.
//! The paired losses must be evaluator-only and computed after the producer commits
//! its choice. Resource units are declared units, not automatically energy.
//!
//! Run unit tests without adding this file to the Cargo workspace:
//! rustc --edition=2024 --test tools/vision-decision-value-reference.rs -o /tmp/vision-dvpc-tests
//! /tmp/vision-dvpc-tests

#![deny(unsafe_code)]

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionMode {
    Cheap,
    Escalated,
}

/// A host/evaluator-recorded charge in one declared, run-consistent unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceAccountingV1 {
    /// Example: normalized-compute-unit-v1 or host-microseconds-v1.
    pub unit_id: String,
    /// Missing is unknown; never impute it as zero.
    pub charged_units: Option<u64>,
    /// Hard ceiling declared before the trial.
    pub budget_units: Option<u64>,
}

/// Evaluator-only paired outcomes. Never expose this value to the producer:
/// it contains both counterfactual losses. trial_id is not a world/entity identity.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorTrialV1 {
    pub trial_id: u64,
    pub chosen_mode: PerceptionMode,
    pub cheap_task_loss: f64,
    pub escalated_task_loss: f64,
    /// Preregistered tolerance for treating a tiny task-loss difference as neutral.
    pub neutral_tolerance: f64,
    pub resource: ResourceAccountingV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStatus {
    NotDeclared,
    UnknownCharge,
    WithinBudget,
    Exceeded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrialAssessmentV1 {
    pub trial_id: u64,
    pub chosen_mode: PerceptionMode,
    /// cheap_task_loss - escalated_task_loss; positive means escalation reduced loss.
    /// This is retrospective counterfactual information and remains evaluator-only.
    pub dvpc: f64,
    pub chosen_task_loss: f64,
    /// Present only if escalation was selected; tolerance still defines helpful/harmful.
    pub realized_escalation_value: Option<f64>,
    pub harmful_escalation: bool,
    pub beneficial_escalation_left_unused: bool,
    pub budget_status: BudgetStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunBudgetStatus {
    NotDeclared,
    Unknown,
    WithinBudget,
    Exceeded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummaryV1 {
    pub trial_count: usize,
    pub escalation_count: usize,
    pub helpful_escalations: usize,
    pub harmful_escalations: usize,
    pub neutral_escalations: usize,
    pub beneficial_escalations_left_unused: usize,
    pub total_chosen_task_loss: f64,
    /// Sum of potential paired benefits; not the selected policy's realized gain.
    pub sum_potential_dvpc: f64,
    /// None if any charge is missing. These are declared resource units, not joules.
    pub total_charged_units: Option<u128>,
    pub resource_unit_id: Option<String>,
    pub budget_status: RunBudgetStatus,
    /// Sorted by trial_id for stable output.
    pub trials: Vec<TrialAssessmentV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssessmentError {
    EmptyRun,
    DuplicateTrialId(u64),
    InvalidResourceUnit(u64),
    MismatchedResourceUnits,
    InvalidLoss(u64),
    InvalidNeutralTolerance(u64),
    AggregateOverflow,
}

pub fn assess_trial(trial: &EvaluatorTrialV1) -> Result<TrialAssessmentV1, AssessmentError> {
    if !trial.cheap_task_loss.is_finite()
        || trial.cheap_task_loss < 0.0
        || !trial.escalated_task_loss.is_finite()
        || trial.escalated_task_loss < 0.0
    {
        return Err(AssessmentError::InvalidLoss(trial.trial_id));
    }
    if !trial.neutral_tolerance.is_finite() || trial.neutral_tolerance < 0.0 {
        return Err(AssessmentError::InvalidNeutralTolerance(trial.trial_id));
    }
    if trial.resource.unit_id.trim().is_empty() {
        return Err(AssessmentError::InvalidResourceUnit(trial.trial_id));
    }

    let dvpc = trial.cheap_task_loss - trial.escalated_task_loss;
    if !dvpc.is_finite() {
        return Err(AssessmentError::AggregateOverflow);
    }

    let chosen_task_loss = match trial.chosen_mode {
        PerceptionMode::Cheap => trial.cheap_task_loss,
        PerceptionMode::Escalated => trial.escalated_task_loss,
    };
    let (harmful_escalation, beneficial_escalation_left_unused, realized_escalation_value) =
        match trial.chosen_mode {
            PerceptionMode::Escalated => (
                dvpc < -trial.neutral_tolerance,
                false,
                Some(dvpc),
            ),
            PerceptionMode::Cheap => (false, dvpc > trial.neutral_tolerance, None),
        };
    let budget_status = match (trial.resource.charged_units, trial.resource.budget_units) {
        (_, None) => BudgetStatus::NotDeclared,
        (None, Some(_)) => BudgetStatus::UnknownCharge,
        (Some(charged), Some(budget)) if charged <= budget => BudgetStatus::WithinBudget,
        (Some(_), Some(_)) => BudgetStatus::Exceeded,
    };

    Ok(TrialAssessmentV1 {
        trial_id: trial.trial_id,
        chosen_mode: trial.chosen_mode,
        dvpc,
        chosen_task_loss,
        realized_escalation_value,
        harmful_escalation,
        beneficial_escalation_left_unused,
        budget_status,
    })
}

/// Aggregate a complete evaluator run. Input order cannot change report order or
/// summation order. Missing charges and mixed units block resource claims.
pub fn summarize_run(trials: &[EvaluatorTrialV1]) -> Result<RunSummaryV1, AssessmentError> {
    if trials.is_empty() {
        return Err(AssessmentError::EmptyRun);
    }

    let mut sorted_trials: Vec<&EvaluatorTrialV1> = trials.iter().collect();
    sorted_trials.sort_by_key(|trial| trial.trial_id);
    let mut ids = BTreeSet::new();
    for trial in &sorted_trials {
        if !ids.insert(trial.trial_id) {
            return Err(AssessmentError::DuplicateTrialId(trial.trial_id));
        }
    }

    let mut assessments = Vec::with_capacity(trials.len());
    for trial in &sorted_trials {
        assessments.push(assess_trial(trial)?);
    }

    let unit_ids: BTreeSet<&str> = sorted_trials
        .iter()
        .map(|trial| trial.resource.unit_id.as_str())
        .collect();
    if unit_ids.len() != 1 {
        return Err(AssessmentError::MismatchedResourceUnits);
    }
    let resource_unit_id = unit_ids
        .iter()
        .next()
        .copied()
        .filter(|unit| !unit.trim().is_empty())
        .ok_or(AssessmentError::MismatchedResourceUnits)?
        .to_owned();

    let mut escalation_count = 0;
    let mut helpful_escalations = 0;
    let mut harmful_escalations = 0;
    let mut neutral_escalations = 0;
    let mut beneficial_escalations_left_unused = 0;
    let mut total_chosen_task_loss = 0.0_f64;
    let mut sum_potential_dvpc = 0.0_f64;
    let mut total_charged = 0_u128;
    let mut all_charges_present = true;
    let mut any_budget_declared = false;
    let mut any_budget_missing = false;
    let mut any_budget_unknown = false;
    let mut any_budget_exceeded = false;

    for (trial, assessment) in sorted_trials.iter().zip(assessments.iter()) {
        total_chosen_task_loss += assessment.chosen_task_loss;
        sum_potential_dvpc += assessment.dvpc;
        if !total_chosen_task_loss.is_finite() || !sum_potential_dvpc.is_finite() {
            return Err(AssessmentError::AggregateOverflow);
        }

        match trial.resource.charged_units {
            Some(charged) => {
                total_charged = total_charged
                    .checked_add(u128::from(charged))
                    .ok_or(AssessmentError::AggregateOverflow)?;
            }
            None => all_charges_present = false,
        }

        match assessment.budget_status {
            BudgetStatus::NotDeclared => any_budget_missing = true,
            BudgetStatus::UnknownCharge => {
                any_budget_declared = true;
                any_budget_unknown = true;
            }
            BudgetStatus::WithinBudget => any_budget_declared = true,
            BudgetStatus::Exceeded => {
                any_budget_declared = true;
                any_budget_exceeded = true;
            }
        }

        match trial.chosen_mode {
            PerceptionMode::Cheap => {
                if assessment.beneficial_escalation_left_unused {
                    beneficial_escalations_left_unused += 1;
                }
            }
            PerceptionMode::Escalated => {
                escalation_count += 1;
                if assessment.dvpc > trial.neutral_tolerance {
                    helpful_escalations += 1;
                } else if assessment.dvpc < -trial.neutral_tolerance {
                    harmful_escalations += 1;
                } else {
                    neutral_escalations += 1;
                }
            }
        }
    }

    let budget_status = if any_budget_exceeded {
        RunBudgetStatus::Exceeded
    } else if any_budget_unknown || (any_budget_declared && any_budget_missing) {
        RunBudgetStatus::Unknown
    } else if any_budget_declared {
        RunBudgetStatus::WithinBudget
    } else {
        RunBudgetStatus::NotDeclared
    };

    Ok(RunSummaryV1 {
        trial_count: assessments.len(),
        escalation_count,
        helpful_escalations,
        harmful_escalations,
        neutral_escalations,
        beneficial_escalations_left_unused,
        total_chosen_task_loss,
        sum_potential_dvpc,
        total_charged_units: all_charges_present.then_some(total_charged),
        resource_unit_id: Some(resource_unit_id),
        budget_status,
        trials: assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(
        id: u64,
        choice: PerceptionMode,
        cheap_loss: f64,
        escalated_loss: f64,
        charged: Option<u64>,
        budget: Option<u64>,
    ) -> EvaluatorTrialV1 {
        EvaluatorTrialV1 {
            trial_id: id,
            chosen_mode: choice,
            cheap_task_loss: cheap_loss,
            escalated_task_loss: escalated_loss,
            neutral_tolerance: 0.01,
            resource: ResourceAccountingV1 {
                unit_id: "normalized-compute-unit-v1".to_owned(),
                charged_units: charged,
                budget_units: budget,
            },
        }
    }

    #[test]
    fn helpful_harmful_and_neutral_escalation_are_distinguished() {
        let helpful = assess_trial(&trial(1, PerceptionMode::Escalated, 10.0, 6.0, Some(3), Some(4))).unwrap();
        assert_eq!(helpful.dvpc, 4.0);
        assert_eq!(helpful.chosen_task_loss, 6.0);
        assert!(!helpful.harmful_escalation);

        let harmful = assess_trial(&trial(2, PerceptionMode::Escalated, 4.0, 9.0, Some(3), Some(4))).unwrap();
        assert_eq!(harmful.dvpc, -5.0);
        assert_eq!(harmful.chosen_task_loss, 9.0);
        assert!(harmful.harmful_escalation);

        let neutral = assess_trial(&trial(3, PerceptionMode::Escalated, 4.0, 4.005, Some(3), Some(4))).unwrap();
        assert!((neutral.dvpc + 0.005).abs() < 1e-12);
        assert!(!neutral.harmful_escalation);
        assert_eq!(neutral.budget_status, BudgetStatus::WithinBudget);
    }

    #[test]
    fn cheap_choice_marks_missed_opportunity_without_claiming_escalation_happened() {
        let result = assess_trial(&trial(4, PerceptionMode::Cheap, 10.0, 6.0, Some(1), Some(2))).unwrap();
        assert!(result.beneficial_escalation_left_unused);
        assert_eq!(result.realized_escalation_value, None);
        assert_eq!(result.chosen_task_loss, 10.0);
    }

    #[test]
    fn budget_overrun_is_explicit_and_not_averaged_away() {
        let summary = summarize_run(&[
            trial(1, PerceptionMode::Cheap, 10.0, 8.0, Some(1), Some(2)),
            trial(2, PerceptionMode::Escalated, 10.0, 12.0, Some(5), Some(4)),
        ]).unwrap();
        assert_eq!(summary.budget_status, RunBudgetStatus::Exceeded);
        assert_eq!(summary.harmful_escalations, 1);
        assert_eq!(summary.total_charged_units, Some(6));
    }

    #[test]
    fn missing_charge_blocks_resource_claim_instead_of_becoming_zero() {
        let summary = summarize_run(&[
            trial(1, PerceptionMode::Cheap, 5.0, 4.0, Some(1), Some(2)),
            trial(2, PerceptionMode::Escalated, 5.0, 6.0, None, Some(4)),
        ]).unwrap();
        assert_eq!(summary.total_charged_units, None);
        assert_eq!(summary.budget_status, RunBudgetStatus::Unknown);
    }

    #[test]
    fn duplicate_ids_invalid_loss_and_bad_tolerance_fail_closed() {
        let duplicate = [
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None),
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None),
        ];
        assert_eq!(summarize_run(&duplicate), Err(AssessmentError::DuplicateTrialId(1)));
        assert_eq!(
            assess_trial(&trial(5, PerceptionMode::Cheap, f64::NAN, 1.0, Some(1), None)),
            Err(AssessmentError::InvalidLoss(5))
        );
        let mut bad = trial(6, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        bad.neutral_tolerance = f64::INFINITY;
        assert_eq!(assess_trial(&bad), Err(AssessmentError::InvalidNeutralTolerance(6)));
    }

    #[test]
    fn mismatched_units_fail_closed_and_a_budget_requires_a_charge_to_pass() {
        let mut other = trial(2, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        other.resource.unit_id = "host-microseconds-v1".to_owned();
        assert_eq!(
            summarize_run(&[
                trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None),
                other,
            ]),
            Err(AssessmentError::MismatchedResourceUnits)
        );

        let missing = assess_trial(&trial(3, PerceptionMode::Cheap, 1.0, 1.0, None, Some(2))).unwrap();
        assert_eq!(missing.budget_status, BudgetStatus::UnknownCharge);
    }

    #[test]
    fn summary_is_input_order_independent_and_preserves_mixed_outcomes() {
        let a = trial(8, PerceptionMode::Escalated, 4.0, 8.0, Some(3), Some(4));
        let b = trial(2, PerceptionMode::Escalated, 10.0, 5.0, Some(2), Some(4));
        let forward = summarize_run(&[a.clone(), b.clone()]).unwrap();
        let reverse = summarize_run(&[b, a]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.trials[0].trial_id, 2);
        assert_eq!(forward.helpful_escalations, 1);
        assert_eq!(forward.harmful_escalations, 1);
        assert_eq!(forward.sum_potential_dvpc, 1.0);
    }

    #[test]
    fn empty_run_fails_closed() {
        assert_eq!(summarize_run(&[]), Err(AssessmentError::EmptyRun));
    }
}
