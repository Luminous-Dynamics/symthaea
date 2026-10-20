// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline evaluator arithmetic for decision value of perception compute (DVPC).
//! This is not an online allocator or oracle API. Paired losses are evaluator-only
//! and may be calculated only after the producer commits its choice.
//! Resource units are declared units, not implicitly energy.
//!
//! Standalone tests (no Cargo workspace membership or third-party dependencies):
//! rustc --edition=2024 --test tools/vision-decision-value-reference.rs -o /tmp/vision-dvpc-tests
//! /tmp/vision-dvpc-tests

#![deny(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionMode { Cheap, Escalated }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceAccountingV1 {
    pub unit_id: String,
    /// Missing is unknown; never impute it as zero.
    pub charged_units: Option<u64>,
    /// Optional hard ceiling for this individual trial.
    pub trial_budget_units: Option<u64>,
    /// Optional hard ceiling for the entire run; must be identical on all trial rows.
    pub run_budget_units: Option<u128>,
}

/// Evaluator-only paired outcomes. Never expose this to the producer: it contains both
/// counterfactual losses. trial_id is a case identifier, not a world/entity identity.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorTrialV1 {
    pub trial_id: u64,
    /// Only trials with the same declared loss profile may be aggregated.
    pub loss_profile_id: String,
    pub chosen_mode: PerceptionMode,
    pub cheap_task_loss: f64,
    pub escalated_task_loss: f64,
    /// Preregistered tolerance for treating tiny paired differences as neutral.
    pub neutral_tolerance: f64,
    pub resource: ResourceAccountingV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStatus { NotDeclared, UnknownCharge, WithinBudget, Exceeded }

#[derive(Debug, Clone, PartialEq)]
pub struct TrialAssessmentV1 {
    pub trial_id: u64,
    pub chosen_mode: PerceptionMode,
    /// cheap_task_loss - escalated_task_loss; positive means escalation reduced loss.
    /// Retrospective counterfactual information; evaluator-only.
    pub dvpc: f64,
    pub chosen_task_loss: f64,
    /// Present only if escalation was selected.
    pub realized_escalation_value: Option<f64>,
    pub harmful_escalation: bool,
    pub beneficial_escalation_left_unused: bool,
    pub budget_status: BudgetStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunBudgetStatus { NotDeclared, Unknown, WithinBudget, Exceeded }

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummaryV1 {
    pub loss_profile_id: String,
    pub trial_count: usize,
    pub escalation_count: usize,
    pub helpful_escalations: usize,
    pub harmful_escalations: usize,
    pub neutral_escalations: usize,
    pub beneficial_escalations_left_unused: usize,
    pub total_chosen_task_loss: f64,
    /// Potential paired benefits across all rows, not the policy's realized gain.
    pub sum_potential_dvpc: f64,
    /// None when any charge is missing; abstract resource units, not joules.
    pub total_charged_units: Option<u128>,
    pub resource_unit_id: String,
    pub run_budget_units: Option<u128>,
    pub budget_status: RunBudgetStatus,
    /// Sorted by trial_id for stable report ordering.
    pub trials: Vec<TrialAssessmentV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssessmentError {
    EmptyRun,
    DuplicateTrialId(u64),
    InvalidResourceUnit(u64),
    MismatchedResourceUnits,
    InvalidLossProfile(u64),
    MismatchedLossProfiles,
    MismatchedRunBudgets,
    InvalidLoss(u64),
    InvalidNeutralTolerance(u64),
    AggregateOverflow,
}

pub fn assess_trial(trial: &EvaluatorTrialV1) -> Result<TrialAssessmentV1, AssessmentError> {
    if !trial.cheap_task_loss.is_finite()
        || trial.cheap_task_loss < 0.0
        || !trial.escalated_task_loss.is_finite()
        || trial.escalated_task_loss < 0.0 {
        return Err(AssessmentError::InvalidLoss(trial.trial_id));
    }
    if !trial.neutral_tolerance.is_finite() || trial.neutral_tolerance < 0.0 {
        return Err(AssessmentError::InvalidNeutralTolerance(trial.trial_id));
    }
    if trial.loss_profile_id.trim().is_empty() {
        return Err(AssessmentError::InvalidLossProfile(trial.trial_id));
    }
    if trial.resource.unit_id.trim().is_empty() {
        return Err(AssessmentError::InvalidResourceUnit(trial.trial_id));
    }

    let dvpc = trial.cheap_task_loss - trial.escalated_task_loss;
    if !dvpc.is_finite() { return Err(AssessmentError::AggregateOverflow); }

    let chosen_task_loss = match trial.chosen_mode {
        PerceptionMode::Cheap => trial.cheap_task_loss,
        PerceptionMode::Escalated => trial.escalated_task_loss,
    };
    let (harmful, unused_benefit, realized_value) = match trial.chosen_mode {
        PerceptionMode::Escalated => (
            dvpc < -trial.neutral_tolerance,
            false,
            Some(dvpc),
        ),
        PerceptionMode::Cheap => (
            false,
            dvpc > trial.neutral_tolerance,
            None,
        ),
    };
    let budget_status = match (trial.resource.charged_units, trial.resource.trial_budget_units) {
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
        realized_escalation_value: realized_value,
        harmful_escalation: harmful,
        beneficial_escalation_left_unused: unused_benefit,
        budget_status,
    })
}

/// Aggregate one declared task/loss profile with a common resource unit and run cap.
/// Missing charges and mixed profile/units block claims rather than being imputed.
pub fn summarize_run(trials: &[EvaluatorTrialV1]) -> Result<RunSummaryV1, AssessmentError> {
    if trials.is_empty() { return Err(AssessmentError::EmptyRun); }

    let mut sorted: Vec<&EvaluatorTrialV1> = trials.iter().collect();
    sorted.sort_by_key(|trial| trial.trial_id);
    for pair in sorted.windows(2) {
        if pair[0].trial_id == pair[1].trial_id {
            return Err(AssessmentError::DuplicateTrialId(pair[0].trial_id));
        }
    }
    for trial in &sorted { assess_trial(trial)?; }

    let first = sorted.first().ok_or(AssessmentError::EmptyRun)?;
    let loss_profile_id = first.loss_profile_id.clone();
    let resource_unit_id = first.resource.unit_id.clone();
    let run_budget_units = first.resource.run_budget_units;
    for trial in &sorted {
        if trial.loss_profile_id != loss_profile_id {
            return Err(AssessmentError::MismatchedLossProfiles);
        }
        if trial.resource.unit_id != resource_unit_id {
            return Err(AssessmentError::MismatchedResourceUnits);
        }
        if trial.resource.run_budget_units != run_budget_units {
            return Err(AssessmentError::MismatchedRunBudgets);
        }
    }

    let mut assessments = Vec::with_capacity(sorted.len());
    for trial in &sorted { assessments.push(assess_trial(trial)?); }

    let mut escalation_count = 0;
    let mut helpful = 0;
    let mut harmful = 0;
    let mut neutral = 0;
    let mut unused_benefit = 0;
    let mut total_loss = 0.0_f64;
    let mut sum_dvpc = 0.0_f64;
    let mut total_charged = 0_u128;
    let mut all_charges_present = true;
    let mut per_trial_budget_declared = false;
    let mut per_trial_budget_missing = false;
    let mut budget_unknown = false;
    let mut budget_exceeded = false;

    for (trial, assessment) in sorted.iter().zip(assessments.iter()) {
        total_loss += assessment.chosen_task_loss;
        sum_dvpc += assessment.dvpc;
        if !total_loss.is_finite() || !sum_dvpc.is_finite() {
            return Err(AssessmentError::AggregateOverflow);
        }
        match trial.resource.charged_units {
            Some(units) => {
                total_charged = total_charged.checked_add(u128::from(units))
                    .ok_or(AssessmentError::AggregateOverflow)?;
            }
            None => all_charges_present = false,
        }
        if trial.resource.trial_budget_units.is_some() {
            per_trial_budget_declared = true;
        } else {
            per_trial_budget_missing = true;
        }
        match assessment.budget_status {
            BudgetStatus::NotDeclared => {}
            BudgetStatus::UnknownCharge => budget_unknown = true,
            BudgetStatus::WithinBudget => {}
            BudgetStatus::Exceeded => budget_exceeded = true,
        }
        match trial.chosen_mode {
            PerceptionMode::Cheap => {
                if assessment.beneficial_escalation_left_unused { unused_benefit += 1; }
            }
            PerceptionMode::Escalated => {
                escalation_count += 1;
                if assessment.dvpc > trial.neutral_tolerance {
                    helpful += 1;
                } else if assessment.dvpc < -trial.neutral_tolerance {
                    harmful += 1;
                } else {
                    neutral += 1;
                }
            }
        }
    }

    if run_budget_units.is_some() && !all_charges_present { budget_unknown = true; }
    if let Some(cap) = run_budget_units {
        if all_charges_present && total_charged > cap { budget_exceeded = true; }
    }
    let budget_status = if budget_exceeded {
        RunBudgetStatus::Exceeded
    } else if budget_unknown || (per_trial_budget_declared && per_trial_budget_missing) {
        RunBudgetStatus::Unknown
    } else if run_budget_units.is_some() || per_trial_budget_declared {
        RunBudgetStatus::WithinBudget
    } else {
        RunBudgetStatus::NotDeclared
    };

    Ok(RunSummaryV1 {
        loss_profile_id,
        trial_count: assessments.len(),
        escalation_count,
        helpful_escalations: helpful,
        harmful_escalations: harmful,
        neutral_escalations: neutral,
        beneficial_escalations_left_unused: unused_benefit,
        total_chosen_task_loss: total_loss,
        sum_potential_dvpc: sum_dvpc,
        total_charged_units: all_charges_present.then_some(total_charged),
        resource_unit_id,
        run_budget_units,
        budget_status,
        trials: assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(id: u64, choice: PerceptionMode, cheap: f64, expensive: f64,
             charged: Option<u64>, per_trial_budget: Option<u64>) -> EvaluatorTrialV1 {
        EvaluatorTrialV1 {
            trial_id: id,
            loss_profile_id: "object-persistence-loss-v1".to_owned(),
            chosen_mode: choice,
            cheap_task_loss: cheap,
            escalated_task_loss: expensive,
            neutral_tolerance: 0.01,
            resource: ResourceAccountingV1 {
                unit_id: "normalized-compute-unit-v1".to_owned(),
                charged_units: charged,
                trial_budget_units: per_trial_budget,
                run_budget_units: None,
            },
        }
    }

    #[test]
    fn helpful_harmful_and_neutral_escalation_are_distinguished() {
        let good = assess_trial(&trial(1, PerceptionMode::Escalated, 10.0, 6.0, Some(3), Some(4))).unwrap();
        assert_eq!(good.dvpc, 4.0);
        assert_eq!(good.chosen_task_loss, 6.0);
        assert!(!good.harmful_escalation);

        let bad = assess_trial(&trial(2, PerceptionMode::Escalated, 4.0, 9.0, Some(3), Some(4))).unwrap();
        assert_eq!(bad.dvpc, -5.0);
        assert_eq!(bad.chosen_task_loss, 9.0);
        assert!(bad.harmful_escalation);

        let neutral = assess_trial(&trial(3, PerceptionMode::Escalated, 4.0, 4.005, Some(3), Some(4))).unwrap();
        assert!((neutral.dvpc + 0.005).abs() < 1e-12);
        assert!(!neutral.harmful_escalation);
    }

    #[test]
    fn cheap_choice_marks_missed_opportunity_without_claiming_escalation_happened() {
        let result = assess_trial(&trial(4, PerceptionMode::Cheap, 10.0, 6.0, Some(1), Some(2))).unwrap();
        assert!(result.beneficial_escalation_left_unused);
        assert_eq!(result.realized_escalation_value, None);
        assert_eq!(result.chosen_task_loss, 10.0);
    }

    #[test]
    fn individual_and_total_budget_overruns_are_explicit() {
        let mut a = trial(1, PerceptionMode::Cheap, 10.0, 8.0, Some(3), Some(4));
        let mut b = trial(2, PerceptionMode::Escalated, 10.0, 12.0, Some(3), Some(4));
        a.resource.run_budget_units = Some(5);
        b.resource.run_budget_units = Some(5);
        let summary = summarize_run(&[a, b]).unwrap();
        assert_eq!(summary.budget_status, RunBudgetStatus::Exceeded);
        assert_eq!(summary.harmful_escalations, 1);
        assert_eq!(summary.total_charged_units, Some(6));
    }

    #[test]
    fn missing_charge_blocks_resource_claim_instead_of_becoming_zero() {
        let mut a = trial(1, PerceptionMode::Cheap, 5.0, 4.0, Some(1), Some(2));
        let mut b = trial(2, PerceptionMode::Escalated, 5.0, 6.0, None, Some(4));
        a.resource.run_budget_units = Some(6);
        b.resource.run_budget_units = Some(6);
        let summary = summarize_run(&[a, b]).unwrap();
        assert_eq!(summary.total_charged_units, None);
        assert_eq!(summary.budget_status, RunBudgetStatus::Unknown);
    }

    #[test]
    fn duplicate_ids_invalid_values_and_mismatched_loss_profiles_fail_closed() {
        let duplicate = [
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None),
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None),
        ];
        assert_eq!(summarize_run(&duplicate), Err(AssessmentError::DuplicateTrialId(1)));
        assert_eq!(assess_trial(&trial(5, PerceptionMode::Cheap, f64::NAN, 1.0, Some(1), None)),
                   Err(AssessmentError::InvalidLoss(5)));
        let mut bad = trial(6, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        bad.neutral_tolerance = f64::INFINITY;
        assert_eq!(assess_trial(&bad), Err(AssessmentError::InvalidNeutralTolerance(6)));
        let mut other = trial(2, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        other.loss_profile_id = "different-loss-v1".to_owned();
        assert_eq!(summarize_run(&[
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None), other
        ]), Err(AssessmentError::MismatchedLossProfiles));
    }

    #[test]
    fn mixed_units_and_run_budget_profiles_fail_closed() {
        let mut other = trial(2, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        other.resource.unit_id = "host-microseconds-v1".to_owned();
        assert_eq!(summarize_run(&[
            trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None), other
        ]), Err(AssessmentError::MismatchedResourceUnits));

        let mut a = trial(1, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        let b = trial(2, PerceptionMode::Cheap, 1.0, 1.0, Some(1), None);
        a.resource.run_budget_units = Some(10);
        assert_eq!(summarize_run(&[a, b]), Err(AssessmentError::MismatchedRunBudgets));
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
