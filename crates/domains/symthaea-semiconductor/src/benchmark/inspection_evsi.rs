// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bayesian expected-value-of-sample-information (EVSI) for active metrology.
//!
//! The policy compares the optimal action before measurement with the expected
//! optimal action after a binary measurement. It is a finite-candidate
//! decision oracle, not a risk estimator, fab scheduler, or equipment controller.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.evsi-subject.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.evsi-report.v1";
pub const MAX_CANDIDATES_EXACT: usize = 18;
const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiBudgetV1 {
    pub max_minutes: f64,
    pub max_measurement_cost: f64,
    pub max_per_group: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiCandidateV1 {
    pub candidate_id: String,
    pub group_id: String,
    pub prior_defect_probability: f64,
    pub test_sensitivity: f64,
    pub test_specificity: f64,
    pub defect_loss: f64,
    pub intervention_cost: f64,
    pub false_intervention_cost: f64,
    /// Fraction of defect loss avoided by intervention, conditional on defect.
    pub actionability_probability: f64,
    pub measurement_cost: f64,
    pub measurement_minutes: f64,
    pub delay_hours: f64,
    pub delay_cost_per_hour: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiSubjectV1 {
    pub schema: String,
    pub case_id: String,
    pub budget: EvsiBudgetV1,
    pub candidates: Vec<EvsiCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct EvsiError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvsiActionV1 {
    NoAction,
    Intervene,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiOutcomeDecisionV1 {
    pub outcome_probability: f64,
    /// None only when the outcome has exactly zero probability.
    pub posterior_defect_probability: Option<f64>,
    /// None only when the outcome has exactly zero probability.
    pub selected_action: Option<EvsiActionV1>,
    pub conditional_minimum_expected_loss: f64,
    pub probability_weighted_expected_loss: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiCandidateReportV1 {
    pub candidate_id: String,
    pub group_id: String,
    pub prior_defect_probability: f64,
    pub pre_measurement_action: EvsiActionV1,
    pub expected_loss_without_measurement: f64,
    pub positive_outcome: EvsiOutcomeDecisionV1,
    pub negative_outcome: EvsiOutcomeDecisionV1,
    pub expected_loss_after_measurement_pre_cost: f64,
    pub gross_evsi: f64,
    pub measurement_cost: f64,
    pub delay_penalty: f64,
    pub net_evsi: f64,
    pub measurement_minutes: f64,
    pub selected: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiComparatorV1 {
    pub selected_candidate_ids: Vec<String>,
    pub total_net_evsi: f64,
    pub total_measurement_cost: f64,
    pub total_measurement_minutes: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiClaimLimitsV1 {
    pub bayes_update_and_decision_arithmetic_are_exact_for_declared_inputs: bool,
    pub risk_test_characteristics_actionability_and_costs_are_not_calibrated_here: bool,
    pub unmeasured_subjects_remain_unknown: bool,
    pub optimizer_is_exact_only_for_this_finite_candidate_set: bool,
    pub does_not_schedule_or_execute_physical_measurement: bool,
    pub does_not_authorize_hold_release_or_process_control: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiReportV1 {
    pub schema: String,
    pub case_id: String,
    pub subject_sha256: String,
    pub budget: EvsiBudgetV1,
    pub selected_candidate_ids: Vec<String>,
    pub selected_total_net_evsi: f64,
    pub selected_total_measurement_cost: f64,
    pub selected_total_measurement_minutes: f64,
    pub risk_only_comparator: EvsiComparatorV1,
    pub candidates: Vec<EvsiCandidateReportV1>,
    pub claim_limits: EvsiClaimLimitsV1,
}

#[derive(Clone, Debug)]
struct EvaluatedCandidate {
    report: EvsiCandidateReportV1,
}

#[derive(Clone, Debug)]
struct SubsetScore {
    selected_ids: Vec<String>,
    utility: f64,
    cost: f64,
    minutes: f64,
}

fn require_text(value: &str, field: &str) -> Result<(), EvsiError> {
    if value.trim().is_empty() {
        return Err(EvsiError(format!("{field}: expected non-empty string")));
    }
    Ok(())
}

fn finite(value: f64, field: &str) -> Result<(), EvsiError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvsiError(format!("{field}: expected finite number")))
    }
}

fn non_negative(value: f64, field: &str) -> Result<(), EvsiError> {
    finite(value, field)?;
    if value < 0.0 {
        return Err(EvsiError(format!("{field}: must be non-negative")));
    }
    Ok(())
}

fn probability(value: f64, field: &str) -> Result<(), EvsiError> {
    finite(value, field)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(EvsiError(format!("{field}: must be in [0, 1]")));
    }
    Ok(())
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_subject(subject: &EvsiSubjectV1) -> Result<(), EvsiError> {
    if subject.schema != INPUT_SCHEMA_V1 {
        return Err(EvsiError("schema: unsupported or missing".into()));
    }
    require_text(&subject.case_id, "case_id")?;
    non_negative(subject.budget.max_minutes, "budget.max_minutes")?;
    non_negative(
        subject.budget.max_measurement_cost,
        "budget.max_measurement_cost",
    )?;
    if subject.budget.max_per_group == 0 {
        return Err(EvsiError("budget.max_per_group: expected positive integer".into()));
    }
    if subject.candidates.is_empty() {
        return Err(EvsiError("candidates: expected non-empty array".into()));
    }
    if subject.candidates.len() > MAX_CANDIDATES_EXACT {
        return Err(EvsiError(format!(
            "candidates: exact subset oracle is limited to {MAX_CANDIDATES_EXACT}"
        )));
    }

    let mut seen = BTreeSet::new();
    for (index, candidate) in subject.candidates.iter().enumerate() {
        require_text(
            &candidate.candidate_id,
            &format!("candidates[{index}].candidate_id"),
        )?;
        require_text(
            &candidate.group_id,
            &format!("candidates[{index}].group_id"),
        )?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(EvsiError(format!(
                "candidates[{index}].candidate_id: duplicate"
            )));
        }

        for (field, value) in [
            ("prior_defect_probability", candidate.prior_defect_probability),
            ("test_sensitivity", candidate.test_sensitivity),
            ("test_specificity", candidate.test_specificity),
            ("actionability_probability", candidate.actionability_probability),
        ] {
            probability(value, &format!("{}.{}", candidate.candidate_id, field))?;
        }
        for (field, value) in [
            ("defect_loss", candidate.defect_loss),
            ("intervention_cost", candidate.intervention_cost),
            ("false_intervention_cost", candidate.false_intervention_cost),
            ("measurement_cost", candidate.measurement_cost),
            ("measurement_minutes", candidate.measurement_minutes),
            ("delay_hours", candidate.delay_hours),
            ("delay_cost_per_hour", candidate.delay_cost_per_hour),
        ] {
            non_negative(value, &format!("{}.{}", candidate.candidate_id, field))?;
        }
    }
    Ok(())
}

fn minimum_loss_action(
    defect_probability: f64,
    candidate: &EvsiCandidateV1,
) -> (EvsiActionV1, f64) {
    let no_action_loss = defect_probability * candidate.defect_loss;
    let intervene_loss = candidate.intervention_cost
        + (1.0 - defect_probability) * candidate.false_intervention_cost
        + defect_probability
            * (1.0 - candidate.actionability_probability)
            * candidate.defect_loss;

    // Stable tie-break: do not intervene when the expected losses are equal.
    if intervene_loss < no_action_loss - EPSILON {
        (EvsiActionV1::Intervene, intervene_loss)
    } else {
        (EvsiActionV1::NoAction, no_action_loss)
    }
}

fn outcome_decision(
    outcome_probability: f64,
    posterior_probability: Option<f64>,
    candidate: &EvsiCandidateV1,
) -> Result<EvsiOutcomeDecisionV1, EvsiError> {
    finite(outcome_probability, "outcome_probability")?;
    if !(0.0..=1.0).contains(&outcome_probability) {
        return Err(EvsiError("outcome_probability: must be in [0, 1]".into()));
    }
    match posterior_probability {
        None => {
            if outcome_probability != 0.0 {
                return Err(EvsiError(
                    "posterior: missing for a positive-probability outcome".into(),
                ));
            }
            Ok(EvsiOutcomeDecisionV1 {
                outcome_probability: 0.0,
                posterior_defect_probability: None,
                selected_action: None,
                conditional_minimum_expected_loss: 0.0,
                probability_weighted_expected_loss: 0.0,
            })
        }
        Some(posterior) => {
            probability(posterior, "posterior_defect_probability")?;
            if outcome_probability == 0.0 {
                return Err(EvsiError(
                    "posterior: supplied for a zero-probability outcome".into(),
                ));
            }
            let (action, conditional_loss) = minimum_loss_action(posterior, candidate);
            finite(conditional_loss, "conditional_minimum_expected_loss")?;
            let weighted_loss = outcome_probability * conditional_loss;
            finite(weighted_loss, "probability_weighted_expected_loss")?;
            Ok(EvsiOutcomeDecisionV1 {
                outcome_probability: rounded(outcome_probability),
                posterior_defect_probability: Some(rounded(posterior)),
                selected_action: Some(action),
                conditional_minimum_expected_loss: rounded(conditional_loss),
                probability_weighted_expected_loss: rounded(weighted_loss),
            })
        }
    }
}

fn evaluate_candidate(
    candidate: &EvsiCandidateV1,
) -> Result<EvsiCandidateReportV1, EvsiError> {
    let p = candidate.prior_defect_probability;
    let (pre_action, pre_loss) = minimum_loss_action(p, candidate);

    let positive_probability =
        p * candidate.test_sensitivity + (1.0 - p) * (1.0 - candidate.test_specificity);
    let negative_probability =
        p * (1.0 - candidate.test_sensitivity) + (1.0 - p) * candidate.test_specificity;
    finite(positive_probability, "positive_outcome_probability")?;
    finite(negative_probability, "negative_outcome_probability")?;
    if (positive_probability + negative_probability - 1.0).abs() > 1.0e-10 {
        return Err(EvsiError(
            "measurement outcome probabilities do not sum to one".into(),
        ));
    }

    let positive_posterior = if positive_probability == 0.0 {
        None
    } else {
        Some((p * candidate.test_sensitivity / positive_probability).clamp(0.0, 1.0))
    };
    let negative_posterior = if negative_probability == 0.0 {
        None
    } else {
        Some((p * (1.0 - candidate.test_sensitivity) / negative_probability).clamp(0.0, 1.0))
    };

    let positive = outcome_decision(positive_probability, positive_posterior, candidate)?;
    let negative = outcome_decision(negative_probability, negative_posterior, candidate)?;
    let post_measurement_loss =
        positive.probability_weighted_expected_loss + negative.probability_weighted_expected_loss;
    let gross_evsi = pre_loss - post_measurement_loss;
    if !gross_evsi.is_finite() {
        return Err(EvsiError("gross_evsi: non-finite value".into()));
    }
    // Bayes-optimal post-measurement decisions cannot be worse before direct
    // cost; tolerate only negligible floating-point error.
    if gross_evsi < -1.0e-9 {
        return Err(EvsiError(
            "gross_evsi: negative value violates information non-degradation".into(),
        ));
    }
    let gross_evsi = gross_evsi.max(0.0);
    let delay_penalty = candidate.delay_hours * candidate.delay_cost_per_hour;
    let net_evsi = gross_evsi - candidate.measurement_cost - delay_penalty;
    for (name, value) in [
        ("expected_loss_without_measurement", pre_loss),
        ("expected_loss_after_measurement_pre_cost", post_measurement_loss),
        ("delay_penalty", delay_penalty),
        ("net_evsi", net_evsi),
    ] {
        finite(value, name)?;
    }

    Ok(EvsiCandidateReportV1 {
        candidate_id: candidate.candidate_id.clone(),
        group_id: candidate.group_id.clone(),
        prior_defect_probability: rounded(p),
        pre_measurement_action: pre_action,
        expected_loss_without_measurement: rounded(pre_loss),
        positive_outcome: positive,
        negative_outcome: negative,
        expected_loss_after_measurement_pre_cost: rounded(post_measurement_loss),
        gross_evsi: rounded(gross_evsi),
        measurement_cost: rounded(candidate.measurement_cost),
        delay_penalty: rounded(delay_penalty),
        net_evsi: rounded(net_evsi),
        measurement_minutes: rounded(candidate.measurement_minutes),
        selected: false,
    })
}

fn subset_is_better(candidate: &SubsetScore, incumbent: &SubsetScore) -> bool {
    if candidate.utility > incumbent.utility + EPSILON {
        return true;
    }
    if incumbent.utility > candidate.utility + EPSILON {
        return false;
    }
    if candidate.cost < incumbent.cost - EPSILON {
        return true;
    }
    if incumbent.cost < candidate.cost - EPSILON {
        return false;
    }
    if candidate.minutes < incumbent.minutes - EPSILON {
        return true;
    }
    if incumbent.minutes < candidate.minutes - EPSILON {
        return false;
    }
    candidate.selected_ids < incumbent.selected_ids
}

fn exact_subset(
    candidates: &[EvsiCandidateReportV1],
    budget: &EvsiBudgetV1,
) -> SubsetScore {
    let subset_count = 1usize << candidates.len();
    let mut best = SubsetScore {
        selected_ids: Vec::new(),
        utility: 0.0,
        cost: 0.0,
        minutes: 0.0,
    };

    for mask in 1usize..subset_count {
        let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut selected_ids = Vec::new();
        let mut utility = 0.0;
        let mut cost = 0.0;
        let mut minutes = 0.0;
        let mut feasible = true;

        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                continue;
            }
            let group_count = group_counts.entry(candidate.group_id.as_str()).or_default();
            *group_count += 1;
            if *group_count > budget.max_per_group || candidate.net_evsi <= 0.0 {
                feasible = false;
                break;
            }
            utility += candidate.net_evsi;
            cost += candidate.measurement_cost;
            minutes += candidate.measurement_minutes;
            selected_ids.push(candidate.candidate_id.clone());
        }
        if !feasible || cost > budget.max_measurement_cost || minutes > budget.max_minutes {
            continue;
        }

        let scored = SubsetScore {
            selected_ids,
            utility,
            cost,
            minutes,
        };
        if subset_is_better(&scored, &best) {
            best = scored;
        }
    }
    best
}

fn risk_only(
    candidates: &[EvsiCandidateReportV1],
    budget: &EvsiBudgetV1,
) -> EvsiComparatorV1 {
    let mut ordered: Vec<&EvsiCandidateReportV1> = candidates.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .prior_defect_probability
            .total_cmp(&left.prior_defect_probability)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let mut selected_candidate_ids = Vec::new();
    let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut total_net_evsi = 0.0;
    let mut total_measurement_cost = 0.0;
    let mut total_measurement_minutes = 0.0;
    for candidate in ordered {
        if candidate.net_evsi <= 0.0 {
            continue;
        }
        if group_counts
            .get(candidate.group_id.as_str())
            .copied()
            .unwrap_or(0)
            >= budget.max_per_group
        {
            continue;
        }
        let next_cost = total_measurement_cost + candidate.measurement_cost;
        let next_minutes = total_measurement_minutes + candidate.measurement_minutes;
        if next_cost <= budget.max_measurement_cost && next_minutes <= budget.max_minutes {
            selected_candidate_ids.push(candidate.candidate_id.clone());
            *group_counts.entry(candidate.group_id.as_str()).or_default() += 1;
            total_measurement_cost = next_cost;
            total_measurement_minutes = next_minutes;
            total_net_evsi += candidate.net_evsi;
        }
    }
    selected_candidate_ids.sort();
    EvsiComparatorV1 {
        selected_candidate_ids,
        total_net_evsi: rounded(total_net_evsi),
        total_measurement_cost: rounded(total_measurement_cost),
        total_measurement_minutes: rounded(total_measurement_minutes),
    }
}

/// Evaluates exact Bayesian EVSI for a finite candidate set.
///
/// EVSI is the minimum expected loss before measurement minus the expected
/// minimum conditional loss after the measurement, before direct measurement
/// cost. Net EVSI subtracts measurement cost and delay penalty.
pub fn evaluate_evsi_v1(subject: &EvsiSubjectV1) -> Result<EvsiReportV1, EvsiError> {
    validate_subject(subject)?;
    let subject_bytes = serde_json::to_vec(subject)
        .map_err(|error| EvsiError(format!("subject serialization failed: {error}")))?;

    let mut candidates = subject
        .candidates
        .iter()
        .map(evaluate_candidate)
        .collect::<Result<Vec<_>, _>>()?;
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let best = exact_subset(&candidates, &subject.budget);
    let selected: BTreeSet<&str> = best.selected_ids.iter().map(String::as_str).collect();
    for candidate in &mut candidates {
        candidate.selected = selected.contains(candidate.candidate_id.as_str());
    }
    let risk = risk_only(&candidates, &subject.budget);

    Ok(EvsiReportV1 {
        schema: OUTPUT_SCHEMA_V1.into(),
        case_id: subject.case_id.clone(),
        subject_sha256: sha256_hex(&subject_bytes),
        budget: subject.budget.clone(),
        selected_candidate_ids: best.selected_ids,
        selected_total_net_evsi: rounded(best.utility),
        selected_total_measurement_cost: rounded(best.cost),
        selected_total_measurement_minutes: rounded(best.minutes),
        risk_only_comparator: risk,
        candidates,
        claim_limits: EvsiClaimLimitsV1 {
            bayes_update_and_decision_arithmetic_are_exact_for_declared_inputs: true,
            risk_test_characteristics_actionability_and_costs_are_not_calibrated_here: true,
            unmeasured_subjects_remain_unknown: true,
            optimizer_is_exact_only_for_this_finite_candidate_set: true,
            does_not_schedule_or_execute_physical_measurement: true,
            does_not_authorize_hold_release_or_process_control: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &str,
        group: &str,
        prior: f64,
        sensitivity: f64,
        specificity: f64,
        measurement_cost: f64,
    ) -> EvsiCandidateV1 {
        EvsiCandidateV1 {
            candidate_id: id.into(),
            group_id: group.into(),
            prior_defect_probability: prior,
            test_sensitivity: sensitivity,
            test_specificity: specificity,
            defect_loss: 100.0,
            intervention_cost: 4.0,
            false_intervention_cost: 10.0,
            actionability_probability: 0.9,
            measurement_cost,
            measurement_minutes: 5.0,
            delay_hours: 0.0,
            delay_cost_per_hour: 0.0,
        }
    }

    fn subject(candidates: Vec<EvsiCandidateV1>) -> EvsiSubjectV1 {
        EvsiSubjectV1 {
            schema: INPUT_SCHEMA_V1.into(),
            case_id: "EVSI-KA-001".into(),
            budget: EvsiBudgetV1 {
                max_minutes: 10.0,
                max_measurement_cost: 10.0,
                max_per_group: 1,
            },
            candidates,
        }
    }

    fn report(input: &EvsiSubjectV1) -> EvsiReportV1 {
        evaluate_evsi_v1(input).expect("EVSI subject evaluates")
    }

    #[test]
    fn uninformative_measurement_has_zero_gross_evsi() {
        // Sensitivity equals false-positive rate (1 - specificity).
        let input = subject(vec![candidate("A", "LOT-A", 0.2, 0.8, 0.2, 0.0)]);
        let result = report(&input);
        assert_eq!(result.candidates[0].gross_evsi, 0.0);
        assert!(result.candidates[0].net_evsi <= 0.0);
    }

    #[test]
    fn perfect_measurement_has_non_negative_gross_evsi() {
        let input = subject(vec![candidate("A", "LOT-A", 0.2, 1.0, 1.0, 0.0)]);
        let result = report(&input);
        assert!(result.candidates[0].gross_evsi >= 0.0);
        assert!(
            result.candidates[0].expected_loss_after_measurement_pre_cost
                <= result.candidates[0].expected_loss_without_measurement + 1.0e-9
        );
    }

    #[test]
    fn positive_measurement_cost_can_make_net_evsi_negative() {
        let input = subject(vec![candidate("A", "LOT-A", 0.2, 1.0, 1.0, 1000.0)]);
        let result = report(&input);
        assert!(result.candidates[0].gross_evsi > 0.0);
        assert!(result.candidates[0].net_evsi < 0.0);
        assert!(result.selected_candidate_ids.is_empty());
    }

    #[test]
    fn bayesian_posteriors_and_outcome_probabilities_are_consistent() {
        let input = subject(vec![candidate("A", "LOT-A", 0.2, 0.8, 0.9, 0.0)]);
        let result = report(&input);
        let row = &result.candidates[0];
        assert!((row.positive_outcome.outcome_probability
            + row.negative_outcome.outcome_probability
            - 1.0)
            .abs()
            < 1.0e-12);
        assert_eq!(row.positive_outcome.posterior_defect_probability, Some(0.695652173913));
        assert_eq!(row.negative_outcome.posterior_defect_probability, Some(0.047619047619));
    }

    #[test]
    fn zero_probability_outcomes_have_no_posterior_or_action() {
        let input = subject(vec![candidate("A", "LOT-A", 0.0, 0.0, 1.0, 0.0)]);
        let result = report(&input);
        let row = &result.candidates[0];
        assert_eq!(row.positive_outcome.outcome_probability, 0.0);
        assert_eq!(row.positive_outcome.posterior_defect_probability, None);
        assert_eq!(row.positive_outcome.selected_action, None);
        assert_eq!(row.negative_outcome.posterior_defect_probability, Some(0.0));
    }

    #[test]
    fn zero_cost_information_cannot_raise_optimal_expected_loss() {
        let input = subject(vec![candidate("A", "LOT-A", 0.4, 0.65, 0.8, 0.0)]);
        let result = report(&input);
        let row = &result.candidates[0];
        assert!(
            row.expected_loss_after_measurement_pre_cost
                <= row.expected_loss_without_measurement + 1.0e-9
        );
    }

    #[test]
    fn candidate_order_does_not_change_result() {
        let a = candidate("A", "LOT-A", 0.3, 0.9, 0.95, 1.0);
        let b = candidate("B", "LOT-B", 0.1, 0.8, 0.9, 0.5);
        let forward = report(&subject(vec![a.clone(), b.clone()]));
        let reverse = report(&subject(vec![b, a]));
        assert_eq!(forward.selected_candidate_ids, reverse.selected_candidate_ids);
        assert_eq!(forward.candidates, reverse.candidates);
    }

    #[test]
    fn duplicate_candidate_identity_is_rejected() {
        let input = subject(vec![
            candidate("A", "LOT-A", 0.3, 0.9, 0.95, 1.0),
            candidate("A", "LOT-B", 0.1, 0.8, 0.9, 0.5),
        ]);
        assert!(evaluate_evsi_v1(&input).is_err());
    }

    #[test]
    fn impossible_probability_and_negative_cost_reject() {
        let mut invalid_probability = subject(vec![candidate("A", "LOT-A", 0.3, 0.9, 0.95, 1.0)]);
        invalid_probability.candidates[0].test_sensitivity = 1.1;
        assert!(evaluate_evsi_v1(&invalid_probability).is_err());

        let mut invalid_cost = subject(vec![candidate("A", "LOT-A", 0.3, 0.9, 0.95, 1.0)]);
        invalid_cost.candidates[0].measurement_cost = -1.0;
        assert!(evaluate_evsi_v1(&invalid_cost).is_err());
    }

    #[test]
    fn exact_optimizer_respects_group_capacity_and_budget() {
        let mut first = candidate("A", "LOT-A", 0.3, 0.95, 0.95, 1.0);
        let mut second = candidate("B", "LOT-A", 0.25, 0.95, 0.95, 1.0);
        let mut third = candidate("C", "LOT-C", 0.2, 0.95, 0.95, 1.0);
        first.defect_loss = 500.0;
        second.defect_loss = 300.0;
        third.defect_loss = 250.0;
        let mut input = subject(vec![first, second, third]);
        input.budget.max_minutes = 10.0;
        input.budget.max_measurement_cost = 2.0;
        let result = report(&input);
        assert!(result.selected_candidate_ids.len() <= 2);
        assert!(result.selected_total_measurement_cost <= 2.0);
        assert!(result.selected_total_measurement_minutes <= 10.0);
        assert!(!(result.selected_candidate_ids.contains(&"A".into())
            && result.selected_candidate_ids.contains(&"B".into())));
    }

    #[test]
    fn exact_oracle_rejects_more_than_eighteen_candidates() {
        let candidates = (0..19)
            .map(|index| {
                candidate(
                    &format!("C-{index:02}"),
                    &format!("G-{index:02}"),
                    0.2,
                    0.9,
                    0.95,
                    1.0,
                )
            })
            .collect();
        assert!(evaluate_evsi_v1(&subject(candidates)).is_err());
    }
}
