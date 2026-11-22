// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact finite-candidate inspection-decision reference for active metrology.
//!
//! This evaluator ranks and selects inspections under declared resource
//! constraints. It does not estimate candidate risks, measure wafers, approve
//! holds/releases, or authorize equipment actions.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.inspection-decision.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.inspection-decision-report.v1";
pub const MAX_CANDIDATES_EXACT: usize = 18;
const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionBudgetV1 {
    pub max_minutes: f64,
    pub max_cash_cost: f64,
    pub max_per_group: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionCandidateV1 {
    pub candidate_id: String,
    pub group_id: String,
    pub risk_probability: f64,
    pub test_sensitivity: f64,
    pub test_specificity: f64,
    pub actionability_probability: f64,
    pub exposed_units: f64,
    pub avoidable_loss_per_defect: f64,
    pub false_hold_cost_per_unit: f64,
    pub inspection_cost: f64,
    pub inspection_minutes: f64,
    pub delay_hours: f64,
    pub delay_cost_per_hour: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionDecisionSubjectV1 {
    pub schema: String,
    pub case_id: String,
    pub budget: InspectionBudgetV1,
    pub candidates: Vec<InspectionCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct InspectionDecisionError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionCandidateLedgerV1 {
    pub candidate_id: String,
    pub group_id: String,
    pub risk_probability: f64,
    pub expected_prevented_loss: f64,
    pub expected_false_hold_cost: f64,
    pub inspection_cost: f64,
    pub delay_penalty: f64,
    pub residual_missed_defect_loss: f64,
    pub expected_net_value: f64,
    pub net_value_per_minute: Option<f64>,
    pub inspection_minutes: f64,
    pub selected: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionRiskOnlyComparatorV1 {
    pub selected_candidate_ids: Vec<String>,
    pub expected_net_value: f64,
    pub inspection_cost: f64,
    pub inspection_minutes: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionDecisionBudgetReportV1 {
    pub max_minutes: f64,
    pub max_cash_cost: f64,
    pub max_per_group: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionDecisionClaimLimitsV1 {
    pub expected_values_are_model_based_not_observed: bool,
    pub uninspected_units_remain_unknown: bool,
    pub policy_optimum_is_exact_only_for_this_finite_candidate_set: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectionDecisionReportV1 {
    pub schema: String,
    pub case_id: String,
    pub selection_policy: String,
    pub budget: InspectionDecisionBudgetReportV1,
    pub selected_candidate_ids: Vec<String>,
    pub selected_expected_net_value: f64,
    pub selected_inspection_cost: f64,
    pub selected_inspection_minutes: f64,
    pub risk_only_comparator: InspectionRiskOnlyComparatorV1,
    pub candidate_ledger: Vec<InspectionCandidateLedgerV1>,
    pub risk_only_ranking: Vec<String>,
    pub expected_value_ranking: Vec<String>,
    pub claim_limits: InspectionDecisionClaimLimitsV1,
}

#[derive(Clone, Debug)]
struct CandidateValue {
    candidate_id: String,
    group_id: String,
    risk_probability: f64,
    expected_prevented_loss: f64,
    expected_false_hold_cost: f64,
    inspection_cost: f64,
    delay_penalty: f64,
    residual_missed_defect_loss: f64,
    expected_net_value: f64,
    net_value_per_minute: Option<f64>,
    inspection_minutes: f64,
}

#[derive(Clone, Debug)]
struct SubsetScore {
    selected_ids: Vec<String>,
    value: f64,
    cash_cost: f64,
    minutes: f64,
}

fn require_text(value: &str, field: &str) -> Result<(), InspectionDecisionError> {
    if value.trim().is_empty() {
        return Err(InspectionDecisionError(format!(
            "{field}: expected non-empty string"
        )));
    }
    Ok(())
}

fn finite(value: f64, field: &str) -> Result<(), InspectionDecisionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InspectionDecisionError(format!(
            "{field}: expected finite number"
        )))
    }
}

fn non_negative(value: f64, field: &str) -> Result<(), InspectionDecisionError> {
    finite(value, field)?;
    if value < 0.0 {
        return Err(InspectionDecisionError(format!(
            "{field}: must be non-negative"
        )));
    }
    Ok(())
}

fn probability(value: f64, field: &str) -> Result<(), InspectionDecisionError> {
    finite(value, field)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(InspectionDecisionError(format!(
            "{field}: must be in [0, 1]"
        )));
    }
    Ok(())
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn validate_subject(subject: &InspectionDecisionSubjectV1) -> Result<(), InspectionDecisionError> {
    if subject.schema != INPUT_SCHEMA_V1 {
        return Err(InspectionDecisionError(
            "schema: unsupported or missing".into(),
        ));
    }
    require_text(&subject.case_id, "case_id")?;
    non_negative(subject.budget.max_minutes, "budget.max_minutes")?;
    non_negative(subject.budget.max_cash_cost, "budget.max_cash_cost")?;
    if subject.budget.max_per_group == 0 {
        return Err(InspectionDecisionError(
            "budget.max_per_group: expected positive integer".into(),
        ));
    }
    if subject.candidates.is_empty() {
        return Err(InspectionDecisionError(
            "candidates: expected non-empty array".into(),
        ));
    }
    if subject.candidates.len() > MAX_CANDIDATES_EXACT {
        return Err(InspectionDecisionError(format!(
            "candidates: maximum {MAX_CANDIDATES_EXACT} for exhaustive reference oracle"
        )));
    }

    let mut seen = BTreeSet::new();
    for (index, candidate) in subject.candidates.iter().enumerate() {
        require_text(
            &candidate.candidate_id,
            &format!("candidates[{index}].candidate_id"),
        )?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(InspectionDecisionError(format!(
                "candidates[{index}].candidate_id: duplicate"
            )));
        }
        require_text(
            &candidate.group_id,
            &format!("candidates[{index}].group_id"),
        )?;

        probability(
            candidate.risk_probability,
            &format!("{}.risk_probability", candidate.candidate_id),
        )?;
        probability(
            candidate.test_sensitivity,
            &format!("{}.test_sensitivity", candidate.candidate_id),
        )?;
        probability(
            candidate.test_specificity,
            &format!("{}.test_specificity", candidate.candidate_id),
        )?;
        probability(
            candidate.actionability_probability,
            &format!("{}.actionability_probability", candidate.candidate_id),
        )?;

        for (field, value) in [
            ("exposed_units", candidate.exposed_units),
            (
                "avoidable_loss_per_defect",
                candidate.avoidable_loss_per_defect,
            ),
            (
                "false_hold_cost_per_unit",
                candidate.false_hold_cost_per_unit,
            ),
            ("inspection_cost", candidate.inspection_cost),
            ("inspection_minutes", candidate.inspection_minutes),
            ("delay_hours", candidate.delay_hours),
            ("delay_cost_per_hour", candidate.delay_cost_per_hour),
        ] {
            non_negative(value, &format!("{}.{field}", candidate.candidate_id))?;
        }
    }
    Ok(())
}

fn candidate_value(candidate: &InspectionCandidateV1) -> CandidateValue {
    let expected_prevented_loss = candidate.risk_probability
        * candidate.test_sensitivity
        * candidate.actionability_probability
        * candidate.exposed_units
        * candidate.avoidable_loss_per_defect;
    let expected_false_hold_cost = (1.0 - candidate.risk_probability)
        * (1.0 - candidate.test_specificity)
        * candidate.exposed_units
        * candidate.false_hold_cost_per_unit;
    let delay_penalty = candidate.delay_hours * candidate.delay_cost_per_hour;
    let expected_net_value = expected_prevented_loss
        - expected_false_hold_cost
        - candidate.inspection_cost
        - delay_penalty;
    let residual_missed_defect_loss = candidate.risk_probability
        * (1.0 - candidate.test_sensitivity * candidate.actionability_probability)
        * candidate.exposed_units
        * candidate.avoidable_loss_per_defect;
    let net_value_per_minute = if candidate.inspection_minutes > 0.0 {
        Some(expected_net_value / candidate.inspection_minutes)
    } else {
        None
    };

    CandidateValue {
        candidate_id: candidate.candidate_id.clone(),
        group_id: candidate.group_id.clone(),
        risk_probability: candidate.risk_probability,
        expected_prevented_loss,
        expected_false_hold_cost,
        inspection_cost: candidate.inspection_cost,
        delay_penalty,
        residual_missed_defect_loss,
        expected_net_value,
        net_value_per_minute,
        inspection_minutes: candidate.inspection_minutes,
    }
}

fn subset_is_better(candidate: &SubsetScore, incumbent: &SubsetScore) -> bool {
    if candidate.value > incumbent.value + EPSILON {
        return true;
    }
    if incumbent.value > candidate.value + EPSILON {
        return false;
    }
    if candidate.cash_cost < incumbent.cash_cost - EPSILON {
        return true;
    }
    if incumbent.cash_cost < candidate.cash_cost - EPSILON {
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

fn exact_subset(rows: &[CandidateValue], budget: &InspectionBudgetV1) -> SubsetScore {
    let subsets = 1usize << rows.len();
    let mut best = SubsetScore {
        selected_ids: Vec::new(),
        value: 0.0,
        cash_cost: 0.0,
        minutes: 0.0,
    };

    for mask in 1usize..subsets {
        let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut value = 0.0;
        let mut cash_cost = 0.0;
        let mut minutes = 0.0;
        let mut selected_ids = Vec::new();
        let mut feasible = true;

        for (index, row) in rows.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                continue;
            }
            let count = group_counts.entry(row.group_id.as_str()).or_default();
            *count += 1;
            if *count > budget.max_per_group {
                feasible = false;
                break;
            }
            value += row.expected_net_value;
            cash_cost += row.inspection_cost;
            minutes += row.inspection_minutes;
            selected_ids.push(row.candidate_id.clone());
        }
        if !feasible
            || value <= 0.0
            || cash_cost > budget.max_cash_cost
            || minutes > budget.max_minutes
        {
            continue;
        }
        // rows are canonicalized by candidate identity before enumeration.
        let candidate = SubsetScore {
            selected_ids,
            value,
            cash_cost,
            minutes,
        };
        if subset_is_better(&candidate, &best) {
            best = candidate;
        }
    }
    best
}

fn risk_only(rows: &[CandidateValue], budget: &InspectionBudgetV1) -> SubsetScore {
    let mut ordered: Vec<&CandidateValue> = rows.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .risk_probability
            .total_cmp(&left.risk_probability)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let mut selected_ids = Vec::new();
    let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut value = 0.0;
    let mut cash_cost = 0.0;
    let mut minutes = 0.0;
    for row in ordered {
        if row.expected_net_value <= 0.0 {
            continue;
        }
        if group_counts
            .get(row.group_id.as_str())
            .copied()
            .unwrap_or(0)
            >= budget.max_per_group
        {
            continue;
        }
        let next_cost = cash_cost + row.inspection_cost;
        let next_minutes = minutes + row.inspection_minutes;
        if next_cost <= budget.max_cash_cost && next_minutes <= budget.max_minutes {
            selected_ids.push(row.candidate_id.clone());
            *group_counts.entry(row.group_id.as_str()).or_default() += 1;
            cash_cost = next_cost;
            minutes = next_minutes;
            value += row.expected_net_value;
        }
    }
    selected_ids.sort();
    SubsetScore {
        selected_ids,
        value,
        cash_cost,
        minutes,
    }
}

/// Evaluate inspection candidates using the exact finite-set v1 policy.
///
/// For each candidate, expected net value is:
///
/// p × sensitivity × actionability × exposed_units × avoidable_loss
/// − expected_false_hold_cost − inspection_cost − delay_penalty.
///
/// This module only selects which inspections to consider under declared
/// constraints. Expected values remain model-based and require calibration.
pub fn evaluate_inspection_decision_v1(
    subject: &InspectionDecisionSubjectV1,
) -> Result<InspectionDecisionReportV1, InspectionDecisionError> {
    validate_subject(subject)?;

    let mut rows: Vec<CandidateValue> = subject.candidates.iter().map(candidate_value).collect();
    rows.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    for row in &rows {
        for (field, value) in [
            ("expected_prevented_loss", row.expected_prevented_loss),
            ("expected_false_hold_cost", row.expected_false_hold_cost),
            ("delay_penalty", row.delay_penalty),
            ("expected_net_value", row.expected_net_value),
            (
                "residual_missed_defect_loss",
                row.residual_missed_defect_loss,
            ),
        ] {
            finite(value, &format!("{}.{}", row.candidate_id, field))?;
        }
        if let Some(value) = row.net_value_per_minute {
            finite(value, &format!("{}.net_value_per_minute", row.candidate_id))?;
        }
    }

    let best = exact_subset(&rows, &subject.budget);
    let risk = risk_only(&rows, &subject.budget);
    let selected: BTreeSet<&str> = best.selected_ids.iter().map(String::as_str).collect();

    let candidate_ledger = rows
        .iter()
        .map(|row| InspectionCandidateLedgerV1 {
            candidate_id: row.candidate_id.clone(),
            group_id: row.group_id.clone(),
            risk_probability: rounded(row.risk_probability),
            expected_prevented_loss: rounded(row.expected_prevented_loss),
            expected_false_hold_cost: rounded(row.expected_false_hold_cost),
            inspection_cost: rounded(row.inspection_cost),
            delay_penalty: rounded(row.delay_penalty),
            residual_missed_defect_loss: rounded(row.residual_missed_defect_loss),
            expected_net_value: rounded(row.expected_net_value),
            net_value_per_minute: row.net_value_per_minute.map(rounded),
            inspection_minutes: rounded(row.inspection_minutes),
            selected: selected.contains(row.candidate_id.as_str()),
        })
        .collect();

    let mut risk_only_ranking = rows.clone();
    risk_only_ranking.sort_by(|left, right| {
        right
            .risk_probability
            .total_cmp(&left.risk_probability)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let mut expected_value_ranking = rows.clone();
    expected_value_ranking.sort_by(|left, right| {
        right
            .expected_net_value
            .total_cmp(&left.expected_net_value)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    Ok(InspectionDecisionReportV1 {
        schema: OUTPUT_SCHEMA_V1.into(),
        case_id: subject.case_id.clone(),
        selection_policy: "exact-exhaustive-net-value-maximization".into(),
        budget: InspectionDecisionBudgetReportV1 {
            max_minutes: rounded(subject.budget.max_minutes),
            max_cash_cost: rounded(subject.budget.max_cash_cost),
            max_per_group: subject.budget.max_per_group,
        },
        selected_candidate_ids: best.selected_ids,
        selected_expected_net_value: rounded(best.value),
        selected_inspection_cost: rounded(best.cash_cost),
        selected_inspection_minutes: rounded(best.minutes),
        risk_only_comparator: InspectionRiskOnlyComparatorV1 {
            selected_candidate_ids: risk.selected_ids,
            expected_net_value: rounded(risk.value),
            inspection_cost: rounded(risk.cash_cost),
            inspection_minutes: rounded(risk.minutes),
        },
        candidate_ledger,
        risk_only_ranking: risk_only_ranking
            .into_iter()
            .map(|row| row.candidate_id.clone())
            .collect(),
        expected_value_ranking: expected_value_ranking
            .into_iter()
            .map(|row| row.candidate_id.clone())
            .collect(),
        claim_limits: InspectionDecisionClaimLimitsV1 {
            expected_values_are_model_based_not_observed: true,
            uninspected_units_remain_unknown: true,
            policy_optimum_is_exact_only_for_this_finite_candidate_set: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/inspection-decision-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/inspection-decision-known-answer-v1.expected.json"
    );

    fn subject() -> InspectionDecisionSubjectV1 {
        serde_json::from_str(INPUT).expect("inspection fixture parses")
    }

    fn report_value(subject: &InspectionDecisionSubjectV1) -> Value {
        serde_json::to_value(evaluate_inspection_decision_v1(subject).expect("fixture evaluates"))
            .expect("report serializes")
    }

    #[test]
    fn reproduces_frozen_python_known_answer() {
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected report parses");
        assert_eq!(report_value(&subject()), expected);
    }

    #[test]
    fn selects_value_aware_subset_not_risk_only_subset() {
        let report = evaluate_inspection_decision_v1(&subject()).expect("fixture evaluates");
        assert_eq!(report.selected_candidate_ids, ["A", "C", "F"]);
        assert_eq!(
            report.risk_only_comparator.selected_candidate_ids,
            ["A", "B", "D", "F"]
        );
        assert_eq!(report.selected_expected_net_value, 1286.16);
        assert_eq!(report.risk_only_comparator.expected_net_value, 607.05);
    }

    #[test]
    fn candidate_input_order_does_not_change_report() {
        let mut original = subject();
        let mut permuted = subject();
        permuted.candidates.reverse();
        assert_eq!(report_value(&original), report_value(&permuted));
        original
            .candidates
            .sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
        assert_eq!(report_value(&original), report_value(&permuted));
    }

    #[test]
    fn duplicate_candidate_identity_is_rejected() {
        let mut input = subject();
        input.candidates[1].candidate_id = input.candidates[0].candidate_id.clone();
        assert!(evaluate_inspection_decision_v1(&input).is_err());
    }

    #[test]
    fn invalid_probability_is_rejected() {
        let mut input = subject();
        input.candidates[0].risk_probability = 1.01;
        assert!(evaluate_inspection_decision_v1(&input).is_err());
    }

    #[test]
    fn negative_cost_is_rejected() {
        let mut input = subject();
        input.candidates[0].inspection_cost = -0.01;
        assert!(evaluate_inspection_decision_v1(&input).is_err());
    }

    #[test]
    fn zero_time_has_null_net_value_per_minute() {
        let mut input = subject();
        input.candidates[0].inspection_minutes = 0.0;
        let report = evaluate_inspection_decision_v1(&input).expect("zero time is valid");
        let row = report
            .candidate_ledger
            .iter()
            .find(|row| row.candidate_id == "A")
            .unwrap();
        assert_eq!(row.net_value_per_minute, None);
    }

    #[test]
    fn exact_oracle_rejects_more_than_eighteen_candidates() {
        let mut input = subject();
        let template = input.candidates[0].clone();
        input.candidates = (0..19)
            .map(|index| {
                let mut candidate = template.clone();
                candidate.candidate_id = format!("C-{index:02}");
                candidate.group_id = format!("G-{index:02}");
                candidate
            })
            .collect();
        assert!(evaluate_inspection_decision_v1(&input).is_err());
    }

    #[test]
    fn empty_candidate_set_is_rejected() {
        let mut input = subject();
        input.candidates.clear();
        assert!(evaluate_inspection_decision_v1(&input).is_err());
    }
}
