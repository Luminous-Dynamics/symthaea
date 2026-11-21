// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Budget-aware finite-candidate inspection allocation reference.
//!
//! This module evaluates an explicit model-based expected-net-value formula.
//! It is an exact exhaustive oracle for at most 18 candidates, not a production
//! scheduler and not an inspection authorization.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const INPUT_SCHEMA: &str = "luminous.semi-bench.inspection-decision.v1";
pub const OUTPUT_SCHEMA: &str = "luminous.semi-bench.inspection-decision-report.v1";
const MAX_EXHAUSTIVE_CANDIDATES: usize = 18;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InspectionDecisionSubjectV1 {
    pub schema: String,
    pub case_id: String,
    pub budget: InspectionBudget,
    pub candidates: Vec<InspectionCandidate>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InspectionBudget {
    pub max_minutes: f64,
    pub max_cash_cost: f64,
    pub max_per_group: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InspectionCandidate {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InspectionCandidateLedgerRow {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RiskOnlyComparator {
    pub selected_candidate_ids: Vec<String>,
    pub expected_net_value: f64,
    pub inspection_cost: f64,
    pub inspection_minutes: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectionClaimLimits {
    pub expected_values_are_model_based_not_observed: bool,
    pub uninspected_units_remain_unknown: bool,
    pub policy_optimum_is_exact_only_for_this_finite_candidate_set: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InspectionDecisionReportV1 {
    pub schema: String,
    pub case_id: String,
    pub selection_policy: String,
    pub budget: InspectionBudgetReport,
    pub selected_candidate_ids: Vec<String>,
    pub selected_expected_net_value: f64,
    pub selected_inspection_cost: f64,
    pub selected_inspection_minutes: f64,
    pub risk_only_comparator: RiskOnlyComparator,
    pub candidate_ledger: Vec<InspectionCandidateLedgerRow>,
    pub risk_only_ranking: Vec<String>,
    pub expected_value_ranking: Vec<String>,
    pub claim_limits: InspectionClaimLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InspectionBudgetReport {
    pub max_minutes: f64,
    pub max_cash_cost: f64,
    pub max_per_group: usize,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum InspectionDecisionError {
    #[error("unsupported or missing inspection decision schema")]
    UnsupportedSchema,
    #[error("case_id must be non-empty")]
    MissingCaseId,
    #[error("budget fields must be finite and non-negative")]
    InvalidBudget,
    #[error("max_per_group must be a positive integer")]
    InvalidGroupCapacity,
    #[error("candidates must be a non-empty array")]
    EmptyCandidates,
    #[error("reference oracle supports at most 18 candidates")]
    TooManyCandidates,
    #[error("candidate identity is empty or duplicated: {0}")]
    DuplicateOrEmptyCandidateId(String),
    #[error("candidate group identity is empty: {0}")]
    MissingGroupId(String),
    #[error("candidate {candidate_id} has invalid probability field {field}")]
    InvalidProbability {
        candidate_id: String,
        field: &'static str,
    },
    #[error("candidate {candidate_id} has invalid non-negative field {field}")]
    InvalidNonNegative {
        candidate_id: String,
        field: &'static str,
    },
    #[error("candidate {0} produced a non-finite derived utility")]
    NonFiniteDerivedValue(String),
}

#[derive(Debug, Clone)]
struct EvaluatedCandidate {
    source: InspectionCandidate,
    expected_prevented_loss: f64,
    expected_false_hold_cost: f64,
    delay_penalty: f64,
    residual_missed_defect_loss: f64,
    expected_net_value: f64,
    net_value_per_minute: Option<f64>,
}

fn round12(value: f64) -> f64 {
    (value * 1.0e12).round() / 1.0e12
}

fn is_finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn evaluate_candidate(
    candidate: InspectionCandidate,
) -> Result<EvaluatedCandidate, InspectionDecisionError> {
    let probabilities = [
        ("risk_probability", candidate.risk_probability),
        ("test_sensitivity", candidate.test_sensitivity),
        ("test_specificity", candidate.test_specificity),
        ("actionability_probability", candidate.actionability_probability),
    ];
    for (field, value) in probabilities {
        if !is_probability(value) {
            return Err(InspectionDecisionError::InvalidProbability {
                candidate_id: candidate.candidate_id.clone(),
                field,
            });
        }
    }

    let nonnegative = [
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
    ];
    for (field, value) in nonnegative {
        if !is_finite_nonnegative(value) {
            return Err(InspectionDecisionError::InvalidNonNegative {
                candidate_id: candidate.candidate_id.clone(),
                field,
            });
        }
    }

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

    if ![
        expected_prevented_loss,
        expected_false_hold_cost,
        delay_penalty,
        expected_net_value,
        residual_missed_defect_loss,
    ]
    .into_iter()
    .all(f64::is_finite)
        || net_value_per_minute.is_some_and(|value| !value.is_finite())
    {
        return Err(InspectionDecisionError::NonFiniteDerivedValue(
            candidate.candidate_id.clone(),
        ));
    }

    Ok(EvaluatedCandidate {
        source: candidate,
        expected_prevented_loss,
        expected_false_hold_cost,
        delay_penalty,
        residual_missed_defect_loss,
        expected_net_value,
        net_value_per_minute,
    })
}

fn choose_risk_only(
    candidates: &[EvaluatedCandidate],
    max_minutes: f64,
    max_cost: f64,
    max_per_group: usize,
) -> (Vec<String>, f64, f64, f64) {
    let mut ordered: Vec<&EvaluatedCandidate> = candidates.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .source
            .risk_probability
            .total_cmp(&left.source.risk_probability)
            .then_with(|| left.source.candidate_id.cmp(&right.source.candidate_id))
    });

    let mut selected = Vec::new();
    let mut group_counts = BTreeMap::<String, usize>::new();
    let (mut total_value, mut total_cost, mut total_minutes) = (0.0, 0.0, 0.0);
    for candidate in ordered {
        if candidate.expected_net_value <= 0.0 {
            continue;
        }
        let group_count = group_counts
            .get(&candidate.source.group_id)
            .copied()
            .unwrap_or(0);
        if group_count >= max_per_group {
            continue;
        }
        let next_cost = total_cost + candidate.source.inspection_cost;
        let next_minutes = total_minutes + candidate.source.inspection_minutes;
        if next_cost <= max_cost && next_minutes <= max_minutes {
            selected.push(candidate.source.candidate_id.clone());
            group_counts.insert(candidate.source.group_id.clone(), group_count + 1);
            total_cost = next_cost;
            total_minutes = next_minutes;
            total_value += candidate.expected_net_value;
        }
    }
    selected.sort();
    (selected, total_value, total_cost, total_minutes)
}

fn subset_is_better(
    value: f64,
    cost: f64,
    minutes: f64,
    ids: &[String],
    best_value: f64,
    best_cost: f64,
    best_minutes: f64,
    best_ids: &[String],
) -> bool {
    const EPSILON: f64 = 1.0e-12;

    if value > best_value + EPSILON {
        return true;
    }
    if (value - best_value).abs() > EPSILON {
        return false;
    }
    if cost < best_cost - EPSILON {
        return true;
    }
    if (cost - best_cost).abs() > EPSILON {
        return false;
    }
    if minutes < best_minutes - EPSILON {
        return true;
    }
    if (minutes - best_minutes).abs() > EPSILON {
        return false;
    }
    best_ids.is_empty() || ids < best_ids
}

pub fn evaluate_inspection_decision(
    subject: &InspectionDecisionSubjectV1,
) -> Result<InspectionDecisionReportV1, InspectionDecisionError> {
    if subject.schema != INPUT_SCHEMA {
        return Err(InspectionDecisionError::UnsupportedSchema);
    }
    if subject.case_id.trim().is_empty() {
        return Err(InspectionDecisionError::MissingCaseId);
    }
    if !is_finite_nonnegative(subject.budget.max_minutes)
        || !is_finite_nonnegative(subject.budget.max_cash_cost)
    {
        return Err(InspectionDecisionError::InvalidBudget);
    }
    if subject.budget.max_per_group == 0 {
        return Err(InspectionDecisionError::InvalidGroupCapacity);
    }
    if subject.candidates.is_empty() {
        return Err(InspectionDecisionError::EmptyCandidates);
    }
    if subject.candidates.len() > MAX_EXHAUSTIVE_CANDIDATES {
        return Err(InspectionDecisionError::TooManyCandidates);
    }

    let mut candidate_ids = HashSet::with_capacity(subject.candidates.len());
    let mut candidates = Vec::with_capacity(subject.candidates.len());
    for candidate in &subject.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.as_str())
        {
            return Err(InspectionDecisionError::DuplicateOrEmptyCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
        if candidate.group_id.trim().is_empty() {
            return Err(InspectionDecisionError::MissingGroupId(
                candidate.candidate_id.clone(),
            ));
        }
        candidates.push(evaluate_candidate(candidate.clone())?);
    }

    // Exact finite-candidate oracle under budget and per-group constraints.
    let mut best_value = 0.0;
    let mut best_cost = 0.0;
    let mut best_minutes = 0.0;
    let mut best_ids = Vec::<String>::new();

    for mask in 1usize..(1usize << candidates.len()) {
        let mut group_counts = BTreeMap::<String, usize>::new();
        let (mut value, mut cost, mut minutes) = (0.0, 0.0, 0.0);
        let mut selected_ids = Vec::new();
        let mut feasible = true;

        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                continue;
            }
            let count = group_counts
                .get(&candidate.source.group_id)
                .copied()
                .unwrap_or(0);
            if count >= subject.budget.max_per_group {
                feasible = false;
                break;
            }
            group_counts.insert(candidate.source.group_id.clone(), count + 1);

            value += candidate.expected_net_value;
            cost += candidate.source.inspection_cost;
            minutes += candidate.source.inspection_minutes;
            selected_ids.push(candidate.source.candidate_id.clone());
        }
        if !feasible
            || value <= 0.0
            || cost > subject.budget.max_cash_cost
            || minutes > subject.budget.max_minutes
        {
            continue;
        }

        selected_ids.sort();
        if subset_is_better(
            value,
            cost,
            minutes,
            &selected_ids,
            best_value,
            best_cost,
            best_minutes,
            &best_ids,
        ) {
            best_value = value;
            best_cost = cost;
            best_minutes = minutes;
            best_ids = selected_ids;
        }
    }

    let (risk_ids, risk_value, risk_cost, risk_minutes) = choose_risk_only(
        &candidates,
        subject.budget.max_minutes,
        subject.budget.max_cash_cost,
        subject.budget.max_per_group,
    );

    let selected_set: HashSet<&str> = best_ids.iter().map(String::as_str).collect();
    let mut candidate_ledger: Vec<InspectionCandidateLedgerRow> = candidates
        .iter()
        .map(|candidate| InspectionCandidateLedgerRow {
            candidate_id: candidate.source.candidate_id.clone(),
            group_id: candidate.source.group_id.clone(),
            risk_probability: round12(candidate.source.risk_probability),
            expected_prevented_loss: round12(candidate.expected_prevented_loss),
            expected_false_hold_cost: round12(candidate.expected_false_hold_cost),
            inspection_cost: round12(candidate.source.inspection_cost),
            delay_penalty: round12(candidate.delay_penalty),
            residual_missed_defect_loss: round12(candidate.residual_missed_defect_loss),
            expected_net_value: round12(candidate.expected_net_value),
            net_value_per_minute: candidate.net_value_per_minute.map(round12),
            inspection_minutes: round12(candidate.source.inspection_minutes),
            selected: selected_set.contains(candidate.source.candidate_id.as_str()),
        })
        .collect();
    candidate_ledger.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let mut risk_ranking: Vec<&EvaluatedCandidate> = candidates.iter().collect();
    risk_ranking.sort_by(|left, right| {
        right
            .source
            .risk_probability
            .total_cmp(&left.source.risk_probability)
            .then_with(|| left.source.candidate_id.cmp(&right.source.candidate_id))
    });
    let mut value_ranking: Vec<&EvaluatedCandidate> = candidates.iter().collect();
    value_ranking.sort_by(|left, right| {
        right
            .expected_net_value
            .total_cmp(&left.expected_net_value)
            .then_with(|| left.source.candidate_id.cmp(&right.source.candidate_id))
    });

    Ok(InspectionDecisionReportV1 {
        schema: OUTPUT_SCHEMA.to_owned(),
        case_id: subject.case_id.clone(),
        selection_policy: "exact-exhaustive-net-value-maximization".to_owned(),
        budget: InspectionBudgetReport {
            max_minutes: round12(subject.budget.max_minutes),
            max_cash_cost: round12(subject.budget.max_cash_cost),
            max_per_group: subject.budget.max_per_group,
        },
        selected_candidate_ids: best_ids,
        selected_expected_net_value: round12(best_value),
        selected_inspection_cost: round12(best_cost),
        selected_inspection_minutes: round12(best_minutes),
        risk_only_comparator: RiskOnlyComparator {
            selected_candidate_ids: risk_ids,
            expected_net_value: round12(risk_value),
            inspection_cost: round12(risk_cost),
            inspection_minutes: round12(risk_minutes),
        },
        candidate_ledger,
        risk_only_ranking: risk_ranking
            .into_iter()
            .map(|candidate| candidate.source.candidate_id.clone())
            .collect(),
        expected_value_ranking: value_ranking
            .into_iter()
            .map(|candidate| candidate.source.candidate_id.clone())
            .collect(),
        claim_limits: InspectionClaimLimits {
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

    fn read_subject() -> InspectionDecisionSubjectV1 {
        serde_json::from_str(INPUT).expect("inspection decision fixture")
    }

    fn assert_json_numeric_equivalent(actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Number(left), Value::Number(right)) => {
                assert_eq!(left.as_f64(), right.as_f64());
            }
            (Value::Array(left), Value::Array(right)) => {
                assert_eq!(left.len(), right.len());
                for (left, right) in left.iter().zip(right) {
                    assert_json_numeric_equivalent(left, right);
                }
            }
            (Value::Object(left), Value::Object(right)) => {
                assert_eq!(
                    left.keys().collect::<Vec<_>>(),
                    right.keys().collect::<Vec<_>>()
                );
                for (key, left_value) in left {
                    assert_json_numeric_equivalent(
                        left_value,
                        right.get(key).expect("matching key"),
                    );
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn inspection_decision_matches_python_known_answer() {
        let subject = read_subject();
        let report = evaluate_inspection_decision(&subject).expect("valid inspection subject");
        let actual = serde_json::to_value(report).expect("serialize decision report");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_json_numeric_equivalent(&actual, &expected);
    }

    #[test]
    fn expected_value_policy_beats_matched_budget_risk_only_policy() {
        let report =
            evaluate_inspection_decision(&read_subject()).expect("valid inspection subject");
        assert_eq!(report.selected_candidate_ids, vec!["A", "C", "F"]);
        assert_eq!(
            report.risk_only_comparator.selected_candidate_ids,
            vec!["A", "B", "D", "F"]
        );
        assert!(report.selected_expected_net_value > report.risk_only_comparator.expected_net_value);
    }

    #[test]
    fn group_capacity_and_budgets_are_never_exceeded() {
        let subject = read_subject();
        let report = evaluate_inspection_decision(&subject).expect("valid inspection subject");
        assert!(report.selected_inspection_minutes <= subject.budget.max_minutes);
        assert!(report.selected_inspection_cost <= subject.budget.max_cash_cost);
        let mut groups = HashSet::new();
        for row in report.candidate_ledger.iter().filter(|row| row.selected) {
            assert!(groups.insert(row.group_id.as_str()));
        }
    }

    #[test]
    fn candidate_order_does_not_change_report() {
        let mut subject = read_subject();
        let first = evaluate_inspection_decision(&subject).expect("valid input");
        subject.candidates.reverse();
        let second = evaluate_inspection_decision(&subject).expect("valid reordered input");
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_probability_out_of_range() {
        let mut subject = read_subject();
        subject.candidates[0].risk_probability = 1.01;
        assert!(matches!(
            evaluate_inspection_decision(&subject),
            Err(InspectionDecisionError::InvalidProbability {
                field: "risk_probability",
                ..
            })
        ));
    }

    #[test]
    fn rejects_negative_cost() {
        let mut subject = read_subject();
        subject.candidates[0].inspection_cost = -1.0;
        assert!(matches!(
            evaluate_inspection_decision(&subject),
            Err(InspectionDecisionError::InvalidNonNegative {
                field: "inspection_cost",
                ..
            })
        ));
    }

    #[test]
    fn rejects_duplicate_candidate_identity() {
        let mut subject = read_subject();
        subject.candidates[1].candidate_id = subject.candidates[0].candidate_id.clone();
        assert!(matches!(
            evaluate_inspection_decision(&subject),
            Err(InspectionDecisionError::DuplicateOrEmptyCandidateId(_))
        ));
    }

    #[test]
    fn rejects_zero_group_capacity() {
        let mut subject = read_subject();
        subject.budget.max_per_group = 0;
        assert_eq!(
            evaluate_inspection_decision(&subject),
            Err(InspectionDecisionError::InvalidGroupCapacity)
        );
    }

    #[test]
    fn rejects_more_than_eighteen_candidates() {
        let mut subject = read_subject();
        let original = subject.candidates.clone();
        for _ in 1..4 {
            subject.candidates.extend(original.clone());
        }
        assert_eq!(
            evaluate_inspection_decision(&subject),
            Err(InspectionDecisionError::TooManyCandidates)
        );
    }

    #[test]
    fn empty_positive_value_set_selects_nothing() {
        let mut subject = read_subject();
        for candidate in &mut subject.candidates {
            candidate.avoidable_loss_per_defect = 0.0;
        }
        let report = evaluate_inspection_decision(&subject).expect("valid empty-positive case");
        assert!(report.selected_candidate_ids.is_empty());
        assert_eq!(report.selected_expected_net_value, 0.0);
    }
}
