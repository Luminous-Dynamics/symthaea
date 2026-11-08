// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only validator for household-goods resilience option fixtures.
//!
//! This validates cross-references and challenge-oracle invariants. It does not
//! compute a robust frontier, minimax regret, operational cost, risk, or ROI.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::scenario_campaign::EMBEDDED_SCENARIO_CAMPAIGN;

pub const EMBEDDED_RESILIENCE_OPTIONS: &str =
    include_str!("../../../../docs/research/fixtures/habitat-resilience-options-v0.1.json");

const REQUIRED_OPTION_IDS: [&str; 6] = [
    "LEAN_BASELINE",
    "QUALIFIED_SUPPLY_ALTERNATIVES",
    "WORKCELL_AND_SKILL_REDUNDANCY",
    "ESSENTIAL_UTILITY_CONTINUITY",
    "MODULAR_REPAIR_COMMONALITY",
    "INTEGRATED_FLEXIBLE_MICROFACTORY",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResilienceOptionIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResilienceOptionReport {
    pub issues: Vec<ResilienceOptionIssue>,
    pub option_count: usize,
    pub scenario_count: usize,
    pub outcome_count: usize,
}

impl ResilienceOptionReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_resilience_options_str(
    input: &str,
) -> Result<ResilienceOptionReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_resilience_options(&value))
}

pub fn validate_resilience_options(root: &Value) -> ResilienceOptionReport {
    let mut issues = Vec::new();

    expect_string(root, "schema_version", "0.1.0", "", &mut issues);
    expect_string(
        root,
        "benchmark_id",
        "HABITAT-RESILIENCE-OPTIONS-001",
        "",
        &mut issues,
    );
    expect_string(
        root,
        "parent_program",
        "HABITAT-VERTICAL-001",
        "",
        &mut issues,
    );
    expect_string(
        root,
        "evidence_class",
        "synthetic_decision_challenge_not_empirical",
        "",
        &mut issues,
    );

    if !root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .is_some_and(|text| {
            text.contains("do not estimate actual capital expense")
                && text.contains("No strategy is approved")
        })
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "fixture must disallow real capital, resilience, and strategy-approval claims",
        );
    }

    let expected_scenarios = source_scenario_ids(&mut issues);
    let scenario_ids = read_declared_scenario_ids(root, &mut issues);
    if let (Some(expected), Some(actual)) = (&expected_scenarios, &scenario_ids) {
        if expected != actual {
            push(
                &mut issues,
                "SOURCE_SCENARIO_SET_MISMATCH",
                "scenario_ids",
                "scenario set must exactly match the embedded disruption campaign",
            );
        }
    }

    let mut options_by_id: HashMap<String, &Value> = HashMap::new();
    let options = root.get("options").and_then(Value::as_array);
    let option_count = options.map_or(0, Vec::len);
    if let Some(options) = options {
        if options.len() != REQUIRED_OPTION_IDS.len() {
            push(
                &mut issues,
                "OPTION_COVERAGE_INCOMPLETE",
                "options",
                format!(
                    "expected exactly {} frozen options, found {}",
                    REQUIRED_OPTION_IDS.len(),
                    options.len()
                ),
            );
        }
        for (index, option) in options.iter().enumerate() {
            let path = format!("options[{index}]");
            let Some(id) = nonempty_string(option, "option_id") else {
                push(
                    &mut issues,
                    "OPTION_ID_MISSING",
                    format!("{path}.option_id"),
                    "option_id must be a non-empty string",
                );
                continue;
            };
            if !REQUIRED_OPTION_IDS.contains(&id) {
                push(
                    &mut issues,
                    "OPTION_ID_UNEXPECTED",
                    format!("{path}.option_id"),
                    format!("option {id:?} is outside the frozen fixture version"),
                );
            }
            if options_by_id.insert(id.to_owned(), option).is_some() {
                push(
                    &mut issues,
                    "OPTION_ID_DUPLICATE",
                    format!("{path}.option_id"),
                    format!("option ID {id:?} is duplicated"),
                );
            }

            let investment = option
                .get("incremental_investment_synthetic_minor_units")
                .and_then(Value::as_u64);
            if investment.is_none() {
                push(
                    &mut issues,
                    "OPTION_INVESTMENT_INVALID",
                    format!("{path}.incremental_investment_synthetic_minor_units"),
                    "investment must be a non-negative integer in explicitly synthetic units",
                );
            }
            expect_string(
                option,
                "state",
                "hypothetical_option_not_qualified",
                &path,
                &mut issues,
            );

            if !option
                .get("qualification_gates")
                .and_then(Value::as_array)
                .is_some_and(|gates| {
                    !gates.is_empty()
                        && gates.iter().all(|gate| {
                            gate.as_str().is_some_and(|value| !value.trim().is_empty())
                        })
                })
            {
                push(
                    &mut issues,
                    "OPTION_QUALIFICATION_GATES_MISSING",
                    format!("{path}.qualification_gates"),
                    "every option needs explicit prerequisite qualification gates",
                );
            }

            let declared_mitigations = option
                .get("mitigated_scenario_ids")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<HashSet<_>>()
                });
            if declared_mitigations.is_none() {
                push(
                    &mut issues,
                    "OPTION_MITIGATION_SET_MISSING",
                    format!("{path}.mitigated_scenario_ids"),
                    "mitigated_scenario_ids must be an array",
                );
            } else if let (Some(declared), Some(scenarios)) =
                (declared_mitigations.as_ref(), scenario_ids.as_ref())
            {
                if !declared.is_subset(scenarios) {
                    push(
                        &mut issues,
                        "OPTION_MITIGATION_UNKNOWN_SCENARIO",
                        format!("{path}.mitigated_scenario_ids"),
                        "option references a scenario not in the frozen campaign",
                    );
                }
            }
        }
    } else {
        push(
            &mut issues,
            "OPTIONS_MISSING",
            "options",
            "options must be an array",
        );
    }

    let required_option_set: HashSet<String> = REQUIRED_OPTION_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    let actual_option_set: HashSet<String> = options_by_id.keys().cloned().collect();
    if actual_option_set != required_option_set {
        push(
            &mut issues,
            "OPTION_REQUIRED_SET_MISMATCH",
            "options",
            "option IDs must exactly match the frozen six-option set",
        );
    }

    let mut outcomes_by_key: HashMap<(String, String), &Value> = HashMap::new();
    let outcomes = root.get("scenario_outcomes").and_then(Value::as_array);
    let outcome_count = outcomes.map_or(0, Vec::len);
    if let Some(outcomes) = outcomes {
        for (index, outcome) in outcomes.iter().enumerate() {
            let path = format!("scenario_outcomes[{index}]");
            let Some(option_id) = nonempty_string(outcome, "option_id") else {
                push(
                    &mut issues,
                    "OUTCOME_OPTION_MISSING",
                    format!("{path}.option_id"),
                    "outcome must name an option",
                );
                continue;
            };
            let Some(scenario_id) = nonempty_string(outcome, "scenario_id") else {
                push(
                    &mut issues,
                    "OUTCOME_SCENARIO_MISSING",
                    format!("{path}.scenario_id"),
                    "outcome must name a scenario",
                );
                continue;
            };

            if !options_by_id.contains_key(option_id) {
                push(
                    &mut issues,
                    "OUTCOME_OPTION_UNKNOWN",
                    format!("{path}.option_id"),
                    format!("unknown option {option_id:?}"),
                );
            }
            if scenario_ids
                .as_ref()
                .is_none_or(|ids| !ids.contains(scenario_id))
            {
                push(
                    &mut issues,
                    "OUTCOME_SCENARIO_UNKNOWN",
                    format!("{path}.scenario_id"),
                    format!("unknown scenario {scenario_id:?}"),
                );
            }

            let key = (option_id.to_owned(), scenario_id.to_owned());
            if outcomes_by_key.insert(key, outcome).is_some() {
                push(
                    &mut issues,
                    "OUTCOME_PAIR_DUPLICATE",
                    format!("{path}.option_id"),
                    format!(
                        "duplicate option/scenario pair ({option_id:?}, {scenario_id:?})"
                    ),
                );
            }

            expect_string(
                outcome,
                "precondition_status",
                "synthetic_gate_conditional_not_observed",
                &path,
                &mut issues,
            );
            expect_string(
                outcome,
                "hard_gate_disposition",
                "must_be_rechecked_by_authoritative_qualifier",
                &path,
                &mut issues,
            );
            expect_string(
                outcome,
                "status",
                "synthetic_known_answer_only",
                &path,
                &mut issues,
            );
            validate_interval(
                outcome,
                "lost_accepted_unit_hours_bounds",
                &path,
                "SERVICE_LOSS_INTERVAL_INVALID",
                &mut issues,
            );
            validate_interval(
                outcome,
                "recovery_hours_bounds",
                &path,
                "RECOVERY_INTERVAL_INVALID",
                &mut issues,
            );
        }
    } else {
        push(
            &mut issues,
            "OUTCOMES_MISSING",
            "scenario_outcomes",
            "scenario_outcomes must be an array",
        );
    }

    if let Some(scenarios) = scenario_ids.as_ref() {
        let expected_count = options_by_id.len().saturating_mul(scenarios.len());
        if outcome_count != expected_count {
            push(
                &mut issues,
                "OUTCOME_MATRIX_INCOMPLETE",
                "scenario_outcomes",
                format!(
                    "expected {} option/scenario cells, found {}",
                    expected_count, outcome_count
                ),
            );
        }
        for option_id in options_by_id.keys() {
            for scenario_id in scenarios {
                if !outcomes_by_key.contains_key(&(option_id.clone(), scenario_id.clone())) {
                    push(
                        &mut issues,
                        "OUTCOME_PAIR_MISSING",
                        "scenario_outcomes",
                        format!(
                            "missing outcome for option {option_id:?}, scenario {scenario_id:?}"
                        ),
                    );
                }
            }
        }
    }

    validate_conditional_oracle(
        &options_by_id,
        &outcomes_by_key,
        scenario_ids.as_ref(),
        &mut issues,
    );
    validate_decision_boundary(root, &mut issues);
    reject_probability_invention(root, &mut issues);

    issues.sort_by(|a, b| {
        (&a.code, &a.path, &a.message).cmp(&(&b.code, &b.path, &b.message))
    });

    ResilienceOptionReport {
        issues,
        option_count,
        scenario_count: scenario_ids.as_ref().map_or(0, HashSet::len),
        outcome_count,
    }
}

fn source_scenario_ids(
    issues: &mut Vec<ResilienceOptionIssue>,
) -> Option<HashSet<String>> {
    let campaign: Value = match serde_json::from_str(EMBEDDED_SCENARIO_CAMPAIGN) {
        Ok(value) => value,
        Err(error) => {
            push(
                issues,
                "EMBEDDED_CAMPAIGN_INVALID",
                "source_campaign",
                format!("could not parse embedded campaign: {error}"),
            );
            return None;
        }
    };
    let Some(scenarios) = campaign.get("scenarios").and_then(Value::as_array) else {
        push(
            issues,
            "EMBEDDED_CAMPAIGN_SCENARIOS_MISSING",
            "source_campaign.scenarios",
            "embedded campaign scenarios must be an array",
        );
        return None;
    };
    Some(
        scenarios
            .iter()
            .filter_map(|entry| entry.get("scenario_id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect(),
    )
}

fn read_declared_scenario_ids(
    root: &Value,
    issues: &mut Vec<ResilienceOptionIssue>,
) -> Option<HashSet<String>> {
    let Some(items) = root.get("scenario_ids").and_then(Value::as_array) else {
        push(
            issues,
            "SCENARIO_IDS_MISSING",
            "scenario_ids",
            "scenario_ids must be an array",
        );
        return None;
    };
    let mut ids = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let Some(id) = item.as_str().filter(|text| !text.trim().is_empty()) else {
            push(
                issues,
                "SCENARIO_ID_INVALID",
                format!("scenario_ids[{index}]"),
                "scenario IDs must be non-empty strings",
            );
            continue;
        };
        if !ids.insert(id.to_owned()) {
            push(
                issues,
                "SCENARIO_ID_DUPLICATE",
                format!("scenario_ids[{index}]"),
                format!("scenario {id:?} is duplicated"),
            );
        }
    }
    Some(ids)
}

fn validate_conditional_oracle(
    options: &HashMap<String, &Value>,
    outcomes: &HashMap<(String, String), &Value>,
    scenario_ids: Option<&HashSet<String>>,
    issues: &mut Vec<ResilienceOptionIssue>,
) {
    let Some(scenario_ids) = scenario_ids else {
        return;
    };
    let baseline_id = "LEAN_BASELINE";
    for scenario_id in scenario_ids {
        let Some(baseline) = outcomes.get(&(baseline_id.to_owned(), scenario_id.clone())) else {
            continue;
        };
        for (option_id, option) in options {
            if option_id == baseline_id {
                continue;
            }
            let Some(outcome) = outcomes.get(&(option_id.clone(), scenario_id.clone())) else {
                continue;
            };
            let declared_mitigations: HashSet<String> = option
                .get("mitigated_scenario_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();

            let loss_base = interval_pair(baseline, "lost_accepted_unit_hours_bounds");
            let loss_candidate = interval_pair(outcome, "lost_accepted_unit_hours_bounds");
            let recovery_base = interval_pair(baseline, "recovery_hours_bounds");
            let recovery_candidate = interval_pair(outcome, "recovery_hours_bounds");
            let (Some(loss_base), Some(loss_candidate), Some(recovery_base), Some(recovery_candidate)) =
                (loss_base, loss_candidate, recovery_base, recovery_candidate)
            else {
                continue;
            };

            if declared_mitigations.contains(scenario_id) {
                let improves_loss = loss_candidate.0 <= loss_base.0
                    && loss_candidate.1 <= loss_base.1
                    && (loss_candidate.0 < loss_base.0 || loss_candidate.1 < loss_base.1);
                let improves_recovery = recovery_candidate.0 <= recovery_base.0
                    && recovery_candidate.1 <= recovery_base.1
                    && (recovery_candidate.0 < recovery_base.0
                        || recovery_candidate.1 < recovery_base.1);
                if !improves_loss || !improves_recovery {
                    push(
                        issues,
                        "DECLARED_MITIGATION_NOT_REFLECTED",
                        format!("scenario_outcomes[{option_id},{scenario_id}]"),
                        "a declared mitigation must improve both synthetic challenge metrics in the frozen oracle",
                    );
                }
            } else if loss_candidate != loss_base || recovery_candidate != recovery_base {
                push(
                    issues,
                    "UNDECLARED_MITIGATION_INJECTED",
                    format!("scenario_outcomes[{option_id},{scenario_id}]"),
                    "unrelated scenarios must retain baseline bounds in this localized synthetic oracle",
                );
            }
        }
    }
}

fn validate_decision_boundary(root: &Value, issues: &mut Vec<ResilienceOptionIssue>) {
    let Some(paradigm) = root.get("decision_paradigm") else {
        push(
            issues,
            "DECISION_PARADIGM_MISSING",
            "decision_paradigm",
            "decision paradigm declaration is required",
        );
        return;
    };
    if paradigm.get("no_probability_assumption").and_then(Value::as_bool) != Some(true) {
        push(
            issues,
            "PROBABILITY_POLICY_INVALID",
            "decision_paradigm.no_probability_assumption",
            "fixture must explicitly prohibit invented scenario probabilities",
        );
    }
    if paradigm.get("no_monetization_of_service_losses").and_then(Value::as_bool) != Some(true) {
        push(
            issues,
            "SERVICE_MONETIZATION_POLICY_INVALID",
            "decision_paradigm.no_monetization_of_service_losses",
            "lost service must remain separate from investment cost",
        );
    }
    if paradigm.get("qualification_gates_precede_credit").and_then(Value::as_bool) != Some(true) {
        push(
            issues,
            "QUALIFICATION_POLICY_INVALID",
            "decision_paradigm.qualification_gates_precede_credit",
            "option benefits must be conditional on qualified prerequisites",
        );
    }
    let Some(comparison) = paradigm.get("result_comparison").and_then(Value::as_str) else {
        push(
            issues,
            "COMPARISON_OWNER_MISSING",
            "decision_paradigm.result_comparison",
            "existing robust-frontier/minimax-regret ownership must be documented",
        );
        return;
    };
    if !comparison.contains("existing") || !comparison.contains("not implement a parallel optimizer") {
        push(
            issues,
            "COMPARISON_OWNER_INVALID",
            "decision_paradigm.result_comparison",
            "fixture must explicitly compose existing optimization owners",
        );
    }
}

fn reject_probability_invention(root: &Value, issues: &mut Vec<ResilienceOptionIssue>) {
    fn walk(
        value: &Value,
        path: &str,
        issues: &mut Vec<ResilienceOptionIssue>,
    ) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    if [
                        "scenario_probability",
                        "probability_distribution",
                        "probability_weights",
                        "expected_annual_loss",
                        "expected_roi",
                        "return_on_investment",
                    ]
                    .contains(&key.as_str())
                    {
                        push(
                            issues,
                            "UNJUSTIFIED_PROBABILITY_OR_ROI",
                            format!("{path}.{key}"),
                            "probabilities and ROI are not admitted by this fixture contract",
                        );
                    }
                    walk(child, &format!("{path}.{key}"), issues);
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    walk(child, &format!("{path}[{index}]"), issues);
                }
            }
            _ => {}
        }
    }
    walk(root, "$", issues);
}

fn validate_interval(
    object: &Value,
    key: &str,
    prefix: &str,
    code: &'static str,
    issues: &mut Vec<ResilienceOptionIssue>,
) {
    let path = format!("{prefix}.{key}");
    let Some(interval) = object.get(key) else {
        push(
            issues,
            code,
            path,
            "required interval is missing",
        );
        return;
    };
    let lower = interval.get("lower").and_then(Value::as_u64);
    let upper = interval.get("upper").and_then(Value::as_u64);
    let (Some(lower), Some(upper)) = (lower, upper) else {
        push(
            issues,
            code,
            path,
            "interval endpoints must be non-negative integers",
        );
        return;
    };
    if lower > upper {
        push(
            issues,
            code,
            path,
            "lower interval endpoint must not exceed upper endpoint",
        );
    }
}

fn interval_pair(object: &Value, key: &str) -> Option<(u64, u64)> {
    let interval = object.get(key)?;
    Some((
        interval.get("lower")?.as_u64()?,
        interval.get("upper")?.as_u64()?,
    ))
}

fn expect_string(
    object: &Value,
    key: &str,
    expected: &str,
    prefix: &str,
    issues: &mut Vec<ResilienceOptionIssue>,
) {
    if object.get(key).and_then(Value::as_str) != Some(expected) {
        let path = if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}.{key}")
        };
        push(
            issues,
            "FIELD_VALUE_UNEXPECTED",
            path,
            format!("expected {expected:?}"),
        );
    }
}

fn nonempty_string<'a>(object: &'a Value, key: &str) -> Option<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn push(
    issues: &mut Vec<ResilienceOptionIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ResilienceOptionIssue {
        code,
        path: path.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(EMBEDDED_RESILIENCE_OPTIONS).unwrap()
    }

    fn has_code(report: &ResilienceOptionReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_fixture_has_exact_six_by_eight_matrix() {
        let report = validate_resilience_options(&fixture());
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.option_count, 6);
        assert_eq!(report.scenario_count, 8);
        assert_eq!(report.outcome_count, 48);
    }

    #[test]
    fn source_scenario_set_must_match_campaign() {
        let mut value = fixture();
        value["scenario_ids"].as_array_mut().unwrap().pop();
        assert!(has_code(
            &validate_resilience_options(&value),
            "SOURCE_SCENARIO_SET_MISMATCH"
        ));
    }

    #[test]
    fn duplicate_option_id_is_rejected() {
        let mut value = fixture();
        value["options"][1]["option_id"] = json!("LEAN_BASELINE");
        assert!(has_code(
            &validate_resilience_options(&value),
            "OPTION_ID_DUPLICATE"
        ));
    }

    #[test]
    fn duplicate_option_scenario_outcome_is_rejected() {
        let mut value = fixture();
        let first = value["scenario_outcomes"][0].clone();
        value["scenario_outcomes"][1] = first;
        assert!(has_code(
            &validate_resilience_options(&value),
            "OUTCOME_PAIR_DUPLICATE"
        ));
    }

    #[test]
    fn missing_option_scenario_cell_is_rejected() {
        let mut value = fixture();
        value["scenario_outcomes"].as_array_mut().unwrap().pop();
        let report = validate_resilience_options(&value);
        assert!(has_code(&report, "OUTCOME_MATRIX_INCOMPLETE"));
        assert!(has_code(&report, "OUTCOME_PAIR_MISSING"));
    }

    #[test]
    fn reversed_service_loss_interval_is_rejected() {
        let mut value = fixture();
        value["scenario_outcomes"][0]["lost_accepted_unit_hours_bounds"] =
            json!({"lower": 90, "upper": 10});
        assert!(has_code(
            &validate_resilience_options(&value),
            "SERVICE_LOSS_INTERVAL_INVALID"
        ));
    }

    #[test]
    fn unmitigated_scenario_must_match_baseline_bounds() {
        let mut value = fixture();
        value["scenario_outcomes"][16]["lost_accepted_unit_hours_bounds"]["upper"] =
            json!(1);
        assert!(has_code(
            &validate_resilience_options(&value),
            "UNDECLARED_MITIGATION_INJECTED"
        ));
    }

    #[test]
    fn declared_mitigation_must_improve_both_synthetic_metrics() {
        let mut value = fixture();
        let baseline_loss = value["scenario_outcomes"][0]["lost_accepted_unit_hours_bounds"].clone();
        let baseline_recovery = value["scenario_outcomes"][0]["recovery_hours_bounds"].clone();
        value["scenario_outcomes"][8]["lost_accepted_unit_hours_bounds"] = baseline_loss;
        value["scenario_outcomes"][8]["recovery_hours_bounds"] = baseline_recovery;
        assert!(has_code(
            &validate_resilience_options(&value),
            "DECLARED_MITIGATION_NOT_REFLECTED"
        ));
    }

    #[test]
    fn option_without_qualification_gates_is_rejected() {
        let mut value = fixture();
        value["options"][1]["qualification_gates"] = json!([]);
        assert!(has_code(
            &validate_resilience_options(&value),
            "OPTION_QUALIFICATION_GATES_MISSING"
        ));
    }

    #[test]
    fn synthetic_option_cannot_claim_real_qualification() {
        let mut value = fixture();
        value["options"][1]["state"] = json!("qualified_for_production");
        assert!(has_code(
            &validate_resilience_options(&value),
            "FIELD_VALUE_UNEXPECTED"
        ));
    }

    #[test]
    fn scenario_probability_cannot_be_added_without_evidence() {
        let mut value = fixture();
        value["scenario_probability"] = json!(0.5);
        assert!(has_code(
            &validate_resilience_options(&value),
            "UNJUSTIFIED_PROBABILITY_OR_ROI"
        ));
    }

    #[test]
    fn option_cost_does_not_erase_hard_gate_recheck() {
        let mut value = fixture();
        value["scenario_outcomes"][0]["hard_gate_disposition"] = json!("passed");
        assert!(has_code(
            &validate_resilience_options(&value),
            "FIELD_VALUE_UNEXPECTED"
        ));
    }

    #[test]
    fn report_order_is_deterministic() {
        let mut value = fixture();
        value["scenario_outcomes"][0]["status"] = json!("measured");
        let first = validate_resilience_options(&value);
        let second = validate_resilience_options(&value);
        assert_eq!(first, second);
    }
}
