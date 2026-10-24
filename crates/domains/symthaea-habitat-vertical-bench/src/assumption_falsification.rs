// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Definition-only qualifier for household-goods factory assumption falsification.
//!
//! It verifies dependency coverage and experiment-definition integrity. It does
//! not select an experiment, calculate decision value, admit observations, or
//! qualify any material, supplier, worker, machine, facility, or process.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::resilience_options::EMBEDDED_RESILIENCE_OPTIONS;
use crate::scenario_campaign::EMBEDDED_SCENARIO_CAMPAIGN;

pub const EMBEDDED_HABITAT_ASSUMPTIONS: &str =
    include_str!("../../../../docs/research/fixtures/habitat-assumption-falsification-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssumptionFalsificationIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssumptionFalsificationReport {
    pub issues: Vec<AssumptionFalsificationIssue>,
    pub assumption_count: usize,
    pub experiment_count: usize,
    pub critical_assumption_count: usize,
}

impl AssumptionFalsificationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_assumption_falsification_str(
    input: &str,
) -> Result<AssumptionFalsificationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_assumption_falsification(&value))
}

/// Validate the frozen factory assumption/test definitions.
///
/// A successful result says only that references resolve, critical assumptions
/// have falsifiers and experiment candidates, and the fixture remains unrun.
/// It does not pick the cheapest or best test.
pub fn validate_assumption_falsification(
    root: &Value,
) -> AssumptionFalsificationReport {
    let mut issues = Vec::new();
    expect_string(root, "schema_version", "0.1.0", "", &mut issues);
    expect_string(
        root,
        "benchmark_id",
        "HABITAT-ASSUMPTION-FALSIFICATION-001",
        "",
        &mut issues,
    );
    expect_string(root, "parent_program", "HABITAT-VERTICAL-001", "", &mut issues);
    expect_string(
        root,
        "evidence_class",
        "proposed_experiment_definitions_not_execution_evidence",
        "",
        &mut issues,
    );

    if !root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .is_some_and(|text| {
            text.contains("contains no observations or selected experiment")
                && text.contains("physical safety")
        })
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "fixture must state that it contains no results or qualification evidence",
        );
    }

    let option_ids = embedded_option_ids(&mut issues);
    let scenario_ids = embedded_scenario_ids(&mut issues);
    validate_decision_paradigm(root, &mut issues);

    let assumptions = root.get("assumptions").and_then(Value::as_array);
    let assumption_count = assumptions.map_or(0, Vec::len);
    let mut critical_assumption_count = 0;
    let mut assumptions_by_id: HashMap<String, &Value> = HashMap::new();

    let Some(assumptions) = assumptions else {
        push(
            &mut issues,
            "ASSUMPTIONS_MISSING",
            "assumptions",
            "assumptions must be an array",
        );
        issues.sort_by(|a, b| {
            (&a.code, &a.path, &a.message).cmp(&(&b.code, &b.path, &b.message))
        });
        return AssumptionFalsificationReport {
            issues,
            assumption_count,
            experiment_count: 0,
            critical_assumption_count,
        };
    };

    for (index, assumption) in assumptions.iter().enumerate() {
        let path = format!("assumptions[{index}]");
        let Some(id) = nonempty_string(assumption, "assumption_id") else {
            push(
                &mut issues,
                "ASSUMPTION_ID_MISSING",
                format!("{path}.assumption_id"),
                "assumption_id must be a non-empty string",
            );
            continue;
        };
        if assumptions_by_id.insert(id.to_owned(), assumption).is_some() {
            push(
                &mut issues,
                "ASSUMPTION_ID_DUPLICATE",
                format!("{path}.assumption_id"),
                format!("duplicate assumption ID {id:?}"),
            );
        }
        if assumption.get("revision").and_then(Value::as_u64).is_none_or(|v| v == 0) {
            push(
                &mut issues,
                "ASSUMPTION_REVISION_INVALID",
                format!("{path}.revision"),
                "revision must be a positive integer",
            );
        }

        for field in ["claim", "scope", "consequence_if_false", "falsifier"] {
            require_nonempty_string(assumption, field, &path, &mut issues);
        }

        match nonempty_string(assumption, "status") {
            Some("unknown" | "known_synthetic_only") => {}
            _ => push(
                &mut issues,
                "ASSUMPTION_STATUS_UNSUPPORTED",
                format!("{path}.status"),
                "v0.1 only permits unknown or known_synthetic_only",
            ),
        }
        match nonempty_string(assumption, "criticality") {
            Some("critical") => critical_assumption_count += 1,
            Some("major") => {}
            _ => push(
                &mut issues,
                "ASSUMPTION_CRITICALITY_INVALID",
                format!("{path}.criticality"),
                "criticality must be critical or major",
            ),
        }

        let evidence_status = assumption
            .get("current_evidence")
            .and_then(|e| e.get("status"))
            .and_then(Value::as_str);
        if evidence_status.is_none_or(|status| status.trim().is_empty()) {
            push(
                &mut issues,
                "CURRENT_EVIDENCE_STATE_MISSING",
                format!("{path}.current_evidence.status"),
                "current evidence status must be explicit even when none is admitted",
            );
        }
        if assumption
            .get("current_evidence")
            .and_then(|e| e.get("references"))
            .and_then(Value::as_array)
            .is_none()
        {
            push(
                &mut issues,
                "CURRENT_EVIDENCE_REFERENCES_MISSING",
                format!("{path}.current_evidence.references"),
                "current evidence references must be an array",
            );
        }

        validate_reference_list(
            assumption,
            "depends_on_options",
            option_ids.as_ref(),
            &path,
            "OPTION_DEPENDENCY_UNKNOWN",
            &mut issues,
        );
        validate_reference_list(
            assumption,
            "depends_on_scenarios",
            scenario_ids.as_ref(),
            &path,
            "SCENARIO_DEPENDENCY_UNKNOWN",
            &mut issues,
        );
    }

    let experiments = root.get("experiment_candidates").and_then(Value::as_array);
    let experiment_count = experiments.map_or(0, Vec::len);
    let mut experiment_ids = HashSet::new();
    let mut experiment_coverage: HashMap<String, usize> = HashMap::new();

    if let Some(experiments) = experiments {
        for (index, experiment) in experiments.iter().enumerate() {
            let path = format!("experiment_candidates[{index}]");
            let Some(id) = nonempty_string(experiment, "experiment_id") else {
                push(
                    &mut issues,
                    "EXPERIMENT_ID_MISSING",
                    format!("{path}.experiment_id"),
                    "experiment_id must be a non-empty string",
                );
                continue;
            };
            if !experiment_ids.insert(id.to_owned()) {
                push(
                    &mut issues,
                    "EXPERIMENT_ID_DUPLICATE",
                    format!("{path}.experiment_id"),
                    format!("duplicate experiment ID {id:?}"),
                );
            }

            for field in [
                "title",
                "stage",
                "purpose",
                "if_assumption_holds",
                "if_assumption_fails",
                "if_indeterminate",
                "cost_time_status",
                "outcome_model_status",
                "safety_authority",
            ] {
                require_nonempty_string(experiment, field, &path, &mut issues);
            }

            if experiment
                .get("tests_assumptions")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
            {
                push(
                    &mut issues,
                    "EXPERIMENT_ASSUMPTIONS_MISSING",
                    format!("{path}.tests_assumptions"),
                    "experiment must identify the assumptions it discriminates",
                );
            } else if let Some(refs) = experiment.get("tests_assumptions").and_then(Value::as_array) {
                let mut local_refs = HashSet::new();
                for (ref_index, reference) in refs.iter().enumerate() {
                    let Some(assumption_id) = reference.as_str().filter(|s| !s.trim().is_empty()) else {
                        push(
                            &mut issues,
                            "EXPERIMENT_ASSUMPTION_REF_INVALID",
                            format!("{path}.tests_assumptions[{ref_index}]"),
                            "assumption references must be non-empty string IDs",
                        );
                        continue;
                    };
                    if !local_refs.insert(assumption_id.to_owned()) {
                        push(
                            &mut issues,
                            "EXPERIMENT_ASSUMPTION_REF_DUPLICATE",
                            format!("{path}.tests_assumptions[{ref_index}]"),
                            format!("assumption {assumption_id:?} is repeated within the experiment"),
                        );
                    }
                    if !assumptions_by_id.contains_key(assumption_id) {
                        push(
                            &mut issues,
                            "EXPERIMENT_ASSUMPTION_UNKNOWN",
                            format!("{path}.tests_assumptions[{ref_index}]"),
                            format!("unknown assumption {assumption_id:?}"),
                        );
                    } else {
                        *experiment_coverage.entry(assumption_id.to_owned()).or_default() += 1;
                    }
                }
            }

            for field in ["controlled_variables", "observations_to_collect"] {
                if experiment
                    .get(field)
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty)
                {
                    push(
                        &mut issues,
                        "EXPERIMENT_DEFINITION_INCOMPLETE",
                        format!("{path}.{field}"),
                        "field must be a non-empty array",
                    );
                }
            }

            expect_string(experiment, "execution_state", "not_executed", &path, &mut issues);
            expect_string(experiment, "selection_state", "not_selected", &path, &mut issues);
            expect_string(
                experiment,
                "cost_time_status",
                "not_measured",
                &path,
                &mut issues,
            );
            expect_string(
                experiment,
                "outcome_model_status",
                "qualitative_discriminator_requires_preregistration",
                &path,
                &mut issues,
            );

            if experiment.get("measured_cost").is_some_and(|value| !value.is_null())
                || experiment.get("measured_duration").is_some_and(|value| !value.is_null())
                || experiment.get("observed_outcomes").is_some_and(|value| !value.is_null())
            {
                push(
                    &mut issues,
                    "EXECUTION_RESULT_IN_DEFINITION",
                    &path,
                    "definition-only fixture must not contain measured costs, durations, or outcomes",
                );
            }

            let stage = nonempty_string(experiment, "stage").unwrap_or_default();
            let physical_stage = ["controlled_physical_test", "bounded_sample", "controlled_repair_trial"]
                .iter()
                .any(|physical| stage.contains(*physical));
            if physical_stage {
                let approval = nonempty_string(experiment, "safety_authority").unwrap_or_default();
                if !approval.contains("not_approved") && !approval.contains("not approved") {
                    push(
                        &mut issues,
                        "PHYSICAL_TEST_APPROVAL_IMPLIED",
                        format!("{path}.safety_authority"),
                        "physical/sample tests must remain explicitly unapproved in this definition fixture",
                    );
                }
            }
        }
    } else {
        push(
            &mut issues,
            "EXPERIMENT_CANDIDATES_MISSING",
            "experiment_candidates",
            "experiment_candidates must be an array",
        );
    }

    for (id, assumption) in &assumptions_by_id {
        if assumption.get("criticality").and_then(Value::as_str) == Some("critical")
            && experiment_coverage.get(id).copied().unwrap_or(0) == 0
        {
            push(
                &mut issues,
                "CRITICAL_ASSUMPTION_WITHOUT_TEST",
                format!("assumptions[{id}].falsifier"),
                "every critical assumption needs at least one candidate experiment",
            );
        }
    }

    validate_global_execution_state(root, experiment_count, &mut issues);
    reject_ungrounded_quantitative_claims(root, &mut issues);

    issues.sort_by(|a, b| {
        (&a.code, &a.path, &a.message).cmp(&(&b.code, &b.path, &b.message))
    });
    AssumptionFalsificationReport {
        issues,
        assumption_count,
        experiment_count,
        critical_assumption_count,
    }
}

fn embedded_option_ids(issues: &mut Vec<AssumptionFalsificationIssue>) -> Option<HashSet<String>> {
    let fixture: Value = match serde_json::from_str(EMBEDDED_RESILIENCE_OPTIONS) {
        Ok(value) => value,
        Err(error) => {
            push(issues, "EMBEDDED_OPTIONS_INVALID", "source_options", error.to_string());
            return None;
        }
    };
    let Some(options) = fixture.get("options").and_then(Value::as_array) else {
        push(issues, "EMBEDDED_OPTIONS_MISSING", "source_options.options", "options must be an array");
        return None;
    };
    Some(
        options
            .iter()
            .filter_map(|option| option.get("option_id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect(),
    )
}

fn embedded_scenario_ids(issues: &mut Vec<AssumptionFalsificationIssue>) -> Option<HashSet<String>> {
    let fixture: Value = match serde_json::from_str(EMBEDDED_SCENARIO_CAMPAIGN) {
        Ok(value) => value,
        Err(error) => {
            push(issues, "EMBEDDED_SCENARIOS_INVALID", "source_scenarios", error.to_string());
            return None;
        }
    };
    let Some(scenarios) = fixture.get("scenarios").and_then(Value::as_array) else {
        push(issues, "EMBEDDED_SCENARIOS_MISSING", "source_scenarios.scenarios", "scenarios must be an array");
        return None;
    };
    Some(
        scenarios
            .iter()
            .filter_map(|scenario| scenario.get("scenario_id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect(),
    )
}

fn validate_decision_paradigm(
    root: &Value,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    let Some(paradigm) = root.get("decision_paradigm") else {
        push(issues, "DECISION_PARADIGM_MISSING", "decision_paradigm", "decision paradigm is required");
        return;
    };
    for key in [
        "assumptions_are_not_facts",
        "unknown_is_not_zero",
        "interval_is_not_probability",
        "evidence_hash_is_not_truth",
        "no_scalar_confidence",
        "no_probabilities_without_admitted_model",
    ] {
        if paradigm.get(key).and_then(Value::as_bool) != Some(true) {
            push(
                issues,
                "DECISION_SEMANTICS_MISSING",
                format!("decision_paradigm.{key}"),
                "this invariant must be explicitly true",
            );
        }
    }
    expect_string(paradigm, "selection_state", "not_selected", "decision_paradigm", issues);
    if !paradigm.get("selected_experiment_id").is_some_and(Value::is_null) {
        push(
            issues,
            "SELECTED_EXPERIMENT_MUST_REMAIN_NULL",
            "decision_paradigm.selected_experiment_id",
            "no experiment may be presented as selected by this definition fixture",
        );
    }
    expect_string(
        paradigm,
        "quantitative_experiment_costs",
        "not_measured",
        "decision_paradigm",
        issues,
    );
    expect_string(
        paradigm,
        "outcome_probabilities",
        "not_supplied",
        "decision_paradigm",
        issues,
    );
}

fn validate_global_execution_state(
    root: &Value,
    experiment_count: usize,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    let Some(status) = root.get("execution_status") else {
        push(issues, "EXECUTION_STATUS_MISSING", "execution_status", "global execution status is required");
        return;
    };
    if status.get("experiments_executed").and_then(Value::as_u64) != Some(0) {
        push(
            issues,
            "EXECUTION_COUNT_NOT_ZERO",
            "execution_status.experiments_executed",
            "definition-only fixture must report zero executed experiments",
        );
    }
    if status.get("observations_admitted").and_then(Value::as_u64) != Some(0) {
        push(
            issues,
            "OBSERVATION_COUNT_NOT_ZERO",
            "execution_status.observations_admitted",
            "definition-only fixture must report zero admitted observations",
        );
    }
    expect_string(status, "selection_decision", "not_run", "execution_status", issues);
    expect_string(status, "rust_qualifier", "to_be_added_not_executed", "execution_status", issues);
    expect_string(status, "physical_trials", "none", "execution_status", issues);
    if experiment_count == 0 {
        push(
            issues,
            "EXPERIMENT_COUNT_ZERO",
            "experiment_candidates",
            "at least one experiment candidate is required",
        );
    }
}

fn validate_reference_list(
    object: &Value,
    key: &str,
    allowed: Option<&HashSet<String>>,
    prefix: &str,
    code: &'static str,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    let Some(refs) = object.get(key).and_then(Value::as_array) else {
        push(issues, "DEPENDENCY_LIST_MISSING", format!("{prefix}.{key}"), "dependency list must be an array");
        return;
    };
    if refs.is_empty() {
        push(issues, "DEPENDENCY_LIST_EMPTY", format!("{prefix}.{key}"), "dependency list must not be empty");
    }
    let mut seen = HashSet::new();
    for (index, value) in refs.iter().enumerate() {
        let Some(reference) = value.as_str().filter(|reference| !reference.trim().is_empty()) else {
            push(issues, "DEPENDENCY_REF_INVALID", format!("{prefix}.{key}[{index}]"), "dependency reference must be non-empty text");
            continue;
        };
        if !seen.insert(reference.to_owned()) {
            push(issues, "DEPENDENCY_REF_DUPLICATE", format!("{prefix}.{key}[{index}]"), format!("duplicate reference {reference:?}"));
        }
        if allowed.is_some_and(|allowed| !allowed.contains(reference)) {
            push(issues, code, format!("{prefix}.{key}[{index}]"), format!("unknown source reference {reference:?}"));
        }
    }
}

fn reject_ungrounded_quantitative_claims(
    root: &Value,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    fn walk(value: &Value, path: &str, issues: &mut Vec<AssumptionFalsificationIssue>) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let always_forbidden_result = [
                        "measured_cost",
                        "measured_duration",
                        "observed_outcomes",
                    ]
                    .contains(&key.as_str());
                    let model_value_key = [
                        "scenario_probability",
                        "failure_probability",
                        "outcome_probability",
                        "outcome_probabilities",
                        "quantitative_experiment_costs",
                        "expected_value",
                        "expected_roi",
                        "expected_annual_loss",
                    ]
                    .contains(&key.as_str());
                    let explicit_unprovided = child.is_null()
                        || child.as_str().is_some_and(|status| {
                            [
                                "not_supplied",
                                "not_measured",
                                "not_run",
                                "none",
                                "intentionally_not_computed",
                            ]
                            .contains(&status)
                        });
                    if always_forbidden_result && !child.is_null()
                        || model_value_key && !explicit_unprovided
                    {
                        push(
                            issues,
                            "UNGROUNDED_QUANTITATIVE_CLAIM",
                            format!("{path}.{key}"),
                            "quantitative experiment results/probabilities are not admitted in this definition fixture",
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

fn require_nonempty_string(
    object: &Value,
    key: &str,
    prefix: &str,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    if !nonempty_string(object, key).is_some() {
        push(
            issues,
            "FIELD_STRING_MISSING",
            format!("{prefix}.{key}"),
            "field must be a non-empty string",
        );
    }
}

fn nonempty_string<'a>(object: &'a Value, key: &str) -> Option<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn expect_string(
    object: &Value,
    key: &str,
    expected: &str,
    prefix: &str,
    issues: &mut Vec<AssumptionFalsificationIssue>,
) {
    if object.get(key).and_then(Value::as_str) != Some(expected) {
        let path = if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}.{key}")
        };
        push(issues, "FIELD_VALUE_UNEXPECTED", path, format!("expected {expected:?}"));
    }
}

fn push(
    issues: &mut Vec<AssumptionFalsificationIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(AssumptionFalsificationIssue {
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
        serde_json::from_str(EMBEDDED_HABITAT_ASSUMPTIONS).unwrap()
    }

    fn has_code(report: &AssumptionFalsificationReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_fixture_is_complete_and_unexecuted() {
        let report = validate_assumption_falsification(&fixture());
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.assumption_count, 9);
        assert_eq!(report.experiment_count, 10);
        assert!(report.critical_assumption_count >= 6);
    }

    #[test]
    fn every_critical_assumption_has_a_candidate_experiment() {
        let value = fixture();
        let report = validate_assumption_falsification(&value);
        assert!(report.is_valid(), "{:?}", report.issues);
        for assumption in value["assumptions"].as_array().unwrap() {
            if assumption["criticality"] == "critical" {
                let id = assumption["assumption_id"].as_str().unwrap();
                assert!(value["experiment_candidates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|experiment| experiment["tests_assumptions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|reference| reference.as_str() == Some(id))));
            }
        }
    }

    #[test]
    fn unknown_option_dependency_is_rejected() {
        let mut value = fixture();
        value["assumptions"][0]["depends_on_options"][0] = json!("NOT_A_REAL_OPTION");
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "OPTION_DEPENDENCY_UNKNOWN"
        ));
    }

    #[test]
    fn unknown_scenario_dependency_is_rejected() {
        let mut value = fixture();
        value["assumptions"][0]["depends_on_scenarios"][0] = json!("SC-UNKNOWN");
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "SCENARIO_DEPENDENCY_UNKNOWN"
        ));
    }

    #[test]
    fn critical_assumption_without_candidate_test_is_rejected() {
        let mut value = fixture();
        value["experiment_candidates"].as_array_mut().unwrap().retain(|item| {
            !item["tests_assumptions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|id| id.as_str() == Some("HAF-SUPPLY-001"))
        });
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "CRITICAL_ASSUMPTION_WITHOUT_TEST"
        ));
    }

    #[test]
    fn observed_result_cannot_be_added_to_definition_fixture() {
        let mut value = fixture();
        value["experiment_candidates"][0]["observed_outcomes"] = json!({"pass": true});
        let report = validate_assumption_falsification(&value);
        assert!(has_code(&report, "EXECUTION_RESULT_IN_DEFINITION"));
        assert!(has_code(&report, "UNGROUNDED_QUANTITATIVE_CLAIM"));
    }

    #[test]
    fn physical_experiment_cannot_be_marked_approved_by_definition() {
        let mut value = fixture();
        value["experiment_candidates"][1]["safety_authority"] = json!("approved");
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "PHYSICAL_TEST_APPROVAL_IMPLIED"
        ));
    }

    #[test]
    fn experiment_cannot_be_silently_selected() {
        let mut value = fixture();
        value["decision_paradigm"]["selected_experiment_id"] = json!("HAF-EXP-01");
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "SELECTED_EXPERIMENT_MUST_REMAIN_NULL"
        ));
    }

    #[test]
    fn unsupported_probability_claim_is_rejected() {
        let mut value = fixture();
        value["decision_paradigm"]["outcome_probabilities"] = json!({"SC-01-SUPPLIER-REJECTION": 0.4});
        let report = validate_assumption_falsification(&value);
        assert!(has_code(&report, "FIELD_VALUE_UNEXPECTED"));
        assert!(has_code(&report, "UNGROUNDED_QUANTITATIVE_CLAIM"));
    }

    #[test]
    fn duplicate_experiment_id_is_rejected() {
        let mut value = fixture();
        let first_id = value["experiment_candidates"][0]["experiment_id"].clone();
        value["experiment_candidates"][1]["experiment_id"] = first_id;
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "EXPERIMENT_ID_DUPLICATE"
        ));
    }

    #[test]
    fn unknown_experiment_assumption_reference_is_rejected() {
        let mut value = fixture();
        value["experiment_candidates"][0]["tests_assumptions"] = json!(["MISSING-ASSUMPTION"]);
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "EXPERIMENT_ASSUMPTION_UNKNOWN"
        ));
    }

    #[test]
    fn measured_experiment_cost_cannot_be_fabricated() {
        let mut value = fixture();
        value["experiment_candidates"][0]["measured_cost"] = json!(0);
        assert!(has_code(
            &validate_assumption_falsification(&value),
            "EXECUTION_RESULT_IN_DEFINITION"
        ));
    }

    #[test]
    fn reversing_valid_assumption_and_experiment_order_preserves_report() {
        let mut value = fixture();
        let original = validate_assumption_falsification(&value);
        value["assumptions"].as_array_mut().unwrap().reverse();
        value["experiment_candidates"].as_array_mut().unwrap().reverse();
        let reversed = validate_assumption_falsification(&value);
        assert!(original.is_valid(), "{:?}", original.issues);
        assert_eq!(original, reversed);
    }
}
