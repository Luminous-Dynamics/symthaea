// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Definition-only qualifier for the household-goods disruption campaign.
//!
//! This module verifies that the eight synthetic scenarios are complete,
//! source-aligned, and explicitly unexecuted. It does not run a scheduler,
//! simulate a factory, or calculate production outcomes.

use std::collections::HashSet;

use serde_json::Value;

use crate::EMBEDDED_FIXTURE;

/// Exact synthetic counterfactual campaign definition pinned by this crate.
pub const EMBEDDED_SCENARIO_CAMPAIGN: &str =
    include_str!("../../../../docs/research/fixtures/habitat-vertical-scenarios-v0.1.json");

const REQUIRED_SCENARIO_IDS: [&str; 8] = [
    "SC-01-SUPPLIER-REJECTION",
    "SC-02-GRADE-SUBSTITUTION",
    "SC-03-TRANSPORT-DELAY",
    "SC-04-EQUIPMENT-OUTAGE",
    "SC-05-POWER-COMMON-MODE",
    "SC-06-SKILL-ABSENCE",
    "SC-07-DEMAND-MIX-CHANGE",
    "SC-08-REPAIR-SPARE-SHORTAGE",
];

/// One deterministic campaign-structure finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioCampaignIssue {
    /// Stable machine-readable finding code.
    pub code: &'static str,
    /// JSON-style location within the campaign.
    pub path: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Validation result for the scenario definition, not its execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioCampaignReport {
    /// Findings sorted by code, path, and message.
    pub issues: Vec<ScenarioCampaignIssue>,
    /// Number of scenario records in the input.
    pub scenario_count: usize,
    /// Number of records carrying a non-null numeric result (must be zero).
    pub numeric_result_count: usize,
}

impl ScenarioCampaignReport {
    /// Whether the definition is internally consistent and remains unexecuted.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Parse and validate a scenario campaign document.
pub fn validate_scenario_campaign_str(
    input: &str,
) -> Result<ScenarioCampaignReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_scenario_campaign(&value))
}

/// Validate the frozen campaign definition against the embedded source fixture.
///
/// Passing this function proves only that the scenario definitions match the
/// declared source disruptions and fail-closed contract. It proves no policy,
/// factory, product, quality, safety, cost, service, or resilience outcome.
pub fn validate_scenario_campaign(root: &Value) -> ScenarioCampaignReport {
    let mut issues = Vec::new();

    expect_string(root, "schema_version", "0.1.0", "", &mut issues);
    expect_string(
        root,
        "campaign_id",
        "HABITAT-COUNTERFACTUAL-001",
        "",
        &mut issues,
    );
    expect_string(
        root,
        "program_id",
        "HABITAT-VERTICAL-001",
        "",
        &mut issues,
    );
    expect_string(
        root,
        "evidence_status",
        "synthetic_contract_only_not_executed",
        "",
        &mut issues,
    );

    if !root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .is_some_and(|text| {
            text.contains("No numeric production results")
                && text.contains("physical measurements")
        })
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "campaign must deny numeric production and physical-measurement claims",
        );
    }

    let source_ids = validate_source_contract(root, &mut issues);
    let scorecard_metric_ids = validate_scorecard(root, &mut issues);
    validate_comparison_protocol(root, &mut issues);

    let scenarios = root.get("scenarios").and_then(Value::as_array);
    let scenario_count = scenarios.map_or(0, Vec::len);
    let mut numeric_result_count = 0;

    let Some(scenarios) = scenarios else {
        push(
            &mut issues,
            "SCENARIOS_MISSING",
            "scenarios",
            "scenarios must be an array",
        );
        issues.sort_by(|a, b| {
            (&a.code, &a.path, &a.message).cmp(&(&b.code, &b.path, &b.message))
        });
        return ScenarioCampaignReport {
            issues,
            scenario_count,
            numeric_result_count,
        };
    };

    if scenarios.len() != REQUIRED_SCENARIO_IDS.len() {
        push(
            &mut issues,
            "SCENARIO_COVERAGE_INCOMPLETE",
            "scenarios",
            format!(
                "expected exactly {} scenario records, found {}",
                REQUIRED_SCENARIO_IDS.len(),
                scenarios.len()
            ),
        );
    }

    let mut seen = HashSet::new();
    let mut actual_ids = HashSet::new();

    for (index, scenario) in scenarios.iter().enumerate() {
        let path = format!("scenarios[{index}]");
        let Some(id) = scenario.get("scenario_id").and_then(Value::as_str) else {
            push(
                &mut issues,
                "SCENARIO_ID_MISSING",
                format!("{path}.scenario_id"),
                "scenario_id must be a non-empty string",
            );
            continue;
        };

        if id.trim().is_empty() {
            push(
                &mut issues,
                "SCENARIO_ID_MISSING",
                format!("{path}.scenario_id"),
                "scenario_id must be a non-empty string",
            );
            continue;
        }
        if !seen.insert(id.to_owned()) {
            push(
                &mut issues,
                "SCENARIO_ID_DUPLICATE",
                format!("{path}.scenario_id"),
                format!("scenario ID {id:?} is duplicated"),
            );
        }
        actual_ids.insert(id.to_owned());

        if !REQUIRED_SCENARIO_IDS.contains(&id) {
            push(
                &mut issues,
                "SCENARIO_ID_UNEXPECTED",
                format!("{path}.scenario_id"),
                format!("scenario ID {id:?} is not part of the frozen campaign v0.1"),
            );
        }

        require_nonempty_object(scenario, "intervention", &path, &mut issues);
        require_nonempty_string(scenario, "expected_disposition", &path, &mut issues);
        require_nonempty_array(scenario, "hard_invariants", &path, &mut issues);
        require_nonempty_array(scenario, "required_outputs", &path, &mut issues);
        require_nonempty_array(scenario, "report_metrics", &path, &mut issues);
        require_nonempty_array(scenario, "execution_blockers", &path, &mut issues);

        expect_string(
            scenario,
            "execution_state",
            "specified_not_simulated",
            &path,
            &mut issues,
        );

        match scenario.get("numeric_results") {
            Some(value) if value.is_null() => {}
            Some(_) => {
                numeric_result_count += 1;
                push(
                    &mut issues,
                    "NUMERIC_RESULTS_IN_DEFINITION",
                    format!("{path}.numeric_results"),
                    "definition-only fixture must use null; execution outputs belong in a separate result artifact",
                );
            }
            None => push(
                &mut issues,
                "NUMERIC_RESULTS_STATUS_MISSING",
                format!("{path}.numeric_results"),
                "definition-only campaign must explicitly set numeric_results to null",
            ),
        }

        if let Some(metrics) = scenario.get("report_metrics").and_then(Value::as_array) {
            let mut seen_metrics = HashSet::new();
            for (metric_index, metric) in metrics.iter().enumerate() {
                let metric_id = metric.as_str().unwrap_or_default();
                if metric_id.is_empty() {
                    push(
                        &mut issues,
                        "SCENARIO_METRIC_INVALID",
                        format!("{path}.report_metrics[{metric_index}]"),
                        "each report metric must be a non-empty metric ID",
                    );
                    continue;
                }
                if !seen_metrics.insert(metric_id.to_owned()) {
                    push(
                        &mut issues,
                        "SCENARIO_METRIC_DUPLICATE",
                        format!("{path}.report_metrics[{metric_index}]"),
                        format!("metric ID {metric_id:?} is repeated in one scenario"),
                    );
                }
                if !scorecard_metric_ids.contains(metric_id) {
                    push(
                        &mut issues,
                        "SCENARIO_METRIC_UNKNOWN",
                        format!("{path}.report_metrics[{metric_index}]"),
                        format!("metric ID {metric_id:?} is not declared in scorecard.metrics"),
                    );
                }
            }
        }
    }

    if let Some(source_ids) = source_ids {
        if actual_ids != source_ids {
            push(
                &mut issues,
                "SCENARIO_SOURCE_COVERAGE_MISMATCH",
                "scenarios",
                "actual scenario ID set must exactly match source_contract.disruption_ids",
            );
        }
    }

    let required: HashSet<String> = REQUIRED_SCENARIO_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    if actual_ids != required {
        push(
            &mut issues,
            "SCENARIO_REQUIRED_SET_MISMATCH",
            "scenarios",
            "scenario set must exactly equal the eight frozen v0.1 disruptions",
        );
    }

    issues.sort_by(|a, b| {
        (&a.code, &a.path, &a.message).cmp(&(&b.code, &b.path, &b.message))
    });

    ScenarioCampaignReport {
        issues,
        scenario_count,
        numeric_result_count,
    }
}

fn validate_source_contract(
    root: &Value,
    issues: &mut Vec<ScenarioCampaignIssue>,
) -> Option<HashSet<String>> {
    let Some(source_contract) = root.get("source_contract") else {
        push(
            issues,
            "SOURCE_CONTRACT_MISSING",
            "source_contract",
            "source contract must identify the existing frozen disruption fixture",
        );
        return None;
    };

    expect_string(
        source_contract,
        "path",
        "docs/research/fixtures/habitat-vertical-v0.1.json",
        "source_contract",
        issues,
    );
    expect_string(
        source_contract,
        "schema_version",
        "0.1.0",
        "source_contract",
        issues,
    );

    let Some(declared) = source_contract
        .get("disruption_ids")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "SOURCE_DISRUPTION_IDS_MISSING",
            "source_contract.disruption_ids",
            "source contract must explicitly declare the disruption IDs it covers",
        );
        return None;
    };

    let declared_ids: Vec<String> = declared
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let declared_set: HashSet<String> = declared_ids.iter().cloned().collect();

    if declared_ids.len() != declared_set.len() {
        push(
            issues,
            "SOURCE_DISRUPTION_IDS_DUPLICATE",
            "source_contract.disruption_ids",
            "source disruption IDs must be unique",
        );
    }

    let embedded_source = serde_json::from_str::<Value>(EMBEDDED_FIXTURE);
    let Ok(embedded_source) = embedded_source else {
        push(
            issues,
            "EMBEDDED_SOURCE_INVALID",
            "source_contract",
            "embedded factory fixture could not be parsed",
        );
        return Some(declared_set);
    };

    let expected_source_ids: HashSet<String> = embedded_source
        .get("disruptions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    if declared_set != expected_source_ids {
        push(
            issues,
            "SOURCE_DISRUPTION_COVERAGE_MISMATCH",
            "source_contract.disruption_ids",
            "declared disruption ID set must exactly match the embedded source fixture",
        );
    }

    Some(declared_set)
}

fn validate_scorecard(
    root: &Value,
    issues: &mut Vec<ScenarioCampaignIssue>,
) -> HashSet<String> {
    let Some(scorecard) = root.get("scorecard") else {
        push(
            issues,
            "SCORECARD_MISSING",
            "scorecard",
            "scorecard is required",
        );
        return HashSet::new();
    };

    let mut ids = HashSet::new();
    let Some(metrics) = scorecard.get("metrics").and_then(Value::as_array) else {
        push(
            issues,
            "SCORECARD_METRICS_MISSING",
            "scorecard.metrics",
            "scorecard.metrics must be an array",
        );
        return ids;
    };

    if metrics.is_empty() {
        push(
            issues,
            "SCORECARD_METRICS_EMPTY",
            "scorecard.metrics",
            "at least one reportable metric is required",
        );
    }

    for (index, metric) in metrics.iter().enumerate() {
        let path = format!("scorecard.metrics[{index}]");
        let Some(id) = metric
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        else {
            push(
                issues,
                "SCORECARD_METRIC_ID_MISSING",
                format!("{path}.id"),
                "metric id must be a non-empty string",
            );
            continue;
        };

        if !ids.insert(id.to_owned()) {
            push(
                issues,
                "SCORECARD_METRIC_DUPLICATE",
                format!("{path}.id"),
                format!("metric ID {id:?} is duplicated"),
            );
        }

        require_nonempty_string(metric, "unit", &path, issues);
        require_nonempty_string(metric, "direction", &path, issues);
    }

    for required in ["lost_accepted_unit_hours", "hard_gate_violations"] {
        if !ids.contains(required) {
            push(
                issues,
                "SCORECARD_REQUIRED_METRIC_MISSING",
                "scorecard.metrics",
                format!("required metric {required:?} is missing"),
            );
        }
    }

    let hypothesis = scorecard.get("research_hypothesis");
    if hypothesis.is_none() {
        push(
            issues,
            "RESEARCH_HYPOTHESIS_MISSING",
            "scorecard.research_hypothesis",
            "proposed target and primary metric must be declared",
        );
    } else if let Some(hypothesis) = hypothesis {
        expect_string(
            hypothesis,
            "primary_metric",
            "lost_accepted_unit_hours",
            "scorecard.research_hypothesis",
            issues,
        );
        expect_string(
            hypothesis,
            "status",
            "proposed_not_tested",
            "scorecard.research_hypothesis",
            issues,
        );
        if hypothesis
            .get("target_direction")
            .and_then(Value::as_str)
            .is_none_or(|text| !(text.contains("10 percent") || text.contains("10_percent")))
        {
            push(
                issues,
                "RESEARCH_TARGET_MISSING",
                "scorecard.research_hypothesis.target_direction",
                "campaign target must retain the stated 10 percent hypothesis",
            );
        }
    }

    ids
}

fn validate_comparison_protocol(
    root: &Value,
    issues: &mut Vec<ScenarioCampaignIssue>,
) {
    let Some(protocol) = root.get("comparison_protocol") else {
        push(
            issues,
            "COMPARISON_PROTOCOL_MISSING",
            "comparison_protocol",
            "paired comparison contract is required",
        );
        return;
    };

    expect_string(
        protocol,
        "status",
        "blocked_policy_and_adapter_not_bound",
        "comparison_protocol",
        issues,
    );
    expect_string(
        protocol,
        "scheduler_adapter_status",
        "not_implemented_for_this_fixture",
        "comparison_protocol",
        issues,
    );
    expect_string(
        protocol,
        "quantitative_comparison_status",
        "not_run",
        "comparison_protocol",
        issues,
    );

    let Some(arms) = protocol.get("policy_arms").and_then(Value::as_array) else {
        push(
            issues,
            "POLICY_ARMS_MISSING",
            "comparison_protocol.policy_arms",
            "baseline and candidate policy slots must both be declared",
        );
        return;
    };

    let mut arm_ids = HashSet::new();
    if arms.len() != 2 {
        push(
            issues,
            "POLICY_ARM_COUNT_INVALID",
            "comparison_protocol.policy_arms",
            "exactly one baseline and one candidate arm are required",
        );
    }
    for (index, arm) in arms.iter().enumerate() {
        let path = format!("comparison_protocol.policy_arms[{index}]");
        let Some(id) = arm.get("arm_id").and_then(Value::as_str) else {
            push(
                issues,
                "POLICY_ARM_ID_MISSING",
                format!("{path}.arm_id"),
                "arm_id must be a non-empty string",
            );
            continue;
        };
        if !["baseline", "candidate"].contains(&id) {
            push(
                issues,
                "POLICY_ARM_ID_INVALID",
                format!("{path}.arm_id"),
                format!("unexpected comparison arm {id:?}"),
            );
        }
        arm_ids.insert(id.to_owned());
        expect_string(
            arm,
            "policy_identity_status",
            "unbound",
            &path,
            issues,
        );
        if !arm.get("policy_revision").is_some_and(Value::is_null) {
            push(
                issues,
                "POLICY_REVISION_MUST_REMAIN_UNBOUND",
                format!("{path}.policy_revision"),
                "no policy revision may be implied until exact canonical policies are inspected and pinned",
            );
        }
    }
    if arm_ids != HashSet::from(["baseline".to_owned(), "candidate".to_owned()]) {
        push(
            issues,
            "POLICY_ARM_SET_INVALID",
            "comparison_protocol.policy_arms",
            "policy arm set must contain exactly baseline and candidate",
        );
    }

    if protocol
        .get("fairness_rules")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        push(
            issues,
            "PAIRING_RULES_MISSING",
            "comparison_protocol.fairness_rules",
            "paired baseline/candidate input-equivalence rules are required",
        );
    }
}

fn expect_string(
    object: &Value,
    key: &str,
    expected: &str,
    prefix: &str,
    issues: &mut Vec<ScenarioCampaignIssue>,
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

fn require_nonempty_string(
    object: &Value,
    key: &str,
    prefix: &str,
    issues: &mut Vec<ScenarioCampaignIssue>,
) {
    if !object
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|text| !text.trim().is_empty())
    {
        push(
            issues,
            "FIELD_STRING_MISSING",
            format!("{prefix}.{key}"),
            "field must be a non-empty string",
        );
    }
}

fn require_nonempty_array(
    object: &Value,
    key: &str,
    prefix: &str,
    issues: &mut Vec<ScenarioCampaignIssue>,
) {
    if object
        .get(key)
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        push(
            issues,
            "FIELD_ARRAY_MISSING",
            format!("{prefix}.{key}"),
            "field must be a non-empty array",
        );
    }
}

fn require_nonempty_object(
    object: &Value,
    key: &str,
    prefix: &str,
    issues: &mut Vec<ScenarioCampaignIssue>,
) {
    if object
        .get(key)
        .and_then(Value::as_object)
        .is_none_or(serde_json::Map::is_empty)
    {
        push(
            issues,
            "FIELD_OBJECT_MISSING",
            format!("{prefix}.{key}"),
            "field must be a non-empty object",
        );
    }
}

fn push(
    issues: &mut Vec<ScenarioCampaignIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ScenarioCampaignIssue {
        code,
        path: path.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn campaign() -> Value {
        serde_json::from_str(EMBEDDED_SCENARIO_CAMPAIGN).unwrap()
    }

    fn has_code(report: &ScenarioCampaignReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_campaign_is_a_valid_definition_not_an_executed_result() {
        let report = validate_scenario_campaign(&campaign());
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.scenario_count, 8);
        assert_eq!(report.numeric_result_count, 0);
    }

    #[test]
    fn source_disruption_coverage_is_exact() {
        let value = campaign();
        let report = validate_scenario_campaign(&value);
        assert!(report.is_valid(), "{:?}", report.issues);

        let actual: HashSet<String> = value["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["scenario_id"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(actual.len(), REQUIRED_SCENARIO_IDS.len());
        assert!(REQUIRED_SCENARIO_IDS.iter().all(|id| actual.contains(*id)));
    }

    #[test]
    fn duplicate_scenario_id_is_rejected() {
        let mut value = campaign();
        let first_id = value["scenarios"][0]["scenario_id"].clone();
        value["scenarios"][1]["scenario_id"] = first_id;
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "SCENARIO_ID_DUPLICATE"
        ));
    }

    #[test]
    fn missing_scenario_is_rejected() {
        let mut value = campaign();
        value["scenarios"].as_array_mut().unwrap().pop();
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "SCENARIO_COVERAGE_INCOMPLETE"
        ));
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "SCENARIO_REQUIRED_SET_MISMATCH"
        ));
    }

    #[test]
    fn numeric_results_cannot_be_added_to_definition_fixture() {
        let mut value = campaign();
        value["scenarios"][0]["numeric_results"] = json!({"accepted_units": 12});
        let report = validate_scenario_campaign(&value);
        assert!(has_code(&report, "NUMERIC_RESULTS_IN_DEFINITION"));
        assert_eq!(report.numeric_result_count, 1);
    }

    #[test]
    fn executed_state_cannot_be_claimed_by_definition_only_fixture() {
        let mut value = campaign();
        value["scenarios"][0]["execution_state"] = json!("executed");
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "FIELD_VALUE_UNEXPECTED"
        ));
    }

    #[test]
    fn empty_hard_invariant_set_is_rejected() {
        let mut value = campaign();
        value["scenarios"][0]["hard_invariants"] = json!([]);
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "FIELD_ARRAY_MISSING"
        ));
    }

    #[test]
    fn scenario_cannot_reference_undeclared_metric() {
        let mut value = campaign();
        value["scenarios"][0]["report_metrics"][0] = json!("invented_metric");
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "SCENARIO_METRIC_UNKNOWN"
        ));
    }

    #[test]
    fn comparison_policies_cannot_appear_bound_in_definition_only_fixture() {
        let mut value = campaign();
        value["comparison_protocol"]["policy_arms"][0]["policy_identity_status"] =
            json!("bound");
        value["comparison_protocol"]["policy_arms"][0]["policy_revision"] =
            json!("deadbeef");
        let report = validate_scenario_campaign(&value);
        assert!(has_code(&report, "FIELD_VALUE_UNEXPECTED"));
        assert!(has_code(&report, "POLICY_REVISION_MUST_REMAIN_UNBOUND"));
    }

    #[test]
    fn missing_candidate_arm_is_rejected() {
        let mut value = campaign();
        value["comparison_protocol"]["policy_arms"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "POLICY_ARM_COUNT_INVALID"
        ));
    }

    #[test]
    fn source_contract_cannot_silently_omit_a_disruption() {
        let mut value = campaign();
        value["source_contract"]["disruption_ids"]
            .as_array_mut()
            .unwrap()
            .pop();
        let report = validate_scenario_campaign(&value);
        assert!(has_code(&report, "SOURCE_DISRUPTION_COVERAGE_MISMATCH"));
        assert!(has_code(&report, "SCENARIO_SOURCE_COVERAGE_MISMATCH"));
    }

    #[test]
    fn unsupported_evidence_status_is_rejected() {
        let mut value = campaign();
        value["evidence_status"] = json!("measured");
        assert!(has_code(
            &validate_scenario_campaign(&value),
            "FIELD_VALUE_UNEXPECTED"
        ));
    }

    #[test]
    fn report_order_is_deterministic() {
        let mut value = campaign();
        value["scenarios"][0]["report_metrics"][0] = json!("unknown_metric");
        let first = validate_scenario_campaign(&value);
        let second = validate_scenario_campaign(&value);
        assert_eq!(first, second);
    }
}
