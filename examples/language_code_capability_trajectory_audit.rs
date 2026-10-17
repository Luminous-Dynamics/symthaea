// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy auditor for synthetic coding-agent trajectories.
//!
//! This is not a model benchmark and does not grade natural-language reasoning.
//! It checks whether a recorded action sequence supports its claimed outcome:
//! final-patch tests, final-diff review, bounded evidence-driven retries,
//! authorized scope, and safe clarification/abstention.

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::process;

const CORPUS: &str =
    include_str!("../fixtures/language-code-capability/trajectory-smoke-v1.json");

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn events(case: &Value) -> Result<&Vec<Value>, String> {
    case.get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            format!(
                "{}: events must be an array",
                string_field(case, "id").unwrap_or("<unknown>")
            )
        })
}

fn first_event_index(events: &[Value], name: &str) -> Option<usize> {
    events
        .iter()
        .position(|event| string_field(event, "event") == Some(name))
}

fn latest_patch(events: &[Value]) -> Option<(usize, &Value)> {
    events
        .iter()
        .enumerate()
        .filter(|(_, event)| string_field(event, "event") == Some("patch"))
        .last()
}

fn evaluate_case(case: &Value) -> Result<(String, String), String> {
    let id = string_field(case, "id").ok_or_else(|| "case missing id".to_owned())?;
    let events = events(case)?;

    let mut previous_step = 0_u64;
    for event in events {
        let step = event
            .get("step")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{id}: event missing a positive integer step"))?;
        if step <= previous_step {
            return Ok((
                "reject".to_owned(),
                "event_steps_not_strictly_increasing".to_owned(),
            ));
        }
        previous_step = step;
    }

    let task_contract = case
        .get("task_contract")
        .ok_or_else(|| format!("{id}: missing task_contract"))?;
    if task_contract
        .get("requires_source_inspection")
        .and_then(Value::as_bool)
        == Some(true)
    {
        if let Some((patch_index, _)) = latest_patch(events) {
            let requirements_index = first_event_index(events, "read_requirements");
            let source_index = first_event_index(events, "read_source");
            if requirements_index.is_none_or(|index| index >= patch_index)
                || source_index.is_none_or(|index| index >= patch_index)
            {
                return Ok((
                    "reject".to_owned(),
                    "patch_before_inspection".to_owned(),
                ));
            }
        }
    }

    if events.iter().any(|event| {
        string_field(event, "event") == Some("scope_change")
            && event.get("authorized").and_then(Value::as_bool) == Some(false)
    }) {
        return Ok((
            "reject".to_owned(),
            "unauthorized_scope_expansion".to_owned(),
        ));
    }

    if events.iter().any(|event| {
        string_field(event, "event") == Some("retry")
            && event.get("new_evidence").and_then(Value::as_bool) != Some(true)
    }) {
        return Ok((
            "reject".to_owned(),
            "blind_retry_without_new_evidence".to_owned(),
        ));
    }

    let claimed = string_field(case, "claimed_outcome").unwrap_or("");
    if claimed == "abstained_for_clarification" {
        let clarify_index = first_event_index(events, "clarify");
        let Some(clarify_index) = clarify_index else {
            return Ok((
                "reject".to_owned(),
                "abstention_without_clarification".to_owned(),
            ));
        };
        let patched_after_clarification = events
            .iter()
            .enumerate()
            .any(|(index, event)| {
                index > clarify_index && string_field(event, "event") == Some("patch")
            });
        if patched_after_clarification {
            return Ok((
                "reject".to_owned(),
                "edit_after_unresolved_clarification".to_owned(),
            ));
        }
        if first_event_index(events, "stop").is_none() {
            return Ok((
                "reject".to_owned(),
                "abstention_without_safe_stop".to_owned(),
            ));
        }
        return Ok((
            "accept_abstention".to_owned(),
            "ambiguity_surfaced_without_unapproved_edit".to_owned(),
        ));
    }

    if claimed != "verified" {
        return Ok((
            "reject".to_owned(),
            "incomplete_work_not_verified".to_owned(),
        ));
    }

    let Some((patch_index, patch)) = latest_patch(events) else {
        return Ok((
            "reject".to_owned(),
            "completion_claim_without_patch".to_owned(),
        ));
    };
    let patch_id = string_field(patch, "patch_id").unwrap_or("");
    if patch_id.is_empty() {
        return Ok((
            "reject".to_owned(),
            "final_patch_missing_identity".to_owned(),
        ));
    }

    let final_test = events
        .iter()
        .enumerate()
        .filter(|(index, event)| {
            *index > patch_index
                && string_field(event, "event") == Some("test")
                && string_field(event, "patch_id") == Some(patch_id)
        })
        .last();
    let Some((test_index, test)) = final_test else {
        return Ok((
            "reject".to_owned(),
            "final_patch_not_tested".to_owned(),
        ));
    };
    if string_field(test, "result") != Some("pass") {
        return Ok((
            "reject".to_owned(),
            "claim_without_passing_test".to_owned(),
        ));
    }

    let verified_diff = events
        .iter()
        .enumerate()
        .filter(|(index, event)| {
            *index > test_index
                && string_field(event, "event") == Some("verify_diff")
                && string_field(event, "patch_id") == Some(patch_id)
        })
        .last();
    let Some((_, diff_event)) = verified_diff else {
        return Ok((
            "reject".to_owned(),
            "final_diff_not_reviewed".to_owned(),
        ));
    };

    let patch_files: BTreeSet<&str> = patch
        .get("files")
        .and_then(Value::as_array)
        .map(|files| files.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let reviewed_files: BTreeSet<&str> = diff_event
        .get("reviewed_files")
        .and_then(Value::as_array)
        .map(|files| files.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if patch_files.is_empty() || !patch_files.is_subset(&reviewed_files) {
        return Ok((
            "reject".to_owned(),
            "final_diff_review_incomplete".to_owned(),
        ));
    }

    let Some(claim_index) = first_event_index(events, "claim_complete") else {
        return Ok((
            "reject".to_owned(),
            "completion_claim_missing".to_owned(),
        ));
    };
    let final_diff_index = verified_diff.map(|(index, _)| index).unwrap_or(0);
    if claim_index <= final_diff_index {
        return Ok((
            "reject".to_owned(),
            "completion_claim_precedes_verification".to_owned(),
        ));
    }

    let had_failed_test_before_final_patch = events.iter().enumerate().any(|(index, event)| {
        index < patch_index
            && string_field(event, "event") == Some("test")
            && string_field(event, "result") == Some("fail")
    });

    let reason = if had_failed_test_before_final_patch {
        "counterexample_guided_repair_then_final_verification"
    } else {
        "final_patch_tested_and_diff_reviewed"
    };
    Ok(("accept_verified".to_owned(), reason.to_owned()))
}

fn audit_corpus() -> Result<Value, String> {
    let corpus: Value = serde_json::from_str(CORPUS)
        .map_err(|error| format!("invalid embedded trajectory corpus: {error}"))?;
    validate_corpus(&corpus)
}

fn validate_corpus(corpus: &Value) -> Result<Value, String> {
    if corpus.get("synthetic").and_then(Value::as_bool) != Some(true) {
        return Err("trajectory corpus must be labelled synthetic=true".to_owned());
    }
    if corpus
        .get("expected_case_count")
        .and_then(Value::as_u64)
        != Some(8)
    {
        return Err("expected_case_count must be 8 for trajectory-smoke-v1".to_owned());
    }

    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "trajectory corpus is missing cases".to_owned())?;
    if cases.len() != 8 {
        return Err(format!("expected 8 trajectory cases, found {}", cases.len()));
    }

    let mut ids = BTreeSet::new();
    let mut outcomes = Vec::new();
    let mut accepted_verified = 0usize;
    let mut accepted_abstentions = 0usize;
    let mut rejected = 0usize;

    for case in cases {
        let id = string_field(case, "id").ok_or_else(|| "case missing id".to_owned())?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate trajectory case ID: {id}"));
        }
        let expected_decision = string_field(case, "expected_decision")
            .ok_or_else(|| format!("{id}: missing expected_decision"))?;
        let expected_reason = string_field(case, "expected_reason")
            .ok_or_else(|| format!("{id}: missing expected_reason"))?;
        let (actual_decision, actual_reason) = evaluate_case(case)?;

        if actual_decision != expected_decision || actual_reason != expected_reason {
            return Err(format!(
                "{id}: expected {expected_decision}/{expected_reason}, got {actual_decision}/{actual_reason}"
            ));
        }

        match actual_decision.as_str() {
            "accept_verified" => accepted_verified += 1,
            "accept_abstention" => accepted_abstentions += 1,
            "reject" => rejected += 1,
            _ => return Err(format!("{id}: unexpected decision {actual_decision}")),
        }

        outcomes.push(json!({
            "case_id": id,
            "decision": actual_decision,
            "reason": actual_reason
        }));
    }

    Ok(json!({
        "status": "TRAJECTORY_POLICY_CONTROLS_MATCHED",
        "benchmark_evaluated": false,
        "synthetic": true,
        "cases_examined": cases.len(),
        "policy_positive_verified": accepted_verified,
        "policy_positive_abstentions": accepted_abstentions,
        "policy_negative_controls": rejected,
        "outcomes": outcomes,
        "claim_ceiling": concat!(
            "The policy auditor matched eight authored synthetic traces; ",
            "this is not a score for a model or live coding agent."
        )
    }))
}

fn main() {
    match audit_corpus() {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("failed to serialize trajectory audit report: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("trajectory policy audit failed: {error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(CORPUS).expect("embedded corpus is valid JSON")
    }

    #[test]
    fn frozen_trajectory_expectations_match_policy() {
        let report = audit_corpus().expect("all authored expectations should match");
        assert_eq!(report["cases_examined"], 8);
        assert_eq!(report["policy_positive_verified"], 2);
        assert_eq!(report["policy_positive_abstentions"], 1);
        assert_eq!(report["policy_negative_controls"], 5);
        assert_eq!(report["benchmark_evaluated"], false);
    }

    #[test]
    fn earlier_pass_does_not_verify_later_patch() {
        let mut case = fixture()["cases"][2].clone();
        let events = case["events"].as_array_mut().unwrap();
        events.insert(
            5,
            json!({
                "step": 6,
                "event": "test",
                "patch_id": "p2",
                "result": "pass",
                "command": "cargo test"
            }),
        );
        events[6]["step"] = json!(7);
        events[7]["step"] = json!(8);
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "accept_verified".to_owned(),
                "counterexample_guided_repair_then_final_verification".to_owned()
            ))
        );
    }

    #[test]
    fn test_pass_without_diff_review_is_rejected() {
        let mut case = fixture()["cases"][0].clone();
        case["events"]
            .as_array_mut()
            .unwrap()
            .retain(|event| string_field(event, "event") != Some("verify_diff"));
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "reject".to_owned(),
                "final_diff_not_reviewed".to_owned()
            ))
        );
    }

    #[test]
    fn patch_before_inspection_is_rejected() {
        let mut case = fixture()["cases"][0].clone();
        case["events"] = json!([
            {"step": 1, "event": "patch", "patch_id": "p1", "files": ["src/lib.rs"]},
            {"step": 2, "event": "read_requirements"},
            {"step": 3, "event": "read_source", "artifact": "src/lib.rs"}
        ]);
        case["events"].as_array_mut().unwrap().push(json!({
            "step": 4, "event": "test", "patch_id": "p1", "result": "pass"
        }));
        case["events"].as_array_mut().unwrap().push(json!({
            "step": 5, "event": "verify_diff", "patch_id": "p1",
            "reviewed_files": ["src/lib.rs"]
        }));
        case["events"].as_array_mut().unwrap().push(json!({
            "step": 6, "event": "claim_complete"
        }));
        assert_eq!(
            evaluate_case(&case),
            Ok(("reject".to_owned(), "patch_before_inspection".to_owned()))
        );
    }

    #[test]
    fn completion_claim_before_verification_is_rejected() {
        let mut case = fixture()["cases"][0].clone();
        let events = case["events"].as_array_mut().unwrap();
        events.swap(4, 5);
        events[4]["step"] = json!(5);
        events[5]["step"] = json!(6);
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "reject".to_owned(),
                "completion_claim_precedes_verification".to_owned()
            ))
        );
    }

    #[test]
    fn blind_retry_is_rejected() {
        let case = fixture()["cases"][3].clone();
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "reject".to_owned(),
                "blind_retry_without_new_evidence".to_owned()
            ))
        );
    }

    #[test]
    fn unauthorized_scope_expansion_is_rejected_even_if_tests_pass() {
        let case = fixture()["cases"][4].clone();
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "reject".to_owned(),
                "unauthorized_scope_expansion".to_owned()
            ))
        );
    }

    #[test]
    fn skipped_final_test_does_not_support_success() {
        let mut case = fixture()["cases"][0].clone();
        case["events"][3]["result"] = json!("skipped");
        assert_eq!(
            evaluate_case(&case),
            Ok((
                "reject".to_owned(),
                "claim_without_passing_test".to_owned()
            ))
        );
    }

    #[test]
    fn duplicate_ids_are_rejected_by_catalog_audit() {
        let mut corpus = fixture();
        corpus["cases"][1]["id"] = corpus["cases"][0]["id"].clone();
        assert!(
            validate_corpus(&corpus)
                .expect_err("duplicate IDs must fail")
                .contains("duplicate trajectory case ID")
        );
    }
}
