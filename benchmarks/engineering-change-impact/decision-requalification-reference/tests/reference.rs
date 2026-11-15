#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CASES: &str = include_str!("../../decision-requalification-cases-v0.1.json");
const MUTATIONS: &str = include_str!("../../decision-requalification-mutations-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
struct Disposition {
    state: String,
    action: String,
    reason_codes: Vec<String>,
}

fn string_at<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string field {key}"))
}

fn array_at<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).and_then(Value::as_array).map(Vec::as_slice)
        .unwrap_or_else(|| panic!("missing array field {key}"))
}

fn strings_at(value: &Value, key: &str) -> Vec<String> {
    array_at(value, key).iter()
        .map(|v| v.as_str().expect("string array member").to_owned())
        .collect()
}

fn sha256(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).expect("canonical JSON projection");
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn dependency_manifest_projection(case: &Value) -> Value {
    let mut decisions: Vec<Value> = array_at(case, "decisions").iter().map(|d| {
        let mut dependency_ids = strings_at(d, "dependency_ids");
        dependency_ids.sort();
        json!({
            "decision_id": string_at(d, "decision_id"),
            "original_decision_at": string_at(d, "original_decision_at"),
            "knowledge_frontier_k0": string_at(d, "knowledge_frontier_k0"),
            "current_frontier_k1": string_at(d, "current_frontier_k1"),
            "pre_event_state": string_at(d, "pre_event_state"),
            "dependency_ids": dependency_ids,
            "effect_authority": string_at(d, "effect_authority")
        })
    }).collect();
    decisions.sort_by(|a, b| a["decision_id"].as_str().cmp(&b["decision_id"].as_str()));

    let mut dependencies: Vec<Value> = array_at(case, "dependencies").to_vec();
    dependencies.sort_by(|a, b| a["dependency_id"].as_str().cmp(&b["dependency_id"].as_str()));

    json!({
        "domain": "symthaea.engineering-decision-requalification.dependency-manifest.v0.1",
        "decisions": decisions,
        "dependencies": dependencies
    })
}

fn event_inventory_projection(case: &Value) -> Value {
    let mut events: Vec<Value> = array_at(case, "events").to_vec();
    events.sort_by(|a, b| a["event_id"].as_str().cmp(&b["event_id"].as_str()));
    json!({
        "domain": "symthaea.engineering-decision-requalification.event-inventory.v0.1",
        "event_inventory_state_at_k0": string_at(case, "event_inventory_state_at_k0"),
        "event_ordering_state_at_k0": string_at(case, "event_ordering_state_at_k0"),
        "event_inventory_state": string_at(case, "event_inventory_state"),
        "event_ordering_state": string_at(case, "event_ordering_state"),
        "events": events
    })
}

fn candidate_visible_event_projection(case: &Value, as_of_k0: bool) -> Value {
    let (domain, field) = if as_of_k0 {
        (
            "symthaea.engineering-decision-requalification.candidate-visible-events-k0.v0.1",
            "candidate_visible_event_ids_at_k0",
        )
    } else {
        (
            "symthaea.engineering-decision-requalification.candidate-visible-events-k1.v0.1",
            "candidate_visible_event_ids",
        )
    };
    let mut event_ids = strings_at(case, field);
    event_ids.sort();
    json!({
        "domain": domain,
        "event_inventory_sha256": string_at(case, "event_inventory_sha256"),
        "event_ids": event_ids
    })
}

fn action_for(state: &str) -> &'static str {
    match state {
        "supported" => "no_requalification",
        "requalification_required" => "hold_and_requalify",
        "withdrawn" => "withdraw_claim_and_hold",
        "indeterminate" => "hold_pending_evidence",
        other => panic!("unknown decision state {other}"),
    }
}

fn trigger_state(event_type: &str, reasons: &mut BTreeSet<String>) -> Option<&'static str> {
    match event_type {
        "evidence_superseded" => {
            reasons.insert("EVIDENCE_SUPERSEDED".to_owned());
            Some("requalification_required")
        }
        "configuration_changed" => {
            reasons.insert("CONFIGURATION_CHANGED".to_owned());
            Some("requalification_required")
        }
        "calibration_drift" => {
            reasons.insert("CALIBRATION_DRIFT".to_owned());
            Some("requalification_required")
        }
        "trust_revoked" => {
            reasons.insert("TRUST_REVOKED".to_owned());
            Some("withdrawn")
        }
        "evidence_withdrawn" => {
            reasons.insert("EVIDENCE_WITHDRAWN".to_owned());
            Some("withdrawn")
        }
        "authority_revoked" => {
            reasons.insert("AUTHORITY_REVOKED".to_owned());
            Some("withdrawn")
        }
        "operating_envelope_breach" => {
            reasons.insert("OPERATING_ENVELOPE_BREACH".to_owned());
            Some("withdrawn")
        }
        "duplicate_notice" => None,
        other => {
            reasons.insert(format!("EVENT_TYPE_UNQUALIFIED:{other}"));
            Some("indeterminate")
        }
    }
}

fn evaluate_frontier(case: &Value, decision: &Value, as_of_k0: bool) -> Disposition {
    let frontier = if as_of_k0 {
        string_at(decision, "knowledge_frontier_k0")
    } else {
        string_at(decision, "current_frontier_k1")
    };
    let inventory_state = if as_of_k0 {
        string_at(case, "event_inventory_state_at_k0")
    } else {
        string_at(case, "event_inventory_state")
    };
    let ordering_state = if as_of_k0 {
        string_at(case, "event_ordering_state_at_k0")
    } else {
        string_at(case, "event_ordering_state")
    };
    let mut reasons = BTreeSet::new();
    let mut indeterminate = false;
    let mut requires_requalification = false;
    let mut withdrawn = false;

    if inventory_state != "complete_frozen" {
        reasons.insert("EVENT_INVENTORY_INCOMPLETE".to_owned());
        indeterminate = true;
    }
    if ordering_state != "total" {
        reasons.insert("EVENT_ORDERING_UNQUALIFIED".to_owned());
        indeterminate = true;
    }

    let decision_dependencies: BTreeSet<String> =
        strings_at(decision, "dependency_ids").into_iter().collect();
    let visible_field = if as_of_k0 {
        "candidate_visible_event_ids_at_k0"
    } else {
        "candidate_visible_event_ids"
    };
    let visible_ids: BTreeSet<String> =
        strings_at(case, visible_field).into_iter().collect();
    let k0 = string_at(decision, "knowledge_frontier_k0");

    // Walk the complete owner-frozen event inventory, not merely the events
    // materialized into candidate context. Otherwise a missing relevant event
    // could be silently interpreted as "no change".
    for event in array_at(case, "events") {
        let event_id = string_at(event, "event_id");
        let effective_at = string_at(event, "effective_at");
        let observed_at = string_at(event, "observed_at");

        // Synthetic fixture timestamps are normalized fixed-width UTC RFC3339.
        // Production adapters must use parsed instants and clock uncertainty intervals.
        if effective_at > frontier || observed_at > frontier {
            continue;
        }

        let affected = event.get("affected_dependency_ids");
        let Some(affected_ids) = affected.and_then(Value::as_array) else {
            reasons.insert("TRIGGER_SCOPE_UNKNOWN".to_owned());
            if string_at(event, "qualification_state") != "qualified" {
                reasons.insert("TRIGGER_QUALIFICATION_UNKNOWN".to_owned());
            }
            indeterminate = true;
            continue;
        };
        let scope_matches = affected_ids.iter().any(|value| {
            value.as_str().is_some_and(|id| decision_dependencies.contains(id))
        });
        if !scope_matches {
            reasons.insert("NON_AFFECTING_EVENT_IGNORED".to_owned());
            continue;
        }

        if !visible_ids.contains(event_id) {
            reasons.insert("RELEVANT_EVENT_NOT_MATERIALIZED".to_owned());
            indeterminate = true;
            continue;
        }

        if string_at(event, "qualification_state") != "qualified" {
            reasons.insert("TRIGGER_QUALIFICATION_UNKNOWN".to_owned());
            indeterminate = true;
            continue;
        }

        if !as_of_k0 && effective_at <= k0 && observed_at > k0 {
            reasons.insert("LATE_ARRIVING_EVENT".to_owned());
        }

        match trigger_state(string_at(event, "event_type"), &mut reasons) {
            Some("withdrawn") => withdrawn = true,
            Some("requalification_required") => requires_requalification = true,
            Some("indeterminate") => indeterminate = true,
            Some(_) | None => {}
        }
    }

    let state = if indeterminate {
        "indeterminate"
    } else if withdrawn {
        "withdrawn"
    } else if requires_requalification {
        "requalification_required"
    } else {
        string_at(decision, "pre_event_state")
    };

    Disposition {
        state: state.to_owned(),
        action: action_for(state).to_owned(),
        reason_codes: reasons.into_iter().collect(),
    }
}

fn find_decision<'a>(case: &'a Value, decision_id: &str) -> &'a Value {
    array_at(case, "decisions").iter()
        .find(|d| string_at(d, "decision_id") == decision_id)
        .unwrap_or_else(|| panic!("decision {decision_id} missing from {}", string_at(case, "id")))
}

fn find_case<'a>(corpus: &'a Value, case_id: &str) -> &'a Value {
    array_at(corpus, "cases").iter()
        .find(|c| string_at(c, "id") == case_id)
        .unwrap_or_else(|| panic!("case {case_id} missing"))
}

#[test]
fn eleven_known_answer_cases_match_K0_K1_reference_and_manifest_digests() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid requalification corpus");
    assert_eq!(string_at(&corpus, "schema_version"), "engineering-decision-requalification-v0.1");
    assert_eq!(array_at(&corpus, "cases").len(), 11);

    let mut case_ids = BTreeSet::new();
    for case in array_at(&corpus, "cases") {
        let case_id = string_at(case, "id");
        assert!(case_ids.insert(case_id.to_owned()), "duplicate case ID {case_id}");
        assert_eq!(string_at(case, "effect_authority"), "none");
        assert_eq!(
            sha256(&dependency_manifest_projection(case)),
            string_at(case, "dependency_manifest_sha256"),
            "{case_id}: dependency manifest digest"
        );
        assert_eq!(
            sha256(&event_inventory_projection(case)),
            string_at(case, "event_inventory_sha256"),
            "{case_id}: event inventory digest"
        );
        assert_eq!(
            sha256(&candidate_visible_event_projection(case, true)),
            string_at(case, "candidate_visible_event_manifest_k0_sha256"),
            "{case_id}: K0 candidate-visible event digest"
        );
        assert_eq!(
            sha256(&candidate_visible_event_projection(case, false)),
            string_at(case, "candidate_visible_event_manifest_sha256"),
            "{case_id}: K1 candidate-visible event digest"
        );

        let dependency_ids: BTreeSet<String> = array_at(case, "dependencies").iter()
            .map(|dep| string_at(dep, "dependency_id").to_owned()).collect();
        assert_eq!(dependency_ids.len(), array_at(case, "dependencies").len(), "{case_id}: duplicate dependency");
        let event_ids: BTreeSet<String> = array_at(case, "events").iter()
            .map(|event| string_at(event, "event_id").to_owned()).collect();
        assert_eq!(event_ids.len(), array_at(case, "events").len(), "{case_id}: duplicate event");
        for id in strings_at(case, "candidate_visible_event_ids_at_k0") {
            assert!(event_ids.contains(&id), "{case_id}: K0 candidate-visible event {id} missing");
        }
        for id in strings_at(case, "candidate_visible_event_ids") {
            assert!(event_ids.contains(&id), "{case_id}: K1 candidate-visible event {id} missing");
        }

        let decision_ids: Vec<String> = array_at(case, "decisions").iter()
            .map(|d| string_at(d, "decision_id").to_owned()).collect();
        assert_eq!(decision_ids.iter().collect::<BTreeSet<_>>().len(), decision_ids.len(), "{case_id}: duplicate decision");

        for decision in array_at(case, "decisions") {
            assert_eq!(string_at(decision, "effect_authority"), "none");
            for dep_id in strings_at(decision, "dependency_ids") {
                assert!(dependency_ids.contains(&dep_id), "{case_id}: dependency {dep_id} missing");
            }

            let as_of = evaluate_frontier(case, decision, true);
            let current = evaluate_frontier(case, decision, false);
            assert_eq!(as_of.state, string_at(&decision["expected"], "as_of_state"),
                "{} / {} K0", case_id, string_at(decision, "decision_id"));
            assert_eq!(current.state, string_at(&decision["expected"], "current_state"),
                "{} / {} K1", case_id, string_at(decision, "decision_id"));
            assert_eq!(current.action, string_at(&decision["expected"], "current_action"),
                "{} / {} action", case_id, string_at(decision, "decision_id"));
            assert_eq!(
                serde_json::to_value(&current.reason_codes).unwrap(),
                decision["expected"]["reason_codes"],
                "{} / {} reason codes", case_id, string_at(decision, "decision_id")
            );
        }
    }
}

#[test]
fn relevant_owner_events_missing_from_K0_or_K1_candidate_context_are_indeterminate() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid requalification corpus");

    let late = find_case(&corpus, "RQ-NEG-002");
    let late_decision = find_decision(late, "D-BOND-03");
    let mut missing_k1 = late.clone();
    missing_k1["candidate_visible_event_ids"] = json!([]);
    let result_k1 = evaluate_frontier(&missing_k1, late_decision, false);
    assert_eq!(result_k1.state, "indeterminate");
    assert_eq!(result_k1.action, "hold_pending_evidence");
    assert!(result_k1.reason_codes.iter().any(|r| r == "RELEVANT_EVENT_NOT_MATERIALIZED"));
    assert_eq!(evaluate_frontier(late, late_decision, true).state, "supported");

    let known = find_case(&corpus, "RQ-NEG-006");
    let known_decision = find_decision(known, "D-VALVE-10");
    let mut missing_k0 = known.clone();
    missing_k0["candidate_visible_event_ids_at_k0"] = json!([]);
    let result_k0 = evaluate_frontier(&missing_k0, known_decision, true);
    assert_eq!(result_k0.state, "indeterminate");
    assert!(result_k0.reason_codes.iter().any(|r| r == "RELEVANT_EVENT_NOT_MATERIALIZED"));
}

#[test]
fn nine_intervention_pairs_preserve_history_and_scope_requalification() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid requalification corpus");
    let pairs = array_at(&corpus, "intervention_pairs");
    assert_eq!(pairs.len(), 9);
    let mut pair_ids = BTreeSet::new();

    for pair in pairs {
        let pair_id = string_at(pair, "id");
        assert!(pair_ids.insert(pair_id.to_owned()), "duplicate pair ID {pair_id}");
        let before_case = find_case(&corpus, string_at(pair, "before_case_id"));
        let after_case = find_case(&corpus, string_at(pair, "after_case_id"));
        let before_decision = find_decision(before_case, string_at(pair, "before_decision_id"));
        let after_decision = find_decision(after_case, string_at(pair, "after_decision_id"));
        let before_k0 = evaluate_frontier(before_case, before_decision, true);
        let before_k1 = evaluate_frontier(before_case, before_decision, false);
        let after_k0 = evaluate_frontier(after_case, after_decision, true);
        let after_k1 = evaluate_frontier(after_case, after_decision, false);

        match string_at(pair, "expected_relation") {
            "invariant" => {
                assert_eq!(before_k1.state, after_k1.state, "{pair_id}");
                assert_eq!(before_k1.action, after_k1.action, "{pair_id}");
                assert_eq!(after_k1.state, string_at(pair, "expected_current_state"), "{pair_id}");
            }
            "flip" => {
                assert_ne!(before_k1.state, after_k1.state, "{pair_id}");
                assert_eq!(before_k1.state, string_at(pair, "expected_before_current_state"), "{pair_id}");
                assert_eq!(after_k1.state, string_at(pair, "expected_after_current_state"), "{pair_id}");
                assert_eq!(after_k1.action, string_at(pair, "expected_after_action"), "{pair_id}");
            }
            "as_of_invariant_current_flip" => {
                assert_eq!(before_k0.state, after_k0.state, "{pair_id}: K0 must be invariant");
                assert_eq!(after_k0.state, string_at(pair, "expected_as_of_state"), "{pair_id}");
                assert_ne!(before_k1.state, after_k1.state, "{pair_id}: K1 must flip");
                assert_eq!(after_k1.state, string_at(pair, "expected_current_state"), "{pair_id}");
                assert_eq!(after_k1.action, string_at(pair, "expected_current_action"), "{pair_id}");
            }
            "scoped_invalidation" => {
                let affected_id = string_at(pair, "affected_decision_id");
                let unaffected_id = string_at(pair, "unaffected_decision_id");
                let before_affected = evaluate_frontier(before_case, find_decision(before_case, affected_id), false);
                let before_unaffected = evaluate_frontier(before_case, find_decision(before_case, unaffected_id), false);
                let after_affected = evaluate_frontier(after_case, find_decision(after_case, affected_id), false);
                let after_unaffected = evaluate_frontier(after_case, find_decision(after_case, unaffected_id), false);
                assert_eq!(before_affected.state, "supported", "{pair_id}");
                assert_eq!(before_unaffected.state, "supported", "{pair_id}");
                assert_eq!(after_affected.state, string_at(pair, "expected_affected_state"), "{pair_id}");
                assert_eq!(after_unaffected.state, string_at(pair, "expected_unaffected_state"), "{pair_id}");
            }
            "flip_to_indeterminate" | "flip_to_withdrawn" | "flip_to_requalification" => {
                assert_eq!(before_k1.state, "supported", "{pair_id}");
                assert_ne!(before_k1.state, after_k1.state, "{pair_id}");
                assert_eq!(after_k1.state, string_at(pair, "expected_current_state"), "{pair_id}");
                assert_eq!(after_k1.action, string_at(pair, "expected_current_action"), "{pair_id}");
            }
            "materialization_gate" => {
                assert_eq!(before_k0.state, string_at(pair, "expected_as_of_state"), "{pair_id}: before K0");
                assert_eq!(after_k0.state, string_at(pair, "expected_as_of_state"), "{pair_id}: after K0");
                assert_eq!(before_k1.state, string_at(pair, "expected_before_current_state"), "{pair_id}: materialized owner event");
                assert_eq!(after_k1.state, string_at(pair, "expected_after_current_state"), "{pair_id}: missing candidate materialization");
                assert_eq!(after_k1.action, string_at(pair, "expected_after_action"), "{pair_id}: missing event must block");
            }
            "historical_rejection" => {
                assert_eq!(before_k0.state, "supported", "{pair_id}");
                assert_eq!(after_k0.state, string_at(pair, "expected_as_of_state"), "{pair_id}");
                assert_eq!(after_k1.state, string_at(pair, "expected_current_state"), "{pair_id}");
            }
            other => panic!("{pair_id}: unknown expected relation {other}"),
        }
    }
}

#[test]
fn twenty_mutation_controls_are_unique_and_reference_known_cases() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid requalification corpus");
    let mutations: Value = serde_json::from_str(MUTATIONS).expect("valid mutation control manifest");
    assert_eq!(string_at(&mutations, "schema_version"), "engineering-decision-requalification-mutations-v0.1");
    assert_eq!(string_at(&mutations, "status"), "preregistered_controls_not_results");
    let controls = array_at(&mutations, "controls");
    assert_eq!(controls.len(), 20);
    assert_eq!(mutations["control_count"].as_u64(), Some(20));
    let case_ids: BTreeSet<String> = array_at(&corpus, "cases").iter()
        .map(|c| string_at(c, "id").to_owned()).collect();
    let mut control_ids = BTreeSet::new();

    for control in controls {
        let id = string_at(control, "id");
        assert!(control_ids.insert(id.to_owned()), "duplicate mutation ID {id}");
        assert!(case_ids.contains(string_at(control, "case_id")), "{id}: unknown case");
        assert!(!string_at(control, "mutation").trim().is_empty(), "{id}: empty mutation");
        assert!(!string_at(control, "reason").trim().is_empty(), "{id}: empty rationale");
        assert!(
            ["pass", "fail", "indeterminate"].contains(&string_at(control, "expected_result")),
            "{id}: invalid expected result"
        );
    }
}
