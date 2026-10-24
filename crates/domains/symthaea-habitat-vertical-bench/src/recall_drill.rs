// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only recall-response drill for HABITAT-VERTICAL-001.
//!
//! It derives affected terminal products from a validated causal event log,
//! requires one typed response per terminal target, and refuses to call a
//! drill closed while any affected target is unresolved. It is not a real
//! recall execution system and does not establish source authenticity.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::Value;

pub const EMBEDDED_RECALL_DRILL: &str =
    include_str!("../../../../docs/research/fixtures/habitat-recall-closure-v0.1.json");

const CONTAINED_DISPOSITIONS: [&str; 3] = [
    "quarantined_at_facility",
    "withdrawal_confirmed_at_recipient",
    "customer_stop_use_confirmed",
];
const ALLOWED_DISPOSITIONS: [&str; 6] = [
    "quarantined_at_facility",
    "withdrawal_confirmed_at_recipient",
    "customer_stop_use_confirmed",
    "not_located",
    "action_pending",
    "confirmed_not_affected",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallReport {
    pub issues: Vec<RecallIssue>,
    pub source_lot_id: String,
    pub affected_terminal_ids: Vec<String>,
    pub target_count: usize,
    pub response_record_count: usize,
    pub confirmed_contained_count: usize,
    pub unresolved_affected_count: usize,
    pub false_positive_control_count: usize,
    pub closure_status: String,
}

impl RecallReport {
    /// True means the synthetic drill definition and oracle are internally consistent.
    /// It does not mean a real recall is closed or that any physical action occurred.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_recall_drill_str(
    drill_input: &str,
    base_input: &str,
    event_input: &str,
) -> Result<RecallReport, serde_json::Error> {
    let drill: Value = serde_json::from_str(drill_input)?;
    let base: Value = serde_json::from_str(base_input)?;
    let events: Value = serde_json::from_str(event_input)?;
    Ok(validate_recall_drill(&drill, &base, &events))
}

pub fn validate_recall_drill(
    drill: &Value,
    base: &Value,
    events: &Value,
) -> RecallReport {
    let mut issues = Vec::new();
    expect_string(drill, "schema_version", "0.1.0", &mut issues);
    expect_string(drill, "campaign_id", "HABITAT-RECALL-CLOSURE-001", &mut issues);
    expect_string(drill, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(
        drill,
        "evidence_status",
        "synthetic_rehearsal_contract_only_not_executed",
        &mut issues,
    );
    if !drill
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .is_some_and(|s| s.contains("does not prove a real recall"))
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "the drill must state that it does not prove a real recall or physical containment",
        );
    }

    let source_lot_id = drill
        .get("source_lot_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let mut node_kinds = BTreeMap::<String, String>::new();
    if let Some(nodes) = base.get("traceability_nodes").and_then(Value::as_array) {
        for node in nodes {
            if let (Some(id), Some(kind)) = (
                node.get("node_id").and_then(Value::as_str),
                node.get("node_kind").and_then(Value::as_str),
            ) {
                node_kinds.insert(id.to_owned(), kind.to_owned());
            }
        }
    } else {
        push(
            &mut issues,
            "BASE_TRACEABILITY_NODES_MISSING",
            "base.traceability_nodes",
            "the base fixture must declare traceability nodes",
        );
    }

    if node_kinds.get(&source_lot_id).map(String::as_str) != Some("material_lot") {
        push(
            &mut issues,
            "SOURCE_LOT_UNKNOWN",
            "source_lot_id",
            format!("source lot {source_lot_id:?} is not a declared material_lot"),
        );
    }

    let Some(target_values) = drill.get("terminal_target_ids").and_then(Value::as_array) else {
        push(
            &mut issues,
            "TERMINAL_TARGET_SCOPE_MISSING",
            "terminal_target_ids",
            "the recall target universe must be explicit",
        );
        return finish(issues, source_lot_id, Vec::new(), 0, 0, 0, 0, 0);
    };
    let mut target_ids = Vec::new();
    let mut target_set = BTreeSet::new();
    for (index, value) in target_values.iter().enumerate() {
        match value.as_str().filter(|s| !s.trim().is_empty()) {
            Some(id) => {
                if !target_set.insert(id.to_owned()) {
                    push(
                        &mut issues,
                        "DUPLICATE_TERMINAL_TARGET",
                        format!("terminal_target_ids[{index}]"),
                        format!("target {id:?} is listed more than once"),
                    );
                }
                if !node_kinds.contains_key(id) {
                    push(
                        &mut issues,
                        "TERMINAL_TARGET_UNKNOWN",
                        format!("terminal_target_ids[{index}]"),
                        format!("target {id:?} does not exist in the base traceability graph"),
                    );
                }
                target_ids.push(id.to_owned());
            }
            None => push(
                &mut issues,
                "TERMINAL_TARGET_ID_MISSING",
                format!("terminal_target_ids[{index}]"),
                "each target must be a non-empty string",
            ),
        }
    }

    let actual_terminal_scope: BTreeSet<String> = node_kinds
        .iter()
        .filter_map(|(id, kind)| {
            (kind == "packaged_unit" || kind == "control_packaged_unit").then(|| id.clone())
        })
        .collect();
    if target_set != actual_terminal_scope {
        push(
            &mut issues,
            "TERMINAL_TARGET_SCOPE_MISMATCH",
            "terminal_target_ids",
            "target universe must exactly cover all packaged and control-packaged unit nodes in the base fixture",
        );
    }

    // Do not derive recall results from an event stream which failed its own
    // causal, identity, timestamp, or object-lineage qualifier.
    let replay = crate::event_replay::validate_event_log(events, base);
    if !replay.is_valid() {
        for finding in &replay.issues {
            push(
                &mut issues,
                "EVENT_SOURCE_INVALID",
                format!("events.{}", finding.path),
                format!("{}: {}", finding.code, finding.message),
            );
        }
        return finish(
            issues,
            source_lot_id,
            Vec::new(),
            target_ids.len(),
            0,
            0,
            0,
            0,
        );
    }

    let mut events_by_id: HashMap<String, &Value> = HashMap::new();
    if let Some(rows) = events.get("events").and_then(Value::as_array) {
        for event in rows {
            if let Some(id) = event.get("event_id").and_then(Value::as_str) {
                events_by_id.insert(id.to_owned(), event);
            }
        }
    }

    let mut affected_objects = HashSet::<String>::new();
    affected_objects.insert(source_lot_id.clone());
    for event_id in &replay.replay_event_ids {
        let Some(event) = events_by_id.get(event_id) else {
            push(
                &mut issues,
                "REPLAY_EVENT_LOOKUP_FAILED",
                format!("events[{event_id}]"),
                "a canonical replay event id was not found in the source log",
            );
            continue;
        };
        let inputs = string_array_field(event, "input_ids");
        if inputs.iter().any(|id| affected_objects.contains(id)) {
            for output_id in string_array_field(event, "output_ids") {
                affected_objects.insert(output_id);
            }
        }
    }

    let mut affected_terminal_ids: Vec<String> = target_set
        .iter()
        .filter(|id| affected_objects.contains(id.as_str()))
        .cloned()
        .collect();
    affected_terminal_ids.sort();

    let expected_affected: BTreeSet<String> = read_string_array(
        drill.get("expected_affected_terminal_ids"),
        "expected_affected_terminal_ids",
        &mut issues,
    )
    .into_iter()
    .collect();
    let computed_affected: BTreeSet<String> = affected_terminal_ids.iter().cloned().collect();
    if expected_affected != computed_affected {
        push(
            &mut issues,
            "AFFECTED_TARGET_KNOWN_ANSWER_MISMATCH",
            "expected_affected_terminal_ids",
            format!(
                "expected {:?} but causal propagation derives {:?}",
                expected_affected, computed_affected
            ),
        );
    }

    let Some(records) = drill.get("response_records").and_then(Value::as_array) else {
        push(
            &mut issues,
            "RESPONSE_RECORDS_MISSING",
            "response_records",
            "response records must be an explicit array",
        );
        return finish(
            issues,
            source_lot_id,
            affected_terminal_ids,
            target_ids.len(),
            0,
            0,
            0,
            0,
        );
    };

    let affected_set: HashSet<&str> = affected_terminal_ids.iter().map(String::as_str).collect();
    let mut response_by_target = HashMap::<String, &Value>::new();
    let mut response_record_ids = HashSet::<String>::new();
    let mut response_record_count = 0;
    let mut confirmed_contained_count = 0;
    let mut unresolved_affected_count = 0;
    let mut false_positive_control_count = 0;

    for (index, record) in records.iter().enumerate() {
        let path = format!("response_records[{index}]");
        let record_id = record
            .get("response_record_id")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty());
        match record_id {
            Some(id) => {
                if !response_record_ids.insert(id.to_owned()) {
                    push(
                        &mut issues,
                        "DUPLICATE_RESPONSE_RECORD_ID",
                        format!("{path}.response_record_id"),
                        format!("response record id {id:?} is duplicated"),
                    );
                }
            }
            None => push(
                &mut issues,
                "RESPONSE_RECORD_ID_MISSING",
                format!("{path}.response_record_id"),
                "each response record requires a stable non-empty id",
            ),
        }

        let Some(target_id) = record
            .get("target_id")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        else {
            push(
                &mut issues,
                "RESPONSE_TARGET_ID_MISSING",
                format!("{path}.target_id"),
                "each response record requires a non-empty target id",
            );
            continue;
        };
        if !target_set.contains(target_id) {
            push(
                &mut issues,
                "RESPONSE_TARGET_OUT_OF_SCOPE",
                format!("{path}.target_id"),
                format!("response target {target_id:?} is outside the declared terminal target scope"),
            );
            continue;
        }
        if response_by_target.insert(target_id.to_owned(), record).is_some() {
            push(
                &mut issues,
                "DUPLICATE_RESPONSE_TARGET",
                format!("{path}.target_id"),
                format!("target {target_id:?} has more than one response record"),
            );
            continue;
        }
        response_record_count += 1;

        if record.get("evidence_status").and_then(Value::as_str) != Some("synthetic_only") {
            push(
                &mut issues,
                "RESPONSE_EVIDENCE_OVERCLAIM",
                format!("{path}.evidence_status"),
                "response states in this fixture must remain synthetic_only",
            );
        }
        if record
            .get("source_record_id")
            .and_then(Value::as_str)
            .is_none_or(|s| s.trim().is_empty())
        {
            push(
                &mut issues,
                "SOURCE_RECORD_ID_MISSING",
                format!("{path}.source_record_id"),
                "each declared response needs a source_record_id, even in this synthetic challenge",
            );
        }

        let disposition = record
            .get("disposition")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !ALLOWED_DISPOSITIONS.contains(&disposition) {
            push(
                &mut issues,
                "UNKNOWN_RESPONSE_DISPOSITION",
                format!("{path}.disposition"),
                format!("unsupported response disposition {disposition:?}"),
            );
            if affected_set.contains(target_id) {
                unresolved_affected_count += 1;
            }
            continue;
        }

        if affected_set.contains(target_id) {
            if CONTAINED_DISPOSITIONS.contains(&disposition) {
                confirmed_contained_count += 1;
            } else {
                unresolved_affected_count += 1;
                if disposition == "confirmed_not_affected" {
                    push(
                        &mut issues,
                        "AFFECTED_TARGET_MISCLASSIFIED",
                        format!("{path}.disposition"),
                        format!("causally affected target {target_id:?} cannot be declared unaffected"),
                    );
                }
            }
        } else if disposition != "confirmed_not_affected" {
            if CONTAINED_DISPOSITIONS.contains(&disposition) {
                false_positive_control_count += 1;
                push(
                    &mut issues,
                    "FALSE_POSITIVE_CONTROL_ACTION",
                    format!("{path}.disposition"),
                    format!("unaffected control target {target_id:?} was assigned a containment/withdrawal action"),
                );
            } else {
                push(
                    &mut issues,
                    "CONTROL_OUTCOME_UNRESOLVED",
                    format!("{path}.disposition"),
                    format!("unaffected control target {target_id:?} lacks an explicit confirmed_not_affected state"),
                );
            }
        }
    }

    for target_id in &target_ids {
        if !response_by_target.contains_key(target_id) {
            push(
                &mut issues,
                "MISSING_TARGET_RESPONSE",
                format!("response_records[{target_id}]"),
                "each terminal target requires exactly one explicit response record",
            );
            if affected_set.contains(target_id.as_str()) {
                unresolved_affected_count += 1;
            }
        }
    }

    let closure_status = if unresolved_affected_count > 0 {
        "OPEN_UNRESOLVED"
    } else if false_positive_control_count > 0 {
        "OPEN_FALSE_POSITIVE"
    } else if issues.is_empty() {
        "CLOSED_IN_SYNTHETIC_ORACLE"
    } else {
        "INVALID"
    }
    .to_owned();

    compare_u64(
        drill,
        "/expected_results/target_count",
        target_ids.len() as u64,
        &mut issues,
    );
    compare_u64(
        drill,
        "/expected_results/response_record_count",
        response_record_count as u64,
        &mut issues,
    );
    compare_u64(
        drill,
        "/expected_results/confirmed_contained_count",
        confirmed_contained_count as u64,
        &mut issues,
    );
    compare_u64(
        drill,
        "/expected_results/unresolved_affected_count",
        unresolved_affected_count as u64,
        &mut issues,
    );
    compare_u64(
        drill,
        "/expected_results/false_positive_control_count",
        false_positive_control_count as u64,
        &mut issues,
    );
    if drill.pointer("/expected_results/closure_status").and_then(Value::as_str)
        != Some(closure_status.as_str())
    {
        push(
            &mut issues,
            "RECALL_SUMMARY_MISMATCH",
            "expected_results.closure_status",
            format!(
                "expected {:?}, computed {:?}",
                drill.pointer("/expected_results/closure_status").and_then(Value::as_str),
                closure_status
            ),
        );
    }

    finish(
        issues,
        source_lot_id,
        affected_terminal_ids,
        target_ids.len(),
        response_record_count,
        confirmed_contained_count,
        unresolved_affected_count,
        false_positive_control_count,
    )
}

fn finish(
    mut issues: Vec<RecallIssue>,
    source_lot_id: String,
    mut affected_terminal_ids: Vec<String>,
    target_count: usize,
    response_record_count: usize,
    confirmed_contained_count: usize,
    unresolved_affected_count: usize,
    false_positive_control_count: usize,
) -> RecallReport {
    issues.sort_by(|a, b| {
        a.code
            .cmp(b.code)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.message.cmp(&b.message))
    });
    affected_terminal_ids.sort();
    let closure_status = if unresolved_affected_count > 0 {
        "OPEN_UNRESOLVED"
    } else if false_positive_control_count > 0 {
        "OPEN_FALSE_POSITIVE"
    } else if issues.is_empty() {
        "CLOSED_IN_SYNTHETIC_ORACLE"
    } else {
        "INVALID"
    }
    .to_owned();
    RecallReport {
        issues,
        source_lot_id,
        affected_terminal_ids,
        target_count,
        response_record_count,
        confirmed_contained_count,
        unresolved_affected_count,
        false_positive_control_count,
        closure_status,
    }
}

fn push(
    issues: &mut Vec<RecallIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(RecallIssue {
        code,
        path: path.into(),
        message: message.into(),
    });
}

fn expect_string(
    root: &Value,
    field: &str,
    expected: &str,
    issues: &mut Vec<RecallIssue>,
) {
    match root.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => {}
        Some(actual) => push(
            issues,
            "UNEXPECTED_FIELD_VALUE",
            field,
            format!("expected {expected:?}, got {actual:?}"),
        ),
        None => push(
            issues,
            "MISSING_REQUIRED_STRING",
            field,
            format!("expected required string {expected:?}"),
        ),
    }
}

fn read_string_array(
    value: Option<&Value>,
    path: &str,
    issues: &mut Vec<RecallIssue>,
) -> Vec<String> {
    let Some(values) = value.and_then(Value::as_array) else {
        push(
            issues,
            "STRING_ARRAY_MISSING",
            path,
            "expected an array of string identifiers",
        );
        return Vec::new();
    };
    let mut result = Vec::new();
    for (index, value) in values.iter().enumerate() {
        match value.as_str().filter(|s| !s.trim().is_empty()) {
            Some(item) => result.push(item.to_owned()),
            None => push(
                issues,
                "STRING_ARRAY_VALUE_INVALID",
                format!("{path}[{index}]"),
                "each identifier must be a non-empty string",
            ),
        }
    }
    result
}

fn string_array_field(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn compare_u64(
    root: &Value,
    pointer: &str,
    actual: u64,
    issues: &mut Vec<RecallIssue>,
) {
    match root.pointer(pointer).and_then(Value::as_u64) {
        Some(expected) if expected == actual => {}
        Some(expected) => push(
            issues,
            "RECALL_SUMMARY_MISMATCH",
            pointer.trim_start_matches('/').replace('/', "."),
            format!("expected {expected}, computed {actual}"),
        ),
        None => push(
            issues,
            "RECALL_SUMMARY_MISSING",
            pointer.trim_start_matches('/').replace('/', "."),
            "expected non-negative integer known answer is missing",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run() -> RecallReport {
        validate_recall_drill_str(
            EMBEDDED_RECALL_DRILL,
            crate::EMBEDDED_FIXTURE,
            crate::event_replay::EMBEDDED_EVENT_LOG,
        )
        .expect("embedded JSON must parse")
    }

    fn parsed() -> (Value, Value) {
        (
            serde_json::from_str(EMBEDDED_RECALL_DRILL).expect("drill JSON"),
            serde_json::from_str(crate::event_replay::EMBEDDED_EVENT_LOG).expect("event JSON"),
        )
    }

    fn has_code(report: &RecallReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_known_answer_is_consistent_but_recall_remains_open() {
        let report = run();
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.affected_terminal_ids, vec!["SYN-UNIT-PACKED-A", "SYN-UNIT-PACKED-B"]);
        assert_eq!(report.target_count, 3);
        assert_eq!(report.response_record_count, 3);
        assert_eq!(report.confirmed_contained_count, 1);
        assert_eq!(report.unresolved_affected_count, 1);
        assert_eq!(report.false_positive_control_count, 0);
        assert_eq!(report.closure_status, "OPEN_UNRESOLVED");
    }

    #[test]
    fn causal_event_propagation_reaches_both_affected_terminal_units() {
        let report = run();
        assert_eq!(report.source_lot_id, "SYN-LOT-LAYER-001");
        assert_eq!(report.affected_terminal_ids.len(), 2);
        assert!(!report.affected_terminal_ids.contains(&"SYN-UNRELATED-PACKED-CONTROL".to_owned()));
    }

    #[test]
    fn event_arrival_order_does_not_change_recall_result() {
        let (drill, mut events) = parsed();
        events["events"].as_array_mut().unwrap().reverse();
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.affected_terminal_ids, vec!["SYN-UNIT-PACKED-A", "SYN-UNIT-PACKED-B"]);
        assert_eq!(report.closure_status, "OPEN_UNRESOLVED");
    }

    #[test]
    fn unknown_source_lot_fails_closed() {
        let (mut drill, events) = parsed();
        drill["source_lot_id"] = json!("SYN-LOT-NOT-DECLARED");
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "SOURCE_LOT_UNKNOWN"));
    }

    #[test]
    fn tampered_affected_target_known_answer_is_rejected() {
        let (mut drill, events) = parsed();
        drill["expected_affected_terminal_ids"] = json!(["SYN-UNIT-PACKED-A"]);
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "AFFECTED_TARGET_KNOWN_ANSWER_MISMATCH"));
    }

    #[test]
    fn missing_response_for_affected_target_keeps_closure_open() {
        let (mut drill, events) = parsed();
        drill["response_records"].as_array_mut().unwrap().retain(|row| row["target_id"] != "SYN-UNIT-PACKED-B");
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "MISSING_TARGET_RESPONSE"));
        assert_eq!(report.unresolved_affected_count, 1);
    }

    #[test]
    fn duplicate_response_for_target_is_rejected() {
        let (mut drill, events) = parsed();
        let duplicate = drill["response_records"][0].clone();
        drill["response_records"].as_array_mut().unwrap().push(duplicate);
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "DUPLICATE_RESPONSE_TARGET"));
    }

    #[test]
    fn response_for_unknown_target_is_rejected() {
        let (mut drill, events) = parsed();
        drill["response_records"].as_array_mut().unwrap().push(json!({
            "response_record_id":"RR-UNKNOWN",
            "target_id":"SYN-UNIT-OUTSIDE-SCOPE",
            "disposition":"not_located",
            "evidence_status":"synthetic_only",
            "source_record_id":"SYN-RESPONSE-UNKNOWN"
        }));
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "RESPONSE_TARGET_OUT_OF_SCOPE"));
    }

    #[test]
    fn unrelated_control_must_not_be_quarantined_by_the_known_answer() {
        let (mut drill, events) = parsed();
        for row in drill["response_records"].as_array_mut().unwrap() {
            if row["target_id"] == "SYN-UNRELATED-PACKED-CONTROL" {
                row["disposition"] = json!("quarantined_at_facility");
            }
        }
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "FALSE_POSITIVE_CONTROL_ACTION"));
        assert_eq!(report.false_positive_control_count, 1);
    }

    #[test]
    fn affected_target_cannot_be_marked_confirmed_not_affected() {
        let (mut drill, events) = parsed();
        for row in drill["response_records"].as_array_mut().unwrap() {
            if row["target_id"] == "SYN-UNIT-PACKED-A" {
                row["disposition"] = json!("confirmed_not_affected");
            }
        }
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "AFFECTED_TARGET_MISCLASSIFIED"));
    }

    #[test]
    fn unknown_response_disposition_is_rejected() {
        let (mut drill, events) = parsed();
        drill["response_records"][0]["disposition"] = json!("probably_fine");
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "UNKNOWN_RESPONSE_DISPOSITION"));
    }

    #[test]
    fn physical_response_evidence_claim_is_rejected() {
        let (mut drill, events) = parsed();
        drill["response_records"][0]["evidence_status"] = json!("verified_physical");
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "RESPONSE_EVIDENCE_OVERCLAIM"));
    }

    #[test]
    fn invalid_event_source_prevents_recall_computation() {
        let (drill, mut events) = parsed();
        events["events"][0]["caused_by"] = json!(["EV-NOT-DECLARED"]);
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "EVENT_SOURCE_INVALID"));
        assert!(report.affected_terminal_ids.is_empty());
    }

    #[test]
    fn missing_claim_ceiling_is_rejected() {
        let (mut drill, events) = parsed();
        drill["claim_ceiling"] = json!("This proves a real recall closed.");
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "CLAIM_CEILING_MISSING"));
    }

    #[test]
    fn terminal_scope_must_include_every_packaged_control() {
        let (mut drill, events) = parsed();
        drill["terminal_target_ids"] = json!(["SYN-UNIT-PACKED-A", "SYN-UNIT-PACKED-B"]);
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "TERMINAL_TARGET_SCOPE_MISMATCH"));
    }

    #[test]
    fn summary_known_answer_is_independently_checked() {
        let (mut drill, events) = parsed();
        drill["expected_results"]["unresolved_affected_count"] = json!(0);
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "RECALL_SUMMARY_MISMATCH"));
    }

    #[test]
    fn duplicate_response_record_identity_is_rejected() {
        let (mut drill, events) = parsed();
        drill["response_records"][1]["response_record_id"] = drill["response_records"][0]["response_record_id"].clone();
        let base: Value = serde_json::from_str(crate::EMBEDDED_FIXTURE).unwrap();
        let report = validate_recall_drill(&drill, &base, &events);
        assert!(has_code(&report, "DUPLICATE_RESPONSE_RECORD_ID"));
    }
}
