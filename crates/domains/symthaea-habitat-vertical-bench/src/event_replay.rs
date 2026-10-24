// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic event-log replay and lineage qualification for HABITAT-VERTICAL-001.
//!
//! This module checks event identity, causal references, source-record clocks,
//! deterministic replay order, and object ancestry against the frozen synthetic
//! traceability graph. It does not authenticate real sources, calculate
//! cryptographic digests, reconcile quantities, schedule work, or authorize
//! physical production.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::Value;

pub const EMBEDDED_EVENT_LOG: &str =
    include_str!("../../../../docs/research/fixtures/habitat-event-replay-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    pub issues: Vec<ReplayIssue>,
    pub unique_event_count: usize,
    pub duplicate_delivery_count: usize,
    pub replay_event_ids: Vec<String>,
}

impl ReplayReport {
    /// No findings means only that this synthetic replay contract is consistent.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_event_log_str(
    input: &str,
    base_fixture: &str,
) -> Result<ReplayReport, serde_json::Error> {
    let root: Value = serde_json::from_str(input)?;
    let base: Value = serde_json::from_str(base_fixture)?;
    Ok(validate_event_log(&root, &base))
}

pub fn validate_event_log(root: &Value, base: &Value) -> ReplayReport {
    let mut issues = Vec::new();
    expect_string(root, "schema_version", "0.1.0", &mut issues);
    expect_string(root, "campaign_id", "HABITAT-EVENT-REPLAY-001", &mut issues);
    expect_string(root, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(
        root,
        "evidence_status",
        "synthetic_contract_only_not_executed",
        &mut issues,
    );

    if !root.get("claim_ceiling").and_then(Value::as_str)
        .is_some_and(|s| s.contains("no physical event authenticity"))
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "the fixture must make the physical-evidence claim ceiling explicit",
        );
    }
    if root.pointer("/interoperability/conformance").and_then(Value::as_str)
        != Some("not_claimed")
    {
        push(
            &mut issues,
            "INTEROP_CONFORMANCE_OVERCLAIM",
            "interoperability.conformance",
            "the simplified event model must not claim GS1 EPCIS conformance",
        );
    }
    if root.pointer("/replay_contract/digest_status").and_then(Value::as_str)
        != Some("not_computed")
        || root.pointer("/replay_contract/quantity_semantics").and_then(Value::as_str)
            != Some("not_modeled")
        || root.pointer("/replay_contract/source_authentication").and_then(Value::as_str)
            != Some("not_modeled")
    {
        push(
            &mut issues,
            "UNMODELED_CONTROLS_OVERCLAIMED",
            "replay_contract",
            "digest, quantity and source-authentication limits must remain explicit",
        );
    }

    let known_objects: HashSet<String> = base
        .get("traceability_nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| node.get("node_id").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect();
    let known_operations: HashSet<String> = base
        .get("synthetic_process_graph")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|operation| operation.get("operation_id").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect();

    let Some(rows) = root.get("events").and_then(Value::as_array) else {
        push(
            &mut issues,
            "EVENTS_ARRAY_MISSING",
            "events",
            "events must be a JSON array",
        );
        return report(issues, 0, 0, Vec::new());
    };

    let mut events: BTreeMap<String, Value> = BTreeMap::new();
    let mut duplicate_delivery_count = 0;
    for (index, event) in rows.iter().enumerate() {
        let path = format!("events[{index}]");
        let Some(id) = event.get("event_id").and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        else {
            push(
                &mut issues,
                "EVENT_ID_MISSING",
                format!("{path}.event_id"),
                "every event delivery requires a stable non-empty event_id",
            );
            continue;
        };

        if let Some(existing) = events.get(id) {
            if same_event_payload(existing, event) {
                duplicate_delivery_count += 1;
            } else {
                push(
                    &mut issues,
                    "CONFLICTING_EVENT_ID",
                    format!("{path}.event_id"),
                    format!("event id {id:?} was redelivered with a conflicting payload"),
                );
            }
            continue;
        }
        events.insert(id.to_owned(), event.clone());
    }

    for (id, event) in &events {
        validate_event_shape(id, event, &known_objects, &known_operations, &mut issues);
    }

    let mut indegree: HashMap<String, usize> =
        events.keys().map(|id| (id.clone(), 0)).collect();
    let mut children: HashMap<String, Vec<String>> = HashMap::new();

    for (id, event) in &events {
        let causes = read_string_array(event, "caused_by", &format!("events[{id}].caused_by"), &mut issues);
        let event_time = event.get("event_time_utc").and_then(Value::as_str);
        let mut seen_causes = HashSet::new();
        for cause_id in causes {
            if !seen_causes.insert(cause_id.clone()) {
                push(
                    &mut issues,
                    "DUPLICATE_CAUSAL_REFERENCE",
                    format!("events[{id}].caused_by"),
                    format!("cause {cause_id:?} appears more than once"),
                );
                continue;
            }
            let Some(cause) = events.get(&cause_id) else {
                push(
                    &mut issues,
                    "UNKNOWN_CAUSAL_REFERENCE",
                    format!("events[{id}].caused_by"),
                    format!("cause event {cause_id:?} does not exist in this event log"),
                );
                continue;
            };
            let cause_time = cause.get("event_time_utc").and_then(Value::as_str);
            if let (Some(before), Some(after)) = (cause_time, event_time) {
                if valid_utc_timestamp(before) && valid_utc_timestamp(after) && before > after {
                    push(
                        &mut issues,
                        "CAUSAL_TIME_INVERSION",
                        format!("events[{id}].caused_by"),
                        format!("cause {cause_id:?} has an event time later than its dependent"),
                    );
                }
            }
            *indegree.entry(id.clone()).or_insert(0) += 1;
            children.entry(cause_id).or_default().push(id.clone());
        }
    }

    for list in children.values_mut() {
        list.sort();
    }
    let mut ready = BTreeSet::new();
    for (id, degree) in &indegree {
        if *degree == 0 {
            ready.insert((event_time_key(&events[id]), id.clone()));
        }
    }

    let mut replay_event_ids = Vec::with_capacity(events.len());
    while let Some((_, id)) = ready.iter().next().cloned() {
        ready.remove(&(event_time_key(&events[&id]), id.clone()));
        replay_event_ids.push(id.clone());
        if let Some(descendants) = children.get(&id) {
            for child in descendants {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert((event_time_key(&events[child]), child.clone()));
                    }
                }
            }
        }
    }
    if replay_event_ids.len() != events.len() {
        push(
            &mut issues,
            "CAUSAL_CYCLE",
            "events",
            "causal references contain a cycle; a complete replay order cannot be formed",
        );
    } else {
        validate_object_ancestry(&events, &replay_event_ids, &mut issues);
    }

    validate_known_answer(
        root,
        &events,
        duplicate_delivery_count,
        &replay_event_ids,
        &mut issues,
    );

    report(issues, events.len(), duplicate_delivery_count, replay_event_ids)
}

fn validate_event_shape(
    id: &str,
    event: &Value,
    known_objects: &HashSet<String>,
    known_operations: &HashSet<String>,
    issues: &mut Vec<ReplayIssue>,
) {
    let path = format!("events[{id}]");
    for field in [
        "event_type",
        "event_time_utc",
        "recorded_at_utc",
        "ingested_at_utc",
        "location_id",
        "actor_id",
        "business_step",
        "disposition",
        "source_revision",
        "evidence_status",
        "payload_digest_status",
    ] {
        if event.get(field).and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
            push(
                issues,
                "EVENT_FIELD_MISSING",
                format!("{path}.{field}"),
                "required event fields must be non-empty strings",
            );
        }
    }

    let event_time = event.get("event_time_utc").and_then(Value::as_str);
    let recorded_time = event.get("recorded_at_utc").and_then(Value::as_str);
    let ingested_time = event.get("ingested_at_utc").and_then(Value::as_str);
    for (field, value) in [
        ("event_time_utc", event_time),
        ("recorded_at_utc", recorded_time),
        ("ingested_at_utc", ingested_time),
    ] {
        if !value.is_some_and(valid_utc_timestamp) {
            push(
                issues,
                "INVALID_UTC_TIMESTAMP",
                format!("{path}.{field}"),
                "timestamp must use the strict YYYY-MM-DDTHH:MM:SSZ form",
            );
        }
    }
    if let (Some(event_time), Some(recorded_time)) = (event_time, recorded_time) {
        if valid_utc_timestamp(event_time)
            && valid_utc_timestamp(recorded_time)
            && recorded_time < event_time
        {
            push(
                issues,
                "SOURCE_CLOCK_INVERSION",
                format!("{path}.recorded_at_utc"),
                "source-recorded time predates the event time; clock trust is unresolved",
            );
        }
    }
    if let (Some(recorded_time), Some(ingested_time)) = (recorded_time, ingested_time) {
        if valid_utc_timestamp(recorded_time)
            && valid_utc_timestamp(ingested_time)
            && ingested_time < recorded_time
        {
            push(
                issues,
                "INGEST_CLOCK_INVERSION",
                format!("{path}.ingested_at_utc"),
                "ingestion time predates the source-recorded time",
            );
        }
    }

    if event.get("evidence_status").and_then(Value::as_str) != Some("synthetic_only") {
        push(
            issues,
            "EVIDENCE_STATUS_OVERCLAIM",
            format!("{path}.evidence_status"),
            "this fixture may only admit explicitly synthetic evidence status",
        );
    }
    if event.get("payload_digest_status").and_then(Value::as_str) != Some("not_computed") {
        push(
            issues,
            "DIGEST_STATUS_OVERCLAIM",
            format!("{path}.payload_digest_status"),
            "cryptographic payload digests are not computed by this qualifier",
        );
    }

    let event_type = event.get("event_type").and_then(Value::as_str).unwrap_or_default();
    let subjects = read_string_array(event, "subject_ids", &format!("{path}.subject_ids"), issues);
    let inputs = read_string_array(event, "input_ids", &format!("{path}.input_ids"), issues);
    let outputs = read_string_array(event, "output_ids", &format!("{path}.output_ids"), issues);
    let causes = read_string_array(event, "caused_by", &format!("{path}.caused_by"), issues);

    let bad_shape = match event_type {
        "receiving" => !subjects.is_empty() && inputs.is_empty() && outputs.is_empty() && causes.is_empty(),
        "inspection_disposition" => !subjects.is_empty() && inputs.is_empty() && outputs.is_empty() && !causes.is_empty(),
        "transformation" | "final_inspection" | "packaging" => {
            subjects.is_empty() && !inputs.is_empty() && !outputs.is_empty() && !causes.is_empty()
        }
        "dispatch" => !subjects.is_empty() && inputs.is_empty() && outputs.is_empty() && !causes.is_empty(),
        _ => {
            push(
                issues,
                "UNKNOWN_EVENT_TYPE",
                format!("{path}.event_type"),
                "event type is not in the closed synthetic contract vocabulary",
            );
            false
        }
    };
    if !bad_shape {
        push(
            issues,
            "EVENT_SHAPE_INVALID",
            &path,
            "subjects/inputs/outputs/causal references do not match the event-type contract",
        );
    }

    let disposition = event.get("disposition").and_then(Value::as_str).unwrap_or_default();
    let disposition_ok = match event_type {
        "receiving" => disposition == "quarantined",
        "inspection_disposition" => {
            disposition == "released_for_synthetic_fixture" || disposition == "rejected"
        }
        "transformation" => disposition == "completed",
        "final_inspection" => disposition == "passed_for_synthetic_fixture" || disposition == "failed",
        "packaging" => disposition == "packaged_for_synthetic_fixture",
        "dispatch" => disposition == "dispatched_for_synthetic_fixture",
        _ => false,
    };
    if !disposition_ok {
        push(
            issues,
            "DISPOSITION_NOT_ALLOWED",
            format!("{path}.disposition"),
            "disposition is not allowed for this event type in the synthetic protocol",
        );
    }

    if let Some(operation_id) = event.get("operation_id").and_then(Value::as_str) {
        if !known_operations.contains(operation_id) {
            push(
                issues,
                "UNKNOWN_PROCESS_OPERATION",
                format!("{path}.operation_id"),
                format!("operation {operation_id:?} does not exist in the frozen source process graph"),
            );
        }
    }

    for field_ids in [&subjects, &inputs, &outputs] {
        for object_id in field_ids.iter() {
            if !known_objects.contains(object_id) {
                push(
                    issues,
                    "UNKNOWN_TRACEABILITY_OBJECT",
                    &path,
                    format!("object {object_id:?} is absent from the frozen traceability graph"),
                );
            }
        }
    }
}

fn validate_object_ancestry(
    events: &BTreeMap<String, Value>,
    replay_ids: &[String],
    issues: &mut Vec<ReplayIssue>,
) {
    let mut ancestor_events: HashMap<String, HashSet<String>> = HashMap::new();
    let mut ancestor_objects: HashMap<String, HashSet<String>> = HashMap::new();

    for id in replay_ids {
        let event = &events[id];
        let causes = read_string_array(event, "caused_by", &format!("events[{id}].caused_by"), issues);
        let mut inherited_events = HashSet::new();
        let mut inherited_objects = HashSet::new();
        for cause_id in causes {
            if !events.contains_key(&cause_id) {
                continue;
            }
            inherited_events.insert(cause_id.clone());
            if let Some(ancestors) = ancestor_events.get(&cause_id) {
                inherited_events.extend(ancestors.iter().cloned());
            }
            if let Some(objects) = ancestor_objects.get(&cause_id) {
                inherited_objects.extend(objects.iter().cloned());
            }
            if let Some(parent) = events.get(&cause_id) {
                for field in ["subject_ids", "input_ids", "output_ids"] {
                    if let Some(values) = parent.get(field).and_then(Value::as_array) {
                        inherited_objects.extend(values.iter().filter_map(Value::as_str).map(ToOwned::to_owned));
                    }
                }
            }
        }

        let event_type = event.get("event_type").and_then(Value::as_str).unwrap_or_default();
        let inputs = read_string_array(event, "input_ids", &format!("events[{id}].input_ids"), issues);
        for object_id in &inputs {
            if !inherited_objects.contains(object_id) {
                push(
                    issues,
                    "INPUT_OUTSIDE_CAUSAL_ANCESTRY",
                    format!("events[{id}].input_ids"),
                    format!("input {object_id:?} is not present in any declared causal ancestor"),
                );
            }
        }
        if event_type != "receiving" {
            let subjects = read_string_array(event, "subject_ids", &format!("events[{id}].subject_ids"), issues);
            for object_id in &subjects {
                if !inherited_objects.contains(object_id) {
                    push(
                        issues,
                        "SUBJECT_OUTSIDE_CAUSAL_ANCESTRY",
                        format!("events[{id}].subject_ids"),
                        format!("subject {object_id:?} is not present in any declared causal ancestor"),
                    );
                }
            }
        }
        for field in ["subject_ids", "input_ids", "output_ids"] {
            if let Some(values) = event.get(field).and_then(Value::as_array) {
                inherited_objects.extend(values.iter().filter_map(Value::as_str).map(ToOwned::to_owned));
            }
        }
        ancestor_events.insert(id.clone(), inherited_events);
        ancestor_objects.insert(id.clone(), inherited_objects);
    }
}

fn validate_known_answer(
    root: &Value,
    events: &BTreeMap<String, Value>,
    duplicate_delivery_count: usize,
    replay_ids: &[String],
    issues: &mut Vec<ReplayIssue>,
) {
    let expected_count = root.pointer("/expected_known_answer/unique_event_count").and_then(Value::as_u64);
    if expected_count != Some(events.len() as u64) {
        push(
            issues,
            "KNOWN_ANSWER_EVENT_COUNT_MISMATCH",
            "expected_known_answer.unique_event_count",
            "unique event count does not match the frozen known answer",
        );
    }
    let expected_duplicates = root.pointer("/expected_known_answer/exact_redelivery_count").and_then(Value::as_u64);
    if expected_duplicates != Some(duplicate_delivery_count as u64) {
        push(
            issues,
            "KNOWN_ANSWER_DUPLICATE_COUNT_MISMATCH",
            "expected_known_answer.exact_redelivery_count",
            "exact-redelivery count does not match the frozen known answer",
        );
    }
    let expected_ids = root.pointer("/expected_known_answer/replay_event_ids")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect::<Vec<_>>());
    if expected_ids.as_deref() != Some(replay_ids) {
        push(
            issues,
            "KNOWN_ANSWER_REPLAY_ORDER_MISMATCH",
            "expected_known_answer.replay_event_ids",
            "replayed event order differs from the frozen deterministic known answer",
        );
    }
}

fn same_event_payload(left: &Value, right: &Value) -> bool {
    let mut left = left.clone();
    let mut right = right.clone();
    if let Some(object) = left.as_object_mut() {
        object.remove("ingested_at_utc");
    }
    if let Some(object) = right.as_object_mut() {
        object.remove("ingested_at_utc");
    }
    left == right
}

fn read_string_array(
    root: &Value,
    field: &str,
    path: &str,
    issues: &mut Vec<ReplayIssue>,
) -> Vec<String> {
    let Some(values) = root.get(field).and_then(Value::as_array) else {
        push(
            issues,
            "EVENT_ARRAY_FIELD_MISSING",
            path,
            "event identity and lineage arrays must exist, even when empty",
        );
        return Vec::new();
    };
    let mut result = Vec::with_capacity(values.len());
    let mut seen = HashSet::new();
    for (index, value) in values.iter().enumerate() {
        match value.as_str().filter(|s| !s.trim().is_empty()) {
            Some(id) => {
                if !seen.insert(id.to_owned()) {
                    push(
                        issues,
                        "DUPLICATE_ARRAY_IDENTIFIER",
                        format!("{path}[{index}]"),
                        format!("identifier {id:?} appears more than once in the same array"),
                    );
                }
                result.push(id.to_owned());
            }
            None => push(
                issues,
                "INVALID_ARRAY_IDENTIFIER",
                format!("{path}[{index}]"),
                "array members must be non-empty strings",
            ),
        }
    }
    result
}

fn event_time_key(event: &Value) -> String {
    event.get("event_time_utc")
        .and_then(Value::as_str)
        .unwrap_or("9999-99-99T99:99:99Z")
        .to_owned()
}

fn valid_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || !bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
        })
    {
        return false;
    }

    let Some(year) = value[0..4].parse::<u32>().ok() else {
        return false;
    };
    let Some(month) = value[5..7].parse::<u32>().ok() else {
        return false;
    };
    let Some(day) = value[8..10].parse::<u32>().ok() else {
        return false;
    };
    let Some(hour) = value[11..13].parse::<u32>().ok() else {
        return false;
    };
    let Some(minute) = value[14..16].parse::<u32>().ok() else {
        return false;
    };
    let Some(second) = value[17..19].parse::<u32>().ok() else {
        return false;
    };

    if year == 0 || !(1..=12).contains(&month) || !(0..=23).contains(&hour)
        || minute > 59 || second > 59
    {
        return false;
    }
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn expect_string(root: &Value, field: &str, expected: &str, issues: &mut Vec<ReplayIssue>) {
    if root.get(field).and_then(Value::as_str) != Some(expected) {
        push(
            issues,
            "EXPECTED_STRING_MISMATCH",
            field,
            format!("required value is {expected:?}"),
        );
    }
}

fn push(
    issues: &mut Vec<ReplayIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ReplayIssue { code, path: path.into(), message: message.into() });
}

fn report(
    mut issues: Vec<ReplayIssue>,
    unique_event_count: usize,
    duplicate_delivery_count: usize,
    replay_event_ids: Vec<String>,
) -> ReplayReport {
    issues.sort_by(|a, b| a.code.cmp(b.code).then_with(|| a.path.cmp(&b.path)).then_with(|| a.message.cmp(&b.message)));
    ReplayReport { issues, unique_event_count, duplicate_delivery_count, replay_event_ids }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_pair() -> (Value, Value) {
        (
            serde_json::from_str(EMBEDDED_EVENT_LOG).expect("event fixture JSON"),
            serde_json::from_str(crate::EMBEDDED_FIXTURE).expect("base fixture JSON"),
        )
    }

    fn event_index(root: &Value, id: &str) -> usize {
        root["events"].as_array().expect("events")
            .iter().position(|event| event["event_id"].as_str() == Some(id))
            .expect("event id exists")
    }

    fn has_code(report: &ReplayReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_event_log_matches_known_answer() {
        let (root, base) = fixture_pair();
        let report = validate_event_log(&root, &base);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.unique_event_count, 22);
        assert_eq!(report.duplicate_delivery_count, 0);
    }

    #[test]
    fn arrival_order_permutation_preserves_canonical_replay() {
        let (mut root, base) = fixture_pair();
        let before = validate_event_log(&root, &base);
        root["events"].as_array_mut().expect("events").reverse();
        let after = validate_event_log(&root, &base);
        assert!(before.is_valid(), "{:?}", before.issues);
        assert!(after.is_valid(), "{:?}", after.issues);
        assert_eq!(before.replay_event_ids, after.replay_event_ids);
    }

    #[test]
    fn exact_redelivery_is_idempotent() {
        let (mut root, base) = fixture_pair();
        let duplicate = root["events"][0].clone();
        let mut duplicate = duplicate;
        duplicate["ingested_at_utc"] = Value::String("2026-10-19T11:00:00Z".to_owned());
        root["events"].as_array_mut().expect("events").push(duplicate);
        root["expected_known_answer"]["exact_redelivery_count"] = Value::from(1);
        let report = validate_event_log(&root, &base);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.unique_event_count, 22);
        assert_eq!(report.duplicate_delivery_count, 1);
    }

    #[test]
    fn conflicting_duplicate_identity_fails_closed() {
        let (mut root, base) = fixture_pair();
        let mut conflict = root["events"][0].clone();
        conflict["disposition"] = Value::String("rejected".to_owned());
        root["events"].as_array_mut().expect("events").push(conflict);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "CONFLICTING_EVENT_ID"));
    }

    #[test]
    fn unknown_causal_reference_is_not_ignored() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-09-CUT-FABRIC");
        root["events"][index]["caused_by"] = serde_json::json!(["EV-MISSING"]);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "UNKNOWN_CAUSAL_REFERENCE"));
    }

    #[test]
    fn causal_cycle_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["caused_by"] = serde_json::json!(["EV-18-DISPATCH-B"]);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "CAUSAL_CYCLE"));
    }

    #[test]
    fn missing_actor_fails_closed() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index].as_object_mut().expect("event").remove("actor_id");
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "EVENT_FIELD_MISSING"));
    }

    #[test]
    fn malformed_timestamp_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["event_time_utc"] = Value::String("yesterday".to_owned());
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "INVALID_UTC_TIMESTAMP"));
    }

    #[test]
    fn impossible_calendar_timestamp_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["event_time_utc"] = Value::String("2026-02-30T08:00:00Z".to_owned());
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "INVALID_UTC_TIMESTAMP"));
    }

    #[test]
    fn source_recorded_time_before_event_time_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["recorded_at_utc"] = Value::String("2026-10-18T07:59:59Z".to_owned());
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "SOURCE_CLOCK_INVERSION"));
    }

    #[test]
    fn unknown_graph_object_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["subject_ids"] = serde_json::json!(["REAL-LOT-NOT-IN-FIXTURE"]);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "UNKNOWN_TRACEABILITY_OBJECT"));
    }

    #[test]
    fn input_without_causal_ancestor_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-09-CUT-FABRIC");
        root["events"][index]["caused_by"] = serde_json::json!([]);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "INPUT_OUTSIDE_CAUSAL_ANCESTRY"));
    }

    #[test]
    fn unknown_process_operation_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-09-CUT-FABRIC");
        root["events"][index]["operation_id"] = Value::String("OP-DOES-NOT-EXIST".to_owned());
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "UNKNOWN_PROCESS_OPERATION"));
    }

    #[test]
    fn transformation_without_outputs_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-09-CUT-FABRIC");
        root["events"][index]["output_ids"] = serde_json::json!([]);
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "EVENT_SHAPE_INVALID"));
    }

    #[test]
    fn physical_evidence_or_digest_claim_is_rejected() {
        let (mut root, base) = fixture_pair();
        let index = event_index(&root, "EV-01-RECEIVE-FABRIC");
        root["events"][index]["evidence_status"] = Value::String("observed_physical".to_owned());
        root["events"][index]["payload_digest_status"] = Value::String("verified".to_owned());
        let report = validate_event_log(&root, &base);
        assert!(has_code(&report, "EVIDENCE_STATUS_OVERCLAIM"));
        assert!(has_code(&report, "DIGEST_STATUS_OVERCLAIM"));
    }
}
