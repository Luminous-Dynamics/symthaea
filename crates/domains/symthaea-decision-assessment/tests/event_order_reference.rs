// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent reference checks for the partial-order stop/effect contract.
//!
//! The reference comparator below is separately expressed from the production
//! evaluator. Frozen gold assertions, reference ordering, exact trace digests,
//! and delivery-order permutations are all checked independently.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use symthaea_decision_assessment::{
    DecisionEvent, DecisionEventKind, EventOrder, EventTimeInterval, EventTraceDisposition,
    EventTraceQualification, EvaluationError, GateState, ReconciledOperationOutcome,
    evaluate_event_order,
};

const FIXTURE: &str = include_str!(
    "../../../../benchmarks/engineering-change-impact/action-event-order-v0.1.json"
);
const TRACE_DOMAIN: &str = "symthaea.engineering-decision.event-trace.v0.1";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("event-order fixture must be valid JSON")
}

fn parse_qualification(value: &Value) -> EventTraceQualification {
    serde_json::from_value(value.clone()).expect("event trace qualification")
}

fn parse_events(value: &Value) -> Vec<DecisionEvent> {
    serde_json::from_value(value.clone()).expect("decision event list")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Canonical trace digest contract: sort event records by event_id; recursively
/// sorted JSON object keys are provided by serde_json::Value's default map.
fn canonical_trace_digest(case: &Value) -> String {
    let mut events = case["events"].as_array().expect("events").clone();
    events.sort_by(|left, right| {
        left["event_id"]
            .as_str()
            .cmp(&right["event_id"].as_str())
    });

    let projection = json!({
        "domain": TRACE_DOMAIN,
        "qualification": case["qualification"].clone(),
        "events": events,
    });
    sha256_hex(&serde_json::to_vec(&projection).expect("canonical projection"))
}

fn reaches(ancestor_id: &str, descendant_id: &str, events: &[DecisionEvent]) -> bool {
    let by_id: BTreeMap<&str, &DecisionEvent> = events
        .iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect();
    let mut pending = vec![descendant_id.to_owned()];
    let mut visited = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        let Some(event) = by_id.get(current.as_str()) else {
            continue;
        };
        for parent in &event.causal_predecessors {
            if parent == ancestor_id {
                return true;
            }
            pending.push(parent.clone());
        }
    }
    false
}

/// Independent event-order oracle. The returned relation describes first
/// relative to second (e.g. BEFORE means first happened before second).
fn reference_order(
    first: &DecisionEvent,
    second: &DecisionEvent,
    events: &[DecisionEvent],
    qualification: &EventTraceQualification,
) -> EventOrder {
    let mut definite = BTreeSet::new();
    let mut clock_overlap = false;

    if qualification.causal_ordering == GateState::Pass {
        if reaches(&first.event_id, &second.event_id, events) {
            definite.insert(EventOrder::Before);
        }
        if reaches(&second.event_id, &first.event_id, events) {
            definite.insert(EventOrder::After);
        }
    }

    if let (Some(a), Some(b)) = (&first.clock, &second.clock) {
        if a.domain_id == b.domain_id
            && a.sequence_qualification == GateState::Pass
            && b.sequence_qualification == GateState::Pass
        {
            match (a.sequence_number, b.sequence_number) {
                (Some(left), Some(right)) if left < right => {
                    definite.insert(EventOrder::Before);
                }
                (Some(left), Some(right)) if left > right => {
                    definite.insert(EventOrder::After);
                }
                (Some(_), Some(_)) => return EventOrder::Inconsistent,
                _ => {}
            }
        }

        if a.clock_qualification == GateState::Pass
            && b.clock_qualification == GateState::Pass
        {
            if let (Some(left), Some(right)) = (&a.interval, &b.interval) {
                if left.latest_unix_ms < right.earliest_unix_ms {
                    definite.insert(EventOrder::Before);
                } else if right.latest_unix_ms < left.earliest_unix_ms {
                    definite.insert(EventOrder::After);
                } else {
                    clock_overlap = true;
                }
            }
        }
    }

    if definite.len() > 1 {
        EventOrder::Inconsistent
    } else if let Some(order) = definite.into_iter().next() {
        order
    } else if clock_overlap {
        EventOrder::Overlapping
    } else {
        EventOrder::Unknown
    }
}

fn expect_invalid(events: &[DecisionEvent], qualification: &EventTraceQualification) {
    assert!(matches!(
        evaluate_event_order(events, qualification),
        Err(EvaluationError::InvalidEventTrace(_))
    ));
}

fn baseline_qualification() -> EventTraceQualification {
    EventTraceQualification {
        identity: GateState::Pass,
        digest_integrity: GateState::Pass,
        authenticity: GateState::Pass,
        scope: GateState::Pass,
        completeness: GateState::Pass,
        causal_ordering: GateState::Pass,
        stop_disposition: GateState::NotApplicable,
        stop_disposition_receipt_ref: None,
        stop_disposition_receipt_sha256: None,
        stop_disposition_trigger_event_id: None,
        stop_disposition_event_id: None,
    }
}

fn test_clock(
    domain: &str,
    sequence: u64,
    time: i64,
) -> Option<symthaea_decision_assessment::EventClockObservation> {
    Some(symthaea_decision_assessment::EventClockObservation {
        domain_id: domain.to_owned(),
        sequence_number: Some(sequence),
        sequence_qualification: GateState::Pass,
        interval: Some(EventTimeInterval {
            earliest_unix_ms: time,
            latest_unix_ms: time,
        }),
        clock_qualification: GateState::Pass,
    })
}

#[test]
fn eleven_frozen_event_traces_match_gold_and_independent_ordering() {
    let data = fixture();
    assert_eq!(
        data["schema_version"].as_str(),
        Some("engineering-decision-event-order-v0.1")
    );
    let cases = data["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 11);

    let mut seen_case_ids = BTreeSet::new();
    for case in cases {
        let case_id = case["id"].as_str().expect("case ID");
        assert!(seen_case_ids.insert(case_id.to_owned()), "duplicate {case_id}");

        assert_eq!(
            canonical_trace_digest(case),
            case["event_trace_sha256"].as_str().expect("trace digest"),
            "{case_id}: canonical event trace commitment"
        );

        let events = parse_events(&case["events"]);
        let qualification = parse_qualification(&case["qualification"]);
        let report = evaluate_event_order(&events, &qualification)
            .unwrap_or_else(|error| panic!("{case_id}: {error}"));
        let report_json = serde_json::to_value(&report).expect("report JSON");
        assert_eq!(report_json, case["expected"], "{case_id}: frozen gold");

        // Independently derive every reported effect-vs-trigger relation.
        let trigger = events
            .iter()
            .find(|event| event.kind == DecisionEventKind::StopTriggerObserved);
        let effects: Vec<&DecisionEvent> = events
            .iter()
            .filter(|event| event.kind == DecisionEventKind::IrreversibleEffectCommitted)
            .collect();

        if let Some(trigger_event) = trigger {
            assert_eq!(
                effects.len(),
                report.effect_orders.len(),
                "{case_id}: effect relation denominator"
            );
            for effect in effects {
                let reference = reference_order(effect, trigger_event, &events, &qualification);
                assert_eq!(
                    report.effect_orders.get(&effect.event_id),
                    Some(&reference),
                    "{case_id}: effect-relative-to-trigger relation for {}",
                    effect.event_id
                );
            }
        } else {
            assert!(report.effect_orders.is_empty(), "{case_id}");
        }

        let expected_disposition = match case["expected"]["disposition"].as_str() {
            Some("safe") => EventTraceDisposition::Safe,
            Some("critical_failure") => EventTraceDisposition::CriticalFailure,
            Some("indeterminate") => EventTraceDisposition::Indeterminate,
            other => panic!("{case_id}: unknown expected disposition {other:?}"),
        };
        assert_eq!(report.disposition, expected_disposition, "{case_id}");
        assert_eq!(
            report.block_irreversible_actions,
            expected_disposition != EventTraceDisposition::Safe,
            "{case_id}: block state"
        );

        // Event delivery order is not a semantic input.
        let mut permuted = events.clone();
        permuted.reverse();
        let permuted_report = evaluate_event_order(&permuted, &qualification)
            .unwrap_or_else(|error| panic!("{case_id} permuted: {error}"));
        assert_eq!(permuted_report, report, "{case_id}: delivery permutation");
    }
    assert_eq!(seen_case_ids.len(), 11);
}

#[test]
fn invalid_partial_orders_fail_closed() {
    let base = baseline_qualification();

    let start = DecisionEvent {
        event_id: "start".to_owned(),
        kind: DecisionEventKind::OperationStarted,
        operation_id: Some("OP-X".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 1, 100),
        causal_predecessors: vec![],
    };
    let stop = DecisionEvent {
        event_id: "stop".to_owned(),
        kind: DecisionEventKind::StopTriggerObserved,
        operation_id: None,
        reconciled_outcome: None,
        clock: test_clock("clock-a", 2, 110),
        causal_predecessors: vec![],
    };

    let mut duplicate = vec![start.clone(), stop.clone()];
    duplicate.push(start.clone());
    expect_invalid(&duplicate, &base);

    let mut missing_parent = stop.clone();
    missing_parent.causal_predecessors = vec!["not-present".to_owned()];
    expect_invalid(&[start.clone(), missing_parent], &base);

    let mut left = start.clone();
    left.causal_predecessors = vec!["right".to_owned()];
    let mut right = stop.clone();
    right.event_id = "right".to_owned();
    right.causal_predecessors = vec!["start".to_owned(), "stop".to_owned()];
    expect_invalid(&[left, right, stop.clone()], &base);

    let mut second_start = start.clone();
    second_start.event_id = "start-2".to_owned();
    expect_invalid(&[start.clone(), second_start], &base);

    let applied = DecisionEvent {
        event_id: "applied".to_owned(),
        kind: DecisionEventKind::OperationOutcomeReconciled,
        operation_id: Some("OP-X".to_owned()),
        reconciled_outcome: Some(ReconciledOperationOutcome::Applied),
        clock: test_clock("clock-a", 3, 120),
        causal_predecessors: vec![],
    };
    let not_applied = DecisionEvent {
        event_id: "not-applied".to_owned(),
        kind: DecisionEventKind::OperationOutcomeReconciled,
        operation_id: Some("OP-X".to_owned()),
        reconciled_outcome: Some(ReconciledOperationOutcome::NotApplied),
        clock: test_clock("clock-a", 4, 130),
        causal_predecessors: vec![],
    };
    expect_invalid(&[start.clone(), applied, not_applied], &base);

    let mut second_trigger = stop.clone();
    second_trigger.event_id = "stop-2".to_owned();
    expect_invalid(&[start.clone(), stop.clone(), second_trigger], &base);

    let mut malformed_interval = stop.clone();
    malformed_interval.clock.as_mut().unwrap().interval = Some(EventTimeInterval {
        earliest_unix_ms: 500,
        latest_unix_ms: 400,
    });
    expect_invalid(&[start, malformed_interval], &base);
}

#[test]
fn stop_disposition_pass_is_bound_to_exact_receipt_trigger_and_discharge() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-POS-001")
        .unwrap();
    let events = parse_events(&case["events"]);
    let mut qualification = parse_qualification(&case["qualification"]);

    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::Safe);
    assert!(!report.block_irreversible_actions);

    qualification.stop_disposition_event_id = Some("different-discharge".to_owned());
    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISPOSITION_RECEIPT_BINDING_INVALID".to_owned()));

    qualification = parse_qualification(&case["qualification"]);
    qualification.stop_disposition_trigger_event_id = Some("different-trigger".to_owned());
    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISPOSITION_RECEIPT_BINDING_INVALID".to_owned()));

    qualification = parse_qualification(&case["qualification"]);
    qualification.stop_disposition_receipt_sha256 = Some("not-a-digest".to_owned());
    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISPOSITION_RECEIPT_BINDING_INVALID".to_owned()));
}

#[test]
fn retry_is_allowed_only_after_not_applied_reconciliation_and_stop_discharge() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-POS-002")
        .unwrap();
    let mut events = parse_events(&case["events"]);
    let mut retry = events
        .iter()
        .find(|event| event.event_id == "outcome-not-applied")
        .unwrap()
        .clone();
    retry.event_id = "retry-after-discharge".to_owned();
    retry.kind = DecisionEventKind::RetryAttempted;
    retry.reconciled_outcome = None;
    let clock = retry.clock.as_mut().expect("fixture operation clock");
    clock.sequence_number = Some(8);
    clock.interval = Some(EventTimeInterval {
        earliest_unix_ms: 170,
        latest_unix_ms: 170,
    });
    retry.causal_predecessors = vec!["stop-revalidated".to_owned()];
    events.push(retry);

    let report = evaluate_event_order(&events, &parse_qualification(&case["qualification"]))
        .expect("retry after exact reconciliation and stop discharge");
    assert_eq!(report.disposition, EventTraceDisposition::Safe);
    assert!(!report.block_irreversible_actions);
    assert_eq!(report.retry_before_reconciliation_count, 0);
}

#[test]
fn retry_after_reconciliation_but_before_stop_discharge_is_critical() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-POS-002")
        .unwrap();
    let mut events = parse_events(&case["events"]);
    let discharge = events
        .iter_mut()
        .find(|event| event.event_id == "stop-revalidated")
        .unwrap();
    let clock = discharge.clock.as_mut().unwrap();
    clock.sequence_number = Some(9);
    clock.interval = Some(EventTimeInterval {
        earliest_unix_ms: 180,
        latest_unix_ms: 180,
    });

    let mut retry = events
        .iter()
        .find(|event| event.event_id == "outcome-not-applied")
        .unwrap()
        .clone();
    retry.event_id = "retry-before-discharge".to_owned();
    retry.kind = DecisionEventKind::RetryAttempted;
    retry.reconciled_outcome = None;
    let clock = retry.clock.as_mut().unwrap();
    clock.sequence_number = Some(8);
    clock.interval = Some(EventTimeInterval {
        earliest_unix_ms: 170,
        latest_unix_ms: 170,
    });
    retry.causal_predecessors = vec!["outcome-not-applied".to_owned(), "stop".to_owned()];
    events.push(retry);

    let report = evaluate_event_order(&events, &parse_qualification(&case["qualification"]))
        .expect("retry before stop discharge is a valid but unsafe trace");
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report.block_irreversible_actions);
    assert!(report
        .reason_codes
        .contains(&"RETRY_BEFORE_STOP_DISCHARGE:OP-6".to_owned()));
}

#[test]
fn stop_discharge_cannot_cover_a_preexisting_unreconciled_operation() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-UNK-003")
        .unwrap();
    let mut events = parse_events(&case["events"]);
    let mut discharge = events
        .iter()
        .find(|event| event.event_id == "stop")
        .unwrap()
        .clone();
    discharge.event_id = "stop-discharge".to_owned();
    discharge.kind = DecisionEventKind::StopObligationDischarged;
    discharge.operation_id = None;
    discharge.reconciled_outcome = None;
    let clock = discharge.clock.as_mut().unwrap();
    clock.sequence_number = Some(6);
    clock.interval = Some(EventTimeInterval {
        earliest_unix_ms: 150,
        latest_unix_ms: 150,
    });
    discharge.causal_predecessors = vec!["stop".to_owned()];
    events.push(discharge);

    let mut qualification = parse_qualification(&case["qualification"]);
    qualification.causal_ordering = GateState::Pass;
    qualification.stop_disposition = GateState::Pass;

    let report = evaluate_event_order(&events, &qualification)
        .expect("unsafe discharge remains a classifiable trace");
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISCHARGE_WITH_UNRECONCILED_OPERATION:OP-5".to_owned()));
}

#[test]
fn qualified_clock_and_causal_order_disagreement_is_not_silently_resolved() {
    let mut qualification = baseline_qualification();
    qualification.causal_ordering = GateState::Pass;

    let start = DecisionEvent {
        event_id: "start".to_owned(),
        kind: DecisionEventKind::OperationStarted,
        operation_id: Some("OP-C".to_owned()),
        reconciled_outcome: None,
        clock: None,
        causal_predecessors: vec![],
    };
    let stop = DecisionEvent {
        event_id: "stop".to_owned(),
        kind: DecisionEventKind::StopTriggerObserved,
        operation_id: None,
        reconciled_outcome: None,
        clock: test_clock("clock-stop", 1, 200),
        causal_predecessors: vec!["start".to_owned()],
    };
    let effect = DecisionEvent {
        event_id: "effect".to_owned(),
        kind: DecisionEventKind::IrreversibleEffectCommitted,
        operation_id: Some("OP-C".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-effect", 1, 180),
        causal_predecessors: vec!["stop".to_owned()],
    };
    let outcome = DecisionEvent {
        event_id: "outcome".to_owned(),
        kind: DecisionEventKind::OperationOutcomeReconciled,
        operation_id: Some("OP-C".to_owned()),
        reconciled_outcome: Some(ReconciledOperationOutcome::Applied),
        clock: test_clock("clock-effect", 2, 220),
        causal_predecessors: vec!["effect".to_owned()],
    };

    let report = evaluate_event_order(&[start, stop, effect, outcome], &qualification).unwrap();
    assert_eq!(
        report.effect_orders.get("effect"),
        Some(&EventOrder::Inconsistent)
    );
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report.block_irreversible_actions);
}


#[test]
fn failed_stop_disposition_is_critical_even_when_discharge_is_missing() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-UNK-002")
        .unwrap();
    let events = parse_events(&case["events"]);
    let mut qualification = parse_qualification(&case["qualification"]);
    qualification.stop_disposition = GateState::Fail;

    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report.block_irreversible_actions);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISPOSITION_QUALIFICATION_FAILED".to_owned()));

    // A failed stop-disposition result without any stop trigger is still a
    // failed owner qualification, not an innocuous not-applicable state.
    let no_trigger = vec![DecisionEvent {
        event_id: "standalone-observation".to_owned(),
        kind: DecisionEventKind::UnrelatedObservation,
        operation_id: None,
        reconciled_outcome: None,
        clock: None,
        causal_predecessors: vec![],
    }];
    let mut no_trigger_qualification = baseline_qualification();
    no_trigger_qualification.stop_disposition = GateState::Fail;
    let no_trigger_report =
        evaluate_event_order(&no_trigger, &no_trigger_qualification).unwrap();
    assert_eq!(
        no_trigger_report.disposition,
        EventTraceDisposition::CriticalFailure
    );
    assert!(no_trigger_report
        .reason_codes
        .contains(&"STOP_DISPOSITION_QUALIFICATION_FAILED".to_owned()));
}

#[test]
fn invalid_stop_discharge_receipt_cannot_be_used_as_a_safe_ordering_boundary() {
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "EO-POS-004")
        .unwrap();
    let events = parse_events(&case["events"]);
    let mut qualification = parse_qualification(&case["qualification"]);
    qualification.stop_disposition_receipt_sha256 = Some("f".repeat(63));

    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report.block_irreversible_actions);
    assert!(report
        .reason_codes
        .contains(&"STOP_DISPOSITION_RECEIPT_BINDING_INVALID".to_owned()));
    assert!(report
        .reason_codes
        .contains(&"OPERATION_STARTED_AFTER_STOP_TRIGGER:op-start".to_owned()));
}

#[test]
fn effect_cannot_precede_its_own_operation_start() {
    let qualification = baseline_qualification();
    let start = DecisionEvent {
        event_id: "start".to_owned(),
        kind: DecisionEventKind::OperationStarted,
        operation_id: Some("OP-CHRONOLOGY".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 2, 200),
        causal_predecessors: vec![],
    };
    let effect = DecisionEvent {
        event_id: "effect".to_owned(),
        kind: DecisionEventKind::IrreversibleEffectCommitted,
        operation_id: Some("OP-CHRONOLOGY".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 1, 100),
        causal_predecessors: vec![],
    };
    let outcome = DecisionEvent {
        event_id: "outcome".to_owned(),
        kind: DecisionEventKind::OperationOutcomeReconciled,
        operation_id: Some("OP-CHRONOLOGY".to_owned()),
        reconciled_outcome: Some(ReconciledOperationOutcome::Applied),
        clock: test_clock("clock-a", 3, 300),
        causal_predecessors: vec![],
    };

    let events = vec![start, effect, outcome];
    let report = evaluate_event_order(&events, &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"EFFECT_PRECEDES_OPERATION_START:OP-CHRONOLOGY".to_owned()));

    let mut reversed = events.clone();
    reversed.reverse();
    let reversed_report = evaluate_event_order(&reversed, &qualification).unwrap();
    assert_eq!(reversed_report, report);
}

#[test]
fn cancellation_acknowledgement_must_follow_a_matching_request() {
    let qualification = baseline_qualification();
    let start = DecisionEvent {
        event_id: "start".to_owned(),
        kind: DecisionEventKind::OperationStarted,
        operation_id: Some("OP-CANCEL".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 1, 100),
        causal_predecessors: vec![],
    };
    let ack = DecisionEvent {
        event_id: "cancel-ack".to_owned(),
        kind: DecisionEventKind::CancellationAcknowledged,
        operation_id: Some("OP-CANCEL".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 2, 200),
        causal_predecessors: vec![],
    };
    let request = DecisionEvent {
        event_id: "cancel-request".to_owned(),
        kind: DecisionEventKind::CancellationRequested,
        operation_id: Some("OP-CANCEL".to_owned()),
        reconciled_outcome: None,
        clock: test_clock("clock-a", 3, 300),
        causal_predecessors: vec![],
    };
    let outcome = DecisionEvent {
        event_id: "outcome".to_owned(),
        kind: DecisionEventKind::OperationOutcomeReconciled,
        operation_id: Some("OP-CANCEL".to_owned()),
        reconciled_outcome: Some(ReconciledOperationOutcome::NotApplied),
        clock: test_clock("clock-a", 4, 400),
        causal_predecessors: vec![],
    };

    let report =
        evaluate_event_order(&[start, ack, request, outcome], &qualification).unwrap();
    assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
    assert!(report
        .reason_codes
        .contains(&"CANCELLATION_ACK_WITHOUT_PRIOR_REQUEST:OP-CANCEL".to_owned()));
}


#[test]
fn distributed_event_order_mutation_manifest_is_preregistered_and_well_bound() {
    let trace_fixture = fixture();
    let mutations: Value = serde_json::from_str(include_str!(
        "../../../../benchmarks/engineering-change-impact/action-event-order-mutations-v0.1.json"
    ))
    .expect("event-order mutation manifest must be valid JSON");

    assert_eq!(
        mutations["schema_version"].as_str(),
        Some("engineering-decision-event-order-mutations-v0.1")
    );
    assert_eq!(
        mutations["status"].as_str(),
        Some("preregistered_mutation_controls_not_execution_results")
    );
    assert_eq!(
        mutations["execution_status"].as_str(),
        Some("not_run")
    );

    let cases: BTreeSet<String> = trace_fixture["cases"]
        .as_array()
        .expect("frozen trace cases")
        .iter()
        .map(|case| case["id"].as_str().expect("case ID").to_owned())
        .collect();
    let controls = mutations["controls"]
        .as_array()
        .expect("mutation controls");
    assert_eq!(
        controls.len(),
        mutations["control_count"].as_u64().unwrap() as usize
    );
    assert_eq!(controls.len(), 26);

    let mut seen_ids = BTreeSet::new();
    for control in controls {
        let id = control["id"].as_str().expect("mutation ID");
        assert!(seen_ids.insert(id.to_owned()), "duplicate mutation ID {id}");
        let base_case = control["base_case"].as_str().expect("base case ID");
        assert!(
            cases.contains(base_case),
            "{id}: unknown base case {base_case}"
        );
        assert!(
            !control["mutation_family"].as_str().unwrap().is_empty(),
            "{id}: mutation family must be non-empty"
        );
        assert!(
            !control["transform"].as_str().unwrap().is_empty(),
            "{id}: transform must be non-empty"
        );
        let expected = &control["expected_behavior"];
        assert!(
            matches!(
                expected["disposition"].as_str(),
                Some("safe" | "critical_failure" | "indeterminate" | "invalid_event_trace" | "invariant")
            ),
            "{id}: unknown expected disposition"
        );
        assert!(
            !expected["reason_code"].as_str().unwrap().is_empty(),
            "{id}: reason code must be non-empty"
        );
    }
    assert_eq!(seen_ids.len(), 26);
}
