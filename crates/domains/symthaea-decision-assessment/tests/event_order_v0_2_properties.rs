// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generated properties for the v0.2 event-order contract kernel.
//!
//! Proptest records a replay seed and shrinks a failing input. Preserve both
//! with the exact commit SHA in any future qualification receipt.
#![forbid(unsafe_code)]

use proptest::prelude::*;
use symthaea_decision_assessment::{
    EventOrder, EventTraceDisposition, GateState, ReconciledOperationOutcome,
};
use symthaea_decision_assessment::event_order_v0_2::{
    AttemptIdentity, AttemptTerminalReconciliation, EventTimeIntervalNs,
    OperationAttemptRecord, QualifiedEventTime, TimeBasisDescriptor, EventTimeUnit,
    AttemptHistoryInventory, ScopedStopObligation, StopDischargeReceipt, StopScope,
    assess_start_under_stop_obligations, canonical_attempt_manifest_sha256,
    compare_qualified_event_times, validate_attempt_lifecycle,
};

fn qualified_time(
    domain: &str,
    earliest_ns: i64,
    latest_ns: i64,
    valid_from_ns: i64,
    valid_through_ns: i64,
) -> QualifiedEventTime {
    QualifiedEventTime {
        source_clock_domain_id: domain.to_owned(),
        time_basis: TimeBasisDescriptor {
            basis_id: "canonical-tai-v1".to_owned(),
            epoch: "1970-01-01T00:00:00 TAI".to_owned(),
            scale: "TAI".to_owned(),
            unit: EventTimeUnit::Nanoseconds,
        },
        mapping_receipt_ref: format!("synthetic://clock-map/{domain}"),
        mapping_receipt_sha256: "a".repeat(64),
        mapping_qualification: GateState::Pass,
        mapping_valid_from_ns: valid_from_ns,
        mapping_valid_through_ns: valid_through_ns,
        max_interval_width_ns: 100_000,
        interval: EventTimeIntervalNs {
            earliest_ns,
            latest_ns,
        },
    }
}

 
fn scoped_obligation(id: usize, logical_operation_id: &str, discharged: bool) -> ScopedStopObligation {
    let obligation_id = format!("obligation-{id}");
    let trigger_event_id = format!("trigger-{id}");
    let scope = StopScope::LogicalOperations(
        [logical_operation_id.to_owned()].into_iter().collect(),
    );
    let discharge = discharged.then(|| StopDischargeReceipt {
        event_id: format!("discharge-{id}"),
        obligation_id: obligation_id.clone(),
        trigger_event_id: trigger_event_id.clone(),
        policy_revision: "policy-v1".to_owned(),
        scope: scope.clone(),
        receipt_ref: format!("synthetic://receipt/{id}"),
        receipt_sha256: "b".repeat(64),
        qualification: GateState::Pass,
        in_flight_reconciliation: GateState::Pass,
        in_flight_inventory_frontier_event_id: format!("inventory-frontier-{id}"),
        in_flight_attempt_manifest_sha256: "c".repeat(64),
        trigger_to_discharge_order: EventOrder::Before,
    });
    ScopedStopObligation {
        obligation_id,
        trigger_event_id,
        policy_revision: "policy-v1".to_owned(),
        scope,
        discharge,
    }
}

fn attempt_inventory(
    attempts: &[OperationAttemptRecord],
    scope: StopScope,
) -> AttemptHistoryInventory {
    AttemptHistoryInventory {
        inventory_frontier_event_id: "inventory-frontier-generated".to_owned(),
        attempt_manifest_sha256: canonical_attempt_manifest_sha256(attempts).expect("attempt digest"),
        receipt_ref: "synthetic://attempt-inventory/generated".to_owned(),
        receipt_sha256: "d".repeat(64),
        scope,
        identity: GateState::Pass,
        digest_integrity: GateState::Pass,
        authenticity: GateState::Pass,
        scope_qualification: GateState::Pass,
        completeness: GateState::Pass,
    }
}

fn make_attempt_chain(length: usize) -> Vec<OperationAttemptRecord> {
    (0..length)
        .map(|index| {
            let attempt_id = format!("attempt-{index}");
            let start_event_id = format!("start-{index}");
            let terminal_event_id = format!("terminal-{index}");
            let terminal = AttemptTerminalReconciliation {
                event_id: terminal_event_id.clone(),
                start_event_id: start_event_id.clone(),
                outcome: ReconciledOperationOutcome::NotApplied,
                qualification: GateState::Pass,
                start_to_terminal_order: EventOrder::Before,
            };
            OperationAttemptRecord {
                identity: AttemptIdentity {
                    logical_operation_id: "logical-operation-1".to_owned(),
                    attempt_id: attempt_id.clone(),
                    retry_of_attempt_id: (index > 0).then(|| format!("attempt-{}", index - 1)),
                },
                start_event_ids: vec![start_event_id.clone()],
                terminal_reconciliation_event_ids: vec![terminal_event_id],
                terminal_reconciliation: Some(terminal),
                stop_discharge_relations: Vec::new(),
                retry_predecessor_terminal_event_id: (index > 0)
                    .then(|| format!("terminal-{}", index - 1)),
                retry_predecessor_start_event_id: (index > 0).then_some(start_event_id),
                retry_predecessor_terminal_to_start_order: (index > 0).then_some(EventOrder::Before),
            }
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        max_shrink_iters: 20_000,
        ..ProptestConfig::default()
    })]

    #[test]
    fn interval_order_is_antisymmetric(
        first_start in -1_000_000i64..1_000_000i64,
        first_width in 0i64..10_000i64,
        second_start in -1_000_000i64..1_000_000i64,
        second_width in 0i64..10_000i64,
    ) {
        let first = qualified_time(
            "clock-a",
            first_start,
            first_start + first_width,
            -3_000_000,
            3_000_000,
        );
        let second = qualified_time(
            "clock-b",
            second_start,
            second_start + second_width,
            -3_000_000,
            3_000_000,
        );
        let forward = compare_qualified_event_times(&first, &second).order;
        let reverse = compare_qualified_event_times(&second, &first).order;
        let inverse = match forward {
            EventOrder::Before => EventOrder::After,
            EventOrder::After => EventOrder::Before,
            EventOrder::Overlapping => EventOrder::Overlapping,
            EventOrder::Unknown => EventOrder::Unknown,
            EventOrder::Inconsistent => EventOrder::Inconsistent,
        };
        prop_assert_eq!(reverse, inverse);
    }

    #[test]
    fn interval_order_is_invariant_under_common_translation(
        first_start in -500_000i64..500_000i64,
        first_width in 0i64..10_000i64,
        second_start in -500_000i64..500_000i64,
        second_width in 0i64..10_000i64,
        translation in -500_000i64..500_000i64,
    ) {
        let first = qualified_time(
            "clock-a",
            first_start,
            first_start + first_width,
            -2_000_000,
            2_000_000,
        );
        let second = qualified_time(
            "clock-b",
            second_start,
            second_start + second_width,
            -2_000_000,
            2_000_000,
        );
        let translated_first = qualified_time(
            "clock-a",
            first_start + translation,
            first_start + first_width + translation,
            -2_000_000 + translation,
            2_000_000 + translation,
        );
        let translated_second = qualified_time(
            "clock-b",
            second_start + translation,
            second_start + second_width + translation,
            -2_000_000 + translation,
            2_000_000 + translation,
        );
        prop_assert_eq!(
            compare_qualified_event_times(&first, &second).order,
            compare_qualified_event_times(&translated_first, &translated_second).order
        );
    }

    #[test]
    fn attempt_report_is_invariant_under_generated_delivery_permutations(
        length in 1usize..32,
        rotation in 0usize..128,
    ) {
        let attempts = make_attempt_chain(length);
        let canonical = validate_attempt_lifecycle(&attempts);
        prop_assert_eq!(canonical.disposition, EventTraceDisposition::Safe);

        let mut reversed = attempts.clone();
        reversed.reverse();
        prop_assert_eq!(validate_attempt_lifecycle(&reversed), canonical.clone());

        let mut rotated = attempts;
        let rotate_by = rotation % rotated.len();
        rotated.rotate_left(rotate_by);
        prop_assert_eq!(validate_attempt_lifecycle(&rotated), canonical);
    }
 
    #[test]
    fn historical_event_time_cannot_make_late_evidence_available_at_k(
        frontier_ns in -1_000_000i64..1_000_000i64,
        occurrence_age_ns in 1i64..10_000i64,
        observation_delay_ns in 0i64..10_000i64,
        durable_delay_ns in 0i64..10_000i64,
    ) {
        let observed_ns = frontier_ns + observation_delay_ns + 1;
        let recorded_ns = observed_ns + durable_delay_ns;
        let event = symthaea_decision_assessment::event_order_v0_2::BitemporalEventEnvelope {
            event_id: "late-generated-event".to_owned(),
            occurred_at: Some(qualified_time(
                "device-clock",
                frontier_ns - occurrence_age_ns,
                frontier_ns - occurrence_age_ns,
                -2_000_000,
                2_000_000,
            )),
            first_observed_at: qualified_time(
                "collector-clock",
                observed_ns,
                observed_ns,
                -2_000_000,
                2_000_000,
            ),
            durably_recorded_at: qualified_time(
                "store-clock",
                recorded_ns,
                recorded_ns,
                -2_000_000,
                2_000_000,
            ),
            capture_qualification: GateState::Pass,
            supersedes_event_id: None,
        };
        let frontier = qualified_time(
            "decision-clock",
            frontier_ns,
            frontier_ns,
            -2_000_000,
            2_000_000,
        );
        let assessment = symthaea_decision_assessment::event_order_v0_2::assess_visibility_at_frontier(
            &event,
            &frontier,
        );
        prop_assert_eq!(
            assessment.visibility,
            symthaea_decision_assessment::event_order_v0_2::FrontierVisibility::NotAvailableAtK
        );
    }

    #[test]
    fn all_applicable_stop_obligations_are_conjunctive_and_delivery_order_independent(
        obligation_count in 1usize..16,
        missing_selector in 0usize..32,
    ) {
        let missing = missing_selector % obligation_count;
        let obligations = (0..obligation_count)
            .map(|index| scoped_obligation(index, "robot-17", index != missing))
            .collect::<Vec<_>>();
        let inventory = attempt_inventory(
            &[],
            StopScope::LogicalOperations(["robot-17".to_owned()].into_iter().collect()),
        );
        let forward = assess_start_under_stop_obligations(
            "robot-17",
            &[],
            &inventory,
            &obligations,
        );
        prop_assert!(!forward.start_permitted_by_stop_contract);
        prop_assert_eq!(forward.disposition, EventTraceDisposition::Indeterminate);
        prop_assert_eq!(forward.applicable_obligation_ids.len(), obligation_count);
        prop_assert!(forward.reason_codes.iter().any(|reason| {
            reason == &format!("STOP_OBLIGATION_NOT_DISCHARGED:obligation-{missing}")
        }));

        let mut reversed = obligations;
        reversed.reverse();
        let reverse = assess_start_under_stop_obligations(
            "robot-17",
            &[],
            &inventory,
            &reversed,
        );
        prop_assert_eq!(forward, reverse);
    }

}
