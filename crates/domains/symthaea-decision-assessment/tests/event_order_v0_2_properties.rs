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
        mapping_valid_from_unix_ns: valid_from_ns,
        mapping_valid_through_unix_ns: valid_through_ns,
        max_interval_width_ns: 100_000,
        interval: EventTimeIntervalNs {
            earliest_ns,
            latest_ns,
        },
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
}
