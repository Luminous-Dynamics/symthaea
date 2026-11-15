use std::collections::BTreeMap;

use proptest::prelude::*;
use symthaea_decision_assessment::event_order_v0_2::{
    EventTimeIntervalNs, EventTimeUnit, QualifiedEventTime, TimeBasisDescriptor,
};
use symthaea_decision_assessment::event_order_v0_3::{
    canonical_schedule_bytes, canonical_schedule_sha256, generate_cancellation_commit_race_schedule,
    generate_capture_clock_fault_schedule, generate_clock_mapping_fault_schedule, generate_clocked_observation_persistence_schedule,
    generate_delivery_schedule, project_bitemporal_event, replay_virtual_schedule, validate_schedule,
    ActionClockTimeReference, ActionEvidence, CancellationCommitOrder, CaptureClockFault, ClockMappingFault,
    ScheduledAction, SchedulerDisposition, ScheduleAction, VirtualSchedule,
};

fn owner_qualified_resolutions(
    schedule: &symthaea_decision_assessment::event_order_v0_3::VirtualSchedule,
) -> BTreeMap<(String, u64), QualifiedEventTime> {
    let mut result = BTreeMap::new();
    for item in &schedule.actions {
        let (mapping_action_id, local_ns, normalized_ns) = match &item.action {
            ScheduleAction::EventProduced { clock_mapping_action_id, source_timestamp_ns, .. } => {
                (clock_mapping_action_id, *source_timestamp_ns, 500_i64)
            }
            ScheduleAction::EventObserved { clock_mapping_action_id, observed_at_ns, .. } => {
                (clock_mapping_action_id, *observed_at_ns, 1_000_i64)
            }
            ScheduleAction::EventPersisted { clock_mapping_action_id, persisted_at_ns, .. } => {
                (clock_mapping_action_id, *persisted_at_ns, 1_200_i64)
            }
            _ => continue,
        };
        let mapping_action = schedule.actions.iter()
            .find(|candidate| candidate.action_id == *mapping_action_id)
            .expect("exact source clock mapping action exists");
        let (clock_domain, receipt_ref, receipt_sha256, valid_from, valid_through) = match &mapping_action.action {
            ScheduleAction::ClockAdvance {
                clock_domain, mapping_receipt_ref, mapping_receipt_sha256,
                mapping_valid_from_ns, mapping_valid_through_ns, ..
            } => (
                clock_domain.clone(),
                mapping_receipt_ref.clone(),
                mapping_receipt_sha256.clone(),
                i64::try_from(*mapping_valid_from_ns).expect("mapping validity start fits i64"),
                i64::try_from(*mapping_valid_through_ns).expect("mapping validity end fits i64"),
            ),
            _ => panic!("clock mapping action has wrong type"),
        };
        result.insert(
            (mapping_action_id.clone(), local_ns),
            QualifiedEventTime {
                source_clock_domain_id: clock_domain,
                time_basis: TimeBasisDescriptor {
                    basis_id: "owner-normalized-tai-v1".to_owned(),
                    epoch: "tai-epoch-0".to_owned(),
                    scale: "TAI".to_owned(),
                    unit: EventTimeUnit::Nanoseconds,
                },
                mapping_receipt_ref: receipt_ref,
                mapping_receipt_sha256: receipt_sha256,
                mapping_qualification: symthaea_decision_assessment::GateState::Pass,
                mapping_valid_from_ns: valid_from,
                mapping_valid_through_ns: valid_through,
                max_interval_width_ns: 0,
                interval: EventTimeIntervalNs {
                    earliest_ns: normalized_ns,
                    latest_ns: normalized_ns,
                },
            },
        );
    }
    result
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn canonical_schedule_is_invariant_under_action_permutation(
        seed in any::<u64>(),
        event_count in 1usize..96,
    ) {
        let schedule = generate_delivery_schedule(seed, event_count).expect("generated schedule");
        let mut permuted = schedule.clone();
        permuted.actions.reverse();
        prop_assert_eq!(
            canonical_schedule_bytes(&schedule).expect("canonical bytes"),
            canonical_schedule_bytes(&permuted).expect("canonical bytes"),
        );
        prop_assert_eq!(
            canonical_schedule_sha256(&schedule).expect("schedule digest"),
            canonical_schedule_sha256(&permuted).expect("schedule digest"),
        );
    }

    #[test]
    fn cancellation_commit_race_order_is_explicit(seed in any::<u64>()) {
        let commit_first = generate_cancellation_commit_race_schedule(
            seed,
            CancellationCommitOrder::CommitBeforeAcknowledgement,
        );
        let acknowledgement_first = generate_cancellation_commit_race_schedule(
            seed,
            CancellationCommitOrder::CommitAfterAcknowledgement,
        );
        let commit_first_report = replay_virtual_schedule(&commit_first).expect("commit-first replay");
        let acknowledgement_first_report =
            replay_virtual_schedule(&acknowledgement_first).expect("acknowledgement-first replay");

        prop_assert!(commit_first_report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::AcceptedCancellationAfterCommit
        }));
        prop_assert!(acknowledgement_first_report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::CommitAfterAcceptedCancellation
        }));
        prop_assert!(commit_first_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::StopDischarged { .. })
                && record.disposition == SchedulerDisposition::Applied
        }));
        prop_assert!(acknowledgement_first_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::StopDischarged { .. })
                && record.disposition == SchedulerDisposition::Applied
        }));
    }

    #[test]
    fn clock_mapping_faults_never_become_applied(seed in any::<u64>(), fault_tag in 0u8..3) {
        let (fault, expected) = match fault_tag {
            0 => (ClockMappingFault::Unqualified, SchedulerDisposition::ClockMappingUnqualified),
            1 => (ClockMappingFault::Expired, SchedulerDisposition::ClockMappingExpired),
            _ => (ClockMappingFault::StaleReference, SchedulerDisposition::StaleClockMappingReference),
        };
        let schedule = generate_clock_mapping_fault_schedule(seed, fault);
        let report = replay_virtual_schedule(&schedule).expect("fault schedule replay");
        prop_assert!(report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventProduced { .. })
                && record.disposition == expected
        }));
    }

    #[test]
    fn observer_and_store_calibration_faults_never_become_applied(
        seed in any::<u64>(),
        fault_tag in 0u8..8,
    ) {
        let (fault, expected, target) = match fault_tag {
            0 => (CaptureClockFault::ObserverUnqualified, SchedulerDisposition::ClockMappingUnqualified, "observer"),
            1 => (CaptureClockFault::ObserverExpired, SchedulerDisposition::ClockMappingExpired, "observer"),
            2 => (CaptureClockFault::ObserverNotYetValid, SchedulerDisposition::ClockMappingNotYetValid, "observer"),
            3 => (CaptureClockFault::ObserverValidityInvalid, SchedulerDisposition::ClockMappingValidityInvalid, "observer"),
            4 => (CaptureClockFault::StoreUnqualified, SchedulerDisposition::ClockMappingUnqualified, "store"),
            5 => (CaptureClockFault::StoreExpired, SchedulerDisposition::ClockMappingExpired, "store"),
            6 => (CaptureClockFault::StoreNotYetValid, SchedulerDisposition::ClockMappingNotYetValid, "store"),
            _ => (CaptureClockFault::StoreValidityInvalid, SchedulerDisposition::ClockMappingValidityInvalid, "store"),
        };
        let schedule = generate_capture_clock_fault_schedule(seed, fault)
            .expect("capture-clock fault schedule");
        validate_schedule(&schedule).expect("fault remains structurally valid");
        let report = replay_virtual_schedule(&schedule).expect("fault replay");
        let disposition = report.records.iter().find_map(|record| match (&record.action, target) {
            (ScheduleAction::EventObserved { .. }, "observer") => Some(record.disposition),
            (ScheduleAction::EventPersisted { .. }, "store") => Some(record.disposition),
            _ => None,
        }).expect("capture action");
        prop_assert_eq!(disposition, expected);
        prop_assert_ne!(disposition, SchedulerDisposition::Applied);
    }

    #[test]
    fn action_owner_time_references_bind_exact_payloads_and_active_clock_maps(seed in any::<u64>()) {
        let mut schedule = generate_delivery_schedule(seed, 4).expect("generated schedule");
        let event_index = schedule.actions.iter().position(|item| {
            matches!(&item.action, ScheduleAction::EventProduced { event_id, .. } if event_id == "event-000002")
        }).expect("third event");
        let (mapping_id, timestamp_ns) = match &schedule.actions[event_index].action {
            ScheduleAction::EventProduced { clock_mapping_action_id, source_timestamp_ns, .. } => {
                (clock_mapping_action_id.clone(), *source_timestamp_ns)
            }
            _ => unreachable!("selected event"),
        };
        schedule.actions[event_index].owner_time_ref = Some(ActionClockTimeReference {
            clock_mapping_action_id: mapping_id.clone(),
            local_timestamp_ns: timestamp_ns,
        });
        validate_schedule(&schedule).expect("exact map and timestamp validate");
        let exact_report = replay_virtual_schedule(&schedule).expect("exact map replay");
        let event_action_id = schedule.actions[event_index].action_id.clone();
        prop_assert!(exact_report.records.iter().any(|record| {
            record.action_id == event_action_id && record.disposition == SchedulerDisposition::Applied
        }));

        let (previous_mapping, previous_timestamp) = schedule.actions.iter().find_map(|item| {
            if let ScheduleAction::EventProduced { event_id, clock_mapping_action_id, source_timestamp_ns, .. } = &item.action {
                (event_id == "event-000000").then_some((clock_mapping_action_id.clone(), *source_timestamp_ns))
            } else {
                None
            }
        }).expect("prior source mapping and tick");

        // Native event clock payloads cannot be substituted independently of
        // owner_time_ref; reject at the schedule contract boundary.
        let mut mismatched_payload = schedule.clone();
        let mismatched = mismatched_payload.actions.iter_mut()
            .find(|item| item.action_id == event_action_id)
            .expect("target event remains");
        mismatched.owner_time_ref.as_mut().expect("owner time ref").local_timestamp_ns += 1;
        prop_assert!(matches!(
            validate_schedule(&mismatched_payload),
            Err(symthaea_decision_assessment::event_order_v0_3::ReplayError::ActionEvidenceMismatch("owner_time_payload_binding"))
        ));

        // For actions without a duplicated native clock field, an earlier
        // mapping in the same series is structurally known but stale at replay.
        let mut stale = schedule.clone();
        let ordinal = stale.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let at_ns = stale.actions.iter().map(|item| item.at_ns).max().unwrap_or(0) + 1;
        let target_action_id = format!("action-{ordinal:06}");
        stale.actions.push(ScheduledAction {
            action_id: target_action_id.clone(),
            at_ns,
            ordinal,
            owner_time_ref: Some(ActionClockTimeReference {
                clock_mapping_action_id: previous_mapping,
                local_timestamp_ns: previous_timestamp,
            }),
            evidence: Some(ActionEvidence::AttemptStart { retry_of_attempt_id: None }),
            action: ScheduleAction::EffectAttempted {
                operation_id: "owner-time-probe".to_owned(),
                attempt_id: "stale-map-probe".to_owned(),
            },
        });
        let stale_action = stale.actions.last().expect("probe action");
        validate_schedule(&stale).expect("known prior map is structurally valid");
        let stale_report = replay_virtual_schedule(&stale).expect("stale map replay");
        prop_assert!(stale_report.records.iter().any(|record| {
            record.action_id == target_action_id
                && record.disposition == SchedulerDisposition::StaleClockMappingReference
        }));

        let mut missing = schedule;
        let missing_ordinal = missing.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let missing_at = missing.actions.iter().map(|item| item.at_ns).max().unwrap_or(0) + 1;
        missing.actions.push(ScheduledAction {
            action_id: format!("action-{missing_ordinal:06}"),
            at_ns: missing_at,
            ordinal: missing_ordinal,
            owner_time_ref: Some(ActionClockTimeReference {
                clock_mapping_action_id: "missing-clock-map".to_owned(),
                local_timestamp_ns: timestamp_ns,
            }),
            evidence: Some(ActionEvidence::AttemptStart { retry_of_attempt_id: None }),
            action: ScheduleAction::EffectAttempted {
                operation_id: "owner-time-probe".to_owned(),
                attempt_id: "missing-map-probe".to_owned(),
            },
        });
        prop_assert!(validate_schedule(&missing).is_err());
    }

    #[test]
    fn replay_is_repeatable_for_identical_seed_and_input(
        seed in any::<u64>(),
        event_count in 1usize..96,
    ) {
        let schedule = generate_delivery_schedule(seed, event_count).expect("generated schedule");
        let first = replay_virtual_schedule(&schedule).expect("first replay");
        let second = replay_virtual_schedule(&schedule).expect("second replay");
        prop_assert_eq!(first, second);
    }

    #[test]
    fn bitemporal_projection_uses_only_separate_owner_clock_resolutions(seed in any::<u64>()) {
        let schedule = generate_clocked_observation_persistence_schedule(seed)
            .expect("generated clocked history");
        let resolutions = owner_qualified_resolutions(&schedule);
        let projection = project_bitemporal_event(
            &schedule,
            "event-000000",
            "worker-after-reconnect",
            "durable-store-0",
            &resolutions,
        ).expect("projection assessment");
        prop_assert_eq!(projection.disposition, symthaea_decision_assessment::EventTraceDisposition::Safe);
        let envelope = projection.envelope.expect("bitemporal envelope");
        prop_assert_eq!(envelope.first_observed_at.interval.earliest_ns, 1_000);
        prop_assert_eq!(envelope.durably_recorded_at.interval.earliest_ns, 1_200);
        prop_assert!(envelope.durably_recorded_at.interval.earliest_ns
            > envelope.first_observed_at.interval.earliest_ns);
    }

    #[test]
    fn missing_observer_clock_resolution_never_projects_an_envelope(seed in any::<u64>()) {
        let schedule = generate_clocked_observation_persistence_schedule(seed)
            .expect("generated clocked history");
        let mut resolutions = owner_qualified_resolutions(&schedule);
        let observation_key = schedule.actions.iter().find_map(|item| {
            if let ScheduleAction::EventObserved { clock_mapping_action_id, observed_at_ns, .. } = &item.action {
                Some((clock_mapping_action_id.clone(), *observed_at_ns))
            } else {
                None
            }
        }).expect("observer key");
        resolutions.remove(&observation_key);
        let projection = project_bitemporal_event(
            &schedule,
            "event-000000",
            "worker-after-reconnect",
            "durable-store-0",
            &resolutions,
        ).expect("projection assessment");
        prop_assert_eq!(projection.disposition, symthaea_decision_assessment::EventTraceDisposition::Indeterminate);
        prop_assert!(projection.envelope.is_none());
    }

}

#[test]
fn action_evidence_cannot_be_substituted_across_action_kinds() {
    let mut schedule = generate_delivery_schedule(0x2026_1124, 4).expect("schedule");
    let target = schedule.actions.iter_mut().find(|item| {
        matches!(item.action, ScheduleAction::EventProduced { .. })
    }).expect("event production action");
    target.evidence = Some(ActionEvidence::EffectCommit {
        receipt_ref: "synthetic://wrong-kind".to_owned(),
        receipt_sha256: "a".repeat(64),
    });
    assert!(matches!(
        validate_schedule(&schedule),
        Err(symthaea_decision_assessment::event_order_v0_3::ReplayError::ActionEvidenceMismatch("action_variant"))
    ));
}

#[test]
fn frozen_v0_3_1_observer_store_clock_fixture_replays_and_projects() {
    let fixture = include_str!("../../../../benchmarks/engineering-change-impact/event-order-v0.3.2-clocked-observation-persistence.json");
    let schedule: VirtualSchedule = serde_json::from_str(fixture).expect("frozen schedule parses");
    validate_schedule(&schedule).expect("frozen v0.3.1 schedule validates");
    let replay = replay_virtual_schedule(&schedule).expect("frozen schedule replays");
    assert_eq!(replay.event_count, 1);
    assert_eq!(replay.observed_count, 1);
    assert_eq!(replay.persisted_count, 1);
    assert!(replay.records.iter().any(|record| {
        matches!(record.action, ScheduleAction::EventObserved { .. })
            && record.disposition == SchedulerDisposition::Applied
    }));
    assert!(replay.records.iter().any(|record| {
        matches!(record.action, ScheduleAction::EventPersisted { .. })
            && record.disposition == SchedulerDisposition::Applied
    }));

    let resolutions = owner_qualified_resolutions(&schedule);
    let projected = project_bitemporal_event(
        &schedule, "event-0001", "observer-0", "ledger-0", &resolutions,
    ).expect("bitemporal projection");
    assert_eq!(
        projected.disposition,
        symthaea_decision_assessment::EventTraceDisposition::Safe
    );
    let envelope = projected.envelope.expect("qualified envelope");
    assert_eq!(envelope.occurred_at.expect("qualified occurrence").interval.earliest_ns, 500);
    assert_eq!(envelope.first_observed_at.interval.earliest_ns, 1_000);
    assert_eq!(envelope.durably_recorded_at.interval.earliest_ns, 1_200);
}
