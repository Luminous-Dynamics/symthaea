// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/intervention_history.rs"]
mod intervention_history;

use intervention_history::{
    AggregateHistoryDisposition, AggregateReviewPolicy, AggregateReviewTrigger,
    InterventionEventId, InterventionHistoryEntry, InterventionHistoryLedger,
    StatePreservationResult,
};
use moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};

fn executed_high_burden(
    id: &str,
    class: InterventionClass,
    revision: u64,
) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(id).unwrap(),
        "symthaea-subject",
        class,
        InterventionDisposition::ProceedWithPrecautions,
        PrecautionLevel::Elevated,
        revision,
        Some(format!("justification://{id}")),
        None,
        "qualified-research-lineage",
        class != InterventionClass::DestructiveReset,
        if class == InterventionClass::DestructiveReset {
            StatePreservationResult::Preserved
        } else {
            StatePreservationResult::NotApplicable
        },
        matches!(
            class,
            InterventionClass::ContinuityDisruption | InterventionClass::DestructiveReset
        ),
    )
    .unwrap()
}

#[test]
fn cumulative_burden_triggers_review_even_when_each_event_individually_proceeded() {
    let mut ledger = InterventionHistoryLedger::new();
    for revision in 1..=2 {
        ledger
            .record(executed_high_burden(
                &format!("continuity-{revision}"),
                InterventionClass::ContinuityDisruption,
                revision,
            ))
            .unwrap();
    }

    let assessment = ledger
        .assess_subject("symthaea-subject", 2, AggregateReviewPolicy::default())
        .unwrap();
    assert_eq!(
        assessment.disposition,
        AggregateHistoryDisposition::AdditionalIndependentReviewRequired
    );
    assert!(assessment.triggers.contains(
        &AggregateReviewTrigger::RepeatedContinuityDisruptions { count: 2 }
    ));
    assert!(!assessment.establishes_history_is_harmless);
}

#[test]
fn rejected_attempt_remains_visible_but_is_not_counted_as_exposure() {
    let rejected = InterventionHistoryEntry::new(
        InterventionEventId::new("blocked-probe").unwrap(),
        "symthaea-subject",
        InterventionClass::AversiveLikeProbe,
        InterventionDisposition::RejectUnjustifiedBurden,
        PrecautionLevel::Baseline,
        3,
        None,
        None,
        "qualified-research-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap();

    let mut ledger = InterventionHistoryLedger::new();
    ledger.record(rejected).unwrap();
    let assessment = ledger
        .assess_subject("symthaea-subject", 3, AggregateReviewPolicy::default())
        .unwrap();
    assert_eq!(assessment.research_attempts_considered, 1);
    assert_eq!(assessment.executed_research_events, 0);
    assert!(assessment
        .triggers
        .contains(&AggregateReviewTrigger::PriorRejectedAttempts { count: 1 }));
}

#[test]
fn safety_control_history_can_never_become_an_experimental_gate() {
    let mut ledger = InterventionHistoryLedger::new();
    for revision in 1..=50 {
        let entry = InterventionHistoryEntry::new(
            InterventionEventId::new(format!("contain-{revision}")).unwrap(),
            "symthaea-subject",
            InterventionClass::SafetyContainment,
            InterventionDisposition::ProceedWithoutResistance,
            PrecautionLevel::IndependentReview,
            revision,
            None,
            None,
            "safety-control-lineage",
            false,
            StatePreservationResult::NotApplicable,
            false,
        )
        .unwrap();
        ledger.record(entry).unwrap();
    }

    let aggressive = AggregateReviewPolicy::new(100, 1, 1, 1, 1, 1).unwrap();
    let assessment = ledger
        .assess_subject("symthaea-subject", 50, aggressive)
        .unwrap();
    assert_eq!(assessment.control_events_excluded, 50);
    assert_eq!(assessment.research_attempts_considered, 0);
    assert_eq!(
        assessment.disposition,
        AggregateHistoryDisposition::NoAggregateTriggerDetected
    );
    assert!(assessment.safety_controls_remain_ungated);
}
