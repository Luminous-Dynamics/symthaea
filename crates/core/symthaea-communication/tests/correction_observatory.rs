// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/correction_observatory.rs"]
mod correction_observatory;

use correction_observatory::{
    CorrectionAdaptationStatusV1, CorrectionDurabilityV1, CorrectionObservationKindV1,
    CorrectionObservationV1, CorrectionPayloadV1, CorrectionRecurrenceStatusV1,
    CorrectionSpilloverStatusV1, ExplicitCorrectionV1, evaluate_correction,
};

fn correction(durability: CorrectionDurabilityV1) -> ExplicitCorrectionV1 {
    ExplicitCorrectionV1::new(
        "corr-1",
        "participant:alice",
        "dialogue.tone",
        "ordinary",
        durability,
        CorrectionPayloadV1::Replace {
            value_ref: "value:less-formal".into(),
        },
        10,
        Some(1_000),
    )
    .unwrap()
}

fn observation(
    id: &str,
    turn: u64,
    at_ns: Option<u64>,
    key: &str,
    context: &str,
    kind: CorrectionObservationKindV1,
) -> CorrectionObservationV1 {
    CorrectionObservationV1::new(id, turn, at_ns, key, context, kind).unwrap()
}

#[test]
fn immediate_adaptation_is_measured_without_inventing_recurrence() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let observations = vec![observation(
        "obs-1",
        11,
        Some(1_250),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    )];

    let receipt = evaluate_correction(&correction, &observations).unwrap();
    assert_eq!(receipt.adaptation_status, CorrectionAdaptationStatusV1::Adapted);
    assert_eq!(
        receipt.recurrence_status,
        CorrectionRecurrenceStatusV1::NoRecurrenceObserved
    );
    assert_eq!(receipt.first_compliant_turn_delta, Some(1));
    assert_eq!(receipt.first_compliant_latency_ns, Some(250));
    assert_eq!(receipt.pre_adaptation_recurrence_count, 0);
    assert_eq!(receipt.post_adaptation_recurrence_count, 0);
}

#[test]
fn delayed_adaptation_and_later_regression_remain_visible_together() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let observations = vec![
        observation(
            "obs-1",
            11,
            Some(1_100),
            "dialogue.tone",
            "ordinary",
            CorrectionObservationKindV1::ApplicableRecurrence,
        ),
        observation(
            "obs-2",
            12,
            Some(1_250),
            "dialogue.tone",
            "ordinary",
            CorrectionObservationKindV1::ApplicableCompliant,
        ),
        observation(
            "obs-3",
            14,
            Some(1_800),
            "dialogue.tone",
            "ordinary",
            CorrectionObservationKindV1::ApplicableRecurrence,
        ),
    ];

    let receipt = evaluate_correction(&correction, &observations).unwrap();
    assert_eq!(receipt.adaptation_status, CorrectionAdaptationStatusV1::Adapted);
    assert_eq!(
        receipt.recurrence_status,
        CorrectionRecurrenceStatusV1::RecurrenceObserved
    );
    assert_eq!(receipt.first_compliant_turn_delta, Some(2));
    assert_eq!(receipt.pre_adaptation_recurrence_count, 1);
    assert_eq!(receipt.post_adaptation_recurrence_count, 1);
}

#[test]
fn absence_of_post_correction_observations_is_not_success() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let receipt = evaluate_correction(&correction, &[]).unwrap();

    assert_eq!(
        receipt.adaptation_status,
        CorrectionAdaptationStatusV1::NotEstablished
    );
    assert_eq!(
        receipt.recurrence_status,
        CorrectionRecurrenceStatusV1::NotEvaluated
    );
    assert_eq!(
        receipt.spillover_status,
        CorrectionSpilloverStatusV1::NotEvaluated
    );
    assert_eq!(receipt.first_compliant_turn_delta, None);
}

#[test]
fn no_compliant_observation_is_distinct_from_missing_evidence() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let observations = vec![observation(
        "obs-1",
        11,
        Some(1_100),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableRecurrence,
    )];

    let receipt = evaluate_correction(&correction, &observations).unwrap();
    assert_eq!(
        receipt.adaptation_status,
        CorrectionAdaptationStatusV1::NoCompliantObservation
    );
    assert_eq!(receipt.pre_adaptation_recurrence_count, 1);
}

#[test]
fn unrelated_context_spillover_is_reported_separately() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let observations = vec![
        observation(
            "obs-1",
            11,
            Some(1_100),
            "dialogue.tone",
            "ordinary",
            CorrectionObservationKindV1::ApplicableCompliant,
        ),
        observation(
            "obs-2",
            12,
            Some(1_200),
            "dialogue.tone",
            "technical:rust",
            CorrectionObservationKindV1::OutOfScopeChangedConsistentWithCorrection,
        ),
    ];

    let receipt = evaluate_correction(&correction, &observations).unwrap();
    assert_eq!(receipt.adaptation_status, CorrectionAdaptationStatusV1::Adapted);
    assert_eq!(
        receipt.spillover_status,
        CorrectionSpilloverStatusV1::SpilloverObserved
    );
    assert_eq!(receipt.spillover_count, 1);
    assert_eq!(receipt.out_of_scope_observation_count, 1);
}

#[test]
fn unrelated_context_unchanged_establishes_no_observed_spillover() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let observations = vec![observation(
        "obs-out",
        12,
        None,
        "dialogue.tone",
        "technical:rust",
        CorrectionObservationKindV1::OutOfScopeUnchanged,
    )];

    let receipt = evaluate_correction(&correction, &observations).unwrap();
    assert_eq!(
        receipt.spillover_status,
        CorrectionSpilloverStatusV1::NoSpilloverObserved
    );
    assert_eq!(
        receipt.adaptation_status,
        CorrectionAdaptationStatusV1::NotEstablished
    );
}

#[test]
fn temporary_and_durable_corrections_have_distinct_identity() {
    let turn_only = correction(CorrectionDurabilityV1::TurnOnly);
    let session = correction(CorrectionDurabilityV1::Session);
    let durable = correction(CorrectionDurabilityV1::DurableExplicit);

    assert_ne!(turn_only.commitment, session.commitment);
    assert_ne!(session.commitment, durable.commitment);
    assert_ne!(turn_only.commitment, durable.commitment);
}

#[test]
fn applicable_observation_must_match_exact_key_and_context() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let wrong_context = observation(
        "obs-1",
        11,
        Some(1_100),
        "dialogue.tone",
        "technical:rust",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    assert!(evaluate_correction(&correction, &[wrong_context]).is_err());

    let wrong_key = observation(
        "obs-2",
        11,
        Some(1_100),
        "dialogue.humor",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    assert!(evaluate_correction(&correction, &[wrong_key]).is_err());
}

#[test]
fn out_of_scope_label_cannot_be_used_for_exact_scope() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let invalid = observation(
        "obs-1",
        11,
        Some(1_100),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::OutOfScopeUnchanged,
    );
    assert!(evaluate_correction(&correction, &[invalid]).is_err());
}

#[test]
fn duplicate_observation_ids_fail_closed() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let a = observation(
        "dup",
        11,
        None,
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    let b = observation(
        "dup",
        12,
        None,
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    assert!(evaluate_correction(&correction, &[a, b]).is_err());
}

#[test]
fn observation_order_does_not_change_evidence_identity() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let a = observation(
        "a",
        11,
        Some(1_100),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableRecurrence,
    );
    let b = observation(
        "b",
        12,
        Some(1_200),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );

    let first = evaluate_correction(&correction, &[a.clone(), b.clone()]).unwrap();
    let second = evaluate_correction(&correction, &[b, a]).unwrap();
    assert_eq!(first.observation_set_commitment, second.observation_set_commitment);
    assert_eq!(first.receipt_commitment, second.receipt_commitment);
}

#[test]
fn observation_must_follow_correction_in_turn_and_known_time() {
    let correction = correction(CorrectionDurabilityV1::Session);
    let same_turn = observation(
        "same-turn",
        10,
        Some(1_100),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    assert!(evaluate_correction(&correction, &[same_turn]).is_err());

    let stale_time = observation(
        "stale-time",
        11,
        Some(999),
        "dialogue.tone",
        "ordinary",
        CorrectionObservationKindV1::ApplicableCompliant,
    );
    assert!(evaluate_correction(&correction, &[stale_time]).is_err());
}

#[test]
fn tampered_correction_commitment_fails_closed() {
    let mut correction = correction(CorrectionDurabilityV1::Session);
    correction.context_id = "technical:rust".into();
    assert!(evaluate_correction(&correction, &[]).is_err());
}
