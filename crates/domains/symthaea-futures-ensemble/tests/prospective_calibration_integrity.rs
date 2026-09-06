// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for prospective calibration-assignment integrity.
//!
//! A calibration bucket is a function of the forecast probability and a committed
//! calibration policy. It must not become a post-outcome degree of freedom merely
//! because the raw resolution schema stores an optional bucket label.

use symthaea_futures_core::{AbstentionReason, ForecastPayload, OutcomeRegion, OutcomeSpaceId};
use symthaea_futures_ledger::prospective::{
    EvaluationProtocol, ForecastAttemptDecision, ProspectiveAttemptCommitment,
    ProspectiveAttemptResolution, ProspectiveError,
};
use symthaea_futures_ledger::v2::{
    ExternalReference, ForecastCommitmentId, ForecastCoordinate, ForecastResolutionId,
    ForecastSpan, ForecastWindow, ObservationLineage, ObservedSnapshotRef,
};

#[derive(Debug, PartialEq)]
enum CalibrationIntegrityError {
    Prospective(ProspectiveError),
    NotForecastResolution,
    UnknownEvaluationProtocol(String),
    BucketMismatch {
        stored: Option<String>,
        expected: Option<String>,
    },
}

impl From<ProspectiveError> for CalibrationIntegrityError {
    fn from(value: ProspectiveError) -> Self {
        Self::Prospective(value)
    }
}

/// Derive the calibration assignment using only information fixed before the
/// outcome exists. `eval-v1` uses ten equal-width buckets of P(Boolean(true)).
fn expected_calibration_bucket(
    attempt: &ProspectiveAttemptCommitment,
) -> Result<Option<String>, CalibrationIntegrityError> {
    if attempt.evaluation_protocol().protocol_version() != "eval-v1" {
        return Err(CalibrationIntegrityError::UnknownEvaluationProtocol(
            attempt.evaluation_protocol().protocol_version().to_string(),
        ));
    }

    let ForecastAttemptDecision::Forecast(payload) = attempt.decision() else {
        return Err(CalibrationIntegrityError::NotForecastResolution);
    };

    let mut saw_boolean = false;
    let mut p_true = 0.0;
    for branch in payload.branches() {
        if let OutcomeRegion::Boolean(value) = &branch.outcome {
            saw_boolean = true;
            if *value {
                p_true += branch.probability.get();
            }
        }
    }
    if !saw_boolean {
        return Ok(None);
    }

    let bucket = ((p_true * 10.0).floor() as usize).min(9);
    Ok(Some(format!("ptrue-decile-{bucket:02}")))
}

fn verify_calibration_assignment(
    attempt: &ProspectiveAttemptCommitment,
    resolution: &ProspectiveAttemptResolution,
) -> Result<Option<String>, CalibrationIntegrityError> {
    resolution.validate_against(attempt)?;

    let ProspectiveAttemptResolution::Forecast {
        calibration_bucket,
        ..
    } = resolution
    else {
        return Err(CalibrationIntegrityError::NotForecastResolution);
    };

    let expected = expected_calibration_bucket(attempt)?;
    let stored = calibration_bucket
        .as_ref()
        .map(|label| label.as_str().to_string());
    if stored != expected {
        return Err(CalibrationIntegrityError::BucketMismatch { stored, expected });
    }
    Ok(expected)
}

fn lineage(snapshot: &str) -> ObservationLineage {
    ObservationLineage::observed(vec![
        ObservedSnapshotRef::new("public-statistics", snapshot, "sha256:fixture").unwrap(),
    ])
    .unwrap()
}

fn window() -> ForecastWindow {
    ForecastWindow::new(
        ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
        ForecastSpan::ordinal_steps("calendar-month", 3).unwrap(),
    )
    .unwrap()
}

fn boolean_payload(p_true: f64) -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("binary-economic-event".into()),
        vec![
            (p_true, OutcomeRegion::Boolean(true), vec![]),
            (1.0 - p_true, OutcomeRegion::Boolean(false), vec![]),
        ],
        0.0,
    )
    .unwrap()
}

fn interval_payload() -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("continuous-economic-outcome".into()),
        vec![
            (0.5, OutcomeRegion::interval(0.0, 0.0).unwrap(), vec![]),
            (0.5, OutcomeRegion::interval(1.0, 1.0).unwrap(), vec![]),
        ],
        0.0,
    )
    .unwrap()
}

fn attempt_with_protocol(
    payload: ForecastPayload,
    protocol_version: &str,
) -> ProspectiveAttemptCommitment {
    ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new("attempt-calibration-integrity").unwrap(),
        lineage("input-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
        window(),
        "observation-policy-v1",
        "sha256:inputs",
        vec!["model-v1".into()],
        vec!["generator-v1".into()],
        None,
        vec![ExternalReference::new("test.claim", "claim-1").unwrap()],
        EvaluationProtocol::new(protocol_version, "brier", "selective-risk-v1").unwrap(),
        ForecastAttemptDecision::Forecast(payload),
        "",
    )
    .unwrap()
}

fn attempt(payload: ForecastPayload) -> ProspectiveAttemptCommitment {
    attempt_with_protocol(payload, "eval-v1")
}

fn resolution(
    attempt: &ProspectiveAttemptCommitment,
    id: &str,
    actual: OutcomeRegion,
    calibration_bucket: Option<&str>,
) -> ProspectiveAttemptResolution {
    ProspectiveAttemptResolution::resolve_forecast(
        ForecastResolutionId::new(id).unwrap(),
        attempt,
        lineage("outcome-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
        actual,
        0.0,
        calibration_bucket.map(str::to_string),
        "",
    )
    .unwrap()
}

#[test]
fn identical_forecast_has_same_bucket_for_true_and_false_outcomes() {
    let attempt = attempt(boolean_payload(0.7));
    let expected = expected_calibration_bucket(&attempt).unwrap();
    assert_eq!(expected.as_deref(), Some("ptrue-decile-07"));

    let true_resolution = resolution(
        &attempt,
        "resolution-true",
        OutcomeRegion::Boolean(true),
        expected.as_deref(),
    );
    let false_resolution = resolution(
        &attempt,
        "resolution-false",
        OutcomeRegion::Boolean(false),
        expected.as_deref(),
    );

    assert_eq!(
        verify_calibration_assignment(&attempt, &true_resolution).unwrap(),
        expected
    );
    assert_eq!(
        verify_calibration_assignment(&attempt, &false_resolution).unwrap(),
        expected
    );
}

#[test]
fn post_outcome_bucket_choice_passes_linkage_but_fails_calibration_integrity() {
    let attempt = attempt(boolean_payload(0.7));
    let resolution = resolution(
        &attempt,
        "resolution-forged-bucket",
        OutcomeRegion::Boolean(true),
        Some("ptrue-decile-02"),
    );

    // Raw ledger linkage does not own calibration semantics.
    resolution.validate_against(&attempt).unwrap();

    match verify_calibration_assignment(&attempt, &resolution) {
        Err(CalibrationIntegrityError::BucketMismatch { stored, expected }) => {
            assert_eq!(stored.as_deref(), Some("ptrue-decile-02"));
            assert_eq!(expected.as_deref(), Some("ptrue-decile-07"));
        }
        other => panic!("expected bucket mismatch, got {other:?}"),
    }
}

#[test]
fn missing_required_boolean_bucket_gets_no_calibration_evidence_credit() {
    let attempt = attempt(boolean_payload(0.7));
    let resolution = resolution(
        &attempt,
        "resolution-missing-bucket",
        OutcomeRegion::Boolean(true),
        None,
    );
    assert!(matches!(
        verify_calibration_assignment(&attempt, &resolution),
        Err(CalibrationIntegrityError::BucketMismatch { .. })
    ));
}

#[test]
fn boundary_probabilities_map_deterministically() {
    let zero = attempt(boolean_payload(0.0));
    let one = attempt(boolean_payload(1.0));
    assert_eq!(
        expected_calibration_bucket(&zero).unwrap().as_deref(),
        Some("ptrue-decile-00")
    );
    assert_eq!(
        expected_calibration_bucket(&one).unwrap().as_deref(),
        Some("ptrue-decile-09")
    );
}

#[test]
fn continuous_forecast_does_not_invent_boolean_calibration_bucket() {
    let attempt = attempt(interval_payload());
    assert_eq!(expected_calibration_bucket(&attempt).unwrap(), None);
    let resolution = resolution(
        &attempt,
        "resolution-continuous",
        OutcomeRegion::interval(0.0, 0.0).unwrap(),
        None,
    );
    assert_eq!(
        verify_calibration_assignment(&attempt, &resolution).unwrap(),
        None
    );
}

#[test]
fn unknown_evaluation_protocol_cannot_define_calibration_semantics_by_accident() {
    let attempt = attempt_with_protocol(boolean_payload(0.7), "eval-v2-unqualified");
    assert_eq!(
        expected_calibration_bucket(&attempt),
        Err(CalibrationIntegrityError::UnknownEvaluationProtocol(
            "eval-v2-unqualified".into()
        ))
    );
}

#[test]
fn abstention_is_not_assigned_a_forecast_calibration_bucket() {
    let attempt = ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new("attempt-calibration-abstention").unwrap(),
        lineage("input-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
        window(),
        "observation-policy-v1",
        "sha256:inputs",
        vec!["model-v1".into()],
        vec!["generator-v1".into()],
        None,
        vec![],
        EvaluationProtocol::new("eval-v1", "brier", "selective-risk-v1").unwrap(),
        ForecastAttemptDecision::Abstain(AbstentionReason::OutOfDistributionScenario),
        "",
    )
    .unwrap();
    let resolution = ProspectiveAttemptResolution::resolve_abstention(
        ForecastResolutionId::new("resolution-calibration-abstention").unwrap(),
        &attempt,
        lineage("outcome-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
        OutcomeRegion::Boolean(false),
        "",
    )
    .unwrap();

    assert_eq!(
        verify_calibration_assignment(&attempt, &resolution),
        Err(CalibrationIntegrityError::NotForecastResolution)
    );
}
