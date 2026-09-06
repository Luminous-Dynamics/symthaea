// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for prospective resolution score integrity.
//!
//! The ledger intentionally records a finite reported score without depending upward
//! on the calibration crate. This test proves the missing higher-layer theorem: a
//! prospective resolution receives score evidence credit only when the reported score
//! is reproducible from the committed `ForecastPayload`, realized outcome, and the
//! precommitted recognized evaluation protocol and scoring rule.

use symthaea_futures_calibration::ScoringError;
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
enum ScoreIntegrityError {
    Prospective(ProspectiveError),
    NotForecastResolution,
    UnknownEvaluationProtocol(String),
    UnknownScoringRule(String),
    Scoring(ScoringError),
    ScoreMismatch { stored: u64, recomputed: u64 },
}

impl From<ProspectiveError> for ScoreIntegrityError {
    fn from(value: ProspectiveError) -> Self {
        Self::Prospective(value)
    }
}

impl From<ScoringError> for ScoreIntegrityError {
    fn from(value: ScoringError) -> Self {
        Self::Scoring(value)
    }
}

fn region_contains(region: &OutcomeRegion, actual: &OutcomeRegion) -> bool {
    match (region, actual) {
        (OutcomeRegion::Interval(bin), OutcomeRegion::Interval(observed)) => {
            bin.contains(observed.midpoint())
        }
        (left, right) => left == right,
    }
}

fn probability_of(payload: &ForecastPayload, actual: &OutcomeRegion) -> f64 {
    payload
        .branches()
        .iter()
        .filter(|branch| region_contains(&branch.outcome, actual))
        .map(|branch| branch.probability.get())
        .sum()
}

fn brier(payload: &ForecastPayload, actual: &OutcomeRegion) -> f64 {
    let classes: Vec<&OutcomeRegion> = payload
        .branches()
        .iter()
        .map(|branch| &branch.outcome)
        .collect();
    let covered = classes.iter().any(|class| region_contains(class, actual));

    let mut sum = 0.0;
    for class in classes {
        let p = probability_of(payload, class);
        let observed = if region_contains(class, actual) { 1.0 } else { 0.0 };
        sum += (p - observed).powi(2);
    }
    if !covered {
        sum += 1.0;
    }
    let unsupported = payload.unsupported_mass().get();
    if unsupported > 0.0 {
        sum += unsupported.powi(2);
    }
    sum
}

fn log_score(payload: &ForecastPayload, actual: &OutcomeRegion) -> f64 {
    -probability_of(payload, actual).max(1e-9).ln()
}

fn interval_midpoint(region: &OutcomeRegion) -> Option<f64> {
    match region {
        OutcomeRegion::Interval(interval) => Some(interval.midpoint()),
        _ => None,
    }
}

fn crps(payload: &ForecastPayload, actual: &OutcomeRegion) -> Result<f64, ScoringError> {
    let Some(observed) = interval_midpoint(actual) else {
        return Err(ScoringError::ActualNotIntervalShaped);
    };
    let atoms: Vec<(f64, f64)> = payload
        .branches()
        .iter()
        .filter_map(|branch| {
            interval_midpoint(&branch.outcome)
                .map(|location| (location, branch.probability.get()))
        })
        .collect();
    if atoms.is_empty() {
        return Err(ScoringError::NoIntervalAtoms);
    }

    let expected_abs_error: f64 = atoms
        .iter()
        .map(|&(location, probability)| probability * (location - observed).abs())
        .sum();
    let mut expected_pairwise_spread = 0.0;
    for &(left, left_probability) in &atoms {
        for &(right, right_probability) in &atoms {
            expected_pairwise_spread +=
                left_probability * right_probability * (left - right).abs();
        }
    }
    Ok(expected_abs_error - 0.5 * expected_pairwise_spread)
}

fn verify_reported_score(
    attempt: &ProspectiveAttemptCommitment,
    resolution: &ProspectiveAttemptResolution,
) -> Result<f64, ScoreIntegrityError> {
    resolution.validate_against(attempt)?;

    let ForecastAttemptDecision::Forecast(payload) = attempt.decision() else {
        return Err(ScoreIntegrityError::NotForecastResolution);
    };
    let ProspectiveAttemptResolution::Forecast {
        actual_continuation,
        score,
        evaluation_protocol,
        ..
    } = resolution
    else {
        return Err(ScoreIntegrityError::NotForecastResolution);
    };

    if evaluation_protocol.protocol_version() != "eval-v1" {
        return Err(ScoreIntegrityError::UnknownEvaluationProtocol(
            evaluation_protocol.protocol_version().to_string(),
        ));
    }

    let recomputed = match evaluation_protocol.scoring_rule() {
        "brier" => brier(payload, actual_continuation),
        "log-score" => log_score(payload, actual_continuation),
        "crps" => crps(payload, actual_continuation)?,
        other => return Err(ScoreIntegrityError::UnknownScoringRule(other.to_string())),
    };
    let stored = score.get();
    if stored.to_bits() != recomputed.to_bits() {
        return Err(ScoreIntegrityError::ScoreMismatch {
            stored: stored.to_bits(),
            recomputed: recomputed.to_bits(),
        });
    }
    Ok(recomputed)
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

fn boolean_payload() -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("inflation-target".into()),
        vec![
            (0.7, OutcomeRegion::Boolean(true), vec![]),
            (0.3, OutcomeRegion::Boolean(false), vec![]),
        ],
        0.0,
    )
    .unwrap()
}

fn interval_payload() -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("inflation-rate".into()),
        vec![
            (0.5, OutcomeRegion::interval(0.0, 0.0).unwrap(), vec![]),
            (0.5, OutcomeRegion::interval(10.0, 10.0).unwrap(), vec![]),
        ],
        0.0,
    )
    .unwrap()
}

fn attempt_with_protocol(
    payload: ForecastPayload,
    protocol_version: &str,
    scoring_rule: &str,
) -> ProspectiveAttemptCommitment {
    ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new("attempt-score-integrity").unwrap(),
        lineage("input-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
        window(),
        "observation-policy-v1",
        "sha256:inputs",
        vec!["model-v1".into()],
        vec!["generator-v1".into()],
        None,
        vec![ExternalReference::new("test.claim", "claim-1").unwrap()],
        EvaluationProtocol::new(protocol_version, scoring_rule, "selective-risk-v1").unwrap(),
        ForecastAttemptDecision::Forecast(payload),
        "",
    )
    .unwrap()
}

fn attempt(payload: ForecastPayload, scoring_rule: &str) -> ProspectiveAttemptCommitment {
    attempt_with_protocol(payload, "eval-v1", scoring_rule)
}

fn forecast_resolution(
    attempt: &ProspectiveAttemptCommitment,
    actual: OutcomeRegion,
    reported_score: f64,
) -> ProspectiveAttemptResolution {
    ProspectiveAttemptResolution::resolve_forecast(
        ForecastResolutionId::new("resolution-score-integrity").unwrap(),
        attempt,
        lineage("outcome-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
        actual,
        reported_score,
        None,
        "",
    )
    .unwrap()
}

#[test]
fn correct_precommitted_brier_score_is_reproducible() {
    let payload = boolean_payload();
    let actual = OutcomeRegion::Boolean(true);
    let expected = brier(&payload, &actual);
    let attempt = attempt(payload, "brier");
    let resolution = forecast_resolution(&attempt, actual, expected);
    let verified = verify_reported_score(&attempt, &resolution).unwrap();
    assert_eq!(verified.to_bits(), expected.to_bits());
}

#[test]
fn rounded_display_score_gets_no_exact_score_evidence_credit() {
    let attempt = attempt(boolean_payload(), "brier");
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), 0.18);
    assert!(matches!(
        verify_reported_score(&attempt, &resolution),
        Err(ScoreIntegrityError::ScoreMismatch { .. })
    ));
}

#[test]
fn forged_finite_score_passes_linkage_but_fails_score_integrity() {
    let attempt = attempt(boolean_payload(), "brier");
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), 0.19);

    // This is intentional: the ledger owns record/linkage validity, not score semantics.
    resolution.validate_against(&attempt).unwrap();

    match verify_reported_score(&attempt, &resolution) {
        Err(ScoreIntegrityError::ScoreMismatch { stored, recomputed }) => {
            assert_ne!(stored, recomputed);
        }
        other => panic!("expected score mismatch, got {other:?}"),
    }
}

#[test]
fn unknown_evaluation_protocol_gets_no_score_evidence_credit() {
    let payload = boolean_payload();
    let expected = brier(&payload, &OutcomeRegion::Boolean(true));
    let attempt = attempt_with_protocol(payload, "eval-v2-unqualified", "brier");
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), expected);
    assert_eq!(
        verify_reported_score(&attempt, &resolution),
        Err(ScoreIntegrityError::UnknownEvaluationProtocol(
            "eval-v2-unqualified".into()
        ))
    );
}

#[test]
fn unknown_precommitted_scoring_rule_gets_no_score_evidence_credit() {
    let attempt = attempt(boolean_payload(), "brier-vague-custom");
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), 0.18);
    assert_eq!(
        verify_reported_score(&attempt, &resolution),
        Err(ScoreIntegrityError::UnknownScoringRule(
            "brier-vague-custom".into()
        ))
    );
}

#[test]
fn log_score_is_recomputed_from_the_committed_payload() {
    let attempt = attempt(boolean_payload(), "log-score");
    let expected = -0.7f64.ln();
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), expected);
    assert_eq!(
        verify_reported_score(&attempt, &resolution)
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
}

#[test]
fn crps_is_recomputed_without_any_clock_metadata() {
    let payload = interval_payload();
    let actual = OutcomeRegion::interval(0.0, 0.0).unwrap();
    let expected = crps(&payload, &actual).unwrap();
    let attempt = attempt(payload, "crps");
    let resolution = forecast_resolution(&attempt, actual, expected);
    assert_eq!(
        verify_reported_score(&attempt, &resolution)
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
}

#[test]
fn crps_shape_error_prevents_score_evidence_credit() {
    let attempt = attempt(boolean_payload(), "crps");
    let resolution = forecast_resolution(&attempt, OutcomeRegion::Boolean(true), 0.0);
    assert_eq!(
        verify_reported_score(&attempt, &resolution),
        Err(ScoreIntegrityError::Scoring(
            ScoringError::ActualNotIntervalShaped
        ))
    );
}

#[test]
fn abstention_resolution_cannot_be_misclassified_as_scored_forecast() {
    let attempt = ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new("attempt-abstention-integrity").unwrap(),
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
        ForecastResolutionId::new("resolution-abstention-integrity").unwrap(),
        &attempt,
        lineage("outcome-vintage"),
        ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
        OutcomeRegion::Boolean(false),
        "",
    )
    .unwrap();

    resolution.validate_against(&attempt).unwrap();
    assert_eq!(
        verify_reported_score(&attempt, &resolution),
        Err(ScoreIntegrityError::NotForecastResolution)
    );
}
