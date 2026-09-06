// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for time-neutral proper scoring.
//!
//! This test deliberately does NOT change the production scoring API. It proves
//! that the current Brier/log/CRPS semantics can be reproduced directly from a
//! `ForecastPayload` without constructing fake tick/horizon metadata. If this
//! exact-head experiment qualifies, a later implementation PR can refactor the
//! scorer around a shared probability-surface kernel with evidence that the
//! numerical ordering relation is preserved.

use symthaea_futures_calibration::{
    BrierScore, Crps, FiniteScore, LogScore, ScoringError, ScoringRule,
};
use symthaea_futures_core::{
    ForecastDistribution, ForecastPayload, Horizon, OutcomeRegion, OutcomeSpaceId,
};

fn region_contains(region: &OutcomeRegion, actual: &OutcomeRegion) -> bool {
    match (region, actual) {
        (OutcomeRegion::Interval(bin), OutcomeRegion::Interval(observed)) => {
            bin.contains(observed.midpoint())
        }
        (left, right) => left == right,
    }
}

fn probability_of(payload: &ForecastPayload, target: &OutcomeRegion) -> f64 {
    payload
        .branches()
        .iter()
        .filter(|branch| region_contains(&branch.outcome, target))
        .map(|branch| branch.probability.get())
        .sum()
}

fn distinct_outcomes(payload: &ForecastPayload) -> Vec<OutcomeRegion> {
    let mut seen = Vec::new();
    for branch in payload.branches() {
        if !seen.contains(&branch.outcome) {
            seen.push(branch.outcome.clone());
        }
    }
    seen
}

fn score_brier(payload: &ForecastPayload, actual: &OutcomeRegion) -> Result<FiniteScore, ScoringError> {
    let classes = distinct_outcomes(payload);
    let covered = classes.iter().any(|class| region_contains(class, actual));

    let mut sum = 0.0;
    for class in &classes {
        let probability = probability_of(payload, class);
        let observed = if region_contains(class, actual) { 1.0 } else { 0.0 };
        sum += (probability - observed).powi(2);
    }
    if !covered {
        sum += 1.0;
    }
    let unsupported = payload.unsupported_mass().get();
    if unsupported > 0.0 {
        sum += unsupported.powi(2);
    }
    FiniteScore::new(sum)
}

fn score_log(
    rule: &LogScore,
    payload: &ForecastPayload,
    actual: &OutcomeRegion,
) -> Result<FiniteScore, ScoringError> {
    FiniteScore::new(-probability_of(payload, actual).max(rule.epsilon).ln())
}

fn interval_midpoint(region: &OutcomeRegion) -> Option<f64> {
    match region {
        OutcomeRegion::Interval(interval) => Some(interval.midpoint()),
        _ => None,
    }
}

fn score_crps(payload: &ForecastPayload, actual: &OutcomeRegion) -> Result<FiniteScore, ScoringError> {
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
    for &(left_location, left_probability) in &atoms {
        for &(right_location, right_probability) in &atoms {
            expected_pairwise_spread += left_probability
                * right_probability
                * (left_location - right_location).abs();
        }
    }
    FiniteScore::new(expected_abs_error - 0.5 * expected_pairwise_spread)
}

fn pair(
    branches: Vec<(f64, OutcomeRegion)>,
    unsupported_mass: f64,
) -> (ForecastDistribution, ForecastPayload) {
    let raw: Vec<_> = branches
        .into_iter()
        .map(|(probability, outcome)| (probability, outcome, Vec::new()))
        .collect();
    let outcome_space = OutcomeSpaceId("payload_scoring_parity".into());
    let distribution = ForecastDistribution::try_from_raw(
        73,
        Horizon(19),
        outcome_space.clone(),
        raw.clone(),
        unsupported_mass,
    )
    .expect("legacy parity fixture must be valid");
    let payload = ForecastPayload::try_from_raw(outcome_space, raw, unsupported_mass)
        .expect("neutral parity fixture must be valid");
    (distribution, payload)
}

#[test]
fn brier_payload_matches_legacy_bitwise() {
    for &(p_true, unsupported) in &[(1.0, 0.0), (0.7, 0.0), (0.35, 0.15), (0.0, 0.25)] {
        let p_false = 1.0 - p_true - unsupported;
        let (distribution, payload) = pair(
            vec![
                (p_true, OutcomeRegion::Boolean(true)),
                (p_false, OutcomeRegion::Boolean(false)),
            ],
            unsupported,
        );
        for actual in [OutcomeRegion::Boolean(true), OutcomeRegion::Boolean(false)] {
            let legacy = BrierScore.score(&distribution, &actual).unwrap().get();
            let neutral = score_brier(&payload, &actual).unwrap().get();
            assert_eq!(legacy.to_bits(), neutral.to_bits());
        }
    }
}

#[test]
fn log_payload_matches_legacy_bitwise() {
    let rule = LogScore::default();
    for &p_true in &[1.0, 0.75, 0.25, 0.0] {
        let (distribution, payload) = pair(
            vec![
                (p_true, OutcomeRegion::Boolean(true)),
                (1.0 - p_true, OutcomeRegion::Boolean(false)),
            ],
            0.0,
        );
        for actual in [OutcomeRegion::Boolean(true), OutcomeRegion::Boolean(false)] {
            let legacy = rule.score(&distribution, &actual).unwrap().get();
            let neutral = score_log(&rule, &payload, &actual).unwrap().get();
            assert_eq!(legacy.to_bits(), neutral.to_bits());
        }
    }
}

#[test]
fn crps_payload_matches_legacy_bitwise() {
    let (distribution, payload) = pair(
        vec![
            (0.2, OutcomeRegion::point(-5.0).unwrap()),
            (0.5, OutcomeRegion::point(0.0).unwrap()),
            (0.3, OutcomeRegion::point(8.0).unwrap()),
        ],
        0.0,
    );
    for observed in [-9.0, 0.0, 3.0, 20.0] {
        let actual = OutcomeRegion::point(observed).unwrap();
        let legacy = Crps.score(&distribution, &actual).unwrap().get();
        let neutral = score_crps(&payload, &actual).unwrap().get();
        assert_eq!(legacy.to_bits(), neutral.to_bits());
    }
}

#[test]
fn payload_path_preserves_crps_shape_errors() {
    let (distribution, payload) = pair(
        vec![
            (0.5, OutcomeRegion::Boolean(true)),
            (0.5, OutcomeRegion::Boolean(false)),
        ],
        0.0,
    );

    let boolean_actual = OutcomeRegion::Boolean(true);
    assert_eq!(
        Crps.score(&distribution, &boolean_actual),
        score_crps(&payload, &boolean_actual)
    );

    let interval_actual = OutcomeRegion::point(0.0).unwrap();
    assert_eq!(
        Crps.score(&distribution, &interval_actual),
        score_crps(&payload, &interval_actual)
    );
}

#[test]
fn neutral_scoring_requires_no_temporal_metadata() {
    let payload = ForecastPayload::try_from_raw(
        OutcomeSpaceId("no_time".into()),
        vec![
            (0.6, OutcomeRegion::Boolean(true), vec![]),
            (0.4, OutcomeRegion::Boolean(false), vec![]),
        ],
        0.0,
    )
    .unwrap();

    let score = score_brier(&payload, &OutcomeRegion::Boolean(true)).unwrap();
    assert!((score.get() - 0.32).abs() < 1e-12);
}
