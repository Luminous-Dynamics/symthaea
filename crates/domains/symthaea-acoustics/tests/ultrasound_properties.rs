// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-call invariants for the research-only ultrasound equations.
//!
//! These tests check mathematical behavior, not probe performance, exposure
//! safety, hardware acquisition, or clinical validity.

use symthaea_acoustics::ultrasound::{
    assess_sampling_plan, estimate_axial_resolution_from_pulse_cycles_m,
    estimate_wavelength_m, minimum_nyquist_sample_rate_hz, UltrasoundModelError,
};
use symthaea_acoustics::ultrasound::phantom::{
    PhantomError, PointReflector, PointReflectorPhantom,
};

const SOUND_SPEED_M_S: f64 = 1_540.0;
const CENTER_FREQUENCY_HZ: f64 = 5_000_000.0;

#[test]
fn wavelength_scales_linearly_with_sound_speed_and_inversely_with_frequency() {
    let base = estimate_wavelength_m(CENTER_FREQUENCY_HZ, SOUND_SPEED_M_S).unwrap();
    let doubled_speed = estimate_wavelength_m(CENTER_FREQUENCY_HZ, 2.0 * SOUND_SPEED_M_S).unwrap();
    let doubled_frequency =
        estimate_wavelength_m(2.0 * CENTER_FREQUENCY_HZ, SOUND_SPEED_M_S).unwrap();

    assert!((doubled_speed / base - 2.0).abs() < 1e-12);
    assert!((doubled_frequency / base - 0.5).abs() < 1e-12);
}

#[test]
fn idealized_axial_resolution_scales_with_pulse_cycles() {
    let one_cycle =
        estimate_axial_resolution_from_pulse_cycles_m(CENTER_FREQUENCY_HZ, SOUND_SPEED_M_S, 1.0)
            .unwrap();
    let four_cycles =
        estimate_axial_resolution_from_pulse_cycles_m(CENTER_FREQUENCY_HZ, SOUND_SPEED_M_S, 4.0)
            .unwrap();

    assert!((four_cycles / one_cycle - 4.0).abs() < 1e-12);
}

#[test]
fn axial_resolution_scales_inversely_with_frequency_for_fixed_medium_and_cycles() {
    let base =
        estimate_axial_resolution_from_pulse_cycles_m(CENTER_FREQUENCY_HZ, SOUND_SPEED_M_S, 2.0)
            .unwrap();
    let doubled_frequency =
        estimate_axial_resolution_from_pulse_cycles_m(
            2.0 * CENTER_FREQUENCY_HZ,
            SOUND_SPEED_M_S,
            2.0,
        )
            .unwrap();

    assert!((doubled_frequency / base - 0.5).abs() < 1e-12);
}

#[test]
fn sampling_boundary_is_inclusive_at_the_mathematical_nyquist_minimum() {
    let highest_frequency_hz = 8_000_000.0;
    let minimum = minimum_nyquist_sample_rate_hz(highest_frequency_hz).unwrap();
    let at_boundary =
        assess_sampling_plan(minimum, highest_frequency_hz, Some("report-id")).unwrap();
    let below_boundary =
        assess_sampling_plan(
            minimum * (1.0 - f64::EPSILON * 2.0),
            highest_frequency_hz,
            Some("report-id"),
        )
            .unwrap();

    assert!(at_boundary.nyquist_condition_satisfied);
    assert!(!below_boundary.nyquist_condition_satisfied);
    assert!(at_boundary.passes_minimum_checks());
    assert!(!below_boundary.passes_minimum_checks());
}

#[test]
fn finite_inputs_that_overflow_a_derived_value_fail_closed() {
    assert_eq!(
        estimate_wavelength_m(1e-308, 1e308),
        Err(UltrasoundModelError::DerivedValueNonFinite("wavelength_m"))
    );
    assert_eq!(
        minimum_nyquist_sample_rate_hz(f64::MAX),
        Err(UltrasoundModelError::DerivedValueNonFinite(
            "minimum_nyquist_sample_rate_hz"
        ))
    );
}

#[test]
fn sampling_evidence_reference_is_presence_only_not_a_verified_artifact() {
    let assessment =
        assess_sampling_plan(20_000_000.0, 8_000_000.0, Some("unverified-placeholder")).unwrap();

    assert!(assessment.anti_alias_evidence_reference_supplied);
    // The API deliberately reports only a supplied reference; it does not verify it.
    // Do not interpret this assertion as evidence that a real filter report exists.
    assert!(assessment.passes_minimum_checks());
}

#[test]
fn analytic_echo_delay_scales_with_depth_and_inverse_sound_speed() {
    let shallow = PointReflectorPhantom::new(
        SOUND_SPEED_M_S,
        vec![PointReflector::new(0.01, 0.5).unwrap()],
    )
    .unwrap();
    let shallow_delay = shallow.echo_time_s(0.01).unwrap();
    let deep_delay = shallow.echo_time_s(0.02).unwrap();

    let faster_medium = PointReflectorPhantom::new(
        2.0 * SOUND_SPEED_M_S,
        vec![PointReflector::new(0.01, 0.5).unwrap()],
    )
    .unwrap();
    let faster_delay = faster_medium.echo_time_s(0.01).unwrap();

    assert!((deep_delay / shallow_delay - 2.0).abs() < 1e-12);
    assert!((faster_delay / shallow_delay - 0.5).abs() < 1e-12);
}

#[test]
fn synthetic_rf_fixture_preserves_known_echo_time_and_sample_position() {
    let phantom = PointReflectorPhantom::new(
        SOUND_SPEED_M_S,
        vec![PointReflector::new(0.01, 0.5).unwrap()],
    )
    .unwrap();
    let trace = phantom
        .simulate_rf_trace(CENTER_FREQUENCY_HZ, 20_000_000.0, 2.0, 25e-6, 1_000)
        .unwrap();
    let echo = trace.expected_echoes().first().unwrap();
    let expected_time_s = 2.0 * 0.01 / SOUND_SPEED_M_S;
    let expected_fractional_index = expected_time_s * 20_000_000.0;

    assert!((echo.arrival_time_s() - expected_time_s).abs() < 1e-15);
    assert!((echo.fractional_sample_index() - expected_fractional_index).abs() < 1e-12);
    assert_eq!(trace.samples().len(), 500);
    assert!(trace.samples().iter().all(|sample| sample.is_finite()));
}

#[test]
fn canonical_rf_bytes_commit_to_parameters_echo_truth_and_samples() {
    let phantom = PointReflectorPhantom::new(
        SOUND_SPEED_M_S,
        vec![PointReflector::new(0.01, 0.5).unwrap()],
    )
    .unwrap();
    let trace = phantom
        .simulate_rf_trace(CENTER_FREQUENCY_HZ, 20_000_000.0, 2.0, 25e-6, 1_000)
        .unwrap();
    let first = trace.canonical_bytes().unwrap();
    let second = trace.canonical_bytes().unwrap();

    assert_eq!(&first[..8], b"SYMRF001");
    assert_eq!(first.len(), 64 + 32 + 500 * 8);
    assert_eq!(first, second);
}

#[test]
fn synthetic_rf_fixture_rejects_unbounded_or_undersampled_requests() {
    let phantom = PointReflectorPhantom::new(
        SOUND_SPEED_M_S,
        vec![PointReflector::new(0.01, 0.5).unwrap()],
    )
    .unwrap();

    assert_eq!(
        phantom.simulate_rf_trace(CENTER_FREQUENCY_HZ, 9_000_000.0, 2.0, 25e-6, 1_000),
        Err(PhantomError::SampleRateBelowNyquist)
    );
    assert_eq!(
        phantom.simulate_rf_trace(CENTER_FREQUENCY_HZ, 20_000_000.0, 2.0, 25e-6, 0),
        Err(PhantomError::SampleBudgetZero)
    );
    assert_eq!(
        phantom.simulate_rf_trace(CENTER_FREQUENCY_HZ, 20_000_000.0, 2.0, 25e-6, 100),
        Err(PhantomError::SampleCountExceedsBudget)
    );
}
