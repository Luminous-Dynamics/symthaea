// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public-API reference cases for finite-sample TV-radius calibration.
//! Expected empirical distances below are hand-derived independently of the module.

use symthaea_operations_research::aviation_probability_calibration::{
    calibrate_tv_radius, calibrate_tv_radius_envelope, CalibrationFrequency, RadiusCalibrationError,
    RegimeCalibration, PROBABILITY_SCALE_PPM,
};

#[test]
fn hand_calculated_ten_percent_tv_shift_is_reported_exactly() {
    let report = calibrate_tv_radius(
        &[
            CalibrationFrequency { scenario_id: "normal".into(), nominal_probability_ppm: 500_000, observed_count: 40 },
            CalibrationFrequency { scenario_id: "disrupted".into(), nominal_probability_ppm: 500_000, observed_count: 60 },
        ],
        50_000,
    ).unwrap();
    assert_eq!(report.calibration_episode_count, 100);
    assert_eq!(report.empirical_tv_distance_ppm, 100_000);
    assert!(report.required_tv_radius_ppm >= 100_000);
}

#[test]
fn perfect_empirical_match_does_not_claim_zero_uncertainty() {
    let report = calibrate_tv_radius(
        &[
            CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 500_000, observed_count: 50 },
            CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 500_000, observed_count: 50 },
        ],
        50_000,
    ).unwrap();
    assert_eq!(report.empirical_tv_distance_ppm, 0);
    assert!(report.sampling_allowance_ppm > 0);
    assert!(report.required_tv_radius_ppm > 0);
}

#[test]
fn confidence_level_and_sample_size_change_only_the_allowance_as_expected() {
    let small = [
        CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 500_000, observed_count: 50 },
        CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 500_000, observed_count: 50 },
    ];
    let mut large = small.clone();
    for row in &mut large { row.observed_count *= 100; }
    let small_high_conf = calibrate_tv_radius(&small, 10_000).unwrap();
    let small_low_conf = calibrate_tv_radius(&small, 100_000).unwrap();
    let large_high_conf = calibrate_tv_radius(&large, 10_000).unwrap();

    assert!(small_high_conf.sampling_allowance_ppm >= small_low_conf.sampling_allowance_ppm);
    assert!(large_high_conf.sampling_allowance_ppm <= small_high_conf.sampling_allowance_ppm);
    assert_eq!(large_high_conf.empirical_tv_distance_ppm, small_high_conf.empirical_tv_distance_ppm);
}

#[test]
fn calibration_bound_never_exceeds_one_and_records_saturation() {
    let report = calibrate_tv_radius(
        &[
            CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 500_000, observed_count: 1 },
            CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 500_000, observed_count: 0 },
        ],
        50_000,
    ).unwrap();
    assert_eq!(report.required_tv_radius_ppm, PROBABILITY_SCALE_PPM);
    assert!(report.capped_at_maximum_radius);
}

#[test]
fn invalid_empirical_input_fails_closed() {
    let result = calibrate_tv_radius(
        &[CalibrationFrequency {
            scenario_id: "only".into(),
            nominal_probability_ppm: 900_000,
            observed_count: 1,
        }],
        50_000,
    );
    assert!(matches!(
        result,
        Err(RadiusCalibrationError::NominalProbabilityMassMismatch { .. })
    ));
}


#[test]
fn regime_envelope_is_the_maximum_across_heterogeneous_calibration_sets() {
    let regimes = [
        RegimeCalibration {
            regime_id: "nominal".into(),
            frequencies: vec![
                CalibrationFrequency { scenario_id: "routine".into(), nominal_probability_ppm: 600_000, observed_count: 60 },
                CalibrationFrequency { scenario_id: "capacity".into(), nominal_probability_ppm: 300_000, observed_count: 30 },
                CalibrationFrequency { scenario_id: "weather".into(), nominal_probability_ppm: 100_000, observed_count: 10 },
            ],
        },
        RegimeCalibration {
            regime_id: "shifted".into(),
            frequencies: vec![
                CalibrationFrequency { scenario_id: "routine".into(), nominal_probability_ppm: 600_000, observed_count: 40 },
                CalibrationFrequency { scenario_id: "capacity".into(), nominal_probability_ppm: 300_000, observed_count: 30 },
                CalibrationFrequency { scenario_id: "weather".into(), nominal_probability_ppm: 100_000, observed_count: 30 },
            ],
        },
    ];
    let envelope = calibrate_tv_radius_envelope(&regimes, 50_000).unwrap();
    assert_eq!(envelope.per_regime_failure_probability_ppm, 25_000);
    assert_eq!(envelope.regimes[0].regime_id, "nominal");
    assert_eq!(envelope.regimes[1].regime_id, "shifted");
    assert_eq!(
        envelope.maximum_required_tv_radius_ppm,
        envelope.regimes.iter().map(|r| r.report.required_tv_radius_ppm).max().unwrap()
    );
    assert!(envelope.regimes[1].report.empirical_tv_distance_ppm > 0);
}

#[test]
fn regime_envelope_rejects_unallocatable_family_confidence_budget() {
    let regimes = [
        RegimeCalibration {
            regime_id: "airport-a".into(),
            frequencies: vec![CalibrationFrequency {
                scenario_id: "routine".into(),
                nominal_probability_ppm: PROBABILITY_SCALE_PPM,
                observed_count: 10,
            }],
        },
        RegimeCalibration {
            regime_id: "airport-b".into(),
            frequencies: vec![CalibrationFrequency {
                scenario_id: "routine".into(),
                nominal_probability_ppm: PROBABILITY_SCALE_PPM,
                observed_count: 10,
            }],
        },
    ];
    assert!(matches!(
        calibrate_tv_radius_envelope(&regimes, 1),
        Err(RadiusCalibrationError::FailureProbabilityTooSmallForRegimeCount { .. })
    ));
}
