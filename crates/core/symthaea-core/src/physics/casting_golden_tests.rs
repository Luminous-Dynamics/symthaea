// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen CASTING-001C numerical oracle tests.
//!
//! The expected values below are fixed independently of the candidate model APIs.
//! They are synthetic arithmetic fixtures, not measured foundry observations.

use super::casting::{
    thermal_modulus_m, CastingModelError, ChvorinovConformalCalibrator, ChvorinovModel,
    ChvorinovObservation, CastingValidationStatus,
};

fn training_fixture() -> Vec<ChvorinovObservation> {
    // Exact relation: t[s] = 1_200_000 [s/m^2] * m[m]^2.
    // Inputs are picked so the expected outputs are straightforward to verify.
    vec![
        ChvorinovObservation::new(0.001, 1.2),
        ChvorinovObservation::new(0.002, 4.8),
        ChvorinovObservation::new(0.004, 19.2),
    ]
}

fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:.15e} differs from expected {expected:.15e} by more than {tolerance:.3e}"
    );
}

#[test]
fn frozen_fit_recovers_independent_known_answer_parameters() {
    let model = ChvorinovModel::calibrate(&training_fixture()).unwrap();

    assert_abs_close(model.exponent(), 2.0, 1e-10);
    assert_abs_close(model.mold_constant_s_per_m_pow_n(), 1_200_000.0, 1e-5);
    assert_eq!(model.sample_count(), 3);
    assert!(model.fit_r_squared() > 0.999_999_99);
}

#[test]
fn frozen_point_predictions_match_reference_values() {
    let model = ChvorinovModel::calibrate(&training_fixture()).unwrap();

    for (modulus_m, expected_seconds) in [
        (0.001, 1.2),
        (0.002, 4.8),
        (0.003, 10.8),
        (0.004, 19.2),
    ] {
        let prediction = model.estimate_solidification_time_s(modulus_m).unwrap();
        assert_abs_close(prediction, expected_seconds, 1e-7);
    }
}

#[test]
fn declared_fixed_exponent_baseline_matches_same_oracle() {
    let model =
        ChvorinovModel::calibrate_with_fixed_exponent(&training_fixture(), 2.0).unwrap();

    assert_abs_close(model.exponent(), 2.0, 1e-12);
    assert_abs_close(model.mold_constant_s_per_m_pow_n(), 1_200_000.0, 1e-5);
    assert_abs_close(
        model.estimate_solidification_time_s(0.003).unwrap(),
        10.8,
        1e-7,
    );
}

#[test]
fn si_geometry_ratio_matches_frozen_reference() {
    // 6e-6 m^3 / 2e-3 m^2 = 3e-3 m.
    assert_abs_close(thermal_modulus_m(6e-6, 2e-3).unwrap(), 0.003, 1e-12);
    assert!(matches!(
        thermal_modulus_m(0.0, 2e-3),
        Err(CastingModelError::InvalidGeometry)
    ));
    assert!(matches!(
        thermal_modulus_m(6e-6, f64::NAN),
        Err(CastingModelError::InvalidGeometry)
    ));
}

#[test]
fn frozen_conformal_quantile_and_interval_match_reference_values() {
    let model = ChvorinovModel::calibrate(&training_fixture()).unwrap();
    // Against the frozen model, the point predictions are 2.7, 7.5, and 14.7 s.
    // Observations are 1.1x, 0.95x, and 1.1x those values, respectively.
    let calibration = [
        ChvorinovObservation::new(0.0015, 2.97),
        ChvorinovObservation::new(0.0025, 7.125),
        ChvorinovObservation::new(0.0035, 16.17),
    ];
    let calibrator =
        ChvorinovConformalCalibrator::calibrate(&model, &calibration, 0.25).unwrap();

    assert_abs_close(
        calibrator.log_residual_quantile(),
        1.1_f64.ln(),
        1e-8,
    );
    assert_eq!(calibrator.calibration_sample_count(), 3);

    let interval = calibrator.predict_interval(&model, 0.003).unwrap();
    assert_abs_close(interval.estimate_s, 10.8, 1e-7);
    assert_abs_close(interval.lower_s, 9.818181818181818, 1e-7);
    assert_abs_close(interval.upper_s, 11.88, 1e-7);
    assert_abs_close(interval.nominal_coverage, 0.75, 1e-12);
}

#[test]
fn intentional_data_corruption_is_detected_by_frozen_prediction_oracle() {
    let mut corrupted = training_fixture();
    corrupted[2] = ChvorinovObservation::new(0.004, 21.12);

    let changed_model = ChvorinovModel::calibrate(&corrupted).unwrap();
    let changed_prediction = changed_model.estimate_solidification_time_s(0.003).unwrap();

    // A 10% change to a training measurement must not silently preserve the golden answer.
    assert!((changed_prediction - 10.8).abs() > 0.1);
}

#[test]
fn unsupported_domain_is_an_abstention_not_an_extrapolated_prediction() {
    let model = ChvorinovModel::calibrate(&training_fixture()).unwrap();

    assert!(matches!(
        model.estimate_solidification_time_s(0.0009),
        Err(CastingModelError::OutsideCalibrationDomain { .. })
    ));

    let report = model
        .evaluate_holdout(&[
            ChvorinovObservation::new(0.0005, 0.3),
            ChvorinovObservation::new(0.003, 10.8),
        ])
        .unwrap();

    assert_eq!(report.total_count, 2);
    assert_eq!(report.prediction_count, 1);
    assert_eq!(report.abstained_out_of_domain_count, 1);
    assert_abs_close(report.coverage_fraction, 0.5, 1e-12);
    assert_eq!(
        report.points[0].status,
        CastingValidationStatus::AbstainedOutsideCalibrationDomain
    );
    assert_eq!(report.points[0].predicted_time_s, None);
}
