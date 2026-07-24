// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_statistics::*;

#[test]
fn qr_rejects_wide_and_rank_deficient_matrices() {
    let wide = DenseMatrix::try_from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
    assert!(matches!(
        QrDecomposition::try_factor(&wide),
        Err(StatisticsError::InvalidShape { .. })
    ));
    let deficient =
        DenseMatrix::try_from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]]).unwrap();
    assert!(matches!(
        QrDecomposition::try_factor(&deficient),
        Err(StatisticsError::SingularMatrix { .. })
    ));
}

#[test]
fn ridge_rejects_invalid_penalties_and_constant_standardization() {
    let predictors = vec![vec![1.0], vec![1.0], vec![1.0]];
    let outcomes = [1.0, 2.0, 3.0];
    assert!(matches!(
        try_ridge_regression(
            &predictors,
            &outcomes,
            RidgeRegressionOptions {
                lambda: -1.0,
                ..Default::default()
            },
        ),
        Err(StatisticsError::NegativeValue { .. })
    ));
    assert!(matches!(
        try_ridge_regression(&predictors, &outcomes, RidgeRegressionOptions::default()),
        Err(StatisticsError::DegenerateSample { .. })
    ));
}

#[test]
fn influence_diagnostics_refuse_saturated_designs() {
    let predictors = vec![vec![0.0], vec![1.0]];
    assert!(matches!(
        try_regression_diagnostics(&predictors, &[0.0, 1.0], true),
        Err(StatisticsError::InsufficientSamples { .. })
    ));
}

#[test]
fn dependent_covariance_requires_clusters_and_valid_lags() {
    let predictors = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let outcomes = [0.0, 1.0, 2.0, 3.1];
    assert!(matches!(
        try_cluster_robust_inference(
            &predictors,
            &outcomes,
            &[1, 1, 1, 1],
            true,
            ClusterCorrection::Cr1,
        ),
        Err(StatisticsError::InsufficientSamples { .. })
    ));
    assert!(matches!(
        try_newey_west_inference(&predictors, &outcomes, true, outcomes.len()),
        Err(StatisticsError::InvalidLag { .. })
    ));
}

#[test]
fn isotonic_calibration_rejects_nonpositive_weights() {
    assert!(matches!(
        try_isotonic_calibration(&[0.1, 0.9], &[false, true], Some(&[1.0, 0.0])),
        Err(StatisticsError::NonPositiveValue { .. })
    ));
}

#[test]
fn conformal_contracts_reject_invalid_scales_and_probabilities() {
    assert!(matches!(
        try_normalized_conformal_regression(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 0.0], 0.9,),
        Err(StatisticsError::InvalidScale { .. })
    ));
    let classifier = try_binary_conformal_classification(&[0.2, 0.8], &[false, true], 0.9).unwrap();
    assert!(matches!(
        classifier.try_prediction_set(1.1),
        Err(StatisticsError::InvalidProbability { .. })
    ));
}

#[test]
fn aipw_requires_strict_overlap_and_both_arms() {
    assert!(matches!(
        try_augmented_inverse_probability_ate(
            &[1.0, 2.0],
            &[true, false],
            &[1.0, 0.5],
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.95,
        ),
        Err(StatisticsError::InvalidProbability { .. })
    ));
    assert!(matches!(
        try_augmented_inverse_probability_ate(
            &[1.0, 2.0],
            &[true, true],
            &[0.5, 0.5],
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.95,
        ),
        Err(StatisticsError::DegenerateSample { .. })
    ));
}

#[test]
fn did_requires_complete_design_cells() {
    assert!(matches!(
        try_repeated_cross_section_difference_in_differences(
            &[1.0, 2.0, 3.0],
            &[false, true, true],
            &[false, false, true],
            0.95,
        ),
        Err(StatisticsError::DegenerateSample { .. })
    ));
}

#[test]
fn blocked_randomization_rejects_singleton_blocks() {
    assert!(matches!(
        try_blocked_random_assignment(&[1, 1, 2], 0.5, 7),
        Err(StatisticsError::DegenerateSample { .. })
    ));
}

#[test]
fn randomization_falls_back_to_seeded_monte_carlo() {
    let outcomes: Vec<f64> = (0..20).map(|value| value as f64).collect();
    let treated: Vec<bool> = (0..20).map(|index| index >= 10).collect();
    let options = RandomizationTestOptions {
        maximum_exact_states: 1,
        monte_carlo_iterations: 100,
        seed: 99,
        ..Default::default()
    };
    let first = try_randomization_test_ate(&outcomes, &treated, options).unwrap();
    let second = try_randomization_test_ate(&outcomes, &treated, options).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.method, RandomizationMethod::MonteCarlo);
}

#[test]
fn density_rejects_invalid_or_unidentified_bandwidths() {
    assert!(matches!(
        try_gaussian_kernel_density(&[1.0, 2.0], Bandwidth::Fixed(0.0)),
        Err(StatisticsError::InvalidScale { .. })
    ));
    assert!(matches!(
        try_gaussian_kernel_density(&[1.0, 1.0, 1.0], Bandwidth::Scott),
        Err(StatisticsError::DegenerateSample { .. })
    ));
}

#[test]
fn dkw_and_residual_tests_validate_domains() {
    assert!(matches!(
        try_dkw_band(0, 0.95),
        Err(StatisticsError::EmptySample)
    ));
    assert!(matches!(
        try_ljung_box_test(&[1.0, -1.0, 1.0, -1.0], 2, 2),
        Err(StatisticsError::InvalidDegreesOfFreedom { .. })
    ));
}

#[test]
fn bca_requires_enough_observations_and_iterations() {
    assert!(matches!(
        try_bca_bootstrap_mean(&[1.0, 2.0], 100, 0.95, 1),
        Err(StatisticsError::InsufficientSamples { .. })
    ));
    assert!(matches!(
        try_bca_bootstrap_mean(&[1.0, 2.0, 3.0], 0, 0.95, 1),
        Err(StatisticsError::InvalidIterations { .. })
    ));
}
