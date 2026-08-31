// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_statistics::DenseMatrix;
use symthaea_statistics::{
    CoxRegressionOptions, CusumChart, EwmaChart, HeteroskedasticityCorrection, MetaAnalysisModel,
    PoissonRegressionOptions, StatisticsError, ThresholdObjective, try_binary_discrimination,
    try_cox_regression, try_cross_validation_summary, try_fit_beta_moments,
    try_inverse_probability_ate, try_k_fold, try_kaplan_meier, try_meta_analysis,
    try_poisson_regression, try_principal_components, try_propensity_matching,
    try_robust_linear_inference, try_select_threshold, try_stratified_k_fold, try_symmetric_eigen,
};

#[test]
fn deterministic_partitions_are_stable_and_disjoint() {
    let first = try_k_fold(101, 7, 0x1234).unwrap();
    let second = try_k_fold(101, 7, 0x1234).unwrap();
    assert_eq!(first, second);
    for fold in first {
        assert!(
            fold.training
                .iter()
                .all(|index| !fold.testing.contains(index))
        );
    }
}

#[test]
fn stratification_fails_when_minority_cannot_cover_folds() {
    let error = try_stratified_k_fold(&[true, false, false, false], 3, 1).unwrap_err();
    assert!(matches!(error, StatisticsError::DegenerateSample { .. }));
}

#[test]
fn predictive_metrics_reject_non_finite_scores() {
    assert!(try_binary_discrimination(&[0.1, f64::NAN], &[false, true]).is_err());
    assert!(try_cross_validation_summary(&[0.1, f64::INFINITY]).is_err());
}

#[test]
fn models_fail_closed_on_degenerate_designs() {
    let predictors = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
    assert!(
        try_poisson_regression(
            &predictors,
            &[1, 2, 3, 4],
            PoissonRegressionOptions::default()
        )
        .is_err()
    );
    assert!(
        try_robust_linear_inference(
            &predictors,
            &[1.0, 2.0, 3.0, 4.0],
            true,
            HeteroskedasticityCorrection::Hc3
        )
        .is_err()
    );
}

#[test]
fn symmetric_eigen_rejects_asymmetry() {
    let matrix = DenseMatrix::try_from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    assert!(try_symmetric_eigen(&matrix, 1e-12, 100).is_err());
}

#[test]
fn pca_rejects_zero_total_variance() {
    let data = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
    assert!(try_principal_components(&data).is_err());
}

#[test]
fn survival_rejects_negative_times_and_all_censored_log_models() {
    assert!(try_kaplan_meier(&[-1.0, 1.0], &[true, false], 0.95).is_err());
    let predictors = vec![vec![0.0], vec![1.0], vec![2.0]];
    assert!(
        try_cox_regression(
            &predictors,
            &[1.0, 2.0, 3.0],
            &[false; 3],
            CoxRegressionOptions::default()
        )
        .is_err()
    );
}

#[test]
fn inverse_probability_weights_require_strict_overlap() {
    let error = try_inverse_probability_ate(
        &[1.0, 2.0, 3.0, 4.0],
        &[true, true, false, false],
        &[1.0, 0.5, 0.5, 0.0],
        0.95,
    )
    .unwrap_err();
    assert!(matches!(error, StatisticsError::InvalidProbability { .. }));
}

#[test]
fn matching_is_reproducible_and_caliper_bounded() {
    let outcomes = [3.0, 5.0, 1.0, 3.0, 2.0, 4.0];
    let treated = [true, true, false, false, false, true];
    let propensity = [0.2, 0.8, 0.2, 0.8, 0.5, 0.52];
    let first =
        try_propensity_matching(&outcomes, &treated, &propensity, 0.05, false, 0.95).unwrap();
    let second =
        try_propensity_matching(&outcomes, &treated, &propensity, 0.05, false, 0.95).unwrap();
    assert_eq!(first, second);
    assert!(
        first
            .pairs
            .iter()
            .all(|pair| pair.propensity_distance <= 0.05)
    );
}

#[test]
fn distribution_fits_reject_impossible_moments() {
    assert!(try_fit_beta_moments(&[0.5, 0.5, 0.5]).is_err());
}

#[test]
fn meta_analysis_rejects_zero_standard_errors() {
    assert!(
        try_meta_analysis(
            &[0.0, 1.0],
            &[0.0, 0.1],
            MetaAnalysisModel::FixedEffect,
            0.95
        )
        .is_err()
    );
}

#[test]
fn decision_costs_must_be_meaningful() {
    assert!(
        try_select_threshold(
            &[0.1, 0.9],
            &[false, true],
            ThresholdObjective::MisclassificationCost {
                false_positive_cost: 0.0,
                false_negative_cost: 0.0,
            }
        )
        .is_err()
    );
}

#[test]
fn process_charts_do_not_mutate_after_invalid_input() {
    let mut ewma = EwmaChart::try_new(0.0, 1.0, 0.2, 3.0).unwrap();
    let before = ewma;
    assert!(ewma.try_push(f64::NAN).is_err());
    assert_eq!(ewma, before);

    let mut cusum = CusumChart::try_new(0.0, 1.0, 0.5, 5.0).unwrap();
    let before = cusum;
    assert!(cusum.try_push(f64::INFINITY).is_err());
    assert_eq!(cusum, before);
}
