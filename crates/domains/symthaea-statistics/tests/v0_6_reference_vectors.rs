// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_statistics::*;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

fn predictors() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![3.0, 1.0],
        vec![2.0, 3.0],
    ]
}

const OUTCOMES: [f64; 8] = [1.2, 3.1, 0.4, 2.5, 4.6, 1.7, 6.4, 3.0];

#[test]
fn qr_ols_and_influence_match_numpy_statsmodels() {
    let fit = try_multiple_linear_regression(&predictors(), &OUTCOMES, true, 0.95).unwrap();
    let expected = [1.1670454545454567, 2.0093749999999986, -0.7255681818181818];
    for (&actual, expected) in fit.coefficients.iter().zip(expected) {
        assert_close(actual, expected, 2e-12);
    }
    let diagnostics = try_regression_diagnostics(&predictors(), &OUTCOMES, true).unwrap();
    assert_close(diagnostics.mean_squared_error, 0.007232954545454501, 2e-13);
    assert_close(diagnostics.press, 0.0935781763791766, 2e-11);
    let expected_leverage = [
        0.4090909090909091,
        0.3153409090909091,
        0.3522727272727274,
        0.13352272727272732,
        0.22727272727272732,
        0.2926136363636364,
        0.6335227272727273,
        0.6363636363636364,
    ];
    for (&actual, expected) in diagnostics.leverage.iter().zip(expected_leverage) {
        assert_close(actual, expected, 2e-11);
    }
}

#[test]
fn ridge_matches_independent_standardized_solution() {
    let fit = try_ridge_regression(
        &predictors(),
        &OUTCOMES,
        RidgeRegressionOptions {
            lambda: 2.5,
            include_intercept: true,
            standardize_predictors: true,
        },
    )
    .unwrap();
    let expected = [1.4949381326066773, 1.4336759311279288, -0.3773627080147452];
    for (&actual, expected) in fit.coefficients.iter().zip(expected) {
        assert_close(actual, expected, 2e-11);
    }
    assert_close(fit.effective_degrees_of_freedom, 2.479493704352424, 2e-11);
    assert_close(fit.sum_squared_error, 2.2529202706405593, 2e-11);
    assert_close(fit.generalized_cross_validation, 0.5913954659681645, 2e-11);
}

#[test]
fn collinearity_and_bp_match_statsmodels() {
    let vif = try_variance_inflation_factors(&predictors()).unwrap();
    assert_close(vif.values[0], 1.171875, 2e-12);
    assert_close(vif.values[1], 1.171875, 2e-12);
    let bp = try_breusch_pagan_test(&predictors(), &OUTCOMES).unwrap();
    assert_close(bp.statistic, 2.520840397631898, 2e-10);
    assert_close(bp.p_value, 0.28353486045216547, 2e-8);
}

#[test]
fn dependent_covariances_match_statsmodels() {
    let clusters = [0, 0, 1, 1, 2, 2, 3, 3];
    let cluster = try_cluster_robust_inference(
        &predictors(),
        &OUTCOMES,
        &clusters,
        true,
        ClusterCorrection::Cr1,
    )
    .unwrap();
    let expected_cluster = [
        [
            0.0002960726037078103,
            -0.000025814309563264434,
            -0.000167067686944705,
        ],
        [
            -0.000025814309563263506,
            0.0016019886531777492,
            -0.0008977784334254702,
        ],
        [
            -0.00016706768694470448,
            -0.0008977784334254705,
            0.0006210406925053872,
        ],
    ];
    for row in 0..3 {
        for column in 0..3 {
            assert_close(
                cluster.covariance.try_get(row, column).unwrap(),
                expected_cluster[row][column],
                2e-11,
            );
        }
    }

    let hac = try_newey_west_inference(&predictors(), &OUTCOMES, true, 2).unwrap();
    let expected_hac = [
        [
            0.0005678054187379672,
            -0.000020476928068968296,
            -0.0002548604219649197,
        ],
        [
            -0.000020476928068968167,
            0.0005508255761517027,
            -0.00023799062223024705,
        ],
        [
            -0.0002548604219649195,
            -0.00023799062223024713,
            0.0002333831943438749,
        ],
    ];
    for row in 0..3 {
        for column in 0..3 {
            assert_close(
                hac.covariance.try_get(row, column).unwrap(),
                expected_hac[row][column],
                2e-11,
            );
        }
    }
}

#[test]
fn isotonic_and_conformal_match_reference_algorithms() {
    let calibrator = try_isotonic_calibration(
        &[0.1, 0.2, 0.2, 0.4, 0.6, 0.8, 0.9],
        &[false, true, false, false, true, true, true],
        Some(&[1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0]),
    )
    .unwrap();
    let predictions = calibrator
        .try_predict_many(&[0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
        .unwrap();
    assert_eq!(predictions, vec![0.0, 0.5, 0.5, 0.5, 1.0, 1.0]);

    let conformal = try_split_conformal_regression(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[0.2, 0.8, 2.4, 2.8, 4.5, 4.7],
        0.8,
    )
    .unwrap();
    assert_close(conformal.radius, 0.5, 1e-15);
    assert_eq!(
        conformal.try_interval(6.0).unwrap(),
        Interval {
            low: 5.5,
            high: 6.5
        }
    );
}

#[test]
fn causal_estimators_match_manual_and_statsmodels_references() {
    let aipw = try_augmented_inverse_probability_ate(
        &[4.0, 5.0, 1.0, 2.0, 6.0, 2.0],
        &[true, true, false, false, true, false],
        &[0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
        &[1.5, 2.0, 1.0, 2.0, 3.0, 2.0],
        &[4.0, 5.0, 3.0, 4.0, 6.0, 4.0],
        0.95,
    )
    .unwrap();
    assert_close(aipw.average_treatment_effect, 2.4166666666666665, 1e-13);
    assert_close(aipw.standard_error, 0.2006932429798716, 2e-13);

    let did = try_repeated_cross_section_difference_in_differences(
        &[1.0, 1.2, 2.0, 2.2, 1.5, 1.7, 4.5, 4.7],
        &[false, false, true, true, false, false, true, true],
        &[false, false, false, false, true, true, true, true],
        0.95,
    )
    .unwrap();
    assert_close(did.effect, 2.0, 2e-12);
    assert_close(did.standard_error, 0.2, 2e-12);
    assert_close(did.t_statistic, 10.0, 2e-10);
}

#[test]
fn exact_randomization_and_density_match_reference_values() {
    let randomization = try_randomization_test_ate(
        &[0.0, 1.0, 10.0, 11.0],
        &[false, false, true, true],
        RandomizationTestOptions::default(),
    )
    .unwrap();
    assert_close(randomization.p_value, 1.0 / 3.0, 1e-15);

    let density =
        try_gaussian_kernel_density(&[-2.0, -1.0, 1.0, 2.0], Bandwidth::Fixed(0.5)).unwrap();
    assert_close(density.try_pdf(0.75).unwrap(), 0.1852332088252571, 2e-14);
    assert_close(density.try_cdf(0.75).unwrap(), 0.5786286389957912, 2e-13);
    assert_close(
        try_dkw_band(100, 0.95).unwrap().epsilon,
        0.13581015157406195,
        2e-15,
    );
}

#[test]
fn residual_and_bca_diagnostics_match_external_references() {
    let residuals = [0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.1, -0.4, 0.25, -0.15];
    let ljung_box = try_ljung_box_test(&residuals, 3, 0).unwrap();
    assert_close(ljung_box.statistic, 22.95561839540373, 2e-11);
    assert_close(ljung_box.p_value, 4.125239496919034e-5, 2e-10);
    let jarque_bera = try_jarque_bera_test(&residuals).unwrap();
    assert_close(jarque_bera.statistic, 0.8133633473934934, 2e-12);
    assert_close(jarque_bera.skewness, -0.04628563051774504, 2e-12);
    assert_close(jarque_bera.excess_kurtosis, -1.3940956191648703, 2e-12);

    let bca = try_bca_bootstrap_mean(&[1.0, 2.0, 3.0, 10.0], 256, 0.9, 42).unwrap();
    assert_close(bca.bias_correction, 0.058553946597514346, 2e-12);
    assert_close(bca.acceleration, 0.0848528137423857, 2e-12);
    assert_close(bca.interval.low, 1.75, 2e-12);
    assert_close(bca.interval.high, 7.962667273429325, 2e-11);
}
