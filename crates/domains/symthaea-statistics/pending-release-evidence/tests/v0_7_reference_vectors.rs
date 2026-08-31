use symthaea_statistics::{
    ElasticNetOptions, NegativeBinomialRegressionOptions, try_cochran_q,
    try_elastic_net_regression, try_kendall_tau_b, try_mcnemar_test, try_negative_binomial_cdf,
    try_negative_binomial_log_pmf, try_negative_binomial_regression, try_negative_binomial_sf,
    try_theil_sen_regression,
};

#[test]
fn negative_binomial_distribution_matches_scipy() {
    let log_pmf = try_negative_binomial_log_pmf(12, 7.5, 0.35).unwrap();
    let cdf = try_negative_binomial_cdf(12, 7.5, 0.35).unwrap();
    let sf = try_negative_binomial_sf(12, 7.5, 0.35).unwrap();
    assert!((log_pmf - (-3.295_629_258_780_280_2)).abs() < 2.0e-12);
    assert!((cdf - 0.844_479_919_949_704).abs() < 2.0e-12);
    assert!((sf - 0.155_520_080_050_296_02).abs() < 2.0e-12);
}

#[test]
fn negative_binomial_regression_matches_statsmodels() {
    let predictors = (0..16).map(|index| vec![index as f64]).collect::<Vec<_>>();
    let counts = [1, 0, 3, 1, 5, 2, 7, 3, 10, 4, 14, 6, 19, 8, 26, 11];
    let fit = try_negative_binomial_regression(
        &predictors,
        &counts,
        NegativeBinomialRegressionOptions {
            dispersion: 0.8,
            ridge: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((fit.coefficients[0] - 0.249_192_56).abs() < 2.0e-6);
    assert!((fit.coefficients[1] - 0.191_984_06).abs() < 2.0e-6);
    assert!((fit.deviance - 6.242_553_710_441_211).abs() < 2.0e-7);
}

#[test]
fn elastic_net_matches_standardized_sklearn_objective() {
    let predictors = (0..40)
        .map(|index| vec![index as f64 - 20.0, ((index * 17) % 11) as f64 - 5.0])
        .collect::<Vec<_>>();
    let outcomes = predictors
        .iter()
        .map(|row| 2.0 + 3.0 * row[0])
        .collect::<Vec<_>>();
    let fit = try_elastic_net_regression(
        &predictors,
        &outcomes,
        ElasticNetOptions {
            lambda: 0.2,
            l1_ratio: 1.0,
            tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((fit.coefficients[0] - 1.991_337_038_363_515_4).abs() < 2.0e-9);
    assert!((fit.coefficients[1] - 2.982_674_08).abs() < 2.0e-8);
    assert!(fit.coefficients[2].abs() < 1.0e-12);
}

#[test]
fn robust_ranks_and_paired_tests_match_reference_libraries() {
    let theil = try_theil_sen_regression(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[1.0, 3.0, 5.0, 7.0, 9.0, 500.0],
        0.8,
    )
    .unwrap();
    let kendall = try_kendall_tau_b(&[1.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 4.0]).unwrap();
    let mcnemar = try_mcnemar_test(1, 9).unwrap();
    let cochran = try_cochran_q(&[
        vec![false, false, true],
        vec![false, true, true],
        vec![false, false, true],
        vec![true, true, true],
        vec![false, true, true],
        vec![false, false, true],
    ])
    .unwrap();
    assert!((theil.slope - 2.0).abs() < 1.0e-12);
    assert!((theil.intercept - 1.0).abs() < 1.0e-12);
    assert!((kendall.tau_b - 0.8).abs() < 1.0e-12);
    assert!((mcnemar.p_value_exact_two_sided - 0.021_484_375).abs() < 1.0e-12);
    assert!((cochran.statistic - 7.6).abs() < 1.0e-12);
    assert!((cochran.p_value - 0.022_370_771_856_165_598).abs() < 2.0e-12);
}
