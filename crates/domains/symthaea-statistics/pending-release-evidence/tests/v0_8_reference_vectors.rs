use symthaea_statistics::{
    BlockBootstrapMethod, DirichletMultinomialModel, KappaWeight, LocalLevelModel,
    MulticlassConfusion, try_bland_altman, try_circular_summary, try_concordance_correlation,
    try_cronbach_alpha, try_dirichlet_covariance, try_dirichlet_log_pdf, try_dirichlet_mean,
    try_generalized_pareto_cdf, try_generalized_pareto_log_pdf, try_generalized_pareto_quantile,
    try_generalized_pareto_sf, try_hill_estimator, try_intraclass_correlations,
    try_local_level_filter, try_local_level_smoother, try_multinomial_log_pmf,
    try_one_sample_hotelling, try_rayleigh_test, try_standardized_cronbach_alpha,
    try_two_sample_hotelling, try_weighted_kappa,
};

#[test]
fn categorical_probability_matches_scipy() {
    let point = [0.2, 0.3, 0.5];
    let concentration = [2.0, 3.0, 5.0];
    let log_pdf = try_dirichlet_log_pdf(&point, &concentration).unwrap();
    assert!((log_pdf - 2.140_654_225_847_825_4).abs() < 5.0e-12);
    assert_eq!(try_dirichlet_mean(&concentration).unwrap(), point);
    let covariance = try_dirichlet_covariance(&concentration).unwrap();
    assert!((covariance.try_get(0, 0).unwrap() - 0.014_545_454_545_454_545).abs() < 1.0e-15);
    assert!((covariance.try_get(1, 2).unwrap() - (-0.013_636_363_636_363_636)).abs() < 1.0e-15);

    let multinomial = try_multinomial_log_pmf(&[3, 2, 1], &[0.5, 0.3, 0.2]).unwrap();
    assert!((multinomial - (-2.002_480_500_543_706_3)).abs() < 5.0e-12);

    let mut model = DirichletMultinomialModel::try_new(vec![1.0, 2.0, 3.0]).unwrap();
    model.try_observe_counts(&[2, 1, 0]).unwrap();
    let log_mass = model.try_log_predictive_mass(&[1, 2, 0]).unwrap();
    assert!((log_mass - (-2.215_573_716_004_416_7)).abs() < 5.0e-12);
}

#[test]
fn multiclass_metrics_match_sklearn() {
    let actual = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2];
    let predicted = [0, 0, 1, 1, 1, 0, 2, 2, 1, 2];
    let matrix = MulticlassConfusion::try_from_labels(&actual, &predicted, 3).unwrap();
    assert!((matrix.accuracy().unwrap() - 0.7).abs() < 1.0e-15);
    assert!((matrix.macro_precision().unwrap() - 0.722_222_222_222_222_2).abs() < 1.0e-15);
    assert!((matrix.macro_recall().unwrap() - 0.694_444_444_444_444_3).abs() < 1.0e-15);
    assert!((matrix.macro_f1().unwrap() - 0.698_412_698_412_698_5).abs() < 1.0e-15);
    assert!((matrix.cohen_kappa().unwrap() - 0.552_238_805_970_149_3).abs() < 1.0e-15);
}

#[test]
fn agreement_and_reliability_match_reference_calculations() {
    let left = [0, 0, 1, 1, 2, 2, 3, 3];
    let right = [0, 1, 1, 2, 2, 3, 3, 3];
    assert!(
        (try_weighted_kappa(&left, &right, 4, KappaWeight::Unweighted)
            .unwrap()
            .kappa
            - 0.5)
            .abs()
            < 1.0e-15
    );
    assert!(
        (try_weighted_kappa(&left, &right, 4, KappaWeight::Linear)
            .unwrap()
            .kappa
            - 0.7)
            .abs()
            < 1.0e-15
    );
    assert!(
        (try_weighted_kappa(&left, &right, 4, KappaWeight::Quadratic)
            .unwrap()
            .kappa
            - 0.85)
            .abs()
            < 1.0e-15
    );

    let bland = try_bland_altman(
        &[10.1, 11.7, 9.8, 13.2, 12.5],
        &[9.9, 11.4, 10.0, 12.8, 12.2],
        0.95,
    )
    .unwrap();
    assert!((bland.bias - 0.199_999_999_999_999_65).abs() < 1.0e-15);
    assert!((bland.difference_standard_deviation - 0.234_520_787_991_170_82).abs() < 1.0e-15);
    assert!((bland.limits_of_agreement.low - (-0.259_652_298_088_648_76)).abs() < 2.0e-7);
    assert!((bland.limits_of_agreement.high - 0.659_652_298_088_648_1).abs() < 2.0e-7);

    let concordance =
        try_concordance_correlation(&[1.0, 2.0, 3.0, 4.0, 6.0], &[1.1, 1.8, 3.2, 3.9, 5.5])
            .unwrap();
    assert!((concordance - 0.987_012_987_012_987_1).abs() < 1.0e-15);

    let reliability = vec![
        vec![1.0, 2.0, 1.0],
        vec![2.0, 3.0, 2.0],
        vec![3.0, 4.0, 4.0],
        vec![4.0, 5.0, 4.0],
        vec![5.0, 6.0, 5.0],
    ];
    assert!((try_cronbach_alpha(&reliability).unwrap() - 0.991_189_427_312_775_4).abs() < 1.0e-15);
    assert!(
        (try_standardized_cronbach_alpha(&reliability).unwrap() - 0.991_468_065_613_079_3).abs()
            < 1.0e-15
    );

    let ratings = vec![
        vec![9.0, 8.0, 9.0],
        vec![6.0, 7.0, 6.0],
        vec![8.0, 8.0, 7.0],
        vec![5.0, 6.0, 5.0],
        vec![7.0, 7.0, 8.0],
    ];
    let icc = try_intraclass_correlations(&ratings).unwrap();
    assert!((icc.one_way_random - 0.820_359_281_437_125_9).abs() < 1.0e-15);
    assert!((icc.two_way_random_agreement - 0.818_181_818_181_818_1).abs() < 1.0e-15);
    assert!((icc.two_way_mixed_consistency - 0.789_473_684_210_526_2).abs() < 1.0e-15);
}

#[test]
fn circular_statistics_match_reference_values() {
    let angles = [350.0_f64, 5.0, 10.0, 15.0, 355.0].map(f64::to_radians);
    let summary = try_circular_summary(&angles).unwrap();
    assert!((summary.mean_direction - 0.052_472_624_648_468_29).abs() < 1.0e-15);
    assert!((summary.mean_resultant_length - 0.986_944_548_857_179).abs() < 1.0e-15);
    assert!((summary.circular_standard_deviation - 0.162_119_848_440_878_96).abs() < 1.0e-15);
    let rayleigh = try_rayleigh_test(&angles).unwrap();
    assert!((rayleigh.statistic - 4.870_297_712_594_502).abs() < 1.0e-14);
    assert!((rayleigh.p_value - 0.000_437_483_907_960_549_9).abs() < 1.0e-15);
}

#[test]
fn local_level_filter_matches_independent_kalman_recursion() {
    let model = LocalLevelModel {
        process_variance: 0.2,
        observation_variance: 0.5,
        initial_mean: 0.0,
        initial_variance: 2.0,
    };
    let filter = try_local_level_filter(&[1.0, 1.4, 0.9, 1.8], model).unwrap();
    assert!((filter.log_likelihood - (-4.853_790_667_756_668)).abs() < 1.0e-14);
    let last = filter.steps.last().unwrap();
    assert!((last.filtered_mean - 1.384_737_363_726_461_7).abs() < 1.0e-15);
    assert!((last.filtered_variance - 0.234_886_025_768_087_2).abs() < 1.0e-15);
    let smoother = try_local_level_smoother(&[1.0, 1.4, 0.9, 1.8], model).unwrap();
    let expected = [
        1.053_320_118_929_633_3,
        1.179_980_178_394_45,
        1.218_632_309_236_868_3,
        1.384_737_363_726_461_7,
    ];
    for (&actual, &reference) in smoother.smoothed_means.iter().zip(&expected) {
        assert!((actual - reference).abs() < 2.0e-15);
    }
}

#[test]
fn extreme_value_functions_match_scipy() {
    for (shape, log_pdf, cdf, sf, quantile) in [
        (
            0.2,
            -1.531_718_834_810_897_4,
            0.502_823_264_701_710_3,
            0.497_176_735_298_289_7,
            5.848_931_924_611_135_5,
        ),
        (
            0.0,
            -1.443_147_180_559_945_4,
            0.527_633_447_258_985_3,
            0.472_366_552_741_014_7,
            4.605_170_185_988_092,
        ),
        (
            -0.2,
            -1.343_222_898_551_045,
            0.556_294_687_5,
            0.443_705_312_499_999_95,
            3.690_426_555_198_068,
        ),
    ] {
        assert!(
            (try_generalized_pareto_log_pdf(1.5, shape, 2.0).unwrap() - log_pdf).abs() < 1.0e-14
        );
        assert!((try_generalized_pareto_cdf(1.5, shape, 2.0).unwrap() - cdf).abs() < 1.0e-14);
        assert!((try_generalized_pareto_sf(1.5, shape, 2.0).unwrap() - sf).abs() < 1.0e-14);
        assert!(
            (try_generalized_pareto_quantile(0.9, shape, 2.0).unwrap() - quantile).abs() < 1.0e-13
        );
    }
    let hill = try_hill_estimator(&[1.0, 1.3, 1.7, 2.1, 3.0, 4.5, 7.0, 11.0], 3).unwrap();
    assert!((hill.tail_index - 0.850_681_984_208_542_9).abs() < 1.0e-15);
    assert!((hill.pareto_exponent - 1.175_527_422_189_832_2).abs() < 1.0e-15);
}

#[test]
fn hotelling_tests_match_numpy_and_scipy() {
    let left = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![4.0, 5.0],
        vec![5.0, 4.0],
        vec![3.0, 3.0],
    ];
    let one = try_one_sample_hotelling(&left, &[0.0, 0.0]).unwrap();
    assert!((one.t_squared - 20.0).abs() < 1.0e-12);
    assert!((one.f_statistic - 7.5).abs() < 1.0e-12);
    assert!((one.p_value - 0.068_041_381_743_977_17).abs() < 2.0e-12);

    let right = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![2.0, 2.0],
        vec![1.0, 2.0],
        vec![2.0, 1.0],
    ];
    let two = try_two_sample_hotelling(&left, &right).unwrap();
    assert!((two.t_squared - 6.0).abs() < 1.0e-12);
    assert!((two.f_statistic - 2.625).abs() < 1.0e-12);
    assert!((two.p_value - 0.141_047_966_604_026_5).abs() < 2.0e-12);
}

#[test]
fn block_bootstrap_scheme_is_represented_in_results() {
    let method = BlockBootstrapMethod::Stationary {
        expected_block_length: 3.0,
    };
    let result =
        symthaea_statistics::try_block_bootstrap_mean(&[1.0, 2.0, 3.0, 4.0], 32, 0.9, 42, method)
            .unwrap();
    assert_eq!(result.method, method);
    assert_eq!(result.replicates.len(), 32);
}
