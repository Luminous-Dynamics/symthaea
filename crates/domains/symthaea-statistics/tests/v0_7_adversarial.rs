use symthaea_statistics::{
    ElasticNetOptions, ImputationStrategy, NegativeBinomialRegressionOptions, StratumSample,
    try_audit_missing, try_beta_binomial_empirical_bayes, try_cumulative_incidence,
    try_elastic_net_regression, try_holt_linear_trend, try_impute_numeric,
    try_negative_binomial_regression, try_stratified_mean,
};

#[test]
fn elastic_net_rejects_constant_standardized_predictors() {
    let error = try_elastic_net_regression(
        &[vec![1.0], vec![1.0], vec![1.0]],
        &[1.0, 2.0, 3.0],
        ElasticNetOptions::default(),
    )
    .unwrap_err();
    assert!(format!("{error}").contains("varying predictors"));
}

#[test]
fn negative_binomial_regression_rejects_zero_events_and_dispersion() {
    assert!(
        try_negative_binomial_regression(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[0, 0, 0],
            NegativeBinomialRegressionOptions::default(),
        )
        .is_err()
    );
    assert!(
        try_negative_binomial_regression(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[1, 2, 3],
            NegativeBinomialRegressionOptions {
                dispersion: f64::NAN,
                ..Default::default()
            },
        )
        .is_err()
    );
}

#[test]
fn missing_data_never_silently_imputes_an_unobserved_column() {
    let table = vec![vec![Some(1.0), None], vec![Some(2.0), None]];
    let audit = try_audit_missing(&table).unwrap();
    assert_eq!(audit.missing_by_column, vec![0, 2]);
    assert!(try_impute_numeric(&table, ImputationStrategy::Mean).is_err());
}

#[test]
fn stratified_estimator_rejects_samples_larger_than_population() {
    let values = [1.0, 2.0, 3.0];
    assert!(
        try_stratified_mean(
            &[StratumSample {
                values: &values,
                population_size: 2,
            }],
            0.95,
        )
        .is_err()
    );
}

#[test]
fn cumulative_incidence_requires_a_real_target_event() {
    assert!(try_cumulative_incidence(&[1.0, 2.0], &[0, 2], 1).is_err());
    assert!(try_cumulative_incidence(&[1.0, 2.0], &[1, 2], 0).is_err());
}

#[test]
fn empirical_beta_prior_requires_successes_and_failures() {
    assert!(try_beta_binomial_empirical_bayes(&[0, 0], &[10, 20], 0.95).is_err());
    assert!(try_beta_binomial_empirical_bayes(&[10, 20], &[10, 20], 0.95).is_err());
}

#[test]
fn smoothing_rejects_invalid_parameters_and_nonfinite_series() {
    assert!(try_holt_linear_trend(&[1.0, 2.0, 3.0], 0.0, 0.5, 1.0).is_err());
    assert!(try_holt_linear_trend(&[1.0, f64::NAN, 3.0], 0.5, 0.5, 1.0).is_err());
}
