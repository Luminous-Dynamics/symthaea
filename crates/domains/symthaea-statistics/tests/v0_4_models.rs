// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_statistics::{
    Alternative, BernoulliSprt, BetaBinomialModel, BoundedMeanConfidenceSequence, Contingency2x2,
    GammaPoissonModel, HuberRegressionOptions, LogisticRegressionOptions, SeedSequence,
    SequentialDecision, try_audit_table, try_beta_cdf, try_beta_quantile,
    try_correlation_sample_size, try_gamma_cdf, try_gamma_quantile, try_huber_regression,
    try_information_criteria, try_likelihood_ratio_test, try_logistic_regression,
    try_multiple_linear_regression, try_odds_ratio_interval, try_standardized_mean_difference,
    try_two_sample_mean_sample_size,
};

#[test]
fn beta_and_gamma_quantiles_round_trip() {
    for &(probability, alpha, beta) in &[(0.1, 2.0, 5.0), (0.5, 4.0, 4.0), (0.9, 8.0, 2.0)] {
        let quantile = try_beta_quantile(probability, alpha, beta).unwrap();
        assert!((try_beta_cdf(quantile, alpha, beta).unwrap() - probability).abs() < 1e-10);
    }
    for &(probability, shape, rate) in &[(0.1, 0.8, 2.0), (0.5, 4.0, 1.5), (0.99, 10.0, 0.5)] {
        let quantile = try_gamma_quantile(probability, shape, rate).unwrap();
        assert!((try_gamma_cdf(quantile, shape, rate).unwrap() - probability).abs() < 1e-10);
    }
}

#[test]
fn multiple_models_share_prediction_contracts() {
    let predictors: Vec<Vec<f64>> = (0..12).map(|value| vec![value as f64]).collect();
    let mut outcomes: Vec<f64> = predictors.iter().map(|row| 2.0 + 3.0 * row[0]).collect();
    outcomes[11] += 50.0;

    let ordinary = try_multiple_linear_regression(&predictors, &outcomes, true, 0.95).unwrap();
    let robust =
        try_huber_regression(&predictors, &outcomes, HuberRegressionOptions::default()).unwrap();
    assert!((robust.coefficients[1] - 3.0).abs() < (ordinary.coefficients[1] - 3.0).abs());
    assert!(robust.observation_weights[11] < 1.0);
}

#[test]
fn logistic_model_exposes_likelihood_for_comparison() {
    let predictors = vec![
        vec![-2.0],
        vec![-1.5],
        vec![-1.0],
        vec![-0.5],
        vec![0.0],
        vec![0.5],
        vec![1.0],
        vec![1.5],
        vec![2.0],
        vec![2.5],
    ];
    let outcomes = [
        false, false, false, true, false, true, true, true, true, true,
    ];
    let fit = try_logistic_regression(
        &predictors,
        &outcomes,
        LogisticRegressionOptions {
            ridge: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fit.try_probability(&[-1.0]).unwrap() < fit.try_probability(&[1.0]).unwrap());
    let criteria =
        try_information_criteria(fit.log_likelihood, fit.coefficients.len(), fit.n).unwrap();
    assert!((criteria.aic - fit.aic).abs() < 1e-10);
}

#[test]
fn conjugate_models_accumulate_without_raw_data() {
    let mut prevalence = BetaBinomialModel::try_new(1.0, 1.0).unwrap();
    prevalence.try_update(30, 100).unwrap();
    prevalence.try_update(15, 50).unwrap();
    assert!((prevalence.mean() - 46.0 / 152.0).abs() < 1e-12);

    let mut rate = GammaPoissonModel::try_new(1.0, 1.0).unwrap();
    rate.try_update(20, 10.0).unwrap();
    assert!((rate.mean_rate() - 21.0 / 11.0).abs() < 1e-12);
}

#[test]
fn sequential_rules_stop_without_optional_peeking() {
    let mut sprt = BernoulliSprt::try_new(0.3, 0.7, 0.05, 0.05).unwrap();
    while sprt.decision() == SequentialDecision::Continue {
        sprt.try_observe(true).unwrap();
    }
    assert_eq!(sprt.decision(), SequentialDecision::AcceptAlternative);
    assert!(sprt.try_observe(true).is_err());

    let mut sequence = BoundedMeanConfidenceSequence::try_new(0.0, 1.0, 0.05).unwrap();
    for _ in 0..2_000 {
        sequence.try_observe(0.75).unwrap();
    }
    assert_eq!(
        sequence.try_decision_above(0.5).unwrap(),
        SequentialDecision::AcceptAlternative
    );
}

#[test]
fn power_designs_meet_declared_targets() {
    let mean_design =
        try_two_sample_mean_sample_size(0.5, 0.05, 0.8, Alternative::TwoSided).unwrap();
    assert!(mean_design.power >= 0.8);
    let correlation_design =
        try_correlation_sample_size(0.3, 0.05, 0.8, Alternative::TwoSided).unwrap();
    assert!(correlation_design.power >= 0.8);
}

#[test]
fn effect_and_likelihood_results_are_self_consistent() {
    let effect =
        try_standardized_mean_difference(&[4.0, 5.0, 6.0, 7.0], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(effect.cohens_d > effect.hedges_g);

    let ratio = try_odds_ratio_interval(Contingency2x2::new(12, 5, 3, 10).unwrap(), 0.95).unwrap();
    assert!(ratio.confidence_interval.contains(ratio.estimate));

    let likelihood = try_likelihood_ratio_test(-120.0, 2, -110.0, 4).unwrap();
    assert!(likelihood.p_value < 0.001);
}

#[test]
fn semantic_seeds_and_audits_are_stable() {
    let seeds = SeedSequence::from_label("experiment-alpha");
    assert_eq!(seeds.derive("bootstrap"), seeds.derive("bootstrap"));
    assert_ne!(seeds.derive("bootstrap"), seeds.derive("permutation"));

    let audit = try_audit_table(&[vec![1.0, 2.0], vec![1.0, f64::NAN], vec![1.0, 4.0]]).unwrap();
    assert_eq!(audit.complete_finite_rows, 2);
    assert_eq!(audit.constant_columns(), vec![0]);
}
