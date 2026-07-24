use symthaea_statistics::{
    BlockBootstrapMethod, DirichletMultinomialModel, KappaWeight, LocalLevelModel,
    MulticlassConfusion, try_block_bootstrap_mean, try_cronbach_alpha, try_dirichlet_log_pdf,
    try_generalized_pareto_quantile, try_generalized_pareto_sf, try_intraclass_correlations,
    try_local_level_filter, try_one_sample_hotelling, try_weighted_kappa,
};

#[test]
fn dirichlet_requires_an_open_normalized_simplex() {
    assert!(try_dirichlet_log_pdf(&[0.0, 1.0], &[1.0, 1.0]).is_err());
    assert!(try_dirichlet_log_pdf(&[0.2, 0.2], &[1.0, 1.0]).is_err());
    assert!(try_dirichlet_log_pdf(&[0.5, 0.5], &[0.0, 1.0]).is_err());
}

#[test]
fn dirichlet_multinomial_updates_are_transactional_on_shape_error() {
    let mut model = DirichletMultinomialModel::try_new(vec![1.0, 1.0, 1.0]).unwrap();
    assert!(model.try_observe_counts(&[1, 2]).is_err());
    assert_eq!(model.counts(), &[0, 0, 0]);
}

#[test]
fn multiclass_labels_are_range_checked() {
    assert!(MulticlassConfusion::try_from_labels(&[0, 3], &[0, 1], 3).is_err());
    assert!(MulticlassConfusion::try_new(2, vec![1.0, -1.0, 0.0, 1.0]).is_err());
}

#[test]
fn kappa_rejects_zero_expected_disagreement() {
    assert!(try_weighted_kappa(&[1, 1, 1], &[1, 1, 1], 3, KappaWeight::Quadratic).is_err());
}

#[test]
fn reliability_rejects_unidentified_variance_components() {
    let constant = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
    assert!(try_cronbach_alpha(&constant).is_err());
    assert!(try_intraclass_correlations(&constant).is_err());
}

#[test]
fn dependent_bootstrap_rejects_invalid_block_configuration() {
    assert!(
        try_block_bootstrap_mean(
            &[1.0, 2.0, 3.0],
            10,
            0.95,
            1,
            BlockBootstrapMethod::Moving { block_length: 0 },
        )
        .is_err()
    );
    assert!(
        try_block_bootstrap_mean(
            &[1.0, 2.0, 3.0],
            10,
            0.95,
            1,
            BlockBootstrapMethod::Stationary {
                expected_block_length: 0.5
            },
        )
        .is_err()
    );
}

#[test]
fn local_level_model_rejects_nonphysical_variances() {
    assert!(
        try_local_level_filter(
            &[1.0, 2.0],
            LocalLevelModel {
                process_variance: -1.0,
                observation_variance: 1.0,
                initial_mean: 0.0,
                initial_variance: 1.0,
            },
        )
        .is_err()
    );
    assert!(
        try_local_level_filter(
            &[1.0, 2.0],
            LocalLevelModel {
                process_variance: 1.0,
                observation_variance: 0.0,
                initial_mean: 0.0,
                initial_variance: 1.0,
            },
        )
        .is_err()
    );
}

#[test]
fn finite_endpoint_tail_is_exact() {
    assert_eq!(try_generalized_pareto_sf(4.0, -0.5, 2.0).unwrap(), 0.0);
    assert_eq!(
        try_generalized_pareto_quantile(1.0, -0.5, 2.0).unwrap(),
        4.0
    );
}

#[test]
fn hotelling_rejects_singular_covariance() {
    let collinear = vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
        vec![4.0, 8.0],
    ];
    assert!(try_one_sample_hotelling(&collinear, &[0.0, 0.0]).is_err());
}
