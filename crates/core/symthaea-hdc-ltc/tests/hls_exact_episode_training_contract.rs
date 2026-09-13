// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ContinuousHV, CurrentOnlyTrainingError, ExactEpisodeTrainingConfig, HlsConfig,
    HolographicLiquidCell, StateTrackingBenchmark, StateTrackingBenchmarkConfig,
    StateTrackingCodec, evaluate_current_only_episode, train_current_only_episode,
};

fn world(seed: u64, historical_query_rate: f64) -> StateTrackingBenchmark {
    StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 4,
        objects: 6,
        locations: 3,
        events: 64,
        query_every: 4,
        historical_query_rate,
        seed,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap()
}

fn cell(dim: usize, seed: u64) -> HolographicLiquidCell {
    HolographicLiquidCell::try_new(
        HlsConfig {
            dim,
            state_norm_limit: f32::INFINITY,
            ..HlsConfig::default()
        },
        seed,
    )
    .unwrap()
}

#[test]
fn public_current_only_training_loop_keeps_evaluation_held_out_and_outcome_agnostic() {
    let train = world(1, 0.0);
    let test = world(2, 0.0);
    let dim = 96;
    let codec = StateTrackingCodec::from_benchmark_config(dim, &train.config, 77).unwrap();
    let mut cell = cell(dim, 88);

    let original_parameters = cell.parameters();
    let frozen = evaluate_current_only_episode(&cell, &test, &codec, 1e-4).unwrap();
    assert_eq!(cell.parameters(), original_parameters);

    let training = train_current_only_episode(
        &mut cell,
        &train,
        &codec,
        &ExactEpisodeTrainingConfig {
            learning_rate: 1e-3,
            gradient_norm_clip: 0.25,
            parameter_abs_bound: 2.0,
            loss_epsilon: 1e-4,
        },
    )
    .unwrap();
    assert!(training.pre_update.mean_loss.is_finite());
    assert!(training.clipped_gradient_norm <= 0.25001);
    assert_ne!(cell.parameters(), original_parameters);

    let trained_parameters = cell.parameters();
    let trained = evaluate_current_only_episode(&cell, &test, &codec, 1e-4).unwrap();
    assert_eq!(cell.parameters(), trained_parameters);
    assert!(frozen.mean_loss.is_finite());
    assert!(trained.mean_loss.is_finite());
    assert_eq!(frozen.query_count, trained.query_count);

    // Deliberately no assertion that training must improve the held-out result.
    // The sign and magnitude of that change are experimental evidence.
}

#[test]
fn historical_benchmark_fails_before_state_or_parameter_mutation() {
    let benchmark = world(3, 1.0);
    assert!(benchmark.queries.iter().all(|query| query.is_historical()));

    let dim = 96;
    let codec = StateTrackingCodec::from_benchmark_config(dim, &benchmark.config, 77).unwrap();
    let mut cell = cell(dim, 89);
    cell.set_state(ContinuousHV::new_random(dim, 90).scale(0.2))
        .unwrap();

    let state_before = cell.state().clone();
    let parameters_before = cell.parameters();
    let error = train_current_only_episode(
        &mut cell,
        &benchmark,
        &codec,
        &ExactEpisodeTrainingConfig::default(),
    )
    .unwrap_err();

    match error {
        CurrentOnlyTrainingError::HistoricalQueriesUnsupported { count } => {
            assert_eq!(count, benchmark.queries.len());
        }
        other => panic!("unexpected error: {other}"),
    }

    assert_eq!(cell.state(), &state_before);
    assert_eq!(cell.parameters(), parameters_before);
}
