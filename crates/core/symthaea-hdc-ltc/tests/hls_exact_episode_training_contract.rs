// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ExactEpisodeTrainingConfig, HlsConfig, HolographicLiquidCell, StateTrackingBenchmark,
    StateTrackingBenchmarkConfig, StateTrackingCodec, evaluate_associative_episode,
    train_exact_episode,
};

fn world(seed: u64) -> StateTrackingBenchmark {
    StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 4,
        objects: 6,
        locations: 3,
        events: 64,
        query_every: 4,
        historical_query_rate: 0.5,
        seed,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap()
}

#[test]
fn public_exact_training_loop_keeps_evaluation_held_out_and_outcome_agnostic() {
    let train = world(1);
    let test = world(2);
    let dim = 96;
    let codec = StateTrackingCodec::from_benchmark_config(dim, &train.config, 77).unwrap();
    let mut cell = HolographicLiquidCell::try_new(
        HlsConfig {
            dim,
            state_norm_limit: f32::INFINITY,
            ..HlsConfig::default()
        },
        88,
    )
    .unwrap();

    let original_parameters = cell.parameters();
    let frozen = evaluate_associative_episode(&cell, &test, &codec, 1e-4).unwrap();
    assert_eq!(cell.parameters(), original_parameters);

    let training = train_exact_episode(
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
    let trained = evaluate_associative_episode(&cell, &test, &codec, 1e-4).unwrap();
    assert_eq!(cell.parameters(), trained_parameters);
    assert!(frozen.mean_loss.is_finite());
    assert!(trained.mean_loss.is_finite());
    assert_eq!(frozen.query_count, trained.query_count);

    // Deliberately no assertion that training must improve the held-out result.
    // The sign and magnitude of that change are experimental evidence.
}
