// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public API contract for the first frozen-reservoir state-tracking diagnostic.

use symthaea_hdc_ltc::{
    FrozenTrackingEvalConfig, HlsConfig, HolographicLiquidCell, StateTrackingBenchmark,
    StateTrackingBenchmarkConfig, evaluate_frozen_reservoir,
};

fn benchmark(seed: u64) -> StateTrackingBenchmark {
    StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 4,
        objects: 8,
        locations: 3,
        events: 128,
        query_every: 4,
        historical_query_rate: 0.75,
        seed,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap()
}

#[test]
fn held_out_world_evaluation_reports_query_only_control() {
    let train = benchmark(100);
    let test = benchmark(200);
    let reservoir = HolographicLiquidCell::try_new(
        HlsConfig {
            dim: 128,
            ..HlsConfig::default()
        },
        300,
    )
    .unwrap();

    let result = evaluate_frozen_reservoir(
        &reservoir,
        &train,
        &test,
        FrozenTrackingEvalConfig { codec_seed: 400 },
    )
    .unwrap();

    assert_eq!(result.score.total, test.queries.len());
    assert_eq!(result.query_only_score.total, test.queries.len());
    assert!((0.0..=1.0).contains(&result.score.accuracy()));
    assert!((0.0..=1.0).contains(&result.query_only_score.accuracy()));
    assert!(result.accuracy_gain_over_query_only().is_finite());
    assert!(result.compositional_gain_over_query_only().is_finite());
    assert!(result.historical_gain_over_query_only().is_finite());
}
