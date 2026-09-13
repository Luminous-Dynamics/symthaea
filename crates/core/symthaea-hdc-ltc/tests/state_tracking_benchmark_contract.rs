// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public API contract for the model-agnostic HLS state-tracking benchmark.

use symthaea_hdc_ltc::{
    StateTrackingBenchmark, StateTrackingBenchmarkConfig, TrackingQueryKind,
};

#[test]
fn public_generator_is_reproducible_and_contains_required_query_classes() {
    let config = StateTrackingBenchmarkConfig {
        entities: 8,
        objects: 16,
        locations: 4,
        events: 512,
        query_every: 8,
        historical_query_rate: 1.0,
        seed: 0xC0FFEE,
        ..StateTrackingBenchmarkConfig::default()
    };

    let a = StateTrackingBenchmark::generate(config.clone()).unwrap();
    let b = StateTrackingBenchmark::generate(config).unwrap();
    assert_eq!(a.events, b.events);
    assert_eq!(a.queries, b.queries);

    assert!(a.queries.iter().any(|query| query.is_historical()));
    assert!(a.queries.iter().any(|query| {
        matches!(query.kind, TrackingQueryKind::EntityLocation { .. })
    }));
    assert!(a.queries.iter().any(|query| {
        matches!(query.kind, TrackingQueryKind::ObjectOwner { .. })
    }));
    assert!(a.queries.iter().any(|query| {
        matches!(query.kind, TrackingQueryKind::ObjectLocation { .. })
    }));
}

#[test]
fn public_oracle_and_scorer_are_self_consistent() {
    let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        events: 256,
        query_every: 8,
        historical_query_rate: 1.0,
        seed: 123,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap();

    let predictions = benchmark
        .queries
        .iter()
        .map(|query| {
            benchmark
                .oracle_answer(query.as_of_event, &query.kind)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let score = benchmark.score(&predictions).unwrap();
    assert_eq!(score.accuracy(), 1.0);
    assert_eq!(score.current_accuracy(), 1.0);
    assert_eq!(score.historical_accuracy(), 1.0);
    assert_eq!(score.compositional_accuracy(), 1.0);
}
