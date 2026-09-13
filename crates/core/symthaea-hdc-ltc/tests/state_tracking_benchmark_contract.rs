// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public API contract for the model-agnostic HLS state-tracking benchmark.

use symthaea_hdc_ltc::{
    StateTrackingBenchmark, StateTrackingBenchmarkConfig, TrackingEventKind, TrackingQueryKind,
};
use std::collections::HashSet;

#[test]
fn public_generator_is_reproducible_observable_and_contains_required_query_classes() {
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
    assert_eq!(a.initialization_events, 24);
    assert_eq!(a.events.len(), 24 + 512);

    let mut entities = HashSet::new();
    let mut objects = HashSet::new();
    for event in a.events.iter().take(a.initialization_events) {
        match event.kind {
            TrackingEventKind::MoveEntity { entity, .. } => {
                assert!(entities.insert(entity));
            }
            TrackingEventKind::TransferObject { object, .. } => {
                assert!(objects.insert(object));
            }
        }
    }
    assert_eq!(entities.len(), 8);
    assert_eq!(objects.len(), 16);

    let first_observed = a.first_fully_observed_event().unwrap();
    assert!(a
        .queries
        .iter()
        .all(|query| query.asked_after_event >= a.initialization_events));
    assert!(a
        .queries
        .iter()
        .all(|query| query.as_of_event >= first_observed));

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
