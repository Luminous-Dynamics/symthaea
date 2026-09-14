// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use symthaea_hdc_ltc::{
    StateTrackingBenchmark, StateTrackingBenchmarkConfig, TrackingQuery, TrackingQueryKind,
};

fn benchmark(rate: f64) -> StateTrackingBenchmark {
    StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 4,
        objects: 6,
        locations: 3,
        events: 64,
        query_every: 4,
        historical_query_rate: rate,
        seed: 3,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap()
}

fn grouped_queries(benchmark: &StateTrackingBenchmark) -> BTreeMap<usize, Vec<&TrackingQuery>> {
    let mut grouped = BTreeMap::<usize, Vec<&TrackingQuery>>::new();
    for query in &benchmark.queries {
        grouped
            .entry(query.asked_after_event)
            .or_default()
            .push(query);
    }
    grouped
}

fn assert_three_current_base_queries(group: &[&TrackingQuery]) {
    let current = group
        .iter()
        .copied()
        .filter(|query| !query.is_historical())
        .collect::<Vec<_>>();
    assert_eq!(current.len(), 3);
    assert_eq!(
        current
            .iter()
            .filter(|query| matches!(query.kind, TrackingQueryKind::EntityLocation { .. }))
            .count(),
        1
    );
    assert_eq!(
        current
            .iter()
            .filter(|query| matches!(query.kind, TrackingQueryKind::ObjectOwner { .. }))
            .count(),
        1
    );
    assert_eq!(
        current
            .iter()
            .filter(|query| matches!(query.kind, TrackingQueryKind::ObjectLocation { .. }))
            .count(),
        1
    );
}

#[test]
fn zero_historical_rate_emits_only_three_current_queries_per_cadence() {
    let benchmark = benchmark(0.0);
    let grouped = grouped_queries(&benchmark);
    assert!(!grouped.is_empty());

    for group in grouped.values() {
        assert_eq!(group.len(), 3);
        assert_three_current_base_queries(group);
        assert!(group.iter().all(|query| !query.is_historical()));
    }

    assert_eq!(
        benchmark
            .queries
            .iter()
            .filter(|query| query.is_historical())
            .count(),
        0
    );
}

#[test]
fn unit_historical_rate_appends_one_historical_query_per_cadence() {
    let benchmark = benchmark(1.0);
    let grouped = grouped_queries(&benchmark);
    assert!(!grouped.is_empty());

    for group in grouped.values() {
        assert_eq!(group.len(), 4);
        assert_three_current_base_queries(group);
        let historical = group
            .iter()
            .copied()
            .filter(|query| query.is_historical())
            .collect::<Vec<_>>();
        assert_eq!(historical.len(), 1);
        assert!(matches!(
            historical[0].kind,
            TrackingQueryKind::ObjectLocation { .. }
        ));
        assert!(historical[0].as_of_event < historical[0].asked_after_event);
    }

    let historical_count = benchmark
        .queries
        .iter()
        .filter(|query| query.is_historical())
        .count();
    let current_count = benchmark.queries.len() - historical_count;
    assert_eq!(historical_count, grouped.len());
    assert_eq!(current_count, grouped.len() * 3);
    assert_eq!(benchmark.queries.len(), grouped.len() * 4);
    assert_eq!(historical_count * 4, benchmark.queries.len());
}
