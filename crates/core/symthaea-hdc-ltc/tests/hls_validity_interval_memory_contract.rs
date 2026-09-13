// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    StateTrackingBenchmark, StateTrackingBenchmarkConfig, StateTrackingCodec,
    StateTrackingValidityArchive, TemporalAxis, UnitaryRole, ValidityIntervalMemory,
};

#[test]
fn public_validity_memory_respects_checkpoint_predecessor_boundaries() {
    let dim = 4096;
    let axis = TemporalAxis::new(dim, 900).unwrap();
    let key = UnitaryRole::new(dim, 901);
    let values = [
        UnitaryRole::new(dim, 910),
        UnitaryRole::new(dim, 911),
        UnitaryRole::new(dim, 912),
    ];
    let mut memory = ValidityIntervalMemory::new(dim).unwrap();
    memory.write_span(&axis, &key, &values[0], 0, 4).unwrap();
    memory.write_span(&axis, &key, &values[1], 4, 9).unwrap();
    memory.write_span(&axis, &key, &values[2], 9, 15).unwrap();

    for checkpoint in 0..15 {
        let expected = if checkpoint < 4 {
            0
        } else if checkpoint < 9 {
            1
        } else {
            2
        };
        let result = memory.cleanup(&axis, &key, &values, checkpoint).unwrap();
        assert_eq!(result.best_index, expected, "checkpoint={checkpoint}");
        assert!(result.margin > 0.0, "checkpoint={checkpoint}, result={result:?}");
    }
}

#[test]
fn public_archive_matches_seeded_historical_oracle_including_two_hop_queries() {
    let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 4,
        objects: 6,
        locations: 3,
        events: 48,
        query_every: 3,
        historical_query_rate: 1.0,
        seed: 920,
        ..Default::default()
    })
    .unwrap();
    let codec = StateTrackingCodec::from_benchmark_config(16_384, &benchmark.config, 921).unwrap();
    let archive = StateTrackingValidityArchive::build(&benchmark, &codec, 922).unwrap();
    let evaluation = archive.evaluate_historical(&benchmark, &codec).unwrap();

    assert_eq!(archive.spans_written(), benchmark.events.len());
    assert!(evaluation.total > 0);
    assert!(evaluation.object_location_total > 0);
    assert_eq!(evaluation.correct, evaluation.total);
    assert_eq!(evaluation.object_location_correct, evaluation.object_location_total);
    assert!(evaluation.smallest_margin > 0.0, "evaluation={evaluation:?}");
}
