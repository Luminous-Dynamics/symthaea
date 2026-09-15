// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    HlsConfig, HlsEligibilityTrace, HolographicLiquidCell, StateTrackingBenchmark,
    StateTrackingBenchmarkConfig, StateTrackingCodec, associative_query, step_with_eligibility,
};

#[test]
fn associative_loss_supplies_full_hls_parameter_gradient() {
    let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        entities: 5,
        objects: 8,
        locations: 4,
        events: 80,
        query_every: 4,
        historical_query_rate: 1.0,
        seed: 101,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap();
    let dim = 128;
    let codec = StateTrackingCodec::from_benchmark_config(dim, &benchmark.config, 202).unwrap();
    let mut cell = HolographicLiquidCell::try_new(
        HlsConfig {
            dim,
            state_norm_limit: f32::INFINITY,
            ..HlsConfig::default()
        },
        303,
    )
    .unwrap();
    let mut trace = HlsEligibilityTrace::zeros(dim);

    let first_query = &benchmark.queries[0];
    let mut previous_time = 0.0_f64;
    for event in benchmark
        .events
        .iter()
        .take(first_query.asked_after_event + 1)
    {
        let dt = event.time - previous_time;
        let input = codec.encode_event(event, dt).unwrap();
        step_with_eligibility(&mut cell, &mut trace, dt as f32, &input).unwrap();
        previous_time = event.time;
    }

    let asked_time = benchmark.events[first_query.asked_after_event].time;
    let readout = associative_query(&codec, cell.state(), first_query, asked_time, 1e-4).unwrap();
    let gradient = trace
        .parameter_gradient(&readout.state_learning_signal)
        .unwrap();

    assert!(readout.loss.is_finite());
    assert_eq!(gradient.scalar_count(), cell.parameter_count());
    assert!(gradient.l2_norm().is_finite());
}

#[test]
fn perfect_associative_memory_is_recovered_without_decoder_training() {
    let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
        events: 40,
        query_every: 4,
        seed: 9,
        ..StateTrackingBenchmarkConfig::default()
    })
    .unwrap();
    let codec = StateTrackingCodec::from_benchmark_config(256, &benchmark.config, 10).unwrap();
    let query = &benchmark.queries[0];
    let asked_time = benchmark.events[query.asked_after_event].time;
    let key = codec.query_key(query, asked_time).unwrap();
    let answer = codec.answer_symbol(query.expected).unwrap();
    let memory = key.bind(answer);

    let result = associative_query(&codec, &memory, query, asked_time, 1e-6).unwrap();
    assert_eq!(result.decoded, query.expected);
    assert!(result.loss < 1e-5);
    assert!(result.state_learning_signal.norm().is_finite());
}
