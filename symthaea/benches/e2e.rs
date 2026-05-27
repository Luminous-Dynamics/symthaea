// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Performance Benchmarks
//!
//! Measures complete pipeline latency from input to output:
//! - Input processing -> HDC encoding -> LTC step -> Φ calculation -> Output
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench e2e
//! cargo bench --bench e2e -- full_pipeline
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::ConnectivityCalculator, unified_hv::ContinuousHV,
};

// =============================================================================
// FULL PIPELINE BENCHMARK
// =============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_full_pipeline");
    group.sample_size(20);

    // Target: <200ms end-to-end
    group.bench_function("complete_cycle", |b| {
        let calc = ConnectivityCalculator::new();

        b.iter(|| {
            // 1. Input: Create random input vector (simulates text embedding)
            let input = ContinuousHV::random(HDC_DIMENSION, 42);

            // 2. HDC Encoding: Create topology representation
            let topo = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);

            // 3. LTC Step: Would process through neural network
            // (Using topology representations as proxy)
            let processed = &topo.node_representations;

            // 4. Φ Calculation: Compute integrated information
            let phi_result = calc.algebraic_connectivity(processed);

            // 5. Output: Return result
            black_box((input, phi_result))
        })
    });

    group.finish();
}

// =============================================================================
// CONCURRENT Φ CALCULATIONS
// =============================================================================

fn bench_concurrent_phi(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_concurrent_phi");
    group.sample_size(10);

    for n_concurrent in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*n_concurrent as u64));

        group.bench_with_input(
            BenchmarkId::new("concurrent_phi", n_concurrent),
            n_concurrent,
            |b, &n| {
                let calc = ConnectivityCalculator::new();

                // Pre-generate topologies
                let topologies: Vec<_> = (0..n)
                    .map(|i| ConsciousnessTopology::ring(8, HDC_DIMENSION, i as u64))
                    .collect();

                b.iter(|| {
                    // Compute Φ for all topologies
                    let results: Vec<_> = topologies
                        .iter()
                        .map(|topo| calc.algebraic_connectivity(&topo.node_representations))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// LATENCY BREAKDOWN
// =============================================================================

fn bench_latency_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_latency_breakdown");
    group.sample_size(30);

    // Measure each component separately
    group.bench_function("hdc_encode", |b| {
        b.iter(|| {
            let hv = ContinuousHV::random(HDC_DIMENSION, 42);
            black_box(hv)
        })
    });

    group.bench_function("topology_create", |b| {
        b.iter(|| {
            let topo = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
            black_box(topo)
        })
    });

    group.bench_function("phi_compute_8node", |b| {
        let calc = ConnectivityCalculator::new();
        let topo = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
        b.iter(|| {
            let result = calc.algebraic_connectivity(&topo.node_representations);
            black_box(result)
        })
    });

    group.bench_function("phi_compute_16node", |b| {
        let calc = ConnectivityCalculator::new();
        let topo = ConsciousnessTopology::ring(16, HDC_DIMENSION, 42);
        b.iter(|| {
            let result = calc.algebraic_connectivity(&topo.node_representations);
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// PARTNERSHIP Φ_DYAD BENCHMARK
// =============================================================================

fn bench_phi_dyad(c: &mut Criterion) {
    use symthaea::hdc::ContinuousHV;
    use symthaea::hdc::relational_consciousness::{
        RelationMode, RelationalAssessment, RelationshipStage,
    };
    use symthaea::partnership::{DyadInput, DyadWeights, HumanPartnerModel, PhiDyadCalculator};

    let mut group = c.benchmark_group("e2e_phi_dyad");
    group.sample_size(20);

    group.bench_function("compute_phi_dyad", |b| {
        let ai_states: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, 1_000 + i))
            .collect();
        let human_states: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, 2_000 + i))
            .collect();

        let assessment = RelationalAssessment {
            agent_a: "ai".to_string(),
            agent_b: "human".to_string(),
            phi_relation: 0.4,
            stage: RelationshipStage::Attunement,
            synchrony: 0.6,
            turn_taking_quality: 0.7,
            mutual_information: 0.5,
            mode: RelationMode::IThou,
            num_interactions: 10,
            relationship_age: 100.0,
            explanation: "benchmark".to_string(),
        };
        let human_model = HumanPartnerModel::new("human");
        let calc = PhiDyadCalculator::new();
        let weights = DyadWeights::default();

        b.iter(|| {
            let input = DyadInput {
                ai_states: &ai_states,
                human_states: &human_states,
                relational: &assessment,
                human_model: &human_model,
                weights,
            };
            let result = calc.compute(&input);
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    e2e_benches,
    bench_full_pipeline,
    bench_concurrent_phi,
    bench_latency_breakdown,
    bench_phi_dyad,
);

criterion_main!(e2e_benches);
