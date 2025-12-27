//! Full System Benchmarks for Symthaea-HLB
//!
//! Comprehensive benchmarks to validate performance claims and
//! establish baseline metrics for all consciousness components.
//!
//! ## Benchmark Categories
//!
//! 1. **HDC Operations** - HV16 bind, bundle, similarity, search
//! 2. **LTC Dynamics** - Network step timing, sparse operations
//! 3. **Φ Computation** - Integrated information measurement
//! 4. **Temporal Reasoning** - Allen algebra operations
//! 5. **End-to-End Pipeline** - Complete query processing
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench                              # Run all benchmarks
//! cargo bench -- hv16                      # Run HV16 benchmarks only
//! cargo bench -- consciousness_benchmarks  # Run this benchmark file
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use symthaea::hdc::{HV16, simd_hv16::SimdHV16};
use symthaea::consciousness::{
    hierarchical_ltc::{HierarchicalLTC, HierarchicalConfig},
    temporal_primitives::{TemporalReasoner, TemporalInterval, AllenRelation, TemporalConfig},
};

// ============================================================================
// HDC BENCHMARKS
// ============================================================================

fn bench_hv16_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16_Operations");

    // Setup
    let hv1 = HV16::random(42);
    let hv2 = HV16::random(43);
    let hvs: Vec<HV16> = (0..100).map(|i| HV16::random(i)).collect();

    // Bind operation
    group.bench_function("bind", |b| {
        b.iter(|| black_box(hv1.bind(&hv2)))
    });

    // Bundle operation (2 vectors)
    group.bench_function("bundle_2", |b| {
        b.iter(|| black_box(HV16::bundle(&[hv1.clone(), hv2.clone()])))
    });

    // Bundle operation (10 vectors)
    let hvs_10: Vec<HV16> = hvs.iter().take(10).cloned().collect();
    group.bench_function("bundle_10", |b| {
        b.iter(|| black_box(HV16::bundle(&hvs_10)))
    });

    // Similarity
    group.bench_function("similarity", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });

    // Permute
    group.bench_function("permute_100", |b| {
        b.iter(|| black_box(hv1.permute(100)))
    });

    // Popcount
    group.bench_function("popcount", |b| {
        b.iter(|| black_box(hv1.popcount()))
    });

    group.finish();
}

fn bench_simd_hv16_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdHV16_Operations");

    // Setup
    let hv1 = SimdHV16::random(42);
    let hv2 = SimdHV16::random(43);
    let hvs: Vec<SimdHV16> = (0..100).map(|i| SimdHV16::random(i)).collect();

    // Bind operation
    group.bench_function("bind", |b| {
        b.iter(|| black_box(hv1.bind(&hv2)))
    });

    // Bundle operation (2 vectors)
    group.bench_function("bundle_2", |b| {
        b.iter(|| black_box(SimdHV16::bundle(&[hv1.clone(), hv2.clone()])))
    });

    // Bundle operation (10 vectors)
    let hvs_10: Vec<SimdHV16> = hvs.iter().take(10).cloned().collect();
    group.bench_function("bundle_10", |b| {
        b.iter(|| black_box(SimdHV16::bundle(&hvs_10)))
    });

    // Similarity
    group.bench_function("similarity", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });

    // Top-k search
    group.bench_function("top_k_10_in_100", |b| {
        b.iter(|| black_box(symthaea::hdc::simd_hv16::top_k_similar(&hv1, &hvs, 10)))
    });

    // Batch similarity
    group.bench_function("batch_similarity_100", |b| {
        b.iter(|| black_box(symthaea::hdc::simd_hv16::batch_similarity(&hv1, &hvs)))
    });

    group.finish();
}

// ============================================================================
// LTC BENCHMARKS
// ============================================================================

fn bench_hierarchical_ltc(c: &mut Criterion) {
    let mut group = c.benchmark_group("HierarchicalLTC");
    group.sample_size(50); // LTC operations can be slow

    // Small network (256 neurons)
    let config_small = HierarchicalConfig {
        num_circuits: 4,
        circuit_size: 64,
        global_size: 64,
        local_sparsity: 0.15,
        inter_sparsity: 0.1,
        dt: 0.01,
        parallel: true,
    };

    // Medium network (1024 neurons - default)
    let config_medium = HierarchicalConfig::default();

    // Large network (4096 neurons)
    let config_large = HierarchicalConfig {
        num_circuits: 64,
        circuit_size: 64,
        global_size: 256,
        ..HierarchicalConfig::default()
    };

    // Benchmark different network sizes
    for (name, config) in [
        ("small_256", config_small),
        ("medium_1024", config_medium.clone()),
        ("large_4096", config_large),
    ] {
        let mut ltc = HierarchicalLTC::new(config).expect("Failed to create HierarchicalLTC");
        group.bench_function(BenchmarkId::new("step", name), |b| {
            b.iter(|| {
                black_box(ltc.step().unwrap())
            })
        });
    }

    // Benchmark consciousness metrics extraction
    let mut ltc = HierarchicalLTC::new(config_medium).expect("Failed to create HierarchicalLTC");
    ltc.step().unwrap(); // Initialize state

    group.bench_function("estimate_phi", |b| {
        b.iter(|| black_box(ltc.estimate_phi()))
    });

    group.bench_function("workspace_access", |b| {
        b.iter(|| black_box(ltc.workspace_access()))
    });

    group.bench_function("binding_coherence", |b| {
        b.iter(|| black_box(ltc.binding_coherence()))
    });

    group.finish();
}

// ============================================================================
// TEMPORAL REASONING BENCHMARKS
// ============================================================================

fn bench_temporal_reasoning(c: &mut Criterion) {
    let mut group = c.benchmark_group("TemporalReasoning");

    // Setup reasoner
    let config = TemporalConfig::default();
    let reasoner = TemporalReasoner::new(config.clone());

    // Create intervals (named interval_a/interval_b to avoid conflict with benchmark closure var)
    let interval_a = TemporalInterval::new("A", 0.0, 1.0).unwrap();
    let interval_b = TemporalInterval::new("B", 1.5, 2.5).unwrap();

    // Benchmark relation computation
    group.bench_function("compute_relation", |bencher| {
        bencher.iter(|| black_box(reasoner.compute_relation(&interval_a, &interval_b)))
    });

    // Benchmark composition
    group.bench_function("compose_relations", |bencher| {
        bencher.iter(|| black_box(reasoner.compose(AllenRelation::Precedes, AllenRelation::Precedes)))
    });

    // Benchmark encoding
    group.bench_function("encode_relation", |bencher| {
        bencher.iter(|| black_box(reasoner.encode_relation(AllenRelation::Overlaps)))
    });

    // Benchmark statement encoding
    group.bench_function("encode_statement", |bencher| {
        bencher.iter(|| black_box(reasoner.encode_statement(&interval_a, AllenRelation::Precedes, &interval_b)))
    });

    // Benchmark with many intervals
    let mut large_reasoner = TemporalReasoner::new(config);
    for i in 0..100 {
        let start = i as f64 * 1.5;
        let interval = TemporalInterval::new(format!("interval_{}", i), start, start + 1.0).unwrap();
        large_reasoner.add_interval(interval);
    }

    group.bench_function("binding_window_search_100", |b| {
        b.iter(|| black_box(large_reasoner.find_binding_candidates(50.0)))
    });

    group.finish();
}

// ============================================================================
// INTEGRATED INFORMATION (Φ) BENCHMARKS
// ============================================================================

fn bench_phi_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Phi_Computation");

    use symthaea::hdc::integrated_information::IntegratedInformation;

    let mut phi_computer = IntegratedInformation::new();

    // Small system (2 components)
    let components_2: Vec<HV16> = (0..2).map(|i| HV16::random(i)).collect();
    group.bench_function("phi_2_components", |b| {
        b.iter(|| black_box(phi_computer.compute_phi(&components_2)))
    });

    // Medium system (5 components)
    let components_5: Vec<HV16> = (0..5).map(|i| HV16::random(i)).collect();
    group.bench_function("phi_5_components", |b| {
        b.iter(|| black_box(phi_computer.compute_phi(&components_5)))
    });

    // Large system (10 components)
    let components_10: Vec<HV16> = (0..10).map(|i| HV16::random(i)).collect();
    group.bench_function("phi_10_components", |b| {
        b.iter(|| black_box(phi_computer.compute_phi(&components_10)))
    });

    group.finish();
}

// ============================================================================
// END-TO-END PIPELINE BENCHMARKS
// ============================================================================

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("EndToEnd_Pipeline");
    group.sample_size(20); // Full pipeline is slow

    // Setup LTC component
    let mut ltc = HierarchicalLTC::new(HierarchicalConfig::default())
        .expect("Failed to create HierarchicalLTC");

    // Simulate full pipeline: Input → HDC → LTC → Φ
    group.bench_function("input_to_consciousness", |b| {
        b.iter(|| {
            // 1. Create input HDC representation
            let concepts: Vec<HV16> = (0..5).map(|i| HV16::random(i)).collect();
            let _bound = concepts.iter().skip(1).fold(concepts[0].clone(), |acc, hv| acc.bind(hv));

            // 2. Step LTC
            let _ = ltc.step();

            // 3. Extract consciousness metrics
            let phi = ltc.estimate_phi();
            let binding = ltc.binding_coherence();
            let workspace = ltc.workspace_access();

            black_box((phi, binding, workspace))
        })
    });

    // Throughput benchmark (how many queries per second)
    group.throughput(Throughput::Elements(1));
    group.bench_function("queries_per_second", |b| {
        b.iter(|| {
            let concepts: Vec<HV16> = (0..3).map(|i| HV16::random(i)).collect();
            let _ = ltc.step();
            black_box(concepts[0].similarity(&concepts[1]))
        })
    });

    group.finish();
}

// ============================================================================
// SCALABILITY BENCHMARKS
// ============================================================================

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalability");
    group.sample_size(20);

    // HV16 search scalability
    for size in [100, 1000, 10000] {
        let query = SimdHV16::random(42);
        let corpus: Vec<SimdHV16> = (0..size).map(|i| SimdHV16::random(i as u64)).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_top_10_search", size),
            &corpus,
            |b, corpus| {
                b.iter(|| black_box(symthaea::hdc::simd_hv16::top_k_similar(&query, corpus, 10)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// CRITERION MAIN
// ============================================================================

criterion_group!(
    hdc_benches,
    bench_hv16_operations,
    bench_simd_hv16_operations,
);

criterion_group!(
    consciousness_benches,
    bench_hierarchical_ltc,
    bench_phi_computation,
);

criterion_group!(
    temporal_benches,
    bench_temporal_reasoning,
);

criterion_group!(
    system_benches,
    bench_end_to_end,
    bench_scalability,
);

criterion_main!(
    hdc_benches,
    consciousness_benches,
    temporal_benches,
    system_benches,
);
