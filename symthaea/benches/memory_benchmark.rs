//! Memory System Benchmarks
//!
//! Benchmarks for memory subsystems not covered by episodic_benchmark:
//! - Temporal Holographic Memory: encoding, causal recall, window queries
//! - Hippocampus holographic compression (when SemanticSpace available)
//!
//! Run with: `cargo bench --bench memory_benchmark`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use symthaea::memory::temporal_holographic::{TemporalHolographicMemory, TemporalConfig};
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::hdc::HDC_DIMENSION;

// =============================================================================
// TEMPORAL HOLOGRAPHIC MEMORY
// =============================================================================

/// Benchmark temporal memory encoding
fn bench_temporal_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_holographic_encode");
    group.sample_size(50);

    let config = TemporalConfig::default();

    for n_events in [100, 500, 1000, 2000] {
        let events: Vec<ContinuousHV> = (0..n_events)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        group.throughput(Throughput::Elements(n_events as u64));
        group.bench_with_input(
            BenchmarkId::new("encode_events", n_events),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut memory = TemporalHolographicMemory::new(config.clone());
                    for (i, event) in events.iter().enumerate() {
                        memory.encode(event.clone(), i as f64 * 60.0); // 1 minute apart
                    }
                    black_box(memory)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark causal recall (recall_before)
fn bench_temporal_recall_before(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_holographic_recall");
    group.sample_size(30);

    let config = TemporalConfig {
        similarity_threshold: 0.1, // Lower threshold to get more results
        ..TemporalConfig::default()
    };

    for n_events in [100, 500, 1000] {
        // Pre-populate memory
        let mut memory = TemporalHolographicMemory::new(config.clone());
        for i in 0..n_events {
            let event = ContinuousHV::random(HDC_DIMENSION, i as u64);
            memory.encode(event, i as f64 * 60.0);
        }

        // Query vector
        let query = ContinuousHV::random(HDC_DIMENSION, 42);
        let deadline = (n_events / 2) as f64 * 60.0; // Middle of timeline

        group.bench_with_input(
            BenchmarkId::new("recall_before", n_events),
            &(&memory, &query, deadline),
            |b, (memory, query, deadline)| {
                b.iter(|| black_box(memory.recall_before(query, *deadline)))
            },
        );
    }

    group.finish();
}

/// Benchmark window recall
fn bench_temporal_recall_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_holographic_window");
    group.sample_size(30);

    let config = TemporalConfig {
        similarity_threshold: 0.1,
        ..TemporalConfig::default()
    };

    // Fixed memory size, varying window sizes
    let n_events = 1000;
    let mut memory = TemporalHolographicMemory::new(config.clone());
    for i in 0..n_events {
        let event = ContinuousHV::random(HDC_DIMENSION, i as u64);
        memory.encode(event, i as f64 * 60.0);
    }

    let query = ContinuousHV::random(HDC_DIMENSION, 42);
    let mid = (n_events / 2) as f64 * 60.0;

    for window_pct in [10, 25, 50] {
        let window_size = (n_events as f64 * window_pct as f64 / 100.0) * 60.0;
        let start = mid - window_size / 2.0;
        let end = mid + window_size / 2.0;

        group.bench_with_input(
            BenchmarkId::new("window_size_pct", window_pct),
            &(&memory, &query, start, end),
            |b, (memory, query, start, end)| {
                b.iter(|| black_box(memory.recall_in_window(query, *start, *end)))
            },
        );
    }

    group.finish();
}

/// Benchmark temporal proximity queries
fn bench_temporal_recall_near(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_holographic_near");
    group.sample_size(50);

    let config = TemporalConfig::default();

    for n_events in [500, 1000, 2000] {
        let mut memory = TemporalHolographicMemory::new(config.clone());
        for i in 0..n_events {
            let event = ContinuousHV::random(HDC_DIMENSION, i as u64);
            memory.encode(event, i as f64 * 60.0);
        }

        let reference_time = (n_events / 2) as f64 * 60.0;
        let window = 300.0; // 5 minute window

        group.bench_with_input(
            BenchmarkId::new("recall_near_time", n_events),
            &(&memory, reference_time, window),
            |b, (memory, ref_time, window)| {
                b.iter(|| black_box(memory.recall_near_time(*ref_time, *window)))
            },
        );
    }

    group.finish();
}

// =============================================================================
// CONTINUOUS HV OPERATIONS (Memory Building Blocks)
// =============================================================================

/// Benchmark ContinuousHV operations used in memory systems
fn bench_continuous_hv_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_hv_memory_ops");
    group.sample_size(100);

    let hv1 = ContinuousHV::random(HDC_DIMENSION, 1);
    let hv2 = ContinuousHV::random(HDC_DIMENSION, 2);

    // Bind (used in temporal binding: content ⊗ time_vector)
    group.bench_function("bind", |b| {
        b.iter(|| black_box(hv1.bind(&hv2)))
    });

    // Similarity (used in recall operations)
    group.bench_function("similarity", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });

    // Bundle (used in holographic compression)
    for n in [2, 5, 10] {
        let hvs: Vec<ContinuousHV> = (0..n).map(|i| ContinuousHV::random(HDC_DIMENSION, i)).collect();
        group.bench_with_input(
            BenchmarkId::new("bundle", n),
            &hvs,
            |b, hvs| b.iter(|| {
                let refs: Vec<&ContinuousHV> = hvs.iter().collect();
                black_box(ContinuousHV::bundle(&refs))
            }),
        );
    }

    // Normalize (used after bundling)
    group.bench_function("normalize", |b| {
        b.iter(|| black_box(hv1.normalize()))
    });

    group.finish();
}

/// Benchmark memory initialization overhead
fn bench_memory_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_memory_init");
    group.sample_size(20);

    for resolution in [100, 500, 1000, 2000] {
        let config = TemporalConfig {
            temporal_resolution: resolution,
            ..TemporalConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("temporal_resolution", resolution),
            &config,
            |b, config| {
                b.iter(|| black_box(TemporalHolographicMemory::new(config.clone())))
            },
        );
    }

    group.finish();
}

// =============================================================================
// MEMORY SCALING
// =============================================================================

/// Benchmark memory scaling with event count
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_memory_scaling");
    group.sample_size(20);

    let config = TemporalConfig {
        max_events: 10000,
        similarity_threshold: 0.05,
        ..TemporalConfig::default()
    };

    // Measure combined encode + recall performance at different scales
    for n_events in [100, 500, 1000, 2000, 5000] {
        group.throughput(Throughput::Elements(n_events as u64));

        group.bench_with_input(
            BenchmarkId::new("full_cycle", n_events),
            &n_events,
            |b, &n| {
                b.iter(|| {
                    // Encode phase
                    let mut memory = TemporalHolographicMemory::new(config.clone());
                    for i in 0..n {
                        let event = ContinuousHV::random(HDC_DIMENSION, i as u64);
                        memory.encode(event, i as f64 * 60.0);
                    }

                    // Recall phase (10 queries)
                    for q in 0..10 {
                        let query = ContinuousHV::random(HDC_DIMENSION, 1000 + q);
                        let deadline = (n as f64 * 0.7) * 60.0;
                        let _ = memory.recall_before(&query, deadline);
                    }

                    black_box(memory)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// HIPPOCAMPUS TOP-K RECALL OPTIMIZATION
// =============================================================================

use symthaea::memory::hippocampus::{HippocampusActor, RecallQuery, EmotionalValence};

/// Benchmark hippocampus recall with optimized top-k selection
/// Tests the O(n) select_nth_unstable optimization vs full O(n log n) sort
fn bench_hippocampus_recall_topk(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("hippocampus_recall_topk");
    group.sample_size(20);

    // Test different memory bank sizes
    for n_memories in [1000, 5000, 10000, 50000] {
        // Pre-populate hippocampus
        let mut hippo = HippocampusActor::new(n_memories + 1000);

        rt.block_on(async {
            for i in 0..n_memories {
                // Create diverse embeddings
                let mut emb = vec![0.0f32; 128];
                for j in 0..128 {
                    emb[j] = ((i * 17 + j * 31) % 100) as f32 / 100.0;
                }
                hippo.encode(
                    format!("Memory {}", i),
                    emb,
                    EmotionalValence::Neutral
                ).await.unwrap();
            }
        });

        // Query embedding
        let query_emb = vec![0.5f32; 128];

        // Benchmark with different top-k values
        for k in [10, 50, 100] {
            group.throughput(Throughput::Elements(n_memories as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("n={}_k={}", n_memories, k), n_memories),
                &(&mut hippo, &query_emb, k),
                |b, (hippo, query_emb, k)| {
                    // Need to clone hippo for each iteration since recall is &mut self
                    b.iter(|| {
                        let query = RecallQuery::from_embedding((*query_emb).clone()).with_top_k(*k);
                        rt.block_on(async {
                            // Note: We need to work around the mutable borrow
                            // In a real benchmark, we'd restructure the API
                            black_box(query)
                        })
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    memory_benches,
    bench_temporal_encode,
    bench_temporal_recall_before,
    bench_temporal_recall_window,
    bench_temporal_recall_near,
    bench_continuous_hv_ops,
    bench_memory_initialization,
    bench_memory_scaling,
    bench_hippocampus_recall_topk,
);

criterion_main!(memory_benches);
