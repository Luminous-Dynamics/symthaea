//! Benchmark: Episodic Memory Optimizations
//!
//! Compares original O(n²+) operations with revolutionary O(n log n) optimized versions.
//!
//! Run with: `cargo bench --bench episodic_benchmark`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

use symthaea::memory::{
    EpisodicTrace,
    optimized_episodic::{
        consolidate_optimized,
        consolidate_with_decay,
        discover_coactivation_optimized,
        reconstruct_causal_chain_optimized,
        batch_retrieve,
        TemporalIndex,
    },
};
use symthaea::memory::episodic_engine::RetrievalEvent;

/// Create test EpisodicTrace with realistic data
fn create_test_trace(id: u64, strength: f32, timestamp_secs: u64) -> EpisodicTrace {
    EpisodicTrace {
        id,
        timestamp: Duration::from_secs(timestamp_secs),
        content: format!("Test memory {} with some realistic content about system operations and consciousness", id),
        tags: vec!["test".to_string(), "benchmark".to_string(), format!("tag_{}", id % 10)],
        emotion: (id as f32 / 100.0).sin(),
        chrono_semantic_vector: vec![0i8; 256],
        emotional_binding_vector: vec![0i8; 256],
        temporal_vector: (0..256).map(|i| (i as f32 * id as f32 / 256.0).cos()).collect(),
        semantic_vector: (0..256).map(|i| (i as f32 * id as f32 / 256.0).sin()).collect(),
        recall_count: (id % 10) as usize,
        strength,
        attention_weight: 0.5 + (id as f32 / 200.0).sin() * 0.3,
        encoding_strength: 10,
        retrieval_history: (0..5).map(|i| RetrievalEvent {
            retrieved_at: Duration::from_secs(timestamp_secs + i * 60),
            query_context: format!("Query {}", i),
            retrieval_method: "benchmark".to_string(),
            retrieval_strength: 0.8,
            content_matched: true,
        }).collect(),
        reliability_score: 0.9,
        has_drifted: false,
        last_modified: Duration::from_secs(timestamp_secs),
        // Week 17 Day 7: Intent-aware fields
        intent: None,
        goal_id: None,
        goal_progress_contribution: 0.0,
        is_goal_completion: false,
    }
}

/// Benchmark consolidation: O(n²) original vs O(n log n) optimized
fn bench_consolidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Consolidation Comparison");

    for size in [100, 250, 500, 1000] {
        // Create buffer with varying strengths
        let buffer: Vec<EpisodicTrace> = (0..size)
            .map(|i| create_test_trace(i as u64, i as f32 / size as f32, i as u64 * 10))
            .collect();

        let excess = size / 5; // Remove 20% of memories
        let target_size = size - excess;

        // Benchmark ORIGINAL O(n²) approach
        group.bench_with_input(
            BenchmarkId::new("Original_O(n²)", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    let mut buf = buffer.clone();
                    // Simulate original while-loop approach
                    while buf.len() > target_size {
                        if let Some((idx, _)) = buf.iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
                        {
                            buf.remove(idx);
                        } else {
                            break;
                        }
                    }
                    buf
                })
            },
        );

        // Benchmark OPTIMIZED O(n log n) approach
        group.bench_with_input(
            BenchmarkId::new("Optimized_O(nlogn)", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    let mut buf = buffer.clone();
                    consolidate_optimized(&mut buf, target_size);
                    buf
                })
            },
        );
    }

    group.finish();
}

/// Benchmark coactivation detection: O(n²m²) original vs O(n² log m) optimized
fn bench_coactivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Coactivation Comparison");
    group.sample_size(20); // Fewer samples for slow tests

    for size in [50, 100, 200] {
        // Create buffer with retrieval histories
        let buffer: Vec<EpisodicTrace> = (0..size)
            .map(|i| {
                let mut trace = create_test_trace(i as u64, 1.0, i as u64 * 100);
                // Add more retrieval events for realistic testing
                trace.retrieval_history = (0..10).map(|j| RetrievalEvent {
                    retrieved_at: Duration::from_secs(i as u64 * 100 + j * 30 + (i * j) % 50),
                    query_context: format!("Query {}:{}", i, j),
                    retrieval_method: "benchmark".to_string(),
                    retrieval_strength: 0.8,
                    content_matched: true,
                }).collect();
                trace
            })
            .collect();

        // Benchmark ORIGINAL O(n²m²) approach
        group.bench_with_input(
            BenchmarkId::new("Original_O(n²m²)", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    // Simulate original nested loop approach
                    let mut count = 0usize;
                    for memory in buffer {
                        for other_memory in buffer {
                            if memory.id >= other_memory.id { continue; }
                            for event in &memory.retrieval_history {
                                for other_event in &other_memory.retrieval_history {
                                    let interval = if event.retrieved_at > other_event.retrieved_at {
                                        event.retrieved_at - other_event.retrieved_at
                                    } else {
                                        other_event.retrieved_at - event.retrieved_at
                                    };
                                    if interval.as_secs() < 300 {
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                    count
                })
            },
        );

        // Benchmark OPTIMIZED O(n² log m) approach
        group.bench_with_input(
            BenchmarkId::new("Optimized_O(n²logm)", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    discover_coactivation_optimized(buffer, 2, 300)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark causal chain reconstruction: Clone-heavy vs Zero-clone
fn bench_causal_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("Causal Chain Comparison");

    for size in [100, 500, 1000] {
        // Create temporal sequence of memories
        let buffer: Vec<EpisodicTrace> = (0..size)
            .map(|i| create_test_trace(i as u64, 1.0, i as u64 * 60)) // 1 minute apart
            .collect();

        let effect_id = (size - 1) as u64; // Last memory

        // Benchmark ORIGINAL clone-heavy approach (simulated)
        group.bench_with_input(
            BenchmarkId::new("Original_Clone", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    // Simulate original approach with cloning
                    let effect = buffer.iter().find(|t| t.id == effect_id).unwrap();
                    let mut chain = vec![effect.clone()]; // Clone #1
                    let mut current_time = effect.timestamp;

                    for _ in 0..5 {
                        let candidates: Vec<&EpisodicTrace> = buffer.iter()
                            .filter(|t| t.timestamp < current_time) // O(n) scan
                            .collect();

                        if candidates.is_empty() { break; }

                        // Find best (just take first for simulation)
                        let best = candidates[0];
                        chain.insert(0, best.clone()); // Clone + O(k) shift
                        current_time = best.timestamp;
                    }
                    chain
                })
            },
        );

        // Benchmark OPTIMIZED zero-clone approach
        group.bench_with_input(
            BenchmarkId::new("Optimized_ZeroCopy", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    reconstruct_causal_chain_optimized(buffer, effect_id, 5, 0.0)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark temporal index: O(n) linear scan vs O(log n) binary search
fn bench_temporal_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("Temporal Index Comparison");

    for size in [100, 500, 1000, 2000] {
        let buffer: Vec<EpisodicTrace> = (0..size)
            .map(|i| create_test_trace(i as u64, 1.0, i as u64 * 60))
            .collect();

        let query_time = Duration::from_secs((size as u64 / 2) * 60);

        // Benchmark ORIGINAL O(n) linear filter
        group.bench_with_input(
            BenchmarkId::new("Original_O(n)", size),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    buffer.iter()
                        .filter(|t| t.timestamp < query_time)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Benchmark OPTIMIZED O(log n) with temporal index
        let idx = TemporalIndex::build(&buffer);
        group.bench_with_input(
            BenchmarkId::new("Optimized_O(logn)", size),
            &idx,
            |b, idx| {
                b.iter(|| {
                    idx.memories_before(query_time)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark batch retrieval: k×O(n) vs O(n)
fn bench_batch_retrieve(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Retrieve Comparison");

    for size in [500, 1000, 2000] {
        let buffer: Vec<EpisodicTrace> = (0..size)
            .map(|i| create_test_trace(i as u64, 1.0, i as u64))
            .collect();

        // Request 50 specific IDs
        let ids: Vec<u64> = (0..50).map(|i| (i * (size as u64 / 50)) as u64).collect();

        // Benchmark ORIGINAL k×O(n) approach
        group.bench_with_input(
            BenchmarkId::new("Original_kO(n)", size),
            &(&buffer, &ids),
            |b, (buffer, ids)| {
                b.iter(|| {
                    let results: Vec<Option<&EpisodicTrace>> = ids.iter()
                        .map(|&id| buffer.iter().find(|t| t.id == id))
                        .collect();
                    results
                })
            },
        );

        // Benchmark OPTIMIZED O(n) single-pass approach
        group.bench_with_input(
            BenchmarkId::new("Optimized_O(n)", size),
            &(&buffer, &ids),
            |b, (buffer, ids)| {
                b.iter(|| {
                    batch_retrieve(buffer, ids)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_consolidation,
    bench_coactivation,
    bench_causal_chain,
    bench_temporal_index,
    bench_batch_retrieve,
);

criterion_main!(benches);
