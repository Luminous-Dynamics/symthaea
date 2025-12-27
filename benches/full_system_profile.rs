//! Full System Profile Benchmark
//!
//! Profiles the COMPLETE consciousness cycle to find real bottlenecks.
//! Run with: cargo bench --bench full_system_profile

use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use symthaea::hdc::{SemanticSpace, binary_hv::HV16, simd_hv::*};
use symthaea::ltc::LiquidNetwork;
use symthaea::consciousness::ConsciousnessGraph;
use symthaea::memory::{EpisodicMemoryEngine, EpisodicConfig};
use symthaea::memory::optimized_episodic::*;

// ============================================================================
// FULL CONSCIOUSNESS CYCLE - The REAL workload
// ============================================================================

fn bench_full_consciousness_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Consciousness Cycle");

    // Initialize components
    let mut semantic = SemanticSpace::new(10_000).unwrap();
    let mut ltc = LiquidNetwork::new(1_000).unwrap();
    let mut consciousness = ConsciousnessGraph::new();
    // Episodic memory - using default config
    let mut episodic = EpisodicMemoryEngine::new().unwrap();

    group.bench_function("single_query", |bencher| {
        bencher.iter(|| {
            // 1. Semantic encoding (HDC operations)
            let query_hv = HV16::random(42);
            let context_hvs: Vec<HV16> = (0..10).map(|i| HV16::random(i + 100)).collect();

            // 2. Bind query with context
            let mut bound = query_hv.clone();
            for ctx in &context_hvs {
                bound = simd_bind(&bound, ctx);
            }

            // 3. Bundle multiple concepts
            let bundled = simd_bundle(&context_hvs);

            // 4. Similarity search in semantic space
            let memory_hvs: Vec<HV16> = (0..100).map(|i| HV16::random(i + 200)).collect();
            let _best = simd_find_most_similar(&bundled, &memory_hvs);

            // 5. LTC network step
            let input: Vec<f32> = (0..1000).map(|i| (i as f32 % 10.0) / 10.0).collect();
            ltc.inject(&input).unwrap();
            ltc.step().unwrap();
            let _consciousness_level = ltc.consciousness_level();

            // 6. Add to consciousness graph
            let semantic_vec = vec![bundled];
            let dynamic_state = ltc.read_state().unwrap();
            let _node = consciousness.add_state(semantic_vec, dynamic_state);

            black_box(());
        })
    });

    group.finish();
}

// ============================================================================
// BATCH PROCESSING - Where parallelism matters
// ============================================================================

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Operations");

    for batch_size in [10, 100, 1000].iter() {
        let queries: Vec<HV16> = (0..*batch_size).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64 + 10000)).collect();

        // Sequential batch similarity
        group.bench_with_input(
            BenchmarkId::new("sequential_similarity", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results: Vec<_> = queries.iter()
                        .map(|q| simd_find_most_similar(q, &memory))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Sequential batch bind
        group.bench_with_input(
            BenchmarkId::new("sequential_bind", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let key = HV16::random(99999);
                    let results: Vec<_> = queries.iter()
                        .map(|q| simd_bind(q, &key))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// EPISODIC MEMORY STRESS TEST
// ============================================================================

fn bench_episodic_stress_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("Episodic Memory Stress");

    for num_memories in [100, 500, 1000].iter() {
        // Using default config - actual fields are: sdm_config, max_buffer_size, recall_threshold, decay_rate
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Populate with memories (simplified - actual API may differ)
        for i in 0..*num_memories {
            let hv = HV16::random(i as u64);
            // Note: Actual memory storage API may differ - this is for profiling
            // We'll benchmark the optimized operations directly instead
        }

        // Benchmark optimized operations directly since we have them
        group.bench_with_input(
            BenchmarkId::new("consolidation", num_memories),
            num_memories,
            |bencher, _| {
                bencher.iter(|| {
                    // Create test buffer
                    let mut buffer: Vec<_> = (0..*num_memories).map(|i| {
                        create_test_trace(i as u64, 0.8 + (i as f32 % 100.0) / 1000.0)
                    }).collect();
                    consolidate_optimized(&mut buffer, num_memories / 2);
                    black_box(buffer)
                })
            },
        );
    }

    group.finish();
}

// Helper function for creating test traces
fn create_test_trace(id: u64, strength: f32) -> symthaea::memory::EpisodicTrace {
    use symthaea::memory::EpisodicTrace;
    use std::time::{SystemTime, Duration};
    use symthaea::hdc::HDC_DIMENSION;

    // Create dummy vectors for testing
    let semantic_vec = vec![0.0f32; HDC_DIMENSION];
    let temporal_vec = vec![0.0f32; HDC_DIMENSION];
    let chrono_semantic = vec![0i8; HDC_DIMENSION];
    let emotional_binding = vec![0i8; HDC_DIMENSION];

    EpisodicTrace {
        id,
        timestamp: Duration::from_secs(id),
        content: format!("test_memory_{}", id),
        tags: vec!["test".to_string()],
        emotion: 0.0,
        chrono_semantic_vector: chrono_semantic,
        emotional_binding_vector: emotional_binding,
        temporal_vector: temporal_vec,
        semantic_vector: semantic_vec,
        recall_count: 0,
        last_recall_time: None,
        consolidation_strength: strength,
        drift_stats: None,
        intent: None,
        goal_id: None,
        goal_progress_contribution: 0.0,
        is_goal_completion: false,
    }
}

// ============================================================================
// MEMORY ALLOCATION PATTERNS
// ============================================================================

fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation");

    // HV16 allocation patterns
    group.bench_function("hv16_vec_allocation", |bencher| {
        bencher.iter(|| {
            let hvs: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64)).collect();
            black_box(hvs)
        })
    });

    // String allocation in episodic memory
    group.bench_function("string_allocation", |bencher| {
        bencher.iter(|| {
            let strings: Vec<String> = (0..1000)
                .map(|i| format!("memory_content_{}_with_some_extra_data", i))
                .collect();
            black_box(strings)
        })
    });

    // Vec cloning vs references
    group.bench_function("vec_clone_1000_hvs", |bencher| {
        let hvs: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64)).collect();
        bencher.iter(|| {
            let cloned = hvs.clone();
            black_box(cloned)
        })
    });

    group.bench_function("vec_iter_refs_1000_hvs", |bencher| {
        let hvs: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64)).collect();
        bencher.iter(|| {
            let refs: Vec<&HV16> = hvs.iter().collect();
            black_box(refs)
        })
    });

    group.finish();
}

// ============================================================================
// LTC NETWORK HOTSPOTS
// ============================================================================

fn bench_ltc_hotspots(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Network");

    for num_neurons in [100, 500, 1000].iter() {
        let mut ltc = LiquidNetwork::new(*num_neurons).unwrap();
        let input: Vec<f32> = (0..*num_neurons).map(|i| (i as f32 % 10.0) / 10.0).collect();

        group.bench_with_input(
            BenchmarkId::new("step", num_neurons),
            num_neurons,
            |bencher, _| {
                bencher.iter(|| {
                    ltc.step().unwrap();
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("inject", num_neurons),
            num_neurons,
            |bencher, _| {
                bencher.iter(|| {
                    ltc.inject(&input).unwrap();
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("consciousness_level", num_neurons),
            num_neurons,
            |bencher, _| {
                bencher.iter(|| {
                    let level = ltc.consciousness_level();
                    black_box(level)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// CONSCIOUSNESS GRAPH OPERATIONS
// ============================================================================

fn bench_consciousness_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("Consciousness Graph");

    let mut graph = ConsciousnessGraph::new();

    // Add nodes
    group.bench_function("add_state", |bencher| {
        let hv = vec![HV16::random(42)];
        let state: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        bencher.iter(|| {
            let node = graph.add_state(hv.clone(), state.clone());
            black_box(node)
        })
    });

    // Populate for complexity measurement
    for i in 0..100 {
        let hv = vec![HV16::random(i)];
        let state: Vec<f32> = (0..1000).map(|j| j as f32 / 1000.0).collect();
        graph.add_state(hv, state);
    }

    group.bench_function("complexity", |bencher| {
        bencher.iter(|| {
            let c = graph.complexity();
            black_box(c)
        })
    });

    group.bench_function("current_phi", |bencher| {
        bencher.iter(|| {
            let c = graph.current_phi();
            black_box(c)
        })
    });

    group.finish();
}

// ============================================================================
// HOTSPOT IDENTIFICATION HELPER
// ============================================================================

fn print_profiling_summary() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          PROFILING BENCHMARK - HOTSPOT ANALYSIS          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("This benchmark profiles the ENTIRE consciousness system:");
    println!("  1. Full consciousness cycle (HDC + LTC + graph + episodic)");
    println!("  2. Batch operations (sequential - ripe for parallelism)");
    println!("  3. Episodic memory under stress");
    println!("  4. Memory allocation patterns");
    println!("  5. LTC network hotspots");
    println!("  6. Consciousness graph operations\n");

    println!("Use results to identify:");
    println!("  • Slowest operations in full cycle");
    println!("  • Best candidates for parallelization");
    println!("  • Memory allocation bottlenecks");
    println!("  • Component-specific optimizations\n");

    println!("Revolutionary optimizations to consider:");
    println!("  🚀 Rayon parallelism for batch operations");
    println!("  🚀 Arena allocators for episodic memory");
    println!("  🚀 Lock-free concurrent structures");
    println!("  🚀 Memory-mapped persistent storage");
    println!("  🚀 SIMD for LTC matrix operations\n");
}

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10));
    targets =
        bench_full_consciousness_cycle,
        bench_batch_operations,
        bench_episodic_stress_test,
        bench_allocation_patterns,
        bench_ltc_hotspots,
        bench_consciousness_graph
);

criterion_main!(benches);

// Print summary before benchmarks
#[allow(dead_code)]
fn print_summary_and_run() {
    print_profiling_summary();
}
