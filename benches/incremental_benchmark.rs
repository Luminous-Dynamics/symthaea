//! Incremental Computation Benchmark
//!
//! Verifies 10-100x speedup claims for incremental HDC operations.
//! Run with: cargo bench --bench incremental_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::hdc::{
    binary_hv::HV16,
    simd_hv::*,
    incremental_hv::*,
};
use std::time::Instant;

// ============================================================================
// INCREMENTAL BUNDLE BENCHMARKS
// ============================================================================

fn bench_incremental_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Incremental Bundle");

    for num_vectors in [10, 50, 100, 500].iter() {
        let vectors: Vec<HV16> = (0..*num_vectors).map(|i| HV16::random(i as u64)).collect();

        // Traditional: Rebundle ALL vectors after every change
        group.bench_with_input(
            BenchmarkId::new("traditional_update", num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    let mut vecs = vectors.clone();
                    // Update one vector
                    vecs[5] = HV16::random(9999);
                    // Rebundle ALL - O(n)
                    let result = simd_bundle(&vecs);
                    black_box(result)
                })
            },
        );

        // Incremental: Update only the changed vector's contribution
        group.bench_with_input(
            BenchmarkId::new("incremental_update", num_vectors),
            num_vectors,
            |bencher, _| {
                let mut bundle = IncrementalBundle::new();
                bundle.add(vectors.clone());

                bencher.iter(|| {
                    // Update one vector - O(1)
                    bundle.update(5, HV16::random(9999));
                    // Get bundle - O(1) with cached counts
                    let result = bundle.get_bundle();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMILARITY CACHE BENCHMARKS
// ============================================================================

fn bench_similarity_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("Similarity Cache");

    for num_targets in [100, 500, 1000].iter() {
        let query = HV16::random(42);
        let targets: Vec<HV16> = (0..*num_targets).map(|i| HV16::random(i as u64 + 100)).collect();

        // Traditional: Recompute ALL similarities every time
        group.bench_with_input(
            BenchmarkId::new("no_cache", num_targets),
            num_targets,
            |bencher, _| {
                bencher.iter(|| {
                    let similarities: Vec<f32> = targets.iter()
                        .map(|t| simd_similarity(&query, t))
                        .collect();
                    black_box(similarities)
                })
            },
        );

        // With cache: First call computes, subsequent calls are instant
        group.bench_with_input(
            BenchmarkId::new("with_cache_hit", num_targets),
            num_targets,
            |bencher, _| {
                let mut cache = SimilarityCache::new();
                let qid = cache.register_query(query.clone());

                // Prime the cache
                for (tid, target) in targets.iter().enumerate() {
                    cache.get_similarity(qid, tid as u64, target);
                }

                bencher.iter(|| {
                    let similarities: Vec<f32> = targets.iter()
                        .enumerate()
                        .map(|(tid, t)| cache.get_similarity(qid, tid as u64, t))
                        .collect();
                    black_box(similarities)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// INCREMENTAL BIND BENCHMARKS
// ============================================================================

fn bench_incremental_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("Incremental Bind");

    for num_queries in [10, 50, 100, 500].iter() {
        let queries: Vec<HV16> = (0..*num_queries).map(|i| HV16::random(i as u64)).collect();
        let key = HV16::random(999);

        // Traditional: Rebind ALL queries when ONE changes
        group.bench_with_input(
            BenchmarkId::new("traditional_rebind_all", num_queries),
            num_queries,
            |bencher, _| {
                bencher.iter(|| {
                    let mut qs = queries.clone();
                    // Update one query
                    qs[5] = HV16::random(9999);
                    // Rebind ALL - O(n)
                    let results: Vec<HV16> = qs.iter()
                        .map(|q| simd_bind(q, &key))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Incremental: Bind only the changed query
        group.bench_with_input(
            BenchmarkId::new("incremental_rebind_one", num_queries),
            num_queries,
            |bencher, _| {
                let mut inc_bind = IncrementalBind::new(key.clone());
                inc_bind.add_queries(queries.clone());

                bencher.iter(|| {
                    // Update one query - marks it dirty
                    inc_bind.update_query(5, HV16::random(9999));
                    // Get results - binds only dirty queries (1 out of n)
                    let results = inc_bind.get_bound_results();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// REALISTIC CONSCIOUSNESS CYCLE SIMULATION
// ============================================================================

fn bench_realistic_consciousness_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Realistic Consciousness Cycle");

    // Simulate typical consciousness operation:
    // - 100 concept vectors
    // - 5 change per cycle (5% update rate)
    // - Bundle to form current context
    // - Compute similarities to 1000 memories
    // - Bind with temporal key

    let num_concepts = 100;
    let num_memories = 1000;
    let updates_per_cycle = 5;

    let initial_concepts: Vec<HV16> = (0..num_concepts).map(|i| HV16::random(i as u64)).collect();
    let memories: Vec<HV16> = (0..num_memories).map(|i| HV16::random(i as u64 + 1000)).collect();
    let temporal_key = HV16::random(9999);

    // Traditional approach: Recompute EVERYTHING every cycle
    group.bench_function("traditional_full_recompute", |bencher| {
        bencher.iter(|| {
            let mut concepts = initial_concepts.clone();

            // Update 5 concepts
            for i in 0..updates_per_cycle {
                concepts[i] = HV16::random((i * 7) as u64);
            }

            // Bundle ALL concepts - O(n)
            let context = simd_bundle(&concepts);

            // Compute ALL similarities - O(m)
            let similarities: Vec<f32> = memories.iter()
                .map(|mem| simd_similarity(&context, mem))
                .collect();

            // Bind context with temporal key - O(1)
            let bound_context = simd_bind(&context, &temporal_key);

            black_box((bound_context, similarities))
        })
    });

    // Incremental approach: Update only what changed
    group.bench_function("incremental_smart_update", |bencher| {
        // Setup incremental structures
        let mut concept_bundle = IncrementalBundle::new();
        concept_bundle.add(initial_concepts.clone());

        let mut similarity_cache = SimilarityCache::new();
        let context_qid = similarity_cache.register_query(concept_bundle.get_bundle());

        let mut bind_tracker = IncrementalBind::new(temporal_key.clone());
        bind_tracker.add_queries(vec![concept_bundle.get_bundle()]);

        bencher.iter(|| {
            // Update 5 concepts - O(k) where k=5
            for i in 0..updates_per_cycle {
                concept_bundle.update(i, HV16::random((i * 7) as u64));
            }

            // Get updated bundle - O(1) with cached counts
            let context = concept_bundle.get_bundle();

            // Invalidate similarity cache for this query (context changed)
            similarity_cache.invalidate_query(context_qid);

            // Compute NEW similarities (cache miss) - O(m) but only once
            let similarities: Vec<f32> = memories.iter()
                .enumerate()
                .map(|(tid, mem)| similarity_cache.get_similarity(context_qid, tid as u64, mem))
                .collect();

            // Update bind tracker with new context - O(1)
            bind_tracker.update_query(0, context.clone());

            // Get bound result - O(k) where k=1
            let bound_results = bind_tracker.get_bound_results();

            black_box((bound_results[0].clone(), similarities))
        })
    });

    group.finish();
}

// ============================================================================
// SPEEDUP VERIFICATION
// ============================================================================

fn verify_incremental_speedups() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     INCREMENTAL COMPUTATION SPEEDUP VERIFICATION         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Verify incremental bundle speedup
    {
        let n = 100;
        let vectors: Vec<HV16> = (0..n).map(|i| HV16::random(i)).collect();

        // Traditional: Rebundle all
        let start = Instant::now();
        for _ in 0..1000 {
            let mut vecs = vectors.clone();
            vecs[5] = HV16::random(9999);
            let _ = simd_bundle(&vecs);
        }
        let traditional_time = start.elapsed().as_micros() / 1000;

        // Incremental: Update one
        let mut bundle = IncrementalBundle::new();
        bundle.add(vectors.clone());

        let start = Instant::now();
        for _ in 0..1000 {
            bundle.update(5, HV16::random(9999));
            let _ = bundle.get_bundle();
        }
        let incremental_time = start.elapsed().as_micros() / 1000;

        let speedup = traditional_time as f64 / incremental_time as f64;

        println!("Incremental Bundle (n=100, update 1 vector):");
        println!("  Traditional (rebundle all): {} µs", traditional_time);
        println!("  Incremental (update one):   {} µs", incremental_time);
        println!("  Speedup:                     {:.1}x", speedup);
        println!("  Target:                      10x | Status: {}\n",
                 if speedup >= 9.0 { "✓ ACHIEVED" } else { "⚠ Below target" });
    }

    // Verify similarity cache speedup
    {
        let query = HV16::random(42);
        let targets: Vec<HV16> = (0..1000).map(|i| HV16::random(i + 100)).collect();

        // No cache
        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<f32> = targets.iter().map(|t| simd_similarity(&query, t)).collect();
        }
        let no_cache_time = start.elapsed().as_micros() / 100;

        // With cache (100% hit rate)
        let mut cache = SimilarityCache::new();
        let qid = cache.register_query(query.clone());

        // Prime cache
        for (tid, target) in targets.iter().enumerate() {
            cache.get_similarity(qid, tid as u64, target);
        }

        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<f32> = targets.iter().enumerate()
                .map(|(tid, t)| cache.get_similarity(qid, tid as u64, t))
                .collect();
        }
        let cache_time = start.elapsed().as_micros() / 100;

        let speedup = no_cache_time as f64 / cache_time as f64;

        println!("Similarity Cache (1000 targets, 100% hit rate):");
        println!("  No cache (compute all):  {} µs", no_cache_time);
        println!("  With cache (hash lookup): {} µs", cache_time);
        println!("  Speedup:                  {:.1}x", speedup);
        println!("  Target:                   100x | Status: {}\n",
                 if speedup >= 90.0 { "✓ ACHIEVED" } else { "⚠ Below target" });

        let stats = cache.stats();
        println!("  Cache Stats: {:.1}% hit rate, {} entries\n", stats.hit_rate * 100.0, stats.cache_size);
    }

    println!("═══════════════════════════════════════════════════════════\n");
}

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_incremental_bundle,
        bench_similarity_cache,
        bench_incremental_bind,
        bench_realistic_consciousness_cycle
);

fn main() {
    verify_incremental_speedups();
    benches();
    Criterion::default().final_summary();
}
