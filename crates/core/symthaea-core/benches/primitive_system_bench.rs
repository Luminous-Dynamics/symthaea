// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Benchmarks for PrimitiveSystem Operations
//!
//! Run with: `cargo bench --bench primitive_system_bench`
//!
//! Benchmarks:
//! - System initialization
//! - Primitive lookup
//! - LSH search
//! - Composition operations (bind, bundle, sequence)
//! - Similarity matrix computation
//! - Cache performance

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::{
    CompositionAlgebra, CompositionCache, PrimitiveSystem, PrimitiveTier,
};

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_system_init(c: &mut Criterion) {
    c.bench_function("primitive_system_new", |b| {
        b.iter(|| {
            let system = PrimitiveSystem::new();
            black_box(system.count())
        });
    });
}

fn bench_global_access(c: &mut Criterion) {
    // Warm up the global
    let _ = PrimitiveSystem::global();

    c.bench_function("primitive_system_global", |b| {
        b.iter(|| {
            let system = PrimitiveSystem::global();
            black_box(system.count())
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOKUP BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_primitive_lookup(c: &mut Criterion) {
    let system = PrimitiveSystem::global();

    let mut group = c.benchmark_group("Primitive_Lookup");

    // Single lookup
    group.bench_function("single_get", |b| {
        b.iter(|| black_box(system.get("CAUSE")));
    });

    // Miss lookup
    group.bench_function("miss_get", |b| {
        b.iter(|| black_box(system.get("NONEXISTENT_PRIMITIVE")));
    });

    // Tier query
    group.bench_function("get_tier_mathematical", |b| {
        b.iter(|| black_box(system.get_tier(PrimitiveTier::Mathematical)));
    });

    group.bench_function("get_tier_consciousness", |b| {
        b.iter(|| black_box(system.get_tier(PrimitiveTier::Consciousness)));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_composition_ops(c: &mut Criterion) {
    let system = PrimitiveSystem::global();
    let cause = system.get("CAUSE").unwrap();
    let effect = system.get("EFFECT").unwrap();
    let mass = system.get("MASS").unwrap();
    let energy = system.get("ENERGY").unwrap();
    let force = system.get("FORCE").unwrap();

    let mut group = c.benchmark_group("Composition_Ops");

    // Bind (XOR)
    group.bench_function("bind_two", |b| {
        b.iter(|| black_box(cause.encoding.bind(&effect.encoding)));
    });

    // Bundle (majority vote)
    group.bench_function("bundle_three", |b| {
        let encodings = vec![
            mass.encoding.clone(),
            energy.encoding.clone(),
            force.encoding.clone(),
        ];
        b.iter(|| black_box(BinaryHV::bundle(&encodings)));
    });

    // Sequence (bind chain)
    group.bench_function("sequence_three", |b| {
        b.iter(|| {
            let step1 = cause.encoding.bind(&cause.encoding.permute(1));
            let step2 = step1.bind(&effect.encoding.permute(2));
            black_box(step2.bind(&mass.encoding.permute(3)))
        });
    });

    // Permute
    group.bench_function("permute", |b| {
        b.iter(|| black_box(cause.encoding.permute(100)));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMILARITY BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_similarity(c: &mut Criterion) {
    let system = PrimitiveSystem::global();
    let cause = system.get("CAUSE").unwrap();
    let effect = system.get("EFFECT").unwrap();

    let mut group = c.benchmark_group("Similarity");

    // Single similarity computation
    group.bench_function("binary_hv_similarity", |b| {
        b.iter(|| black_box(cause.encoding.similarity(&effect.encoding)));
    });

    // Find similar by name (brute force)
    group.bench_function("find_similar_k5", |b| {
        b.iter(|| black_box(system.find_similar("CAUSE", 5)));
    });

    group.bench_function("find_similar_k10", |b| {
        b.iter(|| black_box(system.find_similar("CAUSE", 10)));
    });

    // Find similar by encoding
    group.bench_function("find_similar_to_encoding_k5", |b| {
        b.iter(|| black_box(system.find_similar_to_encoding(&cause.encoding, 5)));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// LSH INDEX BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_lsh_search(c: &mut Criterion) {
    let system = PrimitiveSystem::global();
    let cause = system.get("CAUSE").unwrap();

    let mut group = c.benchmark_group("LSH_Search");

    // Build LSH index via system helper
    group.bench_function("lsh_build_16_bands", |b| {
        b.iter(|| black_box(system.build_lsh_index(16, 64)));
    });

    // Query LSH index (get candidates)
    let lsh = system.build_lsh_index(16, 64);
    group.bench_function("lsh_query_candidates", |b| {
        b.iter(|| black_box(lsh.query_candidates(&cause.encoding)));
    });

    // Full LSH-accelerated similarity search
    group.bench_function("find_similar_lsh_k5", |b| {
        b.iter(|| black_box(system.find_similar_lsh(&cause.encoding, 5, &lsh)));
    });

    group.bench_function("find_similar_lsh_k10", |b| {
        b.iter(|| black_box(system.find_similar_lsh(&cause.encoding, 10, &lsh)));
    });

    // Compare: brute force vs LSH
    group.bench_function("brute_force_k5", |b| {
        b.iter(|| black_box(system.find_similar_to_encoding(&cause.encoding, 5)));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CACHE BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_cache(c: &mut Criterion) {
    let system = PrimitiveSystem::global();

    let mut group = c.benchmark_group("Cache");

    // Cache miss then hit pattern
    group.bench_function("cache_bind_miss_then_hit", |b| {
        b.iter(|| {
            let mut cache = CompositionCache::new(100);
            // Miss
            let _ = cache.bind_cached(system, "CAUSE", "EFFECT");
            // Hit
            black_box(cache.bind_cached(system, "CAUSE", "EFFECT"))
        });
    });

    // Pre-warmed cache hits
    let mut warm_cache = CompositionCache::new(100);
    let _ = warm_cache.bind_cached(system, "CAUSE", "EFFECT");

    group.bench_function("cache_bind_hit_only", |b| {
        b.iter(|| black_box(warm_cache.bind_cached(system, "CAUSE", "EFFECT")));
    });

    // Bundle caching
    group.bench_function("cache_bundle", |b| {
        b.iter(|| {
            let mut cache = CompositionCache::new(100);
            let names: Vec<&str> = vec!["MASS", "ENERGY", "FORCE"];
            black_box(cache.bundle_cached(system, &names))
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMILARITY MATRIX BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_similarity_matrix(c: &mut Criterion) {
    let system = PrimitiveSystem::global();

    let mut group = c.benchmark_group("Similarity_Matrix");

    // Small matrix (5x5)
    let small_names: Vec<&str> = vec!["CAUSE", "EFFECT", "MASS", "ENERGY", "FORCE"];

    group.bench_function("matrix_5x5", |b| {
        b.iter(|| black_box(system.similarity_matrix(&small_names)));
    });

    // Medium matrix (full tier) - get names from tier
    let math_primitives = system.get_tier(PrimitiveTier::Mathematical);
    if !math_primitives.is_empty() {
        let math_names: Vec<&str> = math_primitives.iter().map(|p| p.name.as_str()).collect();
        group.bench_function("matrix_math_tier", |b| {
            b.iter(|| black_box(system.similarity_matrix(&math_names)));
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITION ALGEBRA BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_algebra(c: &mut Criterion) {
    let system = PrimitiveSystem::global();

    let mut group = c.benchmark_group("Composition_Algebra");

    group.bench_function("algebra_define", |b| {
        b.iter(|| {
            let mut algebra = CompositionAlgebra::new();
            let _ = algebra.define("TEST_COMP", "CAUSE ^ EFFECT", system);
            black_box(algebra.list().len())
        });
    });

    // Pre-built algebra
    let mut algebra = CompositionAlgebra::new();
    let _ = algebra.define("CAUSALITY", "CAUSE ^ EFFECT", system);

    group.bench_function("algebra_get", |b| {
        b.iter(|| black_box(algebra.get("CAUSALITY").cloned()));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH OPERATIONS BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_batch_ops(c: &mut Criterion) {
    let system = PrimitiveSystem::global();

    // Create pairs for batch_bind
    let pairs: Vec<(&str, &str)> =
        vec![("CAUSE", "EFFECT"), ("MASS", "ENERGY"), ("BEFORE", "AFTER")];

    let mut group = c.benchmark_group("Batch_Operations");

    group.bench_function("batch_bind_3_pairs", |b| {
        b.iter(|| black_box(system.batch_bind(&pairs)));
    });

    // Multiple individual find_similar calls (simulating batch)
    let names = ["CAUSE", "EFFECT", "MASS"];
    group.bench_function("multi_find_similar_3_k3", |b| {
        b.iter(|| {
            for name in &names {
                black_box(system.find_similar(name, 3));
            }
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CRITERION MAIN
// ═══════════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_system_init,
    bench_global_access,
    bench_primitive_lookup,
    bench_composition_ops,
    bench_similarity,
    bench_lsh_search,
    bench_cache,
    bench_similarity_matrix,
    bench_algebra,
    bench_batch_ops,
);

criterion_main!(benches);
