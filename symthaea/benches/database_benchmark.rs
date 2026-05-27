// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database Benchmark Suite
//!
//! Measures store and search latency for SQLite and LanceDB backends
//! at various data scales. The N=500 threshold is where SQLite's
//! LSH-accelerated search activates, and should be visible as a
//! performance step-change.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench database_benchmark
//! cargo bench --bench database_benchmark -- --quick     # Fewer samples
//! cargo bench --bench database_benchmark --features lancedb-backend  # Include Lance
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea::databases::{ConsciousnessDatabase, MemoryRecord, MemoryType, SqliteMemory};
use symthaea_core::hdc::binary_hv::BinaryHV;

fn make_record(id: &str, seed: u64) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        memory_type: MemoryType::Episodic,
        encoding: BinaryHV::random(seed),
        content: format!("Benchmark record {id}"),
        timestamp_ms: 1700000000 + seed,
        valence: 0.5,
        arousal: 0.3,
        psi: 0.65,
        topics: vec!["bench".to_string()],
        metadata: "{}".to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    }
}

/// Pre-populate a SQLite database with `n` records.
fn populate_sqlite(n: usize) -> SqliteMemory {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let db = SqliteMemory::in_memory().unwrap();
    rt.block_on(async {
        for i in 0..n {
            db.store(make_record(&format!("pre-{i}"), i as u64 * 7))
                .await
                .unwrap();
        }
    });
    db
}

// ============================================================================
// SQLite Store Benchmarks
// ============================================================================

fn bench_sqlite_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_store");
    group.sample_size(30);

    for n in [100, 500, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let db = populate_sqlite(n);
            let rt = tokio::runtime::Runtime::new().unwrap();
            let mut counter = n as u64;

            b.iter(|| {
                counter += 1;
                let record = make_record(&format!("new-{counter}"), counter * 13);
                rt.block_on(async { db.store(black_box(record)).await.unwrap() });
            });
        });
    }

    group.finish();
}

// ============================================================================
// SQLite Search Benchmarks
// ============================================================================

fn bench_sqlite_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_search_top10");
    group.sample_size(30);

    // N<500 = brute force, N>=500 = LSH-accelerated
    for n in [100, 500, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let db = populate_sqlite(n);
            let rt = tokio::runtime::Runtime::new().unwrap();
            let query = BinaryHV::random(999);

            b.iter(|| {
                rt.block_on(async {
                    let results = db.search_similar(black_box(&query), 10).await.unwrap();
                    black_box(results);
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// SQLite Filtered Search Benchmarks
// ============================================================================

fn bench_sqlite_search_filtered(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_search_filtered_top10");
    group.sample_size(30);

    for n in [500, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let db = populate_sqlite(n);
            let rt = tokio::runtime::Runtime::new().unwrap();
            let query = BinaryHV::random(999);

            b.iter(|| {
                rt.block_on(async {
                    let results = db
                        .search_similar_filtered(
                            black_box(&query),
                            10,
                            Some("memory_type = 'episodic'"),
                        )
                        .await
                        .unwrap();
                    black_box(results);
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// LanceDB Benchmarks (feature-gated)
// ============================================================================

#[cfg(feature = "lancedb-backend")]
mod lance_benches {
    use super::*;
    use symthaea::databases::LanceMemory;

    fn populate_lance(n: usize) -> LanceMemory {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let db = LanceMemory::in_memory().await.unwrap();
            for i in 0..n {
                db.store(make_record(&format!("pre-{i}"), i as u64 * 7))
                    .await
                    .unwrap();
            }
            db
        })
    }

    pub fn bench_lance_store(c: &mut Criterion) {
        let mut group = c.benchmark_group("lance_store");
        group.sample_size(20);

        for n in [100, 500, 1000] {
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
                let db = populate_lance(n);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut counter = n as u64;

                b.iter(|| {
                    counter += 1;
                    let record = make_record(&format!("new-{counter}"), counter * 13);
                    rt.block_on(async { db.store(black_box(record)).await.unwrap() });
                });
            });
        }

        group.finish();
    }

    pub fn bench_lance_search(c: &mut Criterion) {
        let mut group = c.benchmark_group("lance_search_top10");
        group.sample_size(20);

        for n in [100, 500, 1000] {
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
                let db = populate_lance(n);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let query = BinaryHV::random(999);

                b.iter(|| {
                    rt.block_on(async {
                        let results = db.search_similar(black_box(&query), 10).await.unwrap();
                        black_box(results);
                    });
                });
            });
        }

        group.finish();
    }
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(not(feature = "lancedb-backend"))]
criterion_group!(
    benches,
    bench_sqlite_store,
    bench_sqlite_search,
    bench_sqlite_search_filtered,
);

#[cfg(feature = "lancedb-backend")]
criterion_group!(
    benches,
    bench_sqlite_store,
    bench_sqlite_search,
    bench_sqlite_search_filtered,
    lance_benches::bench_lance_store,
    lance_benches::bench_lance_search,
);

criterion_main!(benches);
