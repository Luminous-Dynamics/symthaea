//! HDC Performance Benchmarks
//!
//! Validates the performance claims from README.md:
//! - HDC Encoding: ~0.05ms (20,000 ops/sec)
//! - HDC Recall: ~0.10ms (10,000 ops/sec)
//! - Binding (XOR): ~10ns
//! - Similarity (Hamming): ~20ns

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::{SemanticSpace, HDC_DIMENSION};

/// Benchmark HV16 random generation (deterministic from seed)
fn bench_hv16_random(c: &mut Criterion) {
    c.bench_function("HV16::random", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            black_box(HV16::random(seed))
        })
    });
}

/// Benchmark HV16 binding (XOR operation)
/// Claim: ~10ns on modern CPU
fn bench_hv16_bind(c: &mut Criterion) {
    let hv_a = HV16::random(42);
    let hv_b = HV16::random(43);

    c.bench_function("HV16::bind (XOR)", |bencher| {
        bencher.iter(|| black_box(hv_a.bind(&hv_b)))
    });
}

/// Benchmark HV16 similarity (Hamming distance)
/// Claim: ~20ns
fn bench_hv16_similarity(c: &mut Criterion) {
    let hv_a = HV16::random(42);
    let hv_b = HV16::random(43);

    c.bench_function("HV16::similarity (Hamming)", |bencher| {
        bencher.iter(|| black_box(hv_a.similarity(&hv_b)))
    });
}

/// Benchmark HV16 bundling (majority vote)
fn bench_hv16_bundle(c: &mut Criterion) {
    let vectors: Vec<HV16> = (0..10).map(|i| HV16::random(i)).collect();

    c.bench_function("HV16::bundle (10 vectors)", |b| {
        b.iter(|| black_box(HV16::bundle(black_box(&vectors))))
    });
}

/// Benchmark SemanticSpace creation
fn bench_semantic_space_new(c: &mut Criterion) {
    c.bench_function("SemanticSpace::new (16K dim)", |b| {
        b.iter(|| {
            black_box(SemanticSpace::new(HDC_DIMENSION))
        })
    });
}

/// Benchmark semantic encoding
/// Claim: ~0.05ms (50μs)
fn bench_semantic_encode(c: &mut Criterion) {
    // Test with a simple encoding that doesn't require mutable borrow
    c.bench_function("SemanticSpace::encode (short)", |bencher| {
        let mut space = SemanticSpace::new(HDC_DIMENSION).unwrap();
        let text = "install nginx";
        bencher.iter(|| black_box(space.encode(black_box(text)).unwrap()))
    });

    c.bench_function("SemanticSpace::encode (medium)", |bencher| {
        let mut space = SemanticSpace::new(HDC_DIMENSION).unwrap();
        let text = "search firefox browser package";
        bencher.iter(|| black_box(space.encode(black_box(text)).unwrap()))
    });

    c.bench_function("SemanticSpace::encode (long)", |bencher| {
        let mut space = SemanticSpace::new(HDC_DIMENSION).unwrap();
        let text = "configure network settings and enable docker service";
        bencher.iter(|| black_box(space.encode(black_box(text)).unwrap()))
    });
}

/// Benchmark batch operations (realistic workload)
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Operations");

    // Create 100 random vectors
    let vectors: Vec<HV16> = (0..100).map(|i| HV16::random(i)).collect();

    // Benchmark finding most similar (linear scan)
    let query = HV16::random(1000);
    group.bench_function("find_most_similar (100 vectors)", |b| {
        b.iter(|| {
            let mut best_sim = -1.0f32;
            let mut _best_idx = 0;
            for (i, v) in vectors.iter().enumerate() {
                let sim = query.similarity(v);
                if sim > best_sim {
                    best_sim = sim;
                    _best_idx = i;
                }
            }
            black_box(best_sim)
        })
    });

    // Benchmark binding chain
    group.bench_function("bind_chain (10 vectors)", |b| {
        b.iter(|| {
            let mut result = vectors[0];
            for v in vectors[1..10].iter() {
                result = result.bind(v);
            }
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark memory footprint validation
fn bench_memory_footprint(c: &mut Criterion) {
    c.bench_function("HV16 memory size validation", |b| {
        b.iter(|| {
            let v = HV16::random(42);
            // HV16 should be exactly 256 bytes
            assert_eq!(std::mem::size_of_val(&v), 256);
            black_box(v)
        })
    });
}

criterion_group!(
    benches,
    bench_hv16_random,
    bench_hv16_bind,
    bench_hv16_similarity,
    bench_hv16_bundle,
    bench_semantic_space_new,
    bench_semantic_encode,
    bench_batch_operations,
    bench_memory_footprint,
);

criterion_main!(benches);
