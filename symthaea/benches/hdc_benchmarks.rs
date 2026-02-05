//! HDC Core Operations Benchmark Suite
//!
//! Comprehensive benchmarks for Hyperdimensional Computing (HDC) operations
//! using binary hypervectors (HV16).
//!
//! ## Operations Benchmarked
//!
//! - `HV16::random()` - Deterministic random vector generation (BLAKE3-based)
//! - `HV16::bind()` - XOR binding operation (concept composition)
//! - `HV16::similarity()` - Hamming similarity calculation
//! - `HV16::hamming_distance()` - Hamming distance measurement
//! - `HV16::bundle()` - Majority vote bundling
//! - `HV16::permute()` - Bit rotation for sequence encoding
//! - `HV16::invert()` - Bitwise NOT operation
//!
//! ## Usage
//!
//! ```bash
//! # Run all HDC benchmarks
//! cargo bench --bench hdc_benchmarks
//!
//! # Save baseline for comparison
//! cargo bench --bench hdc_benchmarks -- --save-baseline main
//!
//! # Compare against baseline
//! cargo bench --bench hdc_benchmarks -- --baseline main
//! ```
//!
//! ## Performance Targets (Release Mode)
//!
//! | Operation      | Target   | Notes                           |
//! |----------------|----------|----------------------------------|
//! | random()       | <500ns   | BLAKE3 hash + XOF expansion      |
//! | bind()         | <20ns    | SIMD XOR (AVX2)                  |
//! | similarity()   | <30ns    | SIMD XOR + popcount              |
//! | hamming()      | <30ns    | SIMD XOR + popcount              |
//! | bundle(5)      | <2us     | Majority vote over 5 vectors     |
//! | permute(1)     | <100ns   | Word-level bit rotation          |
//! | invert()       | <20ns    | SIMD NOT                         |

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use symthaea_core::hdc::binary_hv::HV16;

// =============================================================================
// HV16::random() GENERATION BENCHMARKS
// =============================================================================

fn bench_random_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_random");
    group.throughput(Throughput::Elements(1));

    // Single random vector generation
    group.bench_function("single", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            black_box(HV16::random(seed))
        })
    });

    // Batch generation (common pattern in initialization)
    for count in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &n| {
            b.iter(|| {
                let vectors: Vec<HV16> = (0..n).map(|i| HV16::random(i as u64)).collect();
                black_box(vectors)
            })
        });
    }

    // Basis vector generation (used in graph encoding)
    group.bench_function("basis", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            idx = idx.wrapping_add(1);
            black_box(HV16::basis(idx))
        })
    });

    group.finish();
}

// =============================================================================
// HV16::bind() OPERATION BENCHMARKS
// =============================================================================

fn bench_bind_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_bind");
    group.throughput(Throughput::Elements(1));

    let hv1 = HV16::random(42);
    let hv2 = HV16::random(43);

    // Single bind operation (SIMD)
    group.bench_function("simd", |b| {
        b.iter(|| black_box(hv1.bind(&hv2)))
    });

    // Scalar bind for comparison
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(hv1.bind_scalar(&hv2)))
    });

    // Chained binding (A bind B bind C bind D)
    let hv3 = HV16::random(44);
    let hv4 = HV16::random(45);
    group.bench_function("chain_4", |b| {
        b.iter(|| black_box(hv1.bind(&hv2).bind(&hv3).bind(&hv4)))
    });

    // Self-inverse property verification (A bind A = 0)
    group.bench_function("self_inverse", |b| {
        b.iter(|| black_box(hv1.bind(&hv1)))
    });

    // Unbinding (recover B from A_bind_B using A)
    let bound = hv1.bind(&hv2);
    group.bench_function("unbind", |b| {
        b.iter(|| black_box(bound.bind(&hv1)))
    });

    group.finish();
}

// =============================================================================
// HV16::similarity() CALCULATION BENCHMARKS
// =============================================================================

fn bench_similarity_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_similarity");
    group.throughput(Throughput::Elements(1));

    let hv1 = HV16::random(42);
    let hv2 = HV16::random(43);

    // Single similarity calculation (SIMD)
    group.bench_function("simd", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });

    // Scalar similarity for comparison
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(hv1.similarity_scalar(&hv2)))
    });

    // Self-similarity (should be 1.0)
    group.bench_function("self", |b| {
        b.iter(|| black_box(hv1.similarity(&hv1)))
    });

    // Similarity with inverted (should be 0.0)
    let inv = hv1.invert();
    group.bench_function("inverse", |b| {
        b.iter(|| black_box(hv1.similarity(&inv)))
    });

    // Batch similarity (query against codebook)
    let codebook: Vec<HV16> = (0..100).map(|i| HV16::random(i)).collect();
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            let sims: Vec<f32> = codebook.iter().map(|v| hv1.similarity(v)).collect();
            black_box(sims)
        })
    });

    // Find most similar in codebook
    group.bench_function("argmax_100", |b| {
        b.iter(|| {
            let (idx, sim) = codebook
                .iter()
                .enumerate()
                .map(|(i, v)| (i, hv1.similarity(v)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            black_box((idx, sim))
        })
    });

    group.finish();
}

// =============================================================================
// HV16::hamming_distance() BENCHMARKS
// =============================================================================

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_hamming");
    group.throughput(Throughput::Elements(1));

    let hv1 = HV16::random(42);
    let hv2 = HV16::random(43);

    // Single hamming distance (SIMD)
    group.bench_function("simd", |b| {
        b.iter(|| black_box(hv1.hamming_distance(&hv2)))
    });

    // Scalar hamming distance for comparison
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(hv1.hamming_distance_scalar(&hv2)))
    });

    // Self distance (should be 0)
    group.bench_function("self", |b| {
        b.iter(|| black_box(hv1.hamming_distance(&hv1)))
    });

    // Distance to inverted (should be 16384)
    let inv = hv1.invert();
    group.bench_function("inverse", |b| {
        b.iter(|| black_box(hv1.hamming_distance(&inv)))
    });

    // Pairwise distances in small set
    let vectors: Vec<HV16> = (0..10).map(|i| HV16::random(i)).collect();
    group.bench_function("pairwise_10", |b| {
        b.iter(|| {
            let mut total = 0u32;
            for i in 0..vectors.len() {
                for j in (i + 1)..vectors.len() {
                    total += vectors[i].hamming_distance(&vectors[j]);
                }
            }
            black_box(total)
        })
    });

    group.finish();
}

// =============================================================================
// HV16::bundle() OPERATION BENCHMARKS
// =============================================================================

fn bench_bundle_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_bundle");

    // Bundle with varying number of vectors
    for count in [3, 5, 10, 50, 100] {
        let vectors: Vec<HV16> = (0..count).map(|i| HV16::random(i as u64)).collect();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("standard", count), &vectors, |b, vecs| {
            b.iter(|| black_box(HV16::bundle(vecs)))
        });

        group.bench_with_input(BenchmarkId::new("safe", count), &vectors, |b, vecs| {
            b.iter(|| black_box(HV16::bundle_safe(vecs)))
        });

        group.bench_with_input(BenchmarkId::new("normalized", count), &vectors, |b, vecs| {
            b.iter(|| black_box(HV16::bundle_normalized(vecs)))
        });
    }

    // Empty bundle (edge case)
    group.bench_function("empty", |b| {
        b.iter(|| black_box(HV16::bundle(&[])))
    });

    // Single vector bundle (should return same vector)
    let single = HV16::random(42);
    group.bench_function("single", |b| {
        b.iter(|| black_box(HV16::bundle(&[single])))
    });

    group.finish();
}

// =============================================================================
// HV16::permute() OPERATION BENCHMARKS
// =============================================================================

fn bench_permute_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_permute");
    group.throughput(Throughput::Elements(1));

    let hv = HV16::random(42);

    // Various shift amounts
    for shift in [1, 8, 64, 128, 1024, 8192] {
        group.bench_with_input(BenchmarkId::new("fast", shift), &shift, |b, &s| {
            b.iter(|| black_box(hv.permute(s)))
        });

        group.bench_with_input(BenchmarkId::new("legacy", shift), &shift, |b, &s| {
            b.iter(|| black_box(hv.permute_legacy(s)))
        });
    }

    // Zero shift (identity, should be fast)
    group.bench_function("identity", |b| {
        b.iter(|| black_box(hv.permute(0)))
    });

    // Full cycle (shift by DIM)
    group.bench_function("full_cycle", |b| {
        b.iter(|| black_box(hv.permute(HV16::DIM)))
    });

    // Sequence encoding pattern (common use case)
    let hv2 = HV16::random(43);
    let hv3 = HV16::random(44);
    group.bench_function("sequence_3", |b| {
        b.iter(|| {
            // Encode sequence: hv1 at position 0, hv2 at position 1, hv3 at position 2
            let encoded = hv.bind(&hv2.permute(1)).bind(&hv3.permute(2));
            black_box(encoded)
        })
    });

    group.finish();
}

// =============================================================================
// HV16::invert() OPERATION BENCHMARKS
// =============================================================================

fn bench_invert_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_invert");
    group.throughput(Throughput::Elements(1));

    let hv = HV16::random(42);

    // Single invert (SIMD)
    group.bench_function("simd", |b| {
        b.iter(|| black_box(hv.invert()))
    });

    // Scalar invert for comparison
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(hv.invert_scalar()))
    });

    // Double invert (should equal original)
    group.bench_function("double", |b| {
        b.iter(|| black_box(hv.invert().invert()))
    });

    group.finish();
}

// =============================================================================
// AUXILIARY OPERATIONS BENCHMARKS
// =============================================================================

fn bench_auxiliary_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_auxiliary");

    let hv = HV16::random(42);

    // Popcount
    group.bench_function("popcount", |b| {
        b.iter(|| black_box(hv.popcount()))
    });

    // Density calculation
    group.bench_function("density", |b| {
        b.iter(|| black_box(hv.density()))
    });

    // Get single bit
    group.bench_function("get_bit", |b| {
        let mut pos = 0usize;
        b.iter(|| {
            pos = (pos + 1) % HV16::DIM;
            black_box(hv.get_bit(pos))
        })
    });

    // Bipolar conversion (to f32 array)
    group.bench_function("to_bipolar", |b| {
        b.iter(|| black_box(hv.to_bipolar()))
    });

    // From bipolar conversion
    let bipolar = hv.to_bipolar();
    group.bench_function("from_bipolar", |b| {
        b.iter(|| black_box(HV16::from_bipolar(&bipolar)))
    });

    // Ensure density (rebalancing)
    let saturated = HV16::ones();
    group.bench_function("ensure_density", |b| {
        b.iter(|| black_box(saturated.ensure_density(0.4, 0.6)))
    });

    // Add noise
    group.bench_function("add_noise_10pct", |b| {
        b.iter(|| black_box(hv.add_noise(0.1, 123)))
    });

    group.finish();
}

// =============================================================================
// MEMORY AND ALLOCATION BENCHMARKS
// =============================================================================

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_memory");

    // Clone operation
    let hv = HV16::random(42);
    group.bench_function("clone", |b| {
        b.iter(|| black_box(hv.clone()))
    });

    // Default (zero) creation
    group.bench_function("default", |b| {
        b.iter(|| black_box(HV16::default()))
    });

    // Zero creation (const)
    group.bench_function("zero", |b| {
        b.iter(|| black_box(HV16::zero()))
    });

    // Ones creation (const)
    group.bench_function("ones", |b| {
        b.iter(|| black_box(HV16::ones()))
    });

    // From bits
    let bits = vec![0xFFFFFFFFFFFFFFFFu64; 256];
    group.bench_function("from_bits", |b| {
        b.iter(|| black_box(HV16::from_bits(&bits)))
    });

    // Equality check
    let hv2 = HV16::random(42);
    group.bench_function("equality", |b| {
        b.iter(|| black_box(hv == hv2))
    });

    // Hash (for HashMap usage)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    group.bench_function("hash", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            hv.hash(&mut hasher);
            black_box(hasher.finish())
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    hdc_benches,
    bench_random_generation,
    bench_bind_operation,
    bench_similarity_calculation,
    bench_hamming_distance,
    bench_bundle_operation,
    bench_permute_operation,
    bench_invert_operation,
    bench_auxiliary_operations,
    bench_memory_operations,
);

criterion_main!(hdc_benches);
