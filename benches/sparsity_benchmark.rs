/*!
Sparsity Measurement Benchmark

CRITICAL QUESTION: Are our HV16 vectors actually sparse?

If yes (>70% zeros): Sparse representations could give 10-100x speedup!
If no (<30% zeros): Dense SIMD is optimal, proceed to GPU.

This benchmark measures REAL sparsity in consciousness operations:
1. Random vectors (baseline - should be ~50% ones)
2. Bundled vectors (semantic combination - might be sparse)
3. Permuted vectors (after transformation)
4. Similarity-filtered vectors (after thresholding)
5. Real consciousness cycle vectors (after operations)

Results will inform whether to implement sparse representations.
*/

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use symthaea::hdc::binary_hv::HV16;
use std::collections::HashMap;

// =============================================================================
// Sparsity Measurement Utilities
// =============================================================================

/// Measure sparsity of a vector (percentage of zeros)
fn measure_sparsity(hv: &HV16) -> f64 {
    let data = &hv.0;  // Access inner [u8; 256] array directly
    let total_bits = data.len() * 8; // 2048 bits

    let mut one_count = 0;
    for &byte in data {
        one_count += byte.count_ones() as usize;
    }

    let zero_count = total_bits - one_count;
    (zero_count as f64 / total_bits as f64) * 100.0
}

/// Measure sparsity distribution across many vectors
fn measure_sparsity_distribution(vectors: &[HV16]) -> HashMap<String, f64> {
    let mut stats = HashMap::new();

    let sparsities: Vec<f64> = vectors.iter()
        .map(|v| measure_sparsity(v))
        .collect();

    let sum: f64 = sparsities.iter().sum();
    let mean = sum / sparsities.len() as f64;

    let variance = sparsities.iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f64>() / sparsities.len() as f64;
    let std_dev = variance.sqrt();

    let min = sparsities.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = sparsities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    stats.insert("mean".to_string(), mean);
    stats.insert("std_dev".to_string(), std_dev);
    stats.insert("min".to_string(), min);
    stats.insert("max".to_string(), max);

    stats
}

// =============================================================================
// Test 1: Random Vectors (Baseline)
// =============================================================================

fn measure_random_sparsity(c: &mut Criterion) {
    c.bench_function("sparsity_random_vectors", |b| {
        b.iter(|| {
            let vectors: Vec<HV16> = (0..1000)
                .map(|i| HV16::random(i as u64))
                .collect();

            let stats = measure_sparsity_distribution(&vectors);
            black_box(stats)
        })
    });

    // Also measure outside benchmark for reporting
    let vectors: Vec<HV16> = (0..1000)
        .map(|i| HV16::random(i as u64))
        .collect();
    let stats = measure_sparsity_distribution(&vectors);

    println!("\n📊 Random Vectors Sparsity:");
    println!("  Mean: {:.2}% zeros", stats.get("mean").unwrap());
    println!("  Std Dev: {:.2}%", stats.get("std_dev").unwrap());
    println!("  Range: {:.2}% - {:.2}%", stats.get("min").unwrap(), stats.get("max").unwrap());
    println!("  Expected: ~50% (random should have 50% ones, 50% zeros)");
}

// =============================================================================
// Test 2: Bundled Vectors (Semantic Combination)
// =============================================================================

fn measure_bundled_sparsity(c: &mut Criterion) {
    c.bench_function("sparsity_bundled_vectors", |b| {
        b.iter(|| {
            let mut bundled_vectors = Vec::new();

            // Create 100 bundles of varying sizes
            for bundle_size in [2, 5, 10, 20, 50, 100].iter() {
                for _trial in 0..20 {
                    let components: Vec<HV16> = (0..*bundle_size)
                        .map(|i| HV16::random((i * 1000) as u64))
                        .collect();

                    let bundled = HV16::bundle(&components);
                    bundled_vectors.push(bundled);
                }
            }

            let stats = measure_sparsity_distribution(&bundled_vectors);
            black_box(stats)
        })
    });

    // Measure outside benchmark
    let mut bundled_vectors = Vec::new();
    for bundle_size in [2, 5, 10, 20, 50, 100].iter() {
        for trial in 0..20 {
            let components: Vec<HV16> = (0..*bundle_size)
                .map(|i| HV16::random((i * 1000 + trial) as u64))
                .collect();
            bundled_vectors.push(HV16::bundle(&components));
        }
    }

    let stats = measure_sparsity_distribution(&bundled_vectors);
    println!("\n📊 Bundled Vectors Sparsity:");
    println!("  Mean: {:.2}% zeros", stats.get("mean").unwrap());
    println!("  Std Dev: {:.2}%", stats.get("std_dev").unwrap());
    println!("  Range: {:.2}% - {:.2}%", stats.get("min").unwrap(), stats.get("max").unwrap());
    println!("  Hypothesis: Bundling might increase sparsity (voting → majority 0s or 1s)");
    println!("  Sparse if: >70% zeros");
}

// =============================================================================
// Test 3: Bound Vectors (XOR Operations)
// =============================================================================

fn measure_bound_sparsity(c: &mut Criterion) {
    c.bench_function("sparsity_bound_vectors", |b| {
        b.iter(|| {
            let mut bound_vectors = Vec::new();

            for i in 0..1000 {
                let v1 = HV16::random(i as u64);
                let v2 = HV16::random((i + 1000) as u64);
                let bound = v1.bind(&v2);
                bound_vectors.push(bound);
            }

            let stats = measure_sparsity_distribution(&bound_vectors);
            black_box(stats)
        })
    });

    // Measure outside benchmark
    let mut bound_vectors = Vec::new();
    for i in 0..1000 {
        let v1 = HV16::random(i as u64);
        let v2 = HV16::random((i + 1000) as u64);
        bound_vectors.push(v1.bind(&v2));
    }

    let stats = measure_sparsity_distribution(&bound_vectors);
    println!("\n📊 Bound (XOR) Vectors Sparsity:");
    println!("  Mean: {:.2}% zeros", stats.get("mean").unwrap());
    println!("  Std Dev: {:.2}%", stats.get("std_dev").unwrap());
    println!("  Range: {:.2}% - {:.2}%", stats.get("min").unwrap(), stats.get("max").unwrap());
    println!("  Expected: ~50% (XOR of random vectors → random)");
}

// =============================================================================
// Test 4: Permuted Vectors
// =============================================================================

fn measure_permuted_sparsity(c: &mut Criterion) {
    c.bench_function("sparsity_permuted_vectors", |b| {
        b.iter(|| {
            let mut permuted_vectors = Vec::new();

            for i in 0..1000 {
                let v = HV16::random(i as u64);
                let permuted = v.permute(42);
                permuted_vectors.push(permuted);
            }

            let stats = measure_sparsity_distribution(&permuted_vectors);
            black_box(stats)
        })
    });

    // Measure outside benchmark
    let mut permuted_vectors = Vec::new();
    for i in 0..1000 {
        let v = HV16::random(i as u64);
        permuted_vectors.push(v.permute(42));
    }

    let stats = measure_sparsity_distribution(&permuted_vectors);
    println!("\n📊 Permuted Vectors Sparsity:");
    println!("  Mean: {:.2}% zeros", stats.get("mean").unwrap());
    println!("  Std Dev: {:.2}%", stats.get("std_dev").unwrap());
    println!("  Range: {:.2}% - {:.2}%", stats.get("min").unwrap(), stats.get("max").unwrap());
    println!("  Expected: ~50% (permutation preserves bit ratios)");
}

// =============================================================================
// Test 5: Realistic Consciousness Cycle
// =============================================================================

fn measure_consciousness_cycle_sparsity(c: &mut Criterion) {
    c.bench_function("sparsity_consciousness_cycle", |b| {
        b.iter(|| {
            // Simulate consciousness cycle operations
            let concepts: Vec<HV16> = (0..100)
                .map(|i| HV16::random(i as u64))
                .collect();

            let memories: Vec<HV16> = (0..1000)
                .map(|i| HV16::random((i + 1000) as u64))
                .collect();

            // Bundle concepts
            let query = HV16::bundle(&concepts);

            // Bind query with context
            let context = HV16::random(9999);
            let contextualized = query.bind(&context);

            // Permute for different semantic roles
            let permuted = contextualized.permute(7);

            // Collect all intermediate vectors
            let mut all_vectors = vec![query, contextualized, permuted];
            all_vectors.extend(concepts);
            all_vectors.extend(memories);

            let stats = measure_sparsity_distribution(&all_vectors);
            black_box(stats)
        })
    });

    // Detailed measurement
    let concepts: Vec<HV16> = (0..100)
        .map(|i| HV16::random(i as u64))
        .collect();

    let memories: Vec<HV16> = (0..1000)
        .map(|i| HV16::random((i + 1000) as u64))
        .collect();

    let query = HV16::bundle(&concepts);
    let context = HV16::random(9999);
    let contextualized = query.bind(&context);
    let permuted = contextualized.permute(7);

    println!("\n📊 Consciousness Cycle Vectors Sparsity:");
    println!("  Concepts (raw): {:.2}% zeros", measure_sparsity(&concepts[0]));
    println!("  Query (bundled): {:.2}% zeros", measure_sparsity(&query));
    println!("  Contextualized (bound): {:.2}% zeros", measure_sparsity(&contextualized));
    println!("  Permuted: {:.2}% zeros", measure_sparsity(&permuted));
    println!("  Memories (raw): {:.2}% zeros", measure_sparsity(&memories[0]));
}

// =============================================================================
// Benchmark Groups
// =============================================================================

criterion_group! {
    name = sparsity_benches;
    config = Criterion::default()
        .sample_size(10);  // Small sample size, we care about distribution
    targets =
        measure_random_sparsity,
        measure_bundled_sparsity,
        measure_bound_sparsity,
        measure_permuted_sparsity,
        measure_consciousness_cycle_sparsity,
}

criterion_main!(sparsity_benches);

// =============================================================================
// Expected Results & Decision Logic
// =============================================================================

/*
DECISION CRITERIA:

IF mean sparsity > 70%:
  → **Implement sparse representations!**
  → Expected speedup: 10-100x
  → Store only non-zero bit positions
  → Sparse operations on compressed format

IF mean sparsity 40-70%:
  → **Hybrid approach**
  → Dense for most operations (SIMD fast)
  → Sparse for large vector stores (memory savings)
  → Case-by-case decision

IF mean sparsity < 40%:
  → **Dense is optimal**
  → SIMD operations already fast
  → Sparse overhead would slow things down
  → Proceed to GPU acceleration

HYPOTHESIS:
- Random vectors: ~50% sparsity (1024 ones, 1024 zeros)
- Bundled vectors: Might be slightly more sparse (majority voting)
- Bound vectors: ~50% sparsity (XOR preserves randomness)
- Permuted: Same as input
- Real cycle: Depends on semantic structure

REALITY: Will be discovered after running this benchmark!
*/
