//! SIMD Integration Tests
//!
//! Tests the SIMD-accelerated SimdHV16 operations with timing measurements.
//! Run with: cargo test --test simd_test -- --nocapture

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv16::{SimdHV16, batch_similarity, top_k_similar};
use std::time::Instant;

// ============================================================================
// CORRECTNESS TESTS
// ============================================================================

#[test]
fn test_simd_bind_correctness() {
    println!("\n=== SIMD Bind Correctness ===\n");

    let a = SimdHV16::random(42);
    let b = SimdHV16::random(43);

    // Test bind correctness via conversion to HV16
    let result = a.bind(&b);
    let a_hv16: HV16 = (&a).into();
    let b_hv16: HV16 = (&b).into();
    let result_via_hv16 = a_hv16.bind(&b_hv16);
    let result_hv16: HV16 = (&result).into();

    assert_eq!(result_hv16.0, result_via_hv16.0, "SIMD bind must match HV16 bind");
    println!("  SimdHV16.bind() matches HV16.bind()");
}

#[test]
fn test_simd_similarity_correctness() {
    println!("\n=== SIMD Similarity Correctness ===\n");

    let a = SimdHV16::random(42);
    let b = SimdHV16::random(43);

    let simd_sim = a.similarity(&b);

    // Compare with HV16 similarity
    let a_hv16: HV16 = (&a).into();
    let b_hv16: HV16 = (&b).into();
    let hv16_sim = a_hv16.similarity(&b_hv16);

    let diff = (simd_sim - hv16_sim).abs();
    assert!(diff < 0.001, "Similarity mismatch: {} vs {} (diff: {})", simd_sim, hv16_sim, diff);
    println!("  SimdHV16.similarity() matches HV16.similarity() (diff: {:.6})", diff);
}

#[test]
fn test_simd_bundle_correctness() {
    println!("\n=== SIMD Bundle Correctness ===\n");

    let vectors: Vec<SimdHV16> = (0..10).map(|i| SimdHV16::random(i + 100)).collect();
    let simd_bundle = SimdHV16::bundle(&vectors);

    // Compare with HV16 bundle
    let hv16_vectors: Vec<HV16> = vectors.iter().map(|v| v.into()).collect();
    let hv16_bundle = HV16::bundle(&hv16_vectors);
    let simd_bundle_as_hv16: HV16 = (&simd_bundle).into();

    let sim = simd_bundle_as_hv16.similarity(&hv16_bundle);
    assert!(sim > 0.95, "Bundle mismatch: similarity = {}", sim);
    println!("  SimdHV16::bundle() matches HV16::bundle() (similarity: {:.4})", sim);
}

#[test]
fn test_simd_hamming_distance() {
    println!("\n=== SIMD Hamming Distance ===\n");

    let a = SimdHV16::random(1);
    let b = SimdHV16::random(2);

    let hamming = a.hamming_distance(&b);

    // Convert to HV16 and verify
    let a_hv16: HV16 = (&a).into();
    let b_hv16: HV16 = (&b).into();

    // Calculate expected hamming from similarity
    // similarity = (DIM - hamming) / DIM
    let expected_sim = (16384 - hamming) as f32 / 16384.0;
    let actual_sim = a_hv16.similarity(&b_hv16);

    let diff = (expected_sim - actual_sim).abs();
    assert!(diff < 0.001, "Hamming distance inconsistent with similarity: {}", diff);
    println!("  Hamming distance: {} (consistent with similarity)", hamming);
}

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

#[test]
fn test_simd_bind_performance() {
    println!("\n=== SIMD Bind Performance ===\n");

    let a = SimdHV16::random(1234);
    let b = SimdHV16::random(5678);

    // Warm-up
    for _ in 0..1000 {
        let _ = a.bind(&b);
    }

    // Benchmark SimdHV16
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(std::hint::black_box(&a).bind(std::hint::black_box(&b)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 100_000;

    // Benchmark HV16 for comparison
    let a_hv16: HV16 = (&a).into();
    let b_hv16: HV16 = (&b).into();

    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(std::hint::black_box(&a_hv16).bind(std::hint::black_box(&b_hv16)));
    }
    let hv16_time = start.elapsed();
    let hv16_ns = hv16_time.as_nanos() / 100_000;

    let speedup = hv16_ns as f64 / simd_ns as f64;
    println!("  HV16:     {} ns/op", hv16_ns);
    println!("  SimdHV16: {} ns/op", simd_ns);
    println!("  Speedup:  {:.1}x", speedup);
    println!("  Target:   <50ns | Status: {}", if simd_ns <= 50 { "ACHIEVED" } else { "Not yet" });
}

#[test]
fn test_simd_similarity_performance() {
    println!("\n=== SIMD Similarity Performance ===\n");

    let a = SimdHV16::random(1234);
    let b = SimdHV16::random(5678);

    // Warm-up
    for _ in 0..1000 {
        let _ = a.similarity(&b);
    }

    // Benchmark SimdHV16
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(std::hint::black_box(&a).similarity(std::hint::black_box(&b)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 100_000;

    // Benchmark HV16 for comparison
    let a_hv16: HV16 = (&a).into();
    let b_hv16: HV16 = (&b).into();

    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(std::hint::black_box(&a_hv16).similarity(std::hint::black_box(&b_hv16)));
    }
    let hv16_time = start.elapsed();
    let hv16_ns = hv16_time.as_nanos() / 100_000;

    let speedup = hv16_ns as f64 / simd_ns as f64;
    println!("  HV16:     {} ns/op", hv16_ns);
    println!("  SimdHV16: {} ns/op", simd_ns);
    println!("  Speedup:  {:.1}x", speedup);
    println!("  Target:   <125ns | Status: {}", if simd_ns <= 125 { "ACHIEVED" } else { "Not yet" });
}

#[test]
fn test_simd_bundle_performance() {
    println!("\n=== SIMD Bundle Performance (10 vectors) ===\n");

    let vectors: Vec<SimdHV16> = (0..10).map(|i| SimdHV16::random(i + 200)).collect();

    // Warm-up
    for _ in 0..100 {
        let _ = SimdHV16::bundle(&vectors);
    }

    // Benchmark SimdHV16
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = std::hint::black_box(SimdHV16::bundle(std::hint::black_box(&vectors)));
    }
    let simd_time = start.elapsed();
    let simd_us = simd_time.as_micros() / 10_000;
    let simd_ns = simd_time.as_nanos() / 10_000;

    // Benchmark HV16 for comparison
    let hv16_vectors: Vec<HV16> = vectors.iter().map(|v| v.into()).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = std::hint::black_box(HV16::bundle(std::hint::black_box(&hv16_vectors)));
    }
    let hv16_time = start.elapsed();
    let hv16_us = hv16_time.as_micros() / 10_000;
    let hv16_ns = hv16_time.as_nanos() / 10_000;

    let speedup = hv16_ns as f64 / simd_ns as f64;
    println!("  HV16:     {} us ({} ns)", hv16_us, hv16_ns);
    println!("  SimdHV16: {} us ({} ns)", simd_us, simd_ns);
    println!("  Speedup:  {:.1}x", speedup);
    println!("  Target:   <1us | Status: {}", if simd_us <= 1 { "ACHIEVED" } else { "Not yet" });
}

#[test]
fn test_simd_batch_similarity_performance() {
    println!("\n=== Batch Similarity Performance (100 vectors) ===\n");

    let query = SimdHV16::random(999);
    let targets: Vec<SimdHV16> = (0..100).map(|i| SimdHV16::random(i + 300)).collect();

    // Warm-up
    for _ in 0..10 {
        let _ = batch_similarity(&query, &targets);
    }

    // Benchmark batch similarity
    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = std::hint::black_box(batch_similarity(std::hint::black_box(&query), std::hint::black_box(&targets)));
    }
    let batch_time = start.elapsed();
    let batch_us = batch_time.as_micros() / 1_000;

    println!("  Batch similarity (100 vectors): {} us", batch_us);
    println!("  Per-comparison: {:.1} ns", (batch_us as f64 * 1000.0) / 100.0);
}

#[test]
fn test_simd_top_k_performance() {
    println!("\n=== Top-K Search Performance (100 vectors, k=5) ===\n");

    let query = SimdHV16::random(999);
    let targets: Vec<SimdHV16> = (0..100).map(|i| SimdHV16::random(i + 400)).collect();

    // Warm-up
    for _ in 0..10 {
        let _ = top_k_similar(&query, &targets, 5);
    }

    // Benchmark top-k
    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = std::hint::black_box(top_k_similar(std::hint::black_box(&query), std::hint::black_box(&targets), 5));
    }
    let topk_time = start.elapsed();
    let topk_us = topk_time.as_micros() / 1_000;

    let result = top_k_similar(&query, &targets, 5);
    println!("  Top-5 search (100 vectors): {} us", topk_us);
    println!("  Best match: index {} with similarity {:.4}", result[0].0, result[0].1);
}

// ============================================================================
// SIMD CAPABILITY REPORT
// ============================================================================

#[test]
fn test_simd_capabilities_report() {
    println!("\n=== SIMD Capabilities ===\n");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  Architecture: x86_64");
        println!("  AVX2: {}", if is_x86_feature_detected!("avx2") { "Available" } else { "Not available" });
        println!("  AVX:  {}", if is_x86_feature_detected!("avx") { "Available" } else { "Not available" });
        println!("  SSE4.2: {}", if is_x86_feature_detected!("sse4.2") { "Available" } else { "Not available" });
        println!("  SSE4.1: {}", if is_x86_feature_detected!("sse4.1") { "Available" } else { "Not available" });
        println!("  SSE2: {}", if is_x86_feature_detected!("sse2") { "Available" } else { "Not available" });
        println!("  POPCNT: {}", if is_x86_feature_detected!("popcnt") { "Available" } else { "Not available" });
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  Architecture: aarch64 (ARM64)");
        println!("  NEON: Available (standard on ARM64)");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  Architecture: Other");
        println!("  SIMD: Using fallback implementations");
    }

    println!("\n  SimdHV16 dimension: {} bits", SimdHV16::DIM);
    println!("  SimdHV16 word count: {} (u64)", SimdHV16::WORDS);
}
