//! Standalone SIMD Integration Test
//!
//! Tests the SIMD-accelerated HV16 operations with timing measurements.
//! Run with: cargo test --test simd_test -- --nocapture

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::{
    simd_bind, simd_similarity, simd_bundle, simd_hamming_distance,
    simd_find_most_similar, has_avx2, has_sse2, simd_capabilities,
};
use symthaea::hdc::optimized_hv::{
    bundle_optimized, similarity_optimized, bind_optimized,
};
use std::time::Instant;

#[test]
fn test_simd_correctness() {
    println!("\n=== SIMD Correctness Tests ===\n");

    let a = HV16::random(42);
    let b = HV16::random(43);

    // Test bind correctness
    let scalar_bind = a.bind(&b);
    let simd_result = simd_bind(&a, &b);
    assert_eq!(scalar_bind.0, simd_result.0, "SIMD bind must match scalar");
    println!("✓ Bind: SIMD matches scalar");

    // Test similarity correctness
    let scalar_sim = a.similarity(&b);
    let simd_sim = simd_similarity(&a, &b);
    let diff = (scalar_sim - simd_sim).abs();
    assert!(diff < 0.001, "SIMD similarity must match scalar: {} vs {}", scalar_sim, simd_sim);
    println!("✓ Similarity: SIMD matches scalar (diff: {:.6})", diff);

    // Test bundle correctness
    let vectors: Vec<HV16> = (0..10).map(|i| HV16::random(i as u64 + 100)).collect();
    let scalar_bundle = HV16::bundle(&vectors);
    let simd_bundle_result = simd_bundle(&vectors);
    let sim = simd_similarity(&scalar_bundle, &simd_bundle_result);
    assert!(sim > 0.95, "SIMD bundle should match scalar: similarity={}", sim);
    println!("✓ Bundle: SIMD matches scalar (similarity: {:.4})", sim);

    println!("\n✅ All correctness tests passed!\n");
}

#[test]
fn test_simd_performance() {
    println!("\n=== SIMD Performance Tests ===\n");
    println!("{}\n", simd_capabilities());

    let a = HV16::random(1234);
    let b = HV16::random(5678);

    // Warm-up
    for _ in 0..1000 {
        let _ = a.bind(&b);
        let _ = simd_bind(&a, &b);
    }

    // =========================================================================
    // BIND BENCHMARK
    // =========================================================================
    println!("--- Bind Operation ---");

    // Original bind
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(a.bind(std::hint::black_box(&b)));
    }
    let original_time = start.elapsed();
    let original_ns = original_time.as_nanos() / 100_000;

    // SIMD bind
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(simd_bind(std::hint::black_box(&a), std::hint::black_box(&b)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 100_000;

    let speedup = original_ns as f64 / simd_ns as f64;
    println!("  Original: {} ns/op", original_ns);
    println!("  SIMD:     {} ns/op", simd_ns);
    println!("  Speedup:  {:.1}x", speedup);
    println!("  Target:   <10ns | Status: {}",
             if simd_ns <= 10 { "✓ ACHIEVED" } else { "⚠ Not yet" });

    // =========================================================================
    // SIMILARITY BENCHMARK
    // =========================================================================
    println!("\n--- Similarity Operation ---");

    // Original similarity
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(a.similarity(std::hint::black_box(&b)));
    }
    let original_time = start.elapsed();
    let original_ns = original_time.as_nanos() / 100_000;

    // SIMD similarity
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(simd_similarity(std::hint::black_box(&a), std::hint::black_box(&b)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 100_000;

    let speedup = original_ns as f64 / simd_ns as f64;
    println!("  Original: {} ns/op", original_ns);
    println!("  SIMD:     {} ns/op", simd_ns);
    println!("  Speedup:  {:.1}x", speedup);
    println!("  Target:   <25ns | Status: {}",
             if simd_ns <= 25 { "✓ ACHIEVED" } else { "⚠ Not yet" });

    // =========================================================================
    // BUNDLE BENCHMARK
    // =========================================================================
    println!("\n--- Bundle Operation (10 vectors) ---");

    let vectors: Vec<HV16> = (0..10).map(|i| HV16::random(i as u64 + 200)).collect();

    // Original bundle
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = std::hint::black_box(HV16::bundle(std::hint::black_box(&vectors)));
    }
    let original_time = start.elapsed();
    let original_us = original_time.as_micros() / 10_000;
    let original_ns = original_time.as_nanos() / 10_000;

    // Optimized bundle
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = std::hint::black_box(bundle_optimized(std::hint::black_box(&vectors)));
    }
    let optimized_time = start.elapsed();
    let optimized_ns = optimized_time.as_nanos() / 10_000;

    // SIMD bundle
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = std::hint::black_box(simd_bundle(std::hint::black_box(&vectors)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 10_000;
    let simd_us = simd_ns / 1000;

    let speedup = original_ns as f64 / simd_ns as f64;
    println!("  Original:  {} ns ({} µs)", original_ns, original_us);
    println!("  Optimized: {} ns", optimized_ns);
    println!("  SIMD:      {} ns ({} µs)", simd_ns, simd_us);
    println!("  Speedup:   {:.1}x", speedup);
    println!("  Target:    <5µs | Status: {}",
             if simd_us <= 5 { "✓ ACHIEVED" } else { "⚠ Not yet" });

    // =========================================================================
    // HAMMING DISTANCE BENCHMARK
    // =========================================================================
    println!("\n--- Hamming Distance ---");

    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = std::hint::black_box(simd_hamming_distance(std::hint::black_box(&a), std::hint::black_box(&b)));
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / 100_000;
    println!("  SIMD Hamming: {} ns/op", simd_ns);

    // =========================================================================
    // SIMILARITY SEARCH BENCHMARK
    // =========================================================================
    println!("\n--- Similarity Search (100 vectors) ---");

    let memory: Vec<HV16> = (0..100).map(|i| HV16::random(i as u64 + 300)).collect();
    let query = HV16::random(999);

    // Linear search
    let start = Instant::now();
    for _ in 0..1_000 {
        let mut best_idx = 0;
        let mut best_sim = f32::MIN;
        for (idx, v) in memory.iter().enumerate() {
            let sim = v.similarity(&query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }
        std::hint::black_box((best_idx, best_sim));
    }
    let linear_time = start.elapsed();
    let linear_us = linear_time.as_micros() / 1_000;

    // SIMD search
    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = std::hint::black_box(simd_find_most_similar(std::hint::black_box(&query), std::hint::black_box(&memory)));
    }
    let simd_time = start.elapsed();
    let simd_us = simd_time.as_micros() / 1_000;

    let speedup = linear_us as f64 / simd_us as f64;
    println!("  Linear: {} µs", linear_us);
    println!("  SIMD:   {} µs", simd_us);
    println!("  Speedup: {:.1}x", speedup);

    println!("\n✅ Performance tests complete!\n");
}

#[test]
fn test_simd_capabilities_report() {
    println!("\n=== CPU SIMD Capabilities ===");
    println!("AVX2: {}", if has_avx2() { "✓ Available" } else { "✗ Not available" });
    println!("SSE2: {}", if has_sse2() { "✓ Available" } else { "✗ Not available" });
    println!("\n{}\n", simd_capabilities());
}
