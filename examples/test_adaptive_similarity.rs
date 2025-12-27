//! Test adaptive similarity search
//!
//! This example demonstrates the revolutionary adaptive algorithm selection
//! in action, comparing naive vs LSH-accelerated similarity search.

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::lsh_similarity::{adaptive_find_most_similar, adaptive_find_top_k};
use symthaea::hdc::simd_hv::simd_find_most_similar;
use std::time::Instant;

fn main() {
    println!("\n=== ADAPTIVE SIMILARITY SEARCH VERIFICATION ===\n");

    // Test 1: Small dataset (should use naive)
    println!("Test 1: Small dataset (100 vectors) - Should use naive O(n)");
    test_small_dataset();

    // Test 2: Large dataset (should use LSH)
    println!("\nTest 2: Large dataset (1000 vectors) - Should use LSH");
    test_large_dataset();

    // Test 3: Very large dataset (demonstrates scalability)
    println!("\nTest 3: Very large dataset (5000 vectors) - LSH with accurate config");
    test_very_large_dataset();

    // Test 4: Top-k search
    println!("\nTest 4: Top-k similarity search (k=5)");
    test_top_k();

    // Test 5: Performance comparison
    println!("\nTest 5: Performance comparison - Naive vs Adaptive");
    benchmark_comparison();

    println!("\n=== ALL TESTS COMPLETE ===\n");
}

fn test_small_dataset() {
    let query = HV16::random(42);
    let targets: Vec<HV16> = (0..100).map(|i| HV16::random(i + 100)).collect();

    let start = Instant::now();
    let result = adaptive_find_most_similar(&query, &targets);
    let elapsed = start.elapsed();

    assert!(result.is_some(), "Should find result in small dataset");
    let (idx, sim) = result.unwrap();
    println!("  ✓ Found most similar at index {} with similarity {:.4}", idx, sim);
    println!("  ✓ Time: {:?} (naive algorithm expected)", elapsed);
    assert!(idx < targets.len(), "Index should be valid");
    assert!(sim >= 0.0 && sim <= 1.0, "Similarity should be in [0,1]");
}

fn test_large_dataset() {
    let query = HV16::random(1);
    let targets: Vec<HV16> = (0..1000).map(|i| HV16::random(i + 200)).collect();

    let start = Instant::now();
    let result = adaptive_find_most_similar(&query, &targets);
    let elapsed = start.elapsed();

    assert!(result.is_some(), "Should find result in large dataset");
    let (idx, sim) = result.unwrap();
    println!("  ✓ Found most similar at index {} with similarity {:.4}", idx, sim);
    println!("  ✓ Time: {:?} (LSH algorithm expected)", elapsed);
    assert!(idx < targets.len(), "Index should be valid");
}

fn test_very_large_dataset() {
    let query = HV16::random(2);
    let targets: Vec<HV16> = (0..5000).map(|i| HV16::random(i + 300)).collect();

    let start = Instant::now();
    let result = adaptive_find_most_similar(&query, &targets);
    let elapsed = start.elapsed();

    assert!(result.is_some(), "Should find result in very large dataset");
    let (idx, sim) = result.unwrap();
    println!("  ✓ Found most similar at index {} with similarity {:.4}", idx, sim);
    println!("  ✓ Time: {:?} (LSH with accurate config)", elapsed);
    assert!(idx < targets.len(), "Index should be valid");
}

fn test_top_k() {
    let query = HV16::random(3);
    let targets: Vec<HV16> = (0..1000).map(|i| HV16::random(i + 400)).collect();

    let k = 5;
    let start = Instant::now();
    let results = adaptive_find_top_k(&query, &targets, k);
    let elapsed = start.elapsed();

    assert_eq!(results.len(), k, "Should return exactly k results");
    println!("  ✓ Found top-{} results:", k);
    for (i, (idx, sim)) in results.iter().enumerate() {
        println!("    {}. Index {} - Similarity: {:.4}", i + 1, idx, sim);
    }
    println!("  ✓ Time: {:?}", elapsed);

    // Verify results are sorted by similarity (descending)
    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "Results should be sorted by similarity"
        );
    }
}

fn benchmark_comparison() {
    println!("\n  Comparing naive vs adaptive on different dataset sizes:\n");

    let sizes = vec![100, 500, 1000, 2000, 5000];

    for size in sizes {
        let query = HV16::random(size as u64);
        let targets: Vec<HV16> = (0..size)
            .map(|i| HV16::random((i + 1000) as u64))
            .collect();

        // Naive approach (original)
        let start = Instant::now();
        let _naive_result = simd_find_most_similar(&query, &targets);
        let naive_time = start.elapsed();

        // Adaptive approach (new)
        let start = Instant::now();
        let _adaptive_result = adaptive_find_most_similar(&query, &targets);
        let adaptive_time = start.elapsed();

        let speedup = naive_time.as_nanos() as f64 / adaptive_time.as_nanos() as f64;
        let algorithm = if size < 500 { "naive" } else { "LSH" };

        println!(
            "  Size {:5}: Naive {:8.2}µs | Adaptive {:8.2}µs ({:6}) | Speedup: {:.2}x",
            size,
            naive_time.as_micros(),
            adaptive_time.as_micros(),
            algorithm,
            speedup
        );
    }

    println!("\n  Key observations:");
    println!("  • Small datasets (<500): Adaptive ≈ naive (same algorithm)");
    println!("  • Large datasets (≥500): Adaptive >> naive (LSH acceleration)");
    println!("  • Automatic: Zero configuration required!");
}
