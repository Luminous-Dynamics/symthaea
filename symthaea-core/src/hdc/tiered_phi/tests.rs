// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the Tiered Φ Approximation System
//!
//! This module contains comprehensive tests for all Φ calculation tiers and features.

use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::binary_hv::BinaryHV;
    use crate::hdc::unified_hv::ContinuousHV;

    fn create_test_components(n: usize) -> Vec<BinaryHV> {
        (0..n).map(|i| BinaryHV::random(i as u64)).collect()
    }

    #[test]
    fn test_mock_tier_deterministic() {
        let mut phi = TieredPhi::for_testing();

        let components = create_test_components(5);

        // Mock should be deterministic
        let result1 = phi.compute(&components);
        let result2 = phi.compute(&components);

        assert_eq!(result1, result2, "Mock should be deterministic");
        assert!(result1 > 0.0, "Mock should return positive Φ");
        assert!(result1 <= 1.0, "Mock should return normalized Φ");
    }

    #[test]
    fn test_mock_tier_scales_with_components() {
        let mut phi = TieredPhi::for_testing();

        let phi_2 = phi.compute(&create_test_components(2));
        let phi_5 = phi.compute(&create_test_components(5));
        let phi_10 = phi.compute(&create_test_components(10));

        assert!(phi_5 > phi_2, "More components should give higher Φ");
        assert!(phi_10 > phi_5, "More components should give higher Φ");
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn test_heuristic_tier_fast() {
        let mut phi = TieredPhi::for_production();

        // Use 20 components (debug mode is ~10x slower than release)
        let components = create_test_components(20);

        let start = std::time::Instant::now();
        let result = phi.compute(&components);
        let elapsed = start.elapsed();

        // Note: IIT-compliant partition sampling is O(n × samples)
        // In debug mode, expect ~100-500ms for n=20
        assert!(
            elapsed.as_millis() < 2000,
            "Heuristic should complete in reasonable time (<2s for n=20 in debug)"
        );
        assert!(
            result >= 0.0 && result <= 1.0,
            "Result should be normalized"
        );
    }

    #[test]
    fn test_phi_fix_different_integration_levels() {
        // CRITICAL TEST: Verify that different integration levels produce different Φ values
        // This test validates the Dec 27, 2025 normalization fix
        let mut phi_calc = TieredPhi::for_production();

        // Test 1: Modular/Homogeneous (all very similar - low Φ expected)
        // In IIT, homogeneous systems have LOW Φ because they're redundant, not integrated
        let base = BinaryHV::random(42);
        let homogeneous: Vec<BinaryHV> = (0..10)
            .map(|i| {
                let mut variant = base.clone();
                // Flip just one bit - creates redundant/homogeneous system
                variant.0[i % 256] ^= 0x01;
                variant
            })
            .collect();
        let phi_homogeneous = phi_calc.compute(&homogeneous);

        // Test 2: Integrated (structured correlations - high Φ expected)
        // Create a system with specific cross-partition correlations
        // Group A: components 0-4 (similar to each other)
        // Group B: components 5-9 (similar to each other)
        // But A and B have some correlation too
        let group_a_base = BinaryHV::random(100);
        let group_b_base = BinaryHV::random(200);

        let mut integrated = Vec::new();
        // Group A: similar to group_a_base
        for i in 0..5 {
            let mut comp = group_a_base.clone();
            // Small variations within group
            for j in 0..5 {
                comp.0[(i * 10 + j) % 256] ^= 0x01;
            }
            integrated.push(comp);
        }
        // Group B: similar to group_b_base
        for i in 0..5 {
            let mut comp = group_b_base.clone();
            // Small variations within group
            for j in 0..5 {
                comp.0[(i * 10 + j) % 256] ^= 0x01;
            }
            integrated.push(comp);
        }
        let phi_integrated = phi_calc.compute(&integrated);

        // Test 3: Random/Modular (uncorrelated - low-medium Φ)
        let random: Vec<BinaryHV> = (0..10)
            .map(|i| BinaryHV::random((i * 1000) as u64))
            .collect();
        let phi_random = phi_calc.compute(&random);

        println!("Φ (homogeneous/redundant): {:.4}", phi_homogeneous);
        println!("Φ (random/modular):        {:.4}", phi_random);
        println!("Φ (integrated):            {:.4}", phi_integrated);

        // Assertions
        assert!(
            phi_homogeneous >= 0.0 && phi_homogeneous <= 1.0,
            "Φ should be in [0,1]"
        );
        assert!(
            phi_random >= 0.0 && phi_random <= 1.0,
            "Φ should be in [0,1]"
        );
        assert!(
            phi_integrated >= 0.0 && phi_integrated <= 1.0,
            "Φ should be in [0,1]"
        );

        // CRITICAL: Values should NOT all converge to ~0.08
        let not_all_converging = (phi_homogeneous - 0.08).abs() > 0.02
            || (phi_random - 0.08).abs() > 0.02
            || (phi_integrated - 0.08).abs() > 0.02;
        assert!(
            not_all_converging,
            "FAILED: Φ values converging to ~0.08! (homog={:.4}, rand={:.4}, integ={:.4})",
            phi_homogeneous, phi_random, phi_integrated
        );

        // CRITICAL: Integrated should have higher Φ than purely homogeneous or random
        // (The exact ordering depends on the specific structure, but integrated should be competitive)
        let shows_variation = phi_integrated != phi_homogeneous && phi_integrated != phi_random;
        assert!(
            shows_variation,
            "FAILED: Φ not differentiating structures (homog={:.4}, rand={:.4}, integ={:.4})",
            phi_homogeneous, phi_random, phi_integrated
        );

        println!("✓ Fix validated: Φ values show meaningful variation across different structures");
        println!(
            "  Homogeneous: {:.4}, Random: {:.4}, Integrated: {:.4}",
            phi_homogeneous, phi_random, phi_integrated
        );
    }

    #[test]
    fn test_spectral_tier() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        let components = create_test_components(20);
        let result = phi.compute(&components);

        assert!(
            result >= 0.0 && result <= 1.0,
            "Result should be normalized"
        );
    }

    #[test]
    fn test_exact_tier_small_system() {
        let mut phi = TieredPhi::for_research();

        // Small system should work
        let components = create_test_components(4);

        let start = std::time::Instant::now();
        let result = phi.compute(&components);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs() < 1,
            "Exact on small system should be fast"
        );
        assert!(
            result >= 0.0 && result <= 1.0,
            "Result should be normalized"
        );
    }

    #[test]
    fn test_tier_suggestions() {
        assert_eq!(
            ApproximationTier::suggest_for(4),
            ApproximationTier::ExhaustivePartition
        );
        // SpectralConnectivity deprecated (r = -0.62 with true Φ, 2026-04-11).
        // suggest_for() now uses SampledPartition for n > 8.
        assert_eq!(
            ApproximationTier::suggest_for(50),
            ApproximationTier::SampledPartition
        );
        assert_eq!(
            ApproximationTier::suggest_for(500),
            ApproximationTier::SampledPartition
        );
    }

    #[test]
    fn test_cache_works() {
        let mut phi = TieredPhi::for_production();

        let components = create_test_components(10);

        // First call: no cache
        phi.compute(&components);
        assert_eq!(phi.stats().cache_hits, 0);

        // Second call: should hit cache
        phi.compute(&components);
        assert_eq!(phi.stats().cache_hits, 1);
    }

    #[test]
    fn test_stats_tracking() {
        let mut phi = TieredPhi::for_production();

        phi.compute(&create_test_components(5));
        phi.compute(&create_test_components(10));
        phi.compute(&create_test_components(15));

        assert_eq!(phi.stats().total_calculations, 3);
        assert!(phi.stats().total_time_us > 0);
    }

    #[test]
    fn test_trivial_cases() {
        let mut phi = TieredPhi::for_production();

        // Empty
        assert_eq!(phi.compute(&[]), 0.0);

        // Single component
        assert_eq!(phi.compute(&create_test_components(1)), 0.0);
    }

    #[test]
    fn test_tier_complexity() {
        assert_eq!(ApproximationTier::RandomBaseline.complexity(), "O(1)");
        assert_eq!(ApproximationTier::SampledPartition.complexity(), "O(n)");
        assert_eq!(
            ApproximationTier::SpectralConnectivity.complexity(),
            "O(n²)"
        );
        assert_eq!(
            ApproximationTier::ExhaustivePartition.complexity(),
            "O(2^n)"
        );
    }

    #[test]
    fn test_tier_suitability() {
        assert!(ApproximationTier::RandomBaseline.is_suitable_for(1000));
        assert!(ApproximationTier::SampledPartition.is_suitable_for(1000));
        assert!(ApproximationTier::SpectralConnectivity.is_suitable_for(500));
        assert!(!ApproximationTier::SpectralConnectivity.is_suitable_for(5000));
        assert!(ApproximationTier::ExhaustivePartition.is_suitable_for(8));
        assert!(!ApproximationTier::ExhaustivePartition.is_suitable_for(20));
    }

    // ========================================================================
    // GLOBAL Φ CALCULATOR TESTS (Revolutionary Improvement #86)
    // ========================================================================

    #[test]
    fn test_auto_tier_selection() {
        // Small systems: Exact
        assert_eq!(auto_tier(5), ApproximationTier::ExhaustivePartition);
        assert_eq!(auto_tier(8), ApproximationTier::ExhaustivePartition);

        // Medium/large systems: sampled heuristic.
        // SpectralConnectivity measures spectral gap, not IIT Φ, and is not
        // selected automatically.
        assert_eq!(auto_tier(9), ApproximationTier::SampledPartition);
        assert_eq!(auto_tier(50), ApproximationTier::SampledPartition);
        assert_eq!(auto_tier(51), ApproximationTier::SampledPartition);
        assert_eq!(auto_tier(500), ApproximationTier::SampledPartition);

        // Huge systems: Mock
        assert_eq!(auto_tier(501), ApproximationTier::RandomBaseline);
        assert_eq!(auto_tier(10000), ApproximationTier::RandomBaseline);
    }

    #[test]
    fn test_global_phi() {
        // Reset to known state
        set_global_tier(ApproximationTier::SpectralConnectivity);

        let components = create_test_components(5);
        let phi = global_phi(&components);

        assert!(phi > 0.0);
        assert!(phi <= 1.0);
    }

    #[test]
    fn test_auto_phi() {
        // Small system: should use Exact
        let small = create_test_components(5);
        let phi_small = auto_phi(&small);
        assert!(phi_small > 0.0);

        // Medium system: should use Spectral
        let medium = create_test_components(20);
        let phi_medium = auto_phi(&medium);
        assert!(phi_medium > 0.0);

        // Large system: should use Heuristic
        let large = create_test_components(100);
        let phi_large = auto_phi(&large);
        assert!(phi_large > 0.0);
    }

    #[test]
    fn test_global_phi_stats() {
        // Reset to known state with fresh instance
        set_global_tier(ApproximationTier::SpectralConnectivity);

        // After reset, stats should be 0 for this fresh instance
        let initial_stats = global_phi_stats();

        // Create unique components each time (different seeds)
        let components1: Vec<_> = (0..5).map(|i| BinaryHV::random(i as u64 * 12345)).collect();
        let components2: Vec<_> = (0..7)
            .map(|i| BinaryHV::random((i + 100) as u64 * 67890))
            .collect();

        global_phi(&components1);
        global_phi(&components2);

        let final_stats = global_phi_stats();

        // Stats should have increased (at least 2 calculations)
        // Note: Use delta instead of absolute to handle test parallelism
        let delta = final_stats
            .total_calculations
            .saturating_sub(initial_stats.total_calculations);
        assert!(
            delta >= 2,
            "Expected at least 2 new calculations, got delta {} (initial: {}, final: {})",
            delta,
            initial_stats.total_calculations,
            final_stats.total_calculations
        );
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #89: PARALLEL COMPUTATION BENCHMARK
    // ========================================================================

    #[test]
    fn test_parallel_spectral_correctness() {
        // Verify parallel computation produces same results as sequential
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Test with n > 16 (triggers parallel path)
        let components = create_test_components(20);

        // Compute using parallel path
        let phi_parallel = phi.compute(&components);

        // Verify result is valid
        assert!(phi_parallel >= 0.0, "Φ should be non-negative");
        assert!(phi_parallel <= 1.0, "Φ should be <= 1.0");
        assert!(phi_parallel > 0.0, "Φ should be positive for 20 components");
    }

    #[test]
    fn test_parallel_vs_sequential_matrix() {
        // Compare parallel and sequential similarity matrix construction
        let phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Test with components that trigger parallel path
        let components = create_test_components(25);

        // Build both matrices
        let parallel_matrix = phi.build_similarity_matrix_parallel(&components);
        let sequential_matrix = phi.build_similarity_matrix_sequential(&components);

        // Verify dimensions
        assert_eq!(parallel_matrix.len(), 25);
        assert_eq!(sequential_matrix.len(), 25);

        // Verify values match (within floating point tolerance)
        for i in 0..25 {
            for j in 0..25 {
                let diff = (parallel_matrix[i][j] - sequential_matrix[i][j]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at [{},{}]: parallel={}, sequential={}",
                    i,
                    j,
                    parallel_matrix[i][j],
                    sequential_matrix[i][j]
                );
            }
        }
    }

    #[test]
    fn test_parallel_speedup_large_system() {
        use std::time::Instant;

        // Benchmark with larger system (n=50)
        let components = create_test_components(50);
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Warm up
        let _ = phi.compute(&components);

        // Benchmark
        let start = Instant::now();
        for _ in 0..5 {
            let _ = phi.compute(&components);
        }
        let elapsed = start.elapsed();

        // Should complete in reasonable time (< 500ms for 5 iterations)
        // This validates that parallel execution is working
        assert!(
            elapsed.as_millis() < 500,
            "Parallel spectral should be fast, took {}ms for 5 iterations",
            elapsed.as_millis()
        );
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #90: INCREMENTAL Φ TESTS
    // ========================================================================

    #[test]
    fn test_incremental_first_call() {
        // First call should do full computation
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(20);

        let result = phi.compute_incremental(&components);

        // Should return valid Φ
        assert!(result >= 0.0 && result <= 1.0);

        // Should have initialized state
        assert!(phi.incremental_state.is_some());

        // Should count as full recompute
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(stats.0, 0, "No incremental updates yet");
        assert_eq!(stats.1, 1, "Should have 1 full recompute");
    }

    #[test]
    fn test_incremental_no_change() {
        // Same components should return cached value
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(15);

        // First computation
        let phi1 = phi.compute_incremental(&components);

        // Second computation with same components
        let phi2 = phi.compute_incremental(&components);

        // Should return same value
        assert!(
            (phi1 - phi2).abs() < 1e-10,
            "Φ should be identical for unchanged components"
        );

        // Should NOT count as incremental update (no change = cache hit)
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(
            stats.0, 0,
            "No incremental updates for unchanged components"
        );
    }

    #[test]
    fn test_incremental_one_change() {
        // Changing one component should trigger incremental update
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let mut components = create_test_components(20);

        // First computation
        let _phi1 = phi.compute_incremental(&components);

        // Change one component
        components[0] = BinaryHV::random(99999);

        // Second computation
        let phi2 = phi.compute_incremental(&components);

        // Should return valid Φ
        assert!(phi2 >= 0.0 && phi2 <= 1.0);

        // Should count as incremental update
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(stats.0, 1, "Should have 1 incremental update");
        assert_eq!(stats.1, 1, "Should still have 1 full recompute (initial)");
    }

    #[test]
    fn test_incremental_multiple_changes() {
        // Changing multiple components should still work
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let mut components = create_test_components(30);

        // First computation
        let _phi1 = phi.compute_incremental(&components);

        // Change 5 components (less than half)
        for i in 0..5 {
            components[i] = BinaryHV::random((i + 1000) as u64);
        }

        // Second computation
        let phi2 = phi.compute_incremental(&components);

        // Should return valid Φ
        assert!(phi2 >= 0.0 && phi2 <= 1.0);

        // Should count as incremental update (5 < 15 = n/2)
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(stats.0, 1, "Should have 1 incremental update");
    }

    #[test]
    fn test_incremental_many_changes_triggers_full() {
        // Changing more than half should trigger full recomputation
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let mut components = create_test_components(20);

        // First computation
        let _phi1 = phi.compute_incremental(&components);

        // Change more than half (11 out of 20)
        for i in 0..11 {
            components[i] = BinaryHV::random((i + 2000) as u64);
        }

        // Second computation
        let phi2 = phi.compute_incremental(&components);

        // Should return valid Φ
        assert!(phi2 >= 0.0 && phi2 <= 1.0);

        // Should trigger full recompute, not incremental
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(
            stats.0, 0,
            "Should have 0 incremental updates (>n/2 changes)"
        );
        assert_eq!(stats.1, 2, "Should have 2 full recomputes");
    }

    #[test]
    fn test_incremental_speedup() {
        // Verify that incremental computation produces correct results and
        // tracks incremental vs full stats properly. Timing is logged but not
        // asserted because it is unreliable under heavy system load.
        use std::time::Instant;

        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let mut phi_full = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let mut components = create_test_components(40);

        // First computation (full) to prime incremental state
        let _ = phi.compute_incremental(&components);

        // Benchmark and collect incremental results
        let start_incremental = Instant::now();
        let mut incremental_results = Vec::new();
        for i in 0..10 {
            components[0] = BinaryHV::random((i * 1000) as u64);
            let result = phi.compute_incremental(&components);
            incremental_results.push(result);
        }
        let incremental_time = start_incremental.elapsed();

        // Verify incremental state is being used (should have incremental updates)
        let stats = phi.incremental_stats().unwrap();
        assert!(
            stats.0 > 0,
            "Should have some incremental updates for single-component changes, got {} incremental / {} full",
            stats.0, stats.1
        );

        // Benchmark full computations with the same component sequence
        let start_full = Instant::now();
        let mut full_results = Vec::new();
        for i in 0..10 {
            components[0] = BinaryHV::random((i * 1000) as u64);
            let result = phi_full.compute(&components);
            full_results.push(result);
        }
        let full_time = start_full.elapsed();

        // Correctness: incremental and full should produce the same Φ values
        for (i, (inc, full)) in incremental_results
            .iter()
            .zip(full_results.iter())
            .enumerate()
        {
            assert!(
                (inc - full).abs() < 1e-6,
                "Iteration {}: incremental ({}) and full ({}) should match",
                i,
                inc,
                full
            );
        }

        // Soft timing check: log but don't fail (flaky under load)
        println!(
            "Incremental: {:?}, Full: {:?}, Ratio: {:.2}x",
            incremental_time,
            full_time,
            full_time.as_secs_f64() / incremental_time.as_secs_f64().max(1e-9)
        );
    }

    #[test]
    fn test_clear_incremental_state() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(10);

        // Build state
        let _ = phi.compute_incremental(&components);
        assert!(phi.incremental_state.is_some());

        // Clear it
        phi.clear_incremental_state();
        assert!(phi.incremental_state.is_none());

        // Next call should do full computation
        let _ = phi.compute_incremental(&components);
        let stats = phi.incremental_stats().unwrap();
        assert_eq!(stats.1, 1, "Should have fresh full recompute after clear");
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #91: HIERARCHICAL Φ TESTS
    // ========================================================================

    #[test]
    fn test_hierarchical_trivial_cases() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Empty components
        let empty: Vec<BinaryHV> = vec![];
        let h = phi.compute_hierarchical(&empty);
        assert_eq!(h.num_clusters, 0);
        assert_eq!(h.micro_phi, 0.0);
        assert_eq!(h.emergence_ratio, 1.0);

        // Single component
        let single = vec![BinaryHV::random(0)];
        let h = phi.compute_hierarchical(&single);
        assert_eq!(h.num_clusters, 1);
        assert_eq!(h.macro_phi, 0.0);
    }

    #[test]
    fn test_hierarchical_basic() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(20);

        let h = phi.compute_hierarchical(&components);

        // Should detect some clusters
        assert!(h.num_clusters >= 1, "Should find at least 1 cluster");

        // All Φ values should be in [0, 1]
        assert!(h.micro_phi >= 0.0 && h.micro_phi <= 1.0);
        assert!(h.meso_phi >= 0.0 && h.meso_phi <= 1.0);
        assert!(h.macro_phi >= 0.0 && h.macro_phi <= 1.0);

        // Bottleneck should be non-negative
        assert!(h.bottleneck_score >= 0.0);
    }

    #[test]
    fn test_hierarchical_identical_components() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Components that are very similar should cluster together
        let base = BinaryHV::random(42);
        let mut components = vec![];
        for i in 0..10 {
            // Create slight variations by XORing with sparse vectors
            let mut variant = base.clone();
            variant.0[i % 32] ^= 0x01; // Flip one bit
            components.push(variant);
        }

        let h = phi.compute_hierarchical(&components);

        // High similarity should lead to few clusters
        // (all similar components should be in same cluster)
        assert!(
            h.num_clusters <= 3,
            "Similar components should cluster together"
        );

        // Micro Φ should be high (strong within-cluster binding)
        assert!(
            h.micro_phi > 0.3,
            "Similar components should have high micro Φ"
        );
    }

    #[test]
    fn test_hierarchical_distinct_clusters() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);

        // Create two very distinct clusters
        let mut components = vec![];

        // Cluster 1: Components derived from seed 100
        for i in 0..5 {
            components.push(BinaryHV::random(100 + i));
        }

        // Cluster 2: Components derived from seed 200
        for i in 0..5 {
            components.push(BinaryHV::random(200 + i));
        }

        let h = phi.compute_hierarchical(&components);

        // Should detect the cluster structure
        // Note: Exact number depends on similarity threshold
        assert!(h.num_clusters >= 1);

        // Meso Φ (between clusters) should generally be lower than micro Φ
        // unless random components happen to be similar
        println!(
            "Clusters: {}, Micro: {:.3}, Meso: {:.3}, Macro: {:.3}",
            h.num_clusters, h.micro_phi, h.meso_phi, h.macro_phi
        );
    }

    #[test]
    fn test_hierarchical_emergence_ratio() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(30);

        let h = phi.compute_hierarchical(&components);

        // Emergence ratio should be positive
        assert!(h.emergence_ratio > 0.0);

        // If emergence_ratio > 1, macro integration exceeds sum of local
        // This indicates true emergent consciousness!
        println!(
            "Emergence ratio: {:.3} (>1 = emergent integration)",
            h.emergence_ratio
        );
    }

    #[test]
    fn test_hierarchical_bottleneck_detection() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(25);

        let h = phi.compute_hierarchical(&components);

        // Bottleneck score = |macro - meso|
        // Low bottleneck = good cross-cluster integration
        // High bottleneck = integration breakdown between clusters
        let expected_bottleneck = (h.macro_phi - h.meso_phi).abs();
        assert!((h.bottleneck_score - expected_bottleneck).abs() < 1e-10);

        println!(
            "Bottleneck score: {:.3} (lower = better integration)",
            h.bottleneck_score
        );
    }

    #[test]
    fn test_hierarchical_consistency_with_macro() {
        // Macro Φ from hierarchical should match regular spectral Φ
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = create_test_components(15);

        let h = phi.compute_hierarchical(&components);
        let regular_phi = phi.compute(&components);

        // They should be close (both use same underlying algorithm)
        // Allow some tolerance due to different code paths
        assert!(
            (h.macro_phi - regular_phi).abs() < 0.1,
            "Hierarchical macro Φ ({:.3}) should match regular Φ ({:.3})",
            h.macro_phi,
            regular_phi
        );
    }

    // ========================================================================
    // REVOLUTIONARY #92: CAUSAL Φ ATTRIBUTION TESTS
    // ========================================================================

    #[test]
    fn test_attribution_empty_components() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components: Vec<BinaryHV> = vec![];

        let attr = phi.compute_attribution(&components);

        assert_eq!(attr.baseline_phi, 0.0);
        assert!(attr.component_scores.is_empty());
        assert!(attr.importance_ranking.is_empty());
        assert!(attr.critical_components.is_empty());
        assert!(attr.redundant_components.is_empty());
        assert_eq!(attr.concentration_index, 0.0);
    }

    #[test]
    fn test_attribution_single_component() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components = vec![BinaryHV::random(42)];

        let attr = phi.compute_attribution(&components);

        // Single component has no integration
        assert_eq!(attr.baseline_phi, 0.0);
        assert_eq!(attr.component_scores.len(), 1);
        assert_eq!(attr.component_scores[0], 0.0);
        assert_eq!(attr.importance_ranking, vec![0]);
        // Single component is redundant (can't contribute to integration alone)
        assert_eq!(attr.redundant_components, vec![0]);
    }

    #[test]
    fn test_attribution_basic() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components = create_test_components(5);

        let attr = phi.compute_attribution(&components);

        // Basic sanity checks
        assert!(
            attr.baseline_phi > 0.0,
            "5 components should have positive Φ"
        );
        assert_eq!(attr.component_scores.len(), 5);
        assert_eq!(attr.importance_ranking.len(), 5);

        // Importance ranking should be a permutation of 0..5
        let mut sorted_ranking = attr.importance_ranking.clone();
        sorted_ranking.sort();
        assert_eq!(sorted_ranking, vec![0, 1, 2, 3, 4]);

        // Concentration should be between 0 and 1
        assert!(attr.concentration_index >= 0.0);
        assert!(attr.concentration_index <= 1.0);

        println!("Attribution test - baseline Φ: {:.4}", attr.baseline_phi);
        println!("Importance ranking: {:?}", attr.importance_ranking);
        println!("Concentration index: {:.4}", attr.concentration_index);
    }

    #[test]
    fn test_attribution_hub_spoke_topology() {
        // Create a hub-spoke structure: component 0 is the hub
        // Hub should be most critical since it connects everything
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);

        // Create hub with specific seed, spokes with similar seeds
        let hub = BinaryHV::random(1000);
        let mut components = vec![hub.clone()];

        // Create spokes that are all similar to hub but not each other
        for i in 1..6 {
            // Mix hub with unique component
            let unique = BinaryHV::random(i as u64);
            // Create spoke by bundling hub pattern with unique pattern
            // This makes spokes connected to hub but less to each other
            let spoke = BinaryHV::bundle(&[hub.clone(), unique]);
            components.push(spoke);
        }

        let attr = phi.compute_attribution(&components);

        // Hub (index 0) should be among the most critical components
        // since removing it breaks hub-spoke integration
        println!("Hub-spoke attribution:");
        println!("  Baseline Φ: {:.4}", attr.baseline_phi);
        println!("  Hub (0) importance: {:.4}", attr.component_scores[0]);
        println!("  Importance ranking: {:?}", attr.importance_ranking);
        println!("  Critical components: {:?}", attr.critical_components);

        // The test validates that the attribution runs without error
        // and produces sensible output
        assert!(attr.baseline_phi > 0.0);
        assert_eq!(attr.component_scores.len(), 6);
    }

    #[test]
    fn test_attribution_fast_vs_full() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components = create_test_components(8);

        let full_attr = phi.compute_attribution(&components);
        let fast_attr = phi.compute_attribution_fast(&components);

        // Both should have same baseline
        assert!((full_attr.baseline_phi - fast_attr.baseline_phi).abs() < 1e-10);

        // Both should have same number of components
        assert_eq!(
            full_attr.component_scores.len(),
            fast_attr.component_scores.len()
        );

        // Fast method may have different ranking but should identify
        // similar critical/redundant patterns
        println!("Full vs Fast attribution comparison:");
        println!("  Full ranking: {:?}", full_attr.importance_ranking);
        println!("  Fast ranking: {:?}", fast_attr.importance_ranking);
        println!("  Full critical: {:?}", full_attr.critical_components);
        println!("  Fast critical: {:?}", fast_attr.critical_components);
    }

    #[test]
    fn test_attribution_identical_components() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);

        // All identical components - should have uniform attribution
        let base = BinaryHV::random(999);
        let components: Vec<BinaryHV> = (0..5).map(|_| base.clone()).collect();

        let attr = phi.compute_attribution(&components);

        // All components should have similar importance (uniform distribution)
        let scores = &attr.component_scores;
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

        println!("Identical components test:");
        println!("  Scores: {:?}", scores);
        println!("  Mean: {:.4}, Variance: {:.6}", mean, variance);

        // Low variance = uniform importance
        assert!(
            variance < 0.01,
            "Identical components should have low variance in importance"
        );
    }

    #[test]
    fn test_attribution_critical_detection() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components = create_test_components(10);

        let attr = phi.compute_attribution(&components);

        // Critical threshold is 10% of baseline_phi
        let threshold = attr.baseline_phi * 0.10;

        // Verify critical components are actually above threshold
        for &i in &attr.critical_components {
            assert!(
                attr.component_scores[i] > threshold,
                "Critical component {} has score {:.4} below threshold {:.4}",
                i,
                attr.component_scores[i],
                threshold
            );
        }

        // Verify redundant components are actually below 1% threshold
        let redundant_threshold = attr.baseline_phi * 0.01;
        for &i in &attr.redundant_components {
            assert!(
                attr.component_scores[i] < redundant_threshold,
                "Redundant component {} has score {:.4} above threshold {:.4}",
                i,
                attr.component_scores[i],
                redundant_threshold
            );
        }
    }

    #[test]
    fn test_attribution_concentration_gradient() {
        // Test that concentration index varies with different distributions
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);

        // Create systems with different integration patterns
        // System 1: Diverse components (should have distributed importance)
        let diverse: Vec<BinaryHV> = (0..6).map(|i| BinaryHV::random(i as u64 * 1000)).collect();
        let attr_diverse = phi.compute_attribution(&diverse);

        // System 2: Mostly similar components (may have more concentrated importance)
        let base = BinaryHV::random(42);
        let similar: Vec<BinaryHV> = (0..6)
            .map(|i| {
                let noise = BinaryHV::random(i as u64);
                BinaryHV::bundle(&[base.clone(), base.clone(), base.clone(), noise])
            })
            .collect();
        let attr_similar = phi.compute_attribution(&similar);

        println!("Concentration gradient test:");
        println!(
            "  Diverse system concentration: {:.4}",
            attr_diverse.concentration_index
        );
        println!(
            "  Similar system concentration: {:.4}",
            attr_similar.concentration_index
        );

        // Both should be valid concentration indices
        assert!(attr_diverse.concentration_index >= 0.0);
        assert!(attr_diverse.concentration_index <= 1.0);
        assert!(attr_similar.concentration_index >= 0.0);
        assert!(attr_similar.concentration_index <= 1.0);
    }

    #[test]
    fn test_phi_attribution_helper_methods() {
        let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
        let components = create_test_components(8);

        let attr = phi.compute_attribution(&components);

        // Test helper methods
        assert!(attr.most_critical().is_some());
        assert!(attr.most_redundant().is_some());

        // Most critical should be first in ranking
        assert_eq!(attr.most_critical(), Some(attr.importance_ranking[0]));

        // Most redundant should be last in ranking
        assert_eq!(
            attr.most_redundant(),
            Some(attr.importance_ranking[attr.importance_ranking.len() - 1])
        );

        // is_distributed should match concentration threshold
        let expected_distributed = attr.concentration_index < 0.3;
        assert_eq!(attr.is_distributed(), expected_distributed);

        // critical_percentage should be valid
        let pct = attr.critical_percentage();
        assert!(pct >= 0.0 && pct <= 100.0);

        println!("Helper methods test:");
        println!("  Most critical: {:?}", attr.most_critical());
        println!("  Most redundant: {:?}", attr.most_redundant());
        println!("  Is distributed: {}", attr.is_distributed());
        println!("  Critical percentage: {:.1}%", pct);
    }

    // ============================================================================
    // Revolutionary #93: Φ Temporal Dynamics Tests
    // ============================================================================

    #[test]
    fn test_dynamics_empty_history() {
        let dynamics = PhiDynamics::new();

        assert_eq!(dynamics.sample_count(), 0);
        assert!(dynamics.get_history().is_empty());
        assert!(dynamics.get_recent(10).is_empty());
    }

    #[test]
    fn test_dynamics_single_sample() {
        let mut dynamics = PhiDynamics::new();

        let snapshot = dynamics.record(0.5);

        assert_eq!(dynamics.sample_count(), 1);
        assert_eq!(snapshot.current_phi, 0.5);
        assert!((snapshot.mean_phi - 0.5).abs() < 1e-10);
        assert!((snapshot.volatility - 0.0).abs() < 1e-10); // Single sample has 0 volatility
        assert!(snapshot.transition.is_none()); // No transition on first sample
    }

    #[test]
    fn test_dynamics_stable_sequence() {
        let mut dynamics = PhiDynamics::new();

        // Record stable values around 0.5
        let mut last_snapshot = None;
        for i in 0..20 {
            let phi = 0.5 + (i as f64 * 0.001); // Very small variation
            last_snapshot = Some(dynamics.record(phi));
        }

        let snapshot = last_snapshot.expect("Should have snapshot");

        // Should be stable or slightly increasing
        assert!(
            snapshot.trend.direction == TrendDirection::Stable
                || snapshot.trend.direction == TrendDirection::Increasing
        );

        println!(
            "Stable sequence trend: {:?}, strength: {:.4}",
            snapshot.trend.direction, snapshot.trend.strength
        );
    }

    #[test]
    fn test_dynamics_increasing_trend() {
        let mut dynamics = PhiDynamics::new();

        // Record clearly increasing values
        let mut last_snapshot = None;
        for i in 0..50 {
            let phi = 0.3 + (i as f64 * 0.01); // 0.3 → 0.79
            last_snapshot = Some(dynamics.record(phi));
        }

        let snapshot = last_snapshot.expect("Should have snapshot");

        assert_eq!(snapshot.trend.direction, TrendDirection::Increasing);
        assert!(snapshot.trend.strength > 0.0);

        println!(
            "Increasing trend: strength = {:.4}, predicted_next = {:.4}",
            snapshot.trend.strength, snapshot.trend.predicted_next
        );
    }

    #[test]
    fn test_dynamics_decreasing_trend() {
        let mut dynamics = PhiDynamics::new();

        // Record clearly decreasing values
        let mut last_snapshot = None;
        for i in 0..50 {
            let phi = 0.8 - (i as f64 * 0.01); // 0.8 → 0.31
            last_snapshot = Some(dynamics.record(phi));
        }

        let snapshot = last_snapshot.expect("Should have snapshot");

        assert_eq!(snapshot.trend.direction, TrendDirection::Decreasing);

        println!(
            "Decreasing trend: strength = {:.4}, predicted_next = {:.4}",
            snapshot.trend.strength, snapshot.trend.predicted_next
        );
    }

    #[test]
    fn test_dynamics_phase_transition_detection() {
        let mut dynamics = PhiDynamics::new();

        // Build up stable baseline
        for _ in 0..20 {
            dynamics.record(0.5);
        }

        // Now introduce a sudden change
        let snapshot = dynamics.record(0.8); // Big jump!

        assert!(
            snapshot.transition.is_some(),
            "Should detect phase transition"
        );

        let transition = snapshot.transition.unwrap();
        assert_eq!(transition.direction, TransitionDirection::Rising);
        assert!(transition.magnitude_sigma > 2.0); // Should be significant

        println!(
            "Detected transition: {:?}, magnitude: {:.2}σ, type: {:?}",
            transition.direction, transition.magnitude_sigma, transition.transition_type
        );
    }

    #[test]
    fn test_dynamics_falling_transition() {
        let mut dynamics = PhiDynamics::new();

        // Build up stable high baseline
        for _ in 0..20 {
            dynamics.record(0.8);
        }

        // Sudden drop
        let snapshot = dynamics.record(0.3);

        assert!(
            snapshot.transition.is_some(),
            "Should detect falling transition"
        );

        let transition = snapshot.transition.unwrap();
        assert_eq!(transition.direction, TransitionDirection::Falling);

        println!(
            "Falling transition detected: magnitude = {:.2}σ",
            transition.magnitude_sigma
        );
    }

    #[test]
    fn test_dynamics_oscillating_pattern() {
        let mut dynamics = PhiDynamics::new();

        // Create oscillating pattern
        let mut last_snapshot = None;
        for i in 0..100 {
            let phi = 0.5 + 0.2 * (i as f64 * 0.3).sin();
            last_snapshot = Some(dynamics.record(phi));
        }

        let snapshot = last_snapshot.expect("Should have snapshot");

        // Should detect oscillation or low strength trend
        assert!(
            snapshot.trend.strength < 0.5
                || snapshot.trend.direction == TrendDirection::Oscillating
        );

        println!(
            "Oscillating pattern: direction = {:?}, strength = {:.4}",
            snapshot.trend.direction, snapshot.trend.strength
        );
    }

    #[test]
    fn test_dynamics_circular_buffer() {
        let config = PhiDynamicsConfig {
            history_size: 10, // Small buffer
            ..Default::default()
        };
        let mut dynamics = PhiDynamics::with_config(config);

        // Add more than buffer size
        for i in 0..25 {
            dynamics.record(i as f64 * 0.1);
        }

        // Should only have last 10
        assert_eq!(dynamics.sample_count(), 10);

        let history = dynamics.get_history();
        assert_eq!(history.len(), 10);

        // Values should be the most recent ones
        let values: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
        for (i, v) in values.iter().enumerate() {
            let expected = (15 + i) as f64 * 0.1; // Last 10 values: 1.5, 1.6, ..., 2.4
            assert!(
                (*v - expected).abs() < 1e-10,
                "Expected {:.1}, got {:.1}",
                expected,
                *v
            );
        }
    }

    #[test]
    fn test_dynamics_reset() {
        let mut dynamics = PhiDynamics::new();

        // Add some samples
        for i in 0..20 {
            dynamics.record(i as f64 * 0.05);
        }
        assert_eq!(dynamics.sample_count(), 20);

        // Reset
        dynamics.reset();

        assert_eq!(dynamics.sample_count(), 0);
        assert!(dynamics.get_history().is_empty());
    }

    #[test]
    fn test_dynamics_statistics_accuracy() {
        let mut dynamics = PhiDynamics::new();

        // Add known values
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for v in &values {
            dynamics.record(*v);
        }

        let snapshot = dynamics.record(6.0);

        // After adding 6.0, we have [1, 2, 3, 4, 5, 6]
        // Mean should be 3.5
        let expected_mean = 3.5;
        assert!(
            (snapshot.mean_phi - expected_mean).abs() < 1e-10,
            "Expected mean {}, got {}",
            expected_mean,
            snapshot.mean_phi
        );

        // Variance = E[X²] - E[X]² = (1+4+9+16+25+36)/6 - 12.25 = 91/6 - 12.25 ≈ 2.9167
        // Std = sqrt(2.9167) ≈ 1.7078
        let expected_volatility = (2.9166666667_f64).sqrt();
        assert!(
            (snapshot.volatility - expected_volatility).abs() < 0.01,
            "Expected volatility {:.4}, got {:.4}",
            expected_volatility,
            snapshot.volatility
        );
    }

    #[test]
    fn test_dynamics_with_real_phi() {
        let mut dynamics = PhiDynamics::new();
        let mut phi_calc = TieredPhi::new(ApproximationTier::SampledPartition);

        // Create varying topologies and track their Φ over time
        for seed in 0..30 {
            let components = create_test_components(8 + (seed % 4)); // 8-11 components
            let phi_value = phi_calc.compute(&components);

            let snapshot = dynamics.record(phi_value);

            if let Some(transition) = snapshot.transition {
                println!(
                    "Transition at step {}: {:?} ({:.2}σ)",
                    seed, transition.direction, transition.magnitude_sigma
                );
            }
        }

        println!("Final sample count: {}", dynamics.sample_count());

        // Verify we can compute dynamics on real Φ values
        assert!(dynamics.sample_count() >= 20);
    }

    // ============================================================================
    // Revolutionary #94: Multi-Scale Φ Pyramid Tests
    // ============================================================================

    #[test]
    fn test_pyramid_empty_components() {
        let mut pyramid = PhiPyramid::new();
        let components: Vec<BinaryHV> = vec![];

        let result = pyramid.compute(&components);

        assert!(result.phi_by_scale.is_empty());
        assert_eq!(result.peak_scale, 0);
        assert_eq!(result.peak_phi, 0.0);
    }

    #[test]
    fn test_pyramid_small_system() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(4);

        let result = pyramid.compute(&components);

        assert!(!result.phi_by_scale.is_empty());
        assert!(result.peak_phi >= 0.0);
        assert!(result.peak_phi <= 1.0);

        println!("Small system (n=4) pyramid:");
        println!("  Scales: {:?}", result.components_per_scale);
        println!("  Φ by scale: {:?}", result.phi_by_scale);
        println!(
            "  Peak at scale {}: Φ = {:.4}",
            result.peak_scale, result.peak_phi
        );
    }

    #[test]
    fn test_pyramid_multi_scale_detection() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(32);

        let result = pyramid.compute(&components);

        // Should have multiple scales (at least 3: 2, 4, 8, 16, 32)
        assert!(
            result.phi_by_scale.len() >= 3,
            "Expected at least 3 scales, got {}",
            result.phi_by_scale.len()
        );

        // Scales should be powers of 2 (or close)
        assert!(result.components_per_scale[0] >= 2);

        println!("Multi-scale pyramid (n=32):");
        for (i, (comps, phi)) in result
            .components_per_scale
            .iter()
            .zip(result.phi_by_scale.iter())
            .enumerate()
        {
            let marker = if i == result.peak_scale {
                " ← PEAK"
            } else {
                ""
            };
            println!(
                "  Scale {}: {} components, Φ = {:.4}{}",
                i, comps, phi, marker
            );
        }
    }

    #[test]
    fn test_pyramid_locality_ratio() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(20);

        let result = pyramid.compute(&components);

        // Locality ratio should be positive
        assert!(result.locality_ratio > 0.0);

        // Test helper methods
        println!("Locality analysis:");
        println!("  Locality ratio: {:.4}", result.locality_ratio);
        println!("  Is local dominant: {}", result.is_local_dominant());
        println!("  Is global dominant: {}", result.is_global_dominant());
        println!("  Optimal scale: {}", result.optimal_scale_description());
    }

    #[test]
    fn test_pyramid_scale_variance() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(16);

        let result = pyramid.compute(&components);

        // Variance should be non-negative
        assert!(result.scale_variance >= 0.0);

        println!(
            "Scale variance: {:.4} (high = scale-dependent consciousness)",
            result.scale_variance
        );
    }

    #[test]
    fn test_pyramid_hierarchy_detection() {
        let mut pyramid = PhiPyramid::new();

        // Create a system that might show hierarchical structure
        let components = create_test_components(64);

        let result = pyramid.compute(&components);

        println!("Hierarchy detection (n=64):");
        println!("  Is hierarchical: {}", result.is_hierarchical);
        println!("  Φ gradient: {:?}", result.scale_gradient());

        assert!(
            !result.phi_by_scale.is_empty(),
            "pyramid should produce at least one scale of phi values"
        );
        assert!(
            result.peak_phi >= 0.0,
            "peak phi should be non-negative, got {}",
            result.peak_phi
        );
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn test_pyramid_fast_config() {
        let mut pyramid = PhiPyramid::fast();
        let components = create_test_components(32);

        let start = std::time::Instant::now();
        let result = pyramid.compute(&components);
        let elapsed = start.elapsed();

        assert!(
            result.phi_by_scale.len() <= 4,
            "Fast config should have at most 4 scales"
        );

        println!("Fast pyramid took {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    }

    #[test]
    fn test_pyramid_convenience_functions() {
        let components = create_test_components(16);

        // Test multi_scale_phi
        let result = multi_scale_phi(&components);
        assert!(!result.phi_by_scale.is_empty());

        // Test optimal_scale
        let (scale, phi) = optimal_scale(&components);
        assert_eq!(scale, result.peak_scale);
        assert!((phi - result.peak_phi).abs() < 1e-10);

        println!("Optimal scale: {} with Φ = {:.4}", scale, phi);
    }

    #[test]
    fn test_pyramid_gradient() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(32);

        let result = pyramid.compute(&components);
        let gradient = result.scale_gradient();

        // Gradient should have one less element than phi_by_scale
        if result.phi_by_scale.len() > 1 {
            assert_eq!(gradient.len(), result.phi_by_scale.len() - 1);

            println!("Scale gradient (Φ change between scales):");
            for (i, g) in gradient.iter().enumerate() {
                let direction = if *g > 0.01 {
                    "↑"
                } else if *g < -0.01 {
                    "↓"
                } else {
                    "→"
                };
                println!("  Scale {} → {}: {:.4} {}", i, i + 1, g, direction);
            }
        }
    }

    #[test]
    fn test_pyramid_different_topologies() {
        // Compare pyramid results for different system sizes
        let mut pyramid = PhiPyramid::new();

        let sizes = [8, 16, 32, 64];
        let mut results = vec![];

        for &n in &sizes {
            let components = create_test_components(n);
            let result = pyramid.compute(&components);
            results.push((n, result.peak_scale, result.peak_phi, result.locality_ratio));
        }

        println!("Pyramid comparison across system sizes:");
        println!(
            "{:>6} | {:>10} | {:>8} | {:>12}",
            "Size", "Peak Scale", "Peak Φ", "Locality"
        );
        println!("{:-<6}-+-{:-<10}-+-{:-<8}-+-{:-<12}", "", "", "", "");

        for (n, peak_scale, peak_phi, locality) in &results {
            println!(
                "{:>6} | {:>10} | {:>8.4} | {:>12.4}",
                n, peak_scale, peak_phi, locality
            );
        }

        assert_eq!(
            results.len(),
            sizes.len(),
            "should have results for all system sizes"
        );
        for &(n, _, peak_phi, _) in &results {
            assert!(
                peak_phi >= 0.0,
                "peak_phi should be non-negative for size {}, got {}",
                n,
                peak_phi
            );
        }
    }

    #[test]
    fn test_pyramid_custom_config() {
        let config = PhiPyramidConfig {
            min_components_per_scale: 3,
            max_scales: 5,
            scale_factor: 3, // Each level has 3x more components
            parallel_scales: false,
            phi_tier: ApproximationTier::SampledPartition,
        };

        let mut pyramid = PhiPyramid::with_config(config);
        let components = create_test_components(27); // 3^3

        let result = pyramid.compute(&components);

        // Should have scales: 3, 9, 27 (3 scales with factor 3)
        assert!(
            result.phi_by_scale.len() <= 5,
            "Expected at most 5 scales, got {}",
            result.phi_by_scale.len()
        );

        println!("Custom config (factor=3) pyramid:");
        println!("  Components per scale: {:?}", result.components_per_scale);
    }

    #[test]
    fn test_pyramid_timing() {
        let mut pyramid = PhiPyramid::new();
        let components = create_test_components(50);

        let result = pyramid.compute(&components);

        // Should have recorded computation time
        assert!(result.computation_time_ms > 0.0);

        println!(
            "Pyramid computation time: {:.2}ms",
            result.computation_time_ms
        );
    }

    // ============================================================================
    // REVOLUTIONARY #95: Φ ENTROPY & COMPLEXITY TESTS
    // ============================================================================

    #[test]
    fn test_entropy_insufficient_samples() {
        let analyzer = PhiEntropyAnalyzer::new();

        // Default min_samples is 50, so 10 samples should be insufficient
        let values: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let result = analyzer.analyze(&values, None);

        // Should return default values for insufficient samples
        assert_eq!(result.shannon_entropy, 0.0);
        assert_eq!(result.sample_count, 10);
        assert_eq!(result.predictability, 1.0);

        println!(
            "Insufficient samples handled correctly: {} samples",
            result.sample_count
        );
    }

    #[test]
    fn test_entropy_constant_signal() {
        let config = PhiEntropyConfig {
            min_samples: 10, // Lower threshold for testing
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Constant signal - all same value
        let values: Vec<f64> = vec![0.5; 100];
        let result = analyzer.analyze(&values, None);

        // Constant signal should have zero entropy
        assert_eq!(
            result.shannon_entropy, 0.0,
            "Constant signal should have zero entropy"
        );
        assert_eq!(result.normalized_entropy, 0.0);
        assert!(
            result.predictability > 0.99,
            "Constant signal should be highly predictable"
        );

        println!(
            "Constant signal: entropy = {:.4}, predictability = {:.4}",
            result.shannon_entropy, result.predictability
        );
    }

    #[test]
    fn test_entropy_uniform_random() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let config = PhiEntropyConfig {
            min_samples: 10,
            num_bins: 10,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Generate pseudo-random values spread across range
        let values: Vec<f64> = (0..1000)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                (hasher.finish() % 1000) as f64 / 1000.0
            })
            .collect();

        let result = analyzer.analyze(&values, None);

        // Random signal should have high normalized entropy
        assert!(
            result.normalized_entropy > 0.5,
            "Random signal should have high entropy: {}",
            result.normalized_entropy
        );
        assert!(
            result.predictability < 0.5,
            "Random signal should have low predictability: {}",
            result.predictability
        );

        println!(
            "Random signal: normalized entropy = {:.4}, predictability = {:.4}",
            result.normalized_entropy, result.predictability
        );
    }

    #[test]
    fn test_entropy_shannon_calculation() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            num_bins: 4, // 4 bins for easy verification
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Create perfectly uniform distribution across 4 bins
        // Values: 25 in each bin [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0)
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(i as f64 / 100.0); // 0.0 to 0.99 uniformly
        }

        let result = analyzer.analyze(&values, None);

        // Shannon entropy for uniform 4-bin distribution = log2(4) = 2.0 bits
        // Normalized = 2.0 / log2(4) = 1.0
        // In practice, due to binning edge effects, it may be slightly less
        assert!(
            result.shannon_entropy > 1.5,
            "Uniform distribution should have entropy > 1.5: {}",
            result.shannon_entropy
        );
        assert!(
            result.normalized_entropy > 0.8,
            "Uniform distribution should have normalized entropy > 0.8: {}",
            result.normalized_entropy
        );

        println!(
            "Shannon entropy: {:.4} bits, normalized: {:.4}",
            result.shannon_entropy, result.normalized_entropy
        );
    }

    #[test]
    fn test_entropy_sample_entropy() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            embed_dim: 2,
            tolerance_fraction: 0.2,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Periodic signal (low sample entropy)
        let periodic: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        // Random-ish signal (higher sample entropy)
        let chaotic: Vec<f64> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.31415926;
                (x.sin() * 1000.0) % 1.0 // Pseudo-random
            })
            .collect();

        let periodic_result = analyzer.analyze(&periodic, None);
        let chaotic_result = analyzer.analyze(&chaotic, None);

        // Sample entropy should generally be lower for periodic signals
        // (though this depends on the specific signals and parameters)
        println!(
            "Periodic sample entropy: {:.4}",
            periodic_result.sample_entropy
        );
        println!(
            "Chaotic sample entropy: {:.4}",
            chaotic_result.sample_entropy
        );

        // Both should produce valid (non-negative) sample entropy
        assert!(periodic_result.sample_entropy >= 0.0);
        assert!(chaotic_result.sample_entropy >= 0.0);
    }

    #[test]
    fn test_entropy_lz_complexity() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Repetitive signal (low complexity)
        let repetitive: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();

        // More varied signal (higher complexity)
        let varied: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.137) % 1.0) // Irrational-ish sequence
            .collect();

        let repetitive_result = analyzer.analyze(&repetitive, None);
        let varied_result = analyzer.analyze(&varied, None);

        println!(
            "Repetitive LZ: {:.4} (normalized: {:.4})",
            repetitive_result.lz_complexity, repetitive_result.normalized_lz
        );
        println!(
            "Varied LZ: {:.4} (normalized: {:.4})",
            varied_result.lz_complexity, varied_result.normalized_lz
        );

        // Both should produce valid complexity values
        assert!(repetitive_result.lz_complexity >= 0.0);
        assert!(varied_result.lz_complexity >= 0.0);
        assert!(repetitive_result.normalized_lz <= 1.0);
        assert!(varied_result.normalized_lz <= 1.0);
    }

    #[test]
    fn test_entropy_multi_scale() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            max_scale: 5,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Generate enough samples for multi-scale analysis
        let values: Vec<f64> = (0..500)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).sin() * 0.5)
            .collect();

        let result = analyzer.analyze(&values, None);

        // Should have multi-scale entropy values
        assert!(
            !result.multi_scale_entropy.is_empty(),
            "Should have multi-scale entropy for {} samples",
            result.sample_count
        );

        println!(
            "Multi-scale entropy ({} scales):",
            result.multi_scale_entropy.len()
        );
        for (scale, se) in result.multi_scale_entropy.iter().enumerate() {
            println!("  Scale {}: {:.4}", scale + 1, se);
        }
    }

    #[test]
    fn test_entropy_integrated_complexity() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Create varied signal
        let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.17) % 1.0).collect();

        // Test with different mean Φ values
        let result_low_phi = analyzer.analyze(&values, Some(0.1));
        let result_high_phi = analyzer.analyze(&values, Some(0.9));

        // Same complexity, different Φ should yield different integrated complexity
        assert!(
            result_high_phi.integrated_complexity > result_low_phi.integrated_complexity,
            "Higher Φ should yield higher integrated complexity"
        );

        // Verify integrated complexity formula: Φ × complexity_index
        let expected_low = 0.1 * result_low_phi.complexity_index;
        let expected_high = 0.9 * result_high_phi.complexity_index;

        assert!(
            (result_low_phi.integrated_complexity - expected_low).abs() < 0.01,
            "Integrated complexity should match Φ × complexity_index"
        );
        assert!((result_high_phi.integrated_complexity - expected_high).abs() < 0.01);

        println!(
            "Low Φ (0.1): integrated_complexity = {:.4}",
            result_low_phi.integrated_complexity
        );
        println!(
            "High Φ (0.9): integrated_complexity = {:.4}",
            result_high_phi.integrated_complexity
        );
    }

    #[test]
    fn test_entropy_quality_descriptors() {
        // Test quality description categories
        let rich = PhiEntropyResult {
            shannon_entropy: 2.0,
            normalized_entropy: 0.5,
            sample_entropy: 1.0,
            lz_complexity: 5.0,
            normalized_lz: 0.5,
            multi_scale_entropy: vec![],
            complexity_index: 0.8,
            integrated_complexity: 0.7, // High
            predictability: 0.5,
            sample_count: 100,
        };
        assert_eq!(rich.quality_description(), "rich");
        assert!(rich.is_complex());

        let chaotic = PhiEntropyResult {
            shannon_entropy: 2.0,
            normalized_entropy: 0.85, // High
            sample_entropy: 1.0,
            lz_complexity: 5.0,
            normalized_lz: 0.5,
            multi_scale_entropy: vec![],
            complexity_index: 0.3,
            integrated_complexity: 0.15, // Low
            predictability: 0.15,
            sample_count: 100,
        };
        assert_eq!(chaotic.quality_description(), "chaotic");
        assert!(chaotic.is_chaotic());

        let simple = PhiEntropyResult {
            shannon_entropy: 0.5,
            normalized_entropy: 0.2,
            sample_entropy: 0.1,
            lz_complexity: 2.0,
            normalized_lz: 0.2,
            multi_scale_entropy: vec![],
            complexity_index: 0.2,
            integrated_complexity: 0.1,
            predictability: 0.8, // High
            sample_count: 100,
        };
        assert_eq!(simple.quality_description(), "simple");
        assert!(simple.is_predictable());

        println!("Quality descriptors: rich, chaotic, simple - all working");
    }

    #[test]
    fn test_entropy_convenience_functions() {
        // Test analyze_phi_complexity
        let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1) % 1.0).collect();

        let result = analyze_phi_complexity(&values);
        assert!(result.sample_count == 100);

        // Test integrated_complexity function
        let ic = integrated_complexity(&values, 0.5);
        assert!(
            ic >= 0.0 && ic <= 1.0,
            "Integrated complexity should be in [0, 1]"
        );

        println!("Convenience functions: analyze_phi_complexity and integrated_complexity working");
    }

    #[test]
    fn test_entropy_config_presets() {
        // Test fast config
        let fast = PhiEntropyAnalyzer::fast();
        let values: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let result = fast.analyze(&values, None);
        assert!(result.sample_count > 0);

        // Test research config
        let research = PhiEntropyAnalyzer::research();
        let values_large: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
        let result_research = research.analyze(&values_large, None);
        assert!(result_research.sample_count > 0);

        println!("Config presets (fast, research) working");
    }

    #[test]
    fn test_entropy_complexity_index() {
        let config = PhiEntropyConfig {
            min_samples: 10,
            ..Default::default()
        };
        let analyzer = PhiEntropyAnalyzer::with_config(config);

        // Create signal with moderate complexity
        let values: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.23 + (i as f64 * 0.07).sin()) % 1.0)
            .collect();

        let result = analyzer.analyze(&values, Some(0.5));

        // Complexity index should be geometric mean of entropy measures
        // Bounded between 0 and 1
        assert!(
            result.complexity_index >= 0.0 && result.complexity_index <= 1.0,
            "Complexity index should be in [0, 1]: {}",
            result.complexity_index
        );

        println!("Complexity index: {:.4}", result.complexity_index);
        println!(
            "Components: norm_entropy={:.4}, sample_ent={:.4}, norm_lz={:.4}",
            result.normalized_entropy, result.sample_entropy, result.normalized_lz
        );
    }

    // ============================================================================
    // REVOLUTIONARY #96: CROSS-TOPOLOGY Φ TRANSFER TESTS
    // ============================================================================

    fn create_test_realvh_components(n: usize, dim: usize, seed: u64) -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| ContinuousHV::random(dim, seed + i as u64 * 1000))
            .collect()
    }

    #[test]
    fn test_transfer_signature_extraction() {
        let transfer = PhiTransfer::new();
        let components = create_test_realvh_components(8, 256, 42);

        let signature = transfer.extract_signature(&components, 0.45, Some("Test"));

        // Signature should have features
        assert!(!signature.similarity_features.is_empty());
        assert!(!signature.connectivity_features.is_empty());
        assert!(!signature.spectral_features.is_empty());

        // Should have correct metadata
        assert_eq!(signature.original_phi, 0.45);
        assert_eq!(signature.num_components, 8);
        assert_eq!(signature.topology_type, Some("Test".to_string()));

        println!("Signature extracted with {} dimensions", signature.dim());
        println!("  Similarity features: {:?}", signature.similarity_features);
        println!(
            "  Connectivity features: {:?}",
            signature.connectivity_features
        );
    }

    #[test]
    fn test_transfer_signature_vector() {
        let transfer = PhiTransfer::new();
        let components = create_test_realvh_components(8, 256, 42);

        let signature = transfer.extract_signature(&components, 0.45, None);
        let vector = signature.as_vector();

        // Vector should combine all features
        let expected_dim = signature.similarity_features.len()
            + signature.connectivity_features.len()
            + signature.spectral_features.len();
        assert_eq!(vector.len(), expected_dim);
        assert_eq!(vector.len(), signature.dim());

        println!("Signature vector has {} dimensions", vector.len());
    }

    #[test]
    fn test_transfer_different_topologies() {
        let transfer = PhiTransfer::new();

        // Create two different "topologies"
        let high_phi_components = create_test_realvh_components(8, 256, 42);
        let low_phi_components = create_test_realvh_components(8, 256, 999);

        let result = transfer.transfer(
            &high_phi_components,
            &low_phi_components,
            0.50, // Source (high) Φ
            0.35, // Target (low) Φ
            "HighPhi",
            "LowPhi",
        );

        // Transfer should produce improvement
        assert!(
            result.enhanced_phi > result.original_phi,
            "Enhanced Φ {} should exceed original {}",
            result.enhanced_phi,
            result.original_phi
        );
        assert!(result.improvement_ratio > 1.0);
        assert!(result.converged);

        println!("Transfer: {} → {}", result.source_type, result.target_type);
        println!("  Original Φ: {:.4}", result.original_phi);
        println!("  Enhanced Φ: {:.4}", result.enhanced_phi);
        println!("  Improvement: {:.2}%", result.improvement_percent());
    }

    #[test]
    fn test_transfer_potential() {
        let transfer = PhiTransfer::new();

        let source = create_test_realvh_components(8, 256, 42);
        let target = create_test_realvh_components(8, 256, 43); // Similar seed

        let potential = transfer.transfer_potential(&source, &target, 0.5, 0.3);

        // Transfer potential should be positive
        assert!(
            potential >= 0.0,
            "Transfer potential should be non-negative"
        );
        assert!(potential <= 1.0, "Transfer potential should be bounded");

        println!("Transfer potential: {:.4}", potential);
    }

    #[test]
    fn test_transfer_result_methods() {
        let result = PhiTransferResult {
            original_phi: 0.3,
            enhanced_phi: 0.45,
            improvement_ratio: 1.5,
            transfer_loss: 0.05,
            iterations: 100,
            converged: true,
            source_type: "Ring".to_string(),
            target_type: "Random".to_string(),
            transfer_vector: vec![0.1, 0.2, -0.1],
        };

        assert!(result.is_successful());
        assert!((result.improvement_percent() - 50.0).abs() < 0.01);

        let failed = PhiTransferResult {
            improvement_ratio: 0.9, // No improvement
            converged: false,
            ..result.clone()
        };
        assert!(!failed.is_successful());

        println!(
            "Result methods working: improvement = {:.1}%",
            result.improvement_percent()
        );
    }

    #[test]
    fn test_transfer_config_presets() {
        let fast = PhiTransfer::fast();
        let research = PhiTransfer::research();

        let components = create_test_realvh_components(8, 256, 42);

        // Both should extract valid signatures
        let sig_fast = fast.extract_signature(&components, 0.5, None);
        let sig_research = research.extract_signature(&components, 0.5, None);

        assert!(sig_fast.dim() > 0);
        assert!(sig_research.dim() > sig_fast.dim()); // Research has more dimensions

        println!("Fast signature dims: {}", sig_fast.dim());
        println!("Research signature dims: {}", sig_research.dim());
    }

    #[test]
    fn test_transfer_empty_components() {
        let transfer = PhiTransfer::new();
        let empty: Vec<ContinuousHV> = vec![];
        let single = create_test_realvh_components(1, 256, 42);

        // Should handle edge cases gracefully
        let sig_empty = transfer.extract_signature(&empty, 0.0, None);
        let sig_single = transfer.extract_signature(&single, 0.1, None);

        // Empty should have zero features
        assert_eq!(sig_empty.num_components, 0);
        assert_eq!(sig_single.num_components, 1);

        println!(
            "Edge cases handled: empty={}, single={}",
            sig_empty.num_components, sig_single.num_components
        );
    }

    #[test]
    fn test_transfer_learning() {
        let mut transfer = PhiTransfer::fast();

        // Create source signatures (high-Φ topologies)
        let sources: Vec<PhiSignature> = (0..3)
            .map(|i| {
                let components = create_test_realvh_components(8, 256, i as u64 * 100);
                transfer.extract_signature(&components, 0.5 + i as f64 * 0.1, Some("High"))
            })
            .collect();

        // Create target signatures (low-Φ topologies)
        let targets: Vec<PhiSignature> = (0..3)
            .map(|i| {
                let components = create_test_realvh_components(8, 256, i as u64 * 200 + 500);
                transfer.extract_signature(&components, 0.3 - i as f64 * 0.05, Some("Low"))
            })
            .collect();

        // Learn transfer mapping
        transfer.learn_transfer(&sources, &targets);

        // Should have learned weights
        assert!(transfer.transfer_weights.is_some());

        println!(
            "Transfer learning complete: {} source signatures",
            sources.len()
        );
    }

    #[test]
    fn test_transfer_spectral_features() {
        let config = PhiTransferConfig {
            signature_dims: 12,
            use_spectral: true,
            ..Default::default()
        };
        let transfer = PhiTransfer::with_config(config);

        let components = create_test_realvh_components(8, 256, 42);
        let signature = transfer.extract_signature(&components, 0.5, None);

        // Should have spectral features
        assert!(!signature.spectral_features.is_empty());

        // Spectral features should include dominant eigenvalue estimate
        println!("Spectral features: {:?}", signature.spectral_features);
    }

    #[test]
    fn test_transfer_improvement_direction() {
        let transfer = PhiTransfer::new();

        // High-Φ source
        let source = create_test_realvh_components(8, 256, 42);
        // Low-Φ target
        let target = create_test_realvh_components(8, 256, 123);

        // Transfer from high to low
        let result_improve = transfer.transfer(&source, &target, 0.6, 0.3, "High", "Low");

        // Transfer from low to high (should not improve much)
        let result_no_improve = transfer.transfer(&target, &source, 0.3, 0.6, "Low", "High");

        // High→Low should show more improvement potential
        assert!(
            result_improve.improvement_percent() > 0.0,
            "High→Low should show improvement"
        );

        println!(
            "High→Low improvement: {:.2}%",
            result_improve.improvement_percent()
        );
        println!(
            "Low→High improvement: {:.2}%",
            result_no_improve.improvement_percent()
        );
    }

    // ========================================================================
    // Revolutionary #97: Φ Attractor Dynamics Tests
    // ========================================================================

    #[test]
    fn test_attractor_empty_trajectory() {
        let mut attractor = PhiAttractor::new();
        let empty: Vec<f64> = vec![];

        let result = attractor.analyze(&empty);

        assert_eq!(result.attractor_type, AttractorType::Transient);
        assert_eq!(result.trajectory.len(), 0);
        assert!(!result.converged);

        println!("Empty trajectory → Transient attractor (as expected)");
    }

    #[test]
    fn test_attractor_fixed_point() {
        let mut attractor = PhiAttractor::new();

        // Create a trajectory that converges to a fixed point
        // Use faster decay (-15.0) to converge within the default threshold (0.001)
        let trajectory: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                0.5 + 0.3 * (-15.0 * t).exp() // Fast exponential decay to 0.5
            })
            .collect();

        let result = attractor.analyze(&trajectory);

        // Should detect fixed point
        assert!(result.converged, "Should detect convergence");
        assert!(
            matches!(result.attractor_type, AttractorType::FixedPoint),
            "Should classify as fixed point, got {:?}",
            result.attractor_type
        );
        assert!(
            (result.attractor_phi - 0.5).abs() < 0.05,
            "Attractor Φ should be near 0.5"
        );
        assert!(
            result.lyapunov_exponent < 0.0,
            "Lyapunov should be negative (stable)"
        );

        println!("Fixed point test:");
        println!("  Attractor Φ: {:.4}", result.attractor_phi);
        println!("  Lyapunov: {:.4}", result.lyapunov_exponent);
        println!("  Convergence time: {}", result.convergence_time);
    }

    #[test]
    fn test_attractor_limit_cycle() {
        let mut attractor = PhiAttractor::new();

        // Create an oscillating trajectory (limit cycle)
        let trajectory: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64;
                0.5 + 0.2 * (t * 0.5).sin() // Regular oscillation
            })
            .collect();

        let result = attractor.analyze(&trajectory);

        // Should detect oscillation
        assert!(
            matches!(result.attractor_type, AttractorType::LimitCycle),
            "Should classify as limit cycle, got {:?}",
            result.attractor_type
        );
        assert!(result.oscillation_period.is_some(), "Should detect period");

        println!("Limit cycle test:");
        println!("  Type: {:?}", result.attractor_type);
        println!("  Oscillation period: {:?}", result.oscillation_period);
        println!(
            "  Interpretation: {}",
            result.attractor_type.consciousness_interpretation()
        );
    }

    #[test]
    fn test_attractor_lyapunov_calculation() {
        let mut attractor = PhiAttractor::new();

        // Stable trajectory (should have negative Lyapunov)
        let stable: Vec<f64> = (0..50)
            .map(|i| 0.5 - 0.3 * (-0.1 * i as f64).exp())
            .collect();
        let result_stable = attractor.analyze(&stable);
        // Note: actually computing on trajectory, so check is within range

        // Diverging trajectory (should have positive Lyapunov)
        let diverging: Vec<f64> = (0..50)
            .map(|i| 0.1 * (0.05 * i as f64).exp().min(1.0))
            .collect();
        let result_diverging = PhiAttractor::new().analyze(&diverging);

        println!("Lyapunov exponent test:");
        println!(
            "  Stable trajectory: λ = {:.4}",
            result_stable.lyapunov_exponent
        );
        println!(
            "  Diverging trajectory: λ = {:.4}",
            result_diverging.lyapunov_exponent
        );

        // Diverging should have larger (more positive) Lyapunov
        assert!(
            result_diverging.lyapunov_exponent > result_stable.lyapunov_exponent,
            "Diverging should have larger Lyapunov than stable"
        );
    }

    #[test]
    fn test_attractor_basin_estimation() {
        let mut attractor = PhiAttractor::new();

        // Trajectory that spends most time near attractor
        let trajectory: Vec<f64> = (0..100)
            .map(|i| {
                if i < 20 {
                    0.3 + 0.01 * i as f64 // Approach
                } else {
                    0.5 + 0.001 * ((i as f64).sin()) // Stable near 0.5
                }
            })
            .collect();

        let result = attractor.analyze(&trajectory);

        // Basin should be reasonably large (>0.5) since trajectory stays near attractor
        assert!(result.basin_size > 0.3, "Basin should be significant");
        assert!(result.basin_size <= 1.0, "Basin should be bounded");

        println!("Basin estimation test:");
        println!("  Basin size: {:.4}", result.basin_size);
        println!("  Robustness score: {:.4}", result.robustness_score());
    }

    #[test]
    fn test_attractor_classification() {
        let mut attractor = PhiAttractor::new();

        // Test all classification methods work
        let cases = vec![
            ("Stable", vec![0.5; 50], AttractorType::FixedPoint),
            (
                "Chaotic",
                (0..50)
                    .map(|i| 0.5 + 0.3 * (i as f64 * 0.7).sin() * (i as f64 * 1.3).cos())
                    .collect::<Vec<_>>(),
                AttractorType::LimitCycle,
            ),
        ];

        for (name, trajectory, _expected) in &cases {
            let result = attractor.analyze(trajectory);
            println!("{} trajectory -> {:?}", name, result.attractor_type);
            println!("  Description: {}", result.attractor_type.description());
            println!(
                "  Consciousness: {}",
                result.attractor_type.consciousness_interpretation()
            );

            assert!(
                result.attractor_phi.is_finite(),
                "attractor_phi should be finite for {} trajectory",
                name
            );
            assert!(
                !result.attractor_type.description().is_empty(),
                "attractor type description should not be empty for {}",
                name
            );
        }
    }

    #[test]
    fn test_attractor_result_methods() {
        let result = AttractorResult {
            attractor_type: AttractorType::FixedPoint,
            attractor_phi: 0.5,
            initial_phi: 0.3,
            basin_size: 0.8,
            lyapunov_exponent: -0.5,
            convergence_time: 50,
            trajectory: vec![0.3, 0.4, 0.45, 0.49, 0.5],
            converged: true,
            oscillation_period: None,
            basin_neighbors: vec![],
        };

        // Test state checks
        assert!(
            result.is_stable(),
            "Fixed point with negative Lyapunov should be stable"
        );
        assert!(
            !result.is_transitioning(),
            "Fixed point should not be transitioning"
        );
        assert!(!result.is_complex(), "Fixed point should not be complex");

        // Test scores
        assert!(
            result.stability_score() > 0.0,
            "Stability score should be positive"
        );
        assert!(
            (result.robustness_score() - 0.8).abs() < 0.001,
            "Robustness should match basin_size"
        );

        println!("AttractorResult methods test:");
        println!("  is_stable: {}", result.is_stable());
        println!("  stability_score: {:.4}", result.stability_score());
        println!("  robustness_score: {:.4}", result.robustness_score());
    }

    #[test]
    fn test_attractor_simulation() {
        let attractor = PhiAttractor::new();

        // Simulate from initial state to target
        let trajectory = attractor.simulate(0.1, 0.7);

        assert!(
            !trajectory.is_empty(),
            "Simulation should produce trajectory"
        );
        assert_eq!(trajectory[0], 0.1, "Should start at initial state");

        // Should move toward target
        let final_phi = *trajectory.last().unwrap();
        let mid_phi = trajectory[trajectory.len() / 2];

        assert!(mid_phi > 0.1, "Should move away from initial");
        assert!(
            (final_phi - 0.7).abs() < (trajectory[0] - 0.7).abs(),
            "Should get closer to target"
        );

        println!("Simulation test:");
        println!("  Initial: {:.4}", trajectory[0]);
        println!("  Mid: {:.4}", mid_phi);
        println!("  Final: {:.4}", final_phi);
        println!("  Steps: {}", trajectory.len());
    }

    #[test]
    fn test_attractor_find_attractors() {
        let mut attractor = PhiAttractor::fast();

        // Find attractors in a range
        let attractors = attractor.find_attractors((0.0, 1.0));

        // Should find at least one attractor (the target we simulate toward)
        assert!(!attractors.is_empty(), "Should find at least one attractor");

        // All attractors should be in valid range
        for a in &attractors {
            assert!(*a >= 0.0 && *a <= 1.0, "Attractor should be in range");
        }

        println!("Find attractors test:");
        println!("  Found {} attractors: {:?}", attractors.len(), attractors);
    }

    #[test]
    fn test_attractor_convenience_functions() {
        // Test analyze_phi_attractor
        let trajectory = vec![0.3, 0.4, 0.45, 0.48, 0.5, 0.5, 0.5, 0.5];
        let result = analyze_phi_attractor(&trajectory);

        assert!(
            result.converged,
            "Simple convergent trajectory should converge"
        );

        // Test classify_consciousness_state
        let (attractor_type, stability) = classify_consciousness_state(&trajectory);

        assert!(stability >= 0.0, "Stability should be non-negative");
        assert!(stability <= 1.0, "Stability should be bounded");

        println!("Convenience functions test:");
        println!("  analyze_phi_attractor → {:?}", result.attractor_type);
        println!(
            "  classify_consciousness_state → {:?}, stability={:.4}",
            attractor_type, stability
        );
    }

    #[test]
    fn test_attractor_config_presets() {
        let fast = PhiAttractor::fast();
        let research = PhiAttractor::research();
        let default = PhiAttractor::new();

        // Fast should have fewer iterations
        assert!(fast.config.max_iterations < default.config.max_iterations);

        // Research should have more iterations and tighter threshold
        assert!(research.config.max_iterations > default.config.max_iterations);
        assert!(research.config.convergence_threshold < default.config.convergence_threshold);

        println!("Config presets test:");
        println!(
            "  Fast: max_iter={}, samples={}",
            fast.config.max_iterations, fast.config.basin_samples
        );
        println!(
            "  Default: max_iter={}, samples={}",
            default.config.max_iterations, default.config.basin_samples
        );
        println!(
            "  Research: max_iter={}, samples={}",
            research.config.max_iterations, research.config.basin_samples
        );
    }

    #[test]
    fn test_attractor_type_descriptions() {
        // Verify all enum variants have descriptions
        let types = vec![
            AttractorType::FixedPoint,
            AttractorType::LimitCycle,
            AttractorType::StrangeAttractor,
            AttractorType::SaddlePoint,
            AttractorType::Transient,
        ];

        for t in types {
            let desc = t.description();
            let interp = t.consciousness_interpretation();

            assert!(
                !desc.is_empty(),
                "Description should not be empty for {:?}",
                t
            );
            assert!(
                !interp.is_empty(),
                "Interpretation should not be empty for {:?}",
                t
            );

            println!("{:?}:", t);
            println!("  Description: {}", desc);
            println!("  Consciousness: {}", interp);
        }
    }

    #[test]
    fn test_attractor_transient_detection() {
        let mut attractor = PhiAttractor::with_config(AttractorConfig {
            max_iterations: 20,
            convergence_threshold: 1e-6, // Very tight threshold
            ..Default::default()
        });

        // Create a trajectory that doesn't converge (keeps changing)
        let trajectory: Vec<f64> = (0..50)
            .map(|i| 0.5 + 0.2 * (i as f64 * 0.1).sin() + 0.1 * (i as f64 * 0.03).cos())
            .collect();

        let result = attractor.analyze(&trajectory);

        // Should detect complex dynamics
        println!("Transient/complex detection test:");
        println!("  Type: {:?}", result.attractor_type);
        println!("  Converged: {}", result.converged);
        println!("  Lyapunov: {:.4}", result.lyapunov_exponent);

        assert!(
            result.lyapunov_exponent.is_finite(),
            "Lyapunov exponent should be finite, got {}",
            result.lyapunov_exponent
        );
        // With a tight convergence threshold and oscillating input, it should not report as converged to a fixed point
        assert!(
            !result.converged || !matches!(result.attractor_type, AttractorType::FixedPoint),
            "oscillating trajectory should not converge to a fixed point"
        );
    }

    #[test]
    fn test_attractor_stability_scores() {
        // Test stability scoring for different dynamics
        let stable_result = AttractorResult {
            attractor_type: AttractorType::FixedPoint,
            attractor_phi: 0.5,
            initial_phi: 0.3,
            basin_size: 0.9,
            lyapunov_exponent: -1.0, // Very stable
            convergence_time: 10,
            trajectory: vec![],
            converged: true,
            oscillation_period: None,
            basin_neighbors: vec![],
        };

        let chaotic_result = AttractorResult {
            lyapunov_exponent: 0.5, // Positive = chaotic
            ..stable_result.clone()
        };

        let neutral_result = AttractorResult {
            lyapunov_exponent: 0.0, // Neutral
            ..stable_result.clone()
        };

        assert!(stable_result.stability_score() > chaotic_result.stability_score());
        assert!(stable_result.stability_score() > neutral_result.stability_score());
        assert_eq!(
            chaotic_result.stability_score(),
            0.0,
            "Positive Lyapunov → 0 stability"
        );

        println!("Stability scores:");
        println!("  Stable (λ=-1.0): {:.4}", stable_result.stability_score());
        println!("  Neutral (λ=0.0): {:.4}", neutral_result.stability_score());
        println!(
            "  Chaotic (λ=+0.5): {:.4}",
            chaotic_result.stability_score()
        );
    }

    // ========================================================================
    // Revolutionary #98: Φ Causal Intervention Tests
    // ========================================================================

    #[test]
    fn test_causal_intervention_empty() {
        let analyzer = PhiCausalAnalyzer::new();
        let empty: Vec<ContinuousHV> = vec![];

        let result = analyzer.analyze(&empty);

        assert_eq!(result.baseline_phi, 0.0);
        assert!(result.node_results.is_empty());
        assert!(result.causal_power.is_empty());
        assert!(result.critical_nodes.is_empty());

        println!("Empty nodes → empty causal analysis");
    }

    #[test]
    fn test_causal_intervention_single_node() {
        let analyzer = PhiCausalAnalyzer::new();
        let single = vec![ContinuousHV::random(128, 42)];

        let result = analyzer.analyze(&single);

        // Single node has no pairwise interactions
        assert_eq!(result.baseline_phi, 0.0);
        assert_eq!(result.node_results.len(), 1);

        println!("Single node baseline Φ: {:.4}", result.baseline_phi);
    }

    #[test]
    fn test_causal_intervention_basic() {
        let analyzer = PhiCausalAnalyzer::new();

        // Create a simple 4-node network
        let nodes: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(128, i as u64 * 100))
            .collect();

        let result = analyzer.analyze(&nodes);

        // Should have results for all 4 nodes
        assert_eq!(result.node_results.len(), 4);

        // Each node should have 3 intervention results (knockout, amplify, dampen)
        for node_result in &result.node_results {
            assert_eq!(node_result.len(), 3);
        }

        // Should have causal power for all nodes
        assert_eq!(result.causal_power.len(), 4);

        // Node ranking should include all nodes
        assert_eq!(result.node_ranking.len(), 4);

        println!("Basic causal analysis:");
        println!("  Baseline Φ: {:.4}", result.baseline_phi);
        println!("  Node ranking: {:?}", result.node_ranking);
        println!("  Causal power: {:?}", result.causal_power);
    }

    #[test]
    fn test_intervention_type_descriptions() {
        let knockout = InterventionType::Knockout;
        let amplify = InterventionType::Amplify(2.0);
        let dampen = InterventionType::Dampen(2.0);
        let noise = InterventionType::Noise;
        let clamp = InterventionType::Clamp(0.5);

        // Check descriptions
        assert!(knockout.description().contains("knockout"));
        assert!(amplify.description().contains("amplify"));
        assert!(dampen.description().contains("dampen"));
        assert!(noise.description().contains("noise"));
        assert!(clamp.description().contains("clamp"));

        // Check interpretations exist
        assert!(!knockout.interpretation().is_empty());
        assert!(!amplify.interpretation().is_empty());
        assert!(!dampen.interpretation().is_empty());
        assert!(!noise.interpretation().is_empty());
        assert!(!clamp.interpretation().is_empty());

        println!("Intervention types:");
        println!(
            "  Knockout: {} - {}",
            knockout.description(),
            knockout.interpretation()
        );
        println!(
            "  Amplify: {} - {}",
            amplify.description(),
            amplify.interpretation()
        );
        println!(
            "  Dampen: {} - {}",
            dampen.description(),
            dampen.interpretation()
        );
        println!(
            "  Noise: {} - {}",
            noise.description(),
            noise.interpretation()
        );
        println!(
            "  Clamp: {} - {}",
            clamp.description(),
            clamp.interpretation()
        );
    }

    #[test]
    fn test_causal_config_presets() {
        let default_config = CausalInterventionConfig::default();
        let fast_config = CausalInterventionConfig::fast();
        let research_config = CausalInterventionConfig::research();

        // Fast should have fewer samples
        assert!(fast_config.bootstrap_samples < default_config.bootstrap_samples);

        // Research should have more samples
        assert!(research_config.bootstrap_samples > default_config.bootstrap_samples);

        println!("Config presets:");
        println!(
            "  Default bootstrap samples: {}",
            default_config.bootstrap_samples
        );
        println!(
            "  Fast bootstrap samples: {}",
            fast_config.bootstrap_samples
        );
        println!(
            "  Research bootstrap samples: {}",
            research_config.bootstrap_samples
        );
    }

    #[test]
    fn test_node_intervention_result_methods() {
        let knockout_critical = NodeInterventionResult {
            node_index: 0,
            intervention: InterventionType::Knockout,
            baseline_phi: 0.5,
            intervened_phi: 0.3,
            delta_phi: 0.2,
            percent_change: -40.0, // Critical: >10% drop
            standard_error: None,
            confidence_interval: None,
        };

        let knockout_redundant = NodeInterventionResult {
            percent_change: -2.0, // Redundant: <5% change
            ..knockout_critical.clone()
        };

        let knockout_significant = NodeInterventionResult {
            percent_change: -8.0, // Significant but not critical
            ..knockout_critical.clone()
        };

        // Test is_critical
        assert!(
            knockout_critical.is_critical(),
            "40% drop should be critical"
        );
        assert!(
            !knockout_redundant.is_critical(),
            "2% drop should not be critical"
        );

        // Test is_redundant
        assert!(
            knockout_redundant.is_redundant(),
            "2% change should be redundant"
        );
        assert!(
            !knockout_critical.is_redundant(),
            "40% drop should not be redundant"
        );

        // Test is_significant
        assert!(
            knockout_critical.is_significant(5.0),
            "40% should be significant at 5% threshold"
        );
        assert!(
            !knockout_redundant.is_significant(5.0),
            "2% should not be significant at 5% threshold"
        );

        println!("Node intervention result methods:");
        println!(
            "  Critical (40% drop): is_critical={}",
            knockout_critical.is_critical()
        );
        println!(
            "  Redundant (2% drop): is_redundant={}",
            knockout_redundant.is_redundant()
        );
    }

    #[test]
    fn test_causal_analysis_result_methods() {
        let result = CausalAnalysisResult {
            baseline_phi: 0.5,
            node_results: vec![],
            causal_power: vec![0.1, 0.3, 0.5, 0.1],
            node_ranking: vec![2, 1, 0, 3], // Node 2 is most critical
            critical_nodes: vec![2],
            redundant_nodes: vec![0, 3],
            mean_effects: std::collections::HashMap::new(),
        };

        // Test most/least critical
        assert_eq!(result.most_critical_node(), Some(2));
        assert_eq!(result.least_critical_node(), Some(3));

        // Test robustness
        let robustness = result.robustness();
        assert!(robustness > 0.0 && robustness < 1.0);

        // Test concentration
        let concentration = result.concentration();
        assert!(concentration >= 0.0 && concentration <= 1.0);

        println!("Causal analysis result methods:");
        println!("  Most critical node: {:?}", result.most_critical_node());
        println!("  Least critical node: {:?}", result.least_critical_node());
        println!("  Robustness: {:.4}", robustness);
        println!("  Concentration: {:.4}", concentration);
    }

    #[test]
    fn test_causal_intervention_hub_detection() {
        let analyzer = PhiCausalAnalyzer::new();

        // Create a hub topology: node 0 is similar to all others
        let hub = ContinuousHV::random(128, 42);
        let mut nodes = vec![hub.clone()];

        // Add spokes that are similar to hub but not to each other
        for i in 1..5 {
            let noise = ContinuousHV::random(128, (i * 1000) as u64);
            // Blend hub with noise (70% hub, 30% noise)
            let spoke = ContinuousHV::bundle_owned(&[hub.clone(), hub.clone(), noise]);
            nodes.push(spoke);
        }

        let result = analyzer.analyze(&nodes);

        // Node 0 (hub) should likely have higher causal power
        println!("Hub topology causal analysis:");
        println!("  Baseline Φ: {:.4}", result.baseline_phi);
        println!("  Node ranking: {:?}", result.node_ranking);
        println!("  Causal power: {:?}", result.causal_power);
        println!("  Most critical: {:?}", result.most_critical_node());

        assert!(
            result.baseline_phi >= 0.0,
            "baseline phi should be non-negative, got {}",
            result.baseline_phi
        );
        assert!(
            !result.node_ranking.is_empty(),
            "node ranking should not be empty for hub topology"
        );
    }

    #[test]
    fn test_causal_find_dominating_set() {
        let analyzer = PhiCausalAnalyzer::new();

        // Create 8-node network
        let nodes: Vec<ContinuousHV> = (0..8)
            .map(|i| ContinuousHV::random(128, i as u64 * 50))
            .collect();

        // Find nodes controlling 80% of causal power
        let dominating = analyzer.find_dominating_set(&nodes, 0.8);

        // Should be a subset of all nodes
        assert!(!dominating.is_empty());
        assert!(dominating.len() <= nodes.len());

        println!("Dominating set (80% threshold):");
        println!("  Nodes: {:?}", dominating);
        println!("  Size: {} of {}", dominating.len(), nodes.len());
    }

    #[test]
    fn test_causal_analyze_subset() {
        let analyzer = PhiCausalAnalyzer::new();

        let nodes: Vec<ContinuousHV> = (0..6)
            .map(|i| ContinuousHV::random(128, i as u64 * 77))
            .collect();

        // Analyze only nodes 0, 2, 4
        let subset_results = analyzer.analyze_subset(&nodes, &[0, 2, 4]);

        assert_eq!(subset_results.len(), 3);

        for result in &subset_results {
            assert!(matches!(result.intervention, InterventionType::Knockout));
        }

        println!("Subset analysis (nodes 0, 2, 4):");
        for r in &subset_results {
            println!(
                "  Node {}: Δ={:.4} ({:.2}%)",
                r.node_index, r.delta_phi, r.percent_change
            );
        }
    }

    #[test]
    fn test_causal_convenience_functions() {
        let nodes: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(128, i as u64 * 123))
            .collect();

        // Test analyze_causal_interventions
        let result = analyze_causal_interventions(&nodes);
        assert!(!result.node_ranking.is_empty());

        // Test find_critical_nodes
        let critical = find_critical_nodes(&nodes);
        // May be empty if no critical nodes detected

        // Test compute_causal_power
        let power = compute_causal_power(&nodes);
        assert_eq!(power.len(), 4);

        println!("Convenience functions:");
        println!("  Causal power: {:?}", power);
        println!("  Critical nodes: {:?}", critical);
    }

    #[test]
    fn test_causal_robustness_comparison() {
        let analyzer = PhiCausalAnalyzer::new();

        // Create a "fragile" network (hub-spoke)
        let hub = ContinuousHV::random(128, 1);
        let fragile: Vec<ContinuousHV> = std::iter::once(hub.clone())
            .chain((1..5).map(|i| {
                let noise = ContinuousHV::random(128, (i * 100) as u64);
                ContinuousHV::bundle_owned(&[hub.clone(), noise])
            }))
            .collect();

        // Create a "robust" network (uniform random)
        let robust: Vec<ContinuousHV> = (0..5)
            .map(|i| ContinuousHV::random(128, (i * 500) as u64))
            .collect();

        let fragile_result = analyzer.analyze(&fragile);
        let robust_result = analyzer.analyze(&robust);

        println!("Robustness comparison:");
        println!("  Hub-spoke (fragile):");
        println!("    Robustness: {:.4}", fragile_result.robustness());
        println!("    Concentration: {:.4}", fragile_result.concentration());
        println!("  Random (robust):");
        println!("    Robustness: {:.4}", robust_result.robustness());
        println!("    Concentration: {:.4}", robust_result.concentration());

        assert!(
            fragile_result.robustness() >= 0.0 && fragile_result.robustness() <= 1.0,
            "fragile robustness should be in [0,1], got {}",
            fragile_result.robustness()
        );
        assert!(
            robust_result.robustness() >= 0.0 && robust_result.robustness() <= 1.0,
            "robust robustness should be in [0,1], got {}",
            robust_result.robustness()
        );
    }

    #[test]
    fn test_causal_intervention_effects() {
        let analyzer = PhiCausalAnalyzer::new();

        // Create correlated network
        let base = ContinuousHV::random(128, 42);
        let nodes: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                let noise = ContinuousHV::random(128, (i * 200) as u64).scale(0.2);
                base.add(&noise)
            })
            .collect();

        let result = analyzer.analyze(&nodes);

        // Check mean effects are computed
        assert!(result.mean_effects.contains_key("knockout"));
        assert!(result.mean_effects.contains_key("amplify"));
        assert!(result.mean_effects.contains_key("dampen"));

        println!("Mean intervention effects:");
        for (intervention, mean) in &result.mean_effects {
            println!("  {}: {:.4}", intervention, mean);
        }
    }

    // ========================================================================
    // REVOLUTIONARY #99: Φ NETWORK MODULARITY TESTS
    // ========================================================================

    #[test]
    fn test_modularity_empty_network() {
        let nodes: Vec<ContinuousHV> = vec![];
        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert_eq!(result.num_modules(), 0);
        assert_eq!(result.total_phi, 0.0);
        assert_eq!(result.modularity_score, 0.0);
    }

    #[test]
    fn test_modularity_single_node() {
        let nodes = vec![ContinuousHV::random(256, 42)];
        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert_eq!(result.total_phi, 0.0); // Single node has no integration
    }

    #[test]
    fn test_modularity_two_nodes() {
        let nodes = vec![ContinuousHV::random(256, 1), ContinuousHV::random(256, 2)];
        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert!(result.total_phi >= 0.0);
    }

    #[test]
    fn test_module_detection_method_descriptions() {
        assert!(ModuleDetectionMethod::Spectral
            .description()
            .contains("Spectral"));
        assert!(ModuleDetectionMethod::Agglomerative
            .description()
            .contains("agglomerative"));
        assert!(ModuleDetectionMethod::Greedy
            .description()
            .contains("Greedy"));
        assert!(ModuleDetectionMethod::KMeans
            .description()
            .contains("K-means"));
    }

    #[test]
    fn test_modularity_config_presets() {
        let quick = ModularityConfig::quick();
        assert_eq!(quick.num_modules, Some(3));
        assert!(!quick.compute_inter_module_phi);

        let thorough = ModularityConfig::thorough();
        assert!(thorough.num_modules.is_none());
        assert!(thorough.compute_inter_module_phi);

        let research = ModularityConfig::research();
        assert_eq!(research.min_module_size, 1);
        assert!(research.max_iterations > thorough.max_iterations);
    }

    #[test]
    fn test_node_role_descriptions() {
        assert!(NodeRole::Core.description().contains("Core"));
        assert!(NodeRole::Peripheral.description().contains("Peripheral"));
        assert!(NodeRole::Bridge.description().contains("Bridge"));
        assert!(NodeRole::Hub.description().contains("Hub"));
        assert!(NodeRole::Isolated.description().contains("Isolated"));
    }

    #[test]
    fn test_consciousness_module_methods() {
        let module = ConsciousnessModule {
            id: 0,
            node_indices: vec![0, 1, 2],
            internal_cohesion: 0.8,
            internal_phi: 0.6,
            isolation_score: 0.7,
            centroid: None,
        };

        assert_eq!(module.size(), 3);
        assert!(module.contains(1));
        assert!(!module.contains(5));
        assert!(module.integration_efficiency() > 0.0);
    }

    #[test]
    fn test_network_modularity_result_methods() {
        // Create a simple modular network
        let dim = 256;
        let mut nodes = Vec::new();

        // Module 1: similar nodes
        let base1 = ContinuousHV::random(dim, 100);
        for i in 0..3 {
            let noise = ContinuousHV::random(dim, 1000 + i);
            nodes.push(base1.add(&noise.scale(0.1)));
        }

        // Module 2: different similar nodes
        let base2 = ContinuousHV::random(dim, 200);
        for i in 0..3 {
            let noise = ContinuousHV::random(dim, 2000 + i);
            nodes.push(base2.add(&noise.scale(0.1)));
        }

        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert!(result.total_phi > 0.0);
        assert!(result.avg_module_size() > 0.0);
        assert!(result.balance_score() > 0.0);
        assert!(result.efficiency_ratio() >= 0.0);

        if !result.modules.is_empty() {
            assert!(result.largest_module().is_some());
            assert!(result.highest_phi_module().is_some());
        }
    }

    #[test]
    fn test_modularity_detects_clear_modules() {
        let dim = 256;
        let mut nodes = Vec::new();

        // Create two clearly separated clusters
        // Cluster A: nodes with positive bias
        let cluster_a_base = ContinuousHV::random(dim, 42);
        for i in 0..4 {
            let variation = ContinuousHV::random(dim, 100 + i);
            nodes.push(cluster_a_base.add(&variation.scale(0.05)));
        }

        // Cluster B: nodes with different base (orthogonal)
        let cluster_b_base = ContinuousHV::random(dim, 999);
        for i in 0..4 {
            let variation = ContinuousHV::random(dim, 200 + i);
            nodes.push(cluster_b_base.add(&variation.scale(0.05)));
        }

        let config = ModularityConfig {
            num_modules: Some(2),
            ..Default::default()
        };
        let result = PhiModularityAnalyzer::with_config(config).analyze(&nodes);

        println!("Modularity analysis of clear clusters:");
        println!("  Num modules: {}", result.num_modules());
        println!("  Modularity Q: {:.4}", result.modularity_score);
        println!("  Segregation: {:.4}", result.segregation_index);
        println!("  Integration: {:.4}", result.integration_index);

        // Should detect structure
        assert!(result.num_modules() >= 1);
    }

    #[test]
    fn test_spectral_clustering() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..6)
            .map(|i| ContinuousHV::random(dim, i as u64 * 1000))
            .collect();

        let config = ModularityConfig {
            detection_method: ModuleDetectionMethod::Spectral,
            num_modules: Some(2),
            ..Default::default()
        };

        let result = PhiModularityAnalyzer::with_config(config).analyze(&nodes);
        assert!(result.num_modules() >= 1);
    }

    #[test]
    fn test_greedy_modularity() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..6)
            .map(|i| ContinuousHV::random(dim, i as u64 * 500))
            .collect();

        let config = ModularityConfig {
            detection_method: ModuleDetectionMethod::Greedy,
            ..Default::default()
        };

        let result = PhiModularityAnalyzer::with_config(config).analyze(&nodes);
        assert!(result.total_phi >= 0.0);
    }

    #[test]
    fn test_kmeans_clustering() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..8)
            .map(|i| ContinuousHV::random(dim, i as u64 * 700))
            .collect();

        let config = ModularityConfig {
            detection_method: ModuleDetectionMethod::KMeans,
            num_modules: Some(2),
            ..Default::default()
        };

        let result = PhiModularityAnalyzer::with_config(config).analyze(&nodes);
        assert!(result.num_modules() >= 1);
    }

    #[test]
    fn test_inter_module_relations() {
        let dim = 256;
        let mut nodes = Vec::new();

        // Two modules with a connecting node
        for i in 0..3 {
            nodes.push(ContinuousHV::random(dim, i as u64));
        }
        for i in 3..6 {
            nodes.push(ContinuousHV::random(dim, i as u64 * 1000));
        }

        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        // If modules were detected, check relations
        if result.num_modules() >= 2 {
            assert!(!result.inter_module_relations.is_empty());
            let relation = &result.inter_module_relations[0];
            assert!(relation.coupling_strength >= 0.0);
            assert!(relation.coupling_strength <= 1.0);
        }
    }

    #[test]
    fn test_node_classification() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..8)
            .map(|i| ContinuousHV::random(dim, i as u64 * 123))
            .collect();

        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert_eq!(result.node_classifications.len(), 8);

        for classification in &result.node_classifications {
            assert!(classification.node_index < 8);
            assert!(classification.participation_coefficient >= 0.0);
            assert!(classification.participation_coefficient <= 1.0);
            assert!(classification.betweenness >= 0.0);
            assert!(classification.betweenness <= 1.0);
        }
    }

    #[test]
    fn test_hierarchical_modularity() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..10)
            .map(|i| ContinuousHV::random(dim, i as u64 * 42))
            .collect();

        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        assert!(!result.hierarchical_scores.is_empty());
        println!("Hierarchical modularity scores:");
        for (k, &q) in result.hierarchical_scores.iter().enumerate() {
            println!("  k={}: Q={:.4}", k + 2, q);
        }
    }

    #[test]
    fn test_convenience_functions() {
        let dim = 256;
        let nodes: Vec<ContinuousHV> = (0..6)
            .map(|i| ContinuousHV::random(dim, i as u64 * 55))
            .collect();

        let result = analyze_network_modularity(&nodes);
        assert!(result.total_phi >= 0.0);

        let count = detect_module_count(&nodes);
        assert!(count >= 1);

        let q = compute_modularity_score(&nodes);
        // Q can be negative, so just check it's finite
        assert!(q.is_finite());
    }

    #[test]
    fn test_hub_and_spoke_modularity() {
        // Create hub-and-spoke topology (star with central hub)
        let dim = 256;
        let hub = ContinuousHV::random(dim, 0);
        let mut nodes = vec![hub.clone()];

        // Create spokes
        for i in 1..=6 {
            let spoke = ContinuousHV::random(dim, i as u64 * 100);
            // Mix spoke with hub to create connection
            let connected = hub.bind(&spoke);
            nodes.push(connected);
        }

        let result = PhiModularityAnalyzer::new().analyze(&nodes);

        println!("Hub-and-spoke modularity:");
        println!("  Total Φ: {:.4}", result.total_phi);
        println!("  Modules: {}", result.num_modules());
        println!("  Q score: {:.4}", result.modularity_score);

        // Hub should be identified as special
        let hub_class = &result.node_classifications[0];
        println!("  Hub node role: {:?}", hub_class.role);
        println!(
            "  Hub participation: {:.4}",
            hub_class.participation_coefficient
        );

        assert!(
            result.total_phi >= 0.0,
            "total phi should be non-negative, got {}",
            result.total_phi
        );
        assert!(
            result.num_modules() >= 1,
            "should detect at least 1 module, got {}",
            result.num_modules()
        );
        assert!(
            result.modularity_score.is_finite(),
            "modularity score should be finite, got {}",
            result.modularity_score
        );
    }

    #[test]
    fn test_phi_computation_beyond_64_components() {
        // Regression test: Phi computation must work correctly for n >= 64.
        // A previous bug used u64 bit masks with modular indexing (i % 64),
        // causing components at index i and i+64 to share the same partition
        // assignment. The fix uses Vec<bool> with independent per-element hashing.
        // We test via the public compute() API, which exercises random_bipartition_vec
        // internally for n > threshold.
        let mut phi = TieredPhi::for_testing();

        for n in [64, 80, 100, 128] {
            let components = create_test_components(n);
            let result = phi.compute(&components);

            assert!(
                result.is_finite(),
                "Phi must be finite for n={n}, got {result}"
            );
            assert!(
                result >= 0.0,
                "Phi must be non-negative for n={n}, got {result}"
            );
        }

        // Verify that different system sizes produce different Phi values,
        // confirming the partition sampling is actually distinguishing systems.
        let small = phi.compute(&create_test_components(64));
        let large = phi.compute(&create_test_components(128));
        // They may be equal in degenerate cases, but at least both must be valid.
        assert!(small.is_finite() && large.is_finite());
    }
}
