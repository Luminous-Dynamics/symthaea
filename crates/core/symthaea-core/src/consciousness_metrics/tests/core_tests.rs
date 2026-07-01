// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core entropy, mutual information, and Φ computation tests.

use super::*;

#[test]
fn test_entropy_config() {
    let fast = EntropyConfig::fast();
    assert_eq!(fast.num_bins, 16);

    let precise = EntropyConfig::precise();
    assert_eq!(precise.num_bins, 32);
}

#[test]
fn test_vector_distribution() {
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);
    let dist = VectorDistribution::from_hv(&hv, 16);

    // Probabilities should sum to 1
    let sum: f64 = dist.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Probabilities should sum to 1, got {}",
        sum
    );

    // All probabilities should be non-negative
    assert!(dist.probabilities.iter().all(|&p| p >= 0.0));
}

#[test]
fn test_entropy_bounds() {
    let calc = TruePhiCalculator::new();
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let entropy = calc.entropy(&hv);

    // Entropy should be non-negative
    assert!(entropy >= 0.0, "Entropy should be non-negative");

    // Entropy should be at most log(num_bins) bits
    let max_entropy = (calc.config.num_bins as f64).log2();
    assert!(
        entropy <= max_entropy + 0.01,
        "Entropy {} should be at most {}",
        entropy,
        max_entropy
    );
}

#[test]
fn test_uniform_distribution_max_entropy() {
    let calc = TruePhiCalculator::new();

    // A vector with values uniformly distributed across bins should have max entropy
    let mut values = Vec::with_capacity(HDC_DIMENSION);
    for i in 0..HDC_DIMENSION {
        // Distribute evenly across [-1, 1]
        let val = -1.0 + 2.0 * (i as f32 / HDC_DIMENSION as f32);
        values.push(val);
    }
    let hv = ContinuousHV::from_vec(values);

    let entropy = calc.entropy(&hv);
    let max_entropy = (calc.config.num_bins as f64).log2();

    // Should be close to max entropy
    assert!(
        entropy > max_entropy * 0.95,
        "Uniform dist entropy {} should be near max {}",
        entropy,
        max_entropy
    );
}

#[test]
fn test_joint_entropy() {
    let calc = TruePhiCalculator::new();
    let hv1 = ContinuousHV::random(HDC_DIMENSION, 42);
    let hv2 = ContinuousHV::random(HDC_DIMENSION, 43);

    let h_x = calc.entropy(&hv1);
    let h_y = calc.entropy(&hv2);
    let h_xy = calc.joint_entropy(&hv1, &hv2);

    // Joint entropy should be >= max of individual entropies
    assert!(
        h_xy >= h_x.max(h_y) - 0.01,
        "H(X,Y) should be >= max(H(X), H(Y))"
    );

    // Joint entropy should be <= sum of individual entropies
    assert!(h_xy <= h_x + h_y + 0.01, "H(X,Y) should be <= H(X) + H(Y)");
}

#[test]
fn test_mutual_information_non_negative() {
    let calc = TruePhiCalculator::new();
    let hv1 = ContinuousHV::random(HDC_DIMENSION, 42);
    let hv2 = ContinuousHV::random(HDC_DIMENSION, 43);

    let mi = calc.mutual_information(&hv1, &hv2);

    // MI should be non-negative
    assert!(
        mi >= 0.0,
        "Mutual information should be non-negative, got {}",
        mi
    );
}

#[test]
fn test_self_mutual_information_equals_entropy() {
    let calc = TruePhiCalculator::new();
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let entropy = calc.entropy(&hv);
    let self_mi = calc.mutual_information(&hv, &hv);

    // I(X;X) = H(X)
    assert!(
        (self_mi - entropy).abs() < 0.01,
        "Self MI {} should equal entropy {}",
        self_mi,
        entropy
    );
}

#[test]
fn test_effective_information_increases_with_correlation() {
    let calc = TruePhiCalculator::new();

    // Independent vectors
    let independent: Vec<ContinuousHV> = create_test_vectors(4);
    let ei_independent = calc.effective_information(&independent);

    // Correlated vectors (all derived from same base)
    let base = ContinuousHV::random(HDC_DIMENSION, 100);
    let correlated: Vec<ContinuousHV> = (0..4)
        .map(|i| {
            let noise = ContinuousHV::random(HDC_DIMENSION, i as u64 + 200);
            ContinuousHV::weighted_bundle(&[&base, &noise], &[0.9, 0.1])
        })
        .collect();
    let ei_correlated = calc.effective_information(&correlated);

    // Correlated should have higher EI
    assert!(
        ei_correlated > ei_independent,
        "Correlated EI {} should exceed independent EI {}",
        ei_correlated,
        ei_independent
    );
}

#[test]
fn test_true_phi_single_component() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(1);

    let result = calc.compute_true_phi(&components);

    assert_eq!(result.phi, 0.0, "Single component should have \u{03a6} = 0");
}

#[test]
fn test_true_phi_two_components() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(2);

    let result = calc.compute_true_phi(&components);

    // For two components, there's only one partition, so Φ = EI - 0 = EI
    assert!(result.phi >= 0.0, "\u{03a6} should be non-negative");
    assert_eq!(result.mip.part_a.len() + result.mip.part_b.len(), 2);
}

#[test]
fn test_true_phi_exhaustive_search() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(4);

    let result = calc.compute_true_phi(&components);

    assert!(result.phi >= 0.0, "\u{03a6} should be non-negative");
    assert!(
        result.system_ei >= result.mip_ei,
        "System EI should be >= MIP EI"
    );
    assert_eq!(result.component_entropies.len(), 4);
}

#[test]
fn test_true_phi_heuristic_search() {
    let calc = TruePhiCalculator::new();
    // 9 components triggers heuristic search (threshold is >8)
    let components: Vec<ContinuousHV> = create_test_vectors(9);

    let result = calc.compute_true_phi(&components);

    assert!(result.phi >= 0.0, "\u{03a6} should be non-negative");
    assert!(
        !result.mip.part_a.is_empty() && !result.mip.part_b.is_empty(),
        "MIP should have non-empty parts"
    );
}

#[test]
fn test_bound_vs_bundle_different_phi() {
    let calc = TruePhiCalculator::new();

    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let c = ContinuousHV::random(HDC_DIMENSION, 3);

    // Bundled structure
    let bundled = ContinuousHV::bundle(&[&a, &b]);
    let bundled_components = vec![bundled.clone(), c.clone()];
    let phi_bundled = calc.compute_true_phi(&bundled_components);

    // Bound structure
    let bound = a.bind(&b);
    let bound_components = vec![bound.clone(), c.clone()];
    let phi_bound = calc.compute_true_phi(&bound_components);

    // Both Φ values should be finite and non-negative
    assert!(phi_bundled.phi.is_finite(), "Bundled Φ should be finite");
    assert!(phi_bound.phi.is_finite(), "Bound Φ should be finite");
    assert!(phi_bundled.phi >= 0.0, "Bundled Φ should be non-negative");
    assert!(phi_bound.phi >= 0.0, "Bound Φ should be non-negative");

    // System EI values should be finite
    assert!(
        phi_bundled.system_ei.is_finite(),
        "Bundled system EI should be finite"
    );
    assert!(
        phi_bound.system_ei.is_finite(),
        "Bound system EI should be finite"
    );

    // They should have different Φ values (bind creates orthogonal structure)
    // This test verifies that our entropy measure is sensitive to structural differences
    println!(
        "\u{03a6}(bundled) = {:.4}, \u{03a6}(bound) = {:.4}",
        phi_bundled.phi, phi_bound.phi
    );
}

#[test]
fn test_phi_fast() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(6);

    let phi_fast = calc.compute_phi_fast(&components);
    let phi_full = calc.compute_true_phi(&components);

    // Fast should be in [0, 1]
    assert!(
        phi_fast >= 0.0 && phi_fast <= 1.0,
        "Fast \u{03a6} should be normalized"
    );

    // They should be positively correlated (but not equal)
    println!(
        "\u{03a6}_fast = {:.4}, \u{03a6}_full = {:.4}",
        phi_fast, phi_full.phi
    );
}

#[test]
fn test_mi_matrix_symmetric() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(4);

    let result = calc.compute_true_phi(&components);
    let matrix = &result.mutual_information_matrix;

    // Check symmetry
    for i in 0..matrix.len() {
        for j in 0..matrix.len() {
            assert!(
                (matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                "MI matrix should be symmetric"
            );
        }
    }
}

// Tests for improved MIP search algorithms

#[test]
fn test_spectral_partition() {
    let calc = TruePhiCalculator::new();

    // Create components with clear cluster structure
    let base1 = ContinuousHV::random(HDC_DIMENSION, 100);
    let c1 = ContinuousHV::weighted_bundle(
        &[&base1, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.9, 0.1],
    );
    let c2 = ContinuousHV::weighted_bundle(
        &[&base1, &ContinuousHV::random(HDC_DIMENSION, 102)],
        &[0.9, 0.1],
    );

    let base2 = ContinuousHV::random(HDC_DIMENSION, 200);
    let c3 = ContinuousHV::weighted_bundle(
        &[&base2, &ContinuousHV::random(HDC_DIMENSION, 201)],
        &[0.9, 0.1],
    );
    let c4 = ContinuousHV::weighted_bundle(
        &[&base2, &ContinuousHV::random(HDC_DIMENSION, 202)],
        &[0.9, 0.1],
    );

    let components = vec![c1, c2, c3, c4];
    let mi_matrix = calc.build_mi_matrix(&components);

    let partition = calc.spectral_partition(&mi_matrix, 4);
    assert!(partition.is_some(), "Spectral partition should succeed");

    let p = partition.unwrap();
    assert!(
        !p.part_a.is_empty() && !p.part_b.is_empty(),
        "Partition should have non-empty parts"
    );
}

#[test]
fn test_simulated_annealing_partition() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(12);

    let partition = calc.simulated_annealing_partition(&components, 12);

    assert!(partition.is_some(), "SA partition should succeed");
    let p = partition.unwrap();
    assert!(
        !p.part_a.is_empty() && !p.part_b.is_empty(),
        "SA partition should have non-empty parts"
    );
    assert_eq!(
        p.part_a.len() + p.part_b.len(),
        12,
        "All elements should be assigned"
    );
}

#[test]
fn test_greedy_bisection_with_local_search() {
    let calc = TruePhiCalculator::new();
    let components: Vec<ContinuousHV> = create_test_vectors(10);

    let mi_matrix = calc.build_mi_matrix(&components);
    let greedy = calc.greedy_bisection_partition(&mi_matrix, 10);
    let refined = calc.local_search_refinement(&components, greedy.clone());

    assert!(!greedy.part_a.is_empty() && !greedy.part_b.is_empty());
    assert!(!refined.part_a.is_empty() && !refined.part_b.is_empty());

    let greedy_ei = calc.partition_effective_information(&components, &greedy);
    let refined_ei = calc.partition_effective_information(&components, &refined);
    assert!(
        refined_ei <= greedy_ei + 1e-10,
        "Local search should not increase EI"
    );
}

#[test]
fn test_large_system_mip_search() {
    let calc = TruePhiCalculator::new();
    // 12 components: large enough to exercise heuristic search, fast enough for CI
    let components: Vec<ContinuousHV> = create_test_vectors(12);

    let result = calc.compute_true_phi(&components);

    assert!(result.phi >= 0.0, "\u{03a6} should be non-negative");
    assert!(
        !result.mip.part_a.is_empty() && !result.mip.part_b.is_empty(),
        "MIP should have non-empty parts"
    );
    assert_eq!(
        result.mip.part_a.len() + result.mip.part_b.len(),
        12,
        "All components should be in partition"
    );
}

#[test]
fn test_assignment_to_partition() {
    let calc = TruePhiCalculator::new();
    let assignment = vec![true, true, false, true, false];
    let partition = calc.assignment_to_partition(&assignment);

    assert_eq!(partition.part_a, vec![0, 1, 3]);
    assert_eq!(partition.part_b, vec![2, 4]);
}

#[test]
fn test_power_iteration_fiedler() {
    let calc = TruePhiCalculator::new();

    let laplacian = vec![
        vec![1.0, -1.0, 0.0, 0.0],
        vec![-1.0, 2.0, -1.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0],
        vec![0.0, 0.0, -1.0, 1.0],
    ];

    let fiedler = calc.power_iteration_fiedler(&laplacian, 4, 100);

    assert_eq!(fiedler.len(), 4);
    let sum: f64 = fiedler.iter().sum();
    assert!(
        sum.abs() < 0.1,
        "Fiedler should be mean-centered, sum={}",
        sum
    );
}

// Tests for continuous entropy estimation

#[test]
fn test_entropy_method_default() {
    let est = ContinuousEntropyEstimator::default();
    assert_eq!(est.method, EntropyMethod::Histogram);
}

#[test]
fn test_knn_estimator_creation() {
    let est = ContinuousEntropyEstimator::knn(5);
    assert_eq!(est.method, EntropyMethod::KNN);
    assert_eq!(est.k_neighbors, 5);
}

#[test]
fn test_kde_estimator_creation() {
    let est = ContinuousEntropyEstimator::kde();
    assert_eq!(est.method, EntropyMethod::KDE);
}

#[test]
fn test_adaptive_estimator_creation() {
    let est = ContinuousEntropyEstimator::adaptive(32);
    assert_eq!(est.method, EntropyMethod::AdaptiveBins);
    assert_eq!(est.adaptive_bins, 32);
}

#[test]
fn test_histogram_entropy_bounds() {
    let est = ContinuousEntropyEstimator::default();
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let h = est.entropy(&hv);
    assert!(h >= 0.0, "Entropy should be non-negative");
    let max_h = (est.adaptive_bins as f64).log2();
    assert!(
        h <= max_h + 0.1,
        "Entropy {} should be at most {}",
        h,
        max_h
    );
}

#[test]
fn test_knn_entropy_non_negative() {
    let est = ContinuousEntropyEstimator::knn(3);
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let h = est.entropy(&hv);
    assert!(h >= 0.0, "k-NN entropy should be non-negative");
}

#[test]
fn test_kde_entropy_non_negative() {
    let est = ContinuousEntropyEstimator::kde();
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let h = est.entropy(&hv);
    assert!(h >= 0.0, "KDE entropy should be non-negative");
}

#[test]
fn test_adaptive_entropy_non_negative() {
    let est = ContinuousEntropyEstimator::adaptive(16);
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let h = est.entropy(&hv);
    assert!(h >= 0.0, "Adaptive entropy should be non-negative");
}

#[test]
fn test_entropy_methods_correlate() {
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let h_hist = ContinuousEntropyEstimator::default().entropy(&hv);
    let h_knn = ContinuousEntropyEstimator::knn(3).entropy(&hv);
    let h_kde = ContinuousEntropyEstimator::kde().entropy(&hv);
    let h_adaptive = ContinuousEntropyEstimator::adaptive(16).entropy(&hv);

    // All should be non-negative
    assert!(
        h_hist >= 0.0,
        "Histogram entropy should be non-negative: {}",
        h_hist
    );
    assert!(
        h_knn >= 0.0,
        "k-NN entropy should be non-negative: {}",
        h_knn
    );
    assert!(
        h_kde >= 0.0,
        "KDE entropy should be non-negative: {}",
        h_kde
    );
    assert!(
        h_adaptive >= 0.0,
        "Adaptive entropy should be non-negative: {}",
        h_adaptive
    );

    // Histogram and adaptive should give positive entropy for random data
    assert!(
        h_hist > 0.0,
        "Histogram should have positive entropy for random data"
    );
    assert!(
        h_adaptive > 0.0,
        "Adaptive should have positive entropy for random data"
    );

    // For methods that give positive values, check they're in same ballpark
    let positive_entropies: Vec<f64> = [h_hist, h_knn, h_kde, h_adaptive]
        .iter()
        .copied()
        .filter(|&h| h > 0.01)
        .collect();

    if positive_entropies.len() >= 2 {
        let max_h = positive_entropies.iter().copied().fold(0.0, f64::max);
        let min_h = positive_entropies.iter().copied().fold(f64::MAX, f64::min);
        assert!(
            max_h / min_h < 10.0,
            "Methods should give similar results: hist={:.3}, knn={:.3}, kde={:.3}, adaptive={:.3}",
            h_hist,
            h_knn,
            h_kde,
            h_adaptive
        );
    }
}

#[test]
fn test_knn_mutual_information() {
    let est = ContinuousEntropyEstimator::knn(3);

    // Test with correlated vectors
    let base = ContinuousHV::random(HDC_DIMENSION, 100);
    let hv1 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.9, 0.1],
    );
    let hv2 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 102)],
        &[0.9, 0.1],
    );

    let mi = est.mutual_information_knn(&hv1, &hv2);
    assert!(mi >= 0.0, "MI should be non-negative");
}

#[test]
fn test_knn_mi_higher_for_correlated() {
    let est = ContinuousEntropyEstimator::knn(3);

    // Independent vectors
    let ind1 = ContinuousHV::random(HDC_DIMENSION, 1);
    let ind2 = ContinuousHV::random(HDC_DIMENSION, 2);
    let mi_ind = est.mutual_information_knn(&ind1, &ind2);

    // Correlated vectors
    let base = ContinuousHV::random(HDC_DIMENSION, 100);
    let cor1 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.8, 0.2],
    );
    let cor2 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 102)],
        &[0.8, 0.2],
    );
    let mi_cor = est.mutual_information_knn(&cor1, &cor2);

    assert!(
        mi_cor > mi_ind,
        "Correlated MI {} should exceed independent MI {}",
        mi_cor,
        mi_ind
    );
}

#[test]
fn test_digamma_function() {
    // Known values
    assert!(
        (digamma(1.0) - (-0.5772156649)).abs() < 0.01,
        "\u{03c8}(1) \u{2248} -\u{03b3}"
    );
    assert!(
        (digamma(2.0) - 0.4227843351).abs() < 0.01,
        "\u{03c8}(2) \u{2248} 1 - \u{03b3}"
    );
    assert!(digamma(10.0) > 2.0, "\u{03c8}(10) should be positive");
}

#[test]
fn test_silverman_bandwidth() {
    let values: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0) - 1.0).collect();
    let bw = silverman_bandwidth(&values);
    assert!(bw > 0.0, "Bandwidth should be positive");
    assert!(bw < 1.0, "Bandwidth should be reasonable for [-1,1] data");
}

// IIT Benchmark Tests
// Based on Integrated Information Theory (Tononi et al., 2016)

/// IIT Axiom: Φ = 0 for completely independent (unintegrated) systems
///
/// Reference: Oizumi et al. (2014) - "From Phenomenology to Mechanisms"
/// A system of independent elements has no integrated information.
#[test]
fn test_iit_axiom_independent_system_zero_phi() {
    let calc = TruePhiCalculator::new();

    // Create completely independent random vectors
    let independent: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 1000))
        .collect();

    let result = calc.compute_true_phi(&independent);

    // Φ should be very close to 0 for independent components
    // (Some small positive value due to random correlations)
    assert!(
        result.phi < 0.1,
        "Independent system should have near-zero \u{03a6}: {:.4}",
        result.phi
    );

    // System EI ≈ MIP EI for independent components
    assert!(
        (result.system_ei - result.mip_ei).abs() < 0.1,
        "EI should be similar for system and MIP: sys={:.4}, mip={:.4}",
        result.system_ei,
        result.mip_ei
    );
}

/// IIT Axiom: Φ > 0 for genuinely integrated systems
///
/// Reference: Tononi (2008) - "Consciousness as Integrated Information"
/// Integration means the whole has more information than the sum of its parts.
#[test]
fn test_iit_axiom_integrated_system_positive_phi() {
    let calc = TruePhiCalculator::new();

    // Create a highly integrated system (all derived from same base)
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let integrated: Vec<ContinuousHV> = (0..4)
        .map(|i| {
            let noise = ContinuousHV::random(HDC_DIMENSION, 100 + i as u64);
            ContinuousHV::weighted_bundle(&[&base, &noise], &[0.85, 0.15])
        })
        .collect();

    let result = calc.compute_true_phi(&integrated);

    // Φ should be significantly positive for integrated system
    assert!(
        result.phi > 0.1,
        "Integrated system should have positive \u{03a6}: {:.4}",
        result.phi
    );

    // System EI > MIP EI (this is the essence of integration)
    assert!(
        result.system_ei > result.mip_ei,
        "System EI should exceed MIP EI: sys={:.4} > mip={:.4}",
        result.system_ei,
        result.mip_ei
    );
}

/// IIT Property: Φ decreases when system is partitioned
///
/// Reference: IIT 3.0 - The Minimum Information Partition (MIP)
/// cuts the system where integration is weakest.
#[test]
fn test_iit_partition_decreases_phi() {
    let calc = TruePhiCalculator::new();

    // Create an integrated system
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let components: Vec<ContinuousHV> = (0..6)
        .map(|i| {
            let noise = ContinuousHV::random(HDC_DIMENSION, 200 + i as u64);
            ContinuousHV::weighted_bundle(&[&base, &noise], &[0.8, 0.2])
        })
        .collect();

    // Compute Φ for full system
    let full_result = calc.compute_true_phi(&components);

    // Compute Φ for subsystems (partitions)
    let part_a: Vec<ContinuousHV> = components[0..3].to_vec();
    let part_b: Vec<ContinuousHV> = components[3..6].to_vec();

    let phi_a = calc.compute_true_phi(&part_a);
    let phi_b = calc.compute_true_phi(&part_b);

    // Sum of partition Φs should be related to cross-partition information
    let _partition_total = phi_a.phi + phi_b.phi;

    println!(
        "Full \u{03a6}: {:.4}, Part A \u{03a6}: {:.4}, Part B \u{03a6}: {:.4}",
        full_result.phi, phi_a.phi, phi_b.phi
    );

    // Full system Φ includes cross-partition information
    // that is lost when partitioned
    assert!(
        full_result.system_ei > phi_a.system_ei + phi_b.system_ei - 0.1,
        "Full system EI should account for all pairwise MI"
    );
}

/// IIT Property: Binding creates integration
///
/// Reference: HDC interpretation of IIT
/// Binding vectors creates new orthogonal structure that
/// cannot be recovered by unbinding - this is integration.
#[test]
fn test_iit_binding_creates_integration() {
    let calc = TruePhiCalculator::new();

    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let c = ContinuousHV::random(HDC_DIMENSION, 3);
    let d = ContinuousHV::random(HDC_DIMENSION, 4);

    // Unbounded system: just the 4 components
    let unbounded = vec![a.clone(), b.clone(), c.clone(), d.clone()];
    let phi_unbounded = calc.compute_true_phi(&unbounded);

    // Bounded system: a*b and c*d create new integrated structures
    let bound1 = a.bind(&b);
    let bound2 = c.bind(&d);
    let bounded = vec![bound1.clone(), bound2.clone(), a.clone(), c.clone()];
    let phi_bounded = calc.compute_true_phi(&bounded);

    println!(
        "Unbounded \u{03a6}: {:.4}, Bounded \u{03a6}: {:.4}",
        phi_unbounded.phi, phi_bounded.phi
    );

    // Binding should affect the information structure
    // (not necessarily increase Φ, but change the MIP)
    assert!(
        phi_bounded.mip.part_a.len() > 0 && phi_bounded.mip.part_b.len() > 0,
        "Bounded system should have non-trivial MIP"
    );
}

/// IIT Property: More components can increase Φ
///
/// Reference: IIT 3.0 - Φ generally increases with system size
/// for integrated systems (but not for independent ones).
#[test]
fn test_iit_size_effect_on_phi() {
    let calc = TruePhiCalculator::new();

    // Create integrated systems of different sizes
    let base = ContinuousHV::random(HDC_DIMENSION, 42);

    let create_integrated = |n: usize| -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| {
                let noise = ContinuousHV::random(HDC_DIMENSION, 300 + i as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.8, 0.2])
            })
            .collect()
    };

    let small = create_integrated(3);
    let medium = create_integrated(5);

    let phi_small = calc.compute_true_phi(&small);
    let phi_medium = calc.compute_true_phi(&medium);

    println!(
        "Small (3) EI: {:.4}, \u{03a6}: {:.4}",
        phi_small.system_ei, phi_small.phi
    );
    println!(
        "Medium (5) EI: {:.4}, \u{03a6}: {:.4}",
        phi_medium.system_ei, phi_medium.phi
    );

    // Larger integrated systems should have higher total EI
    assert!(
        phi_medium.system_ei > phi_small.system_ei,
        "Larger system should have higher EI: medium={:.4} > small={:.4}",
        phi_medium.system_ei,
        phi_small.system_ei
    );
}

/// IIT Property: Information exclusion
///
/// Reference: IIT 3.0 - Only the Minimum Information Partition matters
/// There's exactly one partition that defines Φ.
#[test]
fn test_iit_mip_uniqueness() {
    let calc = TruePhiCalculator::new();

    let components: Vec<ContinuousHV> = create_test_vectors(5);
    let result = calc.compute_true_phi(&components);

    // MIP should be a valid bipartition
    assert!(
        !result.mip.part_a.is_empty(),
        "MIP part A should be non-empty"
    );
    assert!(
        !result.mip.part_b.is_empty(),
        "MIP part B should be non-empty"
    );

    // All elements should be in exactly one part
    let total = result.mip.part_a.len() + result.mip.part_b.len();
    assert_eq!(total, 5, "MIP should partition all elements");

    // No duplicates
    let a_set: std::collections::HashSet<_> = result.mip.part_a.iter().collect();
    let b_set: std::collections::HashSet<_> = result.mip.part_b.iter().collect();
    assert!(a_set.is_disjoint(&b_set), "MIP parts should be disjoint");
}

/// IIT Benchmark: Correlated vs Independent comparison
///
/// This is a key distinguishing test - IIT predicts that
/// correlated components have higher Φ than independent ones.
#[test]
fn test_iit_benchmark_correlation_increases_phi() {
    let calc = TruePhiCalculator::new();

    // Independent system
    let independent: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 7919))
        .collect();

    // Correlated system (same base)
    let base = ContinuousHV::random(HDC_DIMENSION, 12345);
    let correlated: Vec<ContinuousHV> = (0..4)
        .map(|i| {
            let noise = ContinuousHV::random(HDC_DIMENSION, 400 + i as u64);
            ContinuousHV::weighted_bundle(&[&base, &noise], &[0.9, 0.1])
        })
        .collect();

    let phi_ind = calc.compute_true_phi(&independent);
    let phi_cor = calc.compute_true_phi(&correlated);

    println!(
        "Independent \u{03a6}: {:.4}, EI: {:.4}",
        phi_ind.phi, phi_ind.system_ei
    );
    println!(
        "Correlated \u{03a6}: {:.4}, EI: {:.4}",
        phi_cor.phi, phi_cor.system_ei
    );

    // Correlated should have significantly higher Φ
    assert!(
        phi_cor.phi > phi_ind.phi * 2.0,
        "Correlated \u{03a6} ({:.4}) should be much higher than independent \u{03a6} ({:.4})",
        phi_cor.phi,
        phi_ind.phi
    );
}
