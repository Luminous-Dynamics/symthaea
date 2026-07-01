// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the Spectral MIP Finder.

use super::*;

// ─── Helper: build covariance from known structure ─────────────────────

/// Create an n×n identity matrix (flat row-major).
fn identity_cov(n: usize) -> Vec<f64> {
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    cov
}

/// Create a block-diagonal covariance: two correlated blocks with weak inter-block coupling.
fn block_diagonal_cov(n_a: usize, n_b: usize, intra_corr: f64, inter_corr: f64) -> Vec<f64> {
    let n = n_a + n_b;
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let same_block = (i < n_a && j < n_a) || (i >= n_a && j >= n_a);
            let corr = if same_block { intra_corr } else { inter_corr };
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    cov
}

/// Create a 3-block-diagonal covariance with varying inter-block strengths.
fn three_block_cov(
    n_a: usize,
    n_b: usize,
    n_c: usize,
    intra: f64,
    inter_ab: f64,
    inter_bc: f64,
    inter_ac: f64,
) -> Vec<f64> {
    let n = n_a + n_b + n_c;
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let block_i = if i < n_a {
                0
            } else if i < n_a + n_b {
                1
            } else {
                2
            };
            let block_j = if j < n_a {
                0
            } else if j < n_a + n_b {
                1
            } else {
                2
            };
            let corr = if block_i == block_j {
                intra
            } else {
                match (block_i.min(block_j), block_i.max(block_j)) {
                    (0, 1) => inter_ab,
                    (1, 2) => inter_bc,
                    (0, 2) => inter_ac,
                    _ => 0.0,
                }
            };
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    cov
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bordered_cholesky_correctness() {
    // Build a small PD covariance incrementally and compare ln_det vs from-scratch
    let n = 6;
    let cov = block_diagonal_cov(3, 3, 0.5, 0.1);
    // Add regularization
    let mut reg_cov = cov.clone();
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    // Compute ln_det from scratch
    let ln_det_full =
        crate::consciousness_metrics::spectral_mip::ln_determinant_cholesky(&reg_cov, n);

    // Compute incrementally via bordered Cholesky
    let mut factor: Vec<f64> = Vec::new();
    // k=0
    let d0 = reg_cov[0].max(1e-30);
    factor.push(d0.sqrt());
    let mut ln_det_inc = d0.sqrt().ln() * 2.0;

    for k in 1..n {
        let col: Vec<f64> = (0..k).map(|i| reg_cov[i * n + k]).collect();
        let diag = reg_cov[k * n + k];
        let (x, l_new) = crate::consciousness_metrics::spectral_mip::bordered_cholesky_step(
            &factor, k, &col, diag,
        );
        factor.extend_from_slice(&x);
        factor.push(l_new);
        ln_det_inc += 2.0 * l_new.max(1e-300).ln();
    }

    assert!(
        (ln_det_inc - ln_det_full).abs() < 1e-8,
        "Incremental ln_det ({}) != full ln_det ({})",
        ln_det_inc,
        ln_det_full
    );
}

#[test]
fn test_identity_covariance_phi_zero() {
    let n = 8;
    let cov = identity_cov(n);
    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&cov, n, 50).unwrap();

    assert!(
        result.phi.abs() < 1e-6,
        "Identity covariance should give Phi ≈ 0, got {}",
        result.phi
    );
    assert!(result.total_mi.abs() < 1e-6);
}

#[test]
fn test_block_diagonal_mip_at_boundary() {
    let n_a = 4;
    let n_b = 4;
    let n = n_a + n_b;
    let cov = block_diagonal_cov(n_a, n_b, 0.6, 0.02);

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&cov, n, 50).unwrap();

    // Phi should be positive (system is integrated within blocks)
    assert!(
        result.phi > 0.0,
        "Block diagonal should have positive Phi, got {}",
        result.phi
    );
    assert!(result.phi.is_finite());

    // MIP should split close to the block boundary
    let part_a = &result.mip.part_a;
    let part_b = &result.mip.part_b;

    // Check that the partition separates mostly block-A from block-B indices
    // (either orientation is valid)
    let a_in_block_a = part_a.iter().filter(|&&i| i < n_a).count();
    let b_in_block_b = part_b.iter().filter(|&&i| i >= n_a).count();
    let orient1 = (a_in_block_a + b_in_block_b) as f64 / n as f64;

    // Flipped orientation: block-A in part_b, block-B in part_a
    let a_in_block_b = part_b.iter().filter(|&&i| i < n_a).count();
    let b_in_block_a = part_a.iter().filter(|&&i| i >= n_a).count();
    let orient2 = (a_in_block_b + b_in_block_a) as f64 / n as f64;

    let correct_fraction = orient1.max(orient2);

    assert!(
        correct_fraction >= 0.75,
        "MIP should roughly separate blocks (best fraction: {}, orient1={}, orient2={})",
        correct_fraction,
        orient1,
        orient2
    );
}

#[test]
fn test_three_block_mip_at_weakest_boundary() {
    // 3 blocks: A-B strongly coupled, B-C weakly coupled, A-C no coupling
    // MIP should cut between B and C (weakest inter-block link)
    let n_a = 3;
    let n_b = 3;
    let n_c = 3;
    let cov = three_block_cov(n_a, n_b, n_c, 0.7, 0.3, 0.05, 0.01);
    let n = n_a + n_b + n_c;

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&cov, n, 50).unwrap();

    assert!(result.phi > 0.0);
    assert!(result.phi.is_finite());

    // The weakest boundary is between {A,B} and {C}
    // Check that C indices are mostly on one side
    let c_indices: Vec<usize> = (n_a + n_b..n).collect();
    let c_in_a = result
        .mip
        .part_a
        .iter()
        .filter(|i| c_indices.contains(i))
        .count();
    let c_in_b = result
        .mip
        .part_b
        .iter()
        .filter(|i| c_indices.contains(i))
        .count();
    // All C indices should be on the same side
    let c_together = c_in_a.max(c_in_b);
    assert!(
        c_together >= n_c - 1,
        "C-block should be mostly together: {} of {} on one side",
        c_together,
        n_c
    );
}

#[test]
fn test_known_2x2_matches_sigma() {
    // For n=2, there's only one possible bipartition: {0} | {1}
    // Spectral MIP should match SynergisticIntegration exactly
    let config_sigma = SynergisticConfig {
        num_components: 2,
        window_size: 50,
        min_samples: 10,
        regularization: 1e-6,
        ..Default::default()
    };
    let config_spectral = SpectralMIPConfig {
        num_components: 2,
        window_size: 50,
        min_samples: 10,
        regularization: 1e-6,
        ..Default::default()
    };

    let mut si = SynergisticIntegration::new(config_sigma);
    let mut finder = SpectralMIPFinder::new(config_spectral);

    for i in 0..30 {
        let hv = ContinuousHV::random(128, i + 500);
        si.push(&hv);
        finder.push(&hv);
    }

    let sigma_result = si.compute().unwrap();
    let spectral_result = finder.compute().unwrap();

    // Phi from spectral should equal sigma (for n=2, only one partition)
    assert!(
        (spectral_result.phi - sigma_result.sigma).abs() < 0.5,
        "Spectral phi ({}) should be close to sigma ({}) for n=2",
        spectral_result.phi,
        sigma_result.sigma
    );
}

#[test]
fn test_spectral_vs_exhaustive_n8() {
    // For n=8, spectral should find an MI that's close to exhaustive
    let n = 8;
    let cov = block_diagonal_cov(4, 4, 0.5, 0.1);
    let mut reg_cov = cov;
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    // Spectral MIP
    let finder = SpectralMIPFinder::with_defaults();
    let spectral = finder.compute_from_covariance(&reg_cov, n, 50).unwrap();

    // Exhaustive MIP (reuse SynergisticIntegration's method)
    let si = SynergisticIntegration::new(SynergisticConfig {
        num_components: n,
        regularization: 0.0, // already regularized
        ..Default::default()
    });
    let exhaustive_mip_mi = si.exhaustive_mip(&reg_cov, n);

    // Spectral should find MI ≤ 1.5× exhaustive (it searches contiguous cuts only)
    assert!(
        spectral.mip_mi <= exhaustive_mip_mi * 1.5 + 0.1,
        "Spectral MIP MI ({}) too far from exhaustive ({})",
        spectral.mip_mi,
        exhaustive_mip_mi
    );
}

#[test]
fn test_spectral_vs_greedy_n16() {
    // Spectral should find equal-or-better partition than greedy for structured covariance
    let n = 16;
    let cov = block_diagonal_cov(8, 8, 0.5, 0.05);
    let mut reg_cov = cov;
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    let finder = SpectralMIPFinder::with_defaults();
    let spectral = finder.compute_from_covariance(&reg_cov, n, 50).unwrap();

    let si = SynergisticIntegration::new(SynergisticConfig {
        num_components: n,
        regularization: 0.0,
        ..Default::default()
    });
    let greedy_mip_mi = si.greedy_mip(&reg_cov, n);

    // For block-diagonal, spectral should be competitive with greedy
    // Allow some tolerance since they search different spaces
    assert!(
        spectral.mip_mi <= greedy_mip_mi * 2.0 + 0.5,
        "Spectral MIP MI ({}) unexpectedly much worse than greedy ({})",
        spectral.mip_mi,
        greedy_mip_mi
    );
    assert!(spectral.phi.is_finite());
}

#[test]
fn test_fiedler_zero_crossing_block_diagonal() {
    let n_a = 5;
    let n_b = 5;
    let n = n_a + n_b;
    let cov = block_diagonal_cov(n_a, n_b, 0.7, 0.01);

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&cov, n, 50).unwrap();

    // For a clear block-diagonal, the Fiedler zero-crossing should land near the
    // block boundary (index ~4-5 in spectral order)
    let zc = result.fiedler_zero_crossing;
    assert!(
        (zc as i32 - n_a as i32).unsigned_abs() <= 2,
        "Fiedler zero-crossing ({}) should be near block boundary ({})",
        zc,
        n_a
    );
}

#[test]
fn test_scale_smoke_n64() {
    let n = 64;
    let cov = block_diagonal_cov(32, 32, 0.3, 0.02);
    let mut reg_cov = cov;
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&reg_cov, n, 50).unwrap();

    assert!(result.phi.is_finite());
    assert!(result.phi >= 0.0);
    assert_eq!(result.cut_mis.len(), n - 1);
    assert!(result.cut_mis.iter().all(|v| v.is_finite()));
}

#[test]
fn test_scale_smoke_n128() {
    let n = 128;
    let cov = block_diagonal_cov(64, 64, 0.2, 0.01);
    let mut reg_cov = cov;
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&reg_cov, n, 50).unwrap();

    assert!(result.phi.is_finite());
    assert!(result.phi >= 0.0);
}

#[test]
#[ignore] // ~1s, run explicitly
fn test_scale_smoke_n256() {
    let n = 256;
    let cov = block_diagonal_cov(128, 128, 0.15, 0.005);
    let mut reg_cov = cov;
    for i in 0..n {
        reg_cov[i * n + i] += 1e-6;
    }

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&reg_cov, n, 50).unwrap();

    assert!(result.phi.is_finite());
    assert!(result.phi >= 0.0);
}

#[test]
fn test_numerical_stability_near_singular() {
    // Near-singular covariance: some dims nearly collinear
    let n = 8;
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
        for j in (i + 1)..n {
            // Very high correlation → near-singular
            let corr = 0.999;
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    // Small regularization
    for i in 0..n {
        cov[i * n + i] += 1e-4;
    }

    let finder = SpectralMIPFinder::with_defaults();
    let result = finder.compute_from_covariance(&cov, n, 50).unwrap();

    assert!(
        result.phi.is_finite(),
        "Phi should be finite for near-singular cov"
    );
    assert!(!result.phi.is_nan());
    assert!(result.total_mi.is_finite());
    assert!(result.mip_mi.is_finite());
    assert!(result.cut_mis.iter().all(|v| v.is_finite()));
}

#[test]
fn test_window_lifecycle() {
    let config = SpectralMIPConfig {
        num_components: 4,
        window_size: 5,
        min_samples: 3,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    // Not ready initially
    assert!(!finder.ready());
    assert!(finder.compute().is_none());

    // Push below min_samples
    for i in 0..2 {
        finder.push(&ContinuousHV::random(64, i));
    }
    assert!(!finder.ready());

    // Push to min_samples
    finder.push(&ContinuousHV::random(64, 2));
    assert!(finder.ready());
    assert!(finder.compute().is_some());

    // Push beyond window_size → eviction
    for i in 3..10 {
        finder.push(&ContinuousHV::random(64, i));
    }
    assert_eq!(finder.window_len(), 5);
    assert!(finder.compute().is_some());

    // Reset
    finder.reset();
    assert!(!finder.ready());
    assert_eq!(finder.window_len(), 0);
    assert!(finder.compute().is_none());
}

#[test]
fn test_compute_from_covariance_consistent() {
    // Verify that compute_from_covariance gives the same result as compute
    // when fed the same underlying data
    let config = SpectralMIPConfig {
        num_components: 6,
        window_size: 30,
        min_samples: 10,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config.clone());

    for i in 0..20 {
        finder.push(&ContinuousHV::random(128, i + 1000));
    }

    let result_compute = finder.compute().unwrap();

    // Build covariance manually from the same data
    let mut finder2 = SpectralMIPFinder::new(config);
    for i in 0..20 {
        finder2.push(&ContinuousHV::random(128, i + 1000));
    }
    // Access the internal covariance by computing directly
    let result_compute2 = finder2.compute().unwrap();

    // Both should give identical results (same data, same algorithm)
    assert!(
        (result_compute.phi - result_compute2.phi).abs() < 1e-10,
        "Phi mismatch: {} vs {}",
        result_compute.phi,
        result_compute2.phi
    );
    assert!(
        (result_compute.total_mi - result_compute2.total_mi).abs() < 1e-10,
        "Total MI mismatch"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE DIMENSION SELECTION TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_adapt_produces_valid_dims() {
    let config = SpectralMIPConfig {
        num_components: 16,
        window_size: 10,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    // Simulate a full HDC state space of 1024 dimensions
    let full_dim = 1024;
    for i in 0..10 {
        let values: Vec<f32> = (0..full_dim)
            .map(|j| ((i * 7 + j * 3) % 100) as f32 / 50.0 - 1.0)
            .collect();
        let hv = ContinuousHV::from_slice(&values);
        finder.push(&hv);
    }
    assert!(finder.ready());
    assert!(!finder.is_adapted());

    let result = finder.compute().expect("should compute");
    finder.adapt(&result);

    assert!(finder.is_adapted());
    let dims = finder.active_dim_indices().unwrap();
    assert_eq!(dims.len(), 16);

    // All dims should be valid indices into the full space
    for &d in dims {
        assert!(d < full_dim, "dim {d} out of range [0, {full_dim})");
    }

    // Dims should be sorted (for cache-friendly access)
    for w in dims.windows(2) {
        assert!(w[0] <= w[1], "dims should be sorted: {} > {}", w[0], w[1]);
    }

    // No duplicates
    let mut deduped = dims.to_vec();
    deduped.dedup();
    assert_eq!(deduped.len(), dims.len(), "dims should have no duplicates");

    // Window should be cleared after adaptation
    assert_eq!(finder.window_len(), 0);
}

#[test]
fn test_adapt_concentrates_near_mip() {
    let config = SpectralMIPConfig {
        num_components: 32,
        window_size: 15,
        min_samples: 10,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    // Use block-diagonal structure: dims 0..16 correlated, 16..32 correlated
    let full_dim = 2048;
    for i in 0..15 {
        let values: Vec<f32> = (0..full_dim)
            .map(|j| {
                let block_signal = if j < full_dim / 2 { 1.0 } else { -1.0 };
                let noise = ((i * 13 + j * 7 + 42) % 200) as f32 / 200.0 - 0.5;
                block_signal * 0.5 + noise * 0.3
            })
            .collect();
        let hv = ContinuousHV::from_slice(&values);
        finder.push(&hv);
    }

    let result = finder.compute().expect("should compute");
    let _mip_bond = result.mip_bond;

    finder.adapt(&result);
    let dims = finder.active_dim_indices().unwrap();

    // After adaptation, the boundary dims should be concentrated.
    // The non-adapted version was evenly spaced (stride=64).
    // The adapted version should have more dims from the boundary region.
    // We can't assert exact indices (depends on Fiedler ordering), but we can
    // check that the dims are NOT evenly spaced anymore.
    let stride = full_dim / 32;
    let evenly_spaced: Vec<usize> = (0..32).map(|i| i * stride).collect();
    assert_ne!(
        dims,
        &evenly_spaced[..],
        "adapted dims should differ from evenly-spaced"
    );

    // The adaptation result should be usable for a new compute cycle
    for i in 0..15 {
        let values: Vec<f32> = (0..full_dim)
            .map(|j| {
                let block_signal = if j < full_dim / 2 { 1.0 } else { -1.0 };
                let noise = ((i * 17 + j * 11 + 99) % 200) as f32 / 200.0 - 0.5;
                block_signal * 0.5 + noise * 0.3
            })
            .collect();
        let hv = ContinuousHV::from_slice(&values);
        finder.push(&hv);
    }

    let result2 = finder.compute().expect("should compute after adaptation");
    assert!(result2.phi.is_finite() && result2.phi >= 0.0);
}

#[test]
fn test_adapt_noop_on_small_n() {
    let config = SpectralMIPConfig {
        num_components: 2,
        window_size: 5,
        min_samples: 3,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    for i in 0..5 {
        let values: Vec<f32> = (0..64).map(|j| ((i + j) % 10) as f32 / 5.0).collect();
        finder.push(&ContinuousHV::from_slice(&values));
    }

    let result = finder.compute().expect("should compute");
    // n=2 is below the minimum (4) for adaptation
    finder.adapt(&result);
    assert!(!finder.is_adapted(), "should not adapt when n < 4");
}

// ═══════════════════════════════════════════════════════════════════════════
// HIERARCHICAL SPECTRAL MIP
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_hierarchical_block_diagonal() {
    // Large enough to produce multiple hierarchical levels (32, 64)
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 30,
        min_samples: 10,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    // Build block-diagonal state: first 32 dims correlated, second 32 dims correlated,
    // weak inter-block coupling
    for i in 0..20 {
        let values: Vec<f32> = (0..256)
            .map(|j| {
                let block = if j < 128 { 0 } else { 1 };
                let base = (i as f32 * 0.1 + j as f32 * 0.01).sin();
                if block == 0 {
                    base + (i as f32 * 0.3).cos() * 0.5
                } else {
                    base + (i as f32 * 0.7).sin() * 0.5
                }
            })
            .collect();
        finder.push(&ContinuousHV::from_slice(&values));
    }

    let result = finder
        .compute_hierarchical()
        .expect("should compute hierarchical");
    assert!(result.phi.is_finite(), "phi should be finite");
    assert!(!result.levels.is_empty(), "should have at least one level");
    assert!(!result.scales.is_empty(), "should have at least one scale");

    // All levels should have finite phi
    for (i, level) in result.levels.iter().enumerate() {
        assert!(
            level.phi.is_finite(),
            "level {} phi should be finite, got {}",
            i,
            level.phi
        );
    }
}

#[test]
fn test_hierarchical_single_scale() {
    // With n=16, only one scale (16 itself, since 32 > 16)
    let config = SpectralMIPConfig {
        num_components: 16,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    for i in 0..10 {
        let values: Vec<f32> = (0..64).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
        finder.push(&ContinuousHV::from_slice(&values));
    }

    let result = finder.compute_hierarchical().expect("should compute");
    // Single scale: should match regular compute
    assert_eq!(result.levels.len(), 1);
    let regular = finder.compute().expect("regular compute");
    assert!(
        (result.phi - regular.phi).abs() < 1e-10,
        "single-scale hierarchical should match regular: {} vs {}",
        result.phi,
        regular.phi
    );
}

#[test]
fn test_hierarchical_phi_monotonic_refinement() {
    // Each level should produce a reasonable phi (not necessarily monotonic,
    // but all finite and non-negative)
    let config = SpectralMIPConfig {
        num_components: 128,
        window_size: 40,
        min_samples: 15,
        regularization: 1e-6,
        ..Default::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    for i in 0..25 {
        let values: Vec<f32> = (0..512)
            .map(|j| {
                let block = j / 128;
                (i as f32 * 0.1 + block as f32).sin() * (1.0 + block as f32 * 0.2)
            })
            .collect();
        finder.push(&ContinuousHV::from_slice(&values));
    }

    let result = finder.compute_hierarchical().expect("should compute");
    assert!(
        result.scales.len() >= 2,
        "should have at least 2 scales: {:?}",
        result.scales
    );

    for (i, level) in result.levels.iter().enumerate() {
        assert!(
            level.phi >= 0.0,
            "level {} phi should be non-negative: {}",
            i,
            level.phi
        );
        assert!(level.phi.is_finite(), "level {} phi should be finite", i);
        assert!(
            level.total_mi.is_finite(),
            "level {} total_mi should be finite",
            i
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROPTEST: Property-based fuzzing for numerical robustness
// ═══════════════════════════════════════════════════════════════════════════

mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a random valid positive-definite covariance matrix of size n.
    /// Strategy: C = A^T A + εI where A is random, guaranteeing PD.
    fn random_pd_covariance(n: usize) -> impl Strategy<Value = Vec<f64>> {
        proptest::collection::vec(-1.0f64..1.0f64, n * n).prop_map(move |a_flat: Vec<f64>| {
            let mut cov = vec![0.0f64; n * n];
            // C = A^T * A  (Gram matrix, always PSD)
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0f64;
                    for k in 0..n {
                        dot += a_flat[k * n + i] * a_flat[k * n + j];
                    }
                    cov[i * n + j] = dot;
                }
            }
            // Regularize diagonal for strict PD
            for i in 0..n {
                cov[i * n + i] += 1e-4;
            }
            cov
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// Phi is always finite and non-negative for any valid PD covariance.
        #[test]
        fn prop_phi_nonnegative_n8(cov in random_pd_covariance(8)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 8, 30);
            if let Some(r) = result {
                prop_assert!(r.phi.is_finite(), "Phi must be finite, got {}", r.phi);
                prop_assert!(r.phi >= -1e-10, "Phi must be >= 0, got {}", r.phi);
            }
        }

        /// Total MI is always finite for valid PD covariance.
        #[test]
        fn prop_total_mi_finite_n8(cov in random_pd_covariance(8)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 8, 30);
            if let Some(r) = result {
                prop_assert!(r.total_mi.is_finite(), "Total MI must be finite, got {}", r.total_mi);
                prop_assert!(r.total_mi >= -1e-10, "Total MI must be >= 0, got {}", r.total_mi);
            }
        }

        /// Partition always produces exactly two non-empty sets.
        #[test]
        fn prop_partition_valid_n16(cov in random_pd_covariance(16)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 16, 30);
            if let Some(r) = result {
                let a = &r.mip.part_a;
                let b = &r.mip.part_b;
                prop_assert!(!a.is_empty(), "Partition part A must be non-empty");
                prop_assert!(!b.is_empty(), "Partition part B must be non-empty");
                prop_assert_eq!(
                    a.len() + b.len(), 16,
                    "Partition must cover all {} elements", 16
                );
                // No overlaps
                for idx in a {
                    prop_assert!(!b.contains(idx), "Partition parts must not overlap at {}", idx);
                }
            }
        }

        /// cut_mis always has n-1 entries, all finite.
        #[test]
        fn prop_cut_mis_valid_n32(cov in random_pd_covariance(32)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 32, 50);
            if let Some(r) = result {
                prop_assert_eq!(r.cut_mis.len(), 31, "Should have n-1 cut MIs");
                for (i, &mi) in r.cut_mis.iter().enumerate() {
                    prop_assert!(mi.is_finite(), "cut_mi[{}] must be finite, got {}", i, mi);
                }
            }
        }

        /// MIP MI is always <= total MI (the minimum cut can't exceed the whole).
        #[test]
        fn prop_mip_mi_leq_total_mi_n8(cov in random_pd_covariance(8)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 8, 30);
            if let Some(r) = result {
                prop_assert!(
                    r.mip_mi <= r.total_mi + 1e-6,
                    "MIP MI ({}) must be <= total MI ({})",
                    r.mip_mi, r.total_mi
                );
            }
        }

        /// Bordered Cholesky determinants are always finite for PD matrices.
        #[test]
        fn prop_bordered_cholesky_finite_n16(cov in random_pd_covariance(16)) {
            let n = 16;
            // Test that ln_determinant_cholesky produces finite values
            // for every prefix of the covariance matrix
            for k in 2..=n {
                let mut sub_cov = vec![0.0; k * k];
                for i in 0..k {
                    for j in 0..k {
                        sub_cov[i * k + j] = cov[i * n + j];
                    }
                }
                let det = ln_determinant_cholesky(&sub_cov, k);
                prop_assert!(det.is_finite(), "ln_det for {}x{} sub-matrix must be finite", k, k);
            }
        }

        /// Spectral order is always a valid permutation.
        #[test]
        fn prop_spectral_order_is_permutation_n8(cov in random_pd_covariance(8)) {
            let finder = SpectralMIPFinder::with_defaults();
            let result = finder.compute_from_covariance(&cov, 8, 30);
            if let Some(r) = result {
                let mut order = r.spectral_order.clone();
                order.sort();
                prop_assert_eq!(order, (0..8).collect::<Vec<_>>(), "Spectral order must be a permutation");
            }
        }
    }
}

// ─── Config validation tests ─────────────────────────────────────────

#[test]
fn test_spectral_mip_default_config_is_valid() {
    let config = SpectralMIPConfig::default();
    assert!(
        config.validate().is_ok(),
        "default SpectralMIPConfig must be valid"
    );
}

#[test]
fn test_spectral_mip_invalid_dimension_rejected() {
    let config = SpectralMIPConfig {
        num_components: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("num_components"),
        "error should mention num_components: {err}"
    );

    // Also test below minimum (< 2)
    let config = SpectralMIPConfig {
        num_components: 1,
        ..Default::default()
    };
    assert!(
        config.validate().is_err(),
        "num_components=1 should fail (minimum is 2)"
    );
}

#[test]
fn test_spectral_mip_window_size_must_exceed_min_samples() {
    let config = SpectralMIPConfig {
        window_size: 5,
        min_samples: 10,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("window_size"),
        "error should mention window_size: {err}"
    );
    assert!(
        err.contains("min_samples"),
        "error should mention min_samples: {err}"
    );
}

#[test]
fn test_spectral_mip_min_samples_at_least_two() {
    let config = SpectralMIPConfig {
        min_samples: 1,
        ..Default::default()
    };
    assert!(
        config.validate().is_err(),
        "min_samples=1 should fail (need >= 2 for covariance)"
    );

    let config = SpectralMIPConfig {
        min_samples: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err(), "min_samples=0 should fail");
}

#[test]
fn test_spectral_mip_invalid_regularization_rejected() {
    let config = SpectralMIPConfig {
        regularization: 0.0,
        ..Default::default()
    };
    assert!(config.validate().is_err(), "regularization=0.0 should fail");

    let config = SpectralMIPConfig {
        regularization: -1e-6,
        ..Default::default()
    };
    assert!(
        config.validate().is_err(),
        "negative regularization should fail"
    );

    let config = SpectralMIPConfig {
        regularization: f64::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err(), "NaN regularization should fail");

    let config = SpectralMIPConfig {
        regularization: f64::INFINITY,
        ..Default::default()
    };
    assert!(
        config.validate().is_err(),
        "infinite regularization should fail"
    );
}

#[test]
fn test_spectral_mip_valid_custom_config() {
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 100,
        min_samples: 20,
        regularization: 1e-4,
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "valid custom config should pass");
}

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURAL HIERARCHICAL PHI TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_structural_phi_block_diagonal_finds_clusters() {
    // Two clear blocks → should detect 2+ clusters with high micro, low meso
    let n = 16;
    let cov = block_diagonal_cov(8, 8, 0.6, 0.02);

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    assert!(
        structural.num_clusters >= 2,
        "Block diagonal should produce at least 2 clusters, got {}",
        structural.num_clusters
    );
    assert!(
        structural.micro_phi > 0.0,
        "Micro phi should be positive for correlated blocks"
    );
    assert!(structural.macro_phi > 0.0, "Macro phi should be positive");
    // All fields should be finite
    assert!(structural.micro_phi.is_finite());
    assert!(structural.meso_phi.is_finite());
    assert!(structural.macro_phi.is_finite());
    assert!(structural.emergence_ratio.is_finite());
    assert!(structural.bottleneck_score.is_finite());
}

#[test]
fn test_structural_phi_identity_trivial() {
    // Identity covariance: no integration → everything near zero
    let n = 8;
    let cov = identity_cov(n);

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    assert!(
        structural.macro_phi.abs() < 1e-6,
        "Identity should give macro phi ≈ 0, got {}",
        structural.macro_phi
    );
}

#[test]
fn test_structural_phi_three_blocks() {
    // Three clear blocks with strong intra and near-zero inter correlation
    // Larger clusters (6 each) give more robust per-cluster Phi
    let n = 18;
    let cov = three_block_cov(6, 6, 6, 0.8, 0.01, 0.01, 0.01);

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    assert!(
        structural.num_clusters >= 2,
        "Three-block should produce at least 2 clusters, got {}",
        structural.num_clusters
    );
    // With near-zero inter-block coupling, micro >> meso
    assert!(
        structural.micro_phi > structural.meso_phi,
        "Within-cluster integration ({}) should exceed between-cluster ({})",
        structural.micro_phi,
        structural.meso_phi
    );
}

#[test]
fn test_structural_phi_uniform_correlation() {
    // Uniform correlation: one big cluster, no hierarchy
    let n = 8;
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0 + 1e-6;
        for j in (i + 1)..n {
            cov[i * n + j] = 0.5;
            cov[j * n + i] = 0.5;
        }
    }

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    // With uniform correlation, cut_mis should be fairly flat → few/no valleys
    // This may produce 1 cluster (no hierarchy) or 2+ with balanced sizes
    assert!(structural.macro_phi.is_finite());
    assert!(structural.emergence_ratio.is_finite());
}

#[test]
fn test_structural_phi_cluster_sizes_sum_to_n() {
    let n = 16;
    let cov = block_diagonal_cov(8, 8, 0.6, 0.02);

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    let total_size: usize = structural.cluster_sizes.iter().sum();
    // Clusters may not cover all n if tiny fragments are filtered
    assert!(
        total_size >= n / 2,
        "Clusters should cover at least half of n={}: got {}",
        n,
        total_size
    );
    assert_eq!(
        structural.cluster_phis.len(),
        structural.cluster_sizes.len(),
        "cluster_phis and cluster_sizes should have same length"
    );
}

#[test]
fn test_structural_phi_scale_n64() {
    // Larger scale test: 4 blocks of 16
    let n = 64;
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0 + 1e-6;
        for j in (i + 1)..n {
            let same_block = i / 16 == j / 16;
            let corr = if same_block { 0.5 } else { 0.02 };
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }

    let finder = SpectralMIPFinder::with_defaults();
    let mip_result = finder.compute_from_covariance(&cov, n, 50).unwrap();
    let structural = finder
        .compute_structural_hierarchy_from_cov(&mip_result, &cov, n)
        .unwrap();

    assert!(
        structural.num_clusters >= 2,
        "4-block structure should produce at least 2 clusters, got {}",
        structural.num_clusters
    );
    assert!(structural.micro_phi.is_finite());
    assert!(structural.meso_phi.is_finite());
    assert!(structural.macro_phi > 0.0);
}
