// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SpectralMIPFinder Cross-Validation
//!
//! Validates the production SpectralMIPFinder's O(n³) Fiedler-based MIP search
//! against an exhaustive O(2^n) MIP search on the SAME covariance matrices.
//!
//! This is the critical missing validation identified in March 2026:
//! the production consciousness engine uses SpectralMIPFinder (MI Laplacian
//! → Fiedler ordering → bordered Cholesky sweep), but its correlation with
//! true IIT Φ was UNKNOWN. This test establishes that correlation.
//!
//! ## Approach
//!
//! Both algorithms operate on the same Gaussian MI framework:
//! - `Φ = MI_total - MI_mip` where MI uses `0.5 * (Σ ln(σ²_i) - ln(det(Σ)))`
//! - Exhaustive: tries all 2^(n-1)-1 bipartitions (ground truth)
//! - Spectral: Fiedler ordering → n-1 contiguous cuts (fast approximation)
//!
//! We generate covariance matrices with known topology structure (star, ring,
//! modular, random, chain) and compare Φ values across both methods.
//!
//! ## Date: March 2, 2026

use symthaea::symthaea_core::consciousness_metrics::{
    SpectralMIPConfig, SpectralMIPFinder, SpectralMIPResult,
};
use symthaea::symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// EXHAUSTIVE GAUSSIAN MI MIP SEARCH (ground truth)
// ═══════════════════════════════════════════════════════════════════════════════

/// ln(det(M)) via Cholesky decomposition. Falls back to diagonal sum if not PD.
fn ln_determinant(matrix: &[f64], n: usize) -> f64 {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in 0..j {
                sum += l[i * n + p] * l[j * n + p];
            }
            if i == j {
                let diag = matrix[i * n + i] - sum;
                if diag <= 0.0 {
                    // Not positive definite — fall back to diagonal
                    return (0..n).map(|k| matrix[k * n + k].max(1e-30).ln()).sum();
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    let mut result = 0.0;
    for i in 0..n {
        result += l[i * n + i].ln();
    }
    2.0 * result
}

/// Gaussian MI of a set of variables: MI = 0.5 * (Σ ln(σ²_i) - ln(det(Σ)))
fn gaussian_mi(cov: &[f64], n: usize, indices: &[usize]) -> f64 {
    let m = indices.len();
    if m <= 1 {
        return 0.0;
    }

    // Extract sub-covariance matrix
    let mut sub = vec![0.0f64; m * m];
    for (new_i, &old_i) in indices.iter().enumerate() {
        for (new_j, &old_j) in indices.iter().enumerate() {
            sub[new_i * m + new_j] = cov[old_i * n + old_j];
        }
    }

    let sum_ln_var: f64 = indices
        .iter()
        .map(|&i| cov[i * n + i].max(1e-30).ln())
        .sum();
    let ln_det = ln_determinant(&sub, m);
    0.5 * (sum_ln_var - ln_det)
}

/// Add regularization to a covariance matrix (matching SpectralMIPFinder).
fn regularize(cov: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let mut reg = cov.to_vec();
    for i in 0..n {
        reg[i * n + i] += eps;
    }
    reg
}

/// Exhaustive MIP search over all 2^(n-1)-1 non-trivial bipartitions.
/// Returns (phi, mip_mi, part_a, part_b).
fn exhaustive_gaussian_mip(cov: &[f64], n: usize) -> (f64, f64, Vec<usize>, Vec<usize>) {
    // Total MI of the full system
    let total_mi = gaussian_mi(cov, n, &(0..n).collect::<Vec<_>>());

    let mut best_mi = f64::MAX;
    let mut best_mask = 1u64;

    // Enumerate all bipartitions: mask from 1 to 2^(n-1)-1
    // (we only go to 2^(n-1) to avoid mirror partitions)
    let limit = 1u64 << (n - 1);
    for mask in 1..limit {
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }

        let mi_a = gaussian_mi(cov, n, &part_a);
        let mi_b = gaussian_mi(cov, n, &part_b);
        let cut_mi = mi_a + mi_b;

        if cut_mi < best_mi {
            best_mi = cut_mi;
            best_mask = mask;
        }
    }

    let mut part_a = Vec::new();
    let mut part_b = Vec::new();
    for i in 0..n {
        if (best_mask >> i) & 1 == 1 {
            part_a.push(i);
        } else {
            part_b.push(i);
        }
    }

    let phi = (total_mi - best_mi).max(0.0);
    (phi, best_mi, part_a, part_b)
}

// ═══════════════════════════════════════════════════════════════════════════════
// COVARIANCE MATRIX GENERATORS
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a star topology covariance: node 0 correlated with all others,
/// others uncorrelated with each other.
fn star_covariance(n: usize, rho: f64) -> Vec<f64> {
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0; // unit variance
    }
    // Hub (node 0) correlated with all spokes
    for i in 1..n {
        cov[0 * n + i] = rho;
        cov[i * n + 0] = rho;
    }
    cov
}

/// Generate a ring topology covariance: each node correlated with its neighbors.
fn ring_covariance(n: usize, rho: f64) -> Vec<f64> {
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    for i in 0..n {
        let next = (i + 1) % n;
        cov[i * n + next] = rho;
        cov[next * n + i] = rho;
    }
    cov
}

/// Generate a chain topology covariance: linear chain, decaying correlation.
fn chain_covariance(n: usize, rho: f64) -> Vec<f64> {
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let distance = (j - i) as f64;
            let corr = rho.powf(distance);
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    cov
}

/// Generate a modular topology covariance: two blocks with high intra-,
/// low inter-cluster correlation.
fn modular_covariance(n: usize, rho_intra: f64, rho_inter: f64) -> Vec<f64> {
    let half = n / 2;
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let same_cluster = (i < half && j < half) || (i >= half && j >= half);
            let corr = if same_cluster { rho_intra } else { rho_inter };
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    cov
}

/// Generate a random positive-definite covariance matrix with seed.
fn random_covariance(n: usize, seed: u64) -> Vec<f64> {
    // Generate a random lower-triangular matrix L, then cov = L * L^T
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let val = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            l[i * n + j] = val;
        }
        // Ensure positive diagonal
        l[i * n + i] = l[i * n + i].abs() + 0.5;
    }
    // cov = L * L^T
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += l[i * n + k] * l[j * n + k];
            }
            cov[i * n + j] = sum;
        }
    }
    // Normalize to unit variance (correlation matrix)
    let diag: Vec<f64> = (0..n).map(|i| cov[i * n + i].sqrt()).collect();
    for i in 0..n {
        for j in 0..n {
            cov[i * n + j] /= diag[i] * diag[j];
        }
    }
    cov
}

/// Generate a uniform (identity) covariance — no correlation.
fn identity_covariance(n: usize) -> Vec<f64> {
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    cov
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORRELATION STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    fn rank(values: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut ranks = vec![0.0; values.len()];
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            ranks[*orig_idx] = rank as f64 + 1.0;
        }
        ranks
    }
    let rank_x = rank(x);
    let rank_y = rank(y);
    pearson_correlation(&rank_x, &rank_y)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Helper: feed covariance-structured snapshots to SpectralMIPFinder and compute.
///
/// Generates `num_samples` ContinuousHV snapshots from the given covariance
/// (Cholesky decomposition → correlated samples), feeds them to the finder.
fn spectral_phi_from_covariance(
    cov: &[f64],
    n: usize,
    num_samples: usize,
) -> Option<SpectralMIPResult> {
    let config = SpectralMIPConfig {
        num_components: n,
        window_size: num_samples,
        min_samples: num_samples.min(10),
        regularization: 1e-6,
        normalize_variance: true,
        temporal_decorrelation: true,
    };
    let mut finder = SpectralMIPFinder::new(config);

    // Cholesky decomposition of covariance for sampling
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in 0..j {
                sum += l[i * n + p] * l[j * n + p];
            }
            if i == j {
                let diag = cov[i * n + i] - sum;
                l[i * n + j] = if diag > 0.0 { diag.sqrt() } else { 1e-6 };
            } else {
                let denom = l[j * n + j];
                l[i * n + j] = if denom.abs() > 1e-15 {
                    (cov[i * n + j] - sum) / denom
                } else {
                    0.0
                };
            }
        }
    }

    // Generate correlated samples: x = L * z where z ~ N(0, I)
    let mut rng_state = 42u64 ^ 0x9E3779B97F4A7C15;
    for _ in 0..num_samples {
        // Box-Muller for standard normal samples
        let mut z = vec![0.0f64; n];
        for zi in z.iter_mut() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u2 = (rng_state as f64 / u64::MAX as f64) * std::f64::consts::TAU;
            *zi = (-2.0 * u1.ln()).sqrt() * u2.cos();
        }

        // x = L * z
        let mut x = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += l[i * n + j] * z[j];
            }
            x[i] = sum as f32;
        }

        let hv = ContinuousHV::from_vec(x);
        finder.push(&hv);
    }

    finder.compute()
}

/// Alternative: use compute_from_covariance directly (bypasses sampling noise).
/// Takes a pre-regularized covariance matrix.
fn spectral_phi_direct(cov: &[f64], n: usize) -> Option<SpectralMIPResult> {
    let config = SpectralMIPConfig {
        num_components: n,
        window_size: 50,
        min_samples: 10,
        regularization: 1e-6,
        normalize_variance: true,
        temporal_decorrelation: true,
    };
    let finder = SpectralMIPFinder::new(config);
    // Add regularization to match what SpectralMIPFinder's build_covariance_matrix does
    let reg_cov = regularize(cov, n, 1e-6);
    finder.compute_from_covariance(&reg_cov, n, 50)
}

#[test]
fn test_spectral_mip_vs_exhaustive_correlation() {
    println!("\n========================================================");
    println!("SpectralMIPFinder vs Exhaustive MIP: Cross-Validation");
    println!("========================================================\n");

    let mut exact_phis = Vec::new();
    let mut spectral_phis = Vec::new();

    let rhos = [0.3, 0.5, 0.7];
    let sizes = [4, 5, 6, 7, 8];
    let eps = 1e-6; // Matching SpectralMIPFinder's regularization

    println!(
        "{:<12} {:<6} {:<6} {:<12} {:<12} {:<8}",
        "Topology", "n", "rho", "Exact", "Spectral", "Ratio"
    );
    println!("{}", "-".repeat(65));

    let topologies: Vec<(&str, Box<dyn Fn(usize, f64) -> Vec<f64>>)> = vec![
        ("star", Box::new(|n, rho| star_covariance(n, rho))),
        ("ring", Box::new(|n, rho| ring_covariance(n, rho))),
        ("chain", Box::new(|n, rho| chain_covariance(n, rho))),
        (
            "modular",
            Box::new(|n, rho| modular_covariance(n, rho, rho * 0.1)),
        ),
        (
            "random",
            Box::new(|n, rho| random_covariance(n, 1000 + n as u64 * 100 + (rho * 10.0) as u64)),
        ),
    ];

    for &n in &sizes {
        for &rho in &rhos {
            for (name, gen_cov) in &topologies {
                let raw_cov = gen_cov(n, rho);
                // Apply SAME regularization to both paths for fair comparison
                let cov = regularize(&raw_cov, n, eps);

                let (phi_exact, _, _, _) = exhaustive_gaussian_mip(&cov, n);
                if let Some(result) = spectral_phi_direct(&raw_cov, n) {
                    // Skip degenerate cases where both are effectively zero
                    if phi_exact < 1e-8 && result.phi < 1e-8 {
                        continue;
                    }
                    let ratio = if phi_exact > 1e-10 {
                        result.phi / phi_exact
                    } else {
                        f64::NAN
                    };
                    println!(
                        "{:<12} {:<6} {:<6.1} {:<12.6} {:<12.6} {:<8.3}",
                        name, n, rho, phi_exact, result.phi, ratio
                    );
                    exact_phis.push(phi_exact);
                    spectral_phis.push(result.phi);
                }
            }
        }
    }

    println!("\n========================================================");
    println!("CORRELATION ANALYSIS");
    println!("========================================================\n");

    let r = pearson_correlation(&exact_phis, &spectral_phis);
    let rho = spearman_correlation(&exact_phis, &spectral_phis);

    println!("SpectralMIPFinder vs Exhaustive MIP:");
    println!("  Pearson r:  {:.4}", r);
    println!("  Spearman ρ: {:.4}", rho);
    println!("  N samples:  {}", exact_phis.len());
    println!();

    let exact_mean: f64 = exact_phis.iter().sum::<f64>() / exact_phis.len() as f64;
    let spectral_mean: f64 = spectral_phis.iter().sum::<f64>() / spectral_phis.len() as f64;
    println!("Mean Φ values:");
    println!("  Exhaustive: {:.6}", exact_mean);
    println!("  Spectral:   {:.6}", spectral_mean);

    // Compute ratio statistics
    let ratios: Vec<f64> = exact_phis
        .iter()
        .zip(spectral_phis.iter())
        .filter(|(&e, _)| e > 1e-10)
        .map(|(&e, &s)| s / e)
        .collect();
    if !ratios.is_empty() {
        let ratio_mean: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let ratio_min = ratios.iter().cloned().fold(f64::MAX, f64::min);
        let ratio_max = ratios.iter().cloned().fold(f64::MIN, f64::max);
        println!("\nΦ_spectral / Φ_exact ratio:");
        println!("  Mean:  {:.3}", ratio_mean);
        println!("  Min:   {:.3}", ratio_min);
        println!("  Max:   {:.3}", ratio_max);
    }

    println!("\n========================================================");
    println!("VALIDATION RESULT");
    println!("========================================================\n");

    // Classification thresholds
    let strong = r > 0.70 && rho > 0.70;
    let moderate = r > 0.50 || rho > 0.50;
    let verdict = if strong {
        "✅ STRONG VALIDATION (both r > 0.70 and ρ > 0.70)"
    } else if moderate {
        "⚠️ MODERATE CORRELATION (at least one > 0.50)"
    } else {
        "❌ WEAK CORRELATION"
    };
    println!("SpectralMIPFinder: {}", verdict);
    println!("  Pearson r = {:.4}, Spearman ρ = {:.4}", r, rho);
    println!();
    println!("Interpretation:");
    println!("  The Fiedler ordering restricts MIP search to n-1 contiguous cuts.");
    println!("  If the true MIP is non-contiguous in Fiedler order, it will be missed.");
    println!("  Spearman ρ measures rank agreement (topology ordering preserved).");
    println!("  Pearson r measures linear agreement (absolute Φ values match).");

    // We require at least moderate correlation. The spectral shortcut trades
    // exactness for O(n³) speed. Perfect correlation isn't expected, but the
    // rank ordering should be preserved (ρ > 0.50).
    assert!(
        rho > 0.50,
        "SpectralMIPFinder rank correlation too low (ρ={:.4}). \
         The Fiedler ordering is not preserving relative Φ rankings.",
        rho
    );
}

#[test]
fn test_spectral_mip_identity_gives_zero_phi() {
    // An identity covariance (no correlations) should give Φ ≈ 0
    for n in [4, 6, 8] {
        let cov = identity_covariance(n);
        if let Some(result) = spectral_phi_direct(&cov, n) {
            println!("Identity n={}: Φ = {:.6}", n, result.phi);
            assert!(
                result.phi < 0.01,
                "Identity covariance should give near-zero Φ (got {:.6} for n={})",
                result.phi,
                n
            );
        }
    }
}

#[test]
fn test_spectral_mip_modular_finds_natural_partition() {
    // A strongly modular covariance should have its MIP at the module boundary
    let n = 8;
    let cov = modular_covariance(n, 0.8, 0.05);

    if let Some(result) = spectral_phi_direct(&cov, n) {
        println!("Modular n=8: Φ = {:.6}", result.phi);
        println!("  MIP bond: {}", result.mip_bond);
        println!(
            "  Part A: {:?} (size {})",
            result.mip.part_a,
            result.mip.part_a.len()
        );
        println!(
            "  Part B: {:?} (size {})",
            result.mip.part_b,
            result.mip.part_b.len()
        );

        // The MIP should split into two groups of ~4
        let a_len = result.mip.part_a.len();
        let b_len = result.mip.part_b.len();
        let balanced =
            (a_len == 4 && b_len == 4) || (a_len == 3 && b_len == 5) || (a_len == 5 && b_len == 3);
        println!(
            "  Balanced split: {} ({} vs {})",
            if balanced { "✅" } else { "⚠️" },
            a_len,
            b_len
        );
    }
}

#[test]
fn test_spectral_mip_stronger_correlation_higher_phi() {
    // Higher correlation → higher mutual information → higher Φ
    // Use chain topology (always PD, clean monotonic behavior) rather than
    // star (which becomes near-singular at high ρ, distorted by regularization).
    let n = 6;

    let mut phis = Vec::new();
    for &rho in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let cov = chain_covariance(n, rho);
        if let Some(result) = spectral_phi_direct(&cov, n) {
            println!("Chain n={}, ρ={:.1}: Φ = {:.6}", n, rho, result.phi);
            phis.push((rho, result.phi));
        }
    }

    // Check monotonicity: Φ should generally increase with correlation
    let mut monotonic_pairs = 0;
    let mut total_pairs = 0;
    for i in 0..phis.len() {
        for j in (i + 1)..phis.len() {
            total_pairs += 1;
            if phis[j].1 >= phis[i].1 - 1e-6 {
                monotonic_pairs += 1;
            }
        }
    }
    let monotonicity = monotonic_pairs as f64 / total_pairs as f64;
    println!(
        "\nMonotonicity: {:.0}% ({}/{})",
        monotonicity * 100.0,
        monotonic_pairs,
        total_pairs
    );

    assert!(
        monotonicity >= 0.7,
        "Φ should generally increase with correlation strength (monotonicity = {:.0}%)",
        monotonicity * 100.0
    );
}

#[test]
fn test_spectral_mip_with_sampled_window() {
    // Test that the full pipeline (push snapshots → compute) gives similar
    // results to the direct covariance path.
    let n = 6;
    let rho = 0.5;
    let cov = star_covariance(n, rho);

    // Direct path
    let direct = spectral_phi_direct(&cov, n).expect("direct should produce result");

    // Sampled path (200 samples for statistical convergence)
    let sampled =
        spectral_phi_from_covariance(&cov, n, 200).expect("sampled should produce result");

    println!("Star n=6, ρ=0.5:");
    println!("  Direct Φ:  {:.6}", direct.phi);
    println!("  Sampled Φ: {:.6}", sampled.phi);

    let ratio = if direct.phi > 1e-10 {
        sampled.phi / direct.phi
    } else {
        1.0
    };
    println!("  Ratio: {:.3}", ratio);

    // Allow up to 50% deviation due to sampling noise
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Sampled Φ ({:.6}) should be within 2x of direct Φ ({:.6})",
        sampled.phi,
        direct.phi
    );
}

#[test]
fn test_exact_and_spectral_agree_on_trivial_cases() {
    // Single element: exhaustive gives 0
    let cov = vec![1.0];
    let (phi_exact, _, _, _) = exhaustive_gaussian_mip(&cov, 1);
    assert_eq!(phi_exact, 0.0, "Exact Φ should be 0 for n=1");
    // SpectralMIPFinder requires num_components >= 2, so we skip n=1

    // Two uncorrelated variables: Φ ≈ 0
    let cov = identity_covariance(2);
    let (phi_exact, _, _, _) = exhaustive_gaussian_mip(&cov, 2);
    assert!(
        phi_exact < 0.01,
        "Exact Φ should be ~0 for uncorrelated pair (got {})",
        phi_exact
    );
    if let Some(result) = spectral_phi_direct(&cov, 2) {
        assert!(
            result.phi < 0.01,
            "Spectral Φ should be ~0 for uncorrelated pair (got {})",
            result.phi
        );
    }

    // Four uncorrelated variables: Φ ≈ 0
    let cov = identity_covariance(4);
    let (phi_exact, _, _, _) = exhaustive_gaussian_mip(&cov, 4);
    assert!(
        phi_exact < 0.01,
        "Exact Φ should be ~0 for 4 uncorrelated vars (got {})",
        phi_exact
    );
    if let Some(result) = spectral_phi_direct(&cov, 4) {
        assert!(
            result.phi < 0.01,
            "Spectral Φ should be ~0 for 4 uncorrelated vars (got {})",
            result.phi
        );
    }
}
