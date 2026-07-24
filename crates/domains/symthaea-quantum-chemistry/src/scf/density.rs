// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Density matrix construction for RHF.
//!
//! P_μν = 2 Σ_{i=occupied} C_μi C_νi
//!
//! The factor of 2 accounts for double occupancy (alpha + beta electrons).

/// Build the density matrix from MO coefficients.
///
/// `coefficients` is n_basis × n_mo (row-major: C[μ * n_mo + i]).
/// `n_occupied` is the number of doubly-occupied orbitals.
///
/// Returns P as n_basis × n_basis (row-major).
pub fn build_density_matrix(
    coefficients: &[f64],
    n_basis: usize,
    n_mo: usize,
    n_occupied: usize,
) -> Vec<f64> {
    let mut p = vec![0.0; n_basis * n_basis];

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let mut sum = 0.0;
            for i in 0..n_occupied {
                sum += coefficients[mu * n_mo + i] * coefficients[nu * n_mo + i];
            }
            p[mu * n_basis + nu] = 2.0 * sum;
        }
    }

    p
}

/// Build a single-spin density matrix (UHF) from MO coefficients.
///
/// P^σ_μν = Σ_{i=occupied in spin σ} C_μi C_νi
///
/// Unlike `build_density_matrix` (RHF), there is NO factor of 2 here --
/// each spin channel's orbitals are singly occupied. Phase Q2, 2026-07-16.
pub fn build_density_matrix_unrestricted(
    coefficients: &[f64],
    n_basis: usize,
    n_mo: usize,
    n_occupied: usize,
) -> Vec<f64> {
    let mut p = vec![0.0; n_basis * n_basis];

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let mut sum = 0.0;
            for i in 0..n_occupied {
                sum += coefficients[mu * n_mo + i] * coefficients[nu * n_mo + i];
            }
            p[mu * n_basis + nu] = sum;
        }
    }

    p
}

/// Compute the RMS change between two density matrices.
pub fn density_rms_change(p_new: &[f64], p_old: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n * n {
        let diff = p_new[i] - p_old[i];
        sum += diff * diff;
    }
    (sum / (n * n) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_trace() {
        // For 1 occupied orbital with normalized coefficients [1/√2, 1/√2]
        // P should have trace = 2 (2 electrons)
        let s2 = 1.0 / 2.0_f64.sqrt();
        let coeffs = vec![s2, 0.0, s2, 0.0]; // 2x2, one MO
        let p = build_density_matrix(&coeffs, 2, 2, 1);

        let trace = p[0] + p[3]; // P[0,0] + P[1,1]
        assert!(
            (trace - 2.0).abs() < 1e-10,
            "Density trace = {}, expected 2.0",
            trace
        );
    }

    #[test]
    fn test_density_symmetry() {
        let coeffs = vec![0.8, 0.2, 0.6, 0.7];
        let p = build_density_matrix(&coeffs, 2, 2, 1);
        assert!((p[1] - p[2]).abs() < 1e-14, "P should be symmetric");
    }

    #[test]
    fn test_unrestricted_density_trace_has_no_factor_of_two() {
        // Same single-MO coefficients as test_density_trace, but the
        // single-spin density's trace should be 1.0, not 2.0 (Phase Q2).
        let s2 = 1.0 / 2.0_f64.sqrt();
        let coeffs = vec![s2, 0.0, s2, 0.0];
        let p = build_density_matrix_unrestricted(&coeffs, 2, 2, 1);
        let trace = p[0] + p[3];
        assert!(
            (trace - 1.0).abs() < 1e-10,
            "Unrestricted density trace = {}, expected 1.0",
            trace
        );
    }

    #[test]
    fn test_restricted_density_is_twice_unrestricted_for_same_coefficients() {
        let coeffs = vec![0.8, 0.2, 0.6, 0.7];
        let p_r = build_density_matrix(&coeffs, 2, 2, 1);
        let p_u = build_density_matrix_unrestricted(&coeffs, 2, 2, 1);
        for i in 0..4 {
            assert!(
                (p_r[i] - 2.0 * p_u[i]).abs() < 1e-14,
                "P_restricted[{i}]={} should equal 2*P_unrestricted[{i}]={}",
                p_r[i],
                p_u[i]
            );
        }
    }
}
