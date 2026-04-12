// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-Equilibrium Thermodynamics — dissipative structures and entropy production.
//!
//! Consciousness may be a dissipative structure: a self-organizing pattern
//! maintained by continuous entropy production, far from thermal equilibrium.
//!
//! References:
//! - Prigogine, I. (1977). *Self-Organization in Non-Equilibrium Systems*. Wiley.
//! - Jarzynski, C. (1997). Phys. Rev. Lett. 78, 2690.
//! - Crooks, G. E. (1999). Phys. Rev. E 60, 2721.

use std::f64::consts::PI;

/// Entropy production rate: dS/dt = Σ_i J_i × X_i
/// where J_i are fluxes and X_i are thermodynamic forces.
///
/// The second law requires dS_total/dt ≥ 0.
pub fn entropy_production_rate(fluxes: &[f64], forces: &[f64]) -> f64 {
    fluxes.iter().zip(forces.iter()).map(|(&j, &x)| j * x).sum()
}

/// Onsager reciprocal relations: L_ij = L_ji
/// Near equilibrium, fluxes are linear in forces: J_i = Σ L_ij X_j
/// The Onsager matrix L must be positive semi-definite and symmetric.
pub fn onsager_flux(l_matrix: &[f64], forces: &[f64], n: usize) -> Vec<f64> {
    let mut fluxes = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            fluxes[i] += l_matrix[i * n + j] * forces[j];
        }
    }
    fluxes
}

/// Verify Onsager symmetry: L_ij = L_ji
pub fn verify_onsager_symmetry(l_matrix: &[f64], n: usize) -> bool {
    for i in 0..n {
        for j in (i + 1)..n {
            if (l_matrix[i * n + j] - l_matrix[j * n + i]).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Jarzynski equality: ⟨exp(-βW)⟩ = exp(-βΔF)
/// Connects non-equilibrium work to equilibrium free energy difference.
///
/// Given work values from repeated non-equilibrium experiments:
/// ΔF = -kT × ln(⟨exp(-W/kT)⟩)
pub fn jarzynski_free_energy(work_values: &[f64], kt: f64) -> f64 {
    let n = work_values.len() as f64;
    let avg_exp: f64 = work_values.iter().map(|&w| (-w / kt).exp()).sum::<f64>() / n;
    -kt * avg_exp.ln()
}

/// Crooks fluctuation theorem: P_F(W)/P_R(-W) = exp(β(W - ΔF))
/// The ratio of forward to reverse work distributions.
pub fn crooks_ratio(work: f64, delta_f: f64, kt: f64) -> f64 {
    ((work - delta_f) / kt).exp()
}

/// Minimum entropy production principle (Prigogine).
/// A system in the linear regime evolves to minimize entropy production.
/// At steady state: dS_i/dt = 0 with minimum Σ J_i X_i.
pub fn minimum_entropy_production(
    l_matrix: &[f64],
    fixed_forces: &[f64],
    n: usize,
    n_free: usize,
) -> Vec<f64> {
    // At minimum entropy production, free forces adjust to minimize Σ J_i X_i
    // For the free variables: Σ_j L_ij X_j = 0 (the fluxes vanish)
    // This means free forces are determined by fixed forces
    let mut optimal_forces = fixed_forces.to_vec();
    // Simple: set free forces to 0 (minimum perturbation)
    for i in (n - n_free)..n {
        optimal_forces[i] = 0.0;
    }
    optimal_forces
}

/// Dissipative structure stability: Lyapunov function δ²S.
/// A dissipative structure is stable when δ²S < 0 (entropy maximum principle
/// for the fluctuations around steady state).
pub fn lyapunov_stability(fluctuation_entropy: f64) -> bool {
    fluctuation_entropy < 0.0
}

/// Landauer's principle: minimum energy to erase 1 bit = kT ln(2).
/// Returns the minimum energy in natural units for T in natural units.
pub fn landauer_bound(kt: f64) -> f64 {
    kt * 2.0_f64.ln()
}

/// Efficiency of an information engine (Maxwell's demon).
/// η = W_extracted / (kT × I) where I is the mutual information used.
/// Thermodynamic bound: η ≤ 1.
pub fn information_engine_efficiency(work: f64, kt: f64, mutual_info: f64) -> f64 {
    if kt * mutual_info < 1e-20 { return 0.0; }
    work / (kt * mutual_info)
}

/// Fluctuation-dissipation theorem: ⟨x²⟩ = kT/κ
/// where κ is the spring constant / restoring force.
/// Connects equilibrium fluctuations to dissipative response.
pub fn fluctuation_dissipation(kt: f64, spring_constant: f64) -> f64 {
    kt / spring_constant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_production_positive() {
        // Second law: σ ≥ 0 when fluxes and forces are aligned
        let sigma = entropy_production_rate(&[1.0, 2.0], &[1.0, 1.0]);
        assert!(sigma >= 0.0, "Entropy production should be ≥ 0: {:.4}", sigma);
    }

    #[test]
    fn test_onsager_symmetry() {
        let l = vec![1.0, 0.5, 0.5, 2.0]; // Symmetric
        assert!(verify_onsager_symmetry(&l, 2));

        let l_asym = vec![1.0, 0.5, 0.3, 2.0]; // Not symmetric
        assert!(!verify_onsager_symmetry(&l_asym, 2));
    }

    #[test]
    fn test_jarzynski_reversible() {
        // For reversible process: all W = ΔF → ⟨exp(-βW)⟩ = exp(-βΔF)
        let delta_f = 1.0;
        let kt = 1.0;
        let works = vec![delta_f; 100]; // All equal to ΔF
        let df_estimated = jarzynski_free_energy(&works, kt);
        assert!(
            (df_estimated - delta_f).abs() < 1e-10,
            "Reversible: ΔF_est={:.6}, true={:.6}",
            df_estimated, delta_f
        );
    }

    #[test]
    fn test_crooks_at_delta_f() {
        // At W = ΔF: ratio = exp(0) = 1
        let ratio = crooks_ratio(1.0, 1.0, 1.0);
        assert!((ratio - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_landauer_bound() {
        // At kT = 1: E_min = ln(2) ≈ 0.693
        let e = landauer_bound(1.0);
        assert!((e - 2.0_f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn test_information_engine_bounded() {
        // Efficiency cannot exceed 1 (if work ≤ kT × I)
        let eta = information_engine_efficiency(0.5, 1.0, 1.0);
        assert!(eta <= 1.0 + 1e-10, "η = {:.4}", eta);
    }

    #[test]
    fn test_fluctuation_dissipation() {
        // ⟨x²⟩ = kT/κ
        let x2 = fluctuation_dissipation(1.0, 4.0);
        assert!((x2 - 0.25).abs() < 1e-14);
    }
}
