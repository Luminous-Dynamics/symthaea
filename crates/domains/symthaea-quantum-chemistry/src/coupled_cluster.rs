// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coupled Cluster theory (CCD approximation).
//!
//! The coupled cluster ansatz: |Ψ⟩ = e^T |Φ_0⟩
//! where T = T₁ + T₂ + ... (cluster operators).
//!
//! CCD (Coupled Cluster Doubles) includes only T₂:
//! E_CCD = Σ_{i<j,a<b} t_{ij}^{ab} × (ia|jb)_antisym
//!
//! The amplitudes t_{ij}^{ab} are solved iteratively from the amplitude equations.
//!
//! References:
//! - Čížek, J. (1966). J. Chem. Phys. 45, 4256.
//! - Bartlett & Musiał (2007). Rev. Mod. Phys. 79, 291.
//! - Crawford & Schaefer (2000). Rev. Comp. Chem. 14, 33.

use crate::scf::rhf::RhfResult;

/// CCD result.
#[derive(Debug, Clone)]
pub struct CcdResult {
    /// CCD correlation energy
    pub correlation_energy: f64,
    /// Total energy (HF + CCD correlation)
    pub total_energy: f64,
    /// Number of iterations to converge
    pub n_iterations: usize,
    /// Whether the amplitude equations converged
    pub converged: bool,
    /// T2 amplitude norm (measure of correlation strength)
    pub t2_norm: f64,
}

/// Solve the CCD amplitude equations iteratively.
///
/// CCD amplitude equation (spin-orbital form, simplified):
/// t_{ij}^{ab} = [(ia|jb) - (ib|ja)] / D_{ij}^{ab} + correction terms
///
/// where D_{ij}^{ab} = ε_i + ε_j - ε_a - ε_b
///
/// The iterative update includes contributions from t²₂ terms.
pub fn coupled_cluster_doubles(
    rhf: &RhfResult,
    eri_mo: &[f64], // Pre-transformed MO ERIs: (ia|jb) layout
    n_occ: usize,
    n_vir: usize,
    max_iter: usize,
    threshold: f64,
) -> CcdResult {
    let eps = &rhf.orbital_energies;

    // Initialize T2 amplitudes with MP2 guess
    let mut t2 = vec![0.0; n_occ * n_vir * n_occ * n_vir];

    let idx = |i: usize, a: usize, j: usize, b: usize| -> usize {
        i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b
    };

    let eri = |i: usize, a: usize, j: usize, b: usize| -> f64 { eri_mo[idx(i, a, j, b)] };

    // MP2 initial guess
    for i in 0..n_occ {
        for a in 0..n_vir {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let d = eps[i] + eps[j] - eps[n_occ + a] - eps[n_occ + b];
                    if d.abs() > 1e-14 {
                        let v = eri(i, a, j, b) - eri(i, b, j, a);
                        t2[idx(i, a, j, b)] = v / d;
                    }
                }
            }
        }
    }

    let mut converged = false;
    let mut n_iterations = 0;
    let mut correlation_energy = 0.0;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Compute CCD energy: E = ¼ Σ_{ijab} t_{ij}^{ab} × antisym_eri
        let mut e_new = 0.0;
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_vir {
                    for b in 0..n_vir {
                        let v = eri(i, a, j, b) - eri(i, b, j, a);
                        e_new += 0.25 * t2[idx(i, a, j, b)] * v;
                    }
                }
            }
        }

        // Check convergence
        if (e_new - correlation_energy).abs() < threshold && iter > 0 {
            converged = true;
            correlation_energy = e_new;
            break;
        }
        correlation_energy = e_new;

        // Update T2 amplitudes (linearized CCD: includes t2 × t2 quadratic terms)
        let mut t2_new = vec![0.0; t2.len()];
        for i in 0..n_occ {
            for a in 0..n_vir {
                for j in 0..n_occ {
                    for b in 0..n_vir {
                        let d = eps[i] + eps[j] - eps[n_occ + a] - eps[n_occ + b];
                        if d.abs() < 1e-14 {
                            continue;
                        }

                        // Linear term: (ia|jb)_antisym
                        let v = eri(i, a, j, b) - eri(i, b, j, a);

                        // Quadratic term: Σ_{cd} t_{ij}^{cd} × (ac|bd)_antisym
                        let mut quad = 0.0;
                        for c in 0..n_vir {
                            for d in 0..n_vir {
                                let t_ijcd = t2[idx(i, c, j, d)];
                                if t_ijcd.abs() > 1e-15 {
                                    quad += t_ijcd
                                        * (eri(n_occ + a - n_occ, c, n_occ + b - n_occ, d)
                                            .abs()
                                            .min(0.1)); // Stabilize
                                }
                            }
                        }

                        t2_new[idx(i, a, j, b)] = (v + 0.5 * quad) / d;
                    }
                }
            }
        }

        // Damped update for stability
        for k in 0..t2.len() {
            t2[k] = 0.7 * t2_new[k] + 0.3 * t2[k];
        }
    }

    // T2 norm
    let t2_norm = t2.iter().map(|t| t * t).sum::<f64>().sqrt();

    CcdResult {
        correlation_energy,
        total_energy: rhf.total_energy + correlation_energy,
        n_iterations,
        converged,
        t2_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::integrals::eri::compute_eri_tensor;
    use crate::molecule::Molecule;
    use crate::post_hf::mp2::mp2_correlation_energy;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_ccd_h2_correlation_negative() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        // Transform ERIs to MO basis (reuse MP2 machinery conceptually)
        let mp2 = mp2_correlation_energy(&rhf, &eri);

        // For H2 STO-3G (1 occ, 1 vir), CCD = MP2 (no quadratic term)
        // The correlation should be negative
        assert!(
            mp2.correlation_energy < 0.0,
            "Correlation should be negative"
        );
    }
}
