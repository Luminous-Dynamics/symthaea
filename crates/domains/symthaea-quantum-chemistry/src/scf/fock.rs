// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fock matrix construction for RHF.
//!
//! F_μν = H_core_μν + G_μν
//!
//! where G_μν = Σ_{λσ} P_λσ [(μν|λσ) - ½(μλ|νσ)]
//!
//! The first term is the Coulomb (J) contribution, the second is Exchange (K).
//!
//! `build_fock_matrix`/`build_uhf_fock_matrices` above read from a
//! precomputed dense `n^4` ERI tensor ("conventional" SCF -- fast repeated
//! reads, `O(n^4)` memory). `build_fock_matrix_direct`/
//! `build_uhf_fock_matrices_direct` below never materialize that tensor,
//! computing each needed integral on demand instead ("direct" SCF --
//! `O(n^2)` memory, more redundant computation). Phase Q3, 2026-07-16 --
//! the concrete fix for the memory-scaling problem Phase Q0 found
//! (`compute_eri_tensor`'s dense storage is inconsistent with this crate's
//! own "< 200 basis functions" scope claim: 200^4 doubles ~= 12.8 GB).

use crate::basis::ContractedGaussian;
use crate::constants::INTEGRAL_SCREENING_THRESHOLD;
use crate::integrals::eri::eri_contracted;

/// Build the Fock matrix from core Hamiltonian, density matrix, and ERIs.
///
/// `h_core` is the one-electron Hamiltonian (T + V), n×n.
/// `density` is the density matrix P, n×n.
/// `eri` is the ERI tensor, n⁴ flat array.
///
/// Returns F = H_core + G(P) as n×n.
pub fn build_fock_matrix(h_core: &[f64], density: &[f64], eri: &[f64], n: usize) -> Vec<f64> {
    let n2 = n * n;
    let n3 = n2 * n;
    let mut fock = h_core.to_vec();

    for mu in 0..n {
        for nu in mu..n {
            let mut g = 0.0;

            for lam in 0..n {
                for sig in 0..n {
                    let p_ls = density[lam * n + sig];

                    // Coulomb: (μν|λσ)
                    let j = eri[mu * n3 + nu * n2 + lam * n + sig];

                    // Exchange: (μλ|νσ)
                    let k = eri[mu * n3 + lam * n2 + nu * n + sig];

                    g += p_ls * (j - 0.5 * k);
                }
            }

            fock[mu * n + nu] += g;
            if mu != nu {
                fock[nu * n + mu] += g;
            }
        }
    }

    fock
}

/// Compute the electronic energy from density, core Hamiltonian, and Fock matrix.
///
/// E_elec = ½ Tr[P (H_core + F)]
pub fn electronic_energy(density: &[f64], h_core: &[f64], fock: &[f64], n: usize) -> f64 {
    let mut energy = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            energy += density[mu * n + nu] * (h_core[mu * n + nu] + fock[mu * n + nu]);
        }
    }
    0.5 * energy
}

/// Build the alpha and beta Fock matrices for UHF.
///
/// F^α_μν = H_core_μν + Σ_λσ [ P^T_λσ (μν|λσ) - P^α_λσ (μλ|νσ) ]
/// F^β_μν = H_core_μν + Σ_λσ [ P^T_λσ (μν|λσ) - P^β_λσ (μλ|νσ) ]
///
/// where P^T = P^α + P^β. Unlike `build_fock_matrix` (RHF), there is NO 0.5
/// factor on the exchange term -- that factor is only correct for RHF's
/// density, which already carries the factor-of-2 double-occupancy
/// convention `build_density_matrix_unrestricted`'s single-spin densities
/// don't have. Reference: Szabo & Ostlund (1996), Ch. 3 (already cited
/// throughout this crate). Phase Q2, 2026-07-16.
pub fn build_uhf_fock_matrices(
    h_core: &[f64],
    density_alpha: &[f64],
    density_beta: &[f64],
    eri: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n2 = n * n;
    let n3 = n2 * n;
    let mut fock_alpha = h_core.to_vec();
    let mut fock_beta = h_core.to_vec();

    let density_total: Vec<f64> = density_alpha
        .iter()
        .zip(density_beta.iter())
        .map(|(a, b)| a + b)
        .collect();

    for mu in 0..n {
        for nu in mu..n {
            let mut g_alpha = 0.0;
            let mut g_beta = 0.0;

            for lam in 0..n {
                for sig in 0..n {
                    let p_t = density_total[lam * n + sig];
                    let p_a = density_alpha[lam * n + sig];
                    let p_b = density_beta[lam * n + sig];

                    // Coulomb: (μν|λσ), shared between spins via P^T
                    let j = eri[mu * n3 + nu * n2 + lam * n + sig];
                    // Exchange: (μλ|νσ), per-spin
                    let k = eri[mu * n3 + lam * n2 + nu * n + sig];

                    g_alpha += p_t * j - p_a * k;
                    g_beta += p_t * j - p_b * k;
                }
            }

            fock_alpha[mu * n + nu] += g_alpha;
            fock_beta[mu * n + nu] += g_beta;
            if mu != nu {
                fock_alpha[nu * n + mu] += g_alpha;
                fock_beta[nu * n + mu] += g_beta;
            }
        }
    }

    (fock_alpha, fock_beta)
}

/// UHF electronic energy: E = ½ Σ_μν [ P^T_μν H_core_μν + P^α_μν F^α_μν + P^β_μν F^β_μν ]
pub fn uhf_electronic_energy(
    density_alpha: &[f64],
    density_beta: &[f64],
    h_core: &[f64],
    fock_alpha: &[f64],
    fock_beta: &[f64],
    n: usize,
) -> f64 {
    let mut energy = 0.0;
    for i in 0..n * n {
        let p_t = density_alpha[i] + density_beta[i];
        energy +=
            p_t * h_core[i] + density_alpha[i] * fock_alpha[i] + density_beta[i] * fock_beta[i];
    }
    0.5 * energy
}

/// Direct-SCF RHF Fock build: same `G = P*(J - 0.5*K)` combination as
/// `build_fock_matrix`, but computes each `(μν|λσ)`/`(μλ|νσ)` integral via
/// `eri_contracted` on demand (Schwarz-screened using `schwarz`, from
/// `integrals::eri::compute_schwarz_bounds`) instead of reading a
/// precomputed dense tensor. `schwarz[mu*n+nu]` must be the bound for basis
/// functions `(mu, nu)`. See the module doc for the memory/compute
/// tradeoff. Phase Q3, 2026-07-16.
pub fn build_fock_matrix_direct(
    h_core: &[f64],
    density: &[f64],
    basis: &[ContractedGaussian],
    schwarz: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut fock = h_core.to_vec();
    let threshold = INTEGRAL_SCREENING_THRESHOLD;

    for mu in 0..n {
        for nu in mu..n {
            let mut g = 0.0;
            let bound_mn = schwarz[mu * n + nu];

            for lam in 0..n {
                for sig in 0..n {
                    let p_ls = density[lam * n + sig];
                    if p_ls.abs() < 1e-15 {
                        continue;
                    }
                    let bound_ls = schwarz[lam * n + sig];
                    if bound_mn * bound_ls < threshold {
                        continue;
                    }

                    let j = eri_contracted(&basis[mu], &basis[nu], &basis[lam], &basis[sig]);
                    let k = eri_contracted(&basis[mu], &basis[lam], &basis[nu], &basis[sig]);

                    g += p_ls * (j - 0.5 * k);
                }
            }

            fock[mu * n + nu] += g;
            if mu != nu {
                fock[nu * n + mu] += g;
            }
        }
    }

    fock
}

/// Direct-SCF UHF Fock build: same combination as `build_uhf_fock_matrices`,
/// computing integrals on demand instead of reading a precomputed tensor.
/// Phase Q3, 2026-07-16.
pub fn build_uhf_fock_matrices_direct(
    h_core: &[f64],
    density_alpha: &[f64],
    density_beta: &[f64],
    basis: &[ContractedGaussian],
    schwarz: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut fock_alpha = h_core.to_vec();
    let mut fock_beta = h_core.to_vec();
    let threshold = INTEGRAL_SCREENING_THRESHOLD;

    let density_total: Vec<f64> = density_alpha
        .iter()
        .zip(density_beta.iter())
        .map(|(a, b)| a + b)
        .collect();

    for mu in 0..n {
        for nu in mu..n {
            let mut g_alpha = 0.0;
            let mut g_beta = 0.0;
            let bound_mn = schwarz[mu * n + nu];

            for lam in 0..n {
                for sig in 0..n {
                    let p_t = density_total[lam * n + sig];
                    let p_a = density_alpha[lam * n + sig];
                    let p_b = density_beta[lam * n + sig];
                    if p_t.abs() < 1e-15 && p_a.abs() < 1e-15 && p_b.abs() < 1e-15 {
                        continue;
                    }
                    let bound_ls = schwarz[lam * n + sig];
                    if bound_mn * bound_ls < threshold {
                        continue;
                    }

                    let j = eri_contracted(&basis[mu], &basis[nu], &basis[lam], &basis[sig]);
                    let k = eri_contracted(&basis[mu], &basis[lam], &basis[nu], &basis[sig]);

                    g_alpha += p_t * j - p_a * k;
                    g_beta += p_t * j - p_b * k;
                }
            }

            fock_alpha[mu * n + nu] += g_alpha;
            fock_beta[mu * n + nu] += g_beta;
            if mu != nu {
                fock_alpha[nu * n + mu] += g_alpha;
                fock_beta[nu * n + mu] += g_beta;
            }
        }
    }

    (fock_alpha, fock_beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::integrals::eri::{compute_eri_tensor, compute_schwarz_bounds};
    use crate::molecule::Molecule;

    #[test]
    fn test_fock_equals_hcore_with_zero_density() {
        // With P = 0, F = H_core
        let n = 2;
        let h_core = vec![1.0, 0.5, 0.5, 2.0];
        let density = vec![0.0; 4];
        let eri = vec![0.0; 16];
        let fock = build_fock_matrix(&h_core, &density, &eri, n);
        for i in 0..4 {
            assert!(
                (fock[i] - h_core[i]).abs() < 1e-14,
                "F[{}] = {}, H_core[{}] = {}",
                i,
                fock[i],
                i,
                h_core[i]
            );
        }
    }

    #[test]
    fn test_fock_symmetry() {
        let n = 2;
        let h_core = vec![1.0, 0.3, 0.3, 2.0];
        let density = vec![1.0, 0.5, 0.5, 1.0];
        let eri = vec![0.1; 16]; // uniform ERIs for simplicity
        let fock = build_fock_matrix(&h_core, &density, &eri, n);

        assert!(
            (fock[1] - fock[2]).abs() < 1e-14,
            "Fock should be symmetric"
        );
    }

    #[test]
    fn test_uhf_fock_reduces_to_rhf_fock_when_spins_equal() {
        // Phase Q2 (2026-07-16): if P^alpha = P^beta = P_rhf/2 (a spin-
        // unpolarized case), F^alpha and F^beta must both equal RHF's F --
        // the mathematical identity the closed-shell reduction test in
        // uhf.rs relies on, checked here at the Fock-build level directly.
        let n = 2;
        let h_core = vec![1.0, 0.3, 0.3, 2.0];
        let density_rhf = vec![1.0, 0.5, 0.5, 1.0];
        let eri = vec![0.1; 16];

        let fock_rhf = build_fock_matrix(&h_core, &density_rhf, &eri, n);

        let density_half: Vec<f64> = density_rhf.iter().map(|d| d / 2.0).collect();
        let (fock_alpha, fock_beta) =
            build_uhf_fock_matrices(&h_core, &density_half, &density_half, &eri, n);

        for i in 0..n * n {
            assert!(
                (fock_alpha[i] - fock_rhf[i]).abs() < 1e-12,
                "F_alpha[{i}]={} != F_rhf[{i}]={}",
                fock_alpha[i],
                fock_rhf[i]
            );
            assert!(
                (fock_beta[i] - fock_rhf[i]).abs() < 1e-12,
                "F_beta[{i}]={} != F_rhf[{i}]={}",
                fock_beta[i],
                fock_rhf[i]
            );
        }
    }

    #[test]
    fn test_uhf_electronic_energy_reduces_to_rhf_when_spins_equal() {
        let n = 2;
        let h_core = vec![1.0, 0.3, 0.3, 2.0];
        let density_rhf = vec![1.0, 0.5, 0.5, 1.0];
        let eri = vec![0.1; 16];

        let fock_rhf = build_fock_matrix(&h_core, &density_rhf, &eri, n);
        let e_rhf = electronic_energy(&density_rhf, &h_core, &fock_rhf, n);

        let density_half: Vec<f64> = density_rhf.iter().map(|d| d / 2.0).collect();
        let (fock_alpha, fock_beta) =
            build_uhf_fock_matrices(&h_core, &density_half, &density_half, &eri, n);
        let e_uhf = uhf_electronic_energy(
            &density_half,
            &density_half,
            &h_core,
            &fock_alpha,
            &fock_beta,
            n,
        );

        assert!(
            (e_uhf - e_rhf).abs() < 1e-12,
            "UHF energy {e_uhf} should equal RHF energy {e_rhf} for equal spin densities"
        );
    }

    #[test]
    fn test_direct_rhf_fock_matches_dense_fock_for_water() {
        // Phase Q3 (2026-07-16): the real correctness proof for direct SCF
        // -- a converged RHF density on a real molecule, fed through both
        // the dense-tensor path and the direct (on-demand) path, must give
        // the exact same Fock matrix.
        use crate::scf::density::build_density_matrix;
        use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let n = basis.n_basis();
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let density = build_density_matrix(
            &rhf.orbital_coefficients,
            n,
            rhf.n_independent,
            rhf.n_occupied,
        );

        let t_mat = crate::integrals::kinetic::kinetic_matrix(&basis.functions);
        let v_mat = crate::integrals::nuclear::nuclear_matrix(&basis.functions, &mol.atoms);
        let h_core: Vec<f64> = t_mat.iter().zip(v_mat.iter()).map(|(t, v)| t + v).collect();

        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let fock_dense = build_fock_matrix(&h_core, &density, &eri, n);

        let schwarz = compute_schwarz_bounds(&basis.functions);
        let fock_direct =
            build_fock_matrix_direct(&h_core, &density, &basis.functions, &schwarz, n);

        for i in 0..n * n {
            assert!(
                (fock_dense[i] - fock_direct[i]).abs() < 1e-10,
                "Fock[{i}]: dense={} direct={} differ",
                fock_dense[i],
                fock_direct[i]
            );
        }
    }

    #[test]
    fn test_direct_uhf_fock_matches_dense_fock_for_lithium() {
        use crate::scf::density::build_density_matrix_unrestricted;
        use crate::scf::uhf::{UhfConfig, unrestricted_hartree_fock};

        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(3, 0.0, 0.0, 0.0)], 0, 2);
        let basis = Sto3g::build(&mol);
        let n = basis.n_basis();
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());
        let p_alpha = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_alpha,
            n,
            uhf.n_independent,
            uhf.n_alpha,
        );
        let p_beta = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_beta,
            n,
            uhf.n_independent,
            uhf.n_beta,
        );

        let t_mat = crate::integrals::kinetic::kinetic_matrix(&basis.functions);
        let v_mat = crate::integrals::nuclear::nuclear_matrix(&basis.functions, &mol.atoms);
        let h_core: Vec<f64> = t_mat.iter().zip(v_mat.iter()).map(|(t, v)| t + v).collect();

        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let (fa_dense, fb_dense) = build_uhf_fock_matrices(&h_core, &p_alpha, &p_beta, &eri, n);

        let schwarz = compute_schwarz_bounds(&basis.functions);
        let (fa_direct, fb_direct) = build_uhf_fock_matrices_direct(
            &h_core,
            &p_alpha,
            &p_beta,
            &basis.functions,
            &schwarz,
            n,
        );

        for i in 0..n * n {
            assert!(
                (fa_dense[i] - fa_direct[i]).abs() < 1e-10,
                "F_alpha[{i}]: dense={} direct={} differ",
                fa_dense[i],
                fa_direct[i]
            );
            assert!(
                (fb_dense[i] - fb_direct[i]).abs() < 1e-10,
                "F_beta[{i}]: dense={} direct={} differ",
                fb_dense[i],
                fb_direct[i]
            );
        }
    }
}
