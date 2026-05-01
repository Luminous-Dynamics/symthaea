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

#[cfg(test)]
mod tests {
    use super::*;

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
}
