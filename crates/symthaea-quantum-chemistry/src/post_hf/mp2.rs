// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MP2 (Møller-Plesset second-order perturbation theory) correlation energy.
//!
//! MP2 captures the leading electron correlation correction beyond HF.
//! E_MP2 = Σ_{i<j,a<b} |(ia|jb) - (ib|ja)|² / (ε_i + ε_j - ε_a - ε_b)
//!
//! where i,j are occupied and a,b are virtual orbital indices.
//!
//! Reference: Møller & Plesset (1934). Phys. Rev. 46, 618.

use crate::scf::rhf::RhfResult;

/// Result of an MP2 calculation.
#[derive(Debug, Clone)]
pub struct Mp2Result {
    /// MP2 correlation energy (always negative)
    pub correlation_energy: f64,
    /// HF energy (from the preceding RHF calculation)
    pub hf_energy: f64,
    /// Total MP2 energy = HF + correlation
    pub total_energy: f64,
    /// Same-spin (SS) contribution
    pub same_spin: f64,
    /// Opposite-spin (OS) contribution
    pub opposite_spin: f64,
}

/// Compute the MP2 correlation energy from an RHF result.
///
/// Requires the ERI tensor in the AO basis and the MO coefficients + energies
/// from the converged HF calculation.
///
/// The ERIs are transformed from AO to MO basis on-the-fly using the
/// MO coefficients (4-index transformation).
pub fn mp2_correlation_energy(rhf: &RhfResult, eri_ao: &[f64]) -> Mp2Result {
    let n = rhf.n_basis;
    let n_mo = rhf.n_independent;
    let n_occ = rhf.n_occupied;
    let n_vir = n_mo - n_occ;
    let c = &rhf.orbital_coefficients;
    let eps = &rhf.orbital_energies;

    // Transform ERIs from AO to MO basis: (pq|rs) = Σ_{μνλσ} C_μp C_νq (μν|λσ) C_λr C_σs
    // For MP2 we only need (ia|jb) where i,j=occupied and a,b=virtual.
    // Do a partial 4-index transformation for efficiency.

    // Half-transform: (μν|jb) = Σ_σ C_σb Σ_λ C_λj (μν|λσ)
    // Then: (ia|jb) = Σ_ν C_νa Σ_μ C_μi (μν|jb)

    let n3 = n * n * n;
    let n2 = n * n;

    // Step 1: First quarter transform on index 4 (σ → b)
    // (μν|λb) = Σ_σ C_σb (μν|λσ)  for b in virtual orbitals
    let mut eri_1 = vec![0.0; n * n * n * n_vir];
    for mu in 0..n {
        for nu in 0..n {
            for lam in 0..n {
                for b in 0..n_vir {
                    let b_mo = n_occ + b; // virtual orbital index
                    let mut val = 0.0;
                    for sig in 0..n {
                        val += c[sig * n_mo + b_mo] * eri_ao[mu * n3 + nu * n2 + lam * n + sig];
                    }
                    eri_1[mu * n * n * n_vir + nu * n * n_vir + lam * n_vir + b] = val;
                }
            }
        }
    }

    // Step 2: Second quarter transform on index 3 (λ → j)
    // (μν|jb) = Σ_λ C_λj (μν|λb)  for j in occupied orbitals
    let mut eri_2 = vec![0.0; n * n * n_occ * n_vir];
    for mu in 0..n {
        for nu in 0..n {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for lam in 0..n {
                        val += c[lam * n_mo + j]
                            * eri_1[mu * n * n * n_vir + nu * n * n_vir + lam * n_vir + b];
                    }
                    eri_2[mu * n * n_occ * n_vir + nu * n_occ * n_vir + j * n_vir + b] = val;
                }
            }
        }
    }

    // Step 3: Third quarter transform on index 2 (ν → a)
    // (μa|jb) = Σ_ν C_νa (μν|jb)  for a in virtual orbitals
    let mut eri_3 = vec![0.0; n * n_vir * n_occ * n_vir];
    for mu in 0..n {
        for a in 0..n_vir {
            let a_mo = n_occ + a;
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for nu in 0..n {
                        val += c[nu * n_mo + a_mo]
                            * eri_2[mu * n * n_occ * n_vir + nu * n_occ * n_vir + j * n_vir + b];
                    }
                    eri_3[mu * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b] = val;
                }
            }
        }
    }

    // Step 4: Fourth quarter transform on index 1 (μ → i)
    // (ia|jb) = Σ_μ C_μi (μa|jb)  for i in occupied orbitals
    let mut eri_mo = vec![0.0; n_occ * n_vir * n_occ * n_vir];
    for i in 0..n_occ {
        for a in 0..n_vir {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for mu in 0..n {
                        val += c[mu * n_mo + i]
                            * eri_3
                                [mu * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b];
                    }
                    eri_mo[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b] = val;
                }
            }
        }
    }

    // Step 5: Compute MP2 energy
    // E_MP2 = Σ_{i<j, a<b} |t_ij^ab|² × D_ij^ab
    // where t_ij^ab = (ia|jb) - (ib|ja)  (antisymmetrized)
    // and D_ij^ab = 1/(ε_i + ε_j - ε_a - ε_b)
    //
    // Equivalently (summing over all i,j,a,b with factor):
    // E_MP2 = Σ_{ijab} (ia|jb) [2(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)

    let mut e_os = 0.0; // opposite-spin
    let mut e_ss = 0.0; // same-spin

    let eri_idx =
        |i: usize, a: usize, j: usize, b: usize| -> f64 {
            eri_mo[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b]
        };

    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let iajb = eri_idx(i, a, j, b);
                    let ibja = eri_idx(i, b, j, a);
                    let denom = eps[i] + eps[j] - eps[n_occ + a] - eps[n_occ + b];

                    if denom.abs() < 1e-14 {
                        continue;
                    }

                    // OS: (ia|jb)² / D
                    e_os += iajb * iajb / denom;

                    // SS: (ia|jb)((ia|jb) - (ib|ja)) / D
                    e_ss += iajb * (iajb - ibja) / denom;
                }
            }
        }
    }

    // Total: E_MP2 = E_OS + E_SS (with proper counting)
    let correlation = e_os + e_ss;

    Mp2Result {
        correlation_energy: correlation,
        hf_energy: rhf.total_energy,
        total_energy: rhf.total_energy + correlation,
        same_spin: e_ss,
        opposite_spin: e_os,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::sto3g::Sto3g;
    use crate::basis::BasisSetProvider;
    use crate::integrals::eri::compute_eri_tensor;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{restricted_hartree_fock, RhfConfig};

    #[test]
    fn test_mp2_h2() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let mp2 = mp2_correlation_energy(&rhf, &eri);

        // MP2 correlation should be negative
        assert!(
            mp2.correlation_energy < 0.0,
            "MP2 correlation should be negative: {}",
            mp2.correlation_energy
        );

        // For H2 STO-3G, correlation is small (~-0.01 to -0.05 Hartree)
        assert!(
            mp2.correlation_energy.abs() < 0.1,
            "H2 MP2 correlation = {:.6}, should be small",
            mp2.correlation_energy
        );

        // Total should be lower than HF
        assert!(
            mp2.total_energy < mp2.hf_energy,
            "MP2 total ({:.6}) should be lower than HF ({:.6})",
            mp2.total_energy,
            mp2.hf_energy
        );
    }

    #[test]
    fn test_mp2_water() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let mp2 = mp2_correlation_energy(&rhf, &eri);

        assert!(mp2.correlation_energy < 0.0);
        // Water STO-3G MP2 correlation ≈ -0.035 Hartree
        assert!(
            mp2.correlation_energy.abs() < 0.5,
            "H2O MP2 correlation = {:.6}",
            mp2.correlation_energy
        );
    }

    #[test]
    fn test_mp2_spin_components() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let mp2 = mp2_correlation_energy(&rhf, &eri);

        // Same-spin + opposite-spin = total correlation
        assert!(
            (mp2.same_spin + mp2.opposite_spin - mp2.correlation_energy).abs() < 1e-12,
            "SS({:.6}) + OS({:.6}) != total({:.6})",
            mp2.same_spin,
            mp2.opposite_spin,
            mp2.correlation_energy
        );
    }
}
