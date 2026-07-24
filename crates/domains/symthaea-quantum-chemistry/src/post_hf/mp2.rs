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
//!
//! ## Frozen core / SCS-MP2 (Phase Q5b, 2026-07-17)
//!
//! `mp2_correlation_energy_frozen_core` and `scs_mp2_correlation_energy` are
//! purely additive extensions -- `mp2_correlation_energy` itself and its
//! existing caller are unchanged. `standard_frozen_core_count`'s table is
//! verified in this crate's own tests only for Z≤9 (H/He/Li/C/N/O/F, this
//! crate's entire currently-exercised molecule set); the Na-Ar row is
//! included for table completeness but not independently tested here.

use crate::molecule::Molecule;
use crate::scf::rhf::RhfResult;

/// Standard frozen-core spatial-orbital count for element `z`: the usual
/// "freeze through the previous noble gas" convention (0 for H/He, 1 for
/// Li-Ne, 5 for Na-Ar). See the module doc's scope note -- verified here
/// only for Z≤9.
pub fn standard_frozen_core_count(z: u8) -> usize {
    match z {
        1..=2 => 0,
        3..=10 => 1,
        11..=18 => 5,
        _ => 0,
    }
}

/// Sum of `standard_frozen_core_count` over every atom in a molecule.
pub fn total_frozen_core(molecule: &Molecule) -> usize {
    molecule
        .atoms
        .iter()
        .map(|a| standard_frozen_core_count(a.atomic_number))
        .sum()
}

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
                            * eri_3[mu * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b];
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

    let eri_idx = |i: usize, a: usize, j: usize, b: usize| -> f64 {
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

/// Frozen-core MP2 correlation energy (Phase Q5b, 2026-07-17): identical to
/// `mp2_correlation_energy`, except the `i`/`j` summation starts at
/// `n_frozen` instead of `0` -- core orbitals are still AO→MO transformed
/// (kept simple: same transform as the full calculation, no `occ_start`
/// offset threaded through it) but excluded from the correlation sum.
/// `n_frozen == 0` reduces exactly to `mp2_correlation_energy`'s result
/// (verified in tests). Pass `total_frozen_core(molecule)` for the standard
/// convention, or any smaller value.
pub fn mp2_correlation_energy_frozen_core(
    rhf: &RhfResult,
    eri_ao: &[f64],
    n_frozen: usize,
) -> Mp2Result {
    let n = rhf.n_basis;
    let n_mo = rhf.n_independent;
    let n_occ = rhf.n_occupied;
    let n_vir = n_mo - n_occ;
    let c = &rhf.orbital_coefficients;
    let eps = &rhf.orbital_energies;
    let n3 = n * n * n;
    let n2 = n * n;

    let mut eri_1 = vec![0.0; n * n * n * n_vir];
    for mu in 0..n {
        for nu in 0..n {
            for lam in 0..n {
                for b in 0..n_vir {
                    let b_mo = n_occ + b;
                    let mut val = 0.0;
                    for sig in 0..n {
                        val += c[sig * n_mo + b_mo] * eri_ao[mu * n3 + nu * n2 + lam * n + sig];
                    }
                    eri_1[mu * n * n * n_vir + nu * n * n_vir + lam * n_vir + b] = val;
                }
            }
        }
    }
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
    let mut eri_mo = vec![0.0; n_occ * n_vir * n_occ * n_vir];
    for i in 0..n_occ {
        for a in 0..n_vir {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for mu in 0..n {
                        val += c[mu * n_mo + i]
                            * eri_3[mu * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b];
                    }
                    eri_mo[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b] = val;
                }
            }
        }
    }

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    let eri_idx = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        eri_mo[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b]
    };

    for i in n_frozen..n_occ {
        for j in n_frozen..n_occ {
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let iajb = eri_idx(i, a, j, b);
                    let ibja = eri_idx(i, b, j, a);
                    let denom = eps[i] + eps[j] - eps[n_occ + a] - eps[n_occ + b];

                    if denom.abs() < 1e-14 {
                        continue;
                    }

                    e_os += iajb * iajb / denom;
                    e_ss += iajb * (iajb - ibja) / denom;
                }
            }
        }
    }

    let correlation = e_os + e_ss;

    Mp2Result {
        correlation_energy: correlation,
        hf_energy: rhf.total_energy,
        total_energy: rhf.total_energy + correlation,
        same_spin: e_ss,
        opposite_spin: e_os,
    }
}

/// Spin-Component-Scaled MP2 (Phase Q5b, 2026-07-17; Grimme, 2003, J. Chem.
/// Phys. 118, 9095): `E_SCS-MP2 = 1.2 × E_OS + (1/3) × E_SS`, using the
/// standard published coefficients. Trivial linear combination of an
/// already-computed, already-tested `Mp2Result` -- no new integrals.
pub fn scs_mp2_correlation_energy(mp2: &Mp2Result) -> f64 {
    1.2 * mp2.opposite_spin + (1.0 / 3.0) * mp2.same_spin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::integrals::eri::compute_eri_tensor;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

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

    #[test]
    fn test_standard_frozen_core_count_z_le_9() {
        // Phase Q5b (2026-07-17): the only table rows this crate's own tests
        // exercise -- H/He (0 core) and Li-F (1 core).
        assert_eq!(standard_frozen_core_count(1), 0); // H
        assert_eq!(standard_frozen_core_count(2), 0); // He
        assert_eq!(standard_frozen_core_count(3), 1); // Li
        assert_eq!(standard_frozen_core_count(6), 1); // C
        assert_eq!(standard_frozen_core_count(7), 1); // N
        assert_eq!(standard_frozen_core_count(8), 1); // O
        assert_eq!(standard_frozen_core_count(9), 1); // F
    }

    #[test]
    fn test_total_frozen_core_lih() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(3, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(1, 0.0, 0.0, 3.015),
        ]);
        assert_eq!(total_frozen_core(&mol), 1); // Li's 1s core, H has none
    }

    /// Exact zero-frozen identity (Phase Q5b): `n_frozen=0` must reduce
    /// exactly to `mp2_correlation_energy`'s existing result -- a real
    /// structural identity on the new code path, not a tautology.
    #[test]
    fn test_frozen_core_zero_matches_full_mp2_exactly() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let full = mp2_correlation_energy(&rhf, &eri);
        let frozen_zero = mp2_correlation_energy_frozen_core(&rhf, &eri, 0);

        assert_eq!(full.correlation_energy, frozen_zero.correlation_energy);
        assert_eq!(full.same_spin, frozen_zero.same_spin);
        assert_eq!(full.opposite_spin, frozen_zero.opposite_spin);
    }

    /// Exact hand-derived identity for H2/STO-3G (Phase Q5b, mirrors Q5's
    /// CIS 1x1 check): n_occ=1,n_vir=1 means the only term has i=j=0,a=b=0,
    /// so (ib|ja) is literally the same lookup as (ia|jb) -- making the
    /// same-spin contribution `(ia|jb)*((ia|jb)-(ib|ja))` exactly zero by
    /// construction, not approximately. A real, provable "tight benchmark."
    #[test]
    fn test_mp2_h2_sto3g_same_spin_exactly_zero() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        assert_eq!(rhf.n_occupied, 1);
        assert_eq!(rhf.n_independent - rhf.n_occupied, 1);

        let mp2 = mp2_correlation_energy(&rhf, &eri);
        assert_eq!(mp2.same_spin, 0.0);
        assert_eq!(mp2.correlation_energy, mp2.opposite_spin);
        assert!(mp2.opposite_spin < 0.0);
    }

    /// Provable frozen-core monotonicity inequality (Phase Q5b): converged
    /// canonical HF always has every occupied orbital energy below every
    /// virtual orbital energy, so D=eps_i+eps_j-eps_a-eps_b < 0 always, and
    /// the OS term (ia|jb)^2/D is non-positive for every individual
    /// i,j,a,b. Restricting i,j to a strict subset (freezing the core) can
    /// only remove non-positive terms, so frozen-core OS must be >= full OS
    /// (both negative; frozen-core strictly less negative) -- a real,
    /// derivable inequality, not an empirical hunch.
    #[test]
    fn test_lih_frozen_core_os_less_negative_than_full() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(3, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(1, 0.0, 0.0, 3.015),
        ]);
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        assert_eq!(rhf.n_occupied, 2);

        let full = mp2_correlation_energy(&rhf, &eri);
        let frozen = mp2_correlation_energy_frozen_core(&rhf, &eri, 1);

        assert!(full.opposite_spin < 0.0);
        assert!(frozen.opposite_spin < 0.0);
        assert!(
            frozen.opposite_spin >= full.opposite_spin,
            "frozen-core OS ({}) should be >= (less negative than) full OS ({})",
            frozen.opposite_spin,
            full.opposite_spin
        );
    }

    /// SCS-MP2 correctness (Phase Q5b): independently recomputes the
    /// standard linear combination and checks it matches exactly -- catches
    /// a swapped-coefficient bug, not a tautology.
    #[test]
    fn test_scs_mp2_matches_standard_coefficients() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let mp2 = mp2_correlation_energy(&rhf, &eri);
        let scs = scs_mp2_correlation_energy(&mp2);
        let expected = 1.2 * mp2.opposite_spin + (1.0 / 3.0) * mp2.same_spin;

        assert_eq!(scs, expected);
    }
}
