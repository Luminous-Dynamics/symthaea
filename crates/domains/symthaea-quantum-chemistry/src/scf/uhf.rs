// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unrestricted Hartree-Fock (UHF) self-consistent field solver.
//!
//! Solves two coupled Roothaan-Hall equations, one per spin channel:
//! F^α C^α = S C^α ε^α, F^β C^β = S C^β ε^β
//!
//! where F^α and F^β depend on BOTH spin densities through the shared
//! Coulomb term (see `scf::fock::build_uhf_fock_matrices`). Unlike RHF,
//! alpha and beta electrons occupy independent sets of spatial orbitals --
//! this is what makes UHF able to represent open-shell systems (radicals,
//! triplets, doublets, any `multiplicity != 1` molecule) that RHF cannot
//! (see `scf::rhf::restricted_hartree_fock`'s Phase Q0 guard).
//!
//! Phase Q2, 2026-07-16. Built as the direct fix for the gap Q0's
//! multiplicity guard found but didn't close. Reuses RHF's one-electron
//! integrals, ERI tensor, and canonical-orthogonalization machinery
//! directly (all spin-independent); density-build and Fock-build are
//! spin-specific (see `scf::density::build_density_matrix_unrestricted`,
//! `scf::fock::build_uhf_fock_matrices`) since RHF's versions hardcode a
//! factor-of-2 double-occupancy convention that doesn't apply per spin
//! channel.
//!
//! References:
//! - Szabo & Ostlund (1996). *Modern Quantum Chemistry*. Chapter 3.
//! - Pople & Nesbet (1954). J. Chem. Phys. 22, 571 (UHF).

use crate::basis::BasisSet;
use crate::constants::{MAX_SCF_ITERATIONS, SCF_DENSITY_THRESHOLD, SCF_ENERGY_THRESHOLD};
use crate::integrals::eri::{compute_eri_tensor, compute_schwarz_bounds};
use crate::integrals::kinetic::kinetic_matrix;
use crate::integrals::nuclear::nuclear_matrix;
use crate::integrals::overlap::overlap_matrix;
use crate::molecule::Molecule;
use crate::scf::density::{build_density_matrix_unrestricted, density_rms_change};
use crate::scf::diis::Diis;
use crate::scf::fock::{
    build_uhf_fock_matrices, build_uhf_fock_matrices_direct, uhf_electronic_energy,
};
use crate::scf::generalized_eigen::{canonical_orthogonalization, solve_generalized_eigen};

/// Configuration for the UHF solver. Same shape as `RhfConfig`.
#[derive(Debug, Clone)]
pub struct UhfConfig {
    pub max_iterations: usize,
    pub energy_convergence: f64,
    pub density_convergence: f64,
    pub use_diis: bool,
    /// Direct SCF (Phase Q3, 2026-07-16) -- see `RhfConfig::direct`'s doc
    /// comment for the memory/compute tradeoff. Default `false`.
    pub direct: bool,
}

impl Default for UhfConfig {
    fn default() -> Self {
        Self {
            max_iterations: MAX_SCF_ITERATIONS,
            energy_convergence: SCF_ENERGY_THRESHOLD,
            density_convergence: SCF_DENSITY_THRESHOLD,
            use_diis: true,
            direct: false,
        }
    }
}

/// Result of a UHF calculation.
#[derive(Debug, Clone)]
pub struct UhfResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion: f64,
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    /// C[μ * n_mo + i], same layout as `RhfResult::orbital_coefficients`.
    pub orbital_coefficients_alpha: Vec<f64>,
    pub orbital_coefficients_beta: Vec<f64>,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub n_iterations: usize,
    pub converged: bool,
    pub n_basis: usize,
    pub n_independent: usize,
    /// S(S+1) from the molecule's declared multiplicity.
    pub spin_squared_expected: f64,
    /// <S^2> actually computed from the converged alpha/beta orbitals --
    /// equals `spin_squared_expected` only for a spin-pure solution; real
    /// UHF solutions are usually somewhat spin-contaminated (higher than
    /// expected). See the module doc and `compute_spin_squared` below.
    pub spin_squared_computed: f64,
}

/// Split `n_electrons`/`multiplicity` into (n_alpha, n_beta).
///
/// `multiplicity = 2S + 1`, so `n_alpha - n_beta = multiplicity - 1` and
/// `n_alpha + n_beta = n_electrons`. Panics (matching this crate's
/// established panic-based fail-closed convention -- see Q0's
/// `restricted_hartree_fock`/`Molecule::n_electrons` guards) if the
/// combination is inconsistent: a non-integer split (e.g. even electron
/// count with even multiplicity), or `n_beta` would be negative (an
/// unreachable multiplicity for that electron count). `n_alpha` is always
/// the majority-spin count by this crate's convention.
fn alpha_beta_split(n_electrons: usize, multiplicity: u32) -> (usize, usize) {
    let n_electrons = n_electrons as i64;
    let unpaired = multiplicity as i64 - 1;
    assert!(
        unpaired >= 0,
        "invalid multiplicity {multiplicity}: must be >= 1"
    );
    let sum_minus_diff = n_electrons - unpaired;
    assert!(
        sum_minus_diff >= 0 && sum_minus_diff % 2 == 0,
        "inconsistent electronic state: {n_electrons} electrons cannot have multiplicity \
         {multiplicity} (n_alpha - n_beta = {unpaired}, but that requires an electron count \
         with the same parity)"
    );
    let n_beta = sum_minus_diff / 2;
    let n_alpha = n_beta + unpaired;
    (n_alpha as usize, n_beta as usize)
}

/// Cross spin-orbital overlap <ψ_i^α|ψ_j^β> = Σ_μν C^α_μi S_μν C^β_νj,
/// then <S²>_computed = S(S+1) + n_beta - Σ_ij |<ψ_i^α|ψ_j^β>|² (standard
/// UHF spin-contamination formula, e.g. Szabo & Ostlund).
fn compute_spin_squared(
    s_mat: &[f64],
    c_alpha: &[f64],
    c_beta: &[f64],
    n_basis: usize,
    n_mo: usize,
    n_alpha: usize,
    n_beta: usize,
) -> f64 {
    let s_exact = {
        let s = (n_alpha as f64 - n_beta as f64) / 2.0;
        s * (s + 1.0)
    };

    let mut overlap_sum_sq = 0.0;
    for i in 0..n_alpha {
        for j in 0..n_beta {
            let mut overlap_ij = 0.0;
            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    overlap_ij +=
                        c_alpha[mu * n_mo + i] * s_mat[mu * n_basis + nu] * c_beta[nu * n_mo + j];
                }
            }
            overlap_sum_sq += overlap_ij * overlap_ij;
        }
    }

    s_exact + n_beta as f64 - overlap_sum_sq
}

/// Run an Unrestricted Hartree-Fock calculation.
///
/// Works for any `multiplicity` (including 1 -- see the closed-shell
/// reduction tests below, which verify UHF gives the exact same energy as
/// RHF when alpha and beta densities are forced equal). Panics on an
/// inconsistent electron-count/multiplicity combination (see
/// `alpha_beta_split`).
pub fn unrestricted_hartree_fock(
    molecule: &Molecule,
    basis: &BasisSet,
    config: &UhfConfig,
) -> UhfResult {
    let n = basis.n_basis();
    let (n_alpha, n_beta) = alpha_beta_split(molecule.n_electrons(), molecule.multiplicity);
    let v_nn = molecule.nuclear_repulsion_energy();

    // Step 1: One-electron integrals (spin-independent).
    let s_mat = overlap_matrix(&basis.functions);
    let t_mat = kinetic_matrix(&basis.functions);
    let v_mat = nuclear_matrix(&basis.functions, &molecule.atoms);
    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Step 2: Two-electron integrals (spin-independent) -- dense-tensor
    // (default) or direct (Phase Q3, 2026-07-16) mode.
    let (eri, _eri_computed, _eri_screened) = if config.direct {
        (Vec::new(), 0, 0)
    } else {
        compute_eri_tensor(&basis.functions)
    };
    let schwarz = if config.direct {
        compute_schwarz_bounds(&basis.functions)
    } else {
        Vec::new()
    };
    let build_fock = |h_core: &[f64], p_a: &[f64], p_b: &[f64]| -> (Vec<f64>, Vec<f64>) {
        if config.direct {
            build_uhf_fock_matrices_direct(h_core, p_a, p_b, &basis.functions, &schwarz, n)
        } else {
            build_uhf_fock_matrices(h_core, p_a, p_b, &eri, n)
        }
    };

    // Step 3: Canonical orthogonalization (spin-independent -- S doesn't
    // depend on spin).
    let (x_mat, n_ind, _n_disc) = canonical_orthogonalization(&s_mat, n);

    // Step 4: Initial guess -- diagonalize H_core for both spins (core
    // guess: identical starting point, SCF breaks the symmetry).
    let initial = solve_generalized_eigen(&h_core, &x_mat, n, n_ind);
    let mut c_alpha = initial.coefficients.clone();
    let mut c_beta = initial.coefficients;
    let mut eps_alpha = initial.eigenvalues.clone();
    let mut eps_beta = initial.eigenvalues;

    let mut p_alpha = build_density_matrix_unrestricted(&c_alpha, n, n_ind, n_alpha);
    let mut p_beta = build_density_matrix_unrestricted(&c_beta, n, n_ind, n_beta);

    let mut energy_old = 0.0;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut diis_alpha = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };
    let mut diis_beta = if config.use_diis {
        Some(Diis::new(n))
    } else {
        None
    };

    for iter in 0..config.max_iterations {
        n_iterations = iter + 1;

        let (mut fock_alpha, mut fock_beta) = build_fock(&h_core, &p_alpha, &p_beta);

        if let Some(ref mut d) = diis_alpha {
            fock_alpha = d.extrapolate(&fock_alpha, &p_alpha, &s_mat);
        }
        if let Some(ref mut d) = diis_beta {
            fock_beta = d.extrapolate(&fock_beta, &p_beta, &s_mat);
        }

        let e_elec = uhf_electronic_energy(&p_alpha, &p_beta, &h_core, &fock_alpha, &fock_beta, n);
        let e_total = e_elec + v_nn;

        let result_alpha = solve_generalized_eigen(&fock_alpha, &x_mat, n, n_ind);
        c_alpha = result_alpha.coefficients;
        eps_alpha = result_alpha.eigenvalues;

        let result_beta = solve_generalized_eigen(&fock_beta, &x_mat, n, n_ind);
        c_beta = result_beta.coefficients;
        eps_beta = result_beta.eigenvalues;

        let p_alpha_new = build_density_matrix_unrestricted(&c_alpha, n, n_ind, n_alpha);
        let p_beta_new = build_density_matrix_unrestricted(&c_beta, n, n_ind, n_beta);

        let d_rms_alpha = density_rms_change(&p_alpha_new, &p_alpha, n);
        let d_rms_beta = density_rms_change(&p_beta_new, &p_beta, n);
        let de = (e_total - energy_old).abs();

        p_alpha = p_alpha_new;
        p_beta = p_beta_new;
        energy_old = e_total;

        if de < config.energy_convergence
            && d_rms_alpha < config.density_convergence
            && d_rms_beta < config.density_convergence
            && iter > 0
        {
            converged = true;
            break;
        }
    }

    let (fock_alpha_final, fock_beta_final) = build_fock(&h_core, &p_alpha, &p_beta);
    let e_elec = uhf_electronic_energy(
        &p_alpha,
        &p_beta,
        &h_core,
        &fock_alpha_final,
        &fock_beta_final,
        n,
    );

    let spin_squared_expected = {
        let s = (n_alpha as f64 - n_beta as f64) / 2.0;
        s * (s + 1.0)
    };
    let spin_squared_computed =
        compute_spin_squared(&s_mat, &c_alpha, &c_beta, n, n_ind, n_alpha, n_beta);

    UhfResult {
        total_energy: e_elec + v_nn,
        electronic_energy: e_elec,
        nuclear_repulsion: v_nn,
        orbital_energies_alpha: eps_alpha,
        orbital_energies_beta: eps_beta,
        orbital_coefficients_alpha: c_alpha,
        orbital_coefficients_beta: c_beta,
        n_alpha,
        n_beta,
        n_iterations,
        converged,
        n_basis: n,
        n_independent: n_ind,
        spin_squared_expected,
        spin_squared_computed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_alpha_beta_split_singlet() {
        assert_eq!(alpha_beta_split(10, 1), (5, 5));
    }

    #[test]
    fn test_alpha_beta_split_doublet() {
        // Li atom: 3 electrons, doublet ground state.
        assert_eq!(alpha_beta_split(3, 2), (2, 1));
    }

    #[test]
    fn test_alpha_beta_split_triplet() {
        // O atom in its triplet ground state: 8 electrons.
        assert_eq!(alpha_beta_split(8, 3), (5, 3));
    }

    #[test]
    #[should_panic(expected = "inconsistent electronic state")]
    fn test_alpha_beta_split_rejects_impossible_combination() {
        // 10 electrons (even) can't have multiplicity 2 (needs odd n_alpha-n_beta parity).
        alpha_beta_split(10, 2);
    }

    #[test]
    fn test_uhf_h2_matches_rhf_for_closed_shell() {
        // Phase Q2's primary correctness test: no external reference needed
        // -- for a genuinely closed-shell molecule, UHF's alpha and beta
        // densities converge to be identical, and the equations
        // mathematically collapse to RHF's. H2/STO-3G's RHF energy is
        // already validated against Szabo & Ostlund in rhf.rs's own tests.
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged);
        assert_eq!(uhf.n_alpha, 1);
        assert_eq!(uhf.n_beta, 1);
        assert!(
            (uhf.total_energy - rhf.total_energy).abs() < 1e-8,
            "UHF energy {} should match RHF energy {} for closed-shell H2",
            uhf.total_energy,
            rhf.total_energy
        );
        // Spin-pure singlet: <S^2> should be exactly 0.
        assert!(
            uhf.spin_squared_computed.abs() < 1e-8,
            "closed-shell UHF should have <S^2>=0, got {}",
            uhf.spin_squared_computed
        );
    }

    #[test]
    fn test_uhf_water_matches_rhf_for_closed_shell() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged);
        assert!(
            (uhf.total_energy - rhf.total_energy).abs() < 1e-8,
            "UHF energy {} should match RHF energy {} for closed-shell water",
            uhf.total_energy,
            rhf.total_energy
        );
    }

    #[test]
    fn test_uhf_lithium_atom_open_shell() {
        // Li atom: 3 electrons, doublet ground state (1s^2 2s^1),
        // n_alpha=2, n_beta=1 -- a real open-shell system RHF now refuses
        // (Phase Q0) and this function can actually compute.
        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(3, 0.0, 0.0, 0.0)], 0, 2);
        let basis = Sto3g::build(&mol);
        let uhf = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());

        assert!(uhf.converged, "Li atom UHF should converge");
        assert_eq!(uhf.n_alpha, 2);
        assert_eq!(uhf.n_beta, 1);
        assert!(uhf.total_energy < 0.0, "Li atom energy should be negative");

        // Internal-consistency invariants (no external reference needed --
        // same discipline as Q0's HF-discrepancy investigation): Tr[P*S]
        // must equal the electron count for each spin.
        let p_alpha = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_alpha,
            uhf.n_basis,
            uhf.n_independent,
            uhf.n_alpha,
        );
        let p_beta = build_density_matrix_unrestricted(
            &uhf.orbital_coefficients_beta,
            uhf.n_basis,
            uhf.n_independent,
            uhf.n_beta,
        );
        let s_mat = overlap_matrix(&basis.functions);
        let n = uhf.n_basis;
        let mut tr_ps_alpha = 0.0;
        let mut tr_ps_beta = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ps_alpha += p_alpha[i * n + j] * s_mat[j * n + i];
                tr_ps_beta += p_beta[i * n + j] * s_mat[j * n + i];
            }
        }
        assert!(
            (tr_ps_alpha - uhf.n_alpha as f64).abs() < 1e-6,
            "Tr[P_alpha*S]={tr_ps_alpha}, expected {}",
            uhf.n_alpha
        );
        assert!(
            (tr_ps_beta - uhf.n_beta as f64).abs() < 1e-6,
            "Tr[P_beta*S]={tr_ps_beta}, expected {}",
            uhf.n_beta
        );

        // Li atom is a well-behaved, textbook-low-contamination doublet --
        // <S^2>_computed should be close to the exact 0.75 (S=1/2). Measured
        // 2026-07-16: 0.750000 to 6 decimals -- Li/STO-3G's 1s core (beta)
        // and 2s valence (unpaired alpha) are well-separated, so this
        // system is essentially perfectly spin-pure. Tolerance kept looser
        // than the measured precision to avoid a brittle exact-equality
        // assertion across platforms/toolchains.
        assert_eq!(uhf.spin_squared_expected, 0.75);
        assert!(
            (uhf.spin_squared_computed - 0.75).abs() < 1e-4,
            "Li atom spin contamination unexpectedly high: <S^2>={}",
            uhf.spin_squared_computed
        );
    }

    #[test]
    fn test_uhf_direct_mode_matches_dense_mode_end_to_end() {
        // Phase Q3 (2026-07-16): full end-to-end direct-vs-dense identity
        // for UHF, using the real open-shell Li atom test case.
        let mol = Molecule::with_charge(vec![crate::molecule::Atom::new(3, 0.0, 0.0, 0.0)], 0, 2);
        let basis = Sto3g::build(&mol);

        let dense = unrestricted_hartree_fock(&mol, &basis, &UhfConfig::default());
        let direct = unrestricted_hartree_fock(
            &mol,
            &basis,
            &UhfConfig {
                direct: true,
                ..Default::default()
            },
        );

        assert!(dense.converged && direct.converged);
        assert!(
            (dense.total_energy - direct.total_energy).abs() < 1e-8,
            "dense={:.10} direct={:.10} should match",
            dense.total_energy,
            direct.total_energy
        );
        assert!(
            (dense.spin_squared_computed - direct.spin_squared_computed).abs() < 1e-6,
            "spin contamination should match between modes"
        );
    }
}
