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
//!
//! ## Status (Phase Q0, 2026-07-16): experimental, not production CCD
//!
//! The theory above is accurately described, but `coupled_cluster_doubles`'s
//! implementation has three real defects, found via direct code review AND
//! empirically confirmed by adding real tests that call the function itself
//! across several real molecules (the previous test suite only checked that
//! a *separate* MP2 function's correlation energy was negative -- it never
//! exercised this function at all):
//!
//! 1. **For any closed-shell system with exactly one occupied spatial
//!    orbital (n_occ=1: H2, HeH+, He...), this function returns EXACTLY
//!    zero correlation energy, always.** With only one occupied index
//!    value, every term forces i=j, and the antisymmetrized quantity
//!    `v = eri(i,a,j,b) - eri(i,b,j,a)` collapses to `eri(0,a,0,b) -
//!    eri(0,b,0,a)`, which is identically zero via the standard real-ERI
//!    permutational symmetry `(pq|rs) = (rs|pq)`. Confirmed via
//!    `test_ccd_h2_sto3g_returns_exact_zero_not_the_old_tests_implicit_mp2_value`.
//!    This also means the OLD test's comment ("H2 STO-3G ... CCD = MP2, no
//!    quadratic term") was doubly wrong -- it never called this function,
//!    and the claim is false regardless: real MP2 for H2/STO-3G is a
//!    genuine nonzero negative number, not zero.
//! 2. **The quadratic (t2 x t2) term's ERI lookup is architecturally
//!    wrong, not just numerically hacky, and panics on real molecules.**
//!    It calls `eri(n_occ + a - n_occ, c, n_occ + b - n_occ, d)` -- i.e.
//!    `eri(a, c, b, d)` -- passing *virtual*-orbital-local indices (`a`,
//!    `b`, ranging `0..n_vir`) into `eri()`'s first and third argument
//!    slots, which the `eri_mo` tensor's layout only has room for
//!    *occupied* indices (`0..n_occ`). This is gated behind `if
//!    t_ijcd.abs() > 1e-15`, and defect #1 above means that guard is
//!    always false for n_occ=1 systems -- so the bug is invisible on the
//!    trivial cases this crate's other tests happen to use. But for any
//!    system with n_occ>1 (so t2 is genuinely nonzero) AND n_vir>n_occ
//!    (so the loop's `a` index can exceed n_occ), it panics with an
//!    out-of-bounds index -- empirically confirmed for LiH/STO-3G
//!    (n_occ=2, n_vir=4) in
//!    `test_ccd_lih_sto3g_quadratic_term_indexing_panics`. The physically
//!    correct quadratic term needs a virtual-virtual-virtual-virtual
//!    `(ac|bd)` integral block that is never computed or passed into this
//!    function at all.
//! 3. **Even where the indexing happens not to go out of bounds** (e.g.
//!    water/STO-3G, where n_vir <= n_occ keeps it in range by
//!    coincidence, not correctness), the value that same expression
//!    computes is run through `.abs().min(0.1)` -- this doesn't just clamp
//!    magnitude for stability (the `// Stabilize` comment's intent), it
//!    discards the integral's *sign* before the clamp, making destructive
//!    interference between amplitudes structurally impossible to
//!    represent.
//!
//! Do not treat this module's output as a validated CCD correlation
//! energy. A real fix needs the correct `(ac|bd)` virtual-block
//! transformation (a genuine additional AO->MO transformation step, not a
//! one-line indexing patch) and removing the sign-destroying `.abs()` --
//! tracked as part of Phase Q5 in
//! `QUANTUM_CHEMISTRY_COMPLETENESS_ROADMAP_2026-07-16.md`, not attempted
//! in this pass.

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
    use crate::basis::basis_631g::Basis631G;
    use crate::basis::sto3g::Sto3g;
    use crate::integrals::eri::compute_eri_tensor;
    use crate::molecule::Molecule;
    use crate::post_hf::mp2::mp2_correlation_energy;
    use crate::scf::rhf::{RhfConfig, RhfResult, restricted_hartree_fock};

    /// AO->MO (ia|jb) transform, identical algorithm/layout to
    /// `post_hf::mp2::mp2_correlation_energy`'s internal transform (that
    /// function doesn't expose its intermediate `eri_mo`, so this is
    /// duplicated here specifically to build real input for
    /// `coupled_cluster_doubles` instead of the old test's indirect
    /// MP2-only check).
    fn ao_to_mo_ovov(rhf: &RhfResult, eri_ao: &[f64]) -> Vec<f64> {
        let n = rhf.n_basis;
        let n_mo = rhf.n_independent;
        let n_occ = rhf.n_occupied;
        let n_vir = n_mo - n_occ;
        let c = &rhf.orbital_coefficients;
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
                                * eri_2
                                    [mu * n * n_occ * n_vir + nu * n_occ * n_vir + j * n_vir + b];
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
                                * eri_3[mu * n_vir * n_occ * n_vir
                                    + a * n_occ * n_vir
                                    + j * n_vir
                                    + b];
                        }
                        eri_mo[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b] = val;
                    }
                }
            }
        }
        eri_mo
    }

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

    // ── Phase Q0 (2026-07-16): real tests exercising `coupled_cluster_doubles`
    // itself, not just MP2's unrelated correlation-energy sign, per the module
    // doc's status block above ──────────────────────────────────────────────

    #[test]
    fn test_ccd_h2_sto3g_returns_exact_zero_not_the_old_tests_implicit_mp2_value() {
        // Empirically measured 2026-07-16 (not guessed): for ANY n_occ=1
        // closed-shell system (H2, HeH+, He...), `coupled_cluster_doubles`
        // returns EXACTLY zero correlation energy. Root cause: with only one
        // occupied spatial orbital, i=j=0 in every term of both the MP2
        // guess and the energy sum, and the antisymmetrized quantity
        // `v = eri(i,a,j,b) - eri(i,b,j,a)` collapses to `eri(0,a,0,b) -
        // eri(0,b,0,a)`, which is identically zero via the standard real-ERI
        // permutational symmetry (pq|rs) = (rs|pq) applied with i=j. This
        // means the OLD test's comment ("H2 STO-3G ... CCD = MP2, no
        // quadratic term") was doubly wrong: it never called this function,
        // AND the claim is false -- real MP2 for H2/STO-3G is a genuine
        // negative number (~-0.023 Ha per `post_hf::mp2`'s own tests), not
        // zero. This structural zero also means t2 is entirely zero after
        // the MP2 guess for n_occ=1 systems, which is *why*
        // `test_ccd_h2_631g_quadratic_term_indexing_is_broken` below doesn't
        // panic despite n_vir=3>1 -- the buggy quadratic-term code path is
        // gated behind `if t_ijcd.abs() > 1e-15`, and t_ijcd is always 0
        // here, so it never executes.
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let eri_mo = ao_to_mo_ovov(&rhf, &eri);
        let n_occ = rhf.n_occupied;
        let n_vir = rhf.n_independent - n_occ;
        assert_eq!((n_occ, n_vir), (1, 1));

        let ccd = coupled_cluster_doubles(&rhf, &eri_mo, n_occ, n_vir, 50, 1e-10);
        assert!(ccd.converged);
        assert_eq!(
            ccd.correlation_energy, 0.0,
            "expected the structural i=j zero collapse described above"
        );
    }

    #[test]
    fn test_ccd_h2_631g_no_panic_because_t2_guess_is_structurally_zero() {
        // Companion to the test above: H2/6-31G also has n_occ=1 (so the
        // same i=j collapse applies) despite n_vir=3>1. Confirms the
        // quadratic term's buggy `eri()` call is gated by a guard that
        // happens to always be false here, NOT that the indexing is
        // actually safe for n_vir>1 in general -- see
        // `test_ccd_lih_sto3g_quadratic_term_indexing_panics` below for a
        // case where the same code genuinely panics.
        let mol = Molecule::h2();
        let basis = Basis631G::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let eri_mo = ao_to_mo_ovov(&rhf, &eri);
        let n_occ = rhf.n_occupied;
        let n_vir = rhf.n_independent - n_occ;
        assert_eq!((n_occ, n_vir), (1, 3));

        let ccd = coupled_cluster_doubles(&rhf, &eri_mo, n_occ, n_vir, 50, 1e-10);
        // 6-31G's larger contraction accumulates floating-point noise around
        // the structural zero (observed ~1e-33), unlike STO-3G's exact 0.0.
        assert!(
            ccd.correlation_energy.abs() < 1e-12,
            "expected ~0.0 (structural zero, see module doc), got {}",
            ccd.correlation_energy
        );
    }

    #[test]
    fn test_ccd_water_sto3g_gives_a_nonzero_number_by_coincidence_not_correctness() {
        // Water/STO-3G: n_occ=5, n_vir=2. n_occ>1 makes i!=j reachable (so
        // t2 is genuinely nonzero after the MP2 guess), but n_vir(2) <=
        // n_occ(5), so the quadratic term's out-of-bounds-prone index
        // (`a` ranging 0..n_vir) never reaches/exceeds n_occ here -- the
        // buggy code path happens to stay in-bounds for THIS specific
        // occupied/virtual ratio, not because it's correct. Locks in the
        // current output as a regression baseline only -- not validated
        // against any external reference.
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let eri_mo = ao_to_mo_ovov(&rhf, &eri);
        let n_occ = rhf.n_occupied;
        let n_vir = rhf.n_independent - n_occ;
        assert_eq!((n_occ, n_vir), (5, 2));

        let ccd = coupled_cluster_doubles(&rhf, &eri_mo, n_occ, n_vir, 50, 1e-10);
        assert!(ccd.converged);
        assert!(
            (ccd.correlation_energy - (-0.001_012_559_6)).abs() < 1e-8,
            "correlation energy drifted from the locked baseline: {}",
            ccd.correlation_energy
        );
    }

    #[test]
    fn test_ccd_lih_sto3g_quadratic_term_indexing_panics() {
        // LiH/STO-3G: n_occ=2, n_vir=4 -- the combination that actually
        // exercises the bug described in the module doc's status block.
        // n_occ>1 makes t2 genuinely nonzero (unlike the n_occ=1 cases
        // above), and n_vir>n_occ means the quadratic term's `a` index
        // (0..n_vir) can exceed n_occ, at which point
        // `eri(n_occ + a - n_occ, c, n_occ + b - n_occ, d)` -- effectively
        // `eri(a, c, b, d)` -- indexes `eri_mo` (sized for occ x vir x occ x
        // vir = 2x4x2x4 = 64) using a VIRTUAL-orbital-local `a` in the slot
        // `eri_mo`'s layout reserves for an OCCUPIED index, going out of
        // bounds. Empirically confirmed to panic 2026-07-16; this is the
        // module's real, common-case failure mode -- most real molecules
        // beyond the most minimal occupied/virtual ratios will hit this.
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(3, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(1, 0.0, 0.0, 3.015),
        ]);
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let eri_mo = ao_to_mo_ovov(&rhf, &eri);
        let n_occ = rhf.n_occupied;
        let n_vir = rhf.n_independent - n_occ;
        assert_eq!((n_occ, n_vir), (2, 4));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            coupled_cluster_doubles(&rhf, &eri_mo, n_occ, n_vir, 50, 1e-10)
        }));
        assert!(
            result.is_err(),
            "expected coupled_cluster_doubles to panic (out-of-bounds eri_mo index) for LiH/STO-3G; \
             if this now passes, the quadratic-term indexing bug documented in the module doc \
             has been fixed and this test should be replaced with a real correctness check"
        );
    }
}
