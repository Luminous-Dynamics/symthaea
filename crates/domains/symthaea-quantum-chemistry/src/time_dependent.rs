// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-Dependent Quantum Mechanics.
//!
//! Solves the time-dependent Schrödinger equation (TDSE):
//! iℏ ∂Ψ/∂t = H Ψ
//!
//! `RabiDynamics` and `time_evolve_eigenstate` are genuine, validated
//! two-level/eigenbasis time-evolution physics. `cis_excitations` is real
//! Configuration Interaction Singles (Phase Q5, 2026-07-17; Coulomb-term
//! factor-of-2 bug fixed and externally validated against pyscf in Phase
//! Q6b, 2026-07-18): it builds and diagonalizes the actual singlet-adapted
//! CIS matrix `A_{ia,jb} = (ε_a - ε_i)δ_ij δ_ab + 2(ia|jb) - (ij|ab)`. The
//! cheaper Koopmans'-theorem-only approximation this function used to be
//! (Phase Q0, 2026-07-16, corrected the doc to disclose this without fixing
//! the implementation) is kept under its own honest name,
//! `koopmans_theorem_excitations`.
//!
//! Scope: RHF-reference singlet CIS only (no triplets/spin-flip, no
//! UHF-based CIS, no TDHF/RPA de-excitation coupling, no TDDFT).
//! `oscillator_strength` remains a rough energy-gap-derived estimate, not a
//! real transition-dipole integral -- this crate has no dipole-moment
//! integral module.
//!
//! References:
//! - Szabo & Ostlund (1996), Chapter 4 (CIS).
//! - Dreuw & Head-Gordon (2005). Chem. Rev. 105, 4009 (TDDFT review).

use crate::scf::generalized_eigen::symmetric_eigen;
use crate::scf::rhf::RhfResult;
use std::f64::consts::PI;

/// Koopmans'-theorem excitation energy result (the cheap approximation --
/// see `koopmans_theorem_excitations`).
#[derive(Debug, Clone)]
pub struct CisExcitation {
    /// Excitation energy (Hartree)
    pub energy: f64,
    /// Excitation energy (eV)
    pub energy_ev: f64,
    /// The i→a transition (occupied → virtual) this energy comes from
    pub from_orbital: usize,
    pub to_orbital: usize,
    /// Oscillator strength (dimensionless)
    pub oscillator_strength: f64,
}

/// A real Configuration Interaction Singles excited state (Phase Q5,
/// 2026-07-17) -- an eigenstate of the CIS matrix, not a single i→a pair.
#[derive(Debug, Clone)]
pub struct CisState {
    /// CIS excitation energy (Hartree) -- an eigenvalue of the CIS matrix.
    pub energy: f64,
    pub energy_ev: f64,
    /// The single-excitation component with the largest |c_ia|² weight --
    /// a real CIS state is generally a linear combination, not literally
    /// this one transition.
    pub dominant_from_orbital: usize,
    pub dominant_to_orbital: usize,
    /// |c_ia|² of the dominant component (eigenvector columns from
    /// `symmetric_eigen` are unit-norm, so this is already normalized --
    /// 1.0 means the state is purely that one i→a excitation).
    pub dominant_weight: f64,
    /// Oscillator strength (dimensionless) -- same disclosed rough
    /// energy-gap estimate as `koopmans_theorem_excitations`, not computed
    /// from real transition-dipole integrals (none exist in this crate).
    pub oscillator_strength: f64,
}

/// A real Random Phase Approximation (RPA/TDHF) excited state (Phase Q6b,
/// 2026-07-18) -- unlike CIS/TDA, includes de-excitation (`Y`) coupling via
/// the `B` matrix. Mirrors `CisState`'s shape for API consistency.
#[derive(Debug, Clone)]
pub struct RpaState {
    /// Excitation energy (Hartree) -- `ω = sqrt(eigenvalue)` of the reduced
    /// symmetric eigenproblem `(A-B)^{1/2}(A+B)(A-B)^{1/2}`.
    pub energy: f64,
    pub energy_ev: f64,
    /// The single-excitation component of `(X+Y)` with the largest weight
    /// (normalized to unit norm -- `(X+Y)` from the reduction isn't
    /// automatically unit-norm the way CIS's raw eigenvectors are).
    pub dominant_from_orbital: usize,
    pub dominant_to_orbital: usize,
    pub dominant_weight: f64,
    /// Same disclosed rough energy-gap estimate as `CisState`'s field --
    /// no real transition-dipole integrals in this crate.
    pub oscillator_strength: f64,
}

/// Result of an RPA/TDHF calculation (Phase Q6b, 2026-07-18). Mirrors
/// `RhfResult`'s `converged: bool` convention: a non-positive-definite
/// `S=A-B` (or a resulting negative `ω²`) is a genuine RPA/Thouless
/// instability -- real physics tied to the underlying RHF reference (e.g.
/// N2/STO-3G's known-unstable reference, Phase Q6a), not a programmer error
/// -- so this reports status via a field rather than panicking.
#[derive(Debug, Clone)]
pub struct RpaResult {
    /// `true` iff every eigenvalue of `S=A-B` exceeds a small positive
    /// threshold AND every resulting `ω²` is non-negative. `states` is only
    /// populated when `true`.
    pub stable: bool,
    /// The smallest eigenvalue found among `S=A-B`'s spectrum -- a negative
    /// value quantifies how unstable the reference is, useful for
    /// diagnostics even when `stable == false`.
    pub min_stability_eigenvalue: f64,
    pub states: Vec<RpaState>,
}

/// `(ia|jb)` AO→MO partial transform, occ×vir×occ×vir layout. Same
/// quarter-transform algorithm as `post_hf::mp2::mp2_correlation_energy`'s
/// internal transform and `coupled_cluster.rs`'s test-only `ao_to_mo_ovov`
/// (each implements it independently -- established precedent in this
/// crate for this specific well-tested pattern rather than shared
/// cross-module infrastructure for a third consumer).
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
    eri_mo
}

/// `(ij|ab)` AO→MO partial transform, occ×occ×vir×vir layout. New layout
/// (not computed anywhere else in this crate) needed for CIS's
/// exchange-like coupling term.
fn ao_to_mo_oovv(rhf: &RhfResult, eri_ao: &[f64]) -> Vec<f64> {
    let n = rhf.n_basis;
    let n_mo = rhf.n_independent;
    let n_occ = rhf.n_occupied;
    let n_vir = n_mo - n_occ;
    let c = &rhf.orbital_coefficients;
    let n3 = n * n * n;
    let n2 = n * n;

    // (μν|λb): σ → b (virtual)
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
    // (μν|ab): λ → a (virtual)
    let mut eri_2 = vec![0.0; n * n * n_vir * n_vir];
    for mu in 0..n {
        for nu in 0..n {
            for a in 0..n_vir {
                let a_mo = n_occ + a;
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for lam in 0..n {
                        val += c[lam * n_mo + a_mo]
                            * eri_1[mu * n * n * n_vir + nu * n * n_vir + lam * n_vir + b];
                    }
                    eri_2[mu * n * n_vir * n_vir + nu * n_vir * n_vir + a * n_vir + b] = val;
                }
            }
        }
    }
    // (μj|ab): ν → j (occupied)
    let mut eri_3 = vec![0.0; n * n_occ * n_vir * n_vir];
    for mu in 0..n {
        for j in 0..n_occ {
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for nu in 0..n {
                        val += c[nu * n_mo + j]
                            * eri_2[mu * n * n_vir * n_vir + nu * n_vir * n_vir + a * n_vir + b];
                    }
                    eri_3[mu * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b] = val;
                }
            }
        }
    }
    // (ij|ab): μ → i (occupied)
    let mut eri_mo = vec![0.0; n_occ * n_occ * n_vir * n_vir];
    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let mut val = 0.0;
                    for mu in 0..n {
                        val += c[mu * n_mo + i]
                            * eri_3[mu * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b];
                    }
                    eri_mo[i * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b] = val;
                }
            }
        }
    }
    eri_mo
}

/// Build the CIS matrix `A_{ia,jb} = (ε_a - ε_i)δ_ij δ_ab + 2(ia|jb) -
/// (ij|ab)`, size `n_ov × n_ov` where `n_ov = n_occ * n_vir`. This is the
/// closed-shell **singlet**-adapted formula (Szabo & Ostlund; Casida 1995;
/// Dreuw & Head-Gordon 2005) -- the factor of 2 on the Coulomb term comes
/// from summing the α and β spin channels, while the exchange-like term only
/// survives for same-spin. Real and symmetric (`(ia|jb)=(jb|ia)` and
/// `(ij|ab)=(ab|ij)` both follow from real-ERI permutational symmetry) --
/// confirmed by a dedicated test, not just assumed.
///
/// **Phase Q6b (2026-07-18) fix**: this formula previously had NO factor of 2
/// on the Coulomb term -- a real bug, not an intentional alternate
/// convention, confirmed via direct comparison against pyscf's `tdscf.TDA`
/// (Q5/Q5b/Q5c had deferred TDHF/RPA three times specifically because this
/// convention question couldn't be checked without an external reference; a
/// working pyscf shell landed in Q6a resolved it). For H2/STO-3G's exact 1×1
/// case, the old formula gave 0.76616466 Ha vs. pyscf's 0.94742258 Ha -- a
/// gap that matches this crate's own `(ia|ia)` integral (0.18125791) to 8
/// decimal places, i.e. exactly one missing copy of the Coulomb term. See
/// `QUANTUM_CHEMISTRY_PHASE_Q6B_CIS_FIX_AND_TDHF_2026-07-18.md`.
fn build_cis_matrix(rhf: &RhfResult, eri_ao: &[f64]) -> (Vec<f64>, usize, usize) {
    let n_occ = rhf.n_occupied;
    let n_vir = rhf.n_independent - n_occ;
    let n_ov = n_occ * n_vir;
    let eps = &rhf.orbital_energies;

    let eri_ovov = ao_to_mo_ovov(rhf, eri_ao);
    let eri_oovv = ao_to_mo_oovv(rhf, eri_ao);

    let ovov = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        eri_ovov[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b]
    };
    let oovv = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        eri_oovv[i * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b]
    };

    let mut matrix = vec![0.0; n_ov * n_ov];
    for i in 0..n_occ {
        for a in 0..n_vir {
            let row = i * n_vir + a;
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let col = j * n_vir + b;
                    let mut val = 2.0 * ovov(i, a, j, b) - oovv(i, j, a, b);
                    if i == j && a == b {
                        val += eps[n_occ + a] - eps[i];
                    }
                    matrix[row * n_ov + col] = val;
                }
            }
        }
    }
    (matrix, n_occ, n_vir)
}

/// Real Configuration Interaction Singles (Phase Q5, 2026-07-17): builds and
/// diagonalizes the actual CIS matrix, returning genuine excited-state
/// energies (not raw orbital-energy differences). See the module doc for
/// scope (RHF-reference singlet CIS only) and `koopmans_theorem_excitations`
/// for the cheaper approximation this function used to silently be.
pub fn cis_excitations(rhf: &RhfResult, eri_ao: &[f64], n_states: usize) -> Vec<CisState> {
    let (matrix, n_occ, n_vir) = build_cis_matrix(rhf, eri_ao);
    let n_ov = n_occ * n_vir;
    if n_ov == 0 {
        return Vec::new();
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen(&matrix, n_ov);

    let mut indices: Vec<usize> = (0..n_ov).collect();
    indices.sort_by(|&p, &q| eigenvalues[p].total_cmp(&eigenvalues[q]));

    indices
        .into_iter()
        .take(n_states)
        .map(|state_idx| {
            let energy = eigenvalues[state_idx];

            // Eigenvectors are column-major: component k of eigenvector
            // `state_idx` is eigenvectors[k * n_ov + state_idx].
            let mut best_component = 0;
            let mut best_weight = -1.0;
            for k in 0..n_ov {
                let c = eigenvectors[k * n_ov + state_idx];
                let w = c * c;
                if w > best_weight {
                    best_weight = w;
                    best_component = k;
                }
            }
            let from_orbital = best_component / n_vir;
            let to_orbital = n_occ + best_component % n_vir;

            let f_approx = 2.0 / 3.0 * energy * 0.5;

            CisState {
                energy,
                energy_ev: energy * 27.211_386,
                dominant_from_orbital: from_orbital,
                dominant_to_orbital: to_orbital,
                dominant_weight: best_weight,
                oscillator_strength: f_approx,
            }
        })
        .collect()
}

/// Build the RPA/TDHF `A` and `B` matrices (Phase Q6b, 2026-07-18):
/// `A_{ia,jb} = (ε_a - ε_i)δ_ij δ_ab + 2(ia|jb) - (ij|ab)` (identical to
/// `build_cis_matrix`'s formula) and `B_{ia,jb} = 2(ia|jb) - (ib|ja)`
/// (Casida 1995; Dreuw & Head-Gordon 2005).
///
/// **Key simplification, verified element-wise against `pyscf.tdscf.TDHF`'s
/// `get_ab()` during Q6b's planning and implementation (not just assumed
/// from the algebra)**: `(ib|ja)` has the same occ×vir/occ×vir index shape
/// as `(ia|jb)` -- both live in `eri_ovov`, just queried with the virtual
/// indices swapped (`ovov(i, b, j, a)` instead of `ovov(i, a, j, b)`). No
/// new AO→MO transform is needed; `B` reuses the exact same tensor as `A`'s
/// Coulomb term. Deliberately duplicates (rather than calls)
/// `build_cis_matrix`'s `A`-construction logic to compute `A` and `B` in one
/// pass over the same `eri_ovov`/`eri_oovv` tensors -- matches this file's
/// own established precedent (see `ao_to_mo_ovov`'s doc comment) of
/// independent implementations for a well-tested pattern over forced sharing.
fn build_rpa_ab_matrices(rhf: &RhfResult, eri_ao: &[f64]) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n_occ = rhf.n_occupied;
    let n_vir = rhf.n_independent - n_occ;
    let n_ov = n_occ * n_vir;
    let eps = &rhf.orbital_energies;

    let eri_ovov = ao_to_mo_ovov(rhf, eri_ao);
    let eri_oovv = ao_to_mo_oovv(rhf, eri_ao);

    let ovov = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        eri_ovov[i * n_vir * n_occ * n_vir + a * n_occ * n_vir + j * n_vir + b]
    };
    let oovv = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        eri_oovv[i * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b]
    };

    let mut a_mat = vec![0.0; n_ov * n_ov];
    let mut b_mat = vec![0.0; n_ov * n_ov];
    for i in 0..n_occ {
        for a in 0..n_vir {
            let row = i * n_vir + a;
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let col = j * n_vir + b;
                    let mut a_val = 2.0 * ovov(i, a, j, b) - oovv(i, j, a, b);
                    if i == j && a == b {
                        a_val += eps[n_occ + a] - eps[i];
                    }
                    a_mat[row * n_ov + col] = a_val;
                    b_mat[row * n_ov + col] = 2.0 * ovov(i, a, j, b) - ovov(i, b, j, a);
                }
            }
        }
    }
    (a_mat, b_mat, n_occ, n_vir)
}

/// Real Random Phase Approximation / TDHF (Phase Q6b, 2026-07-18): unlike
/// `cis_excitations` (the Tamm-Dancoff approximation, `A`-matrix only), this
/// includes de-excitation coupling via the reduction
/// `(A-B)^{1/2}(A+B)(A-B)^{1/2} T = ω² T` (Casida 1995) -- a symmetric
/// eigenproblem reusing `symmetric_eigen` (`scf/generalized_eigen.rs`) twice:
/// once to build `S^{1/2}` from `S = A - B`, once to diagonalize
/// `Ω = S^{1/2}(A+B)S^{1/2}`.
///
/// Stability handling mirrors `RhfResult`'s `converged: bool` convention
/// (see `RpaResult`'s doc): if `S` isn't positive-definite (a real RPA/
/// Thouless instability, e.g. N2/STO-3G's known-unstable RHF reference --
/// Phase Q6a), or if a resulting `ω²` is negative, `stable` is `false` and
/// `states` is empty rather than producing NaN from `sqrt` of a negative
/// number.
pub fn rpa_excitations(rhf: &RhfResult, eri_ao: &[f64], n_states: usize) -> RpaResult {
    let (a_mat, b_mat, n_occ, n_vir) = build_rpa_ab_matrices(rhf, eri_ao);
    let n_ov = n_occ * n_vir;
    if n_ov == 0 {
        return RpaResult {
            stable: true,
            min_stability_eigenvalue: f64::INFINITY,
            states: Vec::new(),
        };
    }

    let mut s_mat = vec![0.0; n_ov * n_ov];
    let mut r_mat = vec![0.0; n_ov * n_ov];
    for idx in 0..n_ov * n_ov {
        s_mat[idx] = a_mat[idx] - b_mat[idx];
        r_mat[idx] = a_mat[idx] + b_mat[idx];
    }

    let (s_eigenvalues, s_eigenvectors) = symmetric_eigen(&s_mat, n_ov);
    let min_stability_eigenvalue = s_eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);

    const STABILITY_THRESHOLD: f64 = 1e-8;
    if min_stability_eigenvalue <= STABILITY_THRESHOLD {
        return RpaResult {
            stable: false,
            min_stability_eigenvalue,
            states: Vec::new(),
        };
    }

    // S^{1/2}[p][q] = sum_k V[p][k] * sqrt(lambda_k) * V[q][k], from
    // S = V * Lambda * V^T with V's columns the (already unit-norm)
    // eigenvectors (component k of eigenvector idx is
    // s_eigenvectors[k * n_ov + idx]).
    let mut s_sqrt = vec![0.0; n_ov * n_ov];
    for p in 0..n_ov {
        for q in 0..n_ov {
            let mut val = 0.0;
            for k in 0..n_ov {
                val += s_eigenvectors[p * n_ov + k]
                    * s_eigenvalues[k].sqrt()
                    * s_eigenvectors[q * n_ov + k];
            }
            s_sqrt[p * n_ov + q] = val;
        }
    }

    // omega = S^{1/2} * R * S^{1/2} (two n_ov x n_ov matrix products).
    let mut temp = vec![0.0; n_ov * n_ov];
    for p in 0..n_ov {
        for q in 0..n_ov {
            let mut val = 0.0;
            for k in 0..n_ov {
                val += s_sqrt[p * n_ov + k] * r_mat[k * n_ov + q];
            }
            temp[p * n_ov + q] = val;
        }
    }
    let mut omega = vec![0.0; n_ov * n_ov];
    for p in 0..n_ov {
        for q in 0..n_ov {
            let mut val = 0.0;
            for k in 0..n_ov {
                val += temp[p * n_ov + k] * s_sqrt[k * n_ov + q];
            }
            omega[p * n_ov + q] = val;
        }
    }

    let (omega_sq, omega_vecs) = symmetric_eigen(&omega, n_ov);
    if omega_sq.iter().any(|&v| v < -STABILITY_THRESHOLD) {
        return RpaResult {
            stable: false,
            min_stability_eigenvalue,
            states: Vec::new(),
        };
    }

    let mut indices: Vec<usize> = (0..n_ov).collect();
    indices.sort_by(|&p, &q| omega_sq[p].total_cmp(&omega_sq[q]));

    let states = indices
        .into_iter()
        .take(n_states)
        .map(|state_idx| {
            let energy = omega_sq[state_idx].max(0.0).sqrt();

            // (X+Y) = S^{1/2} * T (the state_idx-th eigenvector of Omega),
            // then normalize to unit norm before treating squared
            // components as weights (unlike CIS's raw eigenvectors, this
            // isn't automatically unit-norm).
            let mut xy = vec![0.0; n_ov];
            for p in 0..n_ov {
                let mut val = 0.0;
                for k in 0..n_ov {
                    val += s_sqrt[p * n_ov + k] * omega_vecs[k * n_ov + state_idx];
                }
                xy[p] = val;
            }
            let norm: f64 = xy.iter().map(|v| v * v).sum::<f64>().sqrt();

            let mut best_component = 0;
            let mut best_weight = -1.0;
            for k in 0..n_ov {
                let c = if norm > 0.0 { xy[k] / norm } else { 0.0 };
                let w = c * c;
                if w > best_weight {
                    best_weight = w;
                    best_component = k;
                }
            }
            let from_orbital = best_component / n_vir;
            let to_orbital = n_occ + best_component % n_vir;

            let f_approx = 2.0 / 3.0 * energy * 0.5;

            RpaState {
                energy,
                energy_ev: energy * 27.211_386,
                dominant_from_orbital: from_orbital,
                dominant_to_orbital: to_orbital,
                dominant_weight: best_weight,
                oscillator_strength: f_approx,
            }
        })
        .collect();

    RpaResult {
        stable: true,
        min_stability_eigenvalue,
        states,
    }
}

/// Independent-particle (Koopmans'-theorem) excitation energy spectrum --
/// the cheap approximation `cis_excitations` used to silently be before
/// Phase Q5 (2026-07-17) gave it a real CIS-matrix implementation instead.
///
/// Real CIS approximates excited states as single excitations from the HF
/// ground state, |Ψ_excited⟩ = Σ_{i,a} c_{ia} |Φ_i^a⟩, with excitation
/// energies as eigenvalues of the (singlet-adapted) CIS matrix
/// `A_{ia,jb} = (ε_a - ε_i)δ_ij δ_ab + 2(ia|jb) - (ij|ab)`. This function
/// implements only the diagonal term: raw orbital-energy differences
/// `ε_a - ε_i`, with no CIS-matrix construction, no diagonalization, and no
/// two-electron `(ia|jb)`/`(ij|ab)` integrals touched anywhere. This is a
/// real, named approximation (Koopmans' theorem applied to excitations) --
/// it gives the correct qualitative ordering but omits the orbital-relaxation
/// and exchange/Coulomb corrections real CIS provides, and its energies
/// will generally disagree with real CIS results for the same system. Kept
/// (not deleted) because it's cheap and useful in its own right, and because
/// `cis_excitations`'s own test suite uses it as a verification tool (see
/// `test_cis_differs_from_koopmans_when_coupling_present`).
/// `oscillator_strength` below is similarly a rough energy-gap-derived
/// estimate, not computed from real transition-dipole integrals.
pub fn koopmans_theorem_excitations(rhf: &RhfResult, n_excitations: usize) -> Vec<CisExcitation> {
    let n_occ = rhf.n_occupied;
    let n_mo = rhf.n_independent;
    let eps = &rhf.orbital_energies;

    // Simplified CIS: excitation energies = ε_a - ε_i (Koopman's theorem)
    // This neglects electron correlation in the excited state but gives
    // the correct qualitative ordering.
    let mut excitations: Vec<(f64, usize, usize)> = Vec::new();

    for i in 0..n_occ {
        for a in n_occ..n_mo {
            let de = eps[a] - eps[i];
            if de > 0.0 {
                excitations.push((de, i, a));
            }
        }
    }

    // Sort by energy
    excitations.sort_by(|a, b| a.0.total_cmp(&b.0));

    excitations
        .iter()
        .take(n_excitations)
        .map(|&(de, i, a)| {
            // Oscillator strength (simplified: proportional to orbital overlap)
            // f = 2/3 × ΔE × |⟨i|r|a⟩|² — estimated from energy gap
            let f_approx = 2.0 / 3.0 * de * 0.5; // rough estimate

            CisExcitation {
                energy: de,
                energy_ev: de * 27.211_386,
                from_orbital: i,
                to_orbital: a,
                oscillator_strength: f_approx,
            }
        })
        .collect()
}

/// Rabi oscillation: coherent two-level dynamics under driving field.
///
/// P(t) = sin²(Ω_R t / 2) where Ω_R = √(Ω² + Δ²)
///
/// Ω is the coupling strength, Δ is the detuning.
#[derive(Debug, Clone)]
pub struct RabiDynamics {
    /// Rabi frequency Ω_R (a.u.)
    pub rabi_frequency: f64,
    /// Detuning Δ = ω_drive - ω_transition (a.u.)
    pub detuning: f64,
    /// Coupling strength Ω (a.u.)
    pub coupling: f64,
}

impl RabiDynamics {
    /// Create a resonant Rabi system (Δ = 0).
    pub fn resonant(coupling: f64) -> Self {
        Self {
            rabi_frequency: coupling,
            detuning: 0.0,
            coupling,
        }
    }

    /// Create an off-resonant system.
    pub fn off_resonant(coupling: f64, detuning: f64) -> Self {
        let omega_r = (coupling * coupling + detuning * detuning).sqrt();
        Self {
            rabi_frequency: omega_r,
            detuning,
            coupling,
        }
    }

    /// Probability of being in the excited state at time t (a.u.).
    pub fn excited_population(&self, t: f64) -> f64 {
        let theta = self.rabi_frequency * t / 2.0;
        let sin_theta = theta.sin();
        (self.coupling / self.rabi_frequency).powi(2) * sin_theta * sin_theta
    }

    /// Time for complete population inversion (π-pulse time).
    pub fn pi_pulse_time(&self) -> f64 {
        PI / self.rabi_frequency
    }
}

/// Time-evolve a state vector under a time-independent Hamiltonian.
///
/// |Ψ(t)⟩ = exp(-iHt) |Ψ(0)⟩
///
/// For a diagonal Hamiltonian (energy eigenbasis):
/// c_i(t) = c_i(0) × exp(-i E_i t)
///
/// Returns (real parts, imaginary parts) of the evolved coefficients.
pub fn time_evolve_eigenstate(
    coefficients_re: &[f64],
    coefficients_im: &[f64],
    energies: &[f64],
    time: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = coefficients_re.len();
    let mut re_out = vec![0.0; n];
    let mut im_out = vec![0.0; n];

    for i in 0..n {
        let phase = -energies[i] * time;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // (a + bi)(cos + i sin) = (a cos - b sin) + i(a sin + b cos)
        re_out[i] = coefficients_re[i] * cos_p - coefficients_im[i] * sin_p;
        im_out[i] = coefficients_re[i] * sin_p + coefficients_im[i] * cos_p;
    }

    (re_out, im_out)
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
    fn test_cis_excitations_positive() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let exc = cis_excitations(&rhf, &eri, 3);
        assert!(!exc.is_empty(), "Should have excitations");

        for e in &exc {
            assert!(e.energy > 0.0, "Excitation energy should be positive");
            assert!(e.energy_ev > 0.0);
        }
    }

    #[test]
    fn test_cis_ordered() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let exc = cis_excitations(&rhf, &eri, 5);
        for i in 1..exc.len() {
            assert!(exc[i].energy >= exc[i - 1].energy);
        }
    }

    #[test]
    fn test_koopmans_excitations_positive() {
        // Phase Q5 (2026-07-17): the old `cis_excitations` tests, now
        // covering the renamed `koopmans_theorem_excitations` directly.
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let exc = koopmans_theorem_excitations(&rhf, 3);
        assert!(!exc.is_empty(), "Should have excitations");
        for e in &exc {
            assert!(e.energy > 0.0, "Excitation energy should be positive");
            assert!(e.energy_ev > 0.0);
        }
    }

    #[test]
    fn test_koopmans_excitations_ordered() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let exc = koopmans_theorem_excitations(&rhf, 5);
        for i in 1..exc.len() {
            assert!(exc[i].energy >= exc[i - 1].energy);
        }
    }

    /// Exact hand-computable identity (Phase Q5, 2026-07-17; updated Q6b,
    /// 2026-07-18 for the factor-of-2 fix): for H2/STO-3G (n_occ=1, n_vir=1),
    /// the CIS matrix is exactly 1x1, so its only eigenvalue must equal its
    /// single entry, computed independently here via the same transforms:
    /// (ε_a - ε_i) + 2(ia|ia) - (ii|aa). This only proves internal
    /// self-consistency (the code matches its own formula) -- the formula's
    /// correctness against real singlet-CIS physics is separately confirmed
    /// against pyscf's `tdscf.TDA` in `pyscf_cis_tdhf_diagnosis.rs`.
    #[test]
    fn test_cis_h2_sto3g_exact_1x1_identity() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        assert_eq!(rhf.n_occupied, 1);
        assert_eq!(rhf.n_independent - rhf.n_occupied, 1);

        let eri_ovov = ao_to_mo_ovov(&rhf, &eri);
        let eri_oovv = ao_to_mo_oovv(&rhf, &eri);
        // n_occ=1, n_vir=1: only element is (i,a,j,b)=(0,0,0,0).
        let ia_ia = eri_ovov[0];
        let ii_aa = eri_oovv[0];
        let eps = &rhf.orbital_energies;
        let expected = (eps[1] - eps[0]) + 2.0 * ia_ia - ii_aa;

        let states = cis_excitations(&rhf, &eri, 1);
        assert_eq!(states.len(), 1);
        assert!(
            (states[0].energy - expected).abs() < 1e-10,
            "CIS energy {} != hand-derived identity {}",
            states[0].energy,
            expected
        );
        assert_eq!(states[0].dominant_weight, 1.0);
        assert_eq!(states[0].dominant_from_orbital, 0);
        assert_eq!(states[0].dominant_to_orbital, 1);
    }

    /// External validation (Phase Q6b, 2026-07-18): unlike the H2 identity
    /// above (internal self-consistency only), this pins CIS energies to
    /// values independently computed by pyscf's `tdscf.TDA` at the same
    /// geometry/basis (`examples/pyscf_cis_tdhf_diagnosis.rs` +
    /// `nix develop .#qc-verify`), confirming the singlet Coulomb-factor fix
    /// is correct against real external physics, not just internally
    /// consistent. Tolerance matches the ~1e-7 Ha precision level actually
    /// observed across all 5 non-N2 benchmark molecules.
    #[test]
    fn test_cis_h2o_sto3g_matches_pyscf_tda() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let states = cis_excitations(&rhf, &eri, 6);
        assert_eq!(states.len(), 6);

        // pyscf tdscf.TDA(mf), nstates=6, same H2O/STO-3G geometry.
        let pyscf_energies = [
            0.485_178_64,
            0.557_273_53,
            0.616_660_66,
            0.705_611_22,
            0.811_593_33,
            1.069_916_57,
        ];
        for (state, &expected) in states.iter().zip(pyscf_energies.iter()) {
            assert!(
                (state.energy - expected).abs() < 1e-5,
                "CIS energy {} does not match pyscf TDA reference {}",
                state.energy,
                expected
            );
        }
    }

    /// External validation (Phase Q6b, 2026-07-18): re-verifies, not
    /// re-assumes, that N2/STO-3G's negative CIS eigenvalues (Phase Q5) are
    /// unchanged by the singlet-formula fix -- this crate's RHF necessarily
    /// converges to the unstable SCF solution (Q6a), so CIS built on top of
    /// it should still show negative eigenvalues regardless of the Coulomb
    /// term's coefficient. Confirmed directly: `pyscf_cis_tdhf_diagnosis.rs`
    /// shows 3 of 6 states negative under the fixed formula, same as before.
    #[test]
    fn test_cis_n2_sto3g_still_shows_negative_eigenvalues_after_fix() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(7, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(7, 0.0, 0.0, 2.074),
        ]);
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let states = cis_excitations(&rhf, &eri, 6);
        let n_negative = states.iter().filter(|s| s.energy < 0.0).count();
        assert!(
            n_negative > 0,
            "expected N2/STO-3G CIS to still show negative eigenvalues (RHF-reference \
             instability, unrelated to the Coulomb-factor fix), got all-positive energies: {:?}",
            states.iter().map(|s| s.energy).collect::<Vec<_>>()
        );
    }

    /// Diagonal-element external validation of `build_rpa_ab_matrices`
    /// (Phase Q6b, 2026-07-18): compared every raw element of `A`/`B` against
    /// `pyscf.tdscf.TDHF.get_ab()` first and found the *diagonal* elements
    /// matched to ~1e-7 Ha while several *off-diagonal* elements differed by
    /// an exact sign flip, always in matched pairs (e.g. `A[0,2]`/`A[2,0]`
    /// both flipped together). This is not a formula bug -- individual
    /// molecular orbitals from an eigensolver are only defined up to an
    /// arbitrary overall sign, and two independently-implemented SCF solvers
    /// (this crate's Jacobi diagonalization vs. pyscf's LAPACK-based one) can
    /// and do land on different sign choices for some orbitals. Off-diagonal
    /// `(ia|jb)`-type integrals flip sign under an odd-count sign flip of any
    /// one orbital; diagonal `(ia|ia)`-type terms involve each orbital an
    /// even number of times and are immune, exactly matching what was
    /// observed. Confirmed by isolating the diagonal (sign-invariant
    /// regardless of orbital-sign convention) here, and by the fully
    /// convention-independent `rpa_excitations` eigenvalue check below
    /// (verifies `B`'s formula correctly, without this ambiguity).
    #[test]
    fn test_rpa_a_matrix_diagonal_matches_pyscf_get_ab_water_sto3g() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let (a_mat, b_mat, n_occ, n_vir) = build_rpa_ab_matrices(&rhf, &eri);
        assert_eq!(n_occ, 5);
        assert_eq!(n_vir, 2);
        let n_ov = n_occ * n_vir;

        // pyscf tdscf.TDHF(mf).get_ab() diagonal, same H2O/STO-3G geometry
        // (row=i*n_vir+a convention).
        let pyscf_a_diag: [f64; 10] = [
            20.107_376_49,
            20.157_823_56,
            1.463_698_21,
            1.510_281_93,
            0.793_956_14,
            1.054_386_82,
            0.646_826_76,
            0.724_950_97,
            0.485_178_64,
            0.557_273_53,
        ];
        for k in 0..n_ov {
            let idx = k * n_ov + k;
            assert!(
                (a_mat[idx] - pyscf_a_diag[k]).abs() < 1e-5,
                "A[{k},{k}]={} does not match pyscf diagonal {}",
                a_mat[idx],
                pyscf_a_diag[k]
            );
        }

        // Structural, convention-independent checks: A and B are both
        // symmetric (real-ERI permutational symmetry, same argument as
        // build_cis_matrix's symmetry test).
        for p in 0..n_ov {
            for q in 0..n_ov {
                assert!(
                    (a_mat[p * n_ov + q] - a_mat[q * n_ov + p]).abs() < 1e-9,
                    "A not symmetric at ({p},{q})"
                );
                assert!(
                    (b_mat[p * n_ov + q] - b_mat[q * n_ov + p]).abs() < 1e-9,
                    "B not symmetric at ({p},{q})"
                );
            }
        }
    }

    /// Decisive, convention-independent external validation (Phase Q6b,
    /// 2026-07-18): excitation energies from `rpa_excitations` (eigenvalues
    /// of the full `(A-B)^{1/2}(A+B)(A-B)^{1/2}` reduction) compared against
    /// `pyscf.tdscf.TDHF.e` for the same H2O/STO-3G geometry. Unlike raw
    /// matrix elements, eigenvalues are immune to the per-orbital sign
    /// ambiguity documented above (a similarity transform by a diagonal
    /// sign matrix leaves the spectrum unchanged) -- this is the real proof
    /// that `B`'s formula, and the `S`/`R`/`Ω` reduction machinery, are
    /// correct, not just that `A` (already CIS-validated) is correct.
    #[test]
    fn test_rpa_excitations_h2o_sto3g_matches_pyscf_tdhf() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let result = rpa_excitations(&rhf, &eri, 6);
        assert!(result.stable, "H2O/STO-3G RPA should be stable");
        assert_eq!(result.states.len(), 6);

        // pyscf tdscf.TDHF(mf).kernel(), nstates=6, same geometry.
        let pyscf_energies = [
            0.483_640_05,
            0.556_741_29,
            0.612_613_90,
            0.702_849_92,
            0.807_803_22,
            1.047_500_10,
        ];
        for (state, &expected) in result.states.iter().zip(pyscf_energies.iter()) {
            assert!(
                (state.energy - expected).abs() < 1e-5,
                "RPA energy {} does not match pyscf TDHF reference {}",
                state.energy,
                expected
            );
        }
    }

    /// N2/STO-3G's RPA instability path (Phase Q6b, 2026-07-18): this crate's
    /// RHF reference for N2/STO-3G is a genuine but unstable SCF stationary
    /// point (Phase Q6a), which already manifests as negative CIS eigenvalues
    /// (`test_cis_n2_sto3g_still_shows_negative_eigenvalues_after_fix`).
    /// `S=A-B` should therefore also fail to be positive-definite, and
    /// `rpa_excitations` should report `stable=false` with an empty state
    /// list rather than producing NaN from `sqrt` of a negative eigenvalue.
    #[test]
    fn test_rpa_n2_sto3g_reports_unstable() {
        let mol = Molecule::new(vec![
            crate::molecule::Atom::new(7, 0.0, 0.0, 0.0),
            crate::molecule::Atom::new(7, 0.0, 0.0, 2.074),
        ]);
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let result = rpa_excitations(&rhf, &eri, 6);
        assert!(
            !result.stable,
            "expected N2/STO-3G RPA to be reported unstable (min_stability_eigenvalue={}), \
             got stable=true with {} states",
            result.min_stability_eigenvalue,
            result.states.len()
        );
        assert!(result.states.is_empty());
        assert!(
            result.min_stability_eigenvalue < 0.0,
            "expected a negative min_stability_eigenvalue, got {}",
            result.min_stability_eigenvalue
        );
    }

    /// Matrix symmetry (Phase Q5, 2026-07-17): a genuine, nontrivial check
    /// on the two new AO->MO transforms and the assembly code -- follows
    /// from real-ERI permutational symmetry, not a tautology.
    #[test]
    fn test_cis_matrix_is_symmetric_water_sto3g() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        assert_eq!(rhf.n_occupied, 5);
        assert_eq!(rhf.n_independent - rhf.n_occupied, 2);

        let (matrix, n_occ, n_vir) = build_cis_matrix(&rhf, &eri);
        let n_ov = n_occ * n_vir;
        for p in 0..n_ov {
            for q in 0..n_ov {
                assert!(
                    (matrix[p * n_ov + q] - matrix[q * n_ov + p]).abs() < 1e-10,
                    "CIS matrix not symmetric at ({p},{q})"
                );
            }
        }
    }

    /// Coupling actually executes (Phase Q5, 2026-07-17): for water, real
    /// CIS's lowest excitation must differ from Koopmans' -- evidence the
    /// (ia|jb)-(ij|ab) coupling terms are genuinely computed and nonzero,
    /// not a silent no-op. H2 is deliberately not used here (n_ov=1 means
    /// no off-diagonal coupling is even possible -- see the exact-identity
    /// test above instead).
    #[test]
    fn test_cis_differs_from_koopmans_when_coupling_present() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());
        let (eri, _, _) = compute_eri_tensor(&basis.functions);

        let cis = cis_excitations(&rhf, &eri, 1);
        let koopmans = koopmans_theorem_excitations(&rhf, 1);
        assert!(
            (cis[0].energy - koopmans[0].energy).abs() > 1e-6,
            "CIS ({}) should differ meaningfully from Koopmans ({}) when coupling is present",
            cis[0].energy,
            koopmans[0].energy
        );
    }

    #[test]
    fn test_rabi_resonant_pi_pulse() {
        let rabi = RabiDynamics::resonant(0.01); // Ω = 0.01 a.u.
        let t_pi = rabi.pi_pulse_time();
        let pop = rabi.excited_population(t_pi);
        assert!(
            (pop - 1.0).abs() < 1e-10,
            "π-pulse should give P=1: got {}",
            pop
        );
    }

    #[test]
    fn test_rabi_oscillation_period() {
        let rabi = RabiDynamics::resonant(0.01);
        // At t=0, P=0
        assert!(rabi.excited_population(0.0).abs() < 1e-14);
        // At t=π/Ω, P=1
        let p_max = rabi.excited_population(PI / 0.01);
        assert!((p_max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_evolution_norm_preserved() {
        let re = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let im = vec![0.0, 0.0];
        let energies = vec![-1.0, -0.5];

        let (re_t, im_t) = time_evolve_eigenstate(&re, &im, &energies, 10.0);

        let norm: f64 = re_t
            .iter()
            .zip(im_t.iter())
            .map(|(r, i)| r * r + i * i)
            .sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Norm should be preserved: {}",
            norm
        );
    }
}
