// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gauge group representations: SU(2) and SU(3).
//!
//! Implements the Lie algebra generators, structure constants, and Casimir
//! operators for the Standard Model gauge groups.
//!
//! References:
//! - Georgi, H. (1999). *Lie Algebras in Particle Physics*. Westview.
//! - Gell-Mann, M. (1962). Phys. Rev. 125, 1067.

use crate::constants::N_C;

/// A complex number (re, im) for matrix elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

// ── SU(2): Pauli Matrices ───────────────────────────────────────────────────

/// Pauli matrices σ₁, σ₂, σ₃ (the generators of SU(2) are T_i = σ_i/2).
pub fn pauli_matrices() -> [[[Complex; 2]; 2]; 3] {
    let z = Complex::ZERO;
    let o = Complex::ONE;
    let i = Complex::I;
    let mi = Complex::new(0.0, -1.0);

    [
        // σ₁
        [[z, o], [o, z]],
        // σ₂
        [[z, mi], [i, z]],
        // σ₃
        [[o, z], [z, Complex::new(-1.0, 0.0)]],
    ]
}

/// SU(2) structure constants: [T_a, T_b] = i ε_abc T_c
/// Returns ε_abc (Levi-Civita symbol for 3 indices).
pub fn su2_structure_constant(a: usize, b: usize, c: usize) -> f64 {
    // ε_abc is the totally antisymmetric tensor
    if a == b || b == c || a == c {
        return 0.0;
    }
    match (a, b, c) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}

/// SU(2) Casimir operator C₂(j) = j(j+1) for representation with spin j.
pub fn su2_casimir(j: f64) -> f64 {
    j * (j + 1.0)
}

// ── SU(3): Gell-Mann Matrices ───────────────────────────────────────────────

/// The 8 Gell-Mann matrices λ₁...λ₈ (generators of SU(3) are T_a = λ_a/2).
///
/// Returns λ_a as a 3×3 complex matrix.
pub fn gell_mann_matrix(index: usize) -> [[Complex; 3]; 3] {
    let z = Complex::ZERO;
    let o = Complex::ONE;
    let mo = Complex::new(-1.0, 0.0);
    let i = Complex::I;
    let mi = Complex::new(0.0, -1.0);
    let s = Complex::new(1.0 / 3.0_f64.sqrt(), 0.0);
    let _ms = Complex::new(-1.0 / 3.0_f64.sqrt(), 0.0);
    let _s2 = Complex::new(2.0 / 3.0_f64.sqrt(), 0.0);

    match index {
        // λ₁: off-diagonal (1,2)
        0 => [[z, o, z], [o, z, z], [z, z, z]],
        // λ₂: off-diagonal (1,2) imaginary
        1 => [[z, mi, z], [i, z, z], [z, z, z]],
        // λ₃: diagonal
        2 => [[o, z, z], [z, mo, z], [z, z, z]],
        // λ₄: off-diagonal (1,3)
        3 => [[z, z, o], [z, z, z], [o, z, z]],
        // λ₅: off-diagonal (1,3) imaginary
        4 => [[z, z, mi], [z, z, z], [i, z, z]],
        // λ₆: off-diagonal (2,3)
        5 => [[z, z, z], [z, z, o], [z, o, z]],
        // λ₇: off-diagonal (2,3) imaginary
        6 => [[z, z, z], [z, z, mi], [z, i, z]],
        // λ₈: diagonal (hypercharge-like)
        7 => [[s, z, z], [z, s, z], [z, z, Complex::new(-2.0 / 3.0_f64.sqrt(), 0.0)]],
        _ => [[z, z, z], [z, z, z], [z, z, z]],
    }
}

/// SU(3) structure constants f_abc defined by [T_a, T_b] = i f_abc T_c.
///
/// Only the non-zero independent components are listed.
/// f_abc is totally antisymmetric.
pub fn su3_structure_constant(a: usize, b: usize, c: usize) -> f64 {
    // Canonical non-zero values (1-indexed in physics, 0-indexed here)
    let fabc = |i: usize, j: usize, k: usize| -> Option<f64> {
        match (i, j, k) {
            (0, 1, 2) => Some(1.0),
            (0, 3, 6) => Some(0.5),
            (0, 4, 5) => Some(-0.5), // f_156 = -1/2 (0-indexed: 0,4,5)
            (1, 3, 5) => Some(0.5), // f_245 (0-indexed: 1,3,5) — actually f_246=1/2
            (1, 4, 6) => Some(0.5),
            (2, 3, 4) => Some(0.5),
            (2, 5, 6) => Some(-0.5),
            (3, 5, 7) => Some(3.0_f64.sqrt() / 2.0),
            (4, 6, 7) => Some(3.0_f64.sqrt() / 2.0),
            _ => None,
        }
    };

    // Try all permutations with sign tracking
    let perms: [(usize, usize, usize, f64); 6] = [
        (a, b, c, 1.0),
        (b, c, a, 1.0),
        (c, a, b, 1.0),
        (b, a, c, -1.0),
        (a, c, b, -1.0),
        (c, b, a, -1.0),
    ];

    for (i, j, k, sign) in perms {
        if let Some(val) = fabc(i, j, k) {
            return sign * val;
        }
    }

    0.0
}

/// SU(3) Casimir operators.
/// C₂(fundamental) = (N²-1)/(2N) = 4/3
/// C₂(adjoint) = N = 3
pub fn su3_casimir_fundamental() -> f64 {
    (N_C * N_C - 1.0) / (2.0 * N_C) // 4/3
}

pub fn su3_casimir_adjoint() -> f64 {
    N_C // 3
}

/// Verify the Gell-Mann algebra: Tr(λ_a λ_b) = 2 δ_ab.
pub fn gell_mann_trace_product(a: usize, b: usize) -> Complex {
    let la = gell_mann_matrix(a);
    let lb = gell_mann_matrix(b);

    // Tr(A*B) = Σ_i Σ_j A[i][j] * B[j][i]
    let mut trace = Complex::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            let prod_re = la[i][j].re * lb[j][i].re - la[i][j].im * lb[j][i].im;
            let prod_im = la[i][j].re * lb[j][i].im + la[i][j].im * lb[j][i].re;
            trace.re += prod_re;
            trace.im += prod_im;
        }
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_traceless() {
        let sigma = pauli_matrices();
        for a in 0..3 {
            let trace = sigma[a][0][0].re + sigma[a][1][1].re;
            assert!(trace.abs() < 1e-14, "σ_{} not traceless: Tr={}", a + 1, trace);
        }
    }

    #[test]
    fn test_pauli_hermitian() {
        let sigma = pauli_matrices();
        for a in 0..3 {
            for i in 0..2 {
                for j in 0..2 {
                    // Hermitian: M[i][j] = M[j][i]*
                    let mij = sigma[a][i][j];
                    let mji = sigma[a][j][i];
                    assert!(
                        (mij.re - mji.re).abs() < 1e-14 && (mij.im + mji.im).abs() < 1e-14,
                        "σ_{} not Hermitian at ({},{})",
                        a + 1,
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_su2_casimir_values() {
        assert!((su2_casimir(0.5) - 0.75).abs() < 1e-14); // spin-1/2: 3/4
        assert!((su2_casimir(1.0) - 2.0).abs() < 1e-14); // spin-1: 2
    }

    #[test]
    fn test_gell_mann_traceless() {
        for a in 0..8 {
            let lam = gell_mann_matrix(a);
            let trace_re = lam[0][0].re + lam[1][1].re + lam[2][2].re;
            let trace_im = lam[0][0].im + lam[1][1].im + lam[2][2].im;
            assert!(
                trace_re.abs() < 1e-12 && trace_im.abs() < 1e-12,
                "λ_{} not traceless: Tr = ({:.6}, {:.6})",
                a + 1,
                trace_re,
                trace_im
            );
        }
    }

    #[test]
    fn test_gell_mann_trace_normalization() {
        // Tr(λ_a λ_b) = 2 δ_ab
        for a in 0..8 {
            let tr = gell_mann_trace_product(a, a);
            assert!(
                (tr.re - 2.0).abs() < 1e-10 && tr.im.abs() < 1e-10,
                "Tr(λ_{}²) = ({:.6}, {:.6}), expected 2",
                a + 1,
                tr.re,
                tr.im
            );
        }
    }

    #[test]
    fn test_gell_mann_orthogonality() {
        // Tr(λ_a λ_b) = 0 for a ≠ b
        for a in 0..8 {
            for b in (a + 1)..8 {
                let tr = gell_mann_trace_product(a, b);
                assert!(
                    tr.re.abs() < 1e-10 && tr.im.abs() < 1e-10,
                    "Tr(λ_{} λ_{}) = ({:.6}, {:.6}), expected 0",
                    a + 1,
                    b + 1,
                    tr.re,
                    tr.im
                );
            }
        }
    }

    #[test]
    fn test_su3_casimir_fundamental() {
        assert!((su3_casimir_fundamental() - 4.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_su3_casimir_adjoint() {
        assert!((su3_casimir_adjoint() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_structure_constants_antisymmetric() {
        // f_abc = -f_bac
        for a in 0..8 {
            for b in 0..8 {
                for c in 0..8 {
                    let fabc = su3_structure_constant(a, b, c);
                    let fbac = su3_structure_constant(b, a, c);
                    assert!(
                        (fabc + fbac).abs() < 1e-10,
                        "f_{}{}{} + f_{}{}{} = {} (not antisymmetric)",
                        a, b, c, b, a, c, fabc + fbac
                    );
                }
            }
        }
    }

    #[test]
    fn test_f123_equals_one() {
        // f_123 = 1 (0-indexed: f_012 = 1)
        let f = su3_structure_constant(0, 1, 2);
        assert!((f - 1.0).abs() < 1e-14, "f_123 = {}, expected 1", f);
    }
}
