// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Overlap integrals S_ij = ⟨φᵢ|φⱼ⟩.
//!
//! Uses Hermite expansion coefficients: S_ij = E^{ij}_0 * (π/p)^(3/2)
//! for each Cartesian component.

use crate::basis::ContractedGaussian;
use crate::constants::PI_CONST;
use crate::integrals::hermite::hermite_coefficient;

/// Compute overlap integral between two primitive Gaussians.
///
/// S = N_a * N_b * E^{la,lb}_0(x) * E^{ma,mb}_0(y) * E^{na,nb}_0(z) * (π/p)^(3/2)
pub fn overlap_primitive(
    alpha_a: f64,
    la: i32,
    ma: i32,
    na: i32,
    center_a: [f64; 3],
    alpha_b: f64,
    lb: i32,
    mb: i32,
    nb: i32,
    center_b: [f64; 3],
) -> f64 {
    let p = alpha_a + alpha_b;

    let ex = hermite_coefficient(la, lb, 0, alpha_a, alpha_b, center_a[0], center_b[0]);
    let ey = hermite_coefficient(ma, mb, 0, alpha_a, alpha_b, center_a[1], center_b[1]);
    let ez = hermite_coefficient(na, nb, 0, alpha_a, alpha_b, center_a[2], center_b[2]);

    ex * ey * ez * (PI_CONST / p).powf(1.5)
}

/// Compute overlap integral between two contracted Gaussians.
pub fn overlap_integral(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut result = 0.0;

    for pa in &a.primitives {
        let na = pa.normalization();
        for pb in &b.primitives {
            let nb = pb.normalization();

            let s = overlap_primitive(
                pa.alpha,
                pa.l as i32,
                pa.m as i32,
                pa.n as i32,
                pa.center,
                pb.alpha,
                pb.l as i32,
                pb.m as i32,
                pb.n as i32,
                pb.center,
            );

            result += pa.coeff * pb.coeff * na * nb * s;
        }
    }

    result
}

/// Build the complete overlap matrix S for a basis set.
pub fn overlap_matrix(basis: &[ContractedGaussian]) -> Vec<f64> {
    let n = basis.len();
    let mut s = vec![0.0; n * n];

    for i in 0..n {
        for j in i..n {
            let val = overlap_integral(&basis[i], &basis[j]);
            s[i * n + j] = val;
            s[j * n + i] = val;
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;

    #[test]
    fn test_h2_overlap_diagonal() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let s = overlap_matrix(&basis.functions);

        // Diagonal should be ~1.0 (normalized basis)
        assert!(
            (s[0] - 1.0).abs() < 0.02,
            "S[0,0] = {}, expected ~1.0",
            s[0]
        );
        assert!(
            (s[3] - 1.0).abs() < 0.02,
            "S[1,1] = {}, expected ~1.0",
            s[3]
        );
    }

    #[test]
    fn test_h2_overlap_offdiag() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let s = overlap_matrix(&basis.functions);
        let n = basis.n_basis();

        // Off-diagonal S_12 should be positive and < 1 (bonded atoms)
        let s12 = s[0 * n + 1];
        assert!(s12 > 0.0 && s12 < 1.0, "S[0,1] = {}", s12);
    }

    #[test]
    fn test_overlap_symmetry() {
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        let s = overlap_matrix(&basis.functions);
        let n = basis.n_basis();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (s[i * n + j] - s[j * n + i]).abs() < 1e-12,
                    "S not symmetric at ({}, {})",
                    i,
                    j,
                );
            }
        }
    }
}
