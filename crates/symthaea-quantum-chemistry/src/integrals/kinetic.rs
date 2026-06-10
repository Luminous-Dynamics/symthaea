// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kinetic energy integrals T_ij = ⟨φᵢ|-½∇²|φⱼ⟩.
//!
//! Uses the relation: T_ij = -½ [b(2j+1)S(i,j) - 2b²S(i,j+2)]
//! component-wise, where S(i,j+2) means the overlap with angular momentum
//! incremented by 2 along that axis.
//!
//! More precisely, the kinetic integral is the sum of three 1D contributions
//! (x, y, z), each computed via the Obara-Saika scheme.

use crate::basis::ContractedGaussian;
use crate::integrals::overlap::overlap_primitive;

/// Kinetic energy integral between two primitives.
///
/// T = T_x * S_y * S_z + S_x * T_y * S_z + S_x * S_y * T_z
///
/// where T_x = -½ [β(2l_b+1)S(la,lb) - 2β² S(la,lb+2)]
/// and S(la,lb) is the x-component overlap.
///
/// More directly using the Obara-Saika formula:
/// T_ij = β(2l_b+1)S_ij - 2β²S(i, j with l_b→l_b+2) - ½l_b(l_b-1)S(i, j with l_b→l_b-2)
/// (per axis, multiplied by overlap on the other two axes)
fn kinetic_primitive(
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
    // The kinetic energy for each axis involves:
    // T_x = β(2lb+1) S(la,lb) - 2β² S(la,lb+2) - ½lb(lb-1) S(la,lb-2)
    // (and similarly for y, z)
    // Total T = T_x * Sy * Sz + Sx * T_y * Sz + Sx * Sy * T_z

    let term_x = kinetic_1d(
        alpha_a, la, center_a, alpha_b, lb, center_b, ma, mb, na, nb, 0,
    );
    let term_y = kinetic_1d(
        alpha_a, ma, center_a, alpha_b, mb, center_b, la, lb, na, nb, 1,
    );
    let term_z = kinetic_1d(
        alpha_a, na, center_a, alpha_b, nb, center_b, la, lb, ma, mb, 2,
    );

    term_x + term_y + term_z
}

/// Compute 1D kinetic contribution along a specific axis.
/// Returns T_axis * S_other1 * S_other2 via overlap_primitive calls.
fn kinetic_1d(
    alpha_a: f64,
    l_a: i32,
    center_a: [f64; 3],
    alpha_b: f64,
    l_b: i32,
    center_b: [f64; 3],
    // The other angular momentum components
    other_a1: i32,
    other_b1: i32,
    other_a2: i32,
    other_b2: i32,
    axis: usize,
) -> f64 {
    // Build angular momentum vectors for each term
    let build_lmn = |la_mod: i32, axis: usize, _la: i32, oa1: i32, oa2: i32| -> (i32, i32, i32) {
        match axis {
            0 => (la_mod, oa1, oa2),
            1 => (oa1, la_mod, oa2),
            _ => (oa1, oa2, la_mod),
        }
    };

    // Term 1: β(2lb+1) S(la, lb)
    let (la1, ma1, na1) = build_lmn(l_a, axis, l_a, other_a1, other_a2);
    let (lb1, mb1, nb1) = build_lmn(l_b, axis, l_b, other_b1, other_b2);
    let s_base = overlap_primitive(
        alpha_a, la1, ma1, na1, center_a, alpha_b, lb1, mb1, nb1, center_b,
    );
    let t1 = alpha_b * (2 * l_b + 1) as f64 * s_base;

    // Term 2: -2β² S(la, lb+2)
    let (la2, ma2, na2) = build_lmn(l_a, axis, l_a, other_a1, other_a2);
    let (lb2, mb2, nb2) = build_lmn(l_b + 2, axis, l_b + 2, other_b1, other_b2);
    let s_plus2 = overlap_primitive(
        alpha_a, la2, ma2, na2, center_a, alpha_b, lb2, mb2, nb2, center_b,
    );
    let t2 = -2.0 * alpha_b * alpha_b * s_plus2;

    // Term 3: -½ lb(lb-1) S(la, lb-2) — only if lb >= 2
    let t3 = if l_b >= 2 {
        let (la3, ma3, na3) = build_lmn(l_a, axis, l_a, other_a1, other_a2);
        let (lb3, mb3, nb3) = build_lmn(l_b - 2, axis, l_b - 2, other_b1, other_b2);
        let s_minus2 = overlap_primitive(
            alpha_a, la3, ma3, na3, center_a, alpha_b, lb3, mb3, nb3, center_b,
        );
        -0.5 * (l_b * (l_b - 1)) as f64 * s_minus2
    } else {
        0.0
    };

    t1 + t2 + t3
}

/// Compute kinetic energy integral between two contracted Gaussians.
pub fn kinetic_integral(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut result = 0.0;

    for pa in &a.primitives {
        let na = pa.normalization();
        for pb in &b.primitives {
            let nb = pb.normalization();

            let t = kinetic_primitive(
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

            result += pa.coeff * pb.coeff * na * nb * t;
        }
    }

    result
}

/// Build the complete kinetic energy matrix T.
pub fn kinetic_matrix(basis: &[ContractedGaussian]) -> Vec<f64> {
    let n = basis.len();
    let mut t = vec![0.0; n * n];

    for i in 0..n {
        for j in i..n {
            let val = kinetic_integral(&basis[i], &basis[j]);
            t[i * n + j] = val;
            t[j * n + i] = val;
        }
    }

    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;

    #[test]
    fn test_kinetic_positive_diagonal() {
        // Kinetic energy is always positive on the diagonal
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let t = kinetic_matrix(&basis.functions);
        let n = basis.n_basis();

        for i in 0..n {
            assert!(
                t[i * n + i] > 0.0,
                "T[{},{}] = {} should be positive",
                i,
                i,
                t[i * n + i]
            );
        }
    }

    #[test]
    fn test_kinetic_symmetry() {
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        let t = kinetic_matrix(&basis.functions);
        let n = basis.n_basis();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (t[i * n + j] - t[j * n + i]).abs() < 1e-12,
                    "T not symmetric at ({}, {})",
                    i,
                    j,
                );
            }
        }
    }
}
