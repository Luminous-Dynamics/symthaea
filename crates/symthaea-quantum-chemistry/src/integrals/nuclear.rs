// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear attraction integrals V_ij = ⟨φᵢ|-Z/r_C|φⱼ⟩.
//!
//! Uses Hermite Coulomb coefficients R^n_{tuv} which are built on the Boys function.

use crate::basis::ContractedGaussian;
use crate::constants::PI_CONST;
use crate::integrals::hermite::{hermite_coefficient, hermite_coulomb};
use crate::molecule::Atom;

/// Nuclear attraction integral between two primitives for a single nucleus.
fn nuclear_primitive(
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
    nucleus: [f64; 3],
    z_nucleus: f64,
) -> f64 {
    let p = alpha_a + alpha_b;
    let px = (alpha_a * center_a[0] + alpha_b * center_b[0]) / p;
    let py = (alpha_a * center_a[1] + alpha_b * center_b[1]) / p;
    let pz = (alpha_a * center_a[2] + alpha_b * center_b[2]) / p;

    let rpc = [px - nucleus[0], py - nucleus[1], pz - nucleus[2]];

    let mut val = 0.0;

    for t in 0..=(la + lb) {
        let ex = hermite_coefficient(la, lb, t, alpha_a, alpha_b, center_a[0], center_b[0]);
        if ex.abs() < 1e-15 {
            continue;
        }

        for u in 0..=(ma + mb) {
            let ey = hermite_coefficient(ma, mb, u, alpha_a, alpha_b, center_a[1], center_b[1]);
            if ey.abs() < 1e-15 {
                continue;
            }

            for v in 0..=(na + nb) {
                let ez = hermite_coefficient(na, nb, v, alpha_a, alpha_b, center_a[2], center_b[2]);
                if ez.abs() < 1e-15 {
                    continue;
                }

                let r = hermite_coulomb(t, u, v, 0, p, rpc);
                val += ex * ey * ez * r;
            }
        }
    }

    -z_nucleus * (2.0 * PI_CONST / p) * val
}

/// Nuclear attraction integral between two contracted Gaussians for a single nucleus.
pub fn nuclear_attraction_integral(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    nucleus: &Atom,
) -> f64 {
    let mut result = 0.0;
    let z = nucleus.atomic_number as f64;

    for pa in &a.primitives {
        let na = pa.normalization();
        for pb in &b.primitives {
            let nb = pb.normalization();

            let v = nuclear_primitive(
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
                nucleus.position,
                z,
            );

            result += pa.coeff * pb.coeff * na * nb * v;
        }
    }

    result
}

/// Build the complete nuclear attraction matrix V for all nuclei.
pub fn nuclear_matrix(basis: &[ContractedGaussian], atoms: &[Atom]) -> Vec<f64> {
    let n = basis.len();
    let mut v = vec![0.0; n * n];

    for atom in atoms {
        for i in 0..n {
            for j in i..n {
                let val = nuclear_attraction_integral(&basis[i], &basis[j], atom);
                v[i * n + j] += val;
                if i != j {
                    v[j * n + i] += val;
                }
            }
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;

    #[test]
    fn test_nuclear_negative_diagonal() {
        // Nuclear attraction should be negative (attractive potential)
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let v = nuclear_matrix(&basis.functions, &h2.atoms);
        let n = basis.n_basis();

        for i in 0..n {
            assert!(
                v[i * n + i] < 0.0,
                "V[{},{}] = {} should be negative",
                i,
                i,
                v[i * n + i]
            );
        }
    }

    #[test]
    fn test_nuclear_symmetry() {
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        let v = nuclear_matrix(&basis.functions, &water.atoms);
        let n = basis.n_basis();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (v[i * n + j] - v[j * n + i]).abs() < 1e-10,
                    "V not symmetric at ({}, {}): {} vs {}",
                    i,
                    j,
                    v[i * n + j],
                    v[j * n + i],
                );
            }
        }
    }
}
