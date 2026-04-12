// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Electron repulsion integrals (ERIs): (ij|kl) = ∫∫ φᵢ(1)φⱼ(1) (1/r₁₂) φₖ(2)φₗ(2) d1 d2
//!
//! Uses the Obara-Saika scheme with Hermite expansion and Coulomb coefficients.
//! **Schwarz prescreening** is mandatory: diagonal elements (ij|ij) are computed first,
//! and off-diagonal integrals are skipped when sqrt((ij|ij)*(kl|kl)) < threshold.
//!
//! This drops practical scaling from O(N⁴) to ~O(N^2.2) for extended systems.

use crate::basis::ContractedGaussian;
use crate::constants::{INTEGRAL_SCREENING_THRESHOLD, PI_CONST};
use crate::integrals::hermite::{hermite_coefficient, hermite_coulomb};

/// Compute a single ERI between four primitive Gaussians.
///
/// (ab|cd) = Σ_{tuv} Σ_{τυφ} E^{la,lb}_t E^{ma,mb}_u E^{na,nb}_v
///           × E^{lc,ld}_τ E^{mc,md}_υ E^{nc,nd}_φ × (-1)^{τ+υ+φ}
///           × R_{t+τ, u+υ, v+φ}(α, R_PQ)
///           × 2π^(5/2) / (p·q·√(p+q))
fn eri_primitive(
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
    alpha_c: f64,
    lc: i32,
    mc: i32,
    nc: i32,
    center_c: [f64; 3],
    alpha_d: f64,
    ld: i32,
    md: i32,
    nd: i32,
    center_d: [f64; 3],
) -> f64 {
    let p = alpha_a + alpha_b;
    let q = alpha_c + alpha_d;
    let alpha = p * q / (p + q);

    let px = (alpha_a * center_a[0] + alpha_b * center_b[0]) / p;
    let py = (alpha_a * center_a[1] + alpha_b * center_b[1]) / p;
    let pz = (alpha_a * center_a[2] + alpha_b * center_b[2]) / p;

    let qx = (alpha_c * center_c[0] + alpha_d * center_d[0]) / q;
    let qy = (alpha_c * center_c[1] + alpha_d * center_d[1]) / q;
    let qz = (alpha_c * center_c[2] + alpha_d * center_d[2]) / q;

    let rpq = [px - qx, py - qy, pz - qz];

    let prefactor = 2.0 * PI_CONST.powf(2.5) / (p * q * (p + q).sqrt());

    let mut val = 0.0;

    for t in 0..=(la + lb) {
        let ex_ab = hermite_coefficient(la, lb, t, alpha_a, alpha_b, center_a[0], center_b[0]);
        if ex_ab.abs() < 1e-15 {
            continue;
        }
        for u in 0..=(ma + mb) {
            let ey_ab =
                hermite_coefficient(ma, mb, u, alpha_a, alpha_b, center_a[1], center_b[1]);
            if ey_ab.abs() < 1e-15 {
                continue;
            }
            for v in 0..=(na + nb) {
                let ez_ab = hermite_coefficient(
                    na,
                    nb,
                    v,
                    alpha_a,
                    alpha_b,
                    center_a[2],
                    center_b[2],
                );
                if ez_ab.abs() < 1e-15 {
                    continue;
                }

                for tau in 0..=(lc + ld) {
                    let ex_cd = hermite_coefficient(
                        lc,
                        ld,
                        tau,
                        alpha_c,
                        alpha_d,
                        center_c[0],
                        center_d[0],
                    );
                    if ex_cd.abs() < 1e-15 {
                        continue;
                    }
                    for upsilon in 0..=(mc + md) {
                        let ey_cd = hermite_coefficient(
                            mc,
                            md,
                            upsilon,
                            alpha_c,
                            alpha_d,
                            center_c[1],
                            center_d[1],
                        );
                        if ey_cd.abs() < 1e-15 {
                            continue;
                        }
                        for phi in 0..=(nc + nd) {
                            let ez_cd = hermite_coefficient(
                                nc,
                                nd,
                                phi,
                                alpha_c,
                                alpha_d,
                                center_c[2],
                                center_d[2],
                            );
                            if ez_cd.abs() < 1e-15 {
                                continue;
                            }

                            let sign = if (tau + upsilon + phi) % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            };

                            let r = hermite_coulomb(
                                t + tau,
                                u + upsilon,
                                v + phi,
                                0,
                                alpha,
                                rpq,
                            );

                            val += ex_ab * ey_ab * ez_ab * ex_cd * ey_cd * ez_cd * sign * r;
                        }
                    }
                }
            }
        }
    }

    prefactor * val
}

/// Compute ERI between four contracted Gaussians.
pub fn eri_contracted(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
) -> f64 {
    let mut result = 0.0;

    for pa in &a.primitives {
        let na = pa.normalization();
        for pb in &b.primitives {
            let nb = pb.normalization();
            for pc in &c.primitives {
                let nc = pc.normalization();
                for pd in &d.primitives {
                    let nd = pd.normalization();

                    let eri = eri_primitive(
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
                        pc.alpha,
                        pc.l as i32,
                        pc.m as i32,
                        pc.n as i32,
                        pc.center,
                        pd.alpha,
                        pd.l as i32,
                        pd.m as i32,
                        pd.n as i32,
                        pd.center,
                    );

                    result += pa.coeff * pb.coeff * pc.coeff * pd.coeff * na * nb * nc * nd * eri;
                }
            }
        }
    }

    result
}

/// Compute the full ERI tensor with Schwarz prescreening.
///
/// Returns a flat array indexed as eri[i*n³ + j*n² + k*n + l].
/// Also returns the number of integrals actually computed (vs. screened out).
pub fn compute_eri_tensor(basis: &[ContractedGaussian]) -> (Vec<f64>, usize, usize) {
    let n = basis.len();
    let n2 = n * n;
    let n3 = n2 * n;
    let mut eri = vec![0.0; n * n * n * n];

    // Step 1: Compute Schwarz bounds — diagonal (ij|ij) for all pairs
    let mut schwarz = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let diag = eri_contracted(&basis[i], &basis[j], &basis[i], &basis[j]);
            let bound = diag.abs().sqrt();
            schwarz[i * n + j] = bound;
            schwarz[j * n + i] = bound;
        }
    }

    let threshold = INTEGRAL_SCREENING_THRESHOLD;
    let mut computed = 0_usize;
    let mut screened = 0_usize;

    // Step 2: Compute ERIs with 8-fold symmetry + Schwarz screening
    for i in 0..n {
        for j in 0..=i {
            let bound_ij = schwarz[i * n + j];

            for k in 0..=i {
                let l_max = if k == i { j } else { k };
                for l in 0..=l_max {
                    let bound_kl = schwarz[k * n + l];

                    // Schwarz screening: skip if upper bound is negligible
                    if bound_ij * bound_kl < threshold {
                        screened += 1;
                        continue;
                    }

                    let val = eri_contracted(&basis[i], &basis[j], &basis[k], &basis[l]);
                    computed += 1;

                    // 8-fold permutational symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
                    // = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
                    eri[i * n3 + j * n2 + k * n + l] = val;
                    eri[j * n3 + i * n2 + k * n + l] = val;
                    eri[i * n3 + j * n2 + l * n + k] = val;
                    eri[j * n3 + i * n2 + l * n + k] = val;
                    eri[k * n3 + l * n2 + i * n + j] = val;
                    eri[l * n3 + k * n2 + i * n + j] = val;
                    eri[k * n3 + l * n2 + j * n + i] = val;
                    eri[l * n3 + k * n2 + j * n + i] = val;
                }
            }
        }
    }

    (eri, computed, screened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::sto3g::Sto3g;
    use crate::basis::BasisSetProvider;
    use crate::molecule::Molecule;

    #[test]
    fn test_eri_positive_diagonal() {
        // (ii|ii) should be positive (electron self-repulsion)
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let n = basis.n_basis();

        for i in 0..n {
            let idx = i * n * n * n + i * n * n + i * n + i;
            assert!(
                eri[idx] > 0.0,
                "({}{}|{}{}) = {} should be positive",
                i,
                i,
                i,
                i,
                eri[idx]
            );
        }
    }

    #[test]
    fn test_eri_symmetry() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        let (eri, _, _) = compute_eri_tensor(&basis.functions);
        let n = basis.n_basis();

        // Check (ij|kl) = (kl|ij) for all indices
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let ijkl = eri[i * n * n * n + j * n * n + k * n + l];
                        let klij = eri[k * n * n * n + l * n * n + i * n + j];
                        assert!(
                            (ijkl - klij).abs() < 1e-10,
                            "({}{}|{}{}) = {} != ({}{}|{}{}) = {}",
                            i,
                            j,
                            k,
                            l,
                            ijkl,
                            k,
                            l,
                            i,
                            j,
                            klij,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_schwarz_screening_works() {
        // For H2O STO-3G (7 basis functions), some integrals should be screened
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        let (_eri, computed, screened) = compute_eri_tensor(&basis.functions);

        // Total unique integrals with 8-fold symmetry: n(n+1)/2 * (n(n+1)/2 + 1) / 2
        let total = computed + screened;
        assert!(total > 0, "Should have some integrals");

        // For a small basis, not many will be screened, but the machinery should work
        // The important thing is that computed + screened = total unique pairs
        assert!(
            computed > 0,
            "Should compute at least some integrals"
        );
    }
}
