// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hermite Gaussian expansion coefficients for Obara-Saika integrals.
//!
//! The Hermite expansion coefficient E^{ij}_t relates the product of two
//! Cartesian Gaussians to a sum over Hermite Gaussians centered at the
//! weighted midpoint.
//!
//! Reference: Obara & Saika (1986), J. Chem. Phys. 84, 3963.

/// Compute Hermite expansion coefficient E^{ij}_t using bottom-up tabulation.
///
/// The coefficient appears in the McMurchie-Davidson scheme:
/// x_A^i * x_B^j * exp(-a*(x-Ax)^2 - b*(x-Bx)^2)
///   = Σ_t E^{ij}_t * Λ_t(p, P_x)
pub fn hermite_coefficient(
    i: i32,
    j: i32,
    t: i32,
    alpha_a: f64,
    alpha_b: f64,
    xa: f64,
    xb: f64,
) -> f64 {
    if t < 0 || t > i + j {
        return 0.0;
    }

    let p = alpha_a + alpha_b;
    let mu = alpha_a * alpha_b / p;
    let xp = (alpha_a * xa + alpha_b * xb) / p;
    let xpa = xp - xa;
    let xpb = xp - xb;
    let xab = xa - xb;

    let i = i as usize;
    let j = j as usize;
    let t = t as usize;

    // Table: E[ii][jj][tt] for ii=0..=i, jj=0..=j, tt=0..=(ii+jj)
    // Flat indexing: E[ii * (j+1) * max_t + jj * max_t + tt]
    let max_t = i + j + 1;
    let stride_j = max_t;
    let stride_i = (j + 1) * max_t;
    let size = (i + 1) * stride_i;
    let mut e = vec![0.0_f64; size];

    // Base case: E[0][0][0] = exp(-μ * (xa-xb)²)
    e[0] = (-mu * xab * xab).exp();

    // Build up i (increment along first index)
    for ii in 1..=i {
        for tt in 0..=(ii + 0).min(max_t - 1) {
            let tt_i = tt as i32;
            let mut val = xpa * get_e(&e, ii - 1, 0, tt, stride_i, stride_j, max_t);
            if tt_i >= 1 {
                val += (1.0 / (2.0 * p)) * get_e(&e, ii - 1, 0, tt - 1, stride_i, stride_j, max_t);
            }
            if tt + 1 < max_t {
                val += (tt_i + 1) as f64 * get_e(&e, ii - 1, 0, tt + 1, stride_i, stride_j, max_t);
            }
            e[ii * stride_i + 0 * stride_j + tt] = val;
        }
    }

    // Build up j (increment along second index) for each i
    for ii in 0..=i {
        for jj in 1..=j {
            for tt in 0..=(ii + jj).min(max_t - 1) {
                let tt_i = tt as i32;
                let mut val = xpb * get_e(&e, ii, jj - 1, tt, stride_i, stride_j, max_t);
                if tt_i >= 1 {
                    val += (1.0 / (2.0 * p))
                        * get_e(&e, ii, jj - 1, tt - 1, stride_i, stride_j, max_t);
                }
                if tt + 1 < max_t {
                    val += (tt_i + 1) as f64
                        * get_e(&e, ii, jj - 1, tt + 1, stride_i, stride_j, max_t);
                }
                e[ii * stride_i + jj * stride_j + tt] = val;
            }
        }
    }

    e[i * stride_i + j * stride_j + t]
}

fn get_e(
    e: &[f64],
    i: usize,
    j: usize,
    t: usize,
    stride_i: usize,
    stride_j: usize,
    max_t: usize,
) -> f64 {
    let idx = i * stride_i + j * stride_j + t;
    if t >= max_t || idx >= e.len() {
        return 0.0;
    }
    e[idx]
}

/// Compute Hermite Coulomb integral coefficient R^n_{tuv} for nuclear/ERI integrals.
///
/// Uses bottom-up tabulation with flat Vec to avoid stack overflow.
pub fn hermite_coulomb(t: i32, u: i32, v: i32, n: u32, p: f64, rpc: [f64; 3]) -> f64 {
    if t < 0 || u < 0 || v < 0 {
        return 0.0;
    }

    use crate::integrals::boys::boys_function;

    let rpc_sq = rpc[0] * rpc[0] + rpc[1] * rpc[1] + rpc[2] * rpc[2];
    let t = t as usize;
    let u = u as usize;
    let v = v as usize;
    let n = n as usize;

    // Maximum Boys function order needed
    let max_n = n + t + u + v;

    // Flat table: R[tt][uu][vv][nn]
    // Dimensions: (t+1) x (u+1) x (v+1) x (max_n+1)
    let dim_n = max_n + 1;
    let dim_v = v + 1;
    let dim_u = u + 1;
    let stride_v = dim_n;
    let stride_u = dim_v * dim_n;
    let stride_t = dim_u * dim_v * dim_n;
    let size = (t + 1) * stride_t;
    let mut r = vec![0.0_f64; size];

    // Fill base cases: R[0][0][0][m] for all m
    for m in 0..=max_n {
        r[m] = (-2.0 * p).powi(m as i32) * boys_function(m as u32, p * rpc_sq);
    }

    // Build up t (x-axis)
    for tt in 1..=t {
        for m in 0..=(max_n - tt) {
            let mut val = rpc[0] * r[(tt - 1) * stride_t + m + 1];
            if tt >= 2 {
                val += (tt - 1) as f64 * r[(tt - 2) * stride_t + m + 1];
            }
            r[tt * stride_t + m] = val;
        }
    }

    // Build up u (y-axis)
    for uu in 1..=u {
        for tt in 0..=t {
            for m in 0..=(max_n - tt - uu) {
                let mut val = rpc[1] * r[tt * stride_t + (uu - 1) * stride_u + m + 1];
                if uu >= 2 {
                    val += (uu - 1) as f64 * r[tt * stride_t + (uu - 2) * stride_u + m + 1];
                }
                r[tt * stride_t + uu * stride_u + m] = val;
            }
        }
    }

    // Build up v (z-axis)
    for vv in 1..=v {
        for tt in 0..=t {
            for uu in 0..=u {
                for m in 0..=(max_n - tt - uu - vv) {
                    let mut val =
                        rpc[2] * r[tt * stride_t + uu * stride_u + (vv - 1) * stride_v + m + 1];
                    if vv >= 2 {
                        val += (vv - 1) as f64
                            * r[tt * stride_t + uu * stride_u + (vv - 2) * stride_v + m + 1];
                    }
                    r[tt * stride_t + uu * stride_u + vv * stride_v + m] = val;
                }
            }
        }
    }

    r[t * stride_t + u * stride_u + v * stride_v + n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e00_same_center() {
        // E^{00}_0 with A=B: exp(0) = 1
        let e = hermite_coefficient(0, 0, 0, 1.0, 1.0, 0.0, 0.0);
        assert!((e - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_e00_separated() {
        // E^{00}_0 with A=0, B=1, alpha_a=alpha_b=1: exp(-0.5 * 1) = exp(-0.5)
        let e = hermite_coefficient(0, 0, 0, 1.0, 1.0, 0.0, 1.0);
        let expected = (-0.5_f64).exp();
        assert!((e - expected).abs() < 1e-12);
    }

    #[test]
    fn test_out_of_range() {
        // E^{00}_1 = 0 (t > i+j)
        assert_eq!(hermite_coefficient(0, 0, 1, 1.0, 1.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_hermite_coefficient_p_orbital() {
        // E^{10}_0 should be nonzero for p-orbital
        let e = hermite_coefficient(1, 0, 0, 1.0, 1.0, 0.0, 1.0);
        assert!(e.abs() > 1e-10, "E^{{10}}_0 = {} should be nonzero", e);
    }
}
