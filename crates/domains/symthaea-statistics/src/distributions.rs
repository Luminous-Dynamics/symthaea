// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Probability distributions: normal, binomial, Poisson, Student's t, and
//! chi-square. CDFs are built on the real special functions in [`crate::special`].

use crate::special::{betai, erf, gammp, ln_gamma};
use std::f64::consts::PI;

/// Standard-normal probability density φ(z).
pub fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Normal density N(μ, σ²) at `x`.
pub fn normal_pdf_general(x: f64, mu: f64, sigma: f64) -> f64 {
    normal_pdf((x - mu) / sigma) / sigma
}

/// Standard-normal cumulative distribution Φ(z).
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Normal CDF for N(μ, σ²).
pub fn normal_cdf_general(x: f64, mu: f64, sigma: f64) -> f64 {
    normal_cdf((x - mu) / sigma)
}

/// Inverse standard-normal CDF (quantile), via Acklam's rational
/// approximation (max relative error ≈ 1.15e-9 against the true quantile).
///
/// Note: we deliberately do *not* Newton/Halley-refine against [`normal_cdf`],
/// because that CDF is erf-limited (~1e-7); refining against it would pull the
/// result toward the inverse of the *approximate* CDF and lose accuracy.
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let plow = 0.024_25;
    let phigh = 1.0 - plow;
    let x;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    x
}

/// Binomial probability mass P(X = k) for `n` trials, success prob `p`.
pub fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let ln_choose =
        ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0);
    (ln_choose + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

/// Poisson probability mass P(X = k) with rate `lambda`.
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    (-lambda + k as f64 * lambda.ln() - ln_gamma(k as f64 + 1.0)).exp()
}

/// Student's t CDF with `df` degrees of freedom.
pub fn students_t_cdf(t: f64, df: f64) -> f64 {
    // P(T ≤ t) via the regularized incomplete beta.
    let x = df / (df + t * t);
    let ib = 0.5 * betai(df / 2.0, 0.5, x);
    if t >= 0.0 { 1.0 - ib } else { ib }
}

/// Chi-square CDF with `df` degrees of freedom.
pub fn chi_square_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    gammp(df / 2.0, x / 2.0)
}


/// Natural logarithm of the standard-normal cumulative distribution Φ(z).
///
/// Unlike ln(normal_cdf(z)), this remains useful in the far lower tail. The
/// lower tail uses a continued-fraction Mills ratio; the central region uses
/// an integrated power series. The upper tail is evaluated by symmetry with
/// ln1p to avoid subtracting a rounded CDF from one.
pub fn normal_log_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::INFINITY {
        return 0.0;
    }
    if z == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if z < -2.0 {
        let a = -z;
        let mut tail = 0.0;
        // Laplace's continued fraction for the Mills ratio:
        // Phi(-a) / phi(a) = 1 / (a + 1 / (a + 2 / (a + ...))).
        for k in (1..=128).rev() {
            tail = k as f64 / (a + tail);
        }
        let mills_ratio = 1.0 / (a + tail);
        return -0.5 * a * a - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + mills_ratio.ln();
    }
    if z < 0.0 {
        // Integrate exp(-t²/2) from zero to z. This converges rapidly on
        // [-2, 0], where the alternating terms remain well scaled.
        let mut term = z;
        let mut integral = z;
        for n in 1..=80 {
            let nf = n as f64;
            term *= -(z * z / 2.0) * (2.0 * nf - 1.0) / (nf * (2.0 * nf + 1.0));
            integral += term;
            if term.abs() <= 2.0e-16 * integral.abs().max(1.0e-300) {
                break;
            }
        }
        let cdf = 0.5 + integral / (2.0 * std::f64::consts::PI).sqrt();
        return cdf.ln();
    }
    if z == 0.0 {
        return -std::f64::consts::LN_2;
    }
    (-normal_log_cdf(-z).exp()).ln_1p()
}

/// Natural logarithm of the standard-normal survival function 1 - Φ(z).
///
/// This is evaluated as log Φ(-z), never as ln(1 - normal_cdf(z)).
pub fn normal_log_survival(z: f64) -> f64 {
    normal_log_cdf(-z)
}

/// Natural logarithm of the standard-normal probability in (lower, upper].
///
/// Infinite endpoints are supported. Invalid or unordered endpoints return
/// NaN. Tail intervals use the smaller tail, avoiding catastrophic
/// cancellation when both boundaries lie on the same side of zero. Very
/// narrow finite intervals use a local density expansion.
pub fn normal_log_interval(lower: f64, upper: f64) -> f64 {
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return f64::NAN;
    }
    if lower == f64::NEG_INFINITY {
        return normal_log_cdf(upper);
    }
    if upper == f64::INFINITY {
        return normal_log_survival(lower);
    }

    let width = upper - lower;
    let midpoint = lower + width * 0.5;
    if width.is_finite() && width > 0.0
        && width <= 1.0e-5 * (1.0 + midpoint.abs())
        && midpoint.abs() < 1.0e150
    {
        let correction = ((midpoint * width).powi(2) - width * width) / 24.0;
        if correction > -1.0 {
            return -0.5 * midpoint * midpoint
                - 0.5 * (2.0 * std::f64::consts::PI).ln()
                + width.ln()
                + correction.ln_1p();
        }
    }

    if upper <= 0.0 {
        return log_difference_exp(normal_log_cdf(upper), normal_log_cdf(lower));
    }
    if lower >= 0.0 {
        return log_difference_exp(normal_log_survival(lower), normal_log_survival(upper));
    }

    // The interval straddles zero. Both excluded tails are at most one half.
    let left_tail = normal_log_cdf(lower).exp();
    let right_tail = normal_log_survival(upper).exp();
    let mass = 1.0 - left_tail - right_tail;
    if mass > 0.0 { mass.ln() } else { f64::NEG_INFINITY }
}

fn log_difference_exp(log_a: f64, log_b: f64) -> f64 {
    if log_b == f64::NEG_INFINITY {
        return log_a;
    }
    if log_b > log_a {
        return f64::NAN;
    }
    if log_a == log_b {
        return f64::NEG_INFINITY;
    }
    log_a + (-(log_b - log_a).exp()).ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn normal_log_cdf_matches_central_and_extreme_tail_landmarks() {
        let points = [
            (-10.0, -53.231_285_150_512_47),
            (-8.0, -35.013_437_159_914_56),
            (-3.0, -6.607_726_221_510_35),
            (-1.0, -1.841_021_645_009_263_4),
            (0.0, -0.693_147_180_559_945_3),
            (1.0, -0.172_753_779_023_449_9),
            (3.0, -0.001_350_809_964_748_193),
            (8.0, -6.220_960_574_271_74e-16),
        ];
        for (z, expected) in points {
            let actual = normal_log_cdf(z);
            assert!((actual - expected).abs() < 2.0e-10, "z={z}: {actual} != {expected}");
        }
        assert!(normal_log_cdf(-40.0).is_finite());
        assert_eq!(normal_log_cdf(f64::INFINITY), 0.0);
        assert_eq!(normal_log_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn normal_log_interval_is_stable_in_both_tails() {
        let left = normal_log_interval(f64::NEG_INFINITY, -8.0);
        let right = normal_log_interval(8.0, f64::INFINITY);
        assert!(left.is_finite() && right.is_finite());
        assert!((left - right).abs() < 1.0e-12);
        assert!((normal_log_interval(f64::NEG_INFINITY, 0.0).exp() - 0.5).abs() < 1.0e-14);
        assert!((normal_log_interval(0.0, f64::INFINITY).exp() - 0.5).abs() < 1.0e-14);
        let narrow = normal_log_interval(-1.0e-9, 1.0e-9).exp();
        assert!((narrow / (2.0e-9 / (2.0 * std::f64::consts::PI).sqrt()) - 1.0).abs() < 1.0e-12);
        assert!(normal_log_interval(1.0, 1.0).is_nan());
    }

    #[test]
    fn normal_cdf_landmarks() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn normal_quantile_inverts_cdf() {
        assert!((normal_quantile(0.975) - 1.959_963_98).abs() < 1e-6);
        assert!((normal_quantile(0.5)).abs() < 1e-9);
        for &p in &[0.01, 0.1, 0.4, 0.8, 0.99] {
            assert!((normal_cdf(normal_quantile(p)) - p).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn binomial_and_poisson() {
        // C(10,3) (1/2)^10 = 120/1024.
        assert!((binomial_pmf(10, 3, 0.5) - 120.0 / 1024.0).abs() < 1e-12);
        // Row sums to 1.
        let s: f64 = (0..=10).map(|k| binomial_pmf(10, k, 0.3)).sum();
        assert!((s - 1.0).abs() < 1e-10);
        // Poisson(λ=3) at k=2: e^{-3}·9/2.
        assert!((poisson_pmf(2, 3.0) - (-3.0f64).exp() * 4.5).abs() < 1e-12);
    }

    #[test]
    fn t_cdf_symmetry_and_chi_square_closed_form() {
        // t-CDF is symmetric: F(0)=0.5.
        assert!((students_t_cdf(0.0, 10.0) - 0.5).abs() < 1e-9);
        // df=1 is Cauchy: F(1) = 0.75.
        assert!((students_t_cdf(1.0, 1.0) - 0.75).abs() < 1e-6);
        // chi-square df=2: F(x) = 1 - e^{-x/2}.
        assert!((chi_square_cdf(2.0, 2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
    }
}
