// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hypothesis testing and confidence intervals.

use crate::descriptive::{mean, variance};
use crate::distributions::{chi_square_cdf, normal_quantile, students_t_cdf};

/// Result of a t-test: the statistic, its degrees of freedom, and the
/// two-sided p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTest {
    pub t: f64,
    pub df: f64,
    pub p_two_sided: f64,
}

/// A closed interval estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
}

/// Two-sided p-value from a t-statistic: 2·P(T ≥ |t|).
fn two_sided_t_p(t: f64, df: f64) -> f64 {
    2.0 * (1.0 - students_t_cdf(t.abs(), df))
}

/// One-sample t-test of H₀: mean = `mu0`. Needs ≥ 2 samples.
pub fn one_sample_t_test(xs: &[f64], mu0: f64) -> Option<TTest> {
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let m = mean(xs)?;
    let v = variance(xs)?;
    let se = (v / n as f64).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (m - mu0) / se;
    let df = n as f64 - 1.0;
    Some(TTest {
        t,
        df,
        p_two_sided: two_sided_t_p(t, df),
    })
}

/// Welch's two-sample t-test (does not assume equal variances). Needs ≥ 2 in
/// each group.
pub fn welch_t_test(xs: &[f64], ys: &[f64]) -> Option<TTest> {
    if xs.len() < 2 || ys.len() < 2 {
        return None;
    }
    let (nx, ny) = (xs.len() as f64, ys.len() as f64);
    let (mx, my) = (mean(xs)?, mean(ys)?);
    let (vx, vy) = (variance(xs)?, variance(ys)?);
    let sx = vx / nx;
    let sy = vy / ny;
    let se = (sx + sy).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (mx - my) / se;
    // Welch–Satterthwaite degrees of freedom.
    let df = (sx + sy).powi(2) / (sx.powi(2) / (nx - 1.0) + sy.powi(2) / (ny - 1.0));
    Some(TTest {
        t,
        df,
        p_two_sided: two_sided_t_p(t, df),
    })
}

/// Result of a chi-square goodness-of-fit test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquare {
    pub statistic: f64,
    pub df: f64,
    pub p_value: f64,
}

/// Pearson chi-square goodness-of-fit: Σ (observed − expected)² / expected.
/// `df = categories − 1`. Expected counts must be positive and equal length.
pub fn chi_square_gof(observed: &[f64], expected: &[f64]) -> Option<ChiSquare> {
    if observed.len() != expected.len() || observed.len() < 2 {
        return None;
    }
    if expected.iter().any(|&e| e <= 0.0) {
        return None;
    }
    let statistic: f64 = observed
        .iter()
        .zip(expected)
        .map(|(&o, &e)| (o - e).powi(2) / e)
        .sum();
    let df = observed.len() as f64 - 1.0;
    Some(ChiSquare {
        statistic,
        df,
        p_value: 1.0 - chi_square_cdf(statistic, df),
    })
}

/// Confidence interval for a mean using the normal (z) approximation — use when
/// σ is known or n is large. `confidence` e.g. 0.95.
pub fn z_confidence_interval_mean(xs: &[f64], sigma: f64, confidence: f64) -> Option<Interval> {
    let n = xs.len();
    if n == 0 || !(0.0..1.0).contains(&confidence) {
        return None;
    }
    let m = mean(xs)?;
    let z = normal_quantile(0.5 + confidence / 2.0);
    let half = z * sigma / (n as f64).sqrt();
    Some(Interval {
        low: m - half,
        high: m + half,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sample_t_recovers_known_statistic() {
        // Mean 5, want t against mu0=3. n=5, values chosen for a clean variance.
        let xs = [4.0, 5.0, 6.0, 4.0, 6.0];
        let r = one_sample_t_test(&xs, 3.0).unwrap();
        // mean=5, var=1.0, se=sqrt(1/5)=0.4472 → t=(5-3)/0.4472=4.472.
        assert!((r.t - 4.472_136).abs() < 1e-5, "t={}", r.t);
        assert_eq!(r.df, 4.0);
        assert!(r.p_two_sided > 0.0 && r.p_two_sided < 0.05);
    }

    #[test]
    fn identical_groups_give_zero_t_and_p_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let r = welch_t_test(&a, &a).unwrap();
        assert!(r.t.abs() < 1e-12);
        assert!((r.p_two_sided - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chi_square_fair_die() {
        // A perfectly fair observation gives statistic 0, p=1.
        let obs = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let exp = [10.0; 6];
        let r = chi_square_gof(&obs, &exp).unwrap();
        assert!(r.statistic.abs() < 1e-12);
        assert_eq!(r.df, 5.0);
        assert!((r.p_value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn z_interval_width() {
        // 95% CI for mean 0, sigma 1, n=100 → ±1.96·0.1.
        let xs = [0.0; 100];
        let ci = z_confidence_interval_mean(&xs, 1.0, 0.95).unwrap();
        assert!((ci.high - 0.196).abs() < 1e-3, "high={}", ci.high);
        assert!((ci.low + 0.196).abs() < 1e-3);
    }
}
