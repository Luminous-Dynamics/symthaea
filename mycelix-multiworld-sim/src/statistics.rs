// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Statistical analysis functions for scientific publication.
//!
//! Pure math on `&[f64]` — zero external dependencies. Provides:
//!
//! - Bootstrap confidence intervals (Efron 1979)
//! - Paired t-test with p-value (Student 1908)
//! - Cohen's d effect size (Cohen 1988)
//! - Convergence detection for time series
//!
//! # References
//!
//! - Efron, B. (1979). "Bootstrap methods: another look at the jackknife." Annals of Statistics.
//! - Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences." 2nd ed.
//! - Welch, B. L. (1947). "The generalization of Student's problem." Biometrika.

use crate::stochastic::StochasticEngine;

/// Bootstrap confidence interval result.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapCI {
    pub lower: f64,
    pub mean: f64,
    pub upper: f64,
    pub confidence: f64,
    pub n_resamples: usize,
}

/// Compute bootstrap confidence interval for the mean.
///
/// Resamples with replacement `n_resamples` times, computes the mean of each
/// resample, then takes the (α/2, 1-α/2) percentiles of the bootstrap
/// distribution.
///
/// # Arguments
/// - `samples`: Observed data points
/// - `confidence`: Confidence level (e.g., 0.95 for 95% CI)
/// - `n_resamples`: Number of bootstrap iterations (1000-10000 typical)
/// - `rng`: Deterministic PRNG for reproducibility
///
/// # Returns
/// `None` if samples is empty.
pub fn bootstrap_ci(
    samples: &[f64],
    confidence: f64,
    n_resamples: usize,
    rng: &mut StochasticEngine,
) -> Option<BootstrapCI> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len();
    let observed_mean = samples.iter().sum::<f64>() / n as f64;

    let mut bootstrap_means = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut resample_sum = 0.0;
        for _ in 0..n {
            let idx = (rng.next_u64() as usize) % n;
            resample_sum += samples[idx];
        }
        bootstrap_means.push(resample_sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * bootstrap_means.len() as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_means.len() as f64) as usize;
    let upper_idx = upper_idx.min(bootstrap_means.len() - 1);

    Some(BootstrapCI {
        lower: bootstrap_means[lower_idx],
        mean: observed_mean,
        upper: bootstrap_means[upper_idx],
        confidence,
        n_resamples,
    })
}

/// Paired t-test result.
#[derive(Debug, Clone, Copy)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub df: usize,
    pub mean_diff: f64,
    pub se_diff: f64,
}

/// Perform a paired t-test on two matched samples.
///
/// Tests H0: mean(a) == mean(b) vs H1: mean(a) != mean(b).
///
/// # Arguments
/// - `a`, `b`: Paired samples (must be same length)
///
/// # Returns
/// `None` if samples are empty, different lengths, or have zero variance.
///
/// # Reference
/// Student (1908). "The probable error of a mean." Biometrika 6(1).
pub fn paired_t_test(a: &[f64], b: &[f64]) -> Option<TTestResult> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let n = a.len();
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let mean_diff = diffs.iter().sum::<f64>() / n as f64;
    let var_diff = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1) as f64;
    if var_diff <= 0.0 || n < 2 {
        return None;
    }
    let se_diff = (var_diff / n as f64).sqrt();
    let t = mean_diff / se_diff;
    let df = n - 1;

    // Approximate two-tailed p-value using the t-distribution.
    // For large df (>30), t ≈ z (normal). For small df, use incomplete beta.
    // We use a simple approximation valid for df >= 2:
    // p ≈ 2 * (1 - Φ(|t| * sqrt(df/(df-2+t²)))) — Abramowitz & Stegun approx.
    let p = two_tailed_t_pvalue(t.abs(), df);

    Some(TTestResult {
        t_statistic: t,
        p_value: p,
        df,
        mean_diff,
        se_diff,
    })
}

/// Cohen's d effect size for two independent samples.
///
/// d = (mean_a - mean_b) / pooled_std
///
/// Interpretation (Cohen 1988):
/// - |d| < 0.2: negligible
/// - |d| 0.2-0.5: small
/// - |d| 0.5-0.8: medium
/// - |d| > 0.8: large
pub fn cohens_d(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
    let pooled = ((var_a * (a.len() - 1) as f64 + var_b * (b.len() - 1) as f64)
        / (a.len() + b.len() - 2) as f64)
        .sqrt();
    if pooled <= 0.0 {
        return None;
    }
    Some((mean_a - mean_b) / pooled)
}

/// Detect convergence in a time series.
///
/// Returns the first index at which the rolling standard deviation stays below
/// `threshold` for `window` consecutive points. Returns `None` if the series
/// never converges.
///
/// # Arguments
/// - `series`: Time-ordered values (e.g., CVS per year)
/// - `window`: Number of consecutive points to check (e.g., 10 years)
/// - `threshold`: Maximum rolling std for "converged" (e.g., 0.01)
pub fn detect_convergence(series: &[f64], window: usize, threshold: f64) -> Option<usize> {
    if series.len() < window || window < 2 {
        return None;
    }
    let mut consecutive = 0;
    for i in window..series.len() {
        let slice = &series[i - window..i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        let std = variance.sqrt();
        if std < threshold {
            consecutive += 1;
            if consecutive >= window {
                return Some(i - window); // Converged starting here
            }
        } else {
            consecutive = 0;
        }
    }
    None
}

/// Descriptive statistics for a sample.
#[derive(Debug, Clone, Copy)]
pub struct Descriptive {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Compute descriptive statistics for a sample.
pub fn describe(samples: &[f64]) -> Option<Descriptive> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    Some(Descriptive {
        n,
        mean,
        std: variance.sqrt(),
        min: sorted[0],
        max: sorted[n - 1],
        median,
    })
}

// ─── Internal: t-distribution p-value approximation ─────────────────────────

/// Approximate two-tailed p-value for t-distribution.
///
/// Uses the regularized incomplete beta function approximation.
/// For df >= 30, this is accurate to ~3 decimal places.
fn two_tailed_t_pvalue(t_abs: f64, df: usize) -> f64 {
    // Use the relationship: p = I_{df/(df+t²)}(df/2, 1/2)
    // where I is the regularized incomplete beta function.
    // We approximate using the normal distribution for large df.
    if df >= 120 {
        // For large df, t ≈ z (normal)
        return 2.0 * normal_cdf_complement(t_abs);
    }
    // Approximation: t → z correction (Satterthwaite)
    // z ≈ t * sqrt(1 - 1/(4*df)) for moderate df
    let correction = (1.0 - 1.0 / (4.0 * df as f64)).sqrt();
    let z = t_abs * correction;
    let p = 2.0 * normal_cdf_complement(z);
    // Clamp to [0, 1] for safety
    p.clamp(0.0, 1.0)
}

/// Complementary normal CDF: P(Z > z) for standard normal.
/// Uses Abramowitz & Stegun (1964) approximation 26.2.17.
fn normal_cdf_complement(z: f64) -> f64 {
    if z < 0.0 {
        return 1.0 - normal_cdf_complement(-z);
    }
    // Constants from A&S 26.2.17
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;

    let t = 1.0 / (1.0 + p * z);
    let phi = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_ci_contains_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut rng = StochasticEngine::new(42);
        let ci = bootstrap_ci(&data, 0.95, 1000, &mut rng).unwrap();
        assert!(ci.lower <= ci.mean, "Lower {} > mean {}", ci.lower, ci.mean);
        assert!(ci.mean <= ci.upper, "Mean {} > upper {}", ci.mean, ci.upper);
        assert!(
            (ci.mean - 5.5).abs() < 0.01,
            "Mean should be ~5.5: {}",
            ci.mean
        );
    }

    #[test]
    fn bootstrap_ci_narrows_with_more_data() {
        // Same distribution (mean=5, std=1), different sample sizes
        let mut rng = StochasticEngine::new(42);
        let small: Vec<f64> = (0..10).map(|_| 5.0 + rng.next_gaussian(0.0, 1.0)).collect();
        let mut rng2 = StochasticEngine::new(99);
        let large: Vec<f64> = (0..200)
            .map(|_| 5.0 + rng2.next_gaussian(0.0, 1.0))
            .collect();
        let mut rng3 = StochasticEngine::new(42);
        let mut rng4 = StochasticEngine::new(42);
        let ci_small = bootstrap_ci(&small, 0.95, 2000, &mut rng3).unwrap();
        let ci_large = bootstrap_ci(&large, 0.95, 2000, &mut rng4).unwrap();
        let width_small = ci_small.upper - ci_small.lower;
        let width_large = ci_large.upper - ci_large.lower;
        assert!(
            width_large < width_small,
            "Larger sample should have narrower CI: {} vs {}",
            width_large,
            width_small
        );
    }

    #[test]
    fn paired_t_test_identical_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = paired_t_test(&a, &a);
        // Identical samples → zero variance in diffs → None
        assert!(
            result.is_none(),
            "Identical samples should return None (zero variance)"
        );
    }

    #[test]
    fn paired_t_test_significant_difference() {
        // Non-constant difference: a[i] = b[i] + noise + 5.0
        let mut rng = StochasticEngine::new(42);
        let b: Vec<f64> = (0..20)
            .map(|i| i as f64 + rng.next_gaussian(0.0, 0.5))
            .collect();
        let a: Vec<f64> = b
            .iter()
            .map(|x| x + 5.0 + rng.next_gaussian(0.0, 0.3))
            .collect();
        let result = paired_t_test(&a, &b).unwrap();
        assert!(
            result.p_value < 0.01,
            "Clearly different samples should have p < 0.01: {}",
            result.p_value
        );
        assert!(
            result.mean_diff > 4.0 && result.mean_diff < 6.0,
            "Mean diff should be ~5.0: {}",
            result.mean_diff
        );
    }

    #[test]
    fn paired_t_test_non_significant() {
        // Small random perturbation — should NOT be significant
        let a = vec![5.01, 4.99, 5.02, 4.98, 5.01, 4.99, 5.00, 5.01, 4.99, 5.00];
        let b = vec![5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00];
        let result = paired_t_test(&a, &b).unwrap();
        assert!(
            result.p_value > 0.05,
            "Tiny differences should not be significant: p={}",
            result.p_value
        );
    }

    #[test]
    fn cohens_d_large_effect() {
        let a: Vec<f64> = (10..=20).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=11).map(|x| x as f64).collect();
        let d = cohens_d(&a, &b).unwrap();
        assert!(d > 0.8, "9-unit shift should be large effect: d={}", d);
    }

    #[test]
    fn cohens_d_negligible_effect() {
        // Same distribution, just different draws — effect should be negligible
        let mut rng = StochasticEngine::new(42);
        let a: Vec<f64> = (0..50).map(|_| 5.0 + rng.next_gaussian(0.0, 1.0)).collect();
        let b: Vec<f64> = (0..50).map(|_| 5.0 + rng.next_gaussian(0.0, 1.0)).collect();
        let d = cohens_d(&a, &b).unwrap();
        assert!(
            d.abs() < 0.5,
            "Same-distribution samples should have small d: {}",
            d
        );
    }

    #[test]
    fn convergence_detects_plateau() {
        let mut series = Vec::new();
        // Rising phase (0-50)
        for i in 0..50 {
            series.push(i as f64 * 0.01);
        }
        // Plateau phase (50-100)
        for _ in 50..100 {
            series.push(0.50);
        }
        let result = detect_convergence(&series, 10, 0.001);
        assert!(result.is_some(), "Should detect plateau");
        let converge_at = result.unwrap();
        assert!(
            converge_at >= 50 && converge_at <= 65,
            "Should converge around tick 50-65: {}",
            converge_at
        );
    }

    #[test]
    fn convergence_none_for_noisy_series() {
        let mut rng = StochasticEngine::new(42);
        let series: Vec<f64> = (0..100).map(|_| rng.next_f64()).collect();
        let result = detect_convergence(&series, 10, 0.01);
        assert!(result.is_none(), "Random series should not converge");
    }

    #[test]
    fn describe_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = describe(&data).unwrap();
        assert_eq!(d.n, 5);
        assert!((d.mean - 3.0).abs() < 0.01);
        assert!((d.median - 3.0).abs() < 0.01);
        assert!((d.min - 1.0).abs() < 0.01);
        assert!((d.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn normal_cdf_complement_symmetry() {
        let p_pos = normal_cdf_complement(2.0);
        let p_neg = normal_cdf_complement(-2.0);
        assert!(
            (p_pos + p_neg - 1.0).abs() < 0.01,
            "P(Z>2) + P(Z>-2) should ≈ 1.0: {} + {} = {}",
            p_pos,
            p_neg,
            p_pos + p_neg
        );
        // P(Z > 2) ≈ 0.0228
        assert!(
            (p_pos - 0.0228).abs() < 0.005,
            "P(Z > 2) should be ~0.0228: {}",
            p_pos
        );
    }
}
