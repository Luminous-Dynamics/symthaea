// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared mathematical utilities for qualia confidence benchmarks.
//!
//! Provides Lempel-Ziv complexity, curve fitting, and basic statistics
//! used across the consciousness evaluation benchmarks.

/// Deterministic jitter from seed: maps to [-amplitude, +amplitude].
///
/// Uses the same hash pattern as `gwt_asphyxiation.rs` for consistency.
pub fn jitter_from_seed(seed: u64, amplitude: f64) -> f64 {
    let h = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x6A09E667);
    let frac = (h >> 11) as f64 / (1u64 << 53) as f64;
    (frac - 0.5) * 2.0 * amplitude
}

/// Deterministic float in [0, 1) from seed.
pub fn float_from_seed(seed: u64) -> f64 {
    let h = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x6A09E667);
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Lempel-Ziv 76 complexity: count of distinct substrings found by sequential parsing.
///
/// Standard LZ76 algorithm: scan forward extending the current substring until a
/// novel pattern is found, then increment the counter and start a new substring.
pub fn lempel_ziv_76(sequence: &[bool]) -> usize {
    let n = sequence.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 1;
    let mut i = 0; // start of current component
    let mut k = 1; // length being tested

    while i + k <= n {
        // Check if sequence[i..i+k] appears in sequence[0..i+k-1]
        let sub = &sequence[i..i + k];
        let search_end = i + k - 1;

        let mut found = false;
        if search_end > 0 {
            for start in 0..search_end {
                if start + k <= search_end && &sequence[start..start + k] == sub {
                    found = true;
                    break;
                }
            }
        }

        if found {
            k += 1;
            if i + k > n {
                complexity += 1;
                break;
            }
        } else {
            complexity += 1;
            i += k;
            k = 1;
        }
    }

    complexity
}

/// Normalized Lempel-Ziv complexity: LZ76 / (n / log2(n)), range approximately [0, 1].
///
/// The normalizer n/log2(n) is the theoretical maximum for random binary sequences.
pub fn normalized_lz(sequence: &[bool]) -> f64 {
    let n = sequence.len();
    if n < 2 {
        return 0.0;
    }
    let raw = lempel_ziv_76(sequence) as f64;
    let normalizer = n as f64 / (n as f64).log2();
    (raw / normalizer).min(1.0)
}

/// Sigmoid function: L / (1 + exp(-k * (x - x0)))
fn sigmoid(x: f64, l: f64, k: f64, x0: f64) -> f64 {
    l / (1.0 + (-k * (x - x0)).exp())
}

/// Fit a sigmoid curve y = L / (1 + exp(-k*(x - x0))) via grid search.
///
/// Returns (L, k, x0, r_squared). Grid search with one refinement pass.
/// Adequate for small data sets (e.g., 11 noise levels).
pub fn sigmoid_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();

    if ss_tot < 1e-15 {
        return (y_mean, 0.0, (x_min + x_max) / 2.0, 1.0);
    }

    let steps = 20usize;
    let mut best = (y_max, 1.0, (x_min + x_max) / 2.0, f64::NEG_INFINITY);

    // Grid search: 20^3 = 8000 evaluations
    for li in 0..steps {
        let l = y_min + (y_max - y_min) * (li as f64 + 0.5) / steps as f64;
        for ki in 0..steps {
            // k can be negative (decreasing sigmoid) or positive
            let k = -50.0 + 100.0 * (ki as f64 + 0.5) / steps as f64;
            for x0i in 0..steps {
                let x0 = x_min + (x_max - x_min) * (x0i as f64 + 0.5) / steps as f64;

                let ss_res: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (yi - sigmoid(xi, l, k, x0)).powi(2))
                    .sum();
                let r2 = 1.0 - ss_res / ss_tot;

                if r2 > best.3 {
                    best = (l, k, x0, r2);
                }
            }
        }
    }

    // Refinement pass around best parameters
    let (bl, bk, bx0, _) = best;
    let l_range = (y_max - y_min) / steps as f64;
    let k_range = 100.0 / steps as f64;
    let x0_range = (x_max - x_min) / steps as f64;

    for li in 0..steps {
        let l = bl - l_range + 2.0 * l_range * (li as f64 + 0.5) / steps as f64;
        for ki in 0..steps {
            let k = bk - k_range + 2.0 * k_range * (ki as f64 + 0.5) / steps as f64;
            for x0i in 0..steps {
                let x0 = bx0 - x0_range + 2.0 * x0_range * (x0i as f64 + 0.5) / steps as f64;

                let ss_res: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (yi - sigmoid(xi, l, k, x0)).powi(2))
                    .sum();
                let r2 = 1.0 - ss_res / ss_tot;

                if r2 > best.3 {
                    best = (l, k, x0, r2);
                }
            }
        }
    }

    best
}

/// Fit a linear model y = slope*x + intercept via ordinary least squares.
///
/// Returns (slope, intercept, r_squared).
pub fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }

    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let ss_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum();
    let ss_xx: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    if ss_xx < 1e-15 || ss_tot < 1e-15 {
        return (0.0, y_mean, if ss_tot < 1e-15 { 1.0 } else { 0.0 });
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    (slope, intercept, r2)
}

/// Coefficient of variation: std_dev / mean.
pub fn coefficient_of_variation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if mean.abs() < 1e-15 {
        return 0.0;
    }
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt() / mean.abs()
}

/// Pearson product-moment correlation coefficient between two series.
///
/// Returns 0.0 if either series has zero variance or lengths differ.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return 0.0;
    }
    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    if ss_xx < 1e-15 || ss_yy < 1e-15 {
        return 0.0;
    }
    ss_xy / (ss_xx * ss_yy).sqrt()
}

/// Autocorrelation at a given lag.
///
/// Returns Pearson correlation between values[0..n-lag] and values[lag..n].
pub fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    let n = values.len();
    if lag >= n || n < 3 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
    if variance < 1e-15 {
        return 0.0;
    }

    let covariance: f64 = (0..n - lag)
        .map(|i| (values[i] - mean) * (values[i + lag] - mean))
        .sum();

    covariance / variance
}

/// Inverse normal CDF (probit function) via rational approximation.
///
/// Abramowitz & Stegun 26.2.23. Accurate to ~4.5×10⁻⁴. Input clamped to
/// (0.0001, 0.9999) to avoid infinities at 0 and 1.
pub fn inverse_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(0.0001, 0.9999);
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p.ln()).sqrt();
    // Rational approximation coefficients (A&S 26.2.23)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
}

/// Signal detection d-prime from hit rate and false alarm rate.
///
/// Applies 1/(2N) correction when rates are exactly 0.0 or 1.0 to avoid
/// infinite z-scores (Macmillan & Creelman, 2005).
///
/// Returns `(d_prime, criterion_c)`:
/// - d' > 0 means discriminability (higher = better separation)
/// - c < 0 means liberal bias (observer says "yes" more than ideal)
/// - c > 0 means conservative bias
pub fn sdt_dprime(
    hit_rate: f64,
    false_alarm_rate: f64,
    n_signal: usize,
    n_noise: usize,
) -> (f64, f64) {
    // Apply 1/(2N) correction for extreme rates
    let hr = if hit_rate <= 0.0 || hit_rate >= 1.0 {
        let n = n_signal.max(1) as f64;
        hit_rate.clamp(0.5 / n, 1.0 - 0.5 / n)
    } else {
        hit_rate
    };
    let far = if false_alarm_rate <= 0.0 || false_alarm_rate >= 1.0 {
        let n = n_noise.max(1) as f64;
        false_alarm_rate.clamp(0.5 / n, 1.0 - 0.5 / n)
    } else {
        false_alarm_rate
    };

    let z_hr = inverse_normal_cdf(hr);
    let z_far = inverse_normal_cdf(far);

    let d_prime = z_hr - z_far;
    let criterion_c = -(z_hr + z_far) / 2.0;

    (d_prime, criterion_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz_constant_sequence() {
        // All-same: very low complexity
        let seq = vec![false; 100];
        let c = lempel_ziv_76(&seq);
        assert!(
            c <= 5,
            "Constant sequence should have very low complexity: {c}"
        );
    }

    #[test]
    fn test_lz_alternating() {
        // 0101... has low complexity
        let seq: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let c = lempel_ziv_76(&seq);
        assert!(c < 20, "Alternating should be low complexity: {c}");
    }

    #[test]
    fn test_lz_random_higher_than_constant() {
        // Pseudo-random should have higher complexity than constant
        let constant = vec![false; 100];
        let random: Vec<bool> = (0..100u64)
            .map(|i| float_from_seed(i * 7 + 13) > 0.5)
            .collect();
        let c_const = lempel_ziv_76(&constant);
        let c_random = lempel_ziv_76(&random);
        assert!(
            c_random > c_const,
            "Random should be more complex than constant: random={c_random}, constant={c_const}"
        );
    }

    #[test]
    fn test_normalized_lz_range() {
        let seq: Vec<bool> = (0..100u64)
            .map(|i| float_from_seed(i * 7 + 13) > 0.5)
            .collect();
        let nlz = normalized_lz(&seq);
        assert!(nlz > 0.0 && nlz <= 1.0, "Normalized LZ out of range: {nlz}");
    }

    #[test]
    fn test_sigmoid_fit_recovery() {
        // Generate sigmoid data: y = 1.0 / (1 + exp(-10*(x - 0.5)))
        let x: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 1.0 / (1.0 + (-10.0 * (xi - 0.5)).exp()))
            .collect();
        let (l, k, x0, r2) = sigmoid_fit(&x, &y);
        assert!(
            r2 > 0.95,
            "Sigmoid fit should recover high R²: r²={r2}, L={l}, k={k}, x0={x0}"
        );
    }

    #[test]
    fn test_linear_fit_recovery() {
        let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (slope, intercept, r2) = linear_fit(&x, &y);
        assert!((slope - 2.0).abs() < 0.01, "Slope should be ~2: {slope}");
        assert!(
            (intercept - 1.0).abs() < 0.1,
            "Intercept should be ~1: {intercept}"
        );
        assert!(r2 > 0.999, "Perfect linear data should have R²≈1: {r2}");
    }

    #[test]
    fn test_coefficient_of_variation() {
        let values = vec![10.0, 10.0, 10.0, 10.0];
        assert!(coefficient_of_variation(&values).abs() < 1e-10);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cv = coefficient_of_variation(&values);
        assert!(cv > 0.3 && cv < 0.6, "CV should be moderate: {cv}");
    }

    #[test]
    fn test_autocorrelation_zero_lag() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ac = autocorrelation(&values, 0);
        assert!(
            (ac - 1.0).abs() < 1e-10,
            "Lag-0 autocorrelation should be 1.0: {ac}"
        );
    }

    #[test]
    fn test_autocorrelation_random_low() {
        let values: Vec<f64> = (0..100u64).map(|i| float_from_seed(i * 31 + 7)).collect();
        let ac = autocorrelation(&values, 1);
        assert!(
            ac.abs() < 0.3,
            "Random data should have low lag-1 autocorrelation: {ac}"
        );
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect positive: {r}");

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 1e-10, "Perfect negative: {r_neg}");

        // Zero variance returns 0
        let constant = vec![5.0; 5];
        assert_eq!(pearson_correlation(&x, &constant), 0.0);

        // Uncorrelated (approximately)
        let noisy: Vec<f64> = (0..100u64).map(|i| float_from_seed(i * 17 + 3)).collect();
        let noisy2: Vec<f64> = (0..100u64).map(|i| float_from_seed(i * 31 + 7)).collect();
        let r_noise = pearson_correlation(&noisy, &noisy2);
        assert!(
            r_noise.abs() < 0.3,
            "Independent series should be ~0: {r_noise}"
        );
    }

    #[test]
    fn test_inverse_normal_cdf_symmetry() {
        // z(0.5) should be 0
        let z = inverse_normal_cdf(0.5);
        assert!(z.abs() < 0.001, "z(0.5) should be ~0: {z}");

        // Symmetric: z(p) = -z(1-p)
        for &p in &[0.1, 0.2, 0.3, 0.4] {
            let z_low = inverse_normal_cdf(p);
            let z_high = inverse_normal_cdf(1.0 - p);
            assert!(
                (z_low + z_high).abs() < 0.01,
                "z({p}) + z({}) should be ~0: {} + {} = {}",
                1.0 - p,
                z_low,
                z_high,
                z_low + z_high
            );
        }

        // Known values: z(0.975) ≈ 1.96, z(0.84) ≈ 1.0
        let z_975 = inverse_normal_cdf(0.975);
        assert!(
            (z_975 - 1.96).abs() < 0.01,
            "z(0.975) should be ~1.96: {z_975}"
        );
        let z_84 = inverse_normal_cdf(0.84);
        assert!((z_84 - 1.0).abs() < 0.02, "z(0.84) should be ~1.0: {z_84}");
    }

    #[test]
    fn test_sdt_dprime() {
        // Perfect discriminator: hit=1.0, FA=0.0 → very high d'
        let (dp, _) = sdt_dprime(1.0, 0.0, 100, 100);
        assert!(
            dp > 4.0,
            "Perfect discriminator should have very high d': {dp}"
        );

        // Chance performance: hit≈FA → d'≈0
        let (dp_chance, _) = sdt_dprime(0.50, 0.50, 100, 100);
        assert!(
            dp_chance.abs() < 0.1,
            "Chance should give d'≈0: {dp_chance}"
        );

        // Liberal bias: high hit + high FA → negative criterion
        let (_, c_liberal) = sdt_dprime(0.90, 0.60, 100, 100);
        assert!(
            c_liberal < 0.0,
            "Liberal observer should have c<0: {c_liberal}"
        );

        // Conservative bias: low hit + low FA → positive criterion
        let (_, c_conservative) = sdt_dprime(0.40, 0.10, 100, 100);
        assert!(
            c_conservative > 0.0,
            "Conservative observer should have c>0: {c_conservative}"
        );
    }

    #[test]
    fn test_jitter_bounded() {
        for seed in 0..1000u64 {
            let j = jitter_from_seed(seed, 0.05);
            assert!(
                j.abs() <= 0.05 + 1e-10,
                "Jitter should be bounded: seed={seed}, jitter={j}"
            );
        }
    }

    #[test]
    fn test_float_from_seed_range() {
        for seed in 0..1000u64 {
            let f = float_from_seed(seed);
            assert!(f >= 0.0 && f < 1.0, "Float should be in [0,1): {f}");
        }
    }
}
