//! Shared mathematical utilities for qualia confidence benchmarks.
//!
//! Provides Lempel-Ziv complexity, curve fitting, and basic statistics
//! used across the consciousness evaluation benchmarks.

/// Deterministic jitter from seed: maps to [-amplitude, +amplitude].
///
/// Uses the same hash pattern as `gwt_asphyxiation.rs` for consistency.
pub fn jitter_from_seed(seed: u64, amplitude: f64) -> f64 {
    let h = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6A09E667);
    let frac = (h >> 11) as f64 / (1u64 << 53) as f64;
    (frac - 0.5) * 2.0 * amplitude
}

/// Deterministic float in [0, 1) from seed.
pub fn float_from_seed(seed: u64) -> f64 {
    let h = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6A09E667);
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
                if start + k <= search_end {
                    if &sequence[start..start + k] == sub {
                        found = true;
                        break;
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz_constant_sequence() {
        // All-same: very low complexity
        let seq = vec![false; 100];
        let c = lempel_ziv_76(&seq);
        assert!(c <= 5, "Constant sequence should have very low complexity: {c}");
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
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 / (1.0 + (-10.0 * (xi - 0.5)).exp())).collect();
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
        assert!((ac - 1.0).abs() < 1e-10, "Lag-0 autocorrelation should be 1.0: {ac}");
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
