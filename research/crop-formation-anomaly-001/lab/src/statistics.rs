//! Small statistical reference helpers for benchmark qualification.

/// Shannon entropy in bits for a Bernoulli variable with `ones` successes in
/// `total` observations. Returns None for an empty sample or invalid counts.
pub fn binary_entropy(ones: usize, total: usize) -> Option<f64> {
    if total == 0 || ones > total {
        return None;
    }
    let p = ones as f64 / total as f64;
    if p == 0.0 || p == 1.0 {
        return Some(0.0);
    }
    Some(-p * p.log2() - (1.0 - p) * (1.0 - p).log2())
}

/// One-sided empirical p-value for an observed maximum statistic.
///
/// Each null value must be the maximum over the *same complete search family*
/// used for the observation (all decoders, orientations, thresholds, etc.).
/// The plus-one correction prevents a zero p-value with finite simulations.
/// Exchangeability of observed and null statistics is an assumption that must
/// be justified by the experimental design.
pub fn upper_tail_max_p_value(observed_max: f64, null_maxima: &[f64]) -> Option<f64> {
    if !observed_max.is_finite()
        || null_maxima.is_empty()
        || null_maxima.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let exceedances = null_maxima
        .iter()
        .filter(|value| **value >= observed_max)
        .count();
    Some((exceedances + 1) as f64 / (null_maxima.len() + 1) as f64)
}

/// Wilson score interval for a binomial proportion.
///
/// This is intended for benchmark error-rate summaries. It does not repair
/// dependence between observations; cluster-level sampling must be used when
/// examples share an event, source, or acquisition session.
pub fn wilson_interval(successes: usize, total: usize, z: f64) -> Option<(f64, f64)> {
    if total == 0 || successes > total || !z.is_finite() || z <= 0.0 {
        return None;
    }

    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width =
        z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;

    Some(((center - half_width).max(0.0), (center + half_width).min(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_known_answers() {
        assert_eq!(binary_entropy(0, 10), Some(0.0));
        assert_eq!(binary_entropy(10, 10), Some(0.0));
        assert!((binary_entropy(5, 10).expect("valid sample") - 1.0).abs() < 1.0e-12);
        assert_eq!(binary_entropy(11, 10), None);
        assert_eq!(binary_entropy(0, 0), None);
    }

    #[test]
    fn maximum_statistic_uses_plus_one_and_counts_ties() {
        let nulls = [0.1, 0.3, 0.7, 0.9];
        assert_eq!(upper_tail_max_p_value(0.7, &nulls), Some(0.6));
        assert_eq!(upper_tail_max_p_value(1.0, &nulls), Some(0.2));
        assert_eq!(upper_tail_max_p_value(0.0, &nulls), Some(1.0));
        assert_eq!(upper_tail_max_p_value(f64::NAN, &nulls), None);
    }

    #[test]
    fn wilson_interval_contains_observed_fraction() {
        let (lower, upper) = wilson_interval(50, 100, 1.96).expect("valid sample");
        assert!(lower < 0.5);
        assert!(upper > 0.5);
    }

    #[test]
    fn wilson_rejects_invalid_inputs() {
        assert_eq!(wilson_interval(2, 1, 1.96), None);
        assert_eq!(wilson_interval(0, 0, 1.96), None);
        assert_eq!(wilson_interval(2, 3, f64::NAN), None);
    }
}
