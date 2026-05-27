// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Effective population size (Ne) calculations.
//!
//! Ne determines the rate of genetic drift and loss of heterozygosity.
//! Several formulas are provided for different demographic scenarios:
//! sex ratio imbalance, family size variance, and fluctuating population size.

/// Effective population size from census count (trivial: Ne = N).
pub fn ne_census(n: usize) -> f64 {
    n as f64
}

/// Effective population size accounting for unequal sex ratio.
///
/// Ne = 4 * Nf * Nm / (Nf + Nm)
///
/// When sex ratio is 1:1, Ne = N. With imbalanced ratios, Ne < N.
/// Returns 0.0 if either sex count is zero.
pub fn ne_sex_ratio(n_females: usize, n_males: usize) -> f64 {
    let nf = n_females as f64;
    let nm = n_males as f64;
    if nf + nm == 0.0 {
        return 0.0;
    }
    (4.0 * nf * nm) / (nf + nm)
}

/// Effective population size accounting for family size variance.
///
/// Kimura & Crow formula: Ne = (N * k - 1) / (k - 1 + Vk/k)
///
/// Where k = mean offspring per individual, Vk = variance in offspring number.
/// Equal family sizes (Vk = 0) maximizes Ne (up to ~2N).
/// Returns N if mean_offspring is approximately 1.0 (to avoid division issues).
pub fn ne_family_variance(n: usize, mean_offspring: f64, variance_offspring: f64) -> f64 {
    let k = mean_offspring;
    let vk = variance_offspring;
    if k <= 0.0 {
        return 0.0;
    }
    let denominator = k - 1.0 + vk / k;
    if denominator.abs() < 1e-10 {
        return n as f64;
    }
    (n as f64 * k - 1.0) / denominator
}

/// Effective population size for a population with fluctuating size.
///
/// Uses the harmonic mean: Ne = t / Sum(1/Ni)
///
/// The harmonic mean is always <= arithmetic mean, meaning bottlenecks
/// have outsized impact on long-term Ne.
/// Returns 0.0 for empty input or if any size is zero.
pub fn ne_fluctuating(sizes: &[f64]) -> f64 {
    if sizes.is_empty() {
        return 0.0;
    }
    // Guard against zero or negative sizes
    if sizes.iter().any(|&n| n <= 0.0) {
        return 0.0;
    }
    let t = sizes.len() as f64;
    t / sizes.iter().map(|n| 1.0 / n).sum::<f64>()
}

/// Calculate the minimum viable Ne to retain a target fraction of heterozygosity.
///
/// Solves: H(t)/H(0) = (1 - 1/(2*Ne))^t for Ne.
///
/// Approximation: Ne = -t / (2 * ln(target))
///
/// For example, to retain 90% heterozygosity over 100 generations:
/// Ne = -100 / (2 * ln(0.9)) ~ 474
pub fn ne_minimum_viable(target_heterozygosity_retained: f64, generations: u32) -> f64 {
    if target_heterozygosity_retained <= 0.0 || target_heterozygosity_retained >= 1.0 {
        return f64::INFINITY;
    }
    if generations == 0 {
        return 0.0;
    }
    let t = generations as f64;
    -t / (2.0 * target_heterozygosity_retained.ln())
}

/// Calculate the 50/500 rule thresholds.
///
/// Returns (short_term_ne, long_term_ne) where:
/// - short_term_ne = 50 (prevent inbreeding depression)
/// - long_term_ne = 500 (maintain evolutionary potential)
pub fn fifty_five_hundred_rule() -> (f64, f64) {
    (50.0, 500.0)
}

/// Check if a population meets the minimum viable population threshold.
///
/// Uses the 50/500 rule: Ne >= 50 for short-term, Ne >= 500 for long-term.
pub fn meets_mvp_threshold(ne: f64, long_term: bool) -> bool {
    if long_term { ne >= 500.0 } else { ne >= 50.0 }
}

/// Ratio of effective to census population size.
///
/// Typically Ne/N ranges from 0.1 to 0.5 in natural populations.
/// A ratio near 1.0 indicates ideal breeding conditions.
pub fn ne_to_census_ratio(ne: f64, census_n: usize) -> f64 {
    if census_n == 0 {
        return 0.0;
    }
    ne / census_n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ne_census() {
        assert_eq!(ne_census(100), 100.0);
        assert_eq!(ne_census(0), 0.0);
        assert_eq!(ne_census(1), 1.0);
    }

    #[test]
    fn test_ne_sex_ratio_balanced() {
        let ne = ne_sex_ratio(50, 50);
        assert!((ne - 100.0).abs() < 1e-10, "50F + 50M => Ne=100, got {ne}");
    }

    #[test]
    fn test_ne_sex_ratio_imbalanced_lower() {
        let ne = ne_sex_ratio(10, 90);
        assert!(
            ne < 100.0,
            "imbalanced sex ratio should give Ne < N, got {ne}"
        );
        assert!((ne - 36.0).abs() < 0.1, "10F + 90M => Ne=36, got {ne}");
    }

    #[test]
    fn test_ne_sex_ratio_extreme() {
        let ne = ne_sex_ratio(1, 99);
        assert!(
            ne < 5.0,
            "extreme imbalance: Ne should be very low, got {ne}"
        );
    }

    #[test]
    fn test_ne_sex_ratio_zero() {
        assert_eq!(ne_sex_ratio(0, 0), 0.0);
        assert_eq!(ne_sex_ratio(10, 0), 0.0);
        assert_eq!(ne_sex_ratio(0, 10), 0.0);
    }

    #[test]
    fn test_ne_family_variance_poisson() {
        // Poisson: variance = mean = 2.0
        let ne = ne_family_variance(100, 2.0, 2.0);
        // Ne = (100*2 - 1) / (2 - 1 + 2/2) = 199/2 = 99.5
        assert!(
            (ne - 99.5).abs() < 0.1,
            "Poisson family sizes: Ne ~ N, got {ne}"
        );
    }

    #[test]
    fn test_ne_family_variance_equal_families() {
        // Equal family sizes: variance = 0
        let ne = ne_family_variance(100, 2.0, 0.0);
        // Ne = (200 - 1) / (1 + 0) = 199
        assert!(
            ne > 100.0,
            "equal family sizes should give Ne > N, got {ne}"
        );
    }

    #[test]
    fn test_ne_family_variance_high_variance() {
        // High variance: some families much larger
        let ne = ne_family_variance(100, 2.0, 10.0);
        assert!(ne < 100.0, "high variance should reduce Ne, got {ne}");
    }

    #[test]
    fn test_ne_fluctuating_constant() {
        let sizes = vec![100.0; 10];
        let ne = ne_fluctuating(&sizes);
        assert!(
            (ne - 100.0).abs() < 1e-10,
            "constant size => Ne = N, got {ne}"
        );
    }

    #[test]
    fn test_ne_fluctuating_bottleneck() {
        let sizes = vec![100.0, 100.0, 100.0, 10.0, 100.0];
        let ne = ne_fluctuating(&sizes);
        // Harmonic mean dominated by smallest value
        assert!(ne < 100.0, "bottleneck reduces Ne, got {ne}");
        assert!(ne > 10.0, "Ne should be > minimum, got {ne}");
    }

    #[test]
    fn test_ne_fluctuating_harmonic_less_than_arithmetic() {
        let sizes = vec![50.0, 100.0, 200.0];
        let ne = ne_fluctuating(&sizes);
        let arithmetic = sizes.iter().sum::<f64>() / sizes.len() as f64;
        assert!(
            ne < arithmetic,
            "harmonic mean < arithmetic mean: {ne} vs {arithmetic}"
        );
    }

    #[test]
    fn test_ne_fluctuating_empty() {
        assert_eq!(ne_fluctuating(&[]), 0.0);
    }

    #[test]
    fn test_ne_fluctuating_zero_size() {
        assert_eq!(ne_fluctuating(&[100.0, 0.0, 50.0]), 0.0);
    }

    #[test]
    fn test_ne_minimum_viable_90_pct_100_gen() {
        let ne = ne_minimum_viable(0.9, 100);
        // Ne = -100 / (2 * ln(0.9)) ~ 474.56
        assert!(
            (ne - 474.56).abs() < 1.0,
            "90% retention over 100 r#gen: Ne ~ 475, got {ne}"
        );
    }

    #[test]
    fn test_ne_minimum_viable_95_pct_200_gen() {
        let ne = ne_minimum_viable(0.95, 200);
        // Higher retention needs larger Ne
        let ne_90 = ne_minimum_viable(0.90, 200);
        assert!(ne > ne_90, "higher retention needs larger Ne");
    }

    #[test]
    fn test_ne_minimum_viable_edge_cases() {
        assert!(ne_minimum_viable(0.0, 100).is_infinite());
        assert!(ne_minimum_viable(1.0, 100).is_infinite());
        assert_eq!(ne_minimum_viable(0.9, 0), 0.0);
    }

    #[test]
    fn test_fifty_five_hundred_rule() {
        let (short, long) = fifty_five_hundred_rule();
        assert_eq!(short, 50.0);
        assert_eq!(long, 500.0);
    }

    #[test]
    fn test_meets_mvp_threshold() {
        assert!(!meets_mvp_threshold(30.0, false));
        assert!(meets_mvp_threshold(50.0, false));
        assert!(meets_mvp_threshold(100.0, false));
        assert!(!meets_mvp_threshold(100.0, true));
        assert!(meets_mvp_threshold(500.0, true));
        assert!(meets_mvp_threshold(1000.0, true));
    }

    #[test]
    fn test_ne_to_census_ratio() {
        let ratio = ne_to_census_ratio(50.0, 200);
        assert!((ratio - 0.25).abs() < 1e-10);
        assert_eq!(ne_to_census_ratio(50.0, 0), 0.0);
    }

    #[test]
    fn test_ne_always_leq_census_for_sex_ratio() {
        for nf in [1, 5, 10, 25, 50] {
            for nm in [1, 5, 10, 25, 50] {
                let ne = ne_sex_ratio(nf, nm);
                let n = (nf + nm) as f64;
                assert!(
                    ne <= n + 1e-10,
                    "Ne should be <= N for sex ratio: Ne={ne}, N={n}"
                );
            }
        }
    }
}
