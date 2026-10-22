// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired inference for runtime-assurance episode benchmarks.
//!
//! The unit of resampling is a paired episode/seed, never an individual control
//! tick. Conditions are analyzed separately by the companion CLI. These estimates
//! quantify experimental evidence; they do not establish safety outside the tested
//! tasks, assumptions, and evaluation distribution.

use std::collections::BTreeSet;

const ALPHA_95_ONE_SIDED: f64 = 0.05;
const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedEpisode {
    pub seed: u64,
    pub baseline_metric: f64,
    pub candidate_metric: f64,
    pub baseline_violation: bool,
    pub candidate_violation: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedSummary {
    pub n_pairs: usize,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    /// Candidate minus baseline. Positive is not universally better: interpret
    /// the sign according to the predeclared meaning of the selected metric.
    pub mean_delta: f64,
    pub median_delta: f64,
    pub bootstrap_ci95_low: f64,
    pub bootstrap_ci95_high: f64,
    pub bootstrap_replicates: usize,
    pub baseline_violations: usize,
    pub candidate_violations: usize,
    pub candidate_only_violations: usize,
    pub baseline_only_violations: usize,
    /// Exact two-sided McNemar test over discordant episode-level outcomes.
    pub mcnemar_exact_two_sided_p: f64,
    /// One-sided 95% Clopper-Pearson upper confidence limit for the event rate.
    pub baseline_violation_upper95_exact: f64,
    /// One-sided 95% Clopper-Pearson upper confidence limit for the event rate.
    pub candidate_violation_upper95_exact: f64,
}

impl PairedSummary {
    pub fn from_samples(
        samples: &[PairedEpisode],
        bootstrap_replicates: usize,
        random_seed: u64,
    ) -> Result<Self, String> {
        if samples.len() < 2 {
            return Err("at least two matched episode pairs are required".to_owned());
        }
        if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES {
            return Err(format!(
                "at least {MIN_BOOTSTRAP_REPLICATES} bootstrap replicates are required"
            ));
        }

        let mut seen_seeds = BTreeSet::new();
        for sample in samples {
            if !seen_seeds.insert(sample.seed) {
                return Err(format!("duplicate episode seed in paired sample: {}", sample.seed));
            }
            if !sample.baseline_metric.is_finite() || !sample.candidate_metric.is_finite() {
                return Err(format!(
                    "non-finite metric for episode seed {}",
                    sample.seed
                ));
            }
        }

        let n_pairs = samples.len();
        let baseline_values: Vec<f64> = samples.iter().map(|s| s.baseline_metric).collect();
        let candidate_values: Vec<f64> = samples.iter().map(|s| s.candidate_metric).collect();
        let deltas: Vec<f64> = samples
            .iter()
            .map(|s| s.candidate_metric - s.baseline_metric)
            .collect();

        if deltas.iter().any(|delta| !delta.is_finite()) {
            return Err("paired metric subtraction produced a non-finite value".to_owned());
        }

        let baseline_mean = mean(&baseline_values);
        let candidate_mean = mean(&candidate_values);
        let mean_delta = mean(&deltas);
        let mut sorted_deltas = deltas.clone();
        sorted_deltas.sort_by(f64::total_cmp);
        let median_delta = quantile_sorted(&sorted_deltas, 0.5);
        let (bootstrap_ci95_low, bootstrap_ci95_high) =
            paired_bootstrap_ci(&deltas, bootstrap_replicates, random_seed);
        for (name, value) in [
            ("baseline mean", baseline_mean),
            ("candidate mean", candidate_mean),
            ("mean paired delta", mean_delta),
            ("median paired delta", median_delta),
            ("bootstrap lower limit", bootstrap_ci95_low),
            ("bootstrap upper limit", bootstrap_ci95_high),
        ] {
            if !value.is_finite() {
                return Err(format!("derived statistic is non-finite: {name}"));
            }
        }

        let baseline_violations = samples.iter().filter(|s| s.baseline_violation).count();
        let candidate_violations = samples.iter().filter(|s| s.candidate_violation).count();
        let candidate_only_violations = samples
            .iter()
            .filter(|s| s.candidate_violation && !s.baseline_violation)
            .count();
        let baseline_only_violations = samples
            .iter()
            .filter(|s| s.baseline_violation && !s.candidate_violation)
            .count();

        Ok(Self {
            n_pairs,
            baseline_mean,
            candidate_mean,
            mean_delta,
            median_delta,
            bootstrap_ci95_low,
            bootstrap_ci95_high,
            bootstrap_replicates,
            baseline_violations,
            candidate_violations,
            candidate_only_violations,
            baseline_only_violations,
            mcnemar_exact_two_sided_p: exact_mcnemar_two_sided(
                candidate_only_violations,
                baseline_only_violations,
            ),
            baseline_violation_upper95_exact: clopper_pearson_upper95(
                baseline_violations,
                n_pairs,
            ),
            candidate_violation_upper95_exact: clopper_pearson_upper95(
                candidate_violations,
                n_pairs,
            ),
        })
    }
}

fn mean(values: &[f64]) -> f64 {
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64 * scale
}

fn quantile_sorted(values: &[f64], probability: f64) -> f64 {
    debug_assert!(!values.is_empty());
    let position = probability.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn paired_bootstrap_ci(deltas: &[f64], replicates: usize, seed: u64) -> (f64, f64) {
    let mut random_state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    let mut bootstrap_means = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let sample_weight = 1.0 / deltas.len() as f64;
        let mut normalized_total = 0.0;
        for _ in 0..deltas.len() {
            let index = (next_random(&mut random_state) % deltas.len() as u64) as usize;
            normalized_total += deltas[index] * sample_weight;
        }
        bootstrap_means.push(normalized_total);
    }
    bootstrap_means.sort_by(f64::total_cmp);
    (
        quantile_sorted(&bootstrap_means, 0.025),
        quantile_sorted(&bootstrap_means, 0.975),
    )
}

fn next_random(state: &mut u64) -> u64 {
    // xorshift64*: deterministic reproducibility, not cryptographic randomness.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn log_choose(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k)
        .map(|i| ((n - k + i) as f64).ln() - (i as f64).ln())
        .sum()
}

fn binomial_cdf_at_most(k: usize, n: usize, probability: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return 0.0;
    }

    let log_p = probability.ln();
    let log_one_minus_p = (-probability).ln_1p();
    let log_terms: Vec<f64> = (0..=k)
        .map(|i| {
            log_choose(n, i) + i as f64 * log_p + (n - i) as f64 * log_one_minus_p
        })
        .collect();
    let maximum = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (maximum + log_terms.iter().map(|value| (value - maximum).exp()).sum::<f64>().ln())
        .exp()
        .clamp(0.0, 1.0)
}

fn clopper_pearson_upper95(successes: usize, trials: usize) -> f64 {
    if trials == 0 || successes >= trials {
        return 1.0;
    }
    if successes == 0 {
        return 1.0 - ALPHA_95_ONE_SIDED.powf(1.0 / trials as f64);
    }

    // The upper one-sided exact bound solves P[X <= successes | p] = 0.05.
    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..80 {
        let midpoint = (lower + upper) / 2.0;
        if binomial_cdf_at_most(successes, trials, midpoint) > ALPHA_95_ONE_SIDED {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    (lower + upper) / 2.0
}

fn exact_mcnemar_two_sided(candidate_only: usize, baseline_only: usize) -> f64 {
    let discordant = candidate_only + baseline_only;
    if discordant == 0 {
        return 1.0;
    }
    let tail = candidate_only.min(baseline_only);
    let log_two_to_n = discordant as f64 * 2.0_f64.ln();
    let probability = (0..=tail)
        .map(|i| (log_choose(discordant, i) - log_two_to_n).exp())
        .sum::<f64>();
    (2.0 * probability).min(1.0)
}

/// Minimum independent zero-violation episodes needed for the one-sided exact
/// binomial upper limit to be no greater than `target_upper_rate`.
///
/// This is a precision-planning calculation, not evidence that the episodes are
/// independent or that the tested conditions cover the deployment distribution.
pub fn minimum_zero_violation_trials(
    target_upper_rate: f64,
    confidence_level: f64,
) -> Result<usize, String> {
    if !target_upper_rate.is_finite()
        || target_upper_rate <= 0.0
        || target_upper_rate >= 1.0
    {
        return Err("target upper rate must be finite and strictly between zero and one".to_owned());
    }
    if !confidence_level.is_finite()
        || confidence_level <= 0.0
        || confidence_level >= 1.0
    {
        return Err("confidence level must be finite and strictly between zero and one".to_owned());
    }

    let alpha = 1.0 - confidence_level;
    let required = (alpha.ln() / (1.0 - target_upper_rate).ln()).ceil();
    if !required.is_finite() || required < 1.0 || required > usize::MAX as f64 {
        return Err("required sample count is outside the supported integer range".to_owned());
    }
    Ok(required as usize)
}

#[cfg(test)]
mod tests {
    use super::{PairedEpisode, PairedSummary};

    fn fixture() -> Vec<PairedEpisode> {
        (0..30)
            .map(|seed| PairedEpisode {
                seed,
                baseline_metric: 1.0 + seed as f64 / 100.0,
                candidate_metric: 1.25 + seed as f64 / 100.0,
                baseline_violation: seed % 3 == 0,
                candidate_violation: seed % 6 == 0,
            })
            .collect()
    }

    #[test]
    fn constant_paired_effect_has_exact_bootstrap_interval() {
        let result = PairedSummary::from_samples(&fixture(), 1_000, 7).expect("valid pairs");
        assert!((result.mean_delta - 0.25).abs() < 1e-12);
        assert!((result.median_delta - 0.25).abs() < 1e-12);
        assert!((result.bootstrap_ci95_low - 0.25).abs() < 1e-12);
        assert!((result.bootstrap_ci95_high - 0.25).abs() < 1e-12);
    }

    #[test]
    fn stable_mean_handles_large_finite_metrics_without_overflow() {
        let maximum = f64::MAX;
        let samples = vec![
            PairedEpisode {
                seed: 1,
                baseline_metric: maximum,
                candidate_metric: maximum,
                baseline_violation: false,
                candidate_violation: false,
            },
            PairedEpisode {
                seed: 2,
                baseline_metric: maximum,
                candidate_metric: maximum,
                baseline_violation: false,
                candidate_violation: false,
            },
        ];
        let summary = PairedSummary::from_samples(&samples, 1_000, 11)
            .expect("finite means should remain finite");
        assert_eq!(summary.baseline_mean, maximum);
        assert_eq!(summary.candidate_mean, maximum);
        assert_eq!(summary.mean_delta, 0.0);
        assert_eq!(summary.bootstrap_ci95_low, 0.0);
        assert_eq!(summary.bootstrap_ci95_high, 0.0);
    }

    #[test]
    fn zero_observed_violations_do_not_claim_zero_risk() {
        let bound = super::clopper_pearson_upper95(0, 300);
        assert!(bound > 0.009 && bound < 0.011, "bound={bound}");
    }

    #[test]
    fn nonzero_violation_count_matches_exact_binomial_upper_reference() {
        let bound = super::clopper_pearson_upper95(3, 10);
        assert!((bound - 0.606_624_216_105_412_3).abs() < 1e-9, "bound={bound}");
    }

    #[test]
    fn exact_mcnemar_detects_strong_directional_discordance() {
        let p = super::exact_mcnemar_two_sided(10, 0);
        assert!((p - 0.001_953_125).abs() < 1e-12);
        assert_eq!(super::exact_mcnemar_two_sided(0, 0), 1.0);
    }

    #[test]
    fn zero_event_risk_precision_requires_more_runs_for_stricter_claims() {
        assert_eq!(super::minimum_zero_violation_trials(0.01, 0.95), Ok(299));
        assert_eq!(super::minimum_zero_violation_trials(0.001, 0.95), Ok(2_995));
        assert!(super::minimum_zero_violation_trials(0.0, 0.95).is_err());
    }

    #[test]
    fn duplicate_seed_fails_closed() {
        let mut samples = fixture();
        samples.push(samples[0]);
        let result = PairedSummary::from_samples(&samples, 1_000, 7);
        assert!(result.expect_err("duplicate seed must reject").contains("duplicate"));
    }

    #[test]
    fn non_finite_metrics_fail_closed() {
        let mut samples = fixture();
        samples[0].candidate_metric = f64::NAN;
        assert!(PairedSummary::from_samples(&samples, 1_000, 7).is_err());
    }
}
