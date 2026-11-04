// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parametric sampling and failure-accounted bootstrap for ordered UVSD counts.
//!
//! This module is deliberately separate from the nonparametric conditional
//! bootstrap in bootstrap.rs. It generates fresh class-specific categorical
//! observations from a declared Gaussian UVSD model.

use crate::{
    CategoryCount, CategoryTable, CountCorrection, FitOptions, FitStatus, Optimizer,
    UvsdEstimate, fit_with_optimizer,
};
use std::fmt;

use symthaea_statistics::normal_log_interval;

/// Errors encountered while generating a parametric UVSD category table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametricSimulationError {
    InvalidEstimate,
    CategoryCountMismatch,
    InvalidCategoryProbability,
    InvalidProbabilityMass,
    InvalidGeneratedTable,
}

impl fmt::Display for ParametricSimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidEstimate => "the generating UVSD estimate is invalid",
            Self::CategoryCountMismatch => "estimate thresholds do not match the frozen category table",
            Self::InvalidCategoryProbability => "a generated category probability is invalid",
            Self::InvalidProbabilityMass => "generated category probabilities do not sum to one",
            Self::InvalidGeneratedTable => "the generated category counts do not form a valid table",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParametricSimulationError {}

/// Generate one class-stratified table from a frozen Gaussian UVSD model.
///
/// The original signal and noise totals, category order and number of
/// categories are preserved. Category probabilities are computed from the
/// supplied estimate and fixed evidence thresholds; each class is sampled
/// independently with its own multinomial total.
///
/// This function does not fit the generating estimate and does not apply a
/// continuity correction. Callers must pass the exact intended generating
/// parameters and retain the returned seed in their experiment receipt.
pub fn simulate_uvsd_category_table(
    template: &CategoryTable,
    estimate: &UvsdEstimate,
    seed: u64,
) -> Result<CategoryTable, ParametricSimulationError> {
    let Some(validated_estimate) = UvsdEstimate::new(
        estimate.mu,
        estimate.sigma,
        estimate.thresholds.clone(),
    ) else {
        return Err(ParametricSimulationError::InvalidEstimate);
    };
    if validated_estimate.thresholds.len() + 1 != template.categories().len() {
        return Err(ParametricSimulationError::CategoryCountMismatch);
    }

    let noise_probabilities = category_probabilities(
        &validated_estimate.thresholds,
        0.0,
        1.0,
    )?;
    let signal_probabilities = category_probabilities(
        &validated_estimate.thresholds,
        validated_estimate.mu,
        validated_estimate.sigma,
    )?;

    let mut rng = SplitMix64(seed);
    let noise_counts = sample_multinomial(
        template.noise_total(),
        &noise_probabilities,
        &mut rng,
    );
    let signal_counts = sample_multinomial(
        template.signal_total(),
        &signal_probabilities,
        &mut rng,
    );

    let categories = template
        .categories()
        .iter()
        .enumerate()
        .map(|(index, row)| CategoryCount {
            order: row.order,
            signal: signal_counts[index],
            noise: noise_counts[index],
        })
        .collect();

    CategoryTable::new(categories)
        .map_err(|_| ParametricSimulationError::InvalidGeneratedTable)
}

fn category_probabilities(
    thresholds: &[f64],
    mu: f64,
    sigma: f64,
) -> Result<Vec<f64>, ParametricSimulationError> {
    let category_count = thresholds.len() + 1;
    let mut probabilities = Vec::with_capacity(category_count);

    for index in 0..category_count {
        let lower = if index == 0 {
            f64::NEG_INFINITY
        } else {
            thresholds[index - 1]
        };
        let upper = if index == thresholds.len() {
            f64::INFINITY
        } else {
            thresholds[index]
        };
        let standardized_lower = standardize_boundary(lower, mu, sigma);
        let standardized_upper = standardize_boundary(upper, mu, sigma);
        let log_probability = normal_log_interval(standardized_lower, standardized_upper);
        let probability = log_probability.exp();
        if !probability.is_finite() || probability < 0.0 {
            return Err(ParametricSimulationError::InvalidCategoryProbability);
        }
        probabilities.push(probability);
    }

    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 || (total - 1.0).abs() > 1.0e-8 {
        return Err(ParametricSimulationError::InvalidProbabilityMass);
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

fn standardize_boundary(value: f64, mu: f64, sigma: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else if value == f64::INFINITY {
        f64::INFINITY
    } else {
        (value - mu) / sigma
    }
}

fn sample_multinomial(
    total: u64,
    probabilities: &[f64],
    rng: &mut SplitMix64,
) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for probability in probabilities {
        running += probability;
        cumulative.push(running);
    }

    let mut counts = vec![0u64; probabilities.len()];
    for _ in 0..total {
        let draw = rng.next_unit_f64();
        let index = cumulative
            .partition_point(|upper| *upper <= draw)
            .min(counts.len().saturating_sub(1));
        counts[index] += 1;
    }
    counts
}

/// Controls for a parametric bootstrap over fixed signal/noise class totals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParametricBootstrapOptions {
    pub replicates: usize,
    pub seed: u64,
    pub minimum_successful_replicates: usize,
    pub minimum_successful_fraction: f64,
    pub fit_options: FitOptions,
    pub optimizer: Optimizer,
}

impl Default for ParametricBootstrapOptions {
    fn default() -> Self {
        Self {
            replicates: 2_000,
            seed: 0x5350_4543_5041_5241,
            minimum_successful_replicates: 1_600,
            minimum_successful_fraction: 0.90,
            fit_options: FitOptions::default(),
            optimizer: Optimizer::Bfgs,
        }
    }
}

/// Explicit status for parametric bootstrap interval qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametricBootstrapStatus {
    Qualified,
    InsufficientSuccessfulFits,
    NotEvaluable,
}

/// Percentile intervals from the successful parametric bootstrap fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParametricBootstrapIntervals {
    pub mu_95: [f64; 2],
    pub sigma_95: [f64; 2],
    pub d_a_95: [f64; 2],
}

/// Parametric bootstrap failure accounting, separated by fit disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParametricBootstrapFailureCounts {
    pub boundary: usize,
    pub weakly_identified: usize,
    pub not_identifiable: usize,
    pub not_converged: usize,
    pub not_evaluable: usize,
}

/// Result of a model-based parametric bootstrap.
///
/// Only converged interior estimates enter percentile intervals. All failed
/// fits remain visible in the attempted denominator. The generating estimate
/// and class totals are included so detached reports can identify the model
/// and sampling design that produced the replicate distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricBootstrapResult {
    pub status: ParametricBootstrapStatus,
    pub seed: u64,
    pub rng_algorithm: &'static str,
    pub count_correction: CountCorrection,
    pub generating_estimate: Option<UvsdEstimate>,
    pub signal_trials: u64,
    pub noise_trials: u64,
    pub attempted: usize,
    pub successful: usize,
    pub success_rate: f64,
    pub failures: ParametricBootstrapFailureCounts,
    pub intervals: Option<ParametricBootstrapIntervals>,
}

/// Generate and refit parametric bootstrap tables from a supplied generating
/// estimate. This is not the nonparametric conditional bootstrap in
/// bootstrap.rs and is not yet a null-calibrated likelihood-ratio test.
///
/// Inferential use requires the unsmoothed primary likelihood. The separately
/// named Miyoshi compatibility correction is rejected here because it is a
/// reference-reproduction convention, not the primary sampling likelihood.
pub fn parametric_bootstrap_intervals(
    template: &CategoryTable,
    generating_estimate: &UvsdEstimate,
    options: ParametricBootstrapOptions,
) -> ParametricBootstrapResult {
    let generating_estimate = UvsdEstimate::new(
        generating_estimate.mu,
        generating_estimate.sigma,
        generating_estimate.thresholds.clone(),
    );
    let valid_estimate = generating_estimate.as_ref().filter(|estimate| {
        estimate.thresholds.len() + 1 == template.categories().len()
    });

    let invalid_options = options.replicates == 0
        || options.minimum_successful_replicates == 0
        || options.minimum_successful_replicates > options.replicates
        || !options.minimum_successful_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.minimum_successful_fraction)
        || options.minimum_successful_fraction == 0.0
        || options.fit_options.count_correction != CountCorrection::None;

    if valid_estimate.is_none() || invalid_options {
        return ParametricBootstrapResult {
            status: ParametricBootstrapStatus::NotEvaluable,
            seed: options.seed,
            rng_algorithm: "SplitMix64-v1",
            count_correction: options.fit_options.count_correction,
            generating_estimate,
            signal_trials: template.signal_total(),
            noise_trials: template.noise_total(),
            attempted: 0,
            successful: 0,
            success_rate: 0.0,
            failures: ParametricBootstrapFailureCounts::default(),
            intervals: None,
        };
    }

    let generating_estimate = (*valid_estimate.expect("checked above")).clone();
    let mut rng = SplitMix64(options.seed);
    let mut mu = Vec::with_capacity(options.replicates);
    let mut sigma = Vec::with_capacity(options.replicates);
    let mut d_a = Vec::with_capacity(options.replicates);
    let mut failures = ParametricBootstrapFailureCounts::default();

    for _ in 0..options.replicates {
        let replicate_seed = rng.next_u64();
        let simulated = simulate_uvsd_category_table(
            template,
            &generating_estimate,
            replicate_seed,
        );
        let bootstrap_table = match simulated {
            Ok(value) => value,
            Err(_) => {
                failures.not_evaluable += 1;
                continue;
            }
        };

        let fit = fit_with_optimizer(
            &bootstrap_table,
            options.fit_options,
            options.optimizer,
        );
        match fit.status {
            FitStatus::Converged => {
                if let Some(estimate) = fit.estimate {
                    mu.push(estimate.mu);
                    sigma.push(estimate.sigma);
                    d_a.push(estimate.d_a);
                } else {
                    failures.not_evaluable += 1;
                }
            }
            FitStatus::Boundary => failures.boundary += 1,
            FitStatus::WeaklyIdentified => failures.weakly_identified += 1,
            FitStatus::NotIdentifiable => failures.not_identifiable += 1,
            FitStatus::NotConverged => failures.not_converged += 1,
            FitStatus::NotEvaluable => failures.not_evaluable += 1,
        }
    }

    let attempted = options.replicates;
    let successful = mu.len();
    let success_rate = successful as f64 / attempted as f64;
    let qualified = successful >= options.minimum_successful_replicates
        && success_rate >= options.minimum_successful_fraction;
    let intervals = if qualified {
        Some(ParametricBootstrapIntervals {
            mu_95: percentile_interval(&mut mu),
            sigma_95: percentile_interval(&mut sigma),
            d_a_95: percentile_interval(&mut d_a),
        })
    } else {
        None
    };

    ParametricBootstrapResult {
        status: if qualified {
            ParametricBootstrapStatus::Qualified
        } else {
            ParametricBootstrapStatus::InsufficientSuccessfulFits
        },
        seed: options.seed,
        rng_algorithm: "SplitMix64-v1",
        count_correction: options.fit_options.count_correction,
        generating_estimate: Some(generating_estimate),
        signal_trials: template.signal_total(),
        noise_trials: template.noise_total(),
        attempted,
        successful,
        success_rate,
        failures,
        intervals,
    }
}

fn percentile_interval(values: &mut [f64]) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    [linear_quantile(values, 0.025), linear_quantile(values, 0.975)]
}

fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        mantissa as f64 / 9_007_199_254_740_992.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_table() -> CategoryTable {
        CategoryTable::new(vec![
            CategoryCount { order: 0, signal: 2, noise: 15 },
            CategoryCount { order: 1, signal: 7, noise: 23 },
            CategoryCount { order: 2, signal: 14, noise: 25 },
            CategoryCount { order: 3, signal: 23, noise: 20 },
            CategoryCount { order: 4, signal: 29, noise: 12 },
            CategoryCount { order: 5, signal: 25, noise: 5 },
        ])
        .unwrap()
    }

    fn example_estimate() -> UvsdEstimate {
        UvsdEstimate::new(1.2, 1.25, vec![-1.5, -0.8, 0.0, 0.8, 1.5]).unwrap()
    }

    fn options(seed: u64) -> ParametricBootstrapOptions {
        ParametricBootstrapOptions {
            replicates: 6,
            seed,
            minimum_successful_replicates: 1,
            minimum_successful_fraction: 0.10,
            fit_options: FitOptions {
                max_iterations: 300,
                gradient_tolerance: 1.0e-4,
                starts: 2,
                count_correction: CountCorrection::None,
            },
            optimizer: Optimizer::Bfgs,
        }
    }

    #[test]
    fn parametric_simulation_preserves_frozen_class_totals_and_order() {
        let template = example_table();
        let simulated = simulate_uvsd_category_table(
            &template,
            &example_estimate(),
            17,
        )
        .unwrap();
        assert_eq!(simulated.signal_total(), template.signal_total());
        assert_eq!(simulated.noise_total(), template.noise_total());
        assert_eq!(
            simulated.categories().iter().map(|row| row.order).collect::<Vec<_>>(),
            template.categories().iter().map(|row| row.order).collect::<Vec<_>>()
        );
    }

    #[test]
    fn parametric_sampling_is_deterministic_for_a_frozen_seed() {
        let template = example_table();
        let first = simulate_uvsd_category_table(&template, &example_estimate(), 91).unwrap();
        let second = simulate_uvsd_category_table(&template, &example_estimate(), 91).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn parametric_bootstrap_is_deterministic_and_accounts_for_every_attempt() {
        let template = example_table();
        let first = parametric_bootstrap_intervals(
            &template,
            &example_estimate(),
            options(42),
        );
        let second = parametric_bootstrap_intervals(
            &template,
            &example_estimate(),
            options(42),
        );
        assert_eq!(first, second);
        assert_eq!(first.attempted, 6);
        assert_eq!(
            first.successful
                + first.failures.boundary
                + first.failures.weakly_identified
                + first.failures.not_identifiable
                + first.failures.not_converged
                + first.failures.not_evaluable,
            6
        );
    }

    #[test]
    fn parametric_bootstrap_rejects_reference_smoothing_for_inference() {
        let template = example_table();
        let result = parametric_bootstrap_intervals(
            &template,
            &example_estimate(),
            ParametricBootstrapOptions {
                fit_options: FitOptions {
                    count_correction: CountCorrection::MiyoshiReferencePerCategory,
                    ..FitOptions::default()
                },
                ..options(9)
            },
        );
        assert_eq!(result.status, ParametricBootstrapStatus::NotEvaluable);
        assert!(result.intervals.is_none());
        assert_eq!(result.attempted, 0);
    }

    #[test]
    fn invalid_bootstrap_budget_returns_no_interval() {
        let result = parametric_bootstrap_intervals(
            &example_table(),
            &example_estimate(),
            ParametricBootstrapOptions {
                replicates: 0,
                ..options(9)
            },
        );
        assert_eq!(result.status, ParametricBootstrapStatus::NotEvaluable);
        assert!(result.intervals.is_none());
    }
}
