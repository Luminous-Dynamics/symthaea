// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fixed-class-total nonparametric bootstrap for ordered UVSD category tables.
//!
//! Resampling is performed independently within signal and noise classes.
//! Only converged, interior fits enter percentile intervals; every failed
//! status remains visible in the replicate accounting.

use crate::{
    CategoryCount, CategoryTable, CountCorrection, FitOptions, FitStatus, Optimizer,
    fit_with_optimizer,
};

/// Bootstrap execution and qualification controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapOptions {
    /// Number of independent class-stratified bootstrap replicates.
    pub replicates: usize,
    /// Seed for deterministic SplitMix64 resampling.
    pub seed: u64,
    /// Minimum number of converged interior fits required for intervals.
    pub minimum_successful_replicates: usize,
    /// Minimum successful fraction required for interval qualification.
    pub minimum_successful_fraction: f64,
    pub fit_options: FitOptions,
    pub optimizer: Optimizer,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            replicates: 2_000,
            seed: 0x5350_4543_5452_415f,
            minimum_successful_replicates: 1_600,
            minimum_successful_fraction: 0.90,
            fit_options: FitOptions::default(),
            optimizer: Optimizer::Bfgs,
        }
    }
}

/// Explicit result status for the bootstrap procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapStatus {
    Qualified,
    InsufficientSuccessfulFits,
    NotEvaluable,
}

/// Percentile interval endpoints from the successful bootstrap distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapIntervals {
    pub mu_95: [f64; 2],
    pub sigma_95: [f64; 2],
    pub d_a_95: [f64; 2],
}

/// Number of replicates that failed each fit disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BootstrapFailureCounts {
    pub boundary: usize,
    pub weakly_identified: usize,
    pub not_identifiable: usize,
    pub not_converged: usize,
    pub not_evaluable: usize,
}

/// Bootstrap result including unsuccessful fit accounting. A point estimate
/// is not included; callers retain the original full-data fit separately.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapResult {
    pub status: BootstrapStatus,
    pub seed: u64,
    pub count_correction: CountCorrection,
    pub attempted: usize,
    pub successful: usize,
    pub success_rate: f64,
    pub failures: BootstrapFailureCounts,
    pub intervals: Option<BootstrapIntervals>,
}

/// Run a nonparametric multinomial bootstrap conditional on the observed
/// signal/noise class totals. Category boundaries and class totals remain
/// fixed; within each class, individual observed categorical responses are
/// sampled with replacement.
///
/// Only Converged fits enter the percentile distribution. Boundary,
/// weak-identification and optimization failures remain in the denominator
/// and are returned in failures.
pub fn bootstrap_intervals(
    table: &CategoryTable,
    options: BootstrapOptions,
) -> BootstrapResult {
    let correction = options.fit_options.count_correction;
    let invalid = options.replicates == 0
        || options.minimum_successful_replicates == 0
        || options.minimum_successful_replicates > options.replicates
        || !options.minimum_successful_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.minimum_successful_fraction)
        || options.minimum_successful_fraction == 0.0;

    if invalid {
        return BootstrapResult {
            status: BootstrapStatus::NotEvaluable,
            seed: options.seed,
            count_correction: correction,
            attempted: 0,
            successful: 0,
            success_rate: 0.0,
            failures: BootstrapFailureCounts::default(),
            intervals: None,
        };
    }

    let mut rng = SplitMix64(options.seed);
    let mut mu = Vec::with_capacity(options.replicates);
    let mut sigma = Vec::with_capacity(options.replicates);
    let mut d_a = Vec::with_capacity(options.replicates);
    let mut failures = BootstrapFailureCounts::default();

    for _ in 0..options.replicates {
        let signal_counts = resample_class(table.categories(), true, table.signal_total(), &mut rng);
        let noise_counts = resample_class(table.categories(), false, table.noise_total(), &mut rng);
        let categories = table
            .categories()
            .iter()
            .enumerate()
            .map(|(index, row)| CategoryCount {
                order: row.order,
                signal: signal_counts[index],
                noise: noise_counts[index],
            })
            .collect();

        let bootstrap_table = match CategoryTable::new(categories) {
            Ok(value) => value,
            Err(_) => {
                failures.not_evaluable += 1;
                continue;
            }
        };
        let fit = fit_with_optimizer(&bootstrap_table, options.fit_options, options.optimizer);
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
        Some(BootstrapIntervals {
            mu_95: percentile_interval(&mut mu),
            sigma_95: percentile_interval(&mut sigma),
            d_a_95: percentile_interval(&mut d_a),
        })
    } else {
        None
    };

    BootstrapResult {
        status: if qualified {
            BootstrapStatus::Qualified
        } else {
            BootstrapStatus::InsufficientSuccessfulFits
        },
        seed: options.seed,
        count_correction: correction,
        attempted,
        successful,
        success_rate,
        failures,
        intervals,
    }
}

fn resample_class(
    categories: &[CategoryCount],
    signal_class: bool,
    total: u64,
    rng: &mut SplitMix64,
) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(categories.len());
    let mut running = 0u64;
    for category in categories {
        running += if signal_class {
            category.signal
        } else {
            category.noise
        };
        cumulative.push(running);
    }

    let mut sampled = vec![0u64; categories.len()];
    for _ in 0..total {
        let draw = rng.next_below(total);
        let category = cumulative.partition_point(|&upper| upper <= draw);
        if let Some(count) = sampled.get_mut(category) {
            *count += 1;
        }
    }
    sampled
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

    fn next_below(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % upper;
            }
        }
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

    fn options(seed: u64) -> BootstrapOptions {
        BootstrapOptions {
            replicates: 12,
            seed,
            minimum_successful_replicates: 1,
            minimum_successful_fraction: 0.10,
            fit_options: FitOptions {
                max_iterations: 250,
                gradient_tolerance: 1.0e-4,
                starts: 3,
                count_correction: CountCorrection::None,
            },
            optimizer: Optimizer::Bfgs,
        }
    }

    #[test]
    fn bootstrap_is_deterministic_and_accounts_for_every_replicate() {
        let table = example_table();
        let first = bootstrap_intervals(&table, options(42));
        let second = bootstrap_intervals(&table, options(42));
        assert_eq!(first, second);
        assert_eq!(first.attempted, 12);
        assert_eq!(
            first.successful
                + first.failures.boundary
                + first.failures.weakly_identified
                + first.failures.not_identifiable
                + first.failures.not_converged
                + first.failures.not_evaluable,
            12
        );
        assert!(first.success_rate.is_finite());
    }

    #[test]
    fn each_class_resample_preserves_its_original_total() {
        let table = example_table();
        let mut rng = SplitMix64(99);
        let signal = resample_class(table.categories(), true, table.signal_total(), &mut rng);
        let noise = resample_class(table.categories(), false, table.noise_total(), &mut rng);
        assert_eq!(signal.iter().sum::<u64>(), table.signal_total());
        assert_eq!(noise.iter().sum::<u64>(), table.noise_total());
        assert_eq!(signal.len(), table.categories().len());
        assert_eq!(noise.len(), table.categories().len());
    }

    #[test]
    fn bootstrap_requires_minimum_successful_fraction() {
        let table = example_table();
        let mut options = options(7);
        options.minimum_successful_replicates = options.replicates;
        options.minimum_successful_fraction = 1.0;
        let result = bootstrap_intervals(&table, options);
        if result.successful < options.replicates {
            assert_eq!(result.status, BootstrapStatus::InsufficientSuccessfulFits);
            assert!(result.intervals.is_none());
        } else {
            assert_eq!(result.status, BootstrapStatus::Qualified);
            assert!(result.intervals.is_some());
        }
    }

    #[test]
    fn rejects_invalid_bootstrap_options() {
        let result = bootstrap_intervals(
            &example_table(),
            BootstrapOptions {
                replicates: 0,
                ..options(1)
            },
        );
        assert_eq!(result.status, BootstrapStatus::NotEvaluable);
        assert_eq!(result.attempted, 0);
        assert!(result.intervals.is_none());
    }
}
