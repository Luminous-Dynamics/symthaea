// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Held-out synthetic coverage sweep for conditional UVSD bootstrap intervals.
//!
//! This is a calibration runner, not a proof of coverage. It reports intervals
//! returned separately from coverage among returned intervals, and preserves
//! every full-fit disposition in the denominator.

use spectra_uvsd_fit::{
    BootstrapOptions, BootstrapStatus, CategoryCount, CategoryTable, FitOptions, FitStatus,
    Optimizer, bootstrap_intervals, fit,
};
use std::f64::consts::TAU;

const MU_GRID: [f64; 2] = [1.0, 1.5];
const SIGMA_GRID: [f64; 3] = [0.80, 1.0, 1.30];
const CUTS: [f64; 5] = [-1.5, -0.75, 0.0, 0.75, 1.5];
const DEFAULT_OUTER: usize = 50;
const DEFAULT_BOOTSTRAPS: usize = 100;
const DEFAULT_TRIALS_PER_CLASS: usize = 400;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = TAU * self.uniform_open();
        radius * angle.cos()
    }
}

fn argument(name: &str, default: usize) -> usize {
    let prefix = format!("--{name}=");
    std::env::args()
        .skip(1)
        .find_map(|argument| argument.strip_prefix(&prefix).and_then(|value| value.parse().ok()))
        .unwrap_or(default)
}

fn category_for(value: f64) -> usize {
    for (index, threshold) in CUTS.iter().enumerate() {
        if value <= *threshold {
            return index;
        }
    }
    CUTS.len()
}

fn simulate_table(
    mu: f64,
    sigma: f64,
    trials_per_class: usize,
    seed: u64,
) -> CategoryTable {
    let mut rng = SplitMix64(seed);
    let mut rows: Vec<CategoryCount> = (0..=CUTS.len())
        .map(|index| CategoryCount {
            order: index as i16,
            signal: 0,
            noise: 0,
        })
        .collect();

    for _ in 0..trials_per_class {
        let noise = rng.standard_normal();
        let signal = mu + sigma * rng.standard_normal();
        rows[category_for(noise)].noise += 1;
        rows[category_for(signal)].signal += 1;
    }
    CategoryTable::new(rows).expect("simulator emits both classes and ordered categories")
}

#[derive(Default)]
struct CellSummary {
    fit_converged: usize,
    fit_boundary: usize,
    fit_weakly_identified: usize,
    fit_not_identifiable: usize,
    fit_not_converged: usize,
    fit_not_evaluable: usize,
    intervals_returned: usize,
    covered: usize,
    da_width_sum: f64,
    bootstrap_success_rate_sum: f64,
    bootstrap_failed_replicates: usize,
    bootstrap_replicates_attempted: usize,
}

fn main() {
    let outer = argument("outer", DEFAULT_OUTER).max(1);
    let bootstraps = argument("bootstraps", DEFAULT_BOOTSTRAPS).max(1);
    let trials = argument("trials-per-class", DEFAULT_TRIALS_PER_CLASS).max(20);
    let min_successful = ((bootstraps as f64) * 0.80).ceil() as usize;

    let fit_options = FitOptions {
        max_iterations: 600,
        gradient_tolerance: 1.0e-5,
        starts: 3,
        ..FitOptions::default()
    };
    let bootstrap_template = BootstrapOptions {
        replicates: bootstraps,
        seed: 0,
        minimum_successful_replicates: min_successful.max(1),
        minimum_successful_fraction: 0.80,
        fit_options,
        optimizer: Optimizer::Bfgs,
    };

    println!(
        "mu_true,sigma_true,d_a_true,outer_attempted,fit_converged,fit_boundary,fit_weakly_identified,fit_not_identifiable,fit_not_converged,fit_not_evaluable,intervals_returned,d_a_covered,coverage_given_interval,mean_d_a_interval_width,mean_bootstrap_success_rate,mean_bootstrap_failed_replicates"
    );

    for (mu_index, &mu) in MU_GRID.iter().enumerate() {
        for (sigma_index, &sigma) in SIGMA_GRID.iter().enumerate() {
            let true_da = mu / ((1.0 + sigma * sigma) / 2.0).sqrt();
            let mut summary = CellSummary::default();

            for outer_index in 0..outer {
                let cell = (mu_index * SIGMA_GRID.len() + sigma_index) as u64;
                let seed = 0x5350_4543_434f_5600u64
                    .wrapping_add(cell.wrapping_mul(outer as u64))
                    .wrapping_add(outer_index as u64);
                let table = simulate_table(mu, sigma, trials, seed);
                let full_fit = fit(&table, fit_options);

                match full_fit.status {
                    FitStatus::Converged => summary.fit_converged += 1,
                    FitStatus::Boundary => summary.fit_boundary += 1,
                    FitStatus::WeaklyIdentified => summary.fit_weakly_identified += 1,
                    FitStatus::NotIdentifiable => summary.fit_not_identifiable += 1,
                    FitStatus::NotConverged => summary.fit_not_converged += 1,
                    FitStatus::NotEvaluable => summary.fit_not_evaluable += 1,
                }

                if full_fit.status != FitStatus::Converged {
                    continue;
                }

                let mut bootstrap_options = bootstrap_template;
                bootstrap_options.seed = seed ^ 0x9E37_79B9_7F4A_7C15;
                let bootstrap = bootstrap_intervals(&table, bootstrap_options);
                summary.bootstrap_success_rate_sum += bootstrap.success_rate;
                summary.bootstrap_failed_replicates +=
                    bootstrap.attempted.saturating_sub(bootstrap.successful);
                summary.bootstrap_replicates_attempted += bootstrap.attempted;

                if bootstrap.status == BootstrapStatus::Qualified {
                    if let Some(intervals) = bootstrap.intervals {
                        summary.intervals_returned += 1;
                        if intervals.d_a_95[0] <= true_da && true_da <= intervals.d_a_95[1] {
                            summary.covered += 1;
                        }
                        summary.da_width_sum += intervals.d_a_95[1] - intervals.d_a_95[0];
                    }
                }
            }

            let coverage = if summary.intervals_returned > 0 {
                summary.covered as f64 / summary.intervals_returned as f64
            } else {
                f64::NAN
            };
            let width = if summary.intervals_returned > 0 {
                summary.da_width_sum / summary.intervals_returned as f64
            } else {
                f64::NAN
            };
            let returned_bootstraps = summary.fit_converged;
            let mean_bootstrap_success = if returned_bootstraps > 0 {
                summary.bootstrap_success_rate_sum / returned_bootstraps as f64
            } else {
                f64::NAN
            };
            let mean_bootstrap_failed = if summary.bootstrap_replicates_attempted > 0 {
                summary.bootstrap_failed_replicates as f64
                    / returned_bootstraps.max(1) as f64
            } else {
                f64::NAN
            };

            println!(
                "{mu},{sigma},{true_da},{outer},{},{},{},{},{},{},{},{},{coverage},{width},{mean_bootstrap_success},{mean_bootstrap_failed}",
                summary.fit_converged,
                summary.fit_boundary,
                summary.fit_weakly_identified,
                summary.fit_not_identifiable,
                summary.fit_not_converged,
                summary.fit_not_evaluable,
                summary.intervals_returned,
                summary.covered
            );
        }
    }

    eprintln!(
        "calibration sweep: outer={outer}, bootstrap_replicates={bootstraps}, trials_per_class={trials}, min_successful_bootstraps={min_successful}"
    );
    eprintln!("coverage is conditional on full-fit convergence and qualified bootstrap intervals");
}
