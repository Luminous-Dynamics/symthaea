// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen-seed finite-sample recovery comparison for latent UVSD threshold designs.
//!
//! The threshold vectors in this runner are latent evidence-space conditions.
//! This is a conditional Gaussian simulation benchmark, not an executable
//! human-study design recommendation.

use spectra_uvsd_fit::{
    CategoryCount, CategoryTable, FitOptions, FitStatus, Optimizer,
    expected_fisher_information_for_design, fit,
};
use std::f64::consts::TAU;

const MU_GRID: [f64; 4] = [0.5, 1.0, 1.5, 2.0];
const SIGMA_GRID: [f64; 4] = [0.70, 1.0, 1.30, 1.60];
const DEFAULT_OUTER: usize = 50;
const DEFAULT_TRIALS_PER_CLASS: usize = 250;
const Z_975: f64 = 1.959_963_984_540_054;

const DESIGNS: [(&str, [f64; 5]); 4] = [
    (
        "equal_probability",
        [
            -0.967_421_566_101_701,
            -0.430_727_299_295_457,
            0.0,
            0.430_727_299_295_457,
            0.967_421_566_101_701,
        ],
    ),
    ("fixed_spacing", [-1.5, -0.75, 0.0, 0.75, 1.5]),
    ("minimax_candidate", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("tradeoff_candidate", [-0.5, 0.5, 1.0, 1.5, 2.0]),
];

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

fn argument_usize(name: &str, default: usize) -> usize {
    let prefix = format!("--{name}=");
    std::env::args()
        .skip(1)
        .find_map(|argument| {
            argument
                .strip_prefix(&prefix)
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(default)
}

fn argument_u64(name: &str, default: u64) -> u64 {
    let prefix = format!("--{name}=");
    std::env::args()
        .skip(1)
        .find_map(|argument| {
            argument.strip_prefix(&prefix).and_then(|value| {
                value
                    .strip_prefix("0x")
                    .map(|hex| u64::from_str_radix(hex, 16).ok())
                    .unwrap_or_else(|| value.parse::<u64>().ok())
            })
        })
        .unwrap_or(default)
}

fn category_for(value: f64, thresholds: &[f64; 5]) -> usize {
    for (index, threshold) in thresholds.iter().enumerate() {
        if value <= *threshold {
            return index;
        }
    }
    thresholds.len()
}

fn simulate_table(
    mu: f64,
    sigma: f64,
    thresholds: &[f64; 5],
    trials_per_class: usize,
    seed: u64,
) -> Result<CategoryTable, String> {
    let mut rng = SplitMix64(seed);
    let mut rows: Vec<CategoryCount> = (0..=thresholds.len())
        .map(|index| CategoryCount {
            order: index as i16,
            signal: 0,
            noise: 0,
        })
        .collect();

    for _ in 0..trials_per_class {
        let noise = rng.standard_normal();
        let signal = mu + sigma * rng.standard_normal();
        rows[category_for(noise, thresholds)].noise += 1;
        rows[category_for(signal, thresholds)].signal += 1;
    }

    CategoryTable::new(rows).map_err(|error| error.to_string())
}

#[derive(Default)]
struct IntervalSummary {
    returned: usize,
    covered: usize,
    width_sum: f64,
}

impl IntervalSummary {
    fn record(&mut self, interval: Option<[f64; 2]>, truth: f64) {
        if let Some([lower, upper]) = interval {
            if lower.is_finite() && upper.is_finite() && lower <= upper {
                self.returned += 1;
                if lower <= truth && truth <= upper {
                    self.covered += 1;
                }
                self.width_sum += upper - lower;
            }
        }
    }

    fn coverage(&self) -> f64 {
        ratio(self.covered, self.returned)
    }

    fn availability(&self, attempted: usize) -> f64 {
        ratio(self.returned, attempted)
    }

    fn mean_width(&self) -> f64 {
        if self.returned > 0 {
            self.width_sum / self.returned as f64
        } else {
            f64::NAN
        }
    }

    fn coverage_wilson(&self) -> (f64, f64) {
        wilson_interval(self.covered, self.returned)
    }
}

#[derive(Default)]
struct Summary {
    converged: usize,
    boundary: usize,
    weakly_identified: usize,
    not_identifiable: usize,
    not_converged: usize,
    not_evaluable: usize,
    estimate_count: usize,
    mu_error_sum: f64,
    mu_squared_error_sum: f64,
    sigma_error_sum: f64,
    sigma_squared_error_sum: f64,
    da_error_sum: f64,
    da_squared_error_sum: f64,
    mu_interval: IntervalSummary,
    sigma_interval: IntervalSummary,
    da_interval: IntervalSummary,
    joint_interval_returned: usize,
    joint_interval_covered: usize,
}

impl Summary {
    fn record_fit(&mut self, result: spectra_uvsd_fit::FitResult, mu: f64, sigma: f64, true_da: f64) {
        let spectra_uvsd_fit::FitResult {
            status,
            estimate,
            diagnostics,
        } = result;
        let estimate = match status {
            FitStatus::Converged => match estimate {
                Some(value) => {
                    self.converged += 1;
                    value
                }
                None => {
                    self.not_evaluable += 1;
                    return;
                }
            },
            FitStatus::Boundary => {
                self.boundary += 1;
                return;
            }
            FitStatus::WeaklyIdentified => {
                self.weakly_identified += 1;
                return;
            }
            FitStatus::NotIdentifiable => {
                self.not_identifiable += 1;
                return;
            }
            FitStatus::NotConverged => {
                self.not_converged += 1;
                return;
            }
            FitStatus::NotEvaluable => {
                self.not_evaluable += 1;
                return;
            }
        };

        self.estimate_count += 1;
        let mu_error = estimate.mu - mu;
        let sigma_error = estimate.sigma - sigma;
        let da_error = estimate.d_a - true_da;
        self.mu_error_sum += mu_error;
        self.mu_squared_error_sum += mu_error * mu_error;
        self.sigma_error_sum += sigma_error;
        self.sigma_squared_error_sum += sigma_error * sigma_error;
        self.da_error_sum += da_error;
        self.da_squared_error_sum += da_error * da_error;

        let Some(information) = diagnostics.expected_fisher_information else {
            return;
        };
        let mu_interval = information
            .mu_standard_error
            .map(|se| [estimate.mu - Z_975 * se, estimate.mu + Z_975 * se]);
        let sigma_interval = information.sigma_wald_95;
        let da_interval = information.d_a_wald_95;

        self.mu_interval.record(mu_interval, mu);
        self.sigma_interval.record(sigma_interval, sigma);
        self.da_interval.record(da_interval, true_da);

        if let (Some(mu_ci), Some(sigma_ci), Some(da_ci)) =
            (mu_interval, sigma_interval, da_interval)
        {
            let ordered_finite = |interval: [f64; 2]| {
                interval[0].is_finite()
                    && interval[1].is_finite()
                    && interval[0] <= interval[1]
            };
            if ordered_finite(mu_ci) && ordered_finite(sigma_ci) && ordered_finite(da_ci) {
                self.joint_interval_returned += 1;
                if mu_ci[0] <= mu
                    && mu <= mu_ci[1]
                    && sigma_ci[0] <= sigma
                    && sigma <= sigma_ci[1]
                    && da_ci[0] <= true_da
                    && true_da <= da_ci[1]
                {
                    self.joint_interval_covered += 1;
                }
            }
        }
    }
}
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator > 0 {
        numerator as f64 / denominator as f64
    } else {
        f64::NAN
    }
}

fn mean(sum: f64, count: usize) -> f64 {
    if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    }
}

fn rmse(sum_squared_error: f64, count: usize) -> f64 {
    if count > 0 {
        (sum_squared_error / count as f64).sqrt()
    } else {
        f64::NAN
    }
}

fn wilson_interval(successes: usize, attempts: usize) -> (f64, f64) {
    if attempts == 0 {
        return (f64::NAN, f64::NAN);
    }
    let n = attempts as f64;
    let p = successes as f64 / n;
    let z2 = Z_975 * Z_975;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z_975
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn main() {
    let outer = argument_usize("outer", DEFAULT_OUTER).max(1);
    let trials = argument_usize("trials-per-class", DEFAULT_TRIALS_PER_CLASS).max(20);
    let seed_root = argument_u64("seed", 0x5350_4543_4445_5349);

    let fit_options = FitOptions {
        max_iterations: 600,
        gradient_tolerance: 1.0e-5,
        starts: 3,
        ..FitOptions::default()
    };

    println!(
        "design_class,seed_root,rng,design,mu_true,sigma_true,d_a_true,outer_attempted,trials_per_class,fit_converged,fit_boundary,fit_weakly_identified,fit_not_identifiable,fit_not_converged,fit_not_evaluable,estimate_availability,mu_bias,mu_rmse,sigma_bias,sigma_rmse,d_a_bias,d_a_rmse,fisher_rank,fisher_dimension,fisher_condition_at_truth,fisher_d_a_se_at_truth,mu_interval_returned,mu_interval_availability,mu_coverage_conditional,mu_coverage_wilson_low,mu_coverage_wilson_high,mu_mean_width,sigma_interval_returned,sigma_interval_availability,sigma_coverage_conditional,sigma_coverage_wilson_low,sigma_coverage_wilson_high,sigma_mean_width,d_a_interval_returned,d_a_interval_availability,d_a_coverage_conditional,d_a_coverage_wilson_low,d_a_coverage_wilson_high,d_a_mean_width,joint_interval_returned,joint_interval_availability,joint_coverage_conditional"
    );

    for (design_name, thresholds) in DESIGNS.iter() {
        for (mu_index, &mu) in MU_GRID.iter().enumerate() {
            for (sigma_index, &sigma) in SIGMA_GRID.iter().enumerate() {
                let true_da = mu / ((1.0 + sigma * sigma) / 2.0).sqrt();
                let fisher = expected_fisher_information_for_design(
                    mu,
                    sigma,
                    thresholds,
                    trials as u64,
                    trials as u64,
                );

                let (fisher_rank, fisher_dimension, fisher_condition, fisher_da_se) =
                    match fisher {
                        Some(info) => (
                            info.effective_rank as f64,
                            info.parameter_dimension as f64,
                            info.condition_number.unwrap_or(f64::NAN),
                            info.d_a_standard_error.unwrap_or(f64::NAN),
                        ),
                        None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
                    };

                let mut summary = Summary::default();
                for outer_index in 0..outer {
                    let parameter_cell =
                        (mu_index * SIGMA_GRID.len() + sigma_index) as u64;
                    // Common random numbers: every candidate design sees the
                    // same latent Gaussian draws for this parameter cell and
                    // replicate, enabling paired design comparisons.
                    let seed = seed_root
                        ^ parameter_cell.wrapping_shl(32)
                        ^ outer_index as u64;
                    match simulate_table(mu, sigma, thresholds, trials, seed) {
                        Ok(table) => {
                            let result = fit(&table, fit_options);
                            summary.record_fit(result, mu, sigma, true_da);
                        }
                        Err(_) => summary.not_evaluable += 1,
                    }
                }

                let joint_coverage = ratio(
                    summary.joint_interval_covered,
                    summary.joint_interval_returned,
                );
                println!(
                    "LATENT_CONDITIONAL_ONLY,{seed_root},SplitMix64-v1,{design_name},{mu},{sigma},{true_da},{outer},{trials},{},{},{},{},{},{},{},{},{},{},{},{},{},{fisher_rank},{fisher_dimension},{fisher_condition},{fisher_da_se},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    summary.converged,
                    summary.boundary,
                    summary.weakly_identified,
                    summary.not_identifiable,
                    summary.not_converged,
                    summary.not_evaluable,
                    ratio(summary.estimate_count, outer),
                    mean(summary.mu_error_sum, summary.estimate_count),
                    rmse(summary.mu_squared_error_sum, summary.estimate_count),
                    mean(summary.sigma_error_sum, summary.estimate_count),
                    rmse(summary.sigma_squared_error_sum, summary.estimate_count),
                    mean(summary.da_error_sum, summary.estimate_count),
                    rmse(summary.da_squared_error_sum, summary.estimate_count),
                    summary.mu_interval.returned,
                    summary.mu_interval.availability(outer),
                    summary.mu_interval.coverage(),
                    summary.mu_interval.coverage_wilson().0,
                    summary.mu_interval.coverage_wilson().1,
                    summary.mu_interval.mean_width(),
                    summary.sigma_interval.returned,
                    summary.sigma_interval.availability(outer),
                    summary.sigma_interval.coverage(),
                    summary.sigma_interval.coverage_wilson().0,
                    summary.sigma_interval.coverage_wilson().1,
                    summary.sigma_interval.mean_width(),
                    summary.da_interval.returned,
                    summary.da_interval.availability(outer),
                    summary.da_interval.coverage(),
                    summary.da_interval.coverage_wilson().0,
                    summary.da_interval.coverage_wilson().1,
                    summary.da_interval.mean_width(),
                    summary.joint_interval_returned,
                    ratio(summary.joint_interval_returned, outer),
                    joint_coverage
                );
            }
        }
    }

    eprintln!(
        "latent design recovery: outer={outer}, trials_per_class={trials}, seed_root={seed_root:#x}"
    );
    eprintln!(
        "Wald coverage is conditional on returned local intervals; availability and Wilson intervals are reported separately."
    );
    eprintln!(
        "All thresholds are latent evidence-space conditions. This runner does not operationalize participant-facing confidence criteria or RT binning."
    );
}
