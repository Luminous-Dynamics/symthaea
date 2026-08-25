// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical uncertainty and prospective power planning for architecture research.
//!
//! Confirmatory inference treats independently generated environments as the
//! scientific unit of generalization. Nested representation/learner/stream runs
//! quantify within-environment uncertainty, but do not increase the number of
//! independent environments.

use crate::experiment::confirmatory::{PracticalEffect, classify_practical_effect};
use crate::experiment::PairedEstimate;
use crate::harness::analysis::bootstrap_ci_bca;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn looks_like_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Candidate/control outcomes nested within one independently generated world.
///
/// Each run index must represent the same nuisance realization for candidate and
/// control (for example the same representation/learner/stream seed tuple).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NestedEnvironmentResult {
    pub environment_digest: String,
    pub candidate_runs: Vec<f64>,
    pub control_runs: Vec<f64>,
}

impl NestedEnvironmentResult {
    pub fn validate(&self) -> Result<(), String> {
        if !looks_like_digest(&self.environment_digest) {
            return Err("environment digest must be a 32-byte hex digest".into());
        }
        if self.candidate_runs.is_empty() || self.candidate_runs.len() != self.control_runs.len() {
            return Err("candidate/control nested runs must be non-empty and paired".into());
        }
        if self
            .candidate_runs
            .iter()
            .chain(&self.control_runs)
            .any(|value| !value.is_finite())
        {
            return Err("nested run outcomes must be finite".into());
        }
        Ok(())
    }

    pub fn run_count(&self) -> usize {
        self.candidate_runs.len()
    }

    pub fn mean_delta(&self) -> f64 {
        self.candidate_runs
            .iter()
            .zip(&self.control_runs)
            .map(|(candidate, control)| candidate - control)
            .sum::<f64>()
            / self.run_count() as f64
    }
}

fn validate_environment_results(results: &[NestedEnvironmentResult]) -> Result<(), String> {
    if results.len() < 3 {
        return Err("at least three independent environments are required".into());
    }
    let mut seen = BTreeSet::new();
    for result in results {
        result.validate()?;
        if !seen.insert(result.environment_digest.to_ascii_lowercase()) {
            return Err("duplicate environment digest would create pseudoreplication".into());
        }
    }
    Ok(())
}

/// Equal-environment-weight hierarchical bootstrap estimate.
///
/// The bootstrap first samples environments with replacement. For each sampled
/// environment it then samples paired nested runs with replacement and computes
/// one environment mean delta. The final replicate is the equal-weight mean of
/// sampled environment means, so environments with more nuisance runs do not
/// receive more scientific weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalEstimate {
    pub n_environments: usize,
    pub total_nested_pairs: usize,
    pub mean_delta: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub bootstrap_resamples: usize,
}

pub fn hierarchical_environment_delta_percentile(
    results: &[NestedEnvironmentResult],
    n_resamples: usize,
    seed: u64,
) -> Result<HierarchicalEstimate, String> {
    validate_environment_results(results)?;
    if n_resamples < 200 {
        return Err("hierarchical bootstrap requires at least 200 resamples".into());
    }

    let n_environments = results.len();
    let total_nested_pairs = results.iter().map(NestedEnvironmentResult::run_count).sum();
    let mean_delta = results
        .iter()
        .map(NestedEnvironmentResult::mean_delta)
        .sum::<f64>()
        / n_environments as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut bootstrap = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let mut environment_sum = 0.0;
        for _ in 0..n_environments {
            let environment = &results[rng.gen_range(0..n_environments)];
            let mut nested_sum = 0.0;
            for _ in 0..environment.run_count() {
                let run = rng.gen_range(0..environment.run_count());
                nested_sum += environment.candidate_runs[run] - environment.control_runs[run];
            }
            environment_sum += nested_sum / environment.run_count() as f64;
        }
        bootstrap.push(environment_sum / n_environments as f64);
    }

    bootstrap.sort_by(|left, right| left.total_cmp(right));
    Ok(HierarchicalEstimate {
        n_environments,
        total_nested_pairs,
        mean_delta,
        ci95_low: percentile(&bootstrap, 0.025),
        ci95_high: percentile(&bootstrap, 0.975),
        bootstrap_resamples: n_resamples,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDirection {
    MeaningfulGain,
    MeaningfulRegression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProspectivePowerConfig {
    /// Candidate numbers of independent environments, evaluated in ascending order.
    pub environment_counts: Vec<usize>,
    /// Number of nested paired runs planned per future environment.
    pub runs_per_environment: usize,
    /// Monte Carlo future-study simulations per candidate environment count.
    pub simulation_trials: usize,
    /// BCa resamples used inside each simulated future study.
    pub bootstrap_resamples: usize,
    /// Desired probability of clearing the practical-effect gate.
    pub target_power: f64,
    /// Frozen smallest effect size of interest in the primary metric's natural units.
    pub sesoi: f64,
    pub direction: PowerDirection,
    pub seed: u64,
}

impl ProspectivePowerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.environment_counts.is_empty() {
            return Err("at least one candidate environment count is required".into());
        }
        if self.environment_counts.iter().any(|count| *count < 3) {
            return Err("candidate environment counts must be at least three".into());
        }
        if self.environment_counts.windows(2).any(|window| window[0] >= window[1]) {
            return Err("candidate environment counts must be strictly increasing".into());
        }
        if self.runs_per_environment == 0 {
            return Err("runs_per_environment must be positive".into());
        }
        if self.simulation_trials < 100 {
            return Err("prospective power requires at least 100 simulation trials".into());
        }
        if self.bootstrap_resamples < 100 {
            return Err("prospective power requires at least 100 bootstrap resamples".into());
        }
        if !self.target_power.is_finite() || !(0.5..=1.0).contains(&self.target_power) {
            return Err("target power must be finite and between 0.5 and 1.0".into());
        }
        if !self.sesoi.is_finite() || self.sesoi <= 0.0 {
            return Err("SESOI must be finite and strictly positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerPoint {
    pub environments: usize,
    pub successes: usize,
    pub simulation_trials: usize,
    pub estimated_power: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProspectivePowerPlan {
    pub target_power: f64,
    pub sesoi: f64,
    pub direction: PowerDirection,
    pub runs_per_environment: usize,
    pub minimum_environments: Option<usize>,
    pub points: Vec<PowerPoint>,
}

/// Estimate a future CONFIRM sample size using DEV outcomes only.
///
/// This is an empirical Monte Carlo planning tool, not an analytic guarantee.
/// Each future environment is sampled from the DEV environment distribution,
/// then nested paired runs are resampled within that environment. The simulated
/// study is counted as successful only when its environment-level BCa interval
/// satisfies the same SESOI gate used for practical-effect interpretation.
///
/// Freeze the chosen sample size before observing CONFIRM outcomes.
pub fn prospective_power_from_dev(
    dev_results: &[NestedEnvironmentResult],
    config: &ProspectivePowerConfig,
) -> Result<ProspectivePowerPlan, String> {
    validate_environment_results(dev_results)?;
    config.validate()?;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut points = Vec::with_capacity(config.environment_counts.len());
    let mut minimum_environments = None;

    for &environment_count in &config.environment_counts {
        let mut successes = 0usize;

        for _ in 0..config.simulation_trials {
            let mut future_environment_means = Vec::with_capacity(environment_count);
            for _ in 0..environment_count {
                let template = &dev_results[rng.gen_range(0..dev_results.len())];
                let mut delta_sum = 0.0;
                for _ in 0..config.runs_per_environment {
                    let run = rng.gen_range(0..template.run_count());
                    delta_sum += template.candidate_runs[run] - template.control_runs[run];
                }
                future_environment_means.push(delta_sum / config.runs_per_environment as f64);
            }

            let mean_delta = future_environment_means.iter().sum::<f64>()
                / future_environment_means.len() as f64;
            let ci_seed = rng.next_u64();
            let (ci95_low, ci95_high) = bootstrap_ci_bca(
                &future_environment_means,
                config.bootstrap_resamples,
                0.05,
                ci_seed,
            );
            let estimate = PairedEstimate {
                n_pairs: future_environment_means.len(),
                mean_delta,
                ci95_low,
                ci95_high,
            };
            let practical = classify_practical_effect(&estimate, config.sesoi)?;
            let success = matches!(
                (config.direction, practical),
                (PowerDirection::MeaningfulGain, PracticalEffect::MeaningfulGain)
                    | (
                        PowerDirection::MeaningfulRegression,
                        PracticalEffect::MeaningfulRegression
                    )
            );
            if success {
                successes += 1;
            }
        }

        let estimated_power = successes as f64 / config.simulation_trials as f64;
        if minimum_environments.is_none() && estimated_power >= config.target_power {
            minimum_environments = Some(environment_count);
        }
        points.push(PowerPoint {
            environments: environment_count,
            successes,
            simulation_trials: config.simulation_trials,
            estimated_power,
        });
    }

    Ok(ProspectivePowerPlan {
        target_power: config.target_power,
        sesoi: config.sesoi,
        direction: config.direction,
        runs_per_environment: config.runs_per_environment,
        minimum_environments,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u64) -> String {
        format!("{value:064x}")
    }

    fn environment(id: u64, deltas: &[f64]) -> NestedEnvironmentResult {
        NestedEnvironmentResult {
            environment_digest: digest(id),
            candidate_runs: deltas.iter().map(|delta| 0.5 + delta).collect(),
            control_runs: vec![0.5; deltas.len()],
        }
    }

    #[test]
    fn hierarchical_estimate_equal_weights_environments_not_run_counts() {
        let results = vec![
            environment(1, &[0.20]),
            environment(2, &[0.00; 10]),
            environment(3, &[0.10, 0.10]),
        ];
        let estimate = hierarchical_environment_delta_percentile(&results, 500, 42).unwrap();
        assert_eq!(estimate.n_environments, 3);
        assert_eq!(estimate.total_nested_pairs, 13);
        assert!((estimate.mean_delta - 0.10).abs() < 1e-12);
        assert!(estimate.ci95_low <= estimate.mean_delta);
        assert!(estimate.ci95_high >= estimate.mean_delta);
    }

    #[test]
    fn hierarchical_estimate_is_deterministic_for_fixed_seed() {
        let results = vec![
            environment(1, &[0.08, 0.10, 0.12]),
            environment(2, &[0.06, 0.09, 0.11]),
            environment(3, &[0.07, 0.10, 0.13]),
            environment(4, &[0.09, 0.11, 0.14]),
        ];
        let first = hierarchical_environment_delta_percentile(&results, 500, 7).unwrap();
        let second = hierarchical_environment_delta_percentile(&results, 500, 7).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn hierarchical_estimate_rejects_duplicate_environment_identity() {
        let results = vec![
            environment(1, &[0.1]),
            environment(1, &[0.2]),
            environment(2, &[0.3]),
        ];
        assert!(hierarchical_environment_delta_percentile(&results, 500, 42).is_err());
    }

    #[test]
    fn prospective_power_plan_is_deterministic_and_finds_strong_effect() {
        let dev = vec![
            environment(1, &[0.11, 0.12, 0.13]),
            environment(2, &[0.10, 0.12, 0.14]),
            environment(3, &[0.12, 0.13, 0.15]),
            environment(4, &[0.09, 0.11, 0.13]),
            environment(5, &[0.11, 0.13, 0.14]),
            environment(6, &[0.10, 0.12, 0.13]),
        ];
        let config = ProspectivePowerConfig {
            environment_counts: vec![3, 5, 8],
            runs_per_environment: 3,
            simulation_trials: 100,
            bootstrap_resamples: 100,
            target_power: 0.80,
            sesoi: 0.05,
            direction: PowerDirection::MeaningfulGain,
            seed: 99,
        };
        let first = prospective_power_from_dev(&dev, &config).unwrap();
        let second = prospective_power_from_dev(&dev, &config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.points.len(), 3);
        assert!(first.minimum_environments.is_some());
        assert!(first.points.iter().all(|point| (0.0..=1.0).contains(&point.estimated_power)));
    }

    #[test]
    fn prospective_power_rejects_unsorted_environment_counts() {
        let config = ProspectivePowerConfig {
            environment_counts: vec![8, 5],
            runs_per_environment: 1,
            simulation_trials: 100,
            bootstrap_resamples: 100,
            target_power: 0.80,
            sesoi: 0.05,
            direction: PowerDirection::MeaningfulGain,
            seed: 1,
        };
        assert!(config.validate().is_err());
    }
}
