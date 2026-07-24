// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helicopter benchmark harness with fixed actuator authority.
//!
//! `phi` is recorded as an observational variable only. It is deliberately
//! not multiplied into motor commands: doing so would manufacture the claimed
//! relationship between consciousness and recovery by construction.

use crate::controller::HelicopterController;
use crate::encoder::HelicopterHdcEncoder;
use crate::perturbations::{HelicopterPerturbation, PerturbationSchedule};
use crate::simulator::{HelicopterPhysicsSimulator, LandingOutcome, SimpleHelicopterSimulator};
use crate::types::HelicopterConfig;

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

/// Benchmark result for a single hover evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverBenchmark {
    /// Mean altitude error from target (meters).
    pub mean_altitude_error: f64,
    /// Mean control effort (0.0–1.0).
    pub mean_control_effort: f32,
    /// Mean angular speed (rad/s) — lower = more stable.
    pub mean_angular_speed: f64,
    /// Fraction of steps with altitude within ±2m of target.
    pub hover_fraction: f64,
    /// Whether the helicopter crashed (altitude = 0).
    pub crashed: bool,
    /// Steps before crash (or total steps if no crash).
    pub steps_survived: usize,
    /// Consciousness level during evaluation (for correlation analysis).
    pub phi: f64,
}

/// Benchmark result for perturbation recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationRecoveryBenchmark {
    /// Performance before perturbation.
    pub pre_perturbation_error: f64,
    /// Performance during perturbation.
    pub during_perturbation_error: f64,
    /// Performance after perturbation clears.
    pub post_perturbation_error: f64,
    /// Steps to recover to within 2× pre-perturbation error.
    pub recovery_steps: Option<usize>,
    /// Consciousness level observed during recovery. It does not alter authority.
    pub phi: f64,
    /// Whether the vehicle crashed or produced non-finite state.
    pub crashed: bool,
    /// Whether every preregistered measurement window received samples.
    pub complete_windows: bool,
}

/// Run a hover stability benchmark.
///
/// Evaluates hover performance at a target altitude with optional perturbations.
pub fn benchmark_hover(
    config: &HelicopterConfig,
    target_altitude: f64,
    steps: usize,
    phi: f64,
) -> HoverBenchmark {
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut controller = HelicopterController::new(&genesis, config);
    let mut encoder = HelicopterHdcEncoder::new(&genesis, 32);
    let mut sim = SimpleHelicopterSimulator::new();
    sim.reset(target_altitude);

    let dt = config.physics_dt();
    let mut altitude_error_sum = 0.0;
    let mut effort_sum = 0.0f32;
    let mut angular_sum = 0.0;
    let mut hover_count = 0usize;
    let mut steps_survived = 0;

    for step in 0..steps {
        let reference = crate::guidance::FlightReference::hold([0.0, 0.0, target_altitude], 0.0);
        let guidance = crate::guidance::position_hold_command(
            sim.state(),
            &reference,
            &crate::guidance::GuidanceConfig::default(),
        );
        let hv = encoder.encode_with_dt(sim.state(), dt);
        let learned = controller.forward(&hv, dt as f32);
        let cmd = guidance.blend(learned, 0.10);

        sim.step(&cmd, dt);
        steps_survived = step + 1;

        let state = sim.state();
        let flight_error = (state.altitude() - target_altitude).abs();
        altitude_error_sum += flight_error;
        effort_sum += cmd.control_effort();
        angular_sum += state.angular_speed();

        if flight_error < 2.0 {
            hover_count += 1;
        }

        if matches!(sim.landing_contact().outcome, LandingOutcome::Crash) {
            break;
        }
        if !state.is_finite() {
            break;
        }
    }

    let n = steps_survived.max(1) as f64;
    HoverBenchmark {
        mean_altitude_error: altitude_error_sum / n,
        mean_control_effort: effort_sum / n as f32,
        mean_angular_speed: angular_sum / n,
        hover_fraction: hover_count as f64 / n,
        crashed: steps_survived < steps,
        steps_survived,
        phi,
    }
}

/// Run a perturbation recovery benchmark.
///
/// Applies a crosswind perturbation mid-flight and measures recovery.
pub fn benchmark_perturbation_recovery(
    config: &HelicopterConfig,
    target_altitude: f64,
    phi: f64,
) -> PerturbationRecoveryBenchmark {
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut controller = HelicopterController::new(&genesis, config);
    let mut encoder = HelicopterHdcEncoder::new(&genesis, 32);
    let mut sim = SimpleHelicopterSimulator::new();
    sim.reset(target_altitude);

    let dt = config.physics_dt();
    let total_steps = config.steps_per_episode;
    let perturb_start = total_steps / 3;
    let perturb_end = 2 * total_steps / 3;

    let mut pre_error = 0.0;
    let mut during_error = 0.0;
    let mut post_error = 0.0;
    let mut pre_count = 0;
    let mut during_count = 0;
    let mut post_count = 0;
    let mut recovery_step = None;
    let mut pre_mean = 0.0;
    let mut crashed = false;
    let schedule = PerturbationSchedule::new().add(
        HelicopterPerturbation::Crosswind {
            force_n: 2_236.067_977,
        },
        perturb_start,
        Some(perturb_end),
    );

    for step in 0..total_steps {
        let active = schedule.active_at(step);
        sim.set_active_perturbations(&active)
            .expect("benchmark perturbation schedule is valid");

        let reference = crate::guidance::FlightReference::hold([0.0, 0.0, target_altitude], 0.0);
        let guidance = crate::guidance::position_hold_command(
            sim.state(),
            &reference,
            &crate::guidance::GuidanceConfig::default(),
        );
        let hv = encoder.encode_with_dt(sim.state(), dt);
        let learned = controller.forward(&hv, dt as f32);
        let cmd = guidance.blend(learned, 0.10);

        sim.step(&cmd, dt);

        let state = sim.state();
        let horizontal_error = state.position[0].hypot(state.position[1]);
        let altitude_error = (state.altitude() - target_altitude).abs();
        let flight_error = horizontal_error.hypot(altitude_error) + 0.25 * state.angular_speed();

        if step < perturb_start {
            pre_error += flight_error;
            pre_count += 1;
        } else if step < perturb_end {
            during_error += flight_error;
            during_count += 1;
        } else {
            post_error += flight_error;
            post_count += 1;
            // Check recovery: within 2× pre-perturbation error
            if recovery_step.is_none() && flight_error < pre_mean * 2.0 {
                recovery_step = Some(step - perturb_end);
            }
        }

        if step == perturb_start {
            pre_mean = pre_error / pre_count.max(1) as f64;
        }

        if !sim.state().is_finite()
            || matches!(sim.landing_contact().outcome, LandingOutcome::Crash)
        {
            crashed = true;
            break;
        }
    }

    PerturbationRecoveryBenchmark {
        pre_perturbation_error: mean_or_nan(pre_error, pre_count),
        during_perturbation_error: mean_or_nan(during_error, during_count),
        post_perturbation_error: mean_or_nan(post_error, post_count),
        recovery_steps: recovery_step,
        phi,
        crashed,
        complete_windows: pre_count > 0 && during_count > 0 && post_count > 0,
    }
}

/// Run a fixed-authority negative-control sweep across reported phi values.
///
/// Identical dynamics across the sweep are expected unless a separately
/// preregistered cognitive mechanism—not actuator gain—uses phi internally.
pub fn phi_correlation_sweep(
    config: &HelicopterConfig,
    target_altitude: f64,
    steps_per_eval: usize,
) -> Vec<(f64, HoverBenchmark)> {
    let phi_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    phi_values
        .iter()
        .map(|&phi| {
            let result = benchmark_hover(config, target_altitude, steps_per_eval, phi);
            (phi, result)
        })
        .collect()
}

/// Reproducible declaration for a fixed-authority negative-control study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub protocol_version: String,
    pub scenario_id: String,
    pub target_altitude_m: f64,
    pub steps_per_evaluation: usize,
    pub phi_values: Vec<f64>,
    pub seeds: Vec<u64>,
    pub bootstrap_replicates: usize,
    pub actuator_authority_fixed: bool,
}

/// One row in the negative-control evidence table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSample {
    pub seed: u64,
    pub phi: f64,
    pub mean_altitude_error: f64,
    pub mean_control_effort: f32,
    pub crashed: bool,
}

/// Statistical summary of the negative-control study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeControlReport {
    pub manifest: BenchmarkManifest,
    pub samples: Vec<BenchmarkSample>,
    pub pearson_r: Option<f64>,
    pub cluster_bootstrap_ci95: Option<[f64; 2]>,
    /// Largest performance spread across phi values for any single seed.
    pub max_within_seed_altitude_delta: f64,
}

/// Run a multi-seed fixed-authority protocol. Each seed receives the complete
/// phi grid, so phi is balanced within every controller initialization.
pub fn fixed_authority_negative_control(
    config: &HelicopterConfig,
    target_altitude_m: f64,
    steps_per_evaluation: usize,
    seeds: &[u64],
    bootstrap_replicates: usize,
) -> NegativeControlReport {
    let phi_values = vec![0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut samples = Vec::with_capacity(seeds.len() * phi_values.len());

    for &seed in seeds {
        let mut seeded_config = config.clone();
        seeded_config.genesis_phrase = format!("{}-benchmark-seed-{seed}", config.genesis_phrase);
        for &phi in &phi_values {
            let result =
                benchmark_hover(&seeded_config, target_altitude_m, steps_per_evaluation, phi);
            samples.push(BenchmarkSample {
                seed,
                phi,
                mean_altitude_error: result.mean_altitude_error,
                mean_control_effort: result.mean_control_effort,
                crashed: result.crashed,
            });
        }
    }

    let pearson_r = pearson_correlation(
        &samples.iter().map(|sample| sample.phi).collect::<Vec<_>>(),
        &samples
            .iter()
            .map(|sample| sample.mean_altitude_error)
            .collect::<Vec<_>>(),
    );
    let cluster_bootstrap_ci95 =
        cluster_bootstrap_correlation_ci(&samples, seeds, bootstrap_replicates);
    let max_within_seed_altitude_delta = seeds
        .iter()
        .map(|seed| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for sample in samples.iter().filter(|sample| sample.seed == *seed) {
                min = min.min(sample.mean_altitude_error);
                max = max.max(sample.mean_altitude_error);
            }
            if min.is_finite() && max.is_finite() {
                max - min
            } else {
                f64::NAN
            }
        })
        .filter(|delta| delta.is_finite())
        .fold(0.0, f64::max);

    NegativeControlReport {
        manifest: BenchmarkManifest {
            protocol_version: "helicopter-fixed-authority-v1".to_string(),
            scenario_id: "calm-hover-negative-control".to_string(),
            target_altitude_m,
            steps_per_evaluation,
            phi_values,
            seeds: seeds.to_vec(),
            bootstrap_replicates,
            actuator_authority_fixed: true,
        },
        samples,
        pearson_r,
        cluster_bootstrap_ci95,
        max_within_seed_altitude_delta,
    }
}

/// Pearson product-moment correlation, returning `None` when either variable
/// has zero variance or the input is invalid.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 || !x.iter().chain(y).all(|v| v.is_finite()) {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (&x_i, &y_i) in x.iter().zip(y) {
        let dx = x_i - mean_x;
        let dy = y_i - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    let denominator = (variance_x * variance_y).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    }
}

fn cluster_bootstrap_correlation_ci(
    samples: &[BenchmarkSample],
    seeds: &[u64],
    replicates: usize,
) -> Option<[f64; 2]> {
    if seeds.len() < 2 || replicates < 2 {
        return None;
    }
    let mut rng = XorShift64::new(0x4845_4c49_434f_5054);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..seeds.len() {
            let selected_seed = seeds[rng.next_index(seeds.len())];
            for sample in samples.iter().filter(|sample| sample.seed == selected_seed) {
                x.push(sample.phi);
                y.push(sample.mean_altitude_error);
            }
        }
        if let Some(r) = pearson_correlation(&x, &y) {
            estimates.push(r);
        }
    }
    if estimates.len() < 2 {
        return None;
    }
    estimates.sort_by(f64::total_cmp);
    let low = percentile_index(estimates.len(), 0.025);
    let high = percentile_index(estimates.len(), 0.975);
    Some([estimates[low], estimates[high]])
}

fn percentile_index(len: usize, quantile: f64) -> usize {
    (((len - 1) as f64 * quantile).round() as usize).min(len - 1)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_index(&mut self, len: usize) -> usize {
        (self.next_u64() as usize) % len
    }
}

fn mean_or_nan(sum: f64, count: usize) -> f64 {
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_benchmark_runs() {
        let config = HelicopterConfig {
            steps_per_episode: 300,
            ..Default::default()
        };
        let result = benchmark_hover(&config, 20.0, 300, 0.7);
        assert!(result.steps_survived > 0);
        assert!(result.mean_altitude_error.is_finite());
        assert!(result.mean_control_effort >= 0.0);
    }

    #[test]
    fn test_phi_does_not_change_actuator_authority() {
        let config = HelicopterConfig {
            steps_per_episode: 300,
            ..Default::default()
        };
        let high_phi = benchmark_hover(&config, 20.0, 300, 0.8);
        let low_phi = benchmark_hover(&config, 20.0, 300, 0.05);

        assert_eq!(low_phi.mean_control_effort, high_phi.mean_control_effort);
        assert_eq!(low_phi.mean_altitude_error, high_phi.mean_altitude_error);
    }

    #[test]
    fn test_perturbation_recovery_benchmark() {
        let config = HelicopterConfig {
            steps_per_episode: 900,
            ..Default::default()
        };
        let result = benchmark_perturbation_recovery(&config, 20.0, 0.7);
        assert!(result.pre_perturbation_error.is_finite());
        assert!(result.during_perturbation_error.is_finite());
        assert!(result.complete_windows || result.crashed);
    }

    #[test]
    fn test_phi_sweep() {
        let config = HelicopterConfig {
            steps_per_episode: 100,
            ..Default::default()
        };
        let results = phi_correlation_sweep(&config, 20.0, 100);
        assert_eq!(results.len(), 11); // 11 phi values
        let first = &results[0].1;
        let last = &results[10].1;
        assert_eq!(last.mean_control_effort, first.mean_control_effort);
        assert_eq!(last.mean_altitude_error, first.mean_altitude_error);
    }

    #[test]
    fn test_pearson_correlation_known_values() {
        assert_eq!(
            pearson_correlation(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]),
            Some(1.0)
        );
        assert_eq!(
            pearson_correlation(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]),
            Some(-1.0)
        );
        assert_eq!(pearson_correlation(&[1.0, 1.0], &[2.0, 3.0]), None);
    }

    #[test]
    fn test_fixed_authority_report_is_balanced_and_serializable() {
        let config = HelicopterConfig {
            steps_per_episode: 30,
            ..Default::default()
        };
        let report = fixed_authority_negative_control(&config, 20.0, 30, &[1, 2], 20);
        assert_eq!(report.samples.len(), 22);
        assert!(report.manifest.actuator_authority_fixed);
        assert_eq!(report.max_within_seed_altitude_delta, 0.0);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("helicopter-fixed-authority-v1"));
    }
}
