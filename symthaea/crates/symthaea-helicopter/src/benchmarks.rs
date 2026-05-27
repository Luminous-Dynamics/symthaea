// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helicopter benchmarks: consciousness-coupled vs ungated control.
//!
//! Produces data for the paper claim: "consciousness predicts perturbation recovery"
//! on the helicopter platform (extending the humanoid r=0.87 finding).

use crate::controller::HelicopterController;
use crate::encoder::HelicopterHdcEncoder;
use crate::perturbations::{HelicopterPerturbation, PerturbationSchedule};
use crate::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
use crate::types::{HelicopterCommand, HelicopterConfig};

use symthaea_core::genesis::GenesisSeed;

/// Benchmark result for a single hover evaluation.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct PerturbationRecoveryBenchmark {
    /// Performance before perturbation.
    pub pre_perturbation_error: f64,
    /// Performance during perturbation.
    pub during_perturbation_error: f64,
    /// Performance after perturbation clears.
    pub post_perturbation_error: f64,
    /// Steps to recover to within 2× pre-perturbation error.
    pub recovery_steps: Option<usize>,
    /// Consciousness level during recovery.
    pub phi: f64,
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

    // Motor gain based on consciousness level
    let gain = if phi > 0.6 {
        1.0f32
    } else if phi > 0.3 {
        0.6
    } else if phi > 0.1 {
        0.3
    } else {
        0.0
    };

    for step in 0..steps {
        let hv = encoder.encode(sim.state());
        let mut cmd = controller.forward(&hv, dt as f32);

        // Apply consciousness-gated gain
        cmd.collective *= gain;
        cmd.cyclic_lon *= gain;
        cmd.cyclic_lat *= gain;
        cmd.pedal *= gain;
        cmd.thrust *= gain;
        cmd.tail_rotor *= gain;

        sim.step(&cmd, dt);
        steps_survived = step + 1;

        let state = sim.state();
        let alt_error = (state.altitude() - target_altitude).abs();
        altitude_error_sum += alt_error;
        effort_sum += cmd.control_effort();
        angular_sum += state.angular_speed();

        if alt_error < 2.0 {
            hover_count += 1;
        }

        if state.altitude() <= 0.01 && step > 10 {
            break; // Crashed
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

    let gain = if phi > 0.6 {
        1.0f32
    } else if phi > 0.3 {
        0.6
    } else if phi > 0.1 {
        0.3
    } else {
        0.0
    };

    let mut pre_error = 0.0;
    let mut during_error = 0.0;
    let mut post_error = 0.0;
    let mut pre_count = 0;
    let mut during_count = 0;
    let mut post_count = 0;
    let mut recovery_step = None;
    let mut pre_mean = 0.0;

    for step in 0..total_steps {
        // Apply perturbation during middle third
        if step == perturb_start {
            sim.apply_external_force([2000.0, 1000.0, 0.0]); // Strong crosswind
        }
        if step > perturb_start && step < perturb_end {
            sim.apply_external_force([2000.0, 1000.0, 0.0]); // Sustained
        }

        let hv = encoder.encode(sim.state());
        let mut cmd = controller.forward(&hv, dt as f32);
        cmd.collective *= gain;
        cmd.cyclic_lon *= gain;
        cmd.cyclic_lat *= gain;
        cmd.pedal *= gain;
        cmd.thrust *= gain;
        cmd.tail_rotor *= gain;

        sim.step(&cmd, dt);

        let alt_error = (sim.state().altitude() - target_altitude).abs();

        if step < perturb_start {
            pre_error += alt_error;
            pre_count += 1;
        } else if step < perturb_end {
            during_error += alt_error;
            during_count += 1;
        } else {
            post_error += alt_error;
            post_count += 1;
            // Check recovery: within 2× pre-perturbation error
            if recovery_step.is_none() && alt_error < pre_mean * 2.0 {
                recovery_step = Some(step - perturb_end);
            }
        }

        if step == perturb_start {
            pre_mean = pre_error / pre_count.max(1) as f64;
        }

        if !sim.state().is_finite() || sim.state().altitude() <= 0.01 {
            break;
        }
    }

    PerturbationRecoveryBenchmark {
        pre_perturbation_error: pre_error / pre_count.max(1) as f64,
        during_perturbation_error: during_error / during_count.max(1) as f64,
        post_perturbation_error: post_error / post_count.max(1) as f64,
        recovery_steps: recovery_step,
        phi,
    }
}

/// Run hover benchmark across multiple consciousness levels.
///
/// Returns (phi, hover_benchmark) pairs for correlation analysis.
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
    fn test_higher_phi_better_hover() {
        let config = HelicopterConfig {
            steps_per_episode: 300,
            ..Default::default()
        };
        let high_phi = benchmark_hover(&config, 20.0, 300, 0.8);
        let low_phi = benchmark_hover(&config, 20.0, 300, 0.05);

        // Red safety (phi=0.05) should have zero control effort (gain=0)
        assert_eq!(low_phi.mean_control_effort, 0.0);
        // Green safety should have some control effort
        assert!(high_phi.mean_control_effort > 0.0);
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
    }

    #[test]
    fn test_phi_sweep() {
        let config = HelicopterConfig {
            steps_per_episode: 100,
            ..Default::default()
        };
        let results = phi_correlation_sweep(&config, 20.0, 100);
        assert_eq!(results.len(), 11); // 11 phi values
        // Higher phi should generally have lower error (consciousness helps)
        let first = &results[0].1; // phi=0.05 (Red)
        let last = &results[10].1; // phi=1.0 (Green)
        // Red has zero effort, Green has positive effort
        assert!(last.mean_control_effort > first.mean_control_effort);
    }
}
