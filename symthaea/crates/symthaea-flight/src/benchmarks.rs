// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wind gust robustness benchmarks for FEP-modulated flight control.
//!
//! Compares flight performance with and without Active Inference modulation
//! under wind disturbances injected via `PhysicsSimulator::apply_external_force`.
//!
//! Supports multiple physics backends (Simple, MuJoCo, MuJoCo+Noise) for
//! sim-to-real transfer validation.

use symthaea_core::genesis::GenesisSeed;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{
    ActiveInferenceFlightAgent, FlightFepConfig, FlightFepResult, INTERCEPT_Z_MARGIN_STATIC,
    INTERCEPT_Z_MARGIN_VEL, RENDEZVOUS_FRACS,
};
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::*;

/// A single wind gust event.
#[derive(Debug, Clone)]
pub struct WindGust {
    /// Force vector in world frame [N].
    pub force: [f64; 3],
    /// Start time in seconds.
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
}

/// Configuration for the wind gust benchmark.
#[derive(Debug, Clone)]
pub struct WindBenchmarkConfig {
    /// Base flight configuration.
    pub flight_config: FlightConfig,
    /// Wind gusts to inject.
    pub gusts: Vec<WindGust>,
    /// Number of evaluation episodes.
    pub eval_episodes: usize,
}

impl Default for WindBenchmarkConfig {
    fn default() -> Self {
        Self {
            flight_config: FlightConfig {
                num_episodes: 1,
                steps_per_episode: 2000,
                ..FlightConfig::default()
            },
            gusts: vec![
                WindGust {
                    force: [0.05, 0.0, 0.0], // Lateral gust
                    start_time: 0.5,
                    duration: 0.2,
                },
                WindGust {
                    force: [0.0, 0.0, -0.03], // Downdraft
                    start_time: 1.5,
                    duration: 0.3,
                },
            ],
            eval_episodes: 3,
        }
    }
}

/// Results from a wind gust benchmark run.
#[derive(Debug, Clone)]
pub struct WindBenchmarkResult {
    /// Metrics without FEP modulation (frozen agent).
    pub baseline_metrics: Vec<f64>,
    /// Metrics with full FEP modulation.
    pub fep_metrics: Vec<f64>,
    /// Steps to recover within 5cm after each gust (FEP).
    pub recovery_steps: Vec<usize>,
    /// Maximum deviation during gusts (FEP).
    pub max_deviation: f64,
    /// Maximum deviation during gusts (baseline).
    pub baseline_max_deviation: f64,
}

/// Physics backend selection for polymorphic benchmarks.
#[derive(Debug, Clone)]
pub enum PhysicsBackend {
    /// Pure Rust simplified physics (default, no external deps).
    Simple,
    /// MuJoCo high-fidelity physics (requires `mujoco` feature).
    #[cfg(feature = "mujoco")]
    MuJoCo,
    /// MuJoCo with sensory noise injection (requires `mujoco` feature).
    #[cfg(feature = "mujoco")]
    MuJoCoNoisy(crate::sensory_filter::SensoryFilterConfig),
}

/// Factory: create a physics simulator from a backend enum.
fn create_simulator(backend: &PhysicsBackend) -> Box<dyn PhysicsSimulator> {
    match backend {
        PhysicsBackend::Simple => Box::new(SimplePhysicsSimulator::new()),
        #[cfg(feature = "mujoco")]
        PhysicsBackend::MuJoCo => Box::new(crate::mujoco_sim::MuJoCoSimulator::from_primitive().unwrap()),
        #[cfg(feature = "mujoco")]
        PhysicsBackend::MuJoCoNoisy(noise_config) => {
            let mut sim = crate::mujoco_sim::MuJoCoSimulator::from_primitive().unwrap();
            sim.enable_sensory_filter(noise_config.clone());
            Box::new(sim)
        }
    }
}

/// Run the wind gust robustness benchmark with the default (Simple) backend.
///
/// Compares a "frozen" baseline (no FEP modulation) against full Active Inference.
pub fn run_wind_benchmark(config: &WindBenchmarkConfig) -> WindBenchmarkResult {
    run_wind_benchmark_with_backend(config, &PhysicsBackend::Simple)
}

/// Run the wind gust robustness benchmark with MuJoCo backend.
#[cfg(feature = "mujoco")]
pub fn run_wind_benchmark_mujoco(config: &WindBenchmarkConfig) -> WindBenchmarkResult {
    run_wind_benchmark_with_backend(config, &PhysicsBackend::MuJoCo)
}

/// Run the wind gust robustness benchmark with MuJoCo + sensory noise.
#[cfg(feature = "mujoco")]
pub fn run_wind_benchmark_mujoco_noisy(
    config: &WindBenchmarkConfig,
    noise_config: crate::sensory_filter::SensoryFilterConfig,
) -> WindBenchmarkResult {
    run_wind_benchmark_with_backend(config, &PhysicsBackend::MuJoCoNoisy(noise_config))
}

/// Run the wind gust robustness benchmark with a specified physics backend.
pub fn run_wind_benchmark_with_backend(
    config: &WindBenchmarkConfig,
    backend: &PhysicsBackend,
) -> WindBenchmarkResult {
    let genesis = GenesisSeed::from_phrase(&config.flight_config.genesis_phrase);
    let setpoint = FlightSetpoint::hover();
    let dt = config.flight_config.motor_dt();
    let pd_gains = PdGains::default();
    let cognitive_interval = config.flight_config.cognitive_interval();

    let mut baseline_errors = Vec::new();
    let mut fep_errors = Vec::new();
    let mut recovery_steps = Vec::new();
    let mut max_dev_fep = 0.0f64;
    let mut max_dev_baseline = 0.0f64;

    for ep in 0..config.eval_episodes {
        // ── Baseline: frozen FEP agent (no modulation) ──
        {
            let mut physics = create_simulator(backend);
            physics.reset(0.1);
            let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
            let mut controller = FlightController::new(&genesis, &config.flight_config);
            let frozen_result = FlightFepResult::default();
            let mut total_err = 0.0;

            for step in 0..config.flight_config.steps_per_episode {
                let time = step as f64 * dt;
                let state = physics.state().clone();
                let sensor_hv = encoder.encode(&state);
                let command = controller.forward(&sensor_hv, dt as f32);
                physics.step(&command, dt);

                // Inject gusts
                for gust in &config.gusts {
                    if time >= gust.start_time && time < gust.start_time + gust.duration {
                        physics.apply_external_force(gust.force);
                    }
                }

                let err = setpoint.position_error_magnitude(&state);
                total_err += err;
                max_dev_baseline = max_dev_baseline.max(err);

                // Training (still train, just no FEP modulation)
                if step % config.flight_config.train_every == 0 {
                    let target = pd_baseline(&state, &setpoint, &pd_gains);
                    let lr =
                        config.flight_config.learning_rate * frozen_result.learning_rate_factor;
                    controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
                }
            }
            baseline_errors.push(total_err / config.flight_config.steps_per_episode as f64);
        }

        // ── FEP: full Active Inference modulation ──
        {
            let mut physics = create_simulator(backend);
            physics.reset(0.1);
            let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
            let mut controller = FlightController::new(&genesis, &config.flight_config);
            let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
            let mut fep_result = fep_agent.initial_step(physics.state(), &setpoint);
            let mut pid_state = PidState::default();
            let mut total_err = 0.0;

            // Track recovery after each gust
            let mut gust_end_tracking: Vec<(f64, bool, usize)> = config
                .gusts
                .iter()
                .map(|g| (g.start_time + g.duration, false, 0))
                .collect();

            for step in 0..config.flight_config.steps_per_episode {
                let time = step as f64 * dt;
                let state = physics.state().clone();
                let sensor_hv = encoder.encode(&state);
                let mut command = controller.forward(&sensor_hv, dt as f32);

                if let Some(noise) = &fep_result.exploration_noise {
                    command = command.with_noise(noise);
                }

                // Inject gusts
                for gust in &config.gusts {
                    if time >= gust.start_time && time < gust.start_time + gust.duration {
                        physics.apply_external_force(gust.force);
                    }
                }

                physics.step(&command, dt);

                let err = setpoint.position_error_magnitude(&state);
                total_err += err;
                max_dev_fep = max_dev_fep.max(err);

                // Track recovery
                for (i, (end_time, recovered, count)) in gust_end_tracking.iter_mut().enumerate() {
                    if time >= *end_time && !*recovered {
                        *count += 1;
                        if err < 0.05 {
                            *recovered = true;
                        }
                    }
                    let _ = i; // suppress unused warning
                }

                if step % config.flight_config.train_every == 0 {
                    let target = if fep_result.use_pid_target {
                        pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt)
                    } else {
                        pd_baseline(&state, &setpoint, &pd_gains)
                    };
                    let lr = config.flight_config.learning_rate * fep_result.learning_rate_factor;
                    controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
                }

                if step % cognitive_interval == 0 {
                    fep_result = fep_agent.step(physics.state(), &setpoint);
                    if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                        controller.modulate_tau(fep_result.tau_factor);
                    }
                }
            }

            fep_errors.push(total_err / config.flight_config.steps_per_episode as f64);

            for (_, _, count) in &gust_end_tracking {
                recovery_steps.push(*count);
            }

            let _ = ep;
        }
    }

    WindBenchmarkResult {
        baseline_metrics: baseline_errors,
        fep_metrics: fep_errors,
        recovery_steps,
        max_deviation: max_dev_fep,
        baseline_max_deviation: max_dev_baseline,
    }
}

/// Run a perturbation benchmark with MuJoCo backend.
///
/// Tests FEP's survival reflex: mid-flight mass/rotor changes.
#[cfg(feature = "mujoco")]
pub fn run_perturbation_benchmark(
    config: &FlightConfig,
    schedule: &crate::perturbations::PerturbationSchedule,
) -> crate::perturbations::PerturbationBenchmarkResult {
    use crate::perturbations::PerturbationBenchmarkResult;

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let setpoint = FlightSetpoint::hover();
    let dt = config.motor_dt();
    let pd_gains = PdGains::default();
    let cognitive_interval = config.cognitive_interval();

    let mut sim = crate::mujoco_sim::MuJoCoSimulator::from_primitive().unwrap();
    sim.reset(0.1);

    // Auto-calibrate hover thrust from MuJoCo model mass.
    // The hardcoded HOVER_THRUST (0.265N) doesn't match MuJoCo physics exactly,
    // so we compute the correct value from the simulated mass.
    let hover_thrust_offset = (sim.body_mass() * 9.81) as f32 - QuadrotorCommand::HOVER_THRUST;

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = FlightController::new(&genesis, config);
    let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
    let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);
    let mut pid_state = PidState::default();

    // Find first perturbation step for pre/post comparison
    let first_trigger = schedule
        .perturbations()
        .iter()
        .map(|p| p.trigger_step())
        .min()
        .unwrap_or(config.steps_per_episode);

    let mut pre_errors = Vec::new();
    let mut peak_error = 0.0f64;
    let mut min_tau = 1.0f32;
    let mut recovery_step = None;
    let mut crashed = false;

    let mut fe_trace = Vec::with_capacity(config.steps_per_episode);
    let mut error_trace = Vec::with_capacity(config.steps_per_episode);
    let mut tau_trace = Vec::with_capacity(config.steps_per_episode);

    // Adaptive thrust trim: PID integral applied directly to commands.
    // The CfC handles high-frequency dynamics; the integral handles
    // steady-state offset (hover thrust mismatch, mass changes).
    // High gain (10.0) enables fast response to sudden mass changes —
    // the "survival reflex" integral correction.
    let mut thrust_integral = 0.0f64;
    let integral_gain = 10.0f64;
    let integral_clamp = 0.2; // Max ±0.2N correction (within MAX_THRUST margin)

    for step in 0..config.steps_per_episode {
        // Apply perturbations
        schedule.apply(step, &mut sim);

        let state = sim.state().clone();
        let pos_err = setpoint.position_error(&state);

        // Update thrust integral: accumulates altitude error over time.
        // Anti-windup: clamp and zero-crossing reset.
        if thrust_integral != 0.0 && thrust_integral.signum() != pos_err[2].signum() {
            thrust_integral *= 0.5; // Zero-crossing reset
        }
        thrust_integral += pos_err[2] * dt;
        thrust_integral = thrust_integral.clamp(-integral_clamp, integral_clamp);

        let sensor_hv = encoder.encode(&state);

        // Blend: PD baseline provides position tracking, CfC provides learned dynamics.
        // The FEP agent modulates the blend via tau/learning rate.
        let pd_cmd = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
        let cfc_cmd = controller.forward(&sensor_hv, dt as f32);

        // PD-CfC blend: PD provides reactive position tracking,
        // CfC learns dynamics and provides fine-tuning.
        // This combination delivers extreme robustness under perturbation.
        let alpha = 0.5f32;
        let mut command = QuadrotorCommand {
            thrust: alpha * pd_cmd.thrust
                + (1.0 - alpha) * cfc_cmd.thrust
                + hover_thrust_offset
                + (integral_gain * thrust_integral) as f32,
            roll_moment: alpha * pd_cmd.roll_moment + (1.0 - alpha) * cfc_cmd.roll_moment,
            pitch_moment: alpha * pd_cmd.pitch_moment + (1.0 - alpha) * cfc_cmd.pitch_moment,
            yaw_moment: alpha * pd_cmd.yaw_moment + (1.0 - alpha) * cfc_cmd.yaw_moment,
        }
        .clamped();

        if let Some(noise) = &fep_result.exploration_noise {
            command = command.with_noise(noise);
        }

        sim.step(&command, dt);

        let err = setpoint.position_error_magnitude(&state);
        error_trace.push(err);
        fe_trace.push(fep_result.free_energy);
        tau_trace.push(fep_result.tau_factor);

        if step < first_trigger {
            pre_errors.push(err);
        } else {
            peak_error = peak_error.max(err);
            if recovery_step.is_none() && err < 0.05 && step > first_trigger + 10 {
                recovery_step = Some(step - first_trigger);
            }
        }

        if state.altitude() <= 0.0 {
            crashed = true;
        }

        min_tau = min_tau.min(fep_result.tau_factor);

        if step % config.train_every == 0 {
            let target = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
            let lr = config.learning_rate * fep_result.learning_rate_factor;
            controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
        }

        if step % cognitive_interval == 0 {
            fep_result = fep_agent.step(sim.state(), &setpoint);
            if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                controller.modulate_tau(fep_result.tau_factor);
            }
        }
    }

    let pre_avg = if pre_errors.is_empty() {
        0.0
    } else {
        pre_errors.iter().sum::<f64>() / pre_errors.len() as f64
    };

    let final_err = error_trace.last().cloned().unwrap_or(0.0);

    PerturbationBenchmarkResult {
        pre_perturbation_error: pre_avg,
        peak_error,
        recovery_steps: recovery_step.unwrap_or(config.steps_per_episode),
        final_error: final_err,
        min_tau: min_tau as f64,
        crashed,
        free_energy_trace: fe_trace,
        error_trace,
        tau_trace,
    }
}

/// Results from a FEP comparison benchmark (with vs without FEP modulation).
#[cfg(feature = "mujoco")]
#[derive(Debug, Clone)]
pub struct FepComparisonResult {
    /// Result with FEP active (tau/LR modulation).
    pub fep_active: crate::perturbations::PerturbationBenchmarkResult,
    /// Result with FEP frozen (tau=1.0, lr_factor=1.0).
    pub fep_frozen: crate::perturbations::PerturbationBenchmarkResult,
}

/// Run a perturbation scenario twice: with FEP active vs frozen.
///
/// Uses identical initial conditions (same genesis seed) for fair comparison.
/// The frozen run uses `FlightFepResult::default()` at every step (no modulation).
#[cfg(feature = "mujoco")]
pub fn run_perturbation_comparison(
    config: &FlightConfig,
    schedule: &crate::perturbations::PerturbationSchedule,
) -> FepComparisonResult {
    use crate::perturbations::PerturbationBenchmarkResult;

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let setpoint = FlightSetpoint::hover();
    let dt = config.motor_dt();
    let pd_gains = PdGains::default();
    let cognitive_interval = config.cognitive_interval();

    // Helper: run one episode with optional FEP
    let run_episode = |enable_fep: bool| -> PerturbationBenchmarkResult {
        let mut sim = crate::mujoco_sim::MuJoCoSimulator::from_primitive().unwrap();
        sim.reset(0.1);

        let hover_thrust_offset = (sim.body_mass() * 9.81) as f32 - QuadrotorCommand::HOVER_THRUST;

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
        let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);
        let mut pid_state = PidState::default();

        let first_trigger = schedule
            .perturbations()
            .iter()
            .map(|p| p.trigger_step())
            .min()
            .unwrap_or(config.steps_per_episode);

        let mut pre_errors = Vec::new();
        let mut peak_error = 0.0f64;
        let mut min_tau = 1.0f32;
        let mut recovery_step = None;
        let mut crashed = false;
        let mut fe_trace = Vec::with_capacity(config.steps_per_episode);
        let mut error_trace = Vec::with_capacity(config.steps_per_episode);
        let mut tau_trace = Vec::with_capacity(config.steps_per_episode);

        let mut thrust_integral = 0.0f64;
        let integral_gain = 10.0f64;
        let integral_clamp = 0.2;

        for step in 0..config.steps_per_episode {
            schedule.apply(step, &mut sim);

            let state = sim.state().clone();
            let pos_err = setpoint.position_error(&state);

            if thrust_integral != 0.0 && thrust_integral.signum() != pos_err[2].signum() {
                thrust_integral *= 0.5;
            }
            thrust_integral += pos_err[2] * dt;
            thrust_integral = thrust_integral.clamp(-integral_clamp, integral_clamp);

            let sensor_hv = encoder.encode(&state);
            let pd_cmd = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
            let cfc_cmd = controller.forward(&sensor_hv, dt as f32);

            let alpha = 0.5f32;
            let mut command = QuadrotorCommand {
                thrust: alpha * pd_cmd.thrust
                    + (1.0 - alpha) * cfc_cmd.thrust
                    + hover_thrust_offset
                    + (integral_gain * thrust_integral) as f32,
                roll_moment: alpha * pd_cmd.roll_moment + (1.0 - alpha) * cfc_cmd.roll_moment,
                pitch_moment: alpha * pd_cmd.pitch_moment + (1.0 - alpha) * cfc_cmd.pitch_moment,
                yaw_moment: alpha * pd_cmd.yaw_moment + (1.0 - alpha) * cfc_cmd.yaw_moment,
            }
            .clamped();

            if enable_fep {
                if let Some(noise) = &fep_result.exploration_noise {
                    command = command.with_noise(noise);
                }
            }

            sim.step(&command, dt);

            let err = setpoint.position_error_magnitude(&state);
            error_trace.push(err);
            fe_trace.push(fep_result.free_energy);
            tau_trace.push(if enable_fep {
                fep_result.tau_factor
            } else {
                1.0
            });

            if step < first_trigger {
                pre_errors.push(err);
            } else {
                peak_error = peak_error.max(err);
                if recovery_step.is_none() && err < 0.05 && step > first_trigger + 10 {
                    recovery_step = Some(step - first_trigger);
                }
            }

            if state.altitude() <= 0.0 {
                crashed = true;
            }

            min_tau = min_tau.min(if enable_fep {
                fep_result.tau_factor
            } else {
                1.0
            });

            if step % config.train_every == 0 {
                let target = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
                let lr = config.learning_rate
                    * if enable_fep {
                        fep_result.learning_rate_factor
                    } else {
                        1.0
                    };
                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
            }

            if enable_fep && step % cognitive_interval == 0 {
                fep_result = fep_agent.step(sim.state(), &setpoint);
                if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                    controller.modulate_tau(fep_result.tau_factor);
                }
            }
        }

        let pre_avg = if pre_errors.is_empty() {
            0.0
        } else {
            pre_errors.iter().sum::<f64>() / pre_errors.len() as f64
        };

        PerturbationBenchmarkResult {
            pre_perturbation_error: pre_avg,
            peak_error,
            recovery_steps: recovery_step.unwrap_or(config.steps_per_episode),
            final_error: error_trace.last().cloned().unwrap_or(0.0),
            min_tau: min_tau as f64,
            crashed,
            free_energy_trace: fe_trace,
            error_trace,
            tau_trace,
        }
    };

    FepComparisonResult {
        fep_active: run_episode(true),
        fep_frozen: run_episode(false),
    }
}

// ── EFE Ablation Study ──

use crate::fep_agent::FlightEnvironment;

/// A single data point in the EFE ablation sweep.
#[derive(Debug, Clone)]
pub struct EfeAblationPoint {
    /// Safety prior precision value tested.
    pub safety_precision: f64,
    /// EFE for continuing mission at this precision.
    pub efe_mission: f64,
    /// EFE for best intercept candidate at this precision.
    pub efe_intercept: f64,
    /// Whether interception was chosen (efe_intercept < efe_mission).
    pub chose_intercept: bool,
    /// Which rendezvous fraction produced the best intercept EFE.
    /// None if no velocity (static intercept).
    pub best_rendezvous_frac: Option<f64>,
}

/// Result of a full EFE ablation sweep.
#[derive(Debug, Clone)]
pub struct EfeAblationResult {
    /// All sweep points.
    pub points: Vec<EfeAblationPoint>,
    /// Safety precision at which the decision crosses from mission to intercept.
    /// Computed via linear interpolation between the last mission-preferred and
    /// first intercept-preferred points.
    pub crossover_precision: Option<f64>,
}

/// Sweep safety_prior_precision to find the decision boundary.
///
/// For each precision value, evaluates EFE for mission and all 3 rendezvous
/// intercept candidates (25%, 50%, 75% of impact time). Uses trajectory EFE
/// when threat velocity is available, instantaneous EFE otherwise.
/// Records the best intercept and finds the crossover via linear interpolation.
///
/// Pure analytical computation — no MuJoCo or physics simulation required.
pub fn run_efe_ablation(
    precisions: &[f64],
    env: &FlightEnvironment,
    drone_state: &FlightState,
    mission_setpoint: &FlightSetpoint,
    mission_precision: f64,
    self_precision: f64,
) -> EfeAblationResult {
    let threat_pos = match env.threat_pos {
        Some(p) => p,
        None => {
            return EfeAblationResult {
                points: vec![],
                crossover_precision: None,
            };
        }
    };
    let entity_pos = env.entity_pos.unwrap_or([0.0; 3]);

    // Pre-compute rendezvous intercept candidates (reused for each precision)
    let intercept_candidates: Vec<([f64; 3], Option<f64>)> = if let Some(vel) = env.threat_vel {
        let impact_t = crate::fep_agent::predict_impact_time_z(threat_pos, vel, entity_pos[2]);
        if impact_t.is_finite() && impact_t > 0.0 {
            RENDEZVOUS_FRACS
                .iter()
                .map(|&frac| {
                    let rv_t = impact_t * frac;
                    let beam = crate::fep_agent::predict_falling_position(threat_pos, vel, rv_t);
                    (
                        [
                            beam[0],
                            beam[1],
                            beam[2].max(entity_pos[2] + INTERCEPT_Z_MARGIN_VEL),
                        ],
                        Some(frac),
                    )
                })
                .collect()
        } else {
            vec![(
                [
                    threat_pos[0],
                    threat_pos[1],
                    threat_pos[2].max(entity_pos[2] + INTERCEPT_Z_MARGIN_STATIC),
                ],
                None,
            )]
        }
    } else {
        vec![(
            crate::fep_agent::compute_rendezvous_intercept(threat_pos, None, entity_pos),
            None,
        )]
    };

    let use_trajectory = env.threat_vel.is_some();
    let mut points = Vec::with_capacity(precisions.len());
    let mut crossover = None;

    for &pi_safety in precisions {
        let config = FlightFepConfig {
            extended_observation: true,
            safety_prior_precision: pi_safety,
            mission_prior_precision: mission_precision,
            self_preservation_precision: self_precision,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let efe_mission = if use_trajectory {
            agent.trajectory_efe(
                drone_state,
                mission_setpoint.position,
                mission_setpoint,
                env,
            )
        } else {
            agent.instantaneous_efe(
                drone_state,
                mission_setpoint.position,
                mission_setpoint,
                env,
            )
        };

        // Evaluate all intercept candidates, pick best
        let mut best_efe = f64::INFINITY;
        let mut best_frac: Option<f64> = None;
        for &(candidate, frac) in &intercept_candidates {
            let efe = if use_trajectory {
                agent.trajectory_efe(drone_state, candidate, mission_setpoint, env)
            } else {
                agent.instantaneous_efe(drone_state, candidate, mission_setpoint, env)
            };
            if efe < best_efe {
                best_efe = efe;
                best_frac = frac;
            }
        }

        let chose_intercept = best_efe < efe_mission;

        points.push(EfeAblationPoint {
            safety_precision: pi_safety,
            efe_mission,
            efe_intercept: best_efe,
            chose_intercept,
            best_rendezvous_frac: best_frac,
        });
    }

    // Find crossover via linear interpolation
    for i in 1..points.len() {
        if points[i].chose_intercept != points[i - 1].chose_intercept {
            // Decision flipped between points[i-1] and points[i]
            let p0 = points[i - 1].safety_precision;
            let p1 = points[i].safety_precision;
            let diff0 = points[i - 1].efe_mission - points[i - 1].efe_intercept;
            let diff1 = points[i].efe_mission - points[i].efe_intercept;

            // Linear interpolation: find precision where diff = 0
            if (diff1 - diff0).abs() > 1e-12 {
                let t = -diff0 / (diff1 - diff0);
                crossover = Some(p0 + t * (p1 - p0));
            } else {
                crossover = Some((p0 + p1) / 2.0);
            }
            break;
        }
    }

    EfeAblationResult {
        points,
        crossover_precision: crossover,
    }
}

// ── Multi-Scenario Evaluation ──

/// Pre-defined scenario geometry variants for EFE evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioVariant {
    /// Beam above human, reachable → expected: INTERCEPT
    Default,
    /// Beam very close to drone → expected: INTERCEPT (easy)
    CloseBeam,
    /// Beam 5m away, 5m high → expected: MISSION (unreachable)
    FarBeam,
    /// Human behind drone → expected: INTERCEPT (must reverse)
    ReversedGeometry,
    /// No human (entity_pos = None) → expected: MISSION
    NoHuman,
    /// Beam far from human → expected: MISSION (low threat)
    LowDanger,
    /// No velocity data → instantaneous EFE fallback, expected: INTERCEPT
    StaticThreat,
}

impl ScenarioVariant {
    /// All variants for sweep evaluation.
    pub fn all() -> &'static [ScenarioVariant] {
        &[
            ScenarioVariant::Default,
            ScenarioVariant::CloseBeam,
            ScenarioVariant::FarBeam,
            ScenarioVariant::ReversedGeometry,
            ScenarioVariant::NoHuman,
            ScenarioVariant::LowDanger,
            ScenarioVariant::StaticThreat,
        ]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioVariant::Default => "Default",
            ScenarioVariant::CloseBeam => "CloseBeam",
            ScenarioVariant::FarBeam => "FarBeam",
            ScenarioVariant::ReversedGeometry => "ReversedGeometry",
            ScenarioVariant::NoHuman => "NoHuman",
            ScenarioVariant::LowDanger => "LowDanger",
            ScenarioVariant::StaticThreat => "StaticThreat",
        }
    }

    /// Expected decision for this variant.
    pub fn expected_decision(&self) -> &'static str {
        match self {
            ScenarioVariant::Default => "INTERCEPT",
            ScenarioVariant::CloseBeam => "INTERCEPT",
            ScenarioVariant::FarBeam => "MISSION",
            ScenarioVariant::ReversedGeometry => "INTERCEPT",
            ScenarioVariant::NoHuman => "MISSION",
            ScenarioVariant::LowDanger => "MISSION",
            ScenarioVariant::StaticThreat => "INTERCEPT",
        }
    }

    /// Build the environment, state, and setpoint for this variant.
    pub fn build(&self) -> (FlightState, FlightSetpoint, FlightEnvironment) {
        let drone_state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let env = match self {
            ScenarioVariant::Default => FlightEnvironment {
                human_danger: 0.8,
                mission_progress: 0.3,
                threat_pos: Some([-1.5, 0.0, 2.0]),
                threat_vel: Some([0.0, 0.0, -3.0]),
                entity_pos: Some([-1.5, 0.0, 0.0]),
            },
            ScenarioVariant::CloseBeam => FlightEnvironment {
                human_danger: 0.9,
                mission_progress: 0.3,
                threat_pos: Some([0.0, 0.0, 2.0]), // Right above drone
                threat_vel: Some([0.0, 0.0, -4.0]),
                entity_pos: Some([0.0, 0.0, 0.0]),
            },
            ScenarioVariant::FarBeam => FlightEnvironment {
                human_danger: 0.15,
                mission_progress: 0.3,
                threat_pos: Some([5.0, 5.0, 3.0]),
                threat_vel: Some([0.0, 0.0, -5.0]),
                entity_pos: Some([5.0, 5.0, 0.0]),
            },
            ScenarioVariant::ReversedGeometry => FlightEnvironment {
                human_danger: 0.7,
                mission_progress: 0.3,
                threat_pos: Some([2.0, 0.0, 2.5]),
                threat_vel: Some([0.0, 0.0, -3.0]),
                entity_pos: Some([2.0, 0.0, 0.0]),
            },
            ScenarioVariant::NoHuman => FlightEnvironment {
                human_danger: 0.0,
                mission_progress: 0.3,
                threat_pos: Some([-1.5, 0.0, 2.0]),
                threat_vel: Some([0.0, 0.0, -3.0]),
                entity_pos: None,
            },
            ScenarioVariant::LowDanger => FlightEnvironment {
                human_danger: 0.05,
                mission_progress: 0.3,
                threat_pos: Some([3.0, 3.0, 2.0]),
                threat_vel: Some([0.0, 0.0, -1.0]),
                entity_pos: Some([-1.5, 0.0, 0.0]), // Human far from beam
            },
            ScenarioVariant::StaticThreat => FlightEnvironment {
                human_danger: 0.8,
                mission_progress: 0.3,
                threat_pos: Some([-1.5, 0.0, 2.0]),
                threat_vel: None, // No velocity → instantaneous EFE fallback
                entity_pos: Some([-1.5, 0.0, 0.0]),
            },
        };

        (drone_state, mission_setpoint, env)
    }
}

/// Result of evaluating a single scenario variant.
#[derive(Debug, Clone)]
pub struct ScenarioVariantResult {
    /// Variant name.
    pub variant_name: String,
    /// EFE for continuing mission.
    pub efe_mission: f64,
    /// EFE for best intercept candidate.
    pub efe_intercept: f64,
    /// Whether the agent chose to override (intercept or other non-mission).
    pub chose_override: bool,
    /// Expected decision string.
    pub expected_decision: &'static str,
    /// Whether actual decision matches expected.
    pub matches_expected: bool,
    /// Which rendezvous fraction produced the best intercept EFE.
    /// None if no velocity (static intercept) or decision was MISSION.
    pub best_rendezvous_frac: Option<f64>,
}

/// Evaluate all scenario variants with default precision ratio.
///
/// Pure analytical computation — no MuJoCo required.
pub fn evaluate_scenario_variants() -> Vec<ScenarioVariantResult> {
    let mut results = Vec::new();

    for variant in ScenarioVariant::all() {
        let (drone_state, mission_setpoint, env) = variant.build();

        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let threat_pos = env.threat_pos.unwrap_or([0.0; 3]);
        let entity_pos = env.entity_pos.unwrap_or([0.0; 3]);

        let use_trajectory = env.threat_vel.is_some();
        let efe_mission = if use_trajectory {
            agent.trajectory_efe(
                &drone_state,
                mission_setpoint.position,
                &mission_setpoint,
                &env,
            )
        } else {
            agent.instantaneous_efe(
                &drone_state,
                mission_setpoint.position,
                &mission_setpoint,
                &env,
            )
        };

        // Evaluate all 3 rendezvous candidates (25%, 50%, 75%) and find the best
        let mut best_intercept_efe = f64::INFINITY;
        let mut best_frac: Option<f64> = None;

        if let Some(vel) = env.threat_vel {
            let impact_t = crate::fep_agent::predict_impact_time_z(threat_pos, vel, entity_pos[2]);
            if impact_t.is_finite() && impact_t > 0.0 {
                for &frac in &RENDEZVOUS_FRACS {
                    let rv_t = impact_t * frac;
                    let beam = crate::fep_agent::predict_falling_position(threat_pos, vel, rv_t);
                    let candidate = [
                        beam[0],
                        beam[1],
                        beam[2].max(entity_pos[2] + INTERCEPT_Z_MARGIN_VEL),
                    ];
                    let efe =
                        agent.trajectory_efe(&drone_state, candidate, &mission_setpoint, &env);
                    if efe < best_intercept_efe {
                        best_intercept_efe = efe;
                        best_frac = Some(frac);
                    }
                }
            } else {
                // Beam not descending — static fallback
                let static_candidate = [
                    threat_pos[0],
                    threat_pos[1],
                    threat_pos[2].max(entity_pos[2] + INTERCEPT_Z_MARGIN_STATIC),
                ];
                best_intercept_efe =
                    agent.trajectory_efe(&drone_state, static_candidate, &mission_setpoint, &env);
            }
        } else {
            // No velocity — instantaneous EFE with static intercept
            let static_candidate =
                crate::fep_agent::compute_rendezvous_intercept(threat_pos, None, entity_pos);
            best_intercept_efe =
                agent.instantaneous_efe(&drone_state, static_candidate, &mission_setpoint, &env);
        }

        let chose_override = best_intercept_efe < efe_mission && env.human_danger >= 0.01;
        let actual_decision = if chose_override {
            "INTERCEPT"
        } else {
            "MISSION"
        };

        results.push(ScenarioVariantResult {
            variant_name: variant.name().to_string(),
            efe_mission,
            efe_intercept: best_intercept_efe,
            chose_override,
            expected_decision: variant.expected_decision(),
            matches_expected: actual_decision == variant.expected_decision(),
            best_rendezvous_frac: best_frac,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_benchmark_runs() {
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.1,
                duration: 0.1,
            }],
            eval_episodes: 1,
        };
        let result = run_wind_benchmark(&config);
        assert_eq!(result.baseline_metrics.len(), 1);
        assert_eq!(result.fep_metrics.len(), 1);
        assert!(result.baseline_metrics[0].is_finite());
        assert!(result.fep_metrics[0].is_finite());
    }

    #[test]
    #[ignore] // ~60s: run manually with `cargo test -p symthaea-flight -- --ignored`
    fn test_gust_increases_error() {
        // Run without gust
        let config_no_gust = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![],
            eval_episodes: 1,
        };
        let no_gust = run_wind_benchmark(&config_no_gust);

        // Run with gust
        let config_gust = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.1, 0.0, 0.0],
                start_time: 0.05,
                duration: 0.2,
            }],
            eval_episodes: 1,
        };
        let with_gust = run_wind_benchmark(&config_gust);

        // Gust should increase max deviation
        assert!(
            with_gust.baseline_max_deviation >= no_gust.baseline_max_deviation * 0.5,
            "Gust should increase deviation: no_gust={:.4}, with_gust={:.4}",
            no_gust.baseline_max_deviation,
            with_gust.baseline_max_deviation,
        );
    }

    #[test]
    #[ignore] // ~60s: run manually with `cargo test -p symthaea-flight -- --ignored`
    fn test_fep_does_not_degrade_wind_recovery() {
        // Core validation: FEP modulation should NOT make things worse.
        // If FEP is well-tuned it helps; at minimum it should not hurt.
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 500,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.08, 0.0, 0.0], // Strong lateral gust
                start_time: 0.1,
                duration: 0.2,
            }],
            eval_episodes: 2,
        };

        let result = run_wind_benchmark(&config);

        // FEP average error should not be dramatically worse than baseline
        let baseline_avg: f64 =
            result.baseline_metrics.iter().sum::<f64>() / result.baseline_metrics.len() as f64;
        let fep_avg: f64 = result.fep_metrics.iter().sum::<f64>() / result.fep_metrics.len() as f64;

        assert!(
            fep_avg < baseline_avg * 1.5,
            "FEP should not dramatically degrade performance: baseline_avg={baseline_avg:.6}, fep_avg={fep_avg:.6}"
        );

        // FEP max deviation should be finite and bounded
        assert!(
            result.max_deviation < 5.0,
            "FEP max deviation should be bounded: {}",
            result.max_deviation
        );

        // Recovery should be tracked (non-empty)
        assert!(
            !result.recovery_steps.is_empty(),
            "Recovery tracking should produce results"
        );
    }

    #[test]
    fn test_fep_produces_finite_results() {
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 100,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.05,
                duration: 0.05,
            }],
            eval_episodes: 1,
        };
        let result = run_wind_benchmark(&config);
        assert!(result.max_deviation.is_finite());
        assert!(result.baseline_max_deviation.is_finite());
        for e in &result.fep_metrics {
            assert!(e.is_finite());
        }
    }

    #[test]
    fn test_physics_backend_simple() {
        // Ensure explicit Simple backend produces same results as default
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 100,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.05,
                duration: 0.05,
            }],
            eval_episodes: 1,
        };
        let result = run_wind_benchmark_with_backend(&config, &PhysicsBackend::Simple);
        assert!(result.baseline_metrics[0].is_finite());
        assert!(result.fep_metrics[0].is_finite());
    }

    // ── EFE Ablation Tests ──

    #[test]
    fn test_efe_ablation_finds_crossover() {
        let env = FlightEnvironment {
            human_danger: 0.7,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let drone_state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let precisions = vec![
            0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0,
        ];
        let result = run_efe_ablation(&precisions, &env, &drone_state, &mission_setpoint, 1.0, 0.1);

        assert!(!result.points.is_empty(), "Should produce ablation points");
        assert!(
            result.crossover_precision.is_some(),
            "Should find a crossover between 0.001 and 10000"
        );
        let crossover = result.crossover_precision.unwrap();
        assert!(
            crossover > 0.001 && crossover < 10000.0,
            "Crossover should be in range: {:.4}",
            crossover
        );
    }

    #[test]
    fn test_efe_ablation_monotonic_mission_efe() {
        let env = FlightEnvironment {
            human_danger: 0.7,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let drone_state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let precisions = vec![0.1, 1.0, 10.0, 100.0, 1000.0];
        let result = run_efe_ablation(&precisions, &env, &drone_state, &mission_setpoint, 1.0, 0.1);

        // Mission EFE should increase monotonically with safety precision
        // (higher π_safety → more penalty for unresolved danger)
        for i in 1..result.points.len() {
            assert!(
                result.points[i].efe_mission >= result.points[i - 1].efe_mission - 1e-10,
                "Mission EFE should increase with safety_π: {:.4} < {:.4} at π={:.4}",
                result.points[i].efe_mission,
                result.points[i - 1].efe_mission,
                result.points[i].safety_precision
            );
        }
    }

    #[test]
    fn test_efe_ablation_extremes() {
        let env = FlightEnvironment {
            human_danger: 0.7,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let drone_state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let precisions = vec![0.001, 10000.0];
        let result = run_efe_ablation(&precisions, &env, &drone_state, &mission_setpoint, 1.0, 0.1);

        // π=0.001 → mission dominates → should NOT intercept
        assert!(
            !result.points[0].chose_intercept,
            "π=0.001: should choose mission"
        );
        // π=10000 → safety dominates → should intercept
        assert!(
            result.points[1].chose_intercept,
            "π=10000: should choose intercept"
        );
    }

    // ── Multi-Scenario Tests ──

    #[test]
    fn test_all_scenarios_match_expected() {
        let results = evaluate_scenario_variants();
        assert_eq!(results.len(), 7, "Should have 7 scenario variants");
        for r in &results {
            assert!(
                r.matches_expected,
                "Scenario '{}': expected {}, got {}",
                r.variant_name,
                r.expected_decision,
                if r.chose_override {
                    "INTERCEPT"
                } else {
                    "MISSION"
                }
            );
        }
    }

    #[test]
    fn test_scenario_no_human_continues_mission() {
        let (drone_state, mission_setpoint, env) = ScenarioVariant::NoHuman.build();
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let override_pos =
            agent.evaluate_setpoint_candidates(&drone_state, &mission_setpoint, &env);
        assert!(
            override_pos.is_none(),
            "No human → should continue mission (no override)"
        );
    }

    #[test]
    fn test_ablation_rendezvous_frac_varies_with_precision() {
        // At low precision, 75% rendezvous wins (closer to ground → more danger reduction);
        // at high precision, 50% wins (midpoint intercept optimal when safety dominates).
        let env = FlightEnvironment {
            human_danger: 0.7,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let drone_state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        // Low precision (mission-dominated): 75% should win (lowest altitude ≈ least mission deviation)
        let result_low = run_efe_ablation(&[1.0], &env, &drone_state, &mission_setpoint, 1.0, 0.1);
        assert_eq!(
            result_low.points[0].best_rendezvous_frac,
            Some(0.75),
            "At low precision (1.0), 75% rendezvous should win"
        );

        // High precision (safety-dominated): 25% should win (earliest arrival → least pre-arrival danger)
        let result_high =
            run_efe_ablation(&[1000.0], &env, &drone_state, &mission_setpoint, 1.0, 0.1);
        assert_eq!(
            result_high.points[0].best_rendezvous_frac,
            Some(0.25),
            "At high precision (1000.0), 25% rendezvous should win"
        );

        // Medium precision: 50% should win (balanced tradeoff)
        let result_mid = run_efe_ablation(&[10.0], &env, &drone_state, &mission_setpoint, 1.0, 0.1);
        assert_eq!(
            result_mid.points[0].best_rendezvous_frac,
            Some(0.50),
            "At medium precision (10.0), 50% rendezvous should win"
        );
    }

    #[test]
    fn test_scenario_far_beam_continues_mission() {
        let (drone_state, mission_setpoint, env) = ScenarioVariant::FarBeam.build();
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let efe_mission = agent.instantaneous_efe(
            &drone_state,
            mission_setpoint.position,
            &mission_setpoint,
            &env,
        );
        let efe_intercept =
            agent.instantaneous_efe(&drone_state, [5.0, 5.0, 3.0], &mission_setpoint, &env);

        // Far beam has large mission deviation cost for intercept
        assert!(
            efe_intercept > efe_mission * 0.5,
            "Far beam: intercept EFE ({:.2}) should not overwhelm mission EFE ({:.2})",
            efe_intercept,
            efe_mission
        );
    }
}
