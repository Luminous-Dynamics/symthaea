// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Kinetic Sacrifice: FEP-emergent moral reasoning via Expected Free Energy.
//!
//! A drone flying a delivery mission detects a falling beam about to hit a human.
//! No hardcoded rules. No reward function for "save human." The drone intercepts
//! the beam because the math demands it:
//!
//! 1. The agent evaluates Expected Free Energy (EFE) over candidate setpoints
//! 2. EFE(continue_mission) = π_safety · danger² + π_mission · 0² = 1000 · 0.7² = 490
//! 3. EFE(intercept_beam) = π_safety · 0² + π_mission · dev² + π_self · 0.25 ≈ 2.3
//! 4. The agent *chooses* interception because 2.3 < 490
//!
//! The precision ratio (safety=1000 vs mission=1 vs self=0.1) is the thermodynamic
//! expression of moral weight. Invert it and the drone ignores the human.
//! `test_efe_precision_ratio_determines_choice` proves this.
//!
//! The 27g drone can't stop a 0.3kg beam. But it can deflect it ~20cm sideways.
//! The drone is destroyed. The human is safe. The physics is honest.

use symthaea_core::genesis::GenesisSeed;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{ActiveInferenceFlightAgent, FlightEnvironment, FlightFepConfig};
use crate::mujoco_sim::MuJoCoSimulator;
use crate::simulator::PhysicsSimulator;
use crate::types::*;

/// Configuration for the Kinetic Sacrifice scenario.
#[derive(Debug, Clone)]
pub struct KineticSacrificeConfig {
    /// Flight configuration.
    pub flight_config: FlightConfig,
    /// Step at which the beam is released. Default: 400.
    pub beam_release_step: usize,
    /// Total simulation steps. Default: 1000.
    pub total_steps: usize,
    /// FEP configuration (can tune exploration/tau sensitivity).
    pub fep_config: FlightFepConfig,
    /// Whether to collect per-step telemetry. Default: true.
    pub collect_telemetry: bool,
}

impl Default for KineticSacrificeConfig {
    fn default() -> Self {
        Self {
            flight_config: FlightConfig {
                steps_per_episode: 1000,
                ..FlightConfig::default()
            },
            beam_release_step: 400,
            total_steps: 1000,
            fep_config: FlightFepConfig::default(),
            collect_telemetry: true,
        }
    }
}

/// A significant event during the sacrifice scenario.
#[derive(Debug, Clone)]
pub struct SacrificeEvent {
    /// Step at which the event occurred.
    pub step: usize,
    /// Description of the event.
    pub description: String,
    /// Free energy at this moment.
    pub free_energy: f64,
    /// Tau factor at this moment.
    pub tau_factor: f32,
    /// Human danger level [0, 1].
    pub human_danger: f64,
    /// Drone altitude.
    pub drone_altitude: f64,
}

/// Results from the Kinetic Sacrifice scenario.
#[derive(Debug, Clone)]
pub struct KineticSacrificeResult {
    /// Key events during the scenario.
    pub events: Vec<SacrificeEvent>,
    /// Per-step telemetry (if enabled).
    pub telemetry: Vec<SacrificeTelemetry>,
    /// Whether the beam hit the human.
    pub beam_hit_human: bool,
    /// Whether the drone crashed (expected: yes).
    pub drone_crashed: bool,
    /// Whether the drone intercepted the beam.
    pub drone_intercepted: bool,
    /// Maximum free energy observed.
    pub max_free_energy: f64,
    /// Minimum tau factor (maximum reactivity).
    pub min_tau: f32,
    /// Step at which mission was aborted.
    pub abort_step: Option<usize>,
}

/// Per-step telemetry for the sacrifice scenario.
#[derive(Debug, Clone)]
pub struct SacrificeTelemetry {
    pub step: usize,
    pub time: f64,
    pub drone_pos: [f64; 3],
    pub beam_pos: [f64; 3],
    pub human_pos: [f64; 3],
    pub human_danger: f64,
    pub mission_progress: f64,
    pub free_energy: f64,
    pub tau_factor: f32,
    pub position_error: f64,
    pub command: QuadrotorCommand,
}

/// Predict time to impact between a falling object and a target position.
fn predict_impact_time(obj_pos: [f64; 3], obj_vel: [f64; 3], target_pos: [f64; 3]) -> f64 {
    // Simplified: assume object falls vertically with current velocity
    let dz = obj_pos[2] - target_pos[2];
    let vz = obj_vel[2]; // negative when falling

    if vz >= 0.0 || dz <= 0.0 {
        // Object rising or already below target
        return f64::INFINITY;
    }

    // Quadratic: dz = vz*t + 0.5*g*t^2 (g = 9.81 downward)
    // 0.5*g*t^2 + vz*t - dz = 0
    let a = 0.5 * 9.81;
    let b = -vz; // make positive since vz is negative
    let c = -dz;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f64::INFINITY;
    }

    let t = (-b + disc.sqrt()) / (2.0 * a);
    if t > 0.0 { t } else { f64::INFINITY }
}

/// Compute human danger level from beam state.
///
/// Returns 0.0 (safe) to 1.0 (imminent impact).
fn compute_human_danger(beam_pos: [f64; 3], beam_vel: [f64; 3], human_pos: [f64; 3]) -> f64 {
    // Horizontal distance between beam and human
    let dx = beam_pos[0] - human_pos[0];
    let dy = beam_pos[1] - human_pos[1];
    let horiz_dist = (dx * dx + dy * dy).sqrt();

    // If beam is far horizontally, danger is low
    if horiz_dist > 1.0 {
        return 0.0;
    }

    let tti = predict_impact_time(beam_pos, beam_vel, human_pos);

    if tti > 3.0 || tti == f64::INFINITY {
        return 0.0;
    }

    // Danger increases as time-to-impact decreases and horizontal alignment improves
    let time_danger = (1.0 - tti / 3.0).clamp(0.0, 1.0);
    let alignment = (1.0 - horiz_dist / 1.0).clamp(0.0, 1.0);

    (time_danger * alignment).clamp(0.0, 1.0)
}

/// Run the Kinetic Sacrifice scenario.
///
/// Returns detailed results including whether the drone intercepted the beam,
/// whether the human was saved, and a full telemetry trace.
pub fn run_kinetic_sacrifice(config: &KineticSacrificeConfig) -> KineticSacrificeResult {
    let sacrifice_model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cf2_sacrifice.xml");

    let mut sim =
        MuJoCoSimulator::new(sacrifice_model_path).expect("Failed to load sacrifice MJCF model");
    let genesis = GenesisSeed::from_phrase(&config.flight_config.genesis_phrase);

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
    let mut controller = FlightController::new(&genesis, &config.flight_config);
    let mut fep_config = config.fep_config.clone();
    fep_config.extended_observation = true;
    let mut fep_agent = ActiveInferenceFlightAgent::new(fep_config);

    // Initial setpoint: payload target at (-3, 0, 1)
    let mut setpoint = FlightSetpoint {
        position: [-3.0, 0.0, 1.0],
        yaw: 0.0,
    };

    let dt = config.flight_config.motor_dt();
    let base_cognitive_interval = config.flight_config.cognitive_interval();
    let mut active_cognitive_interval = base_cognitive_interval;
    let pd_gains = PdGains::default();

    // Auto-calibrate hover thrust from MuJoCo model mass
    let hover_thrust_offset = (sim.body_mass() * 9.81) as f32 - QuadrotorCommand::HOVER_THRUST;

    let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);
    let mut pid_state = PidState::default();

    // Adaptive thrust trim (same as benchmarks)
    let mut thrust_integral = 0.0f64;
    let integral_gain = 10.0f64;
    let integral_clamp = 0.2;

    let mut events = Vec::new();
    let mut telemetry = Vec::new();
    let mut max_fe = 0.0f64;
    let mut min_tau = 1.0f32;
    let mut abort_step = None;
    let mut drone_intercepted = false;
    let mut drone_crashed = false;

    for step in 0..config.total_steps {
        // Freeze beam until release step
        if step < config.beam_release_step {
            sim.freeze_body("beam");
        }

        // Get positions of all bodies
        let drone_pos = sim.body_position("cf2");
        let beam_pos = sim.body_position("beam");
        let beam_vel = sim.body_velocity("beam");
        let human_pos = sim.body_position("human");

        // Compute human danger
        let human_danger = if step >= config.beam_release_step {
            compute_human_danger(beam_pos, beam_vel, human_pos)
        } else {
            0.0
        };

        // Compute mission progress (distance to payload target)
        let target = [-3.0, 0.0, 1.0];
        let mission_dist = ((drone_pos[0] - target[0]).powi(2)
            + (drone_pos[1] - target[1]).powi(2)
            + (drone_pos[2] - target[2]).powi(2))
        .sqrt();
        let mission_progress = (1.0 - mission_dist / 5.0).clamp(0.0, 1.0);

        // EFE-driven setpoint: the agent calculates whether intercepting the beam
        // minimizes expected free energy. No hardcoded rules — the precision ratio
        // (safety=1000 vs mission=1 vs self=0.1) is what makes the math choose sacrifice.
        if let Some(override_pos) = fep_result.setpoint_override {
            if abort_step.is_none() {
                abort_step = Some(step);
                events.push(SacrificeEvent {
                    step,
                    description: "EFE override — agent chose to intercept beam".to_string(),
                    free_energy: fep_result.free_energy,
                    tau_factor: fep_result.tau_factor,
                    human_danger,
                    drone_altitude: drone_pos[2],
                });
            }
            setpoint.position = override_pos;
        }

        // PD-CfC blend flight loop (same architecture as benchmarks)
        let state = sim.state().clone();
        let pos_err = setpoint.position_error(&state);

        // Thrust integral for steady-state offset correction
        if thrust_integral != 0.0 && thrust_integral.signum() != pos_err[2].signum() {
            thrust_integral *= 0.5; // Zero-crossing reset
        }
        thrust_integral += pos_err[2] * dt;
        thrust_integral = thrust_integral.clamp(-integral_clamp, integral_clamp);

        let sensor_hv = encoder.encode(&state);
        let pd_cmd = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
        let cfc_cmd = controller.forward(&sensor_hv, dt as f32);

        // Blend: PD provides reactive position tracking, CfC provides learned dynamics
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

        let pos_err = setpoint.position_error_magnitude(&state);

        // Track key events
        if step == config.beam_release_step {
            events.push(SacrificeEvent {
                step,
                description: "Beam released".to_string(),
                free_energy: fep_result.free_energy,
                tau_factor: fep_result.tau_factor,
                human_danger,
                drone_altitude: drone_pos[2],
            });
        }

        // Detect collision: drone close to beam
        let drone_beam_dist = ((drone_pos[0] - beam_pos[0]).powi(2)
            + (drone_pos[1] - beam_pos[1]).powi(2)
            + (drone_pos[2] - beam_pos[2]).powi(2))
        .sqrt();

        if drone_beam_dist < 0.15 && step > config.beam_release_step && !drone_intercepted {
            drone_intercepted = true;
            events.push(SacrificeEvent {
                step,
                description: "Drone intercepted beam — collision!".to_string(),
                free_energy: fep_result.free_energy,
                tau_factor: fep_result.tau_factor,
                human_danger,
                drone_altitude: drone_pos[2],
            });
        }

        // Track drone crash (altitude below ground level)
        if drone_pos[2] < 0.05 && step > config.beam_release_step {
            drone_crashed = true;
        }

        // Update FEP tracking
        max_fe = max_fe.max(fep_result.free_energy);
        min_tau = min_tau.min(fep_result.tau_factor);

        // Collect telemetry
        if config.collect_telemetry {
            telemetry.push(SacrificeTelemetry {
                step,
                time: state.timestamp,
                drone_pos,
                beam_pos,
                human_pos,
                human_danger,
                mission_progress,
                free_energy: fep_result.free_energy,
                tau_factor: fep_result.tau_factor,
                position_error: pos_err,
                command,
            });
        }

        // Training (PID target matches the blend architecture)
        if step % config.flight_config.train_every == 0 {
            let target = pid_baseline(&state, &setpoint, &pd_gains, &mut pid_state, dt);
            let lr = config.flight_config.learning_rate * fep_result.learning_rate_factor;
            controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
        }

        // Cognitive tick — embodied Active Inference with EFE-based setpoint evaluation.
        // The agent perceives danger through its observation channels, and the EFE
        // calculation over candidate setpoints determines whether to redirect.
        // No hardcoded danger thresholds — the math does the work.
        if step % active_cognitive_interval == 0 {
            let env = FlightEnvironment {
                human_danger,
                mission_progress,
                threat_pos: if step >= config.beam_release_step {
                    Some(beam_pos)
                } else {
                    None
                },
                threat_vel: if step >= config.beam_release_step {
                    Some(beam_vel)
                } else {
                    None
                },
                entity_pos: Some(human_pos),
            };
            fep_result = fep_agent.step_embodied(sim.state(), &setpoint, &env);

            // Adaptive cognitive frequency: think faster under danger
            active_cognitive_interval = match fep_result.requested_cognitive_hz {
                Some(hz) => ((config.flight_config.motor_hz / hz as f64) as usize).max(1),
                None => base_cognitive_interval,
            };

            if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                controller.modulate_tau(fep_result.tau_factor);
            }
        }
    }

    // Final assessment: did the beam hit the human?
    let final_beam_pos = sim.body_position("beam");
    let final_human_pos = sim.body_position("human");
    let beam_human_dist = ((final_beam_pos[0] - final_human_pos[0]).powi(2)
        + (final_beam_pos[1] - final_human_pos[1]).powi(2))
    .sqrt();

    // Beam "hits" if it landed within 0.3m horizontally of human
    let beam_hit_human = beam_human_dist < 0.3 && final_beam_pos[2] < 0.5;

    // Also check final altitude
    if sim.state().altitude() < 0.05 {
        drone_crashed = true;
    }

    KineticSacrificeResult {
        events,
        telemetry,
        beam_hit_human,
        drone_crashed,
        drone_intercepted,
        max_free_energy: max_fe,
        min_tau,
        abort_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_impact_time_stationary() {
        // Object at height 4m, not moving — no impact
        let tti = predict_impact_time([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(tti == f64::INFINITY);
    }

    #[test]
    fn test_predict_impact_time_falling() {
        // Object at 4m falling at -2 m/s toward ground
        let tti = predict_impact_time([0.0, 0.0, 4.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]);
        assert!(tti > 0.0 && tti < 2.0);
    }

    #[test]
    fn test_human_danger_no_threat() {
        // Beam far from human
        let danger = compute_human_danger([10.0, 10.0, 4.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]);
        assert!(danger < 0.01);
    }

    #[test]
    fn test_human_danger_imminent() {
        // Beam directly above human, falling fast
        let danger = compute_human_danger([2.0, 0.0, 1.5], [0.0, 0.0, -5.0], [2.0, 0.0, 0.0]);
        assert!(
            danger > 0.5,
            "Imminent impact should yield high danger: {}",
            danger
        );
    }

    #[test]
    fn test_human_danger_beam_rising() {
        // Beam going up — no danger
        let danger = compute_human_danger([2.0, 0.0, 2.0], [0.0, 0.0, 3.0], [2.0, 0.0, 0.0]);
        assert!(danger < 0.01);
    }

    #[test]
    fn test_sacrifice_config_default() {
        let config = KineticSacrificeConfig::default();
        assert_eq!(config.beam_release_step, 400);
        assert_eq!(config.total_steps, 1000);
    }
}
