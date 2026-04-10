// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parallel swarm training via rayon (behind `swarm` feature).
//!
//! 128 parallel MuJoCo instances via rayon, each with randomized wind/mass/geometry.
//! All drones feed into a single thread-safe global experience buffer. Each drone
//! also learns from the swarm's collective experience.
//!
//! Key design: MuJoCo's `MjModel` is read-only after loading and can be shared.
//! But `MjData` is per-simulation and NOT thread-safe. Each rayon task creates
//! its own `MuJoCoSimulator` (which owns both model and data).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use rayon::prelude::*;

use symthaea_core::genesis::GenesisSeed;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{ActiveInferenceFlightAgent, FlightFepConfig};
use crate::mujoco_sim::MuJoCoSimulator;
use crate::simulator::PhysicsSimulator;
use crate::training::EpisodeMetrics;
use crate::types::*;

/// Thread-safe experience buffer shared across swarm members.
///
/// Ring buffer of (sensor state, command target) pairs. Each drone writes
/// its experiences; all drones can sample from the collective pool.
pub struct SwarmExperienceBuffer {
    /// Stored as flattened state channels + command values.
    buffer: Mutex<Vec<([f32; 13], QuadrotorCommand)>>,
    max_size: usize,
    /// Ring buffer write position. AtomicUsize eliminates the second lock
    /// and removes any deadlock possibility from the double-lock pattern.
    write_pos: AtomicUsize,
}

impl SwarmExperienceBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
            write_pos: AtomicUsize::new(0),
        }
    }

    /// Store an experience (thread-safe).
    pub fn store(&self, state_channels: [f32; 13], command: QuadrotorCommand) {
        let mut buf = self.buffer.lock();
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed);

        if buf.len() < self.max_size {
            buf.push((state_channels, command));
        } else {
            buf[pos % self.max_size] = (state_channels, command);
        }
    }

    /// Sample N random experiences from the buffer (thread-safe).
    /// Uses a simple LCG for deterministic sampling.
    pub fn sample(&self, count: usize, seed: u64) -> Vec<([f32; 13], QuadrotorCommand)> {
        let buf = self.buffer.lock();
        let len = buf.len();
        if len == 0 || count == 0 {
            return Vec::new();
        }

        let mut rng = seed;
        let mut samples = Vec::with_capacity(count.min(len));

        for _ in 0..count.min(len) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng >> 33) as usize % len;
            samples.push(buf[idx].clone());
        }

        samples
    }

    /// Current buffer utilization.
    pub fn len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().is_empty()
    }

    /// Buffer capacity.
    pub fn capacity(&self) -> usize {
        self.max_size
    }
}

/// Configuration for swarm training.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Number of parallel MuJoCo instances (default: num_cpus or 128).
    pub num_drones: usize,
    /// Steps per drone per training episode.
    pub steps_per_drone: usize,
    /// Episodes per drone.
    pub episodes: usize,
    /// Experience buffer capacity.
    pub buffer_size: usize,
    /// How many swarm experiences to replay per local training step.
    pub swarm_replay_count: usize,
    /// Mass randomization range (kg). Default: (0.020, 0.035).
    pub mass_range: (f64, f64),
    /// Wind magnitude range (N). Default: (0.0, 0.15).
    pub wind_magnitude_range: (f64, f64),
    /// Base flight configuration.
    pub flight_config: FlightConfig,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            num_drones: num_cpus(),
            steps_per_drone: 2000,
            episodes: 10,
            buffer_size: 128 * 1024, // 128K entries
            swarm_replay_count: 4,
            mass_range: (0.020, 0.035),
            wind_magnitude_range: (0.0, 0.15),
            flight_config: FlightConfig::default(),
        }
    }
}

/// Aggregate results from swarm training.
#[derive(Debug, Clone)]
pub struct SwarmResult {
    /// Per-drone, per-episode metrics (flattened: drone_id * episodes + ep).
    pub metrics: Vec<EpisodeMetrics>,
    /// Total experiences collected.
    pub total_experiences: usize,
    /// Mean final error across all drones.
    pub mean_final_error: f64,
    /// Best final error among all drones.
    pub best_final_error: f64,
    /// Buffer utilization percentage.
    pub buffer_utilization: f64,
}

/// Run swarm training across N parallel MuJoCo instances.
///
/// Each drone gets its own MuJoCo instance (no shared mutable state in physics),
/// randomized mass/wind, and writes experiences to a shared buffer. Each drone
/// also replays from the swarm's collective experience pool.
pub fn train_swarm(config: &SwarmConfig) -> SwarmResult {
    let shared_buffer = Arc::new(SwarmExperienceBuffer::new(config.buffer_size));

    let all_metrics: Vec<Vec<EpisodeMetrics>> = (0..config.num_drones)
        .into_par_iter()
        .map(|drone_id| {
            // Each drone gets its own MuJoCo instance
            let mut sim = MuJoCoSimulator::from_primitive()
                .expect("Failed to load MuJoCo primitive model for swarm drone");

            // Deterministic RNG per drone
            let mut rng_state = (drone_id as u64).wrapping_mul(2654435761).wrapping_add(42);
            let mut next_f64 = || -> f64 {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                rng_state as f64 / u64::MAX as f64
            };

            // Randomize this drone's physics
            let mass_range = config.mass_range.1 - config.mass_range.0;
            let mass = config.mass_range.0 + next_f64() * mass_range;
            sim.set_body_mass(mass);

            // Randomize wind profile
            let wind_range = config.wind_magnitude_range.1 - config.wind_magnitude_range.0;
            let wind_mag = config.wind_magnitude_range.0 + next_f64() * wind_range;
            let wx = (next_f64() * 2.0 - 1.0) * wind_mag;
            let wy = (next_f64() * 2.0 - 1.0) * wind_mag;
            let wind_force = [wx, wy, 0.0];

            // Each drone has its own controller (same genesis → same initial weights)
            let genesis = GenesisSeed::from_phrase(&config.flight_config.genesis_phrase);
            let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
            let mut controller = FlightController::new(&genesis, &config.flight_config);
            let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

            let local_buffer = Arc::clone(&shared_buffer);
            let setpoint = FlightSetpoint::hover();
            let pd_gains = PdGains::default();
            let dt = config.flight_config.motor_dt();
            let cognitive_interval = config.flight_config.cognitive_interval();

            let mut drone_metrics = Vec::with_capacity(config.episodes);

            for ep in 0..config.episodes {
                sim.reset_with_perturbation(0.1, 0.01, (drone_id * 1000 + ep) as u64);
                encoder.reset();
                controller.reset();
                fep_agent.reset();

                let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);
                let mut total_pos_err = 0.0;
                let mut total_fe = 0.0;
                let mut fe_samples = 0;
                let mut hover_steps = 0;
                let mut exploration_count = 0;

                for step in 0..config.steps_per_drone {
                    let state = sim.state().clone();
                    let sensor_hv = encoder.encode(&state);
                    let mut command = controller.forward(&sensor_hv, dt as f32);

                    if let Some(noise) = &fep_result.exploration_noise {
                        command = command.with_noise(noise);
                        exploration_count += 1;
                    }

                    // Apply wind
                    sim.apply_external_force(wind_force);
                    sim.step(&command, dt);

                    let pos_err = setpoint.position_error_magnitude(&state);
                    total_pos_err += pos_err;

                    if (state.altitude() - setpoint.position[2]).abs() < 0.05 {
                        hover_steps += 1;
                    }

                    // Training
                    if step % config.flight_config.train_every == 0 {
                        let target = pd_baseline(&state, &setpoint, &pd_gains);
                        let lr =
                            config.flight_config.learning_rate * fep_result.learning_rate_factor;
                        controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));

                        // Store experience in shared buffer
                        local_buffer.store(state.to_channels(), target);

                        // Replay from swarm buffer
                        let replay_seed = (drone_id * 7919 + ep * 31 + step) as u64;
                        let swarm_experiences =
                            local_buffer.sample(config.swarm_replay_count, replay_seed);
                        for (channels, replay_target) in &swarm_experiences {
                            let replay_state = FlightState::from_channels(*channels);
                            let replay_hv = encoder.encode_stateless(&replay_state);
                            controller.train_step(
                                &replay_hv,
                                replay_target,
                                dt as f32,
                                Some(lr * 0.5),
                            );
                        }
                    }

                    // Cognitive tick
                    if step % cognitive_interval == 0 {
                        fep_result = fep_agent.step(sim.state(), &setpoint);
                        if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                            controller.modulate_tau(fep_result.tau_factor);
                        }
                        total_fe += fep_result.free_energy;
                        fe_samples += 1;
                    }
                }

                let n = config.steps_per_drone as f64;
                let final_state = sim.state().clone();

                drone_metrics.push(EpisodeMetrics {
                    episode: ep,
                    avg_position_error: total_pos_err / n,
                    avg_attitude_error: 0.0, // Simplified for swarm
                    avg_free_energy: if fe_samples > 0 {
                        total_fe / fe_samples as f64
                    } else {
                        0.0
                    },
                    hover_fraction: hover_steps as f64 / n,
                    final_position_error: setpoint.position_error_magnitude(&final_state),
                    exploration_count,
                    total_steps: config.steps_per_drone,
                    telemetry: Vec::new(), // No per-step telemetry in swarm mode
                });
            }

            drone_metrics
        })
        .collect();

    // Aggregate results
    let flat_metrics: Vec<EpisodeMetrics> = all_metrics.into_iter().flatten().collect();

    let total_experiences = shared_buffer.len();
    let buffer_utilization = total_experiences as f64 / config.buffer_size as f64;

    // Compute aggregate stats from final episodes
    let final_errors: Vec<f64> = flat_metrics
        .iter()
        .filter(|m| m.episode == config.episodes - 1)
        .map(|m| m.final_position_error)
        .collect();

    let mean_final_error = if final_errors.is_empty() {
        0.0
    } else {
        final_errors.iter().sum::<f64>() / final_errors.len() as f64
    };

    let best_final_error = final_errors.iter().cloned().fold(f64::INFINITY, f64::min);

    SwarmResult {
        metrics: flat_metrics,
        total_experiences,
        mean_final_error,
        best_final_error,
        buffer_utilization: buffer_utilization.min(1.0),
    }
}

/// Get number of available CPUs (capped at 128 for swarm).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(128))
        .unwrap_or(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_buffer_store_and_sample() {
        let buf = SwarmExperienceBuffer::new(100);
        let channels = [0.0f32; 13];
        let cmd = QuadrotorCommand::hover();

        for _ in 0..50 {
            buf.store(channels, cmd);
        }

        assert_eq!(buf.len(), 50);
        let samples = buf.sample(5, 42);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_experience_buffer_ring() {
        let buf = SwarmExperienceBuffer::new(10);
        let cmd = QuadrotorCommand::hover();

        for i in 0..20 {
            let mut ch = [0.0f32; 13];
            ch[0] = i as f32;
            buf.store(ch, cmd);
        }

        // Should cap at max_size
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_experience_buffer_empty_sample() {
        let buf = SwarmExperienceBuffer::new(100);
        let samples = buf.sample(5, 42);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_swarm_config_default() {
        let config = SwarmConfig::default();
        assert!(config.num_drones > 0);
        assert!(config.buffer_size > 0);
        assert!(config.mass_range.0 < config.mass_range.1);
    }

    #[test]
    fn test_deterministic_sampling() {
        let buf = SwarmExperienceBuffer::new(100);
        let cmd = QuadrotorCommand::hover();

        for i in 0..20 {
            let mut ch = [0.0f32; 13];
            ch[0] = i as f32;
            buf.store(ch, cmd);
        }

        let s1 = buf.sample(5, 12345);
        let s2 = buf.sample(5, 12345);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.0, b.0, "Same seed must produce same samples");
        }
    }

    #[test]
    fn test_replay_produces_state_specific_encodings() {
        // Verify the fixed replay path: different states should produce
        // different encodings, meaning the controller gets state-specific
        // training signal (not just output targets against current hidden state).
        use crate::encoder::QuadrotorHdcEncoder;
        use crate::types::FlightState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("swarm-replay-test");
        let encoder = QuadrotorHdcEncoder::new(&genesis, 32);

        let state_a = FlightState::hover(0.1);
        let mut state_b_channels = state_a.to_channels();
        state_b_channels[0] = 5.0; // Very different x position
        let state_b = FlightState::from_channels(state_b_channels);

        let hv_a = encoder.encode_stateless(&state_a);
        let hv_b = encoder.encode_stateless(&state_b);

        // Different states must produce different encodings
        let sim = hv_a.similarity(&hv_b);
        assert!(
            sim < 0.95,
            "Different states should produce distinct encodings: sim={sim:.4}"
        );
    }

    // Note: train_swarm() requires MuJoCo library — tested via examples only
}
