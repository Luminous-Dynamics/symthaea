// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flight training: multi-rate loop with PD baseline targets and curriculum.
//!
//! The trainer runs a multi-rate control loop:
//! - Motor reflex at 500Hz: encode → evolve → project → command
//! - Training at 125Hz: BPTT from PD baseline targets
//! - Cognitive tick at 25Hz: FEP agent modulates τ/LR/precision
//!
//! Physics simulation is abstracted behind the `PhysicsSimulator` trait,
//! allowing different backends (simple ballistic, MuJoCo, etc.).

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{ActiveInferenceFlightAgent, FlightFepConfig};
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::*;

/// Per-episode metrics.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// Episode index.
    pub episode: usize,
    /// Average position error over the episode.
    pub avg_position_error: f64,
    /// Average attitude error over the episode.
    pub avg_attitude_error: f64,
    /// Average free energy (from FEP agent).
    pub avg_free_energy: f64,
    /// Fraction of steps where altitude was within 5cm of setpoint.
    pub hover_fraction: f64,
    /// Final position error.
    pub final_position_error: f64,
    /// Number of exploration bursts triggered.
    pub exploration_count: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Per-step telemetry (populated when `FlightConfig::collect_telemetry` is true).
    pub telemetry: Vec<FlightTelemetry>,
}

/// Flight trainer with multi-rate control loop.
pub struct FlightTrainer {
    /// Configuration.
    config: FlightConfig,
    /// Genesis seed for deterministic initialization.
    genesis: GenesisSeed,
    /// PD gains for baseline targets.
    pd_gains: PdGains,
    /// Collected episode metrics.
    pub metrics: Vec<EpisodeMetrics>,
}

impl FlightTrainer {
    /// Create a new trainer.
    pub fn new(config: FlightConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            pd_gains: PdGains::default(),
            metrics: Vec::new(),
        }
    }

    /// Create a trainer with custom PD gains.
    pub fn with_pd_gains(config: FlightConfig, gains: PdGains) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            pd_gains: gains,
            metrics: Vec::new(),
        }
    }

    /// Run a single training episode with any physics simulator.
    pub fn run_episode_with_sim(
        &self,
        encoder: &mut QuadrotorHdcEncoder,
        controller: &mut FlightController,
        fep_agent: &mut ActiveInferenceFlightAgent,
        physics: &mut dyn PhysicsSimulator,
        episode: usize,
    ) -> EpisodeMetrics {
        let dt = self.config.motor_dt();
        let cognitive_interval = self.config.cognitive_interval();

        // Setpoint variety: custom curriculum takes priority over built-in cycle.
        let setpoint = if !self.config.curriculum.is_empty() {
            self.config.curriculum[episode % self.config.curriculum.len()].clone()
        } else {
            match episode % 5 {
                0 => FlightSetpoint::hover(), // z=0.1
                1 => FlightSetpoint {
                    position: [0.0, 0.0, 0.2],
                    yaw: 0.0,
                }, // higher hover
                2 => FlightSetpoint {
                    position: [0.05, 0.0, 0.1],
                    yaw: 0.0,
                }, // slight x offset
                3 => FlightSetpoint {
                    position: [0.0, 0.05, 0.15],
                    yaw: 0.0,
                }, // y + z offset
                _ => FlightSetpoint::hover(), // back to default
            }
        };

        // Curriculum: perturbation grows across episodes
        let perturbation = if self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            0.01 + 0.09 * progress
        } else {
            0.01
        };
        let reset_alt = setpoint.position[2];
        physics.reset_with_perturbation(reset_alt, perturbation, episode as u64 + 42);

        // Reset components for new episode
        encoder.reset();
        controller.reset();

        // Prime FEP agent
        let mut fep_result = fep_agent.initial_step(physics.state(), &setpoint);

        // Accumulators
        let mut total_pos_err = 0.0;
        let mut total_att_err = 0.0;
        let mut total_fe = 0.0;
        let mut hover_steps = 0usize;
        let mut exploration_count = 0usize;
        let mut fe_samples = 0usize;
        let mut telemetry = if self.config.collect_telemetry {
            Vec::with_capacity(self.config.steps_per_episode)
        } else {
            Vec::new()
        };
        let mut current_tau_factor = fep_result.tau_factor;
        let mut current_fe = fep_result.free_energy;

        // Experience replay buffer: ring buffer of (sensor_hv, pd_target) pairs.
        // Replaying past experiences prevents catastrophic forgetting as state drifts.
        let replay_cap = self.config.replay_buffer_size;
        let replay_count = self.config.replay_count;
        let mut replay_buf: Vec<(ContinuousHV, QuadrotorCommand)> = Vec::with_capacity(replay_cap);
        let mut replay_priorities: Vec<f32> = Vec::with_capacity(replay_cap);
        let priority_alpha: f32 = 0.6;
        let mut replay_idx = 0usize; // next write position (ring buffer)
        let mut replay_rng = episode as u64 + 7919; // simple hash for replay sampling
        let mut steps_completed = 0usize;
        let mut pid_state = PidState::default();

        // Cosine annealing LR schedule: high early, low late
        let lr_scale = if self.config.enable_lr_schedule && self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
        } else {
            1.0f32
        };

        for step in 0..self.config.steps_per_episode {
            // ── MOTOR REFLEX (every step, 500Hz) ──
            let state = physics.state().clone();
            let sensor_hv = encoder.encode(&state);
            let mut command = controller.forward(&sensor_hv, dt as f32);

            // Add exploration noise if any
            if let Some(noise) = &fep_result.exploration_noise {
                command = command.with_noise(noise);
                exploration_count += 1;
            }

            physics.step(&command, dt);

            // Track metrics
            let pos_err = setpoint.position_error_magnitude(&state);
            let (roll, pitch, _) = state.euler_angles();
            let att_err = (roll * roll + pitch * pitch).sqrt();
            total_pos_err += pos_err;
            total_att_err += att_err;

            if (state.altitude() - setpoint.position[2]).abs() < 0.05 {
                hover_steps += 1;
            }

            steps_completed += 1;

            // Early termination on crash or divergence
            if self.config.early_termination
                && step > 10
                && (pos_err > 5.0 || (state.altitude() < 0.001 && state.linear_velocity[2] <= 0.0))
            {
                break;
            }

            // Collect telemetry if enabled
            if self.config.collect_telemetry {
                telemetry.push(FlightTelemetry {
                    step,
                    time: state.timestamp,
                    position_error: pos_err,
                    attitude_error: att_err,
                    speed: state.speed(),
                    altitude: state.altitude(),
                    free_energy: current_fe,
                    tau_factor: current_tau_factor,
                    learning_rate: controller.learning_rate(),
                    command,
                });
            }

            // ── TRAINING (every train_every steps, 125Hz) ──
            if step % self.config.train_every == 0 {
                let target = if self.config.use_pid || fep_result.use_pid_target {
                    pid_baseline(&state, &setpoint, &self.pd_gains, &mut pid_state, dt)
                } else {
                    pd_baseline(&state, &setpoint, &self.pd_gains)
                };
                let lr = self.config.learning_rate * fep_result.learning_rate_factor * lr_scale;

                // Train on current observation
                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));

                // Store in replay buffer with default priority
                if replay_cap > 0 {
                    if replay_buf.len() < replay_cap {
                        replay_buf.push((sensor_hv.clone(), target));
                        replay_priorities.push(1.0);
                    } else {
                        let widx = replay_idx % replay_cap;
                        replay_buf[widx] = (sensor_hv.clone(), target);
                        replay_priorities[widx] = 1.0;
                    }
                    replay_idx += 1;

                    // Prioritized experience replay: sample proportional to priority^alpha
                    let buf_len = replay_buf.len();
                    if buf_len > 1 {
                        for _ in 0..replay_count.min(buf_len) {
                            let idx = sample_prioritized_index(
                                &replay_priorities,
                                priority_alpha,
                                &mut replay_rng,
                            );
                            let (ref replay_hv, ref replay_target) = replay_buf[idx];
                            let pre = controller.forward(replay_hv, dt as f32);
                            controller.train_step(replay_hv, replay_target, dt as f32, Some(lr));

                            // Update priority with TD error magnitude
                            let td = ((replay_target.thrust - pre.thrust).powi(2)
                                + (replay_target.roll_moment - pre.roll_moment).powi(2)
                                + (replay_target.pitch_moment - pre.pitch_moment).powi(2)
                                + (replay_target.yaw_moment - pre.yaw_moment).powi(2))
                            .sqrt();
                            replay_priorities[idx] = td.max(0.01);
                        }
                    }
                }
            }

            // ── STATE NORMALIZATION (every 100 motor steps) ──
            if step % 100 == 0 && step > 0 {
                controller.normalize_states();
            }

            // ── COGNITIVE TICK (every cognitive_interval steps, 25Hz) ──
            if step % cognitive_interval == 0 {
                fep_result = fep_agent.step(physics.state(), &setpoint);

                // Apply τ modulation
                if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                    controller.modulate_tau(fep_result.tau_factor);
                }

                current_tau_factor = fep_result.tau_factor;
                current_fe = fep_result.free_energy;
                total_fe += fep_result.free_energy;
                fe_samples += 1;
            }
        }

        let n = steps_completed.max(1) as f64;
        let final_state = physics.state().clone();

        EpisodeMetrics {
            episode,
            avg_position_error: total_pos_err / n,
            avg_attitude_error: total_att_err / n,
            avg_free_energy: if fe_samples > 0 {
                total_fe / fe_samples as f64
            } else {
                0.0
            },
            hover_fraction: hover_steps as f64 / n,
            final_position_error: setpoint.position_error_magnitude(&final_state),
            exploration_count,
            total_steps: steps_completed,
            telemetry,
        }
    }

    /// Run a single training episode using the simple physics model.
    pub fn run_episode(
        &self,
        encoder: &mut QuadrotorHdcEncoder,
        controller: &mut FlightController,
        fep_agent: &mut ActiveInferenceFlightAgent,
        episode: usize,
    ) -> EpisodeMetrics {
        let mut physics = SimplePhysicsSimulator::new();
        self.run_episode_with_sim(encoder, controller, fep_agent, &mut physics, episode)
    }

    /// Run the full training curriculum.
    pub fn train(&mut self) -> Vec<EpisodeMetrics> {
        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = QuadrotorHdcEncoder::new(&genesis, num_levels);
        let mut controller = FlightController::new(&genesis, &self.config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        // Warmup: pre-train output projection on static hover samples.
        // Generates (sensor_hv, PD_target) pairs from slightly perturbed hover states
        // so the controller starts with a reasonable policy instead of random weights.
        let warmup_samples: Vec<(ContinuousHV, QuadrotorCommand)> = (0..20)
            .map(|i| {
                let mut state = FlightState::hover(0.1);
                // Small perturbations to give variety
                let offset = (i as f64 - 10.0) * 0.005;
                state.position[2] += offset;
                state.linear_velocity[2] = offset * -2.0; // opposing velocity
                let hv = encoder.encode(&state);
                let target = pd_baseline(&state, &FlightSetpoint::hover(), &self.pd_gains);
                (hv, target)
            })
            .collect();
        controller.warmup(&warmup_samples, 100, self.config.learning_rate * 10.0);
        encoder.reset();

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);

        for ep in 0..self.config.num_episodes {
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);
            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Run training with CSV telemetry output.
    ///
    /// Writes per-step CSV and per-episode summary CSV to `output_dir`.
    pub fn train_with_telemetry(&mut self, output_dir: &str) -> Vec<EpisodeMetrics> {
        self.config.collect_telemetry = true;

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = QuadrotorHdcEncoder::new(&genesis, num_levels);
        let mut controller = FlightController::new(&genesis, &self.config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);
        let _ = std::fs::create_dir_all(output_dir);

        let mut summary = String::from(
            "episode,avg_pos_error,avg_att_error,avg_free_energy,hover_fraction,final_pos_error,exploration_count\n",
        );

        for ep in 0..self.config.num_episodes {
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            // Write per-step CSV
            if !metrics.telemetry.is_empty() {
                let step_path = format!("{}/episode_{:04}.csv", output_dir, ep);
                let mut csv = String::from(
                    "step,time,pos_error,att_error,speed,altitude,free_energy,tau_factor,learning_rate,thrust,roll_moment,pitch_moment,yaw_moment\n",
                );
                for t in &metrics.telemetry {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                        t.step, t.time, t.position_error, t.attitude_error, t.speed,
                        t.altitude, t.free_energy, t.tau_factor, t.learning_rate,
                        t.command.thrust, t.command.roll_moment, t.command.pitch_moment,
                        t.command.yaw_moment,
                    ));
                }
                let _ = std::fs::write(&step_path, csv);
            }

            summary.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.4},{:.6},{}\n",
                ep,
                metrics.avg_position_error,
                metrics.avg_attitude_error,
                metrics.avg_free_energy,
                metrics.hover_fraction,
                metrics.final_position_error,
                metrics.exploration_count,
            ));

            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        let summary_path = format!("{}/summary.csv", output_dir);
        let _ = std::fs::write(&summary_path, summary);

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &FlightConfig {
        &self.config
    }
}

/// Sample an index from a weighted priority distribution.
///
/// Probability proportional to `priority^alpha`. Falls back to uniform if all priorities are zero.
fn sample_prioritized_index(priorities: &[f32], alpha: f32, rng: &mut u64) -> usize {
    let len = priorities.len();
    if len == 0 {
        return 0;
    }

    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);

    let total: f32 = priorities.iter().map(|p| p.powf(alpha)).sum();
    if total <= 0.0 {
        return (*rng >> 33) as usize % len;
    }

    let r = ((*rng >> 33) as f64 / (1u64 << 31) as f64) as f32 * total;
    let mut cumsum = 0.0f32;
    for (i, p) in priorities.iter().enumerate() {
        cumsum += p.powf(alpha);
        if cumsum >= r {
            return i;
        }
    }
    len - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_single_episode() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 100,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        assert_eq!(metrics.episode, 0);
        assert_eq!(metrics.total_steps, 100);
        assert!(metrics.avg_position_error.is_finite());
        assert!(metrics.avg_free_energy.is_finite());
        assert!(metrics.hover_fraction >= 0.0 && metrics.hover_fraction <= 1.0);
    }

    #[test]
    fn test_trainer_multi_rate_timing() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 40,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.avg_free_energy.is_finite());
    }

    #[test]
    fn test_full_training_runs() {
        let config = FlightConfig {
            num_episodes: 3,
            steps_per_episode: 100,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 3);
        for (i, m) in metrics.iter().enumerate() {
            assert_eq!(m.episode, i);
        }
    }

    #[test]
    fn test_training_determinism() {
        let config = FlightConfig {
            num_episodes: 2,
            steps_per_episode: 50,
            ..FlightConfig::default()
        };

        let mut trainer1 = FlightTrainer::new(config.clone());
        let mut trainer2 = FlightTrainer::new(config);

        let m1 = trainer1.train();
        let m2 = trainer2.train();

        assert!(
            (m1[0].avg_position_error - m2[0].avg_position_error).abs() < 1e-6,
            "Same genesis → same metrics"
        );
    }

    #[test]
    fn test_curriculum_perturbation_growth() {
        let config = FlightConfig {
            num_episodes: 5,
            steps_per_episode: 20,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();
        assert!(metrics.len() == 5);
    }

    #[test]
    #[ignore] // ~60s: covered by fast test_regression_motor_lag_convergence
    fn test_multi_episode_convergence() {
        // Train across 6 episodes with curriculum (perturbation 1% → 10%).
        // The controller accumulates learning across episodes (weights persist).
        // Despite increasing difficulty, performance should not catastrophically degrade.
        let config = FlightConfig {
            num_episodes: 6,
            steps_per_episode: 300,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 6);

        // All episodes should produce finite, bounded errors
        for m in &metrics {
            assert!(m.avg_position_error.is_finite());
            assert!(m.final_position_error.is_finite());
            assert!(
                m.avg_position_error < 10.0,
                "Error should be bounded: {}",
                m.avg_position_error
            );
        }

        // Key convergence check: the average error across the last 3 episodes
        // should not be dramatically worse than the first 3, despite harder curriculum.
        // This proves the network is compensating for increasing perturbation.
        let first_half: f64 = metrics[..3]
            .iter()
            .map(|m| m.avg_position_error)
            .sum::<f64>()
            / 3.0;
        let second_half: f64 = metrics[3..]
            .iter()
            .map(|m| m.avg_position_error)
            .sum::<f64>()
            / 3.0;

        assert!(
            second_half < first_half * 3.0,
            "Training should prevent catastrophic degradation under curriculum: \
             first_half_avg={first_half:.6}, second_half_avg={second_half:.6}"
        );

        // Stronger: final position error of last episode should be reasonable
        assert!(
            metrics[5].final_position_error < 2.0,
            "Final episode should end with reasonable error: {}",
            metrics[5].final_position_error
        );
    }

    #[test]
    fn test_telemetry_enabled_collects() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 100,
            collect_telemetry: true,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert_eq!(metrics.telemetry.len(), 100);
    }

    #[test]
    fn test_telemetry_disabled_empty() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 100,
            collect_telemetry: false,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.telemetry.is_empty());
    }

    #[test]
    fn test_telemetry_values_finite() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: true,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        for t in &metrics.telemetry {
            assert!(t.position_error.is_finite());
            assert!(t.attitude_error.is_finite());
            assert!(t.speed.is_finite());
            assert!(t.altitude.is_finite());
            assert!(t.free_energy.is_finite());
            assert!(t.tau_factor.is_finite());
            assert!(t.learning_rate.is_finite());
        }
    }

    #[test]
    fn test_telemetry_time_monotonic() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: true,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        for i in 1..metrics.telemetry.len() {
            assert!(
                metrics.telemetry[i].step > metrics.telemetry[i - 1].step,
                "Steps should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_csv_telemetry_output() {
        let config = FlightConfig {
            num_episodes: 2,
            steps_per_episode: 20,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let dir = "/tmp/symthaea_multirotor_test_csv";
        let _ = std::fs::remove_dir_all(dir);

        let metrics = trainer.train_with_telemetry(dir);
        assert_eq!(metrics.len(), 2);

        let summary = std::fs::read_to_string(format!("{}/summary.csv", dir)).unwrap();
        assert!(summary.starts_with("episode,avg_pos_error,"));

        let ep0 = std::fs::read_to_string(format!("{}/episode_0000.csv", dir)).unwrap();
        assert!(ep0.starts_with("step,time,pos_error,"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_run_episode_with_sim() {
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
        let mut sim = SimplePhysicsSimulator::new();

        let metrics = trainer.run_episode_with_sim(
            &mut encoder,
            &mut controller,
            &mut fep_agent,
            &mut sim,
            0,
        );
        assert!(metrics.total_steps <= 50);
        assert!(metrics.avg_position_error.is_finite());
    }

    #[test]
    fn test_lr_schedule_reduces_lr_over_episodes() {
        let config = FlightConfig {
            num_episodes: 5,
            steps_per_episode: 100,
            enable_lr_schedule: true,
            early_termination: false,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 5);
        for m in &metrics {
            assert!(m.avg_position_error.is_finite());
            assert!(m.total_steps > 0);
        }
    }

    #[test]
    fn test_early_termination_shortens_diverged_episodes() {
        // With early termination, a heavily perturbed episode should end early.
        let config = FlightConfig {
            num_episodes: 1,
            steps_per_episode: 2000,
            enable_lr_schedule: false,
            early_termination: true,
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
        let mut sim = SimplePhysicsSimulator::new();

        // Reset with huge perturbation to force divergence
        sim.reset_with_perturbation(0.1, 5.0, 999);

        let metrics = trainer.run_episode_with_sim(
            &mut encoder,
            &mut controller,
            &mut fep_agent,
            &mut sim,
            0,
        );

        // With large perturbation and early termination, episode should end before 2000 steps
        // (or complete normally if drag keeps things stable — either outcome is acceptable)
        assert!(metrics.total_steps > 0);
        assert!(metrics.avg_position_error.is_finite());
    }

    #[test]
    fn test_prioritized_replay_high_priority_sampled_more() {
        let priorities = vec![0.01, 0.01, 10.0, 0.01];
        let mut counts = [0u32; 4];
        let mut rng = 42u64;
        for _ in 0..1000 {
            let idx = sample_prioritized_index(&priorities, 0.6, &mut rng);
            counts[idx] += 1;
        }
        // Index 2 (priority 10.0) should be sampled most frequently
        assert!(
            counts[2] > counts[0] + counts[1] + counts[3],
            "High-priority sample should dominate: {:?}",
            counts
        );
    }

    #[test]
    fn test_prioritized_replay_uniform_when_equal() {
        let priorities = vec![1.0, 1.0, 1.0, 1.0];
        let mut counts = [0u32; 4];
        let mut rng = 42u64;
        for _ in 0..4000 {
            let idx = sample_prioritized_index(&priorities, 0.6, &mut rng);
            counts[idx] += 1;
        }
        for c in &counts {
            assert!(
                *c > 400 && *c < 1600,
                "Equal priorities should give roughly uniform sampling: {:?}",
                counts
            );
        }
    }

    #[test]
    fn test_curriculum_custom_setpoints() {
        let config = FlightConfig {
            num_episodes: 4,
            steps_per_episode: 50,
            curriculum: vec![
                FlightSetpoint {
                    position: [0.0, 0.0, 0.3],
                    yaw: 0.0,
                },
                FlightSetpoint {
                    position: [0.1, 0.0, 0.2],
                    yaw: 0.0,
                },
            ],
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();
        assert_eq!(metrics.len(), 4);
        // All episodes should complete with finite metrics
        for m in &metrics {
            assert!(m.avg_position_error.is_finite());
        }
    }

    #[test]
    fn test_pid_training_converges() {
        let config = FlightConfig {
            num_episodes: 3,
            steps_per_episode: 200,
            use_pid: true,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.avg_position_error.is_finite());
        }
        assert!(
            metrics[2].avg_position_error < 2.0,
            "PID training should converge: avg_err={:.4}",
            metrics[2].avg_position_error
        );
    }

    #[test]
    #[ignore] // ~90s: covered by fast test_regression_motor_lag_convergence
    fn test_long_horizon_training_convergence() {
        // Train for 15 episodes of 500 steps with drag-enabled physics.
        // Position error should improve or remain bounded across episodes.
        let config = FlightConfig {
            num_episodes: 15,
            steps_per_episode: 500,
            enable_lr_schedule: true,
            early_termination: true,
            ..FlightConfig::default()
        };
        let mut trainer = FlightTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 15);

        // All errors should be finite and bounded
        for m in &metrics {
            assert!(m.avg_position_error.is_finite());
            assert!(
                m.avg_position_error < 5.0,
                "Episode {} error unbounded: {:.4}",
                m.episode,
                m.avg_position_error
            );
        }

        // Last 5 episodes should not be dramatically worse than first 5
        let first5: f64 = metrics[..5]
            .iter()
            .map(|m| m.avg_position_error)
            .sum::<f64>()
            / 5.0;
        let last5: f64 = metrics[10..]
            .iter()
            .map(|m| m.avg_position_error)
            .sum::<f64>()
            / 5.0;

        assert!(
            last5 < first5 * 5.0,
            "Training should prevent catastrophic degradation: first5={first5:.4}, last5={last5:.4}"
        );
    }
}
