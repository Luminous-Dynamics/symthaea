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
        let setpoint = FlightSetpoint::hover();
        let dt = self.config.motor_dt();
        let cognitive_interval = self.config.cognitive_interval();

        // Curriculum: perturbation grows across episodes
        let perturbation = if self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            0.01 + 0.09 * progress
        } else {
            0.01
        };
        physics.reset_with_perturbation(0.1, perturbation, episode as u64 + 42);

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
                let target = pd_baseline(&state, &setpoint, &self.pd_gains);
                let lr = self.config.learning_rate * fep_result.learning_rate_factor;
                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
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

        let n = self.config.steps_per_episode as f64;
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
            total_steps: self.config.steps_per_episode,
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
                ep, metrics.avg_position_error, metrics.avg_attitude_error,
                metrics.avg_free_energy, metrics.hover_fraction,
                metrics.final_position_error, metrics.exploration_count,
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
        let dir = "/tmp/symthaea_flight_test_csv";
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
        assert_eq!(metrics.total_steps, 50);
        assert!(metrics.avg_position_error.is_finite());
    }
}
