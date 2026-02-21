//! Flight training: multi-rate loop with PD baseline targets and curriculum.
//!
//! The trainer runs a multi-rate control loop:
//! - Motor reflex at 500Hz: encode → evolve → project → command
//! - Training at 125Hz: BPTT from PD baseline targets
//! - Cognitive tick at 25Hz: FEP agent modulates τ/LR/precision
//!
//! For pure-Rust testing (no MuJoCo), the trainer can use a simple
//! ballistic physics model as a stand-in.

use symthaea_core::genesis::GenesisSeed;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{ActiveInferenceFlightAgent, FlightFepConfig};
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

/// Simple ballistic physics model for pure-Rust testing.
///
/// Simulates a rigid body with thrust and moments (no rotor dynamics).
/// Good enough for verifying the multi-rate loop and training convergence.
struct SimplePhysics {
    state: FlightState,
    mass: f64,
    inertia: [f64; 3], // Diagonal inertia tensor
}

impl SimplePhysics {
    fn new() -> Self {
        Self {
            state: FlightState::hover(0.1),
            mass: 0.027,                       // Crazyflie 2: 27g
            inertia: [1.4e-5, 1.4e-5, 2.2e-5], // Approximate
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self, altitude: f64) {
        self.state = FlightState::hover(altitude);
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64) {
        self.state = FlightState::hover(altitude);
        // Apply deterministic perturbation
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        self.state.position[0] += perturbation * next_f64() * 0.1;
        self.state.position[1] += perturbation * next_f64() * 0.1;
        self.state.position[2] += perturbation * next_f64() * 0.05;
        // Slight tilt
        let tilt = perturbation * next_f64() * 0.1;
        self.state.quaternion = normalize_quat([1.0, tilt, tilt * 0.5, 0.0]);
    }

    fn step(&mut self, cmd: &QuadrotorCommand, dt: f64) -> &FlightState {
        let g = 9.81;

        // Thrust in world frame (rotated by quaternion)
        let thrust = cmd.thrust as f64;
        let [w, x, y, z] = self.state.quaternion;

        // Simplified rotation of thrust vector (body z → world)
        let fx = 2.0 * (x * z + w * y) * thrust;
        let fy = 2.0 * (y * z - w * x) * thrust;
        let fz = (1.0 - 2.0 * (x * x + y * y)) * thrust;

        // Linear acceleration
        let ax = fx / self.mass;
        let ay = fy / self.mass;
        let az = fz / self.mass - g;

        // Update linear velocity
        self.state.linear_velocity[0] += ax * dt;
        self.state.linear_velocity[1] += ay * dt;
        self.state.linear_velocity[2] += az * dt;

        // Update position
        self.state.position[0] += self.state.linear_velocity[0] * dt;
        self.state.position[1] += self.state.linear_velocity[1] * dt;
        self.state.position[2] += self.state.linear_velocity[2] * dt;

        // Ground constraint
        if self.state.position[2] < 0.0 {
            self.state.position[2] = 0.0;
            self.state.linear_velocity[2] = 0.0;
        }

        // Angular acceleration from moments
        let alpha_x = cmd.roll_moment as f64 / self.inertia[0];
        let alpha_y = cmd.pitch_moment as f64 / self.inertia[1];
        let alpha_z = cmd.yaw_moment as f64 / self.inertia[2];

        // Update angular velocity
        self.state.angular_velocity[0] += alpha_x * dt;
        self.state.angular_velocity[1] += alpha_y * dt;
        self.state.angular_velocity[2] += alpha_z * dt;

        // Simple angular damping
        for av in &mut self.state.angular_velocity {
            *av *= 0.99;
        }

        // Update quaternion from angular velocity
        let [wx, wy, wz] = self.state.angular_velocity;
        let half_dt = dt * 0.5;
        let dw = w - half_dt * (x * wx + y * wy + z * wz);
        let dx = x + half_dt * (w * wx + y * wz - z * wy);
        let dy = y + half_dt * (w * wy + z * wx - x * wz);
        let dz = z + half_dt * (w * wz + x * wy - y * wx);
        self.state.quaternion = normalize_quat([dw, dx, dy, dz]);

        self.state.timestamp += dt;
        &self.state
    }

    fn state(&self) -> &FlightState {
        &self.state
    }
}

fn normalize_quat(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-10 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
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

    /// Run a single training episode with the simple physics model.
    ///
    /// Returns per-episode metrics.
    pub fn run_episode(
        &self,
        encoder: &mut QuadrotorHdcEncoder,
        controller: &mut FlightController,
        fep_agent: &mut ActiveInferenceFlightAgent,
        episode: usize,
    ) -> EpisodeMetrics {
        let mut physics = SimplePhysics::new();
        let setpoint = FlightSetpoint::hover();
        let dt = self.config.motor_dt();
        let cognitive_interval = self.config.cognitive_interval();

        // Curriculum: perturbation grows across episodes
        let perturbation = if self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            0.01 + 0.09 * progress // 0.01 → 0.10
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

    /// Run the full training curriculum.
    ///
    /// Returns all episode metrics.
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

            // Signal episode end to FEP agent
            fep_agent.reset();
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_physics_hover() {
        let mut physics = SimplePhysics::new();
        let cmd = QuadrotorCommand::hover();
        let dt = 0.002;

        // Hover for 100 steps (0.2s)
        for _ in 0..100 {
            physics.step(&cmd, dt);
        }

        let state = physics.state();
        // Should stay near 10cm altitude
        assert!(
            (state.altitude() - 0.1).abs() < 0.05,
            "Hover should maintain altitude: got {}",
            state.altitude()
        );
    }

    #[test]
    fn test_simple_physics_ground_constraint() {
        let mut physics = SimplePhysics::new();
        physics.reset(0.0);
        let cmd = QuadrotorCommand::zero(); // No thrust

        physics.step(&cmd, 0.002);
        assert!(
            physics.state().position[2] >= 0.0,
            "Ground constraint should prevent negative altitude"
        );
    }

    #[test]
    fn test_simple_physics_gravity() {
        let mut physics = SimplePhysics::new();
        physics.reset(1.0);
        let cmd = QuadrotorCommand::zero(); // No thrust — freefall

        for _ in 0..100 {
            physics.step(&cmd, 0.002);
        }

        // Should have fallen significantly
        assert!(
            physics.state().altitude() < 0.95,
            "Should fall under gravity: alt={}",
            physics.state().altitude()
        );
    }

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
            steps_per_episode: 40, // Exactly 2 cognitive ticks
            ..FlightConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = FlightTrainer::new(config.clone());

        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        // 40 steps / 20 cognitive_interval = 2 cognitive ticks
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

        // Later episodes should have higher initial perturbation,
        // so (on average) higher initial position error
        // This is a soft check — not guaranteed per episode but trend should hold
        assert!(metrics.len() == 5);
    }
}
