// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid training: multi-rate loop with PD baseline targets and adaptive curriculum.
//!
//! The trainer runs a multi-rate control loop:
//! - Physics at 40Hz: encode -> evolve -> project -> command
//! - Training at 20Hz: BPTT from PD baseline targets
//! - Cognitive tick at 10Hz: FEP agent modulates tau/LR/precision
//!
//! Adaptive curriculum (advances early on mastery, falls back to fixed schedule):
//! - Phase 0 (max 25%): Stand, PD 80%→40% — mastery: standing_reward > threshold
//! - Phase 1 (max 15%): Stand, PD 40%→10% — mastery: reward > 0.90, uprightness > 0.95
//! - Phase 2 (max 50%): Walk, PD 10%→5%, speed 0→1 m/s — mastery: episode_reward > 0.70
//! - Phase 3 (remainder): Run, PD 5%→0%, speed 1→3 m/s

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
use crate::gait::GaitAnalyzer;
use crate::reward;
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::types::*;

/// Per-episode metrics.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// Episode index.
    pub episode: usize,
    /// Average standing reward over the episode.
    pub avg_standing_reward: f64,
    /// Average episode reward (task-specific).
    pub avg_episode_reward: f64,
    /// Average free energy (from FEP agent).
    pub avg_free_energy: f64,
    /// Average head height.
    pub avg_head_height: f64,
    /// Average uprightness.
    pub avg_uprightness: f64,
    /// Average horizontal speed.
    pub avg_horizontal_speed: f64,
    /// Average control effort.
    pub avg_control_effort: f64,
    /// Number of exploration bursts triggered.
    pub exploration_count: usize,
    /// Total steps completed (may be < steps_per_episode if early termination).
    pub total_steps: usize,
    /// Task for this episode.
    pub task: HumanoidTask,
    /// Average maximum foot clearance during swing phases (meters).
    pub avg_foot_clearance: f64,
    /// Minimum maximum foot clearance across all strides (meters).
    pub min_foot_clearance: f64,
    /// Average stride length in meters.
    pub avg_stride_length: f64,
    /// Average cadence in steps per second.
    pub avg_cadence: f64,
    /// Gait asymmetry (0 = symmetric).
    pub gait_asymmetry: f64,
    /// Cost of Transport: energy / (mass × distance). Normalized proxy, not SI.
    pub cost_of_transport: f64,
    /// Step regularity: exp(-CV) of step intervals (1.0 = perfectly regular).
    pub step_regularity: f64,
    /// Foot strike quality: proper heel-strike dorsiflexion + toe-off plantarflexion.
    pub foot_strike_quality: f64,
    /// Per-step telemetry (populated when collect_telemetry is true).
    pub telemetry: Vec<HumanoidTelemetry>,
}

/// Tracks adaptive curriculum state for performance-based phase transitions.
#[derive(Debug, Clone)]
struct CurriculumState {
    /// Current phase: 0=Stand PD↓, 1=Stand autonomy, 2=Walk, 3=Run.
    phase: usize,
    /// Episode where the current phase began.
    phase_start_ep: usize,
    /// Consecutive episodes meeting mastery criteria.
    mastery_streak: usize,
}

impl CurriculumState {
    fn new() -> Self {
        Self {
            phase: 0,
            phase_start_ep: 0,
            mastery_streak: 0,
        }
    }
}

/// Humanoid trainer with multi-rate control loop and curriculum.
pub struct HumanoidTrainer {
    /// Configuration.
    config: HumanoidConfig,
    /// Genesis seed.
    genesis: GenesisSeed,
    /// PD gains.
    pd_gains: HumanoidPdGains,
    /// Collected episode metrics.
    pub metrics: Vec<EpisodeMetrics>,
    /// Optional pre-initialized controller (e.g., from morphological transfer).
    initial_controller: Option<HumanoidController>,
    /// Adaptive curriculum state.
    curriculum_state: CurriculumState,
}

impl HumanoidTrainer {
    /// Create a new trainer.
    pub fn new(config: HumanoidConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            pd_gains: HumanoidPdGains::default(),
            metrics: Vec::new(),
            initial_controller: None,
            curriculum_state: CurriculumState::new(),
        }
    }

    /// Create a trainer with a pre-initialized controller (e.g., from morphological transfer).
    pub fn with_controller(config: HumanoidConfig, controller: HumanoidController) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            pd_gains: HumanoidPdGains::default(),
            metrics: Vec::new(),
            initial_controller: Some(controller),
            curriculum_state: CurriculumState::new(),
        }
    }

    /// Create a trainer that resumes from a saved checkpoint.
    ///
    /// Loads the controller's output projection from a JSON checkpoint file,
    /// reconstructing the network backbone from the stored genesis phrase.
    /// Training continues from the checkpoint's learned state.
    pub fn with_checkpoint(config: HumanoidConfig, checkpoint_path: &str) -> std::io::Result<Self> {
        let controller = HumanoidController::load_checkpoint(checkpoint_path)?;
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Ok(Self {
            config,
            genesis,
            pd_gains: HumanoidPdGains::default(),
            metrics: Vec::new(),
            initial_controller: Some(controller),
            curriculum_state: CurriculumState::new(),
        })
    }

    /// Create a trainer with custom PD gains.
    pub fn with_pd_gains(config: HumanoidConfig, gains: HumanoidPdGains) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            pd_gains: gains,
            metrics: Vec::new(),
            initial_controller: None,
            curriculum_state: CurriculumState::new(),
        }
    }

    /// Determine the task and PD teacher weight for a given episode.
    ///
    /// When `adaptive_curriculum` is enabled, uses performance-based phase
    /// transitions (early advancement on mastery). Otherwise falls back to
    /// fixed schedule based on episode progress.
    fn curriculum(&self, episode: usize) -> (HumanoidTask, f32, f64) {
        let total = self.config.num_episodes;
        if total <= 1 {
            return (self.config.task, 0.8, self.config.effective_target_speed());
        }

        if self.config.adaptive_curriculum {
            self.curriculum_adaptive(episode)
        } else {
            self.curriculum_fixed(episode)
        }
    }

    /// Fixed curriculum: phase boundaries at 25%/40%/90%/100%.
    fn curriculum_fixed(&self, episode: usize) -> (HumanoidTask, f32, f64) {
        let total = self.config.num_episodes;
        let progress = episode as f64 / (total - 1) as f64;

        if progress < 0.25 {
            let phase_progress = progress / 0.25;
            let pd_weight = 0.8 - 0.4 * phase_progress as f32;
            (HumanoidTask::Stand, pd_weight, 0.0)
        } else if progress < 0.40 {
            let phase_progress = (progress - 0.25) / 0.15;
            let pd_weight = 0.4 - 0.3 * phase_progress as f32;
            (HumanoidTask::Stand, pd_weight, 0.0)
        } else if progress < 0.90 {
            let phase_progress = (progress - 0.40) / 0.50;
            let pd_weight = 0.10 - 0.05 * phase_progress as f32;
            let target_speed = phase_progress;
            (HumanoidTask::Walk, pd_weight, target_speed)
        } else {
            let phase_progress = (progress - 0.90) / 0.10;
            let pd_weight = 0.05 * (1.0 - phase_progress as f32);
            let target_speed = 1.0 + phase_progress * 2.0;
            (HumanoidTask::Run, pd_weight, target_speed)
        }
    }

    /// Adaptive curriculum: uses CurriculumState for phase-aware progress.
    ///
    /// Each phase has a max duration (same as fixed boundaries). Progress
    /// within phase = episodes_in_phase / max_duration_for_phase, clamped to [0,1].
    /// Phase transitions happen via `check_phase_advance()` after each episode.
    fn curriculum_adaptive(&self, episode: usize) -> (HumanoidTask, f32, f64) {
        let total = self.config.num_episodes;
        let cs = &self.curriculum_state;

        // Max durations per phase (in episodes)
        let max_durations = [
            (total as f64 * 0.25).ceil() as usize,   // Phase 0: Stand PD↓
            (total as f64 * 0.15).ceil() as usize,   // Phase 1: Stand autonomy
            (total as f64 * 0.50).ceil() as usize,   // Phase 2: Walk
            total.saturating_sub(cs.phase_start_ep), // Phase 3: Run (remainder)
        ];

        let episodes_in_phase = episode.saturating_sub(cs.phase_start_ep);
        let max_dur = max_durations[cs.phase.min(3)].max(1);
        let phase_progress = (episodes_in_phase as f64 / max_dur as f64).min(1.0);

        match cs.phase {
            0 => {
                let pd_weight = 0.8 - 0.4 * phase_progress as f32;
                (HumanoidTask::Stand, pd_weight, 0.0)
            }
            1 => {
                let pd_weight = 0.4 - 0.3 * phase_progress as f32;
                (HumanoidTask::Stand, pd_weight, 0.0)
            }
            2 => {
                let pd_weight = 0.10 - 0.05 * phase_progress as f32;
                let target_speed = phase_progress;
                (HumanoidTask::Walk, pd_weight, target_speed)
            }
            _ => {
                let pd_weight = 0.05 * (1.0 - phase_progress as f32);
                let target_speed = 1.0 + phase_progress * 2.0;
                (HumanoidTask::Run, pd_weight, target_speed)
            }
        }
    }

    /// Check whether the current phase should advance based on episode metrics.
    ///
    /// Advances when: (a) max duration exceeded, or (b) mastery criteria met
    /// for `mastery_streak_required` consecutive episodes after minimum duration.
    /// Returns true if phase advanced.
    fn check_phase_advance(&mut self, ep: usize, metrics: &EpisodeMetrics) -> bool {
        if !self.config.adaptive_curriculum || self.curriculum_state.phase >= 3 {
            return false;
        }

        let total = self.config.num_episodes;
        let min_durations = [10usize, 5, 15, total];
        let max_durations = [
            (total as f64 * 0.25).ceil() as usize,
            (total as f64 * 0.15).ceil() as usize,
            (total as f64 * 0.50).ceil() as usize,
            total,
        ];

        let phase = self.curriculum_state.phase;
        let episodes_in_phase = ep.saturating_sub(self.curriculum_state.phase_start_ep) + 1;

        // Force advance if max duration exceeded
        if episodes_in_phase >= max_durations[phase] {
            self.curriculum_state.phase += 1;
            self.curriculum_state.phase_start_ep = ep + 1;
            self.curriculum_state.mastery_streak = 0;
            return true;
        }

        // Check mastery only if min duration met
        if episodes_in_phase < min_durations[phase] {
            return false;
        }

        let full_episode = metrics.total_steps == self.config.steps_per_episode;
        let mastered = match phase {
            0 => {
                metrics.avg_standing_reward > self.config.standing_mastery_threshold && full_episode
            }
            1 => {
                metrics.avg_standing_reward > 0.90 && metrics.avg_uprightness > 0.95 && full_episode
            }
            2 => {
                let (_, _, target_speed) = self.curriculum_adaptive(ep);
                metrics.avg_episode_reward > 0.70
                    && metrics.avg_horizontal_speed > target_speed * 0.5
            }
            _ => false,
        };

        if mastered {
            self.curriculum_state.mastery_streak += 1;
            if self.curriculum_state.mastery_streak >= self.config.mastery_streak_required {
                self.curriculum_state.phase += 1;
                self.curriculum_state.phase_start_ep = ep + 1;
                self.curriculum_state.mastery_streak = 0;
                return true;
            }
        } else {
            self.curriculum_state.mastery_streak = 0;
        }

        false
    }

    /// Run a single training episode with any physics simulator.
    pub fn run_episode_with_sim(
        &self,
        encoder: &mut HumanoidHdcEncoder,
        controller: &mut HumanoidController,
        fep_agent: &mut ActiveInferenceHumanoidAgent,
        physics: &mut dyn HumanoidPhysicsSimulator,
        episode: usize,
    ) -> EpisodeMetrics {
        let dt = self.config.physics_dt();
        let cognitive_interval = self.config.cognitive_interval();
        let (task, pd_weight, target_speed) = self.curriculum(episode);

        fep_agent.set_task(task);

        // Reset with curriculum perturbation
        let perturbation = if self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            0.01 + 0.09 * progress
        } else {
            0.01
        };
        physics.reset_with_perturbation(perturbation, episode as u64 + 42);

        encoder.reset();
        controller.reset();

        let initial_cmd = HumanoidCommand::zero();
        let mut fep_result = fep_agent.step_with_encoder_pe(physics.state(), &initial_cmd, None);

        // Accumulators
        let mut total_standing_reward = 0.0;
        let mut total_episode_reward = 0.0;
        let mut total_fe = 0.0;
        let mut total_head_height = 0.0;
        let mut total_uprightness = 0.0;
        let mut total_horizontal_speed = 0.0;
        let mut total_control_effort = 0.0;
        let mut exploration_count = 0usize;
        let mut fe_samples = 0usize;
        let mut steps_completed = 0usize;
        let mut current_tau_factor = fep_result.tau_factor;
        let mut current_fe = fep_result.free_energy;
        let mut low_reward_streak = 0u32;

        let mut telemetry = if self.config.collect_telemetry {
            Vec::with_capacity(self.config.steps_per_episode)
        } else {
            Vec::new()
        };

        // Experience replay buffer
        let replay_cap = self.config.replay_buffer_size;
        let replay_count = self.config.replay_count;
        let mut replay_buf: Vec<(ContinuousHV, HumanoidCommand)> = Vec::with_capacity(replay_cap);
        let mut replay_idx = 0usize;
        let mut replay_rng = episode as u64 + 7919;

        // Cosine annealing LR schedule
        let lr_scale = if self.config.enable_lr_schedule && self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
        } else {
            1.0f32
        };

        // Gait quality analyzer (foot clearance + stride tracking)
        let mut gait_analyzer = GaitAnalyzer::new();

        // Horizontal position accumulator (from velocity × dt)
        let mut horizontal_pos = [0.0f64; 2];
        // Mechanical energy accumulator for Cost of Transport
        let mut total_mechanical_energy = 0.0f64;

        // Speed-dependent gait frequency (biomechanical: cadence increases with speed)
        // Walk: 1.2 Hz at 0 m/s → 1.8 Hz at 1 m/s
        // Run: 2.0 Hz at 1 m/s → 2.6 Hz at 3 m/s
        let gait_freq = match task {
            HumanoidTask::Walk => 1.2 + 0.6 * target_speed.min(1.0),
            HumanoidTask::Run => 2.0 + 0.3 * (target_speed - 1.0).clamp(0.0, 2.0),
            HumanoidTask::Stand | HumanoidTask::Reach | HumanoidTask::Grasp => 0.0,
        };

        for step in 0..self.config.steps_per_episode {
            // -- PHYSICS STEP (every step, 40Hz) --
            let state = physics.state().clone();
            let sensor_hv = encoder.encode(&state);
            let mut command = controller.forward(&sensor_hv, dt as f32);

            // Gait phase: fraction of gait cycle at this timestep
            let gait_phase = (step as f64 * dt * gait_freq).fract();

            // Blend with PD teacher during curriculum (task-appropriate baseline)
            if pd_weight > 0.01 {
                let pd_cmd = match task {
                    HumanoidTask::Stand => pd_standing_baseline(&state, &self.pd_gains),
                    HumanoidTask::Walk => {
                        pd_walking_baseline(&state, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Run => {
                        pd_running_baseline(&state, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Reach => pd_reaching_baseline(
                        &state,
                        &self.pd_gains,
                        self.config.object_position,
                        self.config.reach_hand,
                    ),
                    HumanoidTask::Grasp => {
                        let grasp_phase =
                            (step as f64 / self.config.steps_per_episode as f64).min(1.0);
                        pd_grasping_baseline(
                            &state,
                            &self.pd_gains,
                            self.config.object_position,
                            self.config.reach_hand,
                            grasp_phase,
                        )
                    }
                };
                for i in 0..NUM_ACTUATORS {
                    command.torques[i] =
                        pd_weight * pd_cmd.torques[i] + (1.0 - pd_weight) * command.torques[i];
                }
                command = command.clamped();
            }

            // Add exploration noise
            if let Some(noise) = &fep_result.exploration_noise {
                command = command.with_noise(noise);
                exploration_count += 1;
            }

            physics.step(&command, dt);

            // Update position, energy, and gait analyzer with post-step state
            {
                let post_state = physics.state();
                horizontal_pos[0] += post_state.root_linear_velocity[0] * dt;
                horizontal_pos[1] += post_state.root_linear_velocity[1] * dt;
                for i in 0..NUM_ACTUATORS {
                    total_mechanical_energy +=
                        (command.torques[i] as f64 * post_state.joint_velocities[i]).abs() * dt;
                }
                gait_analyzer.update_with_position(
                    post_state,
                    horizontal_pos,
                    post_state.timestamp,
                );
            }

            // Track metrics
            let standing_r = reward::standing_reward(&state);
            let episode_r = reward::episode_reward(&state, &command, &task, target_speed);
            total_standing_reward += standing_r;
            total_episode_reward += episode_r;
            total_head_height += state.head_height;
            total_uprightness += state.uprightness();
            total_horizontal_speed += state.horizontal_speed();
            total_control_effort += command.control_effort() as f64;

            steps_completed += 1;

            // Early termination on fall or sustained degradation
            if self.config.early_termination && step > 10 {
                if state.head_height < 0.5 || state.uprightness() < 0.1 {
                    break;
                }
                // Stop if standing reward stays below 0.3 for 50 consecutive steps
                if standing_r < 0.3 {
                    low_reward_streak += 1;
                    if low_reward_streak >= 50 {
                        break;
                    }
                } else {
                    low_reward_streak = 0;
                }
            }

            // Collect telemetry
            if self.config.collect_telemetry {
                telemetry.push(HumanoidTelemetry {
                    step,
                    time: state.timestamp,
                    head_height: state.head_height,
                    uprightness: state.uprightness(),
                    horizontal_speed: state.horizontal_speed(),
                    standing_reward: standing_r,
                    episode_reward: episode_r,
                    free_energy: current_fe,
                    tau_factor: current_tau_factor,
                    learning_rate: controller.learning_rate(),
                    control_effort: command.control_effort(),
                    r_foot_z: state.extremities[8],
                    l_foot_z: state.extremities[11],
                });
            }

            // -- TRAINING (every train_every steps, 20Hz) --
            if step % self.config.train_every == 0 {
                let target = match task {
                    HumanoidTask::Stand => pd_standing_baseline(&state, &self.pd_gains),
                    HumanoidTask::Walk => {
                        pd_walking_baseline(&state, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Run => {
                        pd_running_baseline(&state, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Reach => pd_reaching_baseline(
                        &state,
                        &self.pd_gains,
                        self.config.object_position,
                        self.config.reach_hand,
                    ),
                    HumanoidTask::Grasp => {
                        let gp = (step as f64 / self.config.steps_per_episode as f64).min(1.0);
                        pd_grasping_baseline(
                            &state,
                            &self.pd_gains,
                            self.config.object_position,
                            self.config.reach_hand,
                            gp,
                        )
                    }
                };
                // Reward-modulated BPTT: scale gradient by standing_reward so the
                // network learns more when upright (clear signal) and less when
                // fallen (noisy signal). Floor at 0.1 to preserve some gradient.
                let reward_mod = (standing_r as f32).max(0.1);
                let lr = controller.learning_rate()
                    * fep_result.learning_rate_factor
                    * lr_scale
                    * reward_mod;

                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));

                // Replay buffer
                if replay_cap > 0 {
                    if replay_buf.len() < replay_cap {
                        replay_buf.push((sensor_hv.clone(), target));
                    } else {
                        replay_buf[replay_idx % replay_cap] = (sensor_hv.clone(), target);
                    }
                    replay_idx += 1;

                    let buf_len = replay_buf.len();
                    if buf_len > 1 {
                        for _ in 0..replay_count.min(buf_len) {
                            replay_rng =
                                replay_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let idx = (replay_rng >> 33) as usize % buf_len;
                            let (ref replay_hv, ref replay_target) = replay_buf[idx];
                            controller.train_step(replay_hv, replay_target, dt as f32, Some(lr));
                        }
                    }
                }
            }

            // -- STATE NORMALIZATION (every 100 steps) --
            if step % 100 == 0 && step > 0 {
                controller.normalize_states();
            }

            // -- COGNITIVE TICK (every cognitive_interval steps, 10Hz) --
            if step % cognitive_interval == 0 {
                // Feed encoder PE if predictive layer is active
                let enc_pe = if encoder.has_predictive_layer() {
                    Some(encoder.prediction_error())
                } else {
                    None
                };
                fep_result = fep_agent.step_with_encoder_pe(physics.state(), &command, enc_pe);

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

        let gait_summary = gait_analyzer.summary();

        // Cost of Transport: energy / (mass × distance)
        let total_distance = (horizontal_pos[0].powi(2) + horizontal_pos[1].powi(2)).sqrt();
        let cost_of_transport = if total_distance > 0.01 {
            total_mechanical_energy / (70.0 * total_distance)
        } else {
            0.0
        };

        EpisodeMetrics {
            episode,
            avg_standing_reward: total_standing_reward / n,
            avg_episode_reward: total_episode_reward / n,
            avg_free_energy: if fe_samples > 0 {
                total_fe / fe_samples as f64
            } else {
                0.0
            },
            avg_head_height: total_head_height / n,
            avg_uprightness: total_uprightness / n,
            avg_horizontal_speed: total_horizontal_speed / n,
            avg_control_effort: total_control_effort / n,
            exploration_count,
            total_steps: steps_completed,
            task,
            avg_foot_clearance: gait_summary.avg_clearance,
            min_foot_clearance: gait_summary.min_clearance,
            avg_stride_length: gait_summary.avg_stride_length,
            avg_cadence: gait_summary.avg_cadence,
            gait_asymmetry: gait_summary.gait_asymmetry,
            cost_of_transport,
            step_regularity: gait_summary.step_regularity,
            foot_strike_quality: gait_summary.foot_strike_quality,
            telemetry,
        }
    }

    /// Run a single training episode using the simple physics model.
    pub fn run_episode(
        &self,
        encoder: &mut HumanoidHdcEncoder,
        controller: &mut HumanoidController,
        fep_agent: &mut ActiveInferenceHumanoidAgent,
        episode: usize,
    ) -> EpisodeMetrics {
        let noise_scale = if self.config.progressive_noise && self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            progress.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut physics = SimpleHumanoidSimulator::new()
            .with_domain_randomization(self.config.domain_randomization)
            .with_actuator_noise(self.config.actuator_noise_std)
            .with_observation_noise(self.config.observation_noise_std)
            .with_terrain_variation(self.config.terrain_variation)
            .with_noise_scale(noise_scale);
        self.run_episode_with_sim(encoder, controller, fep_agent, &mut physics, episode)
    }

    /// Run the full training curriculum.
    pub fn train(&mut self) -> Vec<EpisodeMetrics> {
        self.curriculum_state = CurriculumState::new();

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new(&genesis, num_levels);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let fep_config = HumanoidFepConfig {
            exploration_decay_rate: self.config.exploration_decay_rate,
            ..HumanoidFepConfig::default()
        };
        let mut fep_agent = ActiveInferenceHumanoidAgent::new(fep_config, self.config.task);

        // Warmup: pre-train on static standing samples
        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing();
                let offset = (i as f64 - 10.0) * 0.005;
                state.joint_angles[0] += offset; // Small abdomen perturbation
                let hv = encoder.encode(&state);
                let target = pd_standing_baseline(&state, &self.pd_gains);
                (hv, target)
            })
            .collect();
        controller.warmup(&warmup_samples, 100, self.config.learning_rate * 10.0);
        encoder.reset();

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);

        // Best-checkpoint revert state
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;

        // Phase-transition LR boost: track previous task to detect transitions
        let mut prev_task: Option<HumanoidTask> = None;
        let mut transition_boost_remaining = 0u32;

        for ep in 0..self.config.num_episodes {
            // Detect phase transitions and apply LR boost
            let (current_task, _, _) = self.curriculum(ep);
            if let Some(prev) = prev_task {
                if prev != current_task {
                    // Phase transition: boost LR for 3 episodes
                    transition_boost_remaining = 3;
                }
            }
            prev_task = Some(current_task);

            if transition_boost_remaining > 0 {
                controller.set_learning_rate_scale(3.0);
                transition_boost_remaining -= 1;
            } else {
                controller.set_learning_rate_scale(1.0);
            }

            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            // Track best checkpoint (output projection weights)
            if metrics.avg_standing_reward > best_reward {
                best_reward = metrics.avg_standing_reward;
                best_weights = Some(controller.output_projection());
                consecutive_low = 0;
            } else if metrics.avg_standing_reward < 0.5 {
                consecutive_low += 1;
                // Revert to best checkpoint if 3 consecutive low-reward episodes
                if consecutive_low >= 3 {
                    if let Some((ref weights, ref bias)) = best_weights {
                        controller.set_output_projection(weights, bias);
                        consecutive_low = 0;
                    }
                }
            } else {
                consecutive_low = 0;
            }

            // Adaptive curriculum: check for early phase advancement
            self.check_phase_advance(ep, &metrics);

            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Train and return the trained controller + encoder (for evaluation).
    ///
    /// Unlike `train()`, this returns the actual trained weights — not a fresh genesis.
    /// Uses the same training loop as `train()` but preserves the controller/encoder.
    pub fn train_and_extract(
        &mut self,
    ) -> (Vec<EpisodeMetrics>, HumanoidController, HumanoidHdcEncoder) {
        self.curriculum_state = CurriculumState::new();

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder =
            HumanoidHdcEncoder::new_predictive(&genesis, num_levels, self.config.morphology);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let fep_config = HumanoidFepConfig {
            exploration_decay_rate: self.config.exploration_decay_rate,
            ..HumanoidFepConfig::default()
        };
        let mut fep_agent = ActiveInferenceHumanoidAgent::new(fep_config, self.config.task);

        // Warmup
        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing();
                let offset = (i as f64 - 10.0) * 0.005;
                state.joint_angles[0] += offset;
                let hv = encoder.encode(&state);
                let target = pd_standing_baseline(&state, &self.pd_gains);
                (hv, target)
            })
            .collect();
        controller.warmup(&warmup_samples, 100, self.config.learning_rate * 10.0);
        encoder.reset();

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;

        for ep in 0..self.config.num_episodes {
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            if metrics.avg_episode_reward > best_reward {
                best_reward = metrics.avg_episode_reward;
                best_weights = Some(controller.output_projection());
                consecutive_low = 0;
            } else {
                consecutive_low += 1;
                if consecutive_low >= 3 && best_reward > 0.5 {
                    if let Some((ref w, ref b)) = best_weights {
                        controller.set_output_projection(w, b);
                    }
                    consecutive_low = 0;
                }
            }

            all_metrics.push(metrics);
        }

        self.metrics = all_metrics.clone();
        (all_metrics, controller, encoder)
    }

    /// DAgger training: periodically expose controller to PD-free states.
    ///
    /// Every `dagger_interval` episodes, runs the controller ALONE for `dagger_steps`
    /// steps, collects the resulting (out-of-distribution) states, and trains the
    /// controller on "what PD would output" for those states. This closes the
    /// distribution shift gap between PD-supported training and PD-free evaluation.
    pub fn train_dagger(
        &mut self,
        dagger_interval: usize,
        dagger_steps: usize,
    ) -> (Vec<EpisodeMetrics>, HumanoidController, HumanoidHdcEncoder) {
        self.curriculum_state = CurriculumState::new();

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder =
            HumanoidHdcEncoder::new_predictive(&genesis, num_levels, self.config.morphology);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let fep_config = HumanoidFepConfig {
            exploration_decay_rate: self.config.exploration_decay_rate,
            ..HumanoidFepConfig::default()
        };
        let mut fep_agent = ActiveInferenceHumanoidAgent::new(fep_config, self.config.task);

        // Warmup
        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing();
                let offset = (i as f64 - 10.0) * 0.005;
                state.joint_angles[0] += offset;
                let hv = encoder.encode(&state);
                let target = pd_standing_baseline(&state, &self.pd_gains);
                (hv, target)
            })
            .collect();
        controller.warmup(&warmup_samples, 100, self.config.learning_rate * 10.0);
        encoder.reset();

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;
        let dt = self.config.physics_dt();

        for ep in 0..self.config.num_episodes {
            // Normal training episode (with PD curriculum)
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            // Best-checkpoint logic
            if metrics.avg_episode_reward > best_reward {
                best_reward = metrics.avg_episode_reward;
                best_weights = Some(controller.output_projection());
                consecutive_low = 0;
            } else {
                consecutive_low += 1;
                if consecutive_low >= 3 && best_reward > 0.5 {
                    if let Some((ref w, ref b)) = best_weights {
                        controller.set_output_projection(w, b);
                    }
                    consecutive_low = 0;
                }
            }

            all_metrics.push(metrics);

            // DAgger round: every dagger_interval episodes, run controller ALONE
            // and train on the out-of-distribution states
            if ep > 0 && ep % dagger_interval == 0 {
                let mut dagger_sim = SimpleHumanoidSimulator::new();
                let mut dagger_encoder = HumanoidHdcEncoder::new(&genesis, num_levels);

                // Collect OOD states by running controller alone
                let mut ood_samples: Vec<(ContinuousHV, HumanoidCommand)> = Vec::new();

                for _step in 0..dagger_steps {
                    let state = dagger_sim.state().clone();
                    let hv = dagger_encoder.encode(&state);

                    // What the controller outputs (possibly bad)
                    let _cmd = controller.forward(&hv, dt as f32);

                    // What the PD WOULD output for this state (the "expert" label)
                    let pd_target = pd_standing_baseline(&state, &self.pd_gains);

                    // Collect this (state_encoding, expert_action) pair
                    ood_samples.push((hv, pd_target.clone()));

                    // Step physics with the CONTROLLER's output (not PD)
                    // This generates the actual OOD trajectory
                    dagger_sim.step(&_cmd, dt);

                    // Early exit if completely fallen
                    if dagger_sim.state().head_height < 0.3 {
                        break;
                    }
                }

                // Train on the OOD samples
                let dagger_lr = self.config.learning_rate * 2.0; // Higher LR for correction
                for (hv, target) in &ood_samples {
                    controller.forward(hv, dt as f32);
                    controller.train_step(hv, target, dt as f32, Some(dagger_lr));
                }

                controller.reset(); // Reset network state after DAgger episode
            }
        }

        self.metrics = all_metrics.clone();
        (all_metrics, controller, encoder)
    }

    /// Run training with CSV telemetry output.
    pub fn train_with_telemetry(&mut self, output_dir: &str) -> Vec<EpisodeMetrics> {
        self.config.collect_telemetry = true;
        self.curriculum_state = CurriculumState::new();

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new(&genesis, num_levels);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let fep_config = HumanoidFepConfig {
            exploration_decay_rate: self.config.exploration_decay_rate,
            ..HumanoidFepConfig::default()
        };
        let mut fep_agent = ActiveInferenceHumanoidAgent::new(fep_config, self.config.task);

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);
        let _ = std::fs::create_dir_all(output_dir);

        let mut summary = String::from(
            "episode,task,avg_standing_reward,avg_episode_reward,avg_free_energy,avg_head_height,avg_uprightness,avg_horizontal_speed,avg_control_effort,exploration_count,total_steps\n",
        );

        // Best-checkpoint revert state
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;

        // Phase-transition LR boost
        let mut prev_task: Option<HumanoidTask> = None;
        let mut transition_boost_remaining = 0u32;

        for ep in 0..self.config.num_episodes {
            let (current_task, _, _) = self.curriculum(ep);
            if let Some(prev) = prev_task {
                if prev != current_task {
                    transition_boost_remaining = 3;
                }
            }
            prev_task = Some(current_task);

            if transition_boost_remaining > 0 {
                controller.set_learning_rate_scale(3.0);
                transition_boost_remaining -= 1;
            } else {
                controller.set_learning_rate_scale(1.0);
            }

            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            // Track best checkpoint and revert on sustained degradation
            if metrics.avg_standing_reward > best_reward {
                best_reward = metrics.avg_standing_reward;
                best_weights = Some(controller.output_projection());
                consecutive_low = 0;
            } else if metrics.avg_standing_reward < 0.5 {
                consecutive_low += 1;
                if consecutive_low >= 3 {
                    if let Some((ref weights, ref bias)) = best_weights {
                        controller.set_output_projection(weights, bias);
                        consecutive_low = 0;
                    }
                }
            } else {
                consecutive_low = 0;
            }

            // Adaptive curriculum: check for early phase advancement
            self.check_phase_advance(ep, &metrics);

            if !metrics.telemetry.is_empty() {
                let step_path = format!("{}/episode_{:04}.csv", output_dir, ep);
                let mut csv = String::from(
                    "step,time,head_height,uprightness,horizontal_speed,standing_reward,episode_reward,free_energy,tau_factor,learning_rate,control_effort,r_foot_z,l_foot_z\n",
                );
                for t in &metrics.telemetry {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                        t.step,
                        t.time,
                        t.head_height,
                        t.uprightness,
                        t.horizontal_speed,
                        t.standing_reward,
                        t.episode_reward,
                        t.free_energy,
                        t.tau_factor,
                        t.learning_rate,
                        t.control_effort,
                        t.r_foot_z,
                        t.l_foot_z,
                    ));
                }
                let _ = std::fs::write(&step_path, csv);
            }

            summary.push_str(&format!(
                "{},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                ep,
                metrics.task,
                metrics.avg_standing_reward,
                metrics.avg_episode_reward,
                metrics.avg_free_energy,
                metrics.avg_head_height,
                metrics.avg_uprightness,
                metrics.avg_horizontal_speed,
                metrics.avg_control_effort,
                metrics.exploration_count,
                metrics.total_steps,
            ));

            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        let summary_path = format!("{}/summary.csv", output_dir);
        let _ = std::fs::write(&summary_path, summary);

        // Save checkpoint of trained controller
        let checkpoint_path = format!("{}/checkpoint.json", output_dir);
        let _ = controller.save_checkpoint(&checkpoint_path, &self.config);

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &HumanoidConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_single_episode() {
        let config = HumanoidConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            ..HumanoidConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = HumanoidTrainer::new(config.clone());

        let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = HumanoidController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        assert_eq!(metrics.episode, 0);
        assert!(metrics.total_steps > 0);
        assert!(metrics.avg_standing_reward.is_finite());
        assert!(metrics.avg_free_energy.is_finite());
    }

    #[test]
    fn test_full_training_runs() {
        let config = HumanoidConfig {
            num_episodes: 3,
            steps_per_episode: 50,
            ..HumanoidConfig::default()
        };
        let mut trainer = HumanoidTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 3);
        for (i, m) in metrics.iter().enumerate() {
            assert_eq!(m.episode, i);
        }
    }

    #[test]
    fn test_training_determinism() {
        let config = HumanoidConfig {
            num_episodes: 2,
            steps_per_episode: 30,
            ..HumanoidConfig::default()
        };

        let mut trainer1 = HumanoidTrainer::new(config.clone());
        let mut trainer2 = HumanoidTrainer::new(config);

        let m1 = trainer1.train();
        let m2 = trainer2.train();

        assert!(
            (m1[0].avg_standing_reward - m2[0].avg_standing_reward).abs() < 1e-6,
            "Same genesis -> same metrics"
        );
    }

    #[test]
    fn test_curriculum_phases_fixed() {
        let config = HumanoidConfig {
            num_episodes: 100,
            steps_per_episode: 10,
            adaptive_curriculum: false,
            ..HumanoidConfig::default()
        };
        let trainer = HumanoidTrainer::new(config);

        // Phase 1: Stand with heavy PD (episode 0 = progress 0.0)
        let (task0, pd0, speed0) = trainer.curriculum(0);
        assert_eq!(task0, HumanoidTask::Stand);
        assert!((pd0 - 0.8).abs() < 0.01);
        assert!((speed0 - 0.0).abs() < 0.01);

        // Phase 2: Stand with decaying PD (episode 30 = progress 0.303)
        let (task30, pd30, _) = trainer.curriculum(30);
        assert_eq!(task30, HumanoidTask::Stand);
        assert!(pd30 < 0.4, "PD should be decaying: {pd30}");

        // Phase 3: Walk (episode 50 = progress 0.505)
        let (task50, _pd50, speed50) = trainer.curriculum(50);
        assert_eq!(task50, HumanoidTask::Walk);
        assert!(
            speed50 > 0.0 && speed50 < 1.0,
            "Walk speed ramping: {speed50}"
        );

        // Phase 3 still: Walk extends to 90% (episode 80 = progress 0.808)
        let (task80, _pd80, speed80) = trainer.curriculum(80);
        assert_eq!(task80, HumanoidTask::Walk);
        assert!(
            speed80 > 0.5 && speed80 <= 1.0,
            "Walk speed near end: {speed80}"
        );

        // Phase 4: Run (episode 92 = progress 0.929)
        let (task92, pd92, speed92) = trainer.curriculum(92);
        assert_eq!(task92, HumanoidTask::Run);
        assert!(pd92 < 0.05, "PD nearly zero in Run: {pd92}");
        assert!(speed92 > 1.0, "Run speed > 1 m/s: {speed92}");

        // Final episode: speed capped at 3 m/s
        let (_task99, _pd99, speed99) = trainer.curriculum(99);
        assert!(
            speed99 <= 3.01,
            "Run speed should be capped at ~3 m/s: {speed99}"
        );
    }

    #[test]
    fn test_adaptive_curriculum_phase_advance() {
        let config = HumanoidConfig {
            num_episodes: 100,
            steps_per_episode: 50,
            adaptive_curriculum: true,
            standing_mastery_threshold: 0.85,
            mastery_streak_required: 3,
            ..HumanoidConfig::default()
        };
        let mut trainer = HumanoidTrainer::new(config.clone());

        // Phase 0 at start
        let (task, pd, _) = trainer.curriculum(0);
        assert_eq!(task, HumanoidTask::Stand);
        assert!((pd - 0.8).abs() < 0.01);

        // Simulate mastery: high standing reward for 3 consecutive episodes after min 10
        // Min duration = 10 episodes (ep 0-9), then mastery can fire at ep 9,10,11
        // Streak of 3 completes at ep 11 → phase advances
        let mut advanced_at = None;
        for ep in 0..13 {
            let metrics = EpisodeMetrics {
                episode: ep,
                avg_standing_reward: 0.90,
                avg_episode_reward: 0.90,
                avg_free_energy: 0.5,
                avg_head_height: 1.38,
                avg_uprightness: 0.97,
                avg_horizontal_speed: 0.0,
                avg_control_effort: 0.3,
                exploration_count: 0,
                total_steps: config.steps_per_episode,
                task: HumanoidTask::Stand,
                avg_foot_clearance: 0.0,
                min_foot_clearance: 0.0,
                avg_stride_length: 0.0,
                avg_cadence: 0.0,
                gait_asymmetry: 0.0,
                cost_of_transport: 0.0,
                step_regularity: 0.0,
                foot_strike_quality: 0.0,
                telemetry: Vec::new(),
            };
            if trainer.check_phase_advance(ep, &metrics) {
                advanced_at = Some(ep);
                break;
            }
        }

        assert_eq!(
            advanced_at,
            Some(11),
            "Should advance at ep 11 (min 10 + streak 3 at ep 9,10,11)"
        );

        // After advancement: phase 1 (Stand autonomy ramp)
        assert_eq!(trainer.curriculum_state.phase, 1);
        let (task, pd, _) = trainer.curriculum(12);
        assert_eq!(task, HumanoidTask::Stand);
        assert!((pd - 0.4).abs() < 0.05, "Phase 1 starts at PD ~0.4: {pd}");
    }

    #[test]
    fn test_adaptive_curriculum_respects_minimum() {
        let config = HumanoidConfig {
            num_episodes: 100,
            steps_per_episode: 50,
            adaptive_curriculum: true,
            mastery_streak_required: 1,
            ..HumanoidConfig::default()
        };
        let mut trainer = HumanoidTrainer::new(config.clone());

        // Try to advance at episode 5 (below min=10) — should NOT advance
        let metrics = EpisodeMetrics {
            episode: 5,
            avg_standing_reward: 0.99,
            avg_episode_reward: 0.99,
            avg_free_energy: 0.1,
            avg_head_height: 1.4,
            avg_uprightness: 0.99,
            avg_horizontal_speed: 0.0,
            avg_control_effort: 0.1,
            exploration_count: 0,
            total_steps: config.steps_per_episode,
            task: HumanoidTask::Stand,
            avg_foot_clearance: 0.0,
            min_foot_clearance: 0.0,
            avg_stride_length: 0.0,
            avg_cadence: 0.0,
            gait_asymmetry: 0.0,
            cost_of_transport: 0.0,
            step_regularity: 0.0,
            foot_strike_quality: 0.0,
            telemetry: Vec::new(),
        };
        let advanced = trainer.check_phase_advance(5, &metrics);
        assert!(
            !advanced,
            "Should not advance before min duration (10 episodes)"
        );
        assert_eq!(trainer.curriculum_state.phase, 0);
    }

    #[test]
    fn test_telemetry_enabled() {
        let config = HumanoidConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: true,
            ..HumanoidConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = HumanoidTrainer::new(config.clone());

        let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = HumanoidController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.telemetry.len() > 0);
    }

    #[test]
    fn test_telemetry_disabled() {
        let config = HumanoidConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: false,
            ..HumanoidConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = HumanoidTrainer::new(config.clone());

        let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = HumanoidController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.telemetry.is_empty());
    }

    #[test]
    fn test_multi_episode_bounded_errors() {
        let config = HumanoidConfig {
            num_episodes: 5,
            steps_per_episode: 100,
            ..HumanoidConfig::default()
        };
        let mut trainer = HumanoidTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 5);
        for m in &metrics {
            assert!(m.avg_standing_reward.is_finite());
            assert!(m.avg_episode_reward.is_finite());
            assert!(m.avg_head_height.is_finite());
        }
    }

    #[test]
    fn test_checkpoint_resume_training() {
        let config = HumanoidConfig {
            num_episodes: 3,
            steps_per_episode: 50,
            ..HumanoidConfig::default()
        };
        let dir = "/tmp/symthaea_humanoid_test_resume";
        let _ = std::fs::remove_dir_all(dir);

        // Phase 1: Train and save checkpoint
        let mut trainer1 = HumanoidTrainer::new(config.clone());
        let _ = trainer1.train_with_telemetry(dir);

        let checkpoint_path = format!("{}/checkpoint.json", dir);
        assert!(std::path::Path::new(&checkpoint_path).exists());

        // Phase 2: Resume from checkpoint
        let resume_config = HumanoidConfig {
            num_episodes: 2,
            steps_per_episode: 50,
            ..HumanoidConfig::default()
        };
        let mut trainer2 = HumanoidTrainer::with_checkpoint(resume_config, &checkpoint_path)
            .expect("checkpoint should load");
        let metrics = trainer2.train();

        assert_eq!(metrics.len(), 2);
        for m in &metrics {
            assert!(m.avg_standing_reward.is_finite());
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_cost_of_transport_computed() {
        let config = HumanoidConfig {
            num_episodes: 1,
            steps_per_episode: 100,
            task: HumanoidTask::Walk,
            target_speed: Some(1.0),
            ..HumanoidConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = HumanoidTrainer::new(config.clone());

        let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = HumanoidController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Walk);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        assert!(
            metrics.cost_of_transport.is_finite(),
            "CoT should be finite: {}",
            metrics.cost_of_transport
        );
        assert!(
            metrics.cost_of_transport >= 0.0,
            "CoT should be non-negative: {}",
            metrics.cost_of_transport
        );
    }

    #[test]
    fn test_cost_of_transport_zero_for_standing() {
        // With domain randomization and actuator noise disabled, standing
        // should produce minimal horizontal drift → CoT near 0.
        let config = HumanoidConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            task: HumanoidTask::Stand,
            domain_randomization: false,
            actuator_noise_std: 0.0,
            ..HumanoidConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = HumanoidTrainer::new(config.clone());

        let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = HumanoidController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        // Standing with no noise: minimal drift, CoT should be very low
        assert!(
            metrics.cost_of_transport < 5.0,
            "Standing CoT should be low: {}",
            metrics.cost_of_transport
        );
    }

    #[test]
    fn test_csv_telemetry_output() {
        let config = HumanoidConfig {
            num_episodes: 2,
            steps_per_episode: 20,
            ..HumanoidConfig::default()
        };
        let mut trainer = HumanoidTrainer::new(config);
        let dir = "/tmp/symthaea_humanoid_test_csv";
        let _ = std::fs::remove_dir_all(dir);

        let metrics = trainer.train_with_telemetry(dir);
        assert_eq!(metrics.len(), 2);

        let summary = std::fs::read_to_string(format!("{}/summary.csv", dir)).unwrap();
        assert!(summary.starts_with("episode,task,"));

        let _ = std::fs::remove_dir_all(dir);
    }
}
