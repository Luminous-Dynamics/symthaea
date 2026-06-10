// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid training: multi-rate loop with PD baseline targets and adaptive curriculum.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
use crate::gait::GaitAnalyzer;
use crate::morphology::HumanoidMorphology;
use crate::reward;
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::types::*;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub episode: usize,
    pub avg_standing_reward: f64,
    pub avg_episode_reward: f64,
    pub avg_free_energy: f64,
    pub avg_head_height: f64,
    pub avg_uprightness: f64,
    pub avg_horizontal_speed: f64,
    pub avg_control_effort: f64,
    pub exploration_count: usize,
    pub total_steps: usize,
    pub task: HumanoidTask,
    pub avg_foot_clearance: f64,
    pub min_foot_clearance: f64,
    pub avg_stride_length: f64,
    pub avg_cadence: f64,
    pub gait_asymmetry: f64,
    pub cost_of_transport: f64,
    pub step_regularity: f64,
    pub foot_strike_quality: f64,
    pub telemetry: Vec<HumanoidTelemetry>,
}

#[derive(Debug, Clone)]
struct CurriculumState {
    pub phase: usize,
    pub phase_start_ep: usize,
    pub mastery_streak: usize,
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

pub struct HumanoidTrainer {
    config: HumanoidConfig,
    genesis: GenesisSeed,
    pd_gains: HumanoidPdGains,
    pub metrics: Vec<EpisodeMetrics>,
    initial_controller: Option<HumanoidController>,
    curriculum_state: CurriculumState,
}

impl HumanoidTrainer {
    pub fn new(config: HumanoidConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let pd_gains = HumanoidPdGains::for_morphology(config.morphology);
        Self {
            config,
            genesis,
            pd_gains,
            metrics: Vec::new(),
            initial_controller: None,
            curriculum_state: CurriculumState::new(),
        }
    }

    pub fn with_controller(config: HumanoidConfig, controller: HumanoidController) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let pd_gains = HumanoidPdGains::for_morphology(config.morphology);
        Self {
            config,
            genesis,
            pd_gains,
            metrics: Vec::new(),
            initial_controller: Some(controller),
            curriculum_state: CurriculumState::new(),
        }
    }

    pub fn with_checkpoint(config: HumanoidConfig, checkpoint_path: &str) -> std::io::Result<Self> {
        let controller = HumanoidController::load_checkpoint(checkpoint_path)?;
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let pd_gains = HumanoidPdGains::for_morphology(config.morphology);
        Ok(Self {
            config,
            genesis,
            pd_gains,
            metrics: Vec::new(),
            initial_controller: Some(controller),
            curriculum_state: CurriculumState::new(),
        })
    }

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

    fn curriculum_adaptive(&self, episode: usize) -> (HumanoidTask, f32, f64) {
        let total = self.config.num_episodes;
        let cs = &self.curriculum_state;

        let max_durations = [
            (total as f64 * 0.25).ceil() as usize,
            (total as f64 * 0.15).ceil() as usize,
            (total as f64 * 0.50).ceil() as usize,
            total.saturating_sub(cs.phase_start_ep),
        ];

        let episodes_in_phase = episode.saturating_sub(cs.phase_start_ep);
        let max_dur = max_durations[cs.phase.min(3)].max(1);
        let phase_progress = (episodes_in_phase as f64 / max_dur as f64).min(1.0);

        match cs.phase {
            0 => (HumanoidTask::Stand, 0.8 - 0.4 * phase_progress as f32, 0.0),
            1 => (HumanoidTask::Stand, 0.4 - 0.3 * phase_progress as f32, 0.0),
            2 => (
                HumanoidTask::Walk,
                0.10 - 0.05 * phase_progress as f32,
                phase_progress,
            ),
            _ => (
                HumanoidTask::Run,
                0.05 * (1.0 - phase_progress as f32),
                1.0 + phase_progress * 2.0,
            ),
        }
    }

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

        if episodes_in_phase >= max_durations[phase] {
            self.curriculum_state.phase += 1;
            self.curriculum_state.phase_start_ep = ep + 1;
            self.curriculum_state.mastery_streak = 0;
            return true;
        }

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
        let perturbation = if self.config.num_episodes > 1 {
            0.01 + 0.09 * (episode as f64 / (self.config.num_episodes - 1) as f64)
        } else {
            0.01
        };
        physics.reset_with_perturbation(perturbation, episode as u64 + 42);

        encoder.reset();
        controller.reset();

        let initial_cmd = HumanoidCommand::zero_for(self.config.morphology.num_actuators());
        let mut fep_result = fep_agent.step_with_encoder_pe(physics.state(), &initial_cmd, None);

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

        let replay_cap = self.config.replay_buffer_size;
        let replay_count = self.config.replay_count;
        let mut replay_buf: Vec<(ContinuousHV, HumanoidCommand)> = Vec::with_capacity(replay_cap);
        let mut replay_idx = 0usize;
        let mut replay_rng = episode as u64 + 7919;

        let mut gait_analyzer = GaitAnalyzer::new();
        let mut horizontal_pos = [0.0f64; 2];
        let mut total_mechanical_energy = 0.0f64;

        let gait_freq = match task {
            HumanoidTask::Walk => 1.2 + 0.6 * target_speed.min(1.0),
            HumanoidTask::Run => 2.0 + 0.3 * (target_speed - 1.0).clamp(0.0, 2.0),
            _ => 0.0,
        };

        let lr_scale = if self.config.enable_lr_schedule && self.config.num_episodes > 1 {
            (0.5 * (1.0
                + (std::f64::consts::PI
                    * (episode as f64 / (self.config.num_episodes - 1) as f64))
                    .cos())) as f32
        } else {
            1.0f32
        };

        for step in 0..self.config.steps_per_episode {
            let state = physics.state().clone();
            let sensor_hv = encoder.encode(&state);
            let mut command = controller.forward(&sensor_hv, dt as f32);
            let gait_phase = (step as f64 * dt * gait_freq).fract();

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
                // Fix: Evaluates dynamically over the layout width to safely handle extra hand actuators
                for i in 0..command.torques.len() {
                    command.torques[i] =
                        pd_weight * pd_cmd.torques[i] + (1.0 - pd_weight) * command.torques[i];
                }
                command = command.clamped();
            }

            // Fix: Re-map exploration vectors safely to prevent zipper truncation bugs
            if let Some(noise) = &fep_result.exploration_noise {
                let mut padded_noise = vec![0.0f32; command.torques.len()];
                let limit = noise.len().min(padded_noise.len());
                padded_noise[..limit].copy_from_slice(&noise[..limit]);
                command = command.with_noise(&padded_noise);
                exploration_count += 1;
            }

            physics.step(&command, dt);

            {
                let post_state = physics.state();
                horizontal_pos[0] += post_state.root_linear_velocity[0] * dt;
                horizontal_pos[1] += post_state.root_linear_velocity[1] * dt;
                // Fix: Evaluate dynamic joint bounds for accurate Cost of Transport calculation profiles
                for i in 0..command.torques.len() {
                    total_mechanical_energy +=
                        (command.torques[i] as f64 * post_state.joint_velocities[i]).abs() * dt;
                }
                gait_analyzer.update_with_position(
                    post_state,
                    horizontal_pos,
                    post_state.timestamp,
                );
            }

            let standing_r = reward::standing_reward(&state);
            let episode_r = reward::episode_reward_ext(
                &state,
                &command,
                &task,
                target_speed,
                Some(self.config.object_position),
                Some(self.config.reach_hand),
            );
            total_standing_reward += standing_r;
            total_episode_reward += episode_r;
            total_head_height += state.head_height;
            total_uprightness += state.uprightness();
            total_horizontal_speed += state.horizontal_speed();
            total_control_effort += command.control_effort() as f64;
            steps_completed += 1;

            if self.config.early_termination && step > 10 {
                if state.head_height < 0.5 || state.uprightness() < 0.1 {
                    break;
                }
                if standing_r < 0.3 {
                    low_reward_streak += 1;
                    if low_reward_streak >= 50 {
                        break;
                    }
                } else {
                    low_reward_streak = 0;
                }
            }

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
                let reward_mod = (standing_r as f32).max(0.1);
                let lr = controller.learning_rate()
                    * fep_result.learning_rate_factor
                    * lr_scale
                    * reward_mod;

                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));

                let current_output_hv = controller.network().output().normalize();
                if replay_cap > 0 {
                    if replay_buf.len() < replay_cap {
                        replay_buf.push((current_output_hv, target));
                    } else {
                        replay_buf[replay_idx % replay_cap] = (current_output_hv, target);
                    }
                    replay_idx += 1;

                    let buf_len = replay_buf.len();
                    if buf_len > 1 {
                        for _ in 0..replay_count.min(buf_len) {
                            replay_rng =
                                replay_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let idx = (replay_rng >> 33) as usize % buf_len;
                            let (ref replay_hv, ref replay_target) = replay_buf[idx];
                            controller.train_head_replay(replay_hv, replay_target, lr);
                        }
                    }
                }
            }

            if step % 100 == 0 && step > 0 {
                controller.normalize_states();
            }

            if step % cognitive_interval == 0 {
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

    pub fn run_episode(
        &self,
        encoder: &mut HumanoidHdcEncoder,
        controller: &mut HumanoidController,
        fep_agent: &mut ActiveInferenceHumanoidAgent,
        episode: usize,
    ) -> EpisodeMetrics {
        // Fix: High-fidelity MuJoCo integration path only executes if morphology matches the 21-DOF standard asset sheet
        #[cfg(feature = "mujoco")]
        {
            if self.config.morphology == HumanoidMorphology::Dmc21 {
                if let Ok(mut physics) =
                    crate::simulator::MuJoCoHumanoidSimulator::from_bundled_asset()
                {
                    return self.run_episode_with_sim(
                        encoder,
                        controller,
                        fep_agent,
                        &mut physics,
                        episode,
                    );
                }
            }
        }

        let noise_scale = if self.config.progressive_noise && self.config.num_episodes > 1 {
            (episode as f64 / (self.config.num_episodes - 1) as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut physics = SimpleHumanoidSimulator::new_for(self.config.morphology)
            .with_domain_randomization(self.config.domain_randomization)
            .with_actuator_noise(self.config.actuator_noise_std)
            .with_observation_noise(self.config.observation_noise_std)
            .with_terrain_variation(self.config.terrain_variation)
            .with_noise_scale(noise_scale);
        self.run_episode_with_sim(encoder, controller, fep_agent, &mut physics, episode)
    }

    pub fn train(&mut self) -> Vec<EpisodeMetrics> {
        self.curriculum_state = CurriculumState::new();
        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new_for(&genesis, num_levels, self.config.morphology);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let fep_config = HumanoidFepConfig {
            exploration_decay_rate: self.config.exploration_decay_rate,
            ..HumanoidFepConfig::default()
        };
        let mut fep_agent = ActiveInferenceHumanoidAgent::new(fep_config, self.config.task);

        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing_for(self.config.morphology);
                if !state.joint_angles.is_empty() {
                    state.joint_angles[0] += (i as f64 - 10.0) * 0.005;
                }
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

            self.check_phase_advance(ep, &metrics);
            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }

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

        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing_for(self.config.morphology);
                if !state.joint_angles.is_empty() {
                    state.joint_angles[0] += (i as f64 - 10.0) * 0.005;
                }
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

        let warmup_samples: Vec<(ContinuousHV, HumanoidCommand)> = (0..20)
            .map(|i| {
                let mut state = HumanoidState::standing_for(self.config.morphology);
                if !state.joint_angles.is_empty() {
                    state.joint_angles[0] += (i as f64 - 10.0) * 0.005;
                }
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
            all_metrics.push(metrics.clone());
            self.check_phase_advance(ep, &metrics);

            if ep > 0
                && ep % dagger_interval == 0
                && metrics.avg_standing_reward > self.config.standing_mastery_threshold
            {
                let mut dagger_sim = SimpleHumanoidSimulator::new_for(self.config.morphology);
                let mut dagger_encoder =
                    HumanoidHdcEncoder::new_for(&genesis, num_levels, self.config.morphology);
                let mut ood_samples: Vec<(ContinuousHV, HumanoidCommand)> = Vec::new();

                for _step in 0..dagger_steps {
                    let state = dagger_sim.state().clone();
                    let hv = dagger_encoder.encode(&state);
                    let _cmd = controller.forward(&hv, dt as f32);
                    let pd_target = pd_standing_baseline(&state, &self.pd_gains);

                    let current_output_hv = controller.network().output().normalize();
                    ood_samples.push((current_output_hv, pd_target));
                    dagger_sim.step(&_cmd, dt);
                    if dagger_sim.state().head_height < 0.3 {
                        break;
                    }
                }

                let dagger_lr = self.config.learning_rate
                    * ((12.0 / self.config.morphology.num_actuators() as f64).min(0.05) as f32); // Cast evaluation block to f32
                for (output_hv, target) in &ood_samples {
                    controller.train_head_replay(output_hv, target, dagger_lr);
                }
                controller.reset();
            }
        }

        self.metrics = all_metrics.clone();
        (all_metrics, controller, encoder)
    }

    pub fn train_with_telemetry(&mut self, output_dir: &str) -> Vec<EpisodeMetrics> {
        self.config.collect_telemetry = true;
        self.curriculum_state = CurriculumState::new();
        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new_for(&genesis, num_levels, self.config.morphology);
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

        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;
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

            self.check_phase_advance(ep, &metrics);

            if !metrics.telemetry.is_empty() {
                let step_path = format!("{}/episode_{:04}.csv", output_dir, ep);
                let mut csv = String::from(
                    "step,time,head_height,uprightness,horizontal_speed,standing_reward,episode_reward,free_energy,tau_factor,learning_rate,control_effort,r_foot_z,l_foot_z\n",
                );
                for t in &metrics.telemetry {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                        t.step, t.time, t.head_height, t.uprightness, t.horizontal_speed, t.standing_reward,
                        t.episode_reward, t.free_energy, t.tau_factor, t.learning_rate, t.control_effort, t.r_foot_z, t.l_foot_z,
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

        let _ = std::fs::write(format!("{}/summary.csv", output_dir), summary);
        let _ =
            controller.save_checkpoint(&format!("{}/checkpoint.json", output_dir), &self.config);

        self.metrics = all_metrics.clone();
        all_metrics
    }

    pub fn config(&self) -> &HumanoidConfig {
        &self.config
    }
}
