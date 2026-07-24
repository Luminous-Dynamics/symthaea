// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid training: multi-rate loop with PD baseline targets and adaptive curriculum.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::actuation::ActuationAdapter;
use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::fall_protection::{FallProtectionController, FallProtectionPhase};
use crate::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
use crate::gait::{ContactLockedGaitClock, GaitAnalyzer};
use crate::hierarchical::HierarchicalHumanoidController;
use crate::reward;
use crate::safety::HumanoidSafetyProjector;
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::state_estimation::{FusedHumanoidStateEstimator, ProprioceptiveMeasurement};
use crate::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub safety_interventions: usize,
    pub avg_residual_authority: f64,
    pub avg_balance_effort: f64,
    #[serde(default)]
    pub avg_recovery_effort: f64,
    #[serde(default)]
    pub recovery_interventions: usize,
    #[serde(default)]
    pub avg_capture_margin_m: f64,
    #[serde(default)]
    pub min_capture_margin_m: f64,
    #[serde(default)]
    pub avg_total_normal_force_n: f64,
    #[serde(default)]
    pub planned_recovery_steps: usize,
    #[serde(default)]
    pub infeasible_whole_body_steps: usize,
    #[serde(default)]
    pub avg_whole_body_active_constraints: f64,
    #[serde(default)]
    pub max_whole_body_joint_utilization: f64,
    #[serde(default)]
    pub fall_protection_interventions: usize,
    #[serde(default)]
    pub get_up_attempts: usize,
    #[serde(default)]
    pub successful_get_ups: usize,
    #[serde(default)]
    pub rejected_state_measurements: usize,
    #[serde(default)]
    pub avg_orientation_innovation_rad: f64,
    #[serde(default)]
    pub avg_linear_velocity_innovation_mps: f64,
    #[serde(default)]
    pub avg_maximum_joint_innovation_rad: f64,
    #[serde(default)]
    pub terrain_planned_steps: usize,
    #[serde(default)]
    pub terrain_infeasible_steps: usize,
    #[serde(default)]
    pub avg_terrain_clearance_m: f64,
    #[serde(default)]
    pub inverse_dynamics_fallback_steps: usize,
    #[serde(default)]
    pub avg_inverse_dynamics_iterations: f64,
    #[serde(default)]
    pub max_inverse_dynamics_violation: f64,
    #[serde(default)]
    pub terrain_mpc_replans: usize,
    #[serde(default)]
    pub avg_terrain_mpc_candidates: f64,
    #[serde(default)]
    pub avg_terrain_mpc_cost: f64,
    #[serde(default)]
    pub contact_dynamics_fallback_steps: usize,
    #[serde(default)]
    pub max_contact_dynamics_residual_nm: f64,
    #[serde(default)]
    pub max_contact_acceleration_residual: f64,
    #[serde(default)]
    pub max_contact_friction_utilization: f64,
    #[serde(default)]
    pub contact_solver_budget_misses: usize,
    #[serde(default)]
    pub max_contact_solver_elapsed_us: u64,
    #[serde(default)]
    pub centroidal_model_steps: usize,
    #[serde(default)]
    pub avg_centroidal_authority: f64,
    #[serde(default)]
    pub avg_centroidal_correction_norm: f64,
    #[serde(default)]
    pub max_angular_momentum_norm: f64,
    #[serde(default)]
    pub max_linear_momentum_norm: f64,
    #[serde(default)]
    pub floating_base_model_steps: usize,
    #[serde(default)]
    pub floating_base_converged_steps: usize,
    #[serde(default)]
    pub floating_base_fallback_steps: usize,
    #[serde(default)]
    pub floating_base_solver_budget_misses: usize,
    #[serde(default)]
    pub max_floating_base_dynamics_residual: f64,
    #[serde(default)]
    pub max_floating_base_solver_elapsed_us: u64,
    #[serde(default)]
    pub floating_base_warm_started_steps: usize,
    #[serde(default)]
    pub floating_base_symbolic_reuse_steps: usize,
    #[serde(default)]
    pub max_floating_base_warm_start_active_bounds: usize,
    #[serde(default)]
    pub max_terrain_height_std_m: f64,
    #[serde(default)]
    pub max_terrain_evidence_age_s: f64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainerResumeManifest {
    version: u32,
    config: HumanoidConfig,
    curriculum_state: CurriculumState,
    metrics: Vec<EpisodeMetrics>,
    completed_episodes: usize,
    controller_file: String,
    morphology_schema_id: String,
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
        config
            .validate()
            .expect("invalid humanoid training configuration");
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
        config
            .validate()
            .expect("invalid humanoid training configuration");
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
        config.validate().map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
        })?;
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
        config
            .validate()
            .expect("invalid humanoid training configuration");
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

    /// Persist a coherent resumable training bundle. Recurrent learning must be
    /// disabled until the core HDC-LTC network exposes parameter serialization.
    pub fn save_resume_bundle(
        &self,
        controller: &HumanoidController,
        directory: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        if self.config.enable_recurrent_learning {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "resume bundles require checkpointable head-only learning",
            ));
        }
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;
        let controller_path = directory.join("controller.json");
        let controller_tmp = directory.join("controller.json.tmp");
        let trainer_path = directory.join("trainer.json");
        let trainer_tmp = directory.join("trainer.json.tmp");

        controller.save_checkpoint(&controller_tmp.to_string_lossy(), &self.config)?;
        std::fs::rename(&controller_tmp, &controller_path)?;

        let manifest = TrainerResumeManifest {
            version: 1,
            config: self.config.clone(),
            curriculum_state: self.curriculum_state.clone(),
            metrics: self.metrics.clone(),
            completed_episodes: self.metrics.len(),
            controller_file: "controller.json".to_string(),
            morphology_schema_id: self.config.morphology.schema_id().to_string(),
        };
        let json = serde_json::to_vec_pretty(&manifest).map_err(std::io::Error::other)?;
        std::fs::write(&trainer_tmp, json)?;
        std::fs::rename(&trainer_tmp, &trainer_path)?;
        Ok(())
    }

    /// Restore controller, curriculum position, and recorded metrics together.
    pub fn load_resume_bundle(directory: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let directory = directory.as_ref();
        let json = std::fs::read(directory.join("trainer.json"))?;
        let manifest: TrainerResumeManifest =
            serde_json::from_slice(&json).map_err(std::io::Error::other)?;
        if manifest.version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported trainer resume version {}", manifest.version),
            ));
        }
        manifest.config.validate().map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })?;
        if manifest.morphology_schema_id != manifest.config.morphology.schema_id() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "resume bundle morphology schema does not match its configuration",
            ));
        }
        if manifest.completed_episodes != manifest.metrics.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "resume bundle episode count does not match metrics",
            ));
        }
        let controller = HumanoidController::load_checkpoint(
            &directory.join(&manifest.controller_file).to_string_lossy(),
        )?;
        let genesis = GenesisSeed::from_phrase(&manifest.config.genesis_phrase);
        let pd_gains = HumanoidPdGains::for_morphology(manifest.config.morphology);
        Ok(Self {
            config: manifest.config,
            genesis,
            pd_gains,
            metrics: manifest.metrics,
            initial_controller: Some(controller),
            curriculum_state: manifest.curriculum_state,
        })
    }

    pub fn completed_episodes(&self) -> usize {
        self.metrics.len()
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
        let capabilities = physics.capabilities();
        capabilities
            .validate_for_morphology(self.config.morphology)
            .unwrap_or_else(|error| panic!("incompatible humanoid backend: {error}"));
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
        let mut total_residual_authority = 0.0f64;
        let mut total_balance_effort = 0.0f64;
        let mut total_recovery_effort = 0.0f64;
        let mut recovery_interventions = 0usize;
        let mut total_capture_margin = 0.0f64;
        let mut capture_margin_samples = 0usize;
        let mut min_capture_margin = f64::INFINITY;
        let mut total_normal_force = 0.0f64;
        let mut planned_recovery_steps = 0usize;
        let mut infeasible_whole_body_steps = 0usize;
        let mut total_whole_body_active_constraints = 0usize;
        let mut max_whole_body_joint_utilization = 0.0f64;
        let mut fall_protection_interventions = 0usize;
        let mut get_up_attempts = 0usize;
        let mut successful_get_ups = 0usize;
        let mut rejected_state_measurements = 0usize;
        let mut total_orientation_innovation = 0.0f64;
        let mut total_linear_velocity_innovation = 0.0f64;
        let mut total_maximum_joint_innovation = 0.0f64;
        let mut accepted_state_measurements = 0usize;
        let mut terrain_planned_steps = 0usize;
        let mut terrain_infeasible_steps = 0usize;
        let mut total_terrain_clearance_m = 0.0f64;
        let mut inverse_dynamics_fallback_steps = 0usize;
        let mut total_inverse_dynamics_iterations = 0usize;
        let mut max_inverse_dynamics_violation = 0.0f64;
        let mut terrain_mpc_replans = 0usize;
        let mut total_terrain_mpc_candidates = 0usize;
        let mut total_terrain_mpc_cost = 0.0f64;
        let mut contact_dynamics_fallback_steps = 0usize;
        let mut max_contact_dynamics_residual_nm = 0.0f64;
        let mut max_contact_acceleration_residual = 0.0f64;
        let mut max_contact_friction_utilization = 0.0f64;
        let mut contact_solver_budget_misses = 0usize;
        let mut max_contact_solver_elapsed_us = 0u64;
        let mut centroidal_model_steps = 0usize;
        let mut total_centroidal_authority = 0.0f64;
        let mut total_centroidal_correction_norm = 0.0f64;
        let mut max_angular_momentum_norm = 0.0f64;
        let mut max_linear_momentum_norm = 0.0f64;
        let mut floating_base_model_steps = 0usize;
        let mut floating_base_converged_steps = 0usize;
        let mut floating_base_fallback_steps = 0usize;
        let mut floating_base_solver_budget_misses = 0usize;
        let mut max_floating_base_dynamics_residual = 0.0f64;
        let mut max_floating_base_solver_elapsed_us = 0u64;
        let mut floating_base_warm_started_steps = 0usize;
        let mut floating_base_symbolic_reuse_steps = 0usize;
        let mut max_floating_base_warm_start_active_bounds = 0usize;
        let mut max_terrain_height_std_m = 0.0f64;
        let mut max_terrain_evidence_age_s = 0.0f64;
        let mut previous_fall_phase = FallProtectionPhase::Upright;
        let mut exploration_count = 0usize;
        let mut safety_interventions = 0usize;
        let mut safety_projector = HumanoidSafetyProjector::new(self.config.morphology);
        let hierarchical = HierarchicalHumanoidController::new(self.config.morphology);
        let mut fall_protection = FallProtectionController::new(self.config.morphology);
        let mut state_estimator = FusedHumanoidStateEstimator::new(self.config.morphology);
        state_estimator
            .reset(physics.observation())
            .expect("validated simulator observation must initialize the state estimator");
        let mut observation_sequence = 0u64;
        let actuation_adapter = ActuationAdapter::default();
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
        let mut gait_clock = ContactLockedGaitClock::new();
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
            let contact_frame = physics.contact_frame();
            let raw_observation = physics.observation().clone();
            observation_sequence = observation_sequence.saturating_add(1);
            let observation = if self.config.enable_state_estimation {
                let mut measurement = ProprioceptiveMeasurement::from_simulator(
                    self.config.morphology,
                    observation_sequence,
                    raw_observation.clone(),
                    contact_frame,
                );
                measurement.received_at_s = physics.true_state().timestamp;
                match state_estimator.update(&measurement) {
                    Ok((estimate, report)) => {
                        total_orientation_innovation += report.orientation_innovation_rad;
                        total_linear_velocity_innovation += report.linear_velocity_innovation_mps;
                        total_maximum_joint_innovation +=
                            report.maximum_joint_position_innovation_rad;
                        accepted_state_measurements += 1;
                        estimate.clone()
                    }
                    Err(_) => {
                        rejected_state_measurements += 1;
                        state_estimator.estimate().clone()
                    }
                }
            } else {
                raw_observation
            };
            let sensor_hv = encoder.encode(&observation);
            let learned_residual = controller.forward(&sensor_hv, dt as f32);
            let gait_phase = gait_clock.advance_with_contacts(&contact_frame, dt, gait_freq);
            let baseline = match task {
                HumanoidTask::Stand => pd_standing_baseline(&observation, &self.pd_gains),
                HumanoidTask::Walk => {
                    pd_walking_baseline(&observation, &self.pd_gains, gait_phase, target_speed)
                }
                HumanoidTask::Run => {
                    pd_running_baseline(&observation, &self.pd_gains, gait_phase, target_speed)
                }
                HumanoidTask::Reach => pd_reaching_baseline(
                    &observation,
                    &self.pd_gains,
                    self.config.object_position,
                    self.config.reach_hand,
                ),
                HumanoidTask::Grasp => {
                    let gp = (step as f64 / self.config.steps_per_episode as f64).min(1.0);
                    pd_grasping_baseline(
                        &observation,
                        &self.pd_gains,
                        self.config.object_position,
                        self.config.reach_hand,
                        gp,
                    )
                }
            };
            let (mut command, hierarchy_report) = hierarchical.synthesize_with_environment(
                task,
                &observation,
                &contact_frame,
                &*physics,
                &baseline,
                &learned_residual,
                pd_weight,
                current_fe,
            );
            total_residual_authority += hierarchy_report.residual_authority as f64;
            total_balance_effort += hierarchy_report.balance_effort as f64;
            total_recovery_effort += hierarchy_report.recovery_effort as f64;
            if hierarchy_report.recovery_mode != crate::recovery::RecoveryMode::Nominal {
                recovery_interventions += 1;
            }
            if hierarchy_report.capture_margin_m.is_finite() {
                total_capture_margin += hierarchy_report.capture_margin_m;
                capture_margin_samples += 1;
                min_capture_margin = min_capture_margin.min(hierarchy_report.capture_margin_m);
            }
            total_normal_force += contact_frame.total_normal_force_n();
            if hierarchy_report.planned_step.is_some() {
                planned_recovery_steps += 1;
            }
            if let Some(trajectory) = hierarchy_report.planned_swing {
                terrain_planned_steps += 1;
                total_terrain_clearance_m += trajectory.clearance_m;
                if !trajectory.feasible {
                    terrain_infeasible_steps += 1;
                }
            }
            total_inverse_dynamics_iterations += hierarchy_report.inverse_dynamics_iterations;
            if hierarchy_report.inverse_dynamics_fallback {
                inverse_dynamics_fallback_steps += 1;
            }
            if hierarchy_report.inverse_dynamics_max_violation.is_finite() {
                max_inverse_dynamics_violation = max_inverse_dynamics_violation
                    .max(hierarchy_report.inverse_dynamics_max_violation);
            }
            if hierarchy_report.terrain_horizon_steps > 0 {
                terrain_mpc_replans += 1;
                total_terrain_mpc_candidates += hierarchy_report.terrain_mpc_candidates;
                if hierarchy_report.terrain_mpc_cost.is_finite() {
                    total_terrain_mpc_cost += hierarchy_report.terrain_mpc_cost;
                }
            }
            if hierarchy_report.contact_dynamics_fallback {
                contact_dynamics_fallback_steps += 1;
            }
            if hierarchy_report.contact_dynamics_residual_nm.is_finite() {
                max_contact_dynamics_residual_nm = max_contact_dynamics_residual_nm
                    .max(hierarchy_report.contact_dynamics_residual_nm);
            }
            if hierarchy_report.contact_acceleration_residual.is_finite() {
                max_contact_acceleration_residual = max_contact_acceleration_residual
                    .max(hierarchy_report.contact_acceleration_residual);
            }
            if hierarchy_report.contact_friction_utilization.is_finite() {
                max_contact_friction_utilization = max_contact_friction_utilization
                    .max(hierarchy_report.contact_friction_utilization);
            }
            if hierarchy_report.contact_solver_budget_missed {
                contact_solver_budget_misses += 1;
            }
            max_contact_solver_elapsed_us =
                max_contact_solver_elapsed_us.max(hierarchy_report.contact_solver_elapsed_us);
            if hierarchy_report.centroidal_model_valid {
                centroidal_model_steps += 1;
                total_centroidal_authority += hierarchy_report.centroidal_authority;
                total_centroidal_correction_norm += hierarchy_report.centroidal_correction_norm;
                if hierarchy_report.angular_momentum_norm.is_finite() {
                    max_angular_momentum_norm =
                        max_angular_momentum_norm.max(hierarchy_report.angular_momentum_norm);
                }
                if hierarchy_report.linear_momentum_norm.is_finite() {
                    max_linear_momentum_norm =
                        max_linear_momentum_norm.max(hierarchy_report.linear_momentum_norm);
                }
            }
            if hierarchy_report.floating_base_model_available {
                floating_base_model_steps += 1;
            }
            if hierarchy_report.floating_base_dynamics_converged {
                floating_base_converged_steps += 1;
            }
            if hierarchy_report.floating_base_dynamics_fallback {
                floating_base_fallback_steps += 1;
            }
            if hierarchy_report.floating_base_solver_budget_missed {
                floating_base_solver_budget_misses += 1;
            }
            if hierarchy_report.floating_base_dynamics_residual.is_finite() {
                max_floating_base_dynamics_residual = max_floating_base_dynamics_residual
                    .max(hierarchy_report.floating_base_dynamics_residual);
            }
            max_floating_base_solver_elapsed_us = max_floating_base_solver_elapsed_us
                .max(hierarchy_report.floating_base_solver_elapsed_us);
            if hierarchy_report.floating_base_warm_start_used {
                floating_base_warm_started_steps += 1;
            }
            if hierarchy_report.floating_base_symbolic_pattern_reused {
                floating_base_symbolic_reuse_steps += 1;
            }
            max_floating_base_warm_start_active_bounds = max_floating_base_warm_start_active_bounds
                .max(hierarchy_report.floating_base_warm_start_active_bounds);
            if hierarchy_report.terrain_max_height_std_m.is_finite() {
                max_terrain_height_std_m =
                    max_terrain_height_std_m.max(hierarchy_report.terrain_max_height_std_m);
            }
            if hierarchy_report.terrain_max_evidence_age_s.is_finite() {
                max_terrain_evidence_age_s =
                    max_terrain_evidence_age_s.max(hierarchy_report.terrain_max_evidence_age_s);
            }
            if !hierarchy_report.whole_body_feasible {
                infeasible_whole_body_steps += 1;
            }
            total_whole_body_active_constraints += hierarchy_report.whole_body_active_constraints;
            if hierarchy_report.whole_body_joint_utilization.is_finite() {
                max_whole_body_joint_utilization = max_whole_body_joint_utilization
                    .max(hierarchy_report.whole_body_joint_utilization);
            }

            let (protective_command, fall_report) = if self.config.enable_fall_recovery {
                fall_protection.update(&observation, &contact_frame, dt)
            } else {
                let multi_contacts =
                    crate::multi_contact::MultiContactFrame::from_feet(&contact_frame)
                        .with_protective_candidates(&observation);
                (
                    HumanoidCommand::zero_for(self.config.morphology.num_actuators()),
                    crate::fall_protection::FallProtectionReport {
                        phase: FallProtectionPhase::Upright,
                        orientation: crate::fall_protection::FallOrientation::Upright,
                        phase_elapsed_s: 0.0,
                        intervention: false,
                        protective_effort: 0.0,
                        get_up_progress: 1.0,
                        active_contacts: multi_contacts.active_count(),
                        upper_body_support: multi_contacts.has_upper_body_support(),
                        knee_support: multi_contacts.has_knee_support(),
                        support_polygon_area_m2: multi_contacts.support_polygon_area_m2(),
                    },
                )
            };
            if fall_report.intervention {
                fall_protection_interventions += 1;
                let retained_policy = match fall_report.phase {
                    FallProtectionPhase::Bracing => 0.35,
                    FallProtectionPhase::ImpactProtection
                    | FallProtectionPhase::Settling
                    | FallProtectionPhase::GetUpReady
                    | FallProtectionPhase::Rising => 0.15,
                    FallProtectionPhase::Faulted => 0.0,
                    FallProtectionPhase::Upright => 1.0,
                };
                for (value, protective) in command
                    .torques
                    .iter_mut()
                    .zip(protective_command.torques.iter())
                {
                    *value = (retained_policy * *value + *protective).clamp(-1.0, 1.0);
                }
            }
            if fall_report.phase == FallProtectionPhase::GetUpReady
                && previous_fall_phase != FallProtectionPhase::GetUpReady
            {
                get_up_attempts += 1;
            }
            if fall_report.phase == FallProtectionPhase::Upright
                && previous_fall_phase == FallProtectionPhase::Rising
            {
                successful_get_ups += 1;
            }
            previous_fall_phase = fall_report.phase;

            // Exploration is never injected into bracing, impact protection,
            // settling, or get-up commands. Those modes are deterministic safety
            // behaviors, not policy exploration opportunities.
            if fall_report.phase == FallProtectionPhase::Upright {
                if let Some(noise) = &fep_result.exploration_noise {
                    let mut padded_noise = vec![0.0f32; command.torques.len()];
                    let limit = noise.len().min(padded_noise.len());
                    padded_noise[..limit].copy_from_slice(&noise[..limit]);
                    command = command.with_noise(&padded_noise);
                    exploration_count += 1;
                }
            }

            if self.config.enable_safety_projection {
                let projected = safety_projector.project(
                    &command,
                    &observation,
                    ActuationMode::NormalizedTorque,
                    dt,
                );
                if projected.report.intervened() {
                    safety_interventions += 1;
                }
                command = projected.command;
            }

            let policy_command = command;
            let adapted = actuation_adapter
                .adapt_normalized_torque_intent(
                    &policy_command,
                    &observation,
                    self.config.morphology,
                    physics.actuation_mode(),
                )
                .expect("validated policy command must adapt to backend actuation");
            let applied_command = adapted.command;

            debug_assert!(
                applied_command
                    .validate_for(
                        self.config.morphology.num_actuators(),
                        physics.actuation_mode(),
                    )
                    .is_ok(),
                "adapted command is incompatible with the embodiment backend"
            );
            physics.step(&applied_command, dt);

            let post_state = physics.true_state().clone();
            horizontal_pos[0] += post_state.root_linear_velocity[0] * dt;
            horizontal_pos[1] += post_state.root_linear_velocity[1] * dt;
            // Canonical effort proxy. Physical work is only reported as such by
            // evaluators that also have actuator torque receipts from the backend.
            for i in 0..policy_command
                .torques
                .len()
                .min(post_state.joint_velocities.len())
            {
                total_mechanical_energy +=
                    (policy_command.torques[i] as f64 * post_state.joint_velocities[i]).abs() * dt;
            }
            gait_analyzer.update_with_position(&post_state, horizontal_pos, post_state.timestamp);

            let standing_r = reward::standing_reward(&post_state);
            let episode_r = reward::episode_reward_ext(
                &post_state,
                &policy_command,
                &task,
                target_speed,
                Some(self.config.object_position),
                Some(self.config.reach_hand),
            );
            total_standing_reward += standing_r;
            total_episode_reward += episode_r;
            total_head_height += post_state.head_height;
            total_uprightness += post_state.uprightness();
            total_horizontal_speed += post_state.horizontal_speed();
            total_control_effort += policy_command.control_effort() as f64;
            steps_completed += 1;

            if self.config.early_termination && step > 10 {
                let unrecoverable_fall = if self.config.enable_fall_recovery {
                    fall_report.phase == FallProtectionPhase::Faulted
                } else {
                    post_state.head_height < 0.5 || post_state.uprightness() < 0.1
                };
                if unrecoverable_fall {
                    break;
                }
                if fall_report.phase == FallProtectionPhase::Upright && standing_r < 0.3 {
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
                    time: post_state.timestamp,
                    head_height: post_state.head_height,
                    uprightness: post_state.uprightness(),
                    horizontal_speed: post_state.horizontal_speed(),
                    standing_reward: standing_r,
                    episode_reward: episode_r,
                    free_energy: current_fe,
                    tau_factor: current_tau_factor,
                    learning_rate: controller.learning_rate(),
                    control_effort: policy_command.control_effort(),
                    residual_authority: hierarchy_report.residual_authority,
                    balance_effort: hierarchy_report.balance_effort,
                    recovery_mode: hierarchy_report.recovery_mode,
                    capture_margin_m: hierarchy_report.capture_margin_m,
                    recovery_effort: hierarchy_report.recovery_effort,
                    planned_footstep_world_m: hierarchy_report
                        .planned_step
                        .map(|plan| plan.target_world_m),
                    planned_swing_apex_world_m: hierarchy_report
                        .planned_swing
                        .map(|trajectory| trajectory.apex_world_m),
                    terrain_confidence: hierarchy_report.terrain_confidence,
                    terrain_clearance_m: hierarchy_report.terrain_clearance_m,
                    whole_body_active_constraints: hierarchy_report.whole_body_active_constraints,
                    whole_body_joint_utilization: hierarchy_report.whole_body_joint_utilization,
                    whole_body_objective_residual: hierarchy_report.whole_body_objective_residual,
                    whole_body_feasible: hierarchy_report.whole_body_feasible,
                    inverse_dynamics_iterations: hierarchy_report.inverse_dynamics_iterations,
                    inverse_dynamics_max_violation: hierarchy_report.inverse_dynamics_max_violation,
                    inverse_dynamics_fallback: hierarchy_report.inverse_dynamics_fallback,
                    fall_protection_phase: fall_report.phase,
                    fall_orientation: fall_report.orientation,
                    protective_effort: fall_report.protective_effort,
                    get_up_progress: fall_report.get_up_progress,
                    total_normal_force_n: contact_frame.total_normal_force_n(),
                    center_of_pressure_world_m: contact_frame.center_of_pressure_world_m(),
                    support_phase: hierarchy_report.support_phase,
                    contact_trust: hierarchy_report.contact_trust,
                    r_foot_z: post_state.extremities[8],
                    l_foot_z: post_state.extremities[11],
                });
            }

            if step % self.config.train_every == 0
                && fall_report.phase == FallProtectionPhase::Upright
            {
                let target = match task {
                    HumanoidTask::Stand => pd_standing_baseline(&observation, &self.pd_gains),
                    HumanoidTask::Walk => {
                        pd_walking_baseline(&observation, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Run => {
                        pd_running_baseline(&observation, &self.pd_gains, gait_phase, target_speed)
                    }
                    HumanoidTask::Reach => pd_reaching_baseline(
                        &observation,
                        &self.pd_gains,
                        self.config.object_position,
                        self.config.reach_hand,
                    ),
                    HumanoidTask::Grasp => {
                        let gp = (step as f64 / self.config.steps_per_episode as f64).min(1.0);
                        pd_grasping_baseline(
                            &observation,
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

                if self.config.enable_recurrent_learning {
                    controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
                } else {
                    let output_hv = controller.network().output().normalize();
                    controller.train_head_replay(&output_hv, &target, lr);
                }

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
                fep_result =
                    fep_agent.step_with_encoder_pe(physics.observation(), &policy_command, enc_pe);
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
            safety_interventions,
            avg_residual_authority: total_residual_authority / n,
            avg_balance_effort: total_balance_effort / n,
            avg_recovery_effort: total_recovery_effort / n,
            recovery_interventions,
            avg_capture_margin_m: if capture_margin_samples > 0 {
                total_capture_margin / capture_margin_samples as f64
            } else {
                0.0
            },
            min_capture_margin_m: if min_capture_margin.is_finite() {
                min_capture_margin
            } else {
                0.0
            },
            avg_total_normal_force_n: total_normal_force / n,
            planned_recovery_steps,
            infeasible_whole_body_steps,
            avg_whole_body_active_constraints: total_whole_body_active_constraints as f64 / n,
            max_whole_body_joint_utilization,
            fall_protection_interventions,
            get_up_attempts,
            successful_get_ups,
            rejected_state_measurements,
            avg_orientation_innovation_rad: if accepted_state_measurements > 0 {
                total_orientation_innovation / accepted_state_measurements as f64
            } else {
                0.0
            },
            avg_linear_velocity_innovation_mps: if accepted_state_measurements > 0 {
                total_linear_velocity_innovation / accepted_state_measurements as f64
            } else {
                0.0
            },
            avg_maximum_joint_innovation_rad: if accepted_state_measurements > 0 {
                total_maximum_joint_innovation / accepted_state_measurements as f64
            } else {
                0.0
            },
            terrain_planned_steps,
            terrain_infeasible_steps,
            avg_terrain_clearance_m: if terrain_planned_steps > 0 {
                total_terrain_clearance_m / terrain_planned_steps as f64
            } else {
                0.0
            },
            inverse_dynamics_fallback_steps,
            avg_inverse_dynamics_iterations: total_inverse_dynamics_iterations as f64 / n,
            max_inverse_dynamics_violation,
            terrain_mpc_replans,
            avg_terrain_mpc_candidates: if terrain_mpc_replans > 0 {
                total_terrain_mpc_candidates as f64 / terrain_mpc_replans as f64
            } else {
                0.0
            },
            avg_terrain_mpc_cost: if terrain_mpc_replans > 0 {
                total_terrain_mpc_cost / terrain_mpc_replans as f64
            } else {
                0.0
            },
            contact_dynamics_fallback_steps,
            max_contact_dynamics_residual_nm,
            max_contact_acceleration_residual,
            max_contact_friction_utilization,
            contact_solver_budget_misses,
            max_contact_solver_elapsed_us,
            centroidal_model_steps,
            avg_centroidal_authority: if centroidal_model_steps > 0 {
                total_centroidal_authority / centroidal_model_steps as f64
            } else {
                0.0
            },
            avg_centroidal_correction_norm: if centroidal_model_steps > 0 {
                total_centroidal_correction_norm / centroidal_model_steps as f64
            } else {
                0.0
            },
            max_angular_momentum_norm,
            max_linear_momentum_norm,
            floating_base_model_steps,
            floating_base_converged_steps,
            floating_base_fallback_steps,
            floating_base_solver_budget_misses,
            max_floating_base_dynamics_residual,
            max_floating_base_solver_elapsed_us,
            floating_base_warm_started_steps,
            floating_base_symbolic_reuse_steps,
            max_floating_base_warm_start_active_bounds,
            max_terrain_height_std_m,
            max_terrain_evidence_age_s,
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
        // Gravity-scaled curriculum: early episodes run at reduced gravity
        // (60%) so balance is learnable, ramping to full gravity by 60% of
        // training. (`with_gravity()` existed but was never called from
        // training — the "gravity-scaled curriculum" claim was unsupported
        // until 2026-07, robotics plan Tier 2.2.)
        let gravity_scale = if self.config.num_episodes > 1 {
            let progress = (episode as f64 / (self.config.num_episodes - 1) as f64).clamp(0.0, 1.0);
            (0.6 + 0.4 * (progress / 0.6)).min(1.0)
        } else {
            1.0
        };
        let mut physics = SimpleHumanoidSimulator::new_for(self.config.morphology)
            .with_domain_randomization(self.config.domain_randomization)
            .with_actuator_noise(self.config.actuator_noise_std)
            .with_observation_noise(self.config.observation_noise_std)
            .with_terrain_variation(self.config.terrain_variation)
            .with_noise_scale(noise_scale)
            .with_gravity(gravity_scale);
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
        // This snapshot covers only the output head. It must never be described
        // as a full-model rollback while recurrent parameters continue learning.
        let mut best_head_projection: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;
        let mut prev_task: Option<HumanoidTask> = None;
        let mut transition_boost_remaining = 0u32;

        for ep in 0..self.config.num_episodes {
            let (current_task, _, _) = self.curriculum(ep);
            if let Some(prev) = prev_task
                && prev != current_task
            {
                transition_boost_remaining = 3;
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
                best_head_projection = Some(controller.output_projection());
                consecutive_low = 0;
            } else if metrics.avg_standing_reward < 0.5 {
                consecutive_low += 1;
                if self.config.enable_head_only_rollback
                    && !self.config.enable_recurrent_learning
                    && consecutive_low >= 3
                    && let Some((ref weights, ref bias)) = best_head_projection
                {
                    controller.set_output_projection(weights, bias);
                    consecutive_low = 0;
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
        // This snapshot covers only the output head. It must never be described
        // as a full-model rollback while recurrent parameters continue learning.
        let mut best_head_projection: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;

        for ep in 0..self.config.num_episodes {
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);
            if metrics.avg_episode_reward > best_reward {
                best_reward = metrics.avg_episode_reward;
                best_head_projection = Some(controller.output_projection());
                consecutive_low = 0;
            } else {
                consecutive_low += 1;
                if self.config.enable_head_only_rollback
                    && !self.config.enable_recurrent_learning
                    && consecutive_low >= 3
                    && best_reward > 0.5
                {
                    if let Some((ref w, ref b)) = best_head_projection {
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
        // This snapshot covers only the output head. It must never be described
        // as a full-model rollback while recurrent parameters continue learning.
        let mut best_head_projection: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;
        let dt = self.config.physics_dt();

        for ep in 0..self.config.num_episodes {
            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);
            if metrics.avg_episode_reward > best_reward {
                best_reward = metrics.avg_episode_reward;
                best_head_projection = Some(controller.output_projection());
                consecutive_low = 0;
            } else {
                consecutive_low += 1;
                if self.config.enable_head_only_rollback
                    && !self.config.enable_recurrent_learning
                    && consecutive_low >= 3
                    && best_reward > 0.5
                {
                    if let Some((ref w, ref b)) = best_head_projection {
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
            "episode,task,avg_standing_reward,avg_episode_reward,avg_free_energy,avg_head_height,avg_uprightness,avg_horizontal_speed,avg_control_effort,exploration_count,safety_interventions,total_steps\n",
        );

        let mut best_reward = f64::NEG_INFINITY;
        // This snapshot covers only the output head. It must never be described
        // as a full-model rollback while recurrent parameters continue learning.
        let mut best_head_projection: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;
        let mut prev_task: Option<HumanoidTask> = None;
        let mut transition_boost_remaining = 0u32;

        for ep in 0..self.config.num_episodes {
            let (current_task, _, _) = self.curriculum(ep);
            if let Some(prev) = prev_task
                && prev != current_task
            {
                transition_boost_remaining = 3;
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
                best_head_projection = Some(controller.output_projection());
                consecutive_low = 0;
            } else if metrics.avg_standing_reward < 0.5 {
                consecutive_low += 1;
                if self.config.enable_head_only_rollback
                    && !self.config.enable_recurrent_learning
                    && consecutive_low >= 3
                    && let Some((ref weights, ref bias)) = best_head_projection
                {
                    controller.set_output_projection(weights, bias);
                    consecutive_low = 0;
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
                "{},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
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
                metrics.safety_interventions,
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
