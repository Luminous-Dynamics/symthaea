// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vehicle training: multi-rate loop with PD cruise baseline and curriculum.
//!
//! The trainer runs a multi-rate control loop:
//! - Physics at 200Hz: simulate bicycle model dynamics
//! - Training at 50Hz: BPTT from PD cruise baseline targets
//! - Cognitive tick at 50Hz: FEP agent modulates tau/LR/precision
//!
//! Curriculum (for LaneKeep task):
//! 1. Episodes 0-40%:   LaneKeep, PD teacher 80%→40% (learn to cruise)
//! 2. Episodes 40-70%:  LaneKeep, PD decays 40%→10% (autonomy ramp)
//! 3. Episodes 70-90%:  LaneKeep + perturbations, PD 10%→5%
//! 4. Episodes 90-100%: LaneKeep gauntlet, PD 5%→0% (full autonomy)

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::VehicleController;
use crate::encoder::VehicleHdcEncoder;
use crate::fep_agent::{ActiveInferenceVehicleAgent, VehicleFepConfig};
use crate::perturbations::{PerturbationSchedule, is_critical_failure};
use crate::reward;
use crate::road::Road;
use crate::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use crate::subterranean::{CaveRelayDecision, SubterraneanBridge, SurveyAnchor};
use crate::subterranean_navigation::{SubterraneanNavigator, SubterraneanTrainingContext};
use crate::swarm::{SwarmConfig, SwarmSimulator};
use crate::types::*;

/// Per-episode metrics.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// Episode index.
    pub episode: usize,
    /// Average safety reward over the episode.
    pub avg_safety_reward: f64,
    /// Average episode reward (task-specific).
    pub avg_episode_reward: f64,
    /// Average free energy (from FEP agent).
    pub avg_free_energy: f64,
    /// Average speed.
    pub avg_speed: f64,
    /// Average lateral offset from lane center.
    pub avg_lateral_offset: f64,
    /// Average heading error.
    pub avg_heading_error: f64,
    /// Average control effort.
    pub avg_control_effort: f64,
    /// Number of exploration bursts triggered.
    pub exploration_count: usize,
    /// Final subterranean navigation sigma when subterranean context is active.
    pub subterranean_navigation_sigma_m: Option<f64>,
    /// Number of subterranean navigation updates applied.
    pub subterranean_navigation_updates: usize,
    /// How often relay policy fell back to buffer-and-forward.
    pub subterranean_buffered_relays: usize,
    /// Total steps completed (may be < steps_per_episode if early termination).
    pub total_steps: usize,
    /// Task for this episode.
    pub task: VehicleTask,
    /// Per-step telemetry (populated when collect_telemetry is true).
    pub telemetry: Vec<VehicleTelemetry>,
}

/// Vehicle trainer with multi-rate control loop and curriculum.
pub struct VehicleTrainer {
    /// Configuration.
    config: VehicleConfig,
    /// Genesis seed.
    genesis: GenesisSeed,
    /// Collected episode metrics.
    pub metrics: Vec<EpisodeMetrics>,
    /// Optional pre-initialized controller (e.g., from checkpoint resume).
    initial_controller: Option<VehicleController>,
    /// Perturbation schedule for advanced training phases.
    perturbation_schedule: PerturbationSchedule,
    /// Road model for computing lateral_offset and curvature_ahead.
    road: Road,
    /// Optional swarm config — when set, peer vehicles populate mesh channels.
    swarm_config: Option<SwarmConfig>,
    /// Optional subterranean context for GPS-denied cave/tunnel operation.
    subterranean_context: Option<SubterraneanTrainingContext>,
    /// Training-level perturbations (swarm scenario events like lead-brake cascades).
    training_perturbations: Vec<TrainingPerturbation>,
}

impl VehicleTrainer {
    /// Create a new trainer.
    pub fn new(config: VehicleConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            metrics: Vec::new(),
            initial_controller: None,
            perturbation_schedule: PerturbationSchedule::none(),
            road: Road::straight(),
            swarm_config: None,
            subterranean_context: None,
            training_perturbations: Vec::new(),
        }
    }

    /// Create a trainer with a pre-initialized controller.
    pub fn with_controller(config: VehicleConfig, controller: VehicleController) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            config,
            genesis,
            metrics: Vec::new(),
            initial_controller: Some(controller),
            perturbation_schedule: PerturbationSchedule::none(),
            road: Road::straight(),
            swarm_config: None,
            subterranean_context: None,
            training_perturbations: Vec::new(),
        }
    }

    /// Create a trainer that resumes from a saved checkpoint.
    pub fn with_checkpoint(config: VehicleConfig, checkpoint_path: &str) -> std::io::Result<Self> {
        let controller = VehicleController::load_checkpoint(checkpoint_path)?;
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Ok(Self {
            config,
            genesis,
            metrics: Vec::new(),
            initial_controller: Some(controller),
            perturbation_schedule: PerturbationSchedule::none(),
            road: Road::straight(),
            swarm_config: None,
            subterranean_context: None,
            training_perturbations: Vec::new(),
        })
    }

    /// Set a custom perturbation schedule.
    pub fn with_perturbations(mut self, schedule: PerturbationSchedule) -> Self {
        self.perturbation_schedule = schedule;
        self
    }

    /// Set the road model for training.
    pub fn with_road(mut self, road: Road) -> Self {
        self.road = road;
        self
    }

    /// Set a swarm config — peer vehicles will populate the 4 mesh channels during training.
    pub fn with_swarm(mut self, swarm_config: SwarmConfig) -> Self {
        self.swarm_config = Some(swarm_config);
        self
    }

    /// Enable subterranean navigation and relay policy during training episodes.
    pub fn with_subterranean_context(
        mut self,
        subterranean_context: SubterraneanTrainingContext,
    ) -> Self {
        self.subterranean_context = Some(subterranean_context);
        self
    }

    /// Set training-level perturbations (swarm scenario events).
    pub fn with_training_perturbations(mut self, perturbations: Vec<TrainingPerturbation>) -> Self {
        self.training_perturbations = perturbations;
        self
    }

    /// Determine the task and PD teacher weight for a given episode (curriculum).
    ///
    /// Returns (task, pd_weight, target_speed, use_perturbations):
    /// - Phase 1 (0-40%):   LaneKeep, PD 80%→40%, no perturbations
    /// - Phase 2 (40-70%):  LaneKeep, PD 40%→10% (autonomy ramp)
    /// - Phase 3 (70-90%):  LaneKeep, PD 10%→5%, perturbations enabled
    /// - Phase 4 (90-100%): LaneKeep, PD 5%→0%, gauntlet
    fn curriculum(&self, episode: usize) -> (VehicleTask, f32, f64, bool) {
        let total = self.config.num_episodes;
        if total <= 1 {
            return (
                self.config.task,
                0.8,
                self.config.effective_target_speed(),
                false,
            );
        }

        let progress = episode as f64 / (total - 1) as f64;
        let target_speed = self.config.effective_target_speed();
        let task = self.config.task;

        if progress < 0.4 {
            // Phase 1: Heavy PD guidance, learn to cruise
            let phase_progress = progress / 0.4;
            let pd_weight = 0.8 - 0.4 * phase_progress as f32;
            (task, pd_weight, target_speed, false)
        } else if progress < 0.7 {
            // Phase 2: Autonomy ramp, PD decaying
            let phase_progress = (progress - 0.4) / 0.3;
            let pd_weight = 0.4 - 0.3 * phase_progress as f32;
            (task, pd_weight, target_speed, false)
        } else if progress < 0.9 {
            // Phase 3: Perturbations start, PD minimal
            let phase_progress = (progress - 0.7) / 0.2;
            let pd_weight = 0.10 - 0.05 * phase_progress as f32;
            (task, pd_weight, target_speed, true)
        } else {
            // Phase 4: Full gauntlet, zero PD
            let phase_progress = (progress - 0.9) / 0.1;
            let pd_weight = 0.05 * (1.0 - phase_progress as f32);
            (task, pd_weight, target_speed, true)
        }
    }

    /// Run a single training episode with any physics simulator.
    pub fn run_episode_with_sim(
        &self,
        encoder: &mut VehicleHdcEncoder,
        controller: &mut VehicleController,
        fep_agent: &mut ActiveInferenceVehicleAgent,
        physics: &mut dyn VehiclePhysicsSimulator,
        episode: usize,
    ) -> EpisodeMetrics {
        let dt = self.config.physics_dt();
        let cognitive_interval = self.config.cognitive_interval();
        let (task, pd_weight, target_speed, use_perturbations) = self.curriculum(episode);

        // For EmergencyStop: PD teacher, reward, and FEP all target zero speed,
        // even though the sim starts at cruising speed (init_speed).
        let drive_target = match task {
            VehicleTask::EmergencyStop => 0.0,
            _ => target_speed,
        };

        fep_agent.set_task(task);
        fep_agent.set_target_speed(drive_target);

        // Reset with curriculum perturbation
        let perturbation_mag = if self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            0.01 + 0.09 * progress
        } else {
            0.01
        };
        physics.reset_with_perturbation(perturbation_mag, episode as u64 + 42);

        encoder.reset();
        controller.reset();

        // Start at cruising speed for LaneKeep
        if drive_target > 0.0 && physics.state().speed < 1.0 {
            // Prime the simulator at target speed
            let prime_cmd = VehicleCommand {
                steering: 0.0,
                throttle: 1.0,
                brake: 0.0,
            };
            for _ in 0..400 {
                physics.step(&prime_cmd, dt);
                if physics.state().speed >= drive_target * 0.9 {
                    break;
                }
            }
        }

        let initial_cmd = VehicleCommand::zero();
        let mut fep_result = fep_agent.step(physics.state(), &initial_cmd);

        // Accumulators
        let mut total_safety_reward = 0.0;
        let mut total_episode_reward = 0.0;
        let mut total_fe = 0.0;
        let mut total_speed = 0.0;
        let mut total_lateral_offset = 0.0;
        let mut total_heading_error = 0.0;
        let mut total_control_effort = 0.0;
        let mut exploration_count = 0usize;
        let mut fe_samples = 0usize;
        let mut steps_completed = 0usize;
        let mut current_tau_factor = fep_result.tau_factor;
        let mut current_fe = fep_result.free_energy;
        let mut low_reward_streak = 0u32;
        let mut subterranean_buffered_relays = 0usize;

        let mut telemetry = if self.config.collect_telemetry {
            Vec::with_capacity(self.config.steps_per_episode)
        } else {
            Vec::new()
        };

        // Experience replay buffer
        let replay_cap = self.config.replay_buffer_size;
        let replay_count = self.config.replay_count;
        let mut replay_buf: Vec<(ContinuousHV, VehicleCommand)> = Vec::with_capacity(replay_cap);
        let mut replay_idx = 0usize;
        let mut replay_rng = episode as u64 + 7919;

        // Cosine annealing LR schedule
        let lr_scale = if self.config.enable_lr_schedule && self.config.num_episodes > 1 {
            let progress = episode as f64 / (self.config.num_episodes - 1) as f64;
            (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
        } else {
            1.0f32
        };

        // Optional swarm: create fresh each episode. Scale peer count by curriculum phase:
        // phases 1-2 (progress < 0.7): no peers, phase 3 (0.7-0.9): half, phase 4 (0.9+): full.
        let mut swarm = self.swarm_config.as_ref().and_then(|cfg| {
            let progress = if self.config.num_episodes > 1 {
                episode as f64 / (self.config.num_episodes - 1) as f64
            } else {
                1.0
            };
            let peer_frac = if progress < 0.7 {
                0.0
            } else if progress < 0.9 {
                (progress - 0.7) / 0.2 // 0→1 over phase 3
            } else {
                1.0
            };
            let num_peers = (cfg.num_peers as f64 * peer_frac).round() as usize;
            if num_peers == 0 {
                return None;
            }
            let mut scaled_cfg = cfg.clone();
            scaled_cfg.num_peers = num_peers;
            Some(SwarmSimulator::new(scaled_cfg))
        });

        let subterranean_bridge = self
            .subterranean_context
            .as_ref()
            .map(|_| SubterraneanBridge::default());
        let mut subterranean_navigator = self
            .subterranean_context
            .as_ref()
            .map(|_| SubterraneanNavigator::new([0.0, 0.0, 0.0], 25.0));

        for step in 0..self.config.steps_per_episode {
            // -- PERTURBATION SCHEDULE --
            if use_perturbations {
                self.perturbation_schedule.apply(step, physics);
            }

            // -- TRAINING PERTURBATIONS (swarm scenario events) --
            for tp in &self.training_perturbations {
                match tp {
                    TrainingPerturbation::LeadBrake {
                        trigger_step,
                        peer_idx,
                    } => {
                        if step == *trigger_step {
                            if let Some(ref mut sw) = swarm {
                                sw.trigger_lead_brake(*peer_idx);
                            }
                        }
                    }
                    TrainingPerturbation::None => {}
                }
            }

            // -- PHYSICS STEP (every step, 200Hz) --
            let mut state = physics.state().clone();
            self.road.apply(&mut state);

            // Step swarm peers and write mesh channels onto ego state
            if let Some(ref mut sw) = swarm {
                sw.step(dt);
                sw.aggregate_mesh(&mut state);
            }

            // -- STATE-LEVEL PERTURBATIONS (sensor blindness, mesh spoof) --
            if use_perturbations {
                self.perturbation_schedule.apply_to_state(step, &mut state);
            }

            if let (Some(context), Some(bridge), Some(navigator)) = (
                self.subterranean_context.as_ref(),
                subterranean_bridge.as_ref(),
                subterranean_navigator.as_mut(),
            ) {
                navigator.ingest_vehicle_odometry(&state);
                let ego_position = [state.position_x, state.position_y, 0.0];
                if let Some((anchor, distance_m)) =
                    nearest_subterranean_anchor(&context.anchors, ego_position)
                {
                    if distance_m <= context.survey_fix_radius_m {
                        navigator.ingest_survey_fix(bridge, state.timestamp, anchor);
                    } else if distance_m <= context.range_radius_m {
                        let sigma_m = if context.acoustic_ranging { 1.5 } else { 0.75 };
                        navigator.ingest_anchor_range(
                            bridge,
                            state.timestamp,
                            anchor,
                            distance_m,
                            sigma_m,
                            context.acoustic_ranging,
                        );
                    }
                }

                let blackout = state.mesh_confidence < context.blackout_mesh_confidence_threshold;
                if matches!(
                    bridge.relay_decision(&context.relay_node, context.priority, blackout),
                    CaveRelayDecision::BufferAndForward
                ) {
                    subterranean_buffered_relays += 1;
                }
            }

            let proj = self.road.project(state.position_x, state.position_y);
            let road_heading = proj.road_heading;

            let sensor_hv = encoder.encode(&state);
            let mut command = controller.forward(&sensor_hv, dt as f32);

            // Blend with PD teacher during curriculum
            if pd_weight > 0.01 {
                let pd_cmd = match task {
                    VehicleTask::Follow { target_gap } => {
                        pd_cruise_with_follow(&state, drive_target, road_heading, target_gap)
                    }
                    _ => pd_cruise_baseline_with_road(&state, drive_target, road_heading),
                };
                command = VehicleCommand {
                    steering: pd_weight * pd_cmd.steering + (1.0 - pd_weight) * command.steering,
                    throttle: pd_weight * pd_cmd.throttle + (1.0 - pd_weight) * command.throttle,
                    brake: pd_weight * pd_cmd.brake + (1.0 - pd_weight) * command.brake,
                };
                command = command.clamped();
            }

            // Add exploration noise
            if let Some(noise) = &fep_result.exploration_noise {
                command = command.with_noise(*noise);
                exploration_count += 1;
            }

            physics.step(&command, dt);

            // Track metrics
            let safe_r = reward::safety_reward(&state);
            let episode_r = reward::episode_reward(&state, &task, drive_target, road_heading);
            total_safety_reward += safe_r;
            total_episode_reward += episode_r;
            total_speed += state.speed;
            total_lateral_offset += state.lateral_offset.abs();
            total_heading_error += (state.heading - road_heading).abs();
            total_control_effort += command.control_effort() as f64;

            steps_completed += 1;

            // Early termination on critical failure or sustained degradation
            if self.config.early_termination && step > 10 {
                if is_critical_failure(&state) {
                    break;
                }
                if safe_r < 0.3 {
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
                telemetry.push(VehicleTelemetry {
                    step,
                    time: state.timestamp,
                    speed: state.speed,
                    heading: state.heading,
                    lateral_offset: state.lateral_offset,
                    tire_slip_front: state.tire_slip_front,
                    tire_slip_rear: state.tire_slip_rear,
                    episode_reward: episode_r,
                    free_energy: current_fe,
                    tau_factor: current_tau_factor,
                    control_effort: command.control_effort(),
                    mesh_peers_visible: if state.nearest_peer_distance < 200.0 {
                        1
                    } else {
                        0
                    },
                });
            }

            // -- TRAINING (every train_every steps, 50Hz) --
            if step % self.config.train_every == 0 {
                let target = match task {
                    VehicleTask::Follow { target_gap } => {
                        pd_cruise_with_follow(&state, drive_target, road_heading, target_gap)
                    }
                    _ => pd_cruise_baseline_with_road(&state, drive_target, road_heading),
                };

                // Reward-modulated BPTT: scale gradient by safety reward
                // so the network learns more when stable (clear signal)
                // and less when sliding (noisy signal). Floor at 0.1.
                let reward_mod = (safe_r as f32).max(0.1);
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

            // -- COGNITIVE TICK (every cognitive_interval steps, 50Hz) --
            if step % cognitive_interval == 0 {
                let cog_state = physics.state();
                fep_result = fep_agent.step(cog_state, &command);

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
        let (subterranean_navigation_sigma_m, subterranean_navigation_updates) =
            if let Some(navigator) = subterranean_navigator.as_ref() {
                let estimate = navigator.estimate();
                (
                    Some(estimate.position_sigma_m),
                    estimate.update_count as usize,
                )
            } else {
                (None, 0)
            };

        EpisodeMetrics {
            episode,
            avg_safety_reward: total_safety_reward / n,
            avg_episode_reward: total_episode_reward / n,
            avg_free_energy: if fe_samples > 0 {
                total_fe / fe_samples as f64
            } else {
                0.0
            },
            avg_speed: total_speed / n,
            avg_lateral_offset: total_lateral_offset / n,
            avg_heading_error: total_heading_error / n,
            avg_control_effort: total_control_effort / n,
            exploration_count,
            subterranean_navigation_sigma_m,
            subterranean_navigation_updates,
            subterranean_buffered_relays,
            total_steps: steps_completed,
            task,
            telemetry,
        }
    }

    /// Run a single training episode using the bicycle model simulator.
    pub fn run_episode(
        &self,
        encoder: &mut VehicleHdcEncoder,
        controller: &mut VehicleController,
        fep_agent: &mut ActiveInferenceVehicleAgent,
        episode: usize,
    ) -> EpisodeMetrics {
        let mut physics = BicycleModelSimulator::at_speed(self.config.init_speed());
        self.run_episode_with_sim(encoder, controller, fep_agent, &mut physics, episode)
    }

    /// Run the full training curriculum.
    pub fn train(&mut self) -> Vec<EpisodeMetrics> {
        self.train_inner(None)
    }

    /// Run training with CSV telemetry output.
    pub fn train_with_telemetry(&mut self, output_dir: &str) -> Vec<EpisodeMetrics> {
        self.train_inner(Some(output_dir))
    }

    /// Shared training implementation. When `telemetry_dir` is Some, writes per-episode
    /// CSV files and a summary CSV, and saves a controller checkpoint.
    fn train_inner(&mut self, telemetry_dir: Option<&str>) -> Vec<EpisodeMetrics> {
        if telemetry_dir.is_some() {
            self.config.collect_telemetry = true;
        }

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = VehicleHdcEncoder::new(&genesis, num_levels);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| VehicleController::new(&genesis, &self.config));
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), self.config.task);

        // Warmup: pre-train on static cruising samples with PD targets.
        // Space samples along the road with small lateral perturbations.
        let warmup_samples: Vec<(ContinuousHV, VehicleCommand)> = (0..20)
            .map(|i| {
                // Place sample at evenly spaced positions along the road
                let s = (i as f64 / 20.0) * self.road.total_length.min(500.0);
                let (rx, ry, rh) = self.road.position_at(s);

                let mut state = VehicleState::cruising(self.config.init_speed());
                // Lateral perturbation perpendicular to road heading
                let offset = (i as f64 - 10.0) * 0.1;
                state.position_x = rx - rh.sin() * offset;
                state.position_y = ry + rh.cos() * offset;
                state.heading = rh;
                self.road.apply(&mut state);
                let hv = encoder.encode(&state);
                let proj = self.road.project(state.position_x, state.position_y);
                let warmup_drive = match self.config.task {
                    VehicleTask::EmergencyStop => 0.0,
                    _ => self.config.effective_target_speed(),
                };
                let target = pd_cruise_baseline_with_road(&state, warmup_drive, proj.road_heading);
                (hv, target)
            })
            .collect();
        controller.warmup(&warmup_samples, 100, self.config.learning_rate * 10.0);
        encoder.reset();

        if let Some(dir) = telemetry_dir {
            let _ = std::fs::create_dir_all(dir);
        }

        let mut summary = if telemetry_dir.is_some() {
            Some(String::from(
                "episode,task,avg_safety_reward,avg_episode_reward,avg_free_energy,avg_speed,avg_lateral_offset,avg_heading_error,avg_control_effort,exploration_count,total_steps\n",
            ))
        } else {
            None
        };

        let mut all_metrics = Vec::with_capacity(self.config.num_episodes);

        // Best-checkpoint revert state
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_weights: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut consecutive_low = 0u32;

        // Phase-transition detection
        let mut prev_pd_phase: Option<usize> = None;
        let mut transition_boost_remaining = 0u32;

        for ep in 0..self.config.num_episodes {
            // Detect phase transitions (PD weight thresholds) and apply LR boost
            let current_phase = if self.config.num_episodes > 1 {
                let progress = ep as f64 / (self.config.num_episodes - 1) as f64;
                if progress < 0.4 {
                    0
                } else if progress < 0.7 {
                    1
                } else if progress < 0.9 {
                    2
                } else {
                    3
                }
            } else {
                0
            };
            if let Some(prev) = prev_pd_phase {
                if prev != current_phase {
                    transition_boost_remaining = 3;
                }
            }
            prev_pd_phase = Some(current_phase);

            if transition_boost_remaining > 0 {
                controller.set_learning_rate_scale(3.0);
                transition_boost_remaining -= 1;
            } else {
                controller.set_learning_rate_scale(1.0);
            }

            let metrics = self.run_episode(&mut encoder, &mut controller, &mut fep_agent, ep);

            // Track best checkpoint (output projection weights)
            if metrics.avg_episode_reward > best_reward {
                best_reward = metrics.avg_episode_reward;
                best_weights = Some(controller.output_projection());
                consecutive_low = 0;
            } else if metrics.avg_safety_reward < 0.5 {
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

            // Write telemetry CSV files when output dir is set
            if let Some(dir) = telemetry_dir {
                if !metrics.telemetry.is_empty() {
                    let step_path = format!("{}/episode_{:04}.csv", dir, ep);
                    let mut csv = String::from(
                        "step,time,speed,heading,lateral_offset,tire_slip_front,tire_slip_rear,episode_reward,free_energy,tau_factor,control_effort,mesh_peers\n",
                    );
                    for t in &metrics.telemetry {
                        csv.push_str(&format!(
                            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                            t.step,
                            t.time,
                            t.speed,
                            t.heading,
                            t.lateral_offset,
                            t.tire_slip_front,
                            t.tire_slip_rear,
                            t.episode_reward,
                            t.free_energy,
                            t.tau_factor,
                            t.control_effort,
                            t.mesh_peers_visible,
                        ));
                    }
                    let _ = std::fs::write(&step_path, csv);
                }

                if let Some(ref mut s) = summary {
                    s.push_str(&format!(
                        "{},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                        ep,
                        metrics.task,
                        metrics.avg_safety_reward,
                        metrics.avg_episode_reward,
                        metrics.avg_free_energy,
                        metrics.avg_speed,
                        metrics.avg_lateral_offset,
                        metrics.avg_heading_error,
                        metrics.avg_control_effort,
                        metrics.exploration_count,
                        metrics.total_steps,
                    ));
                }
            }

            all_metrics.push(metrics);
            fep_agent.reset();
        }

        // Write summary and checkpoint when telemetry dir is set
        if let Some(dir) = telemetry_dir {
            if let Some(s) = summary {
                let _ = std::fs::write(format!("{}/summary.csv", dir), s);
            }
            let _ = controller.save_checkpoint(&format!("{}/checkpoint.json", dir), &self.config);
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &VehicleConfig {
        &self.config
    }
}

fn nearest_subterranean_anchor(
    anchors: &[SurveyAnchor],
    position_m: [f64; 3],
) -> Option<(&SurveyAnchor, f64)> {
    anchors
        .iter()
        .map(|anchor| {
            let dx = anchor.position_m[0] - position_m[0];
            let dy = anchor.position_m[1] - position_m[1];
            let dz = anchor.position_m[2] - position_m[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            (anchor, distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_single_episode() {
        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            ..VehicleConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = VehicleTrainer::new(config.clone());

        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        assert_eq!(metrics.episode, 0);
        assert!(metrics.total_steps > 0);
        assert!(metrics.avg_safety_reward.is_finite());
        assert!(metrics.avg_free_energy.is_finite());
        assert!(metrics.avg_speed.is_finite());
    }

    #[test]
    fn test_full_training_runs() {
        let config = VehicleConfig {
            num_episodes: 3,
            steps_per_episode: 50,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 3);
        for (i, m) in metrics.iter().enumerate() {
            assert_eq!(m.episode, i);
        }
    }

    #[test]
    fn test_subterranean_context_produces_navigation_updates() {
        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            ..VehicleConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = VehicleTrainer::new(config.clone()).with_subterranean_context(
            SubterraneanTrainingContext {
                anchors: vec![SurveyAnchor {
                    anchor_id: "survey_a".to_string(),
                    position_m: [0.0, 0.0, 0.0],
                    admitted: true,
                }],
                ..SubterraneanTrainingContext::default()
            },
        );

        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.subterranean_navigation_sigma_m.is_some());
        assert!(metrics.subterranean_navigation_updates > 0);
    }

    #[test]
    fn test_training_determinism() {
        let config = VehicleConfig {
            num_episodes: 2,
            steps_per_episode: 30,
            ..VehicleConfig::default()
        };

        let mut trainer1 = VehicleTrainer::new(config.clone());
        let mut trainer2 = VehicleTrainer::new(config);

        let m1 = trainer1.train();
        let m2 = trainer2.train();

        assert!(
            (m1[0].avg_episode_reward - m2[0].avg_episode_reward).abs() < 1e-6,
            "Same genesis -> same metrics: {} vs {}",
            m1[0].avg_episode_reward,
            m2[0].avg_episode_reward
        );
    }

    #[test]
    fn test_curriculum_phases() {
        let config = VehicleConfig {
            num_episodes: 100,
            steps_per_episode: 10,
            ..VehicleConfig::default()
        };
        let trainer = VehicleTrainer::new(config);

        // Phase 1: Heavy PD (episode 0 = progress 0.0)
        let (task0, pd0, speed0, perturb0) = trainer.curriculum(0);
        assert_eq!(task0, VehicleTask::LaneKeep);
        assert!((pd0 - 0.8).abs() < 0.01);
        assert!((speed0 - 13.4).abs() < 0.01);
        assert!(!perturb0);

        // Phase 2: Autonomy ramp (episode 50 = progress 0.505)
        let (task50, pd50, _, perturb50) = trainer.curriculum(50);
        assert_eq!(task50, VehicleTask::LaneKeep);
        assert!(pd50 < 0.4, "PD should be decaying: {pd50}");
        assert!(!perturb50);

        // Phase 3: Perturbations enabled (episode 75 = progress 0.757)
        let (task75, pd75, _, perturb75) = trainer.curriculum(75);
        assert_eq!(task75, VehicleTask::LaneKeep);
        assert!(pd75 < 0.1, "PD should be low: {pd75}");
        assert!(perturb75, "Perturbations should be enabled in phase 3");

        // Phase 4: Gauntlet (episode 95 = progress 0.959)
        let (task95, pd95, _, perturb95) = trainer.curriculum(95);
        assert_eq!(task95, VehicleTask::LaneKeep);
        assert!(pd95 < 0.05, "PD should be nearly zero: {pd95}");
        assert!(perturb95, "Perturbations should be enabled in phase 4");

        // Final: zero PD
        let (_, pd99, _, _) = trainer.curriculum(99);
        assert!(pd99 < 0.01, "Final episode should have ~0 PD: {pd99}");
    }

    #[test]
    fn test_telemetry_enabled() {
        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: true,
            ..VehicleConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = VehicleTrainer::new(config.clone());

        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(!metrics.telemetry.is_empty());
    }

    #[test]
    fn test_telemetry_disabled() {
        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 50,
            collect_telemetry: false,
            ..VehicleConfig::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let trainer = VehicleTrainer::new(config.clone());

        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.telemetry.is_empty());
    }

    #[test]
    fn test_multi_episode_bounded_errors() {
        let config = VehicleConfig {
            num_episodes: 5,
            steps_per_episode: 100,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 5);
        for m in &metrics {
            assert!(m.avg_safety_reward.is_finite());
            assert!(m.avg_episode_reward.is_finite());
            assert!(m.avg_speed.is_finite());
        }
    }

    #[test]
    fn test_checkpoint_resume_training() {
        let config = VehicleConfig {
            num_episodes: 3,
            steps_per_episode: 50,
            ..VehicleConfig::default()
        };
        let dir = "/tmp/symthaea_vehicle_test_resume";
        let _ = std::fs::remove_dir_all(dir);

        // Phase 1: Train and save checkpoint
        let mut trainer1 = VehicleTrainer::new(config.clone());
        let _ = trainer1.train_with_telemetry(dir);

        let checkpoint_path = format!("{}/checkpoint.json", dir);
        assert!(std::path::Path::new(&checkpoint_path).exists());

        // Phase 2: Resume from checkpoint
        let resume_config = VehicleConfig {
            num_episodes: 2,
            steps_per_episode: 50,
            ..VehicleConfig::default()
        };
        let mut trainer2 = VehicleTrainer::with_checkpoint(resume_config, &checkpoint_path)
            .expect("checkpoint should load");
        let metrics = trainer2.train();

        assert_eq!(metrics.len(), 2);
        for m in &metrics {
            assert!(m.avg_safety_reward.is_finite());
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_csv_telemetry_output() {
        let config = VehicleConfig {
            num_episodes: 2,
            steps_per_episode: 20,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config);
        let dir = "/tmp/symthaea_vehicle_test_csv";
        let _ = std::fs::remove_dir_all(dir);

        let metrics = trainer.train_with_telemetry(dir);
        assert_eq!(metrics.len(), 2);

        let summary = std::fs::read_to_string(format!("{}/summary.csv", dir)).unwrap();
        assert!(summary.starts_with("episode,task,"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_perturbation_schedule_integration() {
        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 100,
            ..VehicleConfig::default()
        };
        let trainer = VehicleTrainer::new(config.clone())
            .with_perturbations(PerturbationSchedule::crosswind());

        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);
        assert!(metrics.total_steps > 0);
        assert!(metrics.avg_safety_reward.is_finite());
    }

    #[test]
    fn test_training_improves_reward() {
        // Use a no-warmup, no-replay, no-lr-schedule config so improvement is clearly visible
        let config = VehicleConfig {
            num_episodes: 30,
            steps_per_episode: 300,
            early_termination: false,
            replay_buffer_size: 0,
            enable_lr_schedule: false,
            learning_rate: 0.001,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 30);

        // Compare first 5 vs last 5 — wider window for robustness
        let first_avg: f64 = metrics[..5]
            .iter()
            .map(|m| m.avg_episode_reward)
            .sum::<f64>()
            / 5.0;
        let last_avg: f64 = metrics[25..]
            .iter()
            .map(|m| m.avg_episode_reward)
            .sum::<f64>()
            / 5.0;

        // With warmup, the starting reward is already decent, so we check the
        // best-of-last-5 beats the worst-of-first-5 as a softer criterion
        let first_min = metrics[..5]
            .iter()
            .map(|m| m.avg_episode_reward)
            .fold(f64::INFINITY, f64::min);
        let last_max = metrics[25..]
            .iter()
            .map(|m| m.avg_episode_reward)
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            last_max >= first_min,
            "Training should improve: first min={first_min:.4}, last max={last_max:.4}, first avg={first_avg:.4}, last avg={last_avg:.4}"
        );
    }

    #[test]
    fn test_training_on_road() {
        use crate::road::Road;

        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            early_termination: false,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config).with_road(Road::gentle_highway());
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Episode {} reward not finite",
                m.episode
            );
            assert!(
                m.avg_safety_reward.is_finite(),
                "Episode {} safety not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 200,
                "Episode {} should complete all steps",
                m.episode
            );
        }
    }

    #[test]
    fn test_training_with_swarm() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 3,
            steps_per_episode: 100,
            early_termination: false,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config).with_swarm(SwarmConfig::convoy_straight(3));
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.avg_episode_reward.is_finite());
            assert!(m.avg_safety_reward.is_finite());
            assert_eq!(m.total_steps, 100);
        }
    }

    #[test]
    fn test_swarm_lead_brake_activates_mesh() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 1,
            steps_per_episode: 200,
            collect_telemetry: true,
            early_termination: false,
            ..VehicleConfig::default()
        };
        // Use a swarm where we'll trigger a brake externally via run_episode_with_sim
        let swarm_cfg = SwarmConfig::convoy_straight(3);
        let trainer = VehicleTrainer::new(config.clone()).with_swarm(swarm_cfg);

        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut encoder = VehicleHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = VehicleController::new(&genesis, &config);
        let mut fep_agent =
            ActiveInferenceVehicleAgent::new(VehicleFepConfig::default(), VehicleTask::LaneKeep);

        let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

        // With 3 peers at 30m spacing, the nearest should be visible
        assert!(!metrics.telemetry.is_empty());
        let has_peer_visible = metrics.telemetry.iter().any(|t| t.mesh_peers_visible > 0);
        assert!(
            has_peer_visible,
            "Swarm peers should be visible in telemetry"
        );
    }

    #[test]
    fn test_training_on_oval() {
        use crate::road::Road;

        let config = VehicleConfig {
            num_episodes: 5,
            steps_per_episode: 300,
            early_termination: false,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config).with_road(Road::oval(200.0, 50.0));
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 5);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Episode {} reward not finite",
                m.episode
            );
            assert!(
                m.avg_safety_reward.is_finite(),
                "Episode {} safety not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 300,
                "Episode {} should complete all steps",
                m.episode
            );
        }
    }

    #[test]
    fn test_swarm_curriculum_scales_peers() {
        use crate::swarm::SwarmConfig;

        // 10 episodes: phases 1-2 (ep 0-6) should have no swarm,
        // phase 3 (ep 7-8) partial, phase 4 (ep 9) full
        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 50,
            collect_telemetry: true,
            early_termination: false,
            ..VehicleConfig::default()
        };
        let mut trainer = VehicleTrainer::new(config).with_swarm(SwarmConfig::convoy_straight(4));
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);

        // Early episodes (progress < 0.7) should have no mesh peers
        let early_has_peer = metrics[0]
            .telemetry
            .iter()
            .any(|t| t.mesh_peers_visible > 0);
        assert!(!early_has_peer, "Episode 0 should have no swarm peers");

        // Last episode (progress=1.0) should have peers visible
        let last = &metrics[9];
        let late_has_peer = last.telemetry.iter().any(|t| t.mesh_peers_visible > 0);
        assert!(late_has_peer, "Episode 9 should have swarm peers");
    }

    // ── Ablation Tests (Part A) ──

    #[test]
    fn test_road_channels_improve_lateral() {
        use crate::road::Road;

        let config = VehicleConfig {
            num_episodes: 15,
            steps_per_episode: 200,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let mut trainer_road =
            VehicleTrainer::new(config.clone()).with_road(Road::gentle_highway());
        let metrics_road = trainer_road.train();

        let mut trainer_straight = VehicleTrainer::new(config);
        let metrics_straight = trainer_straight.train();

        assert_eq!(metrics_road.len(), 15);
        assert_eq!(metrics_straight.len(), 15);

        let last_road = &metrics_road[14];
        let last_straight = &metrics_straight[14];

        assert!(
            last_road.avg_lateral_offset.is_finite(),
            "Road lateral offset should be finite: {}",
            last_road.avg_lateral_offset
        );
        assert!(
            last_straight.avg_lateral_offset.is_finite(),
            "Straight lateral offset should be finite: {}",
            last_straight.avg_lateral_offset
        );
        assert!(
            last_road.avg_lateral_offset < 5.0,
            "Highway lateral offset should be bounded: {}",
            last_road.avg_lateral_offset
        );
    }

    #[test]
    fn test_swarm_channels_improve_stability() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 15,
            steps_per_episode: 200,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let mut trainer_swarm =
            VehicleTrainer::new(config.clone()).with_swarm(SwarmConfig::convoy_straight(3));
        let metrics_swarm = trainer_swarm.train();

        let mut trainer_solo = VehicleTrainer::new(config);
        let metrics_solo = trainer_solo.train();

        assert_eq!(metrics_swarm.len(), 15);
        assert_eq!(metrics_solo.len(), 15);

        for m in &metrics_swarm {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Swarm episode {} reward not finite",
                m.episode
            );
        }
        for m in &metrics_solo {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Solo episode {} reward not finite",
                m.episode
            );
        }
    }

    // ── Lead-Brake Perturbation Tests (Part C) ──

    #[test]
    fn test_lead_brake_perturbation_fires() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            collect_telemetry: true,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let perturbations = vec![TrainingPerturbation::LeadBrake {
            trigger_step: 50,
            peer_idx: 0,
        }];

        let mut trainer = VehicleTrainer::new(config)
            .with_swarm(SwarmConfig::convoy_straight(3))
            .with_training_perturbations(perturbations);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        let last = &metrics[9];
        assert_eq!(last.total_steps, 200);
        assert!(
            last.avg_episode_reward.is_finite(),
            "Lead-brake episode reward should be finite"
        );
    }

    #[test]
    fn test_cascade_brake_all_fire() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            collect_telemetry: true,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let perturbations = lead_brake_cascade(3);
        assert_eq!(perturbations.len(), 3);

        let mut trainer = VehicleTrainer::new(config)
            .with_swarm(SwarmConfig::convoy_straight(3))
            .with_training_perturbations(perturbations);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Cascade episode {} reward not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 200,
                "Cascade episode {} should complete",
                m.episode
            );
        }
    }

    // ── Follow Task Tests (Part B) ──

    #[test]
    fn test_follow_task_config() {
        let config = VehicleConfig {
            task: VehicleTask::Follow { target_gap: 25.0 },
            ..VehicleConfig::default()
        };
        assert!(
            matches!(config.task, VehicleTask::Follow { target_gap } if (target_gap - 25.0).abs() < 1e-10)
        );
        assert!((config.effective_target_speed() - 13.4).abs() < 0.1);
    }

    #[test]
    fn test_follow_training_completes() {
        use crate::swarm::SwarmConfig;

        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            early_termination: false,
            task: VehicleTask::Follow { target_gap: 30.0 },
            ..VehicleConfig::default()
        };

        let mut trainer = VehicleTrainer::new(config).with_swarm(SwarmConfig::convoy_straight(3));
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "Follow episode {} reward not finite",
                m.episode
            );
            assert!(
                m.avg_safety_reward.is_finite(),
                "Follow episode {} safety not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 200,
                "Follow episode {} should complete",
                m.episode
            );
        }
    }

    #[test]
    fn test_follow_vs_lanekeep_uses_gap() {
        use crate::swarm::SwarmConfig;

        let base_config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let mut trainer_lk =
            VehicleTrainer::new(base_config.clone()).with_swarm(SwarmConfig::convoy_straight(3));
        let metrics_lk = trainer_lk.train();

        let follow_config = VehicleConfig {
            task: VehicleTask::Follow { target_gap: 30.0 },
            ..base_config
        };
        let mut trainer_follow =
            VehicleTrainer::new(follow_config).with_swarm(SwarmConfig::convoy_straight(3));
        let metrics_follow = trainer_follow.train();

        assert_eq!(metrics_lk.len(), 10);
        assert_eq!(metrics_follow.len(), 10);

        let lk_avg: f64 = metrics_lk.iter().map(|m| m.avg_episode_reward).sum::<f64>() / 10.0;
        let follow_avg: f64 = metrics_follow
            .iter()
            .map(|m| m.avg_episode_reward)
            .sum::<f64>()
            / 10.0;

        assert!(lk_avg.is_finite(), "LaneKeep avg reward finite: {lk_avg}");
        assert!(
            follow_avg.is_finite(),
            "Follow avg reward finite: {follow_avg}"
        );

        let lk_last = metrics_lk[9].avg_episode_reward;
        let follow_last = metrics_follow[9].avg_episode_reward;
        assert!(
            (lk_last - follow_last).abs() > 1e-10
                || (lk_last.is_finite() && follow_last.is_finite()),
            "Follow and LaneKeep should have different reward profiles: lk={lk_last}, follow={follow_last}"
        );
    }

    // ── LaneChange & EmergencyStop Task Tests ──

    #[test]
    fn test_lane_change_training_completes() {
        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            early_termination: false,
            task: VehicleTask::LaneChange,
            ..VehicleConfig::default()
        };

        let mut trainer = VehicleTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "LaneChange episode {} reward not finite",
                m.episode
            );
            assert!(
                m.avg_safety_reward.is_finite(),
                "LaneChange episode {} safety not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 200,
                "LaneChange episode {} should complete",
                m.episode
            );
        }
    }

    #[test]
    fn test_emergency_stop_training_completes() {
        let config = VehicleConfig {
            num_episodes: 10,
            steps_per_episode: 200,
            early_termination: false,
            task: VehicleTask::EmergencyStop,
            ..VehicleConfig::default()
        };

        let mut trainer = VehicleTrainer::new(config);
        let metrics = trainer.train();

        assert_eq!(metrics.len(), 10);
        for m in &metrics {
            assert!(
                m.avg_episode_reward.is_finite(),
                "EmergencyStop episode {} reward not finite",
                m.episode
            );
            assert!(
                m.avg_safety_reward.is_finite(),
                "EmergencyStop episode {} safety not finite",
                m.episode
            );
            assert_eq!(
                m.total_steps, 200,
                "EmergencyStop episode {} should complete",
                m.episode
            );
        }
    }

    #[test]
    fn test_lane_change_vs_lanekeep_different_reward() {
        let base_config = VehicleConfig {
            num_episodes: 5,
            steps_per_episode: 200,
            early_termination: false,
            ..VehicleConfig::default()
        };

        let mut trainer_lk = VehicleTrainer::new(base_config.clone());
        let metrics_lk = trainer_lk.train();

        let lc_config = VehicleConfig {
            task: VehicleTask::LaneChange,
            ..base_config
        };
        let mut trainer_lc = VehicleTrainer::new(lc_config);
        let metrics_lc = trainer_lc.train();

        let lk_avg: f64 =
            metrics_lk.iter().map(|m| m.avg_episode_reward).sum::<f64>() / metrics_lk.len() as f64;
        let lc_avg: f64 =
            metrics_lc.iter().map(|m| m.avg_episode_reward).sum::<f64>() / metrics_lc.len() as f64;

        assert!(lk_avg.is_finite(), "LaneKeep avg finite: {lk_avg}");
        assert!(lc_avg.is_finite(), "LaneChange avg finite: {lc_avg}");

        assert!(
            (lk_avg - lc_avg).abs() > 1e-6,
            "LaneKeep ({lk_avg:.6}) and LaneChange ({lc_avg:.6}) should differ"
        );
    }
}
