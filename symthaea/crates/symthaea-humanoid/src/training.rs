//! Humanoid training: multi-rate loop with PD baseline targets and curriculum.
//!
//! The trainer runs a multi-rate control loop:
//! - Physics at 40Hz: encode -> evolve -> project -> command
//! - Training at 20Hz: BPTT from PD baseline targets
//! - Cognitive tick at 10Hz: FEP agent modulates tau/LR/precision
//!
//! Curriculum:
//! 1. Episodes 0-50: Stand (PD teacher 80%, HDC-LTC 20%)
//! 2. Episodes 50-100: Stand (PD decays to 20%, HDC-LTC takes over)
//! 3. Episodes 100-150: Walk (target speed ramps 0->1 m/s)
//! 4. Episodes 150-200: Run (target speed ramps 1->10 m/s)

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::HumanoidController;
use crate::encoder::HumanoidHdcEncoder;
use crate::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
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
    /// Per-step telemetry (populated when collect_telemetry is true).
    pub telemetry: Vec<HumanoidTelemetry>,
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
        }
    }

    /// Determine the task and PD teacher weight for a given episode (curriculum).
    ///
    /// Returns (task, pd_weight, target_speed) with smooth transitions:
    /// - Phase 1 (0-40%):  Stand, PD 80%→40%
    /// - Phase 2 (40-60%): Stand, PD 40%→10% (autonomy ramp)
    /// - Phase 3 (60-95%): Walk, PD 10%→5%, speed 0→1 m/s (35 episodes)
    /// - Phase 4 (95-100%): Run, PD 5%→0%, speed 1→3 m/s (5 episodes)
    fn curriculum(&self, episode: usize) -> (HumanoidTask, f32, f64) {
        let total = self.config.num_episodes;
        if total <= 1 {
            return (self.config.task, 0.8, self.config.effective_target_speed());
        }

        let progress = episode as f64 / (total - 1) as f64;

        if progress < 0.4 {
            // Phase 1: Stand with PD guidance decaying 80%→40%
            let phase_progress = progress / 0.4;
            let pd_weight = 0.8 - 0.4 * phase_progress as f32;
            (HumanoidTask::Stand, pd_weight, 0.0)
        } else if progress < 0.6 {
            // Phase 2: Stand with PD decaying 40%→10% (autonomy ramp)
            let phase_progress = (progress - 0.4) / 0.2;
            let pd_weight = 0.4 - 0.3 * phase_progress as f32;
            (HumanoidTask::Stand, pd_weight, 0.0)
        } else if progress < 0.95 {
            // Phase 3: Walk, PD 10%→5%, speed ramps 0→1 m/s
            // Extended to 35% of training (was 25%) for longer Walk practice
            let phase_progress = (progress - 0.6) / 0.35;
            let pd_weight = 0.10 - 0.05 * phase_progress as f32;
            let target_speed = phase_progress * 1.0;
            (HumanoidTask::Walk, pd_weight, target_speed)
        } else {
            // Phase 4: Run, PD 5%→0%, speed ramps 1→3 m/s
            // Shortened to 5% (was 15%) with gentler speed cap (3 m/s vs 5 m/s)
            let phase_progress = (progress - 0.95) / 0.05;
            let pd_weight = 0.05 * (1.0 - phase_progress as f32);
            let target_speed = 1.0 + phase_progress * 2.0;
            (HumanoidTask::Run, pd_weight, target_speed)
        }
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
        let mut fep_result = fep_agent.step(physics.state(), &initial_cmd);

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

        // Gait phase for locomotion baselines (advances each physics step)
        // Walk: ~1.5 Hz gait cycle, Run: ~2.5 Hz gait cycle
        let gait_freq = match task {
            HumanoidTask::Walk => 1.5,
            HumanoidTask::Run => 2.5,
            HumanoidTask::Stand => 0.0,
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
                fep_result = fep_agent.step(physics.state(), &command);

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
        let mut physics = SimpleHumanoidSimulator::new();
        self.run_episode_with_sim(encoder, controller, fep_agent, &mut physics, episode)
    }

    /// Run the full training curriculum.
    pub fn train(&mut self) -> Vec<EpisodeMetrics> {
        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new(&genesis, num_levels);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), self.config.task);

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

            all_metrics.push(metrics.clone());
            fep_agent.reset();
        }

        self.metrics = all_metrics.clone();
        all_metrics
    }

    /// Run training with CSV telemetry output.
    pub fn train_with_telemetry(&mut self, output_dir: &str) -> Vec<EpisodeMetrics> {
        self.config.collect_telemetry = true;

        let genesis = self.genesis.clone();
        let num_levels = self.config.num_levels;
        let mut encoder = HumanoidHdcEncoder::new(&genesis, num_levels);
        let mut controller = self
            .initial_controller
            .take()
            .unwrap_or_else(|| HumanoidController::new(&genesis, &self.config));
        let mut fep_agent =
            ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), self.config.task);

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

            if !metrics.telemetry.is_empty() {
                let step_path = format!("{}/episode_{:04}.csv", output_dir, ep);
                let mut csv = String::from(
                    "step,time,head_height,uprightness,horizontal_speed,standing_reward,episode_reward,free_energy,tau_factor,learning_rate,control_effort\n",
                );
                for t in &metrics.telemetry {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
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
    fn test_curriculum_phases() {
        let config = HumanoidConfig {
            num_episodes: 100,
            steps_per_episode: 10,
            ..HumanoidConfig::default()
        };
        let trainer = HumanoidTrainer::new(config);

        // Phase 1: Stand with heavy PD (episode 0 = progress 0.0)
        let (task0, pd0, speed0) = trainer.curriculum(0);
        assert_eq!(task0, HumanoidTask::Stand);
        assert!((pd0 - 0.8).abs() < 0.01);
        assert!((speed0 - 0.0).abs() < 0.01);

        // Phase 2: Stand with decaying PD (episode 50 = progress 0.505)
        let (task50, pd50, _) = trainer.curriculum(50);
        assert_eq!(task50, HumanoidTask::Stand);
        assert!(pd50 < 0.4, "PD should be decaying: {pd50}");

        // Phase 3: Walk (episode 70 = progress 0.707)
        let (task70, _pd70, speed70) = trainer.curriculum(70);
        assert_eq!(task70, HumanoidTask::Walk);
        assert!(speed70 > 0.0 && speed70 < 1.0, "Walk speed ramping: {speed70}");

        // Phase 3 still: Walk extends to 95% (episode 90 = progress 0.909)
        let (task90, _pd90, speed90) = trainer.curriculum(90);
        assert_eq!(task90, HumanoidTask::Walk);
        assert!(speed90 > 0.5 && speed90 <= 1.0, "Walk speed near end: {speed90}");

        // Phase 4: Run (episode 96 = progress 0.969)
        let (task96, pd96, speed96) = trainer.curriculum(96);
        assert_eq!(task96, HumanoidTask::Run);
        assert!(pd96 < 0.05, "PD nearly zero in Run: {pd96}");
        assert!(speed96 > 1.0, "Run speed > 1 m/s: {speed96}");

        // Final episode: speed capped at 3 m/s
        let (_task99, _pd99, speed99) = trainer.curriculum(99);
        assert!(
            speed99 <= 3.01,
            "Run speed should be capped at ~3 m/s: {speed99}"
        );
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
