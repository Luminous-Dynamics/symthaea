// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-rate training loop for SAR helicopter.
//!
//! 300Hz motor reflex + 20Hz cognitive tick (every 15 physics steps).

use crate::controller::{HelicopterController, pd_hover_baseline};
use crate::encoder::HelicopterHdcEncoder;
use crate::fep_agent::ActiveInferenceHelicopterAgent;
use crate::simulator::{HelicopterPhysicsSimulator, LandingOutcome, SimpleHelicopterSimulator};
use crate::types::{HelicopterConfig, HelicopterTask};

/// Episode metrics for tracking training progress.
#[derive(Debug, Clone, Default)]
pub struct EpisodeMetrics {
    pub mean_altitude_error: f64,
    pub mean_control_effort: f32,
    pub mean_angular_speed: f64,
    pub final_altitude: f64,
    pub steps_survived: usize,
    pub fell: bool,
    /// Ground-contact outcome, preserving safe vs hard vs crash evidence.
    pub landing_outcome: LandingOutcome,
    /// Mean pre-update imitation MSE against the PD hover baseline —
    /// the learning signal. Decreasing across episodes = the controller is
    /// actually learning.
    pub mean_imitation_loss: f32,
    /// Mean variational free energy from the FEP agent over the episode.
    pub mean_free_energy: f64,
}

/// SAR helicopter trainer.
pub struct HelicopterTrainer {
    controller: HelicopterController,
    encoder: HelicopterHdcEncoder,
    fep_agent: ActiveInferenceHelicopterAgent,
    simulator: SimpleHelicopterSimulator,
    config: HelicopterConfig,
    task: HelicopterTask,
}

impl HelicopterTrainer {
    /// Create a new trainer.
    pub fn new(config: HelicopterConfig, task: HelicopterTask) -> Self {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(&config.genesis_phrase);
        let controller = HelicopterController::new(&genesis, &config);
        let encoder = HelicopterHdcEncoder::new(&genesis, 32);
        let fep_agent = ActiveInferenceHelicopterAgent::new();
        let simulator = SimpleHelicopterSimulator::new();

        Self {
            controller,
            encoder,
            fep_agent,
            simulator,
            config,
            task,
        }
    }

    /// Run one training episode: imitation learning toward the PD hover
    /// baseline (`pd_hover_baseline`), with FEP τ / learning-rate modulation.
    ///
    /// Previously this ran a rollout, collected metrics, and never updated a
    /// single weight ("HelicopterTrainer" trained nothing) while the FEP tick
    /// result was discarded (`let _fep_result = ...`). Both loops are now
    /// closed, mirroring symthaea-quadruped's trainer.
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        let dt = self.config.physics_dt();
        let target_alt = self.task.target_altitude();
        let mut metrics = EpisodeMetrics::default();
        let mut altitude_error_sum = 0.0;
        let mut effort_sum = 0.0f32;
        let mut angular_sum = 0.0;
        let mut loss_sum = 0.0f32;
        let mut fe_sum = 0.0f64;
        let mut fe_count = 0usize;
        let mut tau_factor = 1.0f32;

        self.simulator.reset(target_alt);
        self.controller.reset();
        self.encoder.reset();
        self.fep_agent.reset();
        self.controller.set_learning_rate(self.config.learning_rate);

        for step in 0..self.config.steps_per_episode {
            // Encode current state
            let hv = self.encoder.encode_with_dt(self.simulator.state(), dt);

            // Cognitive tick (every N physics steps). FEP loop closed:
            // τ modulates the controller's effective timestep, the LR factor
            // scales the imitation learning rate, free energy is tracked.
            if step % self.config.cognitive_interval == 0 {
                let fep = self.fep_agent.tick(self.simulator.state(), target_alt);
                tau_factor = fep.tau_factor;
                self.controller
                    .set_learning_rate(self.config.learning_rate * fep.learning_rate_factor);
                fe_sum += fep.free_energy;
                fe_count += 1;
            }

            // Controller forward pass (τ-modulated timestep)
            let cmd = self.controller.forward(&hv, dt as f32 * tau_factor);

            // Learning loop closed: supervised update toward the PD
            // hover/attitude baseline for the state the controller just saw.
            let target_cmd = pd_hover_baseline(self.simulator.state(), target_alt);
            loss_sum += self.controller.train_step(&target_cmd);

            // Physics step
            self.simulator.step(&cmd, dt);

            // Track metrics
            let state = self.simulator.state();
            altitude_error_sum += (state.altitude() - target_alt).abs();
            effort_sum += cmd.control_effort();
            angular_sum += state.angular_speed();
            metrics.steps_survived = step + 1;

            // Early termination after any landing; preserve its classification.
            let landing = self.simulator.landing_contact().outcome;
            if !matches!(landing, LandingOutcome::Airborne) {
                metrics.landing_outcome = landing;
                metrics.fell = matches!(landing, LandingOutcome::Crash);
                break;
            }
            if !state.is_finite() {
                metrics.fell = true;
                break;
            }
        }

        let n = metrics.steps_survived as f64;
        metrics.mean_altitude_error = altitude_error_sum / n;
        metrics.mean_control_effort = effort_sum / n as f32;
        metrics.mean_angular_speed = angular_sum / n;
        metrics.final_altitude = self.simulator.state().altitude();
        metrics.mean_imitation_loss = loss_sum / n as f32;
        metrics.mean_free_energy = fe_sum / fe_count.max(1) as f64;

        metrics
    }

    /// Train for N episodes. Returns per-episode metrics.
    pub fn train(&mut self, num_episodes: usize) -> Vec<EpisodeMetrics> {
        (0..num_episodes).map(|_| self.run_episode()).collect()
    }

    /// Access the controller (for checkpointing).
    pub fn controller(&self) -> &HelicopterController {
        &self.controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_runs_to_completion() {
        let mut config = HelicopterConfig::default();
        config.steps_per_episode = 300; // Short episode for testing
        let mut trainer = HelicopterTrainer::new(config, HelicopterTask::Hover);
        let metrics = trainer.run_episode();
        assert!(metrics.steps_survived > 0);
        assert!(metrics.mean_altitude_error.is_finite());
        assert!(
            metrics.mean_imitation_loss > 0.0,
            "learning signal must be live"
        );
        assert!(metrics.mean_free_energy.is_finite());
    }

    #[test]
    fn test_train_multiple_episodes() {
        let mut config = HelicopterConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = HelicopterTrainer::new(config, HelicopterTask::Hover);
        let results = trainer.train(3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        // The trainer must actually LEARN: imitation loss against the PD
        // hover baseline decreases across episodes. This test fails against
        // the old trainer, which never updated a weight.
        let mut config = HelicopterConfig::default();
        config.steps_per_episode = 400;
        config.learning_rate = 0.005; // Faster convergence for a short test
        let mut trainer = HelicopterTrainer::new(config, HelicopterTask::Hover);
        let first = trainer.run_episode().mean_imitation_loss;
        for _ in 0..3 {
            trainer.run_episode();
        }
        let last = trainer.run_episode().mean_imitation_loss;
        assert!(
            last < first,
            "imitation loss must decrease with training: first {first:.6} -> last {last:.6}"
        );
    }
}
