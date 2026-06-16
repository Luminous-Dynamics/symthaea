// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-rate training loop for SAR helicopter.
//!
//! 300Hz motor reflex + 20Hz cognitive tick (every 15 physics steps).

use crate::controller::HelicopterController;
use crate::encoder::HelicopterHdcEncoder;
use crate::fep_agent::ActiveInferenceHelicopterAgent;
use crate::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
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

    /// Run one training episode. Returns metrics.
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        let dt = self.config.physics_dt();
        let target_alt = self.task.target_altitude();
        let mut metrics = EpisodeMetrics::default();
        let mut altitude_error_sum = 0.0;
        let mut effort_sum = 0.0f32;
        let mut angular_sum = 0.0;

        self.simulator.reset(target_alt);
        self.controller.reset();
        self.encoder.reset();
        self.fep_agent.reset();

        for step in 0..self.config.steps_per_episode {
            // Encode current state
            let hv = self.encoder.encode(self.simulator.state());

            // Cognitive tick (every N physics steps)
            if step % self.config.cognitive_interval == 0 {
                let _fep_result = self.fep_agent.tick(self.simulator.state(), target_alt);
            }

            // Controller forward pass
            let cmd = self.controller.forward(&hv, dt as f32);

            // Physics step
            self.simulator.step(&cmd, dt);

            // Track metrics
            let state = self.simulator.state();
            altitude_error_sum += (state.altitude() - target_alt).abs();
            effort_sum += cmd.control_effort();
            angular_sum += state.angular_speed();
            metrics.steps_survived = step + 1;

            // Early termination: crashed or flipped
            if state.altitude() <= 0.01 && step > 10 {
                metrics.fell = true;
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
    }

    #[test]
    fn test_train_multiple_episodes() {
        let mut config = HelicopterConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = HelicopterTrainer::new(config, HelicopterTask::Hover);
        let results = trainer.train(3);
        assert_eq!(results.len(), 3);
    }
}
