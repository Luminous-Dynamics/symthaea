// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Episode-based training for the manipulator arm.

use crate::controller::ManipulatorController;
use crate::encoder::ManipulatorHdcEncoder;
use crate::fep_agent::ActiveInferenceManipulatorAgent;
use crate::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use crate::types::ManipulatorConfig;
use symthaea_core::genesis::GenesisSeed;

/// Metrics from a single training episode.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
}

/// Episode-based trainer for the manipulator.
pub struct ManipulatorTrainer {
    controller: ManipulatorController,
    encoder: ManipulatorHdcEncoder,
    fep: ActiveInferenceManipulatorAgent,
    simulator: SimpleManipulatorSimulator,
    config: ManipulatorConfig,
}

impl ManipulatorTrainer {
    pub fn new(config: ManipulatorConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            controller: ManipulatorController::new(&genesis, &config),
            encoder: ManipulatorHdcEncoder::new(&genesis, 32),
            fep: ActiveInferenceManipulatorAgent::new(),
            simulator: SimpleManipulatorSimulator::new(),
            config,
        }
    }

    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset();
        self.controller.reset();
        let dt = self.config.physics_dt();
        let mut total_effort = 0.0f32;

        for step in 0..self.config.steps_per_episode {
            let hv = self.encoder.encode(self.simulator.state());
            if step % self.config.cognitive_interval == 0 {
                let _ = self.fep.tick(self.simulator.state());
            }
            let cmd = self.controller.forward(&hv, dt as f32);
            self.simulator.step(&cmd, dt);
            total_effort += cmd.control_effort();

            if !self.simulator.state().is_finite() {
                return EpisodeMetrics {
                    mean_effort: total_effort / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                };
            }
        }

        EpisodeMetrics {
            mean_effort: total_effort / self.config.steps_per_episode as f32,
            steps_survived: self.config.steps_per_episode,
            diverged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_completes() {
        let mut config = ManipulatorConfig::default();
        config.steps_per_episode = 200;
        let mut trainer = ManipulatorTrainer::new(config);
        let metrics = trainer.run_episode();
        assert!(!metrics.diverged);
        assert_eq!(metrics.steps_survived, 200);
        assert!(metrics.mean_effort.is_finite());
    }
}
