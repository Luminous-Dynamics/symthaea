// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::ExoskeletonController;
use crate::encoder::ExoskeletonHdcEncoder;
use crate::fep_agent::ActiveInferenceExoAgent;
use crate::simulator::{ExoskeletonPhysicsSimulator, SimpleExoskeletonSimulator};
use crate::types::ExoskeletonConfig;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub fell: bool,
}

pub struct ExoskeletonTrainer {
    controller: ExoskeletonController,
    encoder: ExoskeletonHdcEncoder,
    fep: ActiveInferenceExoAgent,
    simulator: SimpleExoskeletonSimulator,
    config: ExoskeletonConfig,
}

impl ExoskeletonTrainer {
    pub fn new(config: ExoskeletonConfig) -> Self {
        let g = GenesisSeed::from_phrase("exo_train");
        Self {
            controller: ExoskeletonController::new(&g, &config),
            encoder: ExoskeletonHdcEncoder::new(&g, 32),
            fep: ActiveInferenceExoAgent::new(),
            simulator: SimpleExoskeletonSimulator::new(),
            config,
        }
    }
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        let dt = self.config.physics_dt();
        let mut total_e = 0.0f32;
        for step in 0..self.config.steps_per_episode {
            let hv = self.encoder.encode(self.simulator.state());
            if step % self.config.cognitive_interval == 0 {
                let _ = self.fep.tick(self.simulator.state());
            }
            let cmd = self.controller.forward(&hv, dt as f32);
            self.simulator.step(&cmd, dt);
            total_e += cmd.control_effort();
            if !self.simulator.state().is_finite() {
                return EpisodeMetrics {
                    mean_effort: total_e / (step + 1) as f32,
                    steps_survived: step,
                    fell: true,
                };
            }
        }
        let n = self.config.steps_per_episode;
        EpisodeMetrics {
            mean_effort: total_e / n as f32,
            steps_survived: n,
            fell: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_episode() {
        let mut c = ExoskeletonConfig::default();
        c.steps_per_episode = 200;
        let mut t = ExoskeletonTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.fell);
        assert_eq!(m.steps_survived, 200);
    }
}
