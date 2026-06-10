// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::AgribotController;
use crate::encoder::AgribotHdcEncoder;
use crate::fep_agent::ActiveInferenceAgribotAgent;
use crate::simulator::{AgribotPhysicsSimulator, SimpleAgribotSimulator};
use crate::types::AgribotConfig;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
}

pub struct AgribotTrainer {
    controller: AgribotController,
    encoder: AgribotHdcEncoder,
    fep: ActiveInferenceAgribotAgent,
    simulator: SimpleAgribotSimulator,
    config: AgribotConfig,
}

impl AgribotTrainer {
    pub fn new(config: AgribotConfig) -> Self {
        let g = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            controller: AgribotController::new(&g, &config),
            encoder: AgribotHdcEncoder::new(&g, 32),
            fep: ActiveInferenceAgribotAgent::new(),
            simulator: SimpleAgribotSimulator::new(),
            config,
        }
    }
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset();
        self.controller.reset();
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
                    diverged: true,
                };
            }
        }
        EpisodeMetrics {
            mean_effort: total_e / self.config.steps_per_episode as f32,
            steps_survived: self.config.steps_per_episode,
            diverged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_episode() {
        let mut c = AgribotConfig::default();
        c.steps_per_episode = 200;
        let mut t = AgribotTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.diverged);
        assert_eq!(m.steps_survived, 200);
    }
}
