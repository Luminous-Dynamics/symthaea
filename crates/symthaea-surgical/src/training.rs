// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::SurgicalController;
use crate::encoder::SurgicalHdcEncoder;
use crate::fep_agent::ActiveInferenceSurgicalAgent;
use crate::simulator::{SimpleSurgicalSimulator, SurgicalPhysicsSimulator};
use crate::types::SurgicalConfig;
use symthaea_core::genesis::GenesisSeed;
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_force: f64,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
}
pub struct SurgicalTrainer {
    ctrl: SurgicalController,
    enc: SurgicalHdcEncoder,
    fep: ActiveInferenceSurgicalAgent,
    sim: SimpleSurgicalSimulator,
    cfg: SurgicalConfig,
}
impl SurgicalTrainer {
    pub fn new(c: SurgicalConfig) -> Self {
        let g = GenesisSeed::from_phrase("surg_train");
        Self {
            ctrl: SurgicalController::new(&g, &c),
            enc: SurgicalHdcEncoder::new(&g, 32),
            fep: ActiveInferenceSurgicalAgent::new(),
            sim: SimpleSurgicalSimulator::new(),
            cfg: c,
        }
    }
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        let dt = self.cfg.physics_dt();
        let mut tf = 0.0;
        let mut te = 0.0f32;
        for step in 0..self.cfg.steps_per_episode {
            let hv = self.enc.encode(self.sim.state());
            if step % self.cfg.cognitive_interval == 0 {
                let _ = self.fep.tick(self.sim.state());
            }
            let cmd = self.ctrl.forward(&hv, dt as f32);
            self.sim.step(&cmd, dt);
            tf += self.sim.state().force_magnitude();
            te += cmd.control_effort();
            if !self.sim.state().is_finite() {
                return EpisodeMetrics {
                    mean_force: tf / (step + 1) as f64,
                    mean_effort: te / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                };
            }
        }
        let n = self.cfg.steps_per_episode;
        EpisodeMetrics {
            mean_force: tf / n as f64,
            mean_effort: te / n as f32,
            steps_survived: n,
            diverged: false,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ep() {
        let mut c = SurgicalConfig::default();
        c.steps_per_episode = 500;
        let mut t = SurgicalTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.diverged);
    }
}
