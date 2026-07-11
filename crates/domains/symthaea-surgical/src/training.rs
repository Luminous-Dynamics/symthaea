// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::SurgicalController;
use crate::encoder::SurgicalHdcEncoder;
use crate::fep_agent::ActiveInferenceSurgicalAgent;
use crate::reflex::reflex_torques;
use crate::simulator::{SimpleSurgicalSimulator, SurgicalPhysicsSimulator};
use crate::types::SurgicalConfig;
use symthaea_core::genesis::GenesisSeed;
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_force: f64,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
    /// Mean pre-update imitation MSE against the hand-designed
    /// PD-hold-toward-home reflex -- the learning signal (real-trainer
    /// follow-up to SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub mean_imitation_loss: f32,
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
        let mut tau_factor = 1.0f32;
        let mut loss_sum = 0.0f32;
        for step in 0..self.cfg.steps_per_episode {
            let hv = self.enc.encode(self.sim.state());
            if step % self.cfg.cognitive_interval == 0 {
                let fep = self.fep.tick(self.sim.state());
                tau_factor = fep.tau_factor;
            }
            let cmd = self.ctrl.forward(&hv, dt as f32 * tau_factor);
            let target = reflex_torques(self.sim.state());
            loss_sum += self.ctrl.train_step(&target);
            self.sim.step(&cmd, dt);
            tf += self.sim.state().force_magnitude();
            te += cmd.control_effort();
            if !self.sim.state().is_finite() {
                return EpisodeMetrics {
                    mean_force: tf / (step + 1) as f64,
                    mean_effort: te / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                    mean_imitation_loss: loss_sum / (step + 1) as f32,
                };
            }
        }
        let n = self.cfg.steps_per_episode;
        EpisodeMetrics {
            mean_force: tf / n as f64,
            mean_effort: te / n as f32,
            steps_survived: n,
            diverged: false,
            mean_imitation_loss: loss_sum / n as f32,
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
        assert!(m.mean_imitation_loss > 0.0, "learning signal must be live");
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        let mut c = SurgicalConfig::default();
        c.steps_per_episode = 400;
        let mut t = SurgicalTrainer::new(c);
        let first = t.run_episode().mean_imitation_loss;
        for _ in 0..3 {
            t.run_episode();
        }
        let last = t.run_episode().mean_imitation_loss;
        assert!(
            last < first,
            "imitation loss must decrease with training: first {first:.5} -> last {last:.5}"
        );
    }
}
