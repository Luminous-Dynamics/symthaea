// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::AgribotController;
use crate::encoder::AgribotHdcEncoder;
use crate::fep_agent::ActiveInferenceAgribotAgent;
use crate::reflex::reflex_command;
use crate::simulator::{AgribotPhysicsSimulator, SimpleAgribotSimulator};
use crate::types::AgribotConfig;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
    /// Mean pre-update imitation MSE against the hand-designed reflex
    /// target -- the learning signal (Tier 2 of
    /// SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md).
    pub mean_imitation_loss: f32,
    /// Mean FEP free energy over the episode.
    pub mean_free_energy: f64,
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
            simulator: SimpleAgribotSimulator::new(&g),
            config,
        }
    }
    /// Run one training episode: imitation learning toward the
    /// hand-designed reflex policy, with FEP tau modulation. Previously this
    /// ran a rollout, collected metrics, and never updated a single weight
    /// while the FEP tick result was discarded (`let _ =`). Both loops are
    /// now closed.
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset();
        self.controller.reset();
        let dt = self.config.physics_dt();
        let mut total_e = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut fe_sum = 0.0f64;
        let mut fe_count = 0usize;
        let mut tau_factor = 1.0f32;
        for step in 0..self.config.steps_per_episode {
            let hv = self.encoder.encode(self.simulator.state());
            if step % self.config.cognitive_interval == 0 {
                let fep = self.fep.tick(self.simulator.state());
                tau_factor = fep.tau_factor;
                fe_sum += fep.free_energy;
                fe_count += 1;
            }
            let cmd = self.controller.forward(&hv, dt as f32 * tau_factor);
            loss_sum += self
                .controller
                .train_step(&reflex_command(self.simulator.state()));
            self.simulator.step(&cmd, dt);
            total_e += cmd.control_effort();
            if !self.simulator.state().is_finite() {
                return EpisodeMetrics {
                    mean_effort: total_e / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                    mean_imitation_loss: loss_sum / (step + 1) as f32,
                    mean_free_energy: fe_sum / fe_count.max(1) as f64,
                };
            }
        }
        EpisodeMetrics {
            mean_effort: total_e / self.config.steps_per_episode as f32,
            steps_survived: self.config.steps_per_episode,
            diverged: false,
            mean_imitation_loss: loss_sum / self.config.steps_per_episode as f32,
            mean_free_energy: fe_sum / fe_count.max(1) as f64,
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
        assert!(m.mean_imitation_loss > 0.0, "learning signal must be live");
        assert!(m.mean_free_energy.is_finite());
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        let mut c = AgribotConfig::default();
        c.steps_per_episode = 400;
        let mut t = AgribotTrainer::new(c);
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
