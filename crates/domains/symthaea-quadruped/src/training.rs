use crate::controller::QuadrupedController;
use crate::encoder::QuadrupedHdcEncoder;
use crate::fep_agent::ActiveInferenceQuadrupedAgent;
use crate::simulator::{QuadrupedPhysicsSimulator, SimpleQuadrupedSimulator};
use crate::types::QuadrupedConfig;
use symthaea_core::genesis::GenesisSeed;
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_height: f64,
    pub distance: f64,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub fell: bool,
    /// Mean pre-update imitation MSE against the spinal reflex target —
    /// the learning signal. Decreasing across episodes = the controller is
    /// actually learning.
    pub mean_imitation_loss: f32,
    /// Mean FEP free energy over the episode (posture/stability surprise).
    pub mean_free_energy: f64,
}
pub struct QuadrupedTrainer {
    ctrl: QuadrupedController,
    enc: QuadrupedHdcEncoder,
    fep: ActiveInferenceQuadrupedAgent,
    sim: SimpleQuadrupedSimulator,
    cfg: QuadrupedConfig,
}
impl QuadrupedTrainer {
    pub fn new(c: QuadrupedConfig) -> Self {
        let g = GenesisSeed::from_phrase("quad_train");
        Self {
            ctrl: QuadrupedController::new(&g, &c),
            enc: QuadrupedHdcEncoder::new(&g, 32),
            fep: ActiveInferenceQuadrupedAgent::new(),
            sim: SimpleQuadrupedSimulator::new(),
            cfg: c,
        }
    }
    /// Run one training episode: imitation learning toward the simulator's
    /// spinal reflex layer (CPG-PD target torques), with FEP tau modulation.
    ///
    /// Previously this ran a rollout, collected metrics, and never updated a
    /// single weight ("QuadrupedTrainer" trained nothing) while the FEP tick
    /// result was discarded (`let _ =`). Both loops are now closed.
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        let dt = self.cfg.physics_dt();
        let x0 = self.sim.state().base_position[0];
        let mut th = 0.0;
        let mut te = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut fe_sum = 0.0f64;
        let mut fe_count = 0usize;
        let mut tau_factor = 1.0f32;
        for step in 0..self.cfg.steps_per_episode {
            let hv = self.enc.encode(self.sim.state());
            if step % self.cfg.cognitive_interval == 0 {
                // FEP loop closed: tau_factor modulates the controller's
                // effective timestep (posture surprise → slower, more
                // deliberate dynamics); free energy is tracked in metrics.
                let fep = self.fep.tick(self.sim.state());
                tau_factor = fep.tau_factor;
                fe_sum += fep.free_energy;
                fe_count += 1;
            }
            let cmd = self.ctrl.forward(&hv, dt as f32 * tau_factor);
            // Learning loop closed: supervised update toward the spinal
            // reflex torque the simulator is applying at this instant.
            loss_sum += self.ctrl.train_step(&self.sim.reflex_command());
            self.sim.step(&cmd, dt);
            th += self.sim.state().height();
            te += cmd.control_effort();
            if !self.sim.state().is_finite() || self.sim.state().height() < 0.01 {
                return EpisodeMetrics {
                    mean_height: th / (step + 1) as f64,
                    distance: self.sim.state().base_position[0] - x0,
                    mean_effort: te / (step + 1) as f32,
                    steps_survived: step,
                    fell: true,
                    mean_imitation_loss: loss_sum / (step + 1) as f32,
                    mean_free_energy: fe_sum / fe_count.max(1) as f64,
                };
            }
        }
        let n = self.cfg.steps_per_episode;
        EpisodeMetrics {
            mean_height: th / n as f64,
            distance: self.sim.state().base_position[0] - x0,
            mean_effort: te / n as f32,
            steps_survived: n,
            fell: false,
            mean_imitation_loss: loss_sum / n as f32,
            mean_free_energy: fe_sum / fe_count.max(1) as f64,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ep() {
        let mut c = QuadrupedConfig::default();
        c.steps_per_episode = 500;
        let mut t = QuadrupedTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.fell);
        assert!(m.distance > 0.0);
        assert!(m.mean_imitation_loss > 0.0, "learning signal must be live");
        assert!(m.mean_free_energy.is_finite());
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        // The trainer must actually LEARN: imitation loss against the spinal
        // reflex target decreases across episodes. This test fails against
        // the old trainer, which never updated a weight.
        let mut c = QuadrupedConfig::default();
        c.steps_per_episode = 400;
        let mut t = QuadrupedTrainer::new(c);
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
