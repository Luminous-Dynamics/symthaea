use crate::controller::OrbitalController;
use crate::encoder::OrbitalHdcEncoder;
use crate::fep_agent::ActiveInferenceOrbitalAgent;
use crate::reflex::reflex_torques;
use crate::simulator::{OrbitalPhysicsSimulator, SimpleOrbitalSimulator};
use crate::types::OrbitalConfig;
use symthaea_core::genesis::GenesisSeed;
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_sc_rate: f64,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
    /// Mean pre-update imitation MSE against the hand-designed
    /// PD-hold-toward-stowed arm reflex -- the learning signal
    /// (real-trainer follow-up to
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub mean_imitation_loss: f32,
}
pub struct OrbitalTrainer {
    ctrl: OrbitalController,
    enc: OrbitalHdcEncoder,
    fep: ActiveInferenceOrbitalAgent,
    sim: SimpleOrbitalSimulator,
    cfg: OrbitalConfig,
}
impl OrbitalTrainer {
    pub fn new(c: OrbitalConfig) -> Self {
        let g = GenesisSeed::from_phrase("orb_train");
        Self {
            ctrl: OrbitalController::new(&g, &c),
            enc: OrbitalHdcEncoder::new(&g, 32),
            fep: ActiveInferenceOrbitalAgent::new(),
            sim: SimpleOrbitalSimulator::new(),
            cfg: c,
        }
    }
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        let dt = self.cfg.physics_dt();
        let mut tr = 0.0;
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
            tr += self
                .sim
                .state()
                .spacecraft_angular_velocity
                .iter()
                .map(|v| v.abs())
                .sum::<f64>();
            te += cmd.control_effort();
            if !self.sim.state().is_finite() {
                return EpisodeMetrics {
                    mean_sc_rate: tr / (step + 1) as f64,
                    mean_effort: te / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                    mean_imitation_loss: loss_sum / (step + 1) as f32,
                };
            }
        }
        let n = self.cfg.steps_per_episode;
        EpisodeMetrics {
            mean_sc_rate: tr / n as f64,
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
        let mut c = OrbitalConfig::default();
        c.steps_per_episode = 300;
        let mut t = OrbitalTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.diverged);
        assert!(m.mean_imitation_loss > 0.0, "learning signal must be live");
    }

    // `train_step`'s actual convergence guarantee (error toward a fixed
    // target decreases with repeated calls) is proven directly and cheaply
    // in controller.rs's `test_train_step_reduces_error_toward_fixed_target`.
    // An episode-level first-vs-last version was tried here and abandoned:
    // the simulator re-stows the arm (near-zero-error target) every
    // episode, so whole-episode means are dominated by that episode's
    // environmental perturbations rather than by what the controller
    // learned -- noisy even averaged over many episodes, and each episode
    // is expensive enough (16,384-D HDC network per step) that a
    // multi-episode version made this single test take 3+ minutes.
}
