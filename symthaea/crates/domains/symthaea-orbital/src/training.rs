use crate::controller::OrbitalController;
use crate::encoder::OrbitalHdcEncoder;
use crate::fep_agent::ActiveInferenceOrbitalAgent;
use crate::simulator::{OrbitalPhysicsSimulator, SimpleOrbitalSimulator};
use crate::types::OrbitalConfig;
use symthaea_core::genesis::GenesisSeed;
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_sc_rate: f64,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
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
        for step in 0..self.cfg.steps_per_episode {
            let hv = self.enc.encode(self.sim.state());
            if step % self.cfg.cognitive_interval == 0 {
                let _ = self.fep.tick(self.sim.state());
            }
            let cmd = self.ctrl.forward(&hv, dt as f32);
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
                };
            }
        }
        let n = self.cfg.steps_per_episode;
        EpisodeMetrics {
            mean_sc_rate: tr / n as f64,
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
        let mut c = OrbitalConfig::default();
        c.steps_per_episode = 300;
        let mut t = OrbitalTrainer::new(c);
        let m = t.run_episode();
        assert!(!m.diverged);
    }
}
