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
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        let dt = self.cfg.physics_dt();
        let x0 = self.sim.state().base_position[0];
        let mut th = 0.0;
        let mut te = 0.0f32;
        for step in 0..self.cfg.steps_per_episode {
            let hv = self.enc.encode(self.sim.state());
            if step % self.cfg.cognitive_interval == 0 {
                let _ = self.fep.tick(self.sim.state());
            }
            let cmd = self.ctrl.forward(&hv, dt as f32);
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
    }
}
