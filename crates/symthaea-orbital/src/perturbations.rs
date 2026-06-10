use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrbitalPerturbation {
    DebrisImpact {
        force: [f64; 3],
        at_step: usize,
    },
    CommBlackout {
        duration: usize,
        at_step: usize,
    },
    SolarFlare {
        power_reduction: f64,
        at_step: usize,
    },
}
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    pub perturbations: Vec<OrbitalPerturbation>,
}
