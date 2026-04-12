use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuadrupedPerturbation { ExternalPush { force: [f64; 3], at_step: usize }, LegFailure { leg: usize, at_step: usize }, IceFloor { friction: f64, at_step: usize } }
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule { pub perturbations: Vec<QuadrupedPerturbation> }
