use serde::{Deserialize, Serialize};

/// Data passed from host to guest (zkVM)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionInput {
    pub episodes: Vec<Vec<f32>>, // HDC vectors (e.g., 1024D)
    pub tau_scale: f32,
    pub threshold: f32,
}

/// Data committed by the guest as public output
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionOutput {
    pub average_phi: f32,
    pub tau_scale: f32,
    pub episode_count: u32,
}
