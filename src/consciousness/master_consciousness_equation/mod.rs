//! # Master Consciousness Equation
//!
//! This module implements the unified Master Consciousness Equation:
//!
//! **C(t) = σ(softmin(Φ, B, W, A, R, E, K, M, N, Soc; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)**

mod config;
mod embodiment;
mod engine;
mod narrative;
mod social;
mod types;

pub use config::{ComponentWeights, MasterEquationConfig};
pub use embodiment::{EmbodimentDiagnostics, EmbodimentFactor, InteroceptiveState};
pub use engine::MasterConsciousnessEquation;
pub use narrative::{
    FutureScenario, NarrativeCoherence, NarrativeCoherenceDiagnostics, NarrativeEpisode,
    SimulationBranch,
};
pub use social::{AgentModel, SelfModel, SocialEmbedding};
pub use types::{ConsciousnessInputs, ConsciousnessResult};

#[cfg(test)]
mod tests;
