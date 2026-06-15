// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Master Consciousness Equation
//!
//! Standalone crate implementing the unified Master Consciousness Equation:
//!
//! **C(t) = σ(softmin(Φ, B, W, A, R, E, K, M, N, Soc; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)**
//!
//! - **M** (Embodiment Factor) = sensorimotor_prediction_accuracy × interoceptive_coherence
//! - **N** (Narrative Coherence) = autobiographical_integration × future_simulation_depth
//! - **Soc** (Social Embedding) = other_modeling_accuracy × self_other_distinction
//!
//! This crate has zero dependencies on the main `symthaea` crate. It depends only
//! on `std` and `serde`.

mod config;
mod embodiment;
mod engine;
mod narrative;
mod social;
mod stability_auditor;
mod types;

pub use config::{ComponentWeights, MasterEquationConfig};
pub use embodiment::{EmbodimentDiagnostics, EmbodimentFactor, InteroceptiveState};
pub use engine::MasterConsciousnessEquation;
pub use narrative::{
    FutureScenario, NarrativeCoherence, NarrativeCoherenceDiagnostics, NarrativeEpisode,
    SimulationBranch,
};
pub use social::{AgentModel, SelfModel, SocialEmbedding};
pub use stability_auditor::StabilityAuditor;
pub use types::{ConsciousnessInputs, ConsciousnessResult};

#[cfg(test)]
mod tests;
