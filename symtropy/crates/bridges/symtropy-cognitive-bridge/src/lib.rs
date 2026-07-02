// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stable bridge interface for cognitive agents.

use bevy::prelude::*;
use symtropy_core_stable::hdc::ContinuousHV;

pub mod crystallized_agent;

pub use crystallized_agent::SpacetimeCrystallizedAgent;

/// A symbolic predicate derived from hyperdimensional states.
pub trait SymbolicPredicate: Send + Sync {
    fn name(&self) -> String;
    fn evaluate(&self, state_hv: &ContinuousHV) -> bool;
    fn confidence(&self, state_hv: &ContinuousHV) -> f32;
}

/// A stable interface for agents to process vision and interact with the substrate.
pub trait CognitiveBridge: Send + Sync {
    /// Update perception with the latest crystallized vision vector.
    fn update_perception(&mut self, vision_hv: ContinuousHV);

    /// Get the next motor command vector.
    fn act(&mut self, dt: f32) -> Vec<f32>;

    /// Extract symbolic insights from the current internal state.
    fn query_symbols(&self) -> Vec<Box<dyn SymbolicPredicate>>;
}

/// Resource that acts as the stable mediator for agents.
#[derive(Resource)]
pub struct CognitiveBridgeResource {
    pub bridge: Box<dyn CognitiveBridge>,
}
pub mod integration;
