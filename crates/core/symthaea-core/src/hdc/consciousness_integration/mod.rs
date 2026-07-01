// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Integration Module
//!
//! Provides types for consciousness pipeline integration with comprehensive
//! multi-system consciousness architecture.
//!
//! ## Integrated Systems
//!
//! 1. **Metacognitive Monitoring** - Self-awareness of processing quality
//! 2. **Predictive Consciousness** - Active inference and anticipation
//! 3. **Cross-Modal Binding** - Multi-sensory integration
//! 4. **Temporal Binding** - Stream coherence with theta-phase modulation
//! 5. **Emergent Self-Model** - Recursive higher-order thought

mod builder;
mod pipeline;
mod systems;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use types::*;

// Re-export the builder
pub use builder::*;

// Re-export the pipeline
pub use pipeline::*;

// Re-export external types that were previously in scope (needed by tests and consumers)
pub use super::binary_hv::BinaryHV;
pub use super::phi_guided_search::InitializationStrategy;
