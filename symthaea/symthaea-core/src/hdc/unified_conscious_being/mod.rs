// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Conscious Being
//!
//! The complete integration of all consciousness systems into a single coherent entity.
//!
//! ## Key Innovations
//!
//! 1. **A: Real Persistence** - HippocampusActor + UnifiedMind for durable memory
//! 2. **B: Conscious Dialogue** - Φ-gated response generation
//! 3. **C: Unified Agent** - Single coherent consciousness rather than separate systems
//! 4. **D: Do-Calculus** - Pearl's causal intervention for rigorous counterfactuals
//! 5. **E: Consciousness Prosody** - LTC-driven speech with embodied emotion
//! 6. **F: Test Framework** - Comprehensive scenario testing

pub mod being;
pub mod dialogue;
pub mod do_calculus;
mod extensions;
pub mod scenarios;
mod sharing;

#[cfg(test)]
mod tests;

// Re-export all public items for backward compatibility
pub use being::*;
pub use dialogue::*;
pub use do_calculus::*;
pub use scenarios::*;
pub use sharing::*;
