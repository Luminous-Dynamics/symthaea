// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public types returned by the cognitive loop.
//!
//! Decomposed from a monolithic types.rs into thematic sub-modules.
//! All public APIs are preserved via re-exports.

pub mod carryover;
mod output;
mod scheduling;
pub(crate) mod telemetry;

pub use output::*;
pub use scheduling::*;
pub use telemetry::*;

// Re-export crate-visible types
pub use carryover::{ConsciousnessCache, CycleCarryover, QualityMetrics};
// Used by test modules — gate to suppress unused-import warnings in lib builds
#[cfg(test)]
pub(crate) use carryover::{LearningState, UrgencyState};
pub(crate) use scheduling::CycleState;

// ── Substrate / Thermal / Integrity Telemetry ───────────────────────────────
// Moved to the symthaea-cognitive-types crate (2026-07-12) — pure data types
// with no dependency on CognitiveLoopService or manager-owned types.
pub use symthaea_cognitive_types::{
    AttestationDetail, IntegrityTelemetry, SubstrateTelemetry, ThermalTelemetry,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
