// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Tool Gate Subsystem
//!
//! Consciousness-gated tool use with:
//! - **Risk lattice**: ReadOnly < Reversible < Elevated < High < Critical
//! - **Two-signal gating**: Φ_eff AND plan_confidence (INV-8)
//! - **Fallback strategies**: always computed when blocked
//! - **NixOS backward compatibility**: wraps existing PhiGate

pub mod classifier;
pub mod fallback;
pub mod nixos_adapter;
pub mod types;

// Re-export key types
pub use classifier::{classify, gate};
pub use fallback::{fallback_frustration_cost, select_contextual_fallback};
pub use nixos_adapter::tool_descriptor_from_shell_command;
pub use types::{
    FallbackStrategy, GateDecision, GateResult, RiskLevel, ToolCalibration, ToolDescriptor,
};
