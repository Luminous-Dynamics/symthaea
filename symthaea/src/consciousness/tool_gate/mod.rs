//! Conscious Tool Gate Subsystem
//!
//! Consciousness-gated tool use with:
//! - **Risk lattice**: ReadOnly < Reversible < Elevated < High < Critical
//! - **Two-signal gating**: Φ_eff AND plan_confidence (INV-8)
//! - **Fallback strategies**: always computed when blocked
//! - **NixOS backward compatibility**: wraps existing PhiGate

pub mod types;
pub mod classifier;
pub mod fallback;
pub mod nixos_adapter;

// Re-export key types
pub use types::{
    RiskLevel, ToolDescriptor, GateDecision, GateResult,
    FallbackStrategy, ToolCalibration,
};
pub use classifier::{classify, gate};
pub use fallback::{select_contextual_fallback, fallback_frustration_cost};
pub use nixos_adapter::tool_descriptor_from_shell_command;
