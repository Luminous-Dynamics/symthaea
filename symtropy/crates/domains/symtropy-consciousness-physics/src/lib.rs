// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration-physics coupling for the Symtropy engine.
//!
//! This crate couples **integrated information (Φ)** — a formal measure from
//! Integrated Information Theory (IIT, Tononi 2004) — to rigid body dynamics.
//! Φ quantifies how much a system's information exceeds the sum of its parts.
//! It is a mathematical measure of integration, NOT a claim about subjective
//! experience, qualia, or sentience.
//!
//! # Terminology
//!
//! Throughout this crate, "consciousness" refers exclusively to the IIT formal
//! quantity Φ (integrated information). When we say an entity "has consciousness,"
//! we mean its Φ value exceeds a threshold — nothing more. The type names
//! `ConsciousnessField` and `EntityConsciousness` are retained for API stability
//! and discoverability, but should be read as "integration field" and "entity
//! integration state" respectively.
//!
//! # Five Coupling Channels (per FORMAL_SPECIFICATION.md)
//!
//! 1. **Φ → Force**: Motor gain modulation via NRC 4-tier safety (Green/Yellow/Orange/Red)
//! 2. **Φ → Energy**: Φ-gated energy budget (movement, maintenance, collision costs)
//! 3. **Harmony → Impulse**: Sanctuary zones dampen collision impulses (Sacred Stillness)
//! 4. **Harmony → Friction**: 1/r² CEMI-inspired fields modulate friction coefficients
//! 5. **Collision → Φ**: Prediction error feedback reduces motor precision

pub mod active_inference;
pub mod biometrics;
pub mod convergence;
pub mod coupling;
#[cfg(feature = "consciousness-curvature")]
pub mod curvature;
pub mod dimensional_leakage;
pub mod energy;
pub mod fep_gradient;
pub mod harmony_field;
#[cfg(feature = "consciousness-hdc")]
pub mod hdc_context;
pub mod macro_bridge;
pub mod phase_transition;
pub mod prey;
pub mod proc_gen;
#[cfg(feature = "pyphi-validation")]
pub mod pyphi_validation;
pub mod safety;
pub mod sanctuary;
pub mod simple_field;
pub mod spatial_hash;
pub mod thermodynamics;
#[cfg(feature = "wasm")]
pub mod wasm_bindings;

pub use biometrics::{
    BiometricHistory, BiometricNormParams, BiometricSample, ChannelNorm, neutral_inputs,
};
pub use coupling::{ConsciousnessField, EntityConsciousness};
pub use energy::EnergyBudget;
pub use harmony_field::HarmonyField;
pub use macro_bridge::{MacroWorldModifiers, apply_macro_modifiers};
pub use phase_transition::{
    BifurcationConfig, BifurcationResult, CriticalityIndicators, TransitionType,
    run_bifurcation_sweep,
};
pub use proc_gen::{
    ConsciousnessRegime, EnvironmentEvent, PHI_THRESHOLD_EMERGING, PHI_THRESHOLD_INTEGRATED,
    PHI_THRESHOLD_TRANSCENDENT, PhiEnvironmentCoupler, RoomProfile,
};
pub use safety::SafetyTier;
pub use sanctuary::SanctuaryZone;
pub use simple_field::{SimpleCoupledField, SimpleEntity};
pub use thermodynamics::{ThermodynamicConstants, ThermodynamicLedger};
