// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-physics coupling for the Symtropy engine.
//!
//! This is the novel layer that makes this engine unique: consciousness (Φ) is
//! a first-class physics parameter that modulates forces, energy budgets, and
//! collision responses.
//!
//! # Five Coupling Channels (per FORMAL_SPECIFICATION.md)
//!
//! 1. **Φ → Force**: Motor gain modulation via NRC 4-tier safety (Green/Yellow/Orange/Red)
//! 2. **Φ → Energy**: Consciousness-gated energy budget (movement, maintenance, collision costs)
//! 3. **Harmony → Impulse**: Sanctuary zones dampen collision impulses (Sacred Stillness)
//! 4. **Harmony → Friction**: 1/r² CEMI-inspired fields modulate friction coefficients
//! 5. **Collision → Consciousness**: Prediction error feedback reduces motor precision

pub mod active_inference;
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
pub mod prey;
pub mod safety;
pub mod sanctuary;
pub mod spatial_hash;
pub mod thermodynamics;

pub use coupling::{ConsciousnessField, EntityConsciousness};
pub use energy::EnergyBudget;
pub use harmony_field::HarmonyField;
pub use safety::SafetyTier;
pub use sanctuary::SanctuaryZone;
pub use thermodynamics::{ThermodynamicConstants, ThermodynamicLedger};
