// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-physics coupling for the Symtropy engine.
//!
//! This is the novel layer that makes this engine unique: consciousness (Φ) is
//! a first-class physics parameter that modulates forces, energy budgets, and
//! collision responses.
//!
//! # Five Coupling Channels
//!
//! 1. **Φ → Motor gain**: Higher consciousness = more force authority (safety tiers)
//! 2. **Φ → Energy budget**: Consciousness gates available energy per tick
//! 3. **Harmony → Collision**: Sanctuary zones dampen collision impulses
//! 4. **Collective Φ → Global**: Civilization consciousness affects world parameters
//! 5. **Bottleneck → Environment**: Agent's limiting factor triggers environmental response

pub mod coupling;
pub mod energy;
pub mod harmony_field;
pub mod safety;
pub mod sanctuary;
pub mod thermodynamics;

pub use coupling::{ConsciousnessField, EntityConsciousness};
pub use energy::EnergyBudget;
pub use harmony_field::HarmonyField;
pub use safety::SafetyTier;
pub use sanctuary::SanctuaryZone;
pub use thermodynamics::ThermodynamicLedger;
