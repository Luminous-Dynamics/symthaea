// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Wisdom Module - Philosophical Grounding for Computational Cognition
//!
//! Re-exports from the `symthaea-wisdom` sub-crate.
//! See that crate for full documentation.

// Re-export all public items from the sub-crate
pub use symthaea_wisdom::*;

// Re-export submodules for path compatibility (crate::wisdom::autopoiesis::...)
pub use symthaea_wisdom::autopoiesis;
pub use symthaea_wisdom::harmonics;
pub use symthaea_wisdom::meta_cognition;
