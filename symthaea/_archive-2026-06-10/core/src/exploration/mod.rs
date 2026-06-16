// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Exploration Strategies for Symthaea
//!
//! Re-exports from the `symthaea-exploration` sub-crate.
//! See that crate for full documentation.

pub use symthaea_exploration::*;

/// Re-export the sub-crate itself as `surprise_driven` for backwards compat.
pub use symthaea_exploration as surprise_driven;
