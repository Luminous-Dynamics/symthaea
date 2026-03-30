// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kinship web visualization: force-directed graph with bond breathing,
//! gratitude particles, and homeostatic void.
//!
//! Uses Canvas2D with the same SharedState architecture planned for Bevy.
//! The upgrade path to Bevy is a renderer swap, not an architecture change.

pub mod shared_state;
pub mod renderer;
pub mod layout;

pub use shared_state::*;
pub use renderer::KinshipCanvas;
