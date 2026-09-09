// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared, mission-neutral maritime autonomy primitives.
//!
//! This crate intentionally stops below mission-specific applications. It defines
//! platform state, health, authority, degraded-operation and fleet-assurance
//! contracts reusable by AUVs, USVs, research vessels, logistics craft and other
//! maritime platforms.

pub mod authority;
pub mod degraded;
pub mod fleet;
pub mod health;
pub mod prelude;
pub mod state;

pub use authority::*;
pub use degraded::*;
pub use fleet::*;
pub use health::*;
pub use state::*;
