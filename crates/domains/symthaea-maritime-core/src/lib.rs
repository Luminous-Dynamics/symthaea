// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared, mission-neutral maritime autonomy primitives.
//!
//! This crate intentionally stops below mission-specific applications. It defines
//! platform state, health, authority, degraded-operation, fleet-assurance,
//! operating-mode/limit evaluation, resident docking/service and progressive-failure
//! recording contracts reusable by AUVs, USVs, research vessels, logistics craft
//! and other maritime platforms.

pub mod authority;
pub mod degraded;
pub mod docking;
pub mod fleet;
pub mod health;
pub mod operating_mode;
pub mod prelude;
pub mod resilience;
pub mod state;

pub use authority::*;
pub use degraded::*;
pub use docking::*;
pub use fleet::*;
pub use health::*;
pub use operating_mode::*;
pub use resilience::*;
pub use state::*;
