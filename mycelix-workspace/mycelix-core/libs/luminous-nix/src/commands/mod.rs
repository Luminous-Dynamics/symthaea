// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Command execution for NixOS operations
//!
//! This module provides async executors for all NixOS commands,
//! optimized with JSON output parsing where available.

mod executor;
mod types;

pub use executor::CommandExecutor;
pub use types::*;
