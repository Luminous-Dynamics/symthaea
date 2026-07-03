// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Re-export of [`PlatformType`] from `symtropy-robotics-bridge-core`.
//!
//! This crate doesn't define its own platform enum — the canonical
//! definition lives in the permissively-licensed `-core` crate so it can be
//! shared without pulling AGPL code into consumers that only need the enum.
//! This module exists purely so callers can write
//! `symtropy_robotics_bridge::platform::PlatformType`, matching the
//! `agent::RoboticAgent` import convention used across the platform demos
//! and this crate's `examples/`.

pub use symtropy_robotics_bridge_core::PlatformType;
