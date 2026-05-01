// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flight scenarios for demonstrating emergent FEP behavior.
//!
//! Each scenario sets up a multi-body MuJoCo scene and runs the full
//! HDC-LTC-FEP flight pipeline to observe emergent behavior.

pub mod kinetic_sacrifice;

pub use kinetic_sacrifice::{
    run_kinetic_sacrifice, KineticSacrificeConfig, KineticSacrificeResult, SacrificeEvent,
};
