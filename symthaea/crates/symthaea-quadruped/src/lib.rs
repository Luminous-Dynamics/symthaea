// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Legged quadruped — CPG-modulated consciousness-gated locomotion.
//! Trot(>0.6), Walk(0.3-0.6), Freeze(0.1-0.3), Collapse(<0.1).
#![deny(unsafe_code)]
pub mod types; pub mod encoder; pub mod controller; pub mod simulator;
pub mod embodiment; pub mod fep_agent; pub mod perturbations; pub mod training; pub mod plugin;
#[cfg(feature = "symtropy")]
pub mod symtropy_sim;
