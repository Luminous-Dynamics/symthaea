// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-manipulator
//!
//! Consciousness-coupled industrial manipulator arm with moral algebra gating.
//! Part of the Symthaea robotics expansion (Phase 3).
//!
//! ## Architecture
//!
//! 7-DOF articulated arm with Denavit-Hartenberg kinematics, consciousness-gated
//! force limits, and moral algebra integration (non-harm, minimize-collateral).
//!
//! ## Consciousness-Gated Safety
//!
//! | Level | Phi | Force Cap | Workspace |
//! |-------|-----|-----------|-----------|
//! | Green | >0.6 | Full (>10N) | 100% |
//! | Yellow | 0.3–0.6 | 5N max | Inner 80% |
//! | Orange | 0.1–0.3 | 0N | Retreat to home |
//! | Red | <0.1 | Power cut | Gravity-compensated hold |

#![allow(clippy::needless_range_loop)]

pub mod co_assembly;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod kinematics;
pub mod perturbations;
pub mod simulator;
#[cfg(feature = "symtropy")]
pub mod symtropy_sim;
pub mod training;
pub mod types;
pub mod workspace_safety;

pub use co_assembly::{AssemblyTask, CooperativeGrip};
pub use types::*;
