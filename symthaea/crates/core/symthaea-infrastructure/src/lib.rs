// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-infrastructure
//!
//! Consciousness-coupled stationary infrastructure platform.
//!
//! Tier 1 platform goal:
//! - model hubs, microgrids, storage, and routing nodes as embodied agents
//! - treat thermal and load-balancing behavior as part of autonomy
//! - provide the anchor point for future civic dispatch and city metabolism sims

pub mod controller;
pub mod economy;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod plugin;
pub mod simulator;
pub mod spatial_metabolism;
pub mod town_simpoiesis;
pub mod training;
pub mod types;

pub use types::*;
