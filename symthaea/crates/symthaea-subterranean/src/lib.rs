// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-subterranean
//!
//! Consciousness-coupled subterranean scout / boring platform.
//!
//! Tier 1 platform goal:
//! - digging / spoil / thermal load as first-class embodied constraints
//! - intermittent communication and delayed surfacing
//! - geological exploration rather than open-air locomotion

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod plugin;
pub mod simulator;
pub mod training;
pub mod types;

pub use types::*;
