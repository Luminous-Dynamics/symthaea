// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-agribot
//!
//! Consciousness-coupled ecological stewardship platform.
//!
//! Tier 1 platform goal:
//! - soil / water / light / crop-state feedback as embodied constraints
//! - commons-oriented tending rather than extractive throughput optimization
//! - provide the stewardship counterweight to mining / logistics platforms

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod environment;
pub mod fep_agent;
pub mod plugin;
pub mod reflex;
pub mod simulator;
pub mod training;
pub mod types;

pub use types::*;
