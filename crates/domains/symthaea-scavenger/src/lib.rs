// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-scavenger
//!
//! Consciousness-coupled scavenger / disassembly platform.
//!
//! Tier 1 platform goal:
//! - recover material from damaged or obsolete structures
//! - introduce teardown, sorting, and salvage economics to the robotics stack
//! - become the closed-loop counterpart to fabrication and transport platforms

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
