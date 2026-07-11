// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-biota
//!
//! Consciousness-coupled interspecies sanctuary platform.
//!
//! First civic platform goal:
//! - detect animal distress and path-conflict hazards
//! - translate sanctuary and right-of-way signals into embodied action
//! - provide a welfare-first interface between animals and larger civic systems

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
