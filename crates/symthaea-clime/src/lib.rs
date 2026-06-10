// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-clime
//!
//! Consciousness-coupled habitat homeostasis platform.
//!
//! First civic platform goals:
//! - maintain breathable air, thermal comfort, and circadian coherence
//! - degrade gracefully under utility pressure and environmental incidents
//! - model habitat safety before energy optimization

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod plugin;
pub mod simulator;
pub mod training;
pub mod types;

pub use types::*;
