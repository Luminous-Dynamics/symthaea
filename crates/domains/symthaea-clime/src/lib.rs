// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-clime
//!
//! Consciousness-coupled habitat homeostasis platform.
//!
//! First civic platform goals:
//! - maintain breathable air and thermal comfort
//! - degrade gracefully under utility pressure and environmental incidents
//! - model habitat safety before energy optimization
//!
//! Note: "circadian coherence" is aspirational language from the original
//! design doc (`docs/robotics/CLIME_HAZARDS_AND_SENSORIUM.md`). The actual
//! `circadian_mismatch` channel is a single scalar that monotonically
//! accumulates "night-mode pressure" and decays only when the
//! `light_circadian_shift` actuator counteracts it -- there is no real
//! two-process (sleep-pressure + circadian-phase) model, no time-of-day
//! input, and no light-history tracking.

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod plugin;
pub mod reflex;
pub mod simulator;
pub mod training;
pub mod types;

pub use types::*;
