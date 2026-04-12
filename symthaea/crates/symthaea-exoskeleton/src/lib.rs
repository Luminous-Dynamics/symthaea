// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-exoskeleton
//!
//! Consciousness-coupled lower-limb exoskeleton for human augmentation.
//! The ONLY platform where AI shares motor authority with a human user.
//!
//! Phi maps to assistance mode:
//! - Green (>0.6): Predictive assist — anticipates user intent
//! - Yellow (0.3-0.6): Responsive assist — follows with amplification
//! - Orange (0.1-0.3): Transparent — minimal assistance
//! - Red (<0.1): Gravity compensation only — fully backdrivable

#![deny(unsafe_code)]

pub mod types;
pub mod encoder;
pub mod controller;
pub mod simulator;
pub mod embodiment;
pub mod fep_agent;
pub mod perturbations;
pub mod training;
pub mod plugin;
#[cfg(feature = "symtropy")]
pub mod symtropy_sim;
#[cfg(feature = "symtropy")]
pub mod full_frame;
#[cfg(feature = "hal")]
pub mod hal_bridge;
#[cfg(feature = "sensors")]
pub mod sensored_suite;
