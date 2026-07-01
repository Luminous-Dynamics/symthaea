// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-coupled surgical robot — sub-mm RCM-constrained precision.
//! Safety: Green=5N/50mm/s, Yellow=2N/20mm/s, Orange=freeze, Red=retract.
#![deny(unsafe_code)]
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod plugin;
pub mod simulator;
pub mod training;
pub mod types;
