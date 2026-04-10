// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-helicopter
//!
//! Consciousness-coupled SAR helicopter control using unified HDC-LTC + FEP
//! Active Inference. Part of the Symthaea robotics expansion (Phase 1).
//!
//! ## Architecture
//!
//! ```text
//! Sensors (18D) → HelicopterHdcEncoder → ContinuousHV(16,384D)
//!                                              ↓
//!                                   HdcLtcUnifiedNetwork (evolve_closed_form)
//!                                              ↓
//!                                   HelicopterController (output projection 16,384→6)
//!                                              ↓
//!                                   HelicopterCommand [collective, cyclic×2, pedal, thrust, tail]
//!                                              ↓
//!                                   SimpleHelicopterSimulator (rotor dynamics + body physics)
//! ```
//!
//! ## SAR Mission Profile
//!
//! Designed for search-and-rescue operations governed by the Mycelix emergency
//! cluster. Helicopter is registered as an `EmergencyResource` with 24-hour
//! authority expiry. Consciousness gating (Phi) controls motor authority:
//! - Green (Φ > 0.6): full flight authority
//! - Yellow (0.3–0.6): reduced maneuver envelope
//! - Orange (0.1–0.3): hover-only, no translation
//! - Red (Φ < 0.1): autorotation descent, all motors to safe state
//!
//! ## Features
//!
//! - `mujoco` — MuJoCo high-fidelity physics (requires mujoco-rs)

#![allow(clippy::needless_range_loop)]

pub mod benchmarks;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod navigation_estimator;
pub mod perturbations;
pub mod rotor_dynamics;
pub mod sar_mission;
pub mod simulator;
pub mod training;
pub mod transfer;
pub mod types;
pub mod wind_model;

pub use controller::HelicopterController;
pub use encoder::HelicopterHdcEncoder;
pub use fep_agent::ActiveInferenceHelicopterAgent;
pub use navigation_estimator::{HelicopterNavigationEstimate, HelicopterNavigationEstimator};
pub use perturbations::{HelicopterPerturbation, PerturbationSchedule};
pub use simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
pub use training::HelicopterTrainer;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helicopter_command_hover() {
        let cmd = HelicopterCommand::hover();
        assert!(cmd.thrust > 0.0, "hover thrust should be positive");
        assert!(cmd.collective > 0.0, "hover collective should be positive");
    }

    #[test]
    fn channel_names_count() {
        assert_eq!(CHANNEL_NAMES.len(), NUM_STATE_CHANNELS);
    }
}
