// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-multirotor
//!
//! Unified HDC-LTC + FEP Active Inference quadrotor flight control.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 500Hz motor reflex + 25Hz cognitive tick.
//!
//! ## Architecture
//!
//! ```text
//! Sensors → QuadrotorHdcEncoder → ContinuousHV(16384D)
//!                                       ↓
//!                              HdcLtcUnifiedNetwork (evolve_closed_form)
//!                                       ↓
//!                              FlightController (output projection 16384→4)
//!                                       ↓
//!                              QuadrotorCommand [thrust, roll, pitch, yaw]
//!                                       ↓
//!                              MuJoCo Physics Step
//!
//! Every 20th motor step (25Hz):
//!   ActiveInferenceFlightAgent modulates τ, learning rate, prior precision
//! ```
//!
//! ## Features
//!
//! - `mujoco` — MuJoCo high-fidelity physics (requires mujoco-rs)
//! - `mujoco-viewer` — Interactive 3D viewer (requires mujoco-rs + viewer)
//! - `swarm` — Parallel swarm training via rayon + MuJoCo

#![allow(clippy::needless_range_loop)]

pub mod benchmarks;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod formation;
pub mod perturbations;
pub mod simulator;
pub mod training;
pub mod types;

// ── MuJoCo-dependent modules ──
#[cfg(feature = "mujoco")]
pub mod mujoco_sim;
pub mod navigation_estimator;
#[cfg(feature = "mujoco")]
pub mod scenarios;
#[cfg(feature = "mujoco")]
pub mod sensory_filter;
#[cfg(feature = "mujoco")]
pub mod viewer;

// ── Swarm training (MuJoCo + rayon) ──
#[cfg(feature = "swarm")]
pub mod swarm;

pub use controller::FlightController;
pub use encoder::QuadrotorHdcEncoder;
pub use fep_agent::ActiveInferenceFlightAgent;
pub use navigation_estimator::{FlightNavigationEstimate, FlightNavigationEstimator};
pub use perturbations::{FlightPerturbation, PerturbationSchedule};
pub use simulator::{PhysicsSimulator, SimplePhysicsSimulator};
pub use training::FlightTrainer;
pub use types::*;

// Re-export MuJoCo types when feature is enabled
#[cfg(feature = "mujoco")]
pub use mujoco_sim::MuJoCoSimulator;
#[cfg(feature = "mujoco")]
pub use sensory_filter::{SensoryFilter, SensoryFilterConfig};

/// Re-exported mujoco-rs types for downstream consumers (examples, etc.)
#[cfg(feature = "mujoco")]
pub mod mujoco_reexports {
    pub use mujoco_rs::prelude::*;

    #[cfg(feature = "mujoco-renderer")]
    pub use mujoco_rs::renderer::MjRenderer;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrotor_command_hover() {
        let cmd = QuadrotorCommand::hover();
        assert!(cmd.thrust > 0.0, "hover thrust should be positive");
    }

    #[test]
    fn flight_state_hover_altitude() {
        let state = FlightState::hover(5.0);
        assert!((state.position[2] - 5.0).abs() < 1e-6);
    }
}
