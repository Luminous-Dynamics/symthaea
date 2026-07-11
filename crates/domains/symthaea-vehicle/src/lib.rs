// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-vehicle
//!
//! HDC-LTC + FEP Active Inference autonomous vehicle control via bicycle-model dynamics.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 200Hz physics + 50Hz cognitive tick.
//!
//! ## Architecture — TARGET DEPLOYMENT SHAPE, NOT IMPLEMENTED
//!
//! Everything in this diagram outside the "Cognitive Core" box (Coral/Hailo
//! retina coprocessor, Raspberry Pi host, CAN bus actuators, LoRa mesh
//! radio) is aspirational hardware — none of it exists in this crate, which
//! is a pure-Rust simulation. The in-crate mesh (`swarm.rs`) exchanges peer
//! state instantly with no latency/dropout model. (Labeled honestly per the
//! 2026-07 robotics deep review.)
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  "Retina" Coprocessor (Coral TPU / Hailo-8)              │
//! │  Camera/LiDAR → 16,384D ContinuousHV                    │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  Symthaea Cognitive Core (Raspberry Pi, 50Hz)            │
//! │                                                          │
//! │  Sensors (14D) → VehicleHdcEncoder → ContinuousHV(16384D)│
//! │                                            ↓             │
//! │                     HdcLtcUnifiedNetwork (3×8 neurons)   │
//! │                         evolve_closed_form(dt)           │
//! │                                            ↓             │
//! │                     VehicleController (16384→3 projection)│
//! │                                            ↓             │
//! │                     VehicleCommand [steering, throttle,   │
//! │                                     brake — all tanh]    │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  CAN Bus → Vehicle Actuators                             │
//! │  Steering motor, throttle-by-wire, brake-by-wire         │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  LoRa Mesh Radio → Swarm Horizon                         │
//! │  2KB Wisdom Vectors from ±5 vehicles                     │
//! │  ABS engagement propagates at speed of light              │
//! └──────────────────────────────────────────────────────────┘
//!
//! Every 4th physics step (50Hz):
//!   ActiveInferenceAgent modulates τ, learning rate, prior precision
//!   EFE decomposition:
//!     Risk  → minimize entropy (collision/traction loss = spike)
//!     Ambiguity → minimize uncertainty (rain/fog → slow down)
//! ```
//!
//! ## Tasks
//!
//! - **LaneKeep**: Maintain lane position at target velocity (~13.4 m/s / 30 mph)
//! - **Follow**: Adaptive cruise — match lead vehicle speed
//! - **LaneChange**: Execute a safe lane change maneuver
//! - **EmergencyStop**: Bring vehicle to zero from speed
//!
//! ## Perturbation Crucibles
//!
//! - **Black Ice**: Friction drops to 0.05 at step 400
//! - **Crosswind**: 3000N sustained lateral wind at step 300
//! - **Sensor Blindness**: Camera channels zeroed (heavy rain)
//! - **Tire Blowout**: Asymmetric grip loss + yaw moment
//! - **Mesh Spoof**: Byzantine false brake signals (must detect + ignore)
//! - **Gauntlet**: All perturbations at staggered intervals

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod reward;
pub mod road;
pub mod scenarios;
pub mod simulator;
pub mod subterranean;
pub mod subterranean_navigation;
pub mod swarm;
pub mod training;
pub mod types;

pub use controller::VehicleController;
pub use encoder::VehicleHdcEncoder;
pub use fep_agent::{ActiveInferenceVehicleAgent, VehicleFepConfig, VehicleFepResult};
pub use perturbations::{PerturbationSchedule, VehiclePerturbation};
pub use reward::{episode_reward, follow_distance_reward, safety_reward, speed_reward};
pub use road::{Road, RoadSegment};
pub use scenarios::{
    ScenarioConfig, ScenarioResult, default_scenarios, results_to_csv, run_all_scenarios,
    run_scenario,
};
pub use simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
pub use subterranean::{CaveRelayDecision, SubterraneanBridge, SurveyAnchor, TunnelRelayNode};
pub use subterranean_navigation::{
    SubterraneanEstimate, SubterraneanNavigator, SubterraneanTrainingContext,
};
pub use swarm::{SwarmConfig, SwarmSimulator};
pub use training::{EpisodeMetrics, VehicleTrainer};
pub use types::*;
