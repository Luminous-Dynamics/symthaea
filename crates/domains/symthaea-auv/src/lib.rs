// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-auv
//!
//! Consciousness-coupled Autonomous Underwater Vehicle for commons water stewardship.
//! Part of the Symthaea robotics expansion (Phase 2).
//!
//! Implements Elinor Ostrom's commons principles in physical robotic form:
//! the AUV monitors water quality, maps aquifers, and maintains infrastructure
//! under governance of the local water-purity and water-steward Mycelix zomes.
//!
//! ## Architecture
//!
//! ```text
//! Sensors (32D: 24 kinematic + 8 chemical) → AuvHdcEncoder → ContinuousHV(16,384D)
//!                                                  ↓
//!                                       HdcLtcUnifiedNetwork (evolve_closed_form)
//!                                                  ↓
//!                                       AuvController (output projection 16,384→8)
//!                                                  ↓
//!                                       AuvCommand [8 thruster RPMs]
//!                                                  ↓
//!                                       6DOF Hydrodynamic Simulator
//! ```
//!
//! ## Key Physics
//!
//! Unlike aerial platforms, the AUV operates in a drag-dominated environment:
//! - **Added mass**: 6×6 diagonal matrix (fluid accelerated with the vehicle)
//! - **Quadratic drag**: F = -0.5 × ρ × Cd × A × |v| × v
//! - **Buoyancy**: F_b = ρ × g × V (depth-invariant for rigid hull)
//! - **Hydrostatic restoring**: pitch/roll restored by metacentric height
//!
//! ## Chemical Sensors
//!
//! 8 channels mapping 1:1 to the water-purity zome's QualityReading:
//! pH, turbidity, TDS, nitrates, arsenic, lead, chlorine, dissolved oxygen.

#![allow(clippy::needless_range_loop)]

pub mod acoustic_modem;
pub mod chemical_sensors;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod hydrodynamics;
pub mod mission;
pub mod navigation_bridge;
pub mod navigation_estimator;
pub mod perturbations;
pub mod plugin;
pub mod reflex;
pub mod simulator;
pub mod store_forward;
pub mod training;
pub mod transfer;
pub mod types;

pub use controller::AuvController;
pub use encoder::AuvHdcEncoder;
pub use mission::{AuvCommsPolicy, AuvMission, MissionMetrics, MissionTracker, Waypoint};
pub use navigation_bridge::{AuvNavigationBridge, UnderwaterNavigationSample};
pub use navigation_estimator::{AuvNavigationEstimate, AuvNavigationEstimator};
pub use simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
pub use types::*;
