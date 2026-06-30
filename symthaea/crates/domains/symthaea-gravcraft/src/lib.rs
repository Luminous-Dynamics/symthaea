// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-gravcraft — Non-Inertial Metric Perturbation Embodiment
//!
//! Unlike all other Symthaea embodiment platforms (flight, vehicle, helicopter,
//! AUV, humanoid, manipulator) which apply **forces** to move through space,
//! the gravcraft applies **metric perturbations** to spacetime itself.
//!
//! The craft doesn't accelerate — spacetime moves around it.
//!
//! ## Architecture
//!
//! - 3 gravity amplifiers, each with (amplitude, focal_distance, azimuth, elevation)
//! - Amplifiers modify local metric g_μν via Alcubierre-like shaping functions
//! - Position updates via geodesic equation: dx/dt = v, dv/dt = -Γ^i_jk v^j v^k
//! - Consciousness (Phi) directly modulates maximum metric perturbation amplitude
//!
//! ## Safety Gating
//!
//! - Green (Phi > 0.6): full metric authority, all 3 amplifiers
//! - Yellow (0.3-0.6): single amplifier, reduced amplitude
//! - Orange (0.1-0.3): flat metric only, drift
//! - Red (< 0.1): emergency metric neutralization

pub mod amplifier;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod simulator;
pub mod types;

pub use amplifier::*;
pub use controller::*;
pub use embodiment::*;
pub use encoder::*;
pub use simulator::*;
pub use types::*;
