// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-coupled surgical robot — sub-mm RCM-constrained precision.
//!
//! Safety tiers (shared [`MotorSafetyLevel`](symthaea_core::embodiment::MotorSafetyLevel)
//! contract, surgical-specific gain via `types::surgical_torque_gain` /
//! `types::surgical_cautery_allowed`): Green=full torque authority + cautery
//! allowed, Yellow=40% torque gain (no cautery), Orange=freeze (zero torque,
//! no cautery), Red=Retract SafeFallback (withdraw tool tip, close jaw,
//! disable cautery — see `embodiment::SurgicalEmbodiment`'s `SafeFallback`
//! impl). Consent violation from the ethics engine also forces the full
//! Retract pose even at Orange (a stronger signal than phi alone).
//!
//! CORRECTION (2026-07, SafeFallback audit): this doc previously claimed
//! physical force/velocity limits ("Green=5N/50mm/s, Yellow=2N/20mm/s").
//! That was never what the code enforced — verified against `simulator.rs`,
//! there is no Newton or mm/s clamp anywhere in the physics step; only the
//! abstract per-tier `torque_gain` above is applied to commanded joint
//! torques. Fixing the doc to match the code rather than retrofitting real
//! N/mm-s force-limiting into the simulator's physics, which would need a
//! genuine impedance/admittance model (a bigger, higher-risk change than
//! this pass's scope) — flagging this as a known gap, not silently
//! papering over it.
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
