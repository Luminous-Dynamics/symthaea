// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-coupled surgical robot (early stage).
//!
//! RCM STATUS (2026-07, Tier 2.6 — updates the earlier correction note):
//! a remote-center-of-motion constraint about the trocar fulcrum is now
//! implemented and consumed by the physics step. Be precise about what it
//! is: a **soft spring penalty**, not a hard kinematic constraint. The tool
//! shaft is modeled as the mount→tip segment; where it crosses the trocar
//! port plane (`SurgicalConfig::trocar_port_z`, lateral anchor fixed from
//! the home pose), a task-space spring `F = -rcm_stiffness · d` (N/m) is
//! mapped through the finite-difference Jacobian into joint torques pulling
//! the shaft back through the pivot. Residual lateral port displacement
//! therefore scales with applied torque / stiffness (a few mm at defaults) —
//! this is measurably bounded (see `simulator::tests::
//! test_rcm_bounds_port_displacement`) but it is NOT "sub-mm RCM-constrained
//! precision"; do not restore that claim.
//!
//! GEOMETRIC SAFETY CHANNELS (2026-07, Tier 2.6): `critical_structure_distance`
//! and `trocar_compliance` were previously scripted sinusoids/heuristics.
//! They are now derived from actual kinematics: tip-to-structure Euclidean
//! distance against `SurgicalConfig::critical_structure`, and normalized
//! lateral shaft displacement at the port. The FEP anomaly detector
//! (`fep_agent`) therefore keys off real state.
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
//! VELOCITY LIMITS (2026-07, Tier 2.6 — partially restores a claim retracted
//! by the SafeFallback audit): per-tier **tip velocity** limits are now
//! physically enforced in the simulator step (Green 50 mm/s, Yellow 20 mm/s,
//! Orange/Red 5 mm/s — joint velocities are uniformly scaled whenever the
//! kinematic tip velocity would exceed the tier cap; see
//! `types::surgical_tip_speed_limit`). Per-tier **force** limits (the old
//! "Green=5N/Yellow=2N" claim) remain NOT enforced — that would need a real
//! impedance/admittance model, still a known gap.
//!
//! Gravity: this simplified joint-space model has NO gravity term (links are
//! treated as gravity-balanced, as in a counterweighted surgical arm);
//! documented absence, not an oversight.
#![deny(unsafe_code)]
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod plugin;
pub mod reflex;
pub mod simulator;
pub mod training;
pub mod types;
