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
//!
//! The `space_exosuit` supervisory module deliberately adds an independent
//! deterministic assist envelope for EVA research. Phi may influence proposed
//! assistance elsewhere, but it is not an input to that certified-assist
//! admission boundary. Rescue propulsion is likewise isolated behind its own
//! deterministic authority, navigation-validity, delta-v, and no-fire gates.

#![deny(unsafe_code)]

pub mod control_modes;
pub mod controller;
pub mod durability_coupling;
pub mod dust;
pub mod embodiment;
pub mod encoder;
pub mod eva_fidelity_coupling;
pub mod eva_mission;
pub mod fep_agent;
pub mod fault_campaign;
#[cfg(feature = "symtropy")]
pub mod full_frame;
#[cfg(feature = "symtropy")]
pub mod full_frame_environment;
pub mod glove_benchmark;
pub mod glove_dust;
#[cfg(feature = "hal")]
pub mod hal_bridge;
pub mod integrated_adversarial_campaign;
pub mod metabolism;
pub mod perturbations;
pub mod plss;
pub mod plugin;
pub mod power;
pub mod powered_glove;
pub mod pressure_garment;
pub mod pressure_integrity;
pub mod pressure_integrity_fault_campaign;
pub mod pressure_response;
pub mod radiation;
pub mod reduced_gravity;
pub mod reflex;
pub mod rescue_benchmark;
pub mod rescue_fault_campaign;
pub mod rescue_navigation;
pub mod rescue_propulsion;
pub mod resource_optimizer;
pub mod resource_transfer;
pub mod safe_haven;
#[cfg(feature = "sensors")]
pub mod sensored_suite;
pub mod simulator;
pub mod space_exosuit;
pub mod suit_service_fault_campaign;
pub mod suit_service_interface;
pub mod suit_service_session;
pub mod suitport_state_machine;
#[cfg(feature = "symtropy")]
pub mod symtropy_sim;
pub mod training;
pub mod types;
pub mod workout;
