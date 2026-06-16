// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-nuclear — Computational Nuclear Structure
//!
//! Teaches Symthaea nuclear physics from first principles:
//!
//! - **Semi-Empirical Mass Formula** (Bethe-Weizsäcker): binding energies,
//!   mass excess, beta-stability, alpha-decay Q-values, Geiger-Nuttall half-lives
//! - **Woods-Saxon Shell Model**: Numerov radial Schrödinger solver, single-particle
//!   levels, magic number detection, Strutinsky shell corrections
//! - **Island of Stability**: Combined liquid-drop + shell corrections for superheavy
//!   elements, Moscovium (Z=115) isotope evaluation
//! - **HDC Encoder**: Nuclear state → ContinuousHV for integration with the
//!   physics-bridge catalog and cognitive loop
//!
//! ## References
//!
//! - Krane, K. S. (1988). *Introductory Nuclear Physics*. Wiley.
//! - Möller et al. (2016). Nuclear mass table FRDM(2012). *Atomic Data and Nuclear Data Tables*.
//! - Oganessian & Utyonkov (2015). Superheavy element synthesis. *Nuclear Physics A*.
//! - Ring & Schuck (2004). *The Nuclear Many-Body Problem*. Springer.

mod active_learning;
pub mod ame2020;
mod blind_validation;
mod conformal;
pub mod constants;
mod deep_analysis;
pub mod deformation;
mod discoveries;
mod discoveries2;
mod discoveries3;
pub mod discovery;
pub mod duflo_zuker;
pub mod element_embeddings;
pub mod encoder;
mod exploration;
mod figure_data;
pub mod fission_barrier;
pub mod fundamental;
mod hard_validation;
pub mod hdc_mass;
pub mod hdc_strategies;
mod hdc_validation;
pub mod hfb;
pub mod island_stability;
pub mod isotope_properties;
pub mod mass_formula;
pub mod medical_isotopes;
pub mod ml_mass;
pub mod nuclear_forensics;
pub mod reactor;
pub mod rprocess;
pub mod shell_model;
pub mod space_nuclear;
mod symbolic_regression;

pub use ame2020::*;
pub use constants::*;
pub use deformation::*;
pub use discovery::*;
pub use duflo_zuker::*;
pub use encoder::*;
pub use island_stability::*;
pub use mass_formula::*;
pub use shell_model::*;
