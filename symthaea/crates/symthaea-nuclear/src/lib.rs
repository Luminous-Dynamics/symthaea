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

pub mod ame2020;
pub mod constants;
pub mod deformation;
pub mod discovery;
pub mod duflo_zuker;
pub mod encoder;
pub mod ml_mass;
pub mod island_stability;
pub mod mass_formula;
pub mod shell_model;

pub use ame2020::*;
pub use constants::*;
pub use deformation::*;
pub use duflo_zuker::*;
pub use discovery::*;
pub use encoder::*;
pub use island_stability::*;
pub use mass_formula::*;
pub use shell_model::*;
