// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-physiology
//!
//! Human physiology for Symthaea: nutrition/metabolism and pharmacokinetics.
//! Complements the clinical crates (`symthaea-clinical`, `symthaea-therapeutic`)
//! with the quantitative energy-balance and drug-kinetics layer they lacked.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs textbook.
//!
//! ## Scope
//!
//! - [`nutrition`]: BMI, BMR (Mifflin–St Jeor), TDEE, macronutrient energy.
//! - [`pharmacokinetics`]: half-life ↔ rate constant, one-compartment
//!   concentration decay, clearance.
//!
//! ## Example
//!
//! ```
//! use symthaea_physiology::{nutrition::bmi, pharmacokinetics::concentration_at};
//! use symthaea_physiology::pharmacokinetics::elimination_rate_constant;
//! assert!((bmi(70.0, 1.75) - 22.857).abs() < 1e-3);
//! let ke = elimination_rate_constant(4.0);            // 4 h half-life
//! assert!((concentration_at(100.0, ke, 4.0) - 50.0).abs() < 1e-6);
//! ```

pub mod nutrition;
pub mod pharmacokinetics;
