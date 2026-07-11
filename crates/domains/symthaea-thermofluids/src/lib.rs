// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-thermofluids
//!
//! Applied fluid & thermal engineering for Symthaea, completing the engineering
//! stack alongside `symthaea-structural` (mechanics) and `symthaea-circuits`
//! (electrical). The core physics crates have Navier-Stokes and thermodynamics,
//! but no applied hydraulics/heat-transfer layer.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs textbook.
//!
//! ## Scope
//!
//! - [`fluids`]: Reynolds number & regime, Bernoulli head, Darcy-Weisbach head
//!   loss, continuity.
//! - [`thermal`]: Carnot efficiency, Fourier conduction, Newton cooling, engine
//!   work.
//!
//! ## Example
//!
//! ```
//! use symthaea_thermofluids::{fluids, thermal};
//! assert_eq!(fluids::flow_regime(1000.0), fluids::Regime::Laminar);
//! assert!((thermal::carnot_efficiency(300.0, 600.0) - 0.5).abs() < 1e-12);
//! ```

pub mod fluids;
pub mod thermal;
