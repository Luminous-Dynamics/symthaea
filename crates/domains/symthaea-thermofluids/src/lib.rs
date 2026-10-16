// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-thermofluids
//!
//! Applied fluid & thermal engineering for Symthaea, completing the engineering
//! stack alongside `symthaea-structural` (mechanics) and `symthaea-circuits`
//! (electrical). The core physics crates have Navier-Stokes and thermodynamics,
//! but no applied hydraulics/heat-transfer layer.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against
//! textbook reference values.
//!
//! ## Scope
//!
//! - [`fluids`]: Reynolds number & regime, Bernoulli head, Darcy-Weisbach head
//!   loss, continuity.
//! - [`thermal`]: Carnot efficiency, Fourier conduction, Newton cooling, engine
//!   work.
//! - [`cycles`]: validated ideal Carnot, Brayton, Otto, and Diesel cycle
//!   efficiency baselines for analytical screening, not real-engine predictions.
//! - [`energy_balance`]: steady-state first-law closure and gross/net efficiency
//!   accounting with an explicit system boundary and declared tolerance.
//! - [`energy_interval`]: transient interval-integrated first-law closure,
//!   including signed stored-energy change, plus per-interval-validated duty-cycle
//!   aggregation. No engine efficiency is reported when storage is not charge-sustaining.
//!
//! ## Example
//!
//! ```
//! use symthaea_thermofluids::{cycles, energy_balance, energy_interval, fluids, thermal};
//! assert_eq!(fluids::flow_regime(1000.0), fluids::Regime::Laminar);
//! assert!((thermal::carnot_efficiency(300.0, 600.0) - 0.5).abs() < 1e-12);
//! assert!((cycles::carnot_efficiency(300.0, 600.0).unwrap() - 0.5).abs() < 1e-12);
//! let report = energy_balance::EnergyBalance {
//!     input_power_w: 1000.0,
//!     net_useful_output_power_w: 300.0,
//!     auxiliary_power_w: 20.0,
//!     rejected_or_loss_power_w: 680.0,
//! }.evaluate(1e-9).unwrap();
//! assert!((report.net_efficiency - 0.30).abs() < 1e-12);
//! let interval = energy_interval::EnergyIntervalBalance {
//!     input_energy_j: 800.0,
//!     net_useful_output_energy_j: 300.0,
//!     auxiliary_energy_j: 20.0,
//!     rejected_or_loss_energy_j: 680.0,
//!     stored_energy_change_j: -200.0,
//! }.evaluate(1e-9).unwrap();
//! assert!((interval.energy_throughput_j - 1000.0).abs() < 1e-12);
//! ```

pub mod cycles;
pub mod energy_balance;
pub mod energy_interval;
pub mod fluids;
pub mod thermal;
