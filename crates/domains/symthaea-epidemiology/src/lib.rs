// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-epidemiology
//!
//! A self-contained epidemiology layer for Symthaea, starting with the SIR
//! compartment model. The workspace had clinical/medical crates but no
//! infectious-disease dynamics.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form results
//! (R₀, herd-immunity threshold, final size) plus a conservative time-stepped
//! simulation, checked against known values.
//!
//! ## Scope (v0.1)
//!
//! - SIR model: basic reproduction number, herd-immunity threshold, final
//!   epidemic size, Euler simulation with peak tracking.
//!
//! Not yet: SEIR (exposed compartment), age structure, spatial/network models.
//!
//! ## Example
//!
//! ```
//! use symthaea_epidemiology::Sir;
//! let flu = Sir { beta: 0.3, gamma: 0.1 };      // R0 = 3
//! assert!((flu.basic_reproduction_number() - 3.0).abs() < 1e-12);
//! assert!((flu.herd_immunity_threshold() - 0.6667).abs() < 1e-3);
//! assert!(flu.final_size() > 0.9);              // ~94% eventually infected
//! ```

pub mod sir;

pub use sir::{Sir, State};
