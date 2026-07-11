// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-ecology
//!
//! Analytic population ecology — logistic growth, the Lotka-Volterra
//! predator-prey ODEs, and competitive Lotka-Volterra coexistence.
//!
//! **Non-duplication:** this is the closed-form / differential-equation
//! counterpart to `symthaea-alife`'s *agent-based* predator-prey simulation
//! (which derives interaction rates from real per-agent choices). Here the rates
//! are model parameters and the results are analytic (equilibria, conserved
//! quantities, stability conditions). `symthaea-population` is population
//! *genetics*, and `symthaea-biota` is a sanctuary platform — neither models
//! ecological dynamics.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against
//! closed forms (logistic asymptote, LV equilibrium & first integral, the
//! competitive-exclusion condition).
//!
//! ## Example
//!
//! ```
//! use symthaea_ecology::LotkaVolterra;
//! let m = LotkaVolterra { alpha: 1.0, beta: 0.1, delta: 0.075, gamma: 1.5 };
//! // Coexistence equilibrium (γ/δ, α/β) = (20, 10).
//! let (prey, pred) = m.equilibrium();
//! assert!((prey - 20.0).abs() < 1e-12 && (pred - 10.0).abs() < 1e-12);
//! ```

pub mod competition;
pub mod logistic;
pub mod lotka_volterra;

pub use competition::Competition;
pub use lotka_volterra::LotkaVolterra;
