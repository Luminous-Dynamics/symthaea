// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-numerical
//!
//! Pure-`std` numerical methods for the zero-dependency domain-crate ecosystem —
//! the shared home for routines that were being reinvented per-crate (RK4 in
//! `symthaea-ecology` and in the orbital-mechanics code; ad-hoc root finding).
//!
//! Zero dependencies, no `symthaea-core` link. Every routine is checked against
//! a known analytic result.
//!
//! ## Contents
//! - [`roots`] — bisection, Newton-Raphson, secant
//! - [`quadrature`] — composite trapezoidal and Simpson integration
//! - [`ode`] — RK4 integration, scalar ([`ode::rk4`]) and systems
//!   ([`ode::rk4_system`])
//! - [`interpolate`] — Lagrange and piecewise-linear interpolation
//!
//! ## Example
//!
//! ```
//! use symthaea_numerical::roots::newton;
//! // √2 as the root of x² − 2.
//! let r = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-14, 100).unwrap();
//! assert!((r - 2.0_f64.sqrt()).abs() < 1e-12);
//! ```

pub mod interpolate;
pub mod ode;
pub mod quadrature;
pub mod roots;

pub use interpolate::{lagrange, linear};
pub use ode::{rk4, rk4_system};
pub use quadrature::{simpson, trapezoidal};
pub use roots::{bisection, newton, secant};
