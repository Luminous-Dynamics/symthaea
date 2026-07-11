// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-control
//!
//! Classical control theory for Symthaea — PID controllers, second-order
//! response characteristics, and Routh-Hurwitz stability. Connects to the
//! robotics platforms (which have per-platform controllers but no shared
//! control-theory layer).
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form results
//! plus a closed-loop simulation, checked against textbook values.
//!
//! ## Scope
//!
//! - [`pid`]: discrete PID controller.
//! - [`second_order`]: damping regime, overshoot, settling time, damped freq.
//! - [`routh`]: Routh-Hurwitz RHP-root count / stability.
//!
//! ## Example
//!
//! ```
//! use symthaea_control_theory::{routh::is_stable, second_order::SecondOrder};
//! assert!(is_stable(&[1.0, 3.0, 2.0]));                  // (s+1)(s+2)
//! let s = SecondOrder { natural_freq: 1.0, damping_ratio: 0.5 };
//! assert!((s.percent_overshoot() - 16.303).abs() < 0.01);
//! ```

pub mod pid;
pub mod routh;
pub mod second_order;

pub use pid::Pid;
pub use routh::{is_stable, rhp_root_count};
pub use second_order::{Damping, SecondOrder};
