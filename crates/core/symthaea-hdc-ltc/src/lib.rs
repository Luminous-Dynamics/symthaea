// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-ltc
//!
//! Hyperdimensional Liquid Time-Constant neural network with O(1) closed-form
//! temporal evolution.
//!
//! ## Core Innovation
//!
//! Traditional neural networks use weight *matrices* (O(D^2) parameters) and
//! require ODE integration for temporal dynamics (O(steps) per time jump).
//!
//! This crate replaces both:
//! - **Weight matrices** become weight *hypervectors* via HDC binding (element-wise multiply)
//! - **ODE integration** becomes a closed-form exponential interpolation (O(1) per jump)
//!
//! The result: a single neuron with a 16,384-dimensional state that can jump to
//! *any* time horizon in O(D) operations.
//!
//! ## Mathematical Basis
//!
//! The neuron ODE:
//! ```text
//! dx/dt = (-x + f(W . x + U . u)) / tau(||x||)
//! ```
//!
//! Has the closed-form solution:
//! ```text
//! x(t + dt) = sigma * x_inf + (1 - sigma) * x(t)
//! ```
//!
//! Where `x_inf = f(W . x + U . u)` is the equilibrium, and `sigma` is an
//! adaptive gating factor that depends on dt, state, and input.
//!
//! ## Quick Start
//!
//! ```rust
//! use symthaea_hdc_ltc::{ContinuousHV, NeuronConfig, HdcLtcUnifiedNeuron};
//!
//! let config = NeuronConfig { dim: 1024, ..NeuronConfig::default() };
//! let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
//!
//! let input = ContinuousHV::new_random(1024, 123);
//! neuron.evolve_closed_form(0.1, &input);   // 100 ms jump
//! neuron.evolve_closed_form(100.0, &input);  // 100 second jump (same cost!)
//! ```

pub mod config;
pub mod continuous_hv;
pub mod network;
pub mod neuron;

pub use config::{Activation, NetworkConfig, NeuronConfig};
pub use continuous_hv::{ContinuousHV, HDC_DIMENSION};
pub use network::{HdcLtcUnifiedNetwork, StepTimingConfig};
pub use neuron::HdcLtcUnifiedNeuron;
