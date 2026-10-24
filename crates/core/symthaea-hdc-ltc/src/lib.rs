// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-ltc
//!
//! Hyperdimensional Liquid Time-Constant neural network with solver-free CfC
//! updates that cost O(D) per vector step, independent of the elapsed dt.
//!
//! ## Core Innovation
//!
//! Traditional neural networks use weight *matrices* (O(D^2) parameters) and
//! require ODE integration for temporal dynamics (O(steps) per time jump).
//!
//! This crate replaces both:
//! - **Weight matrices** become weight *hypervectors* via HDC binding (element-wise multiply)
//! - **Repeated ODE substeps** are replaced by a closed-form exponential update (O(D)
//!   coordinate work per jump, independent of dt; not O(1) in hypervector dimension)
//!
//! The default state dimension is 16,384; dimensions are configurable, and the
//! scaling benchmark samples through 65,536 (64K). A temporal jump costs O(D), so
//! increasing D increases work even though elapsed time does not add substeps.
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
pub use neuron::{HdcLtcUnifiedNeuron, NeuronStepError};
