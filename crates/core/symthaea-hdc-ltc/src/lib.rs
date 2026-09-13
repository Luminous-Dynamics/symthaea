// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-ltc
//!
//! Hyperdimensional recurrent state with solver-free liquid-time evolution.
//!
//! The crate deliberately separates two kinds of distributed object:
//!
//! - [`ContinuousHV`] carries arbitrary continuous state and learned/modulatory
//!   fields.
//! - [`UnitaryRole`] carries reversible real HDC roles with components in
//!   `{ -1, +1 }`, so role binding is norm- and inner-product-preserving.
//!
//! A neuron update is O(D) in the hypervector dimension and independent of the
//! number of numerical ODE substeps associated with the elapsed `dt`.
//!
//! ## Quick Start
//!
//! ```rust
//! use symthaea_hdc_ltc::{
//!     ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig, UnitaryRole,
//! };
//!
//! let config = NeuronConfig { dim: 1024, ..NeuronConfig::default() };
//! let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
//! let input = ContinuousHV::new_random(1024, 123);
//!
//! neuron.evolve_closed_form(0.1, &input);
//!
//! let role = UnitaryRole::new(1024, 7);
//! let bound = role.bind(neuron.state());
//! let recovered = role.unbind(&bound);
//! assert_eq!(&recovered, neuron.state());
//! ```

pub mod config;
pub mod continuous_hv;
pub mod network;
pub mod neuron;

pub use config::{Activation, NetworkConfig, NeuronConfig};
pub use continuous_hv::{ContinuousHV, HDC_DIMENSION, UnitaryRole};
pub use network::{HdcLtcUnifiedNetwork, StepTimingConfig};
pub use neuron::HdcLtcUnifiedNeuron;
