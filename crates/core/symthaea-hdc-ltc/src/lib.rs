// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-ltc
//!
//! Hyperdimensional recurrent state with solver-free liquid-time evolution.
//!
//! The crate deliberately separates three research surfaces:
//!
//! - [`ContinuousHV`] carries arbitrary continuous distributed state.
//! - [`UnitaryRole`] carries reversible real HDC roles with components in
//!   `{ -1, +1 }`, so role binding is an isometry.
//! - [`HolographicLiquidCell`] is a theorem-bearing research cell whose temporal
//!   update is constructed to commute with `UnitaryRole` binding when state and
//!   input are transformed by the same role.
//!
//! The legacy [`HdcLtcUnifiedNeuron`] remains available so the algebraic research
//! path can be qualified without silently changing production behavior.

pub mod config;
pub mod continuous_hv;
pub mod holographic_liquid;
pub mod network;
pub mod neuron;

pub use config::{Activation, NetworkConfig, NeuronConfig};
pub use continuous_hv::{ContinuousHV, HDC_DIMENSION, UnitaryRole};
pub use holographic_liquid::{HlsActivation, HlsConfig, HlsError, HolographicLiquidCell};
pub use network::{HdcLtcUnifiedNetwork, StepTimingConfig};
pub use neuron::HdcLtcUnifiedNeuron;
