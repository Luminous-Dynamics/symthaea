// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-cosmogenesis
//!
//! Algorithmic semantic manifold organization via ΛCDM-inspired cosmological dynamics.

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod simulator;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod types;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod analyzer;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod complexity;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod phase_detector;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod consolidation;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod evaporator;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod metabolism;

#[cfg(feature = "cognitive-cosmogenesis")]
pub mod resonance;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use simulator::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use types::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use analyzer::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use complexity::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use phase_detector::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use consolidation::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use evaporator::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use metabolism::*;

#[cfg(feature = "cognitive-cosmogenesis")]
pub use resonance::*;
