// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-runtime
//!
//! Unified foundation for Symthaea's holographic liquid brain.
//! Consolidates HDC math, simulation architecting, and causal grounding.

pub mod formal;
pub mod hdc;
pub mod school;

pub use hdc::probabilistic_hv::{PHVConfig, PHVSimilarity, ProbabilisticHV};
pub use hdc::unified_hv::{BinaryHV, ContinuousHV, HDC_DIMENSION, HV};

// Re-export core genesis for convenience
pub use formal::smt_serializer;
pub use formal::z3_bridge;
pub use symthaea_core::genesis::GenesisSeed;
