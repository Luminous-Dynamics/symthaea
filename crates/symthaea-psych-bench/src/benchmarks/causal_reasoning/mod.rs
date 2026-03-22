// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Reasoning benchmarks.
//!
//! Tests HDC-based causal reasoning using composition algebra operations:
//! intervention effects, confound detection, and causal chain inference.

pub mod causal_chain;
pub mod confound_detection;
pub mod intervention;

pub use causal_chain::CausalChainBenchmark;
pub use confound_detection::ConfoundDetectionBenchmark;
pub use intervention::InterventionEffectBenchmark;
