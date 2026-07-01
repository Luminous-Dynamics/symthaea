// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the true Φ (Integrated Information) module.
//!
//! Split into focused sub-modules for maintainability.

// Re-export parent module items and HDC constants so child test modules
// can access them via `use super::*;`
#[allow(unused_imports)]
pub(crate) use super::*;
#[allow(unused_imports)]
pub(crate) use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub(crate) fn create_test_vectors(count: usize) -> Vec<ContinuousHV> {
    (0..count)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect()
}

#[cfg(feature = "parallel")]
mod cache_tests;
mod conceptual_tests;
mod core_tests;
mod iit4_tests;
mod invariant_tests;
mod optimized_tests;
#[cfg(feature = "parallel")]
mod parallel_tests;
mod pyphi_tests;
mod quantum_tests;
mod simd_tests;
mod spectral_mip_tests;
mod temporal_tests;
