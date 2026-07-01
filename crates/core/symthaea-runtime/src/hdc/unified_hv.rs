// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Hypervector Types (Runtime)
//!
//! Re-exports and extends core HDC types for the Symthaea runtime.

pub use symthaea_core::hdc::simd_continuous;
pub use symthaea_core::hdc::{BinaryHV, ContinuousHV, HDC_DIMENSION, HV};

// Note: STRIDE is an internal implementation detail of symthaea-core's
// non-SIMD paths. We expose control functions here.

/// Set the global cognitive stride for similarity calculations.
pub fn set_cognitive_stride(stride: usize) {
    symthaea_core::hdc::unified_hv::STRIDE
        .store(stride.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// Get the current cognitive stride.
pub fn get_cognitive_stride() -> usize {
    symthaea_core::hdc::unified_hv::STRIDE.load(std::sync::atomic::Ordering::Relaxed)
}
