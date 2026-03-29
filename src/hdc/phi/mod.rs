// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi (Φ) - Integrated Information Theory calculations
//!
//! Re-exports phi-related types from symthaea-core.

// Re-export phi types from symthaea-core
pub use symthaea_core::phi_engine::{
    ApproximationTier, CacheStats, CachedPhiEngine, ContinuousPhiCalculator, PhiCalculator,
    PhiEngine, PhiMethod, PhiResult, TieredPhi, TieredPhiConfig,
};
