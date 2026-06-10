// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Master Consciousness Equation
//!
//! Thin re-export of [`symthaea_consciousness_equation`]. The implementation
//! lives in `crates/symthaea-consciousness-equation/` for build-time isolation.
//!
//! **C(t) = σ(softmin(Φ, B, W, A, R, E, K, M, N, Soc; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)**

pub use symthaea_consciousness_equation::*;
