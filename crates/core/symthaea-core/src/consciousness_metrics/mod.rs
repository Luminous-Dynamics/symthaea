// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # True Integrated Information (Φ) via Shannon Entropy
//!
//! This module implements mathematically rigorous Integrated Information Theory (IIT)
//! using actual Shannon entropy measures rather than similarity-based proxies.
//!
//! ## Key Insight
//!
//! ContinuousHV has 16,384 f32 components in [-1, 1]. We treat these as empirical samples
//! and use discretization-based entropy estimation:
//!
//! 1. **Bin components** into K buckets (K=16 or 32)
//! 2. **Build histogram** → probability distribution
//! 3. **Compute Shannon entropy**: H(X) = -Σ p(x) log₂ p(x)
//! 4. **Joint entropy** via 2D binning for H(X,Y)
//! 5. **Mutual information**: I(X;Y) = H(X) + H(Y) - H(X,Y)
//!
//! ## True Φ Calculation
//!
//! ```text
//! Φ = EI(System) - EI(MIP)
//!
//! Where:
//! - EI(System) = Effective Information of whole system
//! - EI(MIP) = Effective Information of Minimum Information Partition
//! - Φ > 0 indicates true integration (cannot be decomposed)
//! ```
//!
//! ## Scientific Basis
//!
//! - Shannon (1948) - "A Mathematical Theory of Communication"
//! - Tononi et al. (2016) - "Integrated Information Theory: From Consciousness to Its Physical Substrate"
//! - Oizumi et al. (2014) - "From the Phenomenology to the Mechanisms of Consciousness"

mod approximate;
mod bounds;
mod calculator;
mod conceptual;
mod consciousness_core;
mod entropy;
mod iit4;
#[cfg(feature = "parallel")]
mod parallel;
mod quantum;
mod simd;
mod spectral_mip;
mod streaming;
mod synergistic;
mod temporal;
mod types;

// Re-export all public items from submodules for backward compatibility
pub use approximate::*;
pub use bounds::*;
pub use calculator::*;
pub use conceptual::*;
pub use consciousness_core::*;
pub use entropy::*;
pub use iit4::*;
#[cfg(feature = "parallel")]
pub use parallel::*;
pub use quantum::*;
pub use simd::*;
pub use spectral_mip::*;
pub use streaming::*;
pub use synergistic::*;
pub use temporal::*;
pub use types::*;

#[cfg(test)]
mod tests;
