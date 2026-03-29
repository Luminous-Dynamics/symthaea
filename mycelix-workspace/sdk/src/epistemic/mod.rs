// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Charter v2.0 with GIS v4.0 Integration
//!
//! Classification system for truth claims used across the Mycelix ecosystem
//! (hApps, SDKs, and external integrations such as Symthaea-HLB).
//!
//! ## 3D Classification (E/N/M)
//! - **E-Axis (Empirical)**: How to verify the claim (E0-E4)
//! - **N-Axis (Normative)**: Who agrees with the claim (N0-N3)
//! - **M-Axis (Materiality)**: How long the claim matters (M0-M3)
//!
//! ## 4D Extended Classification (E/N/M/H) - GIS v4.0
//! - **H-Axis (Harmonic)**: Resonance with Kosmic Song consciousness metrics (H0-H4)

mod claims;
mod cube;

pub use claims::{ClaimBuilder, EpistemicClaim};
pub use cube::{
    EmpiricalLevel, EpistemicClassification, EpistemicClassificationExtended, HarmonicLevel,
    MaterialityLevel, NormativeLevel,
};
