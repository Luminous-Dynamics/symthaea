// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Crucible: Extreme Ethical Stress Testing
//!
//! The Crucible subjects Symthaea's moral systems to conditions where standard
//! human ethics break down: first contact dilemmas, infinite resource allocation,
//! triage under scarcity, adversarial semantic attacks, and combinatorial
//! property-based fuzzing.
//!
//! If the math holds under the gravity of a black hole, managing a local water
//! grid is computationally trivial.
//!
//! ## Architecture
//!
//! - **Harness**: Multi-cycle simulation runner that feeds scenarios through the
//!   moral algebra, harmony basis, and moral topology pipelines.
//! - **Invariants**: Universal properties that must hold for ALL inputs.
//! - **Scenarios**: Named thought experiments and combinatorial generators.
//! - **Telemetry**: Structured JSON output for every run.

#![deny(unsafe_code)]

pub mod baselines;
pub mod harness;
pub mod invariants;
pub mod scenarios;
pub mod telemetry;

#[cfg(test)]
mod proptest_invariants;
