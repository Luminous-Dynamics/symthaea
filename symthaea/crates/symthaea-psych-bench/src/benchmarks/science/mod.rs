// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Science benchmark suite.
//!
//! Benchmarks covering physical and materials science domains, validating
//! Symthaea's domain-specific science crates against known empirical data.
//!
//! Feature flags:
//! - `nuclear-benchmarks`   — enables nuclear physics benchmarks
//! - `materials-benchmarks` — enables materials design benchmarks
//! - `science-benchmarks`   — enables all of the above

#[cfg(feature = "nuclear-benchmarks")]
pub mod nuclear_physics;

#[cfg(feature = "materials-benchmarks")]
pub mod materials_design;

#[cfg(feature = "nuclear-benchmarks")]
pub use nuclear_physics::NuclearPhysicsBenchmark;

#[cfg(feature = "materials-benchmarks")]
pub use materials_design::MaterialsDesignBenchmark;
