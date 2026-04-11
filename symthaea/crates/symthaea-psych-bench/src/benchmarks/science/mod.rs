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

/// ODE Chaos & Stiffness — van der Pol (stiff μ=1000), Lorenz (chaotic),
/// Robertson chemistry (very stiff, conservation law), coupled oscillators
/// (energy conservation). Uses inline Dormand-Prince RK45.
/// Always enabled; requires only symthaea-core.
pub mod ode_chaos;
pub use ode_chaos::OdeChaosBenchmark;

/// Chemistry — molar mass, Hess's Law, Gibbs free energy, ICE tables,
/// Arrhenius kinetics. Tests stoichiometry and thermochemistry against
/// NIST-JANAF values. Always enabled; requires only symthaea-core.
pub mod chemistry;
pub use chemistry::ChemistryBenchmark;
