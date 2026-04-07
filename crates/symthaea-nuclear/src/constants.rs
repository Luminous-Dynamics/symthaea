// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear physics constants for structure calculations.
//!
//! Constants are sourced from Krane (1988), Möller et al. (2016), and CODATA 2022.

// ── Semi-Empirical Mass Formula (Weizsäcker) ───────────────────────────────

/// Volume coefficient (MeV) — bulk nuclear matter binding.
/// Rohlf (1994) parametrization, optimized for heavy nuclei.
pub const A_VOLUME: f64 = 15.56;

/// Surface coefficient (MeV) — reduced binding at surface
pub const A_SURFACE: f64 = 17.23;

/// Coulomb coefficient (MeV) — proton-proton repulsion
pub const A_COULOMB: f64 = 0.7;

/// Asymmetry coefficient (MeV) — Pauli exclusion N≈Z preference
pub const A_ASYMMETRY: f64 = 23.285;

/// Pairing coefficient (MeV) — even-even/odd-odd energy difference
pub const A_PAIRING: f64 = 12.0;

// ── Woods-Saxon Potential ──────────────────────────────────────────────────

/// Potential depth (MeV)
pub const V0_WOODS_SAXON: f64 = 50.0;

/// Nuclear radius parameter (fm) — R = r₀·A^(1/3)
pub const R0_FM: f64 = 1.25;

/// Surface diffuseness (fm)
pub const DIFFUSENESS_FM: f64 = 0.65;

/// Spin-orbit coupling strength relative to V₀
pub const SPIN_ORBIT_FACTOR: f64 = 0.44;

// ── Physical Constants (nuclear units) ─────────────────────────────────────

/// ℏ²/(2m_nucleon) in MeV·fm² — kinetic energy prefactor for Schrödinger equation
/// = (ℏc)² / (2·m_N·c²) = (197.327)² / (2·938.918) ≈ 20.736 MeV·fm²
pub const HBAR2_OVER_2M: f64 = 20.736;

/// ℏc in MeV·fm
pub const HBAR_C: f64 = 197.327;

/// Nucleon mass (average of p,n) in MeV/c²
pub const M_NUCLEON_MEV: f64 = 938.918;

/// Alpha particle binding energy (MeV)
pub const B_ALPHA: f64 = 28.296;

// ── Numerov Solver Parameters ──────────────────────────────────────────────

/// Radial grid step size (fm) — balance accuracy vs. performance
pub const DR_DEFAULT: f64 = 0.05;

/// Maximum radial extent (fm) for integration
pub const R_MAX_FM: f64 = 20.0;

/// Maximum iterations for eigenvalue bisection
pub const MAX_BISECTION_ITER: usize = 100;

/// Energy convergence tolerance (MeV)
pub const ENERGY_TOLERANCE: f64 = 1e-4;

// ── Strutinsky Smoothing ───────────────────────────────────────────────────

/// Smoothing width (MeV) for Strutinsky averaging
pub const GAMMA_STRUTINSKY: f64 = 1.2;

/// Correction polynomial order for Strutinsky method
pub const STRUTINSKY_ORDER: usize = 6;
