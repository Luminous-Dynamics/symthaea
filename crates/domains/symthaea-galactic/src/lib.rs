// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-galactic — Rotation-Curve Model Comparison on SPARC
//!
//! External-validation benchmark: fits real galactic rotation curves from the
//! SPARC catalog (175 late-type galaxies; Lelli, McGaugh & Schombert 2016)
//! with competing gravity models and compares them with honest statistics:
//!
//! - **Newtonian baryonic-only** — the null model (0 free parameters)
//! - **NFW dark-matter halo** — V200, c fit per galaxy (2 free parameters)
//! - **MOND** — radial acceleration relation with universal a0 (0 free parameters)
//! - **Conformal gravity (Mannheim)** — universal linear + quadratic potential
//!   terms (0 free parameters)
//!
//! Plus an HDC+CfC+GLU residual regressor per model (pattern from
//! `symthaea-nuclear`): a correct physics model leaves unlearnable noise,
//! a wrong one leaves learnable structure.
//!
//! **Scope honesty**: this crate tests rotation-curve *phenomenology* only.
//! It says nothing about quantization, ghosts, or unitarity of any gravity
//! theory, and rotation curves are only one line of dark-matter evidence
//! (CMB, lensing, and cluster dynamics are not addressed here). See the
//! crate README for the full framing and known criticisms of each model.
//!
//! ## References
//!
//! - Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass models
//!   for 175 disk galaxies. *AJ*, 152, 157.
//! - McGaugh, S. S., Lelli, F., & Schombert, J. M. (2016). Radial acceleration
//!   relation in rotationally supported galaxies. *PRL*, 117, 201101.
//! - Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The structure of
//!   cold dark matter halos. *ApJ*, 462, 563.
//! - Mannheim, P. D., & O'Brien, J. G. (2012). Fitting galactic rotation
//!   curves with conformal gravity. *PRD*, 85, 124020.
//! - Flanagan, É. É. (2006). Fourth-order Weyl gravity. *PRD*, 74, 023002.
//!   (critique); Hobson, M. P., & Lasenby, A. N. (2021). *PRD*, 104, 064014.
//!   (critique)

pub mod constants;
pub mod encoder;
pub mod fit;
pub mod gravity_models;
pub mod hdc_residual;
pub mod sparc;
#[cfg(test)]
pub(crate) mod test_support;
pub mod validation;

pub use sparc::*;
