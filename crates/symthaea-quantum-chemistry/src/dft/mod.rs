// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Density Functional Theory (DFT) — Kohn-Sham SCF.
//!
//! Replaces exact HF exchange with approximate exchange-correlation functionals
//! evaluated on a numerical grid. Typically more accurate than HF for molecules.
//!
//! ## Available Functionals
//!
//! - **LDA** (Local Density Approximation): Slater exchange + VWN correlation
//! - **PBE** (Generalized Gradient Approximation): Perdew-Burke-Ernzerhof (1996)

pub mod grid;
pub mod lda;
pub mod xc;

pub use grid::{DftGrid, GridPoint};
pub use lda::{SlaterExchange, VwnCorrelation, lda_exchange_correlation};
pub use xc::{DftConfig, DftResult, XcFunctional, kohn_sham_dft};
