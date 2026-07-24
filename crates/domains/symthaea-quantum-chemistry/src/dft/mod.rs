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
//! - **LDA** (Local Density Approximation): Slater exchange + VWN correlation --
//!   the only functional `kohn_sham_dft`'s self-consistent SCF loop supports
//!   (`XcFunctional` has a single `Lda` variant).
//! - **PBE exchange** (Perdew-Burke-Ernzerhof, 1996) -- Phase Q5d, 2026-07-17:
//!   the enhancement-factor formula is implemented (`pbe::PbeExchange`,
//!   constants fetched from libxc, not memorized) and can be evaluated
//!   *post-hoc* on an already-converged density
//!   (`xc::pbe_exchange_energy_posthoc`), but is **not** wired into the SCF
//!   loop as a self-consistent functional -- that needs an additional
//!   gradient-coupling term in the Fock-matrix build, not yet built. PBE
//!   *correlation* is not implemented at all. This doc previously claimed
//!   PBE as a full "Available Functional" alongside LDA before this
//!   distinction existed -- corrected here, same overclaim class Q0 fixed
//!   in the crate root's `lib.rs` doc but had missed in this module's own.

pub mod grid;
pub mod lda;
pub mod pbe;
pub mod xc;

pub use grid::{DftGrid, GridPoint};
pub use lda::{SlaterExchange, VwnCorrelation, lda_exchange_correlation};
pub use pbe::PbeExchange;
pub use xc::{DftConfig, DftResult, XcFunctional, kohn_sham_dft, pbe_exchange_energy_posthoc};
