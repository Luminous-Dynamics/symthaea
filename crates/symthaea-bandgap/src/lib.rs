// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-bandgap — DFT Bandgap Correction via Delta-Learning
//!
//! Applies the same physics-baseline + ML-correction architecture used for
//! nuclear masses and protein folding to semiconductor bandgap prediction.
//!
//! | Nuclear Physics | Materials Science |
//! |-----------------|-------------------|
//! | DZ10 (10 params) | Electronegativity model |
//! | (Z,N) features | Composition features |
//! | Random Forest correction | Random Forest correction |
//! | AME2020 training data | Zhuo et al. 2020 (PBE+HSE06) |
//!
//! ## References
//!
//! - Zhuo, Y. et al. (2020). Predicting the Band Gaps of Inorganic Solids by
//!   Machine Learning. *J. Phys. Chem. Lett.*, 9, 1668-1673.
//! - Borlido, P. et al. (2019). Large-Scale Benchmark of Exchange-Correlation
//!   Functionals for the Determination of Electronic Band Gaps of Solids.
//!   *J. Chem. Theory Comput.*, 15, 5069-5079.
//! - Rajan, A.C. et al. (2018). Machine-Learning-Assisted Accurate Band Gap
//!   Predictions of Functionalized MXene. *Chem. Mater.*, 30, 4031-4038.

pub mod bandgap_baseline;
mod benchmark;
pub mod features;
pub mod ml_bandgap;
pub mod periodic_table;
pub mod training_data;
mod transfer;
