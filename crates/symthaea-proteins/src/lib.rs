// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-proteins — Protein Secondary Structure Prediction
//!
//! Applies the same physics-baseline + ML-correction architecture used for
//! nuclear mass prediction to protein structure. The analogy:
//!
//! | Nuclear Physics | Protein Folding |
//! |-----------------|-----------------|
//! | DZ10 (10 params) | Chou-Fasman propensity tables |
//! | (Z,N) features | Sliding window features |
//! | Random Forest correction | Random Forest correction |
//! | AME2020 training data | CB513 benchmark proteins |
//!
//! ## References
//!
//! - Chou, P.Y. & Fasman, G.D. (1978). Prediction of the secondary structure
//!   of proteins from their amino acid sequence. *Adv. Enzymol.*, 47, 45-148.
//! - Rost, B. & Sander, C. (1993). Prediction of protein secondary structure
//!   at better than 70% accuracy. *J. Mol. Biol.*, 232, 584-599.
//! - Kyte, J. & Doolittle, R.F. (1982). A simple method for displaying the
//!   hydropathic character of a protein. *J. Mol. Biol.*, 157, 105-132.

pub mod amino_acid;
pub mod benchmark;
pub mod cb513;
pub mod chou_fasman;
pub mod features;
pub mod rf_ssp;
