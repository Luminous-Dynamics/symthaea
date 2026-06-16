#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]
//! # Symthaea Population
//!
//! Population genetics with 16,384D HDC genome encoding, effective population
//! calculations, breeding strategy optimization, and Mycelix-compatible governance.
//! Features O(1) population trajectory prediction via CfC closed-form temporal jumps.
//!
//! ## Overview
//!
//! This crate provides:
//! - **HDC Genetics**: Encode genomes as 16,384D hypervectors using bind/bundle operations
//! - **Effective Population**: Ne calculations (sex ratio, family variance, fluctuating)
//! - **Diversity Metrics**: Heterozygosity, allelic richness, HDC diversity
//! - **Inbreeding**: Pedigree-based kinship and HDC kinship estimates
//! - **Breeding Strategy**: Random, minimum kinship, HDC distance, balanced contribution
//! - **Genetic Load**: Deleterious allele tracking and genetic rescue planning
//! - **Simulation**: Multi-generation forward simulation with selection
//! - **Temporal Evolution**: O(1) trajectory prediction via CfC closed-form
//! - **Governance**: Mycelix-compatible tiered decision making
//! - **Ethics**: Moral algebra for reproductive ethics with Eight Harmonies alignment

pub mod breeding_strategy;
pub mod diversity;
pub mod effective_population;
pub mod ethics;
pub mod genetic_load;
pub mod governance;
pub mod hdc_genetics;
pub mod inbreeding;
pub mod simulation;
pub mod temporal_evolution;
pub mod types;

pub use breeding_strategy::*;
pub use diversity::*;
pub use effective_population::*;
pub use ethics::*;
pub use genetic_load::*;
pub use governance::*;
pub use hdc_genetics::*;
pub use inbreeding::*;
pub use simulation::*;
pub use temporal_evolution::*;
pub use types::*;
