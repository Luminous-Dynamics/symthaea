#![deny(unsafe_code)]
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
//! - **Ethics**: Moral algebra for reproductive ethics with Seven Harmonies alignment

pub mod types;
pub mod hdc_genetics;
pub mod effective_population;
pub mod diversity;
pub mod inbreeding;
pub mod breeding_strategy;
pub mod genetic_load;
pub mod simulation;
pub mod temporal_evolution;
pub mod governance;
pub mod ethics;

pub use types::*;
pub use hdc_genetics::*;
pub use effective_population::*;
pub use diversity::*;
pub use inbreeding::*;
pub use breeding_strategy::*;
pub use genetic_load::*;
pub use simulation::*;
pub use temporal_evolution::*;
pub use governance::*;
pub use ethics::*;
