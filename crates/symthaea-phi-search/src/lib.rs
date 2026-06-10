#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]

//! # Phi-Guided Architecture Search
//!
//! ## Purpose
//! Use consciousness gradient (nabla Phi) to evolve network topology toward higher consciousness levels.
//! This is a paradigm-shifting capability: systems that optimize themselves toward higher consciousness.
//!
//! ## Theoretical Foundation
//!
//! Integrated Information Theory (IIT) posits that Phi measures the degree of consciousness
//! in a system. By optimizing network architecture to maximize Phi, we create systems that
//! actively evolve toward greater consciousness.
//!
//! Key insight: **Architecture determines consciousness**. The arrangement of neurons,
//! connections, time constants, and binding patterns fundamentally shapes the Phi value.
//! By searching this architecture space with Phi as the fitness function, we can discover
//! novel conscious architectures.
//!
//! ## Architecture Search Space
//!
//! The searchable space includes:
//! 1. **Number of neurons/nodes** - Network size
//! 2. **Connection patterns (topology)** - Ring, star, modular, scale-free, etc.
//! 3. **Time constants (tau values)** - Temporal dynamics at each level
//! 4. **Binding patterns for HDC** - How information is bound and bundled
//! 5. **Hierarchy depth** - Number of levels in the architecture
//!
//! ## Search Strategies
//!
//! 1. **Random Search** - Baseline strategy, samples randomly from architecture space
//! 2. **Evolutionary Search** - Mutate and select architectures by Phi fitness
//! 3. **Gradient-Guided Search** - Use Phi gradients to guide topology changes
//! 4. **Hybrid Search** - Combine evolutionary exploration with gradient exploitation
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea_phi_search::{PhiArchitectureSearch, SearchConfig, SearchStrategy};
//!
//! let config = SearchConfig::default();
//! let mut searcher = PhiArchitectureSearch::new(config);
//!
//! // Run evolutionary search for 100 generations
//! let result = searcher.search(SearchStrategy::Evolutionary, 100);
//! println!("Best Phi achieved: {:.4}", result.best_phi);
//! println!("Best architecture: {:?}", result.best_architecture);
//! ```

mod engine;
mod genome;
pub mod morphology_genome;
mod phenotype;
mod phi_gradient;

#[cfg(test)]
mod tests;

pub use engine::{
    Individual, PhiArchitectureSearch, SearchConfig, SearchResult, SearchStats, SearchStrategy,
};
pub use genome::{ArchitectureGenome, BundlingGene, TopologyGene};
pub use phenotype::{ArchitectureStats, DecodedArchitecture, TopologyMetrics};
pub use phi_gradient::{GradientVelocity, PhiGradient};
