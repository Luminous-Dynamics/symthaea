// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(unsafe_code)]
//! # CfC-HDC Neuroevolution
//!
//! Evolves CfC neural organisms via Free Energy minimization using HDC genome encoding.
//!
//! A single `BinaryHV` (2,048 bytes) encodes the full CfC neuron configuration as a genome.
//! Mutation is bit-flipping, crossover is uniform byte selection, fitness is FEP free energy.
//!
//! ## Architecture
//!
//! ```text
//! BinaryHV genome (2KB) → decode → UnifiedConfig + NetworkConfig
//!                                          ↓
//!                              HdcLtcUnifiedNetwork + ActiveInferenceAgent
//!                                          ↓
//!                              evaluate N cycles → FreeEnergyComponents
//!                                          ↓
//!                              OrganismFitness (multi-objective)
//!                                          ↓
//!                              Tournament selection → next generation
//! ```
//!
//! ## References
//!
//! - Stanley & Miikkulainen (2002). Evolving Neural Networks through Augmenting Topologies.
//! - Hasani et al. (2021). Liquid Time-constant Networks.
//! - Friston (2010). The free-energy principle: a unified brain theory?
//! - Tononi (2004). An information integration theory of consciousness.

pub mod fitness;
pub mod genome;
pub mod governance;
pub mod organism;
pub mod phi_fitness;
pub mod threshold_genome;
pub mod tournament;

pub use fitness::{FepFitnessBridge, FepFitnessConfig, FitnessWeights, InputStrategy};
pub use genome::{NeuralGenome, NeuralPhenotype};
pub use governance::{GovernanceResult, blend_thresholds, gate_threshold_proposal};
pub use organism::{NeuralOrganism, OrganismFitness, StepResult};
pub use threshold_genome::{ThresholdPhenotype, decode_thresholds, encode_thresholds};
pub use tournament::{
    Checkpoint, EvolutionResult, GenerationSnapshot, NeuroevolutionConfig, NeuroevolutionEngine,
    SpeciesInfo,
};
