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
pub mod organism;
pub mod tournament;

pub use fitness::{FepFitnessBridge, FepFitnessConfig, FitnessWeights, InputStrategy};
pub use genome::{NeuralGenome, NeuralPhenotype};
pub use organism::{NeuralOrganism, OrganismFitness, StepResult};
pub use tournament::{
    EvolutionResult, GenerationSnapshot, NeuroevolutionConfig, NeuroevolutionEngine, SpeciesInfo,
};
