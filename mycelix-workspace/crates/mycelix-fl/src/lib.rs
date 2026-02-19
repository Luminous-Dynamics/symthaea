//! mycelix-fl: Unified WASM-Compatible Federated Learning Computation
//!
//! This crate provides the complete FL pipeline for decentralized federated
//! learning on Holochain. It compiles to `wasm32-unknown-unknown` and runs
//! in every validator node, enabling trustless aggregation consensus.
//!
//! # Architecture
//!
//! The 9-stage pipeline:
//! 1. **Compress**: HyperFeel J-L projection (10M params -> 2KB HV16)
//! 2. **Classify**: E-N-M-H epistemic quality grading
//! 3. **Measure Phi**: Agent coherence gating
//! 4. **Prove**: PoGQ-v4.1 quality score (mandatory for submission)
//! 5. **Detect**: 9-layer Byzantine defense stack
//! 6. **Submit**: Ed25519-signed to DHT
//! 7. **Aggregate**: A2 in HV space (6 methods, no decompression)
//! 8. **Consensus**: RB-BFT commit-reveal
//! 9. **Reward**: Shapley + KREDIT + Ethereum
//!
//! # WASM Constraints
//!
//! - NO ndarray (use `Vec<f32>` and `&[f32]`)
//! - NO tokio (all computation is synchronous)
//! - NO std::time (timestamps passed from host)
//! - Only serde, rand, thiserror, mycelix-fl-core dependencies
//!
//! # Relationship to mycelix-fl-core
//!
//! Core types (GradientUpdate, AggregationMethod, etc.) and canonical algorithms
//! (FedAvg, TrimmedMean, Median, Krum) are re-exported from `mycelix-fl-core`.
//! This crate adds the 9-layer Byzantine defense, HyperFeel compression,
//! epistemic grading, Phi gating, and Holochain integration on top.

/// Re-export mycelix-fl-core for consumers wanting direct access
pub use mycelix_fl_core as fl_core;

pub mod types;
pub mod compression;
pub mod aggregation;
pub mod detection;
pub mod trust;
pub mod epistemic;
pub mod coherence;
pub mod privacy;
pub mod pipeline;
pub mod holochain;

#[cfg(feature = "attacks")]
pub mod attacks;

pub mod hdc_aggregation;

#[cfg(any(feature = "coherence-series", feature = "phi-series"))]
pub mod coherence_series;

#[cfg(feature = "pogq")]
pub mod pogq;

pub use types::*;
pub use pipeline::DecentralizedPipeline;
pub use coherence::{GradientCoherenceConfig, GradientCoherenceGate, CoherenceScore};
