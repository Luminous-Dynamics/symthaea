//! Causal Module
//!
//! Causal reasoning and integration with the cognitive loop.
//!
//! ## Submodules
//!
//! - [`loop_integration`]: Integrates causal discovery into the cognitive loop,
//!   tracking (input, output) pairs and discovering causal structure periodically.

pub mod loop_integration;

pub use loop_integration::{
    CausalLoopEnhancer, CausalEnhancerConfig, CausalGraph, CausalGraphEdge,
    CausalLoopStats, DiscoveredRelationship,
};
