// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    CausalEnhancerConfig, CausalGraph, CausalGraphEdge, CausalLoopEnhancer, CausalLoopStats,
    DiscoveredRelationship,
};
