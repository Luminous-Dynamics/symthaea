// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tiered Φ (Integrated Information) Approximation System
//!
//! Revolutionary improvement: Consciousness measurement at multiple fidelity levels.
//!
//! # Module Structure
//!
//! This module is organized into submodules for maintainability:
//!
//! - [`core`]: Core types and TieredPhi calculator with tier implementations
//! - `dynamics` (planned): Temporal dynamics tracking and attractor analysis
//! - `analysis` (planned): Multi-scale pyramid and entropy/complexity analysis
//! - `advanced` (planned): Cross-topology transfer, causal intervention, and modularity analysis
//!
//! # The Problem
//!
//! Exact Φ calculation requires finding the Minimum Information Partition (MIP),
//! which is NP-hard (O(2^n) for n components). This causes:
//! - Test timeouts (even small systems take too long)
//! - Production latency issues
//! - Inability to scale to large consciousness states
//!
//! # The Solution: Tiered Approximation
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    TIERED Φ APPROXIMATION SYSTEM                         │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   Tier 0: Mock (O(1))                                                    │
//! │   └── Deterministic values for testing                                   │
//! │       └── φ = 0.1 × n + 0.3 (linear in component count)                  │
//! │                                                                          │
//! │   Tier 1: Heuristic (O(n))                                               │
//! │   └── Fast approximation using average similarity                        │
//! │       └── φ ≈ 1 - avg_pairwise_similarity                                │
//! │                                                                          │
//! │   Tier 2: Spectral (O(n²))                                               │
//! │   └── Graph-based approximation using connectivity                       │
//! │       └── φ ≈ algebraic_connectivity(similarity_graph)                   │
//! │                                                                          │
//! │   Tier 3: Exact (O(2^n))                                                 │
//! │   └── Full MIP search (use sparingly!)                                   │
//! │       └── φ = min_partition(information_loss)                            │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::tiered_phi::{TieredPhi, ApproximationTier};
//!
//! // For testing: O(1) deterministic values
//! let mut phi = TieredPhi::new(ApproximationTier::RandomBaseline);
//! assert!(phi.compute(&components) > 0.0);
//!
//! // For production: O(n) fast approximation
//! let mut phi = TieredPhi::new(ApproximationTier::SampledPartition);
//!
//! // For research: O(2^n) exact calculation
//! let mut phi = TieredPhi::new(ApproximationTier::ExhaustivePartition);
//! ```

// =============================================================================
// SUBMODULES
// =============================================================================

pub mod advanced;
pub mod analysis;
pub mod core;
pub mod dynamics;
pub mod streaming;

#[cfg(test)]
mod tests;

// =============================================================================
// RE-EXPORTS FOR BACKWARD COMPATIBILITY
// =============================================================================

// HDC types needed by tests and users
pub use crate::hdc::binary_hv::BinaryHV;
pub use crate::hdc::unified_hv::ContinuousHV;

// Core types (most commonly used)
pub use core::{
    ApproximationTier,
    HierarchicalPhi,
    IncrementalPhiState,
    PhiAttribution,
    TieredPhi,
    TieredPhiConfig,
    TieredPhiStats,
    auto_phi,
    auto_tier,
    // Global functions
    global_phi,
    global_phi_stats,
    set_global_tier,
};

// Dynamics types
pub use dynamics::{
    // Attractor types
    AttractorConfig,
    AttractorResult,
    AttractorType,
    PhaseTransition,
    PhiAttractor,
    PhiDynamics,
    PhiDynamicsConfig,
    PhiDynamicsSnapshot,
    PhiTrend,
    TransitionDirection,
    TransitionType,
    TrendDirection,
    analyze_phi_attractor,
    classify_consciousness_state,
};

// Analysis types
pub use analysis::{
    PhiEntropyAnalyzer,
    // Entropy types
    PhiEntropyConfig,
    PhiEntropyResult,
    PhiPyramid,
    PhiPyramidConfig,
    PhiPyramidResult,
    analyze_phi_complexity,
    integrated_complexity,
    multi_scale_phi,
    optimal_scale,
};

// Advanced types
pub use advanced::{
    CausalAnalysisResult,
    CausalInterventionConfig,
    ConsciousnessModule,
    InterModuleRelation,
    // Intervention types
    InterventionType,
    ModularityConfig,
    // Modularity types
    ModuleDetectionMethod,
    NetworkModularityResult,
    NodeClassification,
    NodeInterventionResult,
    NodeRole,
    PhiCausalAnalyzer,
    PhiModularityAnalyzer,
    PhiSignature,
    PhiTransfer,
    // Transfer types
    PhiTransferConfig,
    PhiTransferResult,
    analyze_causal_interventions,
    analyze_network_modularity,
    compute_causal_power,
    compute_modularity_score,
    compute_transfer_matrix,
    detect_module_count,
    find_critical_nodes,
    transfer_from_ring,
};

// Streaming gradient types
pub use streaming::{
    GradientConfig, GradientEvent, GradientPrecision, GradientStats, OptimizationAction,
    PhiGradient, StreamingPhiGradient, compute_phi_gradient, compute_phi_gradient_fast,
};
