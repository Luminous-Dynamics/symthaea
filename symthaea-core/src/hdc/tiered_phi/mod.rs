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

pub mod core;
pub mod dynamics;
pub mod analysis;
pub mod advanced;
pub mod streaming;

#[cfg(test)]
mod tests;

// =============================================================================
// RE-EXPORTS FOR BACKWARD COMPATIBILITY
// =============================================================================

// HDC types needed by tests and users
pub use crate::hdc::binary_hv::HV16;
pub use crate::hdc::real_hv::RealHV;

// Core types (most commonly used)
pub use core::{
    ApproximationTier,
    TieredPhiConfig,
    TieredPhiStats,
    TieredPhi,
    IncrementalPhiState,
    HierarchicalPhi,
    PhiAttribution,
    // Global functions
    global_phi,
    auto_phi,
    auto_tier,
    set_global_tier,
    global_phi_stats,
};

// Dynamics types
pub use dynamics::{
    PhiDynamicsConfig,
    PhiDynamics,
    PhiDynamicsSnapshot,
    PhiTrend,
    TrendDirection,
    PhaseTransition,
    TransitionDirection,
    TransitionType,
    // Attractor types
    AttractorConfig,
    AttractorType,
    AttractorResult,
    PhiAttractor,
    analyze_phi_attractor,
    classify_consciousness_state,
};

// Analysis types
pub use analysis::{
    PhiPyramidConfig,
    PhiPyramidResult,
    PhiPyramid,
    multi_scale_phi,
    optimal_scale,
    // Entropy types
    PhiEntropyConfig,
    PhiEntropyResult,
    PhiEntropyAnalyzer,
    analyze_phi_complexity,
    integrated_complexity,
};

// Advanced types
pub use advanced::{
    // Transfer types
    PhiTransferConfig,
    PhiSignature,
    PhiTransferResult,
    PhiTransfer,
    transfer_from_ring,
    compute_transfer_matrix,
    // Intervention types
    InterventionType,
    CausalInterventionConfig,
    NodeInterventionResult,
    CausalAnalysisResult,
    PhiCausalAnalyzer,
    analyze_causal_interventions,
    find_critical_nodes,
    compute_causal_power,
    // Modularity types
    ModuleDetectionMethod,
    ModularityConfig,
    ConsciousnessModule,
    InterModuleRelation,
    NodeRole,
    NodeClassification,
    NetworkModularityResult,
    PhiModularityAnalyzer,
    analyze_network_modularity,
    detect_module_count,
    compute_modularity_score,
};

// Streaming gradient types
pub use streaming::{
    GradientPrecision,
    GradientConfig,
    PhiGradient,
    OptimizationAction,
    GradientEvent,
    StreamingPhiGradient,
    GradientStats,
    compute_phi_gradient,
    compute_phi_gradient_fast,
};
