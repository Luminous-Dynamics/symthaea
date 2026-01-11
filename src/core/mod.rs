//! Core Stable API Surface
//!
//! This module provides a small, focused set of re-exports that represent
//! the most stable and broadly useful parts of Symthaea-HLB. It is intended
//! as the primary entry point for external users and downstream projects.
//!
//! The goal is to give you:
//! - A clean way to use Φ measurement and topology tools
//! - Access to unified hypervector types for HDC experiments
//! - A minimal consciousness pipeline and master equation interface
//!
//! Everything here is just a re-export of existing types; no behavior is
//! changed, and all original module paths remain available.

// Φ engine and measurement
pub use crate::phi_engine::{
    PhiEngine,
    PhiMethod,
    PhiResult,
    PhiCalculator,
    ContinuousPhiCalculator,
    TieredPhi,
    ApproximationTier,
    TieredPhiConfig,
    CachedPhiEngine,
    CacheStats,
};

// HDC core types
pub use crate::hdc::unified_hv::{
    ContinuousHV,
    BinaryHV,
    HV,
    HDC_DIMENSION,
};

pub use crate::hdc::consciousness_topology_generators::{
    ConsciousnessTopology,
    TopologyType,
};

// Master Consciousness Equation v2.0
pub use crate::consciousness::consciousness_equation_v2::{
    ConsciousnessEquationV2,
    ConsciousnessStateV2,
    CoreComponent,
};

// Unified consciousness pipeline (minimal loop)
pub use crate::consciousness::unified_consciousness_pipeline::{
    UnifiedConsciousnessPipeline,
    PipelineConfig,
    ConsciousMoment,
    PipelineStatistics,
};


// Consciousness API traits
pub mod traits;
pub use traits::{
    ConsciousnessMetric,
    ConsciousnessState,
    ConsciousnessUpdater,
    ConsciousnessObserver,
    MeasurementResult,
    StateSnapshot,
    Complexity,
    NullObserver,
};

// Domain-agnostic traits (Generalization Refactoring Phase 1)
// These enable Symthaea to work across Consciousness, Task, NixOS domains
pub mod domain_traits;
pub use domain_traits::{
    // Seam 1: Agent abstraction
    State,
    Action,
    Goal,
    HdcEncodable,
    // Seam 2: World Model
    WorldModel,
    // Seam 3: Domain Adapter
    DomainAdapter,
    // Seam 4: Quality Signals (including Φ)
    QualitySignal,
    // Seam 5: Actor Model
    DomainActor,
    ActorObservation,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS DOMAIN TYPES (Generalization Refactoring Phase 1)
// ═══════════════════════════════════════════════════════════════════════════════

// Re-export consciousness-specific types that implement the domain traits
pub use crate::consciousness::recursive_improvement::world_model::{
    // State implementation
    LatentConsciousnessState,
    // Action implementation
    ConsciousnessAction,
    // Goal implementations
    PhiMaximizationGoal,
    CoherenceGoal,
    // Transition type
    ConsciousnessTransition,
};

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE ALIASES FOR BACKWARD COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Type alias: ConsciousnessState is an alias for types implementing State trait.
/// This provides backward compatibility with code using the old name.
pub type ConsciousnessStateAlias = LatentConsciousnessState;

/// Type alias for consciousness-specific world model.
pub type ConsciousnessWorldModel = crate::consciousness::recursive_improvement::world_model::ConsciousnessWorldModel;
