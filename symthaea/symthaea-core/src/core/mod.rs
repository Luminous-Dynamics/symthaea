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
    HV,
    HDC_DIMENSION,
};

pub use crate::hdc::consciousness_topology_generators::{
    ConsciousnessTopology,
    TopologyType,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS PIPELINE — from hdc::consciousness_integration
// ═══════════════════════════════════════════════════════════════════════════════

// Note: ConsciousnessState struct is re-exported as ConsciousnessStateData to
// avoid collision with the ConsciousnessState trait from core::traits.
pub use crate::hdc::consciousness_integration::{
    ConsciousnessState as ConsciousnessStateData,
    ConsciousnessPipeline,
    IntegrationConfig,
    ConsciousnessMetricsReport,
    IntegrationAssessment,
    WorkspaceItem,
    MetaThought,
    BoundObject,
    BindingLevel,
    AlteredStateIndex,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS DASHBOARD — from hdc::consciousness_dashboard
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::consciousness_dashboard::{
    ConsciousnessDashboard,
    DashboardConfig,
    DashboardStatus,
};

// ═══════════════════════════════════════════════════════════════════════════════
// MATH BRIDGE — from hdc::math_bridge
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::math_bridge::{
    UnifiedMathEngine,
    MathValue,
    MathResult,
};

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS SIMULATION BRIDGE — from physics::simulation_bridge
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::physics::simulation_bridge::{
    PhysicsSimulator,
    SimulationAnalysis,
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
