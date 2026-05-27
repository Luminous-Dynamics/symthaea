// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    ApproximationTier, CacheStats, CachedPhiEngine, ContinuousPhiCalculator, PhiCalculator,
    PhiEngine, PhiMethod, PhiResult, TieredPhi, TieredPhiConfig,
};

// HDC core types
pub use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION, HV};

pub use crate::hdc::consciousness_topology_generators::{ConsciousnessTopology, TopologyType};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS PIPELINE — from hdc::consciousness_integration
// ═══════════════════════════════════════════════════════════════════════════════

// Note: ConsciousnessState struct is re-exported as ConsciousnessStateData to
// avoid collision with the ConsciousnessState trait from core::traits.
pub use crate::hdc::consciousness_integration::{
    AlteredStateIndex, BindingLevel, BoundObject, ConsciousnessMetricsReport,
    ConsciousnessPipeline, ConsciousnessState as ConsciousnessStateData, IntegrationAssessment,
    IntegrationConfig, MetaThought, WorkspaceItem,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS DASHBOARD — from hdc::consciousness_dashboard
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::consciousness_dashboard::{
    ConsciousnessDashboard, DashboardConfig, DashboardStatus,
};

// ═══════════════════════════════════════════════════════════════════════════════
// MATH BRIDGE — from hdc::math_bridge
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::math_bridge::{MathResult, MathValue, UnifiedMathEngine};

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS SIMULATION BRIDGE — from physics::simulation_bridge
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::physics::simulation_bridge::{PhysicsSimulator, SimulationAnalysis};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS VERIFICATION — from hdc::consciousness_verifier
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::consciousness_verifier::{
    ConsciousnessVerdict, ConsciousnessVerifier, IITAxiomScores, VerificationReport,
};

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC BRIDGE — from hdc::semantic_bridge
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::semantic_bridge::SemanticBridge;

// ═══════════════════════════════════════════════════════════════════════════════
// Φ FEEDBACK — from hdc::phi_feedback
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::phi_feedback::{FeedbackModulation, PhiFeedbackConfig, PhiFeedbackController};

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE & OBSERVABILITY
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::consciousness_perf::{
    SimdCapabilities, batch_find_similar, batch_similarity_matrix, cluster_by_similarity,
    find_similar, simd_capabilities,
};

pub use crate::observability::{DataPoint, MetricsCollector, MetricsSnapshot};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS SUBSYSTEM PLUGIN ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════════

pub use crate::hdc::consciousness_subsystem::{
    ConsciousnessSubsystem, SubsystemContext, SubsystemError,
};

pub use crate::hdc::consciousness_metacognitive::{
    MetaConsciousnessWrapped, MetacognitiveSubsystem,
};

pub use crate::hdc::consciousness_self_awareness::{
    SelfAwarenessSubsystem, TemporalConsciousnessWrapped,
};

pub use crate::hdc::consciousness_phi_optimization::{
    PhaseTransitionWrapped, PhiOptimizationSubsystem,
};

pub use crate::hdc::consciousness_integration::ConsciousnessPipelineBuilder;
pub use crate::hdc::consciousness_integration::PipelineCheckpoint;
pub use crate::hdc::consciousness_integration::SubsystemCycleReport;
// State view types are available from their home modules:
//   TemporalState -> consciousness_metacognition
//   EmotionalState -> consciousness_continuity
//   IntegrationMetrics -> physics::design_integration

// Consciousness API traits
pub mod traits;
pub use traits::{
    Complexity, ConsciousnessMetric, ConsciousnessObserver, ConsciousnessState,
    ConsciousnessUpdater, MeasurementResult, NullObserver, StateSnapshot,
};

// Domain-agnostic traits (Generalization Refactoring Phase 1)
// These enable Symthaea to work across Consciousness, Task, NixOS domains
pub mod domain_traits;
pub use domain_traits::{
    Action,
    ActorObservation,
    // Seam 5: Actor Model
    DomainActor,
    // Seam 3: Domain Adapter
    DomainAdapter,
    Goal,
    HdcEncodable,
    // Seam 4: Quality Signals (including Φ)
    QualitySignal,
    // Seam 1: Agent abstraction
    State,
    // Seam 2: World Model
    WorldModel,
};
