//! # Recursive Improvement: Self-Modifying Consciousness
//!
//! This module provides recursive self-improvement capabilities including:
//! - Self-modeling and introspection
//! - Meta-cognitive optimization
//! - Dream-mode exploration
//! - Gradient-based architecture optimization
//! - Safe experimentation frameworks
//! - Consciousness world modeling
//!
//! ## MAGI Loop (Minimum AGI Loop) Implementation
//!
//! The MAGI Loop is a falsifiable AGI crossing criterion with 6 steps:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        THE MAGI LOOP                                │
//! │                                                                     │
//! │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
//! │   │ 1. PREDICT  │────▶│ 2. RESOLVE  │────▶│ 3. SELECT   │          │
//! │   │   (World)   │     │ (Calibrate) │     │  (Action)   │          │
//! │   └─────────────┘     └─────────────┘     └──────┬──────┘          │
//! │                                                   │                 │
//! │   ┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐          │
//! │   │ 6. UPDATE   │◀────│ 5. ATTRIB   │◀────│ 4. OBSERVE  │          │
//! │   │   (Safe)    │     │  (Causal)   │     │  (Reality)  │          │
//! │   └──────┬──────┘     └─────────────┘     └─────────────┘          │
//! │          │                                                          │
//! │          └──────────────────────────────────────────────────────────│
//! │                     LOOP BACK TO STEP 1                             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### Key MAGI Loop Components
//!
//! - [`WorldPrediction`]: Falsifiable predictions about external reality
//! - [`BrierScoreTracker`]: Proper calibration with ECE measurement
//! - [`ConstraintGate`]: Safety gate controlling execution mode
//! - [`WorldGroundedSelfModel`]: Integrated system combining all components
//!
//! See `docs/architecture/MAGI_LOOP_SPECIFICATION.md` for full specification.

// ═══════════════════════════════════════════════════════════════════════════
// MAGI Loop Core - Always available (self-contained, no broken dependencies)
// ═══════════════════════════════════════════════════════════════════════════

// Core infrastructure for MAGI Loop
pub mod types;
pub mod core;

// MAGI Loop implementation (World-Grounded Prediction)
pub mod world_prediction;
pub mod calibration;
pub mod constraint_gate;
pub mod magi_integration;
pub mod resolution;
pub mod active_inference_bridge;
pub mod persistence;
pub mod runtime;
pub mod dream_feedback;

// Re-export key types from core infrastructure
pub use types::{
    instant_now, calculate_trend,
    SemanticInput, InputModality, ActionContext, TimeWindow,
};

pub use core::{
    ComponentId, BottleneckType, Bottleneck, ImprovementType,
    MonitorConfig, ComponentMetrics, PerformanceMonitor, AccuracyMetric,
};

// MAGI Loop exports (World-Grounded Prediction)
pub use world_prediction::{
    WorldPrediction, PredictionDomain, OutcomeCategory, Resolution,
    RiskTier, WorldActionContext,
    ResolutionAuthority, ResolutionContract, ContractRegistry,
    DiffTolerance, ResourceExpectation,
};

// Calibration exports (MAGI Loop Step 2)
pub use calibration::{
    BrierScoreTracker, CalibrationConfig, DomainCalibration,
    CalibrationSummary, DomainStats, ResolvedPredictionRecord,
};

// Constraint Gate exports (MAGI Loop Step 3.5 - UPGRADE B)
pub use constraint_gate::{
    ConstraintGate, ConstraintGateConfig, ExecutionMode,
    GateDecision, GateFactor, GateStatistics,
    DryRunReason, SupervisionReason,
};

// MAGI Loop Integration exports
pub use magi_integration::{
    WorldGroundedSelfModel, WorldGroundedConfig,
    CausalAttribution, MagiLoopState, CalibrationQuality,
    // EFE Integration (Phase 3)
    EfeContribution, EfeWeights, CalibratedEfe,
    // Safe Update Protocol (Phase 6)
    SafeUpdate, SafeUpdateManager, SystemSnapshot, ModelUpdate,
    RollbackCondition, UpdateStatus, UpdateStatistics,
};

// Active Inference Bridge exports (MAGI + PAC + Signals)
pub use active_inference_bridge::{
    ActiveInferenceBridge, ActiveInferenceBridgeConfig,
    CouplingQuality, BridgeStatistics,
    MagiActiveInferenceController,
};

// Persistence exports (Epistemic Save File)
pub use persistence::{
    PersistenceManager, PersistenceConfig, MagiStateSnapshot,
    StartupMode, PersistedDomainCalibration, PersistedCausalAttribution,
    GlobalCalibrationStats, PersistedLoopState,
    // High-level integration
    MagiPersistentModel,
};

// Runtime exports (MAGI Loop Heartbeat)
pub use runtime::{
    MagiLoopRuntime, RuntimeConfig, RuntimeState,
    RuntimeSnapshot, RuntimeSignals, RuntimeEvent,
    RuntimeLogEntry, LogLevel,
    PendingPrediction, AutoResolveType,
};

// Dream Feedback exports (Counterfactual Learning)
pub use dream_feedback::{
    DreamFeedbackBridge, DreamInsight, ActionPrior,
    ConfidenceAdjustment, DreamFeedbackStats, hash_context,
};

// ═══════════════════════════════════════════════════════════════════════════
// Legacy Modules - Gated behind full_consciousness feature flag
// ═══════════════════════════════════════════════════════════════════════════

// World modeling and routing - these compile cleanly
#[cfg(feature = "full_consciousness")]
pub mod world_model;
#[cfg(feature = "full_consciousness")]
pub mod routers;

// Self-improvement modules that compile cleanly
#[cfg(feature = "full_consciousness")]
pub mod intrinsic_motivation;
#[cfg(feature = "full_consciousness")]
pub mod meta_cognitive;
#[cfg(feature = "full_consciousness")]
pub mod self_model;

// TODO: These modules have deep structural mismatches with core types and
// need significant refactoring before they can be re-enabled.
//
// architectural_graph - uses ComponentId in match patterns (E0533), tries to
//   move ComponentId out of references (E0507/E0382); ~28 compilation errors
// #[cfg(feature = "full_consciousness")]
// pub mod architectural_graph;
//
// gradient_optimizer - uses ImprovementType variants as struct constructors
//   with fields (E0559), ComponentId pattern matching (E0533), wrong
//   record_phi signature (E0061), non-exhaustive match on BottleneckType (E0004);
//   ~27 compilation errors
// #[cfg(feature = "full_consciousness")]
// pub mod gradient_optimizer;
//
// improvement_generator - ImprovementType variants used as structs with named
//   fields (from, to, threads, component, count, optimization, name, old_value,
//   new_value), ComponentId::from(f64) not implemented, root_cause move error;
//   ~36 compilation errors
// #[cfg(feature = "full_consciousness")]
// pub mod improvement_generator;
//
// recursive_optimizer - imports from architectural_graph, safe_experiment, and
//   improvement_generator (all disabled); wrong arg counts for get_bottlenecks
//   and record_phi; ~7 compilation errors
// #[cfg(feature = "full_consciousness")]
// pub mod recursive_optimizer;
//
// safe_experiment - uses ImprovementType variants as struct patterns with named
//   fields in match arms (E0026), ComponentId::Cache used as pattern (E0533);
//   ~5 compilation errors
// #[cfg(feature = "full_consciousness")]
// pub mod safe_experiment;

// TODO: These modules have broken dependencies and need a type refactor before
// they can be re-enabled. See each module file for specific issues.
//
// benchmark_suite - depends on 7 advanced router types (CausalValidatedRouter,
//   InformationGeometricRouter, etc.) that don't exist in routers.rs yet
// #[cfg(feature = "full_consciousness")]
// pub mod benchmark_suite;
//
// dream_mode - references crate::soul::{WeaverActor, ConceptDiscovery} which
//   don't exist; uses ConsciousnessWorldModel methods (dream(), pending_concepts,
//   consciousness_level()) that don't exist; uses wrong ConsciousnessTransition fields
// #[cfg(feature = "full_consciousness")]
// pub mod dream_mode;
//
// naming_ceremony - references crate::soul::{WeaverActor, ConceptDiscovery} which
//   don't exist; uses non-existent fields on CrystalizedConcept (uid, attractor_signature,
//   activation_count); uses non-existent methods on ConsciousnessWorldModel
// #[cfg(feature = "full_consciousness")]
// pub mod naming_ceremony;
//
// routing_hub - references advanced router types that don't exist in routers.rs;
//   references RoutingStrategy enum that doesn't exist; uses PrimitiveSystem::global()
//   and AdaptivePrimitiveSelector APIs that may have changed
// #[cfg(feature = "full_consciousness")]
// pub mod routing_hub;
//
// semantic_bridge - uses ConsciousnessWorldModel.pending_concepts (doesn't exist),
//   .observe() (should be .observe_transition()), .stats().consciousness_level
//   (field doesn't exist); uses WorldModelConfig.min_training_samples (doesn't exist);
//   builds ConsciousnessTransition with is_real field (should be surprise)
// #[cfg(feature = "full_consciousness")]
// pub mod semantic_bridge;
//
// primitive_semantic_bridge - references crate::hdc::semantic_primitive_encoder::
//   SemanticPrimitiveEncoder which doesn't exist; uses HdcBridge::default() which
//   may not be available; references qwen3 embedder types behind embeddings feature
// #[cfg(feature = "full_consciousness")]
// pub mod primitive_semantic_bridge;

// ── Conditional re-exports for compiled legacy modules ──

#[cfg(feature = "full_consciousness")]
pub use world_model::{
    LatentConsciousnessState, ConsciousnessAction, ActionType,
    ConsciousnessStateDelta, ConsciousnessTransition,
    WorldModelConfig, WorldModelStats, ConsciousnessWorldModel,
};

#[cfg(feature = "full_consciousness")]
pub use routers::{
    RoutingDecision, RouterType, ConsciousnessRouter,
    DirectRouter, PhiMaximizingRouter, ExploratoryRouter, ConsolidatingRouter,
};

#[cfg(feature = "full_consciousness")]
pub use self_model::{SelfModel, SelfModelConfig};
#[cfg(feature = "full_consciousness")]
pub use meta_cognitive::{MetaCognitiveController, MetaCognitiveConfig};
#[cfg(feature = "full_consciousness")]
pub use intrinsic_motivation::{IntrinsicMotivationSystem, MotivationConfig};
