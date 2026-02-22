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
pub mod core;
pub mod types;

// MAGI Loop implementation (World-Grounded Prediction)
pub mod active_inference_bridge;
pub mod calibration;
pub mod constraint_gate;
pub mod dream_feedback;
pub mod magi_integration;
pub mod persistence;
// DEAD CODE: not imported anywhere outside recursive_improvement/. Candidate for removal.
// The resolution execution system (TestSuite, ExitCode, ResourceState, HumanConfirmation
// resolvers) has zero external callers. Its ResolutionResult type is not re-exported.
// pub mod resolution;
pub mod runtime;
pub mod world_prediction;

// Re-export key types from core infrastructure
pub use types::{
    calculate_trend, instant_now, ActionContext, InputModality, SemanticInput, TimeWindow,
};

pub use core::{
    AccuracyMetric, Bottleneck, BottleneckType, ComponentId, ComponentMetrics, ImprovementType,
    MonitorConfig, PerformanceMonitor,
};

// MAGI Loop exports (World-Grounded Prediction)
pub use world_prediction::{
    ContractRegistry, DiffTolerance, OutcomeCategory, PredictionDomain, Resolution,
    ResolutionAuthority, ResolutionContract, ResourceExpectation, RiskTier, WorldActionContext,
    WorldPrediction,
};

// Calibration exports (MAGI Loop Step 2)
pub use calibration::{
    BrierScoreTracker, CalibrationConfig, CalibrationSummary, DomainCalibration, DomainStats,
    ResolvedPredictionRecord,
};

// Constraint Gate exports (MAGI Loop Step 3.5 - UPGRADE B)
pub use constraint_gate::{
    ConstraintGate, ConstraintGateConfig, DryRunReason, ExecutionMode, GateDecision, GateFactor,
    GateStatistics, SupervisionReason,
};

// MAGI Loop Integration exports
pub use magi_integration::{
    CalibratedEfe,
    CalibrationQuality,
    CausalAttribution,
    // EFE Integration (Phase 3)
    EfeContribution,
    EfeWeights,
    MagiLoopState,
    ModelUpdate,
    RollbackCondition,
    // Safe Update Protocol (Phase 6)
    SafeUpdate,
    SafeUpdateManager,
    SystemSnapshot,
    UpdateStatistics,
    UpdateStatus,
    WorldGroundedConfig,
    WorldGroundedSelfModel,
};

// Active Inference Bridge exports (MAGI + PAC + Signals)
pub use active_inference_bridge::{
    ActiveInferenceBridge, ActiveInferenceBridgeConfig, BridgeStatistics, CouplingQuality,
    MagiActiveInferenceController,
};

// Persistence exports (Epistemic Save File)
pub use persistence::{
    GlobalCalibrationStats,
    // High-level integration
    MagiPersistentModel,
    MagiStateSnapshot,
    PersistedCausalAttribution,
    PersistedDomainCalibration,
    PersistedLoopState,
    PersistenceConfig,
    PersistenceManager,
    StartupMode,
};

// Runtime exports (MAGI Loop Heartbeat)
pub use runtime::{
    AutoResolveType, LogLevel, MagiLoopRuntime, PendingPrediction, RuntimeConfig, RuntimeEvent,
    RuntimeLogEntry, RuntimeSignals, RuntimeSnapshot, RuntimeState,
};

// Dream Feedback exports (Counterfactual Learning)
pub use dream_feedback::{
    hash_context, ActionPrior, ConfidenceAdjustment, DreamFeedbackBridge, DreamFeedbackStats,
    DreamInsight,
};

// ═══════════════════════════════════════════════════════════════════════════
// Legacy Modules - All gated behind full_consciousness feature flag
//
// DEAD CODE AUDIT (2026-02-21): Every module below has ZERO external callers
// outside of recursive_improvement/ itself. The only references are:
//   - brain/consciousness_bridge.rs and brain/hippocampus_bridge.rs import
//     world_model types, but those files are orphaned (not declared in brain/mod.rs)
//   - consciousness_driven_evolution.rs imports gradient_optimizer and
//     recursive_optimizer, but it is itself cfg-gated on full_consciousness
//   - All other references are internal to recursive_improvement/
//
// These modules are candidates for removal. Commenting out all pub mod
// declarations and re-exports to reduce dead code surface.
// ═══════════════════════════════════════════════════════════════════════════

// world_model: Consciousness latent space model.
// Required by brain/affective_bridge.rs (ConsciousnessWorldModel, WorldModelStats).
#[cfg(feature = "full_consciousness")]
pub mod world_model;

// DEAD CODE: routers - ConsciousnessRouter variants (Direct, Exploratory, etc.).
// Only used by routing_hub.rs and benchmark_suite.rs (also dead).
// #[cfg(feature = "full_consciousness")]
// pub mod routers;

// DEAD CODE: intrinsic_motivation - Curiosity/competence/autonomy drives.
// Only used internally by self_model.rs (also dead).
// #[cfg(feature = "full_consciousness")]
// pub mod intrinsic_motivation;

// DEAD CODE: meta_cognitive - Resource allocation controller (GWT/ACT-R inspired).
// Only re-exported in mod.rs, zero external callers.
// #[cfg(feature = "full_consciousness")]
// pub mod meta_cognitive;

// DEAD CODE: self_model - Self-modeling with bottleneck detection.
// Only re-exported in mod.rs, zero external callers.
// #[cfg(feature = "full_consciousness")]
// pub mod self_model;

// DEAD CODE: architectural_graph - Component dependency graph for optimization.
// Only used internally by byzantine_collective (also gated/dead).
// #[cfg(feature = "full_consciousness")]
// pub mod architectural_graph;

// DEAD CODE: gradient_optimizer - Adam optimizer for Phi maximization.
// Only caller is consciousness_driven_evolution.rs (itself cfg-gated, no loop callers).
// #[cfg(feature = "full_consciousness")]
// pub mod gradient_optimizer;

// DEAD CODE: improvement_generator - Generates improvement proposals.
// Only used by recursive_optimizer.rs and meta_cognitive.rs (internal).
// #[cfg(feature = "full_consciousness")]
// pub mod improvement_generator;

// DEAD CODE: recursive_optimizer - Orchestrates improvement cycle.
// Only caller is consciousness_driven_evolution.rs (cfg-gated).
// #[cfg(feature = "full_consciousness")]
// pub mod recursive_optimizer;

// DEAD CODE: safe_experiment - Sandboxed experiment runner.
// Zero external callers.
// #[cfg(feature = "full_consciousness")]
// pub mod safe_experiment;

// DEAD CODE: dream_mode - Dream-state exploration mode.
// Zero external callers.
// #[cfg(feature = "full_consciousness")]
// pub mod dream_mode;

// DEAD CODE: naming_ceremony - Concept naming/labeling.
// Zero external callers.
// #[cfg(feature = "full_consciousness")]
// pub mod naming_ceremony;

// DEAD CODE: semantic_bridge - Semantic processing bridge.
// Zero external callers. Re-exports were already disabled.
// #[cfg(feature = "full_consciousness")]
// pub mod semantic_bridge;

// DEAD CODE: benchmark_suite - Consciousness benchmark harness.
// Zero external callers. Needs SemanticPrimitiveEncoder stubs.
// #[cfg(feature = "full_consciousness")]
// pub mod benchmark_suite;

// DEAD CODE: primitive_semantic_bridge - Primitive-to-semantic mapping.
// Zero external callers. Needs SemanticPrimitiveEncoder stubs.
// #[cfg(feature = "full_consciousness")]
// pub mod primitive_semantic_bridge;

// DEAD CODE: routing_hub - Unified routing coordinator.
// Zero external callers. Depends on routers (also dead).
// #[cfg(feature = "full_consciousness")]
// pub mod routing_hub;

// ── Conditional re-exports DISABLED (all source modules commented out above) ──

// world_model re-exports (used by brain/affective_bridge.rs)
#[cfg(feature = "full_consciousness")]
pub use world_model::{ConsciousnessWorldModel, WorldModelStats};

// DEAD CODE: routers re-exports
// #[cfg(feature = "full_consciousness")]
// pub use routers::{
//     ConsciousnessRouter, ConsolidatingRouter, DirectRouter, ExploratoryRouter, PhiMaximizingRouter,
//     RouterType, RoutingDecision,
// };

// DEAD CODE: intrinsic_motivation re-exports
// #[cfg(feature = "full_consciousness")]
// pub use intrinsic_motivation::{IntrinsicMotivationSystem, MotivationConfig};

// DEAD CODE: meta_cognitive re-exports
// #[cfg(feature = "full_consciousness")]
// pub use meta_cognitive::{MetaCognitiveConfig, MetaCognitiveController};

// DEAD CODE: self_model re-exports
// #[cfg(feature = "full_consciousness")]
// pub use self_model::{SelfModel, SelfModelConfig};

// DEAD CODE: semantic_bridge re-exports (were already disabled)
// #[cfg(feature = "full_consciousness")]
// pub use semantic_bridge::{
//     SemanticBridge, SemanticBridgeConfig, SemanticBridgeStats,
//     SemanticInput, ActionContext, ProcessingResult,
// };
