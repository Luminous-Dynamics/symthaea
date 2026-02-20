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
pub mod resolution;
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
// Legacy Modules - Gated behind full_consciousness feature flag
// ═══════════════════════════════════════════════════════════════════════════

// World modeling and routing - these compile cleanly
#[cfg(feature = "full_consciousness")]
pub mod routers;
#[cfg(feature = "full_consciousness")]
pub mod world_model;

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
// architectural_graph - FIXED: ComponentId function calls now include ()
#[cfg(feature = "full_consciousness")]
pub mod architectural_graph;
//
// gradient_optimizer - FIXED: ComponentId pattern matching now uses as_str(),
//   record_phi signature corrected, ComponentId function calls added ()
#[cfg(feature = "full_consciousness")]
pub mod gradient_optimizer;
//
// improvement_generator - ImprovementType now has struct variants (fixed in core.rs)
#[cfg(feature = "full_consciousness")]
pub mod improvement_generator;
//
// recursive_optimizer - depends on improvement_generator and safe_experiment
#[cfg(feature = "full_consciousness")]
pub mod recursive_optimizer;
//
// safe_experiment - ImprovementType struct patterns now match
#[cfg(feature = "full_consciousness")]
pub mod safe_experiment;

// Previously fixed modules
#[cfg(feature = "full_consciousness")]
pub mod dream_mode;
#[cfg(feature = "full_consciousness")]
pub mod naming_ceremony;
#[cfg(feature = "full_consciousness")]
pub mod semantic_bridge;

// These 3 modules need SemanticPrimitiveEncoder type stubs before they can compile.
// Folded into full_consciousness (formerly recursive_improvement_advanced).
#[cfg(feature = "full_consciousness")]
pub mod benchmark_suite;
#[cfg(feature = "full_consciousness")]
pub mod primitive_semantic_bridge;
#[cfg(feature = "full_consciousness")]
pub mod routing_hub;

// ── Conditional re-exports for compiled legacy modules ──

#[cfg(feature = "full_consciousness")]
pub use world_model::{
    ActionType, ConsciousnessAction, ConsciousnessStateDelta, ConsciousnessTransition,
    ConsciousnessWorldModel, LatentConsciousnessState, WorldModelConfig, WorldModelStats,
};

#[cfg(feature = "full_consciousness")]
pub use routers::{
    ConsciousnessRouter, ConsolidatingRouter, DirectRouter, ExploratoryRouter, PhiMaximizingRouter,
    RouterType, RoutingDecision,
};

#[cfg(feature = "full_consciousness")]
pub use intrinsic_motivation::{IntrinsicMotivationSystem, MotivationConfig};
#[cfg(feature = "full_consciousness")]
pub use meta_cognitive::{MetaCognitiveConfig, MetaCognitiveController};
#[cfg(feature = "full_consciousness")]
pub use self_model::{SelfModel, SelfModelConfig};
// semantic_bridge disabled - see module comment above
// #[cfg(feature = "full_consciousness")]
// pub use semantic_bridge::{
//     SemanticBridge, SemanticBridgeConfig, SemanticBridgeStats,
//     SemanticInput, ActionContext, ProcessingResult,
// };
