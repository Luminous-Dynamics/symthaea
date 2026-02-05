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
// Legacy Modules - Gated behind full_consciousness (have broken dependencies)
// ═══════════════════════════════════════════════════════════════════════════

// World modeling and routing (has some broken imports in routers)
#[cfg(feature = "full_consciousness")]
pub mod world_model;
#[cfg(feature = "full_consciousness")]
pub mod routers;

// Self-improvement modules (many have broken dependencies)
#[cfg(feature = "full_consciousness")]
pub mod architectural_graph;
#[cfg(feature = "full_consciousness")]
pub mod benchmark_suite;
#[cfg(feature = "full_consciousness")]
pub mod dream_mode;
#[cfg(feature = "full_consciousness")]
pub mod gradient_optimizer;
#[cfg(feature = "full_consciousness")]
pub mod improvement_generator;
#[cfg(feature = "full_consciousness")]
pub mod intrinsic_motivation;
#[cfg(feature = "full_consciousness")]
pub mod meta_cognitive;
#[cfg(feature = "full_consciousness")]
pub mod naming_ceremony;
#[cfg(feature = "full_consciousness")]
pub mod primitive_semantic_bridge;
#[cfg(feature = "full_consciousness")]
pub mod recursive_optimizer;
#[cfg(feature = "full_consciousness")]
pub mod routing_hub;
#[cfg(feature = "full_consciousness")]
pub mod safe_experiment;
#[cfg(feature = "full_consciousness")]
pub mod self_model;
#[cfg(feature = "full_consciousness")]
pub mod semantic_bridge;

// Conditional re-exports for legacy modules
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
pub use meta_cognitive::{MetaCognitive, MetaCognitiveState};
#[cfg(feature = "full_consciousness")]
pub use dream_mode::{DreamMode, DreamConfig};
#[cfg(feature = "full_consciousness")]
pub use gradient_optimizer::{GradientOptimizer, OptimizationConfig};
#[cfg(feature = "full_consciousness")]
pub use improvement_generator::{ImprovementGenerator, Improvement};
#[cfg(feature = "full_consciousness")]
pub use intrinsic_motivation::{IntrinsicMotivation, MotivationConfig};
#[cfg(feature = "full_consciousness")]
pub use recursive_optimizer::{RecursiveOptimizer, OptimizationResult};
#[cfg(feature = "full_consciousness")]
pub use safe_experiment::{SafeExperiment, ExperimentConfig, ExperimentResult};
#[cfg(feature = "full_consciousness")]
pub use architectural_graph::{ArchitecturalGraph, ArchNode, ArchEdge};
#[cfg(feature = "full_consciousness")]
pub use benchmark_suite::{BenchmarkSuite, BenchmarkResult};
#[cfg(feature = "full_consciousness")]
pub use routing_hub::{RoutingHub, RoutingConfig};
#[cfg(feature = "full_consciousness")]
pub use semantic_bridge::{SemanticBridge, SemanticBridgeConfig};
#[cfg(feature = "full_consciousness")]
pub use primitive_semantic_bridge::PrimitiveSemanticBridge;
#[cfg(feature = "full_consciousness")]
pub use naming_ceremony::{NamingCeremony, NamingConfig};
