// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Recursive Improvement: Self-Modifying Consciousness
//!
//! This module provides recursive self-improvement capabilities:
//! - MAGI Loop: Falsifiable AGI crossing criterion (predict → calibrate → act → observe → attribute → update)
//! - Consciousness world modeling (latent state tracking)
//! - Dream feedback (counterfactual learning)
//! - Active inference bridge (MAGI + PAC + signals)
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
pub mod calibration_analytics;
pub mod constraint_gate;
pub mod dream_feedback;
pub mod magi_integration;
pub mod persistence;
pub mod resolution;
pub mod runtime;
pub mod world_prediction;

// Re-export key types from core infrastructure
pub use types::{
    ActionContext, InputModality, SemanticInput, TimeWindow, calculate_trend, instant_now,
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
    ActionPrior, ConfidenceAdjustment, DreamFeedbackBridge, DreamFeedbackStats, DreamInsight,
    hash_context,
};

// ═══════════════════════════════════════════════════════════════════════════
// World Model — gated behind full_consciousness feature flag
// ═══════════════════════════════════════════════════════════════════════════

// world_model: Consciousness latent space model.
// Required by brain/affective_bridge.rs (ConsciousnessWorldModel, WorldModelStats).
#[cfg(feature = "full_consciousness")]
pub mod world_model;

#[cfg(feature = "full_consciousness")]
pub use world_model::{ConsciousnessWorldModel, WorldModelStats};
