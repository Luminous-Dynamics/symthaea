// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Recursive Improvement: Self-Modifying Consciousness
//!
//! This module provides recursive self-improvement capabilities:
//! - MAGI Loop: Falsifiable AGI crossing criterion (predict → calibrate → act → observe → attribute → update)
//! - Consciousness world modeling (latent state tracking)
//! - Dream feedback (counterfactual learning)
//! - Replay-grounded policy improvement with explicit epistemic boundaries
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

// Replay-grounded recursive improvement. These modules keep exact historical
// replay distinct from model-generated counterfactuals and predictions.
pub mod dream_confidence_gate;
pub mod epistemic_world;
pub mod exact_replay;
pub mod experience_tree;
pub mod replay_policy;
pub mod sym_rsi_candidate_family;
pub mod sym_rsi_experiment;
pub mod sym_rsi_fixtures;
pub mod sym_rsi_fresh_evaluation;
pub mod sym_rsi_grounded_dream;
pub mod sym_rsi_dream_protocol;
pub mod sym_rsi_dream_fresh_evaluation;
pub mod sym_rsi_dream_ood_evaluation;
pub mod sym_rsi_dream_verification;
pub mod sym_rsi_holdout_gate;
pub mod sym_rsi_replay_corpus;
pub mod sym_rsi_replay_selection;
pub mod sym_rsi_runner;

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

// Replay-grounded RSI exports
pub use dream_confidence_gate::DreamConfidenceGate;
pub use epistemic_world::{EpistemicWorldRecord, WorldEvidenceKind};
pub use exact_replay::{ExactReplayWorld, ReplayError};
pub use experience_tree::{
    ExperienceNode, ExperienceNodeId, ExperienceProvenance, ExperienceTree, ExperienceTreeError,
};
pub use replay_policy::{PolicySelectionError, ReplayPolicyScore, select_replay_policy};
pub use sym_rsi_candidate_family::{
    FrozenReplayCorpus, ReplayAcquisitionError, ReplayCorpusAcquisitionReceipt,
    SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA, SYM_RSI_001_INCUMBENT_POLICY_ID,
    SYM_RSI_001_REPLAY_CORPUS_SCHEMA, acquire_canonical_replay_corpus,
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family,
};
pub use sym_rsi_experiment::{
    ArmRunMetrics, DomainSeedPlan, EvaluationSplit, ExperimentArm, ExperimentDomainSpec,
    ExperimentHarnessError, PrimaryContrastKind, PrimaryContrastReceipt,
    SymRsiExperimentManifest, SymRsiRunReceipt, build_primary_contrast,
};
pub use sym_rsi_fixtures::{
    FixtureDomainError, FixtureDomainKind, FixtureState, FixtureTransition,
    SYM_RSI_001_FIXTURE_ADAPTER_VERSION, canonical_sym_rsi_001_fixture_manifest,
};
pub use sym_rsi_fresh_evaluation::{
    FreshCvsADisposition, FreshCvsAReceipt, FreshDomainSummary, FreshEvaluationError,
    FreshPairReceipt, SYM_RSI_001_C_VS_A_ANALYSIS_RULE, SYM_RSI_001_FRESH_EVALUATION_SCHEMA,
    run_fresh_c_vs_a,
};
pub use sym_rsi_grounded_dream::{
    DreamActionPrediction, DreamActionSupport, DreamDecisionRecord, DreamFixtureAction,
    GroundedDreamError, GroundedDreamModel, GroundedDreamPolicy,
    DREAM_ACTION_FINGERPRINT_SEMANTICS,
    DREAM_OVERRIDE_MARGIN, DREAM_RISK_PENALTY, DREAM_STATE_DIM,
    SYM_RSI_001_GROUNDED_DREAM_POLICY_ID, SYM_RSI_001_GROUNDED_DREAM_SCORING_RULE,
    SYM_RSI_001_GROUNDED_DREAM_SCHEMA,
    build_grounded_dream_policy, train_grounded_dream_model,
};
pub use sym_rsi_dream_protocol::{
    DreamExtensionProtocolBinding, DreamProtocolError, DREAM_FRESH_SEEDS,
    DREAM_OOD_SEEDS, DREAM_VERIFICATION_SEEDS, INHERITED_TRAINING_SEEDS,
    SYM_RSI_001D_ANALYSIS_RULE, SYM_RSI_001D_EXPERIMENT_ID,
    SYM_RSI_001D_PROTOCOL_SCHEMA, SYM_RSI_001D_QUALITY_TOLERANCE,
    canonical_sym_rsi_001d_manifest, validate_canonical_sym_rsi_001d_manifest,
    validate_dream_extension_seed_independence,
};
pub use sym_rsi_dream_fresh_evaluation::{
    DreamFreshDisposition, DreamFreshDomainSummary, DreamFreshEvaluationError,
    DreamFreshEvaluationReceipt, DreamFreshPairReceipt,
    SYM_RSI_001D_FRESH_EVALUATION_SCHEMA, run_fresh_d_vs_c,
};
pub use sym_rsi_dream_ood_evaluation::{
    DreamOodDisposition, DreamOodDomainSummary, DreamOodEvaluationError,
    DreamOodEvaluationReceipt, DreamOodPairReceipt,
    SYM_RSI_001D_OOD_EVALUATION_SCHEMA, run_ood_d_vs_c,
};
pub use sym_rsi_dream_verification::{
    DreamVerificationCorpus, DreamVerificationCorpusReceipt, DreamVerificationDecision,
    DreamVerificationDomainSummary, DreamVerificationError, DreamVerificationGateReceipt,
    SYM_RSI_001D_VERIFICATION_CORPUS_SCHEMA, SYM_RSI_001D_VERIFICATION_GATE_SCHEMA,
    acquire_dream_verification_corpus, validate_grounded_dream_on_verification,
};
pub use sym_rsi_holdout_gate::{
    HeldOutReplayGateReceipt, HoldoutGateDecision, HoldoutGateError,
    SYM_RSI_001_HOLDOUT_GATE_SCHEMA, select_canonical_training_candidate,
    validate_selected_candidate_on_holdout,
};
pub use sym_rsi_replay_corpus::{
    ReplayCorpusError, ReplayCorpusEvaluation, ReplayFixtureWorld, ReplayWorldEvaluation,
    merge_observed_traces, score_policy_on_replay_worlds,
};
pub use sym_rsi_replay_selection::{
    FixedHashCandidateSpec, ReplayCandidateAssessment, ReplayIneligibility, ReplaySelectionError,
    ReplaySelectionReceipt, SYM_RSI_001_REPLAY_SELECTION_SCHEMA,
    select_fixed_hash_policy_from_replay,
};
pub use sym_rsi_runner::{
    FixedHashPolicy, FixtureObservedStep, FixturePolicy, FixtureRunTrace, FixtureRunnerError,
    ReceiptDiagnostics, build_fixture_receipt, fixture_action_digest, fixture_state_digest,
    run_fixture_policy,
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
