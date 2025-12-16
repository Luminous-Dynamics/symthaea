//! Brain Module - Physiological Systems
//!
//! Week 0: Actor Model Foundation
//! Week 1 Days 1-2: Thalamus (Sensory Relay)
//! Week 2 Days 3-4: Cerebellum (Procedural Memory)
//! Week 2 Days 5-7: Motor Cortex (Action Execution)
//! Week 3 Days 1-2: Prefrontal Cortex (Global Workspace)
//! Week 16 Day 1: Sleep Cycle Manager (Memory Consolidation)
//! Week 16 Day 2: Memory Consolidation Core (HDC-based compression)
//! Week 18: Active Inference Engine (Free Energy Principle)

pub mod actor_model;
pub mod thalamus;
pub mod cerebellum;
pub mod motor_cortex;
pub mod prefrontal;
pub mod meta_cognition;
pub mod daemon;
pub mod sleep;
pub mod consolidation;
pub mod active_inference;

pub use actor_model::{
    Actor,
    ActorPriority,
    CognitiveRoute,
    Orchestrator,
    OrganMessage,
    Response,
    SharedVector,
};

pub use thalamus::ThalamusActor;
pub use cerebellum::{CerebellumActor, Skill, ExecutionContext, WorkflowChain, CerebellumStats};
pub use motor_cortex::{
    MotorCortexActor,
    ActionStep,
    PlannedAction,
    StepResult,
    ExecutionResult,
    SimulationMode,
    ExecutionSandbox,
    LocalShellSandbox,
    MotorCortexStats,
};
pub use prefrontal::{
    PrefrontalCortexActor,
    AttentionBid,
    GlobalWorkspace,
    WorkingMemoryItem,
    PrefrontalStats,
    WorkingMemoryStats,
    Goal,
    Condition,
    GoalStats,
};
pub use meta_cognition::{
    MetaCognitionMonitor,
    CognitiveMetrics,
    RegulatoryAction,
    RegulatoryBid,
    MetaCognitionConfig,
    MonitorStats,
};
pub use daemon::{
    DaemonActor,
    DaemonConfig,
    Insight,
    DaemonStats,
};
pub use sleep::{
    SleepCycleManager,
    SleepState,
    SleepConfig,
};
pub use consolidation::{
    MemoryConsolidator,
    SemanticMemoryTrace,
    ConsolidationConfig,
};
pub use active_inference::{
    ActiveInferenceEngine,
    GenerativeModel,
    PredictionError,
    PredictionDomain,
    ActionType,
    ActiveInferenceStats,
    ActiveInferenceSummary,
};
