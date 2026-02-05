//! # Ultimate Breakthrough #3: Recursive Self-Improvement
//!
//! This module implements the REVOLUTIONARY recursive self-improvement system
//! where the AI uses causal reasoning to understand and improve its own architecture!
//!
//! ## Module Structure
//!
//! The recursive improvement system is organized into 26 logical submodules:
//!
//! ### Core Modules
//! - `core`: Legacy backward-compatibility re-exports and tests
//! - `types`: Shared types (ComponentId, ImprovementType, etc.)
//!
//! ### Self-Improvement Infrastructure
//! - `architectural_graph`: Causal graph modeling component interactions
//! - `safe_experiment`: Sandboxed testing of improvements
//! - `improvement_generator`: Proposes optimizations based on bottlenecks
//! - `recursive_optimizer`: Coordinates the improvement loop
//! - `gradient_optimizer`: Gradient-based consciousness (Φ) optimization
//! - `intrinsic_motivation`: Curiosity, competence, autonomy drives
//! - `self_model`: Capability estimation and trajectory planning
//! - `world_model`: Latent state and dynamics modeling
//! - `meta_cognitive`: Resource management and attention broadcasting
//!
//! ### Consciousness Routing Paradigms (consolidated in `routers/` module)
//! All routers now live in the `routers/` submodule with shared types:
//! - `routers::PredictiveRouter`: World model predictions
//! - `routers::OscillatoryRouter`: Phase-locked neural oscillation (Phase 5G)
//! - `routers::CausalValidatedRouter`: Causal emergence validation (IIT-based)
//! - `routers::InformationGeometricRouter`: Fisher information geometry
//! - `routers::TopologicalConsciousnessRouter`: Persistent homology
//! - `routers::QuantumCoherenceRouter`: Quantum-inspired coherence
//! - `routers::ActiveInferenceRouter`: Free energy minimization (Friston)
//! - `routers::PredictiveProcessingRouter`: Hierarchical predictive coding
//! - `routers::ASTRouter`: Attention Schema Theory (Graziano)
//! - `routers::MetaRouter`: Paradigm selection via UCB1 bandit
//! - `routers::GlobalWorkspaceRouter`: Global Workspace Theory (Baars)
//! - `routers::ResonantConsciousnessRouter`: HDC+LTC+Resonator fusion (Phase 5H)
//!
//! ### Routing Coordination
//! - `routing_hub`: Unified routing coordination across paradigms
//! - `benchmark_suite`: Performance benchmarking for all routers
//! - `routers`: **All router implementations and shared types**

// Core implementation (being incrementally split)
mod core;

// Shared types (extracted for cleaner dependencies)
pub mod types;

// Architectural causal graph (extracted from core.rs)
pub mod architectural_graph;

// Safe experimentation framework (extracted from core.rs)
pub mod safe_experiment;

// Improvement generator (extracted from core.rs)
pub mod improvement_generator;

// Recursive optimizer coordination loop (extracted from core.rs)
pub mod recursive_optimizer;

// Gradient-based consciousness optimizer (extracted from core.rs)
pub mod gradient_optimizer;

// Intrinsic motivation system (extracted from core.rs)
pub mod intrinsic_motivation;

// Self-modeling consciousness (extracted from core.rs)
pub mod self_model;

// Consciousness world models (extracted from core.rs)
pub mod world_model;

// Meta-cognitive controller (extracted from core.rs)
pub mod meta_cognitive;

// Unified routing hub (extracted from core.rs)
pub mod routing_hub;

// NOTE: Individual *_router.rs files have been consolidated into the routers/ module
// (Phase 5G/5H improvements). All router types are now exported via routers::*

// Benchmark suite (extracted from core.rs)
pub mod benchmark_suite;

// Routers module
pub mod routers;

// Re-export everything from core for backward compatibility
pub use core::*;

// Re-export shared types
pub use types::{
    ComponentId, ImprovementType, AccuracyMetric, BottleneckType, Bottleneck,
    metric_to_component, suggest_latency_fix, suggest_accuracy_fix, calculate_trend,
    instant_now,
};

// Re-export architectural graph types
pub use architectural_graph::{
    ArchitecturalCausalGraph, ComponentNode, ArchitecturalEdge,
    CausalRelationship, PerformanceImpact, CausalChain, GraphStats,
};

// Re-export safe experiment types
pub use safe_experiment::{
    SafeExperiment, SystemSnapshot, ArchitecturalImprovement,
    SuccessCriteria, RollbackCondition, ExperimentStatus,
    ValidationRun, ExperimentConfig,
};

// Re-export improvement generator types
pub use improvement_generator::{
    ImprovementGenerator, ImprovementRecord, ImprovementOutcome,
    ImprovementPatterns, CausalPattern, GeneratorConfig, GeneratorStats,
};

// Re-export recursive optimizer types
pub use recursive_optimizer::{
    RecursiveOptimizer, OptimizationCycle, OptimizerConfig, OptimizerStats,
};

// Re-export gradient optimizer types
pub use gradient_optimizer::{
    ConsciousnessGradientOptimizer, ArchitecturalParameter, ConsciousnessGradient,
    OptimizationObjective, AdamState, GradientOptimizerConfig, GradientOptimizerStats,
    GradientStep,
};

// Re-export intrinsic motivation types
pub use intrinsic_motivation::{
    IntrinsicMotivationSystem, DriveType, DriveState, AutonomousGoal,
    CuriosityModule, CompetenceModule, AutonomyModule,
    MotivationConfig, MotivationStats, MotivatedAction,
};

// Re-export self-model types
pub use self_model::{
    CapabilityDomain, CapabilityEstimate, KnownLimitation, PredictionRecord,
    SelfModelConfig, SelfModel, BehaviorPrediction, CalibrationStats,
    ImprovementTrajectory, ImprovementStep, ImprovementMethod, DesiredSelfState,
    UnifiedImprovementController, ControllerState, ControllerConfig, ControllerStats,
    RecommendedAction, ControllerOutput,
};

// Re-export world-model types
pub use world_model::{
    LatentConsciousnessState, ConsciousnessAction, ConsciousnessTransition,
    ConsciousnessDynamicsModel, RewardPredictor, Counterfactual,
    ConsciousnessWorldModel, WorldModelConfig, WorldModelStats, WorldModelSummary,
};

// Re-export meta-cognitive types
pub use meta_cognitive::{
    CognitiveResourceType, CognitiveResources, SubsystemId, SubsystemHealth,
    MetaGoal, MetaGoalType, MetaCognitiveConfig, MetaCognitiveStats,
    AttentionBroadcast, BroadcastContentType, MetaCognitiveController,
    MetaCognitiveSummary,
};

// Re-export predictive router types
// TODO: These modules are planned but not yet implemented
// pub use predictive_router::{
//     RoutingStrategy, PredictedRoute, RoutingPlan, PredictiveRouterConfig,
//     PredictiveRouterStats, RoutingOutcome, PredictiveRouter,
//     CounterfactualAnalysis, PredictiveRouterSummary,
// };

// Re-export oscillatory router types
// pub use oscillatory_router::{
//     OscillatoryPhase, ProcessingMode, PhaseProcessingProfile, OscillatoryState,
//     PhaseWindow, OscillatoryRouterConfig, OscillatoryRouterStats,
//     ScheduledOperation, PhaseLockedPlan, CombinedRoutingStrategy,
//     OscillatoryRouter, OscillatoryRouterSummary,
// };

// Re-export causal validated router types
// pub use causal_validated_router::{
//     EffectiveInformation, CausalEmergence, EmergenceInterpretation,
//     CausalValidatedConfig, CausalValidatedStats, CausalRoutingMode,
//     CausalValidatedRouter, ValidatedRoutingDecision, CausalValidatedSummary,
// };

// Re-export geometric router types
// pub use geometric_router::{
//     FisherInformationMatrix, ManifoldPoint, Geodesic,
//     GeometricRouterConfig, GeometricRouterStats, GeometricRouterSummary,
//     InformationGeometricRouter, GeometricRoutingDecision,
// };

// Re-export topological router types
// pub use topological_router::{
//     Simplex, PersistenceInterval, PersistenceDiagram,
//     VietorisRipsComplex, PersistentHomology, TopologicalSignature,
//     TopologicalRouterConfig, TopologicalRouterStats, TopologicalRoutingDecision,
//     TopologicalRouterSummary, TopologicalConsciousnessRouter,
// };

// Re-export quantum router types
// pub use quantum_router::{
//     ComplexAmplitude, QuantumStateVector, DensityMatrix, RoutingHamiltonian,
//     QuantumRouterConfig, QuantumRouterStats, QuantumRoutingDecision,
//     QuantumRouterSummary, QuantumCoherenceRouter,
// };

// Re-export active inference router types
// pub use active_inference_router::{
//     BeliefDistribution, GenerativeModel, ExpectedFreeEnergy, Preferences,
//     ActiveInferenceConfig, ActiveInferenceStats, ActiveInferenceDecision,
//     ActiveInferenceSummary, ActiveInferenceRouter,
// };

// Re-export predictive processing router types
// pub use predictive_processing_router::{
//     PredictiveLevel, HierarchicalWeights, PredictiveProcessingConfig,
//     PredictiveProcessingStats, PredictiveProcessingDecision,
//     PredictiveProcessingRouter,
// };

// Re-export AST router types
// pub use ast_router::{
//     AttentionState, AttentionSchema, SocialAttentionModel,
//     ASTRouterConfig, ASTRouterStats, ASTRoutingDecision, ASTRouter,
// };

// Re-export meta-router types
// pub use meta_router::{
//     RoutingParadigm, ContextProfile, ParadigmStats,
//     MetaRouterConfig, MetaRouterStats, MetaRouterDecision, MetaRouter,
// };

// Re-export global workspace router types
// pub use global_workspace_router::{
//     WorkspaceModule, WorkspaceEntry, BroadcastEvent,
//     GlobalWorkspaceConfig, GlobalWorkspaceStats, GlobalWorkspaceDecision,
//     GlobalWorkspaceRouter,
// };

// Re-export routing hub types
pub use routing_hub::{
    RoutingMode, RouterType, UnifiedRoutingDecision, RouterPerformance,
    RoutingHubConfig, ConsciousnessRoutingHub,
};

// Re-export benchmark suite types
pub use benchmark_suite::{
    RouterBenchmark, ComparativeBenchmark, BenchmarkConfig, RouterBenchmarkSuite,
};
