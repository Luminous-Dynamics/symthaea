//! # Ultimate Breakthrough #3: Recursive Self-Improvement
//!
//! This module implements the REVOLUTIONARY recursive self-improvement system
//! where the AI uses causal reasoning to understand and improve its own architecture!
//!
//! ## Module Structure
//!
//! The recursive improvement system is organized into logical submodules:
//!
//! - `core`: Main implementation (being incrementally split)
//! - `types`: Shared types (ComponentId, ImprovementType, etc.)
//! - `performance_monitor`: Performance tracking and bottleneck detection
//!
//! ## Future Modules (TODO: Extract from core.rs)
//!
//! - `architectural_graph`: Causal graph modeling component interactions
//! - `safe_experiments`: Sandboxed testing of improvements
//! - `improvement_generator`: Proposes optimizations
//! - `recursive_optimizer`: Coordinates the improvement loop
//! - `gradient_optimizer`: Gradient-based consciousness optimization
//! - `intrinsic_motivation`: Curiosity, competence, autonomy modules
//! - `self_model`: Capability estimation and trajectory planning
//! - `world_model`: Latent state and dynamics modeling
//! - `meta_cognitive`: Resource management and attention
//! - `predictive_router`: Predictive routing strategies
//! - `oscillatory_router`: Phase-locked processing
//! - `causal_validated_router`: Causal emergence validation
//! - `geometric_router`: Information geometry routing
//! - `topological_router`: Persistent homology routing
//! - `quantum_router`: Quantum coherence routing
//! - `active_inference_router`: Free energy minimization
//! - `routing_hub`: Unified routing coordination
//! - `predictive_processing_router`: Hierarchical predictive coding
//! - `ast_router`: Attention schema theory routing
//! - `benchmark_suite`: Router benchmarking
//! - `meta_router`: Paradigm selection
//! - `global_workspace_router`: Global workspace theory routing

// Core implementation (all 19,853 lines - to be incrementally split)
mod core;

// Shared types (extracted for cleaner dependencies)
pub mod types;

// Routers module
pub mod routers;

// Re-export everything from core for backward compatibility
pub use core::*;

// Re-export shared types
pub use types::{
    ComponentId, ImprovementType, AccuracyMetric, BottleneckType,
    metric_to_component, suggest_latency_fix, suggest_accuracy_fix, calculate_trend,
};
