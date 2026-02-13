//! The cognitive core — world model, active inference, causal reasoning.
//!
//! This is the heart of the NixOS mind. Instead of classifying intents,
//! the mind maintains a generative model of the system and selects actions
//! by minimizing expected free energy.
//!
//! ## Modules
//!
//! - [`world_model`]: Generative model with HDC delta vector transitions
//! - [`active_inference`]: Action selection via expected free energy minimization
//! - [`goal_inference`]: User input → desired system state in HDC space
//! - [`predictive_hierarchy`]: 4-level prediction stack (sensory → goals)
//! - [`working_memory`]: 7-item capacity-limited context with activation decay
//! - [`episodic_memory`]: Φ-gated system event consolidation
//! - [`causal_graph`]: Causal relationships between NixOS options

pub mod world_model;
pub mod hdc_world_model;
pub mod active_inference;
pub mod causal_graph;
pub mod predictive_hierarchy;
pub mod working_memory;
pub mod episodic_memory;
pub mod goal_inference;
pub mod journal_anomaly;

pub use world_model::{NixWorldModel, ActionCategory};
pub use hdc_world_model::{HdcWorldModel, DriftReport, StateProjection};
pub use active_inference::{NixActiveInference, ActionPlan, ScoredAction};
pub use goal_inference::{GoalInference, InferredGoal};
pub use predictive_hierarchy::{PredictiveHierarchy, PredictionLevel};
pub use working_memory::{WorkingMemory, MemoryItem, MemorySource, SavedWorkingMemory};
pub use episodic_memory::{NixEpisodicMemory, SystemEpisode, EpisodeOutcome};
pub use causal_graph::{NixCausalGraph, CausalEdge, RootCauseAnalysis, SideEffectPrediction};
pub use journal_anomaly::JournalAnomalyDetector;
