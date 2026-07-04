// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The cognitive core — world model, active inference, causal reasoning.
//!
//! This is the heart of the NixOS mind. Instead of classifying intents,
//! the mind maintains a generative model of the system and selects actions
//! by minimizing expected free energy.
//!
//! ## Modules
//!
//! - `world_model`: Generative model with HDC delta vector transitions
//! - `active_inference`: Action selection via expected free energy minimization
//! - `goal_inference`: User input → desired system state in HDC space
//! - `predictive_hierarchy`: 4-level prediction stack (sensory → goals)
//! - `working_memory`: 7-item capacity-limited context with activation decay
//! - `episodic_memory`: Φ-gated system event consolidation
//! - `causal_graph`: Causal relationships between NixOS options

pub mod active_inference;
pub mod causal_graph;
pub mod episodic_memory;
pub mod goal_inference;
pub mod hdc_world_model;
pub mod journal_anomaly;
pub mod neural_bridge;
pub mod ollama_bridge;
pub mod predictive_hierarchy;
pub mod working_memory;
pub mod world_model;

pub use active_inference::{ActionPlan, NixActiveInference, ScoredAction};
pub use causal_graph::{CausalEdge, NixCausalGraph, RootCauseAnalysis, SideEffectPrediction};
pub use episodic_memory::{EpisodeOutcome, NixEpisodicMemory, SystemEpisode};
pub use goal_inference::{GoalInference, InferredGoal};
pub use hdc_world_model::{DriftReport, HdcWorldModel, StateProjection};
pub use journal_anomaly::JournalAnomalyDetector;
pub use neural_bridge::{
    BridgeError, EmbeddingProvenance, EmbeddingResult, NeuralBridge, NeuralBridgeConfig,
};
pub use ollama_bridge::{OllamaBridge, OllamaBridgeConfig, OllamaResponse};
pub use predictive_hierarchy::{PredictionLevel, PredictiveHierarchy};
pub use working_memory::{MemoryItem, MemorySource, SavedWorkingMemory, WorkingMemory};
pub use world_model::{ActionCategory, NixWorldModel};
