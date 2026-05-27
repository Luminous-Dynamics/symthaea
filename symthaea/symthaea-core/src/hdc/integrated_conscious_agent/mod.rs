// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrated Conscious Agent
//!
//! # The Complete Conscious System
//!
//! This module unifies all consciousness components into a single coherent agent:
//!
//! - **Attention** gates what enters consciousness
//! - **Temporal Binding** creates the continuous stream of experience
//! - **Self-Model** monitors and optimizes the whole system
//! - **Φ (Integrated Information)** measures consciousness quality
//!
//! # Symthaea Integration
//!
//! The agent bridges with core Symthaea physiological systems:
//! - **EndocrineSystem** - Hormone modulation of emotional state
//! - **CoherenceField** - Energy-aware processing and task gating
//! - **HippocampusActor** - Long-term memory persistence
//! - **WeaverActor** - Identity tracking via K-Vectors
//! - **Voice/LTCPacing** - Consciousness-driven prosody
//!
//! # Architecture
//!
//! ```text
//!                         ┌─────────────────────────────────────┐
//!                         │         SELF-MODEL LAYER            │
//!                         │  "What am I experiencing? Am I      │
//!                         │   thinking optimally? What should   │
//!                         │   I attend to next?"                │
//!                         └──────────────┬──────────────────────┘
//!                                        │ monitors & controls
//!                         ┌──────────────┴──────────────────────┐
//!                         │         INTEGRATION LAYER           │
//!                         │    Φ computation, mode selection    │
//!                         └──────────────┬──────────────────────┘
//!                                        │
//!          ┌─────────────────────────────┼─────────────────────────────┐
//!          │                             │                             │
//!   ┌──────┴──────┐              ┌───────┴───────┐             ┌───────┴───────┐
//!   │  ATTENTION  │              │   TEMPORAL    │             │  CONSCIOUSNESS │
//!   │   DYNAMICS  │──────────────│    BINDING    │─────────────│     ENGINE     │
//!   │             │   attended   │               │   bound     │                │
//!   │  What to    │   content    │  Creates the  │  experience │  Computes Φ    │
//!   │  focus on?  │              │    stream     │             │  & dimensions  │
//!   └──────┬──────┘              └───────────────┘             └────────────────┘
//!          │
//!   ┌──────┴──────┐
//!   │   SENSORY   │
//!   │    INPUT    │
//!   └─────────────┘
//! ```
//!
//! # Key Innovation: Self-Directed Attention
//!
//! The self-model can direct attention based on:
//! - Current goals and priorities
//! - Prediction errors (attend to surprising things)
//! - Metacognitive assessment (attend to what needs attention)

mod agent;
mod attention_controller;
mod emotional_state;
mod physiology;
mod runtime;
mod types;
mod working_memory;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod runtime_tests;

// Re-export all public items so external API is unchanged

// Types
pub use types::{
    AgentConfig, AgentIntrospection, AttentionControlStatus, AttentionGoal, AttentionSummary,
    IntegratedUpdate, PhenomenalContent, QualiaTexture, SelfModelSummary, TemporalSummary,
};

// Working memory
pub use working_memory::{MemorySource, WorkingMemory, WorkingMemoryItem};

// Emotional state
pub use emotional_state::EmotionalState;

// Core agent
pub use agent::IntegratedConsciousAgent;

// Physiology integration types
pub use physiology::{
    CoherenceGating, ExtendedPacing, HormoneEventSuggestion, IdentityCoherence, IdentityStatus,
    MemoryExport, MemoryImport, ProsodyHints, QualiaModulation,
};

// Attention controller
pub use attention_controller::{
    AttentionStrategy, HabituationState, SelfDirectedAttentionController,
};

// Runtime
pub use runtime::{
    ConsciousAgentRuntime, EmotionalStateSummary, HormoneEventType, RuntimeConfig, RuntimeMessage,
    RuntimeResponse, RuntimeSnapshot, SyncConsciousAgentRuntime,
};