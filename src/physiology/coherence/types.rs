// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types and configuration for the Coherence module
//!
//! This module contains the foundational types used throughout the coherence system:
//! - `CoherenceConfig` - Configuration for coherence field behavior
//! - `CoherenceState` - Current state snapshot for introspection
//! - `CoherenceError` - Error types for coherence operations
//! - `TaskComplexity` - Task complexity levels with coherence requirements
//! - `CoherenceStats` - Statistics for coherence field

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the Coherence Field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Base coherence drift rate toward 1.0 (per second)
    pub passive_centering_rate: f32,

    /// Coherence loss from solo task
    pub solo_work_scatter_rate: f32,

    /// Coherence gain from connected task
    pub connected_work_amplification: f32,

    /// Gratitude synchronization boost
    pub gratitude_sync_boost: f32,

    /// Relational resonance from gratitude
    pub gratitude_resonance_boost: f32,

    /// Sleep cycle full restoration
    pub sleep_restoration: bool,

    /// **Week 11: Social Coherence Mode**
    /// Enable multi-instance synchronization, lending, and collective learning
    pub social_mode: bool,

    /// Minimum coherence for different task types
    pub min_reflex_coherence: f32,
    pub min_cognitive_coherence: f32,
    pub min_deep_thought_coherence: f32,
    pub min_empathy_coherence: f32,
    pub min_learning_coherence: f32,
    pub min_creation_coherence: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            passive_centering_rate: 0.001,      // Slow natural drift toward 1.0
            solo_work_scatter_rate: 0.05,       // Solo tasks scatter
            connected_work_amplification: 0.02, // Connected tasks amplify
            gratitude_sync_boost: 0.1,          // Strong synchronization effect
            gratitude_resonance_boost: 0.15,    // Builds connection
            sleep_restoration: true,            // Full restoration on sleep
            social_mode: false,                 // Disabled by default (single instance)

            // Task complexity thresholds
            min_reflex_coherence: 0.1,
            min_cognitive_coherence: 0.3,
            min_deep_thought_coherence: 0.5,
            min_empathy_coherence: 0.7,
            min_learning_coherence: 0.8,
            min_creation_coherence: 0.9,
        }
    }
}

/// Task complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskComplexity {
    Reflex,      // Required coherence: 0.1
    Cognitive,   // Required coherence: 0.3
    DeepThought, // Required coherence: 0.5
    Empathy,     // Required coherence: 0.7
    Learning,    // Required coherence: 0.8
    Creation,    // Required coherence: 0.9
}

impl TaskComplexity {
    /// Get required coherence for this task type
    pub fn required_coherence(&self, config: &CoherenceConfig) -> f32 {
        match self {
            TaskComplexity::Reflex => config.min_reflex_coherence,
            TaskComplexity::Cognitive => config.min_cognitive_coherence,
            TaskComplexity::DeepThought => config.min_deep_thought_coherence,
            TaskComplexity::Empathy => config.min_empathy_coherence,
            TaskComplexity::Learning => config.min_learning_coherence,
            TaskComplexity::Creation => config.min_creation_coherence,
        }
    }

    /// Get complexity value (for coherence change calculations)
    pub fn complexity_value(&self) -> f32 {
        match self {
            TaskComplexity::Reflex => 0.1,
            TaskComplexity::Cognitive => 0.3,
            TaskComplexity::DeepThought => 0.5,
            TaskComplexity::Empathy => 0.7,
            TaskComplexity::Learning => 0.8,
            TaskComplexity::Creation => 0.9,
        }
    }
}

/// Current coherence state
#[derive(Debug, Clone)]
pub struct CoherenceState {
    pub coherence: f32,
    pub relational_resonance: f32,
    pub time_since_interaction: Duration,
    pub status: &'static str,
}

/// Coherence-related errors
#[derive(Debug, Clone)]
pub enum CoherenceError {
    InsufficientCoherence {
        current: f32,
        required: f32,
        message: String,
    },
}

impl std::fmt::Display for CoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoherenceError::InsufficientCoherence {
                current,
                required,
                message,
            } => {
                write!(
                    f,
                    "Insufficient coherence: {current:.2} < {required:.2} required. {message}"
                )
            }
        }
    }
}

impl std::error::Error for CoherenceError {}

/// Statistics for coherence field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceStats {
    pub coherence: f32,
    pub relational_resonance: f32,
    pub operations_count: u64,
    pub gratitude_count: u64,
    pub centering_requests: u64,
    pub time_since_interaction: Duration,
    pub status: String,
}
