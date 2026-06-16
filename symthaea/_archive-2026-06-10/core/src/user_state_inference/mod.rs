// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # User State Inference: Understanding User Context and Needs
//!
//! This module infers user state from interaction patterns, enabling
//! adaptive and empathic responses. It analyzes:
//!
//! - Context (what the user is trying to do)
//! - Cognitive load (how complex their current task is)
//! - Emotional state (frustration, confusion, flow)
//! - Experience level (beginner, intermediate, expert)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  USER STATE INFERENCE                            │
//! │                                                                  │
//! │   User Input                                                     │
//! │       │                                                          │
//! │       ▼                                                          │
//! │   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐│
//! │   │   Context    │ → │   Cognitive  │ → │   Emotional State    ││
//! │   │   Detection  │   │   Load Est.  │   │   Inference          ││
//! │   └──────────────┘   └──────────────┘   └──────────────────────┘│
//! │           │                 │                     │              │
//! │           └────────────────┼─────────────────────┘              │
//! │                            ▼                                     │
//! │                    ┌──────────────────┐                          │
//! │                    │  User State      │                          │
//! │                    │  (composite)     │                          │
//! │                    └──────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Kind of context the user is operating in
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ContextKind {
    /// Exploring or learning about the system
    Exploration,
    /// Actively working on a task
    Task,
    /// Debugging or troubleshooting an issue
    Troubleshooting,
    /// Configuring or customizing the system
    Configuration,
    /// Seeking help or documentation
    Help,
    /// General conversation or chitchat
    Conversation,
    /// System maintenance or administration
    Maintenance,
    /// Development and coding
    Development,
    /// Development work (alias for Development)
    DevWork,
    /// System upgrade operations
    Upgrade,
    /// Handling errors or failures
    ErrorHandling,
    /// Writing documentation or content
    Writing,
    /// Reviewing code or content
    Review,
    /// Planning or architecting
    Planning,
    /// Initial setup or onboarding
    Setup,
    /// Unknown context
    #[default]
    Unknown,
}

impl ContextKind {
    /// Detect context from text
    pub fn detect(text: &str) -> Self {
        let text_lower = text.to_lowercase();

        // Troubleshooting patterns
        if text_lower.contains("error")
            || text_lower.contains("fail")
            || text_lower.contains("broken")
            || text_lower.contains("not working")
            || text_lower.contains("fix")
            || text_lower.contains("debug")
        {
            return ContextKind::Troubleshooting;
        }

        // Help patterns
        if text_lower.contains("help")
            || text_lower.contains("how do i")
            || text_lower.contains("how to")
            || text_lower.contains("what is")
            || text_lower.contains("explain")
            || text_lower.contains("documentation")
        {
            return ContextKind::Help;
        }

        // Configuration patterns
        if text_lower.contains("config")
            || text_lower.contains("setting")
            || text_lower.contains("option")
            || text_lower.contains("preference")
            || text_lower.contains("customize")
            || text_lower.contains("configure")
        {
            return ContextKind::Configuration;
        }

        // Development patterns
        if text_lower.contains("code")
            || text_lower.contains("function")
            || text_lower.contains("implement")
            || text_lower.contains("program")
            || text_lower.contains("develop")
            || text_lower.contains("build")
        {
            return ContextKind::Development;
        }

        // Task patterns
        if text_lower.contains("install")
            || text_lower.contains("remove")
            || text_lower.contains("update")
            || text_lower.contains("run")
            || text_lower.contains("start")
            || text_lower.contains("stop")
        {
            return ContextKind::Task;
        }

        // Exploration patterns
        if text_lower.contains("search")
            || text_lower.contains("find")
            || text_lower.contains("list")
            || text_lower.contains("show")
            || text_lower.contains("available")
            || text_lower.contains("options")
        {
            return ContextKind::Exploration;
        }

        // Maintenance patterns
        if text_lower.contains("clean")
            || text_lower.contains("garbage")
            || text_lower.contains("optimize")
            || text_lower.contains("maintenance")
            || text_lower.contains("backup")
            || text_lower.contains("restore")
        {
            return ContextKind::Maintenance;
        }

        ContextKind::Unknown
    }

    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ContextKind::Exploration => "Exploration",
            ContextKind::Task => "Task Execution",
            ContextKind::Troubleshooting => "Troubleshooting",
            ContextKind::Configuration => "Configuration",
            ContextKind::Help => "Help Seeking",
            ContextKind::Conversation => "Conversation",
            ContextKind::Maintenance => "Maintenance",
            ContextKind::Development => "Development",
            ContextKind::DevWork => "Development Work",
            ContextKind::Upgrade => "System Upgrade",
            ContextKind::ErrorHandling => "Error Handling",
            ContextKind::Writing => "Writing",
            ContextKind::Review => "Review",
            ContextKind::Planning => "Planning",
            ContextKind::Setup => "Setup",
            ContextKind::Unknown => "Unknown",
        }
    }
}

/// User experience level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ExperienceLevel {
    /// New to the system, needs guidance
    #[default]
    Beginner,
    /// Has some experience, familiar with basics
    Intermediate,
    /// Experienced user, prefers efficiency
    Expert,
}

/// Inferred cognitive load
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CognitiveLoadEstimate {
    /// Overall cognitive load (0.0 = low, 1.0 = high)
    pub level: f64,

    /// Confidence in this estimate
    pub confidence: f64,

    /// Primary contributor to load
    pub primary_factor: CognitiveLoadFactor,
}

impl Default for CognitiveLoadEstimate {
    fn default() -> Self {
        Self {
            level: 0.5,
            confidence: 0.3,
            primary_factor: CognitiveLoadFactor::Unknown,
        }
    }
}

/// Factors contributing to cognitive load
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveLoadFactor {
    /// Task complexity
    TaskComplexity,
    /// Information overload
    InformationOverload,
    /// Time pressure
    TimePressure,
    /// Error recovery
    ErrorRecovery,
    /// Learning curve
    LearningCurve,
    /// Multitasking
    Multitasking,
    /// Unknown
    Unknown,
}

/// Complete user state inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    /// Current context
    pub context: ContextKind,

    /// Experience level
    pub experience: ExperienceLevel,

    /// Cognitive load estimate
    pub cognitive_load: CognitiveLoadEstimate,

    /// Frustration level (0.0 = calm, 1.0 = very frustrated)
    pub frustration: f64,

    /// Confidence level (0.0 = confused, 1.0 = confident)
    pub confidence: f64,

    /// Engagement level (0.0 = disengaged, 1.0 = highly engaged)
    pub engagement: f64,

    /// Time since last interaction (seconds)
    pub idle_time_secs: f32,

    /// Interaction count in current session
    pub interaction_count: u32,

    /// Timestamp of this inference
    pub timestamp: u64,
}

impl Default for UserState {
    fn default() -> Self {
        Self {
            context: ContextKind::Unknown,
            experience: ExperienceLevel::Beginner,
            cognitive_load: CognitiveLoadEstimate::default(),
            frustration: 0.0,
            confidence: 0.5,
            engagement: 0.5,
            idle_time_secs: 0.0,
            interaction_count: 0,
            timestamp: 0,
        }
    }
}

impl UserState {
    /// Create a new user state for a given context
    pub fn new(context: ContextKind) -> Self {
        Self {
            context,
            ..Default::default()
        }
    }

    /// Check if user needs help
    pub fn needs_help(&self) -> bool {
        self.frustration > 0.5
            || self.confidence < 0.3
            || self.context == ContextKind::Help
            || self.context == ContextKind::Troubleshooting
    }

    /// Check if user is in flow state
    pub fn is_in_flow(&self) -> bool {
        self.engagement > 0.7 && self.frustration < 0.2 && self.confidence > 0.6
    }

    /// Get recommended response verbosity
    pub fn recommended_verbosity(&self) -> Verbosity {
        match self.experience {
            ExperienceLevel::Beginner => Verbosity::Detailed,
            ExperienceLevel::Intermediate => {
                if self.needs_help() {
                    Verbosity::Detailed
                } else {
                    Verbosity::Normal
                }
            }
            ExperienceLevel::Expert => {
                if self.needs_help() {
                    Verbosity::Normal
                } else {
                    Verbosity::Concise
                }
            }
        }
    }
}

/// Recommended verbosity level for responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verbosity {
    /// Brief, to the point
    Concise,
    /// Balanced detail
    Normal,
    /// Comprehensive explanations
    Detailed,
}

/// User state inference engine
#[derive(Debug)]
pub struct UserStateInference {
    /// Current inferred state
    current_state: UserState,

    /// History of interactions for pattern analysis
    interaction_history: VecDeque<InteractionRecord>,

    /// Last interaction time
    last_interaction: Option<Instant>,

    /// Configuration
    config: InferenceConfig,
}

/// Record of a single interaction
#[derive(Debug, Clone)]
pub struct InteractionRecord {
    /// Input text
    pub text: String,

    /// Detected context
    pub context: ContextKind,

    /// Response time (ms)
    pub response_time_ms: f32,

    /// Whether it resulted in an error
    pub had_error: bool,

    /// Timestamp
    pub timestamp: Instant,
}

/// Configuration for inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum history to keep
    pub max_history: usize,

    /// Weight for recent interactions
    pub recency_weight: f64,

    /// Frustration decay per minute
    pub frustration_decay: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_history: 50,
            recency_weight: 0.8,
            frustration_decay: 0.1,
        }
    }
}

impl UserStateInference {
    /// Create a new inference engine
    pub fn new() -> Self {
        Self {
            current_state: UserState::default(),
            interaction_history: VecDeque::new(),
            last_interaction: None,
            config: InferenceConfig::default(),
        }
    }

    /// Process a new user input and update state
    pub fn process(&mut self, text: &str, had_error: bool) -> &UserState {
        let now = Instant::now();

        // Update idle time
        if let Some(last) = self.last_interaction {
            self.current_state.idle_time_secs = last.elapsed().as_secs_f32();
        }
        self.last_interaction = Some(now);

        // Detect context
        let context = ContextKind::detect(text);
        self.current_state.context = context;

        // Update frustration based on errors
        if had_error {
            self.current_state.frustration = (self.current_state.frustration + 0.2).min(1.0);
        } else {
            // Decay frustration on success
            self.current_state.frustration = (self.current_state.frustration - 0.1).max(0.0);
        }

        // Update confidence based on context
        match context {
            ContextKind::Help | ContextKind::Troubleshooting => {
                self.current_state.confidence = (self.current_state.confidence - 0.1).max(0.0);
            }
            ContextKind::Task | ContextKind::Development => {
                if !had_error {
                    self.current_state.confidence = (self.current_state.confidence + 0.05).min(1.0);
                }
            }
            _ => {}
        }

        // Estimate cognitive load
        self.estimate_cognitive_load(text);

        // Update interaction count
        self.current_state.interaction_count += 1;

        // Record interaction
        let record = InteractionRecord {
            text: text.to_string(),
            context,
            response_time_ms: 0.0,
            had_error,
            timestamp: now,
        };
        self.interaction_history.push_back(record);

        // Trim history if needed
        if self.interaction_history.len() > self.config.max_history {
            self.interaction_history.pop_front();
        }

        // Update timestamp
        self.current_state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        &self.current_state
    }

    /// Estimate cognitive load from text
    fn estimate_cognitive_load(&mut self, text: &str) {
        let word_count = text.split_whitespace().count();
        let has_technical = text.contains("error")
            || text.contains("config")
            || text.contains("derivation")
            || text.contains("flake");

        let load = (word_count as f64 / 50.0).min(0.5)
            + if has_technical { 0.3 } else { 0.0 }
            + self.current_state.frustration * 0.2;

        self.current_state.cognitive_load = CognitiveLoadEstimate {
            level: load.min(1.0),
            confidence: 0.5,
            primary_factor: if has_technical {
                CognitiveLoadFactor::TaskComplexity
            } else if self.current_state.frustration > 0.5 {
                CognitiveLoadFactor::ErrorRecovery
            } else {
                CognitiveLoadFactor::Unknown
            },
        };
    }

    /// Get the current state
    pub fn state(&self) -> &UserState {
        &self.current_state
    }

    /// Update experience level based on observed patterns
    pub fn update_experience(&mut self, level: ExperienceLevel) {
        self.current_state.experience = level;
    }

    /// Reset the state
    pub fn reset(&mut self) {
        self.current_state = UserState::default();
        self.interaction_history.clear();
        self.last_interaction = None;
    }

    /// Record that an error occurred in the interaction
    ///
    /// This increases frustration and cognitive load, indicating
    /// the user may be struggling.
    pub fn record_error(&mut self) {
        self.current_state.frustration = (self.current_state.frustration + 0.15).min(1.0);
        self.current_state.cognitive_load.level =
            (self.current_state.cognitive_load.level + 0.1).min(1.0);
        self.current_state.cognitive_load.primary_factor = CognitiveLoadFactor::ErrorRecovery;
    }

    /// Record that the user initiated an undo action
    ///
    /// This may indicate frustration with the previous action's result.
    pub fn record_undo(&mut self) {
        self.current_state.frustration = (self.current_state.frustration + 0.1).min(1.0);
        // Undo suggests trying to fix something - slight cognitive load increase
        self.current_state.cognitive_load.level =
            (self.current_state.cognitive_load.level + 0.05).min(1.0);
    }

    /// Infer user state from context for empathic unification
    ///
    /// This method provides a simplified inference based on context,
    /// returning a resonant_speech::UserState for empathic processing.
    pub fn infer(&self, context: ContextKind, _locale: &str) -> crate::resonant_speech::UserState {
        // Update context in a temporary state
        let mut inferred = self.current_state.clone();
        inferred.context = context;

        // Context-based adjustments
        match context {
            ContextKind::ErrorHandling | ContextKind::Troubleshooting => {
                inferred.frustration = (inferred.frustration + 0.2).min(1.0);
                inferred.cognitive_load.level = (inferred.cognitive_load.level + 0.2).min(1.0);
            }
            ContextKind::Exploration | ContextKind::Help => {
                inferred.confidence = (inferred.confidence - 0.1).max(0.0);
            }
            ContextKind::Setup | ContextKind::Configuration => {
                inferred.cognitive_load.level = (inferred.cognitive_load.level + 0.1).min(1.0);
            }
            _ => {}
        }

        crate::resonant_speech::UserState::from_inferred(&inferred)
    }
}

impl Default for UserStateInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_context_detection() {
        assert_eq!(
            ContextKind::detect("error: package not found"),
            ContextKind::Troubleshooting
        );
        assert_eq!(
            ContextKind::detect("how do I install vim"),
            ContextKind::Help
        );
        assert_eq!(
            ContextKind::detect("configure network settings"),
            ContextKind::Configuration
        );
        assert_eq!(ContextKind::detect("install firefox"), ContextKind::Task);
    }

    #[test]
    fn test_user_state_inference() {
        let mut inference = UserStateInference::new();

        // Process a troubleshooting input
        let state = inference.process("error: derivation failed", true);
        assert_eq!(state.context, ContextKind::Troubleshooting);
        assert!(state.frustration > 0.0);

        // Process a successful task
        let state = inference.process("install vim", false);
        assert_eq!(state.context, ContextKind::Task);
    }

    #[test]
    fn test_needs_help() {
        let mut state = UserState::default();
        state.frustration = 0.6;
        assert!(state.needs_help());

        state.frustration = 0.0;
        state.confidence = 0.2;
        assert!(state.needs_help());
    }

    #[test]
    fn test_flow_state() {
        let mut state = UserState::default();
        state.engagement = 0.8;
        state.frustration = 0.1;
        state.confidence = 0.7;
        assert!(state.is_in_flow());
    }
}
