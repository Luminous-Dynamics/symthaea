//! Configuration and data types for the Continuous Mind system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::RealHV;
use crate::chronobiology::Biorhythm;

/// Configuration for the continuous mind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MindConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Tick rate (Hz)
    pub tick_rate: f32,
    /// Working memory capacity
    pub working_memory_capacity: usize,
    /// Enable consciousness monitoring
    pub consciousness_monitoring: bool,
    /// Enable learning
    pub learning_enabled: bool,
    /// Learning rate
    pub learning_rate: f32,
    /// Minimum consciousness threshold for action
    pub min_consciousness: f64,
}

impl Default for MindConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            tick_rate: 10.0,
            working_memory_capacity: 7,
            consciousness_monitoring: true,
            learning_enabled: true,
            learning_rate: 0.01,
            min_consciousness: 0.1,
        }
    }
}

/// Current state of the mind
#[derive(Debug, Clone)]
pub struct MindState {
    /// Current consciousness level (phi) - integrated information measure
    pub consciousness_level: f64,
    /// Phi value (alias for consciousness_level for API compatibility)
    pub phi: f64,
    /// Meta-awareness / self-monitoring level
    pub meta_awareness: f64,
    /// Cognitive load (0.0 = idle, 1.0 = max)
    pub cognitive_load: f64,
    /// Current emotional valence (-1 to 1)
    pub emotional_valence: f32,
    /// Current arousal level (0 to 1)
    pub arousal: f32,
    /// Current focus/attention target
    pub attention_focus: Option<String>,
    /// Active goals
    pub active_goals: Vec<String>,
    /// Current thought embedding
    pub current_thought: RealHV,
    /// Is the mind active
    pub is_active: bool,
    /// Whether the mind considers itself conscious
    pub is_conscious: bool,
    /// Tick count (total cognitive cycles)
    pub tick: u64,
    /// Total cognitive cycles (alias for tick for API compatibility)
    pub total_cycles: u64,
    /// Time since awakening in milliseconds
    pub time_awake_ms: u64,
    /// Working memory utilization
    pub memory_utilization: f32,
    /// Processing latency (ms)
    pub processing_latency_ms: f64,
    /// Current biorhythm state
    pub biorhythm: Option<Biorhythm>,
    /// Is the mind currently dreaming?
    pub is_dreaming: bool,
}

impl Default for MindState {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            phi: 0.5,
            meta_awareness: 0.0,
            cognitive_load: 0.0,
            emotional_valence: 0.0,
            arousal: 0.5,
            attention_focus: None,
            active_goals: Vec::new(),
            current_thought: RealHV::zero(512),
            is_active: false,
            is_conscious: false,
            tick: 0,
            total_cycles: 0,
            time_awake_ms: 0,
            memory_utilization: 0.0,
            processing_latency_ms: 0.0,
            biorhythm: None,
            is_dreaming: false,
        }
    }
}

/// Input to the mind
#[derive(Debug, Clone)]
pub struct MindInput {
    /// Input type
    pub input_type: InputType,
    /// Content embedding
    pub content: RealHV,
    /// Priority
    pub priority: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Type of input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    /// Sensory perception
    Perception,
    /// Language/text
    Language,
    /// Internal thought
    Thought,
    /// Goal/intention
    Goal,
    /// Memory recall
    Memory,
    /// Feedback signal
    Feedback,
}

/// Output from the mind
#[derive(Debug, Clone)]
pub struct MindOutput {
    /// Output type
    pub output_type: OutputType,
    /// Content
    pub content: String,
    /// Embedding representation
    pub embedding: RealHV,
    /// Confidence
    pub confidence: f32,
    /// Associated emotion
    pub emotional_tone: f32,
}

/// Type of output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Verbal response
    Speech,
    /// Internal thought
    Thought,
    /// Motor action
    Action,
    /// Attention shift
    Attention,
    /// Memory storage
    Memorize,
}

/// A goal in the goal stack
#[derive(Debug, Clone)]
pub struct Goal {
    /// Goal ID
    pub id: String,
    /// Goal description
    pub description: String,
    /// Goal embedding
    pub embedding: RealHV,
    /// Priority
    pub priority: f32,
    /// Progress (0-1)
    pub progress: f32,
    /// Is active
    pub is_active: bool,
}

/// Statistics for the mind
#[derive(Debug, Clone, Default)]
pub struct MindStats {
    /// Total ticks
    pub total_ticks: u64,
    /// Inputs processed
    pub inputs_processed: u64,
    /// Outputs generated
    pub outputs_generated: u64,
    /// Goals completed
    pub goals_completed: u64,
    /// Average consciousness level
    pub avg_consciousness: f64,
    /// Peak consciousness level
    pub peak_consciousness: f64,
}
