// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration and data types for the Continuous Mind system.

use crate::chronobiology::Biorhythm;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::{ContinuousHV, LiquidHolocell};

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
    /// Enable social coherence (theory of mind for multi-agent)
    pub enable_social_coherence: bool,
    /// Timezone offset in hours from UTC (e.g., -5.0 for CDT, +9.0 for JST).
    /// Used by `ContinuousMind::tick()` for timezone-aware biorhythm.
    #[serde(default)]
    pub timezone_offset_hours: f64,
    /// Enable learned projection for social signals (512D→16384D).
    /// Required for collective consciousness emergence under partial information.
    #[serde(default)]
    pub social_projection_enabled: bool,
    /// Enable mesh peer name resolution (mirrors `CognitiveLoopConfig`'s
    /// same-named toggle -- a separate field since `MindConfig` and
    /// `CognitiveLoopConfig` are independent config structs).
    #[cfg(feature = "mesh")]
    #[serde(default)]
    pub enable_name_resolution: bool,
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
            enable_social_coherence: false,
            timezone_offset_hours: 0.0,
            social_projection_enabled: false,
            #[cfg(feature = "mesh")]
            enable_name_resolution: false,
        }
    }
}

/// Current state of the mind
#[derive(Debug, Clone)]
pub struct MindState {
    /// Current consciousness level (phi) - integrated information measure
    pub consciousness_level: f64,
    /// Ψ — Consciousness estimate (alias for consciousness_level)
    pub psi: f64,
    /// Meta-awareness / self-monitoring level
    pub meta_awareness: f64,
    /// Cognitive load (0.0 = idle, 1.0 = max)
    pub cognitive_load: f64,
    /// Current emotional valence (-1 to 1)
    pub emotional_valence: f32,
    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached)
    pub thermodynamic_load: f32,
    /// Affective bias: cognitive temperature (0.0 to 2.0)
    pub mood_temperature: f32,
    /// Current arousal level (0 to 1)
    pub arousal: f32,
    /// Last brain mutation suggested by swarm (tau scale)
    pub last_mutation_suggestion: Option<f32>,
    /// Current focus/attention target
    pub attention_focus: Option<String>,
    /// Active goals
    pub active_goals: Vec<String>,
    /// Current thought embedding
    pub current_thought: ContinuousHV,
    /// The liquid holocell governing thought dynamics.
    pub holocell: LiquidHolocell,
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
    /// Mesh network telemetry snapshot (only available with `mesh` feature).
    #[cfg(feature = "mesh")]
    pub mesh_telemetry: Option<crate::swarm::mesh::MeshTelemetry>,
    /// Liquid-Mamba semantic prediction error (injected by facade after translation).
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_pe: f32,
    /// Liquid-Mamba effective distillation learning rate.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_lr: f32,
    /// Liquid-Mamba last cached effective rank of projection bottleneck.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_rank: f32,
    /// Liquid-Mamba total generation/distillation cycles completed.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_generation_count: u32,
    /// Confidence in the current perception (0.0-1.0), consumed by the
    /// neural-bridge epistemic-attenuation step in `tick.rs` to dampen the
    /// Phi estimate when perception is uncertain. Not yet populated by any
    /// real perception pathway -- defaults to 1.0 (fully confident) so
    /// attenuation stays inert until something feeds it real data.
    #[cfg(feature = "neural-bridge")]
    pub perception_confidence: f32,
    /// Uncertainty in the current perception, feeding the same
    /// neural-bridge attenuation step's cognitive-load increment. Defaults
    /// to 0.0 (no uncertainty) for the same reason as `perception_confidence`.
    #[cfg(feature = "neural-bridge")]
    pub perception_uncertainty: f32,
}

impl Default for MindState {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            psi: 0.5,
            meta_awareness: 0.0,
            cognitive_load: 0.0,
            emotional_valence: 0.0,
            thermodynamic_load: 0.0,
            mood_temperature: 1.0,
            arousal: 0.5,
            last_mutation_suggestion: None,
            attention_focus: None,
            active_goals: Vec::new(),
            current_thought: ContinuousHV::zero(512),
            holocell: LiquidHolocell::new(42),
            is_active: false,
            is_conscious: false,
            tick: 0,
            total_cycles: 0,
            time_awake_ms: 0,
            memory_utilization: 0.0,
            processing_latency_ms: 0.0,
            biorhythm: None,
            is_dreaming: false,
            #[cfg(feature = "mesh")]
            mesh_telemetry: None,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_pe: 0.0,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_lr: 0.0,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_rank: 0.0,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_generation_count: 0,
            #[cfg(feature = "neural-bridge")]
            perception_confidence: 1.0,
            #[cfg(feature = "neural-bridge")]
            perception_uncertainty: 0.0,
        }
    }
}

use crate::memory::memory_coordinator::MemorySource;

/// Input to the mind
#[derive(Debug, Clone)]
pub struct MindInput {
    /// Input type
    pub input_type: InputType,
    /// Content embedding
    pub content: ContinuousHV,
    /// Priority
    pub priority: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Source of the input
    pub source: MemorySource,
    /// Whether the information has been verified
    pub is_verified: bool,
}

impl MindInput {
    /// Create a new mind input
    pub fn new(input_type: InputType, content: ContinuousHV) -> Self {
        Self {
            input_type,
            content,
            priority: 0.5,
            metadata: HashMap::new(),
            source: MemorySource::Internal,
            is_verified: false,
        }
    }

    /// Set the source
    pub fn with_source(mut self, source: MemorySource) -> Self {
        self.source = source;
        self
    }

    /// Set verification status
    pub fn with_verification(mut self, verified: bool) -> Self {
        self.is_verified = verified;
        self
    }
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
    pub embedding: ContinuousHV,
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
    pub embedding: ContinuousHV,
    /// Priority
    pub priority: f32,
    /// Progress (0-1)
    pub progress: f32,
    /// Is active
    pub is_active: bool,
}

/// A message exchanged between agents via the social coherence system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialMessage {
    /// Sender agent identifier.
    pub agent_id: String,
    /// Observed behavior embedding of the sender.
    pub behavior: ContinuousHV,
    /// Context embedding at time of observation.
    pub context: ContinuousHV,
    /// Optional interaction outcome (positive = cooperative, negative = adversarial).
    /// When `Some`, the message is treated as an interaction record.
    pub interaction_outcome: Option<f32>,
    /// Sender's neuromodulator bath state for empathic coupling.
    /// Science: Feldman (2012) — oxytocin biobehavioral synchrony.
    #[serde(default)]
    pub bath_state: Option<Vec<f32>>,
    /// High-dimensional consciousness state for swarm integration (Phase 5).
    #[cfg(feature = "swarm")]
    #[serde(default)]
    pub swarm_state: Option<symthaea_swarm::SwarmStateMsg>,
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
    /// Dream ticks where adenosine was cleared (Xie 2013)
    pub dream_adenosine_cleared: u64,
    /// Dream ticks where allostatic recovery occurred (McEwen 1998)
    pub dream_allostatic_recovery: u64,
}
