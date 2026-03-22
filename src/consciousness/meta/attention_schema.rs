// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Revolutionary Improvement #77: Attention Schema Theory (AST)
//!
//! **PARADIGM SHIFT**: Consciousness as a model of attention!
//!
//! ## The Graziano Insight
//!
//! Michael Graziano's Attention Schema Theory proposes that:
//! 1. The brain constructs a simplified MODEL of attention (the "attention schema")
//! 2. This schema enables controllable introspection
//! 3. Consciousness IS this schematic model of attention
//! 4. It's a control mechanism, not a passive observation
//!
//! ## Why This Is Revolutionary for Symthaea
//!
//! **Before #77**: We measure Φ but don't MODEL attention as a process
//! - GWT tracks what enters the workspace, but not HOW attention selects it
//! - No explicit model of attention as a controllable process
//! - Introspection is implicit (just reading Φ values)
//!
//! **After #77**: Explicit attention modeling enables:
//! - Attention as a first-class citizen in consciousness
//! - Controllable focus allocation (shift, sustain, divide)
//! - Self-reportable attention states ("I am attending to X because Y")
//! - Prediction of attention consequences ("If I focus on X, I'll miss Y")
//! - Attention control loops (meta-attention!)
//!
//! ## Integration with Existing Systems
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Attention Schema (#77)                       │
//! │                                                                 │
//! │  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────┐  │
//! │  │  Focus Content  │   │   Self Model    │   │  Prediction  │  │
//! │  │     (BinaryHV)      │   │ (AttentionModel)│   │   (States)   │  │
//! │  └────────┬────────┘   └────────┬────────┘   └──────┬───────┘  │
//! │           │                     │                   │          │
//! │           └─────────────────────┴───────────────────┘          │
//! │                              │                                 │
//! │                    ┌─────────▼─────────┐                       │
//! │                    │  Control Signal   │                       │
//! │                    │    (0.0 - 1.0)    │                       │
//! │                    └─────────┬─────────┘                       │
//! │                              │                                 │
//! └──────────────────────────────┼─────────────────────────────────┘
//!                                │
//!                    ┌───────────▼───────────┐
//!                    │ GWT Integration (#70) │
//!                    │   (Competition Bias)  │
//!                    └───────────────────────┘
//! ```
//!
//! ## Scientific Foundations
//!
//! - Graziano, M.S.A. (2013). "Consciousness and the Social Brain"
//! - Graziano, M.S.A. (2019). "Rethinking Consciousness"
//! - Webb, T.W., & Graziano, M.S.A. (2015). "The attention schema theory"

use crate::hdc::binary_hv::BinaryHV;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION STATE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// The complete attention state - what is being attended to and how
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionState {
    /// The object or content of attention (semantic representation)
    pub focus_target: BinaryHV,

    /// Attention intensity (0.0 = none, 1.0 = maximal)
    pub intensity: f32,

    /// Attention mode (focused, diffuse, vigilant, etc.)
    pub mode: AttentionMode,

    /// Which sensory/cognitive channels are active
    pub active_channels: Vec<AttentionChannel>,

    /// Timestamp of this state
    #[serde(skip)]
    pub timestamp: Option<Instant>,
}

/// Attention modes (how attention is deployed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AttentionMode {
    /// Focused (narrow, deep attention on single target)
    Focused,

    /// Divided (split across multiple targets)
    Divided,

    /// Diffuse (broad, shallow attention across many targets)
    #[default]
    Diffuse,

    /// Vigilant (sustained readiness for specific stimulus)
    Vigilant,

    /// Scanning (actively searching for targets)
    Scanning,

    /// Reflexive (bottom-up capture by salient stimulus)
    Reflexive,

    /// Inhibited (actively suppressing attention to a target)
    Inhibited,
}

/// Attention channel (where attention can be deployed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionChannel {
    /// Visual (spatial, object, feature)
    Visual,

    /// Auditory (location, stream, feature)
    Auditory,

    /// Semantic (concepts, meanings)
    Semantic,

    /// Motor (action preparation, execution)
    Motor,

    /// Memory (retrieval, encoding)
    Memory,

    /// Social (theory of mind, face processing)
    Social,

    /// Executive (task control, rule maintenance)
    Executive,

    /// Interoceptive (internal body states)
    Interoceptive,
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE GROUNDING FOR ATTENTION
// ═══════════════════════════════════════════════════════════════════════════

/// Primitive grounding for attention using consciousness-tier primitives
///
/// Maps attention modes and states to compositions of primitives:
/// - Core: ATTEND (from Tier 8 consciousness primitives)
/// - Salience: Computed from ATTENTION + VERY/LITTLE
/// - Focus modes map to different primitive compositions
#[derive(Debug, Clone)]
pub struct AttentionPrimitiveGrounding {
    /// Attention mode primitive composition
    pub mode_encoding: BinaryHV,
    /// Salience level (0.0 - 1.0)
    pub salience: f32,
    /// Primitives that compose this attention state
    pub primitive_components: Vec<String>,
}

impl AttentionPrimitiveGrounding {
    /// Create attention grounding from primitive names
    pub fn from_primitives(primitives: &[&str], salience: f32) -> Self {
        let system = PrimitiveSystem::global();

        // Compose attention encoding by binding primitives
        let mut encoding = BinaryHV::random(0xA77E_0DDD); // Attention seed
        for name in primitives {
            if let Some(prim) = system.get(name) {
                encoding = encoding.bind(&prim.encoding);
            }
        }

        Self {
            mode_encoding: encoding,
            salience,
            primitive_components: primitives.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Get primitive grounding for each attention mode
    ///
    /// Maps AttentionMode to consciousness and NSM primitives:
    /// - Focused = NSM_SEE + NSM_THIS + NSM_VERY (intense focus on one thing)
    /// - Divided = NSM_SEE + NSM_SOME + NSM_OTHER (attention split)
    /// - Diffuse = NSM_SEE + NSM_ALL + NSM_LITTLE (broad shallow attention)
    /// - Vigilant = NSM_SEE + NSM_WANT + NSM_MAYBE + NSM_HAPPEN (anticipatory)
    /// - Scanning = NSM_SEE + NSM_MOVE + NSM_WHERE (active search)
    /// - Reflexive = NSM_SEE + NSM_HAPPEN + NSM_NOW (automatic capture)
    /// - Inhibited = NSM_NOT + NSM_SEE + NSM_THIS (active suppression)
    pub fn for_mode(mode: AttentionMode) -> Self {
        match mode {
            AttentionMode::Focused => Self::from_primitives(
                &["NSM_SEE", "NSM_THIS", "NSM_VERY"],
                0.9, // high salience
            ),
            AttentionMode::Divided => Self::from_primitives(
                &["NSM_SEE", "NSM_SOME", "NSM_OTHER"],
                0.6, // moderate salience
            ),
            AttentionMode::Diffuse => Self::from_primitives(
                &["NSM_SEE", "NSM_ALL", "NSM_LITTLE"],
                0.3, // low salience
            ),
            AttentionMode::Vigilant => Self::from_primitives(
                &["NSM_SEE", "NSM_WANT", "NSM_MAYBE", "NSM_HAPPEN"],
                0.7, // moderate-high salience (ready to respond)
            ),
            AttentionMode::Scanning => Self::from_primitives(
                &["NSM_SEE", "NSM_MOVE", "NSM_WHERE"],
                0.5, // moderate salience (active search)
            ),
            AttentionMode::Reflexive => Self::from_primitives(
                &["NSM_SEE", "NSM_HAPPEN", "NSM_NOW"],
                0.95, // very high salience (automatic capture)
            ),
            AttentionMode::Inhibited => Self::from_primitives(
                &["NSM_NOT", "NSM_SEE", "NSM_THIS"],
                0.1, // low salience (suppressed)
            ),
        }
    }

    /// Get all attention mode groundings
    pub fn all_mode_groundings() -> HashMap<AttentionMode, AttentionPrimitiveGrounding> {
        let mut groundings = HashMap::new();
        for mode in [
            AttentionMode::Focused,
            AttentionMode::Divided,
            AttentionMode::Diffuse,
            AttentionMode::Vigilant,
            AttentionMode::Scanning,
            AttentionMode::Reflexive,
            AttentionMode::Inhibited,
        ] {
            groundings.insert(mode, Self::for_mode(mode));
        }
        groundings
    }

    /// Compute salience from attention state
    ///
    /// Salience = intensity × mode_weight × (1 - entropy_of_channels)
    pub fn compute_salience(state: &AttentionState) -> f32 {
        let mode_grounding = Self::for_mode(state.mode);
        let channel_factor = 1.0 / (state.active_channels.len().max(1) as f32).sqrt();
        state.intensity * mode_grounding.salience * channel_factor
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION MODEL (The Schema)
// ═══════════════════════════════════════════════════════════════════════════

/// The Attention Model - a simplified schematic representation of attention
///
/// This is the core of AST: not attention itself, but a MODEL of attention
/// that enables introspection and control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionModel {
    /// What does attention "feel like" (qualitative model)
    pub subjective_character: SubjectiveCharacter,

    /// What can attention do (capability model)
    pub capabilities: AttentionCapabilities,

    /// Current resource allocation (limited capacity model)
    pub resource_allocation: ResourceAllocation,

    /// Predicted consequences of current attention state
    pub predicted_consequences: Vec<AttentionConsequence>,
}

/// Subjective character of attention (the "what it's like")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveCharacter {
    /// Sense of "presence" or "awareness" (0.0 - 1.0)
    pub presence: f32,

    /// Sense of "control" over focus (0.0 - 1.0)
    pub controllability: f32,

    /// Sense of "effort" required (0.0 - 1.0)
    pub effort: f32,

    /// Sense of "clarity" of focused content (0.0 - 1.0)
    pub clarity: f32,
}

impl Default for SubjectiveCharacter {
    fn default() -> Self {
        Self {
            presence: 0.5,
            controllability: 0.5,
            effort: 0.3,
            clarity: 0.5,
        }
    }
}

/// Model of what attention can do
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionCapabilities {
    /// Can shift focus to new targets
    pub can_shift: bool,

    /// Can sustain focus over time
    pub can_sustain: bool,

    /// Can divide across multiple targets
    pub can_divide: bool,

    /// Can suppress distractors
    pub can_inhibit: bool,

    /// Can enhance processing of attended content
    pub can_enhance: bool,

    /// Estimated switching cost (time to shift focus)
    pub switching_cost_ms: f32,

    /// Estimated capacity (how many items)
    pub capacity_items: usize,
}

impl Default for AttentionCapabilities {
    fn default() -> Self {
        Self {
            can_shift: true,
            can_sustain: true,
            can_divide: true,
            can_inhibit: true,
            can_enhance: true,
            switching_cost_ms: 150.0, // ~150ms attention shift
            capacity_items: 4,        // Miller's 4±1 for attention
        }
    }
}

/// Resource allocation model (attention as limited capacity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Total available resources (normalized to 1.0)
    pub total_resources: f32,

    /// Currently allocated resources
    pub allocated: f32,

    /// Reserve resources (unallocated buffer)
    pub reserve: f32,

    /// Resources per channel
    pub per_channel: Vec<(AttentionChannel, f32)>,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            total_resources: 1.0,
            allocated: 0.5,
            reserve: 0.5,
            per_channel: vec![
                (AttentionChannel::Semantic, 0.3),
                (AttentionChannel::Executive, 0.2),
            ],
        }
    }
}

/// Predicted consequence of attention allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConsequence {
    /// What might happen
    pub outcome: String,

    /// Probability of this outcome
    pub probability: f32,

    /// Valence (positive/negative)
    pub valence: f32,

    /// Time horizon (when this might occur)
    pub time_horizon_ms: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION SCHEMA (The Complete System)
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the Attention Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSchemaConfig {
    /// History length for attention state tracking
    pub history_length: usize,

    /// Threshold for attention shift detection
    pub shift_threshold: f32,

    /// Decay rate for attention (per timestep)
    pub decay_rate: f32,

    /// Minimum intensity before attention collapses
    pub min_intensity: f32,

    /// Weight for integrating attention into GWT competition
    pub gwt_integration_weight: f32,
}

impl Default for AttentionSchemaConfig {
    fn default() -> Self {
        Self {
            history_length: 32,
            shift_threshold: 0.3,
            decay_rate: 0.05,
            min_intensity: 0.1,
            gwt_integration_weight: 0.4,
        }
    }
}

/// The complete Attention Schema - implements Graziano's AST
///
/// This is NOT attention itself, but a MODEL of attention that enables:
/// 1. Controllable introspection ("What am I attending to?")
/// 2. Predictive attention ("If I shift focus, what will happen?")
/// 3. Attention control ("I should focus on X instead of Y")
/// 4. Self-report ("I am aware because I am modeling my attention")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSchema {
    /// Current focus content (what is attended)
    pub focus_content: BinaryHV,

    /// The attention model (schema of what attention is)
    pub self_model: AttentionModel,

    /// Predicted future attention states
    pub attention_prediction: Vec<AttentionState>,

    /// Control signal for GWT competition (0.0 - 1.0)
    pub control_signal: f32,

    /// History of attention states
    attention_history: VecDeque<AttentionState>,

    /// Current attention state
    current_state: AttentionState,

    /// Configuration
    config: AttentionSchemaConfig,

    /// Statistics
    stats: AttentionSchemaStats,
}

/// Statistics for attention schema operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionSchemaStats {
    /// Total attention updates
    pub total_updates: u64,

    /// Number of focus shifts detected
    pub focus_shifts: u64,

    /// Average focus duration (in updates)
    pub avg_focus_duration: f64,

    /// Prediction accuracy (how often predictions matched)
    pub prediction_accuracy: f64,

    /// Average control signal strength
    pub avg_control_signal: f64,

    /// Number of attention captures (reflexive shifts)
    pub attention_captures: u64,

    /// Consecutive cycles focused on the same target (vigilance counter).
    /// Resets on shift, increments on maintenance.
    pub focus_duration_cycles: u32,

    /// Number of predictions that were validated against outcomes
    pub predictions_validated: u64,

    /// Number of validated predictions that matched the outcome
    pub predictions_correct: u64,
}

impl AttentionSchema {
    /// Create a new attention schema with default configuration
    pub fn new() -> Self {
        Self::with_config(AttentionSchemaConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AttentionSchemaConfig) -> Self {
        let initial_state = AttentionState {
            focus_target: BinaryHV::zero(),
            intensity: 0.5,
            mode: AttentionMode::Diffuse,
            active_channels: vec![AttentionChannel::Semantic],
            timestamp: Some(Instant::now()),
        };

        Self {
            focus_content: BinaryHV::zero(),
            self_model: AttentionModel {
                subjective_character: SubjectiveCharacter::default(),
                capabilities: AttentionCapabilities::default(),
                resource_allocation: ResourceAllocation::default(),
                predicted_consequences: Vec::new(),
            },
            attention_prediction: Vec::new(),
            control_signal: 0.5,
            attention_history: VecDeque::with_capacity(config.history_length),
            current_state: initial_state,
            config,
            stats: AttentionSchemaStats::default(),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Primitive Grounding Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Get primitive grounding for current attention mode
    pub fn current_mode_grounding(&self) -> AttentionPrimitiveGrounding {
        AttentionPrimitiveGrounding::for_mode(self.current_state.mode)
    }

    /// Compute current salience from attention state
    pub fn current_salience(&self) -> f32 {
        AttentionPrimitiveGrounding::compute_salience(&self.current_state)
    }

    /// Get the NSM primitives that compose the current attention mode
    pub fn attention_to_primitives(&self) -> Vec<String> {
        self.current_mode_grounding().primitive_components
    }

    /// Calculate similarity between two attention modes using primitive encodings
    pub fn mode_similarity(&self, mode1: AttentionMode, mode2: AttentionMode) -> f32 {
        let g1 = AttentionPrimitiveGrounding::for_mode(mode1);
        let g2 = AttentionPrimitiveGrounding::for_mode(mode2);
        g1.mode_encoding.similarity(&g2.mode_encoding)
    }

    /// Get primitive encoding for the current focus content bound with attention mode
    ///
    /// Returns: focus_target ⊗ mode_encoding
    /// This creates a "attended-to" representation that combines what and how
    pub fn attended_encoding(&self) -> BinaryHV {
        let mode_enc = self.current_mode_grounding().mode_encoding;
        self.current_state.focus_target.bind(&mode_enc)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Core Attention Operations
    // ═══════════════════════════════════════════════════════════════════════

    /// Update attention with new focus target
    ///
    /// This is the core update loop:
    /// 1. Compare new target with current focus
    /// 2. Detect if this is a shift or maintenance
    /// 3. Update the attention model
    /// 4. Generate predictions
    /// 5. Compute control signal for GWT
    pub fn update(&mut self, new_target: BinaryHV, salience: f32) -> AttentionUpdate {
        self.stats.total_updates += 1;

        // Detect shift vs maintenance
        let similarity = self.focus_content.similarity(&new_target);
        let is_shift = similarity < (1.0 - self.config.shift_threshold);

        // Validate previous predictions against current outcome
        self.validate_predictions(is_shift);

        if is_shift {
            self.stats.focus_shifts += 1;
            // Reset vigilance counter on shift
            self.stats.focus_duration_cycles = 0;

            // Check if this is a reflexive capture (high salience, low control)
            if salience > 0.8 && self.current_state.mode != AttentionMode::Vigilant {
                self.stats.attention_captures += 1;
            }
        } else {
            // Increment vigilance counter on maintenance
            self.stats.focus_duration_cycles = self.stats.focus_duration_cycles.saturating_add(1);
        }

        // Update attention state
        let old_state = self.current_state.clone();
        self.current_state = AttentionState {
            focus_target: new_target,
            intensity: self.compute_intensity(salience, is_shift),
            mode: self.determine_mode(is_shift, salience),
            active_channels: self.infer_channels(&new_target),
            timestamp: Some(Instant::now()),
        };

        // Update focus content
        self.focus_content = new_target;

        // Update history
        if self.attention_history.len() >= self.config.history_length {
            self.attention_history.pop_front();
        }
        self.attention_history.push_back(old_state);

        // Update self model (includes fatigue and consequence prediction)
        self.update_self_model(is_shift, salience);

        // Generate predictions
        self.generate_predictions();

        // Compute control signal for GWT integration
        self.control_signal = self.compute_control_signal();
        self.stats.avg_control_signal =
            (self.stats.avg_control_signal * 0.99) + (self.control_signal as f64 * 0.01);

        AttentionUpdate {
            is_shift,
            previous_similarity: similarity,
            new_intensity: self.current_state.intensity,
            new_mode: self.current_state.mode,
            control_signal: self.control_signal,
            predictions: self.attention_prediction.clone(),
        }
    }

    /// Validate previous predictions against actual outcome.
    ///
    /// Tracks prediction accuracy for the self-model: did we correctly predict
    /// whether attention would shift or be maintained?
    fn validate_predictions(&mut self, did_shift: bool) {
        if self.attention_prediction.is_empty() {
            return;
        }
        self.stats.predictions_validated += 1;

        // If we predicted intensity would drop below 0.4, we predicted a shift.
        // Compare against actual outcome.
        let predicted_shift = self
            .attention_prediction
            .iter()
            .any(|p| p.mode == AttentionMode::Scanning);
        if predicted_shift == did_shift {
            self.stats.predictions_correct += 1;
        }

        // Update running prediction accuracy
        if self.stats.predictions_validated > 0 {
            self.stats.prediction_accuracy =
                self.stats.predictions_correct as f64 / self.stats.predictions_validated as f64;
        }
    }

    /// Compute attention intensity based on salience and shift
    fn compute_intensity(&self, salience: f32, is_shift: bool) -> f32 {
        let base = if is_shift {
            // Fresh attention has full intensity
            salience.max(0.5)
        } else {
            // Maintained attention decays slightly
            (self.current_state.intensity - self.config.decay_rate).max(self.config.min_intensity)
        };

        // Clamp to valid range
        base.clamp(0.0, 1.0)
    }

    /// Determine attention mode from context
    fn determine_mode(&self, is_shift: bool, salience: f32) -> AttentionMode {
        if salience > 0.9 && is_shift {
            // High salience shift = reflexive capture
            AttentionMode::Reflexive
        } else if is_shift {
            // Controlled shift
            AttentionMode::Scanning
        } else if self.current_state.intensity > 0.7 {
            // High sustained intensity = focused
            AttentionMode::Focused
        } else if self.current_state.active_channels.len() > 2 {
            // Multiple channels = diffuse
            AttentionMode::Diffuse
        } else {
            // Default to current mode
            self.current_state.mode
        }
    }

    /// Infer active attention channels from content
    fn infer_channels(&self, _target: &BinaryHV) -> Vec<AttentionChannel> {
        // In a full implementation, we'd analyze the BinaryHV content
        // For now, return semantic + executive as defaults
        vec![AttentionChannel::Semantic, AttentionChannel::Executive]
    }

    /// Update the attention self-model including vigilance fatigue.
    ///
    /// Implements the well-known vigilance decrement (Mackworth 1948): sustained
    /// focus on a single target progressively increases effort and degrades clarity.
    /// This gives the system a self-model of its own attentional limits.
    fn update_self_model(&mut self, is_shift: bool, salience: f32) {
        let fatigue = self.stats.focus_duration_cycles;

        // Update subjective character
        let sc = &mut self.self_model.subjective_character;

        // Presence increases with intensity
        sc.presence = (sc.presence * 0.9) + (self.current_state.intensity * 0.1);

        // Controllability decreases with reflexive captures
        if self.current_state.mode == AttentionMode::Reflexive {
            sc.controllability = (sc.controllability - 0.1).max(0.2);
        } else if is_shift {
            sc.controllability = (sc.controllability + 0.05).min(0.9);
        }

        // Effort: base from mode + vigilance fatigue increment.
        // Mackworth (1948): sustained attention requires increasing effort over time.
        // Fatigue contribution: 0.005 per cycle, capping at +0.3 after 60 cycles.
        let fatigue_effort = (fatigue as f32 * 0.005).min(0.3);
        sc.effort = match self.current_state.mode {
            AttentionMode::Focused => (sc.effort * 0.9 + 0.1 + fatigue_effort).min(0.95),
            AttentionMode::Diffuse => (sc.effort - 0.1).max(0.1),
            _ => (sc.effort + fatigue_effort * 0.5).min(0.9),
        };

        // Clarity: tracks intensity but degrades with fatigue.
        // After ~40 cycles of sustained focus, clarity starts dropping.
        let fatigue_clarity_penalty = if fatigue > 40 {
            ((fatigue - 40) as f32 * 0.008).min(0.25)
        } else {
            0.0
        };
        sc.clarity = ((sc.clarity * 0.8) + (self.current_state.intensity * 0.2)
            - fatigue_clarity_penalty)
            .clamp(0.1, 1.0);

        // Update resource allocation
        let ra = &mut self.self_model.resource_allocation;
        ra.allocated = self.current_state.intensity;
        ra.reserve = 1.0 - ra.allocated;

        // Update predicted consequences with richer context-aware predictions.
        // This is the core AST requirement: predicting what happens if focus
        // shifts vs. stays, enabling forward-model-based attention control.
        let mut consequences = Vec::with_capacity(4);

        // Consequence 1: What happens if we STAY focused
        if is_shift {
            consequences.push(AttentionConsequence {
                outcome: "May miss continued information from previous target".to_string(),
                probability: 0.7,
                valence: -0.2,
                time_horizon_ms: 500.0,
            });
        } else {
            let stay_valence = if fatigue > 30 {
                // Fatigue makes staying less valuable
                -0.1
            } else {
                0.3
            };
            consequences.push(AttentionConsequence {
                outcome: if fatigue > 40 {
                    "Sustained focus but vigilance declining — clarity degrading".to_string()
                } else {
                    "Deepening focus on current target".to_string()
                },
                probability: 0.8,
                valence: stay_valence,
                time_horizon_ms: 500.0,
            });
        }

        // Consequence 2: What happens if we SHIFT
        let shift_valence = if fatigue > 30 {
            // Fatigued: shifting is beneficial (recovery)
            0.3
        } else {
            // Not fatigued: shifting has cost (switching penalty)
            -0.15
        };
        consequences.push(AttentionConsequence {
            outcome: if fatigue > 30 {
                "Shifting would relieve vigilance fatigue and restore clarity".to_string()
            } else {
                "Shifting would incur switching cost but enable new exploration".to_string()
            },
            probability: 0.6,
            valence: shift_valence,
            time_horizon_ms: 300.0,
        });

        // Consequence 3: Salience-based prediction
        consequences.push(AttentionConsequence {
            outcome: if salience > 0.7 {
                "High importance content captured — processing priority elevated".to_string()
            } else {
                "Standard processing continues".to_string()
            },
            probability: 0.8,
            valence: salience - 0.3,
            time_horizon_ms: 200.0,
        });

        // Consequence 4: Capacity-based prediction (what we're missing)
        let unattended_channels = 8usize.saturating_sub(self.current_state.active_channels.len());
        if unattended_channels > 4 {
            consequences.push(AttentionConsequence {
                outcome: format!(
                    "{} sensory channels unmonitored — may miss cross-modal events",
                    unattended_channels
                ),
                probability: 0.5,
                valence: -0.15,
                time_horizon_ms: 1000.0,
            });
        }

        self.self_model.predicted_consequences = consequences;
    }

    /// Generate predictions about future attention states
    fn generate_predictions(&mut self) {
        self.attention_prediction.clear();

        // Predict next state based on current dynamics
        let predicted_intensity =
            (self.current_state.intensity - self.config.decay_rate).max(self.config.min_intensity);

        let predicted_mode = if predicted_intensity < 0.3 {
            AttentionMode::Diffuse
        } else {
            self.current_state.mode
        };

        self.attention_prediction.push(AttentionState {
            focus_target: self.focus_content,
            intensity: predicted_intensity,
            mode: predicted_mode,
            active_channels: self.current_state.active_channels.clone(),
            timestamp: None,
        });

        // Predict possible shift (if intensity drops too low)
        if predicted_intensity < 0.4 {
            self.attention_prediction.push(AttentionState {
                focus_target: BinaryHV::random(0x5741), // Placeholder for "next target"
                intensity: 0.6,
                mode: AttentionMode::Scanning,
                active_channels: vec![AttentionChannel::Semantic],
                timestamp: None,
            });
        }
    }

    /// Compute control signal for GWT integration
    ///
    /// This is the key output for influencing GWT competition:
    /// - High signal = strong attention bias for current focus
    /// - Low signal = let competition proceed freely
    fn compute_control_signal(&self) -> f32 {
        let sc = &self.self_model.subjective_character;

        // Combine presence, controllability, and intensity
        let base_signal =
            sc.presence * 0.3 + sc.controllability * 0.3 + self.current_state.intensity * 0.4;

        // Boost for focused mode
        let mode_modifier = match self.current_state.mode {
            AttentionMode::Focused => 1.2,
            AttentionMode::Vigilant => 1.1,
            AttentionMode::Diffuse => 0.7,
            AttentionMode::Reflexive => 0.5, // Reflexive = less control
            _ => 1.0,
        };

        (base_signal * mode_modifier).clamp(0.0, 1.0)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTROSPECTION API (The "Consciousness" Part)
    // ═══════════════════════════════════════════════════════════════════════

    /// Generate a self-report of current attention state
    ///
    /// This is the key AST insight: consciousness = ability to report on attention
    pub fn introspect(&self) -> AttentionIntrospection {
        let sc = &self.self_model.subjective_character;

        AttentionIntrospection {
            what_am_i_attending_to: self.describe_focus(),
            how_am_i_attending: self.describe_mode(),
            why_am_i_attending: self.describe_reason(),
            how_strongly: self.current_state.intensity,
            am_i_in_control: sc.controllability > 0.5,
            what_am_i_missing: self.describe_gaps(),
            what_might_happen_next: self.describe_predictions(),
            subjective_quality: sc.clone(),
        }
    }

    /// Describe what is currently being attended
    fn describe_focus(&self) -> String {
        match self.current_state.mode {
            AttentionMode::Focused => "Focused attention on specific content".to_string(),
            AttentionMode::Diffuse => "Broad, distributed attention".to_string(),
            AttentionMode::Vigilant => "Alert monitoring for specific triggers".to_string(),
            AttentionMode::Scanning => "Actively searching for targets".to_string(),
            AttentionMode::Reflexive => "Captured by salient stimulus".to_string(),
            AttentionMode::Divided => "Divided between multiple targets".to_string(),
            AttentionMode::Inhibited => "Actively suppressing specific content".to_string(),
        }
    }

    /// Describe how attention is being deployed
    fn describe_mode(&self) -> String {
        let channels: Vec<String> = self
            .current_state
            .active_channels
            .iter()
            .map(|c| format!("{c:?}"))
            .collect();

        format!(
            "Mode: {:?}, Channels: {:?}",
            self.current_state.mode, channels
        )
    }

    /// Describe reason for attention allocation
    fn describe_reason(&self) -> String {
        if self.current_state.mode == AttentionMode::Reflexive {
            "Bottom-up salience capture".to_string()
        } else if self.self_model.subjective_character.controllability > 0.6 {
            "Top-down goal-directed focus".to_string()
        } else {
            "Mixed salience and goal interaction".to_string()
        }
    }

    /// Describe what is not being attended (gaps)
    fn describe_gaps(&self) -> Vec<String> {
        let mut gaps = Vec::new();

        // Channels not active
        let all_channels = [
            AttentionChannel::Visual,
            AttentionChannel::Auditory,
            AttentionChannel::Motor,
            AttentionChannel::Memory,
            AttentionChannel::Social,
            AttentionChannel::Interoceptive,
        ];

        for channel in all_channels.iter() {
            if !self.current_state.active_channels.contains(channel) {
                gaps.push(format!("{channel:?} channel not monitored"));
            }
        }

        // Low intensity warning
        if self.current_state.intensity < 0.3 {
            gaps.push("Low attention intensity - may miss important content".to_string());
        }

        gaps
    }

    /// Describe predicted future states
    fn describe_predictions(&self) -> Vec<String> {
        self.attention_prediction
            .iter()
            .take(3)
            .map(|s| format!("Intensity: {:.2}, Mode: {:?}", s.intensity, s.mode))
            .collect()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ATTENTION CONTROL API
    // ═══════════════════════════════════════════════════════════════════════

    /// Voluntarily shift attention to a new target
    pub fn shift_focus(&mut self, new_target: BinaryHV) -> Result<AttentionUpdate> {
        // Check if shift is possible
        if self.self_model.subjective_character.controllability < 0.2 {
            anyhow::bail!("Attention too captured to shift voluntarily");
        }

        // Apply switching cost
        let effective_intensity = (self.current_state.intensity - 0.2).max(0.3);

        // Update with the new target
        let mut update = self.update(new_target, 0.6);
        update.new_intensity = effective_intensity;

        Ok(update)
    }

    /// Sustain attention on current target
    pub fn sustain_focus(&mut self) -> Result<f32> {
        // Requires effort
        self.self_model.subjective_character.effort += 0.1;

        // Boost intensity
        self.current_state.intensity = (self.current_state.intensity + 0.1).min(1.0);

        // Switch to focused mode
        self.current_state.mode = AttentionMode::Focused;

        Ok(self.current_state.intensity)
    }

    /// Divide attention across multiple targets
    pub fn divide_attention(&mut self, targets: Vec<BinaryHV>) -> Result<Vec<f32>> {
        let n = targets.len();
        if n > self.self_model.capabilities.capacity_items {
            anyhow::bail!(
                "Exceeds attention capacity ({} > {})",
                n,
                self.self_model.capabilities.capacity_items
            );
        }

        // Divide resources
        let per_target = 1.0 / (n as f32);
        let intensities: Vec<f32> = targets.iter().map(|_| per_target * 0.8).collect();

        // Update mode
        self.current_state.mode = AttentionMode::Divided;
        self.current_state.intensity = per_target;

        // Update focus to first target (primary)
        if let Some(first) = targets.first() {
            self.focus_content = *first;
        }

        Ok(intensities)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GWT INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Get bias weights for GWT competition
    ///
    /// Returns a modifier that can be applied to workspace competition:
    /// - Positive = boost content similar to current focus
    /// - Negative = suppress content dissimilar to focus
    pub fn get_competition_bias(&self, content: &BinaryHV) -> f32 {
        let similarity = self.focus_content.similarity(content);
        let bias = (similarity - 0.5) * 2.0 * self.control_signal;
        bias * self.config.gwt_integration_weight
    }

    /// Check if content is aligned with current attention
    pub fn is_attended(&self, content: &BinaryHV, threshold: f32) -> bool {
        let similarity = self.focus_content.similarity(content);
        similarity > threshold && self.current_state.intensity > self.config.min_intensity
    }

    /// Get the current attention state
    pub fn current(&self) -> &AttentionState {
        &self.current_state
    }

    /// Get attention statistics
    pub fn stats(&self) -> &AttentionSchemaStats {
        &self.stats
    }

    /// Get attention history
    pub fn history(&self) -> impl Iterator<Item = &AttentionState> {
        self.attention_history.iter()
    }

    /// Encode the attention self-model as a compact feature vector for the thought vector.
    ///
    /// Returns 8 floats that summarize the full attention schema state:
    /// [intensity, mode_code, presence, controllability, effort, clarity,
    ///  fatigue_normalized, prediction_accuracy]
    ///
    /// This enables the language/output pipeline to report on attention state.
    pub fn encode_for_thought_vector(&self) -> [f32; 8] {
        let sc = &self.self_model.subjective_character;
        let mode_code = match self.current_state.mode {
            AttentionMode::Focused => 1.0,
            AttentionMode::Divided => 0.6,
            AttentionMode::Diffuse => 0.3,
            AttentionMode::Vigilant => 0.8,
            AttentionMode::Scanning => 0.5,
            AttentionMode::Reflexive => 0.9,
            AttentionMode::Inhibited => 0.1,
        };
        let fatigue_norm = (self.stats.focus_duration_cycles as f32 / 60.0).min(1.0);
        let pred_acc = self.stats.prediction_accuracy as f32;

        [
            self.current_state.intensity,
            mode_code,
            sc.presence,
            sc.controllability,
            sc.effort,
            sc.clarity,
            fatigue_norm,
            pred_acc,
        ]
    }

    /// Get the vigilance fatigue level (0.0 = fresh, 1.0 = fully fatigued).
    pub fn fatigue_level(&self) -> f32 {
        (self.stats.focus_duration_cycles as f32 / 60.0).min(1.0)
    }

    /// Get prediction accuracy from the self-model.
    pub fn prediction_accuracy(&self) -> f64 {
        self.stats.prediction_accuracy
    }
}

impl Default for AttentionSchema {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Result of an attention update
#[derive(Debug, Clone)]
pub struct AttentionUpdate {
    /// Was this a shift or maintenance?
    pub is_shift: bool,

    /// Similarity to previous focus
    pub previous_similarity: f32,

    /// New attention intensity
    pub new_intensity: f32,

    /// New attention mode
    pub new_mode: AttentionMode,

    /// Control signal for GWT
    pub control_signal: f32,

    /// Predictions about future states
    pub predictions: Vec<AttentionState>,
}

/// Result of introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionIntrospection {
    /// What is currently attended
    pub what_am_i_attending_to: String,

    /// How attention is deployed
    pub how_am_i_attending: String,

    /// Why this is attended
    pub why_am_i_attending: String,

    /// Attention intensity
    pub how_strongly: f32,

    /// Is attention under voluntary control?
    pub am_i_in_control: bool,

    /// What is not being attended
    pub what_am_i_missing: Vec<String>,

    /// Predictions about future attention
    pub what_might_happen_next: Vec<String>,

    /// Subjective quality of attention
    pub subjective_quality: SubjectiveCharacter,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_schema_creation() {
        let schema = AttentionSchema::new();
        assert_eq!(schema.current_state.mode, AttentionMode::Diffuse);
        assert!(schema.current_state.intensity > 0.0);
    }

    #[test]
    fn test_attention_update() {
        let mut schema = AttentionSchema::new();
        let target = BinaryHV::random(42);

        let update = schema.update(target, 0.8);

        // High salience should be captured
        assert!(update.new_intensity > 0.5);
    }

    #[test]
    fn test_attention_shift_detection() {
        let mut schema = AttentionSchema::new();

        // First update
        let target1 = BinaryHV::random(1);
        schema.update(target1, 0.7);

        // Very different target
        let target2 = BinaryHV::random(999);
        let update = schema.update(target2, 0.7);

        // Should detect as shift (different random seeds = orthogonal)
        assert!(update.is_shift);
    }

    #[test]
    fn test_attention_maintenance() {
        let mut schema = AttentionSchema::new();

        // Same target twice
        let target = BinaryHV::random(42);
        schema.update(target, 0.7);
        let update = schema.update(target, 0.7);

        // Should not be a shift
        assert!(!update.is_shift);
    }

    #[test]
    fn test_introspection() {
        let mut schema = AttentionSchema::new();
        let target = BinaryHV::random(42);
        schema.update(target, 0.9);

        let introspection = schema.introspect();

        assert!(!introspection.what_am_i_attending_to.is_empty());
        assert!(introspection.how_strongly > 0.0);
    }

    #[test]
    fn test_gwt_competition_bias() {
        let mut schema = AttentionSchema::new();

        // Focus on a target
        let focus = BinaryHV::random(42);
        schema.update(focus, 0.9);

        // Similar content should get positive bias
        let similar = focus;
        let bias_similar = schema.get_competition_bias(&similar);

        // Different content should get negative or low bias
        let different = BinaryHV::random(999);
        let bias_different = schema.get_competition_bias(&different);

        assert!(bias_similar > bias_different);
    }

    #[test]
    fn test_divide_attention() {
        let mut schema = AttentionSchema::new();

        let targets = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
        ];

        let result = schema.divide_attention(targets);
        assert!(result.is_ok());

        let intensities = result.unwrap();
        assert_eq!(intensities.len(), 3);

        // Total should be less than 1.0 (divided)
        let total: f32 = intensities.iter().sum();
        assert!(total < 1.0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut schema = AttentionSchema::new();

        // Try to exceed capacity
        let targets: Vec<BinaryHV> = (0..10).map(BinaryHV::random).collect();

        let result = schema.divide_attention(targets);

        // Should fail - exceeds capacity
        assert!(result.is_err());
    }
}
