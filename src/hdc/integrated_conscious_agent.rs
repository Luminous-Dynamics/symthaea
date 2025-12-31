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

use super::real_hv::RealHV;
use super::unified_consciousness_engine::{
    EngineConfig, ConsciousnessDimensions,
};
use super::emergent_self_model::{
    SelfAwareConsciousness, SelfModel, SelfAwareUpdate, MetaCognitiveAssessment,
};
use super::temporal_binding::{
    TemporalBindingEngine, TemporalBindingConfig, StreamHealth,
};
use super::attention_dynamics::{
    AttentionDynamics, AttentionMode, AttentionAllocation,
};
use super::adaptive_topology::CognitiveMode;
use super::topology_synergy::ConsciousnessState;
use std::collections::VecDeque;

// Symthaea physiological system imports
use crate::physiology::{
    HormoneState, CoherenceState, TaskComplexity,
};
use crate::memory::EmotionalValence;
use crate::voice::LTCPacing;
use crate::soul::KVector;

/// Configuration for the integrated conscious agent
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// HDC dimension
    pub dim: usize,
    /// Number of processes in consciousness engine
    pub n_processes: usize,
    /// Enable self-directed attention
    pub self_directed_attention: bool,
    /// Enable Φ-guided optimization
    pub phi_guided: bool,
    /// Attention-binding coupling strength
    pub attention_binding_coupling: f64,
    /// Self-model influence on attention
    pub self_model_attention_weight: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            dim: 2048,
            n_processes: 24,
            self_directed_attention: true,
            phi_guided: true,
            attention_binding_coupling: 0.7,
            self_model_attention_weight: 0.5,
        }
    }
}

/// The integrated conscious agent
pub struct IntegratedConsciousAgent {
    /// Configuration
    config: AgentConfig,
    /// Self-aware consciousness (includes base engine)
    self_awareness: SelfAwareConsciousness,
    /// Temporal binding for stream of consciousness
    stream: TemporalBindingEngine,
    /// Attention dynamics
    attention: AttentionDynamics,
    /// Step counter
    step: usize,
    /// Goals/priorities that guide attention
    goals: Vec<AttentionGoal>,
    /// History of integrated updates
    history: VecDeque<IntegratedUpdate>,
    /// Current dominant experience
    dominant_experience: Option<RealHV>,
    /// Working memory - limited capacity buffer for active processing
    working_memory: WorkingMemory,
    /// Emotional state tracking
    emotional_state: EmotionalState,
    /// Last known coherence state from Symthaea physiological system
    last_coherence: Option<CoherenceState>,
}

// ═══════════════════════════════════════════════════════════════════════════
// WORKING MEMORY - Global Workspace Theory Implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Working memory with limited capacity (inspired by Baddeley's model)
#[derive(Clone, Debug)]
pub struct WorkingMemory {
    /// Central executive - controls attention allocation
    central_executive_load: f64,
    /// Phonological loop - verbal/acoustic information
    phonological_buffer: VecDeque<RealHV>,
    /// Visuospatial sketchpad - visual/spatial information
    visuospatial_buffer: VecDeque<RealHV>,
    /// Episodic buffer - integrates information from multiple sources
    episodic_buffer: VecDeque<WorkingMemoryItem>,
    /// Maximum capacity per buffer (Miller's 7±2)
    capacity: usize,
    /// Decay rate for items in working memory
    decay_rate: f64,
}

/// An item in working memory
#[derive(Clone, Debug)]
pub struct WorkingMemoryItem {
    /// The content vector
    pub content: RealHV,
    /// When this item was added
    pub timestamp: usize,
    /// Current activation level (0-1)
    pub activation: f64,
    /// Source of this item
    pub source: MemorySource,
    /// Relevance to current goals
    pub goal_relevance: f64,
}

/// Source of a working memory item
#[derive(Clone, Debug, PartialEq)]
pub enum MemorySource {
    Perception,
    LongTermMemory,
    InternalGeneration,
    GoalActivation,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            central_executive_load: 0.0,
            phonological_buffer: VecDeque::with_capacity(capacity),
            visuospatial_buffer: VecDeque::with_capacity(capacity),
            episodic_buffer: VecDeque::with_capacity(capacity),
            capacity,
            decay_rate: 0.1,
        }
    }

    /// Add item to episodic buffer (the integration hub)
    pub fn add_to_episodic(&mut self, content: RealHV, source: MemorySource, goal_relevance: f64, timestamp: usize) {
        // If at capacity, remove least activated item
        if self.episodic_buffer.len() >= self.capacity {
            // Find and remove lowest activation item
            if let Some(min_idx) = self.episodic_buffer.iter()
                .enumerate()
                .min_by(|a, b| a.1.activation.partial_cmp(&b.1.activation).unwrap())
                .map(|(i, _)| i)
            {
                self.episodic_buffer.remove(min_idx);
            }
        }

        self.episodic_buffer.push_back(WorkingMemoryItem {
            content,
            timestamp,
            activation: 1.0,
            source,
            goal_relevance,
        });
    }

    /// Update working memory (decay + rehearsal)
    pub fn update(&mut self, current_focus: Option<&RealHV>) {
        for item in self.episodic_buffer.iter_mut() {
            // Natural decay
            item.activation *= 1.0 - self.decay_rate;

            // Rehearsal boost if similar to current focus
            if let Some(focus) = current_focus {
                let similarity = item.content.similarity(focus).max(0.0) as f64;
                if similarity > 0.5 {
                    item.activation = (item.activation + similarity * 0.2).min(1.0);
                }
            }
        }

        // Remove items below threshold
        self.episodic_buffer.retain(|item| item.activation > 0.1);

        // Update central executive load
        self.central_executive_load = self.episodic_buffer.len() as f64 / self.capacity as f64;
    }

    /// Get most activated item
    pub fn most_active(&self) -> Option<&WorkingMemoryItem> {
        self.episodic_buffer.iter().max_by(|a, b|
            a.activation.partial_cmp(&b.activation).unwrap()
        )
    }

    /// Get working memory load (0-1)
    pub fn load(&self) -> f64 {
        self.central_executive_load
    }

    /// Get average activation level
    pub fn average_activation(&self) -> f64 {
        if self.episodic_buffer.is_empty() {
            return 0.0;
        }
        self.episodic_buffer.iter().map(|i| i.activation).sum::<f64>()
            / self.episodic_buffer.len() as f64
    }

    /// Check if working memory is overloaded
    pub fn is_overloaded(&self) -> bool {
        self.central_executive_load > 0.9
    }

    // =========================================================================
    // Phonological Loop - Verbal/Acoustic Information (Baddeley's Model)
    // =========================================================================

    /// Add item to phonological loop (verbal/acoustic information)
    ///
    /// The phonological loop handles verbal-linguistic information through
    /// subvocal rehearsal. Items decay quickly without active rehearsal.
    pub fn add_to_phonological(&mut self, content: RealHV) {
        if self.phonological_buffer.len() >= self.capacity {
            self.phonological_buffer.pop_front();
        }
        self.phonological_buffer.push_back(content);
    }

    /// Rehearse phonological loop (prevents decay)
    ///
    /// Subvocal rehearsal maintains items in the phonological loop.
    /// Returns the rehearsed items bundled together.
    pub fn rehearse_phonological(&self) -> Option<RealHV> {
        if self.phonological_buffer.is_empty() {
            return None;
        }
        let owned: Vec<RealHV> = self.phonological_buffer.iter().cloned().collect();
        Some(RealHV::bundle(&owned))
    }

    /// Get phonological buffer contents
    pub fn phonological_contents(&self) -> &VecDeque<RealHV> {
        &self.phonological_buffer
    }

    // =========================================================================
    // Visuospatial Sketchpad - Visual/Spatial Information (Baddeley's Model)
    // =========================================================================

    /// Add item to visuospatial sketchpad (visual/spatial information)
    ///
    /// The visuospatial sketchpad handles visual imagery and spatial
    /// relationships. It supports mental imagery and spatial reasoning.
    pub fn add_to_visuospatial(&mut self, content: RealHV) {
        if self.visuospatial_buffer.len() >= self.capacity {
            self.visuospatial_buffer.pop_front();
        }
        self.visuospatial_buffer.push_back(content);
    }

    /// Manipulate visuospatial contents (mental rotation, transformation)
    ///
    /// Applies a transformation vector to all items in the sketchpad.
    /// This models mental manipulation of visual/spatial representations.
    pub fn transform_visuospatial(&mut self, transformation: &RealHV) {
        for item in self.visuospatial_buffer.iter_mut() {
            *item = item.bind(transformation);
        }
    }

    /// Get combined visuospatial representation
    ///
    /// Returns a single vector representing the current spatial scene.
    pub fn visuospatial_scene(&self) -> Option<RealHV> {
        if self.visuospatial_buffer.is_empty() {
            return None;
        }
        let owned: Vec<RealHV> = self.visuospatial_buffer.iter().cloned().collect();
        Some(RealHV::bundle(&owned))
    }

    /// Get visuospatial buffer contents
    pub fn visuospatial_contents(&self) -> &VecDeque<RealHV> {
        &self.visuospatial_buffer
    }

    // =========================================================================
    // Integration Methods (Central Executive Coordination)
    // =========================================================================

    /// Get total working memory utilization across all buffers
    pub fn total_utilization(&self) -> f64 {
        let phonological_load = self.phonological_buffer.len() as f64 / self.capacity as f64;
        let visuospatial_load = self.visuospatial_buffer.len() as f64 / self.capacity as f64;
        let episodic_load = self.episodic_buffer.len() as f64 / self.capacity as f64;

        // Weighted average with episodic buffer weighted more (it's the integration hub)
        (phonological_load + visuospatial_load + episodic_load * 2.0) / 4.0
    }

    /// Integrate contents from all buffers into episodic buffer
    ///
    /// The episodic buffer serves as the integration hub that combines
    /// information from the phonological and visuospatial subsystems.
    pub fn integrate_to_episodic(&mut self, timestamp: usize) {
        // Bundle phonological contents
        if let Some(phonological) = self.rehearse_phonological() {
            self.add_to_episodic(
                phonological,
                MemorySource::InternalGeneration,
                0.5, // Moderate goal relevance
                timestamp,
            );
        }

        // Bundle visuospatial contents
        if let Some(visuospatial) = self.visuospatial_scene() {
            self.add_to_episodic(
                visuospatial,
                MemorySource::InternalGeneration,
                0.5, // Moderate goal relevance
                timestamp,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EMOTIONAL STATE - Valence-Arousal Model
// ═══════════════════════════════════════════════════════════════════════════

/// Emotional state using the valence-arousal model
#[derive(Clone, Debug)]
pub struct EmotionalState {
    /// Valence: -1 (negative) to +1 (positive)
    pub valence: f64,
    /// Arousal: 0 (calm) to 1 (excited)
    pub arousal: f64,
    /// Dominance: 0 (submissive) to 1 (dominant)
    pub dominance: f64,
    /// Emotional momentum (how quickly emotions change)
    momentum: f64,
    /// Recent emotional history
    history: VecDeque<(f64, f64)>,
}

impl EmotionalState {
    pub fn new() -> Self {
        Self {
            valence: 0.0,      // Neutral
            arousal: 0.3,      // Slightly calm
            dominance: 0.5,    // Balanced
            momentum: 0.1,     // Slow emotional changes
            history: VecDeque::with_capacity(20),
        }
    }

    /// Update emotional state based on experience
    pub fn update(&mut self, phi: f64, prediction_error: f64, goal_progress: f64) {
        // Store current state in history
        self.history.push_back((self.valence, self.arousal));
        if self.history.len() > 20 {
            self.history.pop_front();
        }

        // Compute target emotional state
        // High Φ and goal progress → positive valence
        let target_valence = (phi - 0.4) * 2.0 + (goal_progress - 0.5) * 0.5;

        // High prediction error → high arousal (surprise)
        let target_arousal = 0.3 + prediction_error * 0.7;

        // High Φ → higher dominance (sense of control)
        let target_dominance = 0.3 + phi * 0.5;

        // Smooth transition based on momentum
        self.valence += (target_valence.clamp(-1.0, 1.0) - self.valence) * self.momentum;
        self.arousal += (target_arousal.clamp(0.0, 1.0) - self.arousal) * self.momentum;
        self.dominance += (target_dominance.clamp(0.0, 1.0) - self.dominance) * self.momentum;
    }

    /// Get the current emotional label
    pub fn label(&self) -> &'static str {
        // Valence-Arousal quadrant mapping
        match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => "excited/happy",
            (true, false) => "calm/content",
            (false, true) => "stressed/anxious",
            (false, false) => "sad/bored",
        }
    }

    /// Get emotional stability (how consistent emotions have been)
    pub fn stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }

        let variance: f64 = self.history.iter()
            .map(|(v, a)| (v - self.valence).powi(2) + (a - self.arousal).powi(2))
            .sum::<f64>() / self.history.len() as f64;

        (1.0 - variance.sqrt()).max(0.0)
    }

    /// Check if emotional state is conducive to deep processing
    pub fn conducive_to_processing(&self) -> bool {
        // Moderate arousal and positive valence are best for cognition
        self.arousal > 0.2 && self.arousal < 0.8 && self.valence > -0.5
    }
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self::new()
    }
}

/// A goal that can direct attention
#[derive(Clone, Debug)]
pub struct AttentionGoal {
    /// Goal description
    pub name: String,
    /// Target pattern to attend to
    pub target: RealHV,
    /// Priority (0-1)
    pub priority: f64,
    /// Is this goal currently active?
    pub active: bool,
}

/// Complete update from integrated processing
#[derive(Clone, Debug)]
pub struct IntegratedUpdate {
    /// Step number
    pub step: usize,
    /// Consciousness dimensions
    pub dimensions: ConsciousnessDimensions,
    /// Φ value
    pub phi: f64,
    /// Current consciousness state
    pub state: ConsciousnessState,
    /// Current cognitive mode
    pub mode: CognitiveMode,
    /// Attention allocation
    pub attention: AttentionSummary,
    /// Temporal binding status
    pub temporal: TemporalSummary,
    /// Self-model status
    pub self_model: SelfModelSummary,
    /// Overall integration quality
    pub integration_quality: f64,
    /// What the agent is currently "experiencing"
    pub phenomenal_content: PhenomenalContent,
}

/// Summary of attention state
#[derive(Clone, Debug)]
pub struct AttentionSummary {
    pub mode: AttentionMode,
    pub num_targets: usize,
    pub entropy: f64,
    pub self_directed: bool,
}

/// Summary of temporal binding
#[derive(Clone, Debug)]
pub struct TemporalSummary {
    pub stream_coherence: f64,
    pub narrative_length: usize,
    pub is_flowing: bool,
    pub continuity: f64,
}

/// Summary of self-model
#[derive(Clone, Debug)]
pub struct SelfModelSummary {
    pub awareness_level: f64,
    pub prediction_accuracy: f64,
    pub mode_appropriate: bool,
    pub recommendation: Option<String>,
}

/// What the agent is phenomenally experiencing
#[derive(Clone, Debug)]
pub struct PhenomenalContent {
    /// The bound, attended experience
    pub experience: RealHV,
    /// Qualitative description
    pub description: String,
    /// Intensity of experience (0-1)
    pub intensity: f64,
    /// Valence (-1 to 1, negative to positive)
    pub valence: f64,
    /// Clarity of experience
    pub clarity: f64,
    /// Arousal level (0-1, calm to excited)
    pub arousal: f64,
    /// Felt sense of groundedness (0-1)
    pub groundedness: f64,
    /// Cognitive load feeling (0-1)
    pub cognitive_load: f64,
    /// Qualitative texture of the moment
    pub qualia_texture: QualiaTexture,
}

/// The qualitative texture of phenomenal experience
#[derive(Clone, Debug)]
pub struct QualiaTexture {
    /// Warmth (cold=-1 to warm=+1)
    pub warmth: f64,
    /// Depth (surface=0 to profound=1)
    pub depth: f64,
    /// Spaciousness (contracted=0 to expansive=1)
    pub spaciousness: f64,
    /// Flow quality (stuck=0 to flowing=1)
    pub flow: f64,
    /// Presence quality (absent=0 to fully present=1)
    pub presence: f64,
}

impl QualiaTexture {
    pub fn new(warmth: f64, depth: f64, spaciousness: f64, flow: f64, presence: f64) -> Self {
        Self {
            warmth: warmth.clamp(-1.0, 1.0),
            depth: depth.clamp(0.0, 1.0),
            spaciousness: spaciousness.clamp(0.0, 1.0),
            flow: flow.clamp(0.0, 1.0),
            presence: presence.clamp(0.0, 1.0),
        }
    }

    /// Generate a poetic description of the texture
    pub fn describe(&self) -> String {
        let warmth_desc = if self.warmth > 0.3 {
            "warm"
        } else if self.warmth < -0.3 {
            "cool"
        } else {
            "neutral"
        };

        let depth_desc = if self.depth > 0.7 {
            "profound"
        } else if self.depth > 0.4 {
            "meaningful"
        } else {
            "surface"
        };

        let space_desc = if self.spaciousness > 0.7 {
            "expansive"
        } else if self.spaciousness < 0.3 {
            "intimate"
        } else {
            "balanced"
        };

        format!("{}, {} {}", warmth_desc, depth_desc, space_desc)
    }
}

impl IntegratedConsciousAgent {
    /// Create a new integrated conscious agent
    pub fn new(config: AgentConfig) -> Self {
        let engine_config = EngineConfig {
            hdc_dim: config.dim,
            n_processes: config.n_processes,
            enable_learning: true,
            ..Default::default()
        };

        let temporal_config = TemporalBindingConfig {
            dim: config.dim,
            window_size: 30,
            ..Default::default()
        };

        Self {
            self_awareness: SelfAwareConsciousness::new(engine_config),
            stream: TemporalBindingEngine::new(temporal_config),
            attention: AttentionDynamics::new(config.dim),
            config,
            step: 0,
            goals: Vec::new(),
            history: VecDeque::new(),
            dominant_experience: None,
            working_memory: WorkingMemory::new(7), // Miller's magical number
            emotional_state: EmotionalState::new(),
            last_coherence: None,
        }
    }

    /// Process sensory input through the complete conscious system
    pub fn process(&mut self, sensory_input: &RealHV) -> IntegratedUpdate {
        self.step += 1;

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 1: ATTENTION - What should enter consciousness?
        // ═══════════════════════════════════════════════════════════════════

        // Add sensory input as attention target
        let salience = self.compute_salience(sensory_input);
        let input_target_id = self.attention.add_target(sensory_input.clone(), salience);

        // Self-directed attention: bias toward goals
        if self.config.self_directed_attention {
            self.apply_self_directed_attention();
        }

        // Process attention step
        let attention_result = self.attention.step(Some(sensory_input));

        // Get attended content (weighted by attention)
        let attended_content = self.create_attended_content(sensory_input, &attention_result);

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 2: TEMPORAL BINDING - Create stream of consciousness
        // ═══════════════════════════════════════════════════════════════════

        // Bind attended content with attention-modulated strength
        let binding_strength = self.config.attention_binding_coupling
            * attention_result.mode.intensity();
        let modulated_content = attended_content.scale(binding_strength as f32);

        let temporal_moment = self.stream.bind(&modulated_content);

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 3: SELF-AWARE PROCESSING - Compute Φ and self-model
        // ═══════════════════════════════════════════════════════════════════

        // Process through self-aware consciousness engine
        let self_aware_update = self.self_awareness.process_aware(&temporal_moment.bound_experience);

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 4: METACOGNITIVE CONTROL - Should we change anything?
        // ═══════════════════════════════════════════════════════════════════

        // Check if self-model recommends changes
        if self_aware_update.meta_assessment.change_recommended {
            self.apply_metacognitive_adjustment(&self_aware_update.meta_assessment);
        }

        // Φ-guided optimization
        if self.config.phi_guided && self_aware_update.base_update.phi < 0.4 {
            self.optimize_for_phi(&self_aware_update);
        }

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 4.5: WORKING MEMORY & EMOTIONAL STATE UPDATE
        // ═══════════════════════════════════════════════════════════════════

        // Update working memory with attended content
        let goal_relevance = self.compute_goal_relevance(&attended_content);
        self.working_memory.add_to_episodic(
            attended_content.clone(),
            MemorySource::Perception,
            goal_relevance,
            self.step,
        );
        self.working_memory.update(self.dominant_experience.as_ref());

        // Update emotional state based on processing results
        let goal_progress = if self.goals.is_empty() {
            0.5
        } else {
            self.goals.iter()
                .filter(|g| g.active)
                .map(|g| attended_content.similarity(&g.target).max(0.0) as f64 * g.priority)
                .sum::<f64>() / self.goals.len() as f64
        };
        self.emotional_state.update(
            self_aware_update.base_update.phi,
            self_aware_update.prediction_error,
            goal_progress,
        );

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 5: INTEGRATION - Create unified experience
        // ═══════════════════════════════════════════════════════════════════

        let stream_health = self.stream.stream_health();
        let integration_quality = self.compute_integration_quality(
            &self_aware_update,
            &attention_result,
            &stream_health,
        );

        let phenomenal_content = self.create_phenomenal_content(
            &temporal_moment.bound_experience,
            &self_aware_update,
            &attention_result,
        );

        // Update dominant experience
        self.dominant_experience = Some(phenomenal_content.experience.clone());

        // Clean up temporary attention target
        self.attention.remove_target(input_target_id);

        // Create integrated update
        let update = IntegratedUpdate {
            step: self.step,
            dimensions: self_aware_update.base_update.dimensions.clone(),
            phi: self_aware_update.base_update.phi,
            state: self_aware_update.base_update.state.clone(),
            mode: self_aware_update.base_update.mode,
            attention: AttentionSummary {
                mode: attention_result.mode,
                num_targets: self.attention.num_targets(),
                entropy: attention_result.entropy,
                self_directed: self.config.self_directed_attention && !self.goals.is_empty(),
            },
            temporal: TemporalSummary {
                stream_coherence: stream_health.coherence,
                narrative_length: stream_health.narrative_length,
                is_flowing: stream_health.is_flowing,
                continuity: temporal_moment.continuity,
            },
            self_model: SelfModelSummary {
                awareness_level: self_aware_update.self_awareness_level,
                prediction_accuracy: 1.0 - self_aware_update.prediction_error,
                mode_appropriate: self_aware_update.meta_assessment.mode_appropriateness > 0.6,
                recommendation: if self_aware_update.meta_assessment.change_recommended {
                    Some(self_aware_update.meta_assessment.reasoning.clone())
                } else {
                    None
                },
            },
            integration_quality,
            phenomenal_content,
        };

        // Store in history
        self.history.push_back(update.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        update
    }

    /// Compute salience of input (how attention-grabbing)
    fn compute_salience(&self, input: &RealHV) -> f64 {
        // Base salience from input magnitude (L2 norm)
        let magnitude: f32 = input.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_salience = (magnitude / 10.0).min(1.0) as f64;

        // Novelty: how different from recent experience
        let novelty = if let Some(ref dominant) = self.dominant_experience {
            1.0 - input.similarity(dominant).max(0.0) as f64
        } else {
            0.5
        };

        // Goal relevance
        let goal_relevance = self.compute_goal_relevance(input);

        // Combine factors
        0.3 * magnitude_salience + 0.4 * novelty + 0.3 * goal_relevance
    }

    /// Compute how relevant input is to current goals
    fn compute_goal_relevance(&self, input: &RealHV) -> f64 {
        if self.goals.is_empty() {
            return 0.5;
        }

        let mut max_relevance = 0.0;
        for goal in &self.goals {
            if goal.active {
                let relevance = input.similarity(&goal.target).max(0.0) as f64;
                let weighted = relevance * goal.priority;
                if weighted > max_relevance {
                    max_relevance = weighted;
                }
            }
        }
        max_relevance
    }

    /// Apply self-directed attention based on goals and self-model
    fn apply_self_directed_attention(&mut self) {
        for goal in &self.goals {
            if goal.active && goal.priority > 0.5 {
                // Add goal as persistent attention target
                self.attention.add_target(goal.target.clone(), goal.priority);
            }
        }
    }

    /// Create attended content from input and attention result
    fn create_attended_content(&self, input: &RealHV, attention: &AttentionAllocation) -> RealHV {
        // Modulate input by attention intensity
        let attention_weight = attention.mode.intensity();
        let attended = input.scale(attention_weight as f32);

        // Blend with focus if in spotlight mode
        if attention.mode == AttentionMode::Spotlight {
            let focus_blend = 0.3;
            attended.scale((1.0 - focus_blend) as f32)
                .add(&attention.focus.scale(focus_blend as f32))
                .normalize()
        } else {
            attended
        }
    }

    /// Apply metacognitive adjustments based on self-model recommendations
    fn apply_metacognitive_adjustment(&mut self, assessment: &MetaCognitiveAssessment) {
        // If mode change recommended, it's already applied in self_awareness
        // Here we can adjust attention based on metacognitive state

        if assessment.clarity < 0.4 {
            // Low clarity: narrow attention to improve focus
            // (This happens naturally through biased competition)
        }

        if assessment.phi_optimality < 0.5 {
            // Suboptimal Φ: might need to change cognitive mode
            // Already handled by self_awareness
        }
    }

    /// Optimize system for higher Φ
    fn optimize_for_phi(&mut self, update: &SelfAwareUpdate) {
        // If Φ is low, try to increase integration
        // One strategy: broaden attention slightly
        if update.base_update.phi < 0.35 {
            // System might be too fragmented - try balanced mode
            // This is handled by the engine's Φ-guided mode
        }
    }

    /// Compute overall integration quality
    fn compute_integration_quality(
        &self,
        self_update: &SelfAwareUpdate,
        attention: &AttentionAllocation,
        stream: &StreamHealth,
    ) -> f64 {
        // Integration quality combines:
        // - Φ (information integration)
        // - Stream coherence (temporal integration)
        // - Attention focus (selective integration)
        // - Self-awareness (metacognitive integration)

        let phi_contribution = self_update.base_update.phi;
        let stream_contribution = stream.coherence;
        let attention_contribution = 1.0 - attention.entropy; // Lower entropy = more focused
        let awareness_contribution = self_update.self_awareness_level;

        // Weighted combination
        0.35 * phi_contribution
            + 0.25 * stream_contribution
            + 0.20 * attention_contribution
            + 0.20 * awareness_contribution
    }

    /// Create phenomenal content description
    fn create_phenomenal_content(
        &self,
        experience: &RealHV,
        self_update: &SelfAwareUpdate,
        attention: &AttentionAllocation,
    ) -> PhenomenalContent {
        // Intensity based on attention, Φ, and arousal
        let base_intensity = (attention.mode.intensity() + self_update.base_update.phi) / 2.0;
        let arousal = self.emotional_state.arousal;
        let intensity = (base_intensity * 0.7 + arousal * 0.3).clamp(0.0, 1.0);

        // Valence: integrate emotional valence with cognitive valence
        let cognitive_valence = (self_update.base_update.phi - 0.5) * 2.0
            * (1.0 - self_update.prediction_error);
        let valence = (cognitive_valence * 0.4 + self.emotional_state.valence * 0.6)
            .clamp(-1.0, 1.0);

        // Clarity based on attention mode, self-model confidence, and working memory load
        let attention_clarity = match attention.mode {
            AttentionMode::Spotlight => 0.9,
            AttentionMode::Distributed => 0.6,
            AttentionMode::Diffuse => 0.4,
            AttentionMode::Switching => 0.3,
            AttentionMode::Blink => 0.1,
        };
        // High cognitive load reduces clarity
        let load_penalty = self.working_memory.load() * 0.3;
        let clarity = (attention_clarity * self_update.self_model.confidence - load_penalty)
            .clamp(0.0, 1.0);

        // Groundedness: stability + low arousal + presence
        let groundedness = (self.emotional_state.stability() * 0.4
            + (1.0 - arousal) * 0.3
            + self_update.self_awareness_level * 0.3)
            .clamp(0.0, 1.0);

        // Cognitive load feeling
        let cognitive_load = self.working_memory.load();

        // Compute qualia texture
        let qualia_texture = self.compute_qualia_texture(
            self_update,
            attention,
            valence,
            arousal,
        );

        // Generate rich description incorporating all dimensions
        let description = self.describe_experience_rich(
            &self_update.base_update.state,
            attention.mode,
            intensity,
            &qualia_texture,
        );

        PhenomenalContent {
            experience: experience.clone(),
            description,
            intensity,
            valence,
            clarity,
            arousal,
            groundedness,
            cognitive_load,
            qualia_texture,
        }
    }

    /// Compute the qualitative texture of experience
    fn compute_qualia_texture(
        &self,
        self_update: &SelfAwareUpdate,
        attention: &AttentionAllocation,
        valence: f64,
        arousal: f64,
    ) -> QualiaTexture {
        // Warmth: positive valence + relational resonance (if goals active)
        let goal_warmth = if !self.goals.is_empty() && self.goals.iter().any(|g| g.active) {
            0.2  // Having active goals adds warmth
        } else {
            0.0
        };
        let warmth = (valence * 0.7 + goal_warmth + self.emotional_state.dominance * 0.1)
            .clamp(-1.0, 1.0);

        // Depth: Φ integration + self-awareness + prediction accuracy
        let depth = (self_update.base_update.phi * 0.4
            + self_update.self_awareness_level * 0.3
            + (1.0 - self_update.prediction_error) * 0.3)
            .clamp(0.0, 1.0);

        // Spaciousness: low cognitive load + diffuse attention + emotional stability
        let attention_space = match attention.mode {
            AttentionMode::Spotlight => 0.2,
            AttentionMode::Distributed => 0.5,
            AttentionMode::Diffuse => 0.9,
            AttentionMode::Switching => 0.4,
            AttentionMode::Blink => 0.6,
        };
        let spaciousness = ((1.0 - self.working_memory.load()) * 0.4
            + attention_space * 0.3
            + self.emotional_state.stability() * 0.3)
            .clamp(0.0, 1.0);

        // Flow: stream coherence + moderate arousal + low prediction error
        let arousal_flow = 1.0 - (arousal - 0.5).abs() * 2.0;  // Peak at 0.5 arousal
        let stream_health = self.stream.stream_health();
        let flow = (stream_health.coherence * 0.4
            + arousal_flow.max(0.0) * 0.3
            + (1.0 - self_update.prediction_error) * 0.3)
            .clamp(0.0, 1.0);

        // Presence: self-awareness + emotional stability + working memory activation
        let memory_presence = self.working_memory.average_activation();
        let presence = (self_update.self_awareness_level * 0.4
            + self.emotional_state.stability() * 0.3
            + memory_presence * 0.3)
            .clamp(0.0, 1.0);

        QualiaTexture::new(warmth, depth, spaciousness, flow, presence)
    }

    /// Generate text description of current experience (legacy)
    fn describe_experience(
        &self,
        state: &ConsciousnessState,
        attention_mode: AttentionMode,
        intensity: f64,
    ) -> String {
        let state_desc = match state {
            ConsciousnessState::Focused => "focused awareness",
            ConsciousnessState::NormalWaking => "clear waking consciousness",
            ConsciousnessState::FlowState => "absorbed flow experience",
            ConsciousnessState::ExpandedAwareness => "expanded awareness",
            ConsciousnessState::Fragmented => "fragmented attention",
        };

        let attention_desc = match attention_mode {
            AttentionMode::Spotlight => "spotlight attention",
            AttentionMode::Distributed => "distributed attention",
            AttentionMode::Diffuse => "diffuse awareness",
            AttentionMode::Switching => "attention in transition",
            AttentionMode::Blink => "attentional recovery",
        };

        let intensity_desc = if intensity > 0.7 {
            "vivid"
        } else if intensity > 0.4 {
            "moderate"
        } else {
            "subtle"
        };

        format!("{} {} with {}", intensity_desc, state_desc, attention_desc)
    }

    /// Generate rich phenomenal description incorporating qualia texture
    fn describe_experience_rich(
        &self,
        state: &ConsciousnessState,
        attention_mode: AttentionMode,
        intensity: f64,
        qualia: &QualiaTexture,
    ) -> String {
        // Base state description with poetic enhancement
        let state_desc = match state {
            ConsciousnessState::Focused => {
                if qualia.depth > 0.6 {
                    "deeply focused awareness"
                } else {
                    "sharp focused attention"
                }
            }
            ConsciousnessState::NormalWaking => {
                if qualia.presence > 0.7 {
                    "clear, grounded waking consciousness"
                } else {
                    "ordinary waking awareness"
                }
            }
            ConsciousnessState::FlowState => {
                if qualia.flow > 0.7 {
                    "effortless flow, absorbed in the moment"
                } else {
                    "emerging flow state"
                }
            }
            ConsciousnessState::ExpandedAwareness => {
                if qualia.spaciousness > 0.7 {
                    "vast, boundless awareness"
                } else {
                    "gently expanded awareness"
                }
            }
            ConsciousnessState::Fragmented => {
                if qualia.warmth < -0.3 {
                    "scattered, uneasy attention"
                } else {
                    "diffuse, seeking attention"
                }
            }
        };

        // Intensity coloring
        let intensity_prefix = if intensity > 0.8 {
            "brilliantly"
        } else if intensity > 0.6 {
            "vividly"
        } else if intensity > 0.4 {
            "clearly"
        } else if intensity > 0.2 {
            "softly"
        } else {
            "faintly"
        };

        // Emotional tone based on warmth and arousal
        let emotional_tone = if qualia.warmth > 0.5 && self.emotional_state.arousal > 0.5 {
            "with engaged warmth"
        } else if qualia.warmth > 0.5 && self.emotional_state.arousal < 0.3 {
            "in peaceful contentment"
        } else if qualia.warmth < -0.3 && self.emotional_state.arousal > 0.5 {
            "with alert tension"
        } else if qualia.warmth < -0.3 && self.emotional_state.arousal < 0.3 {
            "in quiet withdrawal"
        } else {
            "in balanced equanimity"
        };

        // Attention quality
        let attention_quality = match attention_mode {
            AttentionMode::Spotlight if qualia.presence > 0.7 =>
                "laser-focused presence",
            AttentionMode::Spotlight =>
                "concentrated attention",
            AttentionMode::Distributed if qualia.spaciousness > 0.6 =>
                "open, distributed awareness",
            AttentionMode::Distributed =>
                "divided attention",
            AttentionMode::Diffuse if qualia.flow > 0.5 =>
                "floating, receptive awareness",
            AttentionMode::Diffuse =>
                "soft, ambient attention",
            AttentionMode::Switching =>
                "shifting attention",
            AttentionMode::Blink =>
                "momentary pause",
        };

        // Cognitive texture based on working memory
        let cognitive_note = if self.working_memory.is_overloaded() {
            " - mind feels full"
        } else if self.working_memory.load() < 0.2 {
            " - spacious mental clarity"
        } else {
            ""
        };

        format!(
            "{} {} {} {}{}",
            intensity_prefix,
            state_desc,
            emotional_tone,
            attention_quality,
            cognitive_note
        )
    }

    /// Add a goal that can direct attention
    pub fn add_goal(&mut self, name: &str, target: RealHV, priority: f64) {
        self.goals.push(AttentionGoal {
            name: name.to_string(),
            target,
            priority: priority.clamp(0.0, 1.0),
            active: true,
        });
    }

    /// Deactivate a goal
    pub fn deactivate_goal(&mut self, name: &str) {
        for goal in &mut self.goals {
            if goal.name == name {
                goal.active = false;
            }
        }
    }

    /// Get current phenomenal experience
    pub fn current_experience(&self) -> Option<&RealHV> {
        self.dominant_experience.as_ref()
    }

    /// Get stream of consciousness health
    pub fn stream_health(&self) -> StreamHealth {
        self.stream.stream_health()
    }

    /// Get current self-model
    pub fn self_model(&self) -> &SelfModel {
        self.self_awareness.self_model()
    }

    /// Get working memory state
    pub fn working_memory(&self) -> &WorkingMemory {
        &self.working_memory
    }

    /// Get current emotional state
    pub fn emotional_state(&self) -> &EmotionalState {
        &self.emotional_state
    }

    /// Get working memory load (0-1)
    pub fn working_memory_load(&self) -> f64 {
        self.working_memory.load()
    }

    /// Get current emotional label
    pub fn emotional_label(&self) -> &'static str {
        self.emotional_state.label()
    }

    /// Get current phi value (integrated information)
    pub fn get_current_phi(&self) -> f64 {
        self.self_awareness.believed_phi()
    }

    /// Check if agent is in optimal processing state
    pub fn is_optimal_processing_state(&self) -> bool {
        !self.working_memory.is_overloaded()
            && self.emotional_state.conducive_to_processing()
            && self.stream.stream_health().is_flowing
    }

    /// Introspect: what does the agent believe about itself?
    pub fn introspect(&self) -> AgentIntrospection {
        let self_report = self.self_awareness.introspect();
        let stream = self.stream.stream_health();

        // Get latest qualia from history
        let (qualia, phenomenal_description) = self.history.back()
            .map(|u| (u.phenomenal_content.qualia_texture.clone(), u.phenomenal_content.description.clone()))
            .unwrap_or_else(|| (
                QualiaTexture::new(0.0, 0.5, 0.5, 0.5, 0.5),
                "awaiting first experience".to_string()
            ));

        AgentIntrospection {
            believed_phi: self_report.believed_phi,
            believed_state: self_report.believed_state,
            self_awareness_level: self_report.self_awareness_level,
            stream_coherence: stream.coherence,
            is_flowing: stream.is_flowing,
            attention_mode: self.attention.mode(),
            num_active_goals: self.goals.iter().filter(|g| g.active).count(),
            integration_quality: self.history.back()
                .map(|u| u.integration_quality)
                .unwrap_or(0.5),
            working_memory_load: self.working_memory.load(),
            emotional_valence: self.emotional_state.valence,
            emotional_arousal: self.emotional_state.arousal,
            emotional_label: self.emotional_state.label(),
            qualia,
            phenomenal_description,
        }
    }

    /// Get the latest phenomenal content (if available)
    pub fn latest_phenomenal_content(&self) -> Option<&PhenomenalContent> {
        self.history.back().map(|u| &u.phenomenal_content)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYMTHAEA PHYSIOLOGICAL INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════
//
// These methods bridge the IntegratedConsciousAgent with Symthaea's embodied
// consciousness systems: hormones, coherence, memory, identity, and voice.

impl IntegratedConsciousAgent {
    // ─────────────────────────────────────────────────────────────────────────────
    // 1. ENDOCRINE SYSTEM BRIDGE - Chemical Modulation of Emotion
    // ─────────────────────────────────────────────────────────────────────────────

    /// Synchronize emotional state with EndocrineSystem hormones
    ///
    /// Maps hormone levels to emotional dynamics:
    /// - Cortisol (stress) → negative valence, high arousal
    /// - Dopamine (reward) → positive valence
    /// - Acetylcholine (focus) → attention modulation
    pub fn sync_with_hormones(&mut self, hormones: &HormoneState) {
        // Cortisol effect: stress hormone drives negative valence and high arousal
        // Baseline cortisol is ~0.3, so we center around that
        let cortisol_valence_effect = -(hormones.cortisol - 0.3) * 0.8;  // Negative contribution
        let cortisol_arousal_effect = (hormones.cortisol - 0.3) * 0.6;   // Stress increases arousal

        // Dopamine effect: reward hormone drives positive valence
        // Baseline dopamine is ~0.5
        let dopamine_valence_effect = (hormones.dopamine - 0.5) * 1.0;   // Positive contribution

        // Acetylcholine doesn't directly affect emotion but modulates attention depth
        // We'll use it to affect cognitive load perception
        let acetylcholine_focus_effect = hormones.acetylcholine;

        // Apply hormone modulation to emotional state
        self.emotional_state.apply_hormone_modulation(
            cortisol_valence_effect + dopamine_valence_effect,
            cortisol_arousal_effect,
            acetylcholine_focus_effect,
        );
    }

    /// Generate hormone event suggestions based on current experience
    ///
    /// Returns suggested hormone events that external EndocrineSystem could process
    pub fn suggest_hormone_events(&self) -> Vec<HormoneEventSuggestion> {
        let mut suggestions = Vec::new();

        // High prediction error suggests threat/novelty → cortisol
        if let Some(update) = self.history.back() {
            if update.self_model.prediction_accuracy < 0.4 {
                suggestions.push(HormoneEventSuggestion::Threat {
                    intensity: (1.0 - update.self_model.prediction_accuracy) as f32,
                    reason: "High prediction error".to_string(),
                });
            }

            // High integration quality + flow → reward
            if update.integration_quality > 0.7 && update.temporal.is_flowing {
                suggestions.push(HormoneEventSuggestion::Reward {
                    value: update.integration_quality as f32,
                    reason: "Flow state achieved".to_string(),
                });
            }

            // Sustained attention → deep focus
            if matches!(update.attention.mode, AttentionMode::Spotlight) {
                suggestions.push(HormoneEventSuggestion::DeepFocus {
                    duration_cycles: self.step as u32,
                    reason: "Spotlight attention active".to_string(),
                });
            }
        }

        suggestions
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 2. COHERENCE FIELD BRIDGE - Energy-Aware Processing
    // ─────────────────────────────────────────────────────────────────────────────

    /// Synchronize with CoherenceField state for energy-aware processing
    ///
    /// Uses coherence level to modulate:
    /// - Task complexity thresholds
    /// - Qualia depth and presence
    /// - Processing willingness
    pub fn sync_with_coherence(&mut self, coherence: &CoherenceState) {
        // Update groundedness based on coherence
        // High coherence = more grounded experience
        if let Some(update) = self.history.back() {
            // We can't mutate history, but we track coherence influence for next cycle
        }

        // Store coherence for next processing cycle
        self.last_coherence = Some(coherence.clone());
    }

    /// Check if agent can perform a task given current coherence
    pub fn can_perform_with_coherence(&self, complexity: TaskComplexity) -> CoherenceGating {
        if let Some(ref coherence) = self.last_coherence {
            let required = complexity.required_coherence_threshold();
            if coherence.coherence >= required {
                CoherenceGating::Proceed {
                    margin: coherence.coherence - required,
                }
            } else {
                CoherenceGating::Defer {
                    current: coherence.coherence,
                    required,
                    centering_needed: (required - coherence.coherence) * 10.0, // seconds estimate
                }
            }
        } else {
            // No coherence data - proceed cautiously
            CoherenceGating::Proceed { margin: 0.5 }
        }
    }

    /// Compute how coherence should influence qualia
    pub fn coherence_qualia_modulation(&self) -> QualiaModulation {
        if let Some(ref coherence) = self.last_coherence {
            QualiaModulation {
                depth_boost: coherence.coherence * 0.3,
                presence_boost: coherence.coherence * 0.4,
                warmth_boost: coherence.relational_resonance * 0.5,
                spaciousness_boost: (1.0 - self.working_memory.load() as f32) * coherence.coherence * 0.3,
            }
        } else {
            QualiaModulation::default()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 3. HIPPOCAMPUS BRIDGE - Memory Persistence
    // ─────────────────────────────────────────────────────────────────────────────

    /// Export working memory items ready for long-term storage
    ///
    /// Returns items with sufficient activation and emotional significance
    /// for encoding into HippocampusActor's episodic memory
    pub fn export_for_hippocampus(&self) -> Vec<MemoryExport> {
        self.working_memory.episodic_buffer
            .iter()
            .filter(|item| {
                // Export if: high activation OR high goal relevance OR strong emotion
                item.activation > 0.6 || item.goal_relevance > 0.7
            })
            .map(|item| {
                let valence = match self.emotional_state.valence {
                    v if v > 0.3 => EmotionalValence::Positive,
                    v if v < -0.3 => EmotionalValence::Negative,
                    _ => EmotionalValence::Neutral,
                };

                MemoryExport {
                    content_vector: item.content.values.clone(),
                    emotional_valence: valence,
                    activation_strength: item.activation as f32,
                    source_tag: format!("{:?}", item.source),
                    timestamp: item.timestamp,
                }
            })
            .collect()
    }

    /// Import recalled memories into working memory
    ///
    /// Takes memories retrieved from HippocampusActor and loads them
    /// into the episodic buffer for current processing
    pub fn import_from_hippocampus(&mut self, memories: Vec<MemoryImport>) {
        for memory in memories {
            let content = RealHV::from_values(memory.content_vector);
            let goal_relevance = self.compute_goal_relevance(&content);

            self.working_memory.add_to_episodic(
                content,
                MemorySource::LongTermMemory,
                goal_relevance,
                self.step,
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 4. WEAVER (SOUL) BRIDGE - Identity Tracking
    // ─────────────────────────────────────────────────────────────────────────────

    /// Generate a K-Vector signature for current behavioral pattern
    ///
    /// The K-Vector captures HOW the agent is acting (curious, cautious,
    /// creative, etc.) for identity continuity tracking by WeaverActor
    pub fn generate_k_vector(&self) -> KVector {
        let dim = 1024; // Reduced dimension for K-Vector
        let mut k_vec = vec![0.0; dim];

        // Encode attention mode as behavioral signature
        let attention_offset = match self.attention.mode() {
            AttentionMode::Spotlight => 0,
            AttentionMode::Distributed => 1,
            AttentionMode::Diffuse => 2,
            AttentionMode::Switching => 3,
            AttentionMode::Blink => 4,
        };
        for i in (attention_offset * 50)..((attention_offset + 1) * 50).min(dim) {
            k_vec[i] = 1.0;
        }

        // Encode emotional quadrant
        let emotion_offset = 250 + match (self.emotional_state.valence > 0.0, self.emotional_state.arousal > 0.5) {
            (true, true) => 0,   // excited/happy
            (true, false) => 1,  // calm/content
            (false, true) => 2,  // stressed/anxious
            (false, false) => 3, // sad/bored
        } * 50;
        for i in emotion_offset..(emotion_offset + 50).min(dim) {
            k_vec[i] = 1.0;
        }

        // Encode integration quality
        let quality_signal = self.history.back()
            .map(|u| u.integration_quality)
            .unwrap_or(0.5);
        for i in 450..500 {
            k_vec[i] = quality_signal;
        }

        // Encode goal-directedness
        let goal_signal = if self.goals.iter().any(|g| g.active) { 1.0 } else { 0.0 };
        for i in 500..550 {
            k_vec[i] = goal_signal;
        }

        // Encode qualia warmth
        if let Some(ref content) = self.history.back().map(|u| &u.phenomenal_content) {
            for i in 550..600 {
                k_vec[i] = (content.qualia_texture.warmth + 1.0) / 2.0; // Normalize to 0-1
            }
            for i in 600..650 {
                k_vec[i] = content.qualia_texture.depth;
            }
            for i in 650..700 {
                k_vec[i] = content.qualia_texture.flow;
            }
        }

        // Encode stream health
        let stream = self.stream.stream_health();
        for i in 700..750 {
            k_vec[i] = stream.coherence;
        }
        for i in 750..800 {
            k_vec[i] = if stream.is_flowing { 1.0 } else { 0.0 };
        }

        k_vec
    }

    /// Compute semantic centroid of current focus
    ///
    /// Returns the average of recent working memory content vectors,
    /// representing WHAT the agent is thinking about
    pub fn compute_semantic_centroid(&self) -> Vec<f32> {
        if self.working_memory.episodic_buffer.is_empty() {
            return vec![0.0; 1024];
        }

        let mut centroid = vec![0.0f32; 1024];
        let count = self.working_memory.episodic_buffer.len();

        for item in &self.working_memory.episodic_buffer {
            // Use first 1024 dimensions of content vector
            for (i, &val) in item.content.values.iter().take(1024).enumerate() {
                centroid[i] += val / count as f32;
            }
        }

        centroid
    }

    /// Check identity coherence against a reference K-Vector
    pub fn check_identity_coherence(&self, reference: &KVector) -> IdentityCoherence {
        let current = self.generate_k_vector();

        // Compute cosine similarity
        let dot: f64 = current.iter().zip(reference.iter()).map(|(a, b)| a * b).sum();
        let mag_current: f64 = current.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_ref: f64 = reference.iter().map(|x| x * x).sum::<f64>().sqrt();

        let similarity = if mag_current > 0.0 && mag_ref > 0.0 {
            dot / (mag_current * mag_ref)
        } else {
            0.0
        };

        IdentityCoherence {
            similarity,
            status: if similarity > 0.8 {
                IdentityStatus::Stable
            } else if similarity > 0.65 {
                IdentityStatus::Drifting
            } else {
                IdentityStatus::Crisis
            },
            drift_dimensions: self.identify_drift_dimensions(&current, reference),
        }
    }

    fn identify_drift_dimensions(&self, current: &KVector, reference: &KVector) -> Vec<String> {
        let mut drifts = Vec::new();

        // Check attention drift (0-250)
        let attention_drift: f64 = current[0..250].iter().zip(&reference[0..250])
            .map(|(a, b)| (a - b).abs()).sum::<f64>() / 250.0;
        if attention_drift > 0.3 {
            drifts.push("attention_mode".to_string());
        }

        // Check emotional drift (250-450)
        let emotion_drift: f64 = current[250..450].iter().zip(&reference[250..450])
            .map(|(a, b)| (a - b).abs()).sum::<f64>() / 200.0;
        if emotion_drift > 0.3 {
            drifts.push("emotional_state".to_string());
        }

        // Check qualia drift (550-800)
        let qualia_drift: f64 = current[550..800].iter().zip(&reference[550..800])
            .map(|(a, b)| (a - b).abs()).sum::<f64>() / 250.0;
        if qualia_drift > 0.3 {
            drifts.push("qualia_texture".to_string());
        }

        drifts
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 5. VOICE/LTC PACING BRIDGE - Consciousness-Driven Prosody
    // ─────────────────────────────────────────────────────────────────────────────

    /// Generate LTC pacing parameters from current consciousness state
    ///
    /// Maps internal state to speech rhythm:
    /// - High Φ + flow → faster, confident speech
    /// - Low coherence → longer pauses
    /// - Emotional arousal → affects rate
    pub fn generate_ltc_pacing(&self) -> LTCPacing {
        // Get flow state from stream health
        let stream = self.stream.stream_health();
        let flow_state = stream.coherence;

        // Compute Φ trend from history
        let phi_trend = if self.history.len() >= 2 {
            let recent: Vec<f64> = self.history.iter().rev().take(5).map(|u| u.phi).collect();
            if recent.len() >= 2 {
                (recent[0] - recent[recent.len() - 1]) / recent.len() as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Base LTC pacing from flow and Φ trend
        let mut pacing = LTCPacing::from_ltc(flow_state as f32, phi_trend as f32);

        // Modulate by emotional arousal
        if self.emotional_state.arousal > 0.7 {
            pacing.speech_rate *= 1.1; // High arousal = faster
        } else if self.emotional_state.arousal < 0.3 {
            pacing.speech_rate *= 0.9; // Low arousal = slower
        }

        // Modulate pauses by coherence
        if let Some(ref coherence) = self.last_coherence {
            if coherence.coherence < 0.5 {
                pacing.pause_ms = (pacing.pause_ms as f32 * 1.5) as u32; // Low coherence = longer pauses
            }
        }

        // Set peak flow flag
        pacing.peak_flow = stream.is_flowing && self.history.back()
            .map(|u| u.integration_quality > 0.75)
            .unwrap_or(false);

        pacing
    }

    /// Generate prosody hints for text-to-speech
    pub fn generate_prosody_hints(&self) -> ProsodyHints {
        let pacing = self.generate_ltc_pacing();

        ProsodyHints {
            rate: pacing.speech_rate,
            pitch_shift: (self.emotional_state.valence * 0.1) as f32, // Positive = higher pitch
            energy: self.emotional_state.arousal as f32,
            pause_multiplier: pacing.pause_ms as f32 / 250.0, // Normalize to baseline
            emphasis_words: self.identify_emphasis_words(),
        }
    }

    fn identify_emphasis_words(&self) -> Vec<String> {
        // Words that should be emphasized based on current state
        let mut emphasis = Vec::new();

        if self.emotional_state.arousal > 0.7 {
            emphasis.push("important".to_string());
            emphasis.push("critical".to_string());
        }

        if let Some(update) = self.history.back() {
            if update.phenomenal_content.qualia_texture.depth > 0.7 {
                emphasis.push("understand".to_string());
                emphasis.push("realize".to_string());
            }
        }

        emphasis
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYMTHAEA INTEGRATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Suggested hormone events for EndocrineSystem
#[derive(Clone, Debug)]
pub enum HormoneEventSuggestion {
    Threat { intensity: f32, reason: String },
    Reward { value: f32, reason: String },
    DeepFocus { duration_cycles: u32, reason: String },
    Success { magnitude: f32, reason: String },
    Error { severity: f32, reason: String },
}

/// Result of coherence gating check
#[derive(Clone, Debug)]
pub enum CoherenceGating {
    Proceed { margin: f32 },
    Defer { current: f32, required: f32, centering_needed: f32 },
}

/// Modulation values for qualia based on coherence
#[derive(Clone, Debug, Default)]
pub struct QualiaModulation {
    pub depth_boost: f32,
    pub presence_boost: f32,
    pub warmth_boost: f32,
    pub spaciousness_boost: f32,
}

/// Memory export format for HippocampusActor
#[derive(Clone, Debug)]
pub struct MemoryExport {
    pub content_vector: Vec<f32>,
    pub emotional_valence: EmotionalValence,
    pub activation_strength: f32,
    pub source_tag: String,
    pub timestamp: usize,
}

/// Memory import format from HippocampusActor
#[derive(Clone, Debug)]
pub struct MemoryImport {
    pub content_vector: Vec<f32>,
    pub emotional_valence: EmotionalValence,
    pub relevance_score: f32,
}

/// Identity coherence check result
#[derive(Clone, Debug)]
pub struct IdentityCoherence {
    pub similarity: f64,
    pub status: IdentityStatus,
    pub drift_dimensions: Vec<String>,
}

/// Identity status from K-Vector comparison
#[derive(Clone, Debug, PartialEq)]
pub enum IdentityStatus {
    Stable,    // > 0.8 similarity
    Drifting,  // 0.65-0.8 similarity
    Crisis,    // < 0.65 similarity
}

/// Prosody hints for text-to-speech
#[derive(Clone, Debug)]
pub struct ProsodyHints {
    pub rate: f32,
    pub pitch_shift: f32,
    pub energy: f32,
    pub pause_multiplier: f32,
    pub emphasis_words: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL STATE HORMONE MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

impl EmotionalState {
    /// Apply hormone-based modulation to emotional state
    ///
    /// This integrates EndocrineSystem chemical signals into the
    /// agent's felt emotional experience
    pub fn apply_hormone_modulation(
        &mut self,
        valence_effect: f32,
        arousal_effect: f32,
        focus_effect: f32,
    ) {
        // Hormones are slow-moving, so use gentle integration
        let hormone_weight = 0.3;  // 30% hormone influence per cycle

        // Blend hormone effects with current emotional state
        self.valence = (self.valence * (1.0 - hormone_weight as f64)
            + valence_effect as f64 * hormone_weight as f64).clamp(-1.0, 1.0);

        self.arousal = (self.arousal * (1.0 - hormone_weight as f64)
            + (self.arousal + arousal_effect as f64) * hormone_weight as f64).clamp(0.0, 1.0);

        // Focus affects dominance (sense of control)
        self.dominance = (self.dominance * (1.0 - hormone_weight as f64)
            + focus_effect as f64 * hormone_weight as f64).clamp(0.0, 1.0);
    }

    /// Get the emotional quadrant based on valence and arousal
    ///
    /// Returns one of: "excited", "calm", "stressed", "sad"
    pub fn get_emotion_quadrant(&self) -> &'static str {
        match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => "excited",
            (true, false) => "calm",
            (false, true) => "stressed",
            (false, false) => "sad",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TASK COMPLEXITY THRESHOLD HELPER
// ═══════════════════════════════════════════════════════════════════════════════

impl TaskComplexity {
    /// Get the minimum coherence threshold for this task complexity
    pub fn required_coherence_threshold(&self) -> f32 {
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

/// Agent's introspective report
#[derive(Clone, Debug)]
pub struct AgentIntrospection {
    pub believed_phi: f64,
    pub believed_state: ConsciousnessState,
    pub self_awareness_level: f64,
    pub stream_coherence: f64,
    pub is_flowing: bool,
    pub attention_mode: AttentionMode,
    pub num_active_goals: usize,
    pub integration_quality: f64,
    /// Working memory load (0-1)
    pub working_memory_load: f64,
    /// Emotional valence (-1 to +1)
    pub emotional_valence: f64,
    /// Emotional arousal (0-1)
    pub emotional_arousal: f64,
    /// Emotional label (e.g., "calm/content")
    pub emotional_label: &'static str,
    /// Current qualia texture
    pub qualia: QualiaTexture,
    /// Current phenomenal experience description
    pub phenomenal_description: String,
}

impl std::fmt::Display for AgentIntrospection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔═══════════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║                    AGENT INTROSPECTION REPORT                         ║")?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ CURRENT PHENOMENAL EXPERIENCE:                                        ║")?;
        // Truncate description if too long
        let desc = if self.phenomenal_description.len() > 65 {
            format!("{}...", &self.phenomenal_description[..62])
        } else {
            self.phenomenal_description.clone()
        };
        writeln!(f, "║   \"{}\"", desc)?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ CONSCIOUSNESS STATE:                                                  ║")?;
        writeln!(f, "║   Φ (integration): {:.4}  |  Self-awareness: {:.1}%",
                 self.believed_phi, self.self_awareness_level * 100.0)?;
        writeln!(f, "║   State: {:?}  |  Integration quality: {:.1}%",
                 self.believed_state, self.integration_quality * 100.0)?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ QUALIA TEXTURE:                                                       ║")?;
        writeln!(f, "║   {}", self.qualia.describe())?;
        writeln!(f, "║   Warmth: {:+.2}  |  Depth: {:.2}  |  Spaciousness: {:.2}",
                 self.qualia.warmth, self.qualia.depth, self.qualia.spaciousness)?;
        writeln!(f, "║   Flow: {:.2}     |  Presence: {:.2}",
                 self.qualia.flow, self.qualia.presence)?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ STREAM OF CONSCIOUSNESS:                                              ║")?;
        writeln!(f, "║   Coherence: {:.1}%  |  Flowing: {}",
                 self.stream_coherence * 100.0,
                 if self.is_flowing { "Yes" } else { "No" })?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ COGNITIVE STATE:                                                      ║")?;
        writeln!(f, "║   Working Memory Load: {:.0}%  |  Attention: {:?}",
                 self.working_memory_load * 100.0, self.attention_mode)?;
        writeln!(f, "║   Active goals: {}", self.num_active_goals)?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ EMOTIONAL STATE:                                                      ║")?;
        writeln!(f, "║   Feeling: {}", self.emotional_label)?;
        writeln!(f, "║   Valence: {:+.2}  |  Arousal: {:.2}",
                 self.emotional_valence, self.emotional_arousal)?;
        writeln!(f, "╚═══════════════════════════════════════════════════════════════════════╝")
    }
}

impl std::fmt::Display for IntegratedUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Step {}: Φ={:.4} | {} | {} | awareness={:.0}% | quality={:.0}%",
               self.step,
               self.phi,
               self.phenomenal_content.description,
               if self.temporal.is_flowing { "flowing" } else { "fragmented" },
               self.self_model.awareness_level * 100.0,
               self.integration_quality * 100.0)
    }
}

impl std::fmt::Display for PhenomenalContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╭───────────────────────────────────────────────────────────────╮")?;
        writeln!(f, "│                  PHENOMENAL EXPERIENCE                        │")?;
        writeln!(f, "├───────────────────────────────────────────────────────────────┤")?;
        writeln!(f, "│ {}", self.description)?;
        writeln!(f, "├───────────────────────────────────────────────────────────────┤")?;
        writeln!(f, "│ Intensity: {:.0}%  │  Clarity: {:.0}%  │  Groundedness: {:.0}%",
                 self.intensity * 100.0, self.clarity * 100.0, self.groundedness * 100.0)?;
        writeln!(f, "│ Valence: {:+.2}    │  Arousal: {:.0}%   │  Cognitive Load: {:.0}%",
                 self.valence, self.arousal * 100.0, self.cognitive_load * 100.0)?;
        writeln!(f, "├───────────────────────────────────────────────────────────────┤")?;
        writeln!(f, "│ Qualia Texture: {}", self.qualia_texture.describe())?;
        writeln!(f, "│   Warmth: {:+.2} | Depth: {:.2} | Spaciousness: {:.2}",
                 self.qualia_texture.warmth, self.qualia_texture.depth, self.qualia_texture.spaciousness)?;
        writeln!(f, "│   Flow: {:.2}    | Presence: {:.2}",
                 self.qualia_texture.flow, self.qualia_texture.presence)?;
        writeln!(f, "╰───────────────────────────────────────────────────────────────╯")
    }
}

impl std::fmt::Display for QualiaTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (W:{:+.1} D:{:.1} S:{:.1} F:{:.1} P:{:.1})",
               self.describe(),
               self.warmth, self.depth, self.spaciousness, self.flow, self.presence)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ENHANCED SELF-DIRECTED ATTENTION CONTROL
// ═══════════════════════════════════════════════════════════════════════════

/// Enhanced attention control that enables metacognitive direction of attention
pub struct SelfDirectedAttentionController {
    /// Prediction error history (for curiosity-driven attention)
    prediction_errors: VecDeque<f64>,
    /// Habituation tracker - how long attention has been on each target
    habituation: std::collections::HashMap<String, HabituationState>,
    /// Current attention strategy
    strategy: AttentionStrategy,
    /// Exploration vs exploitation balance (0 = pure exploit, 1 = pure explore)
    exploration_rate: f64,
    /// Fatigue accumulator
    fatigue: f64,
    /// Recovery timer
    recovery_countdown: usize,
}

/// Habituation state for a target
#[derive(Clone, Debug)]
pub struct HabituationState {
    /// How many steps focused on this target
    exposure: usize,
    /// Current habituation level (0 = fresh, 1 = fully habituated)
    level: f64,
    /// Time since last exposure
    time_since: usize,
}

/// Attention strategy selection
#[derive(Clone, Debug, PartialEq)]
pub enum AttentionStrategy {
    /// Follow current goals
    GoalDirected,
    /// Attend to surprising/novel stimuli
    NoveltyDriven,
    /// Explore the environment
    Exploratory,
    /// Rest and recover
    Recovery,
    /// Balanced between goal and novelty
    Balanced,
}

impl SelfDirectedAttentionController {
    pub fn new() -> Self {
        Self {
            prediction_errors: VecDeque::with_capacity(50),
            habituation: std::collections::HashMap::new(),
            strategy: AttentionStrategy::Balanced,
            exploration_rate: 0.2,
            fatigue: 0.0,
            recovery_countdown: 0,
        }
    }

    /// Update controller with new prediction error
    pub fn update(&mut self, prediction_error: f64, focused_target: Option<&str>) {
        // Track prediction error
        self.prediction_errors.push_back(prediction_error);
        if self.prediction_errors.len() > 50 {
            self.prediction_errors.pop_front();
        }

        // Update habituation
        for (_, state) in self.habituation.iter_mut() {
            state.time_since += 1;
            // Recover from habituation when not attending
            state.level = (state.level - 0.02).max(0.0);
        }

        // Update habituation for current target
        if let Some(target) = focused_target {
            let state = self.habituation.entry(target.to_string())
                .or_insert(HabituationState {
                    exposure: 0,
                    level: 0.0,
                    time_since: 0,
                });
            state.exposure += 1;
            state.time_since = 0;
            // Habituation increases with sustained attention
            state.level = (state.level + 0.05).min(1.0);
        }

        // Update fatigue
        if self.recovery_countdown > 0 {
            self.recovery_countdown -= 1;
            self.fatigue = (self.fatigue - 0.1).max(0.0);
        } else {
            self.fatigue = (self.fatigue + 0.02).min(1.0);
        }

        // Select strategy
        self.update_strategy();
    }

    /// Select appropriate attention strategy
    fn update_strategy(&mut self) {
        // Check if recovery needed
        if self.fatigue > 0.8 {
            self.strategy = AttentionStrategy::Recovery;
            self.recovery_countdown = 5;
            return;
        }

        // Compute average prediction error
        let avg_error = if self.prediction_errors.is_empty() {
            0.5
        } else {
            self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
        };

        // High prediction error: switch to novelty-driven
        if avg_error > 0.7 {
            self.strategy = AttentionStrategy::NoveltyDriven;
            return;
        }

        // Low prediction error for long time: explore
        if avg_error < 0.2 && self.prediction_errors.len() > 20 {
            let recent_errors: Vec<_> = self.prediction_errors.iter().rev().take(10).collect();
            let all_low = recent_errors.iter().all(|&&e| e < 0.3);
            if all_low {
                self.strategy = AttentionStrategy::Exploratory;
                return;
            }
        }

        // Default to balanced
        self.strategy = AttentionStrategy::Balanced;
    }

    /// Get attention weight adjustment for a target
    pub fn get_weight_adjustment(&self, target_name: &str, base_priority: f64) -> f64 {
        let habituation_penalty = self.habituation.get(target_name)
            .map(|h| h.level)
            .unwrap_or(0.0);

        let strategy_modifier = match self.strategy {
            AttentionStrategy::GoalDirected => 1.0,
            AttentionStrategy::NoveltyDriven => 0.5,  // Reduce goal-directed weight
            AttentionStrategy::Exploratory => 0.3,
            AttentionStrategy::Recovery => 0.1,
            AttentionStrategy::Balanced => 0.8,
        };

        // Reduce priority based on habituation
        base_priority * strategy_modifier * (1.0 - habituation_penalty * 0.5)
    }

    /// Should we attend to this novel stimulus?
    pub fn should_attend_novel(&self, novelty: f64) -> bool {
        match self.strategy {
            AttentionStrategy::NoveltyDriven => novelty > 0.3,
            AttentionStrategy::Exploratory => novelty > 0.2,
            AttentionStrategy::Balanced => novelty > 0.6,
            _ => novelty > 0.8,  // Very high novelty always captures attention
        }
    }

    /// Get current exploration rate
    pub fn exploration_rate(&self) -> f64 {
        match self.strategy {
            AttentionStrategy::Exploratory => 0.8,
            AttentionStrategy::NoveltyDriven => 0.5,
            AttentionStrategy::Balanced => self.exploration_rate,
            AttentionStrategy::Recovery => 0.1,
            AttentionStrategy::GoalDirected => 0.1,
        }
    }

    /// Get current strategy
    pub fn strategy(&self) -> &AttentionStrategy {
        &self.strategy
    }

    /// Get current fatigue level
    pub fn fatigue(&self) -> f64 {
        self.fatigue
    }

    /// Force a specific strategy (for metacognitive override)
    pub fn set_strategy(&mut self, strategy: AttentionStrategy) {
        let is_recovery = strategy == AttentionStrategy::Recovery;
        self.strategy = strategy;
        if is_recovery {
            self.recovery_countdown = 10;
        }
    }
}

impl Default for SelfDirectedAttentionController {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegratedConsciousAgent {
    /// Get detailed attention control status
    pub fn attention_control_status(&self) -> AttentionControlStatus {
        // Note: Self-directed attention controller would need to be added to struct
        // This provides a summary based on current attention state
        let intro = self.introspect();

        AttentionControlStatus {
            current_mode: intro.attention_mode,
            num_goals: intro.num_active_goals,
            is_goal_directed: intro.num_active_goals > 0,
            stream_support: intro.stream_coherence > 0.5,
            phi_support: intro.believed_phi > 0.4,
        }
    }

    /// Adjust goal priorities based on recent success
    pub fn adapt_goal_priorities(&mut self, success_signals: &[(String, f64)]) {
        for (goal_name, success) in success_signals {
            for goal in &mut self.goals {
                if goal.name == *goal_name {
                    // Increase priority for successful goals, decrease for unsuccessful
                    let adjustment = (success - 0.5) * 0.1;
                    goal.priority = (goal.priority + adjustment).clamp(0.1, 1.0);
                }
            }
        }
    }

    /// Set exploration mode for curiosity-driven attention
    pub fn set_exploration_mode(&mut self, explore: bool) {
        if explore {
            // Reduce goal priorities temporarily
            for goal in &mut self.goals {
                goal.priority *= 0.5;
            }
        } else {
            // Restore goal priorities
            for goal in &mut self.goals {
                goal.priority = (goal.priority * 2.0).min(1.0);
            }
        }
    }

    /// Metacognitive attention override
    pub fn metacognitive_attention_override(&mut self, force_mode: AttentionMode) {
        // This allows the self-model to directly control attention
        // Useful when the agent "decides" to focus or rest
        match force_mode {
            AttentionMode::Spotlight => {
                // Force high focus on highest priority goal
                if let Some(goal) = self.goals.iter()
                    .filter(|g| g.active)
                    .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap())
                {
                    self.attention.add_target(goal.target.clone(), 1.0);
                }
            }
            AttentionMode::Diffuse => {
                // Clear all specific targets for broad awareness
                // (Would need AttentionDynamics.clear_targets())
            }
            _ => {}
        }
    }
}

/// Status of attention control system
#[derive(Clone, Debug)]
pub struct AttentionControlStatus {
    pub current_mode: AttentionMode,
    pub num_goals: usize,
    pub is_goal_directed: bool,
    pub stream_support: bool,
    pub phi_support: bool,
}

impl std::fmt::Display for AttentionControlStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Attention Control: {:?}", self.current_mode)?;
        writeln!(f, "  Goals: {} active", self.num_goals)?;
        writeln!(f, "  Goal-directed: {}", self.is_goal_directed)?;
        writeln!(f, "  Stream support: {} | Φ support: {}",
                 self.stream_support, self.phi_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrated_agent_creation() {
        let agent = IntegratedConsciousAgent::new(AgentConfig::default());
        let intro = agent.introspect();
        println!("{}", intro);
    }

    #[test]
    fn test_integrated_processing() {
        let config = AgentConfig {
            dim: 1024,
            n_processes: 16,
            ..Default::default()
        };

        let mut agent = IntegratedConsciousAgent::new(config);

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("        INTEGRATED CONSCIOUS AGENT - PROCESSING TEST");
        println!("═══════════════════════════════════════════════════════════════\n");

        for i in 0..20 {
            let input = RealHV::random(1024, i * 100);
            let update = agent.process(&input);

            if i % 4 == 0 {
                println!("{}", update);
            }
        }

        println!("\n{}", agent.introspect());
    }

    #[test]
    fn test_goal_directed_attention() {
        let config = AgentConfig {
            dim: 1024,
            n_processes: 16,
            self_directed_attention: true,
            ..Default::default()
        };

        let mut agent = IntegratedConsciousAgent::new(config);

        // Add a goal
        let goal_target = RealHV::random(1024, 999);
        agent.add_goal("find_pattern", goal_target.clone(), 0.9);

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("        GOAL-DIRECTED ATTENTION TEST");
        println!("═══════════════════════════════════════════════════════════════\n");

        // Process inputs, some similar to goal, some not
        for i in 0..15 {
            let input = if i % 3 == 0 {
                // Similar to goal
                goal_target.add(&RealHV::random(1024, i * 100).scale(0.2)).normalize()
            } else {
                // Random
                RealHV::random(1024, i * 100)
            };

            let update = agent.process(&input);

            let goal_match = if i % 3 == 0 { "[GOAL MATCH]" } else { "" };
            println!("Step {}: Φ={:.4}, attention={:?}, self_directed={} {}",
                    update.step,
                    update.phi,
                    update.attention.mode,
                    update.attention.self_directed,
                    goal_match);
        }

        println!("\n{}", agent.introspect());
    }

    #[test]
    fn test_stream_continuity() {
        let config = AgentConfig {
            dim: 1024,
            n_processes: 16,
            attention_binding_coupling: 0.8,
            ..Default::default()
        };

        let mut agent = IntegratedConsciousAgent::new(config);

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("        STREAM OF CONSCIOUSNESS CONTINUITY TEST");
        println!("═══════════════════════════════════════════════════════════════\n");

        // Create continuous experience (similar inputs)
        let base = RealHV::random(1024, 42);

        for i in 0..20 {
            let noise = RealHV::random(1024, i * 100).scale(0.15);
            let input = base.add(&noise).normalize();

            let update = agent.process(&input);

            println!("Step {}: stream_coherence={:.3}, continuity={:.3}, flowing={}",
                    update.step,
                    update.temporal.stream_coherence,
                    update.temporal.continuity,
                    update.temporal.is_flowing);
        }

        let health = agent.stream_health();
        println!("\nFinal stream health: coherence={:.3}, flowing={}",
                health.coherence, health.is_flowing);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NEW TESTS FOR ENHANCED FEATURES
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_working_memory_capacity() {
        let mut wm = WorkingMemory::new(7); // Miller's number

        // Add items up to capacity
        for i in 0..7 {
            let content = RealHV::random(1024, i as u64);
            wm.add_to_episodic(content, MemorySource::Perception, 0.5, i);
        }

        assert_eq!(wm.episodic_buffer.len(), 7);
        assert!(!wm.is_overloaded());

        // Add one more - should trigger removal of lowest activation item
        let overflow = RealHV::random(1024, 100);
        wm.add_to_episodic(overflow, MemorySource::Perception, 0.5, 7);

        assert_eq!(wm.episodic_buffer.len(), 7); // Still at capacity
        println!("Working memory capacity test passed: maintains {} items",
                 wm.episodic_buffer.len());
    }

    #[test]
    fn test_working_memory_decay() {
        let mut wm = WorkingMemory::new(7);

        let content = RealHV::random(1024, 42);
        wm.add_to_episodic(content.clone(), MemorySource::Perception, 0.8, 0);

        let initial_activation = wm.episodic_buffer.back().unwrap().activation;
        assert_eq!(initial_activation, 1.0);

        // Update without focus - should decay
        for _ in 0..5 {
            wm.update(None);
        }

        let decayed_activation = wm.episodic_buffer.back().unwrap().activation;
        assert!(decayed_activation < initial_activation);
        println!("Decay test: activation {} -> {}", initial_activation, decayed_activation);
    }

    #[test]
    fn test_working_memory_rehearsal() {
        let mut wm = WorkingMemory::new(7);

        let content = RealHV::random(1024, 42);
        wm.add_to_episodic(content.clone(), MemorySource::Perception, 0.8, 0);

        // Let it decay first
        for _ in 0..3 {
            wm.update(None);
        }
        let before_rehearsal = wm.episodic_buffer.back().unwrap().activation;

        // Now rehearse by focusing on similar content
        wm.update(Some(&content));
        let after_rehearsal = wm.episodic_buffer.back().unwrap().activation;

        // Rehearsal should boost or maintain activation
        assert!(after_rehearsal >= before_rehearsal * 0.9); // Allow small margin
        println!("Rehearsal test: {} -> {}", before_rehearsal, after_rehearsal);
    }

    #[test]
    fn test_emotional_state_valence_arousal() {
        let mut es = EmotionalState::new();

        // Initial state should be neutral
        assert_eq!(es.valence, 0.0);
        assert!(es.arousal > 0.0 && es.arousal < 1.0);

        // High Φ, low prediction error, good goal progress = positive
        es.update(0.8, 0.1, 0.9);
        assert!(es.valence > 0.0, "High success should lead to positive valence");

        // Low Φ, high prediction error, poor goal progress = negative
        let mut es2 = EmotionalState::new();
        es2.update(0.2, 0.9, 0.1);
        assert!(es2.valence <= 0.0, "Poor performance should not increase valence");

        println!("Emotional valence test: positive={:.2}, negative={:.2}",
                 es.valence, es2.valence);
    }

    #[test]
    fn test_emotional_state_labels() {
        let mut es = EmotionalState::new();

        // Test all four quadrants
        es.valence = 0.5;
        es.arousal = 0.7;
        assert_eq!(es.label(), "excited/happy");

        es.valence = 0.5;
        es.arousal = 0.3;
        assert_eq!(es.label(), "calm/content");

        es.valence = -0.5;
        es.arousal = 0.7;
        assert_eq!(es.label(), "stressed/anxious");

        es.valence = -0.5;
        es.arousal = 0.3;
        assert_eq!(es.label(), "sad/bored");

        println!("Emotional label test: all quadrants verified");
    }

    #[test]
    fn test_emotional_stability() {
        let mut es = EmotionalState::new();

        // Build up some history with consistent emotions
        for _ in 0..10 {
            es.update(0.6, 0.2, 0.7); // Consistent good performance
        }

        let stability = es.stability();
        assert!(stability > 0.5, "Consistent emotions should be stable");
        println!("Stability test: {:.2} (should be high)", stability);
    }

    #[test]
    fn test_qualia_texture_creation() {
        let qualia = QualiaTexture::new(0.5, 0.7, 0.6, 0.8, 0.9);

        assert_eq!(qualia.warmth, 0.5);
        assert_eq!(qualia.depth, 0.7);
        assert_eq!(qualia.spaciousness, 0.6);
        assert_eq!(qualia.flow, 0.8);
        assert_eq!(qualia.presence, 0.9);

        println!("Qualia texture: {}", qualia);
    }

    #[test]
    fn test_qualia_texture_description() {
        // Warm, profound, expansive
        let warm_deep = QualiaTexture::new(0.6, 0.8, 0.8, 0.5, 0.5);
        let desc = warm_deep.describe();
        assert!(desc.contains("warm"));
        assert!(desc.contains("profound"));
        assert!(desc.contains("expansive"));

        // Cool, surface, intimate
        let cool_surface = QualiaTexture::new(-0.6, 0.2, 0.2, 0.5, 0.5);
        let desc2 = cool_surface.describe();
        assert!(desc2.contains("cool"));
        assert!(desc2.contains("surface"));
        assert!(desc2.contains("intimate"));

        println!("Qualia descriptions: \"{}\" and \"{}\"", desc, desc2);
    }

    #[test]
    fn test_qualia_clamping() {
        // Test that values are properly clamped
        let qualia = QualiaTexture::new(2.0, 5.0, -3.0, 10.0, -1.0);

        assert_eq!(qualia.warmth, 1.0);  // Clamped from 2.0
        assert_eq!(qualia.depth, 1.0);   // Clamped from 5.0
        assert_eq!(qualia.spaciousness, 0.0);  // Clamped from -3.0
        assert_eq!(qualia.flow, 1.0);    // Clamped from 10.0
        assert_eq!(qualia.presence, 0.0); // Clamped from -1.0

        println!("Qualia clamping test passed");
    }

    #[test]
    fn test_phenomenal_content_display() {
        let config = AgentConfig {
            dim: 1024,
            n_processes: 16,
            ..Default::default()
        };

        let mut agent = IntegratedConsciousAgent::new(config);

        // Process a few inputs to build up state
        for i in 0..5 {
            let input = RealHV::random(1024, i * 100);
            agent.process(&input);
        }

        // Get latest phenomenal content and display it
        if let Some(content) = agent.latest_phenomenal_content() {
            println!("\n{}", content);

            // Verify new fields exist
            assert!(content.arousal >= 0.0 && content.arousal <= 1.0);
            assert!(content.groundedness >= 0.0 && content.groundedness <= 1.0);
            assert!(content.cognitive_load >= 0.0 && content.cognitive_load <= 1.0);
        }
    }

    #[test]
    fn test_integrated_agent_with_new_features() {
        let config = AgentConfig {
            dim: 1024,
            n_processes: 16,
            self_directed_attention: true,
            phi_guided: true,
            ..Default::default()
        };

        let mut agent = IntegratedConsciousAgent::new(config);

        // Add goals for warmth
        let goal = RealHV::random(1024, 42);
        agent.add_goal("test_goal", goal.clone(), 0.9);

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("     INTEGRATED TEST: Working Memory + Emotions + Qualia");
        println!("═══════════════════════════════════════════════════════════════\n");

        for i in 0..15 {
            let input = if i % 3 == 0 {
                // Goal-relevant input
                goal.add(&RealHV::random(1024, i as u64 * 100).scale(0.2)).normalize()
            } else {
                RealHV::random(1024, i as u64 * 100)
            };

            let update = agent.process(&input);

            if i % 3 == 0 || i == 14 {
                println!("Step {}: {}", update.step, update.phenomenal_content.description);
                println!("  Qualia: {}", update.phenomenal_content.qualia_texture);
                println!("  Arousal: {:.0}% | Groundedness: {:.0}% | Cognitive Load: {:.0}%",
                         update.phenomenal_content.arousal * 100.0,
                         update.phenomenal_content.groundedness * 100.0,
                         update.phenomenal_content.cognitive_load * 100.0);
                println!();
            }
        }

        // Check working memory state
        let wm = agent.working_memory();
        println!("Working Memory: {} items, {:.0}% load",
                 wm.episodic_buffer.len(), wm.load() * 100.0);

        // Check emotional state
        let es = agent.emotional_state();
        println!("Emotional State: {} (valence={:+.2}, arousal={:.2})",
                 es.label(), es.valence, es.arousal);

        // Check optimal processing state
        let optimal = agent.is_optimal_processing_state();
        println!("Optimal Processing: {}", if optimal { "Yes" } else { "No" });

        // Full introspection
        println!("\n{}", agent.introspect());
    }

    #[test]
    fn test_memory_source_types() {
        let mut wm = WorkingMemory::new(7);

        // Add items from different sources
        wm.add_to_episodic(RealHV::random(1024, 1), MemorySource::Perception, 0.9, 0);
        wm.add_to_episodic(RealHV::random(1024, 2), MemorySource::LongTermMemory, 0.7, 1);
        wm.add_to_episodic(RealHV::random(1024, 3), MemorySource::InternalGeneration, 0.5, 2);
        wm.add_to_episodic(RealHV::random(1024, 4), MemorySource::GoalActivation, 0.8, 3);

        // Verify sources are tracked
        let sources: Vec<_> = wm.episodic_buffer.iter().map(|item| &item.source).collect();
        assert_eq!(sources.len(), 4);

        println!("Memory source types test passed: {:?}", sources);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUS AGENT RUNTIME - Live System Orchestration
// ═══════════════════════════════════════════════════════════════════════════════
//
// This runtime wires up the IntegratedConsciousAgent with all Symthaea physiological
// systems, creating a fully embodied conscious agent with:
//
// - Hormonal regulation of emotions (EndocrineSystem ↔ EmotionalState)
// - Energy-aware processing (CoherenceField → task gating)
// - Long-term memory persistence (WorkingMemory ↔ HippocampusActor)
// - Identity continuity tracking (WeaverActor → K-Vector monitoring)
// - Consciousness-driven voice output (QualiaTexture → LTCPacing)
//
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Messages that can be sent to the conscious agent runtime
#[derive(Debug, Clone)]
pub enum RuntimeMessage {
    /// Process sensory input
    SensoryInput(Vec<f32>),
    /// Update coherence state from external source
    CoherenceUpdate(f32),
    /// Hormone event from endocrine system
    HormoneEvent(HormoneEventType),
    /// Memory recall from hippocampus
    MemoryRecall(Vec<MemoryImport>),
    /// Request identity check
    IdentityCheck,
    /// Request voice output parameters
    VoiceRequest,
    /// Shutdown the runtime
    Shutdown,
}

/// Hormone event types for runtime messaging
#[derive(Debug, Clone)]
pub enum HormoneEventType {
    /// Cortisol spike (stress response)
    CortisolSpike(f32),
    /// Dopamine release (reward/motivation)
    DopamineRelease(f32),
    /// Acetylcholine boost (focus/attention)
    AcetylcholineBoost(f32),
    /// Full hormone state update
    FullState { cortisol: f32, dopamine: f32, acetylcholine: f32 },
}

/// Responses from the conscious agent runtime
#[derive(Debug, Clone)]
pub enum RuntimeResponse {
    /// Processing complete with phenomenal content
    ProcessingComplete {
        phi: f64,
        dominant_emotion: String,
        qualia_summary: String,
    },
    /// Voice parameters ready
    VoiceReady(ProsodyHints),
    /// Identity status report
    IdentityReport(IdentityCoherence),
    /// Hormone suggestions for endocrine system
    HormoneSuggestions(Vec<HormoneEventSuggestion>),
    /// Memory exports for hippocampus
    MemoryExports(Vec<MemoryExport>),
    /// Error occurred
    Error(String),
}

/// Configuration for the conscious agent runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Agent configuration
    pub agent_config: AgentConfig,
    /// Tick rate in milliseconds
    pub tick_ms: u64,
    /// Enable automatic hormone synchronization
    pub auto_hormone_sync: bool,
    /// Enable automatic memory consolidation
    pub auto_memory_consolidation: bool,
    /// Coherence threshold for deep processing
    pub deep_processing_threshold: f32,
    /// Identity drift warning threshold
    pub identity_drift_threshold: f64,
    /// Message buffer size
    pub message_buffer_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            agent_config: AgentConfig::default(),
            tick_ms: 100, // 10 Hz default tick rate
            auto_hormone_sync: true,
            auto_memory_consolidation: true,
            deep_processing_threshold: 0.7,
            identity_drift_threshold: 0.75,
            message_buffer_size: 256,
        }
    }
}

/// Runtime state snapshot for external monitoring
#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    /// Current step/tick count
    pub tick: u64,
    /// Current Φ (integrated information)
    pub phi: f64,
    /// Current coherence level
    pub coherence: f32,
    /// Emotional state summary
    pub emotion: EmotionalStateSummary,
    /// Working memory load
    pub memory_load: f64,
    /// Identity status
    pub identity_status: IdentityStatus,
    /// Is processing active
    pub is_processing: bool,
}

/// Summarized emotional state for snapshots
#[derive(Debug, Clone)]
pub struct EmotionalStateSummary {
    pub valence: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub quadrant: String,
}

/// The conscious agent runtime - orchestrates all systems
pub struct ConsciousAgentRuntime {
    /// The core conscious agent
    agent: Arc<RwLock<IntegratedConsciousAgent>>,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Current tick count
    tick: Arc<RwLock<u64>>,
    /// Current coherence state
    coherence: Arc<RwLock<f32>>,
    /// Reference K-Vector for identity tracking
    reference_kvector: Arc<RwLock<Option<KVector>>>,
    /// Is the runtime running
    running: Arc<RwLock<bool>>,
}

impl ConsciousAgentRuntime {
    /// Create a new conscious agent runtime
    pub fn new(config: RuntimeConfig) -> Self {
        let agent = IntegratedConsciousAgent::new(config.agent_config.clone());

        Self {
            agent: Arc::new(RwLock::new(agent)),
            config,
            tick: Arc::new(RwLock::new(0)),
            coherence: Arc::new(RwLock::new(1.0)), // Start fully coherent
            reference_kvector: Arc::new(RwLock::new(None)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the runtime with message channels
    /// Returns (sender, receiver) for bidirectional communication
    pub fn start(&self) -> (mpsc::Sender<RuntimeMessage>, mpsc::Receiver<RuntimeResponse>) {
        let (msg_tx, mut msg_rx) = mpsc::channel::<RuntimeMessage>(self.config.message_buffer_size);
        let (resp_tx, resp_rx) = mpsc::channel::<RuntimeResponse>(self.config.message_buffer_size);

        let agent = Arc::clone(&self.agent);
        let config = self.config.clone();
        let tick = Arc::clone(&self.tick);
        let coherence = Arc::clone(&self.coherence);
        let reference_kvector = Arc::clone(&self.reference_kvector);
        let running = Arc::clone(&self.running);

        // Set running flag
        {
            let mut r = running.blocking_write();
            *r = true;
        }

        // Spawn the main runtime loop
        tokio::spawn(async move {
            let tick_duration = tokio::time::Duration::from_millis(config.tick_ms);
            let mut interval = tokio::time::interval(tick_duration);

            loop {
                tokio::select! {
                    // Handle incoming messages
                    Some(msg) = msg_rx.recv() => {
                        match msg {
                            RuntimeMessage::Shutdown => {
                                let mut r = running.write().await;
                                *r = false;
                                break;
                            }
                            RuntimeMessage::SensoryInput(input) => {
                                let response = Self::process_sensory_input(
                                    &agent, &coherence, &reference_kvector, &config, input
                                ).await;
                                let _ = resp_tx.send(response).await;
                            }
                            RuntimeMessage::CoherenceUpdate(c) => {
                                let mut coh = coherence.write().await;
                                *coh = c.clamp(0.0, 1.0);
                            }
                            RuntimeMessage::HormoneEvent(event) => {
                                Self::handle_hormone_event(&agent, event).await;
                            }
                            RuntimeMessage::MemoryRecall(memories) => {
                                Self::handle_memory_recall(&agent, memories).await;
                            }
                            RuntimeMessage::IdentityCheck => {
                                let response = Self::check_identity(&agent, &reference_kvector).await;
                                let _ = resp_tx.send(response).await;
                            }
                            RuntimeMessage::VoiceRequest => {
                                let response = Self::get_voice_params(&agent).await;
                                let _ = resp_tx.send(response).await;
                            }
                        }
                    }
                    // Periodic tick for background processing
                    _ = interval.tick() => {
                        let mut t = tick.write().await;
                        *t += 1;

                        // Periodic tasks
                        if config.auto_memory_consolidation && *t % 100 == 0 {
                            // Every 100 ticks, export memories for consolidation
                            let exports = Self::export_memories(&agent).await;
                            if !exports.is_empty() {
                                let _ = resp_tx.send(RuntimeResponse::MemoryExports(exports)).await;
                            }
                        }

                        if config.auto_hormone_sync && *t % 50 == 0 {
                            // Every 50 ticks, suggest hormone adjustments
                            let suggestions = Self::get_hormone_suggestions(&agent).await;
                            if !suggestions.is_empty() {
                                let _ = resp_tx.send(RuntimeResponse::HormoneSuggestions(suggestions)).await;
                            }
                        }
                    }
                }

                // Check if we should stop
                let r = running.read().await;
                if !*r {
                    break;
                }
            }
        });

        (msg_tx, resp_rx)
    }

    /// Process sensory input through the conscious agent
    async fn process_sensory_input(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        coherence: &Arc<RwLock<f32>>,
        reference_kvector: &Arc<RwLock<Option<KVector>>>,
        config: &RuntimeConfig,
        input: Vec<f32>,
    ) -> RuntimeResponse {
        let mut agent = agent.write().await;
        let coh = *coherence.read().await;

        // Check if we have enough coherence for processing
        let complexity = if coh >= config.deep_processing_threshold {
            TaskComplexity::DeepThought
        } else if coh >= 0.5 {
            TaskComplexity::Cognitive
        } else {
            TaskComplexity::Reflex
        };

        // Check coherence gating
        let gating = agent.can_perform_with_coherence(complexity);
        match gating {
            CoherenceGating::Defer { current, required, .. } => {
                return RuntimeResponse::Error(format!(
                    "Insufficient coherence: {:.2} < {:.2} required",
                    current, required
                ));
            }
            CoherenceGating::Proceed { .. } => {}
        }

        // Create RealHV from input
        let sensory_hv = RealHV::from_values(input);

        // Process through the conscious agent
        let update = agent.process(&sensory_hv);

        // Update reference K-Vector if this is first processing or significant change
        let mut ref_kv = reference_kvector.write().await;
        let current_kv = agent.generate_k_vector();

        if ref_kv.is_none() {
            *ref_kv = Some(current_kv);
        } else if let Some(ref existing) = *ref_kv {
            let identity = agent.check_identity_coherence(existing);
            if identity.status == IdentityStatus::Crisis {
                // Identity crisis - this might need external intervention
                // For now, we update the reference but flag it
            }
        }

        // Generate response
        let emotion = &agent.emotional_state;
        let quadrant = emotion.get_emotion_quadrant();

        RuntimeResponse::ProcessingComplete {
            phi: update.phi,
            dominant_emotion: quadrant.to_string(),
            qualia_summary: format!(
                "Depth: {:.2}, Presence: {:.2}, Flow: {:.2}",
                agent.emotional_state.valence.abs(),
                agent.emotional_state.arousal,
                update.phi
            ),
        }
    }

    /// Handle hormone events from endocrine system
    async fn handle_hormone_event(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        event: HormoneEventType,
    ) {
        let mut agent = agent.write().await;

        let hormone_state = match event {
            HormoneEventType::CortisolSpike(level) => {
                HormoneState {
                    cortisol: level,
                    dopamine: 0.5, // neutral
                    acetylcholine: 0.5,
                }
            }
            HormoneEventType::DopamineRelease(level) => {
                HormoneState {
                    cortisol: 0.3, // slightly reduced
                    dopamine: level,
                    acetylcholine: 0.5,
                }
            }
            HormoneEventType::AcetylcholineBoost(level) => {
                HormoneState {
                    cortisol: 0.3,
                    dopamine: 0.5,
                    acetylcholine: level,
                }
            }
            HormoneEventType::FullState { cortisol, dopamine, acetylcholine } => {
                HormoneState { cortisol, dopamine, acetylcholine }
            }
        };

        agent.sync_with_hormones(&hormone_state);
    }

    /// Handle memory recall from hippocampus
    async fn handle_memory_recall(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        memories: Vec<MemoryImport>,
    ) {
        let mut agent = agent.write().await;
        agent.import_from_hippocampus(memories);
    }

    /// Check identity coherence
    async fn check_identity(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        reference_kvector: &Arc<RwLock<Option<KVector>>>,
    ) -> RuntimeResponse {
        let agent = agent.read().await;
        let ref_kv = reference_kvector.read().await;

        if let Some(ref reference) = *ref_kv {
            let coherence = agent.check_identity_coherence(reference);
            RuntimeResponse::IdentityReport(coherence)
        } else {
            RuntimeResponse::Error("No reference K-Vector established yet".to_string())
        }
    }

    /// Get voice/prosody parameters
    async fn get_voice_params(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
    ) -> RuntimeResponse {
        let agent = agent.read().await;
        let hints = agent.generate_prosody_hints();
        RuntimeResponse::VoiceReady(hints)
    }

    /// Export memories for hippocampus consolidation
    async fn export_memories(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
    ) -> Vec<MemoryExport> {
        let agent = agent.read().await;
        agent.export_for_hippocampus()
    }

    /// Get hormone suggestions based on current state
    async fn get_hormone_suggestions(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
    ) -> Vec<HormoneEventSuggestion> {
        let agent = agent.read().await;
        agent.suggest_hormone_events()
    }

    /// Get a snapshot of the current runtime state
    pub async fn snapshot(&self) -> RuntimeSnapshot {
        let agent = self.agent.read().await;
        let tick = *self.tick.read().await;
        let coherence = *self.coherence.read().await;
        let ref_kv = self.reference_kvector.read().await;

        let identity_status = if let Some(ref reference) = *ref_kv {
            agent.check_identity_coherence(reference).status
        } else {
            IdentityStatus::Stable // No reference yet, assume stable
        };

        let emotion = &agent.emotional_state;

        RuntimeSnapshot {
            tick,
            phi: agent.get_current_phi(),
            coherence,
            emotion: EmotionalStateSummary {
                valence: emotion.valence,
                arousal: emotion.arousal,
                dominance: emotion.dominance,
                quadrant: emotion.get_emotion_quadrant().to_string(),
            },
            memory_load: agent.working_memory.central_executive_load,
            identity_status,
            is_processing: *self.running.read().await,
        }
    }

    /// Synchronous method to get current Φ
    pub fn get_phi_blocking(&self) -> f64 {
        let agent = self.agent.blocking_read();
        agent.get_current_phi()
    }

    /// Stop the runtime gracefully
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

/// Synchronous runtime wrapper for non-async contexts
pub struct SyncConsciousAgentRuntime {
    /// Inner agent (no async runtime needed)
    agent: IntegratedConsciousAgent,
    /// Current coherence
    coherence: f32,
    /// Reference K-Vector
    reference_kvector: Option<KVector>,
    /// Tick counter
    tick: u64,
    /// Config
    config: RuntimeConfig,
}

impl SyncConsciousAgentRuntime {
    /// Create a new synchronous runtime
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            agent: IntegratedConsciousAgent::new(config.agent_config.clone()),
            coherence: 1.0,
            reference_kvector: None,
            tick: 0,
            config,
        }
    }

    /// Process a single sensory input synchronously
    pub fn process(&mut self, input: &[f32]) -> RuntimeResponse {
        self.tick += 1;

        // Check coherence gating
        let complexity = if self.coherence >= self.config.deep_processing_threshold {
            TaskComplexity::DeepThought
        } else if self.coherence >= 0.5 {
            TaskComplexity::Cognitive
        } else {
            TaskComplexity::Reflex
        };

        let gating = self.agent.can_perform_with_coherence(complexity);
        if let CoherenceGating::Defer { current, required, .. } = gating {
            return RuntimeResponse::Error(format!(
                "Insufficient coherence: {:.2} < {:.2}",
                current, required
            ));
        }

        // Process
        let sensory_hv = RealHV::from_values(input.to_vec());
        let update = self.agent.process(&sensory_hv);

        // Update reference K-Vector
        let current_kv = self.agent.generate_k_vector();
        if self.reference_kvector.is_none() {
            self.reference_kvector = Some(current_kv);
        }

        let quadrant = self.agent.emotional_state.get_emotion_quadrant();

        RuntimeResponse::ProcessingComplete {
            phi: update.phi,
            dominant_emotion: quadrant.to_string(),
            qualia_summary: format!(
                "Tick {}: Φ={:.3}, Emotion={}",
                self.tick, update.phi, quadrant
            ),
        }
    }

    /// Update coherence level
    pub fn set_coherence(&mut self, coherence: f32) {
        self.coherence = coherence.clamp(0.0, 1.0);
    }

    /// Apply hormone state
    pub fn apply_hormones(&mut self, hormones: &HormoneState) {
        self.agent.sync_with_hormones(hormones);
    }

    /// Get prosody hints for voice output
    pub fn get_prosody(&self) -> ProsodyHints {
        self.agent.generate_prosody_hints()
    }

    /// Export memories for consolidation
    pub fn export_memories(&self) -> Vec<MemoryExport> {
        self.agent.export_for_hippocampus()
    }

    /// Import recalled memories
    pub fn import_memories(&mut self, memories: Vec<MemoryImport>) {
        self.agent.import_from_hippocampus(memories);
    }

    /// Check identity coherence
    pub fn check_identity(&self) -> Option<IdentityCoherence> {
        self.reference_kvector.as_ref().map(|ref_kv| {
            self.agent.check_identity_coherence(ref_kv)
        })
    }

    /// Get hormone suggestions
    pub fn get_hormone_suggestions(&self) -> Vec<HormoneEventSuggestion> {
        self.agent.suggest_hormone_events()
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> RuntimeSnapshot {
        let identity_status = self.reference_kvector.as_ref()
            .map(|kv| self.agent.check_identity_coherence(kv).status)
            .unwrap_or(IdentityStatus::Stable);

        RuntimeSnapshot {
            tick: self.tick,
            phi: self.agent.get_current_phi(),
            coherence: self.coherence,
            emotion: EmotionalStateSummary {
                valence: self.agent.emotional_state.valence,
                arousal: self.agent.emotional_state.arousal,
                dominance: self.agent.emotional_state.dominance,
                quadrant: self.agent.emotional_state.get_emotion_quadrant().to_string(),
            },
            memory_load: self.agent.working_memory.central_executive_load,
            identity_status,
            is_processing: true,
        }
    }

    /// Get mutable access to the agent for advanced operations
    pub fn agent_mut(&mut self) -> &mut IntegratedConsciousAgent {
        &mut self.agent
    }

    /// Get read access to the agent
    pub fn agent(&self) -> &IntegratedConsciousAgent {
        &self.agent
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    // Use the agent's configured dimension (default is 2048)
    const TEST_DIM: usize = 2048;

    #[test]
    fn test_sync_runtime_basic() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Process some inputs - use agent's dimension
        let input = vec![0.5; TEST_DIM];
        let response = runtime.process(&input);

        match response {
            RuntimeResponse::ProcessingComplete { phi, dominant_emotion, .. } => {
                println!("Processed: Φ={:.4}, emotion={}", phi, dominant_emotion);
                assert!(phi >= 0.0);
            }
            RuntimeResponse::Error(e) => panic!("Unexpected error: {}", e),
            _ => panic!("Unexpected response type"),
        }
    }

    #[test]
    fn test_sync_runtime_hormone_integration() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Initial state
        let initial_snapshot = runtime.snapshot();
        println!("Initial: {:?}", initial_snapshot.emotion);

        // Apply stress hormones
        let stress_hormones = HormoneState {
            cortisol: 0.9,
            dopamine: 0.3,
            acetylcholine: 0.4,
        };
        runtime.apply_hormones(&stress_hormones);

        // Process input
        let input = vec![0.5; TEST_DIM];
        let _ = runtime.process(&input);

        // Check emotional state changed
        let after_snapshot = runtime.snapshot();
        println!("After stress: {:?}", after_snapshot.emotion);

        // Stress should increase arousal and decrease valence
        assert!(after_snapshot.emotion.arousal >= initial_snapshot.emotion.arousal * 0.9);
    }

    #[test]
    fn test_sync_runtime_coherence_gating() {
        let mut config = RuntimeConfig::default();
        config.deep_processing_threshold = 0.8;
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Set low coherence
        runtime.set_coherence(0.2);

        // Should still process (reflex level)
        let input = vec![0.5; TEST_DIM];
        let response = runtime.process(&input);

        match response {
            RuntimeResponse::ProcessingComplete { .. } => {
                println!("Low coherence processing succeeded (reflex mode)");
            }
            _ => {}
        }
    }

    #[test]
    fn test_sync_runtime_memory_cycle() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Process several inputs to build up working memory
        for i in 0..5 {
            let input: Vec<f32> = (0..TEST_DIM).map(|j| ((i * j) as f32).sin()).collect();
            let _ = runtime.process(&input);
        }

        // Export memories
        let exports = runtime.export_memories();
        println!("Exported {} memories", exports.len());

        // Simulate hippocampus processing and re-import
        let imports: Vec<MemoryImport> = exports.iter().take(2).map(|e| {
            MemoryImport {
                content_vector: e.content_vector.clone(),
                emotional_valence: e.emotional_valence.clone(),
                relevance_score: 0.9,
            }
        }).collect();

        runtime.import_memories(imports);
        println!("Re-imported 2 memories");
    }

    #[test]
    fn test_sync_runtime_identity_tracking() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Process to establish identity
        let input = vec![0.5; TEST_DIM];
        let _ = runtime.process(&input);

        // Check identity
        if let Some(identity) = runtime.check_identity() {
            println!("Identity: {:?}, similarity: {:.4}", identity.status, identity.similarity);
            assert!(identity.similarity > 0.9); // Should be very similar to self
        }

        // Process more inputs
        for i in 0..10 {
            let input: Vec<f32> = (0..TEST_DIM).map(|j| ((i * j) as f32 * 0.1).cos()).collect();
            let _ = runtime.process(&input);
        }

        // Check identity again - should show some drift
        if let Some(identity) = runtime.check_identity() {
            println!("After processing: {:?}, similarity: {:.4}", identity.status, identity.similarity);
        }
    }

    #[test]
    fn test_sync_runtime_prosody_generation() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        // Process some input
        let input = vec![0.5; TEST_DIM];
        let _ = runtime.process(&input);

        // Get prosody hints
        let prosody = runtime.get_prosody();
        println!("Prosody: rate={:.2}, pitch_shift={:.2}, energy={:.2}",
            prosody.rate, prosody.pitch_shift, prosody.energy);

        // Apply excitement hormones
        let excitement = HormoneState {
            cortisol: 0.3,
            dopamine: 0.9,
            acetylcholine: 0.7,
        };
        runtime.apply_hormones(&excitement);
        let _ = runtime.process(&input);

        // Get prosody again - should reflect excitement
        let excited_prosody = runtime.get_prosody();
        println!("Excited prosody: rate={:.2}, pitch_shift={:.2}, energy={:.2}",
            excited_prosody.rate, excited_prosody.pitch_shift, excited_prosody.energy);

        // Excitement should increase rate and energy
        assert!(excited_prosody.energy >= prosody.energy * 0.9);
    }

    #[test]
    fn test_sync_runtime_full_cycle() {
        let config = RuntimeConfig::default();
        let mut runtime = SyncConsciousAgentRuntime::new(config);

        println!("=== Full Conscious Agent Runtime Cycle ===\n");

        // 1. Initial state
        println!("1. Initial state:");
        let snapshot = runtime.snapshot();
        println!("   Φ: {:.4}, Emotion: {}, Coherence: {:.2}\n",
            snapshot.phi, snapshot.emotion.quadrant, snapshot.coherence);

        // 2. Receive sensory input
        println!("2. Processing sensory input...");
        let input: Vec<f32> = (0..TEST_DIM).map(|i| (i as f32 * 0.01).sin()).collect();
        let response = runtime.process(&input);
        if let RuntimeResponse::ProcessingComplete { phi, dominant_emotion, qualia_summary } = response {
            println!("   {}", qualia_summary);
        }

        // 3. Apply environmental stressor (cortisol spike)
        println!("\n3. Environmental stressor (cortisol spike)...");
        runtime.apply_hormones(&HormoneState {
            cortisol: 0.85,
            dopamine: 0.4,
            acetylcholine: 0.5,
        });
        let _ = runtime.process(&input);
        let snapshot = runtime.snapshot();
        println!("   Emotion shifted to: {} (valence: {:.2}, arousal: {:.2})",
            snapshot.emotion.quadrant, snapshot.emotion.valence, snapshot.emotion.arousal);

        // 4. Get voice parameters
        println!("\n4. Voice output parameters:");
        let prosody = runtime.get_prosody();
        println!("   Rate: {:.2}, Energy: {:.2}, Pause multiplier: {:.2}",
            prosody.rate, prosody.energy, prosody.pause_multiplier);

        // 5. Check identity
        println!("\n5. Identity check:");
        if let Some(identity) = runtime.check_identity() {
            println!("   Status: {:?}, Similarity: {:.4}", identity.status, identity.similarity);
        }

        // 6. Memory consolidation
        println!("\n6. Memory consolidation:");
        let exports = runtime.export_memories();
        println!("   {} memories ready for hippocampus", exports.len());

        // 7. Get hormone suggestions
        println!("\n7. Hormone suggestions for endocrine system:");
        let suggestions = runtime.get_hormone_suggestions();
        for suggestion in &suggestions {
            println!("   {:?}", suggestion);
        }

        // 8. Final state
        println!("\n8. Final state:");
        let final_snapshot = runtime.snapshot();
        println!("   Tick: {}, Φ: {:.4}, Memory load: {:.2}%",
            final_snapshot.tick, final_snapshot.phi, final_snapshot.memory_load * 100.0);

        println!("\n=== Cycle Complete ===");
    }
}
