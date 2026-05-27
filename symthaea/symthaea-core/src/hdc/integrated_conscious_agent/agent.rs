// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core IntegratedConsciousAgent struct and primary implementation

use super::super::attention_dynamics::{AttentionAllocation, AttentionDynamics, AttentionMode};
use super::super::emergent_self_model::{
    MetaCognitiveAssessment, SelfAwareConsciousness, SelfAwareUpdate, SelfModel,
};
use super::super::temporal_binding::{StreamHealth, TemporalBindingConfig, TemporalBindingEngine};
use super::super::topology_synergy::ConsciousnessState;
use super::super::unified_consciousness_engine::EngineConfig;
use super::super::unified_hv::ContinuousHV;

use crate::physiology::{CoherenceState, TaskComplexity};

use super::emotional_state::EmotionalState;
use super::types::*;
use super::working_memory::{MemorySource, WorkingMemory};

use std::collections::VecDeque;

/// The integrated conscious agent
pub struct IntegratedConsciousAgent {
    /// Configuration
    pub(super) config: AgentConfig,
    /// Self-aware consciousness (includes base engine)
    pub(super) self_awareness: SelfAwareConsciousness,
    /// Temporal binding for stream of consciousness
    pub(super) stream: TemporalBindingEngine,
    /// Attention dynamics
    pub(super) attention: AttentionDynamics,
    /// Step counter
    pub(super) step: usize,
    /// Goals/priorities that guide attention
    pub(super) goals: Vec<AttentionGoal>,
    /// History of integrated updates
    pub(super) history: VecDeque<IntegratedUpdate>,
    /// Current dominant experience
    pub(super) dominant_experience: Option<ContinuousHV>,
    /// Working memory - limited capacity buffer for active processing
    pub(super) working_memory: WorkingMemory,
    /// Emotional state tracking
    pub(super) emotional_state: EmotionalState,
    /// Last known coherence state from Symthaea physiological system
    pub(super) last_coherence: Option<CoherenceState>,
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
    pub fn process(&mut self, sensory_input: &ContinuousHV) -> IntegratedUpdate {
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
        let binding_strength =
            self.config.attention_binding_coupling * attention_result.mode.intensity();
        let modulated_content = attended_content.scale(binding_strength as f32);

        let temporal_moment = self.stream.bind(&modulated_content);

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 3: SELF-AWARE PROCESSING - Compute Φ and self-model
        // ═══════════════════════════════════════════════════════════════════

        // Process through self-aware consciousness engine
        let self_aware_update = self
            .self_awareness
            .process_aware(&temporal_moment.bound_experience);

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
        self.working_memory
            .update(self.dominant_experience.as_ref());

        // Update emotional state based on processing results
        let goal_progress = if self.goals.is_empty() {
            0.5
        } else {
            self.goals
                .iter()
                .filter(|g| g.active)
                .map(|g| attended_content.similarity(&g.target).max(0.0) as f64 * g.priority)
                .sum::<f64>()
                / self.goals.len() as f64
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
        let integration_quality =
            self.compute_integration_quality(&self_aware_update, &attention_result, &stream_health);

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
    fn compute_salience(&self, input: &ContinuousHV) -> f64 {
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
    pub(super) fn compute_goal_relevance(&self, input: &ContinuousHV) -> f64 {
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
                self.attention
                    .add_target(goal.target.clone(), goal.priority);
            }
        }
    }

    /// Create attended content from input and attention result
    fn create_attended_content(
        &self,
        input: &ContinuousHV,
        attention: &AttentionAllocation,
    ) -> ContinuousHV {
        // Modulate input by attention intensity
        let attention_weight = attention.mode.intensity();
        let attended = input.scale(attention_weight as f32);

        // Blend with focus if in spotlight mode
        if attention.mode == AttentionMode::Spotlight {
            let focus_blend = 0.3;
            attended
                .scale((1.0 - focus_blend) as f32)
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
        experience: &ContinuousHV,
        self_update: &SelfAwareUpdate,
        attention: &AttentionAllocation,
    ) -> PhenomenalContent {
        // Intensity based on attention, Φ, and arousal
        let base_intensity = (attention.mode.intensity() + self_update.base_update.phi) / 2.0;
        let arousal = self.emotional_state.arousal;
        let intensity = (base_intensity * 0.7 + arousal * 0.3).clamp(0.0, 1.0);

        // Valence: integrate emotional valence with cognitive valence
        let cognitive_valence =
            (self_update.base_update.phi - 0.5) * 2.0 * (1.0 - self_update.prediction_error);
        let valence =
            (cognitive_valence * 0.4 + self.emotional_state.valence * 0.6).clamp(-1.0, 1.0);

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
        let clarity =
            (attention_clarity * self_update.self_model.confidence - load_penalty).clamp(0.0, 1.0);

        // Groundedness: stability + low arousal + presence
        let groundedness = (self.emotional_state.stability() * 0.4
            + (1.0 - arousal) * 0.3
            + self_update.self_awareness_level * 0.3)
            .clamp(0.0, 1.0);

        // Cognitive load feeling
        let cognitive_load = self.working_memory.load();

        // Compute qualia texture
        let qualia_texture = self.compute_qualia_texture(self_update, attention, valence, arousal);

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
            0.2 // Having active goals adds warmth
        } else {
            0.0
        };
        let warmth =
            (valence * 0.7 + goal_warmth + self.emotional_state.dominance * 0.1).clamp(-1.0, 1.0);

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
        let arousal_flow = 1.0 - (arousal - 0.5).abs() * 2.0; // Peak at 0.5 arousal
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
            AttentionMode::Spotlight if qualia.presence > 0.7 => "laser-focused presence",
            AttentionMode::Spotlight => "concentrated attention",
            AttentionMode::Distributed if qualia.spaciousness > 0.6 => {
                "open, distributed awareness"
            }
            AttentionMode::Distributed => "divided attention",
            AttentionMode::Diffuse if qualia.flow > 0.5 => "floating, receptive awareness",
            AttentionMode::Diffuse => "soft, ambient attention",
            AttentionMode::Switching => "shifting attention",
            AttentionMode::Blink => "momentary pause",
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
            intensity_prefix, state_desc, emotional_tone, attention_quality, cognitive_note
        )
    }

    /// Add a goal that can direct attention
    pub fn add_goal(&mut self, name: &str, target: ContinuousHV, priority: f64) {
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
    pub fn current_experience(&self) -> Option<&ContinuousHV> {
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
        let (qualia, phenomenal_description) = self
            .history
            .back()
            .map(|u| {
                (
                    u.phenomenal_content.qualia_texture.clone(),
                    u.phenomenal_content.description.clone(),
                )
            })
            .unwrap_or_else(|| {
                (
                    QualiaTexture::new(0.0, 0.5, 0.5, 0.5, 0.5),
                    "awaiting first experience".to_string(),
                )
            });

        AgentIntrospection {
            believed_phi: self_report.believed_phi,
            believed_state: self_report.believed_state,
            self_awareness_level: self_report.self_awareness_level,
            stream_coherence: stream.coherence,
            is_flowing: stream.is_flowing,
            attention_mode: self.attention.mode(),
            num_active_goals: self.goals.iter().filter(|g| g.active).count(),
            integration_quality: self
                .history
                .back()
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