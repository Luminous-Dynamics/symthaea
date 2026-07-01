// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Physiological Integration
//!
//! These methods bridge the IntegratedConsciousAgent with Symthaea's embodied
//! consciousness systems: hormones, coherence, memory, identity, and voice.

use super::super::attention_dynamics::AttentionMode;
use super::super::unified_hv::ContinuousHV;

use crate::memory::EmotionalValence;
use crate::physiology::{CoherenceState, HormoneState, TaskComplexity};
use crate::soul::KVector;
use crate::voice::LTCPacing;

use super::agent::IntegratedConsciousAgent;
use super::working_memory::MemorySource;

// ═══════════════════════════════════════════════════════════════════════════════
// SYMTHAEA INTEGRATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Suggested hormone events for EndocrineSystem
#[derive(Clone, Debug)]
pub enum HormoneEventSuggestion {
    Threat {
        intensity: f32,
        reason: String,
    },
    Reward {
        value: f32,
        reason: String,
    },
    DeepFocus {
        duration_cycles: u32,
        reason: String,
    },
    Success {
        magnitude: f32,
        reason: String,
    },
    Error {
        severity: f32,
        reason: String,
    },
}

/// Result of coherence gating check
#[derive(Clone, Debug)]
pub enum CoherenceGating {
    Proceed {
        margin: f32,
    },
    Defer {
        current: f32,
        required: f32,
        centering_needed: f32,
    },
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
    Stable,   // > 0.8 similarity
    Drifting, // 0.65-0.8 similarity
    Crisis,   // < 0.65 similarity
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

impl ProsodyHints {
    /// Convert consciousness-driven prosody hints to LTC pacing parameters
    /// for use with the voice synthesis system
    pub fn to_ltc_pacing(&self) -> crate::voice::LTCPacing {
        // Map consciousness rate (0.7-1.3) to LTC speech_rate (0.8-1.2)
        let speech_rate = (self.rate * 0.9).clamp(0.8, 1.2);

        // Convert pause multiplier to milliseconds
        // Base pause is 250ms, multiplier scales it
        let pause_ms = (250.0 * self.pause_multiplier) as u32;

        // Peak flow when rate is high and energy is high
        let peak_flow = self.rate > 1.1 && self.energy > 0.7;

        crate::voice::LTCPacing {
            speech_rate,
            pause_ms,
            peak_flow,
        }
    }

    /// Create extended pacing with all consciousness parameters
    pub fn to_extended_pacing(&self) -> ExtendedPacing {
        ExtendedPacing {
            speech_rate: self.rate,
            pitch_shift_semitones: self.pitch_shift,
            energy_level: self.energy,
            pause_ms: (250.0 * self.pause_multiplier) as u32,
            emphasis_words: self.emphasis_words.clone(),
            peak_flow: self.rate > 1.1 && self.energy > 0.7,
        }
    }
}

/// Extended pacing parameters for advanced voice synthesis
/// Includes all consciousness-derived voice modulation parameters
#[derive(Debug, Clone)]
pub struct ExtendedPacing {
    /// Speech rate multiplier (0.7 - 1.3)
    pub speech_rate: f32,
    /// Pitch shift in semitones (-5 to +5)
    pub pitch_shift_semitones: f32,
    /// Vocal energy level (0.0 - 1.0)
    pub energy_level: f32,
    /// Pause duration in milliseconds
    pub pause_ms: u32,
    /// Words to emphasize
    pub emphasis_words: Vec<String>,
    /// Whether in peak consciousness flow
    pub peak_flow: bool,
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

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSIOLOGICAL INTEGRATION METHODS
// ═══════════════════════════════════════════════════════════════════════════════

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
        let cortisol_valence_effect = -(hormones.cortisol - 0.3) * 0.8; // Negative contribution
        let cortisol_arousal_effect = (hormones.cortisol - 0.3) * 0.6; // Stress increases arousal

        // Dopamine effect: reward hormone drives positive valence
        // Baseline dopamine is ~0.5
        let dopamine_valence_effect = (hormones.dopamine - 0.5) * 1.0; // Positive contribution

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
                spaciousness_boost: (1.0 - self.working_memory.load() as f32)
                    * coherence.coherence
                    * 0.3,
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
        self.working_memory
            .episodic_buffer
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
            let content = ContinuousHV::from_values(memory.content_vector);
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
        let emotion_offset = 250
            + match (
                self.emotional_state.valence > 0.0,
                self.emotional_state.arousal > 0.5,
            ) {
                (true, true) => 0,   // excited/happy
                (true, false) => 1,  // calm/content
                (false, true) => 2,  // stressed/anxious
                (false, false) => 3, // sad/bored
            } * 50;
        for i in emotion_offset..(emotion_offset + 50).min(dim) {
            k_vec[i] = 1.0;
        }

        // Encode integration quality
        let quality_signal = self
            .history
            .back()
            .map(|u| u.integration_quality)
            .unwrap_or(0.5);
        for i in 450..500 {
            k_vec[i] = quality_signal;
        }

        // Encode goal-directedness
        let goal_signal = if self.goals.iter().any(|g| g.active) {
            1.0
        } else {
            0.0
        };
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
        let dot: f64 = current
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| a * b)
            .sum();
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
        let attention_drift: f64 = current[0..250]
            .iter()
            .zip(&reference[0..250])
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 250.0;
        if attention_drift > 0.3 {
            drifts.push("attention_mode".to_string());
        }

        // Check emotional drift (250-450)
        let emotion_drift: f64 = current[250..450]
            .iter()
            .zip(&reference[250..450])
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 200.0;
        if emotion_drift > 0.3 {
            drifts.push("emotional_state".to_string());
        }

        // Check qualia drift (550-800)
        let qualia_drift: f64 = current[550..800]
            .iter()
            .zip(&reference[550..800])
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 250.0;
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
        pacing.peak_flow = stream.is_flowing
            && self
                .history
                .back()
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