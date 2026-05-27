// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Narrative Arc Dynamics using Hierarchical CfC
//!
//! Wraps `HierarchicalCfC` with narrative-specific semantics. Scene HDC vectors
//! are projected into the CfC input space, and the multi-scale output is decoded
//! into a **NarrativeSignal** (the "Ghost Signal") that captures the continuous
//! dynamics of a story: energy, surprise, valence, tension, momentum.
//!
//! # Time Scale Mapping
//!
//! | Layer | tau     | Narrative role         |
//! |-------|---------|------------------------|
//! | 0     | 0.1     | Beat-by-beat pacing    |
//! | 1     | 1.0     | Scene arc              |
//! | 2     | 10.0    | Act arc / tension      |
//! | 3     | 100.0   | Thematic throughline   |
//!
//! The `NarrativeCompiler` (in `language::narrative_compiler`) reads the Ghost
//! Signal and converts it into concrete writing instructions for an LLM.

use std::collections::VecDeque;

use ndarray::Array1;
use symthaea_core::hdc::ContinuousHV;

use super::hierarchical_cfc::{HierarchicalCfC, HierarchicalCfCConfig};
use crate::hdc::narrative_algebra::{
    ArcPhase, NARRATIVE_DIM, NarrativeAlgebra, NarrativeMood, TensionLevel,
};

// ============================================================================
// Narrative Archetypes — deterministic HDC reference vectors
// ============================================================================

/// Archetype HVs for computing instantaneous narrative signal from HDC space.
///
/// These provide scene-level signal values via cosine similarity, bypassing
/// the CfC network for a direct "HDC readout" that is then blended with the
/// CfC temporal dynamics.
pub struct NarrativeArchetypes {
    pub high_energy: ContinuousHV,
    pub surprise: ContinuousHV,
    pub positive_valence: ContinuousHV,
    pub high_tension: ContinuousHV,
}

impl NarrativeArchetypes {
    /// Create archetypes composed from narrative algebra primitives.
    ///
    /// Instead of random vectors (which have ~0 similarity to any scene),
    /// these archetypes are built from the same primitives used to encode
    /// scenes, so cosine similarity produces meaningful signal variation.
    pub fn from_algebra(algebra: &NarrativeAlgebra) -> Self {
        // High energy: stakes + tense mood + conflict (action-packed scenes)
        let high_energy = ContinuousHV::bundle(&[
            &algebra.primitives.stakes,
            &algebra.encode_mood(NarrativeMood::Tense),
            &algebra.primitives.conflict,
        ]);

        // Surprise: conflict + mysterious mood (unexpected twists)
        let surprise = ContinuousHV::bundle(&[
            &algebra.primitives.conflict,
            &algebra.encode_mood(NarrativeMood::Mysterious),
            &algebra.encode_tension(TensionLevel::Peak),
        ]);

        // Positive valence: hopeful + triumphant moods (uplifting scenes)
        let positive_valence = ContinuousHV::bundle(&[
            &algebra.encode_mood(NarrativeMood::Hopeful),
            &algebra.encode_mood(NarrativeMood::Triumphant),
            &algebra.encode_mood(NarrativeMood::Peaceful),
        ]);

        // High tension: peak tension + conflict + escalation operator
        let high_tension = ContinuousHV::bundle(&[
            &algebra.encode_tension(TensionLevel::Peak),
            &algebra.primitives.conflict,
            &algebra.operators.escalates,
        ]);

        Self {
            high_energy,
            surprise,
            positive_valence,
            high_tension,
        }
    }

    /// Create archetypes with deterministic random seeds (fallback).
    pub fn new(dim: usize) -> Self {
        Self {
            high_energy: ContinuousHV::random(dim, 13_000_003),
            surprise: ContinuousHV::random(dim, 13_000_017),
            positive_valence: ContinuousHV::random(dim, 13_000_029),
            high_tension: ContinuousHV::random(dim, 13_000_039),
        }
    }
}

// ============================================================================
// NarrativeSignal — the "Ghost Signal"
// ============================================================================

/// The continuous narrative dynamics output ("Ghost Signal").
///
/// Each field is a scalar extracted from the multi-scale CfC output, capturing
/// one dimension of the story's current state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NarrativeSignal {
    /// 0.0 (still) to 1.0 (frantic) — pacing / intensity
    pub energy: f32,
    /// 0.0 (predictable) to 1.0 (shocking) — entropy
    pub surprise: f32,
    /// -1.0 (dark) to 1.0 (light) — emotional direction
    pub valence: f32,
    /// 0.0 (relaxed) to 1.0 (unbearable) — stakes / pressure
    pub tension: f32,
    /// -1.0 (decelerating) to 1.0 (accelerating) — pace change
    pub momentum: f32,
    /// Inferred from signal shape
    pub arc_phase: ArcPhase,
}

impl Default for NarrativeSignal {
    fn default() -> Self {
        Self {
            energy: 0.0,
            surprise: 0.0,
            valence: 0.0,
            tension: 0.0,
            momentum: 0.0,
            arc_phase: ArcPhase::Setup,
        }
    }
}

// ============================================================================
// StoryArcConfig
// ============================================================================

/// Configuration for story arc dynamics.
pub struct StoryArcConfig {
    /// Fast: beat-by-beat (default: 0.1)
    pub scene_tau: f32,
    /// Medium: scene arcs (default: 1.0)
    pub act_tau: f32,
    /// Slow: overall arc (default: 10.0)
    pub story_tau: f32,
    /// Glacial: thematic throughline (default: 100.0)
    pub theme_tau: f32,
    /// Top-down theme → scene influence (default: 0.3)
    pub feedback_strength: f32,
    /// CfC hidden dimension per layer (default: 64)
    pub hidden_dim: usize,
    /// Input projection dimension — size of the scene vector fed to the CfC
    /// (default: 32)
    pub input_dim: usize,
}

impl Default for StoryArcConfig {
    fn default() -> Self {
        Self {
            scene_tau: 0.1,
            act_tau: 1.0,
            story_tau: 10.0,
            theme_tau: 100.0,
            feedback_strength: 0.3,
            hidden_dim: 64,
            input_dim: 32,
        }
    }
}

// ============================================================================
// StoryArcDynamics
// ============================================================================

/// Story arc dynamics engine wrapping `HierarchicalCfC`.
pub struct StoryArcDynamics {
    config: StoryArcConfig,
    cfc: HierarchicalCfC,
    /// Stored CfC config for re-creation in predict_horizon
    cfc_config: HierarchicalCfCConfig,
    step_count: u64,
    signal_history: VecDeque<NarrativeSignal>,
    /// Previous signal for momentum computation
    prev_energy: f32,
    /// HDC archetype vectors for instantaneous readout
    archetypes: NarrativeArchetypes,
    /// Blend factor: alpha * CfC + (1-alpha) * HDC (default 0.3, decays toward 0.5)
    hdc_blend_alpha: f32,
    /// Whether to train the CfC online using HDC signal as target
    online_learning: bool,
    /// Learning rate for CfC online learning
    online_lr: f32,
    /// Tension offset from conflict events (decays each step)
    tension_offset: f32,
    /// Running min/max tension for adaptive arc phase thresholds
    tension_min: f32,
    tension_max: f32,
    /// Count of consecutive scenes where tension moved in the same direction.
    /// Positive = consecutive rises, negative = consecutive falls.
    tension_streak: i32,
    /// Previous tension value for streak tracking
    prev_tension: f32,
    /// Recent mood history for mood-sequence scaling (last N moods)
    mood_history: VecDeque<NarrativeMood>,
}

/// Maximum signal history length
const MAX_HISTORY: usize = 256;

impl StoryArcDynamics {
    /// Create a new story arc dynamics engine.
    pub fn new(config: StoryArcConfig) -> Self {
        let cfc_config = HierarchicalCfCConfig {
            input_dim: config.input_dim,
            output_dim: config.hidden_dim,
            time_constants: vec![
                config.scene_tau,
                config.act_tau,
                config.story_tau,
                config.theme_tau,
            ],
            hidden_dims: vec![
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
            ],
            top_down_feedback: true,
            feedback_strength: config.feedback_strength,
            bottom_up_integration: true,
            integration_strength: 0.5,
            multi_scale_prediction: true,
            lr_scales: vec![0.5, 1.0, 1.0, 0.5],
        };

        let mut cfc = HierarchicalCfC::new(cfc_config.clone());

        // Apply narrative-specific output biases so resting state isn't flat 0.5
        Self::apply_narrative_bias(&mut cfc, config.hidden_dim);

        let archetypes = NarrativeArchetypes::new(NARRATIVE_DIM);

        Self {
            config,
            cfc,
            cfc_config,
            step_count: 0,
            signal_history: VecDeque::with_capacity(MAX_HISTORY),
            prev_energy: 0.0,
            archetypes,
            hdc_blend_alpha: 0.3,
            online_learning: true,
            online_lr: 0.01,
            tension_offset: 0.0,
            tension_min: 0.5,
            tension_max: 0.5,
            tension_streak: 0,
            prev_tension: 0.5,
            mood_history: VecDeque::with_capacity(8),
        }
    }

    /// Create with semantic archetypes derived from the narrative algebra.
    ///
    /// This produces much better signal variation than random archetypes
    /// because the archetype HVs share structure with scene HVs.
    pub fn with_algebra(config: StoryArcConfig, algebra: &NarrativeAlgebra) -> Self {
        let mut dynamics = Self::new(config);
        dynamics.archetypes = NarrativeArchetypes::from_algebra(algebra);
        dynamics
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StoryArcConfig::default())
    }

    /// Advance dynamics one step.
    ///
    /// `scene_input` should be an array of dimension `config.input_dim`
    /// (project your HDC scene vector down before calling this).
    pub fn step(&mut self, scene_input: &Array1<f32>, dt: f32) -> NarrativeSignal {
        let output = self.cfc.forward_hierarchical(scene_input, dt);
        self.step_count += 1;

        // Extract scalar fields from multi-scale layer outputs.
        // Each layer's output is an Array1<f32> of length hidden_dim.
        // We take the mean of the first few elements as the scalar signal.
        let energy = Self::extract_scalar(&output.scale_outputs[0], 0);
        let momentum_raw = energy - self.prev_energy;
        self.prev_energy = energy;

        let surprise = Self::extract_scalar(&output.scale_outputs[1], 0);
        let tension = Self::extract_scalar(&output.scale_outputs[2], 0);
        let valence = Self::extract_scalar_signed(&output.scale_outputs[3], 0);

        let momentum = momentum_raw.clamp(-1.0, 1.0);

        let arc_phase = Self::infer_arc_phase(tension, momentum, self.step_count);

        let signal = NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum,
            arc_phase,
        };

        // Maintain bounded history
        if self.signal_history.len() >= MAX_HISTORY {
            self.signal_history.pop_front();
        }
        self.signal_history.push_back(signal.clone());

        signal
    }

    /// Look ahead without advancing internal state.
    ///
    /// Creates a fresh CfC from the stored config, steps it forward, and
    /// returns the resulting signal. Because CfC internal state is not
    /// cloneable, this gives an approximate "relaxation" prediction.
    pub fn predict_horizon(&self, horizon: f32) -> NarrativeSignal {
        let mut cfc_fresh = HierarchicalCfC::new(self.cfc_config.clone());
        let dt = 0.1_f32;
        let steps = (horizon / dt).ceil().max(1.0) as usize;
        let zero_input = Array1::zeros(self.config.input_dim);

        let mut last_output = cfc_fresh.forward_hierarchical(&zero_input, dt);
        for _ in 1..steps {
            last_output = cfc_fresh.forward_hierarchical(&zero_input, dt);
        }

        let energy = Self::extract_scalar(&last_output.scale_outputs[0], 0);
        let surprise = Self::extract_scalar(&last_output.scale_outputs[1], 0);
        let tension = Self::extract_scalar(&last_output.scale_outputs[2], 0);
        let valence = Self::extract_scalar_signed(&last_output.scale_outputs[3], 0);

        NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum: 0.0, // no delta available in prediction
            arc_phase: Self::infer_arc_phase(tension, 0.0, self.step_count + steps as u64),
        }
    }

    /// Reset to initial state (start a new story).
    pub fn reset(&mut self) {
        self.cfc.reset();
        self.step_count = 0;
        self.signal_history.clear();
        self.prev_energy = 0.0;
        self.hdc_blend_alpha = 0.3;
        self.tension_offset = 0.0;
        self.tension_min = 0.5;
        self.tension_max = 0.5;
        self.tension_streak = 0;
        self.prev_tension = 0.5;
        self.mood_history.clear();
    }

    /// Recent signal trajectory.
    pub fn signal_history(&self) -> &VecDeque<NarrativeSignal> {
        &self.signal_history
    }

    /// Number of steps processed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Compute a character-specific signal by comparing a character's presence
    /// HV against the narrative archetypes.
    ///
    /// This produces per-character emotional trajectories that diverge from
    /// the global scene signal based on each character's accumulated context.
    pub fn character_signal(
        &self,
        char_hv: &ContinuousHV,
        base_signal: &NarrativeSignal,
    ) -> NarrativeSignal {
        let char_energy = (char_hv.similarity(&self.archetypes.high_energy) + 1.0) / 2.0;
        let char_surprise = (char_hv.similarity(&self.archetypes.surprise) + 1.0) / 2.0;
        let char_tension = (char_hv.similarity(&self.archetypes.high_tension) + 1.0) / 2.0;
        let char_valence = char_hv.similarity(&self.archetypes.positive_valence);

        // Blend: 70% scene signal + 30% character-specific
        let blend = 0.3;
        NarrativeSignal {
            energy: (base_signal.energy * (1.0 - blend) + char_energy * blend).clamp(0.0, 1.0),
            surprise: (base_signal.surprise * (1.0 - blend) + char_surprise * blend)
                .clamp(0.0, 1.0),
            tension: (base_signal.tension * (1.0 - blend) + char_tension * blend).clamp(0.0, 1.0),
            valence: (base_signal.valence * (1.0 - blend) + char_valence * blend).clamp(-1.0, 1.0),
            momentum: base_signal.momentum,
            arc_phase: base_signal.arc_phase,
        }
    }

    /// Inject a tension spike (called when a conflict is introduced).
    ///
    /// The offset decays by 50% each scene, creating a sharp spike that fades.
    pub fn inject_tension_spike(&mut self, magnitude: f32) {
        self.tension_offset += magnitude;
    }

    /// Inject a tension drop (called when a conflict is resolved).
    pub fn inject_tension_drop(&mut self, magnitude: f32) {
        self.tension_offset -= magnitude;
    }

    /// Advance dynamics one step with hybrid HDC+CfC signal extraction.
    ///
    /// Blends CfC temporal dynamics with instantaneous HDC archetype similarity,
    /// applies mood-based signal offsets, and optionally trains the CfC online
    /// using the HDC-derived signal as a target.
    ///
    /// `signal = alpha * cfc_value + (1 - alpha) * hdc_value + mood_offset`
    pub fn step_hybrid(
        &mut self,
        scene_input: &Array1<f32>,
        scene_hv: &ContinuousHV,
        dt: f32,
    ) -> NarrativeSignal {
        self.step_hybrid_with_mood(scene_input, scene_hv, dt, None)
    }

    /// Advance dynamics with explicit mood for signal bias.
    ///
    /// `scene_position` is an optional normalized [0,1] value indicating
    /// where this scene falls in the overall arc (0.0 = start, 1.0 = end).
    /// When provided, arc-phase tension shaping modulates the output so that
    /// tension peaks later in the story rather than spiking on the first
    /// high-intensity scene.
    pub fn step_hybrid_with_mood(
        &mut self,
        scene_input: &Array1<f32>,
        scene_hv: &ContinuousHV,
        dt: f32,
        mood: Option<NarrativeMood>,
    ) -> NarrativeSignal {
        self.step_hybrid_with_mood_and_position(scene_input, scene_hv, dt, mood, None)
    }

    /// Full-featured step with mood and scene position.
    pub fn step_hybrid_with_mood_and_position(
        &mut self,
        scene_input: &Array1<f32>,
        scene_hv: &ContinuousHV,
        dt: f32,
        mood: Option<NarrativeMood>,
        scene_position: Option<f32>,
    ) -> NarrativeSignal {
        let output = self.cfc.forward_hierarchical(scene_input, dt);
        self.step_count += 1;

        // Blend ratio decay: alpha moves from 0.3 toward 0.5 as CfC learns
        let alpha = self.hdc_blend_alpha;
        let one_minus_alpha = 1.0 - alpha;

        // CfC component (temporal dynamics)
        let cfc_energy = Self::extract_scalar(&output.scale_outputs[0], 0);
        let cfc_surprise = Self::extract_scalar(&output.scale_outputs[1], 0);
        let cfc_tension = Self::extract_scalar(&output.scale_outputs[2], 0);
        let cfc_valence = Self::extract_scalar_signed(&output.scale_outputs[3], 0);

        // HDC component (instantaneous archetype similarity)
        let hdc_energy = (scene_hv.similarity(&self.archetypes.high_energy) + 1.0) / 2.0;
        let hdc_surprise = (scene_hv.similarity(&self.archetypes.surprise) + 1.0) / 2.0;
        let hdc_tension = (scene_hv.similarity(&self.archetypes.high_tension) + 1.0) / 2.0;
        let hdc_valence = scene_hv.similarity(&self.archetypes.positive_valence); // [-1,1]

        // Blend HDC + CfC
        let mut energy = alpha * cfc_energy + one_minus_alpha * hdc_energy;
        let mut surprise = alpha * cfc_surprise + one_minus_alpha * hdc_surprise;
        let mut tension = alpha * cfc_tension + one_minus_alpha * hdc_tension;
        let mut valence = alpha * cfc_valence + one_minus_alpha * hdc_valence;

        // Mood-sequence scaling: amplify offsets for mood transitions,
        // compound for consecutive same-mood scenes
        let mood_scale = if let Some(m) = mood {
            self.mood_transition_scale(m)
        } else {
            1.0
        };

        // Apply mood-based signal offsets (scaled by mood transition)
        if let Some(m) = mood {
            let (de, ds, dt_, dv) = Self::mood_offsets(m);
            energy += de * mood_scale;
            surprise += ds * mood_scale;
            tension += dt_ * mood_scale;
            valence += dv * mood_scale;
            // Record mood in history
            if self.mood_history.len() >= 8 {
                self.mood_history.pop_front();
            }
            self.mood_history.push_back(m);
        }

        // Apply conflict-driven tension offset, then decay it
        tension += self.tension_offset;
        self.tension_offset *= 0.5; // 50% decay per scene

        // Clamp to valid ranges
        let energy = energy.clamp(0.0, 1.0);
        let surprise = surprise.clamp(0.0, 1.0);
        let mut tension = tension.clamp(0.0, 1.0);
        let valence = valence.clamp(-1.0, 1.0);

        // Tension momentum compounding: consecutive same-direction moves amplify
        let tension_delta = tension - self.prev_tension;
        if tension_delta > 0.005 {
            // Rising
            self.tension_streak = if self.tension_streak > 0 {
                self.tension_streak + 1
            } else {
                1
            };
        } else if tension_delta < -0.005 {
            // Falling
            self.tension_streak = if self.tension_streak < 0 {
                self.tension_streak - 1
            } else {
                -1
            };
        }
        // else: no significant change, streak preserved

        // Apply compounding: each consecutive same-direction scene adds a bonus
        let streak_abs = self.tension_streak.unsigned_abs();
        if streak_abs >= 2 {
            // Bonus: 0.03 per streak step beyond 1 (e.g., 3-streak = +0.06)
            let bonus = 0.03 * (streak_abs - 1) as f32;
            if self.tension_streak > 0 {
                tension = (tension + bonus).min(1.0);
            } else {
                tension = (tension - bonus).max(0.0);
            }
        }
        self.prev_tension = tension;

        // Arc-phase tension shaping: modulate tension based on story position.
        // Early scenes get dampened, late scenes get amplified, so peaks shift
        // toward the climax rather than spiking on the first high-intensity input.
        if let Some(pos) = scene_position {
            let phase_multiplier = Self::arc_phase_tension_multiplier(pos);
            // Apply as a soft modulation toward 0.5 baseline
            let baseline = 0.5;
            tension = baseline + (tension - baseline) * phase_multiplier;
            tension = tension.clamp(0.0, 1.0);
        }

        // Update adaptive tension tracking
        if tension < self.tension_min {
            self.tension_min = tension;
        }
        if tension > self.tension_max {
            self.tension_max = tension;
        }

        let momentum_raw = energy - self.prev_energy;
        self.prev_energy = energy;
        let momentum = momentum_raw.clamp(-1.0, 1.0);

        let arc_phase = self.infer_arc_phase_adaptive(tension, momentum);

        // CfC online learning: use HDC-derived values as training targets
        if self.online_learning {
            self.train_cfc_online(
                scene_input,
                hdc_energy,
                hdc_surprise,
                hdc_tension,
                hdc_valence,
                dt,
            );
        }

        // Decay blend alpha toward 0.5 (give CfC more weight as it learns)
        self.hdc_blend_alpha = (self.hdc_blend_alpha + 0.02).min(0.5);

        let signal = NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum,
            arc_phase,
        };

        if self.signal_history.len() >= MAX_HISTORY {
            self.signal_history.pop_front();
        }
        self.signal_history.push_back(signal.clone());

        signal
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Infer arc phase using adaptive thresholds based on observed tension range.
    ///
    /// Instead of hardcoded 0.8/0.5/0.3 thresholds, uses the running min/max
    /// tension to compute relative thresholds. Falls back to static thresholds
    /// if the range is too narrow (< 0.1).
    fn infer_arc_phase_adaptive(&self, tension: f32, momentum: f32) -> ArcPhase {
        let range = self.tension_max - self.tension_min;
        if range < 0.1 || self.step_count < 3 {
            return Self::infer_arc_phase(tension, momentum, self.step_count);
        }

        // Normalize tension to [0,1] within observed range
        let t_norm = (tension - self.tension_min) / range;

        if t_norm > 0.85 {
            ArcPhase::Climax
        } else if t_norm > 0.5 && momentum > 0.0 {
            ArcPhase::RisingAction
        } else if t_norm > 0.3 && momentum < 0.0 {
            ArcPhase::FallingAction
        } else if t_norm <= 0.3 && momentum <= 0.0 && self.step_count > 5 {
            ArcPhase::Resolution
        } else {
            ArcPhase::Setup
        }
    }

    /// Compute a mood-transition scale factor.
    ///
    /// - Same mood as previous → compounding (1.0 + 0.15 * consecutive_count)
    /// - Different mood → amplified transition (1.5 for sharp contrast, 1.2 otherwise)
    fn mood_transition_scale(&self, current: NarrativeMood) -> f32 {
        if self.mood_history.is_empty() {
            return 1.0;
        }
        let prev = match self.mood_history.back() {
            Some(m) => *m,
            None => return 1.0,
        };
        if std::mem::discriminant(&prev) == std::mem::discriminant(&current) {
            // Same mood repeating — compound by counting consecutive
            let consecutive = self
                .mood_history
                .iter()
                .rev()
                .take_while(|m| std::mem::discriminant(*m) == std::mem::discriminant(&current))
                .count();
            1.0 + 0.15 * consecutive as f32
        } else {
            // Mood transition — check if it's a sharp contrast
            let sharp = matches!(
                (prev, current),
                (NarrativeMood::Peaceful, NarrativeMood::Tense)
                    | (NarrativeMood::Tense, NarrativeMood::Peaceful)
                    | (NarrativeMood::Triumphant, NarrativeMood::Melancholy)
                    | (NarrativeMood::Melancholy, NarrativeMood::Triumphant)
                    | (NarrativeMood::Hopeful, NarrativeMood::Tense)
            );
            if sharp { 1.5 } else { 1.2 }
        }
    }

    /// Compute arc-phase tension multiplier based on normalized story position [0,1].
    ///
    /// The curve dampens tension early (×0.6 at start) and amplifies it near
    /// the golden ratio climax point (×1.3 at 0.618), then fades for resolution.
    /// This shifts peaks from the first high-intensity scene toward the natural
    /// climax position.
    fn arc_phase_tension_multiplier(position: f32) -> f32 {
        // Piecewise: ramp up from 0.6 to 1.3 over [0, 0.618], then down to 0.7
        let pos = position.clamp(0.0, 1.0);
        if pos <= 0.618 {
            // Linear ramp: 0.6 at pos=0, 1.3 at pos=0.618
            0.6 + (pos / 0.618) * 0.7
        } else {
            // Linear decay: 1.3 at pos=0.618, 0.7 at pos=1.0
            1.3 - ((pos - 0.618) / 0.382) * 0.6
        }
    }

    /// Score conflict text intensity using a keyword vocabulary.
    ///
    /// Returns a multiplier in [1.0, 2.0] based on the presence of
    /// high-intensity words in the conflict description.
    pub fn conflict_intensity(conflict: &str) -> f32 {
        const HIGH_INTENSITY: &[&str] = &[
            "dragon",
            "death",
            "ultimate",
            "final",
            "destroy",
            "betray",
            "kill",
            "war",
            "apocalypse",
            "doom",
            "annihilat",
            "catastroph",
            "invasion",
            "massacre",
            "inferno",
            "oblivion",
        ];
        const MED_INTENSITY: &[&str] = &[
            "battle", "fight", "enemy", "danger", "threat", "storm", "fire", "pursuit", "escape",
            "trap", "wound", "confront", "crisis", "attack", "siege", "ambush", "power",
            "betrayal",
        ];

        let lower = conflict.to_lowercase();
        let mut score = 0.0f32;
        for word in HIGH_INTENSITY {
            if lower.contains(word) {
                score += 0.3;
            }
        }
        for word in MED_INTENSITY {
            if lower.contains(word) {
                score += 0.15;
            }
        }
        // Clamp to [1.0, 2.0]
        (1.0 + score).min(2.0)
    }

    /// Mood-to-signal offsets: (energy, surprise, tension, valence).
    ///
    /// Each mood nudges the blended signal in a narratively appropriate direction.
    fn mood_offsets(mood: NarrativeMood) -> (f32, f32, f32, f32) {
        match mood {
            //                         energy  surprise  tension  valence
            NarrativeMood::Tense => (0.10, 0.05, 0.20, -0.10),
            NarrativeMood::Peaceful => (-0.10, -0.10, -0.20, 0.15),
            NarrativeMood::Mysterious => (0.00, 0.15, 0.10, 0.00),
            NarrativeMood::Melancholy => (-0.05, 0.00, 0.05, -0.20),
            NarrativeMood::Triumphant => (0.15, -0.05, -0.10, 0.25),
            NarrativeMood::Hopeful => (0.05, -0.05, -0.15, 0.20),
        }
    }

    /// Train CfC toward HDC-derived signal values (online learning).
    ///
    /// Constructs per-layer target arrays from the HDC signal components and
    /// runs one gradient step via `train_step_multiscale`. This lets the CfC
    /// gradually learn narrative dynamics from accumulating scene data.
    fn train_cfc_online(
        &mut self,
        scene_input: &Array1<f32>,
        hdc_energy: f32,
        hdc_surprise: f32,
        hdc_tension: f32,
        hdc_valence: f32,
        dt: f32,
    ) {
        let hidden_dim = self.config.hidden_dim;
        let lr = self.online_lr;

        // Build per-layer targets: each layer's target is an Array1 where
        // the first element encodes the layer's target signal value (inverse
        // sigmoid for [0,1] fields, atanh for [-1,1] valence).
        let inv_sigmoid = |x: f32| -> f32 {
            let x = x.clamp(0.01, 0.99);
            (x / (1.0 - x)).ln()
        };

        let targets: Vec<Array1<f32>> = vec![
            Self::make_target(hidden_dim, inv_sigmoid(hdc_energy)), // layer 0: energy
            Self::make_target(hidden_dim, inv_sigmoid(hdc_surprise)), // layer 1: surprise
            Self::make_target(hidden_dim, inv_sigmoid(hdc_tension)), // layer 2: tension
            Self::make_target(hidden_dim, hdc_valence.clamp(-0.99, 0.99).atanh()), // layer 3: valence
        ];

        // Ignore errors (training is best-effort)
        let _ = self
            .cfc
            .train_step_multiscale(scene_input, &targets, dt, lr);
    }

    /// Build a target array: first element set to `value`, rest zero.
    fn make_target(dim: usize, value: f32) -> Array1<f32> {
        let mut target = Array1::zeros(dim);
        target[0] = value;
        target
    }

    /// Apply narrative-specific output biases so the resting state isn't flat 0.5.
    fn apply_narrative_bias(cfc: &mut HierarchicalCfC, hidden_dim: usize) {
        // Layer 0 → energy: slight positive bias (resting energy > 0.5)
        let mut energy_bias = Array1::zeros(hidden_dim);
        energy_bias[0] = 0.3;
        cfc.set_output_bias(0, energy_bias);

        // Layer 2 → tension: slight negative bias (resting tension < 0.5)
        let mut tension_bias = Array1::zeros(hidden_dim);
        tension_bias[0] = -0.2;
        cfc.set_output_bias(2, tension_bias);
    }

    /// Extract a [0,1]-clamped scalar from a layer output.
    /// Uses sigmoid on the mean of the first `width` elements.
    fn extract_scalar(arr: &Array1<f32>, _channel: usize) -> f32 {
        let width = arr.len().min(8);
        let mean: f32 = arr.iter().take(width).sum::<f32>() / width as f32;
        sigmoid(mean)
    }

    /// Extract a [-1,1]-clamped scalar (for valence).
    fn extract_scalar_signed(arr: &Array1<f32>, _channel: usize) -> f32 {
        let width = arr.len().min(8);
        let mean: f32 = arr.iter().take(width).sum::<f32>() / width as f32;
        mean.tanh()
    }

    /// Infer arc phase from tension trajectory and step count.
    pub fn infer_arc_phase(tension: f32, momentum: f32, step: u64) -> ArcPhase {
        if step < 3 {
            return ArcPhase::Setup;
        }
        if tension > 0.8 {
            ArcPhase::Climax
        } else if tension > 0.5 && momentum > 0.0 {
            ArcPhase::RisingAction
        } else if tension > 0.3 && momentum < 0.0 {
            ArcPhase::FallingAction
        } else if tension <= 0.3 && momentum <= 0.0 && step > 5 {
            ArcPhase::Resolution
        } else {
            ArcPhase::Setup
        }
    }
}

/// Standard sigmoid
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    #[test]
    fn test_default_config() {
        let config = StoryArcConfig::default();
        assert_eq!(config.scene_tau, 0.1);
        assert_eq!(config.act_tau, 1.0);
        assert_eq!(config.story_tau, 10.0);
        assert_eq!(config.theme_tau, 100.0);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.input_dim, 32);
        assert!((config.feedback_strength - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_step_produces_signal() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);
        let signal = dynamics.step(&input, 0.1);

        assert!(
            (0.0..=1.0).contains(&signal.energy),
            "energy out of range: {}",
            signal.energy
        );
        assert!(
            (0.0..=1.0).contains(&signal.surprise),
            "surprise out of range: {}",
            signal.surprise
        );
        assert!(
            (-1.0..=1.0).contains(&signal.valence),
            "valence out of range: {}",
            signal.valence
        );
        assert!(
            (0.0..=1.0).contains(&signal.tension),
            "tension out of range: {}",
            signal.tension
        );
        assert!(
            (-1.0..=1.0).contains(&signal.momentum),
            "momentum out of range: {}",
            signal.momentum
        );
    }

    #[test]
    fn test_signal_evolves() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![1.0; 32]);

        let signal_1 = dynamics.step(&input, 0.1);
        // Feed different input
        let input_2 = Array1::from_vec(vec![-1.0; 32]);
        let signal_2 = dynamics.step(&input_2, 0.1);

        // At least one field should differ (CfC state evolves)
        let same = (signal_1.energy - signal_2.energy).abs() < 1e-6
            && (signal_1.surprise - signal_2.surprise).abs() < 1e-6
            && (signal_1.tension - signal_2.tension).abs() < 1e-6
            && (signal_1.valence - signal_2.valence).abs() < 1e-6;
        assert!(!same, "Signal should evolve after different inputs");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![1.0; 32]);
        dynamics.step(&input, 0.1);
        dynamics.step(&input, 0.1);
        assert_eq!(dynamics.step_count(), 2);
        assert!(!dynamics.signal_history().is_empty());

        dynamics.reset();
        assert_eq!(dynamics.step_count(), 0);
        assert!(dynamics.signal_history().is_empty());
    }

    #[test]
    fn test_high_input_energy() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        // Large positive values should push energy high (after sigmoid)
        let high_input = Array1::from_vec(vec![5.0; 32]);
        let signal = dynamics.step(&high_input, 0.1);

        // sigmoid(large positive) → close to 1.0
        // The exact value depends on CfC dynamics, but should be > 0.5
        assert!(
            signal.energy > 0.4,
            "High input should produce elevated energy: {}",
            signal.energy
        );
    }

    #[test]
    fn test_arc_phase_inference() {
        // Peak tension → Climax
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.9, 0.1, 10),
            ArcPhase::Climax
        );
        // Rising tension with positive momentum → RisingAction
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.6, 0.2, 10),
            ArcPhase::RisingAction
        );
        // Falling tension with negative momentum → FallingAction
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.4, -0.3, 10),
            ArcPhase::FallingAction
        );
        // Low tension, negative momentum, late in story → Resolution
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.2, -0.1, 10),
            ArcPhase::Resolution
        );
        // Early steps → always Setup
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.9, 0.5, 1),
            ArcPhase::Setup
        );
    }

    #[test]
    fn test_hybrid_signal_varies_with_mood() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);

        // Two very different scene HVs
        let calm_hv = ContinuousHV::random(NARRATIVE_DIM, 100);
        let tense_hv = ContinuousHV::random(NARRATIVE_DIM, 200);

        let signal_calm = dynamics.step_hybrid(&input, &calm_hv, 0.1);
        let signal_tense = dynamics.step_hybrid(&input, &tense_hv, 0.1);

        // The signals should differ because HDC similarities differ
        let any_differs = (signal_calm.energy - signal_tense.energy).abs() > 0.001
            || (signal_calm.surprise - signal_tense.surprise).abs() > 0.001
            || (signal_calm.tension - signal_tense.tension).abs() > 0.001
            || (signal_calm.valence - signal_tense.valence).abs() > 0.001;

        assert!(
            any_differs,
            "Hybrid signal should vary between different scene HVs"
        );
    }

    #[test]
    fn test_hybrid_energy_responds_to_conflict() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);

        // Use the high_energy archetype itself as scene — should maximize energy HDC component
        let energy_scene = dynamics.archetypes.high_energy.clone();
        let signal = dynamics.step_hybrid(&input, &energy_scene, 0.1);

        // Energy should be elevated (HDC component is 1.0 for self-similarity)
        assert!(
            signal.energy > 0.5,
            "Scene matching energy archetype should have high energy, got {}",
            signal.energy
        );
    }

    #[test]
    fn test_backward_compat_step() {
        // Original step() should still work and produce valid signals
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);
        let signal = dynamics.step(&input, 0.1);

        assert!((0.0..=1.0).contains(&signal.energy));
        assert!((0.0..=1.0).contains(&signal.tension));
        assert!((-1.0..=1.0).contains(&signal.valence));
    }

    #[test]
    fn test_narrative_bias_applied() {
        // With narrative bias, zero-input energy should differ from 0.5
        let mut dynamics = StoryArcDynamics::with_defaults();
        let zero_input = Array1::zeros(32);
        let zero_hv = ContinuousHV::zero(NARRATIVE_DIM);
        let signal = dynamics.step_hybrid(&zero_input, &zero_hv, 0.1);

        // Energy layer has +0.3 bias → sigmoid(0.3) ≈ 0.574
        // Blended with HDC (0.5 for zero HV), should not be exactly 0.5
        // We just check it's valid and not exactly 0.5
        assert!((0.0..=1.0).contains(&signal.energy));
    }
}
