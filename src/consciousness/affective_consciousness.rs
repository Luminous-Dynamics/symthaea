// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Affective Consciousness: Emotional Awareness
//!
//! Based on Damasio's Somatic Marker Hypothesis and core affect theory.
//! Emotions as bodily-felt states that guide decision-making.
//!
//! ## Core Affect Model (Russell's Circumplex)
//! - Valence: pleasure-displeasure dimension
//! - Arousal: activation-deactivation dimension
//! - Dominance: control dimension (optional)

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::PrimitiveSystem;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Core affect state (Russell's circumplex model)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoreAffect {
    /// Valence: -1.0 (unpleasant) to 1.0 (pleasant)
    pub valence: f32,

    /// Arousal: 0.0 (calm/deactivated) to 1.0 (excited/activated)
    pub arousal: f32,

    /// Dominance: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
}

impl CoreAffect {
    /// Create a new core affect
    pub fn new(valence: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
        }
    }

    /// Neutral affect state
    pub fn neutral() -> Self {
        Self::new(0.0, 0.5, 0.0)
    }

    /// Create from valence and arousal only
    pub fn from_va(valence: f32, arousal: f32) -> Self {
        Self::new(valence, arousal, 0.0)
    }

    /// Blend with another affect state
    pub fn blend(&self, other: &CoreAffect, weight: f32) -> CoreAffect {
        let w = weight.clamp(0.0, 1.0);
        CoreAffect::new(
            self.valence * (1.0 - w) + other.valence * w,
            self.arousal * (1.0 - w) + other.arousal * w,
            self.dominance * (1.0 - w) + other.dominance * w,
        )
    }

    /// Distance to another affect state
    pub fn distance(&self, other: &CoreAffect) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dv * dv + da * da + dd * dd).sqrt()
    }

    /// Get the closest emotion category
    pub fn closest_emotion(&self) -> EmotionCategory {
        // Simplified mapping based on valence/arousal quadrants
        if self.valence > 0.3 {
            if self.arousal > 0.6 {
                EmotionCategory::Joy
            } else if self.arousal > 0.3 {
                EmotionCategory::Contentment
            } else {
                EmotionCategory::Serenity
            }
        } else if self.valence < -0.3 {
            if self.arousal > 0.6 {
                if self.dominance < -0.3 {
                    EmotionCategory::Fear
                } else {
                    EmotionCategory::Anger
                }
            } else {
                EmotionCategory::Sadness
            }
        } else if self.arousal > 0.6 {
            EmotionCategory::Surprise
        } else {
            EmotionCategory::Neutral
        }
    }

    /// Alias for `closest_emotion()` (used by AffectiveBridge)
    pub fn to_emotion_category(&self) -> EmotionCategory {
        self.closest_emotion()
    }

    /// Emotional intensity (Euclidean magnitude in VAD space)
    pub fn intensity(&self) -> f64 {
        ((self.valence * self.valence
            + self.arousal * self.arousal
            + self.dominance * self.dominance) as f64)
            .sqrt()
    }

    /// Decay toward neutral by the given rate (0.0 = no decay, 1.0 = instant neutral)
    pub fn decay(&mut self, rate: f64) {
        *self = self.blend(&CoreAffect::neutral(), rate as f32);
    }
}

impl Default for CoreAffect {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Physiological state bridging body to affect
/// Used by embodied_cognition to represent body state
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysiologicalState {
    /// Change in heart rate from baseline
    pub heart_rate_delta: f32,
    /// Skin conductance (sweat response)
    pub skin_conductance: f32,
    /// Muscle tension level
    pub muscle_tension: f32,
    /// Change in breathing rate from baseline
    pub respiration_delta: f32,
    /// Gut feeling / interoceptive signal
    pub gut_feeling: f32,
}

impl Default for PhysiologicalState {
    fn default() -> Self {
        Self {
            heart_rate_delta: 0.0,
            skin_conductance: 0.0,
            muscle_tension: 0.0,
            respiration_delta: 0.0,
            gut_feeling: 0.0,
        }
    }
}

/// Discrete emotion categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionCategory {
    /// High valence, high arousal
    Joy,
    /// High valence, medium arousal
    Contentment,
    /// High valence, low arousal
    Serenity,
    /// Low valence, high arousal, low dominance
    Fear,
    /// Low valence, high arousal, high dominance
    Anger,
    /// Low valence, low arousal
    Sadness,
    /// Neutral valence, high arousal
    Surprise,
    /// Neutral state
    Neutral,
    /// Disgust
    Disgust,
    /// Anticipation
    Anticipation,
    /// Trust
    Trust,
}

impl EmotionCategory {
    /// Get typical core affect for this emotion
    pub fn typical_affect(&self) -> CoreAffect {
        match self {
            EmotionCategory::Joy => CoreAffect::new(0.8, 0.7, 0.5),
            EmotionCategory::Contentment => CoreAffect::new(0.6, 0.4, 0.3),
            EmotionCategory::Serenity => CoreAffect::new(0.5, 0.2, 0.2),
            EmotionCategory::Fear => CoreAffect::new(-0.7, 0.8, -0.6),
            EmotionCategory::Anger => CoreAffect::new(-0.6, 0.8, 0.6),
            EmotionCategory::Sadness => CoreAffect::new(-0.6, 0.2, -0.4),
            EmotionCategory::Surprise => CoreAffect::new(0.1, 0.8, 0.0),
            EmotionCategory::Neutral => CoreAffect::neutral(),
            EmotionCategory::Disgust => CoreAffect::new(-0.6, 0.5, 0.2),
            EmotionCategory::Anticipation => CoreAffect::new(0.3, 0.6, 0.3),
            EmotionCategory::Trust => CoreAffect::new(0.5, 0.3, 0.4),
        }
    }

    /// Get emotion name
    pub fn name(&self) -> &'static str {
        match self {
            EmotionCategory::Joy => "joy",
            EmotionCategory::Contentment => "contentment",
            EmotionCategory::Serenity => "serenity",
            EmotionCategory::Fear => "fear",
            EmotionCategory::Anger => "anger",
            EmotionCategory::Sadness => "sadness",
            EmotionCategory::Surprise => "surprise",
            EmotionCategory::Neutral => "neutral",
            EmotionCategory::Disgust => "disgust",
            EmotionCategory::Anticipation => "anticipation",
            EmotionCategory::Trust => "trust",
        }
    }

    /// Get all emotion categories
    pub fn all() -> Vec<Self> {
        vec![
            EmotionCategory::Joy,
            EmotionCategory::Contentment,
            EmotionCategory::Serenity,
            EmotionCategory::Fear,
            EmotionCategory::Anger,
            EmotionCategory::Sadness,
            EmotionCategory::Surprise,
            EmotionCategory::Neutral,
            EmotionCategory::Disgust,
            EmotionCategory::Anticipation,
            EmotionCategory::Trust,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NSM PRIMITIVE GROUNDING
// ═══════════════════════════════════════════════════════════════════════════

/// NSM primitive grounding for emotion categories
///
/// Maps each emotion to Wierzbicka's semantic primitives, capturing
/// the universal semantic core of emotional experience:
///
/// # Semantic Dimensions
/// - **Valence**: GOOD vs BAD
/// - **Arousal**: VERY, MOVE, DO
/// - **Agency**: CAN, WANT, DO
/// - **Social**: OTHER, I
/// - **Temporal**: NOW, BEFORE, AFTER, HAPPEN
///
/// # Grounding Philosophy
///
/// Emotions are grounded as felt evaluations (FEEL + GOOD/BAD) with
/// additional modifiers capturing arousal, agency, and social dimensions.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct EmotionPrimitiveGrounding {
    /// The emotion category
    pub emotion: EmotionCategory,
    /// NSM primitives that define this emotion
    pub nsm_primitives: Vec<String>,
    /// HDC encoding of the primitive composition
    pub primitive_encoding: BinaryHV,
    /// Core affect coordinates
    pub core_affect: CoreAffect,
    /// Valence dimension (-1 to 1)
    pub valence: f32,
    /// Arousal dimension (0 to 1)
    pub arousal: f32,
}

#[allow(dead_code)]
impl EmotionPrimitiveGrounding {
    /// Create grounding for an emotion with given primitive system
    pub(crate) fn new(emotion: EmotionCategory, primitives: &PrimitiveSystem) -> Self {
        let nsm_primitives = Self::primitives_for_emotion(&emotion);
        let primitive_encoding = Self::encode_primitives(&nsm_primitives, primitives);
        let core_affect = emotion.typical_affect();

        Self {
            emotion,
            nsm_primitives,
            primitive_encoding,
            valence: core_affect.valence,
            arousal: core_affect.arousal,
            core_affect,
        }
    }

    /// Get NSM primitives for each emotion
    ///
    /// Based on Wierzbicka's NSM analysis of emotion concepts:
    /// - Evaluative core: GOOD, BAD
    /// - Experiential: FEEL, BODY
    /// - Volitional: WANT, CAN, DO
    /// - Cognitive: KNOW, THINK
    /// - Temporal: NOW, BEFORE, AFTER, HAPPEN
    /// - Social: I, OTHER, SOMEONE
    fn primitives_for_emotion(emotion: &EmotionCategory) -> Vec<String> {
        match emotion {
            // Joy: intense positive feeling with high energy
            EmotionCategory::Joy => vec![
                "GOOD".to_string(),
                "VERY".to_string(),
                "FEEL".to_string(),
                "I".to_string(),
                "WANT".to_string(),
                "NOW".to_string(),
            ],

            // Contentment: satisfied, fulfilled state
            EmotionCategory::Contentment => vec![
                "GOOD".to_string(),
                "FEEL".to_string(),
                "I".to_string(),
                "NOW".to_string(),
                "HAVE".to_string(),
            ],

            // Serenity: calm, peaceful state
            EmotionCategory::Serenity => vec![
                "GOOD".to_string(),
                "FEEL".to_string(),
                "I".to_string(),
                "NOT".to_string(),
                "MOVE".to_string(),
                "NOW".to_string(),
            ],

            // Fear: threat response, loss of control
            EmotionCategory::Fear => vec![
                "BAD".to_string(),
                "FEEL".to_string(),
                "SOMETHING".to_string(),
                "HAPPEN".to_string(),
                "NOT".to_string(),
                "CAN".to_string(),
            ],

            // Anger: frustrated agency, desire to act against
            EmotionCategory::Anger => vec![
                "BAD".to_string(),
                "FEEL".to_string(),
                "WANT".to_string(),
                "DO".to_string(),
                "OTHER".to_string(),
                "BECAUSE".to_string(),
            ],

            // Sadness: loss, absence of desired state
            EmotionCategory::Sadness => vec![
                "BAD".to_string(),
                "FEEL".to_string(),
                "NOT".to_string(),
                "HAVE".to_string(),
                "WANT".to_string(),
                "BEFORE".to_string(),
            ],

            // Surprise: unexpected event, schema violation
            EmotionCategory::Surprise => vec![
                "FEEL".to_string(),
                "NOT".to_string(),
                "KNOW".to_string(),
                "BEFORE".to_string(),
                "HAPPEN".to_string(),
                "NOW".to_string(),
            ],

            // Neutral: baseline state
            EmotionCategory::Neutral => vec![
                "FEEL".to_string(),
                "NOT".to_string(),
                "VERY".to_string(),
                "NOT".to_string(),
                "GOOD".to_string(),
                "NOT".to_string(),
                "BAD".to_string(),
            ],

            // Disgust: rejection, contamination avoidance
            EmotionCategory::Disgust => vec![
                "BAD".to_string(),
                "FEEL".to_string(),
                "NOT".to_string(),
                "WANT".to_string(),
                "THIS".to_string(),
                "BODY".to_string(),
            ],

            // Anticipation: future-oriented expectation
            EmotionCategory::Anticipation => vec![
                "FEEL".to_string(),
                "WANT".to_string(),
                "KNOW".to_string(),
                "AFTER".to_string(),
                "HAPPEN".to_string(),
                "SOMETHING".to_string(),
            ],

            // Trust: positive social expectation
            EmotionCategory::Trust => vec![
                "GOOD".to_string(),
                "FEEL".to_string(),
                "OTHER".to_string(),
                "CAN".to_string(),
                "DO".to_string(),
                "GOOD".to_string(),
            ],
        }
    }

    /// Encode primitives as HDC vector
    fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
        if primitives.is_empty() {
            return BinaryHV::zero();
        }

        let vectors: Vec<BinaryHV> = primitives
            .iter()
            .map(|name| {
                // Try to get primitive from system by name
                if let Some(p) = system.get(name) {
                    p.encoding
                } else if let Some(p) = system.get(&name.to_lowercase()) {
                    p.encoding
                } else {
                    // Fallback: create deterministic vector from name
                    let seed = name
                        .bytes()
                        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                    BinaryHV::random(seed)
                }
            })
            .collect();

        if vectors.is_empty() {
            return BinaryHV::zero();
        }

        let mut result = vectors[0];
        for v in &vectors[1..] {
            result = BinaryHV::bind(&result, v);
        }
        result
    }

    /// Semantic similarity to another emotion grounding
    pub(crate) fn similarity(&self, other: &Self) -> f64 {
        self.primitive_encoding
            .similarity(&other.primitive_encoding) as f64
    }
}

/// Unified NSM grounding system for affective consciousness
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct AffectiveNSMGrounding {
    /// Grounding for each emotion category
    pub emotion_groundings: HashMap<EmotionCategory, EmotionPrimitiveGrounding>,
    /// Valence axis endpoints (negative to positive)
    pub valence_axis: (BinaryHV, BinaryHV),
    /// Arousal axis endpoints (low to high)
    pub arousal_axis: (BinaryHV, BinaryHV),
}

#[allow(dead_code)]
impl AffectiveNSMGrounding {
    /// Create complete NSM grounding for affective consciousness
    pub(crate) fn new(primitive_system: &PrimitiveSystem) -> Self {
        let mut emotion_groundings = HashMap::new();

        for emotion in EmotionCategory::all() {
            let grounding = EmotionPrimitiveGrounding::new(emotion, primitive_system);
            emotion_groundings.insert(emotion, grounding);
        }

        // Create valence axis from BAD to GOOD
        let bad_vec = Self::encode_simple(&["BAD", "FEEL"], primitive_system);
        let good_vec = Self::encode_simple(&["GOOD", "FEEL"], primitive_system);
        let valence_axis = (bad_vec, good_vec);

        // Create arousal axis from calm to activated
        let low_arousal = Self::encode_simple(&["NOT", "MOVE", "FEEL"], primitive_system);
        let high_arousal = Self::encode_simple(&["VERY", "MOVE", "DO", "FEEL"], primitive_system);
        let arousal_axis = (low_arousal, high_arousal);

        Self {
            emotion_groundings,
            valence_axis,
            arousal_axis,
        }
    }

    fn encode_simple(primitives: &[&str], system: &PrimitiveSystem) -> BinaryHV {
        let vec_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();
        EmotionPrimitiveGrounding::encode_primitives(&vec_primitives, system)
    }

    /// Get grounding for a specific emotion
    pub(crate) fn grounding(
        &self,
        emotion: &EmotionCategory,
    ) -> Option<&EmotionPrimitiveGrounding> {
        self.emotion_groundings.get(emotion)
    }

    /// Find emotions semantically similar to a query vector
    pub(crate) fn find_similar(
        &self,
        query: &BinaryHV,
        threshold: f64,
    ) -> Vec<(&EmotionCategory, f64)> {
        let mut similar: Vec<_> = self
            .emotion_groundings
            .iter()
            .map(|(em, g)| (em, query.similarity(&g.primitive_encoding) as f64))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similar
    }

    /// Semantic distance between two emotions
    pub(crate) fn semantic_distance(&self, e1: &EmotionCategory, e2: &EmotionCategory) -> f64 {
        match (self.grounding(e1), self.grounding(e2)) {
            (Some(g1), Some(g2)) => 1.0 - g1.similarity(g2),
            _ => 1.0,
        }
    }

    /// Project a vector onto the valence dimension
    pub(crate) fn valence_projection(&self, vector: &BinaryHV) -> f64 {
        let neg_sim = vector.similarity(&self.valence_axis.0) as f64;
        let pos_sim = vector.similarity(&self.valence_axis.1) as f64;
        (pos_sim - neg_sim) / 2.0 // Range: -0.5 to 0.5, normalized
    }

    /// Project a vector onto the arousal dimension
    pub(crate) fn arousal_projection(&self, vector: &BinaryHV) -> f64 {
        let low_sim = vector.similarity(&self.arousal_axis.0) as f64;
        let high_sim = vector.similarity(&self.arousal_axis.1) as f64;
        (high_sim - low_sim + 1.0) / 2.0 // Range: 0 to 1
    }
}

/// Configuration for the affective consciousness analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectiveConfig {
    /// How quickly affect decays toward neutral
    pub decay_rate: f32,

    /// Momentum for affect changes
    pub momentum: f32,

    /// History length for affect trajectory
    pub history_size: usize,

    /// Threshold for significant affect change
    pub change_threshold: f32,
}

impl Default for AffectiveConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.1,
            momentum: 0.3,
            history_size: 100,
            change_threshold: 0.2,
        }
    }
}

/// An affective event - something that triggered an emotional response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectiveEvent {
    /// Event description/identifier
    pub description: String,

    /// The affect induced by this event
    pub affect: CoreAffect,

    /// Intensity of the event
    pub intensity: f32,

    /// Timestamp
    pub timestamp: u64,

    /// What triggered this event
    pub trigger: Option<String>,
}

/// The affective consciousness analyzer
#[derive(Debug)]
pub struct AffectiveConsciousnessAnalyzer {
    /// Configuration
    config: AffectiveConfig,

    /// Current affective state
    current_affect: CoreAffect,

    /// Affect velocity (rate of change)
    affect_velocity: CoreAffect,

    /// History of affective states
    affect_history: VecDeque<CoreAffect>,

    /// Recent events
    events: VecDeque<AffectiveEvent>,

    /// Somatic markers: associations between stimuli and affects
    somatic_markers: HashMap<String, CoreAffect>,

    /// Statistics
    stats: AffectiveStats,
}

/// Statistics for the analyzer
#[derive(Debug, Clone, Default)]
pub struct AffectiveStats {
    /// Total updates
    pub updates: u64,

    /// Total events processed
    pub events_processed: u64,

    /// Average valence
    pub avg_valence: f32,

    /// Average arousal
    pub avg_arousal: f32,

    /// Somatic markers learned
    pub markers_learned: usize,
}

impl AffectiveConsciousnessAnalyzer {
    /// Create a new analyzer
    pub fn new(config: AffectiveConfig) -> Self {
        Self {
            config,
            current_affect: CoreAffect::neutral(),
            affect_velocity: CoreAffect::new(0.0, 0.0, 0.0),
            affect_history: VecDeque::new(),
            events: VecDeque::new(),
            somatic_markers: HashMap::new(),
            stats: AffectiveStats::default(),
        }
    }

    /// Process a stimulus and update affective state
    pub fn process_stimulus(
        &mut self,
        stimulus: &str,
        base_affect: Option<CoreAffect>,
    ) -> CoreAffect {
        // Check if we have a somatic marker for this stimulus
        let marker_affect = self.somatic_markers.get(stimulus).copied();

        // Determine induced affect
        let induced = match (marker_affect, base_affect) {
            (Some(m), Some(b)) => m.blend(&b, 0.5),
            (Some(m), None) => m,
            (None, Some(b)) => b,
            (None, None) => CoreAffect::neutral(),
        };

        // Update current affect with momentum
        let target = self.current_affect.blend(&induced, 0.3);
        self.update_affect(target);

        self.current_affect
    }

    /// Update the current affective state
    fn update_affect(&mut self, target: CoreAffect) {
        // Apply momentum
        let new_velocity = CoreAffect::new(
            self.affect_velocity.valence * self.config.momentum
                + (target.valence - self.current_affect.valence) * (1.0 - self.config.momentum),
            self.affect_velocity.arousal * self.config.momentum
                + (target.arousal - self.current_affect.arousal) * (1.0 - self.config.momentum),
            self.affect_velocity.dominance * self.config.momentum
                + (target.dominance - self.current_affect.dominance) * (1.0 - self.config.momentum),
        );
        self.affect_velocity = new_velocity;

        // Update current affect
        self.current_affect = CoreAffect::new(
            self.current_affect.valence + self.affect_velocity.valence,
            self.current_affect.arousal + self.affect_velocity.arousal,
            self.current_affect.dominance + self.affect_velocity.dominance,
        );

        // Record in history
        if self.affect_history.len() >= self.config.history_size {
            self.affect_history.pop_front();
        }
        self.affect_history.push_back(self.current_affect);

        // Update stats
        self.stats.updates += 1;
        let n = self.stats.updates as f32;
        self.stats.avg_valence =
            (self.stats.avg_valence * (n - 1.0) + self.current_affect.valence) / n;
        self.stats.avg_arousal =
            (self.stats.avg_arousal * (n - 1.0) + self.current_affect.arousal) / n;
    }

    /// Record an affective event
    pub fn record_event(&mut self, event: AffectiveEvent) {
        // Process the event's affect
        self.process_stimulus(&event.description, Some(event.affect));

        // Store event
        if self.events.len() >= self.config.history_size {
            self.events.pop_front();
        }
        self.events.push_back(event);

        self.stats.events_processed += 1;
    }

    /// Learn a somatic marker (association between stimulus and affect)
    pub fn learn_marker(&mut self, stimulus: impl Into<String>, affect: CoreAffect) {
        let key = stimulus.into();

        // Update or create marker
        if let Some(existing) = self.somatic_markers.get(&key) {
            // Blend with existing
            let blended = existing.blend(&affect, 0.3);
            self.somatic_markers.insert(key, blended);
        } else {
            self.somatic_markers.insert(key, affect);
            self.stats.markers_learned += 1;
        }
    }

    /// Decay affect toward neutral over time
    pub fn decay(&mut self, dt: f32) {
        let decay_factor = (1.0 - self.config.decay_rate * dt).max(0.0);
        self.current_affect = self
            .current_affect
            .blend(&CoreAffect::neutral(), 1.0 - decay_factor);
    }

    /// Get current affect
    pub fn current_affect(&self) -> CoreAffect {
        self.current_affect
    }

    /// Get current emotion category
    pub fn current_emotion(&self) -> EmotionCategory {
        self.current_affect.closest_emotion()
    }

    /// Get affect trajectory (trend)
    pub fn affect_trend(&self) -> (f32, f32) {
        if self.affect_history.len() < 2 {
            return (0.0, 0.0);
        }

        let recent: Vec<_> = self.affect_history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return (0.0, 0.0);
        }

        let divisor = (recent.len() - 1) as f32;

        let valence_trend: f32 = recent
            .windows(2)
            .map(|w| w[0].valence - w[1].valence)
            .sum::<f32>()
            / divisor;

        let arousal_trend: f32 = recent
            .windows(2)
            .map(|w| w[0].arousal - w[1].arousal)
            .sum::<f32>()
            / divisor;

        // Guard: if any affect values were NaN/Inf, return neutral trend
        let valence_trend = if valence_trend.is_finite() {
            valence_trend
        } else {
            0.0
        };
        let arousal_trend = if arousal_trend.is_finite() {
            arousal_trend
        } else {
            0.0
        };

        (valence_trend, arousal_trend)
    }

    /// Get statistics
    pub fn stats(&self) -> &AffectiveStats {
        &self.stats
    }

    /// Check a somatic marker
    pub fn get_marker(&self, stimulus: &str) -> Option<CoreAffect> {
        self.somatic_markers.get(stimulus).copied()
    }
}

impl Default for AffectiveConsciousnessAnalyzer {
    fn default() -> Self {
        Self::new(AffectiveConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_affect_creation() {
        let affect = CoreAffect::new(0.5, 0.7, 0.0);
        assert!((affect.valence - 0.5).abs() < 0.01);
        assert!((affect.arousal - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_emotion_mapping() {
        let joy = CoreAffect::new(0.8, 0.8, 0.5);
        assert_eq!(joy.closest_emotion(), EmotionCategory::Joy);

        let fear = CoreAffect::new(-0.7, 0.9, -0.7);
        assert_eq!(fear.closest_emotion(), EmotionCategory::Fear);
    }

    #[test]
    fn test_analyzer_processing() {
        let mut analyzer = AffectiveConsciousnessAnalyzer::default();

        let affect = analyzer.process_stimulus("happy event", Some(CoreAffect::new(0.8, 0.6, 0.3)));
        assert!(affect.valence > 0.0);
    }

    #[test]
    fn test_somatic_marker_learning() {
        let mut analyzer = AffectiveConsciousnessAnalyzer::default();

        analyzer.learn_marker("danger", CoreAffect::new(-0.8, 0.9, -0.5));
        let marker = analyzer.get_marker("danger").unwrap();
        assert!(marker.valence < 0.0);
    }
}
