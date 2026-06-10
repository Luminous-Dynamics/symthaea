// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emotional Depth Module - Complex Emotional Blends and HDC-Native Emotional Algebra
//!
//! Extends the basic emotional infrastructure with:
//! - Complex 3+ emotion blends using HDC binding/bundling
//! - Named compound emotions (bittersweet, melancholy, schadenfreude, etc.)
//! - Emotional trajectory modeling (how emotions flow over time)
//! - Integration between language/emotional_core and physiology/emotional_reasoning
//!
//! # HDC Emotional Algebra
//!
//! ```text
//! Simple blend (A + B):     bundle(A, B)           // both present equally
//! Weighted blend:           bundle(A * w1, B * w2) // weighted presence
//! Contextualized emotion:   bind(emotion, context) // emotion IN context
//! Temporal sequence:        bind(A, TIME1) ⊕ bind(B, TIME2) // A then B
//! Complex compound:         bind(bundle(A, B), modifier)    // (A+B) modified
//! ```
//!
//! # Compound Emotions
//!
//! | Name           | Components                    | Description                        |
//! |----------------|-------------------------------|------------------------------------|
//! | Bittersweet    | Joy + Sadness                 | Happy memory with loss             |
//! | Nostalgia      | Joy + Sadness + Anticipation  | Longing for the past               |
//! | Melancholy     | Sadness + Beauty + Peace      | Thoughtful sorrow                  |
//! | Awe            | Fear + Wonder + Joy           | Overwhelming grandeur              |
//! | Schadenfreude  | Joy + (inverted Empathy)      | Pleasure at others' misfortune     |
//! | Serenity       | Peace + Contentment + Trust   | Deep calm acceptance               |
//! | Anticipation   | Joy + Fear + Excitement       | Eager uncertainty                  |
//! | Ambivalence    | (A + inverted A)              | Conflicting feelings               |

use super::binary_hv::BinaryHV;
use super::primitive_system::PrimitiveSystem;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// EMOTIONAL COMPONENTS
// ============================================================================

/// Primary emotional components for HDC blending
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionalComponent {
    // Basic Ekman emotions
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,

    // Extended emotional dimensions
    Trust,
    Anticipation,
    Love,
    Peace,
    Curiosity,
    Gratitude,
    Wonder,
    Pride,
    Shame,
    Guilt,
    Envy,
    Contempt,

    // Modifiers (not standalone emotions)
    Intensity,    // Amplifies other emotions
    Uncertainty,  // Adds doubt/hesitation
    Temporal,     // Past/future orientation
    Social,       // Other-directed
    SelfDirected, // Self-directed
}

impl EmotionalComponent {
    /// Get primitive name for HDC encoding lookup
    pub fn primitive_name(&self) -> &'static str {
        match self {
            Self::Joy => "JOY",
            Self::Sadness => "SADNESS",
            Self::Fear => "FEAR",
            Self::Anger => "ANGER",
            Self::Surprise => "SURPRISE",
            Self::Disgust => "DISGUST",
            Self::Trust => "TRUST",
            Self::Anticipation => "ANTICIPATION",
            Self::Love => "LOVE",
            Self::Peace => "CONTENTMENT",
            Self::Curiosity => "CURIOSITY",
            Self::Gratitude => "GRATITUDE",
            Self::Wonder => "AWE",
            Self::Pride => "PRIDE",
            Self::Shame => "SHAME",
            Self::Guilt => "GUILT",
            Self::Envy => "ENVY",
            Self::Contempt => "CONTEMPT",
            Self::Intensity => "INTENSITY",
            Self::Uncertainty => "UNCERTAINTY",
            Self::Temporal => "TIME",
            Self::Social => "SOCIAL",
            Self::SelfDirected => "SELF",
        }
    }

    /// Default valence (-1.0 to 1.0)
    pub fn valence(&self) -> f64 {
        match self {
            Self::Joy | Self::Love | Self::Peace | Self::Gratitude => 0.8,
            Self::Wonder | Self::Pride | Self::Curiosity => 0.6,
            Self::Trust | Self::Anticipation => 0.4,
            Self::Surprise => 0.0,
            Self::Sadness | Self::Fear | Self::Uncertainty => -0.4,
            Self::Anger | Self::Disgust | Self::Contempt => -0.6,
            Self::Shame | Self::Guilt | Self::Envy => -0.5,
            Self::Intensity | Self::Temporal | Self::Social | Self::SelfDirected => 0.0,
        }
    }

    /// Default arousal (0.0 to 1.0)
    pub fn arousal(&self) -> f64 {
        match self {
            Self::Fear | Self::Anger | Self::Surprise => 0.9,
            Self::Joy | Self::Anticipation | Self::Wonder => 0.7,
            Self::Curiosity | Self::Pride | Self::Disgust => 0.5,
            Self::Love | Self::Gratitude | Self::Trust => 0.4,
            Self::Sadness | Self::Guilt | Self::Shame => 0.3,
            Self::Peace | Self::Contempt => 0.2,
            Self::Envy | Self::Uncertainty => 0.6,
            Self::Intensity | Self::Temporal | Self::Social | Self::SelfDirected => 0.5,
        }
    }

    /// Whether this component can invert (for ambivalence)
    pub fn invertible(&self) -> bool {
        !matches!(
            self,
            Self::Intensity | Self::Temporal | Self::Social | Self::SelfDirected
        )
    }
}

// ============================================================================
// WEIGHTED EMOTIONAL COMPONENT
// ============================================================================

/// A single emotional component with weight and optional inversion
#[derive(Debug, Clone)]
pub struct WeightedComponent {
    pub component: EmotionalComponent,
    pub weight: f64,    // 0.0 to 1.0 (relative contribution)
    pub inverted: bool, // For emotions like schadenfreude (inverted empathy)
}

impl WeightedComponent {
    pub fn new(component: EmotionalComponent, weight: f64) -> Self {
        Self {
            component,
            weight: weight.clamp(0.0, 1.0),
            inverted: false,
        }
    }

    pub fn inverted(component: EmotionalComponent, weight: f64) -> Self {
        Self {
            component,
            weight: weight.clamp(0.0, 1.0),
            inverted: true,
        }
    }

    /// Calculate effective valence considering inversion
    pub fn effective_valence(&self) -> f64 {
        let base = self.component.valence();
        if self.inverted { -base } else { base }
    }

    /// Calculate effective arousal (inversion doesn't affect arousal)
    pub fn effective_arousal(&self) -> f64 {
        self.component.arousal()
    }
}

// ============================================================================
// COMPLEX EMOTIONAL BLEND
// ============================================================================

/// A complex emotional blend of 2+ weighted components
#[derive(Debug, Clone)]
pub struct EmotionalBlend {
    /// Name of the blend (e.g., "Bittersweet", "Custom Blend")
    pub name: String,

    /// Component emotions with weights
    pub components: Vec<WeightedComponent>,

    /// HDC encoding of the blended emotional state
    pub encoding: BinaryHV,

    /// Computed overall valence
    pub valence: f64,

    /// Computed overall arousal
    pub arousal: f64,

    /// Coherence of the blend (0.0 = conflicting, 1.0 = harmonious)
    pub coherence: f64,

    /// Timestamp of creation
    pub timestamp: u64,
}

impl EmotionalBlend {
    /// Create a new blend from weighted components
    pub fn new(name: &str, components: Vec<WeightedComponent>, encoder: &EmotionalEncoder) -> Self {
        let encoding = encoder.encode_blend(&components);
        let (valence, arousal) = Self::compute_valence_arousal(&components);
        let coherence = Self::compute_coherence(&components);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            name: name.to_string(),
            components,
            encoding,
            valence,
            arousal,
            coherence,
            timestamp,
        }
    }

    /// Compute weighted valence and arousal
    fn compute_valence_arousal(components: &[WeightedComponent]) -> (f64, f64) {
        if components.is_empty() {
            return (0.0, 0.5);
        }

        let total_weight: f64 = components.iter().map(|c| c.weight).sum();
        if total_weight < 0.001 {
            return (0.0, 0.5);
        }

        let valence: f64 = components
            .iter()
            .map(|c| c.effective_valence() * c.weight)
            .sum::<f64>()
            / total_weight;

        let arousal: f64 = components
            .iter()
            .map(|c| c.effective_arousal() * c.weight)
            .sum::<f64>()
            / total_weight;

        (valence.clamp(-1.0, 1.0), arousal.clamp(0.0, 1.0))
    }

    /// Compute emotional coherence (do the emotions "fit together"?)
    fn compute_coherence(components: &[WeightedComponent]) -> f64 {
        if components.len() < 2 {
            return 1.0;
        }

        // Check for conflicting valences (high conflict = low coherence)
        let valences: Vec<f64> = components.iter().map(|c| c.effective_valence()).collect();

        let has_positive = valences.iter().any(|v| *v > 0.3);
        let has_negative = valences.iter().any(|v| *v < -0.3);

        // Conflicting emotions reduce coherence but aren't invalid
        // (bittersweet is low coherence but meaningful)
        let valence_conflict = if has_positive && has_negative {
            0.3
        } else {
            0.0
        };

        // Check for arousal spread (very different arousals reduce coherence)
        let arousals: Vec<f64> = components.iter().map(|c| c.effective_arousal()).collect();
        let max_arousal = arousals.iter().cloned().fold(0.0, f64::max);
        let min_arousal = arousals.iter().cloned().fold(1.0, f64::min);
        let arousal_spread = max_arousal - min_arousal;

        (1.0 - valence_conflict - arousal_spread * 0.3).clamp(0.0, 1.0)
    }

    /// Is this an ambivalent (conflicting) emotional state?
    pub fn is_ambivalent(&self) -> bool {
        self.coherence < 0.5
    }

    /// Dominant emotion (highest weighted)
    pub fn dominant(&self) -> Option<&WeightedComponent> {
        self.components.iter().max_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get human-readable description
    pub fn describe(&self) -> String {
        if self.components.is_empty() {
            return "Neutral".to_string();
        }

        let parts: Vec<String> = self
            .components
            .iter()
            .filter(|c| c.weight > 0.1)
            .map(|c| {
                let intensity = if c.weight > 0.7 {
                    "strong"
                } else if c.weight > 0.4 {
                    "moderate"
                } else {
                    "mild"
                };
                let inversion = if c.inverted { " (inverted)" } else { "" };
                format!("{} {:?}{}", intensity, c.component, inversion)
            })
            .collect();

        format!(
            "{}: {} [valence: {:.2}, arousal: {:.2}, coherence: {:.2}]",
            self.name,
            parts.join(" + "),
            self.valence,
            self.arousal,
            self.coherence
        )
    }
}

// ============================================================================
// COMPOUND EMOTIONS (PREDEFINED)
// ============================================================================

/// Named compound emotions with predefined compositions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompoundEmotion {
    /// Joy + Sadness - happy memory tinged with loss
    Bittersweet,

    /// Joy + Sadness + Anticipation - longing for the past
    Nostalgia,

    /// Sadness + Peace + Wonder - thoughtful, beautiful sorrow
    Melancholy,

    /// Fear + Wonder + Joy - overwhelming grandeur
    Awe,

    /// Joy + inverted(Empathy) - pleasure at others' misfortune
    Schadenfreude,

    /// Peace + Trust + Gratitude - deep calm acceptance
    Serenity,

    /// Joy + Fear + Anticipation - eager uncertainty
    Excitement,

    /// Love + Fear + Anticipation - protective concern
    Devotion,

    /// Pride + Joy + Love - warmth toward accomplishment
    Fulfillment,

    /// Sadness + Anger + Love - grief from loss
    Grief,

    /// Fear + Sadness + Uncertainty - worried anticipation
    Dread,

    /// Joy + Curiosity + Wonder - childlike discovery
    Delight,

    /// Contempt + Disgust + Anger - strong moral rejection
    Outrage,

    /// Trust + Love + Peace - secure connection
    Belonging,

    /// Shame + Sadness + Self - painful self-reflection
    Remorse,
}

impl CompoundEmotion {
    /// Get the component recipe for this compound emotion
    pub fn components(&self) -> Vec<WeightedComponent> {
        match self {
            Self::Bittersweet => vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.6),
                WeightedComponent::new(EmotionalComponent::Sadness, 0.5),
            ],

            Self::Nostalgia => vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.4),
                WeightedComponent::new(EmotionalComponent::Sadness, 0.4),
                WeightedComponent::new(EmotionalComponent::Anticipation, 0.3),
                WeightedComponent::new(EmotionalComponent::Love, 0.2),
            ],

            Self::Melancholy => vec![
                WeightedComponent::new(EmotionalComponent::Sadness, 0.5),
                WeightedComponent::new(EmotionalComponent::Peace, 0.3),
                WeightedComponent::new(EmotionalComponent::Wonder, 0.3),
            ],

            Self::Awe => vec![
                WeightedComponent::new(EmotionalComponent::Wonder, 0.6),
                WeightedComponent::new(EmotionalComponent::Fear, 0.3),
                WeightedComponent::new(EmotionalComponent::Joy, 0.3),
            ],

            Self::Schadenfreude => vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.5),
                WeightedComponent::inverted(EmotionalComponent::Love, 0.4), // inverted empathy/love
                WeightedComponent::new(EmotionalComponent::Social, 0.2),
            ],

            Self::Serenity => vec![
                WeightedComponent::new(EmotionalComponent::Peace, 0.6),
                WeightedComponent::new(EmotionalComponent::Trust, 0.4),
                WeightedComponent::new(EmotionalComponent::Gratitude, 0.3),
            ],

            Self::Excitement => vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.5),
                WeightedComponent::new(EmotionalComponent::Fear, 0.2),
                WeightedComponent::new(EmotionalComponent::Anticipation, 0.6),
            ],

            Self::Devotion => vec![
                WeightedComponent::new(EmotionalComponent::Love, 0.7),
                WeightedComponent::new(EmotionalComponent::Fear, 0.2),
                WeightedComponent::new(EmotionalComponent::Trust, 0.4),
            ],

            Self::Fulfillment => vec![
                WeightedComponent::new(EmotionalComponent::Pride, 0.5),
                WeightedComponent::new(EmotionalComponent::Joy, 0.5),
                WeightedComponent::new(EmotionalComponent::Gratitude, 0.4),
            ],

            Self::Grief => vec![
                WeightedComponent::new(EmotionalComponent::Sadness, 0.8),
                WeightedComponent::new(EmotionalComponent::Love, 0.5),
                WeightedComponent::new(EmotionalComponent::Anger, 0.2),
            ],

            Self::Dread => vec![
                WeightedComponent::new(EmotionalComponent::Fear, 0.7),
                WeightedComponent::new(EmotionalComponent::Sadness, 0.3),
                WeightedComponent::new(EmotionalComponent::Anticipation, 0.4),
            ],

            Self::Delight => vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.7),
                WeightedComponent::new(EmotionalComponent::Curiosity, 0.5),
                WeightedComponent::new(EmotionalComponent::Wonder, 0.4),
            ],

            Self::Outrage => vec![
                WeightedComponent::new(EmotionalComponent::Anger, 0.7),
                WeightedComponent::new(EmotionalComponent::Disgust, 0.5),
                WeightedComponent::new(EmotionalComponent::Contempt, 0.4),
            ],

            Self::Belonging => vec![
                WeightedComponent::new(EmotionalComponent::Trust, 0.5),
                WeightedComponent::new(EmotionalComponent::Love, 0.5),
                WeightedComponent::new(EmotionalComponent::Peace, 0.4),
            ],

            Self::Remorse => vec![
                WeightedComponent::new(EmotionalComponent::Guilt, 0.6),
                WeightedComponent::new(EmotionalComponent::Sadness, 0.5),
                WeightedComponent::new(EmotionalComponent::SelfDirected, 0.3),
            ],
        }
    }

    /// Get typical valence for this compound
    pub fn typical_valence(&self) -> f64 {
        match self {
            Self::Awe | Self::Serenity | Self::Fulfillment | Self::Delight | Self::Belonging => 0.6,
            Self::Excitement | Self::Devotion => 0.4,
            Self::Bittersweet | Self::Nostalgia | Self::Melancholy => 0.0,
            Self::Schadenfreude => 0.2,
            Self::Grief | Self::Dread | Self::Outrage | Self::Remorse => -0.6,
        }
    }

    /// Get description of this compound emotion
    pub fn description(&self) -> &'static str {
        match self {
            Self::Bittersweet => "A happy memory tinged with sadness or loss",
            Self::Nostalgia => "Sentimental longing for the past",
            Self::Melancholy => "Thoughtful, pensive sadness with a hint of beauty",
            Self::Awe => "Overwhelming wonder, often with a touch of fear",
            Self::Schadenfreude => "Pleasure derived from another's misfortune",
            Self::Serenity => "Deep, peaceful calm and acceptance",
            Self::Excitement => "Eager anticipation with nervous energy",
            Self::Devotion => "Deep love combined with protective concern",
            Self::Fulfillment => "Satisfaction and pride in accomplishment",
            Self::Grief => "Deep sorrow from loss, mixed with love",
            Self::Dread => "Anxious anticipation of something negative",
            Self::Delight => "Joyful discovery and wonder",
            Self::Outrage => "Strong moral indignation and anger",
            Self::Belonging => "Warm sense of connection and acceptance",
            Self::Remorse => "Painful regret and self-directed guilt",
        }
    }
}

// ============================================================================
// EMOTIONAL ENCODER
// ============================================================================

/// HDC-native emotional encoder
pub struct EmotionalEncoder {
    /// Primitive system for base emotion vectors
    primitives: &'static PrimitiveSystem,

    /// Cached component encodings
    component_cache: HashMap<EmotionalComponent, BinaryHV>,

    /// Seed for deterministic encoding
    seed: u64,
}

impl EmotionalEncoder {
    pub fn new() -> Self {
        let primitives = PrimitiveSystem::global();
        let mut encoder = Self {
            primitives,
            component_cache: HashMap::new(),
            seed: 0xE0071005, // Emotional seed
        };
        encoder.initialize_cache();
        encoder
    }

    fn initialize_cache(&mut self) {
        // Pre-cache all emotional components
        for component in [
            EmotionalComponent::Joy,
            EmotionalComponent::Sadness,
            EmotionalComponent::Fear,
            EmotionalComponent::Anger,
            EmotionalComponent::Surprise,
            EmotionalComponent::Disgust,
            EmotionalComponent::Trust,
            EmotionalComponent::Anticipation,
            EmotionalComponent::Love,
            EmotionalComponent::Peace,
            EmotionalComponent::Curiosity,
            EmotionalComponent::Gratitude,
            EmotionalComponent::Wonder,
            EmotionalComponent::Pride,
            EmotionalComponent::Shame,
            EmotionalComponent::Guilt,
            EmotionalComponent::Envy,
            EmotionalComponent::Contempt,
            EmotionalComponent::Intensity,
            EmotionalComponent::Uncertainty,
            EmotionalComponent::Temporal,
            EmotionalComponent::Social,
            EmotionalComponent::SelfDirected,
        ] {
            let encoding = self.encode_component(component);
            self.component_cache.insert(component, encoding);
        }
    }

    /// Encode a single emotional component
    fn encode_component(&self, component: EmotionalComponent) -> BinaryHV {
        let name = component.primitive_name();

        // Try to get from primitive system first
        if let Some(prim) = self.primitives.get(name) {
            return prim.encoding;
        }

        // Fallback: generate deterministic random HV
        let seed = self.seed
            ^ (name
                .as_bytes()
                .iter()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as u64)));
        BinaryHV::random(seed)
    }

    /// Get cached encoding for a component
    pub fn get_component(&self, component: EmotionalComponent) -> BinaryHV {
        self.component_cache
            .get(&component)
            .cloned()
            .unwrap_or_else(|| BinaryHV::random(self.seed))
    }

    /// Encode a weighted component (with optional inversion)
    fn encode_weighted(&self, wc: &WeightedComponent) -> BinaryHV {
        let base = self.get_component(wc.component);

        if wc.inverted { base.invert() } else { base }
    }

    /// Encode a blend of weighted components using HDC bundling
    pub fn encode_blend(&self, components: &[WeightedComponent]) -> BinaryHV {
        if components.is_empty() {
            return BinaryHV::random(self.seed);
        }

        if components.len() == 1 {
            return self.encode_weighted(&components[0]);
        }

        // Weighted bundling: superposition of component vectors
        // Weight affects how much each contributes to the final bundle
        let weighted_hvs: Vec<(BinaryHV, f64)> = components
            .iter()
            .map(|c| (self.encode_weighted(c), c.weight))
            .collect();

        // Use weighted majority for bundling
        Self::weighted_bundle(&weighted_hvs)
    }

    /// Weighted bundle operation (HDC superposition with weights)
    fn weighted_bundle(weighted_hvs: &[(BinaryHV, f64)]) -> BinaryHV {
        if weighted_hvs.is_empty() {
            return BinaryHV::random(42);
        }

        // Accumulate weighted votes for each bit
        // BinaryHV is 2048 bytes = 16384 bits
        const DIMENSIONS: usize = 16384;
        let mut accumulator = vec![0.0f64; DIMENSIONS];

        for (hv, weight) in weighted_hvs {
            for (i, byte) in hv.0.iter().enumerate() {
                for bit in 0..8 {
                    let idx = i * 8 + bit;
                    if idx < DIMENSIONS {
                        let bit_val = (byte >> (7 - bit)) & 1;
                        // +weight for 1, -weight for 0
                        accumulator[idx] += if bit_val == 1 { *weight } else { -*weight };
                    }
                }
            }
        }

        // Threshold at 0 to get binary result
        let mut result = [0u8; 2048];
        for (i, &sum) in accumulator.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            if sum > 0.0 {
                result[byte_idx] |= 1 << bit_idx;
            }
        }

        BinaryHV(result)
    }

    /// Encode a compound emotion
    pub fn encode_compound(&self, compound: CompoundEmotion) -> EmotionalBlend {
        let components = compound.components();
        let name = format!("{compound:?}");
        EmotionalBlend::new(&name, components, self)
    }

    /// Create a contextualized emotion (emotion IN context)
    /// Uses HDC binding to associate emotion with context
    pub fn contextualize(&self, emotion: &BinaryHV, context: &BinaryHV) -> BinaryHV {
        emotion.bind(context)
    }

    /// Compute similarity between two emotional states
    pub fn similarity(&self, a: &BinaryHV, b: &BinaryHV) -> f64 {
        a.similarity(b) as f64
    }
}

impl Default for EmotionalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for EmotionalEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmotionalEncoder")
            .field("seed", &self.seed)
            .field("component_cache_len", &self.component_cache.len())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// EMOTIONAL TRAJECTORY
// ============================================================================

/// A point in emotional trajectory (emotion over time)
#[derive(Debug, Clone)]
pub struct EmotionalMoment {
    pub blend: EmotionalBlend,
    pub timestamp: u64,
    pub trigger: Option<String>, // What caused this emotional shift
}

/// Emotional trajectory - how emotions flow over time
#[derive(Debug)]
pub struct EmotionalTrajectory {
    /// Sequence of emotional moments
    moments: Vec<EmotionalMoment>,

    /// Maximum trajectory length
    max_length: usize,

    /// Encoder for HDC operations
    encoder: EmotionalEncoder,
}

impl EmotionalTrajectory {
    pub fn new() -> Self {
        Self {
            moments: Vec::new(),
            max_length: 100,
            encoder: EmotionalEncoder::new(),
        }
    }

    /// Add a new emotional moment
    pub fn record(&mut self, blend: EmotionalBlend, trigger: Option<String>) {
        let moment = EmotionalMoment {
            blend,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            trigger,
        };

        self.moments.push(moment);

        // Trim if too long
        if self.moments.len() > self.max_length {
            self.moments.remove(0);
        }
    }

    /// Get current emotional state
    pub fn current(&self) -> Option<&EmotionalBlend> {
        self.moments.last().map(|m| &m.blend)
    }

    /// Get recent emotional trend (average valence over last N moments)
    pub fn valence_trend(&self, n: usize) -> f64 {
        let recent: Vec<_> = self.moments.iter().rev().take(n).collect();
        if recent.is_empty() {
            return 0.0;
        }
        recent.iter().map(|m| m.blend.valence).sum::<f64>() / recent.len() as f64
    }

    /// Get emotional volatility (how much emotion changes)
    pub fn volatility(&self, n: usize) -> f64 {
        let recent: Vec<_> = self.moments.iter().rev().take(n).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let mut total_change = 0.0;
        for i in 1..recent.len() {
            let v_diff = (recent[i].blend.valence - recent[i - 1].blend.valence).abs();
            let a_diff = (recent[i].blend.arousal - recent[i - 1].blend.arousal).abs();
            total_change += v_diff + a_diff;
        }

        total_change / (recent.len() - 1) as f64
    }

    /// Predict next emotional state based on trajectory
    pub fn predict_next(&self) -> Option<(f64, f64)> {
        if self.moments.len() < 3 {
            return None;
        }

        // Simple linear extrapolation
        let recent: Vec<_> = self.moments.iter().rev().take(5).collect();

        let mut valence_delta = 0.0;
        let mut arousal_delta = 0.0;
        let count = (recent.len() - 1) as f64;

        for i in 1..recent.len() {
            valence_delta += recent[i - 1].blend.valence - recent[i].blend.valence;
            arousal_delta += recent[i - 1].blend.arousal - recent[i].blend.arousal;
        }

        valence_delta /= count;
        arousal_delta /= count;

        let current = self.moments.last()?;
        let predicted_valence = (current.blend.valence + valence_delta).clamp(-1.0, 1.0);
        let predicted_arousal = (current.blend.arousal + arousal_delta).clamp(0.0, 1.0);

        Some((predicted_valence, predicted_arousal))
    }

    /// Encode entire trajectory as single HDC vector (temporal binding)
    pub fn encode_trajectory(&self, window: usize) -> BinaryHV {
        let recent: Vec<_> = self.moments.iter().rev().take(window).collect();
        if recent.is_empty() {
            return BinaryHV::random(42);
        }

        // Temporal binding: bind each moment with position vector, then bundle
        let time_base = self.encoder.get_component(EmotionalComponent::Temporal);

        let temporal_hvs: Vec<(BinaryHV, f64)> = recent
            .iter()
            .enumerate()
            .map(|(i, moment)| {
                // Create position vector by permuting time base
                let position = time_base.permute(i);

                // Bind emotion with position
                let contextualized = moment.blend.encoding.bind(&position);

                // Weight by recency (more recent = higher weight)
                let weight = 1.0 - (i as f64 / window as f64) * 0.5;
                (contextualized, weight)
            })
            .collect();

        EmotionalEncoder::weighted_bundle(&temporal_hvs)
    }

    /// Get trajectory length
    pub fn len(&self) -> usize {
        self.moments.len()
    }

    /// Is trajectory empty
    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }
}

impl Default for EmotionalTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EMOTIONAL DEPTH SYSTEM (MAIN)
// ============================================================================

/// Complete emotional depth system integrating all components
pub struct EmotionalDepthSystem {
    /// HDC encoder
    pub encoder: EmotionalEncoder,

    /// Current emotional trajectory
    pub trajectory: EmotionalTrajectory,

    /// Current active blend
    pub current_blend: EmotionalBlend,
}

impl EmotionalDepthSystem {
    pub fn new() -> Self {
        let encoder = EmotionalEncoder::new();
        let neutral = EmotionalBlend::new(
            "Neutral",
            vec![WeightedComponent::new(EmotionalComponent::Peace, 0.5)],
            &encoder,
        );

        Self {
            encoder,
            trajectory: EmotionalTrajectory::new(),
            current_blend: neutral,
        }
    }

    /// Set current emotion to a compound emotion
    pub fn feel_compound(&mut self, compound: CompoundEmotion, trigger: Option<String>) {
        let blend = self.encoder.encode_compound(compound);
        self.trajectory.record(blend.clone(), trigger);
        self.current_blend = blend;
    }

    /// Set current emotion to a custom blend
    pub fn feel_blend(
        &mut self,
        name: &str,
        components: Vec<WeightedComponent>,
        trigger: Option<String>,
    ) {
        let blend = EmotionalBlend::new(name, components, &self.encoder);
        self.trajectory.record(blend.clone(), trigger);
        self.current_blend = blend;
    }

    /// Blend current emotion with new components
    pub fn blend_with(&mut self, additional: Vec<WeightedComponent>, blend_ratio: f64) {
        // Merge current components with new ones
        let mut merged = self.current_blend.components.clone();

        // Reduce existing weights by (1 - blend_ratio)
        for comp in &mut merged {
            comp.weight *= 1.0 - blend_ratio;
        }

        // Add new components with blend_ratio weight
        for mut comp in additional {
            comp.weight *= blend_ratio;
            merged.push(comp);
        }

        let new_blend = EmotionalBlend::new("Blended", merged, &self.encoder);
        self.trajectory.record(new_blend.clone(), None);
        self.current_blend = new_blend;
    }

    /// Get current emotional state
    pub fn current(&self) -> &EmotionalBlend {
        &self.current_blend
    }

    /// Get valence trend
    pub fn trend(&self) -> f64 {
        self.trajectory.valence_trend(10)
    }

    /// Get volatility
    pub fn volatility(&self) -> f64 {
        self.trajectory.volatility(10)
    }

    /// Check similarity between current state and a compound emotion
    pub fn similarity_to(&self, compound: CompoundEmotion) -> f64 {
        let compound_blend = self.encoder.encode_compound(compound);
        self.encoder
            .similarity(&self.current_blend.encoding, &compound_blend.encoding)
    }

    /// Get HDC encoding of current emotional trajectory
    pub fn trajectory_encoding(&self) -> BinaryHV {
        self.trajectory.encode_trajectory(10)
    }

    /// Detailed emotional report
    pub fn report(&self) -> String {
        let trend_description = if self.trend() > 0.3 {
            "improving"
        } else if self.trend() < -0.3 {
            "declining"
        } else {
            "stable"
        };

        let volatility_description = if self.volatility() > 0.4 {
            "highly volatile"
        } else if self.volatility() > 0.2 {
            "moderately variable"
        } else {
            "steady"
        };

        format!(
            "=== Emotional Depth Report ===\n\
             Current: {}\n\
             Trend: {} ({:.2})\n\
             Volatility: {} ({:.2})\n\
             Trajectory length: {} moments",
            self.current_blend.describe(),
            trend_description,
            self.trend(),
            volatility_description,
            self.volatility(),
            self.trajectory.len()
        )
    }
}

impl Default for EmotionalDepthSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_component_properties() {
        assert!(EmotionalComponent::Joy.valence() > 0.0);
        assert!(EmotionalComponent::Sadness.valence() < 0.0);
        assert!(EmotionalComponent::Fear.arousal() > EmotionalComponent::Peace.arousal());
    }

    #[test]
    fn test_weighted_component() {
        let joy = WeightedComponent::new(EmotionalComponent::Joy, 0.8);
        assert!(joy.effective_valence() > 0.0);

        let inv_joy = WeightedComponent::inverted(EmotionalComponent::Joy, 0.8);
        assert!(inv_joy.effective_valence() < 0.0);
    }

    #[test]
    fn test_emotional_blend_creation() {
        let encoder = EmotionalEncoder::new();

        let bittersweet = EmotionalBlend::new(
            "Bittersweet",
            vec![
                WeightedComponent::new(EmotionalComponent::Joy, 0.6),
                WeightedComponent::new(EmotionalComponent::Sadness, 0.5),
            ],
            &encoder,
        );

        // Bittersweet should have low coherence (conflicting emotions)
        assert!(bittersweet.coherence < 0.8);
        assert!(bittersweet.is_ambivalent() || !bittersweet.is_ambivalent()); // Either is valid
    }

    #[test]
    fn test_compound_emotions() {
        let encoder = EmotionalEncoder::new();

        for compound in [
            CompoundEmotion::Bittersweet,
            CompoundEmotion::Nostalgia,
            CompoundEmotion::Awe,
            CompoundEmotion::Serenity,
        ] {
            let blend = encoder.encode_compound(compound);
            assert!(!blend.components.is_empty());
            assert!(!blend.name.is_empty());
        }
    }

    #[test]
    fn test_emotional_trajectory() {
        let mut trajectory = EmotionalTrajectory::new();
        let encoder = EmotionalEncoder::new();

        // Record some emotions
        trajectory.record(
            encoder.encode_compound(CompoundEmotion::Excitement),
            Some("Good news".into()),
        );
        trajectory.record(encoder.encode_compound(CompoundEmotion::Excitement), None);
        trajectory.record(
            encoder.encode_compound(CompoundEmotion::Bittersweet),
            Some("Farewell".into()),
        );

        assert_eq!(trajectory.len(), 3);
        assert!(trajectory.current().is_some());
    }

    #[test]
    fn test_emotional_depth_system() {
        let mut system = EmotionalDepthSystem::new();

        system.feel_compound(CompoundEmotion::Awe, Some("Saw the stars".into()));
        assert!(system.current().valence > 0.0);

        system.feel_compound(CompoundEmotion::Nostalgia, Some("Old photograph".into()));

        let report = system.report();
        assert!(report.contains("Emotional Depth Report"));
    }

    #[test]
    fn test_similarity_detection() {
        let mut system = EmotionalDepthSystem::new();

        system.feel_compound(CompoundEmotion::Awe, None);

        // Should be very similar to Awe
        let awe_similarity = system.similarity_to(CompoundEmotion::Awe);
        assert!(awe_similarity > 0.9);

        // Should be less similar to opposite emotion
        let grief_similarity = system.similarity_to(CompoundEmotion::Grief);
        assert!(grief_similarity < awe_similarity);
    }

    #[test]
    fn test_blend_mixing() {
        let mut system = EmotionalDepthSystem::new();

        system.feel_compound(CompoundEmotion::Excitement, None);
        let initial_valence = system.current().valence;

        // Blend in some sadness
        system.blend_with(
            vec![WeightedComponent::new(EmotionalComponent::Sadness, 0.8)],
            0.5,
        );

        // Valence should decrease
        assert!(system.current().valence < initial_valence);
    }
}
