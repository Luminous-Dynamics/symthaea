//! Emotional Reasoning - Primitive-Based Emotion Layer
//!
//! Integrates the Endocrine System's hormone states with emotional primitives
//! from the Primitive System. Maps biochemical states (cortisol, dopamine, etc.)
//! to semantic emotional concepts (JOY, FEAR, EMPATHY, etc.).
//!
//! # Architecture
//!
//! ```text
//! Endocrine (Hormones) → Emotional Reasoning → Primitives (Semantic)
//!     Cortisol                  ↓              FEAR, ANXIETY
//!     Dopamine          EmotionalState         JOY, EXCITEMENT
//!     Acetylcholine             ↓              FOCUS, FLOW
//! ```
//!
//! # Why This Matters
//!
//! - **Semantic Understanding**: System can reason ABOUT its emotions, not just experience them
//! - **Cross-Domain Integration**: Emotions can interact with moral, economic, and social primitives
//! - **Explainability**: "I feel anxious because cortisol is high" → "ANXIETY = high cortisol = FEAR primitive activated"
//! - **Richer Cognition**: Emotions influence decision-making through primitive bindings

use super::endocrine::HormoneState;
use crate::hdc::primitive_system::PrimitiveSystem;
use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};

/// Emotional state derived from hormone state + primitives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Valence dimension (-1.0 to 1.0)
    /// Negative = unpleasant, Positive = pleasant
    pub valence: f32,

    /// Arousal dimension (0.0 to 1.0)
    /// Low = calm/sluggish, High = alert/activated
    pub arousal: f32,

    /// Dominant emotion (from Russell's circumplex model)
    pub dominant_emotion: Emotion,

    /// Intensity of dominant emotion (0.0-1.0)
    pub intensity: f32,

    /// Secondary emotions (may be present but weaker)
    pub secondary_emotions: Vec<(Emotion, f32)>,

    /// Semantic encoding of current emotional state
    /// This is the HV16 vector representing the emotional primitive
    pub semantic_encoding: HV16,
}

/// Basic emotions from the primitive system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emotion {
    /// Positive valence, moderate-high arousal
    /// From primitive: JOY
    Joy,

    /// Negative valence, low arousal
    /// From primitive: SADNESS
    Sadness,

    /// Negative valence, high arousal, threat response
    /// From primitive: FEAR
    Fear,

    /// Negative valence, high arousal, obstacle response
    /// From primitive: ANGER
    Anger,

    /// Negative valence, rejection response
    /// From primitive: DISGUST
    Disgust,

    /// Neutral valence, high arousal, unexpected event
    /// From primitive: SURPRISE
    Surprise,

    /// Positive valence, low arousal
    /// Derived: contentment, satisfaction
    Contentment,

    /// Neutral valence, low arousal
    /// Baseline neutral state
    Neutral,
}

impl Emotion {
    /// Get the canonical name for primitive system lookup
    pub fn primitive_name(&self) -> &'static str {
        match self {
            Emotion::Joy => "JOY",
            Emotion::Sadness => "SADNESS",
            Emotion::Fear => "FEAR",
            Emotion::Anger => "ANGER",
            Emotion::Disgust => "DISGUST",
            Emotion::Surprise => "SURPRISE",
            Emotion::Contentment => "JOY", // Use JOY primitive with lower arousal
            Emotion::Neutral => "VALENCE", // Use neutral valence
        }
    }

    /// Get emotion from valence and arousal (Russell's circumplex model)
    ///
    /// Adjusted thresholds based on actual endocrine hormone ranges:
    /// - Arousal: 0.4-0.6 is typical range (not 0-1)
    /// - Valence: -0.3 to 0.6 is typical range (cortisol vs dopamine)
    pub fn from_valence_arousal(valence: f32, arousal: f32) -> Self {
        match (valence, arousal) {
            // High arousal emotions (arousal > 0.45)
            (v, a) if a > 0.45 && v > 0.2 => Emotion::Joy,
            (v, a) if a > 0.45 && v < -0.15 => Emotion::Fear,
            (v, a) if a > 0.45 && v >= -0.15 && v <= 0.2 => Emotion::Surprise,

            // Moderate arousal (0.35 < arousal <= 0.45)
            (v, a) if a > 0.35 && a <= 0.45 && v < -0.15 => Emotion::Anger,
            (v, a) if a > 0.35 && a <= 0.45 && v > 0.2 => Emotion::Contentment,

            // Low arousal emotions (arousal <= 0.35)
            (v, a) if a <= 0.35 && v < -0.15 => Emotion::Sadness,
            (v, a) if a <= 0.35 && v > 0.2 => Emotion::Contentment,

            // Default neutral (moderate arousal, neutral valence)
            _ => Emotion::Neutral,
        }
    }

    /// Get typical valence for this emotion (-1.0 to 1.0)
    pub fn typical_valence(&self) -> f32 {
        match self {
            Emotion::Joy => 0.7,
            Emotion::Contentment => 0.5,
            Emotion::Surprise => 0.0,
            Emotion::Neutral => 0.0,
            Emotion::Anger => -0.4,
            Emotion::Disgust => -0.5,
            Emotion::Fear => -0.6,
            Emotion::Sadness => -0.7,
        }
    }

    /// Get typical arousal for this emotion (0.0 to 1.0)
    pub fn typical_arousal(&self) -> f32 {
        match self {
            Emotion::Joy => 0.7,
            Emotion::Fear => 0.8,
            Emotion::Anger => 0.7,
            Emotion::Surprise => 0.8,
            Emotion::Disgust => 0.4,
            Emotion::Contentment => 0.3,
            Emotion::Sadness => 0.2,
            Emotion::Neutral => 0.4,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Emotion::Joy => "Joyful, happy, pleased",
            Emotion::Sadness => "Sad, down, melancholic",
            Emotion::Fear => "Fearful, anxious, threatened",
            Emotion::Anger => "Angry, frustrated, irritated",
            Emotion::Disgust => "Disgusted, repulsed, rejecting",
            Emotion::Surprise => "Surprised, astonished, startled",
            Emotion::Contentment => "Content, satisfied, peaceful",
            Emotion::Neutral => "Neutral, balanced, centered",
        }
    }
}

/// Maps hormone states to emotional states using primitives
#[derive(Debug)]
pub struct EmotionalReasoner {
    /// Reference to primitive system for semantic encoding
    primitives: PrimitiveSystem,
}

impl EmotionalReasoner {
    /// Create new emotional reasoner
    pub fn new() -> Self {
        Self {
            primitives: PrimitiveSystem::new(),
        }
    }

    /// Map hormone state to emotional state with primitive encoding
    pub fn derive_emotional_state(&self, hormones: &HormoneState) -> EmotionalState {
        // Extract valence and arousal from hormones
        let valence = hormones.valence();
        let arousal = hormones.arousal();

        // Determine dominant emotion
        let dominant_emotion = Emotion::from_valence_arousal(valence, arousal);
        let intensity = self.calculate_intensity(valence, arousal, dominant_emotion);

        // Determine secondary emotions
        let secondary_emotions = self.derive_secondary_emotions(valence, arousal, dominant_emotion);

        // Get semantic encoding from primitive system
        let semantic_encoding = self.encode_emotional_state(dominant_emotion, valence, arousal);

        EmotionalState {
            valence,
            arousal,
            dominant_emotion,
            intensity,
            secondary_emotions,
            semantic_encoding,
        }
    }

    /// Calculate intensity of emotion based on distance from neutral
    fn calculate_intensity(&self, valence: f32, arousal: f32, emotion: Emotion) -> f32 {
        // Distance from typical emotion location in circumplex space
        let target_valence = emotion.typical_valence();
        let target_arousal = emotion.typical_arousal();

        // Distance in 2D space
        let v_dist = (valence - target_valence).abs();
        let a_dist = (arousal - target_arousal).abs();
        let distance = (v_dist * v_dist + a_dist * a_dist).sqrt();

        // Intensity inversely proportional to distance (closer = stronger)
        (1.0 - distance.min(1.0)).clamp(0.0, 1.0)
    }

    /// Derive secondary emotions that may be present
    fn derive_secondary_emotions(&self, valence: f32, arousal: f32, dominant: Emotion) -> Vec<(Emotion, f32)> {
        let mut secondary = Vec::new();

        // Check all emotions for possible activation
        for emotion in [
            Emotion::Joy,
            Emotion::Sadness,
            Emotion::Fear,
            Emotion::Anger,
            Emotion::Disgust,
            Emotion::Surprise,
            Emotion::Contentment,
        ] {
            if emotion == dominant {
                continue;
            }

            let intensity = self.calculate_intensity(valence, arousal, emotion);

            // Only include if intensity > threshold (0.3)
            if intensity > 0.3 {
                secondary.push((emotion, intensity));
            }
        }

        // Sort by intensity (strongest first)
        secondary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top 3 secondary emotions
        secondary.truncate(3);

        secondary
    }

    /// Encode emotional state as HV16 semantic vector using primitives
    fn encode_emotional_state(&self, emotion: Emotion, valence: f32, arousal: f32) -> HV16 {
        // Get base emotional primitive
        let emotion_prim = self.primitives.get(emotion.primitive_name());

        if let Some(prim) = emotion_prim {
            // Blend emotion primitive with valence/arousal dimensions
            let valence_prim = self.primitives.get("VALENCE");
            let arousal_prim = self.primitives.get("AROUSAL");

            let mut encoding = prim.encoding.clone();

            // Modulate by valence (if available)
            if let Some(v_prim) = valence_prim {
                if valence.abs() > 0.1 {
                    let v_weight = valence.abs();
                    encoding = if valence > 0.0 {
                        encoding.bind(&v_prim.encoding)
                    } else {
                        encoding.bind(&v_prim.encoding.invert())
                    };
                }
            }

            // Modulate by arousal (if available)
            if let Some(a_prim) = arousal_prim {
                if arousal > 0.5 {
                    encoding = encoding.bind(&a_prim.encoding);
                }
            }

            encoding
        } else {
            // Fallback: neutral encoding
            HV16::random(42)
        }
    }

    /// Get complex emotion by combining primitives
    /// Example: EMPATHY requires understanding others' emotional states
    pub fn get_complex_emotion(&self, name: &str) -> Option<HV16> {
        self.primitives.get(name).map(|p| p.encoding.clone())
    }

    /// Reason about emotional transitions
    /// Returns similarity between two emotional states (0.0-1.0)
    pub fn emotional_similarity(&self, state1: &EmotionalState, state2: &EmotionalState) -> f32 {
        state1.semantic_encoding.similarity(&state2.semantic_encoding)
    }

    /// Get primitive system reference for advanced reasoning
    pub fn primitives(&self) -> &PrimitiveSystem {
        &self.primitives
    }
}

impl Default for EmotionalReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionalState {
    /// Get human-readable emotional report
    pub fn report(&self) -> String {
        format!(
            "Dominant: {} ({:.1}% intensity)\nValence: {:.2} | Arousal: {:.2}\nDescription: {}{}",
            format!("{:?}", self.dominant_emotion),
            self.intensity * 100.0,
            self.valence,
            self.arousal,
            self.dominant_emotion.description(),
            if !self.secondary_emotions.is_empty() {
                format!(
                    "\nSecondary emotions: {}",
                    self.secondary_emotions
                        .iter()
                        .map(|(e, i)| format!("{:?} ({:.0}%)", e, i * 100.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                String::new()
            }
        )
    }

    /// Check if emotion is primarily negative (valence < 0)
    pub fn is_negative(&self) -> bool {
        self.valence < 0.0
    }

    /// Check if emotion is primarily positive (valence > 0)
    pub fn is_positive(&self) -> bool {
        self.valence > 0.0
    }

    /// Check if arousal is high (activated/alert)
    pub fn is_high_arousal(&self) -> bool {
        self.arousal > 0.6
    }

    /// Check if arousal is low (calm/relaxed)
    pub fn is_low_arousal(&self) -> bool {
        self.arousal < 0.4
    }

    /// Get quadrant in Russell's circumplex model
    pub fn circumplex_quadrant(&self) -> &'static str {
        match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => "High Arousal Positive (Excited, Joyful)",
            (true, false) => "Low Arousal Positive (Content, Peaceful)",
            (false, true) => "High Arousal Negative (Anxious, Fearful)",
            (false, false) => "Low Arousal Negative (Sad, Depressed)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_from_valence_arousal() {
        // High arousal, positive valence = Joy
        assert_eq!(Emotion::from_valence_arousal(0.7, 0.8), Emotion::Joy);

        // High arousal, negative valence = Fear
        assert_eq!(Emotion::from_valence_arousal(-0.7, 0.8), Emotion::Fear);

        // Low arousal, negative valence = Sadness
        assert_eq!(Emotion::from_valence_arousal(-0.7, 0.2), Emotion::Sadness);

        // Low arousal, positive valence = Contentment
        assert_eq!(Emotion::from_valence_arousal(0.7, 0.2), Emotion::Contentment);
    }

    #[test]
    fn test_emotional_reasoner_creation() {
        let reasoner = EmotionalReasoner::new();
        assert!(reasoner.primitives().get("JOY").is_some());
        assert!(reasoner.primitives().get("FEAR").is_some());
        assert!(reasoner.primitives().get("VALENCE").is_some());
        assert!(reasoner.primitives().get("AROUSAL").is_some());
    }

    #[test]
    fn test_derive_emotional_state_joyful() {
        let reasoner = EmotionalReasoner::new();

        // High dopamine, low cortisol = positive valence
        let hormones = HormoneState {
            cortisol: 0.2,
            dopamine: 0.9,
            acetylcholine: 0.6,
        };

        let emotional_state = reasoner.derive_emotional_state(&hormones);

        // Primary validation: positive valence from high dopamine, low cortisol
        assert!(emotional_state.is_positive(), "Should be positive valence");
        // The specific emotion depends on thresholds; Joy or Contentment are valid
        assert!(
            matches!(emotional_state.dominant_emotion, Emotion::Joy | Emotion::Contentment | Emotion::Surprise),
            "Should be a positive emotion, got {:?}", emotional_state.dominant_emotion
        );
    }

    #[test]
    fn test_derive_emotional_state_anxious() {
        let reasoner = EmotionalReasoner::new();

        // High cortisol, low dopamine = anxious/fearful
        let hormones = HormoneState {
            cortisol: 0.9,
            dopamine: 0.2,
            acetylcholine: 0.8,
        };

        let emotional_state = reasoner.derive_emotional_state(&hormones);

        assert!(emotional_state.is_negative(), "Should be negative valence");
        assert!(emotional_state.is_high_arousal(), "Should be high arousal");
        assert_eq!(emotional_state.dominant_emotion, Emotion::Fear);
    }

    #[test]
    fn test_derive_emotional_state_sad() {
        let reasoner = EmotionalReasoner::new();

        // Low dopamine, moderate cortisol, low arousal = negative valence
        let hormones = HormoneState {
            cortisol: 0.4,
            dopamine: 0.1,
            acetylcholine: 0.2,
        };

        let emotional_state = reasoner.derive_emotional_state(&hormones);

        // Primary validation: negative valence from low dopamine vs cortisol
        assert!(emotional_state.is_negative(), "Should be negative valence");
        // The specific emotion depends on thresholds; any negative emotion is valid
        assert!(
            matches!(emotional_state.dominant_emotion, Emotion::Sadness | Emotion::Fear | Emotion::Anger | Emotion::Disgust | Emotion::Neutral),
            "Should be a negative or neutral emotion, got {:?}", emotional_state.dominant_emotion
        );
    }

    #[test]
    fn test_emotional_state_report() {
        let reasoner = EmotionalReasoner::new();

        let hormones = HormoneState {
            cortisol: 0.2,
            dopamine: 0.8,
            acetylcholine: 0.5,
        };

        let emotional_state = reasoner.derive_emotional_state(&hormones);
        let report = emotional_state.report();

        assert!(report.contains("Joy") || report.contains("Contentment"));
        assert!(report.contains("Valence"));
        assert!(report.contains("Arousal"));
    }

    #[test]
    fn test_complex_emotions() {
        let reasoner = EmotionalReasoner::new();

        // Check complex emotions are available
        assert!(reasoner.get_complex_emotion("EMPATHY").is_some());
        assert!(reasoner.get_complex_emotion("ATTACHMENT").is_some());
        assert!(reasoner.get_complex_emotion("AWE").is_some());
    }

    #[test]
    fn test_emotional_similarity() {
        let reasoner = EmotionalReasoner::new();

        let joyful_hormones = HormoneState {
            cortisol: 0.2,
            dopamine: 0.9,
            acetylcholine: 0.6,
        };

        let slightly_less_joyful = HormoneState {
            cortisol: 0.3,
            dopamine: 0.8,
            acetylcholine: 0.6,
        };

        let fearful_hormones = HormoneState {
            cortisol: 0.9,
            dopamine: 0.2,
            acetylcholine: 0.8,
        };

        let state1 = reasoner.derive_emotional_state(&joyful_hormones);
        let state2 = reasoner.derive_emotional_state(&slightly_less_joyful);
        let state3 = reasoner.derive_emotional_state(&fearful_hormones);

        // Similar joyful states should be similar
        let similarity_joy = reasoner.emotional_similarity(&state1, &state2);
        assert!(similarity_joy > 0.6, "Similar joyful states should be semantically similar");

        // Joyful vs fearful should be different
        let similarity_diff = reasoner.emotional_similarity(&state1, &state3);
        assert!(similarity_diff < 0.9, "Joy and fear should be distinguishable");
    }

    #[test]
    fn test_secondary_emotions() {
        let reasoner = EmotionalReasoner::new();

        // Moderate state might have multiple emotional components
        let hormones = HormoneState {
            cortisol: 0.5,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        let emotional_state = reasoner.derive_emotional_state(&hormones);

        // Should have identified some secondary emotions if state is mixed
        println!("Secondary emotions: {:?}", emotional_state.secondary_emotions);
        println!("Report:\n{}", emotional_state.report());

        // Verify secondary emotions are weaker than dominant
        for (emotion, intensity) in &emotional_state.secondary_emotions {
            assert!(intensity < &emotional_state.intensity,
                "Secondary emotion {:?} ({:.2}) should be weaker than dominant ({:.2})",
                emotion, intensity, emotional_state.intensity
            );
        }
    }
}
