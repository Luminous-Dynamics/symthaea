// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

/// Emotion contagion - emotional content influences consciousness state
///
/// Detects emotional valence in input and nudges consciousness patterns:
/// - Positive emotions -> Excited, Focused
/// - Negative emotions -> Contemplative
/// - Neutral -> no influence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EmotionContagion {
    /// Current emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    /// High arousal = excited/angry, Low arousal = calm/sad
    pub arousal: f32,

    /// Emotional influence strength (0.0 to 1.0)
    /// How much emotion affects consciousness pattern
    pub influence_strength: f32,

    /// Smoothed valence (EMA)
    smoothed_valence: f32,

    /// Smoothed arousal (EMA)
    smoothed_arousal: f32,
}

impl Default for EmotionContagion {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            influence_strength: 0.3,
            smoothed_valence: 0.0,
            smoothed_arousal: 0.5,
        }
    }
}

impl EmotionContagion {
    /// Positive emotion indicators
    const POSITIVE_WORDS: &'static [&'static str] = &[
        "happy",
        "joy",
        "love",
        "great",
        "wonderful",
        "excellent",
        "amazing",
        "beautiful",
        "fantastic",
        "good",
        "perfect",
        "brilliant",
        "awesome",
        "delighted",
        "excited",
        "pleased",
        "thrilled",
        "grateful",
        "hope",
        "success",
        "win",
        "celebrate",
        "smile",
        "laugh",
        "fun",
        "enjoy",
    ];

    /// Negative emotion indicators
    const NEGATIVE_WORDS: &'static [&'static str] = &[
        "sad",
        "angry",
        "fear",
        "hate",
        "terrible",
        "awful",
        "horrible",
        "bad",
        "wrong",
        "fail",
        "lost",
        "pain",
        "hurt",
        "worry",
        "anxious",
        "stressed",
        "frustrated",
        "disappointed",
        "regret",
        "sorry",
        "grief",
        "cry",
        "suffer",
        "struggle",
        "difficult",
        "problem",
        "error",
    ];

    /// High arousal indicators (excitement/intensity)
    const HIGH_AROUSAL: &'static [&'static str] = &[
        "!",
        "amazing",
        "incredible",
        "urgent",
        "now",
        "immediately",
        "excited",
        "thrilled",
        "furious",
        "terrified",
        "ecstatic",
    ];

    /// Analyze text for emotional content
    pub fn analyze(&mut self, text: &str) {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        // Safe cast: use f64 intermediate to prevent precision loss on large word counts
        let word_count = (words.len().max(1) as f64) as f32;

        // Count emotional indicators (safe casts via f64)
        let positive_count = (Self::POSITIVE_WORDS
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let negative_count = (Self::NEGATIVE_WORDS
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let arousal_count = (Self::HIGH_AROUSAL
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        // Compute raw valence (-1 to 1)
        let total_emotional = positive_count + negative_count;
        let raw_valence = if total_emotional > 0.0 {
            (positive_count - negative_count) / total_emotional
        } else {
            0.0
        };

        // Compute intensity based on proportion of emotional words
        let emotional_density = total_emotional / word_count;
        let intensity = (emotional_density * 3.0).min(1.0); // Scale up, cap at 1

        // Compute arousal (base + exclamation points + high-arousal words)
        // Safe cast via f64 to handle large match counts
        let exclamation_boost = (text.matches('!').count() as f64 * 0.1) as f32;
        let raw_arousal = (0.5 + arousal_count * 0.1 + exclamation_boost).min(1.0);

        // Apply intensity to valence
        self.valence = raw_valence * intensity;

        // Update arousal
        self.arousal = raw_arousal;

        // Smooth with EMA
        let alpha = 0.3;
        self.smoothed_valence = self.smoothed_valence * (1.0 - alpha) + self.valence * alpha;
        self.smoothed_arousal = self.smoothed_arousal * (1.0 - alpha) + self.arousal * alpha;
    }

    /// Get suggested pattern nudge based on emotional state
    /// Returns (pattern_suggestion, strength) where strength is 0-1
    pub fn pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) {
        let valence = self.smoothed_valence;
        let arousal = self.smoothed_arousal;

        // Only nudge if emotion is significant
        if valence.abs() < 0.2 {
            return (None, 0.0);
        }

        let strength = valence.abs() * self.influence_strength;

        let suggested_pattern = if valence > 0.3 && arousal > 0.6 {
            // High positive + high arousal -> Excited
            Some(ConsciousnessPattern::Excited)
        } else if valence > 0.2 && arousal < 0.5 {
            // Positive + calm -> Focused
            Some(ConsciousnessPattern::Focused)
        } else if valence < -0.3 {
            // Negative -> Contemplative (processing/reflecting)
            Some(ConsciousnessPattern::Contemplative)
        } else if valence > 0.2 {
            // Mildly positive -> Exploratory
            Some(ConsciousnessPattern::Exploratory)
        } else {
            None
        };

        (suggested_pattern, strength)
    }

    /// Get emotional valence for voice prosody
    pub fn prosody_valence(&self) -> f32 {
        self.smoothed_valence
    }

    /// Get arousal for voice prosody
    pub fn prosody_arousal(&self) -> f32 {
        self.smoothed_arousal
    }

    /// Get smoothed valence (EMA-filtered)
    pub fn smoothed_valence(&self) -> f32 {
        self.smoothed_valence
    }

    /// Get smoothed arousal (EMA-filtered)
    pub fn smoothed_arousal(&self) -> f32 {
        self.smoothed_arousal
    }

    /// Reset emotional state
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emotion_default_neutral() {
        let ec = EmotionContagion::default();
        assert_eq!(ec.valence, 0.0);
        assert_eq!(ec.arousal, 0.5);
        assert_eq!(ec.smoothed_valence, 0.0);
        assert_eq!(ec.smoothed_arousal, 0.5);
    }

    #[test]
    fn emotion_positive_text_increases_valence() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am so happy and excited, this is wonderful!");
        assert!(
            ec.valence > 0.0,
            "valence should be positive: {}",
            ec.valence
        );
        assert!(ec.smoothed_valence > 0.0);
    }

    #[test]
    fn emotion_negative_text_decreases_valence() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am sad and angry, everything is terrible and awful");
        assert!(
            ec.valence < 0.0,
            "valence should be negative: {}",
            ec.valence
        );
        assert!(ec.smoothed_valence < 0.0);
    }

    #[test]
    fn emotion_neutral_text_near_zero() {
        let mut ec = EmotionContagion::default();
        ec.analyze("twelve purple chairs remain quite still");
        assert!(
            ec.valence.abs() < 0.01,
            "neutral text valence: {}",
            ec.valence
        );
    }

    #[test]
    fn emotion_exclamation_boosts_arousal() {
        let mut ec = EmotionContagion::default();
        ec.analyze("Now!!! Immediately!!!");
        assert!(
            ec.arousal > 0.5,
            "arousal should be elevated: {}",
            ec.arousal
        );
    }

    #[test]
    fn emotion_valence_bounded() {
        let mut ec = EmotionContagion::default();
        // Extreme positive
        ec.analyze("happy joy love great wonderful excellent amazing beautiful fantastic good perfect brilliant awesome");
        assert!(
            ec.valence >= -1.0 && ec.valence <= 1.0,
            "valence out of bounds: {}",
            ec.valence
        );
        // Extreme negative
        ec.analyze("sad angry fear hate terrible awful horrible bad wrong fail");
        assert!(ec.valence >= -1.0 && ec.valence <= 1.0);
    }

    #[test]
    fn emotion_smoothing_lags() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am incredibly happy and excited!");
        let raw = ec.valence;
        let smoothed = ec.smoothed_valence;
        // First analysis: smoothed should lag behind raw (EMA from 0)
        assert!(
            smoothed.abs() < raw.abs(),
            "smoothed {} should lag raw {}",
            smoothed,
            raw
        );
    }

    #[test]
    fn emotion_pattern_nudge_significant_positive() {
        let mut ec = EmotionContagion::default();
        // Build up enough smoothed valence
        for _ in 0..5 {
            ec.analyze("I am so happy and excited, wonderful amazing day!");
        }
        let (pattern, strength) = ec.pattern_nudge();
        assert!(pattern.is_some(), "should suggest a pattern");
        assert!(strength > 0.0, "strength should be positive");
    }

    #[test]
    fn emotion_pattern_nudge_weak_returns_none() {
        let ec = EmotionContagion::default();
        let (pattern, strength) = ec.pattern_nudge();
        assert!(
            pattern.is_none(),
            "neutral state should not suggest a pattern"
        );
        assert_eq!(strength, 0.0);
    }

    #[test]
    fn emotion_reset_restores_default() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am very happy!");
        ec.reset();
        assert_eq!(ec.valence, 0.0);
        assert_eq!(ec.smoothed_valence, 0.0);
    }

    #[test]
    fn emotion_prosody_accessors() {
        let mut ec = EmotionContagion::default();
        ec.analyze("happy wonderful great");
        assert_eq!(ec.prosody_valence(), ec.smoothed_valence);
        assert_eq!(ec.prosody_arousal(), ec.smoothed_arousal);
    }

    #[test]
    fn emotion_empty_input() {
        let mut ec = EmotionContagion::default();
        ec.analyze("");
        // Should not panic, valence stays near zero
        assert!(ec.valence.abs() < 0.01);
    }
}
