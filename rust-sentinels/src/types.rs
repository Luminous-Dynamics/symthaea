// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for consciousness state representation

use serde::{Deserialize, Serialize};

/// High-level consciousness state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsciousnessState {
    /// Deep NREM sleep (N3)
    DeepSleep,
    /// Light NREM sleep (N1, N2)
    LightSleep,
    /// REM sleep state
    Rem,
    /// Drowsy/transitional state
    Drowsy,
    /// Calm wakefulness
    Relaxed,
    /// Active concentration
    Focused,
    /// Deep meditation state
    Meditative,
    /// Optimal performance state
    Flow,
    /// High arousal, negative valence
    Stressed,
}

impl ConsciousnessState {
    /// Returns true if this is a sleep state
    pub fn is_sleep(&self) -> bool {
        matches!(self, Self::DeepSleep | Self::LightSleep | Self::Rem)
    }

    /// Returns true if this is an awake state
    pub fn is_awake(&self) -> bool {
        !self.is_sleep()
    }

    /// Returns a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::DeepSleep => "Deep restorative sleep",
            Self::LightSleep => "Light transitional sleep",
            Self::Rem => "REM dream sleep",
            Self::Drowsy => "Drowsy transitional state",
            Self::Relaxed => "Calm wakeful state",
            Self::Focused => "Active concentration",
            Self::Meditative => "Deep meditative state",
            Self::Flow => "Optimal flow state",
            Self::Stressed => "High stress state",
        }
    }
}

/// Sleep stage classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SleepStage {
    Wake = 0,
    N1 = 1,
    N2 = 2,
    N3 = 3,
    Rem = 4,
}

impl SleepStage {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wake => "Wake",
            Self::N1 => "N1",
            Self::N2 => "N2",
            Self::N3 => "N3",
            Self::Rem => "REM",
        }
    }
}

/// Emotion analysis results (Proof of Joy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    /// Emotional valence: -1 (negative) to +1 (positive)
    pub valence: f32,
    /// Arousal level: 0 (calm) to 1 (excited)
    pub arousal: f32,
    /// Confidence in the measurement
    pub confidence: f32,
}

impl EmotionScore {
    /// Returns the emotion quadrant (Russell's circumplex model)
    pub fn quadrant(&self) -> EmotionQuadrant {
        match (self.valence >= 0.0, self.arousal >= 0.5) {
            (true, true) => EmotionQuadrant::ExcitedPositive,
            (true, false) => EmotionQuadrant::CalmPositive,
            (false, true) => EmotionQuadrant::ExcitedNegative,
            (false, false) => EmotionQuadrant::CalmNegative,
        }
    }

    /// Example emotions for this state
    pub fn example_emotions(&self) -> &'static [&'static str] {
        match self.quadrant() {
            EmotionQuadrant::ExcitedPositive => &["happy", "excited", "elated"],
            EmotionQuadrant::CalmPositive => &["relaxed", "content", "peaceful"],
            EmotionQuadrant::ExcitedNegative => &["angry", "anxious", "stressed"],
            EmotionQuadrant::CalmNegative => &["sad", "bored", "depressed"],
        }
    }
}

/// Emotion quadrant in circumplex model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmotionQuadrant {
    ExcitedPositive,
    CalmPositive,
    ExcitedNegative,
    CalmNegative,
}

/// Sleep analysis results (Proof of Rest)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepScore {
    /// Detected sleep stage
    pub stage: SleepStage,
    /// Confidence in stage classification
    pub confidence: f32,
    /// Relative delta power (0.5-4 Hz)
    pub delta_power: f32,
    /// Overall sleep quality score (0-1)
    pub sleep_quality: f32,
}

impl SleepScore {
    /// Returns true if deep sleep is detected
    pub fn is_deep_sleep(&self) -> bool {
        self.stage == SleepStage::N3
    }

    /// Returns true if awake
    pub fn is_awake(&self) -> bool {
        self.stage == SleepStage::Wake
    }
}

/// Meditation analysis results (Proof of Focus)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeditationScore {
    /// Meditation depth: 0 (surface) to 1 (deep)
    pub depth: f32,
    /// Stability of meditation state
    pub stability: f32,
    /// Relative alpha power (8-13 Hz)
    pub alpha_power: f32,
    /// Relative theta power (4-8 Hz)
    pub theta_power: f32,
    /// Meditation index: (alpha + theta) / beta
    pub meditation_index: f32,
}

impl MeditationScore {
    /// Returns the meditation state category
    pub fn state(&self) -> MeditationState {
        if self.depth > 0.7 {
            MeditationState::Deep
        } else if self.depth > 0.4 {
            MeditationState::Moderate
        } else {
            MeditationState::Light
        }
    }
}

/// Meditation depth category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeditationState {
    Light,
    Moderate,
    Deep,
}

/// Combined Proof of Consciousness result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfConsciousness {
    /// Unix timestamp of analysis
    pub timestamp: f64,
    /// Overall consciousness state
    pub state: ConsciousnessState,
    /// Consciousness level: 0 (deep sleep) to 1 (peak awareness)
    pub consciousness_level: f32,
    /// Wellbeing score: 0 (poor) to 1 (optimal)
    pub wellbeing_score: f32,
    /// Emotion analysis (if performed)
    pub emotion: Option<EmotionScore>,
    /// Sleep analysis (if performed)
    pub sleep: Option<SleepScore>,
    /// Meditation analysis (if performed)
    pub meditation: Option<MeditationScore>,
}

impl ProofOfConsciousness {
    /// Returns a summary string
    pub fn summary(&self) -> String {
        format!(
            "State: {:?} | Consciousness: {:.0}% | Wellbeing: {:.0}%",
            self.state,
            self.consciousness_level * 100.0,
            self.wellbeing_score * 100.0
        )
    }

    /// Returns JSON representation
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}