// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Resonant Speech: Consciousness-Aware Response Generation
//!
//! This module provides speech generation that adapts to user state and
//! cognitive load. It produces responses that resonate with the user's
//! current needs and level of understanding.
//!
//! ## Key Concepts
//!
//! - **Cognitive Load**: Adapts complexity based on user mental state
//! - **User State**: Tracks emotional and contextual factors
//! - **Resonance**: Responses that harmonize with user needs
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::resonant_speech::{ResonantSpeech, CognitiveLoad, UserState};
//!
//! let mut speech = ResonantSpeech::new();
//! let response = speech.generate("Install vim", cognitive_load, user_state);
//! ```

use serde::{Deserialize, Serialize};

// Re-export from user_state_inference for convenience
pub use crate::user_state_inference::{
    ContextKind, ExperienceLevel, UserState as UserStateInferred, Verbosity,
};

/// Cognitive load level for response adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CognitiveLoad {
    /// Low cognitive load - user can handle detailed information
    Low,
    /// Medium cognitive load - balanced responses
    #[default]
    Medium,
    /// High cognitive load - simplified, focused responses
    High,
    /// Overloaded - minimal essential information only
    Overloaded,
}

impl CognitiveLoad {
    /// Create from numeric value (0.0 to 1.0)
    pub fn from_level(level: f64) -> Self {
        if level < 0.25 {
            CognitiveLoad::Low
        } else if level < 0.5 {
            CognitiveLoad::Medium
        } else if level < 0.75 {
            CognitiveLoad::High
        } else {
            CognitiveLoad::Overloaded
        }
    }

    /// Convert to numeric value
    pub fn to_level(&self) -> f64 {
        match self {
            CognitiveLoad::Low => 0.1,
            CognitiveLoad::Medium => 0.4,
            CognitiveLoad::High => 0.7,
            CognitiveLoad::Overloaded => 0.9,
        }
    }

    /// Get recommended response characteristics
    pub fn response_profile(&self) -> ResponseProfile {
        match self {
            CognitiveLoad::Low => ResponseProfile {
                max_sentences: 10,
                use_examples: true,
                include_explanations: true,
                use_technical_terms: true,
                suggested_pause_ms: 0,
            },
            CognitiveLoad::Medium => ResponseProfile {
                max_sentences: 5,
                use_examples: true,
                include_explanations: true,
                use_technical_terms: true,
                suggested_pause_ms: 100,
            },
            CognitiveLoad::High => ResponseProfile {
                max_sentences: 3,
                use_examples: false,
                include_explanations: false,
                use_technical_terms: false,
                suggested_pause_ms: 200,
            },
            CognitiveLoad::Overloaded => ResponseProfile {
                max_sentences: 1,
                use_examples: false,
                include_explanations: false,
                use_technical_terms: false,
                suggested_pause_ms: 300,
            },
        }
    }
}

/// User state for resonant speech (simplified version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    /// Cognitive load level
    pub cognitive_load: CognitiveLoad,

    /// Frustration level (0.0 to 1.0)
    pub frustration: f64,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Experience level
    pub experience: ExperienceLevel,

    /// Current context
    pub context: ContextKind,

    /// Whether user appears rushed
    pub is_rushed: bool,

    /// Whether user is in learning mode
    pub is_learning: bool,

    /// Trust level in Symthaea (0.0 to 1.0)
    pub trust_in_sophia: f64,
}

impl Default for UserState {
    fn default() -> Self {
        Self {
            cognitive_load: CognitiveLoad::Medium,
            frustration: 0.0,
            confidence: 0.5,
            experience: ExperienceLevel::Beginner,
            context: ContextKind::Unknown,
            is_rushed: false,
            is_learning: false,
            trust_in_sophia: 0.5,
        }
    }
}

impl UserState {
    /// Create from UserStateInferred
    pub fn from_inferred(inferred: &UserStateInferred) -> Self {
        Self {
            cognitive_load: CognitiveLoad::from_level(inferred.cognitive_load.level),
            frustration: inferred.frustration,
            confidence: inferred.confidence,
            experience: inferred.experience,
            context: inferred.context,
            // `idle_time_secs < 0.5` alone was an unrealistic threshold for
            // turn-based chat (a single round trip typically exceeds it), so
            // it could almost never fire; explicit urgency language is a
            // direct signal, kept alongside a looser idle-time fallback.
            is_rushed: inferred.urgent_language
                || (inferred.idle_time_secs < 5.0 && inferred.interaction_count > 5),
            is_learning: inferred.context == ContextKind::Help
                || inferred.context == ContextKind::Exploration,
            // Trust starts high and degrades with frustration
            trust_in_sophia: (1.0 - inferred.frustration * 0.5).max(0.0),
        }
    }

    /// Check if empathic response needed
    pub fn needs_empathy(&self) -> bool {
        self.frustration > 0.5 || self.confidence < 0.3
    }

    /// Check if user is overwhelmed
    pub fn is_overwhelmed(&self) -> bool {
        self.cognitive_load == CognitiveLoad::Overloaded
            || (self.cognitive_load == CognitiveLoad::High && self.frustration > 0.5)
    }

    /// Create from neuromodulator bath signals.
    ///
    /// Maps allostatic load and consciousness level (Psi) to cognitive load,
    /// with an oxytocin-mediated empathic override (Feldman 2012).
    pub fn from_neuromod(allostatic_load: f32, psi: f64, arousal: f32, oxytocin: f32) -> Self {
        let load_level = allostatic_load as f64 * 0.6 + (1.0 - psi) * 0.4;
        // High oxytocin + low psi → empathic override (Feldman 2012)
        let cognitive_load = if oxytocin > 0.6 && psi < 0.4 {
            CognitiveLoad::Overloaded
        } else {
            CognitiveLoad::from_level(load_level)
        };
        Self {
            cognitive_load,
            frustration: allostatic_load as f64,
            confidence: psi,
            is_rushed: arousal > 0.7,
            trust_in_sophia: (0.5 + oxytocin as f64 * 0.3).clamp(0.0, 1.0),
            ..Default::default()
        }
    }
}

/// Profile for response generation
#[derive(Debug, Clone)]
pub struct ResponseProfile {
    /// Maximum sentences in response
    pub max_sentences: usize,

    /// Whether to include examples
    pub use_examples: bool,

    /// Whether to include explanations
    pub include_explanations: bool,

    /// Whether to use technical terms
    pub use_technical_terms: bool,

    /// Suggested pause before responding (ms)
    pub suggested_pause_ms: u32,
}

/// Resonant speech generator
#[derive(Debug)]
pub struct ResonantSpeech {
    /// Current user state
    user_state: UserState,

    /// Response templates
    templates: ResponseTemplates,

    /// Statistics
    stats: SpeechStats,
}

/// Response templates for different situations
#[derive(Debug, Clone, Default)]
pub struct ResponseTemplates {
    /// Empathic prefixes for frustrated users
    pub empathic_prefixes: Vec<String>,

    /// Encouraging suffixes for low-confidence users
    pub encouraging_suffixes: Vec<String>,

    /// Acknowledgment phrases
    pub acknowledgments: Vec<String>,
}

impl ResponseTemplates {
    fn default_templates() -> Self {
        Self {
            empathic_prefixes: vec![
                "I understand this can be frustrating.".to_string(),
                "Let's work through this together.".to_string(),
                "I can see this isn't working as expected.".to_string(),
            ],
            encouraging_suffixes: vec![
                "You're doing great!".to_string(),
                "Feel free to ask if anything is unclear.".to_string(),
                "That's a good question.".to_string(),
            ],
            acknowledgments: vec![
                "Got it.".to_string(),
                "I understand.".to_string(),
                "Sure.".to_string(),
            ],
        }
    }
}

/// Statistics for resonant speech
#[derive(Debug, Clone, Default)]
pub struct SpeechStats {
    /// Total responses generated
    pub total_responses: u64,

    /// Empathic responses given
    pub empathic_responses: u64,

    /// Simplified responses (high cognitive load)
    pub simplified_responses: u64,

    /// Average response length (characters)
    pub avg_response_length: f32,
}

impl ResonantSpeech {
    /// Create a new resonant speech generator
    pub fn new() -> Self {
        Self {
            user_state: UserState::default(),
            templates: ResponseTemplates::default_templates(),
            stats: SpeechStats::default(),
        }
    }

    /// Update user state
    pub fn update_state(&mut self, state: UserState) {
        self.user_state = state;
    }

    /// Generate a resonant response
    pub fn generate(&mut self, content: &str, _context: &str) -> String {
        let profile = self.user_state.cognitive_load.response_profile();
        let mut response = String::new();

        // Add empathic prefix if needed
        if self.user_state.needs_empathy() && !self.templates.empathic_prefixes.is_empty() {
            let idx =
                (self.stats.total_responses as usize) % self.templates.empathic_prefixes.len();
            response.push_str(&self.templates.empathic_prefixes[idx]);
            response.push(' ');
            self.stats.empathic_responses += 1;
        }

        // Add acknowledgment for task contexts
        if self.user_state.context == ContextKind::Task
            && !self.templates.acknowledgments.is_empty()
        {
            let idx = (self.stats.total_responses as usize) % self.templates.acknowledgments.len();
            response.push_str(&self.templates.acknowledgments[idx]);
            response.push(' ');
        }

        // Add main content (simplified if needed)
        if self.user_state.is_overwhelmed() {
            // Simplified response
            response.push_str(&self.simplify(content, profile.max_sentences));
            self.stats.simplified_responses += 1;
        } else {
            response.push_str(content);
        }

        // Add encouraging suffix for learning contexts
        if self.user_state.is_learning
            && self.user_state.confidence < 0.5
            && !self.templates.encouraging_suffixes.is_empty()
        {
            let idx =
                (self.stats.total_responses as usize) % self.templates.encouraging_suffixes.len();
            response.push(' ');
            response.push_str(&self.templates.encouraging_suffixes[idx]);
        }

        // Update stats
        self.stats.total_responses += 1;
        let n = self.stats.total_responses as f32;
        self.stats.avg_response_length =
            (self.stats.avg_response_length * (n - 1.0) + response.len() as f32) / n;

        response
    }

    /// Simplify content for high cognitive load
    fn simplify(&self, content: &str, max_sentences: usize) -> String {
        let sentences: Vec<&str> = content
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .take(max_sentences)
            .collect();

        if sentences.is_empty() {
            content.to_string()
        } else {
            sentences.join(". ") + "."
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SpeechStats {
        &self.stats
    }

    /// Get current user state
    pub fn user_state(&self) -> &UserState {
        &self.user_state
    }
}

impl Default for ResonantSpeech {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_load_from_level() {
        assert_eq!(CognitiveLoad::from_level(0.1), CognitiveLoad::Low);
        assert_eq!(CognitiveLoad::from_level(0.4), CognitiveLoad::Medium);
        assert_eq!(CognitiveLoad::from_level(0.7), CognitiveLoad::High);
        assert_eq!(CognitiveLoad::from_level(0.9), CognitiveLoad::Overloaded);
    }

    #[test]
    fn test_user_state_needs_empathy() {
        let mut state = UserState::default();
        assert!(!state.needs_empathy());

        state.frustration = 0.6;
        assert!(state.needs_empathy());

        state.frustration = 0.0;
        state.confidence = 0.2;
        assert!(state.needs_empathy());
    }

    #[test]
    fn test_resonant_speech_generation() {
        let mut speech = ResonantSpeech::new();

        let response = speech.generate("Here is your answer.", "test");
        assert!(!response.is_empty());
    }

    #[test]
    fn test_empathic_response() {
        let mut speech = ResonantSpeech::new();
        speech.user_state.frustration = 0.7;

        let response = speech.generate("Content", "context");
        assert!(response.len() > "Content".len()); // Should have prefix
    }

    #[test]
    fn from_neuromod_low_load_technical() {
        let state = UserState::from_neuromod(0.1, 0.8, 0.3, 0.2);
        // allostatic=0.1, psi=0.8 → load_level = 0.06 + 0.08 = 0.14 → Low
        assert_eq!(state.cognitive_load, CognitiveLoad::Low);
        assert!(!state.is_rushed);
    }

    #[test]
    fn from_neuromod_high_load_empathic() {
        let state = UserState::from_neuromod(0.9, 0.2, 0.5, 0.3);
        // allostatic=0.9, psi=0.2 → load_level = 0.54 + 0.32 = 0.86 → Overloaded
        assert_eq!(state.cognitive_load, CognitiveLoad::Overloaded);
    }

    #[test]
    fn from_neuromod_oxytocin_empathy_override() {
        let state = UserState::from_neuromod(0.3, 0.3, 0.5, 0.8);
        // oxy=0.8 > 0.6, psi=0.3 < 0.4 → override to Overloaded
        assert_eq!(state.cognitive_load, CognitiveLoad::Overloaded);
    }

    #[test]
    fn from_neuromod_trust_from_oxytocin() {
        let state = UserState::from_neuromod(0.3, 0.5, 0.3, 0.5);
        // trust = (0.5 + 0.5 * 0.3) = 0.65
        assert!(
            state.trust_in_sophia > 0.5,
            "trust={}",
            state.trust_in_sophia
        );
        assert!((state.trust_in_sophia - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_simplified_response() {
        let mut speech = ResonantSpeech::new();
        speech.user_state.cognitive_load = CognitiveLoad::Overloaded;

        let long_content = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let response = speech.generate(long_content, "context");

        // Should be simplified
        assert!(response.matches('.').count() <= 2);
    }
}
