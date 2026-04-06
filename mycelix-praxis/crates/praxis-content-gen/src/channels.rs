// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ThoughtChannel presets for educational content generation.
//!
//! These map to Broca's 43-channel ThoughtChannels when a Symthaea runtime is
//! available. Without the runtime, they serve as structured configuration for
//! the [`MockGenerator`](crate::mock::MockGenerator) and future Broca adapters.
//!
//! ## Channel semantics
//!
//! | Channel            | Range     | Educational meaning                        |
//! |--------------------|-----------|--------------------------------------------|
//! | `epistemic_level`  | 0.0-4.0   | 0=certain fact, 4=out of domain            |
//! | `valence`          | -1.0-1.0  | Positive = encouraging, negative = serious |
//! | `arousal`          | 0.0-1.0   | Energy level of the prose                  |
//! | `warmth`           | 0.0-1.0   | Friendliness toward learner                |
//! | `consciousness`    | 0.0-1.0   | Gates verbosity (higher = more concise)    |
//! | `meta_awareness`   | 0.0-1.0   | Self-monitoring during generation          |
//! | `coherence`        | 0.0-1.0   | Logical flow requirement                   |
//! | `domain_familiarity` | 0.0-1.0 | Expertise in the subject matter            |
//! | `confidence`       | 0.0-1.0   | Assertion strength                         |
//! | `time_pressure`    | 0.0-1.0   | Urgency (lower = more thoughtful)          |

use serde::{Deserialize, Serialize};

/// Thought channel configuration for content generation.
///
/// Maps to Broca's ThoughtChannels when a Symthaea runtime is available.
/// The `intent` field maps to the 8-channel one-hot `semantic_intent` vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentChannels {
    /// Semantic intent (maps to Broca's one-hot semantic_intent)
    pub intent: ContentIntent,
    /// Epistemic status: 0.0 (certain) to 4.0 (out of domain)
    pub epistemic_level: f32,
    /// Emotional valence: -1.0 (serious/negative) to 1.0 (encouraging/positive)
    pub valence: f32,
    /// Arousal: 0.0 (calm, methodical) to 1.0 (energetic, exciting)
    pub arousal: f32,
    /// Warmth: 0.0 (formal) to 1.0 (friendly, nurturing)
    pub warmth: f32,
    /// Consciousness level: 0.0 (verbose) to 1.0 (concise, gated)
    pub consciousness: f32,
    /// Meta-awareness: self-monitoring during generation
    pub meta_awareness: f32,
    /// Coherence: logical flow requirement
    pub coherence: f32,
    /// Domain familiarity: expertise in the subject
    pub domain_familiarity: f32,
    /// Response confidence: assertion strength
    pub confidence: f32,
    /// Time pressure: 0.0 (thoughtful) to 1.0 (urgent/brief)
    pub time_pressure: f32,
}

/// Semantic intent for content generation, mapping to Broca's 8-channel
/// one-hot `semantic_intent` vector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContentIntent {
    /// Teach a concept with explanation
    TeachConcept,
    /// Provide a worked example
    GiveExample,
    /// Ask a practice question
    AskQuestion,
    /// Provide a progressive hint
    ProvideHint,
    /// Address a misconception
    ExplainMisconception,
    /// Prompt meta-cognitive reflection
    ReflectOnLearning,
    /// Encourage after struggle or mistake
    Encourage,
}

impl ContentChannels {
    /// Preset for teaching a factual concept (definitions, rules, procedures).
    ///
    /// High confidence, moderate warmth, strong coherence. Epistemic level 0
    /// (certain) because we teach established mathematical facts.
    pub fn teaching_factual() -> Self {
        Self {
            intent: ContentIntent::TeachConcept,
            epistemic_level: 0.0,
            valence: 0.4,
            arousal: 0.3,
            warmth: 0.6,
            consciousness: 0.6,
            meta_awareness: 0.5,
            coherence: 0.9,
            domain_familiarity: 0.95,
            confidence: 0.95,
            time_pressure: 0.2,
        }
    }

    /// Preset for step-by-step worked examples.
    ///
    /// Very high coherence (each step must follow logically), moderate arousal
    /// to keep the student engaged, high domain familiarity.
    pub fn worked_example() -> Self {
        Self {
            intent: ContentIntent::GiveExample,
            epistemic_level: 0.0,
            valence: 0.3,
            arousal: 0.35,
            warmth: 0.5,
            consciousness: 0.7,
            meta_awareness: 0.6,
            coherence: 0.95,
            domain_familiarity: 0.95,
            confidence: 0.9,
            time_pressure: 0.1,
        }
    }

    /// Preset for practice problems.
    ///
    /// Slightly higher arousal (challenge energy), lower warmth (assessment
    /// mode), very high confidence in the correctness of the question.
    pub fn practice_problem() -> Self {
        Self {
            intent: ContentIntent::AskQuestion,
            epistemic_level: 0.0,
            valence: 0.2,
            arousal: 0.4,
            warmth: 0.4,
            consciousness: 0.7,
            meta_awareness: 0.5,
            coherence: 0.9,
            domain_familiarity: 0.95,
            confidence: 0.95,
            time_pressure: 0.3,
        }
    }

    /// Preset for progressive hints.
    ///
    /// `level` controls specificity: 1 = vague nudge, 2 = moderate guidance,
    /// 3 = nearly reveals the answer. Higher levels have more warmth and
    /// lower consciousness (more verbose).
    pub fn hint(level: u8) -> Self {
        let level_f = (level.clamp(1, 3) as f32 - 1.0) / 2.0; // 0.0, 0.5, 1.0
        Self {
            intent: ContentIntent::ProvideHint,
            epistemic_level: 0.0,
            valence: 0.5 + level_f * 0.2,
            arousal: 0.2,
            warmth: 0.6 + level_f * 0.2,
            consciousness: 0.7 - level_f * 0.3, // More verbose at higher levels
            meta_awareness: 0.4,
            coherence: 0.8,
            domain_familiarity: 0.9,
            confidence: 0.5 + level_f * 0.4, // More direct at higher levels
            time_pressure: 0.1,
        }
    }

    /// Preset for addressing misconceptions.
    ///
    /// High meta-awareness (need to understand *why* the student is confused),
    /// high warmth (don't make them feel bad), high coherence.
    pub fn misconception_correction() -> Self {
        Self {
            intent: ContentIntent::ExplainMisconception,
            epistemic_level: 0.0,
            valence: 0.3,
            arousal: 0.3,
            warmth: 0.7,
            consciousness: 0.6,
            meta_awareness: 0.8,
            coherence: 0.9,
            domain_familiarity: 0.95,
            confidence: 0.9,
            time_pressure: 0.1,
        }
    }

    /// Preset for encouragement after a wrong answer.
    ///
    /// High warmth and valence, low time pressure, moderate consciousness
    /// (a few sentences of encouragement, not a lecture).
    pub fn encouragement() -> Self {
        Self {
            intent: ContentIntent::Encourage,
            epistemic_level: 1.0, // Slightly uncertain — we're empathizing, not teaching
            valence: 0.8,
            arousal: 0.4,
            warmth: 0.9,
            consciousness: 0.5,
            meta_awareness: 0.3,
            coherence: 0.7,
            domain_familiarity: 0.5,
            confidence: 0.6,
            time_pressure: 0.0,
        }
    }

    /// Preset for meta-cognitive reflection prompts.
    ///
    /// "What did you learn? How does this connect to what you already know?"
    /// High meta-awareness, moderate warmth, lower confidence (open-ended).
    pub fn reflection() -> Self {
        Self {
            intent: ContentIntent::ReflectOnLearning,
            epistemic_level: 1.5,
            valence: 0.4,
            arousal: 0.2,
            warmth: 0.6,
            consciousness: 0.5,
            meta_awareness: 0.9,
            coherence: 0.7,
            domain_familiarity: 0.7,
            confidence: 0.5,
            time_pressure: 0.0,
        }
    }

    /// Adjust channels for grade level.
    ///
    /// Younger students get warmer tone, lower arousal (calmer), simpler
    /// language (higher consciousness = more concise). `grade_ordinal` uses
    /// the same scale as `GradeLevel::ordinal()` (PreK=0, K=1, Grade1=2, ...).
    pub fn for_grade(mut self, grade_ordinal: u8) -> Self {
        // Normalize: 0 (PreK) to 15 (Adult) -> 0.0 to 1.0
        let maturity = (grade_ordinal as f32 / 15.0).clamp(0.0, 1.0);

        // Younger = warmer, calmer, more encouraging
        self.warmth = (self.warmth + 0.3 * (1.0 - maturity)).clamp(0.0, 1.0);
        self.valence = (self.valence + 0.2 * (1.0 - maturity)).clamp(-1.0, 1.0);
        self.arousal = (self.arousal - 0.1 * (1.0 - maturity)).clamp(0.0, 1.0);

        // Younger = simpler language (higher consciousness = more gated/concise)
        self.consciousness = (self.consciousness + 0.15 * (1.0 - maturity)).clamp(0.0, 1.0);

        self
    }

    /// Returns true if all channel values are within their valid ranges.
    pub fn is_valid(&self) -> bool {
        (0.0..=4.0).contains(&self.epistemic_level)
            && (-1.0..=1.0).contains(&self.valence)
            && (0.0..=1.0).contains(&self.arousal)
            && (0.0..=1.0).contains(&self.warmth)
            && (0.0..=1.0).contains(&self.consciousness)
            && (0.0..=1.0).contains(&self.meta_awareness)
            && (0.0..=1.0).contains(&self.coherence)
            && (0.0..=1.0).contains(&self.domain_familiarity)
            && (0.0..=1.0).contains(&self.confidence)
            && (0.0..=1.0).contains(&self.time_pressure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teaching_factual_valid() {
        let ch = ContentChannels::teaching_factual();
        assert!(ch.is_valid());
        assert_eq!(ch.intent, ContentIntent::TeachConcept);
        assert!(ch.confidence > 0.9);
        assert!(ch.coherence > 0.8);
    }

    #[test]
    fn test_worked_example_valid() {
        let ch = ContentChannels::worked_example();
        assert!(ch.is_valid());
        assert_eq!(ch.intent, ContentIntent::GiveExample);
        assert!(ch.coherence > 0.9);
    }

    #[test]
    fn test_practice_problem_valid() {
        let ch = ContentChannels::practice_problem();
        assert!(ch.is_valid());
        assert_eq!(ch.intent, ContentIntent::AskQuestion);
    }

    #[test]
    fn test_hint_levels_valid() {
        for level in 1..=3 {
            let ch = ContentChannels::hint(level);
            assert!(ch.is_valid(), "hint level {} invalid", level);
            assert_eq!(ch.intent, ContentIntent::ProvideHint);
        }
    }

    #[test]
    fn test_hint_level_progression() {
        let h1 = ContentChannels::hint(1);
        let h2 = ContentChannels::hint(2);
        let h3 = ContentChannels::hint(3);

        // Higher levels should be warmer and more confident (more specific)
        assert!(h3.warmth >= h1.warmth);
        assert!(h3.confidence >= h1.confidence);

        // Higher levels should be less gated (more verbose explanation)
        assert!(h3.consciousness <= h1.consciousness);

        // Level 2 should be between 1 and 3
        assert!(h2.confidence >= h1.confidence);
        assert!(h2.confidence <= h3.confidence);
    }

    #[test]
    fn test_hint_level_clamped() {
        // Levels outside 1-3 should clamp
        let h0 = ContentChannels::hint(0);
        let h1 = ContentChannels::hint(1);
        assert!((h0.confidence - h1.confidence).abs() < f32::EPSILON);

        let h5 = ContentChannels::hint(5);
        let h3 = ContentChannels::hint(3);
        assert!((h5.confidence - h3.confidence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_misconception_correction_valid() {
        let ch = ContentChannels::misconception_correction();
        assert!(ch.is_valid());
        assert!(ch.meta_awareness > 0.7);
        assert!(ch.warmth > 0.6);
    }

    #[test]
    fn test_encouragement_valid() {
        let ch = ContentChannels::encouragement();
        assert!(ch.is_valid());
        assert!(ch.warmth > 0.8);
        assert!(ch.valence > 0.7);
    }

    #[test]
    fn test_reflection_valid() {
        let ch = ContentChannels::reflection();
        assert!(ch.is_valid());
        assert!(ch.meta_awareness > 0.8);
    }

    #[test]
    fn test_for_grade_young_warmer() {
        let base = ContentChannels::teaching_factual();
        let grade3 = base.clone().for_grade(4); // Grade3 ordinal = 4
        let adult = base.for_grade(15);

        assert!(grade3.warmth > adult.warmth, "Grade 3 should be warmer than adult");
        assert!(grade3.valence >= adult.valence, "Grade 3 should be more positive");
    }

    #[test]
    fn test_for_grade_stays_valid() {
        let presets = vec![
            ContentChannels::teaching_factual(),
            ContentChannels::worked_example(),
            ContentChannels::practice_problem(),
            ContentChannels::hint(2),
            ContentChannels::encouragement(),
            ContentChannels::reflection(),
        ];
        for preset in presets {
            for grade in [0, 1, 4, 8, 13, 15] {
                let adjusted = preset.clone().for_grade(grade);
                assert!(adjusted.is_valid(), "Grade {} adjustment invalid", grade);
            }
        }
    }

    #[test]
    fn test_channels_roundtrip() {
        let ch = ContentChannels::teaching_factual().for_grade(4);
        let json = serde_json::to_string(&ch).unwrap();
        let round: ContentChannels = serde_json::from_str(&json).unwrap();
        assert_eq!(round.intent, ContentIntent::TeachConcept);
        assert!((round.confidence - ch.confidence).abs() < f32::EPSILON);
    }
}
