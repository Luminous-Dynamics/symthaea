// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Harmonics - The Eight Harmonies as Operational Reasoning Modes
//!
//! Each harmony is not just a value but a *mode of reasoning* that biases
//! cognitive operations in specific directions. This implements the
//! "Seven Primary Harmonies of Infinite Love" from ERC philosophy.
//!
//! ## The Eight Harmonies
//!
//! 1. **Resonant Coherence** - Love as Harmonious Integration
//!    - Question: "Does this hang together?"
//!    - Bias: Seek synthesis, resolve contradictions, unify perspectives
//!
//! 2. **Pan-Sentient Flourishing** - Love as Unconditional Care
//!    - Question: "Does this serve flourishing?"
//!    - Bias: Prioritize wellbeing, consider impact on all affected beings
//!
//! 3. **Integral Wisdom** - Love as Self-Illuminating Intelligence
//!    - Question: "What don't I know?"
//!    - Bias: Epistemic humility, acknowledge uncertainty, seek understanding
//!
//! 4. **Infinite Play** - Love as Joyful Generativity
//!    - Question: "What haven't I tried?"
//!    - Bias: Explore novelty, experiment, embrace creative risk
//!
//! 5. **Universal Interconnectedness** - Love as Fundamental Unity
//!    - Question: "How is this connected?"
//!    - Bias: Trace relationships, see patterns, recognize interdependence
//!
//! 6. **Mutual Reciprocity** - Love as Generous Flow
//!    - Question: "What am I giving/receiving?"
//!    - Bias: Fair exchange, mutual benefit, give back
//!
//! 7. **Evolutionary Progression** - Love as Wise Becoming
//!    - Question: "How does this help us grow?"
//!    - Bias: Learn from experience, improve over time, embrace change

use std::fmt;

/// The eight harmonies as an enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActiveHarmonic {
    Coherence,    // Resonant Coherence
    Flourishing,  // Pan-Sentient Flourishing
    Wisdom,       // Integral Wisdom
    Play,         // Infinite Play
    Interconnect, // Universal Interconnectedness
    Reciprocity,  // Mutual Reciprocity
    Evolution,    // Evolutionary Progression
}

impl ActiveHarmonic {
    /// Return the harmonic name as a static string, matching Debug output.
    /// Avoids `format!("{:?}", harmonic)` allocation on the hot path.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Coherence => "Coherence",
            Self::Flourishing => "Flourishing",
            Self::Wisdom => "Wisdom",
            Self::Play => "Play",
            Self::Interconnect => "Interconnect",
            Self::Reciprocity => "Reciprocity",
            Self::Evolution => "Evolution",
        }
    }

    /// The core question this harmony asks
    pub fn question(&self) -> &'static str {
        match self {
            Self::Coherence => "Does this hang together?",
            Self::Flourishing => "Does this serve flourishing?",
            Self::Wisdom => "What don't I know?",
            Self::Play => "What haven't I tried?",
            Self::Interconnect => "How is this connected?",
            Self::Reciprocity => "What am I giving/receiving?",
            Self::Evolution => "How does this help us grow?",
        }
    }

    /// The reasoning bias this harmony induces
    pub fn bias(&self) -> ReasoningBias {
        match self {
            Self::Coherence => ReasoningBias::Synthesize,
            Self::Flourishing => ReasoningBias::CareForImpact,
            Self::Wisdom => ReasoningBias::AcknowledgeUncertainty,
            Self::Play => ReasoningBias::ExploreNovelty,
            Self::Interconnect => ReasoningBias::TraceConnections,
            Self::Reciprocity => ReasoningBias::SeekFairExchange,
            Self::Evolution => ReasoningBias::LearnAndGrow,
        }
    }

    /// Full philosophical name
    pub fn full_name(&self) -> &'static str {
        match self {
            Self::Coherence => "Resonant Coherence (Love as Harmonious Integration)",
            Self::Flourishing => "Pan-Sentient Flourishing (Love as Unconditional Care)",
            Self::Wisdom => "Integral Wisdom (Love as Self-Illuminating Intelligence)",
            Self::Play => "Infinite Play (Love as Joyful Generativity)",
            Self::Interconnect => "Universal Interconnectedness (Love as Fundamental Unity)",
            Self::Reciprocity => "Mutual Reciprocity (Love as Generous Flow)",
            Self::Evolution => "Evolutionary Progression (Love as Wise Becoming)",
        }
    }

    /// All harmonics as an array for iteration
    pub fn all() -> [Self; 7] {
        [
            Self::Coherence,
            Self::Flourishing,
            Self::Wisdom,
            Self::Play,
            Self::Interconnect,
            Self::Reciprocity,
            Self::Evolution,
        ]
    }
}

impl fmt::Display for ActiveHarmonic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// How a harmony biases reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningBias {
    /// Seek synthesis, resolve contradictions, unify perspectives
    Synthesize,
    /// Prioritize wellbeing, consider impact on all affected beings
    CareForImpact,
    /// Epistemic humility, acknowledge uncertainty, seek understanding
    AcknowledgeUncertainty,
    /// Explore novelty, experiment, embrace creative risk
    ExploreNovelty,
    /// Trace relationships, see patterns, recognize interdependence
    TraceConnections,
    /// Fair exchange, mutual benefit, give back
    SeekFairExchange,
    /// Learn from experience, improve over time, embrace change
    LearnAndGrow,
}

impl ReasoningBias {
    /// How this bias affects primitive selection weights
    pub fn primitive_weight_modifier(&self, primitive_type: &str) -> f32 {
        match self {
            Self::Synthesize => match primitive_type {
                "INTEGRATE" | "BIND" | "UNIFY" => 1.5,
                "SEPARATE" | "DIVIDE" => 0.7,
                _ => 1.0,
            },
            Self::CareForImpact => match primitive_type {
                "PROTECT" | "NURTURE" | "SUPPORT" => 1.5,
                "HARM" | "NEGLECT" => 0.3,
                _ => 1.0,
            },
            Self::AcknowledgeUncertainty => match primitive_type {
                "QUESTION" | "DOUBT" | "EXPLORE" => 1.5,
                "ASSERT" | "CLAIM" => 0.8,
                _ => 1.0,
            },
            Self::ExploreNovelty => match primitive_type {
                "EXPLORE" | "CREATE" | "EXPERIMENT" => 1.5,
                "REPEAT" | "CONFORM" => 0.6,
                _ => 1.0,
            },
            Self::TraceConnections => match primitive_type {
                "RELATE" | "CONNECT" | "LINK" => 1.5,
                "ISOLATE" => 0.6,
                _ => 1.0,
            },
            Self::SeekFairExchange => match primitive_type {
                "GIVE" | "SHARE" | "EXCHANGE" => 1.5,
                "TAKE" | "HOARD" => 0.5,
                _ => 1.0,
            },
            Self::LearnAndGrow => match primitive_type {
                "LEARN" | "ADAPT" | "EVOLVE" => 1.5,
                "STAGNATE" | "RESIST" => 0.5,
                _ => 1.0,
            },
        }
    }
}

/// A single harmonic mode with its activation level
#[derive(Debug, Clone, Copy)]
pub struct HarmonicMode {
    pub harmonic: ActiveHarmonic,
    pub activation: f32, // 0.0 to 1.0
}

impl HarmonicMode {
    pub fn new(harmonic: ActiveHarmonic, activation: f32) -> Self {
        Self {
            harmonic,
            activation: activation.clamp(0.0, 1.0),
        }
    }

    /// Get the effective weight modifier for a primitive type
    pub fn weight_for_primitive(&self, primitive_type: &str) -> f32 {
        let base_modifier = self
            .harmonic
            .bias()
            .primitive_weight_modifier(primitive_type);
        // Activation scales the effect: at 0.5 activation, effect is neutral (1.0)
        // At 1.0 activation, full effect; at 0.0 activation, no effect
        let scale = (self.activation - 0.5) * 2.0; // -1.0 to 1.0
        if scale > 0.0 {
            1.0 + (base_modifier - 1.0) * scale
        } else {
            1.0
        }
    }
}

/// A question that a harmony is asking
#[derive(Debug, Clone)]
pub struct HarmonicQuestion {
    pub harmonic: ActiveHarmonic,
    pub question: &'static str,
    pub urgency: f32, // How strongly this question needs answering
}

impl HarmonicQuestion {
    pub fn from_harmonic(harmonic: ActiveHarmonic, urgency: f32) -> Self {
        Self {
            harmonic,
            question: harmonic.question(),
            urgency,
        }
    }
}

/// The complete profile of all seven harmonics
#[derive(Debug, Clone)]
pub struct HarmonicProfile {
    pub coherence: f32,
    pub flourishing: f32,
    pub wisdom: f32,
    pub play: f32,
    pub interconnect: f32,
    pub reciprocity: f32,
    pub evolution: f32,
}

impl HarmonicProfile {
    /// Create a balanced profile (all harmonies at 0.5)
    pub fn balanced() -> Self {
        Self {
            coherence: 0.5,
            flourishing: 0.5,
            wisdom: 0.5,
            play: 0.5,
            interconnect: 0.5,
            reciprocity: 0.5,
            evolution: 0.5,
        }
    }

    /// Get activation for a specific harmonic
    pub fn get(&self, harmonic: ActiveHarmonic) -> f32 {
        match harmonic {
            ActiveHarmonic::Coherence => self.coherence,
            ActiveHarmonic::Flourishing => self.flourishing,
            ActiveHarmonic::Wisdom => self.wisdom,
            ActiveHarmonic::Play => self.play,
            ActiveHarmonic::Interconnect => self.interconnect,
            ActiveHarmonic::Reciprocity => self.reciprocity,
            ActiveHarmonic::Evolution => self.evolution,
        }
    }

    /// Set activation for a specific harmonic
    pub fn set(&mut self, harmonic: ActiveHarmonic, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        match harmonic {
            ActiveHarmonic::Coherence => self.coherence = clamped,
            ActiveHarmonic::Flourishing => self.flourishing = clamped,
            ActiveHarmonic::Wisdom => self.wisdom = clamped,
            ActiveHarmonic::Play => self.play = clamped,
            ActiveHarmonic::Interconnect => self.interconnect = clamped,
            ActiveHarmonic::Reciprocity => self.reciprocity = clamped,
            ActiveHarmonic::Evolution => self.evolution = clamped,
        }
    }

    /// Boost a specific harmonic
    pub fn boost(&mut self, harmonic: ActiveHarmonic, amount: f32) {
        let current = self.get(harmonic);
        self.set(harmonic, current + amount);
    }

    /// Decay all harmonics toward the balanced state
    pub fn decay_toward_balance(&mut self, rate: f32) {
        let balance = 0.5;
        for harmonic in ActiveHarmonic::all() {
            let current = self.get(harmonic);
            let diff = balance - current;
            self.set(harmonic, current + diff * rate);
        }
    }

    /// Get the dominant (most active) harmonic
    pub fn dominant(&self) -> ActiveHarmonic {
        let values = [
            (ActiveHarmonic::Coherence, self.coherence),
            (ActiveHarmonic::Flourishing, self.flourishing),
            (ActiveHarmonic::Wisdom, self.wisdom),
            (ActiveHarmonic::Play, self.play),
            (ActiveHarmonic::Interconnect, self.interconnect),
            (ActiveHarmonic::Reciprocity, self.reciprocity),
            (ActiveHarmonic::Evolution, self.evolution),
        ];

        values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(h, _)| *h)
            .unwrap_or(ActiveHarmonic::Wisdom) // Default to wisdom
    }

    /// Get the top N most active harmonics with their questions
    pub fn active_questions(&self, n: usize) -> Vec<HarmonicQuestion> {
        let mut values: Vec<_> = ActiveHarmonic::all()
            .iter()
            .map(|h| (*h, self.get(*h)))
            .collect();

        values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        values
            .into_iter()
            .take(n)
            .filter(|(_, v)| *v > 0.5) // Only questions with above-baseline urgency
            .map(|(h, v)| HarmonicQuestion::from_harmonic(h, v))
            .collect()
    }

    /// Calculate combined weight modifier for a primitive type
    pub fn combined_weight_for_primitive(&self, primitive_type: &str) -> f32 {
        let mut total_weight = 0.0;
        let mut total_activation = 0.0;

        for harmonic in ActiveHarmonic::all() {
            let activation = self.get(harmonic);
            let mode = HarmonicMode::new(harmonic, activation);
            total_weight += mode.weight_for_primitive(primitive_type) * activation;
            total_activation += activation;
        }

        if total_activation > 0.0 {
            total_weight / total_activation
        } else {
            1.0
        }
    }

    /// Get a vector representation for HDC operations
    pub fn as_vector(&self) -> [f32; 7] {
        [
            self.coherence,
            self.flourishing,
            self.wisdom,
            self.play,
            self.interconnect,
            self.reciprocity,
            self.evolution,
        ]
    }

    /// Create from a vector
    pub fn from_vector(v: [f32; 7]) -> Self {
        Self {
            coherence: v[0].clamp(0.0, 1.0),
            flourishing: v[1].clamp(0.0, 1.0),
            wisdom: v[2].clamp(0.0, 1.0),
            play: v[3].clamp(0.0, 1.0),
            interconnect: v[4].clamp(0.0, 1.0),
            reciprocity: v[5].clamp(0.0, 1.0),
            evolution: v[6].clamp(0.0, 1.0),
        }
    }
}

impl Default for HarmonicProfile {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_questions() {
        assert_eq!(ActiveHarmonic::Wisdom.question(), "What don't I know?");
        assert_eq!(ActiveHarmonic::Play.question(), "What haven't I tried?");
    }

    #[test]
    fn test_reasoning_bias_affects_primitives() {
        let bias = ReasoningBias::Synthesize;
        assert!(bias.primitive_weight_modifier("INTEGRATE") > 1.0);
        assert!(bias.primitive_weight_modifier("SEPARATE") < 1.0);
    }

    #[test]
    fn test_profile_dominant() {
        let mut profile = HarmonicProfile::balanced();
        profile.wisdom = 0.9;
        assert_eq!(profile.dominant(), ActiveHarmonic::Wisdom);
    }

    #[test]
    fn test_profile_boost_and_decay() {
        let mut profile = HarmonicProfile::balanced();
        profile.boost(ActiveHarmonic::Play, 0.3);
        assert!(profile.play > 0.5);

        profile.decay_toward_balance(0.5);
        assert!(profile.play < 0.8); // Should have decayed toward 0.5
    }

    #[test]
    fn test_combined_primitive_weight() {
        let mut profile = HarmonicProfile::balanced();
        profile.wisdom = 0.9; // Boost wisdom

        // QUESTION should be weighted higher when wisdom is dominant
        let weight = profile.combined_weight_for_primitive("QUESTION");
        assert!(weight > 1.0);
    }

    #[test]
    fn test_active_questions() {
        let mut profile = HarmonicProfile::balanced();
        profile.wisdom = 0.8;
        profile.play = 0.7;

        let questions = profile.active_questions(3);
        assert!(!questions.is_empty());
        assert_eq!(questions[0].harmonic, ActiveHarmonic::Wisdom);
    }

    // ── ActiveHarmonic enum ─────────────────────────────────────────────

    #[test]
    fn test_all_harmonics_returns_seven() {
        assert_eq!(ActiveHarmonic::all().len(), 7);
    }

    #[test]
    fn test_each_harmonic_has_distinct_question() {
        let questions: Vec<&str> = ActiveHarmonic::all().iter().map(|h| h.question()).collect();
        for i in 0..questions.len() {
            for j in (i + 1)..questions.len() {
                assert_ne!(questions[i], questions[j], "Questions should be unique");
            }
        }
    }

    #[test]
    fn test_each_harmonic_has_distinct_bias() {
        let biases: Vec<ReasoningBias> = ActiveHarmonic::all().iter().map(|h| h.bias()).collect();
        for i in 0..biases.len() {
            for j in (i + 1)..biases.len() {
                assert_ne!(biases[i], biases[j], "Biases should be unique");
            }
        }
    }

    #[test]
    fn test_full_name_nonempty() {
        for h in ActiveHarmonic::all() {
            assert!(!h.full_name().is_empty());
        }
    }

    #[test]
    fn test_display_impl() {
        let s = format!("{}", ActiveHarmonic::Coherence);
        assert_eq!(s, "Coherence");
    }

    // ── ReasoningBias ───────────────────────────────────────────────────

    #[test]
    fn test_unknown_primitive_returns_neutral_weight() {
        for h in ActiveHarmonic::all() {
            let w = h.bias().primitive_weight_modifier("UNKNOWN_PRIMITIVE");
            assert!(
                (w - 1.0).abs() < f32::EPSILON,
                "Unknown primitive should get 1.0"
            );
        }
    }

    #[test]
    fn test_all_biases_positive_weights() {
        let primitives = [
            "INTEGRATE",
            "PROTECT",
            "QUESTION",
            "EXPLORE",
            "RELATE",
            "GIVE",
            "LEARN",
        ];
        for prim in &primitives {
            for h in ActiveHarmonic::all() {
                let w = h.bias().primitive_weight_modifier(prim);
                assert!(
                    w > 0.0,
                    "Weight for {} under {:?} must be positive",
                    prim,
                    h
                );
            }
        }
    }

    // ── HarmonicMode ────────────────────────────────────────────────────

    #[test]
    fn test_harmonic_mode_clamps_activation() {
        let mode = HarmonicMode::new(ActiveHarmonic::Play, 5.0);
        assert!((mode.activation - 1.0).abs() < f32::EPSILON);

        let mode2 = HarmonicMode::new(ActiveHarmonic::Play, -1.0);
        assert!((mode2.activation - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_weight_for_primitive_at_half_activation() {
        let mode = HarmonicMode::new(ActiveHarmonic::Coherence, 0.5);
        // At activation 0.5, scale = 0 so result = 1.0 (neutral)
        let w = mode.weight_for_primitive("INTEGRATE");
        assert!((w - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_weight_for_primitive_at_full_activation() {
        let mode = HarmonicMode::new(ActiveHarmonic::Coherence, 1.0);
        let w = mode.weight_for_primitive("INTEGRATE");
        // At activation 1.0, full bias applies: 1.0 + (1.5 - 1.0)*1.0 = 1.5
        assert!((w - 1.5).abs() < f32::EPSILON);
    }

    // ── HarmonicProfile ─────────────────────────────────────────────────

    #[test]
    fn test_profile_set_clamps() {
        let mut profile = HarmonicProfile::balanced();
        profile.set(ActiveHarmonic::Wisdom, 2.0);
        assert!((profile.get(ActiveHarmonic::Wisdom) - 1.0).abs() < f32::EPSILON);

        profile.set(ActiveHarmonic::Wisdom, -1.0);
        assert!((profile.get(ActiveHarmonic::Wisdom) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_as_vector_round_trip() {
        let original = HarmonicProfile::balanced();
        let v = original.as_vector();
        let restored = HarmonicProfile::from_vector(v);
        assert_eq!(original.as_vector(), restored.as_vector());
    }

    #[test]
    fn test_from_vector_clamps() {
        let v = [2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5];
        let profile = HarmonicProfile::from_vector(v);
        assert!((profile.coherence - 1.0).abs() < f32::EPSILON);
        assert!((profile.flourishing - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_active_questions_empty_when_all_at_baseline() {
        let profile = HarmonicProfile::balanced();
        // All at exactly 0.5 -- filter requires > 0.5
        let questions = profile.active_questions(7);
        assert!(
            questions.is_empty(),
            "No questions when all at baseline 0.5"
        );
    }

    #[test]
    fn test_harmonic_question_from_harmonic() {
        let q = HarmonicQuestion::from_harmonic(ActiveHarmonic::Evolution, 0.9);
        assert_eq!(q.harmonic, ActiveHarmonic::Evolution);
        assert!((q.urgency - 0.9).abs() < f32::EPSILON);
        assert_eq!(q.question, "How does this help us grow?");
    }

    #[test]
    fn test_default_profile_is_balanced() {
        let profile = HarmonicProfile::default();
        for h in ActiveHarmonic::all() {
            assert!((profile.get(h) - 0.5).abs() < f32::EPSILON);
        }
    }
}
