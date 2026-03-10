//! # Symthaea Shared Types
//!
//! Shared types for the Symthaea ecosystem. This crate provides canonical
//! definitions of types that are used across multiple sub-crates, preventing
//! duplication and ensuring consistency.
//!
//! ## Eight Harmonies
//!
//! The [`Harmony`] enum represents the Eight Primary Harmonies of the Kosmic Song,
//! used for value alignment, epistemic analysis, and consciousness evaluation.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Number of harmonies (Eight Harmonies including Sacred Stillness).
pub const N_HARMONIES: usize = 8;

/// The Eight Primary Harmonies of Infinite Love
///
/// Each harmony represents both a value dimension AND an epistemic lens
/// through which knowledge is perceived and evaluated.
///
/// # Value Alignment
///
/// Used by [`EightHarmonies`](crate) evaluators to assess whether actions
/// align with deep ethical and consciousness principles.
///
/// # Epistemic Lens (GIS v4.0)
///
/// Each harmony has an epistemic mode (e.g., "Care-Knowing" for Pan-Sentient
/// Flourishing) and a focus question that guides knowledge assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Harmony {
    /// Resonant Coherence — Integration-Knowing
    /// Harmonious integration, luminous order, boundless creativity
    ResonantCoherence,

    /// Pan-Sentient Flourishing — Care-Knowing
    /// Unconditional care for all sentient beings, intrinsic value, holistic well-being
    PanSentientFlourishing,

    /// Integral Wisdom — Truth-Knowing
    /// Self-illuminating intelligence, embodied knowing, wisdom in action
    IntegralWisdom,

    /// Infinite Play — Creative-Knowing
    /// Joyful generativity, divine play, endless novelty and exploration
    InfinitePlay,

    /// Universal Interconnectedness — Relational-Knowing
    /// Fundamental unity of all existence, empathic resonance across beings
    UniversalInterconnectedness,

    /// Sacred Reciprocity — Exchange-Knowing
    /// Generous flow between beings, mutual upliftment, generative trust
    SacredReciprocity,

    /// Evolutionary Progression — Developmental-Knowing
    /// Wise becoming through time, continuous evolution toward greater consciousness
    EvolutionaryProgression,

    /// Sacred Stillness — Apophatic-Knowing
    /// Rest, silence, release, surrender, the void from which all arises
    SacredStillness,
}

impl Harmony {
    /// Get all harmonies in canonical order
    pub fn all() -> [Harmony; N_HARMONIES] {
        [
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::InfinitePlay,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
            Harmony::EvolutionaryProgression,
            Harmony::SacredStillness,
        ]
    }

    // ========================================================================
    // Value alignment methods (from eight_harmonies)
    // ========================================================================

    /// Get the human-readable name of this harmony
    pub fn name(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Resonant Coherence",
            Harmony::PanSentientFlourishing => "Pan-Sentient Flourishing",
            Harmony::IntegralWisdom => "Integral Wisdom",
            Harmony::InfinitePlay => "Infinite Play",
            Harmony::UniversalInterconnectedness => "Universal Interconnectedness",
            Harmony::SacredReciprocity => "Sacred Reciprocity",
            Harmony::EvolutionaryProgression => "Evolutionary Progression",
            Harmony::SacredStillness => "Sacred Stillness",
        }
    }

    /// Get a description of this harmony's principles
    pub fn description(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => {
                "Harmonious integration, luminous order, boundless creativity"
            }
            Harmony::PanSentientFlourishing => {
                "Unconditional care for all sentient beings, intrinsic value, holistic well-being"
            }
            Harmony::IntegralWisdom => {
                "Self-illuminating intelligence, embodied knowing, wisdom in action"
            }
            Harmony::InfinitePlay => {
                "Joyful generativity, divine play, endless novelty and exploration"
            }
            Harmony::UniversalInterconnectedness => {
                "Fundamental unity of all existence, empathic resonance across beings"
            }
            Harmony::SacredReciprocity => {
                "Generous flow between beings, mutual upliftment, generative trust"
            }
            Harmony::EvolutionaryProgression => {
                "Wise becoming through time, continuous evolution toward greater consciousness"
            }
            Harmony::SacredStillness => {
                "Rest, silence, release, surrender, the void from which all arises"
            }
        }
    }

    /// Get the sacred question for this harmony
    pub fn sacred_question(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Does this create harmony and integration?",
            Harmony::PanSentientFlourishing => "Does this serve the flourishing of all beings?",
            Harmony::IntegralWisdom => "Does this arise from and cultivate wisdom?",
            Harmony::InfinitePlay => "Does this celebrate creativity and joy?",
            Harmony::UniversalInterconnectedness => "Does this honor our fundamental connection?",
            Harmony::SacredReciprocity => "Does this participate in the generous flow of giving?",
            Harmony::EvolutionaryProgression => "Does this contribute to wise evolution?",
            Harmony::SacredStillness => {
                "Does this honor the need for rest, release, and not-knowing?"
            }
        }
    }

    // ========================================================================
    // Epistemic lens methods (from GIS v4.0 ignorance_types)
    // ========================================================================

    /// Base importance weight for this harmony
    ///
    /// These weights reflect the relative importance in the Kosmic Song.
    /// Care-related harmonies (RC, PSF) have highest weights.
    /// Weights sum to 1.0.
    pub fn base_weight(&self) -> f32 {
        match self {
            Harmony::ResonantCoherence => 0.17,
            Harmony::PanSentientFlourishing => 0.17,
            Harmony::IntegralWisdom => 0.13,
            Harmony::InfinitePlay => 0.09,
            Harmony::UniversalInterconnectedness => 0.13,
            Harmony::SacredReciprocity => 0.09,
            Harmony::EvolutionaryProgression => 0.09,
            Harmony::SacredStillness => 0.13,
        }
    }

    /// Get the epistemic mode name for this harmony's way of knowing
    pub fn epistemic_mode(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Integration-Knowing",
            Harmony::PanSentientFlourishing => "Care-Knowing",
            Harmony::IntegralWisdom => "Truth-Knowing",
            Harmony::InfinitePlay => "Creative-Knowing",
            Harmony::UniversalInterconnectedness => "Relational-Knowing",
            Harmony::SacredReciprocity => "Exchange-Knowing",
            Harmony::EvolutionaryProgression => "Developmental-Knowing",
            Harmony::SacredStillness => "Apophatic-Knowing",
        }
    }

    /// Get the focus question for this harmony's epistemic lens
    pub fn focus_question(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "How do the parts relate?",
            Harmony::PanSentientFlourishing => "Who is affected?",
            Harmony::IntegralWisdom => "What is verifiable?",
            Harmony::InfinitePlay => "What possibilities exist?",
            Harmony::UniversalInterconnectedness => "What connections exist?",
            Harmony::SacredReciprocity => "What flows back?",
            Harmony::EvolutionaryProgression => "What is emerging?",
            Harmony::SacredStillness => "What must be released?",
        }
    }

    /// Short code for E/N/M/H notation
    pub fn code(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "RC",
            Harmony::PanSentientFlourishing => "PSF",
            Harmony::IntegralWisdom => "IW",
            Harmony::InfinitePlay => "IP",
            Harmony::UniversalInterconnectedness => "UI",
            Harmony::SacredReciprocity => "SR",
            Harmony::EvolutionaryProgression => "EP",
            Harmony::SacredStillness => "SS",
        }
    }
}

impl std::fmt::Display for Harmony {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_all_returns_expected_count() {
        assert_eq!(Harmony::all().len(), N_HARMONIES);
    }

    #[test]
    fn test_harmony_weights_sum_to_one() {
        let total: f32 = Harmony::all().iter().map(|h| h.base_weight()).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Harmony weights should sum to ~1.0, got {total}",
        );
    }

    #[test]
    fn test_harmony_codes_unique() {
        let codes: Vec<&str> = Harmony::all().iter().map(|h| h.code()).collect();
        for (i, c) in codes.iter().enumerate() {
            for (j, d) in codes.iter().enumerate() {
                if i != j {
                    assert_ne!(c, d, "Codes must be unique");
                }
            }
        }
    }

    #[test]
    fn test_harmony_display_matches_name() {
        for h in Harmony::all() {
            assert_eq!(format!("{h}"), h.name());
        }
    }

    #[test]
    fn test_harmony_serialize_roundtrip() {
        for h in Harmony::all() {
            let json = serde_json::to_string(&h).unwrap();
            let back: Harmony = serde_json::from_str(&json).unwrap();
            assert_eq!(h, back);
        }
    }

    #[test]
    fn test_harmony_name_nonempty() {
        for h in Harmony::all() {
            assert!(!h.name().is_empty());
            assert!(!h.description().is_empty());
            assert!(!h.sacred_question().is_empty());
            assert!(!h.epistemic_mode().is_empty());
            assert!(!h.focus_question().is_empty());
            assert!(!h.code().is_empty());
        }
    }

    #[test]
    fn test_epistemic_modes_are_distinct() {
        let modes: Vec<&str> = Harmony::all().iter().map(|h| h.epistemic_mode()).collect();
        for (i, m) in modes.iter().enumerate() {
            for (j, n) in modes.iter().enumerate() {
                if i != j {
                    assert_ne!(m, n, "Epistemic modes must be unique");
                }
            }
        }
    }

    #[test]
    fn test_sacred_questions_are_distinct() {
        let questions: Vec<&str> = Harmony::all().iter().map(|h| h.sacred_question()).collect();
        for (i, q) in questions.iter().enumerate() {
            for (j, r) in questions.iter().enumerate() {
                if i != j {
                    assert_ne!(q, r, "Sacred questions must be unique");
                }
            }
        }
    }

    #[test]
    fn test_base_weight_positive() {
        for h in Harmony::all() {
            assert!(h.base_weight() > 0.0, "{} weight must be positive", h.name());
        }
    }

    #[test]
    fn test_base_weight_reasonable_range() {
        for h in Harmony::all() {
            let w = h.base_weight();
            assert!(
                w >= 0.05 && w <= 0.25,
                "{} weight {} outside [0.05, 0.25]",
                h.name(),
                w
            );
        }
    }

    #[test]
    fn test_harmony_codes_length() {
        for h in Harmony::all() {
            let len = h.code().len();
            assert!(
                len >= 2 && len <= 3,
                "{} code '{}' length {} not in [2,3]",
                h.name(),
                h.code(),
                len
            );
        }
    }

    #[test]
    fn test_focus_questions_nonempty() {
        for h in Harmony::all() {
            assert!(
                !h.focus_question().is_empty(),
                "{} has empty focus_question",
                h.name()
            );
        }
    }

    #[test]
    fn test_harmony_all_ordering_stable() {
        let first = Harmony::all();
        let second = Harmony::all();
        assert_eq!(first, second, "all() must return same order on repeated calls");
    }

    #[test]
    fn test_harmony_clone_eq() {
        for h in Harmony::all() {
            let cloned = h;
            assert_eq!(h, cloned, "Clone of {:?} must equal original", h);
        }
    }
}
