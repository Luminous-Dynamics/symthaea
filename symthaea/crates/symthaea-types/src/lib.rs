//! # Symthaea Shared Types
//!
//! Shared types for the Symthaea ecosystem. This crate provides canonical
//! definitions of types that are used across multiple sub-crates, preventing
//! duplication and ensuring consistency.
//!
//! ## Seven Harmonies
//!
//! The [`Harmony`] enum represents the Seven Primary Harmonies of the Kosmic Song,
//! used for value alignment, epistemic analysis, and consciousness evaluation.

use serde::{Deserialize, Serialize};

/// The Seven Primary Harmonies of Infinite Love
///
/// Each harmony represents both a value dimension AND an epistemic lens
/// through which knowledge is perceived and evaluated.
///
/// # Value Alignment
///
/// Used by [`SevenHarmonies`](crate) evaluators to assess whether actions
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
}

impl Harmony {
    /// Get all harmonies in canonical order
    pub fn all() -> [Harmony; 7] {
        [
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::InfinitePlay,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
            Harmony::EvolutionaryProgression,
        ]
    }

    // ========================================================================
    // Value alignment methods (from seven_harmonies)
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
        }
    }

    /// Get the sacred question for this harmony
    pub fn sacred_question(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Does this create harmony and integration?",
            Harmony::PanSentientFlourishing => "Does this serve the flourishing of all beings?",
            Harmony::IntegralWisdom => "Does this arise from and cultivate wisdom?",
            Harmony::InfinitePlay => "Does this celebrate creativity and joy?",
            Harmony::UniversalInterconnectedness => {
                "Does this honor our fundamental connection?"
            }
            Harmony::SacredReciprocity => {
                "Does this participate in the generous flow of giving?"
            }
            Harmony::EvolutionaryProgression => "Does this contribute to wise evolution?",
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
            Harmony::ResonantCoherence => 0.20,
            Harmony::PanSentientFlourishing => 0.20,
            Harmony::IntegralWisdom => 0.15,
            Harmony::InfinitePlay => 0.10,
            Harmony::UniversalInterconnectedness => 0.15,
            Harmony::SacredReciprocity => 0.10,
            Harmony::EvolutionaryProgression => 0.10,
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
    fn test_harmony_all_returns_seven() {
        assert_eq!(Harmony::all().len(), 7);
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
}
