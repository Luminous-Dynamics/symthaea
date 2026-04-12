// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
            assert!(
                h.base_weight() > 0.0,
                "{} weight must be positive",
                h.name()
            );
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
        assert_eq!(
            first, second,
            "all() must return same order on repeated calls"
        );
    }

    #[test]
    fn test_harmony_clone_eq() {
        for h in Harmony::all() {
            let cloned = h;
            assert_eq!(h, cloned, "Clone of {:?} must equal original", h);
        }
    }

    // ========================================================================
    // Additional test modules
    // ========================================================================

    mod n_harmonies_constant {
        use super::*;

        #[test]
        fn equals_eight() {
            assert_eq!(N_HARMONIES, 8);
        }

        #[test]
        fn matches_all_len() {
            assert_eq!(N_HARMONIES, Harmony::all().len());
        }
    }

    mod harmony_enum_identity {
        use super::*;

        #[test]
        fn copy_semantics_preserve_identity() {
            let a = Harmony::ResonantCoherence;
            let b = a; // Copy
            assert_eq!(a, b);
            // a is still usable after copy
            assert_eq!(a.name(), "Resonant Coherence");
        }

        #[test]
        fn clone_equals_original() {
            for h in Harmony::all() {
                #[allow(clippy::clone_on_copy)]
                let c = h.clone();
                assert_eq!(h, c);
            }
        }

        #[test]
        fn debug_output_nonempty() {
            for h in Harmony::all() {
                let debug = format!("{:?}", h);
                assert!(!debug.is_empty());
                // Debug should contain variant name (no "Harmony::" prefix in Debug derive)
                assert!(
                    !debug.contains("()"),
                    "Debug should not contain empty parens: {debug}"
                );
            }
        }

        #[test]
        fn debug_contains_variant_name() {
            assert!(format!("{:?}", Harmony::ResonantCoherence).contains("ResonantCoherence"));
            assert!(format!("{:?}", Harmony::SacredStillness).contains("SacredStillness"));
        }

        #[test]
        fn partial_eq_reflexive() {
            for h in Harmony::all() {
                assert_eq!(h, h);
            }
        }

        #[test]
        fn partial_eq_different_variants() {
            assert_ne!(Harmony::ResonantCoherence, Harmony::SacredStillness);
            assert_ne!(Harmony::IntegralWisdom, Harmony::InfinitePlay);
            assert_ne!(
                Harmony::PanSentientFlourishing,
                Harmony::EvolutionaryProgression
            );
        }

        #[test]
        fn eq_symmetric() {
            let a = Harmony::IntegralWisdom;
            let b = Harmony::IntegralWisdom;
            assert_eq!(a, b);
            assert_eq!(b, a);
        }
    }

    mod harmony_hash {
        use super::*;
        use std::collections::HashSet;

        #[test]
        fn all_variants_hashable_and_distinct() {
            let set: HashSet<Harmony> = Harmony::all().iter().copied().collect();
            assert_eq!(set.len(), N_HARMONIES);
        }

        #[test]
        fn same_variant_hashes_to_same_bucket() {
            let mut set = HashSet::new();
            set.insert(Harmony::SacredReciprocity);
            // Inserting the same variant should not increase size
            set.insert(Harmony::SacredReciprocity);
            assert_eq!(set.len(), 1);
        }

        #[test]
        fn hashset_contains_lookup_works() {
            let set: HashSet<Harmony> = Harmony::all().iter().copied().collect();
            for h in Harmony::all() {
                assert!(set.contains(&h), "HashSet should contain {:?}", h);
            }
        }

        #[test]
        fn hashmap_as_key() {
            use std::collections::HashMap;
            let mut map = HashMap::new();
            for (i, h) in Harmony::all().iter().enumerate() {
                map.insert(*h, i);
            }
            assert_eq!(map.len(), N_HARMONIES);
            assert_eq!(map[&Harmony::ResonantCoherence], 0);
            assert_eq!(map[&Harmony::SacredStillness], 7);
        }
    }

    mod harmony_display {
        use super::*;

        #[test]
        fn display_matches_name_for_each_variant() {
            assert_eq!(Harmony::ResonantCoherence.to_string(), "Resonant Coherence");
            assert_eq!(
                Harmony::PanSentientFlourishing.to_string(),
                "Pan-Sentient Flourishing"
            );
            assert_eq!(Harmony::IntegralWisdom.to_string(), "Integral Wisdom");
            assert_eq!(Harmony::InfinitePlay.to_string(), "Infinite Play");
            assert_eq!(
                Harmony::UniversalInterconnectedness.to_string(),
                "Universal Interconnectedness"
            );
            assert_eq!(Harmony::SacredReciprocity.to_string(), "Sacred Reciprocity");
            assert_eq!(
                Harmony::EvolutionaryProgression.to_string(),
                "Evolutionary Progression"
            );
            assert_eq!(Harmony::SacredStillness.to_string(), "Sacred Stillness");
        }

        #[test]
        fn display_in_format_string() {
            let h = Harmony::InfinitePlay;
            let msg = format!("Harmony: {h}");
            assert_eq!(msg, "Harmony: Infinite Play");
        }
    }

    mod harmony_serde {
        use super::*;

        #[test]
        fn serialize_to_string_variant() {
            let json = serde_json::to_string(&Harmony::ResonantCoherence).unwrap();
            assert_eq!(json, "\"ResonantCoherence\"");
        }

        #[test]
        fn serialize_each_variant_is_quoted_string() {
            for h in Harmony::all() {
                let json = serde_json::to_string(&h).unwrap();
                assert!(json.starts_with('"') && json.ends_with('"'));
            }
        }

        #[test]
        fn deserialize_from_string_variant() {
            let h: Harmony = serde_json::from_str("\"SacredStillness\"").unwrap();
            assert_eq!(h, Harmony::SacredStillness);
        }

        #[test]
        fn deserialize_invalid_variant_fails() {
            let result = serde_json::from_str::<Harmony>("\"NotAHarmony\"");
            assert!(result.is_err());
        }

        #[test]
        fn deserialize_number_fails() {
            let result = serde_json::from_str::<Harmony>("42");
            assert!(result.is_err());
        }

        #[test]
        fn deserialize_null_fails() {
            let result = serde_json::from_str::<Harmony>("null");
            assert!(result.is_err());
        }

        #[test]
        fn deserialize_empty_string_fails() {
            let result = serde_json::from_str::<Harmony>("\"\"");
            assert!(result.is_err());
        }

        #[test]
        fn deserialize_case_sensitive() {
            // lowercase should fail
            let result = serde_json::from_str::<Harmony>("\"resonantcoherence\"");
            assert!(result.is_err());
        }

        #[test]
        fn roundtrip_all_variants_via_json_value() {
            for h in Harmony::all() {
                let val = serde_json::to_value(&h).unwrap();
                assert!(val.is_string());
                let back: Harmony = serde_json::from_value(val).unwrap();
                assert_eq!(h, back);
            }
        }

        #[test]
        fn serialize_array_of_harmonies() {
            let all = Harmony::all();
            let json = serde_json::to_string(&all).unwrap();
            let back: Vec<Harmony> = serde_json::from_str(&json).unwrap();
            assert_eq!(back.len(), N_HARMONIES);
            for (a, b) in all.iter().zip(back.iter()) {
                assert_eq!(a, b);
            }
        }

        #[test]
        fn serialize_pretty_roundtrip() {
            let h = Harmony::EvolutionaryProgression;
            let pretty = serde_json::to_string_pretty(&h).unwrap();
            let back: Harmony = serde_json::from_str(&pretty).unwrap();
            assert_eq!(h, back);
        }
    }

    mod harmony_all_method {
        use super::*;

        #[test]
        fn first_is_resonant_coherence() {
            assert_eq!(Harmony::all()[0], Harmony::ResonantCoherence);
        }

        #[test]
        fn last_is_sacred_stillness() {
            assert_eq!(Harmony::all()[N_HARMONIES - 1], Harmony::SacredStillness);
        }

        #[test]
        fn no_duplicates() {
            let all = Harmony::all();
            for (i, a) in all.iter().enumerate() {
                for (j, b) in all.iter().enumerate() {
                    if i != j {
                        assert_ne!(a, b, "Duplicate at indices {i} and {j}");
                    }
                }
            }
        }

        #[test]
        fn iteration_covers_all() {
            let all = Harmony::all();
            assert!(all.contains(&Harmony::ResonantCoherence));
            assert!(all.contains(&Harmony::PanSentientFlourishing));
            assert!(all.contains(&Harmony::IntegralWisdom));
            assert!(all.contains(&Harmony::InfinitePlay));
            assert!(all.contains(&Harmony::UniversalInterconnectedness));
            assert!(all.contains(&Harmony::SacredReciprocity));
            assert!(all.contains(&Harmony::EvolutionaryProgression));
            assert!(all.contains(&Harmony::SacredStillness));
        }
    }

    mod harmony_name_method {
        use super::*;

        #[test]
        fn names_are_distinct() {
            let names: Vec<&str> = Harmony::all().iter().map(|h| h.name()).collect();
            let set: std::collections::HashSet<&&str> = names.iter().collect();
            assert_eq!(set.len(), N_HARMONIES);
        }

        #[test]
        fn names_contain_spaces() {
            // All harmony names are multi-word
            for h in Harmony::all() {
                assert!(
                    h.name().contains(' '),
                    "{:?} name '{}' should contain a space",
                    h,
                    h.name()
                );
            }
        }

        #[test]
        fn names_start_with_uppercase() {
            for h in Harmony::all() {
                let first = h.name().chars().next().unwrap();
                assert!(
                    first.is_uppercase(),
                    "{:?} name '{}' should start uppercase",
                    h,
                    h.name()
                );
            }
        }
    }

    mod harmony_description_method {
        use super::*;

        #[test]
        fn descriptions_are_distinct() {
            let descs: Vec<&str> = Harmony::all().iter().map(|h| h.description()).collect();
            let set: std::collections::HashSet<&&str> = descs.iter().collect();
            assert_eq!(set.len(), N_HARMONIES);
        }

        #[test]
        fn descriptions_are_substantial() {
            for h in Harmony::all() {
                assert!(
                    h.description().len() > 20,
                    "{:?} description too short: '{}'",
                    h,
                    h.description()
                );
            }
        }
    }

    mod harmony_sacred_question_method {
        use super::*;

        #[test]
        fn all_questions_end_with_question_mark() {
            for h in Harmony::all() {
                assert!(
                    h.sacred_question().ends_with('?'),
                    "{:?} sacred_question '{}' should end with '?'",
                    h,
                    h.sacred_question()
                );
            }
        }

        #[test]
        fn all_questions_start_with_does() {
            for h in Harmony::all() {
                assert!(
                    h.sacred_question().starts_with("Does"),
                    "{:?} sacred_question '{}' should start with 'Does'",
                    h,
                    h.sacred_question()
                );
            }
        }
    }

    mod harmony_epistemic_mode_method {
        use super::*;

        #[test]
        fn all_modes_end_with_knowing() {
            for h in Harmony::all() {
                assert!(
                    h.epistemic_mode().ends_with("-Knowing"),
                    "{:?} epistemic_mode '{}' should end with '-Knowing'",
                    h,
                    h.epistemic_mode()
                );
            }
        }

        #[test]
        fn modes_are_distinct() {
            let modes: Vec<&str> = Harmony::all().iter().map(|h| h.epistemic_mode()).collect();
            let set: std::collections::HashSet<&&str> = modes.iter().collect();
            assert_eq!(set.len(), N_HARMONIES);
        }
    }

    mod harmony_focus_question_method {
        use super::*;

        #[test]
        fn all_focus_questions_end_with_question_mark() {
            for h in Harmony::all() {
                assert!(
                    h.focus_question().ends_with('?'),
                    "{:?} focus_question '{}' should end with '?'",
                    h,
                    h.focus_question()
                );
            }
        }

        #[test]
        fn focus_questions_are_distinct() {
            let qs: Vec<&str> = Harmony::all().iter().map(|h| h.focus_question()).collect();
            let set: std::collections::HashSet<&&str> = qs.iter().collect();
            assert_eq!(set.len(), N_HARMONIES);
        }

        #[test]
        fn focus_questions_start_with_interrogative() {
            for h in Harmony::all() {
                let q = h.focus_question();
                assert!(
                    q.starts_with("How") || q.starts_with("What") || q.starts_with("Who"),
                    "{:?} focus_question '{}' should start with How/What/Who",
                    h,
                    q
                );
            }
        }
    }

    mod harmony_code_method {
        use super::*;

        #[test]
        fn specific_code_values() {
            assert_eq!(Harmony::ResonantCoherence.code(), "RC");
            assert_eq!(Harmony::PanSentientFlourishing.code(), "PSF");
            assert_eq!(Harmony::IntegralWisdom.code(), "IW");
            assert_eq!(Harmony::InfinitePlay.code(), "IP");
            assert_eq!(Harmony::UniversalInterconnectedness.code(), "UI");
            assert_eq!(Harmony::SacredReciprocity.code(), "SR");
            assert_eq!(Harmony::EvolutionaryProgression.code(), "EP");
            assert_eq!(Harmony::SacredStillness.code(), "SS");
        }

        #[test]
        fn codes_are_all_uppercase() {
            for h in Harmony::all() {
                assert!(
                    h.code().chars().all(|c| c.is_uppercase()),
                    "{:?} code '{}' should be all uppercase",
                    h,
                    h.code()
                );
            }
        }

        #[test]
        fn codes_are_alphabetic() {
            for h in Harmony::all() {
                assert!(
                    h.code().chars().all(|c| c.is_alphabetic()),
                    "{:?} code '{}' should be all alphabetic",
                    h,
                    h.code()
                );
            }
        }
    }

    mod harmony_base_weight_method {
        use super::*;

        #[test]
        fn care_harmonies_have_highest_weight() {
            let rc_w = Harmony::ResonantCoherence.base_weight();
            let psf_w = Harmony::PanSentientFlourishing.base_weight();
            // RC and PSF should be the highest
            for h in Harmony::all() {
                assert!(
                    h.base_weight() <= rc_w || h.base_weight() <= psf_w,
                    "{:?} weight {} exceeds care harmonies",
                    h,
                    h.base_weight()
                );
            }
        }

        #[test]
        fn rc_and_psf_equal_weight() {
            assert_eq!(
                Harmony::ResonantCoherence.base_weight(),
                Harmony::PanSentientFlourishing.base_weight()
            );
        }

        #[test]
        fn weights_are_finite() {
            for h in Harmony::all() {
                let w = h.base_weight();
                assert!(w.is_finite(), "{:?} weight is not finite", h);
                assert!(!w.is_nan(), "{:?} weight is NaN", h);
            }
        }

        #[test]
        fn specific_weight_values() {
            assert!((Harmony::ResonantCoherence.base_weight() - 0.17).abs() < f32::EPSILON);
            assert!((Harmony::InfinitePlay.base_weight() - 0.09).abs() < f32::EPSILON);
            assert!((Harmony::IntegralWisdom.base_weight() - 0.13).abs() < f32::EPSILON);
            assert!((Harmony::SacredStillness.base_weight() - 0.13).abs() < f32::EPSILON);
        }

        #[test]
        fn three_weight_tiers() {
            // Should have exactly 3 distinct weight levels
            let mut weights: Vec<u32> = Harmony::all()
                .iter()
                .map(|h| (h.base_weight() * 100.0) as u32)
                .collect();
            weights.sort();
            weights.dedup();
            assert_eq!(
                weights.len(),
                3,
                "Expected 3 distinct weight tiers, got {weights:?}"
            );
        }
    }

    mod harmony_cross_method_consistency {
        use super::*;

        #[test]
        fn name_appears_in_description_or_question_context() {
            // Every harmony should have coherent name/description/question
            for h in Harmony::all() {
                // Name, description, and sacred question should all be non-trivially different
                assert_ne!(h.name(), h.description());
                assert_ne!(h.name(), h.sacred_question());
                assert_ne!(h.description(), h.sacred_question());
            }
        }

        #[test]
        fn epistemic_mode_and_code_both_defined_for_all() {
            for h in Harmony::all() {
                assert!(!h.epistemic_mode().is_empty());
                assert!(!h.code().is_empty());
                // They should be different
                assert_ne!(h.epistemic_mode(), h.code());
            }
        }

        #[test]
        fn focus_question_differs_from_sacred_question() {
            for h in Harmony::all() {
                assert_ne!(
                    h.focus_question(),
                    h.sacred_question(),
                    "{:?} has identical focus and sacred questions",
                    h
                );
            }
        }
    }

    mod harmony_size_and_layout {
        use super::*;

        #[test]
        fn enum_is_small() {
            // Harmony should be a simple enum, small enough to copy cheaply
            assert!(
                std::mem::size_of::<Harmony>() <= 2,
                "Harmony should be at most 2 bytes, got {}",
                std::mem::size_of::<Harmony>()
            );
        }

        #[test]
        fn option_harmony_same_size_or_one_byte_more() {
            let h_size = std::mem::size_of::<Harmony>();
            let opt_size = std::mem::size_of::<Option<Harmony>>();
            // Option<Harmony> should use niche optimization or be at most 1 byte larger
            assert!(
                opt_size <= h_size + 1,
                "Option<Harmony> size {opt_size} unexpectedly large vs Harmony size {h_size}"
            );
        }
    }
}
