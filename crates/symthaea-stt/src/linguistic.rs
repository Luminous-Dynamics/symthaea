// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Linguistic Constraints for HDC Speech Recognition
//!
//! This module encodes linguistic structure directly as hypervectors:
//! - Phoneme classes (manner, voicing, place of articulation)
//! - Phonotactic constraints (legal sequences)
//! - Syllable structure templates
//!
//! The key insight: Learn only what's messy (acoustics), encode what's structured (linguistics).

use crate::hdc::{BundleAccumulator, HV16};
use std::collections::HashMap;

/// Phoneme class definitions for English (ARPABET)
pub struct PhonemeClasses {
    /// Individual phoneme HVs (reserved for future phoneme lookup)
    #[allow(dead_code)]
    phonemes: HashMap<String, HV16>,

    // Manner of articulation classes
    pub stop: HV16,
    pub nasal: HV16,
    pub fricative: HV16,
    pub affricate: HV16,
    pub liquid: HV16,
    pub glide: HV16,
    pub vowel: HV16,

    // Voicing classes
    pub voiced: HV16,
    pub voiceless: HV16,

    // Place of articulation classes
    pub labial: HV16,
    pub dental: HV16,
    pub alveolar: HV16,
    pub palatal: HV16,
    pub velar: HV16,
    pub glottal: HV16,

    // Vowel features
    pub high_vowel: HV16,
    pub mid_vowel: HV16,
    pub low_vowel: HV16,
    pub front_vowel: HV16,
    pub central_vowel: HV16,
    pub back_vowel: HV16,

    // Phonotactic position constraints
    pub word_initial: HV16,
    pub word_final: HV16,
}

impl PhonemeClasses {
    /// Build phoneme class HVs from prototype phoneme HVs
    pub fn new(phoneme_hvs: &HashMap<String, HV16>) -> Self {
        // Store phonemes (strip stress markers for class membership)
        let mut phonemes = HashMap::new();
        for (label, hv) in phoneme_hvs {
            let base = strip_stress(label);
            phonemes.entry(base).or_insert_with(|| *hv);
        }

        // Helper to bundle phonemes by name
        let bundle = |names: &[&str]| -> HV16 {
            let mut acc = BundleAccumulator::new();
            for name in names {
                if let Some(hv) = phonemes.get(*name) {
                    acc.add(hv);
                }
            }
            acc.finalize()
        };

        // Manner of articulation
        let stop = bundle(&["P", "B", "T", "D", "K", "G"]);
        let nasal = bundle(&["M", "N", "NG"]);
        let fricative = bundle(&["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]);
        let affricate = bundle(&["CH", "JH"]);
        let liquid = bundle(&["L", "R"]);
        let glide = bundle(&["W", "Y"]);
        let vowel = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW",
        ]);

        // Voicing
        let voiced = bundle(&[
            "B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W", "Y",
            // All vowels are voiced
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW",
        ]);
        let voiceless = bundle(&["P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"]);

        // Place of articulation
        let labial = bundle(&["P", "B", "M", "F", "V", "W"]);
        let dental = bundle(&["TH", "DH"]);
        let alveolar = bundle(&["T", "D", "N", "S", "Z", "L", "R"]);
        let palatal = bundle(&["SH", "ZH", "CH", "JH", "Y"]);
        let velar = bundle(&["K", "G", "NG"]);
        let glottal = bundle(&["HH"]);

        // Vowel height
        let high_vowel = bundle(&["IY", "IH", "UH", "UW"]);
        let mid_vowel = bundle(&["EY", "EH", "AH", "ER", "OW"]);
        let low_vowel = bundle(&["AE", "AA", "AO"]);

        // Vowel frontness
        let front_vowel = bundle(&["IY", "IH", "EY", "EH", "AE"]);
        let central_vowel = bundle(&["AH", "ER"]);
        let back_vowel = bundle(&["UW", "UH", "OW", "AO", "AA"]);

        // Phonotactic position constraints
        // What can start a word (most consonants, all vowels, but NOT NG or ZH)
        let word_initial = bundle(&[
            "P", "B", "T", "D", "K", "G", "M", "N", "F", "V", "TH", "DH", "S", "Z", "SH", "CH",
            "JH", "HH", "L", "R", "W", "Y", "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
            "IH", "IY", "OW", "OY", "UH", "UW",
        ]);

        // What can end a word
        let word_final = bundle(&[
            "P", "B", "T", "D", "K", "G", "M", "N", "NG", "F", "V", "TH", "DH", "S", "Z", "SH",
            "ZH", "CH", "JH", "L", "R", "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
            "IY", "OW", "OY", "UH", "UW",
        ]);

        Self {
            phonemes,
            stop,
            nasal,
            fricative,
            affricate,
            liquid,
            glide,
            vowel,
            voiced,
            voiceless,
            labial,
            dental,
            alveolar,
            palatal,
            velar,
            glottal,
            high_vowel,
            mid_vowel,
            low_vowel,
            front_vowel,
            central_vowel,
            back_vowel,
            word_initial,
            word_final,
        }
    }

    /// Get the feature vector for a phoneme (what classes it belongs to)
    pub fn get_features(&self, phoneme: &str) -> PhonemeFeatures {
        let base = strip_stress(phoneme);

        PhonemeFeatures {
            // Manner
            is_stop: self.is_member(&base, &["P", "B", "T", "D", "K", "G"]),
            is_nasal: self.is_member(&base, &["M", "N", "NG"]),
            is_fricative: self
                .is_member(&base, &["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]),
            is_affricate: self.is_member(&base, &["CH", "JH"]),
            is_liquid: self.is_member(&base, &["L", "R"]),
            is_glide: self.is_member(&base, &["W", "Y"]),
            is_vowel: self.is_vowel(&base),

            // Voicing
            is_voiced: !self.is_member(&base, &["P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"]),

            // Place
            is_labial: self.is_member(&base, &["P", "B", "M", "F", "V", "W"]),
            is_dental: self.is_member(&base, &["TH", "DH"]),
            is_alveolar: self.is_member(&base, &["T", "D", "N", "S", "Z", "L", "R"]),
            is_palatal: self.is_member(&base, &["SH", "ZH", "CH", "JH", "Y"]),
            is_velar: self.is_member(&base, &["K", "G", "NG"]),
        }
    }

    /// Check if acoustic HV is consistent with a phoneme class
    pub fn class_score(&self, acoustic_hv: &HV16, class_hv: &HV16) -> f32 {
        acoustic_hv.similarity(class_hv)
    }

    /// Score a candidate phoneme given class evidence from acoustics
    ///
    /// Returns a multiplier (0.0 to 2.0) based on class consistency
    pub fn class_consistency_score(&self, acoustic_hv: &HV16, candidate_phoneme: &str) -> f32 {
        let features = self.get_features(candidate_phoneme);
        let mut score: f32 = 1.0;

        // Check voicing consistency (most discriminative)
        let voicing_evidence =
            acoustic_hv.similarity(&self.voiced) - acoustic_hv.similarity(&self.voiceless);

        if (features.is_voiced && voicing_evidence > 0.05)
            || (!features.is_voiced && voicing_evidence < -0.05)
        {
            score += 0.3; // Voicing matches acoustic evidence
        } else if features.is_voiced && voicing_evidence < -0.1 {
            score -= 0.3; // Penalty for mismatch
        } else if !features.is_voiced && voicing_evidence > 0.1 {
            score -= 0.3;
        }

        // Check manner consistency
        if features.is_nasal {
            let nasal_score = acoustic_hv.similarity(&self.nasal);
            if nasal_score > 0.1 {
                score += 0.2;
            } else if nasal_score < -0.1 {
                score -= 0.2;
            }
        }

        if features.is_fricative {
            let fric_score = acoustic_hv.similarity(&self.fricative);
            if fric_score > 0.1 {
                score += 0.2;
            }
        }

        // Check place consistency for consonants
        if !features.is_vowel {
            if features.is_alveolar {
                let alv_score = acoustic_hv.similarity(&self.alveolar);
                if alv_score > 0.1 {
                    score += 0.15;
                }
            }
            if features.is_velar {
                let vel_score = acoustic_hv.similarity(&self.velar);
                if vel_score > 0.1 {
                    score += 0.15;
                }
            }
        }

        // Clamp to reasonable range
        score.clamp(0.5, 1.5)
    }

    /// Get candidates that match observed class features
    pub fn filter_by_class(
        &self,
        acoustic_hv: &HV16,
        candidates: &[String],
        threshold: f32,
    ) -> Vec<String> {
        candidates
            .iter()
            .filter(|c| self.class_consistency_score(acoustic_hv, c) >= threshold)
            .cloned()
            .collect()
    }

    fn is_member(&self, phoneme: &str, class: &[&str]) -> bool {
        class.contains(&phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme,
            "AA" | "AE"
                | "AH"
                | "AO"
                | "AW"
                | "AY"
                | "EH"
                | "ER"
                | "EY"
                | "IH"
                | "IY"
                | "OW"
                | "OY"
                | "UH"
                | "UW"
        )
    }
}

/// Feature vector for a phoneme
#[derive(Debug, Clone)]
pub struct PhonemeFeatures {
    // Manner
    pub is_stop: bool,
    pub is_nasal: bool,
    pub is_fricative: bool,
    pub is_affricate: bool,
    pub is_liquid: bool,
    pub is_glide: bool,
    pub is_vowel: bool,

    // Voicing
    pub is_voiced: bool,

    // Place
    pub is_labial: bool,
    pub is_dental: bool,
    pub is_alveolar: bool,
    pub is_palatal: bool,
    pub is_velar: bool,
}

/// Phonotactic bigram constraints
pub struct PhonotacticConstraints {
    /// For each phoneme, bundle of legal followers
    legal_followers: HashMap<String, HV16>,

    /// Phoneme HVs for building constraints
    phonemes: HashMap<String, HV16>,
}

impl PhonotacticConstraints {
    pub fn new(phoneme_hvs: &HashMap<String, HV16>) -> Self {
        let mut phonemes = HashMap::new();
        for (label, hv) in phoneme_hvs {
            let base = strip_stress(label);
            phonemes.entry(base).or_insert_with(|| *hv);
        }

        // Build legal follower bundles
        let legal_followers = Self::build_follower_constraints(&phonemes);

        Self {
            legal_followers,
            phonemes,
        }
    }

    fn build_follower_constraints(phonemes: &HashMap<String, HV16>) -> HashMap<String, HV16> {
        let mut constraints = HashMap::new();

        let bundle = |names: &[&str]| -> HV16 {
            let mut acc = BundleAccumulator::new();
            for name in names {
                if let Some(hv) = phonemes.get(*name) {
                    acc.add(hv);
                }
            }
            acc.finalize()
        };

        // After stops: most things (vowels, liquids, nasals common)
        let after_stop = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", "L", "R", "W", "Y", "M", "N",
        ]);
        for stop in &["P", "B", "T", "D", "K", "G"] {
            constraints.insert(stop.to_string(), after_stop);
        }

        // After nasals: usually vowels
        let after_nasal = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", "T", "D", "K", "G", "Z", "S", // Can have nasal + stop clusters
        ]);
        for nasal in &["M", "N"] {
            constraints.insert(nasal.to_string(), after_nasal);
        }

        // After NG: very restricted (usually vowel or nothing)
        let after_ng = bundle(&[
            "AA", "AE", "AH", "AO", "EH", "ER", "IH", "IY", "UH", "K", "G",
            "Z", // "inks", "ings"
        ]);
        constraints.insert("NG".to_string(), after_ng);

        // After fricatives: vowels, liquids, some stops
        let after_fricative = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", "L", "R", "W", "Y", "T", "P", "K",
        ]);
        for fric in &["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"] {
            constraints.insert(fric.to_string(), after_fricative);
        }

        // After vowels: almost anything
        let after_vowel = bundle(&[
            "P", "B", "T", "D", "K", "G", "M", "N", "NG", "F", "V", "TH", "DH", "S", "Z", "SH",
            "ZH", "CH", "JH", "HH", "L", "R", "W", "Y", "AA", "AE", "AH", "AO", "AW", "AY", "EH",
            "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        ]);
        for vowel in &[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW",
        ] {
            constraints.insert(vowel.to_string(), after_vowel);
        }

        // After liquids
        let after_liquid = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", "P", "B", "T", "D", "K", "G", "M", "N", "S", "Z", "F", "V",
        ]);
        for liquid in &["L", "R"] {
            constraints.insert(liquid.to_string(), after_liquid);
        }

        // After glides
        let after_glide = bundle(&[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW",
        ]);
        for glide in &["W", "Y"] {
            constraints.insert(glide.to_string(), after_glide);
        }

        constraints
    }

    /// Score how likely candidate is to follow prev_phoneme
    pub fn transition_score(&self, prev_phoneme: &str, candidate: &str) -> f32 {
        let prev_base = strip_stress(prev_phoneme);
        let cand_base = strip_stress(candidate);

        if let (Some(legal), Some(cand_hv)) = (
            self.legal_followers.get(&prev_base),
            self.phonemes.get(&cand_base),
        ) {
            // Similarity to legal followers bundle
            let sim = cand_hv.similarity(legal);
            // Map to 0.5-1.5 range (neutral to bonus)
            0.5 + sim.clamp(0.0, 1.0)
        } else {
            1.0 // Neutral if unknown
        }
    }
}

/// Strip stress markers from phoneme (AH0 -> AH)
fn strip_stress(phoneme: &str) -> String {
    // Also strip variant suffix (_0, _1, etc.)
    let base = if let Some(idx) = phoneme.rfind('_') {
        if phoneme[idx + 1..].chars().all(|c| c.is_ascii_digit()) {
            &phoneme[..idx]
        } else {
            phoneme
        }
    } else {
        phoneme
    };

    base.trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_stress() {
        assert_eq!(strip_stress("AH0"), "AH");
        assert_eq!(strip_stress("IY1"), "IY");
        assert_eq!(strip_stress("T"), "T");
        assert_eq!(strip_stress("AH0_0"), "AH");
        assert_eq!(strip_stress("NG_2"), "NG");
    }

    #[test]
    fn test_phoneme_features() {
        let phonemes = HashMap::new(); // Empty for feature test
        let classes = PhonemeClasses::new(&phonemes);

        let t_features = classes.get_features("T");
        assert!(t_features.is_stop);
        assert!(!t_features.is_voiced);
        assert!(t_features.is_alveolar);

        let d_features = classes.get_features("D");
        assert!(d_features.is_stop);
        assert!(d_features.is_voiced);
        assert!(d_features.is_alveolar);

        let n_features = classes.get_features("N");
        assert!(n_features.is_nasal);
        assert!(n_features.is_voiced);
        assert!(n_features.is_alveolar);

        let ng_features = classes.get_features("NG");
        assert!(ng_features.is_nasal);
        assert!(ng_features.is_voiced);
        assert!(ng_features.is_velar);
    }
}
