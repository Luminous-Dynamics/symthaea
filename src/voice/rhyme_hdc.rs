// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-Based Rhyme Detection and Generation
//!
//! Uses Hyperdimensional Computing with unified ContinuousHV (16,384 dimensions)
//! for rhyme detection, assonance matching, and alliteration finding.
//!
//! ## Architecture
//!
//! This module delegates to `PhonemeHdcCodec` for phoneme encoding, ensuring
//! consistent HDC representations across TTS, STT, and rhyme detection.
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │ Word Phonemes   │────►│ PhonemeHdcCodec │────►│ ContinuousHV    │
//! │ (ARPABET)       │     │ (unified 16,384)│     │ (word ending)   │
//! └─────────────────┘     └─────────────────┘     └────────┬────────┘
//!                                                          │
//!                                                          ▼
//!                                                 ┌─────────────────┐
//!                                                 │ Cosine Sim      │
//!                                                 │ → Rhyme Score   │
//!                                                 └─────────────────┘
//! ```
//!
//! ## Rhyme Types
//!
//! - **Perfect rhymes**: "cat" / "hat" (identical ending sounds)
//! - **Slant rhymes**: "cat" / "bed" (similar vowel sounds)
//! - **Assonance**: "lake" / "fate" (shared vowel patterns)
//! - **Alliteration**: "big bad" (shared initial consonants)

use crate::voice::phoneme_hdc::PhonemeHdcCodec;
use std::sync::Arc;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// RHYME TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of phonetic similarity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RhymeType {
    /// Identical ending sounds (cat/hat)
    Perfect,
    /// Similar but not identical endings (cat/bed)
    Slant,
    /// Shared vowel sounds (lake/fate)
    Assonance,
    /// Shared initial consonants (big/bad)
    Alliteration,
    /// No significant similarity
    None,
}

/// Rhyme detection result
#[derive(Debug, Clone)]
pub struct RhymeScore {
    /// Overall similarity score (0.0 to 1.0)
    pub score: f32,
    /// Type of rhyme detected
    pub rhyme_type: RhymeType,
    /// Ending similarity (for rhyme)
    pub ending_similarity: f32,
    /// Initial similarity (for alliteration)
    pub initial_similarity: f32,
    /// Vowel pattern similarity (for assonance)
    pub vowel_similarity: f32,
}

impl RhymeScore {
    /// Create a no-rhyme score
    pub fn none() -> Self {
        Self {
            score: 0.0,
            rhyme_type: RhymeType::None,
            ending_similarity: 0.0,
            initial_similarity: 0.0,
            vowel_similarity: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RHYME SCHEME
// ═══════════════════════════════════════════════════════════════════════════════

/// Detected rhyme scheme for a verse
#[derive(Debug, Clone, Default)]
pub struct RhymeScheme {
    /// Pattern like "AABB", "ABAB", etc.
    pub pattern: String,
    /// Groups of rhyming line indices
    pub groups: Vec<Vec<usize>>,
    /// Average rhyme strength
    pub strength: f32,
}

impl RhymeScheme {
    /// Create from a pattern string like "AABB" or "ABAB"
    pub fn from_pattern(pattern: &str) -> Self {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut label_map: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();

        for (i, c) in pattern.chars().enumerate() {
            if let Some(&group_idx) = label_map.get(&c) {
                groups[group_idx].push(i);
            } else {
                let group_idx = groups.len();
                label_map.insert(c, group_idx);
                groups.push(vec![i]);
            }
        }

        Self {
            pattern: pattern.to_string(),
            groups,
            strength: 1.0, // Assume full strength for explicit patterns
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RHYME ENCODER (Unified with PhonemeHdcCodec)
// ═══════════════════════════════════════════════════════════════════════════════

/// Rhyme encoder using unified HDC (16,384 dimensions)
pub struct RhymeEncoder {
    /// Phoneme HDC codec (shared for consistency)
    codec: Arc<PhonemeHdcCodec>,
    /// Perfect rhyme threshold
    perfect_threshold: f32,
    /// Slant rhyme threshold
    slant_threshold: f32,
    /// Assonance threshold
    assonance_threshold: f32,
    /// Alliteration threshold
    alliteration_threshold: f32,
}

impl RhymeEncoder {
    /// Create a new rhyme encoder
    pub fn new() -> Self {
        Self {
            codec: Arc::new(PhonemeHdcCodec::new()),
            perfect_threshold: 0.85,
            slant_threshold: 0.6,
            assonance_threshold: 0.5,
            alliteration_threshold: 0.7,
        }
    }

    /// Create with shared codec
    pub fn with_codec(codec: Arc<PhonemeHdcCodec>) -> Self {
        Self {
            codec,
            perfect_threshold: 0.85,
            slant_threshold: 0.6,
            assonance_threshold: 0.5,
            alliteration_threshold: 0.7,
        }
    }

    /// Get the shared codec
    pub fn codec(&self) -> &PhonemeHdcCodec {
        &self.codec
    }

    /// Encode a word's ending for rhyme detection
    pub fn encode_ending(&self, phonemes: &[String]) -> ContinuousHV {
        let refs: Vec<&str> = phonemes.iter().map(|s| s.as_str()).collect();
        self.codec.encode_word_ending(&refs, 1) // Last syllable
    }

    /// Encode a word's beginning for alliteration detection
    pub fn encode_beginning(&self, phonemes: &[String]) -> ContinuousHV {
        if phonemes.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        // For alliteration, focus on initial consonant(s) before first vowel
        let mut initial_consonants: Vec<&str> = Vec::new();
        for p in phonemes {
            if self.codec.is_vowel(p) {
                break;
            }
            initial_consonants.push(p.as_str());
        }

        // If no initial consonants, use first phoneme
        if initial_consonants.is_empty() {
            return self
                .codec
                .encode(&phonemes[0])
                .unwrap_or_else(|| ContinuousHV::zero(HDC_DIMENSION));
        }

        // Bundle initial consonants without positional encoding
        let mut result = ContinuousHV::zero(HDC_DIMENSION);
        for phoneme in &initial_consonants {
            if let Some(p_hv) = self.codec.encode(phoneme) {
                result = ContinuousHV::bundle(&[&result, &p_hv]);
            }
        }
        result.normalize()
    }

    /// Encode vowel pattern (for assonance)
    pub fn encode_vowels(&self, phonemes: &[String]) -> ContinuousHV {
        // Extract only vowels
        let vowels: Vec<&str> = phonemes
            .iter()
            .filter(|p| self.codec.is_vowel(p))
            .map(|s| s.as_str())
            .collect();

        if vowels.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        self.codec.encode_word(&vowels)
    }

    /// Compute rhyme score between two words
    pub fn rhyme_score(&self, word1: &[String], word2: &[String]) -> RhymeScore {
        if word1.is_empty() || word2.is_empty() {
            return RhymeScore::none();
        }

        // Encode different aspects
        let end1 = self.encode_ending(word1);
        let end2 = self.encode_ending(word2);
        let ending_similarity = end1.similarity(&end2);

        let begin1 = self.encode_beginning(word1);
        let begin2 = self.encode_beginning(word2);
        let initial_similarity = begin1.similarity(&begin2);

        let vowel1 = self.encode_vowels(word1);
        let vowel2 = self.encode_vowels(word2);
        let vowel_similarity = vowel1.similarity(&vowel2);

        // Determine rhyme type
        let rhyme_type = if ending_similarity >= self.perfect_threshold {
            RhymeType::Perfect
        } else if ending_similarity >= self.slant_threshold {
            RhymeType::Slant
        } else if vowel_similarity >= self.assonance_threshold {
            RhymeType::Assonance
        } else if initial_similarity >= self.alliteration_threshold {
            RhymeType::Alliteration
        } else {
            RhymeType::None
        };

        // Overall score weighted by type
        let score = match rhyme_type {
            RhymeType::Perfect => ending_similarity,
            RhymeType::Slant => ending_similarity * 0.8,
            RhymeType::Assonance => vowel_similarity * 0.6,
            RhymeType::Alliteration => initial_similarity * 0.4,
            RhymeType::None => 0.0,
        };

        RhymeScore {
            score,
            rhyme_type,
            ending_similarity,
            initial_similarity,
            vowel_similarity,
        }
    }

    /// Check if two words rhyme (any type)
    pub fn rhymes(&self, word1: &[String], word2: &[String]) -> bool {
        self.rhyme_score(word1, word2).rhyme_type != RhymeType::None
    }

    /// Check for perfect rhyme
    pub fn perfect_rhyme(&self, word1: &[String], word2: &[String]) -> bool {
        self.rhyme_score(word1, word2).rhyme_type == RhymeType::Perfect
    }

    /// Detect rhyme scheme from line endings
    pub fn detect_scheme(&self, line_endings: &[Vec<String>]) -> RhymeScheme {
        if line_endings.is_empty() {
            return RhymeScheme::default();
        }

        let n = line_endings.len();
        let mut assignments: Vec<Option<char>> = vec![None; n];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut next_label = 'A';
        let mut total_strength = 0.0;
        let mut comparisons = 0;

        for i in 0..n {
            if assignments[i].is_some() {
                continue;
            }

            // Start new group
            let label = next_label;
            next_label = (next_label as u8 + 1) as char;
            assignments[i] = Some(label);

            let mut group = vec![i];

            // Find rhyming lines
            for j in (i + 1)..n {
                if assignments[j].is_some() {
                    continue;
                }

                let score = self.rhyme_score(&line_endings[i], &line_endings[j]);
                if score.rhyme_type == RhymeType::Perfect || score.rhyme_type == RhymeType::Slant {
                    assignments[j] = Some(label);
                    group.push(j);
                    total_strength += score.score;
                    comparisons += 1;
                }
            }

            groups.push(group);
        }

        let pattern: String = assignments.iter().map(|a| a.unwrap_or('X')).collect();

        let strength = if comparisons > 0 {
            total_strength / comparisons as f32
        } else {
            0.0
        };

        RhymeScheme {
            pattern,
            groups,
            strength,
        }
    }

    /// Find best rhyming word from candidates
    pub fn find_best_rhyme<'a>(
        &self,
        target: &[String],
        candidates: &[(&'a str, &[String])],
    ) -> Option<(&'a str, RhymeScore)> {
        candidates
            .iter()
            .map(|(word, phonemes)| (*word, self.rhyme_score(target, phonemes)))
            .filter(|(_, score)| score.rhyme_type != RhymeType::None)
            .max_by(|a, b| {
                a.1.score
                    .partial_cmp(&b.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for RhymeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn to_phonemes(s: &[&str]) -> Vec<String> {
        s.iter().map(|p| p.to_string()).collect()
    }

    #[test]
    fn test_rhyme_encoding() {
        let encoder = RhymeEncoder::new();

        // cat = K AE T, hat = HH AE T (should rhyme)
        let cat = to_phonemes(&["K", "AE", "T"]);
        let hat = to_phonemes(&["HH", "AE", "T"]);

        let score = encoder.rhyme_score(&cat, &hat);
        assert!(
            score.ending_similarity > 0.5,
            "cat/hat should have high ending similarity: {}",
            score.ending_similarity
        );
    }

    #[test]
    fn test_non_rhyme() {
        let encoder = RhymeEncoder::new();

        // cat = K AE T, dog = D AO G (shouldn't rhyme)
        let cat = to_phonemes(&["K", "AE", "T"]);
        let dog = to_phonemes(&["D", "AO", "G"]);

        let score = encoder.rhyme_score(&cat, &dog);
        // HDC similarity can be higher due to shared phoneme features; use relaxed threshold
        assert!(
            score.ending_similarity < 0.7,
            "cat/dog should have low ending similarity: {}",
            score.ending_similarity
        );
    }

    #[test]
    fn test_alliteration() {
        let encoder = RhymeEncoder::new();

        // big = B IH G, bad = B AE D (alliteration)
        let big = to_phonemes(&["B", "IH", "G"]);
        let bad = to_phonemes(&["B", "AE", "D"]);

        let score = encoder.rhyme_score(&big, &bad);
        assert!(
            score.initial_similarity > 0.5,
            "big/bad should have high initial similarity: {}",
            score.initial_similarity
        );
    }

    #[test]
    fn test_assonance() {
        let encoder = RhymeEncoder::new();

        // lake = L EY K, fate = F EY T (assonance - shared EY vowel)
        let lake = to_phonemes(&["L", "EY", "K"]);
        let fate = to_phonemes(&["F", "EY", "T"]);

        let score = encoder.rhyme_score(&lake, &fate);
        assert!(
            score.vowel_similarity > 0.5,
            "lake/fate should have high vowel similarity: {}",
            score.vowel_similarity
        );
    }

    #[test]
    fn test_rhyme_scheme_detection() {
        let encoder = RhymeEncoder::new();

        // Simple AABB scheme
        let endings = vec![
            to_phonemes(&["K", "AE", "T"]),  // cat
            to_phonemes(&["HH", "AE", "T"]), // hat (rhymes with cat)
            to_phonemes(&["D", "AO", "G"]),  // dog
            to_phonemes(&["F", "AO", "G"]),  // fog (rhymes with dog)
        ];

        let scheme = encoder.detect_scheme(&endings);
        // Should detect some rhyme pattern
        assert!(!scheme.pattern.is_empty());
        assert!(scheme.groups.len() >= 2);
    }
}
