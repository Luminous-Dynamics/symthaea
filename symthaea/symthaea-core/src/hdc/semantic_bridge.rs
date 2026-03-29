// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bidirectional Semantic Bridge: Text <-> HV <-> Consciousness
//!
//! This module provides bidirectional mapping between natural language text and
//! hyperdimensional vectors (BinaryHV), with optional consciousness grounding
//! via Phi (Φ) weights.
//!
//! # Architecture
//!
//! ```text
//! Text ("red ball")
//!   │
//!   ▼ encode_phrase()
//! BinaryHV (16,384-bit holographic vector)
//!   │
//!   ▼ decode_hv() / decode_phrase_hv()
//! Text (nearest neighbor reconstruction)
//!   │
//!   ▼ ground_in_consciousness()
//! Φ-weighted semantic space (consciousness-grounded meaning)
//! ```
//!
//! # Key Properties
//!
//! - **Deterministic**: Same word always maps to the same HV via `seed_from_name`
//! - **Compositional**: Phrases are encoded by binding word HVs (order-sensitive)
//! - **Reversible**: decode_hv recovers nearest words from an HV
//! - **Consciousness-grounded**: Φ weights modulate semantic importance

use super::binary_hv::BinaryHV;
use super::deterministic_seeds::seed_from_name;
use std::collections::HashMap;

/// Bidirectional semantic bridge between text and hypervectors.
///
/// Maps words to deterministic BinaryHVs and back, with optional
/// consciousness Φ weighting for grounded semantics.
pub struct SemanticBridge {
    /// word -> HV mapping
    vocabulary: HashMap<String, BinaryHV>,
    /// for nearest-neighbor lookup (linear scan)
    reverse_index: Vec<(String, BinaryHV)>,
    /// consciousness-weighted terms (word -> Φ value)
    phi_weights: HashMap<String, f64>,
}

impl SemanticBridge {
    /// Create an empty semantic bridge with no vocabulary.
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            reverse_index: Vec::new(),
            phi_weights: HashMap::new(),
        }
    }

    /// Create a semantic bridge initialized with a vocabulary.
    ///
    /// Each word gets a deterministic BinaryHV generated from `seed_from_name`.
    ///
    /// # Example
    /// ```ignore
    /// let bridge = SemanticBridge::with_vocabulary(&["cat", "dog", "red", "ball"]);
    /// assert_eq!(bridge.vocabulary_size(), 4);
    /// ```
    pub fn with_vocabulary(words: &[&str]) -> Self {
        let mut bridge = Self::new();
        for &word in words {
            bridge.add_word(word);
        }
        bridge
    }

    /// Add a single word to the vocabulary using a deterministic seed.
    ///
    /// If the word already exists, this is a no-op.
    pub fn add_word(&mut self, word: &str) {
        let normalized = word.to_lowercase();
        if self.vocabulary.contains_key(&normalized) {
            return;
        }
        let seed = seed_from_name(&normalized);
        let hv = BinaryHV::random(seed);
        self.vocabulary.insert(normalized.clone(), hv);
        self.reverse_index.push((normalized, hv));
    }

    /// Look up the HV for a known word.
    ///
    /// Returns `None` if the word is not in the vocabulary.
    pub fn encode_word(&self, word: &str) -> Option<BinaryHV> {
        let normalized = word.to_lowercase();
        self.vocabulary.get(&normalized).copied()
    }

    /// Encode a multi-word phrase by binding word HVs sequentially.
    ///
    /// Words are bound left-to-right with positional permutation to preserve
    /// word order: `permute(w0, 0) XOR permute(w1, 1) XOR permute(w2, 2) ...`
    ///
    /// Unknown words are skipped. If no known words are found, returns `BinaryHV::zero()`.
    pub fn encode_phrase(&self, phrase: &str) -> BinaryHV {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let mut hvs: Vec<BinaryHV> = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if let Some(hv) = self.encode_word(word) {
                // Permute by position to encode word order
                hvs.push(hv.permute(i));
            }
        }

        if hvs.is_empty() {
            return BinaryHV::zero();
        }

        if hvs.len() == 1 {
            return hvs[0];
        }

        // Bundle the positionally-permuted word HVs
        BinaryHV::bundle(&hvs)
    }

    /// Find the nearest words to a given HV by similarity.
    ///
    /// Returns up to `top_k` results as `(word, similarity)` pairs,
    /// sorted by descending similarity.
    pub fn decode_hv(&self, hv: &BinaryHV, top_k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .reverse_index
            .iter()
            .map(|(word, word_hv)| (word.clone(), hv.similarity(word_hv)))
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Attempt to reconstruct a phrase from an HV.
    ///
    /// Greedily selects the most similar words and attempts positional
    /// unbinding. Returns a space-separated string of recovered words.
    pub fn decode_phrase_hv(&self, hv: &BinaryHV, max_words: usize) -> String {
        // Strategy: try positional unbinding for each position and find
        // the best matching word at that position.
        let mut recovered_words: Vec<(usize, String, f32)> = Vec::new();

        for pos in 0..max_words {
            // Create the positional permutation inverse:
            // If encoding used permute(word, pos), we undo with permute(hv, DIM - pos)
            // But for bundled HVs, we can't perfectly unbind. Instead, we check
            // similarity of each vocabulary word's permuted form against the HV.
            let mut best_word = String::new();
            let mut best_sim: f32 = 0.0;

            for (word, word_hv) in &self.reverse_index {
                let permuted = word_hv.permute(pos);
                let sim = hv.similarity(&permuted);
                if sim > best_sim {
                    best_sim = sim;
                    best_word = word.clone();
                }
            }

            // Only include words above a threshold (random similarity ~0.5)
            if best_sim > 0.52 {
                recovered_words.push((pos, best_word, best_sim));
            }
        }

        // Sort by position
        recovered_words.sort_by_key(|(pos, _, _)| *pos);

        // Deduplicate consecutive identical words
        let mut result: Vec<String> = Vec::new();
        for (_, word, _) in &recovered_words {
            if result.last() != Some(word) {
                result.push(word.clone());
            }
        }

        if result.is_empty() {
            // Fallback: just return the single nearest word
            let nearest = self.decode_hv(hv, 1);
            if let Some((word, _)) = nearest.first() {
                return word.clone();
            }
            return String::new();
        }

        result.join(" ")
    }

    /// Associate a consciousness Phi weight with a word.
    ///
    /// Higher Phi values indicate the word is more "conscious" or salient.
    pub fn ground_in_consciousness(&mut self, word: &str, phi: f64) {
        let normalized = word.to_lowercase();
        self.phi_weights.insert(normalized, phi);
    }

    /// Encode a phrase with consciousness Phi weighting.
    ///
    /// Words with higher Phi weights are repeated more times in the bundle,
    /// giving them greater influence on the resulting HV.
    ///
    /// Words without a Phi weight default to 1.0.
    pub fn consciousness_weighted_encode(&self, phrase: &str) -> BinaryHV {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let mut hvs: Vec<BinaryHV> = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if let Some(hv) = self.encode_word(word) {
                let normalized = word.to_lowercase();
                let phi = self.phi_weights.get(&normalized).copied().unwrap_or(1.0);

                // Repeat the HV proportional to Phi weight (minimum 1 copy)
                let copies = (phi.ceil() as usize).max(1);
                let permuted = hv.permute(i);
                for _ in 0..copies {
                    hvs.push(permuted);
                }
            }
        }

        if hvs.is_empty() {
            return BinaryHV::zero();
        }

        if hvs.len() == 1 {
            return hvs[0];
        }

        BinaryHV::bundle(&hvs)
    }

    /// Compute semantic similarity between two words via their HVs.
    ///
    /// Returns `None` if either word is not in the vocabulary.
    pub fn semantic_similarity(&self, a: &str, b: &str) -> Option<f32> {
        let hv_a = self.encode_word(a)?;
        let hv_b = self.encode_word(b)?;
        Some(hv_a.similarity(&hv_b))
    }

    /// Return the number of words in the vocabulary.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl Default for SemanticBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = SemanticBridge::new();
        assert_eq!(bridge.vocabulary_size(), 0);
        assert!(bridge.reverse_index.is_empty());
        assert!(bridge.phi_weights.is_empty());
    }

    #[test]
    fn test_with_vocabulary() {
        let bridge = SemanticBridge::with_vocabulary(&["cat", "dog", "red", "ball", "sky"]);
        assert_eq!(bridge.vocabulary_size(), 5);
        // Each word should have an entry in both vocabulary and reverse_index
        assert_eq!(bridge.reverse_index.len(), 5);
    }

    #[test]
    fn test_encode_word() {
        let bridge = SemanticBridge::with_vocabulary(&["hello", "world"]);
        let hv = bridge.encode_word("hello");
        assert!(hv.is_some());

        // Deterministic: same word always produces same HV
        let hv2 = bridge.encode_word("hello");
        assert_eq!(hv.unwrap(), hv2.unwrap());

        // Case insensitive
        let hv3 = bridge.encode_word("Hello");
        assert_eq!(hv.unwrap(), hv3.unwrap());
    }

    #[test]
    fn test_encode_unknown_word() {
        let bridge = SemanticBridge::with_vocabulary(&["cat"]);
        let result = bridge.encode_word("elephant");
        assert!(result.is_none());
    }

    #[test]
    fn test_encode_phrase() {
        let bridge = SemanticBridge::with_vocabulary(&["red", "ball", "blue", "sky"]);

        let phrase1 = bridge.encode_phrase("red ball");
        let phrase2 = bridge.encode_phrase("red ball");

        // Deterministic
        assert_eq!(phrase1, phrase2);

        // Different phrase should produce different HV
        let phrase3 = bridge.encode_phrase("blue sky");
        assert_ne!(phrase1, phrase3);

        // Order matters (due to positional permutation)
        let phrase4 = bridge.encode_phrase("ball red");
        assert_ne!(phrase1, phrase4);
    }

    #[test]
    fn test_decode_hv_finds_original() {
        let bridge = SemanticBridge::with_vocabulary(&["cat", "dog", "fish", "bird"]);
        let cat_hv = bridge.encode_word("cat").unwrap();

        let results = bridge.decode_hv(&cat_hv, 4);

        // The top result should be "cat" with similarity 1.0
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "cat");
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_decode_phrase() {
        let words = &["the", "cat", "sat", "on", "mat"];
        let bridge = SemanticBridge::with_vocabulary(words);

        let hv = bridge.encode_phrase("cat sat");
        let decoded = bridge.decode_phrase_hv(&hv, 5);

        // The decoded phrase should contain at least one of the original words
        let decoded_lower = decoded.to_lowercase();
        let has_cat = decoded_lower.contains("cat");
        let has_sat = decoded_lower.contains("sat");
        assert!(
            has_cat || has_sat,
            "Decoded '{}' should contain 'cat' or 'sat'",
            decoded
        );
    }

    #[test]
    fn test_ground_in_consciousness() {
        let mut bridge = SemanticBridge::with_vocabulary(&["awareness", "table", "chair"]);
        bridge.ground_in_consciousness("awareness", 5.0);
        bridge.ground_in_consciousness("table", 1.0);

        assert_eq!(bridge.phi_weights.get("awareness"), Some(&5.0));
        assert_eq!(bridge.phi_weights.get("table"), Some(&1.0));
        assert_eq!(bridge.phi_weights.get("chair"), None);
    }

    #[test]
    fn test_consciousness_weighted_encode() {
        let mut bridge = SemanticBridge::with_vocabulary(&["light", "dark", "consciousness"]);
        bridge.ground_in_consciousness("consciousness", 5.0);
        bridge.ground_in_consciousness("light", 1.0);

        let weighted = bridge.consciousness_weighted_encode("light consciousness");
        let unweighted = bridge.encode_phrase("light consciousness");

        // Weighted and unweighted should differ because consciousness gets 5 copies
        assert_ne!(weighted, unweighted);

        // The weighted encoding should be more similar to "consciousness" alone
        let consciousness_hv = bridge.encode_word("consciousness").unwrap();
        let sim_weighted = weighted.similarity(&consciousness_hv.permute(1));
        let sim_unweighted = unweighted.similarity(&consciousness_hv.permute(1));

        assert!(
            sim_weighted > sim_unweighted,
            "Weighted sim ({}) should exceed unweighted sim ({})",
            sim_weighted,
            sim_unweighted
        );
    }

    #[test]
    fn test_add_word() {
        let mut bridge = SemanticBridge::new();
        assert_eq!(bridge.vocabulary_size(), 0);

        bridge.add_word("quantum");
        assert_eq!(bridge.vocabulary_size(), 1);
        assert!(bridge.encode_word("quantum").is_some());

        // Adding same word again should be a no-op
        bridge.add_word("quantum");
        assert_eq!(bridge.vocabulary_size(), 1);

        // Case normalization: adding "Quantum" should also be a no-op
        bridge.add_word("Quantum");
        assert_eq!(bridge.vocabulary_size(), 1);
    }

    #[test]
    fn test_semantic_similarity() {
        let bridge = SemanticBridge::with_vocabulary(&["alpha", "beta", "gamma"]);

        // Self-similarity should be 1.0
        let self_sim = bridge.semantic_similarity("alpha", "alpha");
        assert!(self_sim.is_some());
        assert!((self_sim.unwrap() - 1.0).abs() < 0.001);

        // Cross-similarity between random HVs should be near 0.5
        let cross_sim = bridge.semantic_similarity("alpha", "beta").unwrap();
        assert!(
            cross_sim > 0.45 && cross_sim < 0.55,
            "Random HV similarity should be near 0.5, got {}",
            cross_sim
        );

        // Unknown word returns None
        let missing = bridge.semantic_similarity("alpha", "omega");
        assert!(missing.is_none());
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        let words = &["apple", "banana", "cherry", "date", "elderberry"];
        let bridge = SemanticBridge::with_vocabulary(words);

        for &word in words {
            let hv = bridge.encode_word(word).unwrap();
            let decoded = bridge.decode_hv(&hv, 1);

            assert!(!decoded.is_empty());
            assert_eq!(
                decoded[0].0, word,
                "Roundtrip failed: encoded '{}', decoded '{}'",
                word, decoded[0].0
            );
            assert!(
                (decoded[0].1 - 1.0).abs() < 0.001,
                "Expected similarity 1.0 for exact match, got {}",
                decoded[0].1
            );
        }
    }

    #[test]
    fn test_encode_phrase_with_unknown_words() {
        let bridge = SemanticBridge::with_vocabulary(&["known"]);
        let hv = bridge.encode_phrase("unknown words only");
        // All unknown words -> should return zero HV
        assert_eq!(hv, BinaryHV::zero());

        // Mixed known/unknown
        let mixed = bridge.encode_phrase("a known word");
        assert_ne!(mixed, BinaryHV::zero());
    }

    #[test]
    fn test_default_trait() {
        let bridge = SemanticBridge::default();
        assert_eq!(bridge.vocabulary_size(), 0);
    }
}
