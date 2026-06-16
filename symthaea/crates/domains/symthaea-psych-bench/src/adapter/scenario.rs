// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Text scenario encoding via deterministic hash.
//!
//! Converts text strings into HDC vectors using a simple hash-based encoder.
//! Each unique string maps to a unique (deterministic) hypervector.

use super::StimulusAdapter;
use symthaea_core::hdc::ContinuousHV;

/// A text scenario (sentence, description, or label).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scenario(pub String);

impl Scenario {
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }
}

impl From<&str> for Scenario {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Encodes text scenarios into HDC vectors via deterministic hashing.
///
/// Uses a word-level encoding: each word gets a basis vector from its hash,
/// permuted by position, then all are bundled. This preserves word identity
/// and rough order without requiring an external tokenizer.
pub struct ScenarioAdapter;

impl ScenarioAdapter {
    /// Hash a word into a seed.
    fn word_seed(word: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for b in word.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
        // Avoid seed-0 fixed point in xorshift
        h ^ 0x9E3779B97F4A7C15
    }
}

impl StimulusAdapter for ScenarioAdapter {
    type Stimulus = Scenario;

    fn encode(&self, scenario: &Scenario, dim: usize) -> ContinuousHV {
        let words: Vec<&str> = scenario.0.split_whitespace().collect();
        if words.is_empty() {
            return ContinuousHV::zero(dim);
        }
        let positioned: Vec<ContinuousHV> = words
            .iter()
            .enumerate()
            .map(|(pos, word)| {
                let seed = Self::word_seed(word);
                ContinuousHV::random(dim, seed).permute(pos)
            })
            .collect();
        ContinuousHV::bundle_owned(&positioned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let adapter = ScenarioAdapter;
        let s = Scenario::new("the cat sat on the mat");
        let a = adapter.encode(&s, 512);
        let b = adapter.encode(&s, 512);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_sentences_dissimilar() {
        let adapter = ScenarioAdapter;
        let a = adapter.encode(&Scenario::new("the dog ran fast"), 512);
        let b = adapter.encode(&Scenario::new("quantum physics is complex"), 512);
        assert!(a.similarity(&b).abs() < 0.3);
    }

    #[test]
    fn test_similar_sentences_partially_similar() {
        let adapter = ScenarioAdapter;
        let a = adapter.encode(&Scenario::new("the cat sat on the mat"), 512);
        let b = adapter.encode(&Scenario::new("the cat sat on the rug"), 512);
        // Should have some overlap (shared words) but not perfect
        let sim = a.similarity(&b);
        assert!(sim > 0.3 && sim < 0.99);
    }

    #[test]
    fn test_empty_scenario() {
        let adapter = ScenarioAdapter;
        let result = adapter.encode(&Scenario::new(""), 512);
        assert_eq!(result.values.len(), 512);
    }
}
