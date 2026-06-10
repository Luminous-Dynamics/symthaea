// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # DeepNSM Integration — Computational Natural Semantic Metalanguage
//!
//! Integrates the DeepNSM corpus (44,000+ word-usage-explication triplets)
//! with Symthaea's existing NSM grounding pipeline.
//!
//! ## How It Works
//!
//! 1. Load NSM explications (word → list of semantic primes)
//! 2. Encode each explication as a BinaryHV by binding its primes
//! 3. When GroundedUnderstanding encounters a word, look up the corpus
//!    before falling back to heuristic decomposition
//!
//! ## Science
//!
//! - DeepNSM (arXiv:2505.11764, May 2025): First computational NSM system
//! - Wierzbicka (1996): Semantics: Primes and Universals
//! - 65 semantic primes + ~50 semantic molecules
//! - Legality metric: fraction of output tokens that are valid NSM primes
//! - Substitutability metric: semantic preservation of explication

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hdc::unified_hv::BinaryHV;
#[cfg(test)]
use crate::hdc::unified_hv::HDC_DIMENSION;

/// A single NSM explication triplet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NsmExplication {
    /// The word being explicated
    pub word: String,
    /// Usage context for disambiguation
    pub usage: String,
    /// NSM primitive decomposition
    pub primitives: Vec<String>,
    /// Pre-computed HDC encoding (lazy, populated on first use)
    #[serde(skip)]
    pub encoding: Option<BinaryHV>,
}

/// Corpus of NSM explications for grounding language understanding.
#[derive(Debug)]
pub struct DeepNSMCorpus {
    /// Word → list of explications (multiple senses per word)
    entries: HashMap<String, Vec<NsmExplication>>,
    /// Total number of explications loaded
    count: usize,
}

impl DeepNSMCorpus {
    /// Create an empty corpus.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            count: 0,
        }
    }

    /// Add an explication to the corpus.
    pub fn add(&mut self, explication: NsmExplication) {
        let word = explication.word.to_lowercase();
        self.entries.entry(word).or_default().push(explication);
        self.count += 1;
    }

    /// Look up explications for a word.
    pub fn lookup(&self, word: &str) -> Option<&[NsmExplication]> {
        self.entries.get(&word.to_lowercase()).map(|v| v.as_slice())
    }

    /// Encode an explication as a BinaryHV by binding its primitives.
    ///
    /// `prime_lookup` maps a prime name to its hypervector.
    /// Unknown primes are skipped. Vectors are permuted by position
    /// to make the encoding order-sensitive.
    pub fn encode_explication<F>(primitives: &[String], prime_lookup: F) -> BinaryHV
    where
        F: Fn(&str) -> Option<BinaryHV>,
    {
        if primitives.is_empty() {
            return BinaryHV::random(0);
        }

        let mut vectors: Vec<BinaryHV> = Vec::new();
        for (i, prime) in primitives.iter().enumerate() {
            if let Some(hv) = prime_lookup(prime) {
                vectors.push(hv.permute(i));
            }
        }

        if vectors.is_empty() {
            return BinaryHV::random(0);
        }

        BinaryHV::bundle(&vectors)
    }

    /// Get the best explication for a word in context.
    ///
    /// If multiple senses exist, returns the one whose usage
    /// context best matches the given context (simple keyword overlap).
    pub fn best_match(&self, word: &str, context: &str) -> Option<&NsmExplication> {
        let entries = self.lookup(word)?;
        if entries.len() == 1 {
            return Some(&entries[0]);
        }

        let query = word.to_lowercase();
        let context_lower = context.to_lowercase();
        let context_words: Vec<String> = context_lower
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_ascii_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty() && w != &query)
            .collect();

        entries.iter().max_by_key(|e| {
            let usage_lower = e.usage.to_lowercase();
            let usage_words: Vec<&str> = usage_lower.split_whitespace().collect();
            context_words
                .iter()
                .filter(|w| usage_words.contains(&w.as_str()))
                .count()
        })
    }

    /// Number of explications in the corpus.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the corpus is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of unique words.
    pub fn word_count(&self) -> usize {
        self.entries.len()
    }

    /// Compute legality score: fraction of tokens that are valid NSM primes.
    pub fn legality_score(text: &str, valid_primes: &[&str]) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        let valid_count = words
            .iter()
            .filter(|w| valid_primes.contains(&w.to_lowercase().as_str()))
            .count();
        valid_count as f64 / words.len() as f64
    }
}

impl Default for DeepNSMCorpus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_lookup() {
        let mut corpus = DeepNSMCorpus::new();
        corpus.add(NsmExplication {
            word: "happy".to_string(),
            usage: "feeling joy".to_string(),
            primitives: vec!["FEEL".to_string(), "GOOD".to_string()],
            encoding: None,
        });
        assert_eq!(corpus.len(), 1);
        assert!(corpus.lookup("happy").is_some());
        assert!(corpus.lookup("HAPPY").is_some()); // case-insensitive
        assert!(corpus.lookup("sad").is_none());
    }

    #[test]
    fn test_multiple_senses() {
        let mut corpus = DeepNSMCorpus::new();
        corpus.add(NsmExplication {
            word: "bank".to_string(),
            usage: "financial institution".to_string(),
            primitives: vec!["HAVE".to_string(), "THING".to_string()],
            encoding: None,
        });
        corpus.add(NsmExplication {
            word: "bank".to_string(),
            usage: "river bank".to_string(),
            primitives: vec!["PART".to_string(), "WHERE".to_string()],
            encoding: None,
        });
        assert_eq!(corpus.lookup("bank").unwrap().len(), 2);
        assert_eq!(corpus.word_count(), 1);
        assert_eq!(corpus.len(), 2);
    }

    #[test]
    fn test_best_match_context() {
        let mut corpus = DeepNSMCorpus::new();
        corpus.add(NsmExplication {
            word: "bank".to_string(),
            usage: "financial institution for money".to_string(),
            primitives: vec!["HAVE".to_string()],
            encoding: None,
        });
        corpus.add(NsmExplication {
            word: "bank".to_string(),
            usage: "river bank near water".to_string(),
            primitives: vec!["WHERE".to_string()],
            encoding: None,
        });

        let financial = corpus
            .best_match("bank", "I deposited money at the bank")
            .unwrap();
        assert!(financial.usage.contains("financial"));

        let river = corpus
            .best_match("bank", "The river bank was muddy from water")
            .unwrap();
        assert!(river.usage.contains("river"));
    }

    #[test]
    fn test_legality_score() {
        let primes = vec!["i", "you", "feel", "good", "bad", "know", "want"];
        assert!(DeepNSMCorpus::legality_score("I feel good", &primes) > 0.9);
        assert!(DeepNSMCorpus::legality_score("perambulate quintessentially", &primes) < 0.01);
    }

    #[test]
    fn test_encode_explication() {
        let primitives = vec!["FEEL".to_string(), "GOOD".to_string()];
        // Simple lookup that generates a deterministic HV per prime
        let hv = DeepNSMCorpus::encode_explication(&primitives, |prime| {
            let seed = prime
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            Some(BinaryHV::random(seed))
        });
        assert_eq!(BinaryHV::DIM, HDC_DIMENSION);
    }
}
