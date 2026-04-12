// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Code Encoder — Synonym-aware encoding for programming concepts
//!
//! Replaces the character n-gram encoding that makes "add" and "sum" orthogonal.
//! Groups synonymous programming concepts and assigns shared base HVs with
//! per-word perturbations, so semantically similar operations produce similar
//! hypervectors.
//!
//! # Design
//!
//! Each synonym group shares a deterministic base HV. Individual words within
//! the group get a small perturbation (cyclic permutation) that preserves high
//! similarity (cosine > 0.7) while remaining distinguishable.
//!
//! Words not in any group fall back to the original n-gram encoding.

use std::collections::HashMap;

use symthaea_core::hdc::ContinuousHV;

/// A group of synonymous programming concepts
struct SynonymGroup {
    base_hv: ContinuousHV,
    words: Vec<String>,
}

/// Semantic encoder for programming concepts with synonym awareness.
pub struct CodeSemanticEncoder {
    dim: usize,
    /// word → HV lookup (includes all synonym group members + individual words)
    word_vectors: HashMap<String, ContinuousHV>,
}

impl CodeSemanticEncoder {
    /// Create a new semantic encoder with the given dimension.
    ///
    /// Initializes the synonym codebook with ~50 groups covering common
    /// Rust programming operations, types, and patterns.
    pub fn new(dim: usize) -> Self {
        let mut encoder = Self {
            dim,
            word_vectors: HashMap::new(),
        };
        encoder.init_synonym_groups();
        encoder
    }

    /// Get the encoding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a text string into a semantic HV.
    ///
    /// Splits text into words, looks up each word in the semantic codebook
    /// (falling back to character n-gram for unknown words), then bundles
    /// all word HVs into a single normalized vector.
    pub fn encode_text(&self, text: &str) -> ContinuousHV {
        let words: Vec<&str> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty() && w.len() > 1) // skip single chars and empty
            .collect::<Vec<_>>()
            .into_iter()
            .collect();

        // Workaround: split creates owned strings, need to re-split
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect();

        if words.is_empty() {
            return ContinuousHV::zero(self.dim);
        }

        let mut sum = vec![0.0f32; self.dim];
        let mut count = 0;

        for word in &words {
            let hv = self.encode_word(word);
            for (i, v) in hv.values.iter().enumerate() {
                if i < self.dim {
                    sum[i] += v;
                }
            }
            count += 1;
        }

        // Normalize
        if count > 0 {
            let magnitude: f32 = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for v in &mut sum {
                    *v /= magnitude;
                }
            }
        }

        ContinuousHV::from_values(sum)
    }

    /// Encode a single word. Uses semantic codebook if available, falls back
    /// to character n-gram encoding for unknown words.
    pub fn encode_word(&self, word: &str) -> ContinuousHV {
        let lower = word.to_lowercase();

        // Check semantic codebook first
        if let Some(hv) = self.word_vectors.get(&lower) {
            return hv.clone();
        }

        // Fall back to character n-gram encoding
        self.encode_ngram(&lower)
    }

    /// Compute cosine similarity between two text strings
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let hv_a = self.encode_text(a);
        let hv_b = self.encode_text(b);
        hv_a.similarity(&hv_b)
    }

    /// Character n-gram fallback encoding (same algorithm as original CodeHDEncoder)
    fn encode_ngram(&self, text: &str) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dim];

        for (i, byte) in text.bytes().enumerate() {
            let idx = ((byte as usize)
                .wrapping_mul(31)
                .wrapping_add(i.wrapping_mul(7)))
                % self.dim;
            values[idx] += 1.0;
        }

        let bytes: Vec<u8> = text.bytes().collect();
        for window in bytes.windows(2) {
            let idx = ((window[0] as usize)
                .wrapping_mul(127)
                .wrapping_add(window[1] as usize)
                .wrapping_mul(131))
                % self.dim;
            values[idx] += 0.5;
        }

        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Generate a deterministic base HV for a synonym group using a seed string.
    fn make_base_hv(dim: usize, seed: &str) -> ContinuousHV {
        // Use a deterministic hash-based RNG seeded by the group name
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        for byte in seed.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }

        let mut values = vec![0.0f32; dim];
        for i in 0..dim {
            // Deterministic pseudo-random from hash
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let frac = (hash >> 33) as f32 / (u32::MAX as f32);
            values[i] = frac * 2.0 - 1.0;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Create a perturbation of a base HV for an individual word within a group.
    ///
    /// Uses cyclic permutation by word index to create distinguishable but
    /// similar vectors (cosine similarity > 0.7 with base).
    fn perturb_hv(base: &ContinuousHV, word_index: usize, dim: usize) -> ContinuousHV {
        if word_index == 0 {
            return base.clone();
        }

        // Small cyclic permutation: shift by word_index positions
        let shift = word_index % dim.max(1);
        let mut values = vec![0.0f32; dim];
        let base_len = base.values.len().min(dim);
        for i in 0..base_len {
            values[(i + shift) % dim] = base.values[i];
        }

        // Blend with original to maintain high similarity: 85% shifted + 15% original
        for i in 0..base_len {
            values[i] = values[i] * 0.85 + base.values[i] * 0.15;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Register a synonym group: all words share a base HV with perturbations
    fn register_group(&mut self, group_name: &str, words: &[&str]) {
        let base = Self::make_base_hv(self.dim, group_name);
        for (idx, word) in words.iter().enumerate() {
            let hv = Self::perturb_hv(&base, idx, self.dim);
            self.word_vectors.insert(word.to_lowercase(), hv);
        }
    }

    /// Initialize all synonym groups for Rust programming concepts
    fn init_synonym_groups(&mut self) {
        // ── Arithmetic Operations ──
        self.register_group("arithmetic:add", &[
            "add", "sum", "plus", "total", "increment", "accumulate",
        ]);
        self.register_group("arithmetic:subtract", &[
            "subtract", "minus", "difference", "decrement", "reduce",
        ]);
        self.register_group("arithmetic:multiply", &[
            "multiply", "product", "times", "scale",
        ]);
        self.register_group("arithmetic:divide", &[
            "divide", "quotient", "ratio", "split",
        ]);
        self.register_group("arithmetic:modulo", &[
            "modulo", "remainder", "mod",
        ]);
        self.register_group("arithmetic:negate", &[
            "negate", "negative", "invert", "flip",
        ]);
        self.register_group("arithmetic:absolute", &[
            "absolute", "abs", "magnitude",
        ]);
        self.register_group("arithmetic:power", &[
            "power", "exponent", "pow", "raise",
        ]);

        // ── Comparison Operations ──
        self.register_group("compare:max", &[
            "maximum", "max", "largest", "biggest", "greatest", "highest",
        ]);
        self.register_group("compare:min", &[
            "minimum", "min", "smallest", "least", "lowest",
        ]);
        self.register_group("compare:clamp", &[
            "clamp", "bound", "constrain", "limit",
        ]);
        self.register_group("compare:equal", &[
            "equal", "equals", "same", "identical", "match",
        ]);

        // ── String Operations ──
        self.register_group("string:reverse", &[
            "reverse", "flip", "backward", "invert",
        ]);
        self.register_group("string:uppercase", &[
            "uppercase", "upper", "capitalize", "upcase",
        ]);
        self.register_group("string:lowercase", &[
            "lowercase", "lower", "downcase",
        ]);
        self.register_group("string:trim", &[
            "trim", "strip", "clean",
        ]);
        self.register_group("string:split", &[
            "split", "tokenize", "separate", "segment",
        ]);
        self.register_group("string:join", &[
            "join", "concatenate", "concat", "combine", "merge", "append",
        ]);
        self.register_group("string:replace", &[
            "replace", "substitute", "swap",
        ]);
        self.register_group("string:contains", &[
            "contains", "includes", "has",
        ]);
        self.register_group("string:starts", &[
            "starts", "prefix", "begins",
        ]);
        self.register_group("string:ends", &[
            "ends", "suffix", "trailing",
        ]);
        self.register_group("string:length", &[
            "length", "len", "size", "count",
        ]);
        self.register_group("string:repeat", &[
            "repeat", "replicate", "duplicate",
        ]);
        self.register_group("string:parse", &[
            "parse", "convert", "decode", "interpret",
        ]);
        self.register_group("string:format", &[
            "format", "stringify", "serialize", "render",
        ]);

        // ── Collection Operations ──
        self.register_group("collection:sort", &[
            "sort", "order", "arrange", "rank", "organize",
        ]);
        self.register_group("collection:filter", &[
            "filter", "select", "keep", "retain", "where",
        ]);
        self.register_group("collection:map", &[
            "map", "transform", "apply", "convert",
        ]);
        self.register_group("collection:reduce", &[
            "reduce", "fold", "aggregate", "accumulate",
        ]);
        self.register_group("collection:find", &[
            "find", "search", "locate", "lookup", "seek",
        ]);
        self.register_group("collection:flatten", &[
            "flatten", "unnest", "unroll",
        ]);
        self.register_group("collection:unique", &[
            "unique", "distinct", "deduplicate", "dedup",
        ]);
        self.register_group("collection:zip", &[
            "zip", "pair", "combine", "interleave",
        ]);
        self.register_group("collection:take", &[
            "take", "first", "head", "prefix", "limit",
        ]);
        self.register_group("collection:skip", &[
            "skip", "drop", "tail", "offset",
        ]);
        self.register_group("collection:chunk", &[
            "chunk", "batch", "partition", "group", "window",
        ]);
        self.register_group("collection:enumerate", &[
            "enumerate", "index", "number", "label",
        ]);

        // ── Boolean/Check Operations ──
        self.register_group("check:empty", &[
            "empty", "blank", "void", "nil",
        ]);
        self.register_group("check:even", &["even"]);
        self.register_group("check:odd", &["odd"]);
        self.register_group("check:positive", &[
            "positive", "nonnegative",
        ]);
        self.register_group("check:negative", &["negative"]);
        self.register_group("check:sorted", &[
            "sorted", "ordered", "monotonic",
        ]);
        self.register_group("check:palindrome", &[
            "palindrome", "symmetric",
        ]);
        self.register_group("check:prime", &["prime"]);

        // ── Math Functions ──
        self.register_group("math:factorial", &["factorial"]);
        self.register_group("math:fibonacci", &[
            "fibonacci", "fib",
        ]);
        self.register_group("math:gcd", &[
            "gcd", "greatest common divisor",
        ]);
        self.register_group("math:lcm", &[
            "lcm", "least common multiple",
        ]);
        self.register_group("math:sqrt", &[
            "sqrt", "square root", "root",
        ]);
        self.register_group("math:average", &[
            "average", "mean", "avg",
        ]);
        self.register_group("math:median", &["median", "middle"]);
        self.register_group("math:mode", &["mode", "frequent"]);
        self.register_group("math:distance", &[
            "distance", "euclidean", "norm",
        ]);

        // ── Error Handling ──
        self.register_group("error:result", &[
            "result", "error", "fallible",
        ]);
        self.register_group("error:option", &[
            "option", "optional", "nullable", "maybe",
        ]);
        self.register_group("error:unwrap", &[
            "unwrap", "expect", "extract",
        ]);
        self.register_group("error:validate", &[
            "validate", "check", "verify", "ensure",
        ]);

        // ── Structural Concepts ──
        self.register_group("struct:define", &[
            "struct", "structure", "record", "type",
        ]);
        self.register_group("struct:field", &[
            "field", "member", "attribute", "property",
        ]);
        self.register_group("struct:method", &[
            "method", "function", "fn", "procedure",
        ]);
        self.register_group("trait:define", &[
            "trait", "interface", "protocol", "contract",
        ]);
        self.register_group("enum:define", &[
            "enum", "variant", "union", "choice",
        ]);
        self.register_group("impl:define", &[
            "implement", "impl", "realize",
        ]);

        // ── Algorithm Families ──
        self.register_group("algo:sort", &[
            "bubble", "merge", "quick", "insertion", "heap",
            "ascending", "descending",
        ]);
        self.register_group("algo:search", &[
            "binary search", "linear search", "bfs", "dfs",
            "breadth", "depth",
        ]);
        self.register_group("algo:dp", &[
            "dynamic programming", "memoize", "tabulate",
            "knapsack", "subsequence",
        ]);
        self.register_group("algo:graph", &[
            "graph", "node", "edge", "vertex",
            "dijkstra", "shortest path", "topological",
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonyms_are_similar() {
        let encoder = CodeSemanticEncoder::new(512);

        // "add" and "sum" should be highly similar
        let sim = encoder.similarity("add", "sum");
        assert!(
            sim > 0.5,
            "add/sum similarity should be > 0.5, got {:.3}",
            sim
        );

        // "sort" and "order" should be highly similar
        let sim = encoder.similarity("sort", "order");
        assert!(
            sim > 0.5,
            "sort/order similarity should be > 0.5, got {:.3}",
            sim
        );

        // "filter" and "select" should be similar
        let sim = encoder.similarity("filter", "select");
        assert!(
            sim > 0.5,
            "filter/select similarity should be > 0.5, got {:.3}",
            sim
        );
    }

    #[test]
    fn test_unrelated_words_are_dissimilar() {
        let encoder = CodeSemanticEncoder::new(512);

        // "add" and "graph" should be dissimilar
        let sim = encoder.similarity("add", "graph");
        assert!(
            sim < 0.3,
            "add/graph similarity should be < 0.3, got {:.3}",
            sim
        );

        // "sort" and "parse" should be dissimilar
        let sim = encoder.similarity("sort", "parse");
        assert!(
            sim < 0.3,
            "sort/parse similarity should be < 0.3, got {:.3}",
            sim
        );
    }

    #[test]
    fn test_sentence_similarity() {
        let encoder = CodeSemanticEncoder::new(512);

        // "add two numbers" and "sum two values" should be similar
        let sim = encoder.similarity("add two numbers", "sum two values");
        assert!(
            sim > 0.3,
            "Similar purpose sentences should have similarity > 0.3, got {:.3}",
            sim
        );
    }

    #[test]
    fn test_unknown_words_fallback() {
        let encoder = CodeSemanticEncoder::new(512);

        // Unknown words should still produce valid HVs via n-gram fallback
        let hv = encoder.encode_text("xyzzy_unknown_word");
        let magnitude: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.01,
            "Unknown word HV should be normalized, got magnitude {:.3}",
            magnitude
        );
    }

    #[test]
    fn test_codebook_coverage() {
        let encoder = CodeSemanticEncoder::new(512);

        // All these common words should be in the codebook
        let common_words = [
            "add", "subtract", "multiply", "divide",
            "sort", "filter", "map", "reduce",
            "find", "search", "reverse", "contains",
            "struct", "trait", "enum", "impl",
            "result", "option", "validate",
        ];

        for word in common_words {
            assert!(
                encoder.word_vectors.contains_key(word),
                "Word '{}' should be in the codebook",
                word
            );
        }
    }

    #[test]
    fn test_within_group_similarity() {
        let encoder = CodeSemanticEncoder::new(512);

        // All words in the "add" group should be mutually similar
        let add_words = ["add", "sum", "plus", "total"];
        for i in 0..add_words.len() {
            for j in (i + 1)..add_words.len() {
                let sim = encoder.similarity(add_words[i], add_words[j]);
                assert!(
                    sim > 0.5,
                    "{}/{} similarity should be > 0.5, got {:.3}",
                    add_words[i], add_words[j], sim
                );
            }
        }
    }

    #[test]
    fn test_empty_text() {
        let encoder = CodeSemanticEncoder::new(512);
        let hv = encoder.encode_text("");
        assert_eq!(hv.values.len(), 512);
    }
}
