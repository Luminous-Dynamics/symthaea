// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HV→Code Decoder — Bridge from hypervector space to executable Rust
//!
//! This is the linchpin for emergence. The system can:
//! - Encode code as hypervectors (CodeHDEncoder)
//! - Evolve hypervectors through genetic operations (CodePrimitiveEvolver)
//! - Compose hypervectors via bundle/bind (HDC algebra)
//!
//! But without a DECODER, evolved vectors are meaningless numbers.
//! This module bridges the gap: given an evolved HV, produce executable code.
//!
//! # Architecture
//!
//! ```text
//! Evolved HV (16,384D)
//!     ↓ find K nearest known fragments
//! CodeFragment[] (ranked by similarity)
//!     ↓ compose fragments guided by signature + purpose
//! Candidate source code
//!     ↓ compile + test (fitness evaluation)
//! Executable Rust (or failure → evolve again)
//! ```
//!
//! The codebook grows over time. Every successful generation adds its code
//! as a new fragment. Every discovered pattern from evolution is registered.
//! The system literally builds its own vocabulary of code.

use std::collections::HashMap;

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_encoder::CodeHDEncoder;

/// A known code fragment indexed by its hypervector.
///
/// Fragments are the vocabulary of code generation — small, composable
/// pieces of Rust that can be combined into full solutions.
#[derive(Debug, Clone)]
pub struct CodeFragment {
    /// Human-readable name for this fragment
    pub name: String,
    /// The actual Rust code
    pub code: String,
    /// HDC encoding of this fragment
    pub hv: ContinuousHV,
    /// What kind of operation this fragment performs
    pub category: FragmentCategory,
    /// How many times this fragment has been used successfully
    pub success_count: u32,
    /// How many times it was tried but failed
    pub failure_count: u32,
}

impl CodeFragment {
    /// Bayesian success rate
    pub fn success_rate(&self) -> f32 {
        (self.success_count as f32 + 1.0)
            / (self.success_count as f32 + self.failure_count as f32 + 2.0)
    }
}

/// What kind of operation a fragment performs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentCategory {
    /// Iterator start: .iter(), .chars(), .into_iter()
    IteratorSource,
    /// Iterator adapter: .filter(), .map(), .flat_map()
    IteratorAdapter,
    /// Iterator terminal: .collect(), .sum(), .count(), .any(), .all()
    IteratorTerminal,
    /// Loop construct: for, while, loop
    LoopConstruct,
    /// Accumulator pattern: let mut acc; ... acc += ...; acc
    Accumulator,
    /// Comparison/predicate: ==, !=, >, <, %, is_*
    Predicate,
    /// Type conversion: as, .into(), .to_string(), .parse()
    TypeConversion,
    /// Wrapping: Some(), Ok(), Err()
    Wrapping,
    /// Complete function body (from discovered patterns)
    CompleteBody,
}

/// The decoder: converts hypervectors back to executable code.
pub struct HvCodeDecoder {
    encoder: CodeHDEncoder,
    /// Known code fragments indexed by category
    fragments: Vec<CodeFragment>,
    /// Category → fragment indices for efficient lookup
    by_category: HashMap<FragmentCategory, Vec<usize>>,
}

impl HvCodeDecoder {
    /// Create a new decoder with the standard fragment library.
    pub fn new(dim: usize) -> Self {
        let encoder = CodeHDEncoder::new(dim);
        let mut decoder = Self {
            encoder,
            fragments: Vec::new(),
            by_category: HashMap::new(),
        };
        decoder.seed_standard_fragments();
        decoder
    }

    /// Seed with standard Rust code fragments — the initial vocabulary.
    fn seed_standard_fragments(&mut self) {
        // Iterator sources
        self.register("iter_slice", ".iter()", FragmentCategory::IteratorSource);
        self.register("iter_chars", ".chars()", FragmentCategory::IteratorSource);
        self.register(
            "iter_into",
            ".into_iter()",
            FragmentCategory::IteratorSource,
        );
        self.register("iter_bytes", ".bytes()", FragmentCategory::IteratorSource);
        self.register("iter_lines", ".lines()", FragmentCategory::IteratorSource);
        self.register("iter_range", "(0..n)", FragmentCategory::IteratorSource);
        self.register(
            "iter_range_inc",
            "(1..=n)",
            FragmentCategory::IteratorSource,
        );
        self.register(
            "iter_windows",
            ".windows(n)",
            FragmentCategory::IteratorSource,
        );
        self.register(
            "iter_enumerate",
            ".enumerate()",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "iter_zip",
            ".zip(other.iter())",
            FragmentCategory::IteratorAdapter,
        );

        // Iterator adapters
        self.register(
            "filter_even",
            ".filter(|&x| x % 2 == 0)",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "filter_odd",
            ".filter(|&x| x % 2 != 0)",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "filter_positive",
            ".filter(|&x| x > 0)",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "filter_alpha",
            ".filter(|c| c.is_alphabetic())",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "filter_digit",
            ".filter(|c| c.is_ascii_digit())",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "map_uppercase",
            ".map(|c| c.to_uppercase().next().unwrap())",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "map_lowercase",
            ".map(|c| c.to_lowercase().next().unwrap())",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "map_square",
            ".map(|x| x * x)",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "map_double",
            ".map(|x| x * 2)",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "map_to_string",
            ".map(|x| x.to_string())",
            FragmentCategory::IteratorAdapter,
        );
        self.register(
            "flat_map",
            ".flat_map(|x| x.iter())",
            FragmentCategory::IteratorAdapter,
        );
        self.register("copied", ".copied()", FragmentCategory::IteratorAdapter);
        self.register("cloned", ".cloned()", FragmentCategory::IteratorAdapter);
        self.register("rev", ".rev()", FragmentCategory::IteratorAdapter);
        self.register("skip", ".skip(n)", FragmentCategory::IteratorAdapter);
        self.register("take", ".take(n)", FragmentCategory::IteratorAdapter);
        self.register("peekable", ".peekable()", FragmentCategory::IteratorAdapter);

        // Iterator terminals
        self.register(
            "collect_vec",
            ".collect::<Vec<_>>()",
            FragmentCategory::IteratorTerminal,
        );
        self.register(
            "collect_string",
            ".collect::<String>()",
            FragmentCategory::IteratorTerminal,
        );
        self.register(
            "collect_hashset",
            ".collect::<HashSet<_>>()",
            FragmentCategory::IteratorTerminal,
        );
        self.register("sum", ".sum()", FragmentCategory::IteratorTerminal);
        self.register("product", ".product()", FragmentCategory::IteratorTerminal);
        self.register("count", ".count()", FragmentCategory::IteratorTerminal);
        self.register(
            "any",
            ".any(|x| condition)",
            FragmentCategory::IteratorTerminal,
        );
        self.register(
            "all",
            ".all(|x| condition)",
            FragmentCategory::IteratorTerminal,
        );
        self.register(
            "find",
            ".find(|x| condition)",
            FragmentCategory::IteratorTerminal,
        );
        self.register("max", ".max()", FragmentCategory::IteratorTerminal);
        self.register("min", ".min()", FragmentCategory::IteratorTerminal);
        self.register(
            "fold",
            ".fold(init, |acc, x| acc + x)",
            FragmentCategory::IteratorTerminal,
        );
        self.register(
            "position",
            ".position(|x| condition)",
            FragmentCategory::IteratorTerminal,
        );

        // Predicates
        self.register("eq_zero", "x == 0", FragmentCategory::Predicate);
        self.register("gt_zero", "x > 0", FragmentCategory::Predicate);
        self.register("lt_zero", "x < 0", FragmentCategory::Predicate);
        self.register("mod_eq", "x % n == 0", FragmentCategory::Predicate);
        self.register("is_alpha", "c.is_alphabetic()", FragmentCategory::Predicate);
        self.register(
            "is_digit",
            "c.is_ascii_digit()",
            FragmentCategory::Predicate,
        );
        self.register("is_upper", "c.is_uppercase()", FragmentCategory::Predicate);
        self.register("is_lower", "c.is_lowercase()", FragmentCategory::Predicate);

        // Type conversions
        self.register(
            "to_string",
            ".to_string()",
            FragmentCategory::TypeConversion,
        );
        self.register(
            "to_lowercase",
            ".to_lowercase()",
            FragmentCategory::TypeConversion,
        );
        self.register(
            "to_uppercase",
            ".to_uppercase()",
            FragmentCategory::TypeConversion,
        );
        self.register(
            "parse_int",
            ".parse::<i32>().unwrap_or(0)",
            FragmentCategory::TypeConversion,
        );
        self.register("as_str", ".as_str()", FragmentCategory::TypeConversion);
        self.register("trim", ".trim()", FragmentCategory::TypeConversion);

        // Wrapping
        self.register("some", "Some(x)", FragmentCategory::Wrapping);
        self.register("none", "None", FragmentCategory::Wrapping);
        self.register("ok", "Ok(x)", FragmentCategory::Wrapping);
        self.register("err", "Err(e)", FragmentCategory::Wrapping);
    }

    /// Register a new fragment in the codebook.
    pub fn register(&mut self, name: &str, code: &str, category: FragmentCategory) {
        let hv = self.encoder.encode_name(code);
        let idx = self.fragments.len();
        self.fragments.push(CodeFragment {
            name: name.to_string(),
            code: code.to_string(),
            hv,
            category,
            success_count: 0,
            failure_count: 0,
        });
        self.by_category.entry(category).or_default().push(idx);
    }

    /// Register a complete discovered pattern (from evolution or dream).
    pub fn register_discovery(&mut self, name: &str, code: &str) {
        self.register(name, code, FragmentCategory::CompleteBody);
    }

    /// Decode a hypervector into code by finding the K nearest fragments.
    ///
    /// Returns the top-K fragments ranked by cosine similarity to the query HV.
    pub fn decode_nearest(
        &self,
        query_hv: &ContinuousHV,
        top_k: usize,
    ) -> Vec<(f32, &CodeFragment)> {
        let mut scored: Vec<(f32, &CodeFragment)> = self
            .fragments
            .iter()
            .map(|f| (query_hv.similarity(&f.hv), f))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Decode a hypervector into an iterator chain by finding the best
    /// source → adapters → terminal composition.
    ///
    /// This is the key decoding operation: given an evolved HV, construct
    /// a valid iterator pipeline by selecting the most similar fragment
    /// from each category.
    pub fn decode_iterator_chain(
        &self,
        query_hv: &ContinuousHV,
        input_type: &str,
        return_type: &str,
    ) -> Option<String> {
        // Find best source
        let source = self.best_in_category(query_hv, FragmentCategory::IteratorSource)?;

        // Find best adapters (0-2)
        let adapters = self.top_in_category(query_hv, FragmentCategory::IteratorAdapter, 2);

        // Find best terminal
        let terminal = self.best_in_category(query_hv, FragmentCategory::IteratorTerminal)?;

        // Compose: source + adapters + terminal
        let mut chain = source.code.clone();
        for adapter in &adapters {
            if adapter.0 > 0.3 {
                // Only include adapters with reasonable similarity
                chain.push_str(&adapter.1.code);
            }
        }
        chain.push_str(&terminal.code);

        // Prefix with the input parameter
        let prefix = if input_type.contains("&str") {
            "input"
        } else if input_type.contains("&[") || input_type.contains("Vec") {
            "v"
        } else {
            "input"
        };

        Some(format!("{}{}", prefix, chain))
    }

    /// Find the best fragment in a category by similarity to query.
    fn best_in_category(
        &self,
        query_hv: &ContinuousHV,
        category: FragmentCategory,
    ) -> Option<&CodeFragment> {
        let indices = self.by_category.get(&category)?;
        indices
            .iter()
            .map(|&idx| {
                let f = &self.fragments[idx];
                (query_hv.similarity(&f.hv), f)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, f)| f)
    }

    /// Find top-K fragments in a category.
    pub fn top_in_category(
        &self,
        query_hv: &ContinuousHV,
        category: FragmentCategory,
        k: usize,
    ) -> Vec<(f32, &CodeFragment)> {
        let indices = match self.by_category.get(&category) {
            Some(i) => i,
            None => return Vec::new(),
        };
        let mut scored: Vec<(f32, &CodeFragment)> = indices
            .iter()
            .map(|&idx| {
                let f = &self.fragments[idx];
                (query_hv.similarity(&f.hv), f)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Record success/failure for a fragment (updates Bayesian rates).
    pub fn record_outcome(&mut self, fragment_name: &str, success: bool) {
        if let Some(f) = self.fragments.iter_mut().find(|f| f.name == fragment_name) {
            if success {
                f.success_count += 1;
            } else {
                f.failure_count += 1;
            }
        }
    }

    /// Number of fragments in the codebook.
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Number of discovered patterns (CompleteBody category).
    pub fn discovery_count(&self) -> usize {
        self.by_category
            .get(&FragmentCategory::CompleteBody)
            .map_or(0, |v| v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_fragments() {
        let decoder = HvCodeDecoder::new(512);
        assert!(
            decoder.fragment_count() > 50,
            "Should have 50+ seeded fragments"
        );
    }

    #[test]
    fn test_decode_nearest_finds_similar() {
        let decoder = HvCodeDecoder::new(512);
        // Encode "filter" and find nearest
        let query = decoder.encoder.encode_name(".filter(|x| condition)");
        let results = decoder.decode_nearest(&query, 5);
        assert!(!results.is_empty());
        // Top result should be a filter-related fragment
        assert!(
            results[0].1.name.contains("filter")
                || results[0].1.category == FragmentCategory::IteratorAdapter,
            "Expected filter-like fragment, got: {} ({})",
            results[0].1.name,
            results[0].0
        );
    }

    #[test]
    fn test_register_discovery() {
        let mut decoder = HvCodeDecoder::new(512);
        let before = decoder.discovery_count();
        decoder.register_discovery("divisor_sum", "(1..n).filter(|i| n % i == 0).sum::<u64>()");
        assert_eq!(decoder.discovery_count(), before + 1);
    }

    #[test]
    fn test_decode_iterator_chain() {
        let decoder = HvCodeDecoder::new(512);
        // Encode a concept similar to "reverse string"
        let query = decoder
            .encoder
            .encode_name("reverse characters collect string");
        let chain = decoder.decode_iterator_chain(&query, "&str", "String");
        assert!(chain.is_some(), "Should produce an iterator chain");
        let code = chain.unwrap();
        assert!(code.contains("input"), "Should reference input parameter");
    }

    #[test]
    fn test_record_outcome() {
        let mut decoder = HvCodeDecoder::new(512);
        decoder.record_outcome("iter_slice", true);
        decoder.record_outcome("iter_slice", true);
        decoder.record_outcome("iter_slice", false);
        let frag = decoder
            .fragments
            .iter()
            .find(|f| f.name == "iter_slice")
            .unwrap();
        assert_eq!(frag.success_count, 2);
        assert_eq!(frag.failure_count, 1);
        assert!(frag.success_rate() > 0.5);
    }
}
