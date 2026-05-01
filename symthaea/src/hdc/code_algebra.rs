// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code HDC Algebra — operations on code-as-hypervectors
//!
//! Provides high-level algebraic operations on code encoded as hypervectors:
//! similarity search, analogy reasoning, composition, semantic diff,
//! and pattern abstraction.
//!
//! # Key Operations
//!
//! - **find_similar**: Find functions/types similar to a query pattern
//! - **analogy**: "A is to B as C is to ?" reasoning in code space
//! - **compose**: Combine two code patterns
//! - **semantic_diff**: What changed between code versions?
//! - **abstract_pattern**: Extract common pattern from examples

use symthaea_core::hdc::ContinuousHV;

use super::code_encoder::CodeHDEncoder;
use crate::language::code_parser::ParsedCode;

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    /// Index in the candidate list
    pub index: usize,
    /// Cosine similarity score (-1.0 to 1.0)
    pub similarity: f32,
}

/// Result of a semantic diff
#[derive(Debug, Clone)]
pub struct CodeDiff {
    /// The diff vector in HDC space
    pub diff_hv: ContinuousHV,
    /// Magnitude of change (0.0 = identical, higher = more different)
    pub change_magnitude: f32,
    /// Cosine similarity between old and new (1.0 = identical, 0.0 = orthogonal)
    pub similarity: f32,
}

/// Algebraic operations on code hypervectors
pub struct CodeAlgebra {
    encoder: CodeHDEncoder,
}

impl CodeAlgebra {
    /// Create a new CodeAlgebra with the given encoder
    pub fn new(encoder: CodeHDEncoder) -> Self {
        Self { encoder }
    }

    /// Get a reference to the encoder
    pub fn encoder(&self) -> &CodeHDEncoder {
        &self.encoder
    }

    /// Get a mutable reference to the encoder
    pub fn encoder_mut(&mut self) -> &mut CodeHDEncoder {
        &mut self.encoder
    }

    // ========================================================================
    // Similarity Search
    // ========================================================================

    /// Find the top-k most similar vectors to a query
    pub fn find_similar(
        &self,
        query: &ContinuousHV,
        candidates: &[ContinuousHV],
        top_k: usize,
    ) -> Vec<SimilarityMatch> {
        let mut matches: Vec<SimilarityMatch> = candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| SimilarityMatch {
                index: i,
                similarity: query.similarity(candidate),
            })
            .collect();

        // Sort by similarity descending
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(top_k);
        matches
    }

    /// Find similar entities in a parsed codebase
    pub fn find_similar_in_code(
        &self,
        query: &ContinuousHV,
        parsed_files: &[(String, ParsedCode)],
        top_k: usize,
    ) -> Vec<(String, String, f32)> {
        let mut results = Vec::new();

        for (path, parsed) in parsed_files {
            for entity in parsed.all_entities() {
                let entity_hv = self.encoder.encode_entity(entity);
                let sim = query.similarity(&entity_hv);
                results.push((path.clone(), entity.name.clone(), sim));
            }
        }

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    // ========================================================================
    // Analogy
    // ========================================================================

    /// Compute analogy: "A is to B as C is to ?"
    ///
    /// Returns the vector D such that the relationship A→B
    /// is the same as C→D. In HDC:
    /// D = C ⊕ unbind(A, B) ≈ C + (B - A) in the linear approximation
    pub fn analogy(
        &self,
        a: &ContinuousHV,
        b: &ContinuousHV,
        c: &ContinuousHV,
        candidates: &[ContinuousHV],
    ) -> Vec<SimilarityMatch> {
        // Compute the relationship vector: what transforms A into B?
        // In real-valued HDC: relationship ≈ B * A^(-1) ≈ element-wise B/A
        // Simplified: relationship = B - A (additive analogy)
        let dim = a.values.len();
        let mut target_values = vec![0.0f32; dim];
        for i in 0..dim {
            target_values[i] = c.values[i] + (b.values[i] - a.values[i]);
        }

        // Normalize
        let magnitude: f32 = target_values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut target_values {
                *v /= magnitude;
            }
        }

        let target = ContinuousHV::from_values(target_values);
        self.find_similar(&target, candidates, 5)
    }

    // ========================================================================
    // Composition
    // ========================================================================

    /// Compose two code patterns: bundle them together
    ///
    /// The result is a vector that is similar to both inputs,
    /// representing their combination.
    pub fn compose(&self, a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
        ContinuousHV::bundle_owned(&[a.clone(), b.clone()])
    }

    /// Compose multiple code patterns
    pub fn compose_many(&self, patterns: &[&ContinuousHV]) -> ContinuousHV {
        if patterns.is_empty() {
            return ContinuousHV::zero(self.encoder.dim());
        }
        let owned: Vec<ContinuousHV> = patterns.iter().map(|p| (*p).clone()).collect();
        ContinuousHV::bundle_owned(&owned)
    }

    // ========================================================================
    // Semantic Diff
    // ========================================================================

    /// Compute semantic diff between two code snapshots
    pub fn semantic_diff(&self, old: &ParsedCode, new: &ParsedCode) -> CodeDiff {
        let old_hv = self.encoder.encode_module(old);
        let new_hv = self.encoder.encode_module(new);

        let similarity = old_hv.similarity(&new_hv);

        // Diff vector
        let dim = old_hv.values.len();
        let mut diff_values = vec![0.0f32; dim];
        for i in 0..dim {
            diff_values[i] = new_hv.values[i] - old_hv.values[i];
        }

        let change_magnitude: f32 = diff_values.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Normalize diff
        if change_magnitude > 0.0 {
            for v in &mut diff_values {
                *v /= change_magnitude;
            }
        }

        CodeDiff {
            diff_hv: ContinuousHV::from_values(diff_values),
            change_magnitude,
            similarity,
        }
    }

    /// Compute semantic diff between two pre-encoded vectors
    pub fn semantic_diff_hvs(&self, old_hv: &ContinuousHV, new_hv: &ContinuousHV) -> CodeDiff {
        let similarity = old_hv.similarity(new_hv);

        let dim = old_hv.values.len();
        let mut diff_values = vec![0.0f32; dim];
        for i in 0..dim {
            diff_values[i] = new_hv.values[i] - old_hv.values[i];
        }

        let change_magnitude: f32 = diff_values.iter().map(|v| v * v).sum::<f32>().sqrt();

        if change_magnitude > 0.0 {
            for v in &mut diff_values {
                *v /= change_magnitude;
            }
        }

        CodeDiff {
            diff_hv: ContinuousHV::from_values(diff_values),
            change_magnitude,
            similarity,
        }
    }

    // ========================================================================
    // Pattern Abstraction
    // ========================================================================

    /// Extract the common pattern from multiple implementations.
    ///
    /// The abstract pattern is the centroid (bundled average) of all examples.
    /// It captures what all examples share, filtering out implementation-specific details.
    pub fn abstract_pattern(&self, examples: &[ContinuousHV]) -> ContinuousHV {
        if examples.is_empty() {
            return ContinuousHV::zero(self.encoder.dim());
        }

        ContinuousHV::bundle_owned(examples)
    }

    /// Measure how coherent a set of code patterns are
    /// (how similar they all are to each other)
    pub fn measure_coherence(&self, patterns: &[ContinuousHV]) -> f32 {
        if patterns.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0f32;
        let mut count = 0u32;

        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                total_sim += patterns[i].similarity(&patterns[j]);
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::{Entity, EntityKind, Span};

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    fn make_algebra() -> CodeAlgebra {
        CodeAlgebra::new(CodeHDEncoder::new(512))
    }

    #[test]
    fn test_find_similar() {
        let algebra = make_algebra();
        let query = algebra.encoder().encode_name("sort");
        let candidates: Vec<ContinuousHV> = ["sort_list", "bubble_sort", "hash_map", "binary_tree"]
            .iter()
            .map(|name| algebra.encoder().encode_name(name))
            .collect();

        let results = algebra.find_similar(&query, &candidates, 2);
        assert_eq!(results.len(), 2);
        // The sort-related names should be most similar
        assert!(results[0].similarity >= results[1].similarity);
    }

    #[test]
    fn test_compose() {
        let algebra = make_algebra();
        let a = algebra.encoder().encode_name("parser");
        let b = algebra.encoder().encode_name("validator");

        let composed = algebra.compose(&a, &b);

        // Composed should be similar to both inputs
        let sim_a = composed.similarity(&a);
        let sim_b = composed.similarity(&b);
        assert!(sim_a > 0.0);
        assert!(sim_b > 0.0);
    }

    #[test]
    fn test_semantic_diff() {
        let algebra = make_algebra();

        let mut old = ParsedCode::new("fn a() {}", "rust");
        old.entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));

        let mut new = ParsedCode::new("fn a() {} fn b() {}", "rust");
        new.entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));
        new.entities
            .push(Entity::new(EntityKind::Function, "b", test_span()));

        let diff = algebra.semantic_diff(&old, &new);
        // There should be some change
        assert!(diff.change_magnitude > 0.0);
        // Old and new should still be somewhat similar (a is shared)
        assert!(diff.similarity > -1.0);
    }

    #[test]
    fn test_abstract_pattern() {
        let algebra = make_algebra();

        // Several "sort" functions should abstract to a sort-like pattern
        let examples: Vec<ContinuousHV> = ["bubble_sort", "quick_sort", "merge_sort"]
            .iter()
            .map(|name| algebra.encoder().encode_name(name))
            .collect();

        let pattern = algebra.abstract_pattern(&examples);

        // Pattern should be more similar to sort functions than to unrelated code
        let sort_sim = pattern.similarity(&algebra.encoder().encode_name("heap_sort"));
        let other_sim = pattern.similarity(&algebra.encoder().encode_name("database_connection"));

        assert!(
            sort_sim > other_sim,
            "Abstract pattern should be more similar to sort: {} vs {}",
            sort_sim,
            other_sim
        );
    }

    #[test]
    fn test_measure_coherence() {
        let algebra = make_algebra();

        // Similar patterns should have high coherence
        let similar: Vec<ContinuousHV> = ["sort_a", "sort_b", "sort_c"]
            .iter()
            .map(|n| algebra.encoder().encode_name(n))
            .collect();

        let diverse: Vec<ContinuousHV> = ["sort", "database", "network"]
            .iter()
            .map(|n| algebra.encoder().encode_name(n))
            .collect();

        let similar_coherence = algebra.measure_coherence(&similar);
        let diverse_coherence = algebra.measure_coherence(&diverse);

        assert!(
            similar_coherence > diverse_coherence,
            "Similar patterns should be more coherent: {} vs {}",
            similar_coherence,
            diverse_coherence
        );
    }

    #[test]
    fn test_analogy() {
        let algebra = make_algebra();

        // Simple analogy: "sort" is to "Vec" as ? is to "HashMap"
        let sort = algebra.encoder().encode_name("sort");
        let vec = algebra.encoder().encode_name("Vec");
        let hashmap = algebra.encoder().encode_name("HashMap");

        let candidates: Vec<ContinuousHV> = ["get", "sort", "insert", "contains"]
            .iter()
            .map(|n| algebra.encoder().encode_name(n))
            .collect();

        let results = algebra.analogy(&sort, &vec, &hashmap, &candidates);
        assert!(!results.is_empty());
        // We don't assert specific results since analogy quality depends on dim
    }
}
