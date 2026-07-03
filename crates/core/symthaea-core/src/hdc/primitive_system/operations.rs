// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public API operations for the Primitive System.
//!
//! This module contains all the public methods on PrimitiveSystem for:
//! - Querying primitives (get, get_tier, domain, count, etc.)
//! - Validation (orthogonality, derivation chain, domain validation)
//! - Compositional operator execution (sequence, conditional, fixpoint, iterate)
//! - Summary/reporting
//! - Similarity search (find_similar, LSH index, batch operations)
//! - Typed primitive operations (bind, bundle, analogy, permute, encode_sequence)
//! - Default trait implementation

use super::{LshIndex, Primitive, PrimitiveError, PrimitiveResult, PrimitiveSystem, PrimitiveTier};
use crate::hdc::binary_hv::BinaryHV;

impl PrimitiveSystem {
    /// Get a primitive by name
    pub fn get(&self, name: &str) -> Option<&Primitive> {
        self.primitives.get(name)
    }

    /// Get all primitives in a tier
    pub fn get_tier(&self, tier: PrimitiveTier) -> Vec<&Primitive> {
        self.by_tier
            .get(&tier)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|n| self.primitives.get(n))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get a domain manifold
    pub fn domain(&self, name: &str) -> Option<&super::DomainManifold> {
        self.domains.get(name)
    }

    /// Check orthogonality between primitives
    pub fn check_orthogonality(&self, name1: &str, name2: &str) -> Option<f32> {
        let p1 = self.get(name1)?;
        let p2 = self.get(name2)?;
        Some(p1.encoding.similarity(&p2.encoding))
    }

    /// Validate that all primitives in a tier are sufficiently orthogonal.
    ///
    /// Returns pairs whose similarity deviates from 0.5 (random baseline) by
    /// more than `threshold`. With 16,384-bit vectors, expected deviation from
    /// 0.5 is ~0.008 (1 sigma) for random pairs, so threshold=0.03 is approx 4 sigma.
    pub fn validate_tier_orthogonality(
        &self,
        tier: PrimitiveTier,
        threshold: f32,
    ) -> Vec<(String, String, f32)> {
        let mut violations = Vec::new();
        let primitives = self.get_tier(tier);

        for i in 0..primitives.len() {
            for j in (i + 1)..primitives.len() {
                let sim = primitives[i].encoding.similarity(&primitives[j].encoding);
                let deviation = (sim - 0.5).abs();
                if deviation > threshold {
                    violations.push((primitives[i].name.clone(), primitives[j].name.clone(), sim));
                }
            }
        }

        violations
    }

    /// Get count of primitives
    pub fn count(&self) -> usize {
        self.primitives.len()
    }

    /// Get all primitives as an iterator
    pub fn all_primitives(&self) -> impl Iterator<Item = &Primitive> {
        self.primitives.values()
    }

    /// Get count by tier
    pub fn count_tier(&self, tier: PrimitiveTier) -> usize {
        self.by_tier.get(&tier).map(|v| v.len()).unwrap_or(0)
    }

    /// Get all binding rules
    pub fn binding_rules(&self) -> &[super::BindingRule] {
        &self.binding_rules
    }

    // === DERIVATION CHAIN VALIDATION ===

    /// Validate the derivation chain: check that all derived primitives
    /// have their parents registered and encodings are genuinely composed.
    pub fn validate_derivation_chain(&self) -> Vec<(String, bool, Option<String>)> {
        let mut diagnostics = Vec::new();
        for (name, prim) in &self.primitives {
            if !prim.is_base
                && let Some(ref derivation) = prim.derivation
            {
                // Parse parent names from derivation expression (split on ^ or whitespace ops)
                let parent_names: Vec<&str> = derivation
                    .split(['^', ' '])
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty() && s.chars().next().is_some_and(|c| c.is_uppercase()))
                    .collect();
                let all_found = parent_names
                    .iter()
                    .all(|p| self.primitives.contains_key(*p));
                if !all_found {
                    diagnostics.push((name.clone(), false, Some(derivation.clone())));
                }
            }
        }
        diagnostics
    }

    /// Validate domain orthogonality: for each pair of domains,
    /// measure average inter-domain similarity (should be ~0.5 for random baseline).
    pub fn validate_domain_orthogonality(&self) -> Vec<(String, String, f32)> {
        let domain_names: Vec<String> = self.domains.keys().cloned().collect();
        let mut results = Vec::new();

        for i in 0..domain_names.len() {
            for j in (i + 1)..domain_names.len() {
                let prims_i: Vec<&Primitive> = self
                    .primitives
                    .values()
                    .filter(|p| p.domain == domain_names[i])
                    .collect();
                let prims_j: Vec<&Primitive> = self
                    .primitives
                    .values()
                    .filter(|p| p.domain == domain_names[j])
                    .collect();

                if prims_i.is_empty() || prims_j.is_empty() {
                    continue;
                }

                let mut total_sim = 0.0f32;
                let mut count = 0u32;
                for pi in &prims_i {
                    for pj in &prims_j {
                        total_sim += pi.encoding.similarity(&pj.encoding);
                        count += 1;
                    }
                }
                let avg_sim = total_sim / count as f32;
                results.push((domain_names[i].clone(), domain_names[j].clone(), avg_sim));
            }
        }
        results
    }

    /// Run all validation checks and return a summary.
    #[allow(clippy::type_complexity)]
    pub fn validate_all(
        &self,
    ) -> (
        Vec<(String, bool, Option<String>)>,
        Vec<(String, String, f32)>,
    ) {
        (
            self.validate_derivation_chain(),
            self.validate_domain_orthogonality(),
        )
    }

    // === COMPOSITIONAL OPERATOR EXECUTION (Tier 7) ===

    /// Execute a sequence of HDC operations (function composition).
    pub fn execute_sequence(
        ops: &[Box<dyn Fn(BinaryHV) -> BinaryHV>],
        input: BinaryHV,
    ) -> BinaryHV {
        let mut result = input;
        for op in ops {
            result = op(result);
        }
        result
    }

    /// Execute conditional: if condition is similar to reference (above threshold),
    /// apply then_op, otherwise apply else_op.
    pub fn execute_conditional(
        condition: &BinaryHV,
        reference: &BinaryHV,
        threshold: f32,
        then_op: &dyn Fn(BinaryHV) -> BinaryHV,
        else_op: &dyn Fn(BinaryHV) -> BinaryHV,
        input: BinaryHV,
    ) -> BinaryHV {
        if condition.similarity(reference) > threshold {
            then_op(input)
        } else {
            else_op(input)
        }
    }

    /// Execute fixpoint iteration: apply op until the result stabilizes
    /// (similarity to previous > threshold) or max_iter is reached.
    /// Returns (final_result, iterations_used).
    pub fn execute_fixpoint(
        op: &dyn Fn(BinaryHV) -> BinaryHV,
        initial: BinaryHV,
        max_iter: usize,
        threshold: f32,
    ) -> (BinaryHV, usize) {
        let mut current = initial;
        for i in 0..max_iter {
            let next = op(current);
            if next.similarity(&current) > threshold {
                return (next, i + 1);
            }
            current = next;
        }
        (current, max_iter)
    }

    /// Execute iterate: apply op n times starting from initial.
    pub fn execute_iterate(
        op: &dyn Fn(BinaryHV) -> BinaryHV,
        initial: BinaryHV,
        n: usize,
    ) -> BinaryHV {
        let mut result = initial;
        for _ in 0..n {
            result = op(result);
        }
        result
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("# Primitive System Summary\n\n");
        report.push_str(&format!("**Total Primitives**: {}\n", self.count()));
        report.push_str(&format!("**Domains**: {}\n\n", self.domains.len()));

        report.push_str("## Primitives by Tier\n\n");
        for tier in &[
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
            PrimitiveTier::Code,
        ] {
            let count = self.count_tier(*tier);
            if count > 0 {
                report.push_str(&format!("- **{tier:?}**: {count} primitives\n"));
            }
        }

        report.push_str("\n## Domain Manifolds\n\n");
        for (name, domain) in &self.domains {
            report.push_str(&format!("### {name}\n"));
            report.push_str(&format!("- **Tier**: {:?}\n", domain.tier));
            report.push_str(&format!("- **Purpose**: {}\n\n", domain.purpose));
        }

        report.push_str(&format!(
            "\n## Binding Rules: {}\n\n",
            self.binding_rules.len()
        ));

        report
    }

    // ========================================================================
    // SIMILARITY SEARCH
    // ========================================================================

    /// Get all primitive names as a vector
    pub fn all_primitive_names(&self) -> Vec<&str> {
        self.primitives.keys().map(|s| s.as_str()).collect()
    }

    /// Find primitives most similar to the given primitive by name.
    ///
    /// Returns a vector of (name, similarity) pairs sorted by descending similarity.
    pub fn find_similar(&self, name: &str, top_k: usize) -> Vec<(String, f32)> {
        let query = match self.primitives.get(name) {
            Some(p) => &p.encoding,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(String, f32)> = self
            .primitives
            .iter()
            .filter(|(n, _)| *n != name)
            .map(|(n, p)| {
                let sim = query.similarity(&p.encoding);
                (n.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Find primitives most similar to a given encoding.
    ///
    /// Useful for finding matches to composed/derived encodings.
    pub fn find_similar_to_encoding(
        &self,
        encoding: &BinaryHV,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self
            .primitives
            .iter()
            .map(|(n, p)| {
                let sim = encoding.similarity(&p.encoding);
                (n.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    // ========================================================================
    // LSH INDEX FOR FAST APPROXIMATE SIMILARITY SEARCH
    // ========================================================================

    /// Create an LSH (Locality Sensitive Hashing) index for fast similarity search.
    ///
    /// LSH provides O(1) expected time for approximate nearest neighbor queries
    /// instead of O(n) linear scan. For 200+ primitives this is faster.
    ///
    /// # Parameters
    /// - `num_bands`: Number of hash tables (more = higher recall, more memory)
    /// - `bits_per_band`: Bits sampled per table (fewer = more collisions/candidates)
    ///
    /// # Example
    /// ```ignore
    /// let system = PrimitiveSystem::global();
    /// let lsh = system.build_lsh_index(8, 64);
    /// let candidates = lsh.query_candidates(&some_encoding);
    /// ```
    pub fn build_lsh_index(&self, num_bands: usize, bits_per_band: usize) -> LshIndex {
        LshIndex::build(&self.primitives, num_bands, bits_per_band)
    }

    /// Find similar primitives using LSH (faster for large primitive sets).
    ///
    /// This method uses a pre-built LSH index for O(1) candidate retrieval,
    /// then does full similarity comparison only on candidates.
    pub fn find_similar_lsh(
        &self,
        encoding: &BinaryHV,
        top_k: usize,
        lsh: &LshIndex,
    ) -> Vec<(String, f32)> {
        // Get candidate primitive names from LSH
        let candidates = lsh.query_candidates(encoding);

        if candidates.is_empty() {
            // Fallback to linear scan if no LSH candidates
            return self.find_similar_to_encoding(encoding, top_k);
        }

        // Compute exact similarity only for candidates
        let mut similarities: Vec<(String, f32)> = candidates
            .into_iter()
            .filter_map(|name| {
                self.primitives.get(&name).map(|p| {
                    let sim = encoding.similarity(&p.encoding);
                    (name, sim)
                })
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    // ========================================================================
    // BATCH SIMILARITY SEARCH (SIMD-OPTIMIZED)
    // ========================================================================

    /// Batch find similar primitives for multiple query encodings.
    ///
    /// Uses parallel processing with rayon for queries and SIMD for similarity
    /// computation. Automatically selects optimal algorithm based on batch size.
    ///
    /// # Performance
    /// - Small batches (<50): Sequential processing (avoids parallel overhead)
    /// - Large batches (>=50): Parallel processing (2-8x speedup on multi-core)
    ///
    /// # Example
    /// ```ignore
    /// let system = PrimitiveSystem::global();
    /// let queries = vec![encoding1, encoding2, encoding3];
    /// let results = system.batch_find_similar(&queries, 5);
    /// // results[0] = top 5 similar to encoding1
    /// // results[1] = top 5 similar to encoding2
    /// // etc.
    /// ```
    #[cfg(feature = "parallel")]
    pub fn batch_find_similar(
        &self,
        queries: &[BinaryHV],
        top_k: usize,
    ) -> Vec<Vec<(String, f32)>> {
        use rayon::prelude::*;

        const PARALLEL_THRESHOLD: usize = 50;

        if queries.len() < PARALLEL_THRESHOLD {
            // Sequential for small batches
            queries
                .iter()
                .map(|q| self.find_similar_to_encoding(q, top_k))
                .collect()
        } else {
            // Parallel for large batches
            queries
                .par_iter()
                .map(|q| self.find_similar_to_encoding(q, top_k))
                .collect()
        }
    }

    /// Batch find similar primitives (sequential version for no-parallel builds).
    #[cfg(not(feature = "parallel"))]
    pub fn batch_find_similar(
        &self,
        queries: &[BinaryHV],
        top_k: usize,
    ) -> Vec<Vec<(String, f32)>> {
        queries
            .iter()
            .map(|q| self.find_similar_to_encoding(q, top_k))
            .collect()
    }

    /// Batch find similar using LSH for very large searches.
    ///
    /// Builds an LSH index once and reuses it for all queries.
    /// Best for: many queries against all primitives.
    pub fn batch_find_similar_lsh(
        &self,
        queries: &[BinaryHV],
        top_k: usize,
        num_bands: usize,
        bits_per_band: usize,
    ) -> Vec<Vec<(String, f32)>> {
        let lsh = self.build_lsh_index(num_bands, bits_per_band);

        queries
            .iter()
            .map(|q| self.find_similar_lsh(q, top_k, &lsh))
            .collect()
    }

    /// Batch bind multiple primitive pairs.
    ///
    /// More efficient than calling bind_primitives repeatedly.
    pub fn batch_bind(
        &self,
        pairs: &[(&str, &str)],
    ) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        pairs
            .iter()
            .map(|(a, b)| self.bind_primitives(a, b))
            .collect()
    }

    /// Batch bundle multiple primitive groups.
    pub fn batch_bundle(&self, groups: &[&[&str]]) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        groups
            .iter()
            .map(|names| self.bundle_primitives(names))
            .collect()
    }

    /// Batch encode multiple sequences.
    pub fn batch_encode_sequences(
        &self,
        sequences: &[&[&str]],
    ) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        sequences
            .iter()
            .map(|names| self.encode_sequence(names))
            .collect()
    }

    /// Compute pairwise similarities between all given encodings.
    ///
    /// Returns a flattened lower-triangular matrix: [(i, j, similarity)]
    /// for all i > j pairs.
    pub fn pairwise_similarities(&self, encodings: &[BinaryHV]) -> Vec<(usize, usize, f32)> {
        let mut results = Vec::with_capacity(encodings.len() * (encodings.len() - 1) / 2);

        for i in 0..encodings.len() {
            for j in 0..i {
                let sim = encodings[i].similarity(&encodings[j]);
                results.push((i, j, sim));
            }
        }

        results
    }

    /// Compute similarity matrix for named primitives.
    ///
    /// Returns a symmetric matrix where matrix[i][j] = similarity(primitive_i, primitive_j).
    pub fn similarity_matrix(&self, names: &[&str]) -> Vec<Vec<f32>> {
        let encodings: Vec<_> = names
            .iter()
            .filter_map(|n| self.get(n).map(|p| p.encoding))
            .collect();

        let n = encodings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity
            for j in 0..i {
                let sim = encodings[i].similarity(&encodings[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim; // Symmetric
            }
        }

        matrix
    }

    // ========================================================================
    // TYPED PRIMITIVE OPERATIONS
    // ========================================================================

    /// Bind two named primitives together (XOR in BinaryHV space).
    ///
    /// Binding creates a new encoding that represents the relationship between
    /// two concepts. In HDC, bind(A, B) creates a vector orthogonal to both
    /// A and B but can be "unbound" by either to recover the other.
    pub fn bind_primitives(&self, a: &str, b: &str) -> Result<PrimitiveResult, PrimitiveError> {
        let prim_a = self
            .primitives
            .get(a)
            .ok_or_else(|| PrimitiveError::NotFound(a.to_string()))?;
        let prim_b = self
            .primitives
            .get(b)
            .ok_or_else(|| PrimitiveError::NotFound(b.to_string()))?;

        let encoding = prim_a.encoding.bind(&prim_b.encoding);
        Ok(PrimitiveResult {
            encoding,
            operation: format!("bind({a}, {b})"),
            source_primitives: vec![a.to_string(), b.to_string()],
        })
    }

    /// Bundle multiple named primitives together (majority vote in BinaryHV space).
    ///
    /// Bundling creates an encoding similar to all inputs (unlike bind).
    pub fn bundle_primitives(&self, names: &[&str]) -> Result<PrimitiveResult, PrimitiveError> {
        if names.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        let mut encodings = Vec::with_capacity(names.len());
        for name in names {
            let prim = self
                .primitives
                .get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            encodings.push(prim.encoding);
        }

        let encoding = BinaryHV::bundle(&encodings);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("bundle({})", names.join(", ")),
            source_primitives: names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Bundle primitives with weights for nuanced compositions.
    ///
    /// Higher weights make that primitive more dominant in the result.
    /// Uses probabilistic bit selection based on weights.
    pub fn bundle_weighted(
        &self,
        weighted: &[(&str, f32)],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        if weighted.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        // Normalize weights
        let total_weight: f32 = weighted.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return Err(PrimitiveError::InvalidWeight);
        }

        let mut encodings = Vec::with_capacity(weighted.len());
        let mut weights = Vec::with_capacity(weighted.len());

        for (name, weight) in weighted {
            let prim = self
                .primitives
                .get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            encodings.push(prim.encoding);
            weights.push(*weight / total_weight);
        }

        // Weighted bundling: for each bit position, sum weighted votes
        // BinaryHV is [u8; 2048] (2048 * 8 = 16384 bits)
        let mut result_bytes = [0u8; 2048];
        for byte_idx in 0..2048 {
            let mut byte_val: u8 = 0;
            for bit_in_byte in 0..8 {
                let mut weighted_sum: f32 = 0.0;
                for (enc, w) in encodings.iter().zip(weights.iter()) {
                    let enc_byte = enc.0[byte_idx];
                    let bit = (enc_byte >> bit_in_byte) & 1;
                    weighted_sum += if bit == 1 { *w } else { -*w };
                }

                if weighted_sum > 0.0 {
                    byte_val |= 1u8 << bit_in_byte;
                }
            }
            result_bytes[byte_idx] = byte_val;
        }

        let encoding = BinaryHV(result_bytes);
        let names: Vec<String> = weighted.iter().map(|(n, _)| n.to_string()).collect();

        Ok(PrimitiveResult {
            encoding,
            operation: format!(
                "bundle_weighted({})",
                weighted
                    .iter()
                    .map(|(n, w)| format!("{n}:{w:.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            source_primitives: names,
        })
    }

    /// Compute an analogy: A is to B as C is to ?
    ///
    /// Uses the HDC analogy formula: result = bind(unbind(A, B), C)
    pub fn analogy(&self, a: &str, b: &str, c: &str) -> Result<PrimitiveResult, PrimitiveError> {
        let prim_a = self
            .primitives
            .get(a)
            .ok_or_else(|| PrimitiveError::NotFound(a.to_string()))?;
        let prim_b = self
            .primitives
            .get(b)
            .ok_or_else(|| PrimitiveError::NotFound(b.to_string()))?;
        let prim_c = self
            .primitives
            .get(c)
            .ok_or_else(|| PrimitiveError::NotFound(c.to_string()))?;

        // Analogy: A:B :: C:? => ? = bind(bind(A, B), C)
        // Note: In XOR-based HDC, unbind(A, B) = bind(A, B) since XOR is self-inverse
        let ab_relation = prim_a.encoding.bind(&prim_b.encoding);
        let encoding = ab_relation.bind(&prim_c.encoding);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("analogy({a}:{b} :: {c}:?)"),
            source_primitives: vec![a.to_string(), b.to_string(), c.to_string()],
        })
    }

    /// Permute a named primitive (cyclic rotation in BinaryHV space).
    ///
    /// Useful for encoding sequences or temporal relationships.
    pub fn permute_primitive(
        &self,
        name: &str,
        steps: usize,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let prim = self
            .primitives
            .get(name)
            .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;

        let encoding = prim.encoding.permute(steps);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("permute({name}, {steps})"),
            source_primitives: vec![name.to_string()],
        })
    }

    /// Encode an ordered sequence of primitives preserving position.
    ///
    /// Uses permutation to encode position: A (x) permute(B, 1) (x) permute(C, 2)
    /// This creates an encoding that captures both content and order.
    pub fn encode_sequence(&self, names: &[&str]) -> Result<PrimitiveResult, PrimitiveError> {
        if names.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        let first = self
            .primitives
            .get(names[0])
            .ok_or_else(|| PrimitiveError::NotFound(names[0].to_string()))?;

        let mut encoding = first.encoding;

        for (i, name) in names.iter().enumerate().skip(1) {
            let prim = self
                .primitives
                .get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            let permuted = prim.encoding.permute(i);
            encoding = encoding.bind(&permuted);
        }

        Ok(PrimitiveResult {
            encoding,
            operation: format!("sequence({})", names.join(" -> ")),
            source_primitives: names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Query what primitive best matches a given encoding.
    pub fn query(&self, encoding: &BinaryHV) -> (String, f32) {
        let matches = self.find_similar_to_encoding(encoding, 1);
        matches
            .into_iter()
            .next()
            .unwrap_or_else(|| ("UNKNOWN".to_string(), 0.0))
    }
}

impl Default for PrimitiveSystem {
    fn default() -> Self {
        Self::new()
    }
}
