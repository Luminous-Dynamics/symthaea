// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Knowledge Retrieval — Consciousness-Weighted Queries
//!
//! Implements PathHD-style hyperdimensional knowledge graph retrieval
//! where consciousness state (thought hypervector) is bound with claim
//! vectors to produce consciousness-weighted queries.
//!
//! ## Architecture
//!
//! Facts in the knowledge graph are encoded as 16,384D BinaryHV.
//! Queries bind a thought vector with a query vector using HDC's
//! non-commutative binding (permute between hops for multi-hop paths).
//!
//! ## Science
//!
//! - PathHD (arXiv:2512.09369): Encoder-free KG reasoning with HDC
//! - HDReason (arXiv:2403.05763): 10.6x speedup for KG completion via HDC
//! - Kanerva (2009): Hyperdimensional computing for cognitive architectures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use symthaea_core::hdc::unified_hv::{BinaryHV, HDC_DIMENSION};

/// A fact indexed in the HDC knowledge space.
#[derive(Debug, Clone)]
pub struct IndexedFact {
    /// Unique fact identifier
    pub fact_id: u64,
    /// HDC encoding of the fact content
    pub encoding: BinaryHV,
    /// Domain tag for filtering
    pub domain: String,
    /// Original confidence when indexed
    pub confidence: f32,
}

/// Result of an HDC retrieval query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Fact identifier
    pub fact_id: u64,
    /// Cosine similarity to query (0.0-1.0)
    pub similarity: f32,
    /// Domain of the fact
    pub domain: String,
    /// Original confidence
    pub confidence: f32,
}

/// HDC-based knowledge index for consciousness-weighted retrieval.
///
/// Stores fact encodings as BinaryHV and supports binding-based queries
/// that weight results by the current conscious state.
#[derive(Debug)]
pub struct HdcKnowledgeIndex {
    /// Indexed facts
    facts: Vec<IndexedFact>,
    /// Max capacity
    capacity: usize,
    /// Domain-level indices for fast filtering
    domain_index: HashMap<String, Vec<usize>>,
}

/// Named constant: default index capacity
const DEFAULT_INDEX_CAPACITY: usize = 4096;

/// Named constant: minimum similarity to include in results
const MIN_RETRIEVAL_SIMILARITY: f32 = 0.1;

impl HdcKnowledgeIndex {
    pub fn new() -> Self {
        Self {
            facts: Vec::with_capacity(256),
            capacity: DEFAULT_INDEX_CAPACITY,
            domain_index: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            facts: Vec::with_capacity(capacity.min(1024)),
            capacity,
            domain_index: HashMap::new(),
        }
    }

    /// Index a new fact.
    pub fn insert(&mut self, fact_id: u64, encoding: BinaryHV, domain: &str, confidence: f32) {
        // Evict oldest if at capacity
        if self.facts.len() >= self.capacity {
            let removed = self.facts.remove(0);
            // Clean up domain index
            if let Some(indices) = self.domain_index.get_mut(&removed.domain) {
                indices.retain(|&i| i != 0);
                // Shift all indices down by 1
                for idx in indices.iter_mut() {
                    *idx = idx.saturating_sub(1);
                }
            }
        }

        let idx = self.facts.len();
        self.domain_index
            .entry(domain.to_string())
            .or_default()
            .push(idx);

        self.facts.push(IndexedFact {
            fact_id,
            encoding,
            domain: domain.to_string(),
            confidence,
        });
    }

    /// Bind a thought vector with a claim vector for consciousness-weighted query.
    ///
    /// The binding operation uses XOR (standard BSC binding) to create a query
    /// that retrieves facts related to BOTH the current thought and the claim.
    pub fn bind_query(thought_hv: &BinaryHV, claim_hv: &BinaryHV) -> BinaryHV {
        thought_hv.bind(claim_hv)
    }

    /// Multi-hop query: permute between hops for path-aware retrieval.
    ///
    /// For a 2-hop path A→B→C: bind(A, permute(B))
    /// Non-commutative binding preserves relational structure (PathHD).
    pub fn multi_hop_query(hops: &[&BinaryHV]) -> Option<BinaryHV> {
        if hops.is_empty() {
            return None;
        }
        let mut result = hops[0].clone();
        for (i, hop) in hops.iter().enumerate().skip(1) {
            // Permute by hop index to make binding non-commutative
            let permuted = hop.permute(i);
            result = result.bind(&permuted);
        }
        Some(result)
    }

    /// Retrieve top-k facts most similar to the query vector.
    pub fn retrieve(&self, query_hv: &BinaryHV, k: usize) -> Vec<RetrievalResult> {
        let mut results: Vec<RetrievalResult> = self
            .facts
            .iter()
            .map(|fact| {
                let similarity = query_hv.similarity(&fact.encoding);
                RetrievalResult {
                    fact_id: fact.fact_id,
                    similarity,
                    domain: fact.domain.clone(),
                    confidence: fact.confidence,
                }
            })
            .filter(|r| r.similarity >= MIN_RETRIEVAL_SIMILARITY)
            .collect();

        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results.truncate(k);
        results
    }

    /// Retrieve within a specific domain only.
    pub fn retrieve_in_domain(
        &self,
        query_hv: &BinaryHV,
        domain: &str,
        k: usize,
    ) -> Vec<RetrievalResult> {
        let indices = match self.domain_index.get(domain) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let mut results: Vec<RetrievalResult> = indices
            .iter()
            .filter_map(|&i| self.facts.get(i))
            .map(|fact| {
                let similarity = query_hv.similarity(&fact.encoding);
                RetrievalResult {
                    fact_id: fact.fact_id,
                    similarity,
                    domain: fact.domain.clone(),
                    confidence: fact.confidence,
                }
            })
            .filter(|r| r.similarity >= MIN_RETRIEVAL_SIMILARITY)
            .collect();

        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results.truncate(k);
        results
    }

    /// Consciousness-weighted retrieval: bind thought state with query,
    /// then retrieve. Results reflect both what's relevant to the query
    /// AND what the current conscious state makes salient.
    pub fn retrieve_by_consciousness(
        &self,
        thought_hv: &BinaryHV,
        query_hv: &BinaryHV,
        k: usize,
    ) -> Vec<RetrievalResult> {
        let bound = Self::bind_query(thought_hv, query_hv);
        self.retrieve(&bound, k)
    }

    /// Number of indexed facts.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

impl Default for HdcKnowledgeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(42);

    fn random_hv() -> BinaryHV {
        let seed = SEED.fetch_add(1, Ordering::Relaxed);
        BinaryHV::random(seed)
    }

    #[test]
    fn test_insert_and_retrieve() {
        let mut index = HdcKnowledgeIndex::new();
        let fact_hv = random_hv();
        index.insert(1, fact_hv.clone(), "science", 0.9);

        // Query with the same HV should find it
        let results = index.retrieve(&fact_hv, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].fact_id, 1);
        assert!(results[0].similarity > 0.5);
    }

    #[test]
    fn test_binding_changes_query() {
        let thought = random_hv();
        let claim = random_hv();
        let bound = HdcKnowledgeIndex::bind_query(&thought, &claim);

        // Bound vector should be different from both inputs
        let sim_thought = bound.cosine_similarity(&thought);
        let sim_claim = bound.cosine_similarity(&claim);
        assert!(
            sim_thought < 0.3,
            "bound should differ from thought: {sim_thought}"
        );
        assert!(
            sim_claim < 0.3,
            "bound should differ from claim: {sim_claim}"
        );
    }

    #[test]
    fn test_multi_hop_non_commutative() {
        let a = random_hv();
        let b = random_hv();

        let ab = HdcKnowledgeIndex::multi_hop_query(&[&a, &b]).unwrap();
        let ba = HdcKnowledgeIndex::multi_hop_query(&[&b, &a]).unwrap();

        // Non-commutative: A→B != B→A
        let sim = ab.cosine_similarity(&ba);
        assert!(sim < 0.5, "multi-hop should be non-commutative: {sim}");
    }

    #[test]
    fn test_domain_filtering() {
        let mut index = HdcKnowledgeIndex::new();
        index.insert(1, random_hv(), "physics", 0.9);
        index.insert(2, random_hv(), "biology", 0.8);
        index.insert(3, random_hv(), "physics", 0.7);

        let query = random_hv();
        let physics = index.retrieve_in_domain(&query, "physics", 10);
        let biology = index.retrieve_in_domain(&query, "biology", 10);

        // Should only return facts from the requested domain
        assert!(physics.iter().all(|r| r.domain == "physics"));
        assert!(biology.iter().all(|r| r.domain == "biology"));
    }

    #[test]
    fn test_capacity_eviction() {
        let mut index = HdcKnowledgeIndex::with_capacity(3);
        for i in 0..5 {
            index.insert(i, random_hv(), "test", 0.5);
        }
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_empty_index() {
        let index = HdcKnowledgeIndex::new();
        assert!(index.is_empty());
        let results = index.retrieve(&random_hv(), 5);
        assert!(results.is_empty());
    }
}
