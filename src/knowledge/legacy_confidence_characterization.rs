// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only characterization of legacy knowledge-graph confidence authority.
//!
//! The EKM evidence pipeline deliberately separates observation, evidence admission,
//! evidence mutation, and later belief revision. The pre-existing
//! `EnhancedKnowledgeGraph` predates that separation and contains several direct
//! confidence mutation paths. This module measures those paths on disposable local
//! graphs so migration can be preregistered rather than inferred from code review.
//!
//! Nothing here changes the live `KnowledgeManager`, disables legacy behavior, or
//! grants the EKM ledger authority over the legacy graph.

use super::encoding::FactEncoding;
use super::graph::{ContradictionAlert, EnhancedKnowledgeGraph};
use std::collections::HashMap;
use symthaea_core::hdc::unified_hv::BinaryHV;

#[derive(Debug, Clone, PartialEq)]
pub struct LegacyConfidenceAuthorityProfile {
    /// Confidence increase caused by inserting an HDC-identical fact again.
    pub similarity_corroboration_delta: f32,
    /// Whether the second insertion reused the original FactId.
    pub similarity_corroboration_reused_fact: bool,
    /// Corroboration counter after the duplicate insertion.
    pub similarity_corroboration_count: u32,
    /// Whether ordinary retrieval refreshed `last_accessed_cycle`.
    pub retrieval_refreshes_last_accessed: bool,
    pub retrieval_last_accessed_before: u64,
    pub retrieval_last_accessed_after: u64,
    /// Multiplicative confidence ratio applied to the weaker fact by the legacy
    /// contradiction resolver in the characterization fixture.
    pub contradiction_resolution_ratio: f32,
    /// Observed change produced by the public direct confidence-adjustment API.
    pub direct_adjustment_delta: f32,
    /// Confidence increase produced by topic-based dream consolidation after a
    /// fact was first placed below its initial confidence.
    pub dream_topic_strengthening_delta: f32,
    /// Confidence increase produced by generic causal-fact strengthening after a
    /// causal fact was first placed below its initial confidence.
    pub causal_strengthening_delta: f32,
}

impl LegacyConfidenceAuthorityProfile {
    /// True when every authority channel being characterized is observably active.
    ///
    /// This is a characterization predicate, not a scientific PASS condition.
    pub fn all_characterized_channels_active(&self) -> bool {
        self.similarity_corroboration_delta > 0.0
            && self.similarity_corroboration_reused_fact
            && self.similarity_corroboration_count > 0
            && self.retrieval_refreshes_last_accessed
            && self.contradiction_resolution_ratio < 1.0
            && self.direct_adjustment_delta > 0.0
            && self.dream_topic_strengthening_delta > 0.0
            && self.causal_strengthening_delta > 0.0
    }
}

fn encoding(seed: u64, text: &str, confidence: f32) -> FactEncoding {
    FactEncoding {
        vector: BinaryHV::random(seed),
        role_vectors: HashMap::new(),
        source_text: text.to_string(),
        confidence,
    }
}

/// Run deterministic local fixtures that expose the legacy confidence mutation
/// surface. The returned values describe existing behavior only.
pub fn characterize_legacy_confidence_authority() -> LegacyConfidenceAuthorityProfile {
    // 1. Similarity-based corroboration directly raises confidence.
    let mut corroboration_graph = EnhancedKnowledgeGraph::new(16);
    let duplicate = encoding(7, "same information", 0.40);
    let duplicate_query = duplicate.vector.clone();
    let (first_id, _) = corroboration_graph.insert(duplicate.clone(), 1, None, false);
    let confidence_before_corroboration = corroboration_graph
        .get_fact(first_id)
        .expect("inserted fact")
        .confidence;
    let (second_id, _) = corroboration_graph.insert(duplicate, 2, None, false);
    let corroborated = corroboration_graph
        .get_fact(first_id)
        .expect("corroborated fact");
    let similarity_corroboration_delta =
        corroborated.confidence - confidence_before_corroboration;
    let similarity_corroboration_count = corroborated.corroboration_count;

    // 2. Ordinary retrieval refreshes recency, which later changes decay rate.
    let retrieval_last_accessed_before = corroboration_graph
        .get_fact(first_id)
        .expect("fact before retrieval")
        .last_accessed_cycle;
    let _ = corroboration_graph.search(&duplicate_query, 1, 50);
    let retrieval_last_accessed_after = corroboration_graph
        .get_fact(first_id)
        .expect("fact after retrieval")
        .last_accessed_cycle;

    // 3. Contradiction resolution contracts the currently weaker fact directly.
    let mut contradiction_graph = EnhancedKnowledgeGraph::new(16);
    let (strong_id, _) = contradiction_graph.insert(
        encoding(100, "strong fact", 0.80),
        1,
        None,
        false,
    );
    let (weak_id, _) = contradiction_graph.insert(
        encoding(200, "weak fact", 0.60),
        1,
        None,
        false,
    );
    let weak_before = contradiction_graph
        .get_fact(weak_id)
        .expect("weak fact")
        .confidence;
    let alert = ContradictionAlert {
        new_fact_id: weak_id,
        existing_fact_id: strong_id,
        similarity: 1.0,
        higher_confidence_id: strong_id,
        detected_at_cycle: 2,
    };
    contradiction_graph.resolve_contradictions(&[alert]);
    let weak_after = contradiction_graph
        .get_fact(weak_id)
        .expect("weak fact survives fixture")
        .confidence;
    let contradiction_resolution_ratio = weak_after / weak_before;

    // 4. Public direct adjustment can change confidence without an evidence object.
    let mut adjustment_graph = EnhancedKnowledgeGraph::new(16);
    let (adjust_id, _) = adjustment_graph.insert(
        encoding(300, "adjustable fact", 0.30),
        1,
        None,
        false,
    );
    let direct_before = adjustment_graph
        .get_fact(adjust_id)
        .expect("adjustable fact")
        .confidence;
    adjustment_graph.adjust_confidence(adjust_id, 0.20);
    let direct_after = adjustment_graph
        .get_fact(adjust_id)
        .expect("adjusted fact")
        .confidence;

    // 5. Dream/topic replay can strengthen a fact after confidence was lowered.
    let mut dream_graph = EnhancedKnowledgeGraph::new(16);
    let (dream_id, _) = dream_graph.insert(
        encoding(400, "dream topic fact", 0.80),
        1,
        None,
        false,
    );
    dream_graph.adjust_confidence(dream_id, -0.30);
    let dream_before = dream_graph
        .get_fact(dream_id)
        .expect("dream fact")
        .confidence;
    let strengthened = dream_graph.strengthen_facts_by_topic(&["dream".into()], 0.10);
    debug_assert_eq!(strengthened, 1);
    let dream_after = dream_graph
        .get_fact(dream_id)
        .expect("dream-strengthened fact")
        .confidence;

    // 6. Merely being marked causal allows a separate consolidation boost.
    let mut causal_graph = EnhancedKnowledgeGraph::new(16);
    let (causal_id, _) = causal_graph.insert(
        encoding(500, "causal fact", 0.80),
        1,
        None,
        true,
    );
    causal_graph.adjust_confidence(causal_id, -0.30);
    let causal_before = causal_graph
        .get_fact(causal_id)
        .expect("causal fact")
        .confidence;
    let strengthened = causal_graph.strengthen_causal_facts(0.10);
    debug_assert_eq!(strengthened, 1);
    let causal_after = causal_graph
        .get_fact(causal_id)
        .expect("strengthened causal fact")
        .confidence;

    LegacyConfidenceAuthorityProfile {
        similarity_corroboration_delta,
        similarity_corroboration_reused_fact: first_id == second_id,
        similarity_corroboration_count,
        retrieval_refreshes_last_accessed:
            retrieval_last_accessed_after > retrieval_last_accessed_before,
        retrieval_last_accessed_before,
        retrieval_last_accessed_after,
        contradiction_resolution_ratio,
        direct_adjustment_delta: direct_after - direct_before,
        dream_topic_strengthening_delta: dream_after - dream_before,
        causal_strengthening_delta: causal_after - causal_before,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_confidence_authority_surface_is_explicitly_characterized() {
        let profile = characterize_legacy_confidence_authority();
        assert!(profile.all_characterized_channels_active(), "{profile:?}");
        assert!((profile.similarity_corroboration_delta - 0.10).abs() < 1e-6);
        assert!(profile.similarity_corroboration_reused_fact);
        assert_eq!(profile.similarity_corroboration_count, 1);
        assert_eq!(profile.retrieval_last_accessed_before, 2);
        assert_eq!(profile.retrieval_last_accessed_after, 50);
        assert!((profile.contradiction_resolution_ratio - 0.5).abs() < 1e-6);
        assert!((profile.direct_adjustment_delta - 0.20).abs() < 1e-6);
        assert!((profile.dream_topic_strengthening_delta - 0.10).abs() < 1e-6);
        assert!((profile.causal_strengthening_delta - 0.10).abs() < 1e-6);
    }

    #[test]
    fn characterization_is_deterministic() {
        assert_eq!(
            characterize_legacy_confidence_authority(),
            characterize_legacy_confidence_authority()
        );
    }
}
