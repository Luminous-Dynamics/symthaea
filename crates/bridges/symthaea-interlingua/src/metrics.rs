// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measurement helpers for SCIP fidelity and wire-efficiency experiments.
//!
//! These metrics deliberately report bytes and semantic fidelity rather than
//! claiming token savings. Token counts depend on the concrete LLM tokenizer
//! and must be measured by each model adapter.

use crate::{
    GroundedHdcCodec, HdcPayload, InterchangeError, SparseHdcDelta, canonical_graph_bytes,
};
use serde::{Deserialize, Serialize};
use symthaea_communication::GroundedConceptGraph;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionMetrics {
    pub semantic_hash_matches: bool,
    pub profile_matches: bool,
    pub cosine_similarity: f32,
    pub canonical_graph_bytes: usize,
    pub dense_hdc_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeltaMetrics {
    pub dense_hdc_bytes: usize,
    pub estimated_delta_bytes: usize,
    pub changed_components: usize,
    pub dimension: usize,
    pub changed_fraction: f32,
}

impl DeltaMetrics {
    pub fn byte_ratio(&self) -> f32 {
        if self.dense_hdc_bytes == 0 {
            0.0
        } else {
            self.estimated_delta_bytes as f32 / self.dense_hdc_bytes as f32
        }
    }
}

pub fn measure_projection(
    codec: &GroundedHdcCodec,
    graph: &GroundedConceptGraph,
    payload: &HdcPayload,
) -> Result<ProjectionMetrics, InterchangeError> {
    let verification = codec.verify_projection(graph, payload)?;
    Ok(ProjectionMetrics {
        semantic_hash_matches: verification.semantic_hash_matches,
        profile_matches: verification.profile_matches,
        cosine_similarity: verification.cosine_similarity,
        canonical_graph_bytes: canonical_graph_bytes(graph)?.len(),
        dense_hdc_bytes: payload.values.len() * std::mem::size_of::<f32>(),
    })
}

pub fn measure_delta(delta: &SparseHdcDelta) -> DeltaMetrics {
    let dense_hdc_bytes = delta.dimension * std::mem::size_of::<f32>();
    // Each sparse component carries a u32 index + f32 delta. Metadata is
    // deliberately excluded so comparisons remain representation-independent;
    // callers can add their transport/frame overhead separately.
    let estimated_delta_bytes =
        delta.changes.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>());

    DeltaMetrics {
        dense_hdc_bytes,
        estimated_delta_bytes,
        changed_components: delta.changes.len(),
        dimension: delta.dimension,
        changed_fraction: delta.changed_fraction(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SCIP_HDC_NAMESPACE_V1, SparseHdcDelta};
    use symthaea_communication::{ConceptKind, ConceptNode};

    fn graph(label: &str) -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![ConceptNode {
                id: "concept".into(),
                kind: ConceptKind::Object,
                label: Some(label.into()),
                grounded_by: vec!["obs".into()],
                confidence: 0.9,
            }],
            edges: vec![],
        }
    }

    #[test]
    fn metrics_report_perfect_self_projection_fidelity() {
        let codec = GroundedHdcCodec::new(1024, SCIP_HDC_NAMESPACE_V1);
        let graph = graph("reactor");
        let payload = codec.encode_graph(&graph).unwrap();
        let metrics = measure_projection(&codec, &graph, &payload).unwrap();

        assert!(metrics.semantic_hash_matches);
        assert!(metrics.profile_matches);
        assert!(metrics.cosine_similarity > 0.9999);
        assert_eq!(metrics.dense_hdc_bytes, 4096);
        assert!(metrics.canonical_graph_bytes > 0);
    }

    #[test]
    fn sparse_delta_metrics_do_not_assume_compression() {
        let codec = GroundedHdcCodec::new(256, SCIP_HDC_NAMESPACE_V1);
        let base = codec.encode_graph(&graph("reactor")).unwrap();
        let target = codec.encode_graph(&graph("pump")).unwrap();
        let delta = SparseHdcDelta::between(&base, &target, 0.0).unwrap();
        let metrics = measure_delta(&delta);

        assert_eq!(metrics.dimension, 256);
        assert_eq!(metrics.dense_hdc_bytes, 1024);
        assert!((0.0..=1.0).contains(&metrics.changed_fraction));
        // A semantic change can touch most dimensions. SCIP reports the result
        // rather than asserting that a delta must be smaller.
        assert!(metrics.estimated_delta_bytes <= metrics.changed_components * 8);
    }
}
