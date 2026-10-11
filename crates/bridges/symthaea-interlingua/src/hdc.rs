// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! GroundedConceptGraph -> HDC projection for SCIP.
//!
//! The graph remains canonical. This module produces an associative projection
//! that can be compared, bundled, retrieved, or passed to a model adapter. The
//! projection is accepted as grounded only when its semantic hash resolves to a
//! grounded graph known to the receiver.

use crate::protocol::{
    CognitiveEnvelope, HdcPayload, HdcProfile, InterchangeError, InterchangePayload,
    SemanticProfile, graph_semantic_hash, validate_graph,
};
use symthaea_communication::{ConceptKind, GroundedConceptGraph, Provenance};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

pub const SCIP_HDC_NAMESPACE_V1: &str = "symthaea.scip.v1";
pub const SCIP_HDC_ALGEBRA_V1: &str = "continuous-hadamard+bipolar-atoms+mean-bundle/v1";
pub const SCIP_HDC_ATOM_DERIVATION_V1: &str = "blake3-xof-bipolar/v1";

#[derive(Clone, Debug)]
pub struct GroundedHdcCodec {
    profile: HdcProfile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionVerification {
    pub semantic_hash_matches: bool,
    pub profile_matches: bool,
    pub cosine_similarity: f32,
}

impl ProjectionVerification {
    pub fn is_valid(self, minimum_similarity: f32) -> bool {
        self.semantic_hash_matches
            && self.profile_matches
            && self.cosine_similarity >= minimum_similarity
    }
}

impl Default for GroundedHdcCodec {
    fn default() -> Self {
        Self::standard()
    }
}

impl GroundedHdcCodec {
    pub fn standard() -> Self {
        Self::new(HDC_DIMENSION, SCIP_HDC_NAMESPACE_V1)
    }

    pub fn new(dimension: usize, namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        let fingerprint = profile_fingerprint(dimension, &namespace);
        Self {
            profile: HdcProfile {
                dimension,
                algebra: SCIP_HDC_ALGEBRA_V1.into(),
                atom_derivation: SCIP_HDC_ATOM_DERIVATION_V1.into(),
                namespace,
                codebook_fingerprint: fingerprint,
            },
        }
    }

    pub fn profile(&self) -> &HdcProfile {
        &self.profile
    }

    pub fn encode_graph(
        &self,
        graph: &GroundedConceptGraph,
    ) -> Result<HdcPayload, InterchangeError> {
        validate_graph(graph)?;
        if self.profile.dimension == 0 {
            return Err(InterchangeError::InvalidHdcPayload);
        }

        let mut graph = graph.clone();
        graph.nodes.sort_by(|a, b| a.id.cmp(&b.id));
        graph.edges.sort_by(|a, b| {
            (&a.source, &a.relation, &a.target).cmp(&(&b.source, &b.relation, &b.target))
        });

        let mut records = Vec::with_capacity(graph.nodes.len() + graph.edges.len());

        for node in &graph.nodes {
            let mut fields = vec![
                self.field("record/type", "node"),
                self.field("node/id", &node.id),
                self.field("node/kind", concept_kind_name(&node.kind)),
                self.field("node/confidence", &confidence_bucket(node.confidence)),
            ];
            if let Some(label) = &node.label {
                fields.push(self.field("node/label", label));
            }

            let mut grounding = node.grounded_by.clone();
            grounding.sort();
            fields.extend(
                grounding
                    .iter()
                    .map(|value| self.field("node/grounded-by", value)),
            );
            records.push(mean_bundle(&fields, self.profile.dimension));
        }

        for edge in &graph.edges {
            let mut fields = vec![
                self.field("record/type", "edge"),
                self.field("edge/source", &edge.source),
                self.field("edge/relation", &edge.relation),
                self.field("edge/target", &edge.target),
                self.field("edge/confidence", &confidence_bucket(edge.confidence)),
            ];
            let mut evidence = edge.evidence_ids.clone();
            evidence.sort();
            fields.extend(
                evidence
                    .iter()
                    .map(|value| self.field("edge/evidence", value)),
            );
            records.push(mean_bundle(&fields, self.profile.dimension));
        }

        let vector = mean_bundle(&records, self.profile.dimension);

        Ok(HdcPayload {
            values: vector.values,
            semantic_hash: graph_semantic_hash(&graph)?,
            profile_fingerprint: self.profile.codebook_fingerprint.clone(),
        })
    }

    pub fn envelope_from_graph(
        &self,
        graph: &GroundedConceptGraph,
        confidence: f32,
        provenance: Provenance,
    ) -> Result<CognitiveEnvelope, InterchangeError> {
        let payload = self.encode_graph(graph)?;
        CognitiveEnvelope::new(
            SemanticProfile::hdc(self.profile.clone()),
            InterchangePayload::Hdc(payload),
            confidence,
            provenance,
        )
    }

    pub fn verify_projection(
        &self,
        graph: &GroundedConceptGraph,
        payload: &HdcPayload,
    ) -> Result<ProjectionVerification, InterchangeError> {
        let expected = self.encode_graph(graph)?;
        if payload.values.len() != self.profile.dimension {
            return Err(InterchangeError::InvalidHdcPayload);
        }
        Ok(ProjectionVerification {
            semantic_hash_matches: payload.semantic_hash == expected.semantic_hash,
            profile_matches: payload.profile_fingerprint == self.profile.codebook_fingerprint,
            cosine_similarity: cosine_similarity(&expected.values, &payload.values),
        })
    }

    fn field(&self, role: &str, value: &str) -> ContinuousHV {
        self.atom("role", role).bind(&self.atom("value", value))
    }

    fn atom(&self, domain: &str, value: &str) -> ContinuousHV {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-scip-hdc-atom\0");
        hasher.update(self.profile.namespace.as_bytes());
        hasher.update(b"\0");
        hasher.update(domain.as_bytes());
        hasher.update(b"\0");
        hasher.update(value.as_bytes());

        let mut reader = hasher.finalize_xof();
        let mut bytes = vec![0u8; self.profile.dimension.div_ceil(8)];
        reader.fill(&mut bytes);

        let values = (0..self.profile.dimension)
            .map(|index| {
                let bit = (bytes[index / 8] >> (index % 8)) & 1;
                if bit == 0 { -1.0 } else { 1.0 }
            })
            .collect();
        ContinuousHV::from_vec(values)
    }
}

pub fn profile_fingerprint(dimension: usize, namespace: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-scip-hdc-profile\0");
    hasher.update(dimension.to_string().as_bytes());
    hasher.update(b"\0");
    hasher.update(SCIP_HDC_ALGEBRA_V1.as_bytes());
    hasher.update(b"\0");
    hasher.update(SCIP_HDC_ATOM_DERIVATION_V1.as_bytes());
    hasher.update(b"\0");
    hasher.update(namespace.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn mean_bundle(vectors: &[ContinuousHV], dimension: usize) -> ContinuousHV {
    if vectors.is_empty() {
        return ContinuousHV::zero(dimension);
    }

    let mut values = vec![0.0f32; dimension];
    for vector in vectors {
        debug_assert_eq!(vector.values.len(), dimension);
        for (accumulator, value) in values.iter_mut().zip(&vector.values) {
            *accumulator += *value;
        }
    }
    let inverse = 1.0 / vectors.len() as f32;
    for value in &mut values {
        *value *= inverse;
    }
    ContinuousHV::from_vec(values)
}

fn confidence_bucket(value: f32) -> String {
    ((value.clamp(0.0, 1.0) * 1000.0).round() as u16).to_string()
}

fn concept_kind_name(kind: &ConceptKind) -> &'static str {
    match kind {
        ConceptKind::Agent => "agent",
        ConceptKind::Object => "object",
        ConceptKind::Event => "event",
        ConceptKind::Action => "action",
        ConceptKind::State => "state",
        ConceptKind::Property => "property",
        ConceptKind::Relation => "relation",
        ConceptKind::Unknown => "unknown",
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&left, &right) in a.iter().zip(b) {
        let left = left as f64;
        let right = right as f64;
        dot += left * right;
        aa += left * left;
        bb += right * right;
    }
    let denominator = (aa * bb).sqrt();
    if denominator <= f64::EPSILON {
        if a == b { 1.0 } else { 0.0 }
    } else {
        (dot / denominator).clamp(-1.0, 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_communication::{ConceptEdge, ConceptNode};

    fn graph() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "alice".into(),
                    kind: ConceptKind::Agent,
                    label: Some("Alice".into()),
                    grounded_by: vec!["obs-1".into()],
                    confidence: 0.92,
                },
                ConceptNode {
                    id: "reactor-a".into(),
                    kind: ConceptKind::Object,
                    label: Some("Reactor A".into()),
                    grounded_by: vec!["obs-2".into()],
                    confidence: 0.97,
                },
            ],
            edges: vec![ConceptEdge {
                source: "alice".into(),
                relation: "believes-failure-risk".into(),
                target: "reactor-a".into(),
                evidence_ids: vec!["sensor-s17".into()],
                confidence: 0.82,
            }],
        }
    }

    #[test]
    fn standard_profile_is_stable() {
        let first = GroundedHdcCodec::standard();
        let second = GroundedHdcCodec::standard();
        assert_eq!(first.profile(), second.profile());
        assert_eq!(first.profile().dimension, HDC_DIMENSION);
    }

    #[test]
    fn projection_is_order_independent() {
        let codec = GroundedHdcCodec::new(1024, SCIP_HDC_NAMESPACE_V1);
        let first = graph();
        let mut second = first.clone();
        second.nodes.reverse();
        second.edges[0].evidence_ids.reverse();

        let a = codec.encode_graph(&first).unwrap();
        let b = codec.encode_graph(&second).unwrap();
        assert_eq!(a.semantic_hash, b.semantic_hash);
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn verification_binds_vector_to_grounded_hash() {
        let codec = GroundedHdcCodec::new(1024, SCIP_HDC_NAMESPACE_V1);
        let graph = graph();
        let payload = codec.encode_graph(&graph).unwrap();
        let verification = codec.verify_projection(&graph, &payload).unwrap();
        assert!(verification.is_valid(0.9999));

        let mut changed = graph;
        changed.edges[0].confidence = 0.25;
        let mismatch = codec.verify_projection(&changed, &payload).unwrap();
        assert!(!mismatch.semantic_hash_matches);
    }
}
