// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical content identity for validated evidence-lineage graphs.
//!
//! `EvidenceLineageGraphV1::graph_id` is a legacy producer-supplied wire label.
//! It is validated only for digest shape and MUST NOT be used as governance
//! identity. This module derives identity directly from typed validated graph
//! semantics and explicitly excludes serde/wire representation from authority.

use crate::lineage::{
    CognitiveDerivationKindV1, ValidatedEvidenceLineageGraphV1,
    COGNITIVE_LINEAGE_SCHEMA_VERSION,
};

pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_SCHEMA_VERSION: u16 = 1;
pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_PROFILE_V1: &str =
    "rca-canonical-evidence-lineage-identity-v1";

pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_CONTRACT_V1: &str = concat!(
    "rca-canonical-evidence-lineage-identity-v1\n",
    "input=validated_evidence_lineage_graph_v1\n",
    "identity_source=typed_validated_semantics_not_serde_wire_projection\n",
    "legacy_wire_graph_id_is_explicitly_excluded_from_governance_identity\n",
    "identity_fields=graph_schema+node_schema+evidence_id+sorted_parent_ids+explicit_derivation_kind_tag\n",
    "node_input_order_does_not_change_identity\n",
    "parent_input_order_does_not_change_identity\n",
    "unrelated_node_addition_changes_identity\n",
    "derivation_or_parent_change_changes_identity\n",
    "canonical_identity_derivation_is_infallible_after_validation\n",
    "identity=blake3_explicit_semantic_tree_v1\n",
    "canonical_lineage_identity_is_not_evidence_independence_or_downstream_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-canonical-evidence-lineage-identity-contract:v1\0";
const GRAPH_ID_DOMAIN: &[u8] = b"symthaea:rca-canonical-evidence-lineage-graph:v1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalNodeV1 {
    evidence_id: String,
    parent_ids: Vec<String>,
    derivation_kind: CognitiveDerivationKindV1,
}

pub fn canonical_evidence_lineage_identity_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        CANONICAL_EVIDENCE_LINEAGE_IDENTITY_CONTRACT_V1.as_bytes(),
    )
}

/// Derive a serializer-independent governance identity directly from one
/// validated lineage graph. The producer-supplied wire `graph_id` is ignored.
///
/// Validation has already established the graph/node schema, digest shapes,
/// closed ancestry, unique evidence ids, unique parents, and acyclicity. Identity
/// derivation therefore has no remaining fallible wire-projection step.
pub fn canonical_evidence_lineage_graph_id_v1(
    graph: &ValidatedEvidenceLineageGraphV1,
) -> String {
    let mut nodes = graph
        .nodes()
        .iter()
        .map(|node| {
            let mut parent_ids = node.parent_ids().to_vec();
            parent_ids.sort();
            CanonicalNodeV1 {
                evidence_id: node.evidence_id().to_string(),
                parent_ids,
                derivation_kind: node.derivation_kind(),
            }
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));

    let profile_contract_digest = canonical_evidence_lineage_identity_profile_digest_v1();
    let mut hasher = blake3::Hasher::new();
    hasher.update(GRAPH_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        &profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"identity_schema_version",
        &CANONICAL_EVIDENCE_LINEAGE_IDENTITY_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_bytes(
        &mut hasher,
        b"graph_schema_version",
        &COGNITIVE_LINEAGE_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_count(&mut hasher, b"node_count", nodes.len());
    for node in &nodes {
        hash_bytes(
            &mut hasher,
            b"node_schema_version",
            &COGNITIVE_LINEAGE_SCHEMA_VERSION.to_le_bytes(),
        );
        hash_text(&mut hasher, b"evidence_id", &node.evidence_id);
        hash_text(
            &mut hasher,
            b"derivation_kind",
            derivation_kind_tag(node.derivation_kind),
        );
        hash_count(&mut hasher, b"parent_count", node.parent_ids.len());
        for parent_id in &node.parent_ids {
            hash_text(&mut hasher, b"parent_id", parent_id);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn derivation_kind_tag(kind: CognitiveDerivationKindV1) -> &'static str {
    match kind {
        CognitiveDerivationKindV1::RootObservation => "root_observation",
        CognitiveDerivationKindV1::Retrieval => "retrieval",
        CognitiveDerivationKindV1::Transformation => "transformation",
        CognitiveDerivationKindV1::Inference => "inference",
        CognitiveDerivationKindV1::Simulation => "simulation",
        CognitiveDerivationKindV1::Summary => "summary",
        CognitiveDerivationKindV1::Critique => "critique",
        CognitiveDerivationKindV1::FormalDerivation => "formal_derivation",
        CognitiveDerivationKindV1::Other => "other",
    }
}

fn hash_count(hasher: &mut blake3::Hasher, label: &[u8], count: usize) {
    hash_bytes(hasher, label, &(count as u64).to_le_bytes());
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::{
        EvidenceLineageGraphV1, EvidenceLineageNodeV1, ValidatedEvidenceLineageNodeV1,
    };

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn node(
        id: &str,
        parents: &[&str],
        derivation_kind: CognitiveDerivationKindV1,
    ) -> ValidatedEvidenceLineageNodeV1 {
        EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: id.into(),
            parent_ids: parents.iter().map(|value| (*value).to_string()).collect(),
            derivation_kind,
        }
        .validate()
        .unwrap()
    }

    fn graph(
        legacy_graph_id: &str,
        nodes: Vec<ValidatedEvidenceLineageNodeV1>,
    ) -> ValidatedEvidenceLineageGraphV1 {
        EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: legacy_graph_id.into(),
            nodes,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn explicit_derivation_tags_are_stable() {
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::RootObservation),
            "root_observation"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Retrieval),
            "retrieval"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Transformation),
            "transformation"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Inference),
            "inference"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Simulation),
            "simulation"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Summary),
            "summary"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Critique),
            "critique"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::FormalDerivation),
            "formal_derivation"
        );
        assert_eq!(
            derivation_kind_tag(CognitiveDerivationKindV1::Other),
            "other"
        );
    }

    #[test]
    fn legacy_graph_label_does_not_define_canonical_identity() {
        let nodes = vec![
            node(A, &[], CognitiveDerivationKindV1::RootObservation),
            node(B, &[A], CognitiveDerivationKindV1::Inference),
        ];
        let first = graph(E, nodes.clone());
        let second = graph(F, nodes);
        assert_eq!(
            canonical_evidence_lineage_graph_id_v1(&first),
            canonical_evidence_lineage_graph_id_v1(&second)
        );
    }

    #[test]
    fn node_and_parent_order_do_not_change_identity() {
        let first = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[], CognitiveDerivationKindV1::RootObservation),
                node(C, &[A, B], CognitiveDerivationKindV1::Inference),
            ],
        );
        let second = graph(
            E,
            vec![
                node(C, &[B, A], CognitiveDerivationKindV1::Inference),
                node(B, &[], CognitiveDerivationKindV1::RootObservation),
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
            ],
        );
        assert_eq!(
            canonical_evidence_lineage_graph_id_v1(&first),
            canonical_evidence_lineage_graph_id_v1(&second)
        );
    }

    #[test]
    fn unrelated_node_addition_changes_identity() {
        let first = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[A], CognitiveDerivationKindV1::Inference),
            ],
        );
        let second = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[A], CognitiveDerivationKindV1::Inference),
                node(C, &[], CognitiveDerivationKindV1::RootObservation),
            ],
        );
        assert_ne!(
            canonical_evidence_lineage_graph_id_v1(&first),
            canonical_evidence_lineage_graph_id_v1(&second)
        );
    }

    #[test]
    fn derivation_change_changes_identity() {
        let first = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[A], CognitiveDerivationKindV1::Inference),
            ],
        );
        let second = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[A], CognitiveDerivationKindV1::Transformation),
            ],
        );
        assert_ne!(
            canonical_evidence_lineage_graph_id_v1(&first),
            canonical_evidence_lineage_graph_id_v1(&second)
        );
    }

    #[test]
    fn parent_change_changes_identity() {
        let first = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[], CognitiveDerivationKindV1::RootObservation),
                node(C, &[A], CognitiveDerivationKindV1::Inference),
            ],
        );
        let second = graph(
            F,
            vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[], CognitiveDerivationKindV1::RootObservation),
                node(C, &[B], CognitiveDerivationKindV1::Inference),
            ],
        );
        assert_ne!(
            canonical_evidence_lineage_graph_id_v1(&first),
            canonical_evidence_lineage_graph_id_v1(&second)
        );
    }
}
