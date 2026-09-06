// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical content identity for validated evidence-lineage graphs.
//!
//! `EvidenceLineageGraphV1::graph_id` is a legacy producer-supplied wire label.
//! It is validated only for digest shape and MUST NOT be used as governance
//! identity. This module derives identity from the validated graph contents while
//! explicitly excluding that legacy label.

use crate::lineage::ValidatedEvidenceLineageGraphV1;
use serde_json::Value;

pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_SCHEMA_VERSION: u16 = 1;
pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_PROFILE_V1: &str =
    "rca-canonical-evidence-lineage-identity-v1";

pub const CANONICAL_EVIDENCE_LINEAGE_IDENTITY_CONTRACT_V1: &str = concat!(
    "rca-canonical-evidence-lineage-identity-v1\n",
    "input=validated_evidence_lineage_graph_v1\n",
    "legacy_wire_graph_id_is_explicitly_excluded_from_governance_identity\n",
    "identity_fields=graph_schema+node_schema+evidence_id+sorted_parent_ids+explicit_derivation_kind_tag\n",
    "node_input_order_does_not_change_identity\n",
    "parent_input_order_does_not_change_identity\n",
    "unrelated_node_addition_changes_identity\n",
    "derivation_or_parent_change_changes_identity\n",
    "identity=blake3_explicit_semantic_tree_v1\n",
    "canonical_lineage_identity_is_not_evidence_independence_or_downstream_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-canonical-evidence-lineage-identity-contract:v1\0";
const GRAPH_ID_DOMAIN: &[u8] = b"symthaea:rca-canonical-evidence-lineage-graph:v1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalNodeV1 {
    schema_version: u16,
    evidence_id: String,
    parent_ids: Vec<String>,
    derivation_kind_tag: String,
}

pub fn canonical_evidence_lineage_identity_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        CANONICAL_EVIDENCE_LINEAGE_IDENTITY_CONTRACT_V1.as_bytes(),
    )
}

/// Derive a serializer-order-independent governance identity from a validated
/// lineage graph. `graph_id` from the wire object is deliberately ignored.
pub fn canonical_evidence_lineage_graph_id_v1(
    graph: &ValidatedEvidenceLineageGraphV1,
) -> Result<String, CanonicalLineageIdentityError> {
    let value = serde_json::to_value(graph)
        .map_err(|error| CanonicalLineageIdentityError::Serialization(error.to_string()))?;
    let object = value
        .as_object()
        .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape("graph_object"))?;

    let graph_schema = u16_field(object.get("schema_version"), "schema_version")?;
    let nodes_value = object
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape("nodes"))?;

    let mut nodes = Vec::with_capacity(nodes_value.len());
    for node_value in nodes_value {
        let node = node_value
            .as_object()
            .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape("node"))?;
        let schema_version = u16_field(node.get("schema_version"), "node.schema_version")?;
        let evidence_id = text_field(node.get("evidence_id"), "node.evidence_id")?.to_string();
        let derivation_kind_tag =
            text_field(node.get("derivation_kind"), "node.derivation_kind")?.to_string();
        validate_derivation_tag(&derivation_kind_tag)?;

        let parent_values = node
            .get("parent_ids")
            .and_then(Value::as_array)
            .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape(
                "node.parent_ids",
            ))?;
        let mut parent_ids = parent_values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape(
                        "node.parent_id",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        parent_ids.sort();

        nodes.push(CanonicalNodeV1 {
            schema_version,
            evidence_id,
            parent_ids,
            derivation_kind_tag,
        });
    }
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
        &graph_schema.to_le_bytes(),
    );
    hash_count(&mut hasher, b"node_count", nodes.len());
    for node in &nodes {
        hash_bytes(
            &mut hasher,
            b"node_schema_version",
            &node.schema_version.to_le_bytes(),
        );
        hash_text(&mut hasher, b"evidence_id", &node.evidence_id);
        hash_text(
            &mut hasher,
            b"derivation_kind",
            &node.derivation_kind_tag,
        );
        hash_count(&mut hasher, b"parent_count", node.parent_ids.len());
        for parent_id in &node.parent_ids {
            hash_text(&mut hasher, b"parent_id", parent_id);
        }
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn validate_derivation_tag(tag: &str) -> Result<(), CanonicalLineageIdentityError> {
    if matches!(
        tag,
        "root_observation"
            | "retrieval"
            | "transformation"
            | "inference"
            | "simulation"
            | "summary"
            | "critique"
            | "formal_derivation"
            | "other"
    ) {
        Ok(())
    } else {
        Err(CanonicalLineageIdentityError::UnexpectedDerivationTag(
            tag.to_string(),
        ))
    }
}

fn u16_field(value: Option<&Value>, field: &'static str) -> Result<u16, CanonicalLineageIdentityError> {
    let raw = value
        .and_then(Value::as_u64)
        .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape(field))?;
    u16::try_from(raw).map_err(|_| CanonicalLineageIdentityError::UnexpectedWireShape(field))
}

fn text_field<'a>(
    value: Option<&'a Value>,
    field: &'static str,
) -> Result<&'a str, CanonicalLineageIdentityError> {
    value
        .and_then(Value::as_str)
        .ok_or(CanonicalLineageIdentityError::UnexpectedWireShape(field))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalLineageIdentityError {
    Serialization(String),
    UnexpectedWireShape(&'static str),
    UnexpectedDerivationTag(String),
}

impl std::fmt::Display for CanonicalLineageIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => write!(f, "failed to project validated lineage graph: {error}"),
            Self::UnexpectedWireShape(field) => {
                write!(f, "validated lineage graph has unexpected wire shape at {field}")
            }
            Self::UnexpectedDerivationTag(tag) => {
                write!(f, "validated lineage graph has unknown derivation tag {tag}")
            }
        }
    }
}

impl std::error::Error for CanonicalLineageIdentityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::{
        CognitiveDerivationKindV1, EvidenceLineageGraphV1, EvidenceLineageNodeV1,
        ValidatedEvidenceLineageNodeV1, COGNITIVE_LINEAGE_SCHEMA_VERSION,
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
    fn legacy_graph_label_does_not_define_canonical_identity() {
        let nodes = vec![
            node(A, &[], CognitiveDerivationKindV1::RootObservation),
            node(B, &[A], CognitiveDerivationKindV1::Inference),
        ];
        let first = graph(E, nodes.clone());
        let second = graph(F, nodes);
        assert_eq!(
            canonical_evidence_lineage_graph_id_v1(&first).unwrap(),
            canonical_evidence_lineage_graph_id_v1(&second).unwrap()
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
            canonical_evidence_lineage_graph_id_v1(&first).unwrap(),
            canonical_evidence_lineage_graph_id_v1(&second).unwrap()
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
            canonical_evidence_lineage_graph_id_v1(&first).unwrap(),
            canonical_evidence_lineage_graph_id_v1(&second).unwrap()
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
            canonical_evidence_lineage_graph_id_v1(&first).unwrap(),
            canonical_evidence_lineage_graph_id_v1(&second).unwrap()
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
            canonical_evidence_lineage_graph_id_v1(&first).unwrap(),
            canonical_evidence_lineage_graph_id_v1(&second).unwrap()
        );
    }
}
