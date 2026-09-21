// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed provenance graph for MEL-EPI.
//!
//! Provenance records exact lineage and context. It never propagates epistemic
//! claims by ancestry, reachability, or graph shape.
//!
//! Only derivation relations participate in the acyclic partial order. Typed
//! association/context edges are validated independently and do not become
//! derivation merely because they are directional.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::semantic_evidence::{
    EvidenceSemanticIdV1, SemanticTranscriptError, SemanticTranscriptV1,
};
use crate::source_reference::{EvidenceSourceRefV1, SourceReferenceError};
use crate::validated_projection::ValidatedEvidenceSourceRefV1;

pub const PROVENANCE_GRAPH_VERSION_V1: &str = "melothaea-provenance-graph-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProvenanceNodeClassV1 {
    EvidenceArtifact,
    Activity,
    ProfileOrPolicy,
    DecisionOrReceipt,
    AgentOrAuthorityRef,
}

impl ProvenanceNodeClassV1 {
    fn code(self) -> &'static str {
        match self {
            Self::EvidenceArtifact => "evidence-artifact",
            Self::Activity => "activity",
            Self::ProfileOrPolicy => "profile-or-policy",
            Self::DecisionOrReceipt => "decision-or-receipt",
            Self::AgentOrAuthorityRef => "agent-or-authority-ref",
        }
    }

    fn is_artifact_like(self) -> bool {
        matches!(self, Self::EvidenceArtifact | Self::DecisionOrReceipt)
    }
}

/// Exact endpoint identity/reference for one provenance node.
///
/// This is deliberately stronger than using only an `EvidenceSemanticIdV1`.
/// Two source-native artifacts may project to the same semantic identity while
/// retaining different native commitments or preservation ceilings. Those must
/// remain distinct provenance nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceNodeReferenceV1 {
    SemanticId(EvidenceSemanticIdV1),
    SourceRef(EvidenceSourceRefV1),
}

impl ProvenanceNodeReferenceV1 {
    pub fn semantic_id(&self) -> &EvidenceSemanticIdV1 {
        match self {
            Self::SemanticId(id) => id,
            Self::SourceRef(source) => &source.semantic_id,
        }
    }

    fn validate(&self) -> Result<(), ProvenanceGraphError> {
        match self {
            Self::SemanticId(id) => id
                .validate()
                .map_err(ProvenanceGraphError::InvalidSemanticId),
            Self::SourceRef(source) => source
                .validate()
                .map_err(ProvenanceGraphError::InvalidSourceRef),
        }
    }

    fn kind_code(&self) -> &'static str {
        match self {
            Self::SemanticId(_) => "semantic-id",
            Self::SourceRef(_) => "source-ref",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceNodeV1 {
    pub class: ProvenanceNodeClassV1,
    pub reference: ProvenanceNodeReferenceV1,
}

impl ProvenanceNodeV1 {
    /// Build a source-backed node from an immutable structurally validated
    /// MEL-EPI source reference.
    ///
    /// The serialized graph still stores an ordinary source-ref DTO and remains
    /// non-authoritative after interchange.
    pub fn from_validated_source(
        class: ProvenanceNodeClassV1,
        source: &ValidatedEvidenceSourceRefV1,
    ) -> Self {
        Self {
            class,
            reference: ProvenanceNodeReferenceV1::SourceRef(source.as_raw().clone()),
        }
    }

    pub fn from_semantic_id(
        class: ProvenanceNodeClassV1,
        semantic_id: EvidenceSemanticIdV1,
    ) -> Self {
        Self {
            class,
            reference: ProvenanceNodeReferenceV1::SemanticId(semantic_id),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceDerivationRelationV1 {
    DerivedFrom,
    ProjectedFrom,
    AnalyzedFrom,
    ReportedFrom,
    AggregatedFrom,
}

impl ProvenanceDerivationRelationV1 {
    fn code(self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived-from",
            Self::ProjectedFrom => "projected-from",
            Self::AnalyzedFrom => "analyzed-from",
            Self::ReportedFrom => "reported-from",
            Self::AggregatedFrom => "aggregated-from",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::DerivedFrom => 1,
            Self::ProjectedFrom => 2,
            Self::AnalyzedFrom => 3,
            Self::ReportedFrom => 4,
            Self::AggregatedFrom => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceAssociationRelationV1 {
    GeneratedBy,
    ObservedDuring,
    AssignedUnder,
    AdmittedUnder,
    ExcludedUnder,
    AttestedBy,
    ExecutedBy,
    GovernedBy,
}

impl ProvenanceAssociationRelationV1 {
    fn code(self) -> &'static str {
        match self {
            Self::GeneratedBy => "generated-by",
            Self::ObservedDuring => "observed-during",
            Self::AssignedUnder => "assigned-under",
            Self::AdmittedUnder => "admitted-under",
            Self::ExcludedUnder => "excluded-under",
            Self::AttestedBy => "attested-by",
            Self::ExecutedBy => "executed-by",
            Self::GovernedBy => "governed-by",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::GeneratedBy => 1,
            Self::ObservedDuring => 2,
            Self::AssignedUnder => 3,
            Self::AdmittedUnder => 4,
            Self::ExcludedUnder => 5,
            Self::AttestedBy => 6,
            Self::ExecutedBy => 7,
            Self::GovernedBy => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceRelationV1 {
    Derivation(ProvenanceDerivationRelationV1),
    Association(ProvenanceAssociationRelationV1),
}

impl ProvenanceRelationV1 {
    fn domain_rank(self) -> u8 {
        match self {
            Self::Derivation(_) => 0,
            Self::Association(_) => 1,
        }
    }

    fn relation_rank(self) -> u8 {
        match self {
            Self::Derivation(value) => value.rank(),
            Self::Association(value) => value.rank(),
        }
    }

    fn domain_code(self) -> &'static str {
        match self {
            Self::Derivation(_) => "derivation",
            Self::Association(_) => "association",
        }
    }

    fn relation_code(self) -> &'static str {
        match self {
            Self::Derivation(value) => value.code(),
            Self::Association(value) => value.code(),
        }
    }

    pub fn is_derivation(self) -> bool {
        matches!(self, Self::Derivation(_))
    }
}

/// Direction is semantic and relation-specific.
///
/// For derivation relations V1 uses:
///
/// ```text
/// child/output --DerivedFrom--> parent/input
/// ```
///
/// Endpoints carry the complete node reference. A source-backed endpoint is
/// therefore distinguished by semantic identity + native commitment +
/// preservation ceiling, not semantic identity alone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEdgeV1 {
    pub relation: ProvenanceRelationV1,
    pub from: ProvenanceNodeReferenceV1,
    pub to: ProvenanceNodeReferenceV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceGraphV1 {
    pub graph_version: String,
    pub nodes: Vec<ProvenanceNodeV1>,
    pub edges: Vec<ProvenanceEdgeV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvenanceEndpointV1 {
    From,
    To,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceGraphError {
    WrongVersion { found: String },
    InvalidSemanticId(SemanticTranscriptError),
    InvalidSourceRef(SourceReferenceError),
    NodesNotStrictlyIncreasing { previous_index: usize, next_index: usize },
    EdgesNotStrictlyIncreasing { previous_index: usize, next_index: usize },
    MissingEndpoint {
        edge_index: usize,
        endpoint: ProvenanceEndpointV1,
    },
    SelfEdge { edge_index: usize },
    InvalidRelationEndpointClasses {
        edge_index: usize,
        relation: ProvenanceRelationV1,
        from_class: ProvenanceNodeClassV1,
        to_class: ProvenanceNodeClassV1,
    },
    DerivationCycle,
    Transcript(SemanticTranscriptError),
}

impl fmt::Display for ProvenanceGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongVersion { found } => {
                write!(f, "unsupported provenance graph version: {found}")
            }
            Self::InvalidSemanticId(error) => write!(f, "invalid semantic id: {error}"),
            Self::InvalidSourceRef(error) => write!(f, "invalid source ref: {error}"),
            Self::NodesNotStrictlyIncreasing {
                previous_index,
                next_index,
            } => write!(
                f,
                "provenance nodes must be strictly increasing and unique by exact node-reference bytes: previous index {previous_index}, next index {next_index}"
            ),
            Self::EdgesNotStrictlyIncreasing {
                previous_index,
                next_index,
            } => write!(
                f,
                "provenance edges must be strictly increasing and unique: previous index {previous_index}, next index {next_index}"
            ),
            Self::MissingEndpoint {
                edge_index,
                endpoint,
            } => write!(
                f,
                "provenance edge {edge_index} references missing {endpoint:?} endpoint"
            ),
            Self::SelfEdge { edge_index } => {
                write!(f, "provenance edge {edge_index} is a self-edge")
            }
            Self::InvalidRelationEndpointClasses {
                edge_index,
                relation,
                from_class,
                to_class,
            } => write!(
                f,
                "provenance edge {edge_index} has invalid endpoint classes for {relation:?}: {from_class:?} -> {to_class:?}"
            ),
            Self::DerivationCycle => write!(f, "derivation subgraph contains a cycle"),
            Self::Transcript(error) => write!(f, "provenance transcript error: {error}"),
        }
    }
}

impl std::error::Error for ProvenanceGraphError {}

impl ProvenanceGraphV1 {
    pub fn validate(&self) -> Result<(), ProvenanceGraphError> {
        if self.graph_version != PROVENANCE_GRAPH_VERSION_V1 {
            return Err(ProvenanceGraphError::WrongVersion {
                found: self.graph_version.clone(),
            });
        }

        let node_keys = self
            .nodes
            .iter()
            .map(|node| node_reference_bytes(&node.reference))
            .collect::<Result<Vec<_>, _>>()?;

        for (index, pair) in node_keys.windows(2).enumerate() {
            if pair[0] >= pair[1] {
                return Err(ProvenanceGraphError::NodesNotStrictlyIncreasing {
                    previous_index: index,
                    next_index: index + 1,
                });
            }
        }

        let node_index: BTreeMap<Vec<u8>, usize> = node_keys
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect();

        for (index, edge) in self.edges.iter().enumerate() {
            let from_key = node_reference_bytes(&edge.from)?;
            let to_key = node_reference_bytes(&edge.to)?;

            if from_key == to_key {
                return Err(ProvenanceGraphError::SelfEdge { edge_index: index });
            }

            let from_index = node_index.get(&from_key).copied().ok_or(
                ProvenanceGraphError::MissingEndpoint {
                    edge_index: index,
                    endpoint: ProvenanceEndpointV1::From,
                },
            )?;
            let to_index = node_index.get(&to_key).copied().ok_or(
                ProvenanceGraphError::MissingEndpoint {
                    edge_index: index,
                    endpoint: ProvenanceEndpointV1::To,
                },
            )?;
            let from_class = self.nodes[from_index].class;
            let to_class = self.nodes[to_index].class;

            if !relation_endpoint_classes_valid(edge.relation, from_class, to_class) {
                return Err(ProvenanceGraphError::InvalidRelationEndpointClasses {
                    edge_index: index,
                    relation: edge.relation,
                    from_class,
                    to_class,
                });
            }
        }

        for (index, pair) in self.edges.windows(2).enumerate() {
            if edge_order_key(&pair[0])? >= edge_order_key(&pair[1])? {
                return Err(ProvenanceGraphError::EdgesNotStrictlyIncreasing {
                    previous_index: index,
                    next_index: index + 1,
                });
            }
        }

        derivation_topological_order(self)?;
        Ok(())
    }

    /// Canonical typed transcript for the complete generic graph.
    ///
    /// Deterministic graph bytes establish no truth/authority for any node or
    /// edge. Native/source-specific verifiers remain authoritative.
    pub fn semantic_payload(&self) -> Result<SemanticTranscriptV1, ProvenanceGraphError> {
        self.validate()?;

        let node_bytes = self
            .nodes
            .iter()
            .map(node_semantic_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let edge_bytes = self
            .edges
            .iter()
            .map(edge_semantic_bytes)
            .collect::<Result<Vec<_>, _>>()?;

        let mut payload = SemanticTranscriptV1::new();
        payload
            .push_utf8(1, &self.graph_version)
            .map_err(ProvenanceGraphError::Transcript)?;
        payload
            .push_sequence(2, &node_bytes)
            .map_err(ProvenanceGraphError::Transcript)?;
        payload
            .push_sequence(3, &edge_bytes)
            .map_err(ProvenanceGraphError::Transcript)?;
        Ok(payload)
    }
}

/// Immutable structurally validated provenance graph.
///
/// The wrapper caches canonical graph bytes and a deterministic parent-first
/// topological order for the derivation subgraph. It remains ordinary generic
/// provenance, not source admission and not a claim-propagation mechanism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedProvenanceGraphV1 {
    raw: ProvenanceGraphV1,
    semantic_payload: SemanticTranscriptV1,
    derivation_order: Vec<ProvenanceNodeReferenceV1>,
}

impl ValidatedProvenanceGraphV1 {
    pub fn as_raw(&self) -> &ProvenanceGraphV1 {
        &self.raw
    }

    pub fn into_raw(self) -> ProvenanceGraphV1 {
        self.raw
    }

    pub fn semantic_payload(&self) -> &SemanticTranscriptV1 {
        &self.semantic_payload
    }

    /// Deterministic parent/input-before-child/output order induced solely by
    /// derivation relations. Association/context edges do not affect this order.
    pub fn derivation_topological_order(&self) -> &[ProvenanceNodeReferenceV1] {
        &self.derivation_order
    }

    pub fn node_class(
        &self,
        reference: &ProvenanceNodeReferenceV1,
    ) -> Option<ProvenanceNodeClassV1> {
        let key = node_reference_bytes(reference).ok()?;
        self.raw
            .nodes
            .binary_search_by(|node| {
                node_reference_bytes(&node.reference)
                    .expect("validated graph contains canonical node reference")
                    .cmp(&key)
            })
            .ok()
            .map(|index| self.raw.nodes[index].class)
    }
}

impl TryFrom<ProvenanceGraphV1> for ValidatedProvenanceGraphV1 {
    type Error = ProvenanceGraphError;

    fn try_from(raw: ProvenanceGraphV1) -> Result<Self, Self::Error> {
        raw.validate()?;
        let semantic_payload = raw.semantic_payload()?;
        let derivation_order = derivation_topological_order(&raw)?;
        Ok(Self {
            raw,
            semantic_payload,
            derivation_order,
        })
    }
}

impl TryFrom<&ProvenanceGraphV1> for ValidatedProvenanceGraphV1 {
    type Error = ProvenanceGraphError;

    fn try_from(raw: &ProvenanceGraphV1) -> Result<Self, Self::Error> {
        Self::try_from(raw.clone())
    }
}

fn relation_endpoint_classes_valid(
    relation: ProvenanceRelationV1,
    from: ProvenanceNodeClassV1,
    to: ProvenanceNodeClassV1,
) -> bool {
    match relation {
        ProvenanceRelationV1::Derivation(_) => from.is_artifact_like() && to.is_artifact_like(),
        ProvenanceRelationV1::Association(value) => match value {
            ProvenanceAssociationRelationV1::GeneratedBy => {
                from.is_artifact_like() && to == ProvenanceNodeClassV1::Activity
            }
            ProvenanceAssociationRelationV1::ObservedDuring => {
                from == ProvenanceNodeClassV1::EvidenceArtifact
                    && to == ProvenanceNodeClassV1::Activity
            }
            ProvenanceAssociationRelationV1::AssignedUnder
            | ProvenanceAssociationRelationV1::ExcludedUnder => {
                from == ProvenanceNodeClassV1::EvidenceArtifact
                    && to == ProvenanceNodeClassV1::ProfileOrPolicy
            }
            ProvenanceAssociationRelationV1::AdmittedUnder => {
                from.is_artifact_like() && to == ProvenanceNodeClassV1::ProfileOrPolicy
            }
            ProvenanceAssociationRelationV1::AttestedBy => {
                from.is_artifact_like() && to == ProvenanceNodeClassV1::AgentOrAuthorityRef
            }
            ProvenanceAssociationRelationV1::ExecutedBy => {
                from == ProvenanceNodeClassV1::Activity
                    && to == ProvenanceNodeClassV1::AgentOrAuthorityRef
            }
            ProvenanceAssociationRelationV1::GovernedBy => {
                from == ProvenanceNodeClassV1::Activity
                    && to == ProvenanceNodeClassV1::ProfileOrPolicy
            }
        },
    }
}

fn edge_order_key(
    edge: &ProvenanceEdgeV1,
) -> Result<(u8, u8, Vec<u8>, Vec<u8>), ProvenanceGraphError> {
    Ok((
        edge.relation.domain_rank(),
        edge.relation.relation_rank(),
        node_reference_bytes(&edge.from)?,
        node_reference_bytes(&edge.to)?,
    ))
}

fn semantic_id_bytes(id: &EvidenceSemanticIdV1) -> Result<Vec<u8>, ProvenanceGraphError> {
    id.validate()
        .map_err(ProvenanceGraphError::InvalidSemanticId)?;

    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, &id.digest_algorithm)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(2, &id.transcript_version)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(3, &id.namespace)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(4, &id.schema_version)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(5, &id.profile_id)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(6, &id.record_id)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_sha256_digest_reference(
            7,
            id.digest_bytes()
                .map_err(ProvenanceGraphError::InvalidSemanticId)?,
        )
        .map_err(ProvenanceGraphError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

fn node_reference_bytes(
    reference: &ProvenanceNodeReferenceV1,
) -> Result<Vec<u8>, ProvenanceGraphError> {
    reference.validate()?;
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, reference.kind_code())
        .map_err(ProvenanceGraphError::Transcript)?;
    match reference {
        ProvenanceNodeReferenceV1::SemanticId(id) => payload
            .push_bytes(2, &semantic_id_bytes(id)?)
            .map_err(ProvenanceGraphError::Transcript)?,
        ProvenanceNodeReferenceV1::SourceRef(source) => payload
            .push_bytes(
                2,
                source
                    .semantic_payload()
                    .map_err(ProvenanceGraphError::InvalidSourceRef)?
                    .as_bytes(),
            )
            .map_err(ProvenanceGraphError::Transcript)?,
    }
    Ok(payload.as_bytes().to_vec())
}

fn node_semantic_bytes(node: &ProvenanceNodeV1) -> Result<Vec<u8>, ProvenanceGraphError> {
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, node.class.code())
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_bytes(2, &node_reference_bytes(&node.reference)?)
        .map_err(ProvenanceGraphError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

fn edge_semantic_bytes(edge: &ProvenanceEdgeV1) -> Result<Vec<u8>, ProvenanceGraphError> {
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, edge.relation.domain_code())
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_utf8(2, edge.relation.relation_code())
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_bytes(3, &node_reference_bytes(&edge.from)?)
        .map_err(ProvenanceGraphError::Transcript)?;
    payload
        .push_bytes(4, &node_reference_bytes(&edge.to)?)
        .map_err(ProvenanceGraphError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

/// Return every node in deterministic parent/input-before-child/output order for
/// derivation edges. Association edges are intentionally ignored. Ties are
/// broken by the already-canonical node-vector order.
fn derivation_topological_order(
    graph: &ProvenanceGraphV1,
) -> Result<Vec<ProvenanceNodeReferenceV1>, ProvenanceGraphError> {
    let node_keys = graph
        .nodes
        .iter()
        .map(|node| node_reference_bytes(&node.reference))
        .collect::<Result<Vec<_>, _>>()?;
    let node_index: BTreeMap<Vec<u8>, usize> = node_keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect();

    let mut indegree = vec![0usize; graph.nodes.len()];
    let mut dependents = vec![BTreeSet::<usize>::new(); graph.nodes.len()];

    for (edge_index, edge) in graph.edges.iter().enumerate() {
        if edge.relation.is_derivation() {
            let child_key = node_reference_bytes(&edge.from)?;
            let parent_key = node_reference_bytes(&edge.to)?;
            let child = node_index.get(&child_key).copied().ok_or(
                ProvenanceGraphError::MissingEndpoint {
                    edge_index,
                    endpoint: ProvenanceEndpointV1::From,
                },
            )?;
            let parent = node_index.get(&parent_key).copied().ok_or(
                ProvenanceGraphError::MissingEndpoint {
                    edge_index,
                    endpoint: ProvenanceEndpointV1::To,
                },
            )?;
            indegree[child] += 1;
            dependents[parent].insert(child);
        }
    }

    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(graph.nodes.len());

    while let Some(index) = ready.pop_first() {
        order.push(graph.nodes[index].reference.clone());
        for child in dependents[index].iter().copied() {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                ready.insert(child);
            }
        }
    }

    if order.len() != graph.nodes.len() {
        return Err(ProvenanceGraphError::DerivationCycle);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespaced_code::NamespacedCodeV1;
    use crate::source_reference::{
        EVIDENCE_SOURCE_REF_VERSION_V1, SemanticPreservationV1, SourceNativeCommitmentV1,
    };

    fn id(record: &str, byte: u8) -> EvidenceSemanticIdV1 {
        EvidenceSemanticIdV1::from_sha256_digest(
            "muse.provenance-test",
            "v1",
            "projection-v1",
            record,
            [byte; 32],
        )
        .unwrap()
    }

    fn source_ref_from_id(semantic_id: EvidenceSemanticIdV1, native_byte: u8) -> EvidenceSourceRefV1 {
        EvidenceSourceRefV1 {
            source_ref_version: EVIDENCE_SOURCE_REF_VERSION_V1.into(),
            semantic_id,
            native_commitment: SourceNativeCommitmentV1 {
                scheme_id: NamespacedCodeV1::new("muse.test.commitment-v1").unwrap(),
                sha256_hex: format!("{native_byte:02x}").repeat(32),
            },
            preservation: SemanticPreservationV1::LosslessUnderProfile,
        }
    }

    fn source_ref(record: &str, semantic_byte: u8, native_byte: u8) -> EvidenceSourceRefV1 {
        source_ref_from_id(id(record, semantic_byte), native_byte)
    }

    fn canonical_nodes() -> Vec<ProvenanceNodeV1> {
        let parent = ProvenanceNodeV1 {
            class: ProvenanceNodeClassV1::EvidenceArtifact,
            reference: ProvenanceNodeReferenceV1::SourceRef(source_ref(
                "parent", 0x11, 0x61,
            )),
        };
        let child = ProvenanceNodeV1 {
            class: ProvenanceNodeClassV1::EvidenceArtifact,
            reference: ProvenanceNodeReferenceV1::SourceRef(source_ref(
                "child", 0x22, 0x62,
            )),
        };
        let activity = ProvenanceNodeV1::from_semantic_id(
            ProvenanceNodeClassV1::Activity,
            id("activity", 0x33),
        );
        let policy = ProvenanceNodeV1::from_semantic_id(
            ProvenanceNodeClassV1::ProfileOrPolicy,
            id("policy", 0x44),
        );
        let agent = ProvenanceNodeV1::from_semantic_id(
            ProvenanceNodeClassV1::AgentOrAuthorityRef,
            id("agent", 0x55),
        );
        let mut nodes = vec![parent, child, activity, policy, agent];
        nodes.sort_by_key(|node| node_reference_bytes(&node.reference).unwrap());
        nodes
    }

    fn reference(nodes: &[ProvenanceNodeV1], record: &str) -> ProvenanceNodeReferenceV1 {
        nodes
            .iter()
            .find(|node| node.reference.semantic_id().record_id == record)
            .unwrap()
            .reference
            .clone()
    }

    fn canonical_graph() -> ProvenanceGraphV1 {
        let nodes = canonical_nodes();
        let parent = reference(&nodes, "parent");
        let child = reference(&nodes, "child");
        let activity = reference(&nodes, "activity");
        let policy = reference(&nodes, "policy");
        let agent = reference(&nodes, "agent");

        let mut edges = vec![
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Derivation(
                    ProvenanceDerivationRelationV1::DerivedFrom,
                ),
                from: child.clone(),
                to: parent,
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Association(
                    ProvenanceAssociationRelationV1::GeneratedBy,
                ),
                from: child,
                to: activity.clone(),
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Association(
                    ProvenanceAssociationRelationV1::ExecutedBy,
                ),
                from: activity.clone(),
                to: agent,
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Association(
                    ProvenanceAssociationRelationV1::GovernedBy,
                ),
                from: activity,
                to: policy,
            },
        ];
        edges.sort_by_key(|edge| edge_order_key(edge).unwrap());

        ProvenanceGraphV1 {
            graph_version: PROVENANCE_GRAPH_VERSION_V1.into(),
            nodes,
            edges,
        }
    }

    #[test]
    fn canonical_graph_validates_and_freezes_payload() {
        let graph = canonical_graph();
        graph.validate().unwrap();
        let expected = graph.semantic_payload().unwrap();
        let validated = ValidatedProvenanceGraphV1::try_from(graph).unwrap();
        assert_eq!(validated.semantic_payload(), &expected);
    }

    #[test]
    fn source_node_constructor_accepts_validated_source_ref() {
        let validated = ValidatedEvidenceSourceRefV1::try_from(source_ref(
            "source", 0x66, 0x76,
        ))
        .unwrap();
        let node = ProvenanceNodeV1::from_validated_source(
            ProvenanceNodeClassV1::EvidenceArtifact,
            &validated,
        );
        assert_eq!(node.reference.semantic_id().record_id, "source");
    }

    #[test]
    fn same_semantic_identity_with_different_native_commitments_stays_distinct() {
        let shared = id("same-semantic", 0x77);
        let first = ProvenanceNodeV1 {
            class: ProvenanceNodeClassV1::EvidenceArtifact,
            reference: ProvenanceNodeReferenceV1::SourceRef(source_ref_from_id(
                shared.clone(),
                0x81,
            )),
        };
        let second = ProvenanceNodeV1 {
            class: ProvenanceNodeClassV1::EvidenceArtifact,
            reference: ProvenanceNodeReferenceV1::SourceRef(source_ref_from_id(shared, 0x82)),
        };
        assert_eq!(first.reference.semantic_id(), second.reference.semantic_id());
        assert_ne!(
            node_reference_bytes(&first.reference).unwrap(),
            node_reference_bytes(&second.reference).unwrap()
        );

        let mut graph = ProvenanceGraphV1 {
            graph_version: PROVENANCE_GRAPH_VERSION_V1.into(),
            nodes: vec![first, second],
            edges: Vec::new(),
        };
        graph
            .nodes
            .sort_by_key(|node| node_reference_bytes(&node.reference).unwrap());
        graph.validate().unwrap();
    }

    #[test]
    fn nodes_must_be_canonical_and_unique_by_full_reference() {
        let mut graph = canonical_graph();
        graph.nodes.swap(0, 1);
        assert!(matches!(
            graph.validate(),
            Err(ProvenanceGraphError::NodesNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn missing_endpoint_rejects() {
        let mut graph = canonical_graph();
        let missing = ProvenanceNodeReferenceV1::SemanticId(id("missing", 0x99));
        graph.edges[0].to = missing;
        graph.edges.sort_by_key(|edge| edge_order_key(edge).unwrap());
        assert!(matches!(
            graph.validate(),
            Err(ProvenanceGraphError::MissingEndpoint { .. })
        ));
    }

    #[test]
    fn invalid_relation_endpoint_classes_reject() {
        let mut graph = canonical_graph();
        let artifact = graph
            .nodes
            .iter()
            .find(|node| node.class == ProvenanceNodeClassV1::EvidenceArtifact)
            .unwrap()
            .reference
            .clone();
        let policy = graph
            .nodes
            .iter()
            .find(|node| node.class == ProvenanceNodeClassV1::ProfileOrPolicy)
            .unwrap()
            .reference
            .clone();
        graph.edges = vec![ProvenanceEdgeV1 {
            relation: ProvenanceRelationV1::Association(
                ProvenanceAssociationRelationV1::GovernedBy,
            ),
            from: artifact,
            to: policy,
        }];
        assert!(matches!(
            graph.validate(),
            Err(ProvenanceGraphError::InvalidRelationEndpointClasses { .. })
        ));
    }

    #[test]
    fn derivation_cycle_rejects() {
        let mut graph = canonical_graph();
        let parent = reference(&graph.nodes, "parent");
        let child = reference(&graph.nodes, "child");
        graph.edges = vec![
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Derivation(
                    ProvenanceDerivationRelationV1::DerivedFrom,
                ),
                from: child.clone(),
                to: parent.clone(),
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Derivation(
                    ProvenanceDerivationRelationV1::DerivedFrom,
                ),
                from: parent,
                to: child,
            },
        ];
        graph.edges.sort_by_key(|edge| edge_order_key(edge).unwrap());
        assert_eq!(graph.validate(), Err(ProvenanceGraphError::DerivationCycle));
    }

    #[test]
    fn derivation_order_is_parent_before_child_and_ignores_associations() {
        let validated = ValidatedProvenanceGraphV1::try_from(canonical_graph()).unwrap();
        let order = validated.derivation_topological_order();
        let parent_index = order
            .iter()
            .position(|reference| reference.semantic_id().record_id == "parent")
            .unwrap();
        let child_index = order
            .iter()
            .position(|reference| reference.semantic_id().record_id == "child")
            .unwrap();
        assert!(parent_index < child_index);
    }

    #[test]
    fn graph_payload_changes_when_relation_changes() {
        let left = canonical_graph();
        let mut right = left.clone();
        let edge = right
            .edges
            .iter_mut()
            .find(|edge| edge.relation.is_derivation())
            .unwrap();
        edge.relation = ProvenanceRelationV1::Derivation(
            ProvenanceDerivationRelationV1::ProjectedFrom,
        );
        right.edges.sort_by_key(|edge| edge_order_key(edge).unwrap());
        right.validate().unwrap();
        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn serde_round_trip_returns_raw_graph_then_explicit_validation() {
        let graph = canonical_graph();
        let json = serde_json::to_string(&graph).unwrap();
        let raw: ProvenanceGraphV1 = serde_json::from_str(&json).unwrap();
        let validated = ValidatedProvenanceGraphV1::try_from(raw).unwrap();
        assert!(!validated.semantic_payload().as_bytes().is_empty());
    }
}
