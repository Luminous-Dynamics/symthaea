// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit adapter from a validated scientific qualification projection to
//! the CP-04 typed evidence vocabulary.
//!
//! The adapter is deliberately lossy in one direction: it accepts only the
//! relation/object vocabulary that CP-04 explicitly admits. It never invents
//! facts, maps epistemic edges into dependency edges, or raises the projection
//! authority ceiling.

use crate::{
    AuthorityCeiling, EngineeringObjectId, EngineeringRelation, EngineeringRelationKind,
    QualificationProjection, QualificationProjectionError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const CP04_ADAPTER_SCHEMA: &str = "symthaea.cp-04-qualification-adapter.v1";
pub const CP04_SCHEMA: &str = "cp-04-compute-evidence-dag-v1-1";
pub const CP04_CLAIM_CEILING: &str =
    "typed dependency and invalidation semantics over synthetic/reference workflows only; no physical execution, performance, compiler, accelerator, model, deployment, safety, or operational-authority claim";

const CP04_NODE_TYPES: &[&str] = &[
    "requirement",
    "representation",
    "model",
    "model_parameters",
    "runtime",
    "toolchain",
    "accelerator",
    "deployment_artifact",
    "execution_context",
    "observation",
    "uncertainty",
    "statistics",
    "provenance_reference",
    "currentness",
    "applicability",
    "disposition",
];

/// A CP-04 node carrying the original canonical engineering identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cp04Node {
    pub identity: EngineeringObjectId,
    pub identity_digest: String,
    pub object_kind: String,
}

/// A CP-04 edge carrying the original relation identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cp04Edge {
    pub source_identity_digest: String,
    pub edge_type: String,
    pub target_identity_digest: String,
    pub relation_digest: String,
}

/// Contract-neutral hand-off artifact for a CP-04 consumer.
///
/// Every node and edge is derived from the validated source projection; the
/// adapter does not synthesize a claim, measurement, performance result, or
/// operational authorization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cp04QualificationArtifact {
    pub schema: String,
    pub cp04_schema: String,
    pub source_graph_digest: String,
    pub projection_digest: String,
    pub qualification_policy: String,
    pub authority_ceiling: AuthorityCeiling,
    pub claim_ceiling: String,
    pub nodes: Vec<Cp04Node>,
    pub edges: Vec<Cp04Edge>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cp04AdapterError {
    InvalidProjection(QualificationProjectionError),
    UnsupportedNodeKind(String),
    UnsupportedRelationKind(String),
    MissingEndpoint(String),
    DuplicateNode(String),
    DuplicateEdge(String),
    UnorderedNodes,
    UnorderedEdges,
    InvalidSchema(String),
    InvalidCp04Schema(String),
    InvalidAuthorityCeiling,
    InvalidClaimCeiling,
    InvalidArtifactDigest,
    SourceMismatch,
}

impl fmt::Display for Cp04AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProjection(err) => write!(f, "invalid source projection: {err}"),
            Self::UnsupportedNodeKind(kind) => write!(f, "CP-04 does not admit node kind: {kind}"),
            Self::UnsupportedRelationKind(kind) => {
                write!(f, "CP-04 does not admit relation kind: {kind}")
            }
            Self::MissingEndpoint(digest) => write!(f, "relation endpoint missing from node set: {digest}"),
            Self::DuplicateNode(digest) => write!(f, "duplicate CP-04 node identity: {digest}"),
            Self::DuplicateEdge(digest) => write!(f, "duplicate CP-04 relation identity: {digest}"),
            Self::UnorderedNodes => write!(f, "CP-04 nodes are not canonically ordered"),
            Self::UnorderedEdges => write!(f, "CP-04 edges are not canonically ordered"),
            Self::InvalidSchema(schema) => write!(f, "invalid CP-04 adapter schema: {schema}"),
            Self::InvalidCp04Schema(schema) => write!(f, "invalid CP-04 schema: {schema}"),
            Self::InvalidAuthorityCeiling => write!(f, "CP-04 authority ceiling is not SyntheticQualification"),
            Self::InvalidClaimCeiling => write!(f, "CP-04 claim ceiling does not match the adapter contract"),
            Self::InvalidArtifactDigest => write!(f, "CP-04 adapter artifact digest mismatch"),
            Self::SourceMismatch => write!(f, "CP-04 adapter artifact does not match source projection"),
        }
    }
}

impl std::error::Error for Cp04AdapterError {}

impl From<QualificationProjectionError> for Cp04AdapterError {
    fn from(value: QualificationProjectionError) -> Self {
        Self::InvalidProjection(value)
    }
}

impl Cp04QualificationArtifact {
    /// Adapt a validated qualification projection without adding semantic facts.
    pub fn try_from_projection(
        projection: &QualificationProjection,
    ) -> Result<Self, Cp04AdapterError> {
        projection.validate()?;

        let mut nodes: BTreeMap<String, EngineeringObjectId> = BTreeMap::new();
        let mut edges = Vec::new();

        for relation in projection.relations() {
            let source = &relation.source;
            let target = &relation.target;
            validate_cp04_node(source)?;
            validate_cp04_node(target)?;
            validate_cp04_relation(relation)?;

            let source_digest = source.identity_digest();
            let target_digest = target.identity_digest();
            if nodes.insert(source_digest.clone(), source.clone()).is_some() {
                // Re-inserting an identical identity is expected when multiple
                // edges share a node; the map itself proves identity uniqueness.
            }
            if nodes.insert(target_digest.clone(), target.clone()).is_some() {
                // See source insertion above.
            }

            edges.push(Cp04Edge {
                source_identity_digest: source_digest,
                edge_type: relation.kind.wire_name().to_owned(),
                target_identity_digest: target_digest,
                relation_digest: relation.relation_digest(),
            });
        }

        edges.sort_by(|a, b| {
            (&a.source_identity_digest, &a.edge_type, &a.target_identity_digest, &a.relation_digest)
                .cmp(&(&b.source_identity_digest, &b.edge_type, &b.target_identity_digest, &b.relation_digest))
        });

        let nodes = nodes
            .into_values()
            .map(|identity| {
                let digest = identity.identity_digest();
                Cp04Node {
                    object_kind: identity.object_kind.clone(),
                    identity,
                    identity_digest: digest,
                }
            })
            .collect();

        let mut artifact = Self {
            schema: CP04_ADAPTER_SCHEMA.to_owned(),
            cp04_schema: CP04_SCHEMA.to_owned(),
            source_graph_digest: projection.source_graph_digest().to_owned(),
            projection_digest: projection.projection_digest().to_owned(),
            qualification_policy: projection.qualification_policy().to_owned(),
            authority_ceiling: projection.authority_ceiling(),
            claim_ceiling: CP04_CLAIM_CEILING.to_owned(),
            nodes,
            edges,
            artifact_digest: String::new(),
        };
        artifact.artifact_digest = artifact.compute_digest()?;
        Ok(artifact)
    }

    /// Validate the complete adapter envelope before a downstream hand-off.
    pub fn validate(&self) -> Result<(), Cp04AdapterError> {
        if self.schema != CP04_ADAPTER_SCHEMA {
            return Err(Cp04AdapterError::InvalidSchema(self.schema.clone()));
        }
        if self.cp04_schema != CP04_SCHEMA {
            return Err(Cp04AdapterError::InvalidCp04Schema(self.cp04_schema.clone()));
        }
        if self.claim_ceiling != CP04_CLAIM_CEILING {
            return Err(Cp04AdapterError::InvalidClaimCeiling);
        }
        if self.authority_ceiling != AuthorityCeiling::SyntheticQualification {
            return Err(Cp04AdapterError::InvalidAuthorityCeiling);
        }

        let mut node_digests = BTreeMap::new();
        let mut previous_node_digest: Option<String> = None;
        for node in &self.nodes {
            node.identity.validate().map_err(|_| {
                Cp04AdapterError::MissingEndpoint(node.identity_digest.clone())
            })?;
            let digest = node.identity.identity_digest();
            if digest != node.identity_digest || node.object_kind != node.identity.object_kind {
                return Err(Cp04AdapterError::MissingEndpoint(digest));
            }
            validate_cp04_node(&node.identity)?;
            if node_digests.insert(digest.clone(), ()).is_some() {
                return Err(Cp04AdapterError::DuplicateNode(digest));
            }
            if previous_node_digest.as_deref().is_some_and(|previous| previous >= digest.as_str()) {
                return Err(Cp04AdapterError::UnorderedNodes);
            }
            previous_node_digest = Some(digest);
        }

        let mut previous_edge_key: Option<(String, String, String, String)> = None;
        let mut seen_edge_digests = std::collections::BTreeSet::new();
        for edge in &self.edges {
            if !node_digests.contains_key(&edge.source_identity_digest) {
                return Err(Cp04AdapterError::MissingEndpoint(
                    edge.source_identity_digest.clone(),
                ));
            }
            if !node_digests.contains_key(&edge.target_identity_digest) {
                return Err(Cp04AdapterError::MissingEndpoint(
                    edge.target_identity_digest.clone(),
                ));
            }
            let kind = cp04_relation_kind(&edge.edge_type)?;
            let source = node_digests
                .get(&edge.source_identity_digest)
                .and_then(|_| self.nodes.iter().find(|n| n.identity_digest == edge.source_identity_digest))
                .ok_or_else(|| Cp04AdapterError::MissingEndpoint(edge.source_identity_digest.clone()))?;
            let target = self.nodes
                .iter()
                .find(|n| n.identity_digest == edge.target_identity_digest)
                .ok_or_else(|| Cp04AdapterError::MissingEndpoint(edge.target_identity_digest.clone()))?;
            let relation = EngineeringRelation::new(
                source.identity.clone(),
                target.identity.clone(),
                kind,
            )
            .map_err(|_| Cp04AdapterError::UnsupportedRelationKind(edge.edge_type.clone()))?;
            let computed_relation_digest = relation.relation_digest();
            if computed_relation_digest != edge.relation_digest {
                return Err(Cp04AdapterError::InvalidArtifactDigest);
            }
            if !seen_edge_digests.insert(computed_relation_digest.clone()) {
                return Err(Cp04AdapterError::DuplicateEdge(computed_relation_digest));
            }
            let key = (
                edge.source_identity_digest.clone(), edge.edge_type.clone(),
                edge.target_identity_digest.clone(), edge.relation_digest.clone(),
            );
            if previous_edge_key.as_ref().is_some_and(|previous| previous >= &key) {
                return Err(Cp04AdapterError::UnorderedEdges);
            }
            previous_edge_key = Some(key);
        }

        if self.artifact_digest != self.compute_digest()? {
            return Err(Cp04AdapterError::InvalidArtifactDigest);
        }
        Ok(())
    }

    /// Require the artifact to match the exact validated projection that produced it.
    ///
    /// Internal artifact validation proves envelope integrity; this method additionally
    /// proves lineage preservation against the source projection and therefore prevents
    /// a caller from changing an edge or identity while merely recomputing the envelope
    /// digest.
    pub fn validate_against_projection(
        &self,
        projection: &QualificationProjection,
    ) -> Result<(), Cp04AdapterError> {
        projection.validate()?;
        let expected = Self::try_from_projection(projection)?;
        if self.schema != expected.schema
            || self.cp04_schema != expected.cp04_schema
            || self.source_graph_digest != expected.source_graph_digest
            || self.projection_digest != expected.projection_digest
            || self.qualification_policy != expected.qualification_policy
            || self.authority_ceiling != expected.authority_ceiling
            || self.claim_ceiling != expected.claim_ceiling
            || self.nodes != expected.nodes
            || self.edges != expected.edges
            || self.artifact_digest != expected.artifact_digest
        {
            return Err(Cp04AdapterError::SourceMismatch);
        }
        self.validate()
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, Cp04AdapterError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| Cp04AdapterError::InvalidArtifactDigest)
    }

    fn compute_digest(&self) -> Result<String, Cp04AdapterError> {
        let mut clone = self.clone();
        clone.artifact_digest.clear();
        let bytes =
            serde_json::to_vec(&clone).map_err(|_| Cp04AdapterError::InvalidArtifactDigest)?;
        Ok(hex_digest(&bytes))
    }
}

const CP04_EDGE_TYPES: &[&str] = &[
    "requires",
    "implements",
    "parameterizes",
    "executes_with",
    "compiled_by",
    "runs_on",
    "deploys",
    "executes",
    "observes",
    "quantifies",
    "summarizes",
    "traces_to",
    "currentness_for",
    "applicable_to",
    "derives",
];

fn cp04_relation_kind(wire_name: &str) -> Result<EngineeringRelationKind, Cp04AdapterError> {
    let kind = match wire_name {
        "requires" => EngineeringRelationKind::Requires,
        "implements" => EngineeringRelationKind::Implements,
        "parameterizes" => EngineeringRelationKind::Parameterizes,
        "executes_with" => EngineeringRelationKind::ExecutesWith,
        "compiled_by" => EngineeringRelationKind::CompiledBy,
        "runs_on" => EngineeringRelationKind::RunsOn,
        "deploys" => EngineeringRelationKind::Deploys,
        "executes" => EngineeringRelationKind::Executes,
        "observes" => EngineeringRelationKind::Observes,
        "quantifies" => EngineeringRelationKind::Quantifies,
        "summarizes" => EngineeringRelationKind::Summarizes,
        "traces_to" => EngineeringRelationKind::TracesTo,
        "currentness_for" => EngineeringRelationKind::CurrentnessFor,
        "applicable_to" => EngineeringRelationKind::ApplicableTo,
        "derives" => EngineeringRelationKind::Derives,
        _ => return Err(Cp04AdapterError::UnsupportedRelationKind(wire_name.to_owned())),
    };
    Ok(kind)
}

fn validate_cp04_node(node: &EngineeringObjectId) -> Result<(), Cp04AdapterError> {
    if CP04_NODE_TYPES.contains(&node.object_kind.as_str()) {
        Ok(())
    } else {
        Err(Cp04AdapterError::UnsupportedNodeKind(
            node.object_kind.clone(),
        ))
    }
}

fn validate_cp04_relation(relation: &EngineeringRelation) -> Result<(), Cp04AdapterError> {
    let kind = relation.kind.wire_name();
    if !CP04_EDGE_TYPES.contains(&kind) {
        return Err(Cp04AdapterError::UnsupportedRelationKind(kind.to_owned()));
    }

    let allowed = relation.kind.endpoint_pairs().ok_or_else(|| {
        Cp04AdapterError::UnsupportedRelationKind(kind.to_owned())
    })?;
    if allowed.iter().any(|(source, target)| {
        *source == relation.source.object_kind && *target == relation.target.object_kind
    }) {
        Ok(())
    } else {
        Err(Cp04AdapterError::UnsupportedRelationKind(kind.to_owned()))
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineeringRelation, EngineeringRelationKind};

    const A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn object(kind: &str, id: &str, digest: &str) -> EngineeringObjectId {
        EngineeringObjectId::new("cp04-fixture", kind, id, "1", digest).unwrap()
    }

    fn projection() -> QualificationProjection {
        let requirement = object("requirement", "r", A);
        let representation = object("representation", "rep", B);
        let model = object("model", "m", A);
        let params = object("model_parameters", "p", B);
        let runtime = object("runtime", "rt", A);
        let toolchain = object("toolchain", "tc", B);
        let accelerator = object("accelerator", "acc", A);
        let deployment = object("deployment_artifact", "dep", B);
        let context = object("execution_context", "ctx", A);
        let observation = object("observation", "obs", B);
        let uncertainty = object("uncertainty", "u", A);
        let statistics = object("statistics", "s", B);
        let provenance = object("provenance_reference", "prov", A);
        let currentness = object("currentness", "cur", B);
        let applicability = object("applicability", "app", A);
        let disposition = object("disposition", "d", B);

        let relations = vec![
            (requirement, representation, EngineeringRelationKind::Requires),
            (representation, model, EngineeringRelationKind::Implements),
            (model, params, EngineeringRelationKind::Parameterizes),
            (model, runtime, EngineeringRelationKind::ExecutesWith),
            (runtime, toolchain, EngineeringRelationKind::CompiledBy),
            (runtime, accelerator, EngineeringRelationKind::RunsOn),
            (accelerator, deployment, EngineeringRelationKind::Deploys),
            (deployment, context, EngineeringRelationKind::Executes),
            (context, observation, EngineeringRelationKind::Observes),
            (observation, uncertainty, EngineeringRelationKind::Quantifies),
            (observation, statistics, EngineeringRelationKind::Quantifies),
            (observation, provenance, EngineeringRelationKind::TracesTo),
            (currentness, runtime, EngineeringRelationKind::CurrentnessFor),
            (applicability, context, EngineeringRelationKind::ApplicableTo),
            (statistics, disposition, EngineeringRelationKind::Derives),
        ];

        let mut graph = crate::ScientificLineageGraph::new();
        for (source, target, kind) in relations {
            graph.add_relation(EngineeringRelation::new(source, target, kind).unwrap());
        }
        graph.qualification_projection().unwrap()
    }

    #[test]
    fn checked_in_fixture_round_trips_through_rust_adapter() {
        let fixture = include_str!("../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json");
        let artifact: Cp04QualificationArtifact =
            serde_json::from_str(fixture).expect("checked-in CP-04 fixture must deserialize");
        artifact.validate().expect("checked-in CP-04 fixture must validate");

        let generated = Cp04QualificationArtifact::try_from_projection(
            &projection(),
        ).expect("fixture projection must remain adaptable");
        assert_eq!(artifact, generated);
        assert_eq!(artifact.canonical_bytes().unwrap(), generated.canonical_bytes().unwrap());
    }

    #[test]
    fn mutation_of_identity_field_is_rejected_even_before_envelope_hash_check() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.nodes[0].identity.canonical_identifier.push_str("-tampered");
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn mutation_of_content_digest_is_rejected_by_identity_validation() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.nodes[0].identity.content_digest = "c".repeat(64);
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn source_graph_mutation_cannot_preserve_artifact_identity() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        let original = artifact.artifact_digest.clone();
        artifact.source_graph_digest = "f".repeat(64);
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert_ne!(original, artifact.artifact_digest);
    }

    #[test]
    fn endpoint_mutation_cannot_be_hidden_by_recomputing_envelope_digest() {
        let projection = projection();
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();
        artifact.edges[0].target_identity_digest = artifact.nodes
            .iter()
            .find(|n| n.object_kind == "requirement")
            .unwrap()
            .identity_digest
            .clone();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(artifact.validate().is_err());
        assert!(matches!(
            artifact.validate_against_projection(&projection),
            Err(Cp04AdapterError::SourceMismatch)
        ));
    }

    #[test]
    fn relation_kind_mutation_cannot_be_hidden_by_recomputing_envelope_digest() {
        let projection = projection();
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();
        artifact.edges[0].edge_type = "requires".into();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(artifact.validate().is_err());
        assert!(matches!(
            artifact.validate_against_projection(&projection),
            Err(Cp04AdapterError::SourceMismatch)
        ));
    }

    #[test]
    fn qualification_relation_mutation_changes_all_downstream_identities() {
        let original = projection();
        let mut changed_graph = crate::ScientificLineageGraph::new();
        for (index, relation) in original.relations().cloned().enumerate() {
            if index == 0 {
                let changed_target = EngineeringObjectId::new(
                    relation.target.namespace.clone(),
                    relation.target.object_kind.clone(),
                    relation.target.canonical_identifier.clone(),
                    "2",
                    relation.target.content_digest.clone(),
                ).unwrap();
                changed_graph.add_relation(EngineeringRelation::new(
                    relation.source.clone(), changed_target, relation.kind,
                ).unwrap());
            } else {
                changed_graph.add_relation(relation);
            }
        }
        let changed = changed_graph.qualification_projection().unwrap();
        let original_artifact = Cp04QualificationArtifact::try_from_projection(&original).unwrap();
        let changed_artifact = Cp04QualificationArtifact::try_from_projection(&changed).unwrap();

        assert_ne!(original.source_graph_digest(), changed.source_graph_digest());
        assert_ne!(original.projection_digest(), changed.projection_digest());
        assert_ne!(original_artifact.artifact_digest, changed_artifact.artifact_digest);
    }

    #[test]
    fn projection_relation_mutation_is_detected_at_source_boundary() {
        let projection = projection();
        let artifact = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();
        let mut changed = projection.clone();
        changed.relations.pop();
        changed.projection_digest = changed.compute_projection_digest();
        assert!(changed.validate().is_ok());
        assert!(matches!(
            artifact.validate_against_projection(&changed),
            Err(Cp04AdapterError::SourceMismatch)
        ));
    }

    #[test]
    fn envelope_policy_failures_have_distinct_typed_errors() {
        let mut schema = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        schema.schema = "tampered.schema".into();
        assert!(matches!(schema.validate(), Err(Cp04AdapterError::InvalidSchema(_))));

        let mut cp04_schema = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        cp04_schema.cp04_schema = "tampered.cp04.schema".into();
        assert!(matches!(cp04_schema.validate(), Err(Cp04AdapterError::InvalidCp04Schema(_))));

        let mut claim = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        claim.claim_ceiling = "tampered claim ceiling".into();
        assert!(matches!(claim.validate(), Err(Cp04AdapterError::InvalidClaimCeiling)));
    }

    #[test]
    fn standalone_validation_rejects_noncanonical_node_and_edge_order() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.nodes.reverse();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(matches!(artifact.validate(), Err(Cp04AdapterError::UnorderedNodes)));

        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.edges.reverse();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(matches!(artifact.validate(), Err(Cp04AdapterError::UnorderedEdges)));
    }

    #[test]
    fn standalone_validation_rejects_duplicate_relation_identity() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.edges.push(artifact.edges[0].clone());
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(matches!(artifact.validate(), Err(Cp04AdapterError::DuplicateEdge(_))));
    }

    #[test]
    fn contract_vectors_pin_every_digest_boundary() {
        let fixture = include_str!("../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json");
        let vectors = include_str!("../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1-contract-vectors.json");
        let artifact: Cp04QualificationArtifact = serde_json::from_str(fixture).unwrap();
        let vectors: serde_json::Value = serde_json::from_str(vectors).unwrap();

        assert_eq!(artifact.source_graph_digest, vectors["source_graph_digest"]);
        assert_eq!(artifact.projection_digest, vectors["projection_digest"]);
        assert_eq!(artifact.artifact_digest, vectors["artifact_digest"]);

        let actual_identities: Vec<_> = artifact.nodes.iter().map(|n| n.identity_digest.clone()).collect();
        let mut expected_identities: Vec<String> = vectors["identity_digests"]
            .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
        let mut actual_identities = actual_identities;
        actual_identities.sort();
        expected_identities.sort();
        assert_eq!(actual_identities, expected_identities);

        let actual_relations: Vec<_> = artifact.edges.iter().map(|e| e.relation_digest.clone()).collect();
        let mut expected_relations: Vec<String> = vectors["relation_digests"]
            .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
        let mut actual_relations = actual_relations;
        actual_relations.sort();
        expected_relations.sort();
        assert_eq!(actual_relations, expected_relations);

        let unicode = &vectors["utf8_identity_vector"];
        let identity = EngineeringObjectId::new(
            unicode["namespace"].as_str().unwrap(),
            unicode["object_kind"].as_str().unwrap(),
            unicode["canonical_identifier"].as_str().unwrap(),
            unicode["version"].as_str().unwrap(),
            unicode["content_digest"].as_str().unwrap(),
        ).unwrap();
        assert_eq!(identity.identity_digest(), unicode["identity_digest"]);
    }

    #[test]
    fn adapter_preserves_identity_and_authority_ceiling() {
        let artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        assert_eq!(artifact.authority_ceiling, AuthorityCeiling::SyntheticQualification);
        assert_eq!(artifact.nodes.len(), 16);
        assert_eq!(artifact.edges.len(), 15);
        assert_eq!(artifact.projection_digest.len(), 64);
        artifact.validate().unwrap();
    }

    #[test]
    fn adapter_is_deterministic() {
        let a = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        let b = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.artifact_digest, b.artifact_digest);
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
    }

    #[test]
    fn standalone_validation_checks_edge_semantics_not_only_artifact_hash() {
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection()).unwrap();
        artifact.edges[0].edge_type = "implements".into();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn adapter_rejects_tampered_edge_even_if_envelope_digest_is_recomputed() {
        let projection = projection();
        let mut artifact = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();
        artifact.edges[0].edge_type = "implements".into();
        artifact.artifact_digest = artifact.compute_digest().unwrap();
        assert!(matches!(
            artifact.validate_against_projection(&projection),
            Err(Cp04AdapterError::SourceMismatch)
        ));
    }

    #[test]
    fn epistemic_relation_cannot_cross_adapter_boundary() {
        let source = object("model", "m", A);
        let target = object("model", "claim", B);
        let mut graph = crate::ScientificLineageGraph::new();
        graph.add_relation(EngineeringRelation::new(
            source,
            target,
            EngineeringRelationKind::Supports,
        ).unwrap());
        // The source projection contains no epistemic relation at all.
        let projection = graph.qualification_projection().unwrap();
        let artifact = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();
        assert!(artifact.edges.is_empty());
    }

    #[test]
    fn scientific_only_projection_is_rejected_rather_than_reinterpreted() {
        let source = object("simulation", "sim", A);
        let target = object("prediction", "pred", B);
        let mut graph = crate::ScientificLineageGraph::new();
        graph.add_relation(EngineeringRelation::new(
            source,
            target,
            EngineeringRelationKind::Produces,
        ).unwrap());
        let projection = graph.qualification_projection().unwrap();
        assert!(matches!(
            Cp04QualificationArtifact::try_from_projection(&projection),
            Err(Cp04AdapterError::UnsupportedNodeKind(kind)) if kind == "simulation"
        ));
    }
}
