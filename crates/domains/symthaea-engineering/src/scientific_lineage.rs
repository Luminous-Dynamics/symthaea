// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable cross-domain scientific lineage contract.
//!
//! This graph is deliberately richer than the deterministic qualification
//! projection: epistemic edges remain traversable in the scientific graph,
//! but are excluded from qualification invalidation closure.

use crate::{EngineeringObjectId, EngineeringRelation, EngineeringRelationKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const QUALIFICATION_PROJECTION_SCHEMA: &str = "symthaea.qualification-projection.v1";
pub const QUALIFICATION_POLICY: &str = "symthaea.qualification-policy.v1";
pub const QUALIFICATION_PROJECTION_ARTIFACT_SCHEMA: &str =
    "symthaea.qualification-projection-artifact.v1";

/// A recursive scientific graph. It permits cycles because the DKG/knowledge
/// layer is not itself a qualification DAG.
#[derive(Debug, Clone, Default)]
pub struct ScientificLineageGraph {
    nodes: BTreeMap<EngineeringObjectId, ()>,
    relations: Vec<EngineeringRelation>,
}

impl ScientificLineageGraph {
    pub fn new() -> Self { Self::default() }

    pub fn add_node(&mut self, node: EngineeringObjectId) {
        self.nodes.insert(node, ());
    }

    pub fn add_relation(&mut self, relation: EngineeringRelation) {
        self.nodes.insert(relation.source.clone(), ());
        self.nodes.insert(relation.target.clone(), ());
        self.relations.push(relation);
    }

    pub fn nodes(&self) -> impl Iterator<Item = &EngineeringObjectId> {
        self.nodes.keys()
    }

    pub fn relations(&self) -> impl Iterator<Item = &EngineeringRelation> {
        self.relations.iter()
    }

    pub fn relations_of_kind(
        &self,
        kind: EngineeringRelationKind,
    ) -> impl Iterator<Item = &EngineeringRelation> {
        self.relations.iter().filter(move |r| r.kind == kind)
    }

    /// Build an immutable, validated qualification projection.
    pub fn qualification_projection(&self) -> Result<QualificationProjection, QualificationCycle> {
        self.validate_qualification_acyclic()?;
        let mut relations: Vec<EngineeringRelation> = self.qualification_relations().cloned().collect();
        relations.sort_by(|a, b| a.relation_digest().cmp(&b.relation_digest()));
        let mut projection = QualificationProjection {
            schema: QUALIFICATION_PROJECTION_SCHEMA.to_owned(),
            source_graph_digest: self.graph_digest(),
            qualification_policy: QUALIFICATION_POLICY.to_owned(),
            relations,
            authority_ceiling: AuthorityCeiling::SyntheticQualification,
            projection_digest: String::new(),
        };
        projection.projection_digest = projection.compute_projection_digest();
        Ok(projection)
    }

    /// Project the graph to qualification-admissible relations.
    ///
    /// This is a semantic boundary, not a copy of the full DKG: epistemic
    /// relations such as \`supports\` and \`contradicts\` are intentionally absent.
    pub fn qualification_relations(&self) -> impl Iterator<Item = &EngineeringRelation> {
        self.relations.iter().filter(|r| r.admissible_for_qualification())
    }

    /// Validate that the qualification projection is acyclic.
    ///
    /// The full scientific/DKG graph may be cyclic. Only the explicitly
    /// admitted qualification projection is subject to DAG topology.
    pub fn validate_qualification_acyclic(&self) -> Result<(), QualificationCycle> {
        let mut indegree: BTreeMap<EngineeringObjectId, usize> =
            self.nodes.keys().cloned().map(|n| (n, 0)).collect();
        let mut outgoing: BTreeMap<EngineeringObjectId, Vec<EngineeringObjectId>> =
            BTreeMap::new();

        for relation in self.qualification_relations() {
            outgoing.entry(relation.source.clone()).or_default().push(relation.target.clone());
            *indegree.entry(relation.target.clone()).or_default() += 1;
        }

        let mut queue: VecDeque<_> = indegree
            .iter()
            .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
            .collect();
        let mut visited = 0usize;

        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(targets) = outgoing.get(&node) {
                for target in targets {
                    let degree = indegree.get_mut(target).expect("target exists in node set");
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target.clone());
                    }
                }
            }
        }

        if visited == indegree.len() {
            Ok(())
        } else {
            Err(QualificationCycle { node_count: indegree.len(), visited_count: visited })
        }
    }

    /// Compute deterministic downstream invalidation closure.
    ///
    /// Only qualification-admissible edges participate. Thus adding/removing
    /// epistemic evidence cannot silently invalidate an engineering artifact.
    pub fn qualification_closure(
        &self,
        source: &EngineeringObjectId,
    ) -> BTreeSet<EngineeringObjectId> {
        let mut closure = BTreeSet::new();
        let mut queue = VecDeque::from([source.clone()]);

        while let Some(current) = queue.pop_front() {
            if !closure.insert(current.clone()) {
                continue;
            }

            for relation in self.qualification_relations() {
                if relation.source == current && !closure.contains(&relation.target) {
                    queue.push_back(relation.target.clone());
                }
            }
        }

        closure
    }

    /// Deterministic fingerprint of the complete scientific graph.
    ///
    /// Nodes are included even when isolated, so replay identity cannot silently
    /// collapse two knowledge snapshots merely because they have the same edges.
    pub fn graph_digest(&self) -> String {
        let mut node_digests: Vec<_> = self.nodes.keys().map(EngineeringObjectId::identity_digest).collect();
        let mut relation_digests: Vec<_> =
            self.relations.iter().map(EngineeringRelation::relation_digest).collect();
        node_digests.sort_unstable();
        relation_digests.sort_unstable();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea.scientific-lineage.v2");
        bytes.extend_from_slice(b"nodes\0");
        for digest in node_digests {
            bytes.extend_from_slice(digest.as_bytes());
            bytes.push(0);
        }
        bytes.extend_from_slice(b"relations\0");
        for digest in relation_digests {
            bytes.extend_from_slice(digest.as_bytes());
            bytes.push(0);
        }

        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Explicit ceiling carried by every qualification projection.
///
/// This prevents a projection artifact from being interpreted as operational
/// authority or physical-performance evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityCeiling {
    SyntheticQualification,
}

/// Immutable deterministic projection consumed by downstream qualification.
///
/// The serialized form is a replay artifact: it records the exact source graph
/// snapshot, qualification policy, authority ceiling, and ordered relation set.
/// Deserialization validates the artifact before constructing the trusted type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "QualificationProjectionWire", into = "QualificationProjectionWire")]
pub struct QualificationProjection {
    schema: String,
    source_graph_digest: String,
    qualification_policy: String,
    relations: Vec<EngineeringRelation>,
    authority_ceiling: AuthorityCeiling,
    projection_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct QualificationProjectionWire {
    schema: String,
    source_graph_digest: String,
    qualification_policy: String,
    relations: Vec<EngineeringRelation>,
    authority_ceiling: AuthorityCeiling,
    projection_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationProjectionError {
    InvalidSchema,
    InvalidSourceGraphDigest,
    InvalidPolicy,
    InvalidAuthorityCeiling,
    InvalidRelation,
    UnorderedRelations,
    DuplicateRelation,
    InvalidProjectionDigest,
    CyclicProjection,
}

impl std::fmt::Display for QualificationProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid qualification projection: {:?}", self)
    }
}

impl std::error::Error for QualificationProjectionError {}

impl TryFrom<QualificationProjectionWire> for QualificationProjection {
    type Error = QualificationProjectionError;

    fn try_from(wire: QualificationProjectionWire) -> Result<Self, Self::Error> {
        let projection = Self {
            schema: wire.schema,
            source_graph_digest: wire.source_graph_digest,
            qualification_policy: wire.qualification_policy,
            relations: wire.relations,
            authority_ceiling: wire.authority_ceiling,
            projection_digest: wire.projection_digest,
        };
        projection.validate()?;
        Ok(projection)
    }
}

impl From<QualificationProjection> for QualificationProjectionWire {
    fn from(value: QualificationProjection) -> Self {
        Self {
            schema: value.schema,
            source_graph_digest: value.source_graph_digest,
            qualification_policy: value.qualification_policy,
            relations: value.relations,
            authority_ceiling: value.authority_ceiling,
            projection_digest: value.projection_digest,
        }
    }
}

impl QualificationProjection {
    pub fn relations(&self) -> impl Iterator<Item = &EngineeringRelation> {
        self.relations.iter()
    }

    pub const fn authority_ceiling(&self) -> AuthorityCeiling {
        self.authority_ceiling
    }

    pub fn source_graph_digest(&self) -> &str {
        &self.source_graph_digest
    }

    pub fn qualification_policy(&self) -> &str {
        &self.qualification_policy
    }

    /// Require an exact source knowledge snapshot before replay.
    pub fn source_graph_matches(&self, graph: &ScientificLineageGraph) -> bool {
        self.source_graph_digest == graph.graph_digest()
    }

    /// Deterministic bytes for durable storage or hand-off to CP-04.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(QUALIFICATION_PROJECTION_ARTIFACT_SCHEMA.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.schema.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.source_graph_digest.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.qualification_policy.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.projection_digest.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(b"synthetic-qualification");
        bytes.push(0);
        for relation in &self.relations {
            bytes.extend_from_slice(relation.relation_digest().as_bytes());
            bytes.push(0);
        }
        bytes
    }

    /// Validate and return the artifact bytes used by a downstream consumer.
    pub fn validated_canonical_bytes(&self) -> Result<Vec<u8>, QualificationProjectionError> {
        self.validate()?;
        Ok(self.canonical_bytes())
    }

    pub const fn schema(&self) -> &'static str {
        QUALIFICATION_PROJECTION_SCHEMA
    }

    /// Validate the replay artifact's semantic and cryptographic invariants.
    pub fn validate(&self) -> Result<(), QualificationProjectionError> {
        if self.schema != QUALIFICATION_PROJECTION_SCHEMA {
            return Err(QualificationProjectionError::InvalidSchema);
        }
        if !is_sha256_hex(&self.source_graph_digest) {
            return Err(QualificationProjectionError::InvalidSourceGraphDigest);
        }
        if self.qualification_policy != QUALIFICATION_POLICY {
            return Err(QualificationProjectionError::InvalidPolicy);
        }
        if self.authority_ceiling != AuthorityCeiling::SyntheticQualification {
            return Err(QualificationProjectionError::InvalidAuthorityCeiling);
        }

        let mut previous: Option<String> = None;
        let mut seen = BTreeSet::new();
        for relation in &self.relations {
            if !relation.admissible_for_qualification() {
                return Err(QualificationProjectionError::InvalidRelation);
            }
            EngineeringRelation::new(
                relation.source.clone(),
                relation.target.clone(),
                relation.kind,
            ).map_err(|_| QualificationProjectionError::InvalidRelation)?;
            let digest = relation.relation_digest();
            if !seen.insert(digest.clone()) {
                return Err(QualificationProjectionError::DuplicateRelation);
            }
            if previous.as_ref().is_some_and(|p| p >= &digest) {
                return Err(QualificationProjectionError::UnorderedRelations);
            }
            previous = Some(digest);
        }

        if !projection_is_acyclic(&self.relations) {
            return Err(QualificationProjectionError::CyclicProjection);
        }
        if !is_sha256_hex(&self.projection_digest)
            || self.projection_digest != self.compute_projection_digest()
        {
            return Err(QualificationProjectionError::InvalidProjectionDigest);
        }
        Ok(())
    }

    /// Stable identity of the bounded projection. The source graph snapshot is
    /// deliberately recorded separately: epistemic-only DKG mutations change
    /// the source snapshot without changing qualification projection identity.
    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    fn compute_projection_digest(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(QUALIFICATION_PROJECTION_SCHEMA.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.qualification_policy.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(b"synthetic-qualification");
        bytes.push(0);
        for relation in &self.relations {
            bytes.extend_from_slice(relation.relation_digest().as_bytes());
            bytes.push(0);
        }
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn closure(&self, source: &EngineeringObjectId) -> BTreeSet<EngineeringObjectId> {
        let mut closure = BTreeSet::new();
        let mut queue = VecDeque::from([source.clone()]);
        while let Some(current) = queue.pop_front() {
            if !closure.insert(current.clone()) { continue; }
            for relation in &self.relations {
                if relation.source == current && !closure.contains(&relation.target) {
                    queue.push_back(relation.target.clone());
                }
            }
        }
        closure
    }
}

fn projection_is_acyclic(relations: &[EngineeringRelation]) -> bool {
    let mut indegree: BTreeMap<EngineeringObjectId, usize> = BTreeMap::new();
    let mut outgoing: BTreeMap<EngineeringObjectId, Vec<EngineeringObjectId>> = BTreeMap::new();

    for relation in relations {
        indegree.entry(relation.source.clone()).or_default();
        *indegree.entry(relation.target.clone()).or_default() += 1;
        outgoing.entry(relation.source.clone()).or_default().push(relation.target.clone());
    }

    let mut queue: VecDeque<_> = indegree.iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect();
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(targets) = outgoing.get(&node) {
            for target in targets {
                let degree = indegree.get_mut(target).expect("target exists");
                *degree -= 1;
                if *degree == 0 { queue.push_back(target.clone()); }
            }
        }
    }
    visited == indegree.len()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationCycle {
    pub node_count: usize,
    pub visited_count: usize,
}

impl std::fmt::Display for QualificationCycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qualification projection contains a cycle: visited {} of {} nodes", self.visited_count, self.node_count)
    }
}

impl std::error::Error for QualificationCycle {}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn object(kind: &str, id: &str, digest: &str) -> EngineeringObjectId {
        EngineeringObjectId::new("scientific-fixture", kind, id, "1", digest).unwrap()
    }

    fn rel(
        source: &EngineeringObjectId,
        target: &EngineeringObjectId,
        kind: EngineeringRelationKind,
    ) -> EngineeringRelation {
        EngineeringRelation::new(source.clone(), target.clone(), kind).unwrap()
    }

    fn fixture() -> ScientificLineageGraph {
        let theorem = object("theorem", "energy-balance", A);
        let physical_model = object("physical_model", "thermal-model", B);
        let simulation = object("simulation", "thermal-run", A);
        let prediction = object("prediction", "strength-prediction", B);
        let claim = object("scientific_claim", "claim-1", A);
        let process = object("process", "sintering-route", B);
        let material = object("material", "candidate-1", A);
        let predicted_property = object("predicted_property", "strength-1", B);
        let experiment = object("experiment", "test-1", A);
        let observation = object("observation", "obs-1", B);
        let uncertainty = object("uncertainty", "u-1", A);

        let mut graph = ScientificLineageGraph::new();
        graph.add_relation(rel(&theorem, &physical_model, EngineeringRelationKind::Constrains));
        graph.add_relation(rel(&simulation, &physical_model, EngineeringRelationKind::Instantiates));
        graph.add_relation(rel(&simulation, &prediction, EngineeringRelationKind::Produces));
        graph.add_relation(rel(&prediction, &claim, EngineeringRelationKind::Supports));
        graph.add_relation(rel(&process, &material, EngineeringRelationKind::Produces));
        graph.add_relation(rel(&material, &predicted_property, EngineeringRelationKind::HasProperty));
        graph.add_relation(rel(&experiment, &observation, EngineeringRelationKind::Observes));
        graph.add_relation(rel(&observation, &uncertainty, EngineeringRelationKind::Quantifies));
        graph
    }

    #[test]
    fn executable_fixture_covers_all_cross_domain_boundaries() {
        let graph = fixture();
        assert_eq!(graph.relations().count(), 8);
        assert_eq!(graph.relations_of_kind(EngineeringRelationKind::Supports).count(), 1);
        assert_eq!(graph.relations_of_kind(EngineeringRelationKind::Quantifies).count(), 1);
    }

    #[test]
    fn epistemic_support_is_in_graph_but_never_in_qualification_closure() {
        let graph = fixture();
        let prediction = object("prediction", "strength-prediction", B);
        let claim = object("scientific_claim", "claim-1", A);

        assert_eq!(graph.relations_of_kind(EngineeringRelationKind::Supports).count(), 1);
        assert!(graph.qualification_relations().all(|r| r.kind != EngineeringRelationKind::Supports));

        let closure = graph.qualification_closure(&prediction);
        assert!(closure.contains(&prediction));
        assert!(!closure.contains(&claim));
    }

    #[test]
    fn qualification_closure_reaches_only_semantically_admitted_dependents() {
        let graph = fixture();
        let simulation = object("simulation", "thermal-run", A);
        let prediction = object("prediction", "strength-prediction", B);
        let claim = object("scientific_claim", "claim-1", A);
        let physical_model = object("physical_model", "thermal-model", B);

        let closure = graph.qualification_closure(&simulation);
        assert!(closure.contains(&simulation));
        assert!(closure.contains(&prediction));
        assert!(closure.contains(&physical_model));
        assert!(!closure.contains(&claim));
    }

    #[test]
    fn qualification_projection_is_acyclic_for_scientific_fixture() {
        assert!(fixture().validate_qualification_acyclic().is_ok());
        let projection = fixture().qualification_projection().unwrap();
        assert_eq!(projection.authority_ceiling(), AuthorityCeiling::SyntheticQualification);
        assert_eq!(projection.relations().count(), 7);
    }

    #[test]
    fn dkg_graph_can_contain_epistemic_cycle_without_making_projection_cyclic() {
        let prediction = object("prediction", "p", A);
        let claim = object("scientific_claim", "c", B);
        let mut graph = ScientificLineageGraph::new();

        graph.add_relation(rel(&prediction, &claim, EngineeringRelationKind::Supports));
        graph.add_relation(rel(&claim, &prediction, EngineeringRelationKind::Contradicts));

        assert_eq!(graph.relations().count(), 2);
        assert_eq!(graph.qualification_relations().count(), 0);
        assert!(graph.validate_qualification_acyclic().is_ok());
        assert_eq!(graph.qualification_closure(&prediction).len(), 1);
    }

    #[test]
    fn qualification_cycle_is_rejected_even_when_dkg_cycle_is_allowed() {
        let simulation = object("simulation", "s", A);
        let model = object("model", "m", B);
        let parameters = object("model_parameters", "p", A);
        let mut graph = ScientificLineageGraph::new();
        graph.add_relation(rel(&simulation, &model, EngineeringRelationKind::Instantiates));
        graph.add_relation(rel(&model, &parameters, EngineeringRelationKind::Parameterizes));
        graph.add_relation(rel(&parameters, &simulation, EngineeringRelationKind::DependsOn));

        let err = graph.validate_qualification_acyclic().unwrap_err();
        assert!(err.visited_count < err.node_count);
    }

    #[test]
    fn projection_digest_is_deterministic_and_excludes_epistemic_edges() {
        let graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        assert_eq!(projection.projection_digest(), graph.qualification_projection().unwrap().projection_digest());
        assert!(projection.relations().all(|r| r.kind != EngineeringRelationKind::Supports && r.kind != EngineeringRelationKind::Contradicts));
    }

    #[test]
    fn projection_closure_matches_graph_qualification_closure() {
        let graph = fixture();
        let simulation = object("simulation", "thermal-run", A);
        let projection = graph.qualification_projection().unwrap();
        assert_eq!(projection.closure(&simulation), graph.qualification_closure(&simulation));
    }

    #[test]
    fn adding_epistemic_evidence_does_not_change_projection_identity() {
        let prediction = object("prediction", "p", A);
        let claim = object("scientific_claim", "c", B);
        let mut graph = ScientificLineageGraph::new();
        graph.add_relation(rel(&prediction, &claim, EngineeringRelationKind::Supports));
        let before = graph.qualification_projection().unwrap().projection_digest();
        graph.add_relation(rel(&claim, &prediction, EngineeringRelationKind::Contradicts));
        assert_eq!(before, graph.qualification_projection().unwrap().projection_digest());
    }


    #[test]
    fn canonical_artifact_bytes_are_deterministic_and_validation_gated() {
        let graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        assert_eq!(projection.canonical_bytes(), projection.canonical_bytes());
        assert_eq!(projection.validated_canonical_bytes().unwrap(), projection.canonical_bytes());
    }

    #[test]
    fn projection_is_serializable_and_validated_on_deserialization() {
        let graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        let json = serde_json::to_string(&projection).unwrap();
        let restored: QualificationProjection = serde_json::from_str(&json).unwrap();
        assert_eq!(projection, restored);
        assert!(restored.validate().is_ok());
    }

    #[test]
    fn projection_records_source_snapshot_separately_from_projection_identity() {
        let prediction = object("prediction", "p", A);
        let claim = object("scientific_claim", "c", B);
        let mut graph = ScientificLineageGraph::new();
        graph.add_relation(rel(&prediction, &claim, EngineeringRelationKind::Supports));
        let before = graph.qualification_projection().unwrap();
        graph.add_relation(rel(&claim, &prediction, EngineeringRelationKind::Contradicts));
        let after = graph.qualification_projection().unwrap();
        assert_ne!(before.source_graph_digest(), after.source_graph_digest());
        assert_eq!(before.projection_digest(), after.projection_digest());
    }

    #[test]
    fn isolated_node_changes_source_graph_identity() {
        let mut graph = fixture();
        let before = graph.graph_digest();
        graph.add_node(object("definition", "isolated", A));
        assert_ne!(before, graph.graph_digest());
    }

    #[test]
    fn replay_requires_exact_source_graph_snapshot() {
        let mut graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        assert!(projection.source_graph_matches(&graph));

        graph.add_relation(rel(
            &object("prediction", "extra", A),
            &object("scientific_claim", "extra-claim", B),
            EngineeringRelationKind::Supports,
        ));
        assert!(!projection.source_graph_matches(&graph));
    }

    #[test]
    fn tampered_projection_is_rejected() {
        let graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        let mut json = serde_json::to_value(&projection).unwrap();
        json["qualification_policy"] = serde_json::json!("tampered-policy");
        let result: Result<QualificationProjection, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_relation_is_rejected_by_replay_digest() {
        let graph = fixture();
        let projection = graph.qualification_projection().unwrap();
        let mut json = serde_json::to_value(&projection).unwrap();
        let relation = json["relations"][0]["source"]["canonical_identifier"]
            .as_str()
            .unwrap()
            .to_owned();
        json["relations"][0]["source"]["canonical_identifier"] =
            serde_json::json!(format!("{relation}-tampered"));
        let result: Result<QualificationProjection, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn graph_digest_is_deterministic() {
        assert_eq!(fixture().graph_digest(), fixture().graph_digest());
    }

    #[test]
    fn changing_epistemic_evidence_changes_dkg_digest_without_changing_qualification_closure() {
        let prediction = object("prediction", "p", A);
        let claim = object("scientific_claim", "c", B);
        let mut graph = ScientificLineageGraph::new();

        graph.add_relation(rel(&prediction, &claim, EngineeringRelationKind::Supports));
        let before_digest = graph.graph_digest();
        let before_closure = graph.qualification_closure(&prediction);

        graph.add_relation(rel(&claim, &prediction, EngineeringRelationKind::Contradicts));

        assert_ne!(before_digest, graph.graph_digest());
        assert_eq!(before_closure, graph.qualification_closure(&prediction));
    }
}
