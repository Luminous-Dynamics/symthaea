// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance-checked replication lineage assessment.
//!
//! Replication and independence are not booleans. This module compares a
//! declared `IndependenceProfile` with the shared ancestors actually visible in
//! a `ProvenanceGraph`. Absence of a shared root in one graph is reported only
//! as `NoSharedLineageObserved`; it is not proof of global independence and does
//! not grant scientific authority.

use crate::{
    FramedDigest, IndependenceDimension, IndependenceProfile, ProvenanceGraph, ResearchId,
    Sha256Digest, SharedRoot, SharedRootKind,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPLICATION_LINEAGE_SCHEMA: &str = "symthaea.replication-lineage.v1";
const ASSESSMENT_DOMAIN: &str = "symthaea.replication-lineage.identity.v1";
const REPORT_DOMAIN: &str = "symthaea.replication-lineage-report.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationLineageAssessment {
    pub schema_version: String,
    pub assessment_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    /// Commitment to what equivalence/claim/scope is being replicated.
    pub replication_scope_sha256: Sha256Digest,
    pub provenance_graph_sha256: Sha256Digest,
    pub target_node_id: ResearchId,
    pub replication_node_id: ResearchId,
    pub independence_profile: IndependenceProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationLineageIssue {
    WrongSchemaVersion { found: String },
    ProvenanceGraphBindingMismatch,
    SameTargetAndReplicationNode,
    UnknownTargetNode,
    UnknownReplicationNode,
    TargetSubjectMismatch,
    ReplicationSubjectMismatch,
    UndeclaredSharedAncestor { artifact_sha256: Sha256Digest },
    IndependenceDimensionContradictsSharedRoot {
        dimension: IndependenceDimension,
        root_kind: SharedRootKind,
    },
}

impl ReplicationLineageAssessment {
    pub fn validate_against(&self, graph: &ProvenanceGraph) -> Vec<ReplicationLineageIssue> {
        let mut issues = Vec::new();
        if self.schema_version != REPLICATION_LINEAGE_SCHEMA {
            issues.push(ReplicationLineageIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.provenance_graph_sha256 != graph.graph_sha256() {
            issues.push(ReplicationLineageIssue::ProvenanceGraphBindingMismatch);
        }
        if self.target_node_id == self.replication_node_id {
            issues.push(ReplicationLineageIssue::SameTargetAndReplicationNode);
        }

        match graph.node(&self.target_node_id) {
            None => issues.push(ReplicationLineageIssue::UnknownTargetNode),
            Some(node) if node.subject_sha256() != &self.subject_sha256 => {
                issues.push(ReplicationLineageIssue::TargetSubjectMismatch);
            }
            Some(_) => {}
        }
        match graph.node(&self.replication_node_id) {
            None => issues.push(ReplicationLineageIssue::UnknownReplicationNode),
            Some(node) if node.subject_sha256() != &self.subject_sha256 => {
                issues.push(ReplicationLineageIssue::ReplicationSubjectMismatch);
            }
            Some(_) => {}
        }

        if let Some(observed) = graph.shared_ancestor_artifacts(
            &self.target_node_id,
            &self.replication_node_id,
        ) {
            let declared = self
                .independence_profile
                .shared_roots()
                .iter()
                .map(|root| root.digest.clone())
                .collect::<BTreeSet<_>>();
            for artifact_sha256 in observed.difference(&declared) {
                issues.push(ReplicationLineageIssue::UndeclaredSharedAncestor {
                    artifact_sha256: artifact_sha256.clone(),
                });
            }
        }

        for dimension in self.independence_profile.dimensions() {
            for root in self.independence_profile.shared_roots() {
                if dimension_conflicts_with_root(*dimension, root.kind) {
                    issues.push(
                        ReplicationLineageIssue::IndependenceDimensionContradictsSharedRoot {
                            dimension: *dimension,
                            root_kind: root.kind,
                        },
                    );
                }
            }
        }
        issues
    }

    pub fn freeze_against(
        self,
        graph: &ProvenanceGraph,
    ) -> Result<FrozenReplicationLineageAssessment, Vec<ReplicationLineageIssue>> {
        let issues = self.validate_against(graph);
        if !issues.is_empty() {
            return Err(issues);
        }
        let assessment_sha256 = self.compute_digest();
        Ok(FrozenReplicationLineageAssessment {
            assessment: self,
            assessment_sha256,
        })
    }

    fn compute_digest(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(ASSESSMENT_DOMAIN);
        digest.text(REPLICATION_LINEAGE_SCHEMA);
        digest.text(self.assessment_id.as_str());
        digest.text(self.subject_sha256.as_str());
        digest.text(self.replication_scope_sha256.as_str());
        digest.text(self.provenance_graph_sha256.as_str());
        digest.text(self.target_node_id.as_str());
        digest.text(self.replication_node_id.as_str());
        for dimension in self.independence_profile.dimensions() {
            digest.text("dimension");
            digest.text(independence_dimension_tag(*dimension));
        }
        for root in self.independence_profile.shared_roots() {
            digest.text("shared-root");
            digest.text(shared_root_kind_tag(root.kind));
            digest.text(root.digest.as_str());
        }
        digest.digest()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenReplicationLineageAssessment {
    assessment: ReplicationLineageAssessment,
    assessment_sha256: Sha256Digest,
}

impl FrozenReplicationLineageAssessment {
    pub fn assessment(&self) -> &ReplicationLineageAssessment {
        &self.assessment
    }
    pub fn assessment_sha256(&self) -> &Sha256Digest {
        &self.assessment_sha256
    }
    pub fn report(&self, graph: &ProvenanceGraph) -> ReplicationLineageReport {
        ReplicationLineageReport::derive(self, graph)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReplicationLineageClosure {
    Replay,
    SharedLineage,
    NoSharedLineageObserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LineageEvidenceScope {
    /// The conclusion is limited to ancestry represented in this exact graph.
    GraphLocalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplicationLineageReport {
    assessment_sha256: Sha256Digest,
    provenance_graph_sha256: Sha256Digest,
    closure: ReplicationLineageClosure,
    evidence_scope: LineageEvidenceScope,
    observed_shared_ancestor_artifacts: BTreeSet<Sha256Digest>,
    declared_dimensions: BTreeSet<IndependenceDimension>,
    declared_shared_roots: BTreeSet<SharedRoot>,
    report_sha256: Sha256Digest,
}

impl ReplicationLineageReport {
    fn derive(
        frozen: &FrozenReplicationLineageAssessment,
        graph: &ProvenanceGraph,
    ) -> Self {
        let assessment = frozen.assessment();
        let observed = graph
            .shared_ancestor_artifacts(
                &assessment.target_node_id,
                &assessment.replication_node_id,
            )
            .unwrap_or_default();
        let closure = if assessment
            .independence_profile
            .dimensions()
            .contains(&IndependenceDimension::SameExecutionReplay)
        {
            ReplicationLineageClosure::Replay
        } else if !observed.is_empty() || !assessment.independence_profile.shared_roots().is_empty() {
            ReplicationLineageClosure::SharedLineage
        } else {
            ReplicationLineageClosure::NoSharedLineageObserved
        };
        let dimensions = assessment.independence_profile.dimensions().clone();
        let shared_roots = assessment.independence_profile.shared_roots().clone();
        let graph_sha256 = graph.graph_sha256();
        let report_sha256 = report_digest(
            frozen.assessment_sha256(),
            &graph_sha256,
            closure,
            &observed,
            &dimensions,
            &shared_roots,
        );
        Self {
            assessment_sha256: frozen.assessment_sha256().clone(),
            provenance_graph_sha256: graph_sha256,
            closure,
            evidence_scope: LineageEvidenceScope::GraphLocalOnly,
            observed_shared_ancestor_artifacts: observed,
            declared_dimensions: dimensions,
            declared_shared_roots: shared_roots,
            report_sha256,
        }
    }

    pub fn closure(&self) -> ReplicationLineageClosure {
        self.closure
    }
    pub fn evidence_scope(&self) -> LineageEvidenceScope {
        self.evidence_scope
    }
    pub fn observed_shared_ancestor_artifacts(&self) -> &BTreeSet<Sha256Digest> {
        &self.observed_shared_ancestor_artifacts
    }
    pub fn declared_dimensions(&self) -> &BTreeSet<IndependenceDimension> {
        &self.declared_dimensions
    }
    pub fn declared_shared_roots(&self) -> &BTreeSet<SharedRoot> {
        &self.declared_shared_roots
    }
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }
}

fn dimension_conflicts_with_root(
    dimension: IndependenceDimension,
    root_kind: SharedRootKind,
) -> bool {
    matches!(
        (dimension, root_kind),
        (
            IndependenceDimension::IndependentImplementationSharedInputs,
            SharedRootKind::Implementation
        ) | (
            IndependenceDimension::IndependentDataReduction,
            SharedRootKind::Preprocessing
        ) | (
            IndependenceDimension::IndependentDataset,
            SharedRootKind::RawData
        ) | (
            IndependenceDimension::IndependentOrganization,
            SharedRootKind::Organization
        )
    )
}

fn report_digest(
    assessment_sha256: &Sha256Digest,
    graph_sha256: &Sha256Digest,
    closure: ReplicationLineageClosure,
    observed: &BTreeSet<Sha256Digest>,
    dimensions: &BTreeSet<IndependenceDimension>,
    roots: &BTreeSet<SharedRoot>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(REPORT_DOMAIN);
    digest.text(assessment_sha256.as_str());
    digest.text(graph_sha256.as_str());
    digest.text(replication_closure_tag(closure));
    digest.text("graph-local-only");
    for artifact in observed {
        digest.text("observed-shared-ancestor");
        digest.text(artifact.as_str());
    }
    for dimension in dimensions {
        digest.text("declared-dimension");
        digest.text(independence_dimension_tag(*dimension));
    }
    for root in roots {
        digest.text("declared-shared-root");
        digest.text(shared_root_kind_tag(root.kind));
        digest.text(root.digest.as_str());
    }
    digest.digest()
}

const fn replication_closure_tag(value: ReplicationLineageClosure) -> &'static str {
    match value {
        ReplicationLineageClosure::Replay => "replay",
        ReplicationLineageClosure::SharedLineage => "shared-lineage",
        ReplicationLineageClosure::NoSharedLineageObserved => "no-shared-lineage-observed",
    }
}

const fn independence_dimension_tag(value: IndependenceDimension) -> &'static str {
    match value {
        IndependenceDimension::SameExecutionReplay => "same-execution-replay",
        IndependenceDimension::IndependentExecutionSameBinary => "independent-execution-same-binary",
        IndependenceDimension::IndependentImplementationSharedInputs => {
            "independent-implementation-shared-inputs"
        }
        IndependenceDimension::IndependentMethodSharedRawData => "independent-method-shared-raw-data",
        IndependenceDimension::IndependentDataReduction => "independent-data-reduction",
        IndependenceDimension::IndependentDataset => "independent-dataset",
        IndependenceDimension::IndependentSite => "independent-site",
        IndependenceDimension::IndependentOrganization => "independent-organization",
    }
}

const fn shared_root_kind_tag(value: SharedRootKind) -> &'static str {
    match value {
        SharedRootKind::RawData => "raw-data",
        SharedRootKind::Calibration => "calibration",
        SharedRootKind::Preprocessing => "preprocessing",
        SharedRootKind::Implementation => "implementation",
        SharedRootKind::SolverLibrary => "solver-library",
        SharedRootKind::ModelCheckpoint => "model-checkpoint",
        SharedRootKind::Organization => "organization",
        SharedRootKind::Instrument => "instrument",
        SharedRootKind::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthorityProfile, ProvenanceEdge, ProvenanceEdgeKind, ProvenanceNode,
        ProvenanceNodeKind,
    };

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }
    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }
    fn node(
        id_value: &str,
        kind: ProvenanceNodeKind,
        subject: &Sha256Digest,
        artifact: &str,
    ) -> ProvenanceNode {
        ProvenanceNode::new(
            id(id_value),
            kind,
            subject.clone(),
            sha(artifact),
            None,
            AuthorityProfile::empty(),
        )
    }
    fn edge(source: &str, target: &str) -> ProvenanceEdge {
        ProvenanceEdge {
            source: id(source),
            target: id(target),
            kind: ProvenanceEdgeKind::Consumes,
        }
    }
    fn shared_data_graph(subject: &Sha256Digest) -> ProvenanceGraph {
        ProvenanceGraph::new(
            [
                node("DATA", ProvenanceNodeKind::RawData, subject, "shared-data"),
                node("TARGET", ProvenanceNodeKind::Execution, subject, "target-result"),
                node("REPL", ProvenanceNodeKind::Execution, subject, "replication-result"),
            ],
            [edge("DATA", "TARGET"), edge("DATA", "REPL")],
        )
        .unwrap()
    }
    fn assessment(
        graph: &ProvenanceGraph,
        subject: &Sha256Digest,
        profile: IndependenceProfile,
    ) -> ReplicationLineageAssessment {
        ReplicationLineageAssessment {
            schema_version: REPLICATION_LINEAGE_SCHEMA.into(),
            assessment_id: id("REPL-ASSESS-001"),
            subject_sha256: subject.clone(),
            replication_scope_sha256: sha("same-scientific-claim"),
            provenance_graph_sha256: graph.graph_sha256(),
            target_node_id: id("TARGET"),
            replication_node_id: id("REPL"),
            independence_profile: profile,
        }
    }

    #[test]
    fn shared_graph_ancestor_must_be_declared() {
        let subject = sha("subject");
        let graph = shared_data_graph(&subject);
        let profile = IndependenceProfile::new(
            [IndependenceDimension::IndependentImplementationSharedInputs],
            [],
        )
        .unwrap();
        assert!(assessment(&graph, &subject, profile)
            .freeze_against(&graph)
            .unwrap_err()
            .iter()
            .any(|issue| matches!(
                issue,
                ReplicationLineageIssue::UndeclaredSharedAncestor { .. }
            )));
    }

    #[test]
    fn shared_data_is_visible_as_shared_lineage() {
        let subject = sha("subject");
        let graph = shared_data_graph(&subject);
        let data = sha("shared-data");
        let profile = IndependenceProfile::new(
            [IndependenceDimension::IndependentImplementationSharedInputs],
            [SharedRoot {
                kind: SharedRootKind::RawData,
                digest: data.clone(),
            }],
        )
        .unwrap();
        let report = assessment(&graph, &subject, profile)
            .freeze_against(&graph)
            .unwrap()
            .report(&graph);
        assert_eq!(report.closure(), ReplicationLineageClosure::SharedLineage);
        assert!(report.observed_shared_ancestor_artifacts().contains(&data));
        assert_eq!(report.evidence_scope(), LineageEvidenceScope::GraphLocalOnly);
    }

    #[test]
    fn independent_dataset_label_conflicts_with_shared_raw_data() {
        let subject = sha("subject");
        let graph = shared_data_graph(&subject);
        let profile = IndependenceProfile::new(
            [IndependenceDimension::IndependentDataset],
            [SharedRoot {
                kind: SharedRootKind::RawData,
                digest: sha("shared-data"),
            }],
        )
        .unwrap();
        assert!(assessment(&graph, &subject, profile)
            .freeze_against(&graph)
            .unwrap_err()
            .iter()
            .any(|issue| matches!(
                issue,
                ReplicationLineageIssue::IndependenceDimensionContradictsSharedRoot { .. }
            )));
    }

    #[test]
    fn no_shared_root_is_observation_not_global_independence_proof() {
        let subject = sha("subject");
        let graph = ProvenanceGraph::new(
            [
                node("DATA-A", ProvenanceNodeKind::RawData, &subject, "data-a"),
                node("DATA-B", ProvenanceNodeKind::RawData, &subject, "data-b"),
                node("TARGET", ProvenanceNodeKind::Execution, &subject, "target-result"),
                node("REPL", ProvenanceNodeKind::Execution, &subject, "replication-result"),
            ],
            [edge("DATA-A", "TARGET"), edge("DATA-B", "REPL")],
        )
        .unwrap();
        let profile = IndependenceProfile::new(
            [IndependenceDimension::IndependentDataset],
            [],
        )
        .unwrap();
        let report = assessment(&graph, &subject, profile)
            .freeze_against(&graph)
            .unwrap()
            .report(&graph);
        assert_eq!(
            report.closure(),
            ReplicationLineageClosure::NoSharedLineageObserved
        );
        assert_eq!(report.evidence_scope(), LineageEvidenceScope::GraphLocalOnly);
    }

    #[test]
    fn replay_is_not_independent_replication() {
        let subject = sha("subject");
        let graph = shared_data_graph(&subject);
        let profile = IndependenceProfile::new(
            [IndependenceDimension::SameExecutionReplay],
            [SharedRoot {
                kind: SharedRootKind::RawData,
                digest: sha("shared-data"),
            }],
        )
        .unwrap();
        let report = assessment(&graph, &subject, profile)
            .freeze_against(&graph)
            .unwrap()
            .report(&graph);
        assert_eq!(report.closure(), ReplicationLineageClosure::Replay);
    }

    #[test]
    fn assessment_cannot_compare_node_with_itself() {
        let subject = sha("subject");
        let graph = shared_data_graph(&subject);
        let profile = IndependenceProfile::new(
            [IndependenceDimension::SameExecutionReplay],
            [SharedRoot {
                kind: SharedRootKind::RawData,
                digest: sha("shared-data"),
            }],
        )
        .unwrap();
        let mut draft = assessment(&graph, &subject, profile);
        draft.replication_node_id = draft.target_node_id.clone();
        assert!(draft
            .freeze_against(&graph)
            .unwrap_err()
            .contains(&ReplicationLineageIssue::SameTargetAndReplicationNode));
    }
}
