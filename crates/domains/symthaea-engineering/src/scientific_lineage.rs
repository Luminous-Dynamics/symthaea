// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable cross-domain scientific lineage contract.
//!
//! This graph is deliberately richer than the deterministic qualification
//! projection: epistemic edges remain traversable in the scientific graph,
//! but are excluded from qualification invalidation closure.

use crate::{EngineeringObjectId, EngineeringRelation, EngineeringRelationKind};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
        Ok(QualificationProjection { relations, authority_ceiling: AuthorityCeiling::SyntheticQualification })
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

    /// Deterministic graph fingerprint for replay fixtures.
    pub fn graph_digest(&self) -> String {
        let mut relation_digests: Vec<_> =
            self.relations.iter().map(EngineeringRelation::relation_digest).collect();
        relation_digests.sort_unstable();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea.scientific-lineage.v1");
        for digest in relation_digests {
            bytes.extend_from_slice(digest.as_bytes());
            bytes.push(0);
        }

        use sha2::{Digest, Sha256};
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Explicit ceiling carried by every qualification projection.
///
/// This prevents a projection artifact from being interpreted as operational
/// authority or physical-performance evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityCeiling {
    SyntheticQualification,
}

/// Immutable deterministic projection consumed by downstream qualification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationProjection {
    relations: Vec<EngineeringRelation>,
    authority_ceiling: AuthorityCeiling,
}

impl QualificationProjection {
    pub fn relations(&self) -> impl Iterator<Item = &EngineeringRelation> {
        self.relations.iter()
    }

    pub const fn authority_ceiling(&self) -> AuthorityCeiling {
        self.authority_ceiling
    }

    /// Stable replay identity for the bounded qualification projection.
    pub fn projection_digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea.qualification-projection.v1");
        bytes.extend_from_slice(b"synthetic-qualification");
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
