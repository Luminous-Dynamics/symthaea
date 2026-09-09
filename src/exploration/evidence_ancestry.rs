// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transitive ancestry analysis for declared evidence lineages.
//!
//! Distinct immediate method/data lineage labels can still descend from a shared upstream
//! root. This module follows declared ancestry transitively and can require root diversity
//! before a candidate reaches Pareto comparison.
//!
//! Ancestry remains declared rather than authenticated. Root diversity is not a truth,
//! replication, confidence, reputation, or authority score.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::archive::{ArchiveEntry, ArchiveInsertOutcome, ParetoPolicy};
use super::evidence_admission::{
    DimensionEvidenceBinding, EvidenceAdmissionPolicy, EvidenceSupportReport,
};
use super::evidence_independence::{
    assess_declared_independence, DeclaredIndependenceArchive, DeclaredIndependencePolicy,
    DeclaredIndependenceReport, EvidenceLineageDeclaration, IndependenceError,
    MAX_LINEAGE_ID_LEN,
};
use super::persisted_evidence::VerifiedGenerativityBundle;

pub const MAX_ANCESTRY_EDGES: usize = 4_096;
pub const MAX_PARENTS_PER_LINEAGE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LineageDomain {
    Method,
    Source,
}

/// Declared child→parent lineage relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageAncestryEdge {
    pub domain: LineageDomain,
    pub child_id: String,
    pub parent_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitiveAncestryPolicy {
    pub min_method_root_lineages_per_dimension: usize,
    pub min_source_root_lineages_per_dimension: usize,
    pub max_edges: usize,
    pub max_parents_per_lineage: usize,
}

impl Default for TransitiveAncestryPolicy {
    fn default() -> Self {
        Self {
            min_method_root_lineages_per_dimension: 1,
            min_source_root_lineages_per_dimension: 1,
            max_edges: 1_024,
            max_parents_per_lineage: 8,
        }
    }
}

impl TransitiveAncestryPolicy {
    pub fn validate(&self) -> Result<(), AncestryError> {
        if self.max_edges == 0 || self.max_edges > MAX_ANCESTRY_EDGES {
            return Err(AncestryError::InvalidEdgeLimit(self.max_edges));
        }
        if self.max_parents_per_lineage == 0
            || self.max_parents_per_lineage > MAX_PARENTS_PER_LINEAGE
        {
            return Err(AncestryError::InvalidParentLimit(
                self.max_parents_per_lineage,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootReuse {
    pub root_id: String,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DimensionAncestryDiversity {
    pub dimension: super::evidence_admission::GenerativityDimension,
    pub method_root_lineages: Vec<String>,
    pub source_root_lineages: Vec<String>,
    /// Roots reached by more than one qualified evidence item for this dimension.
    pub reused_method_roots: Vec<RootReuse>,
    pub reused_source_roots: Vec<RootReuse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransitiveAncestryReport {
    pub assessment_subject_id: String,
    pub dimensions: Vec<DimensionAncestryDiversity>,
}

#[derive(Debug)]
pub enum AncestryError {
    InvalidEdgeLimit(usize),
    InvalidParentLimit(usize),
    TooManyEdges { count: usize, max: usize },
    EmptyLineageId(&'static str),
    UnboundedLineageId { field: &'static str, len: usize },
    SelfParent { domain: LineageDomain, id: String },
    DuplicateEdge {
        domain: LineageDomain,
        child: String,
        parent: String,
    },
    TooManyParents {
        domain: LineageDomain,
        child: String,
        count: usize,
        max: usize,
    },
    DuplicateDeclaration(String),
    DuplicateSourceLineage {
        evidence_id: String,
        lineage_id: String,
    },
    MissingDeclaration(String),
    ReachableCycle {
        domain: LineageDomain,
        nodes: Vec<String>,
    },
    InsufficientMethodRootDiversity {
        dimension: super::evidence_admission::GenerativityDimension,
        required: usize,
        observed: usize,
    },
    InsufficientSourceRootDiversity {
        dimension: super::evidence_admission::GenerativityDimension,
        required: usize,
        observed: usize,
    },
    Independence(IndependenceError),
}

impl From<IndependenceError> for AncestryError {
    fn from(value: IndependenceError) -> Self {
        Self::Independence(value)
    }
}

impl std::fmt::Display for AncestryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEdgeLimit(value) => write!(
                f,
                "ancestry edge limit must be within 1..={MAX_ANCESTRY_EDGES}, got {value}"
            ),
            Self::InvalidParentLimit(value) => write!(
                f,
                "ancestry parent limit must be within 1..={MAX_PARENTS_PER_LINEAGE}, got {value}"
            ),
            Self::TooManyEdges { count, max } => {
                write!(f, "ancestry graph has {count} edges; maximum is {max}")
            }
            Self::EmptyLineageId(field) => write!(f, "ancestry field is empty: {field}"),
            Self::UnboundedLineageId { field, len } => write!(
                f,
                "ancestry field {field} is {len} bytes; maximum is {MAX_LINEAGE_ID_LEN}"
            ),
            Self::SelfParent { domain, id } => {
                write!(f, "{domain:?} lineage cannot be its own parent: {id}")
            }
            Self::DuplicateEdge {
                domain,
                child,
                parent,
            } => write!(
                f,
                "duplicate {domain:?} ancestry edge: {child} -> {parent}"
            ),
            Self::TooManyParents {
                domain,
                child,
                count,
                max,
            } => write!(
                f,
                "{domain:?} lineage {child} has {count} parents; maximum is {max}"
            ),
            Self::DuplicateDeclaration(id) => {
                write!(f, "duplicate evidence-lineage declaration: {id}")
            }
            Self::DuplicateSourceLineage {
                evidence_id,
                lineage_id,
            } => write!(
                f,
                "evidence {evidence_id} repeats source lineage {lineage_id}"
            ),
            Self::MissingDeclaration(id) => {
                write!(f, "missing lineage declaration for qualified evidence {id}")
            }
            Self::ReachableCycle { domain, nodes } => write!(
                f,
                "reachable {domain:?} ancestry contains a cycle involving: {}",
                nodes.join(", ")
            ),
            Self::InsufficientMethodRootDiversity {
                dimension,
                required,
                observed,
            } => write!(
                f,
                "dimension {dimension:?} requires {required} declared method roots, observed {observed}"
            ),
            Self::InsufficientSourceRootDiversity {
                dimension,
                required,
                observed,
            } => write!(
                f,
                "dimension {dimension:?} requires {required} declared source roots, observed {observed}"
            ),
            Self::Independence(error) => write!(f, "declared-independence gate failed: {error}"),
        }
    }
}

impl std::error::Error for AncestryError {}

/// Evaluate transitive root diversity over a previously qualified declared-independence
/// report. Only ancestry reachable from evidence used in that report participates.
pub fn assess_transitive_ancestry(
    base: &DeclaredIndependenceReport,
    declarations: &[EvidenceLineageDeclaration],
    edges: &[LineageAncestryEdge],
    policy: TransitiveAncestryPolicy,
) -> Result<TransitiveAncestryReport, AncestryError> {
    policy.validate()?;
    if edges.len() > policy.max_edges {
        return Err(AncestryError::TooManyEdges {
            count: edges.len(),
            max: policy.max_edges,
        });
    }

    let declaration_by_id = declaration_map(declarations)?;
    let parents = build_parent_maps(edges, policy.max_parents_per_lineage)?;

    let used_ids = base
        .dimensions
        .iter()
        .flat_map(|dimension| dimension.qualified_evidence_ids.iter())
        .cloned()
        .collect::<HashSet<_>>();

    for evidence_id in &used_ids {
        if !declaration_by_id.contains_key(evidence_id) {
            return Err(AncestryError::MissingDeclaration(evidence_id.clone()));
        }
    }

    let method_starts = used_ids
        .iter()
        .filter_map(|id| declaration_by_id.get(id).map(|d| d.method_lineage_id.clone()))
        .collect::<HashSet<_>>();
    let source_starts = used_ids
        .iter()
        .flat_map(|id| {
            declaration_by_id
                .get(id)
                .into_iter()
                .flat_map(|d| d.source_lineage_ids.iter().cloned())
        })
        .collect::<HashSet<_>>();

    validate_reachable_acyclic(LineageDomain::Method, &method_starts, &parents)?;
    validate_reachable_acyclic(LineageDomain::Source, &source_starts, &parents)?;

    let mut dimensions = Vec::with_capacity(base.dimensions.len());
    for dimension in &base.dimensions {
        let mut method_root_use = HashMap::<String, HashSet<String>>::new();
        let mut source_root_use = HashMap::<String, HashSet<String>>::new();

        for evidence_id in &dimension.qualified_evidence_ids {
            let declaration = declaration_by_id
                .get(evidence_id)
                .ok_or_else(|| AncestryError::MissingDeclaration(evidence_id.clone()))?;

            for root in roots_for(
                LineageDomain::Method,
                &declaration.method_lineage_id,
                &parents,
            ) {
                method_root_use
                    .entry(root)
                    .or_default()
                    .insert(evidence_id.clone());
            }

            for source in &declaration.source_lineage_ids {
                for root in roots_for(LineageDomain::Source, source, &parents) {
                    source_root_use
                        .entry(root)
                        .or_default()
                        .insert(evidence_id.clone());
                }
            }
        }

        let mut method_root_lineages = method_root_use.keys().cloned().collect::<Vec<_>>();
        method_root_lineages.sort();
        let mut source_root_lineages = source_root_use.keys().cloned().collect::<Vec<_>>();
        source_root_lineages.sort();

        if method_root_lineages.len() < policy.min_method_root_lineages_per_dimension {
            return Err(AncestryError::InsufficientMethodRootDiversity {
                dimension: dimension.dimension,
                required: policy.min_method_root_lineages_per_dimension,
                observed: method_root_lineages.len(),
            });
        }
        if source_root_lineages.len() < policy.min_source_root_lineages_per_dimension {
            return Err(AncestryError::InsufficientSourceRootDiversity {
                dimension: dimension.dimension,
                required: policy.min_source_root_lineages_per_dimension,
                observed: source_root_lineages.len(),
            });
        }

        dimensions.push(DimensionAncestryDiversity {
            dimension: dimension.dimension,
            method_root_lineages,
            source_root_lineages,
            reused_method_roots: reused_roots(method_root_use),
            reused_source_roots: reused_roots(source_root_use),
        });
    }

    Ok(TransitiveAncestryReport {
        assessment_subject_id: base.assessment_subject_id.clone(),
        dimensions,
    })
}

fn declaration_map<'a>(
    declarations: &'a [EvidenceLineageDeclaration],
) -> Result<HashMap<String, &'a EvidenceLineageDeclaration>, AncestryError> {
    let mut result = HashMap::new();
    for declaration in declarations {
        validate_id("evidence_id", &declaration.evidence_id)?;
        validate_id("method_lineage_id", &declaration.method_lineage_id)?;
        let mut seen_sources = HashSet::new();
        for source in &declaration.source_lineage_ids {
            validate_id("source_lineage_id", source)?;
            if !seen_sources.insert(source.as_str()) {
                return Err(AncestryError::DuplicateSourceLineage {
                    evidence_id: declaration.evidence_id.clone(),
                    lineage_id: source.clone(),
                });
            }
        }
        if result
            .insert(declaration.evidence_id.clone(), declaration)
            .is_some()
        {
            return Err(AncestryError::DuplicateDeclaration(
                declaration.evidence_id.clone(),
            ));
        }
    }
    Ok(result)
}

type ParentMaps = HashMap<LineageDomain, HashMap<String, Vec<String>>>;

fn build_parent_maps(
    edges: &[LineageAncestryEdge],
    max_parents: usize,
) -> Result<ParentMaps, AncestryError> {
    let mut maps = ParentMaps::new();
    let mut seen = HashSet::new();

    for edge in edges {
        validate_id("child_id", &edge.child_id)?;
        validate_id("parent_id", &edge.parent_id)?;
        if edge.child_id == edge.parent_id {
            return Err(AncestryError::SelfParent {
                domain: edge.domain,
                id: edge.child_id.clone(),
            });
        }
        if !seen.insert((edge.domain, edge.child_id.clone(), edge.parent_id.clone())) {
            return Err(AncestryError::DuplicateEdge {
                domain: edge.domain,
                child: edge.child_id.clone(),
                parent: edge.parent_id.clone(),
            });
        }

        let parents = maps
            .entry(edge.domain)
            .or_default()
            .entry(edge.child_id.clone())
            .or_default();
        parents.push(edge.parent_id.clone());
        if parents.len() > max_parents {
            return Err(AncestryError::TooManyParents {
                domain: edge.domain,
                child: edge.child_id.clone(),
                count: parents.len(),
                max: max_parents,
            });
        }
        parents.sort();
    }

    Ok(maps)
}

fn validate_reachable_acyclic(
    domain: LineageDomain,
    starts: &HashSet<String>,
    maps: &ParentMaps,
) -> Result<(), AncestryError> {
    let parent_map = maps.get(&domain);
    let mut reachable = HashSet::new();
    let mut stack = starts.iter().cloned().collect::<Vec<_>>();

    while let Some(node) = stack.pop() {
        if !reachable.insert(node.clone()) {
            continue;
        }
        if let Some(parents) = parent_map.and_then(|map| map.get(&node)) {
            stack.extend(parents.iter().cloned());
        }
    }

    if reachable.is_empty() {
        return Ok(());
    }

    let mut indegree = reachable
        .iter()
        .map(|node| (node.clone(), 0usize))
        .collect::<HashMap<_, _>>();
    let mut children_by_parent = HashMap::<String, Vec<String>>::new();

    if let Some(parent_map) = parent_map {
        for child in &reachable {
            if let Some(parents) = parent_map.get(child) {
                for parent in parents {
                    if reachable.contains(parent) {
                        *indegree.get_mut(parent).expect("reachable parent exists") += 1;
                        children_by_parent
                            .entry(child.clone())
                            .or_default()
                            .push(parent.clone());
                    }
                }
            }
        }
    }

    let mut zero = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect::<Vec<_>>();
    let mut processed = 0usize;

    while let Some(node) = zero.pop() {
        processed += 1;
        for parent in children_by_parent.get(&node).into_iter().flatten() {
            let degree = indegree.get_mut(parent).expect("parent indegree exists");
            *degree -= 1;
            if *degree == 0 {
                zero.push(parent.clone());
            }
        }
    }

    if processed != reachable.len() {
        let mut cycle_nodes = indegree
            .into_iter()
            .filter_map(|(node, degree)| (degree > 0).then_some(node))
            .collect::<Vec<_>>();
        cycle_nodes.sort();
        return Err(AncestryError::ReachableCycle {
            domain,
            nodes: cycle_nodes,
        });
    }

    Ok(())
}

fn roots_for(domain: LineageDomain, start: &str, maps: &ParentMaps) -> Vec<String> {
    let parent_map = maps.get(&domain);
    let mut roots = HashSet::new();
    let mut visited = HashSet::new();
    let mut stack = vec![start.to_string()];

    while let Some(node) = stack.pop() {
        if !visited.insert(node.clone()) {
            continue;
        }
        match parent_map.and_then(|map| map.get(&node)) {
            Some(parents) if !parents.is_empty() => stack.extend(parents.iter().cloned()),
            _ => {
                roots.insert(node);
            }
        }
    }

    let mut roots = roots.into_iter().collect::<Vec<_>>();
    roots.sort();
    roots
}

fn reused_roots(root_use: HashMap<String, HashSet<String>>) -> Vec<RootReuse> {
    let mut result = root_use
        .into_iter()
        .filter_map(|(root_id, evidence_ids)| {
            (evidence_ids.len() > 1).then(|| {
                let mut evidence_ids = evidence_ids.into_iter().collect::<Vec<_>>();
                evidence_ids.sort();
                RootReuse {
                    root_id,
                    evidence_ids,
                }
            })
        })
        .collect::<Vec<_>>();
    result.sort_by(|a, b| a.root_id.cmp(&b.root_id));
    result
}

fn validate_id(field: &'static str, value: &str) -> Result<(), AncestryError> {
    if value.trim().is_empty() {
        return Err(AncestryError::EmptyLineageId(field));
    }
    if value.len() > MAX_LINEAGE_ID_LEN {
        return Err(AncestryError::UnboundedLineageId {
            field,
            len: value.len(),
        });
    }
    Ok(())
}

/// Full admission stack through transitive declared ancestry.
#[derive(Debug, Clone, Serialize)]
pub struct TransitiveAncestryArchive {
    inner: DeclaredIndependenceArchive,
    evidence_policy: EvidenceAdmissionPolicy,
    independence_policy: DeclaredIndependencePolicy,
    ancestry_policy: TransitiveAncestryPolicy,
    ancestry_report_by_entry: HashMap<String, TransitiveAncestryReport>,
}

impl TransitiveAncestryArchive {
    pub fn new(
        pareto_policy: ParetoPolicy,
        evidence_policy: EvidenceAdmissionPolicy,
        independence_policy: DeclaredIndependencePolicy,
        ancestry_policy: TransitiveAncestryPolicy,
    ) -> Result<Self, AncestryError> {
        ancestry_policy.validate()?;
        Ok(Self {
            inner: DeclaredIndependenceArchive::new(
                pareto_policy,
                evidence_policy,
                independence_policy,
            )?,
            evidence_policy,
            independence_policy,
            ancestry_policy,
            ancestry_report_by_entry: HashMap::new(),
        })
    }

    pub fn entries(&self) -> &[ArchiveEntry] {
        self.inner.entries()
    }

    pub fn evidence_support_for(&self, entry_id: &str) -> Option<&EvidenceSupportReport> {
        self.inner.evidence_support_for(entry_id)
    }

    pub fn ancestry_report_for(&self, entry_id: &str) -> Option<&TransitiveAncestryReport> {
        self.ancestry_report_by_entry.get(entry_id)
    }

    pub fn qualify_and_insert(
        &mut self,
        entry_id: impl Into<String>,
        niche: impl Into<String>,
        bundle: &VerifiedGenerativityBundle,
        bindings: &[DimensionEvidenceBinding],
        declarations: &[EvidenceLineageDeclaration],
        edges: &[LineageAncestryEdge],
    ) -> Result<ArchiveInsertOutcome, AncestryError> {
        let entry_id = entry_id.into();
        let niche = niche.into();

        let base = assess_declared_independence(
            entry_id.clone(),
            niche.clone(),
            bundle,
            bindings,
            self.evidence_policy,
            declarations,
            self.independence_policy,
        )?;
        let ancestry = assess_transitive_ancestry(
            &base,
            declarations,
            edges,
            self.ancestry_policy,
        )?;

        let outcome = self.inner.qualify_and_insert(
            entry_id.clone(),
            niche,
            bundle,
            bindings,
            declarations,
        )?;

        if let ArchiveInsertOutcome::Inserted { removed_entry_ids } = &outcome {
            for removed in removed_entry_ids {
                self.ancestry_report_by_entry.remove(removed);
            }
            self.ancestry_report_by_entry.insert(entry_id, ancestry);
        }

        Ok(outcome)
    }
}

impl Default for TransitiveAncestryArchive {
    fn default() -> Self {
        Self::new(
            ParetoPolicy::default(),
            EvidenceAdmissionPolicy::default(),
            DeclaredIndependencePolicy::default(),
            TransitiveAncestryPolicy::default(),
        )
        .expect("default transitive ancestry policies are valid")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::evidence_admission::GenerativityDimension;
    use crate::exploration::evidence_binding::EvidencePlaneEnvelope;
    use crate::exploration::generativity::{
        GenerativityAssessment, GenerativityEstimate, GenerativityVector,
    };
    use crate::exploration::persisted_evidence::{
        PersistedEvidenceCapsule, PersistedGenerativityBundle,
    };

    fn vector() -> GenerativityVector {
        let e = |value| GenerativityEstimate::new(value, 0.9).unwrap();
        GenerativityVector {
            immediate_utility: e(0.7),
            epistemic_gain: e(0.7),
            option_value: e(0.7),
            diversity: e(0.7),
            capability_gain: e(0.7),
            diffusion: e(0.7),
            commons_gain: e(0.7),
            regeneration: e(0.7),
            dependency_risk: e(0.2),
            concentration_risk: e(0.2),
            irreversibility_risk: e(0.2),
        }
    }

    fn run(run_id: &str, calls: f64) -> RunEvidence {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("mechanism_calls", calls);
        RunEvidence::new(RunId::new(run_id), &("mode", "active"), declared, measured)
    }

    fn verified_bundle() -> VerifiedGenerativityBundle {
        let mut assessment = GenerativityAssessment::new("alpha", "context", vector());
        let mut capsules = Vec::new();
        for (id, calls) in [("run:1", 4.0), ("run:2", 5.0)] {
            let run = run(id, calls);
            let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
            envelope.bind_qualified(&mut assessment, None).unwrap();
            capsules.push(PersistedEvidenceCapsule::from_run(run, vec![]).unwrap());
        }
        PersistedGenerativityBundle::new(assessment, capsules)
            .verify()
            .unwrap()
    }

    fn bindings() -> Vec<DimensionEvidenceBinding> {
        GenerativityDimension::ALL
            .iter()
            .map(|dimension| DimensionEvidenceBinding {
                dimension: *dimension,
                evidence_ids: vec![
                    "evidence-plane:run:1".into(),
                    "evidence-plane:run:2".into(),
                ],
            })
            .collect()
    }

    fn declarations() -> Vec<EvidenceLineageDeclaration> {
        vec![
            EvidenceLineageDeclaration {
                evidence_id: "evidence-plane:run:1".into(),
                method_lineage_id: "method:a".into(),
                source_lineage_ids: vec!["data:a".into()],
            },
            EvidenceLineageDeclaration {
                evidence_id: "evidence-plane:run:2".into(),
                method_lineage_id: "method:b".into(),
                source_lineage_ids: vec!["data:b".into()],
            },
        ]
    }

    fn evidence_policy() -> EvidenceAdmissionPolicy {
        EvidenceAdmissionPolicy {
            min_qualified_runs_per_dimension: 2,
            ..EvidenceAdmissionPolicy::default()
        }
    }

    fn independence_policy() -> DeclaredIndependencePolicy {
        DeclaredIndependencePolicy {
            min_method_lineages_per_dimension: 2,
            min_source_lineages_per_dimension: 2,
            ..DeclaredIndependencePolicy::default()
        }
    }

    fn root_policy() -> TransitiveAncestryPolicy {
        TransitiveAncestryPolicy {
            min_method_root_lineages_per_dimension: 2,
            min_source_root_lineages_per_dimension: 2,
            ..TransitiveAncestryPolicy::default()
        }
    }

    fn base_report() -> DeclaredIndependenceReport {
        assess_declared_independence(
            "entry",
            "research",
            &verified_bundle(),
            &bindings(),
            evidence_policy(),
            &declarations(),
            independence_policy(),
        )
        .unwrap()
    }

    #[test]
    fn distinct_immediate_labels_do_not_hide_shared_upstream_root() {
        let edges = vec![
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:a".into(),
                parent_id: "method:root".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:b".into(),
                parent_id: "method:root".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:a".into(),
                parent_id: "data:root".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:b".into(),
                parent_id: "data:root".into(),
            },
        ];

        assert!(matches!(
            assess_transitive_ancestry(&base_report(), &declarations(), &edges, root_policy()),
            Err(AncestryError::InsufficientMethodRootDiversity { observed: 1, .. })
        ));
    }

    #[test]
    fn distinct_declared_roots_pass_root_diversity_gate() {
        let edges = vec![
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:a".into(),
                parent_id: "method:root-a".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:b".into(),
                parent_id: "method:root-b".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:a".into(),
                parent_id: "data:root-a".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:b".into(),
                parent_id: "data:root-b".into(),
            },
        ];

        let report =
            assess_transitive_ancestry(&base_report(), &declarations(), &edges, root_policy())
                .unwrap();
        assert!(report.dimensions.iter().all(|dimension| {
            dimension.method_root_lineages.len() == 2
                && dimension.source_root_lineages.len() == 2
        }));
    }

    #[test]
    fn reachable_cycle_fails_closed() {
        let edges = vec![
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:a".into(),
                parent_id: "method:x".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:x".into(),
                parent_id: "method:a".into(),
            },
        ];

        assert!(matches!(
            assess_transitive_ancestry(
                &base_report(),
                &declarations(),
                &edges,
                TransitiveAncestryPolicy::default(),
            ),
            Err(AncestryError::ReachableCycle {
                domain: LineageDomain::Method,
                ..
            })
        ));
    }

    #[test]
    fn unrelated_cycle_does_not_poison_used_evidence() {
        let edges = vec![
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "unused:a".into(),
                parent_id: "unused:b".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "unused:b".into(),
                parent_id: "unused:a".into(),
            },
        ];

        let report = assess_transitive_ancestry(
            &base_report(),
            &declarations(),
            &edges,
            TransitiveAncestryPolicy::default(),
        )
        .unwrap();
        assert_eq!(report.dimensions.len(), GenerativityDimension::ALL.len());
    }
}
