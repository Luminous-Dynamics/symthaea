// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded application of qualified positive-independence attestations.
//!
//! This layer is deliberately stricter than graph reachability. Pairwise
//! independence is not transitive: A independent of B and B independent of C
//! does not prove A independent of C. A lower bound of N distinct origins is
//! raised only by an explicit N-component clique whose every pair is covered by
//! an active, policy-admitted attestation.
//!
//! The attestation allowlist is not authentication. Callers must authenticate
//! attestation authority before invoking this reasoning layer. Known shared-
//! origin evidence always dominates and conflicting attestations fail closed.

use crate::{
    assess_independence, EvidenceLineageRef, IndependenceAssessment,
    IndependenceAssessmentError, IndependenceAttestationSet, IndependenceAttestationSetError,
    IndependenceAuthorityPolicy, LineageRelationship, ObservationEnvelope, ObservationId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Explicit work limits for positive-independence graph reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedIndependenceReasoningPolicy {
    /// Maximum number of shared-origin components considered in one cohort.
    pub max_components: usize,
    /// Maximum active attestations considered after authority admission.
    pub max_active_attestations: usize,
    /// Maximum recursive clique-search states explored.
    pub max_search_states: usize,
}

impl Default for QualifiedIndependenceReasoningPolicy {
    fn default() -> Self {
        Self {
            max_components: 128,
            max_active_attestations: 4_096,
            max_search_states: 100_000,
        }
    }
}

/// Auditable witness for the lower bound established by positive attestations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceCliqueWitness {
    /// One observation-id set for each pairwise-independent shared-origin
    /// component in the witness.
    pub component_observation_ids: Vec<Vec<ObservationId>>,
    /// One active attestation id for every pair in the clique.
    pub attestation_ids: Vec<String>,
}

/// Conservative qualified-independence result for one aligned evidence cohort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedIndependenceAssessment {
    pub base: IndependenceAssessment,
    /// Active, authority-admitted attestations whose two subjects both matched
    /// this cohort and did not contradict known shared-origin evidence.
    pub matched_active_attestations: usize,
    /// Active attestations that did not match both sides of this cohort.
    pub out_of_cohort_attestations: usize,
    /// Structurally valid admitted attestations that were inactive at query time.
    pub inactive_attestations: usize,
    /// Unique component pairs covered by active positive attestations.
    pub attested_component_pairs: usize,
    /// Conservative lower bound after qualified positive evidence is applied.
    /// This can only increase through a pairwise-complete clique witness.
    pub qualified_independent_lower_bound: usize,
    /// Whether the bounded search proved that the returned clique is the largest
    /// clique under the admitted graph. A false value does not invalidate the
    /// returned lower bound; it only means a larger witness may exist.
    pub clique_search_complete: bool,
    pub clique_search_states: usize,
    pub witness: IndependenceCliqueWitness,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QualifiedIndependenceError {
    #[error("base independence assessment failed: {0}")]
    BaseAssessment(String),
    #[error("independence attestation admission failed: {0}")]
    AttestationAdmission(String),
    #[error("qualified-independence reasoning policy must use non-zero limits")]
    InvalidPolicy,
    #[error("cohort has {actual} shared-origin components, limit is {max}")]
    TooManyComponents { actual: usize, max: usize },
    #[error("cohort has {actual} active attestations, limit is {max}")]
    TooManyActiveAttestations { actual: usize, max: usize },
    #[error("attestation `{attestation_id}` lineage `{lineage}` matches multiple shared-origin components")]
    AmbiguousLineageReference {
        attestation_id: String,
        lineage: String,
    },
    #[error("attestation `{attestation_id}` contradicts known shared-origin evidence")]
    AttestationContradictsSharedOrigin { attestation_id: String },
}

/// Apply already-authenticated, locally admitted positive-independence
/// attestations to one semantically aligned observation cohort.
///
/// `IndependenceAuthorityPolicy` admission is intentionally repeated here so a
/// direct caller cannot bypass the authority qualification allowlist. This does
/// **not** authenticate the attestation bytes; authentication must have happened
/// before this call.
pub fn assess_qualified_independence(
    observations: &[ObservationEnvelope],
    attestations: &IndependenceAttestationSet,
    authority_policy: &IndependenceAuthorityPolicy,
    reasoning_policy: &QualifiedIndependenceReasoningPolicy,
    at_unix_ms: u64,
) -> Result<QualifiedIndependenceAssessment, QualifiedIndependenceError> {
    if reasoning_policy.max_components == 0
        || reasoning_policy.max_active_attestations == 0
        || reasoning_policy.max_search_states == 0
    {
        return Err(QualifiedIndependenceError::InvalidPolicy);
    }

    let base = assess_independence(observations).map_err(map_base_error)?;
    attestations
        .validate_with_policy(authority_policy)
        .map_err(map_attestation_error)?;

    let components = shared_origin_components(observations);
    if components.members.len() > reasoning_policy.max_components {
        return Err(QualifiedIndependenceError::TooManyComponents {
            actual: components.members.len(),
            max: reasoning_policy.max_components,
        });
    }

    let active_count = attestations
        .attestations
        .iter()
        .filter(|attestation| attestation.is_active_at(at_unix_ms))
        .count();
    if active_count > reasoning_policy.max_active_attestations {
        return Err(QualifiedIndependenceError::TooManyActiveAttestations {
            actual: active_count,
            max: reasoning_policy.max_active_attestations,
        });
    }

    let mut edge_attestations = BTreeMap::<(usize, usize), BTreeSet<String>>::new();
    let mut matched_active_attestations = 0usize;
    let mut out_of_cohort_attestations = 0usize;
    let mut inactive_attestations = 0usize;

    for attestation in &attestations.attestations {
        if !attestation.is_active_at(at_unix_ms) {
            inactive_attestations += 1;
            continue;
        }

        let left = match_lineage_component(
            &attestation.left,
            observations,
            &components.component_for_observation,
            &attestation.attestation_id,
        )?;
        let right = match_lineage_component(
            &attestation.right,
            observations,
            &components.component_for_observation,
            &attestation.attestation_id,
        )?;

        let (Some(left), Some(right)) = (left, right) else {
            out_of_cohort_attestations += 1;
            continue;
        };

        if left == right {
            return Err(QualifiedIndependenceError::AttestationContradictsSharedOrigin {
                attestation_id: attestation.attestation_id.clone(),
            });
        }

        let edge = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        edge_attestations
            .entry(edge)
            .or_default()
            .insert(attestation.attestation_id.clone());
        matched_active_attestations += 1;
    }

    let graph = build_graph(components.members.len(), &edge_attestations);
    let mut search = CliqueSearch::new(reasoning_policy.max_search_states, components.members.len());
    search.run(&graph);

    let witness = build_witness(&search.best, &components.members, observations, &edge_attestations);
    let qualified_independent_lower_bound = base
        .independent_lower_bound
        .max(search.best.len())
        .min(base.independent_upper_bound);

    Ok(QualifiedIndependenceAssessment {
        base,
        matched_active_attestations,
        out_of_cohort_attestations,
        inactive_attestations,
        attested_component_pairs: edge_attestations.len(),
        qualified_independent_lower_bound,
        clique_search_complete: !search.exhausted,
        clique_search_states: search.states,
        witness,
    })
}

fn map_base_error(error: IndependenceAssessmentError) -> QualifiedIndependenceError {
    QualifiedIndependenceError::BaseAssessment(error.to_string())
}

fn map_attestation_error(error: IndependenceAttestationSetError) -> QualifiedIndependenceError {
    QualifiedIndependenceError::AttestationAdmission(error.to_string())
}

#[derive(Debug)]
struct SharedOriginComponents {
    component_for_observation: Vec<usize>,
    members: Vec<Vec<usize>>,
}

fn shared_origin_components(observations: &[ObservationEnvelope]) -> SharedOriginComponents {
    let mut sets = DisjointSet::new(observations.len());
    for left in 0..observations.len() {
        for right in (left + 1)..observations.len() {
            if matches!(
                observations[left].lineage_relationship(&observations[right]),
                LineageRelationship::SameObservation | LineageRelationship::SharedOrigin
            ) {
                sets.union(left, right);
            }
        }
    }

    let mut root_members = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..observations.len() {
        let root = sets.find(index);
        root_members.entry(root).or_default().push(index);
    }

    let mut component_for_observation = vec![0usize; observations.len()];
    let mut members = Vec::with_capacity(root_members.len());
    for (component, indices) in root_members.into_values().enumerate() {
        for index in &indices {
            component_for_observation[*index] = component;
        }
        members.push(indices);
    }

    SharedOriginComponents {
        component_for_observation,
        members,
    }
}

fn match_lineage_component(
    lineage: &EvidenceLineageRef,
    observations: &[ObservationEnvelope],
    component_for_observation: &[usize],
    attestation_id: &str,
) -> Result<Option<usize>, QualifiedIndependenceError> {
    let mut components = BTreeSet::new();
    for (index, observation) in observations.iter().enumerate() {
        if lineage.matches_observation(observation) {
            components.insert(component_for_observation[index]);
        }
    }

    match components.len() {
        0 => Ok(None),
        1 => Ok(components.into_iter().next()),
        _ => Err(QualifiedIndependenceError::AmbiguousLineageReference {
            attestation_id: attestation_id.to_string(),
            lineage: lineage.canonical_key(),
        }),
    }
}

fn build_graph(
    components: usize,
    edge_attestations: &BTreeMap<(usize, usize), BTreeSet<String>>,
) -> Vec<BTreeSet<usize>> {
    let mut graph = vec![BTreeSet::new(); components];
    for &(left, right) in edge_attestations.keys() {
        graph[left].insert(right);
        graph[right].insert(left);
    }
    graph
}

fn build_witness(
    clique: &[usize],
    component_members: &[Vec<usize>],
    observations: &[ObservationEnvelope],
    edge_attestations: &BTreeMap<(usize, usize), BTreeSet<String>>,
) -> IndependenceCliqueWitness {
    let component_observation_ids = clique
        .iter()
        .map(|component| {
            component_members[*component]
                .iter()
                .map(|index| observations[*index].observation_id.clone())
                .collect()
        })
        .collect();

    let mut attestation_ids = Vec::new();
    for left_index in 0..clique.len() {
        for right_index in (left_index + 1)..clique.len() {
            let left = clique[left_index];
            let right = clique[right_index];
            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            if let Some(ids) = edge_attestations.get(&edge) {
                if let Some(id) = ids.iter().next() {
                    attestation_ids.push(id.clone());
                }
            }
        }
    }
    attestation_ids.sort();

    IndependenceCliqueWitness {
        component_observation_ids,
        attestation_ids,
    }
}

struct CliqueSearch {
    max_states: usize,
    states: usize,
    exhausted: bool,
    best: Vec<usize>,
}

impl CliqueSearch {
    fn new(max_states: usize, component_count: usize) -> Self {
        Self {
            max_states,
            states: 0,
            exhausted: false,
            best: if component_count == 0 { vec![] } else { vec![0] },
        }
    }

    fn run(&mut self, graph: &[BTreeSet<usize>]) {
        let candidates: Vec<usize> = (0..graph.len()).collect();
        let mut current = Vec::new();
        self.visit(graph, current.as_mut_slice(), candidates);
    }

    fn visit(
        &mut self,
        graph: &[BTreeSet<usize>],
        current: &mut [usize],
        mut candidates: Vec<usize>,
    ) {
        if self.states >= self.max_states {
            self.exhausted = true;
            return;
        }
        self.states += 1;

        if current.len() + candidates.len() <= self.best.len() {
            return;
        }

        while let Some(vertex) = candidates.pop() {
            if self.exhausted {
                return;
            }
            if current.len() + candidates.len() + 1 <= self.best.len() {
                return;
            }

            let mut next_current = current.to_vec();
            next_current.push(vertex);
            if next_current.len() > self.best.len() {
                self.best = next_current.clone();
                self.best.sort_unstable();
            }

            let next_candidates = candidates
                .iter()
                .copied()
                .filter(|candidate| graph[vertex].contains(candidate))
                .collect();
            self.visit(graph, next_current.as_mut_slice(), next_candidates);
        }
    }
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] = self.rank[left_root].saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EntityRef, IndependenceAttestation, IndependenceAuthorityQualification, IndependenceBasis,
        ObservationKind, ObservationLineage, ObservationQuality, ObservationSource,
        ObservationValue, INDEPENDENCE_ATTESTATION_SCHEMA_VERSION,
    };

    fn authority() -> IndependenceAuthorityQualification {
        IndependenceAuthorityQualification::new("review-board", "independence-v1")
    }

    fn authority_policy() -> IndependenceAuthorityPolicy {
        IndependenceAuthorityPolicy {
            trusted_qualifications: BTreeSet::from([authority()]),
            ..IndependenceAuthorityPolicy::default()
        }
    }

    fn observation(id: &str, integration: &str) -> ObservationEnvelope {
        ObservationEnvelope::new(
            ObservationId::new(id),
            100,
            100,
            EntityRef::new("site:lab", "service", "api"),
            ObservationKind::Metric,
            "service.capacity",
            ObservationValue::Unsigned(4),
            ObservationSource {
                integration_id: integration.into(),
                collector_id: Some(format!("collector-{integration}")),
                upstream_origin: None,
                measurement_method: format!("method-{integration}"),
                tenant: Some("tenant-a".into()),
            },
            ObservationQuality::observed(1.0),
            ObservationLineage {
                lineage_id: format!("lineage-{integration}"),
                parent_ids: vec![],
                independence_group: None,
                transforms: vec![],
            },
        )
    }

    fn attestation(
        id: &str,
        left_observation: &ObservationEnvelope,
        right_observation: &ObservationEnvelope,
    ) -> IndependenceAttestation {
        let mut left = EvidenceLineageRef::from_observation(left_observation);
        let mut right = EvidenceLineageRef::from_observation(right_observation);
        if left.canonical_key() > right.canonical_key() {
            std::mem::swap(&mut left, &mut right);
        }
        IndependenceAttestation {
            schema_version: INDEPENDENCE_ATTESTATION_SCHEMA_VERSION,
            attestation_id: id.into(),
            left,
            right,
            basis: IndependenceBasis::ReviewedProvenance,
            authority: authority(),
            issued_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        }
    }

    #[test]
    fn chain_of_pairwise_attestations_does_not_prove_three_origins() {
        let a = observation("a", "kubernetes");
        let b = observation("b", "prometheus");
        let c = observation("c", "otlp");
        let set = IndependenceAttestationSet {
            attestations: vec![attestation("ab", &a, &b), attestation("bc", &b, &c)],
        };
        let assessment = assess_qualified_independence(
            &[a, b, c],
            &set,
            &authority_policy(),
            &QualifiedIndependenceReasoningPolicy::default(),
            100,
        )
        .unwrap();
        assert_eq!(assessment.qualified_independent_lower_bound, 2);
        assert_eq!(assessment.witness.component_observation_ids.len(), 2);
    }

    #[test]
    fn triangle_of_pairwise_attestations_proves_three_distinct_origins() {
        let a = observation("a", "kubernetes");
        let b = observation("b", "prometheus");
        let c = observation("c", "otlp");
        let set = IndependenceAttestationSet {
            attestations: vec![
                attestation("ab", &a, &b),
                attestation("bc", &b, &c),
                attestation("ac", &a, &c),
            ],
        };
        let assessment = assess_qualified_independence(
            &[a, b, c],
            &set,
            &authority_policy(),
            &QualifiedIndependenceReasoningPolicy::default(),
            100,
        )
        .unwrap();
        assert_eq!(assessment.qualified_independent_lower_bound, 3);
        assert_eq!(assessment.witness.component_observation_ids.len(), 3);
        assert_eq!(assessment.witness.attestation_ids.len(), 3);
        assert!(assessment.clique_search_complete);
    }

    #[test]
    fn attestation_cannot_override_shared_origin_component() {
        let mut a = observation("a", "kubernetes");
        let mut b = observation("b", "prometheus");
        a.lineage.independence_group = Some("same-source".into());
        b.lineage.independence_group = Some("same-source".into());
        let set = IndependenceAttestationSet {
            attestations: vec![attestation("conflict", &a, &b)],
        };
        assert!(matches!(
            assess_qualified_independence(
                &[a, b],
                &set,
                &authority_policy(),
                &QualifiedIndependenceReasoningPolicy::default(),
                100,
            ),
            Err(QualifiedIndependenceError::AttestationContradictsSharedOrigin { .. })
        ));
    }

    #[test]
    fn future_attestation_cannot_raise_historical_lower_bound() {
        let a = observation("a", "kubernetes");
        let b = observation("b", "prometheus");
        let mut future = attestation("future", &a, &b);
        future.issued_at_unix_ms = 200;
        let set = IndependenceAttestationSet {
            attestations: vec![future],
        };
        let assessment = assess_qualified_independence(
            &[a, b],
            &set,
            &authority_policy(),
            &QualifiedIndependenceReasoningPolicy::default(),
            100,
        )
        .unwrap();
        assert_eq!(assessment.qualified_independent_lower_bound, 1);
        assert_eq!(assessment.inactive_attestations, 1);
    }

    #[test]
    fn out_of_cohort_attestation_is_ignored_not_counted_as_support() {
        let a = observation("a", "kubernetes");
        let b = observation("b", "prometheus");
        let external = observation("external", "otlp");
        let set = IndependenceAttestationSet {
            attestations: vec![attestation("external", &a, &external)],
        };
        let assessment = assess_qualified_independence(
            &[a, b],
            &set,
            &authority_policy(),
            &QualifiedIndependenceReasoningPolicy::default(),
            100,
        )
        .unwrap();
        assert_eq!(assessment.qualified_independent_lower_bound, 1);
        assert_eq!(assessment.out_of_cohort_attestations, 1);
    }

    #[test]
    fn exhausted_search_returns_only_a_valid_witness_lower_bound() {
        let a = observation("a", "kubernetes");
        let b = observation("b", "prometheus");
        let c = observation("c", "otlp");
        let set = IndependenceAttestationSet {
            attestations: vec![
                attestation("ab", &a, &b),
                attestation("bc", &b, &c),
                attestation("ac", &a, &c),
            ],
        };
        let policy = QualifiedIndependenceReasoningPolicy {
            max_search_states: 1,
            ..QualifiedIndependenceReasoningPolicy::default()
        };
        let assessment = assess_qualified_independence(
            &[a, b, c],
            &set,
            &authority_policy(),
            &policy,
            100,
        )
        .unwrap();
        assert!(assessment.qualified_independent_lower_bound >= 1);
        assert!(assessment.qualified_independent_lower_bound <= 3);
        assert!(!assessment.clique_search_complete);
    }
}
