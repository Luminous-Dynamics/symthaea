// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware potential blast-radius analysis over `SystemStateGraphV1`.
//!
//! This module is deliberately read-only. It does not create topology, infer
//! dependencies from names, or claim that a reachable entity will actually fail.
//! It answers a narrower question:
//!
//! > Given explicit dependency semantics, recorded topology, evidence policy,
//! > and a set of affected entities, which other entities are potentially
//! > impacted through those declared relationships?
//!
//! Core non-equivalences:
//!
//! ```text
//! graph reachability != guaranteed outage
//! relation presence != dependency semantics
//! stale evidence != current topology
//! blast-radius estimate != execution authority
//! redundancy unknown != redundancy absent
//! ```

use crate::system_state::{
    CurrentnessStatusV1, EntityId, ObservationId, RelationId, RelationKindV1,
    SystemRelationV1, SystemStateGraphV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

/// Direction in which impact is allowed to propagate across a stored relation.
///
/// For a relation `from --kind--> to`, `Reverse` means failure of `to` may
/// affect `from`. This is the natural direction for `DependsOn`, `RunsOn`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactTraversalV1 {
    Forward,
    Reverse,
    Bidirectional,
}

/// Explicit semantics for one relation kind. Merely having a relation in the
/// graph is not enough to make it an impact-propagation edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactPropagationRuleV1 {
    pub relation_kind: RelationKindV1,
    pub traversal: ImpactTraversalV1,
}

/// Currentness requirement for topology evidence used during propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactEvidenceRequirementV1 {
    /// At least one recorded evidence item is enough. Currentness is still
    /// reported in the result, but does not block propagation.
    AnyRecorded,
    /// At least one evidence item must be fresh at the analysis time.
    FreshOnly,
    /// Fresh or indeterminate evidence may propagate; explicitly stale-only
    /// relations are blocked.
    FreshOrIndeterminate,
}

/// Policy governing which graph relations have impact semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyImpactPolicyV1 {
    pub rules: Vec<ImpactPropagationRuleV1>,
    pub evidence_requirement: ImpactEvidenceRequirementV1,
    pub max_depth: usize,
}

impl DependencyImpactPolicyV1 {
    /// Conservative availability-oriented defaults.
    ///
    /// Only relations whose stored direction naturally means "from relies on
    /// to" are enabled. Ambiguous/soft relations such as `ConnectsTo`,
    /// `ReplicatesTo`, `ObservedBy`, `ConfiguredBy`, and `MemberOf` are excluded
    /// until a caller explicitly declares semantics for its environment.
    pub fn availability_default() -> Self {
        let reverse = |relation_kind| ImpactPropagationRuleV1 {
            relation_kind,
            traversal: ImpactTraversalV1::Reverse,
        };
        Self {
            rules: vec![
                reverse(RelationKindV1::RunsOn),
                reverse(RelationKindV1::DependsOn),
                reverse(RelationKindV1::RoutesVia),
                reverse(RelationKindV1::ResolvesVia),
                reverse(RelationKindV1::AuthenticatesVia),
                reverse(RelationKindV1::AuthorizedBy),
                reverse(RelationKindV1::MountedFrom),
            ],
            evidence_requirement: ImpactEvidenceRequirementV1::AnyRecorded,
            max_depth: 16,
        }
    }

    pub fn validate(&self) -> Result<(), DependencyImpactErrorV1> {
        for (index, rule) in self.rules.iter().enumerate() {
            if self.rules[..index]
                .iter()
                .any(|existing| existing.relation_kind == rule.relation_kind)
            {
                return Err(DependencyImpactErrorV1::DuplicateRule(
                    rule.relation_kind.clone(),
                ));
            }
        }
        Ok(())
    }

    fn rule_for(&self, kind: &RelationKindV1) -> Option<&ImpactPropagationRuleV1> {
        self.rules.iter().find(|rule| &rule.relation_kind == kind)
    }
}

impl Default for DependencyImpactPolicyV1 {
    fn default() -> Self {
        Self::availability_default()
    }
}

/// Currentness summary for evidence attached to one traversed relation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactEvidenceSummaryV1 {
    pub total: usize,
    pub fresh: usize,
    pub stale: usize,
    pub indeterminate: usize,
}

impl ImpactEvidenceSummaryV1 {
    fn allows(self, requirement: ImpactEvidenceRequirementV1) -> bool {
        match requirement {
            ImpactEvidenceRequirementV1::AnyRecorded => self.total > 0,
            ImpactEvidenceRequirementV1::FreshOnly => self.fresh > 0,
            ImpactEvidenceRequirementV1::FreshOrIndeterminate => {
                self.fresh + self.indeterminate > 0
            }
        }
    }
}

/// One edge on a deterministic shortest potential-impact path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactPathStepV1 {
    pub relation_id: RelationId,
    pub relation_kind: RelationKindV1,
    pub from: EntityId,
    pub to: EntityId,
    pub traversal: ImpactTraversalV1,
    pub evidence_ids: Vec<ObservationId>,
    pub evidence: ImpactEvidenceSummaryV1,
}

/// One entity reachable under the declared impact policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PotentialImpactV1 {
    pub entity_id: EntityId,
    pub depth: usize,
    /// Canonical deterministic shortest path from one seed to this entity.
    pub path: Vec<ImpactPathStepV1>,
}

/// Read-only result. "Potentially impacted" is intentionally weaker than
/// "failed" or "will fail".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyImpactAnalysisV1 {
    pub seeds: Vec<EntityId>,
    pub potentially_impacted: Vec<PotentialImpactV1>,
    pub considered_relations: usize,
    pub traversable_relations: usize,
    pub blocked_by_evidence: usize,
    pub excluded_by_policy: usize,
    pub depth_limited: bool,
}

pub fn analyze_dependency_impact_v1(
    graph: &SystemStateGraphV1,
    seeds: impl IntoIterator<Item = EntityId>,
    policy: &DependencyImpactPolicyV1,
    evaluation_time_unix_ms: u64,
) -> Result<DependencyImpactAnalysisV1, DependencyImpactErrorV1> {
    policy.validate()?;

    let mut seeds: Vec<EntityId> = seeds.into_iter().collect();
    seeds.sort();
    seeds.dedup();
    if seeds.is_empty() {
        return Err(DependencyImpactErrorV1::EmptySeeds);
    }
    for seed in &seeds {
        if graph.entity(seed).is_none() {
            return Err(DependencyImpactErrorV1::UnknownSeed(seed.clone()));
        }
    }

    let seed_set: BTreeSet<EntityId> = seeds.iter().cloned().collect();
    let mut visited = BTreeMap::<EntityId, usize>::new();
    let mut paths = BTreeMap::<EntityId, Vec<ImpactPathStepV1>>::new();
    let mut queue = VecDeque::new();
    for seed in &seeds {
        visited.insert(seed.clone(), 0);
        paths.insert(seed.clone(), Vec::new());
        queue.push_back(seed.clone());
    }

    let mut considered = BTreeSet::<RelationId>::new();
    let mut traversable = BTreeSet::<RelationId>::new();
    let mut blocked = BTreeSet::<RelationId>::new();
    let mut excluded = BTreeSet::<RelationId>::new();
    let mut depth_limited = false;

    while let Some(current) = queue.pop_front() {
        let depth = visited[&current];

        for relation in graph.relations() {
            let incident = relation.from == current || relation.to == current;
            if !incident {
                continue;
            }

            let Some(rule) = policy.rule_for(&relation.kind) else {
                excluded.insert(relation.id.clone());
                continue;
            };

            let Some(next) = next_entity_for_rule(relation, &current, rule.traversal) else {
                continue;
            };
            considered.insert(relation.id.clone());

            let evidence = summarize_relation_evidence(
                graph,
                relation,
                evaluation_time_unix_ms,
            );
            if !evidence.allows(policy.evidence_requirement) {
                blocked.insert(relation.id.clone());
                continue;
            }
            traversable.insert(relation.id.clone());

            if depth >= policy.max_depth {
                if !visited.contains_key(&next) {
                    depth_limited = true;
                }
                continue;
            }

            if visited.contains_key(&next) {
                continue;
            }

            let mut path = paths.get(&current).cloned().unwrap_or_default();
            path.push(ImpactPathStepV1 {
                relation_id: relation.id.clone(),
                relation_kind: relation.kind.clone(),
                from: relation.from.clone(),
                to: relation.to.clone(),
                traversal: rule.traversal,
                evidence_ids: relation.evidence.iter().cloned().collect(),
                evidence,
            });
            visited.insert(next.clone(), depth + 1);
            paths.insert(next.clone(), path);
            queue.push_back(next);
        }
    }

    let mut potentially_impacted: Vec<PotentialImpactV1> = visited
        .into_iter()
        .filter(|(entity_id, _)| !seed_set.contains(entity_id))
        .map(|(entity_id, depth)| PotentialImpactV1 {
            path: paths.remove(&entity_id).unwrap_or_default(),
            entity_id,
            depth,
        })
        .collect();
    potentially_impacted.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then_with(|| a.entity_id.cmp(&b.entity_id))
    });

    Ok(DependencyImpactAnalysisV1 {
        seeds,
        potentially_impacted,
        considered_relations: considered.len(),
        traversable_relations: traversable.len(),
        blocked_by_evidence: blocked.len(),
        excluded_by_policy: excluded.len(),
        depth_limited,
    })
}

fn next_entity_for_rule(
    relation: &SystemRelationV1,
    current: &EntityId,
    traversal: ImpactTraversalV1,
) -> Option<EntityId> {
    match traversal {
        ImpactTraversalV1::Forward if &relation.from == current => Some(relation.to.clone()),
        ImpactTraversalV1::Reverse if &relation.to == current => Some(relation.from.clone()),
        ImpactTraversalV1::Bidirectional if &relation.from == current => Some(relation.to.clone()),
        ImpactTraversalV1::Bidirectional if &relation.to == current => Some(relation.from.clone()),
        _ => None,
    }
}

fn summarize_relation_evidence(
    graph: &SystemStateGraphV1,
    relation: &SystemRelationV1,
    evaluation_time_unix_ms: u64,
) -> ImpactEvidenceSummaryV1 {
    let mut summary = ImpactEvidenceSummaryV1::default();
    for evidence_id in &relation.evidence {
        summary.total += 1;
        match graph
            .observation(evidence_id)
            .map(|observation| observation.clock.currentness_at(evaluation_time_unix_ms))
            .unwrap_or(CurrentnessStatusV1::Indeterminate)
        {
            CurrentnessStatusV1::Fresh => summary.fresh += 1,
            CurrentnessStatusV1::Stale => summary.stale += 1,
            CurrentnessStatusV1::Indeterminate => summary.indeterminate += 1,
        }
    }
    summary
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyImpactErrorV1 {
    EmptySeeds,
    UnknownSeed(EntityId),
    DuplicateRule(RelationKindV1),
}

impl fmt::Display for DependencyImpactErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeeds => write!(f, "dependency impact analysis requires at least one seed"),
            Self::UnknownSeed(seed) => write!(f, "dependency impact seed {:?} is not in graph", seed.0),
            Self::DuplicateRule(kind) => write!(f, "duplicate impact propagation rule for {kind:?}"),
        }
    }
}

impl Error for DependencyImpactErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::{
        EntityKindV1, ObservationClockV1, ObservationProvenanceV1, ObservationSourceKindV1,
        SystemObservationV1,
    };

    fn observation(id: &str, observed_at: u64, max_age_ms: Option<u64>) -> SystemObservationV1 {
        SystemObservationV1 {
            id: ObservationId(id.into()),
            subject: EntityId("topology:collector".into()),
            provenance: ObservationProvenanceV1 {
                source_id: "topology:test".into(),
                source_kind: ObservationSourceKindV1::ConfigSnapshot,
                collector: "dependency-impact-test".into(),
                collector_version: Some("1".into()),
                schema_version: None,
                artifact_digest: None,
            },
            clock: ObservationClockV1 {
                event_time_unix_ms: None,
                observed_at_unix_ms: observed_at,
                ingested_at_unix_ms: None,
                max_age_ms,
                clock_uncertainty_ms: None,
            },
            confidence: 1.0,
            facts: BTreeMap::new(),
        }
    }

    fn graph_with_evidence(observed_at: u64, max_age_ms: Option<u64>) -> SystemStateGraphV1 {
        let mut graph = SystemStateGraphV1::new();
        let evidence = observation("obs:topology", observed_at, max_age_ms);
        graph.record_observation(evidence).unwrap();
        for (id, kind) in [
            ("host", EntityKindV1::Host),
            ("service", EntityKindV1::Service),
            ("app", EntityKindV1::Service),
            ("monitor", EntityKindV1::Service),
        ] {
            graph
                .upsert_entity(
                    EntityId(id.into()),
                    kind,
                    BTreeMap::new(),
                    &ObservationId("obs:topology".into()),
                )
                .unwrap();
        }
        graph
            .upsert_relation(
                RelationId("rel:service-host".into()),
                EntityId("service".into()),
                RelationKindV1::RunsOn,
                EntityId("host".into()),
                BTreeMap::new(),
                &ObservationId("obs:topology".into()),
            )
            .unwrap();
        graph
            .upsert_relation(
                RelationId("rel:app-service".into()),
                EntityId("app".into()),
                RelationKindV1::DependsOn,
                EntityId("service".into()),
                BTreeMap::new(),
                &ObservationId("obs:topology".into()),
            )
            .unwrap();
        graph
            .upsert_relation(
                RelationId("rel:observed".into()),
                EntityId("host".into()),
                RelationKindV1::ObservedBy,
                EntityId("monitor".into()),
                BTreeMap::new(),
                &ObservationId("obs:topology".into()),
            )
            .unwrap();
        graph
    }

    #[test]
    fn default_availability_policy_propagates_dependency_failure_in_reverse() {
        let graph = graph_with_evidence(900, Some(200));
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &DependencyImpactPolicyV1::default(),
            1_000,
        )
        .unwrap();
        let ids: Vec<_> = result
            .potentially_impacted
            .iter()
            .map(|impact| (impact.entity_id.0.as_str(), impact.depth))
            .collect();
        assert_eq!(ids, vec![("service", 1), ("app", 2)]);
        assert_eq!(result.excluded_by_policy, 1);
        assert!(!result.depth_limited);
    }

    #[test]
    fn ambiguous_observation_relation_does_not_propagate_by_default() {
        let graph = graph_with_evidence(900, Some(200));
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("monitor".into())],
            &DependencyImpactPolicyV1::default(),
            1_000,
        )
        .unwrap();
        assert!(result.potentially_impacted.is_empty());
    }

    #[test]
    fn custom_policy_can_explicitly_admit_connectivity_semantics() {
        let mut graph = graph_with_evidence(900, Some(200));
        graph
            .upsert_relation(
                RelationId("rel:connect".into()),
                EntityId("host".into()),
                RelationKindV1::ConnectsTo,
                EntityId("monitor".into()),
                BTreeMap::new(),
                &ObservationId("obs:topology".into()),
            )
            .unwrap();
        let mut policy = DependencyImpactPolicyV1::default();
        policy.rules.push(ImpactPropagationRuleV1 {
            relation_kind: RelationKindV1::ConnectsTo,
            traversal: ImpactTraversalV1::Forward,
        });
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &policy,
            1_000,
        )
        .unwrap();
        assert!(result
            .potentially_impacted
            .iter()
            .any(|impact| impact.entity_id == EntityId("monitor".into())));
    }

    #[test]
    fn fresh_only_policy_blocks_stale_topology() {
        let graph = graph_with_evidence(100, Some(100));
        let mut policy = DependencyImpactPolicyV1::default();
        policy.evidence_requirement = ImpactEvidenceRequirementV1::FreshOnly;
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &policy,
            1_000,
        )
        .unwrap();
        assert!(result.potentially_impacted.is_empty());
        assert!(result.blocked_by_evidence >= 1);
    }

    #[test]
    fn depth_limit_is_explicit_in_result() {
        let graph = graph_with_evidence(900, Some(200));
        let mut policy = DependencyImpactPolicyV1::default();
        policy.max_depth = 1;
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &policy,
            1_000,
        )
        .unwrap();
        assert_eq!(result.potentially_impacted.len(), 1);
        assert_eq!(result.potentially_impacted[0].entity_id, EntityId("service".into()));
        assert!(result.depth_limited);
    }

    #[test]
    fn cycles_terminate_without_readding_seeds() {
        let mut graph = graph_with_evidence(900, Some(200));
        graph
            .upsert_relation(
                RelationId("rel:cycle".into()),
                EntityId("host".into()),
                RelationKindV1::DependsOn,
                EntityId("app".into()),
                BTreeMap::new(),
                &ObservationId("obs:topology".into()),
            )
            .unwrap();
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &DependencyImpactPolicyV1::default(),
            1_000,
        )
        .unwrap();
        let ids: BTreeSet<_> = result
            .potentially_impacted
            .iter()
            .map(|impact| impact.entity_id.clone())
            .collect();
        assert_eq!(ids.len(), result.potentially_impacted.len());
        assert!(!ids.contains(&EntityId("host".into())));
    }

    #[test]
    fn deterministic_shortest_path_keeps_relation_evidence() {
        let graph = graph_with_evidence(900, Some(200));
        let result = analyze_dependency_impact_v1(
            &graph,
            [EntityId("host".into())],
            &DependencyImpactPolicyV1::default(),
            1_000,
        )
        .unwrap();
        let app = result
            .potentially_impacted
            .iter()
            .find(|impact| impact.entity_id == EntityId("app".into()))
            .unwrap();
        assert_eq!(app.depth, 2);
        assert_eq!(app.path.len(), 2);
        assert_eq!(app.path[0].relation_id, RelationId("rel:service-host".into()));
        assert_eq!(app.path[1].relation_id, RelationId("rel:app-service".into()));
        assert_eq!(app.path[0].evidence.fresh, 1);
        assert_eq!(app.path[0].evidence_ids, vec![ObservationId("obs:topology".into())]);
    }

    #[test]
    fn unknown_seed_is_rejected() {
        let graph = graph_with_evidence(900, Some(200));
        assert!(matches!(
            analyze_dependency_impact_v1(
                &graph,
                [EntityId("missing".into())],
                &DependencyImpactPolicyV1::default(),
                1_000,
            ),
            Err(DependencyImpactErrorV1::UnknownSeed(_))
        ));
    }

    #[test]
    fn duplicate_relation_semantics_are_rejected() {
        let mut policy = DependencyImpactPolicyV1::default();
        policy.rules.push(ImpactPropagationRuleV1 {
            relation_kind: RelationKindV1::DependsOn,
            traversal: ImpactTraversalV1::Forward,
        });
        assert!(matches!(
            policy.validate(),
            Err(DependencyImpactErrorV1::DuplicateRule(RelationKindV1::DependsOn))
        ));
    }
}
