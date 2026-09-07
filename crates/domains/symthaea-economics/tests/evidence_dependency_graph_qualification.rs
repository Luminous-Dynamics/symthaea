// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only evidence dependency theorem for Economic Science.
//!
//! The core separation is:
//!
//! publication count != evidence lineage count != declared dependency disjointness
//! != verified independent replication.
//!
//! This test intentionally never emits `independent = true`.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DependencyDomain {
    SourceData,
    SourceVintage,
    MeasurementSpecification,
    SamplingDesign,
    TransformationArtifact,
    ModelArtifact,
    EstimatorArtifact,
    IdentificationStrategy,
    OutcomePolicy,
    EvaluationProtocol,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DependencyItem {
    domain: DependencyDomain,
    id: String,
}

impl DependencyItem {
    fn new(domain: DependencyDomain, id: impl Into<String>) -> Result<Self, DependencyError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(DependencyError::EmptyText("dependency id"));
        }
        Ok(Self { domain, id })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DependencyInventory {
    covered_domains: BTreeSet<DependencyDomain>,
    items: BTreeSet<DependencyItem>,
}

impl DependencyInventory {
    fn new(
        covered_domains: impl IntoIterator<Item = DependencyDomain>,
        items: impl IntoIterator<Item = DependencyItem>,
    ) -> Result<Self, DependencyError> {
        let covered_domains: BTreeSet<_> = covered_domains.into_iter().collect();
        if covered_domains.is_empty() {
            return Err(DependencyError::EmptyInventoryScope);
        }

        let mut unique_items = BTreeSet::new();
        for item in items {
            if !covered_domains.contains(&item.domain) {
                return Err(DependencyError::ItemOutsideDeclaredScope(item.domain));
            }
            if !unique_items.insert(item) {
                return Err(DependencyError::DuplicateDependency);
            }
        }

        Ok(Self {
            covered_domains,
            items: unique_items,
        })
    }

    fn covers(&self, domain: DependencyDomain) -> bool {
        self.covered_domains.contains(&domain)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvidenceLineage {
    lineage_id: String,
    contribution_id: String,
    evidence_target_id: String,
    inventory: DependencyInventory,
}

impl EvidenceLineage {
    fn new(
        lineage_id: impl Into<String>,
        contribution_id: impl Into<String>,
        evidence_target_id: impl Into<String>,
        inventory: DependencyInventory,
    ) -> Result<Self, DependencyError> {
        let lineage_id = lineage_id.into();
        let contribution_id = contribution_id.into();
        let evidence_target_id = evidence_target_id.into();
        for (field, value) in [
            ("lineage id", lineage_id.as_str()),
            ("contribution id", contribution_id.as_str()),
            ("evidence target id", evidence_target_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(DependencyError::EmptyText(field));
            }
        }
        Ok(Self {
            lineage_id,
            contribution_id,
            evidence_target_id,
            inventory,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DependencyRelation {
    SameLineage,
    SharedDeclaredDependencies {
        overlaps: Vec<DependencyItem>,
        inventory_complete_for_scope: bool,
    },
    DeclaredDisjointWithinScope {
        scope: Vec<DependencyDomain>,
    },
    IncompleteInventory {
        missing_left: Vec<DependencyDomain>,
        missing_right: Vec<DependencyDomain>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DependencyError {
    EmptyText(&'static str),
    EmptyInventoryScope,
    EmptyComparisonScope,
    DuplicateDependency,
    ItemOutsideDeclaredScope(DependencyDomain),
    EvidenceTargetMismatch,
    LineageIdentityConflict,
    MixedEvidenceTargets,
}

fn missing_domains(
    inventory: &DependencyInventory,
    required_scope: &BTreeSet<DependencyDomain>,
) -> Vec<DependencyDomain> {
    required_scope
        .iter()
        .copied()
        .filter(|domain| !inventory.covers(*domain))
        .collect()
}

fn compare_lineages(
    left: &EvidenceLineage,
    right: &EvidenceLineage,
    required_scope: &BTreeSet<DependencyDomain>,
) -> Result<DependencyRelation, DependencyError> {
    if required_scope.is_empty() {
        return Err(DependencyError::EmptyComparisonScope);
    }
    if left.evidence_target_id != right.evidence_target_id {
        return Err(DependencyError::EvidenceTargetMismatch);
    }

    if left.lineage_id == right.lineage_id {
        if left.inventory != right.inventory {
            return Err(DependencyError::LineageIdentityConflict);
        }
        return Ok(DependencyRelation::SameLineage);
    }

    let missing_left = missing_domains(&left.inventory, required_scope);
    let missing_right = missing_domains(&right.inventory, required_scope);

    let overlaps: Vec<_> = left
        .inventory
        .items
        .intersection(&right.inventory.items)
        .filter(|item| required_scope.contains(&item.domain))
        .cloned()
        .collect();

    if !overlaps.is_empty() {
        return Ok(DependencyRelation::SharedDeclaredDependencies {
            overlaps,
            inventory_complete_for_scope: missing_left.is_empty() && missing_right.is_empty(),
        });
    }

    if !missing_left.is_empty() || !missing_right.is_empty() {
        return Ok(DependencyRelation::IncompleteInventory {
            missing_left,
            missing_right,
        });
    }

    Ok(DependencyRelation::DeclaredDisjointWithinScope {
        scope: required_scope.iter().copied().collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplicationDependencyAudit {
    total_contributions: usize,
    contribution_ids: Vec<String>,
    unique_lineages: usize,
    known_dependency_components: Vec<Vec<String>>,
    incomplete_lineages: Vec<String>,
}

fn audit_replication_set(
    lineages: &[EvidenceLineage],
    required_scope: &BTreeSet<DependencyDomain>,
) -> Result<ReplicationDependencyAudit, DependencyError> {
    if required_scope.is_empty() {
        return Err(DependencyError::EmptyComparisonScope);
    }

    let mut contribution_ids: Vec<_> = lineages
        .iter()
        .map(|lineage| lineage.contribution_id.clone())
        .collect();
    contribution_ids.sort();

    if lineages.is_empty() {
        return Ok(ReplicationDependencyAudit {
            total_contributions: 0,
            contribution_ids,
            unique_lineages: 0,
            known_dependency_components: Vec::new(),
            incomplete_lineages: Vec::new(),
        });
    }

    let target = &lineages[0].evidence_target_id;
    if lineages
        .iter()
        .any(|lineage| &lineage.evidence_target_id != target)
    {
        return Err(DependencyError::MixedEvidenceTargets);
    }

    let mut representatives: BTreeMap<String, &EvidenceLineage> = BTreeMap::new();
    for lineage in lineages {
        match representatives.get(&lineage.lineage_id) {
            Some(existing)
                if existing.inventory != lineage.inventory
                    || existing.evidence_target_id != lineage.evidence_target_id =>
            {
                return Err(DependencyError::LineageIdentityConflict);
            }
            Some(_) => {}
            None => {
                representatives.insert(lineage.lineage_id.clone(), lineage);
            }
        }
    }

    let mut adjacency: BTreeMap<String, BTreeSet<String>> = representatives
        .keys()
        .map(|id| (id.clone(), BTreeSet::new()))
        .collect();

    let representative_values: Vec<_> = representatives.values().copied().collect();
    for (index, left) in representative_values.iter().enumerate() {
        for right in representative_values.iter().skip(index + 1) {
            if matches!(
                compare_lineages(left, right, required_scope)?,
                DependencyRelation::SameLineage
                    | DependencyRelation::SharedDeclaredDependencies { .. }
            ) {
                adjacency
                    .get_mut(&left.lineage_id)
                    .expect("left lineage must exist")
                    .insert(right.lineage_id.clone());
                adjacency
                    .get_mut(&right.lineage_id)
                    .expect("right lineage must exist")
                    .insert(left.lineage_id.clone());
            }
        }
    }

    let incomplete_lineages: Vec<_> = representatives
        .values()
        .filter(|lineage| !missing_domains(&lineage.inventory, required_scope).is_empty())
        .map(|lineage| lineage.lineage_id.clone())
        .collect();

    let mut visited = BTreeSet::new();
    let mut components = Vec::new();
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        let mut stack = vec![start.clone()];
        let mut component = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            component.push(current.clone());
            if let Some(neighbors) = adjacency.get(&current) {
                stack.extend(neighbors.iter().cloned());
            }
        }
        component.sort();
        components.push(component);
    }
    components.sort();

    Ok(ReplicationDependencyAudit {
        total_contributions: lineages.len(),
        contribution_ids,
        unique_lineages: representatives.len(),
        known_dependency_components: components,
        incomplete_lineages,
    })
}

fn full_scope() -> BTreeSet<DependencyDomain> {
    [
        DependencyDomain::SourceData,
        DependencyDomain::SourceVintage,
        DependencyDomain::MeasurementSpecification,
        DependencyDomain::SamplingDesign,
        DependencyDomain::TransformationArtifact,
        DependencyDomain::ModelArtifact,
        DependencyDomain::EstimatorArtifact,
        DependencyDomain::IdentificationStrategy,
        DependencyDomain::OutcomePolicy,
        DependencyDomain::EvaluationProtocol,
    ]
    .into_iter()
    .collect()
}

fn inventory(
    scope: &BTreeSet<DependencyDomain>,
    items: &[(DependencyDomain, &str)],
) -> DependencyInventory {
    DependencyInventory::new(
        scope.iter().copied(),
        items
            .iter()
            .map(|(domain, id)| DependencyItem::new(*domain, *id).unwrap()),
    )
    .unwrap()
}

fn lineage(
    lineage_id: &str,
    contribution_id: &str,
    scope: &BTreeSet<DependencyDomain>,
    items: &[(DependencyDomain, &str)],
) -> EvidenceLineage {
    EvidenceLineage::new(
        lineage_id,
        contribution_id,
        "claim:policy-employment-v1",
        inventory(scope, items),
    )
    .unwrap()
}

#[test]
fn inventory_validation_is_fail_closed() {
    assert_eq!(
        DependencyItem::new(DependencyDomain::SourceData, " "),
        Err(DependencyError::EmptyText("dependency id"))
    );
    assert_eq!(
        DependencyInventory::new(Vec::<DependencyDomain>::new(), Vec::<DependencyItem>::new()),
        Err(DependencyError::EmptyInventoryScope)
    );

    let source_only: BTreeSet<_> = [DependencyDomain::SourceData].into_iter().collect();
    assert_eq!(
        DependencyInventory::new(
            source_only.iter().copied(),
            [DependencyItem::new(DependencyDomain::ModelArtifact, "model:a").unwrap()],
        ),
        Err(DependencyError::ItemOutsideDeclaredScope(
            DependencyDomain::ModelArtifact
        ))
    );

    let duplicate = DependencyItem::new(DependencyDomain::SourceData, "data:a").unwrap();
    assert_eq!(
        DependencyInventory::new(
            source_only.iter().copied(),
            [duplicate.clone(), duplicate],
        ),
        Err(DependencyError::DuplicateDependency)
    );
}

#[test]
fn same_dataset_is_declared_dependency_even_with_different_models() {
    let scope = full_scope();
    let left = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:lfs"),
            (DependencyDomain::ModelArtifact, "model:a"),
        ],
    );
    let right = lineage(
        "lineage:b",
        "paper:b",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:lfs"),
            (DependencyDomain::ModelArtifact, "model:b"),
        ],
    );

    let relation = compare_lineages(&left, &right, &scope).unwrap();
    match relation {
        DependencyRelation::SharedDeclaredDependencies {
            overlaps,
            inventory_complete_for_scope,
        } => {
            assert!(inventory_complete_for_scope);
            assert_eq!(
                overlaps,
                vec![DependencyItem::new(DependencyDomain::SourceData, "data:lfs").unwrap()]
            );
        }
        other => panic!("unexpected relation: {other:?}"),
    }
}

#[test]
fn same_identification_strategy_is_shared_even_with_different_data() {
    let scope = full_scope();
    let left = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:a"),
            (
                DependencyDomain::IdentificationStrategy,
                "identify:iv-tax-shock-v1",
            ),
        ],
    );
    let right = lineage(
        "lineage:b",
        "paper:b",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:b"),
            (
                DependencyDomain::IdentificationStrategy,
                "identify:iv-tax-shock-v1",
            ),
        ],
    );

    assert!(matches!(
        compare_lineages(&left, &right, &scope).unwrap(),
        DependencyRelation::SharedDeclaredDependencies { .. }
    ));
}

#[test]
fn no_overlap_with_incomplete_inventory_does_not_become_disjoint_evidence() {
    let required = full_scope();
    let partial_scope: BTreeSet<_> = [
        DependencyDomain::SourceData,
        DependencyDomain::ModelArtifact,
    ]
    .into_iter()
    .collect();

    let left = lineage(
        "lineage:a",
        "paper:a",
        &partial_scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );
    let right = lineage(
        "lineage:b",
        "paper:b",
        &required,
        &[(DependencyDomain::SourceData, "data:b")],
    );

    match compare_lineages(&left, &right, &required).unwrap() {
        DependencyRelation::IncompleteInventory {
            missing_left,
            missing_right,
        } => {
            assert!(!missing_left.is_empty());
            assert!(missing_right.is_empty());
            assert!(missing_left.contains(&DependencyDomain::IdentificationStrategy));
        }
        other => panic!("unexpected relation: {other:?}"),
    }
}

#[test]
fn fully_declared_no_overlap_is_only_disjoint_within_the_predeclared_scope() {
    let scope = full_scope();
    let left = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:a"),
            (DependencyDomain::ModelArtifact, "model:a"),
        ],
    );
    let right = lineage(
        "lineage:b",
        "paper:b",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:b"),
            (DependencyDomain::ModelArtifact, "model:b"),
        ],
    );

    match compare_lineages(&left, &right, &scope).unwrap() {
        DependencyRelation::DeclaredDisjointWithinScope { scope: observed } => {
            assert_eq!(observed, scope.iter().copied().collect::<Vec<_>>());
        }
        other => panic!("unexpected relation: {other:?}"),
    }
}

#[test]
fn duplicate_publications_of_one_lineage_do_not_create_new_lineages() {
    let scope = full_scope();
    let first = lineage(
        "lineage:a",
        "paper:a1",
        &scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );
    let second = lineage(
        "lineage:a",
        "paper:a2",
        &scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );

    assert_eq!(
        compare_lineages(&first, &second, &scope).unwrap(),
        DependencyRelation::SameLineage
    );

    let audit = audit_replication_set(&[first, second], &scope).unwrap();
    assert_eq!(audit.total_contributions, 2);
    assert_eq!(audit.contribution_ids, vec!["paper:a1", "paper:a2"]);
    assert_eq!(audit.unique_lineages, 1);
    assert_eq!(audit.known_dependency_components, vec![vec!["lineage:a".into()]]);
}

#[test]
fn same_lineage_id_with_conflicting_inventory_fails_closed() {
    let scope = full_scope();
    let first = lineage(
        "lineage:a",
        "paper:a1",
        &scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );
    let second = lineage(
        "lineage:a",
        "paper:a2",
        &scope,
        &[(DependencyDomain::SourceData, "data:b")],
    );

    assert_eq!(
        compare_lineages(&first, &second, &scope),
        Err(DependencyError::LineageIdentityConflict)
    );
    assert_eq!(
        audit_replication_set(&[first, second], &scope),
        Err(DependencyError::LineageIdentityConflict)
    );
}

#[test]
fn known_dependency_graph_preserves_transitive_shared_lineage_clusters() {
    let scope = full_scope();
    let a = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[(DependencyDomain::SourceData, "data:shared-ab")],
    );
    let b = lineage(
        "lineage:b",
        "paper:b",
        &scope,
        &[
            (DependencyDomain::SourceData, "data:shared-ab"),
            (DependencyDomain::EstimatorArtifact, "estimator:shared-bc"),
        ],
    );
    let c = lineage(
        "lineage:c",
        "paper:c",
        &scope,
        &[(DependencyDomain::EstimatorArtifact, "estimator:shared-bc")],
    );
    let d = lineage(
        "lineage:d",
        "paper:d",
        &scope,
        &[(DependencyDomain::SourceData, "data:d")],
    );

    assert!(matches!(
        compare_lineages(&a, &c, &scope).unwrap(),
        DependencyRelation::DeclaredDisjointWithinScope { .. }
    ));

    let audit = audit_replication_set(&[a, b, c, d], &scope).unwrap();
    assert_eq!(audit.total_contributions, 4);
    assert_eq!(audit.unique_lineages, 4);
    assert_eq!(
        audit.known_dependency_components,
        vec![
            vec!["lineage:a".into(), "lineage:b".into(), "lineage:c".into()],
            vec!["lineage:d".into()],
        ]
    );
    assert!(audit.incomplete_lineages.is_empty());
}

#[test]
fn five_publications_with_one_shared_data_line_do_not_become_five_independent_replications() {
    let scope = full_scope();
    let lineages: Vec<_> = (0..5)
        .map(|index| {
            let model_id = format!("model:{index}");
            lineage(
                &format!("lineage:{index}"),
                &format!("paper:{index}"),
                &scope,
                &[
                    (DependencyDomain::SourceData, "data:shared"),
                    (DependencyDomain::ModelArtifact, model_id.as_str()),
                ],
            )
        })
        .collect();

    let audit = audit_replication_set(&lineages, &scope).unwrap();
    assert_eq!(audit.total_contributions, 5);
    assert_eq!(audit.contribution_ids.len(), 5);
    assert_eq!(audit.unique_lineages, 5);
    assert_eq!(audit.known_dependency_components.len(), 1);
    assert_eq!(audit.known_dependency_components[0].len(), 5);
}

#[test]
fn incomplete_lineages_are_retained_as_incomplete_in_set_audit() {
    let required = full_scope();
    let partial_scope: BTreeSet<_> = [DependencyDomain::SourceData].into_iter().collect();
    let incomplete = lineage(
        "lineage:partial",
        "paper:partial",
        &partial_scope,
        &[(DependencyDomain::SourceData, "data:partial")],
    );
    let complete = lineage(
        "lineage:complete",
        "paper:complete",
        &required,
        &[(DependencyDomain::SourceData, "data:complete")],
    );

    let audit = audit_replication_set(&[incomplete, complete], &required).unwrap();
    assert_eq!(audit.incomplete_lineages, vec!["lineage:partial".to_string()]);
}

#[test]
fn evidence_targets_must_match_before_replication_relationship_is_assessed() {
    let scope = full_scope();
    let left = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );
    let right = EvidenceLineage::new(
        "lineage:b",
        "paper:b",
        "claim:different",
        inventory(&scope, &[(DependencyDomain::SourceData, "data:b")]),
    )
    .unwrap();

    assert_eq!(
        compare_lineages(&left, &right, &scope),
        Err(DependencyError::EvidenceTargetMismatch)
    );
    assert_eq!(
        audit_replication_set(&[left, right], &scope),
        Err(DependencyError::MixedEvidenceTargets)
    );
}

#[test]
fn empty_comparison_scope_fails_closed() {
    let scope = full_scope();
    let left = lineage(
        "lineage:a",
        "paper:a",
        &scope,
        &[(DependencyDomain::SourceData, "data:a")],
    );
    let right = lineage(
        "lineage:b",
        "paper:b",
        &scope,
        &[(DependencyDomain::SourceData, "data:b")],
    );
    let empty = BTreeSet::new();

    assert_eq!(
        compare_lineages(&left, &right, &empty),
        Err(DependencyError::EmptyComparisonScope)
    );
    assert_eq!(
        audit_replication_set(&[left, right], &empty),
        Err(DependencyError::EmptyComparisonScope)
    );
}

#[test]
fn dependency_taxonomy_is_not_a_strength_ranking() {
    let domains = full_scope();
    assert_eq!(domains.len(), 10);
    let labels: Vec<_> = domains.iter().map(|domain| format!("{domain:?}")).collect();
    assert!(labels.contains(&"SourceData".to_string()));
    assert!(labels.contains(&"EvaluationProtocol".to_string()));
}
