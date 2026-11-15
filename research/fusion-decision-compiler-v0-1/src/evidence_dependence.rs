//! Observable-scoped solver dependency comparison for FECOMP.
//!
//! This module refines the compiler's coarse dataset/code/assumption groups.
//! It describes declared dependency inventories per output observable. It
//! does not prove that declarations are complete or truthful, authenticate
//! source bytes, estimate covariance, or establish physical independence.

use std::collections::BTreeSet;

pub const DEPENDENCE_CONTRACT_VERSION: &str = "fusion-observable-dependence/0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DependencyKind {
    SolverBuildLineage,
    EquationOrModelFamily,
    EmpiricalClosureOrCorrelation,
    InputDataset,
    CalibrationDataset,
    BoundaryCondition,
    InitialCondition,
    GeometryOrConfiguration,
    PreprocessingOrTransformation,
    MeasurementSystem,
    MeasurementDataset,
    MaterialPropertyDataset,
    SharedScenarioAssumption,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyIdentity {
    /// Stable identity for the source or dependency family.
    pub node_id: String,
    /// Exact source/model revision when one is available.
    pub revision: String,
    pub kind: DependencyKind,
    /// Optional SHA-256 of the referenced payload. A digest is only an
    /// identity declaration until the payload verifier checks the bytes.
    pub content_sha256: Option<String>,
    /// Optional reference to a source/run/evidence receipt.
    pub source_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservableDependency {
    pub observable_id: String,
    pub dependency: DependencyIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunDependencyManifest {
    pub run_id: String,
    /// Observables actually emitted by this run and admitted to comparison.
    pub output_observables: Vec<String>,
    /// Observables for which the producer declares the dependency inventory
    /// complete. This is a declaration, not an independently proven fact.
    pub complete_observables: Vec<String>,
    pub dependencies: Vec<ObservableDependency>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DependencyMatchBasis {
    SameNodeLineage,
    SameContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedDependency {
    pub observable_id: String,
    pub kind: DependencyKind,
    pub first_node_id: String,
    pub first_revision: String,
    pub second_node_id: String,
    pub second_revision: String,
    pub match_basis: DependencyMatchBasis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableDependenceState {
    /// One or more dependencies are known to be shared. Completeness flags
    /// must still be inspected because additional common causes may be absent.
    SharedDependencyFound,
    /// No shared dependency was found, but at least one inventory is incomplete.
    UnknownDependencyInventory,
    /// No shared dependency was found and both parties declare complete
    /// inventories for this observable. This is not proof of physical or
    /// statistical independence.
    NoSharedDependencyInCompleteDeclarations,
    /// One or both runs did not produce the requested observable.
    NotComparable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservableDependenceAssessment {
    pub observable_id: String,
    pub state: ObservableDependenceState,
    pub first_inventory_complete: bool,
    pub second_inventory_complete: bool,
    pub first_dependencies_without_evidence_ref: usize,
    pub second_dependencies_without_evidence_ref: usize,
    pub shared_dependencies: Vec<SharedDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyValidationError {
    EmptyRunId,
    DuplicateRunIdentity(String),
    EmptyOutputObservable,
    DuplicateOutputObservable(String),
    EmptyCompleteObservable,
    DuplicateCompleteObservable(String),
    CompletenessForUnknownObservable(String),
    DependencyForUnknownObservable(String),
    EmptyDependencyNodeId,
    EmptyDependencyRevision,
    EmptySourceEvidenceRef,
    InvalidContentSha256,
    DuplicateDependencyBinding {
        observable_id: String,
        node_id: String,
    },
    ConflictingDependencyKind(String),
    EmptyRequestedObservable,
    DuplicateRequestedObservable(String),
    EmptyRequestedObservables,
}

/// Compare only the requested observable scopes. A shared node ID indicates
/// common lineage even when exact revisions differ. Identical content digests
/// also expose shared payloads even when callers use different local names.
///
/// The result is deliberately not a statistical-independence test. Unknown
/// inventories remain unknown; missing cross-covariance is handled by the
/// separate covariance contract and must never be replaced by zero here.
pub fn compare_observable_dependencies(
    first: &RunDependencyManifest,
    second: &RunDependencyManifest,
    requested_observables: &[String],
) -> Result<Vec<ObservableDependenceAssessment>, Vec<DependencyValidationError>> {
    let mut errors = validate_manifest(first);
    errors.extend(validate_manifest(second));

    if first.run_id == second.run_id {
        errors.push(DependencyValidationError::DuplicateRunIdentity(
            first.run_id.clone(),
        ));
    }
    errors.extend(validate_cross_manifest_kinds(first, second));

    if requested_observables.is_empty() {
        errors.push(DependencyValidationError::EmptyRequestedObservables);
    }
    let mut requested_seen = BTreeSet::new();
    for observable in requested_observables {
        if observable.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyRequestedObservable);
        }
        if !requested_seen.insert(observable.as_str()) {
            errors.push(DependencyValidationError::DuplicateRequestedObservable(
                observable.clone(),
            ));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut assessments = Vec::with_capacity(requested_observables.len());
    for observable_id in requested_observables {
        let first_has_output = first
            .output_observables
            .iter()
            .any(|value| value == observable_id);
        let second_has_output = second
            .output_observables
            .iter()
            .any(|value| value == observable_id);
        let first_complete = first
            .complete_observables
            .iter()
            .any(|value| value == observable_id);
        let second_complete = second
            .complete_observables
            .iter()
            .any(|value| value == observable_id);

        let first_dependencies = dependencies_for(first, observable_id);
        let second_dependencies = dependencies_for(second, observable_id);
        let mut shared_dependencies =
            find_shared_dependencies(observable_id, &first_dependencies, &second_dependencies);

        shared_dependencies.sort_by(|left, right| {
            (
                left.first_node_id.as_str(),
                left.first_revision.as_str(),
                left.second_node_id.as_str(),
                left.second_revision.as_str(),
                left.kind,
                left.match_basis,
            )
                .cmp(&(
                    right.first_node_id.as_str(),
                    right.first_revision.as_str(),
                    right.second_node_id.as_str(),
                    right.second_revision.as_str(),
                    right.kind,
                    right.match_basis,
                ))
        });

        let state = if !first_has_output || !second_has_output {
            ObservableDependenceState::NotComparable
        } else if !shared_dependencies.is_empty() {
            ObservableDependenceState::SharedDependencyFound
        } else if !first_complete || !second_complete {
            ObservableDependenceState::UnknownDependencyInventory
        } else {
            ObservableDependenceState::NoSharedDependencyInCompleteDeclarations
        };

        assessments.push(ObservableDependenceAssessment {
            observable_id: observable_id.clone(),
            state,
            first_inventory_complete: first_complete,
            second_inventory_complete: second_complete,
            first_dependencies_without_evidence_ref: first_dependencies
                .iter()
                .filter(|item| item.dependency.source_evidence_ref.is_none())
                .count(),
            second_dependencies_without_evidence_ref: second_dependencies
                .iter()
                .filter(|item| item.dependency.source_evidence_ref.is_none())
                .count(),
            shared_dependencies,
        });
    }

    assessments.sort_by(|left, right| left.observable_id.cmp(&right.observable_id));
    Ok(assessments)
}

fn validate_manifest(manifest: &RunDependencyManifest) -> Vec<DependencyValidationError> {
    let mut errors = Vec::new();
    if manifest.run_id.trim().is_empty() {
        errors.push(DependencyValidationError::EmptyRunId);
    }

    let mut outputs = BTreeSet::new();
    for observable in &manifest.output_observables {
        if observable.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyOutputObservable);
        }
        if !outputs.insert(observable.as_str()) {
            errors.push(DependencyValidationError::DuplicateOutputObservable(
                observable.clone(),
            ));
        }
    }

    let mut complete = BTreeSet::new();
    for observable in &manifest.complete_observables {
        if observable.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyCompleteObservable);
        }
        if !complete.insert(observable.as_str()) {
            errors.push(DependencyValidationError::DuplicateCompleteObservable(
                observable.clone(),
            ));
        }
        if !outputs.contains(observable.as_str()) {
            errors.push(DependencyValidationError::CompletenessForUnknownObservable(
                observable.clone(),
            ));
        }
    }

    let mut bindings = BTreeSet::new();
    let mut node_kinds = std::collections::BTreeMap::new();
    for item in &manifest.dependencies {
        if item.observable_id.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyOutputObservable);
        }
        if !outputs.contains(item.observable_id.as_str()) {
            errors.push(DependencyValidationError::DependencyForUnknownObservable(
                item.observable_id.clone(),
            ));
        }

        let dependency = &item.dependency;
        if dependency.node_id.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyDependencyNodeId);
        }
        if dependency.revision.trim().is_empty() {
            errors.push(DependencyValidationError::EmptyDependencyRevision);
        }
        if let Some(reference) = dependency.source_evidence_ref.as_deref()
            && reference.trim().is_empty()
        {
            errors.push(DependencyValidationError::EmptySourceEvidenceRef);
        }
        if let Some(digest) = dependency.content_sha256.as_deref()
            && !is_sha256(digest)
        {
            errors.push(DependencyValidationError::InvalidContentSha256);
        }

        if !bindings.insert((item.observable_id.as_str(), dependency.node_id.as_str())) {
            errors.push(DependencyValidationError::DuplicateDependencyBinding {
                observable_id: item.observable_id.clone(),
                node_id: dependency.node_id.clone(),
            });
        }

        if let Some(previous_kind) = node_kinds.insert(&dependency.node_id, dependency.kind)
            && previous_kind != dependency.kind
        {
            errors.push(DependencyValidationError::ConflictingDependencyKind(
                dependency.node_id.clone(),
            ));
        }
    }

    errors
}

fn validate_cross_manifest_kinds(
    first: &RunDependencyManifest,
    second: &RunDependencyManifest,
) -> Vec<DependencyValidationError> {
    let mut errors = Vec::new();
    for left in &first.dependencies {
        for right in &second.dependencies {
            if left.observable_id != right.observable_id {
                continue;
            }
            let same_node = left.dependency.node_id == right.dependency.node_id;
            let same_payload = left.dependency.content_sha256.is_some()
                && right.dependency.content_sha256.is_some()
                && left
                    .dependency
                    .content_sha256
                    .as_deref()
                    .zip(right.dependency.content_sha256.as_deref())
                    .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b));
            if (same_node || same_payload) && left.dependency.kind != right.dependency.kind {
                errors.push(DependencyValidationError::ConflictingDependencyKind(
                    if same_node {
                        left.dependency.node_id.clone()
                    } else {
                        format!(
                            "payload:{}",
                            left.dependency
                                .content_sha256
                                .as_deref()
                                .unwrap_or_default()
                        )
                    },
                ));
            }
        }
    }
    errors
}

fn dependencies_for<'a>(
    manifest: &'a RunDependencyManifest,
    observable_id: &str,
) -> Vec<&'a ObservableDependency> {
    manifest
        .dependencies
        .iter()
        .filter(|item| item.observable_id == observable_id)
        .collect()
}

fn find_shared_dependencies(
    observable_id: &str,
    first: &[&ObservableDependency],
    second: &[&ObservableDependency],
) -> Vec<SharedDependency> {
    let mut shared = Vec::new();
    for left in first {
        for right in second {
            let left_ref = &left.dependency;
            let right_ref = &right.dependency;

            let match_basis = if left_ref.node_id == right_ref.node_id {
                Some(DependencyMatchBasis::SameNodeLineage)
            } else if left_ref.content_sha256.is_some()
                && right_ref.content_sha256.is_some()
                && left_ref
                    .content_sha256
                    .as_deref()
                    .zip(right_ref.content_sha256.as_deref())
                    .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
            {
                Some(DependencyMatchBasis::SameContentDigest)
            } else {
                None
            };

            if let Some(match_basis) = match_basis {
                shared.push(SharedDependency {
                    observable_id: observable_id.to_string(),
                    kind: left_ref.kind,
                    first_node_id: left_ref.node_id.clone(),
                    first_revision: left_ref.revision.clone(),
                    second_node_id: right_ref.node_id.clone(),
                    second_revision: right_ref.revision.clone(),
                    match_basis,
                });
            }
        }
    }
    shared
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dependency(
        node_id: &str,
        revision: &str,
        kind: DependencyKind,
        digest: Option<&str>,
    ) -> DependencyIdentity {
        DependencyIdentity {
            node_id: node_id.to_string(),
            revision: revision.to_string(),
            kind,
            content_sha256: digest.map(str::to_string),
            source_evidence_ref: Some(format!("source-evidence-{node_id}")),
        }
    }

    fn binding(observable_id: &str, dependency: DependencyIdentity) -> ObservableDependency {
        ObservableDependency {
            observable_id: observable_id.to_string(),
            dependency,
        }
    }

    fn manifest(
        run_id: &str,
        outputs: &[&str],
        complete: &[&str],
        dependencies: Vec<ObservableDependency>,
    ) -> RunDependencyManifest {
        RunDependencyManifest {
            run_id: run_id.to_string(),
            output_observables: outputs.iter().map(|value| value.to_string()).collect(),
            complete_observables: complete.iter().map(|value| value.to_string()).collect(),
            dependencies,
        }
    }

    fn compare_one(
        first: &RunDependencyManifest,
        second: &RunDependencyManifest,
        observable: &str,
    ) -> ObservableDependenceAssessment {
        compare_observable_dependencies(first, second, &[observable.to_string()])
            .unwrap()
            .remove(0)
    }

    #[test]
    fn shared_equation_is_scoped_to_only_the_observable_that_uses_it() {
        let equation = dependency(
            "gs-equation-family",
            "rev-4",
            DependencyKind::EquationOrModelFamily,
            None,
        );
        let first = manifest(
            "run-a",
            &["psi", "q95"],
            &["psi", "q95"],
            vec![
                binding("psi", equation.clone()),
                binding(
                    "q95",
                    dependency(
                        "q95-closure-a",
                        "rev-1",
                        DependencyKind::EmpiricalClosureOrCorrelation,
                        None,
                    ),
                ),
            ],
        );
        let second = manifest(
            "run-b",
            &["psi", "q95"],
            &["psi", "q95"],
            vec![
                binding("psi", equation),
                binding(
                    "q95",
                    dependency(
                        "q95-closure-b",
                        "rev-1",
                        DependencyKind::EmpiricalClosureOrCorrelation,
                        None,
                    ),
                ),
            ],
        );

        let assessments =
            compare_observable_dependencies(&first, &second, &["psi".into(), "q95".into()])
                .unwrap();
        assert_eq!(assessments.len(), 2);
        assert_eq!(assessments[0].observable_id, "psi");
        assert_eq!(
            assessments[0].state,
            ObservableDependenceState::SharedDependencyFound
        );
        assert_eq!(
            assessments[1].state,
            ObservableDependenceState::NoSharedDependencyInCompleteDeclarations
        );
    }

    #[test]
    fn same_node_with_different_revisions_is_shared_lineage() {
        let first = manifest(
            "run-a",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency(
                    "solver-project",
                    "commit-a",
                    DependencyKind::SolverBuildLineage,
                    None,
                ),
            )],
        );
        let second = manifest(
            "run-b",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency(
                    "solver-project",
                    "commit-b",
                    DependencyKind::SolverBuildLineage,
                    None,
                ),
            )],
        );

        let result = compare_one(&first, &second, "psi");
        assert_eq!(
            result.state,
            ObservableDependenceState::SharedDependencyFound
        );
        assert_eq!(
            result.shared_dependencies[0].match_basis,
            DependencyMatchBasis::SameNodeLineage
        );
        assert_ne!(
            result.shared_dependencies[0].first_revision,
            result.shared_dependencies[0].second_revision
        );
    }

    #[test]
    fn identical_payload_digest_detects_aliases_with_different_node_ids() {
        let digest = "a".repeat(64);
        let first = manifest(
            "run-a",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency(
                    "local-name-a",
                    "rev-a",
                    DependencyKind::InputDataset,
                    Some(&digest),
                ),
            )],
        );
        let second = manifest(
            "run-b",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency(
                    "local-name-b",
                    "rev-b",
                    DependencyKind::InputDataset,
                    Some(&digest),
                ),
            )],
        );

        let result = compare_one(&first, &second, "psi");
        assert_eq!(
            result.state,
            ObservableDependenceState::SharedDependencyFound
        );
        assert_eq!(
            result.shared_dependencies[0].match_basis,
            DependencyMatchBasis::SameContentDigest
        );
    }

    #[test]
    fn disjoint_complete_inventories_are_only_independent_under_declarations() {
        let first = manifest(
            "run-a",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency("dataset-a", "v1", DependencyKind::InputDataset, None),
            )],
        );
        let second = manifest(
            "run-b",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency("dataset-b", "v1", DependencyKind::InputDataset, None),
            )],
        );

        let result = compare_one(&first, &second, "psi");
        assert_eq!(
            result.state,
            ObservableDependenceState::NoSharedDependencyInCompleteDeclarations
        );
    }

    #[test]
    fn missing_inventory_completeness_never_becomes_independence() {
        let first = manifest(
            "run-a",
            &["psi"],
            &[],
            vec![binding(
                "psi",
                dependency("dataset-a", "v1", DependencyKind::InputDataset, None),
            )],
        );
        let second = manifest("run-b", &["psi"], &["psi"], Vec::new());

        let result = compare_one(&first, &second, "psi");
        assert_eq!(
            result.state,
            ObservableDependenceState::UnknownDependencyInventory
        );
        assert!(!result.first_inventory_complete);
        assert!(result.second_inventory_complete);
    }

    #[test]
    fn known_shared_dependency_remains_visible_when_other_lineage_is_incomplete() {
        let shared = dependency(
            "shared-calibration",
            "cal-v2",
            DependencyKind::CalibrationDataset,
            None,
        );
        let first = manifest("run-a", &["q95"], &[], vec![binding("q95", shared.clone())]);
        let second = manifest("run-b", &["q95"], &["q95"], vec![binding("q95", shared)]);

        let result = compare_one(&first, &second, "q95");
        assert_eq!(
            result.state,
            ObservableDependenceState::SharedDependencyFound
        );
        assert!(!result.first_inventory_complete);
        assert_eq!(result.shared_dependencies.len(), 1);
    }

    #[test]
    fn missing_output_is_not_comparable() {
        let first = manifest("run-a", &["psi"], &["psi"], Vec::new());
        let second = manifest("run-b", &["q95"], &["q95"], Vec::new());

        let result = compare_one(&first, &second, "psi");
        assert_eq!(result.state, ObservableDependenceState::NotComparable);
    }

    #[test]
    fn duplicate_bindings_and_malformed_digests_are_rejected() {
        let item = binding(
            "psi",
            dependency(
                "dataset-a",
                "v1",
                DependencyKind::InputDataset,
                Some("not-a-sha"),
            ),
        );
        let first = manifest(
            "run-a",
            &["psi"],
            &["psi"],
            vec![item.clone(), item.clone()],
        );
        let second = manifest("run-b", &["psi"], &["psi"], Vec::new());

        let errors = compare_observable_dependencies(&first, &second, &["psi".into()]).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, DependencyValidationError::InvalidContentSha256))
        );
        assert!(errors.iter().any(|error| matches!(
            error,
            DependencyValidationError::DuplicateDependencyBinding { .. }
        )));
    }

    #[test]
    fn assessments_are_sorted_and_shared_dependencies_are_deterministic() {
        let first = manifest(
            "run-a",
            &["psi", "q95"],
            &["psi", "q95"],
            vec![
                binding(
                    "psi",
                    dependency("shared-z", "v1", DependencyKind::InputDataset, None),
                ),
                binding(
                    "psi",
                    dependency("shared-a", "v1", DependencyKind::CalibrationDataset, None),
                ),
            ],
        );
        let second = manifest(
            "run-b",
            &["psi", "q95"],
            &["psi", "q95"],
            vec![
                binding(
                    "psi",
                    dependency("shared-a", "v2", DependencyKind::CalibrationDataset, None),
                ),
                binding(
                    "psi",
                    dependency("shared-z", "v2", DependencyKind::InputDataset, None),
                ),
            ],
        );

        let result =
            compare_observable_dependencies(&first, &second, &["q95".into(), "psi".into()])
                .unwrap();
        assert_eq!(result[0].observable_id, "psi");
        assert_eq!(result[1].observable_id, "q95");
        assert_eq!(result[0].shared_dependencies.len(), 2);
        assert_eq!(result[0].shared_dependencies[0].first_node_id, "shared-a");
        assert_eq!(result[0].shared_dependencies[1].first_node_id, "shared-z");
    }
    #[test]
    fn comparing_a_run_to_itself_is_rejected() {
        let run = manifest("run-a", &["psi"], &["psi"], Vec::new());
        let errors = compare_observable_dependencies(&run, &run, &["psi".into()]).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DependencyValidationError::DuplicateRunIdentity(run_id) if run_id == "run-a"
        )));
    }

    #[test]
    fn shared_node_with_conflicting_kinds_is_rejected() {
        let first = manifest(
            "run-a",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency("ambiguous-source", "v1", DependencyKind::InputDataset, None),
            )],
        );
        let second = manifest(
            "run-b",
            &["psi"],
            &["psi"],
            vec![binding(
                "psi",
                dependency(
                    "ambiguous-source",
                    "v1",
                    DependencyKind::CalibrationDataset,
                    None,
                ),
            )],
        );
        let errors = compare_observable_dependencies(&first, &second, &["psi".into()]).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DependencyValidationError::ConflictingDependencyKind(_)
        )));
    }
}
