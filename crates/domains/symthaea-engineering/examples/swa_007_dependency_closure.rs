//! SWA-007: dependency closure and complete revalidation frontier.
//!
//! SWA-006 intentionally reported the first changed dependency. That is useful
//! for a minimal fixture, but an epistemic dependency graph must preserve the
//! complete delta and distinguish which evidence dimensions actually depend on
//! each changed input.
//!
//! This fixture makes both properties executable:
//! - every changed dependency is retained;
//! - each validation dimension declares its dependency closure;
//! - unrelated changes do not invalidate unrelated evidence;
//! - incomplete closure yields Unknown rather than a pass;
//! - historical evidence remains immutable;
//! - no result becomes authorization.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum DependencyKind {
    Model,
    Parameters,
    Solver,
    Scenario,
    Dataset,
    UncertaintyModel,
    ContextOfUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyRevision {
    kind: DependencyKind,
    id: &'static str,
    revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyChange {
    kind: DependencyKind,
    from: Option<u64>,
    to: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyClosure {
    dependencies: &'static [DependencyKind],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum EvidenceDimension {
    Comfort,
    Energy,
    Resilience,
    Uncertainty,
    Applicability,
}

impl EvidenceDimension {
    fn closure(self) -> DependencyClosure {
        match self {
            Self::Comfort => DependencyClosure {
                dependencies: &[
                    DependencyKind::Model,
                    DependencyKind::Parameters,
                    DependencyKind::Solver,
                    DependencyKind::Scenario,
                    DependencyKind::Dataset,
                ],
            },
            Self::Energy => DependencyClosure {
                dependencies: &[
                    DependencyKind::Model,
                    DependencyKind::Parameters,
                    DependencyKind::Solver,
                    DependencyKind::Scenario,
                    DependencyKind::Dataset,
                ],
            },
            Self::Resilience => DependencyClosure {
                dependencies: &[
                    DependencyKind::Model,
                    DependencyKind::Parameters,
                    DependencyKind::Scenario,
                    DependencyKind::Dataset,
                ],
            },
            Self::Uncertainty => DependencyClosure {
                dependencies: &[
                    DependencyKind::Model,
                    DependencyKind::Parameters,
                    DependencyKind::Solver,
                    DependencyKind::Dataset,
                    DependencyKind::UncertaintyModel,
                ],
            },
            Self::Applicability => DependencyClosure {
                dependencies: &[
                    DependencyKind::Model,
                    DependencyKind::Scenario,
                    DependencyKind::Dataset,
                    DependencyKind::ContextOfUse,
                ],
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyFingerprint {
    model: DependencyRevision,
    parameters: DependencyRevision,
    solver: DependencyRevision,
    scenario: DependencyRevision,
    dataset: DependencyRevision,
    uncertainty: DependencyRevision,
    context: DependencyRevision,
}

impl DependencyFingerprint {
    fn revision(
        model: u64,
        parameters: u64,
        solver: u64,
        scenario: u64,
        dataset: u64,
        uncertainty: u64,
        context: u64,
    ) -> Self {
        Self {
            model: DependencyRevision {
                kind: DependencyKind::Model,
                id: "model",
                revision: model,
            },
            parameters: DependencyRevision {
                kind: DependencyKind::Parameters,
                id: "parameters",
                revision: parameters,
            },
            solver: DependencyRevision {
                kind: DependencyKind::Solver,
                id: "solver",
                revision: solver,
            },
            scenario: DependencyRevision {
                kind: DependencyKind::Scenario,
                id: "scenario",
                revision: scenario,
            },
            dataset: DependencyRevision {
                kind: DependencyKind::Dataset,
                id: "dataset",
                revision: dataset,
            },
            uncertainty: DependencyRevision {
                kind: DependencyKind::UncertaintyModel,
                id: "uncertainty",
                revision: uncertainty,
            },
            context: DependencyRevision {
                kind: DependencyKind::ContextOfUse,
                id: "context",
                revision: context,
            },
        }
    }

    fn get(self, kind: DependencyKind) -> DependencyRevision {
        match kind {
            DependencyKind::Model => self.model,
            DependencyKind::Parameters => self.parameters,
            DependencyKind::Solver => self.solver,
            DependencyKind::Scenario => self.scenario,
            DependencyKind::Dataset => self.dataset,
            DependencyKind::UncertaintyModel => self.uncertainty,
            DependencyKind::ContextOfUse => self.context,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ValidationEvidence {
    id: &'static str,
    dimension: EvidenceDimension,
    dependencies: DependencyFingerprint,
    is_authorization: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Freshness {
    Valid,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct FrontierEntry {
    evidence_id: &'static str,
    dimension: EvidenceDimension,
    freshness: Freshness,
    changes: Vec<DependencyChange>,
}

fn changed_dependencies(
    historical: DependencyFingerprint,
    current: DependencyFingerprint,
) -> Option<Vec<DependencyChange>> {
    let mut changes = Vec::new();

    for kind in [
        DependencyKind::Model,
        DependencyKind::Parameters,
        DependencyKind::Solver,
        DependencyKind::Scenario,
        DependencyKind::Dataset,
        DependencyKind::UncertaintyModel,
        DependencyKind::ContextOfUse,
    ] {
        let old = historical.get(kind);
        let new = current.get(kind);

        if old.id != new.id || old.kind != new.kind {
            return None;
        }

        if old.revision != new.revision {
            changes.push(DependencyChange {
                kind,
                from: Some(old.revision),
                to: Some(new.revision),
            });
        }
    }

    Some(changes)
}

fn frontier(
    evidence: &[ValidationEvidence],
    current: DependencyFingerprint,
) -> Vec<FrontierEntry> {
    evidence
        .iter()
        .map(|record| {
            let changes = changed_dependencies(record.dependencies, current);

            match changes {
                None => FrontierEntry {
                    evidence_id: record.id,
                    dimension: record.dimension,
                    freshness: Freshness::Unknown,
                    changes: Vec::new(),
                },
                Some(all_changes) => {
                    let relevant = all_changes
                        .iter()
                        .filter(|change| {
                            record
                                .dimension
                                .closure()
                                .dependencies
                                .contains(&change.kind)
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    FrontierEntry {
                        evidence_id: record.id,
                        dimension: record.dimension,
                        freshness: if relevant.is_empty() {
                            Freshness::Valid
                        } else {
                            Freshness::Stale
                        },
                        changes: relevant,
                    }
                }
            }
        })
        .collect()
}

fn main() {
    let v1 = DependencyFingerprint::revision(1, 1, 1, 1, 1, 1, 1);
    let changed_everything = DependencyFingerprint::revision(2, 2, 2, 2, 2, 2, 2);

    let evidence = [
        ValidationEvidence {
            id: "comfort-001",
            dimension: EvidenceDimension::Comfort,
            dependencies: v1,
            is_authorization: false,
        },
        ValidationEvidence {
            id: "resilience-001",
            dimension: EvidenceDimension::Resilience,
            dependencies: v1,
            is_authorization: false,
        },
        ValidationEvidence {
            id: "uncertainty-001",
            dimension: EvidenceDimension::Uncertainty,
            dependencies: v1,
            is_authorization: false,
        },
        ValidationEvidence {
            id: "applicability-001",
            dimension: EvidenceDimension::Applicability,
            dependencies: v1,
            is_authorization: false,
        },
    ];

    let result = frontier(&evidence, changed_everything);

    assert!(result.iter().all(|entry| entry.freshness == Freshness::Stale));
    assert!(result
        .iter()
        .find(|entry| entry.dimension == EvidenceDimension::Resilience)
        .expect("resilience")
        .changes
        .iter()
        .all(|change| change.kind != DependencyKind::Solver));
    assert!(result
        .iter()
        .find(|entry| entry.dimension == EvidenceDimension::Applicability)
        .expect("applicability")
        .changes
        .iter()
        .any(|change| change.kind == DependencyKind::ContextOfUse));

    // The evidence itself is still the v1 historical record.
    assert_eq!(evidence[0].dependencies, v1);
    assert!(!evidence[0].is_authorization);

    println!(
        "{}",
        serde_json::to_string(&result).expect("frontier serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> ValidationEvidence {
        ValidationEvidence {
            id: "resilience-test",
            dimension: EvidenceDimension::Resilience,
            dependencies: DependencyFingerprint::revision(1, 1, 1, 1, 1, 1, 1),
            is_authorization: false,
        }
    }

    #[test]
    fn complete_delta_keeps_all_changed_dependencies() {
        let old = evidence().dependencies;
        let new = DependencyFingerprint::revision(2, 2, 2, 2, 2, 2, 2);
        let changes = changed_dependencies(old, new).expect("complete provenance");

        assert_eq!(changes.len(), 7);
        assert!(changes.iter().any(|c| c.kind == DependencyKind::Model));
        assert!(changes.iter().any(|c| c.kind == DependencyKind::ContextOfUse));
    }

    #[test]
    fn unrelated_solver_change_does_not_stale_resilience_evidence() {
        let record = evidence();
        let current = DependencyFingerprint::revision(1, 1, 2, 1, 1, 1, 1);
        let result = frontier(&[record], current);

        assert_eq!(result[0].freshness, Freshness::Valid);
        assert!(result[0].changes.is_empty());
    }

    #[test]
    fn parameter_change_does_stale_resilience_evidence() {
        let record = evidence();
        let current = DependencyFingerprint::revision(1, 2, 1, 1, 1, 1, 1);
        let result = frontier(&[record], current);

        assert_eq!(result[0].freshness, Freshness::Stale);
        assert_eq!(result[0].changes.len(), 1);
        assert_eq!(result[0].changes[0].kind, DependencyKind::Parameters);
    }

    #[test]
    fn context_change_only_affects_applicability() {
        let record = evidence();
        let applicability = ValidationEvidence {
            id: "applicability-test",
            dimension: EvidenceDimension::Applicability,
            dependencies: record.dependencies,
            is_authorization: false,
        };
        let current = DependencyFingerprint::revision(1, 1, 1, 1, 1, 1, 2);

        let result = frontier(&[record, applicability], current);

        assert_eq!(result[0].freshness, Freshness::Valid);
        assert_eq!(result[1].freshness, Freshness::Stale);
        assert_eq!(result[1].changes[0].kind, DependencyKind::ContextOfUse);
    }

    #[test]
    fn incomplete_provenance_is_unknown_not_valid() {
        let record = evidence();
        let mut current = record.dependencies;
        current.model = DependencyRevision {
            kind: DependencyKind::Model,
            id: "unknown-model",
            revision: 1,
        };

        let result = frontier(&[record], current);

        assert_eq!(result[0].freshness, Freshness::Unknown);
    }

    #[test]
    fn historical_record_is_unchanged_and_not_authorization() {
        let record = evidence();
        let current = DependencyFingerprint::revision(2, 2, 2, 2, 2, 2, 2);
        let _ = frontier(&[record], current);

        assert_eq!(record.dependencies.model.revision, 1);
        assert!(!record.is_authorization);
    }

    #[test]
    fn frontier_is_deterministic() {
        let record = evidence();
        let current = DependencyFingerprint::revision(2, 1, 1, 1, 1, 1, 1);

        let first = frontier(&[record], current);
        let second = frontier(&[record], current);

        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_string(&first).expect("serialize first"),
            serde_json::to_string(&second).expect("serialize second")
        );
    }
}
