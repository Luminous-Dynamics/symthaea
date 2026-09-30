//! SWA-006: immutable validation evidence + derived freshness.
//!
//! Validation evidence is historical fact. Whether it remains applicable to a
//! current model is derived from the dependency lineage; the historical record
//! is never mutated.
//!
//! This fixture deliberately keeps:
//! - evidence identity immutable;
//! - dependency revisions explicit;
//! - current freshness derived;
//! - invalidation reasons typed;
//! - revalidation requirements separate from evidence;
//! - authorization outside the evidence graph.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum DependencyKind {
    ModelRevision,
    ParameterSetRevision,
    SolverRevision,
    ScenarioRevision,
    DatasetRevision,
    UncertaintyModelRevision,
    ContextOfUseRevision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyRevision {
    kind: DependencyKind,
    id: &'static str,
    revision: u64,
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
    fn for_revision(
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
                kind: DependencyKind::ModelRevision,
                id: "model",
                revision: model,
            },
            parameters: DependencyRevision {
                kind: DependencyKind::ParameterSetRevision,
                id: "parameters",
                revision: parameters,
            },
            solver: DependencyRevision {
                kind: DependencyKind::SolverRevision,
                id: "solver",
                revision: solver,
            },
            scenario: DependencyRevision {
                kind: DependencyKind::ScenarioRevision,
                id: "scenario",
                revision: scenario,
            },
            dataset: DependencyRevision {
                kind: DependencyKind::DatasetRevision,
                id: "dataset",
                revision: dataset,
            },
            uncertainty: DependencyRevision {
                kind: DependencyKind::UncertaintyModelRevision,
                id: "uncertainty",
                revision: uncertainty,
            },
            context: DependencyRevision {
                kind: DependencyKind::ContextOfUseRevision,
                id: "context",
                revision: context,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ValidationEvidence {
    validation_id: &'static str,
    created_for: DependencyFingerprint,
    is_authorization: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum EvidenceFreshness {
    Valid,
    Stale,
    Invalidated,
    RequiresRevalidation,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum InvalidationReason {
    ModelRevisionChanged,
    ParameterSetRevisionChanged,
    SolverRevisionChanged,
    ScenarioRevisionChanged,
    DatasetRevisionChanged,
    UncertaintyModelRevisionChanged,
    ContextOfUseChanged,
    ProvenanceIncomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyChange {
    kind: DependencyKind,
    from_revision: Option<u64>,
    to_revision: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct RevalidationRequirement {
    validation_id: &'static str,
    freshness: EvidenceFreshness,
    reason: Option<InvalidationReason>,
    changed_dependency: Option<DependencyChange>,
}

fn classify_change(
    evidence: ValidationEvidence,
    current: DependencyFingerprint,
) -> (EvidenceFreshness, Option<InvalidationReason>, Option<DependencyChange>) {
    let pairs = [
        (
            evidence.created_for.model,
            current.model,
            InvalidationReason::ModelRevisionChanged,
        ),
        (
            evidence.created_for.parameters,
            current.parameters,
            InvalidationReason::ParameterSetRevisionChanged,
        ),
        (
            evidence.created_for.solver,
            current.solver,
            InvalidationReason::SolverRevisionChanged,
        ),
        (
            evidence.created_for.scenario,
            current.scenario,
            InvalidationReason::ScenarioRevisionChanged,
        ),
        (
            evidence.created_for.dataset,
            current.dataset,
            InvalidationReason::DatasetRevisionChanged,
        ),
        (
            evidence.created_for.uncertainty,
            current.uncertainty,
            InvalidationReason::UncertaintyModelRevisionChanged,
        ),
        (
            evidence.created_for.context,
            current.context,
            InvalidationReason::ContextOfUseChanged,
        ),
    ];

    for (historical, now, reason) in pairs {
        if historical.id != now.id || historical.kind != now.kind {
            return (
                EvidenceFreshness::Unknown,
                Some(InvalidationReason::ProvenanceIncomplete),
                None,
            );
        }

        if historical.revision != now.revision {
            return (
                EvidenceFreshness::Stale,
                Some(reason),
                Some(DependencyChange {
                    kind: now.kind,
                    from_revision: Some(historical.revision),
                    to_revision: Some(now.revision),
                }),
            );
        }
    }

    (EvidenceFreshness::Valid, None, None)
}

fn revalidation_frontier(
    evidence: &[ValidationEvidence],
    current: DependencyFingerprint,
) -> Vec<RevalidationRequirement> {
    evidence
        .iter()
        .map(|record| {
            let (freshness, reason, changed_dependency) = classify_change(*record, current);
            RevalidationRequirement {
                validation_id: record.validation_id,
                freshness,
                reason,
                changed_dependency,
            }
        })
        .collect()
}

fn main() {
    let v1 = DependencyFingerprint::for_revision(1, 1, 1, 1, 1, 1, 1);
    let unchanged = v1;
    let model_v2 = DependencyFingerprint::for_revision(2, 1, 1, 1, 1, 1, 1);
    let solver_v2 = DependencyFingerprint::for_revision(1, 1, 2, 1, 1, 1, 1);

    let evidence = [
        ValidationEvidence {
            validation_id: "validation-swa-006-001",
            created_for: v1,
            is_authorization: false,
        },
        ValidationEvidence {
            validation_id: "validation-swa-006-002",
            created_for: v1,
            is_authorization: false,
        },
    ];

    let unchanged_frontier = revalidation_frontier(&evidence, unchanged);
    assert!(unchanged_frontier
        .iter()
        .all(|r| r.freshness == EvidenceFreshness::Valid));

    let model_frontier = revalidation_frontier(&evidence, model_v2);
    assert!(model_frontier.iter().all(|r| {
        r.freshness == EvidenceFreshness::Stale
            && r.reason == Some(InvalidationReason::ModelRevisionChanged)
    }));

    let solver_frontier = revalidation_frontier(&evidence, solver_v2);
    assert!(solver_frontier.iter().all(|r| {
        r.freshness == EvidenceFreshness::Stale
            && r.reason == Some(InvalidationReason::SolverRevisionChanged)
    }));

    // Historical evidence itself is unchanged.
    assert_eq!(evidence[0].created_for, v1);
    assert!(!evidence[0].is_authorization);

    println!(
        "{}",
        serde_json::to_string(&model_frontier).expect("frontier serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> ValidationEvidence {
        ValidationEvidence {
            validation_id: "validation-test-001",
            created_for: DependencyFingerprint::for_revision(1, 1, 1, 1, 1, 1, 1),
            is_authorization: false,
        }
    }

    #[test]
    fn unchanged_dependencies_keep_evidence_valid() {
        let record = evidence();
        let (freshness, reason, change) = classify_change(record, record.created_for);
        assert_eq!(freshness, EvidenceFreshness::Valid);
        assert_eq!(reason, None);
        assert_eq!(change, None);
    }

    #[test]
    fn model_revision_makes_evidence_stale_without_mutating_history() {
        let record = evidence();
        let current = DependencyFingerprint::for_revision(2, 1, 1, 1, 1, 1, 1);
        let (freshness, reason, change) = classify_change(record, current);

        assert_eq!(freshness, EvidenceFreshness::Stale);
        assert_eq!(reason, Some(InvalidationReason::ModelRevisionChanged));
        assert_eq!(
            change,
            Some(DependencyChange {
                kind: DependencyKind::ModelRevision,
                from_revision: Some(1),
                to_revision: Some(2),
            })
        );
        assert_eq!(record.created_for.model.revision, 1);
    }

    #[test]
    fn parameter_change_is_not_model_form_failure() {
        let record = evidence();
        let current = DependencyFingerprint::for_revision(1, 2, 1, 1, 1, 1, 1);
        let (_, reason, change) = classify_change(record, current);

        assert_eq!(reason, Some(InvalidationReason::ParameterSetRevisionChanged));
        assert_eq!(
            change.expect("parameter change").kind,
            DependencyKind::ParameterSetRevision
        );
    }

    #[test]
    fn uncertainty_change_propagates_independently() {
        let record = evidence();
        let current = DependencyFingerprint::for_revision(1, 1, 1, 1, 1, 2, 1);
        let (_, reason, _) = classify_change(record, current);
        assert_eq!(
            reason,
            Some(InvalidationReason::UncertaintyModelRevisionChanged)
        );
    }

    #[test]
    fn context_change_requires_reassessment() {
        let record = evidence();
        let current = DependencyFingerprint::for_revision(1, 1, 1, 1, 1, 1, 2);
        let (_, reason, _) = classify_change(record, current);
        assert_eq!(reason, Some(InvalidationReason::ContextOfUseChanged));
    }

    #[test]
    fn revalidation_frontier_is_deterministic() {
        let records = [evidence()];
        let current = DependencyFingerprint::for_revision(2, 1, 1, 1, 1, 1, 1);
        let a = revalidation_frontier(&records, current);
        let b = revalidation_frontier(&records, current);

        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&a).expect("serialize a"),
            serde_json::to_string(&b).expect("serialize b")
        );
    }

    #[test]
    fn historical_evidence_never_becomes_authorization() {
        let record = evidence();
        assert!(!record.is_authorization);
    }

    #[test]
    fn provenance_identity_mismatch_is_unknown() {
        let mut current = evidence().created_for;
        current.model = DependencyRevision {
            kind: DependencyKind::ModelRevision,
            id: "different-model",
            revision: 1,
        };

        let (freshness, reason, change) = classify_change(evidence(), current);
        assert_eq!(freshness, EvidenceFreshness::Unknown);
        assert_eq!(
            reason,
            Some(InvalidationReason::ProvenanceIncomplete)
        );
        assert_eq!(change, None);
    }
}
