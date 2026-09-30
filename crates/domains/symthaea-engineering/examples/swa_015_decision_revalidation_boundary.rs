//! SWA-015: deterministic decision revalidation boundary.
//!
//! This fixture connects the SWA-014 invalidation frontier to pending
//! intervention/decision contexts without turning epistemic state into
//! authorization.
//!
//! Boundary:
//! dependency change
//!   -> invalidation propagation
//!   -> evidence/witness freshness
//!   -> decision-context revalidation
//!   -> human/governance authorization (outside this fixture)
//!
//! Invariants:
//! - ReopenForReview != Reject.
//! - RequiresRevalidation != False.
//! - Unknown provenance never becomes Current.
//! - Historical decision context and authorization are immutable.
//! - A decision context is affected only by evidence it explicitly requires.
//! - Contradictory evidence remains visible; this layer does not resolve it.
//! - Output ordering is deterministic.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
enum DependencyKind {
    Model,
    Parameters,
    Solver,
    Scenario,
    Dataset,
    UncertaintyModel,
    ContextOfUse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DependencyChange {
    kind: DependencyKind,
    from: u64,
    to: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EvidenceStatus {
    id: String,
    current: bool,
    unknown: bool,
    requires_revalidation: bool,
    historical_unchanged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum AuthorizationState {
    NotAuthorized,
    AuthorizedByHuman,
    AuthorizedByGovernance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DecisionContext {
    id: String,
    candidate_intervention_id: String,
    required_evidence_ids: Vec<String>,
    historical_authorization: AuthorizationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ReviewStatus {
    Current,
    ReopenForReview,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum RevalidationReason {
    AffectedEvidence {
        evidence_ids: Vec<String>,
        dependency_changes: Vec<DependencyChange>,
    },
    UnknownEvidence {
        evidence_ids: Vec<String>,
    },
    NoAffectedEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DecisionRevalidation {
    decision_id: String,
    candidate_intervention_id: String,
    status: ReviewStatus,
    reason: RevalidationReason,
    historical_authorization: AuthorizationState,
}

fn canonical_changes(changes: &[DependencyChange]) -> Vec<DependencyChange> {
    let mut changes = changes.to_vec();
    changes.sort_by_key(|change| (change.kind, change.from, change.to));
    changes.dedup();
    changes
}

fn canonical_ids(ids: &[String]) -> Vec<String> {
    let mut ids = ids.to_vec();
    ids.sort();
    ids.dedup();
    ids
}

/// Determine whether a pending decision context must be reopened.
///
/// This function intentionally has no authorization output. It can identify
/// an epistemic prerequisite for review, but cannot authorize, reject, or
/// execute the intervention.
fn revalidate_decision(
    decision: &DecisionContext,
    evidence: &[EvidenceStatus],
    changes: &[DependencyChange],
) -> DecisionRevalidation {
    let required: BTreeSet<&str> = decision
        .required_evidence_ids
        .iter()
        .map(String::as_str)
        .collect();

    let mut affected = Vec::new();
    let mut unknown = Vec::new();

    for item in evidence {
        if required.contains(item.id.as_str()) {
            if item.unknown || !item.historical_unchanged {
                unknown.push(item.id.clone());
            } else if item.requires_revalidation {
                affected.push(item.id.clone());
            }
        }
    }

    // A required evidence reference that cannot be found is itself unknown.
    let known: BTreeSet<&str> = evidence.iter().map(|item| item.id.as_str()).collect();
    for id in &required {
        if !known.contains(id) {
            unknown.push((*id).to_owned());
        }
    }

    let affected = canonical_ids(&affected);
    let unknown = canonical_ids(&unknown);
    let changes = canonical_changes(changes);

    let (status, reason) = if !unknown.is_empty() {
        (
            ReviewStatus::Unknown,
            RevalidationReason::UnknownEvidence {
                evidence_ids: unknown,
            },
        )
    } else if !affected.is_empty() {
        (
            ReviewStatus::ReopenForReview,
            RevalidationReason::AffectedEvidence {
                evidence_ids: affected,
                dependency_changes: changes,
            },
        )
    } else {
        (ReviewStatus::Current, RevalidationReason::NoAffectedEvidence)
    };

    DecisionRevalidation {
        decision_id: decision.id.clone(),
        candidate_intervention_id: decision.candidate_intervention_id.clone(),
        status,
        reason,
        historical_authorization: decision.historical_authorization.clone(),
    }
}

fn main() {
    let decision = DecisionContext {
        id: "decision-001".into(),
        candidate_intervention_id: "intervention-heat-pump".into(),
        required_evidence_ids: vec![
            "evidence-comfort".into(),
            "evidence-energy".into(),
            "witness-counterexample".into(),
        ],
        historical_authorization: AuthorizationState::NotAuthorized,
    };

    let changes = vec![DependencyChange {
        kind: DependencyKind::Model,
        from: 7,
        to: 8,
    }];

    let evidence = vec![
        EvidenceStatus {
            id: "evidence-comfort".into(),
            current: false,
            unknown: false,
            requires_revalidation: true,
            historical_unchanged: true,
        },
        EvidenceStatus {
            id: "evidence-energy".into(),
            current: true,
            unknown: false,
            requires_revalidation: false,
            historical_unchanged: true,
        },
        EvidenceStatus {
            id: "witness-counterexample".into(),
            current: false,
            unknown: false,
            requires_revalidation: true,
            historical_unchanged: true,
        },
    ];

    let result = revalidate_decision(&decision, &evidence, &changes);
    assert_eq!(result.status, ReviewStatus::ReopenForReview);

    println!("SWA-015 decision revalidation: {:?}", result.status);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision() -> DecisionContext {
        DecisionContext {
            id: "decision-001".into(),
            candidate_intervention_id: "intervention-001".into(),
            required_evidence_ids: vec!["evidence-a".into(), "evidence-b".into()],
            historical_authorization: AuthorizationState::AuthorizedByHuman,
        }
    }

    fn evidence(
        id: &str,
        requires_revalidation: bool,
        unknown: bool,
    ) -> EvidenceStatus {
        EvidenceStatus {
            id: id.into(),
            current: !requires_revalidation && !unknown,
            unknown,
            requires_revalidation,
            historical_unchanged: true,
        }
    }

    fn model_change() -> DependencyChange {
        DependencyChange {
            kind: DependencyKind::Model,
            from: 7,
            to: 8,
        }
    }

    #[test]
    fn affected_required_evidence_reopens_decision_for_review() {
        let result = revalidate_decision(
            &decision(),
            &[evidence("evidence-a", true, false), evidence("evidence-b", false, false)],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::ReopenForReview);
        assert!(matches!(
            result.reason,
            RevalidationReason::AffectedEvidence { .. }
        ));
    }

    #[test]
    fn unrelated_stale_evidence_does_not_reopen_decision() {
        let result = revalidate_decision(
            &decision(),
            &[
                evidence("evidence-a", false, false),
                evidence("evidence-b", false, false),
                evidence("unrelated", true, false),
            ],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::Current);
        assert_eq!(result.reason, RevalidationReason::NoAffectedEvidence);
    }

    #[test]
    fn unknown_required_evidence_fails_closed() {
        let result = revalidate_decision(
            &decision(),
            &[evidence("evidence-a", false, false)],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::Unknown);
        assert_eq!(
            result.reason,
            RevalidationReason::UnknownEvidence {
                evidence_ids: vec!["evidence-b".into()]
            }
        );
    }

    #[test]
    fn explicitly_unknown_required_evidence_fails_closed() {
        let result = revalidate_decision(
            &decision(),
            &[
                evidence("evidence-a", false, false),
                evidence("evidence-b", false, true),
            ],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::Unknown);
    }

    #[test]
    fn historical_authorization_is_not_changed_by_revalidation() {
        let before = decision();
        let result = revalidate_decision(
            &before,
            &[evidence("evidence-a", true, false), evidence("evidence-b", false, false)],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::ReopenForReview);
        assert_eq!(result.historical_authorization, AuthorizationState::AuthorizedByHuman);
        assert_eq!(before.historical_authorization, AuthorizationState::AuthorizedByHuman);
    }

    #[test]
    fn reopening_does_not_mean_rejection() {
        let result = revalidate_decision(
            &decision(),
            &[evidence("evidence-a", true, false), evidence("evidence-b", false, false)],
            &[model_change()],
        );

        assert_eq!(result.status, ReviewStatus::ReopenForReview);
        assert!(!matches!(result.status, ReviewStatus::Current));
    }

    #[test]
    fn contradiction_remains_a_review_input_not_a_resolution() {
        let result = revalidate_decision(
            &decision(),
            &[
                evidence("evidence-a", true, false),
                evidence("evidence-b", false, false),
            ],
            &[model_change()],
        );

        // The boundary only consumes freshness state. It does not decide
        // whether contradictory evidence wins.
        assert_eq!(result.status, ReviewStatus::ReopenForReview);
    }

    #[test]
    fn duplicate_and_unsorted_inputs_are_deterministic() {
        let first = revalidate_decision(
            &decision(),
            &[evidence("evidence-b", true, false), evidence("evidence-a", true, false)],
            &[
                model_change(),
                model_change(),
                DependencyChange {
                    kind: DependencyKind::Parameters,
                    from: 3,
                    to: 4,
                },
            ],
        );

        let second = revalidate_decision(
            &decision(),
            &[evidence("evidence-a", true, false), evidence("evidence-b", true, false)],
            &[
                DependencyChange {
                    kind: DependencyKind::Parameters,
                    from: 3,
                    to: 4,
                },
                model_change(),
            ],
        );

        assert_eq!(first, second);
    }

    #[test]
    fn historical_evidence_is_not_mutated() {
        let evidence = evidence("evidence-a", true, false);
        let before = evidence.clone();

        let _ = revalidate_decision(
            &decision(),
            &[evidence.clone(), evidence("evidence-b", false, false)],
            &[model_change()],
        );

        assert_eq!(evidence, before);
        assert!(evidence.historical_unchanged);
    }
}
