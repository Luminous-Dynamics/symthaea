//! SWA-016: deterministic intervention/authorization-context binding.
//!
//! SWA-015 answers whether required epistemic evidence is still current.
//! SWA-016 closes the other half of that boundary: the decision context must
//! also be bound to the exact intervention and scenario that were reviewed.
//!
//! A fresh evidence set cannot silently authorize a changed intervention.
//! Likewise, changing an intervention does not rewrite the historical record.
//!
//! Boundary:
//! intervention revision + scenario revision + evidence freshness
//!   -> authorization-context readiness
//!   -> external human/governance authorization
//!
//! This fixture never grants, revokes, or executes physical authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
enum DependencyKind { Model, Parameters, Solver, Scenario, Dataset, UncertaintyModel, ContextOfUse, Intervention }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Revision { kind: DependencyKind, id: String, revision: u64 }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct InterventionSpec {
    id: String,
    revision: u64,
    mechanism: String,
    commitment_months: u32,
    reconfiguration_cost_cents: u64,
    exit_cost_cents: u64,
    reversibility_basis: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ScenarioSpec {
    id: String,
    revision: u64,
    occupancy_profile: String,
    infrastructure_dependency: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EvidenceRef {
    id: String,
    intervention_id: String,
    intervention_revision: u64,
    scenario_id: String,
    scenario_revision: u64,
    requires_revalidation: bool,
    unknown: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AuthorizationContext {
    id: String,
    intervention: Revision,
    scenario: Revision,
    required_evidence_ids: Vec<String>,
    historical_authorization: HistoricalAuthorization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum HistoricalAuthorization { NotAuthorized, AuthorizedByHuman, AuthorizedByGovernance }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Readiness { Current, ReopenForReview, Unknown }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ReadinessReason {
    Current,
    EvidenceRequiresRevalidation { evidence_ids: Vec<String> },
    InterventionRevisionChanged { expected: u64, current: u64 },
    ScenarioRevisionChanged { expected: u64, current: u64 },
    EvidenceBindsToDifferentIntervention { evidence_id: String },
    EvidenceBindsToDifferentScenario { evidence_id: String },
    MissingEvidence { evidence_ids: Vec<String> },
    UnknownEvidence { evidence_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AuthorizationReadiness {
    context_id: String,
    readiness: Readiness,
    reason: ReadinessReason,
    historical_authorization: HistoricalAuthorization,
}

fn canonical_ids(ids: &[String]) -> Vec<String> {
    let mut result = ids.to_vec();
    result.sort();
    result.dedup();
    result
}

/// Bind a decision context to the exact intervention and scenario reviewed.
/// This is a readiness calculation, not an authorization calculation.
fn assess_readiness(
    context: &AuthorizationContext,
    current_intervention: &InterventionSpec,
    current_scenario: &ScenarioSpec,
    evidence: &[EvidenceRef],
) -> AuthorizationReadiness {
    if context.intervention.kind != DependencyKind::Intervention
        || context.scenario.kind != DependencyKind::Scenario
    {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::Unknown,
            reason: ReadinessReason::UnknownEvidence { evidence_ids: Vec::new() },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    if context.intervention.id != current_intervention.id
        || context.intervention.revision != current_intervention.revision
    {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::ReopenForReview,
            reason: ReadinessReason::InterventionRevisionChanged {
                expected: context.intervention.revision,
                current: current_intervention.revision,
            },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    if context.scenario.id != current_scenario.id
        || context.scenario.revision != current_scenario.revision
    {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::ReopenForReview,
            reason: ReadinessReason::ScenarioRevisionChanged {
                expected: context.scenario.revision,
                current: current_scenario.revision,
            },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    let required: BTreeSet<&str> = context.required_evidence_ids.iter().map(String::as_str).collect();
    let known: BTreeSet<&str> = evidence.iter().map(|item| item.id.as_str()).collect();

    let missing = canonical_ids(
        &required.iter().filter(|id| !known.contains(**id)).map(|id| (*id).to_owned()).collect::<Vec<_>>(),
    );
    if !missing.is_empty() {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::Unknown,
            reason: ReadinessReason::MissingEvidence { evidence_ids: missing },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    let mut unknown = Vec::new();
    let mut stale = Vec::new();

    for item in evidence.iter().filter(|item| required.contains(item.id.as_str())) {
        if item.intervention_id != current_intervention.id {
            return AuthorizationReadiness {
                context_id: context.id.clone(),
                readiness: Readiness::ReopenForReview,
                reason: ReadinessReason::EvidenceBindsToDifferentIntervention { evidence_id: item.id.clone() },
                historical_authorization: context.historical_authorization.clone(),
            };
        }
        if item.intervention_revision != current_intervention.revision {
            stale.push(item.id.clone());
        }
        if item.scenario_id != current_scenario.id {
            return AuthorizationReadiness {
                context_id: context.id.clone(),
                readiness: Readiness::ReopenForReview,
                reason: ReadinessReason::EvidenceBindsToDifferentScenario { evidence_id: item.id.clone() },
                historical_authorization: context.historical_authorization.clone(),
            };
        }
        if item.scenario_revision != current_scenario.revision {
            stale.push(item.id.clone());
        }
        if item.unknown {
            unknown.push(item.id.clone());
        } else if item.requires_revalidation {
            stale.push(item.id.clone());
        }
    }

    let unknown = canonical_ids(&unknown);
    if !unknown.is_empty() {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::Unknown,
            reason: ReadinessReason::UnknownEvidence { evidence_ids: unknown },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    let stale = canonical_ids(&stale);
    if !stale.is_empty() {
        return AuthorizationReadiness {
            context_id: context.id.clone(),
            readiness: Readiness::ReopenForReview,
            reason: ReadinessReason::EvidenceRequiresRevalidation { evidence_ids: stale },
            historical_authorization: context.historical_authorization.clone(),
        };
    }

    AuthorizationReadiness {
        context_id: context.id.clone(),
        readiness: Readiness::Current,
        reason: ReadinessReason::Current,
        historical_authorization: context.historical_authorization.clone(),
    }
}

fn main() {
    let intervention = InterventionSpec {
        id: "workspace-intervention-modular-hvac".into(),
        revision: 2,
        mechanism: "reversible-zone-control".into(),
        commitment_months: 3,
        reconfiguration_cost_cents: 25_000,
        exit_cost_cents: 10_000,
        reversibility_basis: "low commitment and bounded exit cost".into(),
    };
    let scenario = ScenarioSpec {
        id: "building-001-summer-profile".into(),
        revision: 4,
        occupancy_profile: "aggregate-zone-profile-v2".into(),
        infrastructure_dependency: "grid-feed-001".into(),
    };
    let context = AuthorizationContext {
        id: "decision-context-001".into(),
        intervention: Revision { kind: DependencyKind::Intervention, id: intervention.id.clone(), revision: intervention.revision },
        scenario: Revision { kind: DependencyKind::Scenario, id: scenario.id.clone(), revision: scenario.revision },
        required_evidence_ids: vec!["comfort-witness".into(), "energy-witness".into()],
        historical_authorization: HistoricalAuthorization::NotAuthorized,
    };
    let evidence = vec![
        EvidenceRef { id: "comfort-witness".into(), intervention_id: intervention.id.clone(), intervention_revision: intervention.revision, scenario_id: scenario.id.clone(), scenario_revision: scenario.revision, requires_revalidation: false, unknown: false },
        EvidenceRef { id: "energy-witness".into(), intervention_id: intervention.id.clone(), intervention_revision: intervention.revision, scenario_id: scenario.id.clone(), scenario_revision: scenario.revision, requires_revalidation: false, unknown: false },
    ];
    let result = assess_readiness(&context, &intervention, &scenario, &evidence);
    assert_eq!(result.readiness, Readiness::Current);
    assert_eq!(result.historical_authorization, HistoricalAuthorization::NotAuthorized);
    println!("{result:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intervention(revision: u64) -> InterventionSpec {
        InterventionSpec { id: "intervention-001".into(), revision, mechanism: "reversible".into(), commitment_months: 3, reconfiguration_cost_cents: 100, exit_cost_cents: 50, reversibility_basis: "bounded exit".into() }
    }
    fn scenario(revision: u64) -> ScenarioSpec {
        ScenarioSpec { id: "scenario-001".into(), revision, occupancy_profile: "aggregate-v1".into(), infrastructure_dependency: "grid-001".into() }
    }
    fn context() -> AuthorizationContext {
        AuthorizationContext {
            id: "context-001".into(),
            intervention: Revision { kind: DependencyKind::Intervention, id: "intervention-001".into(), revision: 1 },
            scenario: Revision { kind: DependencyKind::Scenario, id: "scenario-001".into(), revision: 1 },
            required_evidence_ids: vec!["evidence-a".into(), "evidence-b".into()],
            historical_authorization: HistoricalAuthorization::AuthorizedByHuman,
        }
    }
    fn evidence(intervention_revision: u64, scenario_revision: u64) -> Vec<EvidenceRef> {
        vec![
            EvidenceRef { id: "evidence-a".into(), intervention_id: "intervention-001".into(), intervention_revision, scenario_id: "scenario-001".into(), scenario_revision, requires_revalidation: false, unknown: false },
            EvidenceRef { id: "evidence-b".into(), intervention_id: "intervention-001".into(), intervention_revision, scenario_id: "scenario-001".into(), scenario_revision, requires_revalidation: false, unknown: false },
        ]
    }

    #[test]
    fn unchanged_binding_is_current_but_not_authorization() {
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &evidence(1, 1));
        assert_eq!(result.readiness, Readiness::Current);
        assert_eq!(result.historical_authorization, HistoricalAuthorization::AuthorizedByHuman);
    }

    #[test]
    fn intervention_revision_change_reopens_review() {
        let result = assess_readiness(&context(), &intervention(2), &scenario(1), &evidence(1, 1));
        assert_eq!(result.readiness, Readiness::ReopenForReview);
        assert_eq!(result.reason, ReadinessReason::InterventionRevisionChanged { expected: 1, current: 2 });
    }

    #[test]
    fn scenario_revision_change_reopens_review() {
        let result = assess_readiness(&context(), &intervention(1), &scenario(2), &evidence(1, 1));
        assert_eq!(result.readiness, Readiness::ReopenForReview);
    }

    #[test]
    fn stale_evidence_revision_reopens_review() {
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &evidence(0, 1));
        assert_eq!(result.readiness, Readiness::ReopenForReview);
        assert!(matches!(result.reason, ReadinessReason::EvidenceRequiresRevalidation { .. }));
    }

    #[test]
    fn missing_evidence_is_unknown() {
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &[evidence(1, 1)[0].clone()]);
        assert_eq!(result.readiness, Readiness::Unknown);
        assert!(matches!(result.reason, ReadinessReason::MissingEvidence { .. }));
    }

    #[test]
    fn evidence_from_different_intervention_reopens_review() {
        let mut items = evidence(1, 1);
        items[0].intervention_id = "other-intervention".into();
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &items);
        assert_eq!(result.readiness, Readiness::ReopenForReview);
        assert!(matches!(result.reason, ReadinessReason::EvidenceBindsToDifferentIntervention { .. }));
    }

    #[test]
    fn unknown_evidence_fails_closed() {
        let mut items = evidence(1, 1);
        items[0].unknown = true;
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &items);
        assert_eq!(result.readiness, Readiness::Unknown);
    }

    #[test]
    fn historical_authorization_is_immutable() {
        let before = context();
        let result = assess_readiness(&before, &intervention(2), &scenario(1), &evidence(1, 1));
        assert_eq!(result.readiness, Readiness::ReopenForReview);
        assert_eq!(result.historical_authorization, HistoricalAuthorization::AuthorizedByHuman);
        assert_eq!(before.historical_authorization, HistoricalAuthorization::AuthorizedByHuman);
    }

    #[test]
    fn equivalent_input_order_is_deterministic() {
        let first = assess_readiness(&context(), &intervention(1), &scenario(1), &evidence(1, 1));
        let mut reversed = evidence(1, 1);
        reversed.reverse();
        let second = assess_readiness(&context(), &intervention(1), &scenario(1), &reversed);
        assert_eq!(first, second);
    }

    #[test]
    fn current_does_not_grant_physical_authority() {
        let result = assess_readiness(&context(), &intervention(1), &scenario(1), &evidence(1, 1));
        assert_eq!(result.readiness, Readiness::Current);
    }
}
