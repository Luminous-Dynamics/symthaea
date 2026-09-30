//! SWA-017: end-to-end intervention revalidation lifecycle fixture.
//!
//! Composes the contracts exercised in SWA-011..016 into one deterministic
//! lifecycle. This is a reference scenario, not a production authorization,
//! Holochain zome, building controller, or physical actuation interface.
//!
//! Historical authorization is immutable. Readiness is recomputed against
//! exact intervention/scenario revisions and evidence dependencies.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum EvidenceRelation { Supports, Qualifies, Contradicts }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Readiness { Current, ReopenForReview, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Authorization { NotAuthorized, AuthorizedByHuman, AuthorizedByGovernance }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Intervention {
    id: String,
    revision: u64,
    mechanism: String,
    commitment_months: u32,
    reversible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Scenario {
    id: String,
    revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Evidence {
    id: String,
    intervention_id: String,
    intervention_revision: u64,
    scenario_id: String,
    scenario_revision: u64,
    model_revision: u64,
    relation: EvidenceRelation,
    requires_revalidation: bool,
    unknown: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DecisionContext {
    id: String,
    intervention_id: String,
    intervention_revision: u64,
    scenario_id: String,
    scenario_revision: u64,
    required_evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HistoricalAuthorization {
    context_id: String,
    intervention_id: String,
    intervention_revision: u64,
    scenario_id: String,
    scenario_revision: u64,
    authority: Authorization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LifecycleSnapshot {
    stage: String,
    readiness: Readiness,
    context_id: String,
    intervention_revision: u64,
    scenario_revision: u64,
    evidence_ids: Vec<String>,
    historical_authorization: Authorization,
    physical_actuation: bool,
}

fn readiness(
    context: &DecisionContext,
    intervention: &Intervention,
    scenario: &Scenario,
    evidence: &[Evidence],
) -> Readiness {
    if context.intervention_id != intervention.id
        || context.intervention_revision != intervention.revision
        || context.scenario_id != scenario.id
        || context.scenario_revision != scenario.revision
    {
        return Readiness::ReopenForReview;
    }

    for required in &context.required_evidence {
        let Some(item) = evidence.iter().find(|item| &item.id == required) else {
            return Readiness::Unknown;
        };
        if item.unknown { return Readiness::Unknown; }
        if item.intervention_id != intervention.id
            || item.intervention_revision != intervention.revision
            || item.scenario_id != scenario.id
            || item.scenario_revision != scenario.revision
            || item.requires_revalidation
        {
            return Readiness::ReopenForReview;
        }
    }
    Readiness::Current
}

fn snapshot(
    stage: &str,
    context: &DecisionContext,
    intervention: &Intervention,
    scenario: &Scenario,
    evidence: &[Evidence],
    authorization: &HistoricalAuthorization,
) -> LifecycleSnapshot {
    let mut evidence_ids = evidence.iter().map(|item| item.id.clone()).collect::<Vec<_>>();
    evidence_ids.sort();
    evidence_ids.dedup();
    LifecycleSnapshot {
        stage: stage.to_owned(),
        readiness: readiness(context, intervention, scenario, evidence),
        context_id: context.id.clone(),
        intervention_revision: intervention.revision,
        scenario_revision: scenario.revision,
        evidence_ids,
        historical_authorization: authorization.authority,
        physical_actuation: false,
    }
}

/// Deterministic example lifecycle. Each state is an observation of the
/// review boundary; no state transition performs or implies actuation.
fn lifecycle() -> Vec<LifecycleSnapshot> {
    let v1 = Intervention {
        id: "heat-pump-retrofit".into(), revision: 1,
        mechanism: "zoned_heat_pump".into(), commitment_months: 12, reversible: true,
    };
    let scenario_v1 = Scenario { id: "winter-occupancy-profile".into(), revision: 1 };
    let evidence_v1 = vec![
        Evidence {
            id: "comfort-witness".into(), intervention_id: v1.id.clone(),
            intervention_revision: 1, scenario_id: scenario_v1.id.clone(),
            scenario_revision: 1, model_revision: 7, relation: EvidenceRelation::Supports,
            requires_revalidation: false, unknown: false,
        },
        Evidence {
            id: "energy-counterevidence".into(), intervention_id: v1.id.clone(),
            intervention_revision: 1, scenario_id: scenario_v1.id.clone(),
            scenario_revision: 1, model_revision: 7, relation: EvidenceRelation::Contradicts,
            requires_revalidation: false, unknown: false,
        },
        Evidence {
            id: "comfort-qualification".into(), intervention_id: v1.id.clone(),
            intervention_revision: 1, scenario_id: scenario_v1.id.clone(),
            scenario_revision: 1, model_revision: 7, relation: EvidenceRelation::Qualifies,
            requires_revalidation: false, unknown: false,
        },
    ];
    let context_v1 = DecisionContext {
        id: "review-context-v1".into(), intervention_id: v1.id.clone(),
        intervention_revision: 1, scenario_id: scenario_v1.id.clone(),
        scenario_revision: 1,
        required_evidence: vec!["comfort-witness".into(), "energy-counterevidence".into(),
            "comfort-qualification".into()],
    };
    let authorized_v1 = HistoricalAuthorization {
        context_id: context_v1.id.clone(), intervention_id: v1.id.clone(),
        intervention_revision: 1, scenario_id: scenario_v1.id.clone(),
        scenario_revision: 1, authority: Authorization::AuthorizedByHuman,
    };

    let mut snapshots = vec![snapshot("v1_review_ready", &context_v1, &v1,
        &scenario_v1, &evidence_v1, &authorized_v1)];

    // A model update makes model-derived evidence require revalidation;
    // evidence records and the prior authorization remain historical.
    let mut stale_evidence = evidence_v1.clone();
    for item in &mut stale_evidence {
        if item.model_revision == 7 { item.requires_revalidation = true; }
    }
    snapshots.push(snapshot("model_revision_7_to_8", &context_v1, &v1,
        &scenario_v1, &stale_evidence, &authorized_v1));

    // New intervention revision cannot inherit v1 evidence or authorization.
    let v2 = Intervention { revision: 2, mechanism: "zoned_heat_pump_with_storage".into(),
        ..v1.clone() };
    snapshots.push(snapshot("intervention_v2_with_v1_evidence", &context_v1, &v2,
        &scenario_v1, &stale_evidence, &authorized_v1));

    let context_v2 = DecisionContext { id: "review-context-v2".into(),
        intervention_revision: 2, ..context_v1.clone() };
    let v2_evidence = vec![
        Evidence { id: "v2-comfort-witness".into(), intervention_id: v2.id.clone(),
            intervention_revision: 2, scenario_id: scenario_v1.id.clone(),
            scenario_revision: 1, model_revision: 8, relation: EvidenceRelation::Supports,
            requires_revalidation: false, unknown: false },
        Evidence { id: "v2-energy-counterevidence".into(), intervention_id: v2.id.clone(),
            intervention_revision: 2, scenario_id: scenario_v1.id.clone(),
            scenario_revision: 1, model_revision: 8, relation: EvidenceRelation::Contradicts,
            requires_revalidation: false, unknown: false },
    ];
    let context_v2 = DecisionContext { required_evidence: vec![
        "v2-comfort-witness".into(), "v2-energy-counterevidence".into()],
        ..context_v2 };
    snapshots.push(snapshot("v2_fresh_evidence", &context_v2, &v2,
        &scenario_v1, &v2_evidence, &authorized_v1));

    let authorized_v2 = HistoricalAuthorization { context_id: context_v2.id.clone(),
        intervention_id: v2.id.clone(), intervention_revision: 2,
        scenario_id: scenario_v1.id.clone(), scenario_revision: 1,
        authority: Authorization::NotAuthorized };
    snapshots.push(snapshot("v2_context_awaiting_external_authority", &context_v2,
        &v2, &scenario_v1, &v2_evidence, &authorized_v2));
    snapshots
}

fn main() {
    let trace = lifecycle();
    for item in &trace {
        println!("{}: {:?}, intervention r{}, physical_actuation={}",
            item.stage, item.readiness, item.intervention_revision, item.physical_actuation);
    }
    assert!(trace.iter().all(|item| !item.physical_actuation));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_context_is_current_but_readiness_is_not_authorization() {
        let trace = lifecycle();
        assert_eq!(trace[0].readiness, Readiness::Current);
        assert_eq!(trace[0].historical_authorization, Authorization::AuthorizedByHuman);
        assert!(!trace[0].physical_actuation);
    }

    #[test]
    fn model_revision_reopens_review_without_rewriting_history() {
        let trace = lifecycle();
        assert_eq!(trace[1].readiness, Readiness::ReopenForReview);
        assert_eq!(trace[1].historical_authorization, Authorization::AuthorizedByHuman);
    }

    #[test]
    fn intervention_revision_cannot_inherit_old_evidence() {
        let trace = lifecycle();
        assert_eq!(trace[2].readiness, Readiness::ReopenForReview);
        assert_eq!(trace[2].intervention_revision, 2);
    }

    #[test]
    fn new_revision_requires_new_review_context_and_evidence() {
        let trace = lifecycle();
        assert_eq!(trace[3].readiness, Readiness::Current);
        assert_ne!(trace[0].context_id, trace[3].context_id);
        assert_eq!(trace[4].historical_authorization, Authorization::NotAuthorized);
    }

    #[test]
    fn missing_evidence_is_unknown() {
        let mut trace = lifecycle();
        let mut item = trace.remove(0);
        let context = DecisionContext { required_evidence: vec!["absent".into()], ..DecisionContext {
            id: item.context_id.clone(), intervention_id: "heat-pump-retrofit".into(),
            intervention_revision: 1, scenario_id: "winter-occupancy-profile".into(),
            scenario_revision: 1, required_evidence: vec![] } };
        let intervention = Intervention { id: "heat-pump-retrofit".into(), revision: 1,
            mechanism: "zoned_heat_pump".into(), commitment_months: 12, reversible: true };
        let scenario = Scenario { id: "winter-occupancy-profile".into(), revision: 1 };
        assert_eq!(readiness(&context, &intervention, &scenario, &[]), Readiness::Unknown);
    }

    #[test]
    fn contradiction_and_qualification_survive_lifecycle() {
        let trace = lifecycle();
        assert!(trace[0].evidence_ids.contains(&"energy-counterevidence".into()));
        assert!(trace[0].evidence_ids.contains(&"comfort-qualification".into()));
        assert!(trace[3].evidence_ids.contains(&"v2-energy-counterevidence".into()));
    }

    #[test]
    fn lifecycle_trace_is_deterministic() {
        assert_eq!(lifecycle(), lifecycle());
    }

    #[test]
    fn no_physical_actuation_is_representable_as_a_transition() {
        assert!(lifecycle().iter().all(|item| !item.physical_actuation));
    }

    #[test]
    fn historical_authorization_is_revision_bound() {
        let trace = lifecycle();
        assert_eq!(trace[0].intervention_revision, 1);
        assert_eq!(trace[2].intervention_revision, 2);
        assert_eq!(trace[2].historical_authorization, Authorization::AuthorizedByHuman);
        assert_ne!(trace[0].context_id, trace[3].context_id);
    }
}
