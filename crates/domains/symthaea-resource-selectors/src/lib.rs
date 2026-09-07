// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic control selectors for resource-planning experiments.
//!
//! These selectors deliberately add no new feasibility or authority semantics.
//! They consume already-closed planning contexts and emit only the same
//! `ResourcePlannerProposal` contract validated by `symthaea-resource-recommendation`.
//!
//! They are intended as explicit experimental controls for later HDC/learned
//! selectors, not as claims about the universally correct resource policy.

#![deny(unsafe_code)]

use symthaea_resource_feasible_set::DeterministicFeasibleSet;
use symthaea_resource_pareto::ResourceParetoRanking;
use symthaea_resource_recommendation::{
    validate_resource_planner_proposal, RecommendationError, ResourcePlannerProposal,
    ValidatedResourcePlannerDecision,
};

pub const CANONICAL_FEASIBLE_SELECTOR_ID: &str = "baseline/canonical-feasible-v1";
pub const PARETO_UNIQUE_SELECTOR_ID: &str = "pareto/unique-frontier-v1";
pub const PARETO_CANONICAL_SELECTOR_ID: &str = "pareto/canonical-frontier-id-v1";

/// Objective-free deterministic baseline.
///
/// Chooses the first candidate in #754's canonical feasible ordering. If no
/// feasible candidate exists, explicitly abstains. This is intentionally simple:
/// it exists as a reproducible control, not as an optimized policy.
pub fn canonical_feasible_proposal(
    feasible_set: &DeterministicFeasibleSet,
) -> ResourcePlannerProposal {
    match feasible_set.feasible().first() {
        Some(candidate) => ResourcePlannerProposal::Recommend {
            selector_id: CANONICAL_FEASIBLE_SELECTOR_ID.into(),
            candidate_id: candidate.candidate_id().into(),
        },
        None => ResourcePlannerProposal::Abstain {
            selector_id: CANONICAL_FEASIBLE_SELECTOR_ID.into(),
        },
    }
}

/// Pareto control that refuses to hide an unresolved tradeoff.
///
/// Recommends only when the non-dominated frontier contains exactly one candidate;
/// otherwise it explicitly abstains.
pub fn pareto_unique_frontier_proposal(
    ranking: &ResourceParetoRanking,
) -> ResourcePlannerProposal {
    let frontier = ranking.frontier_ids();
    if frontier.len() == 1 {
        ResourcePlannerProposal::Recommend {
            selector_id: PARETO_UNIQUE_SELECTOR_ID.into(),
            candidate_id: frontier[0].into(),
        }
    } else {
        ResourcePlannerProposal::Abstain {
            selector_id: PARETO_UNIQUE_SELECTOR_ID.into(),
        }
    }
}

/// Pareto benchmark control with an explicit, visible tie-break.
///
/// When several non-dominated alternatives remain, choose the lexicographically
/// smallest candidate ID. This is not presented as substantive preference; the
/// selector ID names the tie-break so benchmark results cannot hide it.
pub fn pareto_canonical_frontier_proposal(
    ranking: &ResourceParetoRanking,
) -> ResourcePlannerProposal {
    match ranking.frontier_ids().into_iter().min() {
        Some(candidate_id) => ResourcePlannerProposal::Recommend {
            selector_id: PARETO_CANONICAL_SELECTOR_ID.into(),
            candidate_id: candidate_id.into(),
        },
        None => ResourcePlannerProposal::Abstain {
            selector_id: PARETO_CANONICAL_SELECTOR_ID.into(),
        },
    }
}

pub fn validated_canonical_feasible(
    feasible_set: &DeterministicFeasibleSet,
) -> Result<ValidatedResourcePlannerDecision, RecommendationError> {
    validate_resource_planner_proposal(feasible_set, canonical_feasible_proposal(feasible_set))
}

pub fn validated_pareto_unique(
    ranking: &ResourceParetoRanking,
) -> Result<ValidatedResourcePlannerDecision, RecommendationError> {
    let feasible_set = ranking.evidence_set().feasible_set();
    validate_resource_planner_proposal(feasible_set, pareto_unique_frontier_proposal(ranking))
}

pub fn validated_pareto_canonical(
    ranking: &ResourceParetoRanking,
) -> Result<ValidatedResourcePlannerDecision, RecommendationError> {
    let feasible_set = ranking.evidence_set().feasible_set();
    validate_resource_planner_proposal(feasible_set, pareto_canonical_frontier_proposal(ranking))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use symthaea_operations_research::ObjectiveDirection;
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::enumerate_feasible_set;
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_objective_evidence::{
        CandidateObjectiveEvidence, ObjectiveEvidenceClass, ObjectiveEvidenceScope,
        ResourceObjectiveEvidenceSet,
    };
    use symthaea_resource_objective_metric::{ObjectiveMetric, ObjectiveStatistic};
    use symthaea_resource_objective_qualification::{
        ObjectiveEvidenceIdentityPolicy, ObjectiveMultipleEvidenceRule,
        ObjectiveQualificationPolicy, ObjectiveQualificationRequest,
    };
    use symthaea_resource_pareto::{rank_feasible_resources, ResourceParetoObjective};
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
        ResourceQualityRequirement,
    };
    use symthaea_resource_quality_evidence::{
        QualityEvidenceClass, QualityEvidenceSet, QualityEvidenceWindow, QualitySubject,
    };
    use symthaea_resource_quality_qualification::{
        EvidenceIdentityPolicy, MultipleEvidenceRule, QualityQualificationPolicy,
    };
    use symthaea_resource_recommendation::ValidatedResourcePlannerDecision;
    use symthaea_resource_topology::{ResourceLink, ResourceTopology};

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn topology() -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology.add_node("source", [ResourcePort { id: "out".into(), direction: PortDirection::Output, capacity: power(100.0) }]).unwrap();
        topology.add_node("sink", [ResourcePort { id: "in".into(), direction: PortDirection::Input, capacity: power(100.0) }]).unwrap();
        topology.add_link(ResourceLink { id: "line".into(), from_node: "source".into(), from_port: "out".into(), to_node: "sink".into(), to_port: "in".into(), capacity: power(100.0), loss_fraction: 0.0 }).unwrap();
        topology
    }

    fn feasible_set(candidate_ids: &[&str]) -> DeterministicFeasibleSet {
        let topology = topology();
        let mut capacities = CapacitySchedule::default();
        for (id, subject) in [
            ("source-cap", CapacitySubject::Port { node_id: "source".into(), port_id: "out".into() }),
            ("link-cap", CapacitySubject::Link { link_id: "line".into() }),
            ("sink-cap", CapacitySubject::Port { node_id: "sink".into(), port_id: "in".into() }),
        ] {
            capacities.add_window(&topology, CapacityWindow { id: id.into(), subject, valid_from: t0(), valid_until: t0() + Duration::hours(1), capacity: power(100.0), semantics: CapacitySemantics::Concurrent }).unwrap();
        }
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile.set_numeric(QualityMetric::TemperatureCelsius, 40.0).unwrap();
        let mut quality = QualityEvidenceSet::default();
        quality.insert(&topology, QualityEvidenceWindow { id: "quality".into(), subject: QualitySubject::Link { link_id: "line".into() }, valid_from: t0(), valid_until: t0() + Duration::hours(1), profile, evidence_class: QualityEvidenceClass::Observed, evidence_ref: "sensor:quality".into() }).unwrap();
        let mut requirement = ResourceQualityRequirement::new(power(1.0).key);
        requirement.require_numeric(NumericQualityConstraint { metric: QualityMetric::TemperatureCelsius, minimum: Some(20.0), maximum: Some(60.0) }).unwrap();
        let qpolicy = QualityQualificationPolicy { allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(), identity_policy: EvidenceIdentityPolicy::Any, minimum_evidence: 1, multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple };
        let candidates = candidate_ids.iter().enumerate().map(|(index, id)| PlannedAllocation { id: (*id).into(), link_id: "line".into(), valid_from: t0(), valid_until: t0() + Duration::hours(1), sent: power(20.0 + index as f64) }).collect();
        enumerate_feasible_set(&topology, &capacities, &AllocationBook::default(), &quality, &requirement, &qpolicy, candidates, 16).unwrap()
    }

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new("energy", "energy.total.joule.v1", "si.joule.v1", ObjectiveStatistic::CandidateTotal).unwrap()
    }

    fn resilience() -> ObjectiveMetric {
        ObjectiveMetric::new("resilience", "resilience.fraction.v1", "ratio.fraction.v1", ObjectiveStatistic::CandidateFraction).unwrap()
    }

    fn policy() -> ObjectiveQualificationPolicy {
        ObjectiveQualificationPolicy { allowed_classes: [ObjectiveEvidenceClass::Observed].into_iter().collect(), identity_policy: ObjectiveEvidenceIdentityPolicy::Any, minimum_evidence: 1, multiple_evidence_rule: ObjectiveMultipleEvidenceRule::RejectMultiple }
    }

    fn claim(id: &str, candidate: &str, metric: ObjectiveMetric, value: f64) -> CandidateObjectiveEvidence {
        CandidateObjectiveEvidence { id: id.into(), candidate_id: candidate.into(), metric, value, evidence_class: ObjectiveEvidenceClass::Observed, evidence_ref: format!("source:{id}"), scope: ObjectiveEvidenceScope::CandidateAggregate }
    }

    fn ranking(values: [(&str, f64, f64); 2]) -> ResourceParetoRanking {
        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible_set(&[values[0].0, values[1].0]));
        for (candidate, energy_value, resilience_value) in values {
            evidence.insert(claim(&format!("{candidate}-energy"), candidate, energy(), energy_value)).unwrap();
            evidence.insert(claim(&format!("{candidate}-resilience"), candidate, resilience(), resilience_value)).unwrap();
        }
        let objectives = vec![
            ResourceParetoObjective::new(energy(), ObjectiveDirection::Minimize),
            ResourceParetoObjective::new(resilience(), ObjectiveDirection::Maximize),
        ];
        let requests = vec![
            ObjectiveQualificationRequest { metric: energy(), policy: policy() },
            ObjectiveQualificationRequest { metric: resilience(), policy: policy() },
        ];
        rank_feasible_resources(&evidence, &objectives, &requests).unwrap()
    }

    #[test]
    fn canonical_baseline_uses_closed_canonical_feasible_order() {
        let set = feasible_set(&["z", "a"]);
        let ValidatedResourcePlannerDecision::Recommend(validated) = validated_canonical_feasible(&set).unwrap() else {
            panic!("baseline should recommend");
        };
        assert_eq!(validated.selector_id(), CANONICAL_FEASIBLE_SELECTOR_ID);
        assert_eq!(validated.candidate_id(), "a");
    }

    #[test]
    fn canonical_baseline_abstains_when_feasible_set_is_empty() {
        let set = feasible_set(&[]);
        let ValidatedResourcePlannerDecision::Abstain(validated) = validated_canonical_feasible(&set).unwrap() else {
            panic!("empty baseline should abstain");
        };
        assert_eq!(validated.selector_id(), CANONICAL_FEASIBLE_SELECTOR_ID);
    }

    #[test]
    fn unique_pareto_frontier_recommends_without_tie_break() {
        let ranking = ranking([("a", 1.0, 0.9), ("b", 2.0, 0.8)]);
        let ValidatedResourcePlannerDecision::Recommend(validated) = validated_pareto_unique(&ranking).unwrap() else {
            panic!("unique frontier should recommend");
        };
        assert_eq!(validated.candidate_id(), "a");
        assert_eq!(validated.selector_id(), PARETO_UNIQUE_SELECTOR_ID);
    }

    #[test]
    fn pareto_unique_abstains_on_real_tradeoff() {
        let ranking = ranking([("cheap", 1.0, 0.5), ("resilient", 5.0, 0.95)]);
        let ValidatedResourcePlannerDecision::Abstain(validated) = validated_pareto_unique(&ranking).unwrap() else {
            panic!("tradeoff frontier should abstain");
        };
        assert_eq!(validated.selector_id(), PARETO_UNIQUE_SELECTOR_ID);
    }

    #[test]
    fn pareto_canonical_tie_break_is_explicit_and_closed_set_validated() {
        let ranking = ranking([("z", 1.0, 0.5), ("a", 5.0, 0.95)]);
        let ValidatedResourcePlannerDecision::Recommend(validated) = validated_pareto_canonical(&ranking).unwrap() else {
            panic!("canonical frontier selector should recommend");
        };
        assert_eq!(validated.candidate_id(), "a");
        assert_eq!(validated.selector_id(), PARETO_CANONICAL_SELECTOR_ID);
    }
}
