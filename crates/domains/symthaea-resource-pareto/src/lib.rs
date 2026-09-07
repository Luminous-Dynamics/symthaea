// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-qualified Pareto ranking over exact resource objective metrics.
//!
//! This crate remains a thin adapter over `symthaea-operations-research`. It adds
//! no optimizer. Its job is to prove that every feasible candidate has a complete
//! qualified vector under one exact metric schema before mapping those values into
//! the generic deterministic Pareto sorter.

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use symthaea_operations_research::{
    rank_pareto, ObjectiveDirection, ParetoCandidate, ParetoError, ParetoObjective,
    ParetoRanking,
};
use symthaea_resource_objective_evidence::ResourceObjectiveEvidenceSet;
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricSchema, ObjectiveMetricSchemaError,
};
use symthaea_resource_objective_qualification::{
    qualify_objective_vector, ObjectiveQualificationRequest, ObjectiveVectorError,
    QualifiedObjectiveVector,
};
use thiserror::Error;

/// Exact metric semantics plus the optimization direction applied to that metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceParetoObjective {
    pub metric: ObjectiveMetric,
    pub direction: ObjectiveDirection,
}

impl ResourceParetoObjective {
    pub fn new(metric: ObjectiveMetric, direction: ObjectiveDirection) -> Self {
        Self { metric, direction }
    }
}

/// Positive non-serializable ranking bound to one exact feasible/evidence/metric
/// context. It exposes fronts and ranks only; there is no winner API.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceParetoRanking {
    evidence_set: ResourceObjectiveEvidenceSet,
    metric_schema: ObjectiveMetricSchema,
    objectives: Vec<ResourceParetoObjective>,
    objective_requests: Vec<ObjectiveQualificationRequest>,
    qualified_vectors: BTreeMap<String, QualifiedObjectiveVector>,
    ranking: ParetoRanking,
}

impl ResourceParetoRanking {
    pub fn evidence_set(&self) -> &ResourceObjectiveEvidenceSet {
        &self.evidence_set
    }

    pub fn metric_schema(&self) -> &ObjectiveMetricSchema {
        &self.metric_schema
    }

    pub fn objectives(&self) -> &[ResourceParetoObjective] {
        &self.objectives
    }

    pub fn objective_requests(&self) -> &[ObjectiveQualificationRequest] {
        &self.objective_requests
    }

    pub fn qualified_vector(&self, candidate_id: &str) -> Option<&QualifiedObjectiveVector> {
        self.qualified_vectors.get(candidate_id)
    }

    pub fn ranking(&self) -> &ParetoRanking {
        &self.ranking
    }

    pub fn frontier_ids(&self) -> Vec<&str> {
        self.ranking
            .first_front()
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect()
    }

    pub fn rank_of(&self, candidate_id: &str) -> Option<usize> {
        self.ranking.rank_of(candidate_id)
    }
}

/// Qualify every member of the complete feasible subset under one exact metric
/// schema, then delegate non-dominated sorting to the existing Pareto primitive.
///
/// A feasible candidate with missing/conflicting/incompatible objective evidence
/// fails the entire ranking; it is never silently dropped from the comparison set.
pub fn rank_feasible_resources(
    evidence_set: &ResourceObjectiveEvidenceSet,
    objectives: &[ResourceParetoObjective],
    requests: &[ObjectiveQualificationRequest],
) -> Result<ResourceParetoRanking, ResourceParetoError> {
    let metric_schema = validate_schema(objectives, requests)?;

    let pareto_objectives: Vec<ParetoObjective> = objectives
        .iter()
        .map(|objective| {
            ParetoObjective::new(
                objective.metric.objective_name.clone(),
                objective.direction,
            )
        })
        .collect();

    let mut vectors = BTreeMap::new();
    let mut pareto_candidates = Vec::new();

    for feasible in evidence_set.feasible_set().feasible() {
        let candidate_id = feasible.candidate_id();
        let vector = qualify_objective_vector(evidence_set, candidate_id, requests).map_err(
            |source| ResourceParetoError::ObjectiveVectorRejected {
                candidate_id: candidate_id.to_owned(),
                source,
            },
        )?;

        if vector.metric_schema() != &metric_schema {
            return Err(ResourceParetoError::QualifiedVectorSchemaMismatch {
                candidate_id: candidate_id.to_owned(),
            });
        }

        let mut values = Vec::with_capacity(objectives.len());
        for objective in objectives {
            let qualified = vector
                .get(&objective.metric.objective_name)
                .ok_or_else(|| ResourceParetoError::QualifiedObjectiveMissing {
                    candidate_id: candidate_id.to_owned(),
                    objective_name: objective.metric.objective_name.clone(),
                })?;
            if qualified.metric() != &objective.metric {
                return Err(ResourceParetoError::QualifiedMetricMismatch {
                    candidate_id: candidate_id.to_owned(),
                    objective_name: objective.metric.objective_name.clone(),
                });
            }
            values.push(qualified.value());
        }

        pareto_candidates.push(ParetoCandidate::new(candidate_id, values));
        vectors.insert(candidate_id.to_owned(), vector);
    }

    let ranking = rank_pareto(&pareto_objectives, &pareto_candidates)
        .map_err(ResourceParetoError::Pareto)?;

    let mut objective_requests = requests.to_vec();
    objective_requests.sort_by(|left, right| {
        left.metric
            .objective_name
            .cmp(&right.metric.objective_name)
    });

    Ok(ResourceParetoRanking {
        evidence_set: evidence_set.clone(),
        metric_schema,
        objectives: objectives.to_vec(),
        objective_requests,
        qualified_vectors: vectors,
        ranking,
    })
}

fn validate_schema(
    objectives: &[ResourceParetoObjective],
    requests: &[ObjectiveQualificationRequest],
) -> Result<ObjectiveMetricSchema, ResourceParetoError> {
    if objectives.is_empty() {
        return Err(ResourceParetoError::NoObjectives);
    }
    if requests.is_empty() {
        return Err(ResourceParetoError::NoObjectiveRequests);
    }

    let metric_schema = ObjectiveMetricSchema::new(
        objectives.iter().map(|objective| objective.metric.clone()),
    )
    .map_err(ResourceParetoError::InvalidMetricSchema)?;

    let request_schema = ObjectiveMetricSchema::new(
        requests.iter().map(|request| request.metric.clone()),
    )
    .map_err(ResourceParetoError::InvalidRequestMetricSchema)?;

    if metric_schema != request_schema {
        return Err(ResourceParetoError::MetricSchemaMismatch);
    }
    Ok(metric_schema)
}

#[derive(Debug, Error)]
pub enum ResourceParetoError {
    #[error("at least one resource Pareto objective is required")]
    NoObjectives,
    #[error("at least one objective qualification request is required")]
    NoObjectiveRequests,
    #[error("invalid Pareto metric schema: {0}")]
    InvalidMetricSchema(ObjectiveMetricSchemaError),
    #[error("invalid qualification-request metric schema: {0}")]
    InvalidRequestMetricSchema(ObjectiveMetricSchemaError),
    #[error("Pareto metric schema and qualification-request metric schema differ")]
    MetricSchemaMismatch,
    #[error("objective vector rejected for feasible candidate {candidate_id}: {source}")]
    ObjectiveVectorRejected {
        candidate_id: String,
        #[source]
        source: ObjectiveVectorError,
    },
    #[error("qualified vector for {candidate_id} does not retain the exact Pareto metric schema")]
    QualifiedVectorSchemaMismatch { candidate_id: String },
    #[error("qualified vector for {candidate_id} is missing objective {objective_name}")]
    QualifiedObjectiveMissing {
        candidate_id: String,
        objective_name: String,
    },
    #[error("qualified metric for {candidate_id}/{objective_name} differs from Pareto schema")]
    QualifiedMetricMismatch {
        candidate_id: String,
        objective_name: String,
    },
    #[error("Pareto ranking failed: {0}")]
    Pareto(#[source] ParetoError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, TimeZone, Utc};
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
    };
    use symthaea_resource_objective_metric::ObjectiveStatistic;
    use symthaea_resource_objective_qualification::{
        ObjectiveEvidenceIdentityPolicy, ObjectiveMultipleEvidenceRule,
        ObjectiveQualificationPolicy,
    };
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

    fn feasible_set() -> symthaea_resource_feasible_set::DeterministicFeasibleSet {
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
        enumerate_feasible_set(
            &topology,
            &capacities,
            &AllocationBook::default(),
            &quality,
            &requirement,
            &qpolicy,
            vec![
                PlannedAllocation { id: "cheap".into(), link_id: "line".into(), valid_from: t0(), valid_until: t0() + Duration::hours(1), sent: power(40.0) },
                PlannedAllocation { id: "resilient".into(), link_id: "line".into(), valid_from: t0(), valid_until: t0() + Duration::hours(1), sent: power(50.0) },
            ],
            8,
        )
        .unwrap()
    }

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new("energy", "energy.total.joule.v1", "si.joule.v1", ObjectiveStatistic::CandidateTotal).unwrap()
    }

    fn resilience() -> ObjectiveMetric {
        ObjectiveMetric::new("resilience", "resilience.fraction.v1", "ratio.fraction.v1", ObjectiveStatistic::CandidateFraction).unwrap()
    }

    fn energy_kwh() -> ObjectiveMetric {
        ObjectiveMetric::new("energy", "energy.total.kwh.v1", "energy.kilowatt_hour.v1", ObjectiveStatistic::CandidateTotal).unwrap()
    }

    fn policy() -> ObjectiveQualificationPolicy {
        ObjectiveQualificationPolicy { allowed_classes: [ObjectiveEvidenceClass::Observed].into_iter().collect(), identity_policy: ObjectiveEvidenceIdentityPolicy::Any, minimum_evidence: 1, multiple_evidence_rule: ObjectiveMultipleEvidenceRule::RejectMultiple }
    }

    fn claim(id: &str, candidate: &str, metric: ObjectiveMetric, value: f64) -> CandidateObjectiveEvidence {
        CandidateObjectiveEvidence { id: id.into(), candidate_id: candidate.into(), metric, value, evidence_class: ObjectiveEvidenceClass::Observed, evidence_ref: format!("source:{id}"), scope: ObjectiveEvidenceScope::CandidateAggregate }
    }

    fn evidence() -> ResourceObjectiveEvidenceSet {
        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible_set());
        evidence.insert(claim("cheap-energy", "cheap", energy(), 2.0)).unwrap();
        evidence.insert(claim("cheap-resilience", "cheap", resilience(), 0.6)).unwrap();
        evidence.insert(claim("resilient-energy", "resilient", energy(), 5.0)).unwrap();
        evidence.insert(claim("resilient-resilience", "resilient", resilience(), 0.95)).unwrap();
        evidence
    }

    fn objectives() -> Vec<ResourceParetoObjective> {
        vec![
            ResourceParetoObjective::new(energy(), ObjectiveDirection::Minimize),
            ResourceParetoObjective::new(resilience(), ObjectiveDirection::Maximize),
        ]
    }

    fn requests() -> Vec<ObjectiveQualificationRequest> {
        vec![
            ObjectiveQualificationRequest { metric: resilience(), policy: policy() },
            ObjectiveQualificationRequest { metric: energy(), policy: policy() },
        ]
    }

    #[test]
    fn exact_semantic_tradeoff_reuses_generic_pareto_frontier() {
        let ranking = rank_feasible_resources(&evidence(), &objectives(), &requests()).unwrap();
        assert_eq!(ranking.frontier_ids(), vec!["cheap", "resilient"]);
        assert_eq!(ranking.metric_schema().get("energy"), Some(&energy()));
    }

    #[test]
    fn request_order_does_not_change_metric_mapping() {
        let a = rank_feasible_resources(&evidence(), &objectives(), &requests()).unwrap();
        let mut reversed = requests();
        reversed.reverse();
        let b = rank_feasible_resources(&evidence(), &objectives(), &reversed).unwrap();
        assert_eq!(a.ranking(), b.ranking());
    }

    #[test]
    fn same_name_different_unit_schema_is_rejected_before_ranking() {
        let mut requests = requests();
        requests.retain(|request| request.metric.objective_name != "energy");
        requests.push(ObjectiveQualificationRequest { metric: energy_kwh(), policy: policy() });
        assert!(matches!(
            rank_feasible_resources(&evidence(), &objectives(), &requests),
            Err(ResourceParetoError::MetricSchemaMismatch)
        ));
    }

    #[test]
    fn missing_evidence_for_one_feasible_candidate_fails_complete_ranking() {
        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible_set());
        evidence.insert(claim("cheap-energy", "cheap", energy(), 2.0)).unwrap();
        evidence.insert(claim("cheap-resilience", "cheap", resilience(), 0.6)).unwrap();
        evidence.insert(claim("resilient-energy", "resilient", energy(), 5.0)).unwrap();
        assert!(matches!(
            rank_feasible_resources(&evidence, &objectives(), &requests()),
            Err(ResourceParetoError::ObjectiveVectorRejected { candidate_id, .. }) if candidate_id == "resilient"
        ));
    }

    #[test]
    fn duplicate_metric_schema_fails_closed() {
        let objectives = vec![
            ResourceParetoObjective::new(energy(), ObjectiveDirection::Minimize),
            ResourceParetoObjective::new(energy(), ObjectiveDirection::Maximize),
        ];
        assert!(matches!(
            rank_feasible_resources(&evidence(), &objectives, &requests()),
            Err(ResourceParetoError::InvalidMetricSchema(_))
        ));
    }
}
