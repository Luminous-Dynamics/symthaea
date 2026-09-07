// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed qualification of exact resource objective metrics.
//!
//! Raw objective claims are rankable only after an explicit evidence policy
//! qualifies one exact candidate + exact `ObjectiveMetric`. Same-name claims with
//! different units/statistics never enter the same qualification domain.
//!
//! There is intentionally no averaging, implicit unit conversion, latest-wins,
//! provenance precedence, Pareto ranking, planner selection, lease, or execution
//! authority.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use symthaea_resource_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidenceClass, ResourceObjectiveEvidenceSet,
};
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricError, ObjectiveMetricSchema, ObjectiveMetricSchemaError,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveEvidenceIdentityPolicy {
    Any,
    EvidenceIdAllowlist(BTreeSet<String>),
    EvidenceRefAllowlist(BTreeSet<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveMultipleEvidenceRule {
    RejectMultiple,
    RequireExactValueAgreement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveQualificationPolicy {
    pub allowed_classes: BTreeSet<ObjectiveEvidenceClass>,
    pub identity_policy: ObjectiveEvidenceIdentityPolicy,
    pub minimum_evidence: usize,
    pub multiple_evidence_rule: ObjectiveMultipleEvidenceRule,
}

impl ObjectiveQualificationPolicy {
    pub fn validate(&self) -> Result<(), ObjectiveQualificationError> {
        if self.allowed_classes.is_empty() {
            return Err(ObjectiveQualificationError::NoAllowedEvidenceClasses);
        }
        if self.minimum_evidence == 0 {
            return Err(ObjectiveQualificationError::ZeroMinimumEvidence);
        }
        match &self.identity_policy {
            ObjectiveEvidenceIdentityPolicy::Any => {}
            ObjectiveEvidenceIdentityPolicy::EvidenceIdAllowlist(values)
            | ObjectiveEvidenceIdentityPolicy::EvidenceRefAllowlist(values) => {
                if values.is_empty() {
                    return Err(ObjectiveQualificationError::EmptyIdentityAllowlist);
                }
                if values.iter().any(|value| value.trim().is_empty()) {
                    return Err(ObjectiveQualificationError::BlankIdentityAllowlistEntry);
                }
            }
        }
        if self.multiple_evidence_rule == ObjectiveMultipleEvidenceRule::RejectMultiple
            && self.minimum_evidence > 1
        {
            return Err(ObjectiveQualificationError::ImpossibleSingleEvidencePolicy {
                minimum_evidence: self.minimum_evidence,
            });
        }
        Ok(())
    }

    fn admits(&self, evidence: &CandidateObjectiveEvidence) -> bool {
        if !self.allowed_classes.contains(&evidence.evidence_class) {
            return false;
        }
        match &self.identity_policy {
            ObjectiveEvidenceIdentityPolicy::Any => true,
            ObjectiveEvidenceIdentityPolicy::EvidenceIdAllowlist(values) => {
                values.contains(&evidence.id)
            }
            ObjectiveEvidenceIdentityPolicy::EvidenceRefAllowlist(values) => {
                values.contains(&evidence.evidence_ref)
            }
        }
    }
}

/// Positive non-serializable scalar retaining its exact semantic metric identity.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedObjectiveValue {
    candidate_id: String,
    metric: ObjectiveMetric,
    value: f64,
    evidence_ids: Vec<String>,
    evidence_refs: Vec<String>,
    evidence_classes: Vec<ObjectiveEvidenceClass>,
    multiple_evidence_rule: ObjectiveMultipleEvidenceRule,
}

impl QualifiedObjectiveValue {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn metric(&self) -> &ObjectiveMetric {
        &self.metric
    }

    pub fn objective_name(&self) -> &str {
        &self.metric.objective_name
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn evidence_ids(&self) -> &[String] {
        &self.evidence_ids
    }

    pub fn evidence_refs(&self) -> &[String] {
        &self.evidence_refs
    }

    pub fn evidence_classes(&self) -> &[ObjectiveEvidenceClass] {
        &self.evidence_classes
    }

    pub fn multiple_evidence_rule(&self) -> ObjectiveMultipleEvidenceRule {
        self.multiple_evidence_rule
    }
}

pub fn qualify_objective_value(
    evidence_set: &ResourceObjectiveEvidenceSet,
    candidate_id: &str,
    metric: &ObjectiveMetric,
    policy: &ObjectiveQualificationPolicy,
) -> Result<QualifiedObjectiveValue, ObjectiveQualificationError> {
    policy.validate()?;
    if candidate_id.trim().is_empty() {
        return Err(ObjectiveQualificationError::EmptyCandidateId);
    }
    metric
        .validate()
        .map_err(ObjectiveQualificationError::InvalidMetric)?;

    if evidence_set
        .feasible_set()
        .feasible_candidate(candidate_id)
        .is_none()
    {
        if evidence_set.feasible_set().rejection(candidate_id).is_some() {
            return Err(ObjectiveQualificationError::RejectedCandidate(
                candidate_id.to_owned(),
            ));
        }
        return Err(ObjectiveQualificationError::CandidateOutsideUniverse(
            candidate_id.to_owned(),
        ));
    }

    let mut admissible: Vec<&CandidateObjectiveEvidence> = evidence_set
        .claims_for_metric(candidate_id, metric)
        .filter(|claim| policy.admits(claim))
        .collect();
    admissible.sort_by(|left, right| left.id.cmp(&right.id));

    if admissible.is_empty() {
        return Err(ObjectiveQualificationError::NoAdmissibleEvidence {
            candidate_id: candidate_id.to_owned(),
            metric_id: metric.metric_id.clone(),
        });
    }
    if admissible.len() < policy.minimum_evidence {
        return Err(ObjectiveQualificationError::InsufficientEvidence {
            candidate_id: candidate_id.to_owned(),
            metric_id: metric.metric_id.clone(),
            minimum: policy.minimum_evidence,
            actual: admissible.len(),
        });
    }

    match policy.multiple_evidence_rule {
        ObjectiveMultipleEvidenceRule::RejectMultiple if admissible.len() != 1 => {
            return Err(ObjectiveQualificationError::MultipleEvidenceRejected {
                candidate_id: candidate_id.to_owned(),
                metric_id: metric.metric_id.clone(),
                count: admissible.len(),
            });
        }
        ObjectiveMultipleEvidenceRule::RequireExactValueAgreement => {
            let expected = admissible[0].value;
            if admissible.iter().skip(1).any(|claim| claim.value != expected) {
                return Err(ObjectiveQualificationError::ConflictingObjectiveValues {
                    candidate_id: candidate_id.to_owned(),
                    metric_id: metric.metric_id.clone(),
                });
            }
        }
        ObjectiveMultipleEvidenceRule::RejectMultiple => {}
    }

    Ok(QualifiedObjectiveValue {
        candidate_id: candidate_id.to_owned(),
        metric: metric.clone(),
        value: admissible[0].value,
        evidence_ids: admissible.iter().map(|claim| claim.id.clone()).collect(),
        evidence_refs: admissible
            .iter()
            .map(|claim| claim.evidence_ref.clone())
            .collect(),
        evidence_classes: admissible
            .iter()
            .map(|claim| claim.evidence_class)
            .collect(),
        multiple_evidence_rule: policy.multiple_evidence_rule,
    })
}

/// One required exact metric and its decision-local evidence policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveQualificationRequest {
    pub metric: ObjectiveMetric,
    pub policy: ObjectiveQualificationPolicy,
}

/// Complete positive vector retaining the exact metric schema used to qualify it.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedObjectiveVector {
    candidate_id: String,
    metric_schema: ObjectiveMetricSchema,
    values: BTreeMap<String, QualifiedObjectiveValue>,
}

impl QualifiedObjectiveVector {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn metric_schema(&self) -> &ObjectiveMetricSchema {
        &self.metric_schema
    }

    pub fn values(&self) -> &BTreeMap<String, QualifiedObjectiveValue> {
        &self.values
    }

    pub fn get(&self, objective_name: &str) -> Option<&QualifiedObjectiveValue> {
        self.values.get(objective_name)
    }
}

pub fn qualify_objective_vector(
    evidence_set: &ResourceObjectiveEvidenceSet,
    candidate_id: &str,
    requests: &[ObjectiveQualificationRequest],
) -> Result<QualifiedObjectiveVector, ObjectiveVectorError> {
    if candidate_id.trim().is_empty() {
        return Err(ObjectiveVectorError::EmptyCandidateId);
    }
    if requests.is_empty() {
        return Err(ObjectiveVectorError::NoObjectives);
    }

    let metric_schema = ObjectiveMetricSchema::new(
        requests.iter().map(|request| request.metric.clone()),
    )
    .map_err(ObjectiveVectorError::InvalidMetricSchema)?;

    let mut ordered: Vec<&ObjectiveQualificationRequest> = requests.iter().collect();
    ordered.sort_by(|left, right| {
        left.metric
            .objective_name
            .cmp(&right.metric.objective_name)
    });

    let mut values = BTreeMap::new();
    for request in ordered {
        let qualified = qualify_objective_value(
            evidence_set,
            candidate_id,
            &request.metric,
            &request.policy,
        )
        .map_err(|source| ObjectiveVectorError::ObjectiveRejected {
            candidate_id: candidate_id.to_owned(),
            metric_id: request.metric.metric_id.clone(),
            source,
        })?;
        values.insert(request.metric.objective_name.clone(), qualified);
    }

    Ok(QualifiedObjectiveVector {
        candidate_id: candidate_id.to_owned(),
        metric_schema,
        values,
    })
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObjectiveQualificationError {
    #[error("at least one objective evidence class must be allowed")]
    NoAllowedEvidenceClasses,
    #[error("minimum objective evidence must be greater than zero")]
    ZeroMinimumEvidence,
    #[error("objective evidence identity allowlist must not be empty")]
    EmptyIdentityAllowlist,
    #[error("objective evidence identity allowlist contains a blank entry")]
    BlankIdentityAllowlistEntry,
    #[error("RejectMultiple cannot require minimum evidence {minimum_evidence}")]
    ImpossibleSingleEvidencePolicy { minimum_evidence: usize },
    #[error("candidate id must not be empty")]
    EmptyCandidateId,
    #[error("invalid objective metric: {0}")]
    InvalidMetric(ObjectiveMetricError),
    #[error("candidate {0} exists in the universe but failed feasibility")]
    RejectedCandidate(String),
    #[error("candidate {0} is outside the retained feasible-set universe")]
    CandidateOutsideUniverse(String),
    #[error("no admissible evidence for {candidate_id}/{metric_id}")]
    NoAdmissibleEvidence { candidate_id: String, metric_id: String },
    #[error("{candidate_id}/{metric_id} has {actual} admissible claims; minimum is {minimum}")]
    InsufficientEvidence {
        candidate_id: String,
        metric_id: String,
        minimum: usize,
        actual: usize,
    },
    #[error("{candidate_id}/{metric_id} has {count} admissible claims under RejectMultiple")]
    MultipleEvidenceRejected {
        candidate_id: String,
        metric_id: String,
        count: usize,
    },
    #[error("admissible objective values conflict for {candidate_id}/{metric_id}")]
    ConflictingObjectiveValues { candidate_id: String, metric_id: String },
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObjectiveVectorError {
    #[error("candidate id must not be empty")]
    EmptyCandidateId,
    #[error("at least one objective must be requested")]
    NoObjectives,
    #[error("invalid objective metric schema: {0}")]
    InvalidMetricSchema(ObjectiveMetricSchemaError),
    #[error("objective metric {metric_id} rejected for candidate {candidate_id}: {source}")]
    ObjectiveRejected {
        candidate_id: String,
        metric_id: String,
        source: ObjectiveQualificationError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::{enumerate_feasible_set, DeterministicFeasibleSet};
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_objective_evidence::{
        CandidateObjectiveEvidence, ObjectiveEvidenceScope,
    };
    use symthaea_resource_objective_metric::{ObjectiveMetric, ObjectiveStatistic};
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

    fn feasible_set() -> DeterministicFeasibleSet {
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
        let quality_policy = QualityQualificationPolicy { allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(), identity_policy: EvidenceIdentityPolicy::Any, minimum_evidence: 1, multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple };
        enumerate_feasible_set(&topology, &capacities, &AllocationBook::default(), &quality, &requirement, &quality_policy, vec![PlannedAllocation { id: "candidate".into(), link_id: "line".into(), valid_from: t0(), valid_until: t0() + Duration::hours(1), sent: power(40.0) }], 8).unwrap()
    }

    fn energy_joule() -> ObjectiveMetric {
        ObjectiveMetric::new("energy", "energy.total.joule.v1", "si.joule.v1", ObjectiveStatistic::CandidateTotal).unwrap()
    }

    fn energy_kwh() -> ObjectiveMetric {
        ObjectiveMetric::new("energy", "energy.total.kwh.v1", "energy.kilowatt_hour.v1", ObjectiveStatistic::CandidateTotal).unwrap()
    }

    fn evidence_set() -> ResourceObjectiveEvidenceSet {
        ResourceObjectiveEvidenceSet::new(feasible_set())
    }

    fn claim(id: &str, metric: ObjectiveMetric, value: f64, class: ObjectiveEvidenceClass) -> CandidateObjectiveEvidence {
        CandidateObjectiveEvidence { id: id.into(), candidate_id: "candidate".into(), metric, value, evidence_class: class, evidence_ref: format!("source:{id}"), scope: ObjectiveEvidenceScope::CandidateAggregate }
    }

    fn policy(classes: impl IntoIterator<Item = ObjectiveEvidenceClass>, rule: ObjectiveMultipleEvidenceRule) -> ObjectiveQualificationPolicy {
        ObjectiveQualificationPolicy { allowed_classes: classes.into_iter().collect(), identity_policy: ObjectiveEvidenceIdentityPolicy::Any, minimum_evidence: 1, multiple_evidence_rule: rule }
    }

    #[test]
    fn exact_metric_claim_qualifies() {
        let mut evidence = evidence_set();
        evidence.insert(claim("estimate", energy_joule(), 12.0, ObjectiveEvidenceClass::Estimated)).unwrap();
        let qualified = qualify_objective_value(&evidence, "candidate", &energy_joule(), &policy([ObjectiveEvidenceClass::Estimated], ObjectiveMultipleEvidenceRule::RejectMultiple)).unwrap();
        assert_eq!(qualified.metric(), &energy_joule());
        assert_eq!(qualified.value(), 12.0);
    }

    #[test]
    fn same_name_different_unit_cannot_satisfy_metric_request() {
        let mut evidence = evidence_set();
        evidence.insert(claim("kwh", energy_kwh(), 12.0, ObjectiveEvidenceClass::Observed)).unwrap();
        assert!(matches!(
            qualify_objective_value(&evidence, "candidate", &energy_joule(), &policy([ObjectiveEvidenceClass::Observed], ObjectiveMultipleEvidenceRule::RejectMultiple)),
            Err(ObjectiveQualificationError::NoAdmissibleEvidence { .. })
        ));
    }

    #[test]
    fn conflicting_exact_metric_values_are_not_averaged() {
        let mut evidence = evidence_set();
        evidence.insert(claim("a", energy_joule(), 10.0, ObjectiveEvidenceClass::Observed)).unwrap();
        evidence.insert(claim("b", energy_joule(), 11.0, ObjectiveEvidenceClass::Attested)).unwrap();
        assert!(matches!(
            qualify_objective_value(&evidence, "candidate", &energy_joule(), &policy([ObjectiveEvidenceClass::Observed, ObjectiveEvidenceClass::Attested], ObjectiveMultipleEvidenceRule::RequireExactValueAgreement)),
            Err(ObjectiveQualificationError::ConflictingObjectiveValues { .. })
        ));
    }

    #[test]
    fn complete_vector_retains_exact_metric_schema() {
        let latency = ObjectiveMetric::new("latency", "latency.p95.second.v1", "si.second.v1", ObjectiveStatistic::PercentileBasisPoints(9500)).unwrap();
        let mut evidence = evidence_set();
        evidence.insert(claim("energy", energy_joule(), 10.0, ObjectiveEvidenceClass::Observed)).unwrap();
        evidence.insert(claim("latency", latency.clone(), 0.02, ObjectiveEvidenceClass::Observed)).unwrap();
        let requests = vec![
            ObjectiveQualificationRequest { metric: latency.clone(), policy: policy([ObjectiveEvidenceClass::Observed], ObjectiveMultipleEvidenceRule::RejectMultiple) },
            ObjectiveQualificationRequest { metric: energy_joule(), policy: policy([ObjectiveEvidenceClass::Observed], ObjectiveMultipleEvidenceRule::RejectMultiple) },
        ];
        let vector = qualify_objective_vector(&evidence, "candidate", &requests).unwrap();
        assert_eq!(vector.metric_schema().get("energy"), Some(&energy_joule()));
        assert_eq!(vector.metric_schema().get("latency"), Some(&latency));
    }

    #[test]
    fn duplicate_objective_name_with_different_metric_fails_vector_schema() {
        let requests = vec![
            ObjectiveQualificationRequest { metric: energy_joule(), policy: policy([ObjectiveEvidenceClass::Observed], ObjectiveMultipleEvidenceRule::RejectMultiple) },
            ObjectiveQualificationRequest { metric: energy_kwh(), policy: policy([ObjectiveEvidenceClass::Observed], ObjectiveMultipleEvidenceRule::RejectMultiple) },
        ];
        assert!(matches!(
            qualify_objective_vector(&evidence_set(), "candidate", &requests),
            Err(ObjectiveVectorError::InvalidMetricSchema(ObjectiveMetricSchemaError::DuplicateObjectiveName(name))) if name == "energy"
        ));
    }
}
