// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Planning-context-bound evidence for exact resource objective metrics.
//!
//! Raw ranking values become eligible for later qualification only when they are
//! bound to one exact feasible candidate and one exact `ObjectiveMetric` semantic
//! identity. A shared human-facing name is not enough: different units/statistics
//! remain different evidence domains.
//!
//! Provenance classes are descriptive, not rank-ordered. This crate does not
//! authenticate evidence, resolve conflicts, convert units, aggregate values, or
//! rank candidates.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use symthaea_resource_feasible_set::DeterministicFeasibleSet;
use symthaea_resource_objective_metric::{ObjectiveMetric, ObjectiveMetricError};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectiveEvidenceClass {
    Declared,
    Estimated,
    Observed,
    Attested,
}

/// V1 evidence describes one aggregate scalar for the complete candidate plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveEvidenceScope {
    CandidateAggregate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateObjectiveEvidence {
    pub id: String,
    pub candidate_id: String,
    pub metric: ObjectiveMetric,
    pub value: f64,
    pub evidence_class: ObjectiveEvidenceClass,
    pub evidence_ref: String,
    pub scope: ObjectiveEvidenceScope,
}

/// Objective evidence tied to one exact deterministic feasible-set context.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceObjectiveEvidenceSet {
    feasible_set: DeterministicFeasibleSet,
    claims: BTreeMap<String, CandidateObjectiveEvidence>,
}

impl ResourceObjectiveEvidenceSet {
    pub fn new(feasible_set: DeterministicFeasibleSet) -> Self {
        Self {
            feasible_set,
            claims: BTreeMap::new(),
        }
    }

    pub fn feasible_set(&self) -> &DeterministicFeasibleSet {
        &self.feasible_set
    }

    pub fn insert(
        &mut self,
        claim: CandidateObjectiveEvidence,
    ) -> Result<(), ObjectiveEvidenceError> {
        validate_claim(&self.feasible_set, &claim)?;
        if self.claims.contains_key(&claim.id) {
            return Err(ObjectiveEvidenceError::DuplicateEvidenceId(claim.id));
        }
        self.claims.insert(claim.id.clone(), claim);
        Ok(())
    }

    pub fn get(&self, evidence_id: &str) -> Option<&CandidateObjectiveEvidence> {
        self.claims.get(evidence_id)
    }

    pub fn claims(&self) -> impl Iterator<Item = &CandidateObjectiveEvidence> {
        self.claims.values()
    }

    /// Exact metric match only. Same objective name with another unit/statistic does
    /// not enter this iterator.
    pub fn claims_for_metric<'a>(
        &'a self,
        candidate_id: &'a str,
        metric: &'a ObjectiveMetric,
    ) -> impl Iterator<Item = &'a CandidateObjectiveEvidence> + 'a {
        self.claims.values().filter(move |claim| {
            claim.candidate_id == candidate_id && claim.metric == *metric
        })
    }

    /// Every distinct exact metric claimed for one candidate, in canonical order.
    pub fn metrics_for(&self, candidate_id: &str) -> BTreeSet<ObjectiveMetric> {
        self.claims
            .values()
            .filter(|claim| claim.candidate_id == candidate_id)
            .map(|claim| claim.metric.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
}

fn validate_claim(
    feasible_set: &DeterministicFeasibleSet,
    claim: &CandidateObjectiveEvidence,
) -> Result<(), ObjectiveEvidenceError> {
    if claim.id.trim().is_empty() {
        return Err(ObjectiveEvidenceError::EmptyEvidenceId);
    }
    if claim.candidate_id.trim().is_empty() {
        return Err(ObjectiveEvidenceError::EmptyCandidateId);
    }
    if claim.evidence_ref.trim().is_empty() {
        return Err(ObjectiveEvidenceError::EmptyEvidenceRef);
    }
    claim
        .metric
        .validate()
        .map_err(ObjectiveEvidenceError::InvalidMetric)?;
    if !claim.value.is_finite() {
        return Err(ObjectiveEvidenceError::NonFiniteObjectiveValue {
            metric_id: claim.metric.metric_id.clone(),
            value: claim.value,
        });
    }

    if feasible_set.feasible_candidate(&claim.candidate_id).is_some() {
        return Ok(());
    }
    if feasible_set.rejection(&claim.candidate_id).is_some() {
        return Err(ObjectiveEvidenceError::RejectedCandidate(
            claim.candidate_id.clone(),
        ));
    }
    Err(ObjectiveEvidenceError::CandidateOutsideUniverse(
        claim.candidate_id.clone(),
    ))
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObjectiveEvidenceError {
    #[error("objective evidence id must not be empty")]
    EmptyEvidenceId,
    #[error("objective evidence candidate id must not be empty")]
    EmptyCandidateId,
    #[error("objective evidence reference must not be empty")]
    EmptyEvidenceRef,
    #[error("invalid objective metric: {0}")]
    InvalidMetric(ObjectiveMetricError),
    #[error("objective metric {metric_id} has non-finite value {value}")]
    NonFiniteObjectiveValue { metric_id: String, value: f64 },
    #[error("candidate {0} exists in the universe but failed feasibility")]
    RejectedCandidate(String),
    #[error("candidate {0} is outside the retained feasible-set universe")]
    CandidateOutsideUniverse(String),
    #[error("duplicate objective evidence id {0}")]
    DuplicateEvidenceId(String),
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
        topology
            .add_node(
                "source",
                [ResourcePort {
                    id: "out".into(),
                    direction: PortDirection::Output,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [ResourcePort {
                    id: "in".into(),
                    direction: PortDirection::Input,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();
        topology
    }

    fn capacities(topology: &ResourceTopology) -> CapacitySchedule {
        let mut schedule = CapacitySchedule::default();
        for (id, subject) in [
            (
                "source-cap",
                CapacitySubject::Port {
                    node_id: "source".into(),
                    port_id: "out".into(),
                },
            ),
            (
                "link-cap",
                CapacitySubject::Link { link_id: "line".into() },
            ),
            (
                "sink-cap",
                CapacitySubject::Port {
                    node_id: "sink".into(),
                    port_id: "in".into(),
                },
            ),
        ] {
            schedule
                .add_window(
                    topology,
                    CapacityWindow {
                        id: id.into(),
                        subject,
                        valid_from: t0(),
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }
        schedule
    }

    fn feasible_set() -> DeterministicFeasibleSet {
        let topology = topology();
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, 40.0)
            .unwrap();
        let mut quality = QualityEvidenceSet::default();
        quality
            .insert(
                &topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link { link_id: "line".into() },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        let mut requirement = ResourceQualityRequirement::new(power(1.0).key);
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(20.0),
                maximum: Some(60.0),
            })
            .unwrap();
        let policy = QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &quality,
            &requirement,
            &policy,
            vec![PlannedAllocation {
                id: "candidate".into(),
                link_id: "line".into(),
                valid_from: t0(),
                valid_until: t0() + Duration::hours(1),
                sent: power(40.0),
            }],
            8,
        )
        .unwrap()
    }

    fn energy_joule() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn energy_kwh() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.kwh.v1",
            "energy.kilowatt_hour.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn claim(id: &str, metric: ObjectiveMetric, value: f64) -> CandidateObjectiveEvidence {
        CandidateObjectiveEvidence {
            id: id.into(),
            candidate_id: "candidate".into(),
            metric,
            value,
            evidence_class: ObjectiveEvidenceClass::Observed,
            evidence_ref: format!("source:{id}"),
            scope: ObjectiveEvidenceScope::CandidateAggregate,
        }
    }

    #[test]
    fn exact_metric_is_retained_with_claim() {
        let mut set = ResourceObjectiveEvidenceSet::new(feasible_set());
        set.insert(claim("e1", energy_joule(), 12.0)).unwrap();
        assert_eq!(set.get("e1").unwrap().metric, energy_joule());
    }

    #[test]
    fn same_name_different_unit_remains_a_distinct_evidence_domain() {
        let mut set = ResourceObjectiveEvidenceSet::new(feasible_set());
        set.insert(claim("joule", energy_joule(), 12.0)).unwrap();
        set.insert(claim("kwh", energy_kwh(), 12.0)).unwrap();
        assert_eq!(
            set.claims_for_metric("candidate", &energy_joule()).count(),
            1
        );
        assert_eq!(set.claims_for_metric("candidate", &energy_kwh()).count(), 1);
        assert_eq!(set.metrics_for("candidate").len(), 2);
    }

    #[test]
    fn duplicate_evidence_identity_fails_closed() {
        let mut set = ResourceObjectiveEvidenceSet::new(feasible_set());
        set.insert(claim("same", energy_joule(), 1.0)).unwrap();
        assert!(matches!(
            set.insert(claim("same", energy_joule(), 1.0)),
            Err(ObjectiveEvidenceError::DuplicateEvidenceId(id)) if id == "same"
        ));
    }

    #[test]
    fn non_finite_value_fails_closed() {
        let mut set = ResourceObjectiveEvidenceSet::new(feasible_set());
        assert!(matches!(
            set.insert(claim("bad", energy_joule(), f64::NAN)),
            Err(ObjectiveEvidenceError::NonFiniteObjectiveValue { .. })
        ));
    }

    #[test]
    fn outside_candidate_cannot_acquire_objective_evidence() {
        let mut set = ResourceObjectiveEvidenceSet::new(feasible_set());
        let mut outside = claim("outside", energy_joule(), 1.0);
        outside.candidate_id = "manufactured".into();
        assert!(matches!(
            set.insert(outside),
            Err(ObjectiveEvidenceError::CandidateOutsideUniverse(id)) if id == "manufactured"
        ));
    }
}
