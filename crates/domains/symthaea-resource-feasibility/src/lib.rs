// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic composition of quantity and full-interval quality feasibility.
//!
//! This crate is intentionally not an optimizer. It answers one local question:
//!
//! > If this exact candidate allocation were added to this exact current planning
//! > book, would it satisfy both temporal quantity conservation and the declared
//! > quality-evidence theorem for its exact topology link?
//!
//! Quantity and quality are evaluated independently so rejection diagnostics can
//! preserve both causes when both fail. A positive witness retains the complete
//! prospective allocation book, preventing a result proven against one load state
//! from floating onto another.

#![deny(unsafe_code)]

use symthaea_resource_allocation::{AdmittedAllocation, AllocationBook, AllocationError, PlannedAllocation};
use symthaea_resource_capacity::CapacitySchedule;
use symthaea_resource_quality::ResourceQualityRequirement;
use symthaea_resource_quality_evidence::{QualityEvidenceSet, QualitySubject};
use symthaea_resource_quality_interval::{
    qualify_quality_interval, QualifiedQualityInterval, QualityIntervalError,
};
use symthaea_resource_quality_qualification::QualityQualificationPolicy;
use symthaea_resource_topology::ResourceTopology;

/// Positive non-serializable witness that one candidate is feasible against one
/// exact prospective planning state.
#[derive(Debug, Clone, PartialEq)]
pub struct FeasibleResourceCandidate {
    candidate_id: String,
    prospective_book: AllocationBook,
    quality: QualifiedQualityInterval,
}

impl FeasibleResourceCandidate {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    /// Complete allocation state used by the positive quantity theorem, including
    /// the newly admitted candidate.
    pub fn prospective_book(&self) -> &AllocationBook {
        &self.prospective_book
    }

    pub fn admitted_allocation(&self) -> &AdmittedAllocation {
        // Construction is private and occurs only after successful admission.
        self.prospective_book
            .get(&self.candidate_id)
            .expect("feasible candidate must retain admitted allocation")
    }

    pub fn quality(&self) -> &QualifiedQualityInterval {
        &self.quality
    }
}

/// Typed rejection retaining quantity and quality causes independently.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceCandidateRejection {
    pub candidate_id: String,
    pub quantity_error: Option<AllocationError>,
    pub quality_error: Option<QualityIntervalError>,
}

impl ResourceCandidateRejection {
    pub fn quantity_failed(&self) -> bool {
        self.quantity_error.is_some()
    }

    pub fn quality_failed(&self) -> bool {
        self.quality_error.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceCandidateAssessment {
    Feasible(FeasibleResourceCandidate),
    Rejected(ResourceCandidateRejection),
}

/// Assess one exact candidate without mutating the caller's allocation book.
///
/// Quality is evaluated on the candidate's exact topology link across the entire
/// allocation interval. Quantity and quality are both attempted, so a candidate
/// that fails both returns both diagnostics rather than whichever happened first.
pub fn assess_resource_candidate(
    topology: &ResourceTopology,
    capacities: &CapacitySchedule,
    base_book: &AllocationBook,
    evidence: &QualityEvidenceSet,
    quality_requirement: &ResourceQualityRequirement,
    quality_policy: &QualityQualificationPolicy,
    candidate: PlannedAllocation,
) -> ResourceCandidateAssessment {
    let candidate_id = candidate.id.clone();

    let mut prospective_book = base_book.clone();
    let quantity_error = prospective_book
        .admit(topology, capacities, candidate.clone())
        .err();

    let quality_subject = QualitySubject::Link {
        link_id: candidate.link_id.clone(),
    };
    let quality_result = qualify_quality_interval(
        topology,
        evidence,
        &quality_subject,
        quality_requirement,
        quality_policy,
        candidate.valid_from,
        candidate.valid_until,
    );

    match (quantity_error, quality_result) {
        (None, Ok(quality)) => ResourceCandidateAssessment::Feasible(FeasibleResourceCandidate {
            candidate_id,
            prospective_book,
            quality,
        }),
        (quantity_error, quality_result) => ResourceCandidateAssessment::Rejected(
            ResourceCandidateRejection {
                candidate_id,
                quantity_error,
                quality_error: quality_result.err(),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use symthaea_resource_capacity::{
        CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
    };
    use symthaea_resource_quality_evidence::{
        QualityEvidenceClass, QualityEvidenceWindow,
    };
    use symthaea_resource_quality_qualification::{
        EvidenceIdentityPolicy, MultipleEvidenceRule,
    };
    use symthaea_resource_topology::ResourceLink;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, value: f64) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity: power(value),
        }
    }

    fn topology() -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology
            .add_node("source", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        topology
            .add_node("sink", [port("in", PortDirection::Input, 100.0)])
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
                CapacitySubject::Link {
                    link_id: "line".into(),
                },
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
                        valid_until: t0() + Duration::hours(2),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }
        schedule
    }

    fn quality_requirement() -> ResourceQualityRequirement {
        let mut requirement = ResourceQualityRequirement::new(power(1.0).key);
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(30.0),
                maximum: Some(60.0),
            })
            .unwrap();
        requirement
    }

    fn quality_policy() -> QualityQualificationPolicy {
        QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        }
    }

    fn quality_evidence(
        topology: &ResourceTopology,
        temp_c: f64,
        start_min: i64,
        end_min: i64,
    ) -> QualityEvidenceSet {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            topology,
            QualityEvidenceWindow {
                id: format!("q-{start_min}-{end_min}"),
                subject: QualitySubject::Link {
                    link_id: "line".into(),
                },
                valid_from: t0() + Duration::minutes(start_min),
                valid_until: t0() + Duration::minutes(end_min),
                profile,
                evidence_class: QualityEvidenceClass::Observed,
                evidence_ref: format!("sensor:{start_min}:{end_min}"),
            },
        )
        .unwrap();
        set
    }

    fn allocation(id: &str, start_min: i64, end_min: i64, value: f64) -> PlannedAllocation {
        PlannedAllocation {
            id: id.into(),
            link_id: "line".into(),
            valid_from: t0() + Duration::minutes(start_min),
            valid_until: t0() + Duration::minutes(end_min),
            sent: power(value),
        }
    }

    #[test]
    fn candidate_is_feasible_only_when_quantity_and_full_interval_quality_pass() {
        let topology = topology();
        let capacities = capacities(&topology);
        let book = AllocationBook::default();
        let evidence = quality_evidence(&topology, 40.0, 0, 60);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 80.0),
        );

        let ResourceCandidateAssessment::Feasible(feasible) = result else {
            panic!("candidate should be feasible");
        };
        assert_eq!(feasible.candidate_id(), "candidate");
        assert_eq!(feasible.prospective_book().len(), 1);
        assert_eq!(feasible.admitted_allocation().allocation().sent.value, 80.0);
        assert_eq!(feasible.quality().segments().len(), 1);
    }

    #[test]
    fn quantity_failure_is_typed_without_hiding_quality_success() {
        let topology = topology();
        let capacities = capacities(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("existing", 0, 60, 70.0))
            .unwrap();
        let evidence = quality_evidence(&topology, 40.0, 0, 60);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 40.0),
        );
        let ResourceCandidateAssessment::Rejected(rejected) = result else {
            panic!("candidate should be rejected");
        };
        assert!(rejected.quantity_failed());
        assert!(!rejected.quality_failed());
    }

    #[test]
    fn quality_failure_is_typed_without_hiding_quantity_success() {
        let topology = topology();
        let capacities = capacities(&topology);
        let book = AllocationBook::default();
        let evidence = quality_evidence(&topology, 90.0, 0, 60);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 40.0),
        );
        let ResourceCandidateAssessment::Rejected(rejected) = result else {
            panic!("candidate should be rejected");
        };
        assert!(!rejected.quantity_failed());
        assert!(rejected.quality_failed());
    }

    #[test]
    fn simultaneous_quantity_and_quality_failures_are_both_retained() {
        let topology = topology();
        let capacities = capacities(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("existing", 0, 60, 90.0))
            .unwrap();
        let evidence = quality_evidence(&topology, 90.0, 0, 60);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 20.0),
        );
        let ResourceCandidateAssessment::Rejected(rejected) = result else {
            panic!("candidate should be rejected");
        };
        assert!(rejected.quantity_failed());
        assert!(rejected.quality_failed());
    }

    #[test]
    fn assessment_never_mutates_callers_base_book() {
        let topology = topology();
        let capacities = capacities(&topology);
        let book = AllocationBook::default();
        let evidence = quality_evidence(&topology, 40.0, 0, 60);

        let _ = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 40.0),
        );
        assert!(book.is_empty());
    }

    #[test]
    fn positive_witness_retains_exact_prior_load_state_plus_candidate() {
        let topology = topology();
        let capacities = capacities(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("existing", 0, 60, 30.0))
            .unwrap();
        let evidence = quality_evidence(&topology, 40.0, 0, 60);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 60.0),
        );
        let ResourceCandidateAssessment::Feasible(feasible) = result else {
            panic!("candidate should be feasible");
        };
        assert_eq!(feasible.prospective_book().len(), 2);
        assert!(feasible.prospective_book().get("existing").is_some());
        assert!(feasible.prospective_book().get("candidate").is_some());
    }

    #[test]
    fn quality_gap_late_in_candidate_interval_rejects_even_when_quantity_passes() {
        let topology = topology();
        let capacities = capacities(&topology);
        let book = AllocationBook::default();
        let evidence = quality_evidence(&topology, 40.0, 0, 30);

        let result = assess_resource_candidate(
            &topology,
            &capacities,
            &book,
            &evidence,
            &quality_requirement(),
            &quality_policy(),
            allocation("candidate", 0, 60, 40.0),
        );
        let ResourceCandidateAssessment::Rejected(rejected) = result else {
            panic!("candidate should be rejected");
        };
        assert!(!rejected.quantity_failed());
        assert!(rejected.quality_failed());
    }
}
