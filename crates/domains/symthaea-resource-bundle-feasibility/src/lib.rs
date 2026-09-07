// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint feasibility for bounded multi-allocation resource bundles.
//!
//! `symthaea-resource-feasible-set` proves individual alternatives against one
//! common base book. This crate answers the distinct multi-allocation question:
//!
//! > Can this exact bounded set of allocations coexist in one prospective book,
//! > while every member also satisfies its own full-interval quality theorem?
//!
//! Positive bundle feasibility is planning evidence only. It is not ranking,
//! recommendation, a Mycelix lease, operating authority, or physical execution.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use symthaea_resource_allocation::{AllocationBook, AllocationError, PlannedAllocation};
use symthaea_resource_capacity::CapacitySchedule;
use symthaea_resource_quality::ResourceQualityRequirement;
use symthaea_resource_quality_evidence::{QualityEvidenceSet, QualitySubject};
use symthaea_resource_quality_interval::{
    qualify_quality_interval, QualifiedQualityInterval, QualityIntervalError,
};
use symthaea_resource_quality_qualification::QualityQualificationPolicy;
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// One allocation plus the exact quality theorem that applies to that member.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedBundleMember {
    pub allocation: PlannedAllocation,
    pub quality_requirement: ResourceQualityRequirement,
    pub quality_policy: QualityQualificationPolicy,
}

/// Bounded multi-allocation proposal. Member ordering is not semantically trusted;
/// assessment canonicalizes by allocation ID before constructing the proof.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedResourceBundle {
    pub id: String,
    pub members: Vec<PlannedBundleMember>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedBundleMemberQuality {
    allocation_id: String,
    quality: QualifiedQualityInterval,
}

impl QualifiedBundleMemberQuality {
    pub fn allocation_id(&self) -> &str {
        &self.allocation_id
    }

    pub fn quality(&self) -> &QualifiedQualityInterval {
        &self.quality
    }
}

/// Positive non-serializable joint-feasibility witness.
#[derive(Debug, Clone, PartialEq)]
pub struct FeasibleResourceBundle {
    bundle_id: String,
    members: Vec<PlannedBundleMember>,
    prospective_book: AllocationBook,
    quality: Vec<QualifiedBundleMemberQuality>,
    max_members: usize,
}

impl FeasibleResourceBundle {
    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }

    /// Exact canonical member set used by the theorem.
    pub fn members(&self) -> &[PlannedBundleMember] {
        &self.members
    }

    /// Exact base-plus-bundle allocation state after all joint quantity admissions.
    pub fn prospective_book(&self) -> &AllocationBook {
        &self.prospective_book
    }

    pub fn qualified_quality(&self) -> &[QualifiedBundleMemberQuality] {
        &self.quality
    }

    pub fn max_members(&self) -> usize {
        self.max_members
    }

    pub fn member_quality(&self, allocation_id: &str) -> Option<&QualifiedQualityInterval> {
        self.quality
            .binary_search_by(|entry| entry.allocation_id.as_str().cmp(allocation_id))
            .ok()
            .map(|index| self.quality[index].quality())
    }
}

/// Deterministic first failing quantity admission under canonical member-ID order.
///
/// `allocation_id` is a diagnostic proof-construction encounter point, not a claim
/// that this member alone caused the collective conflict.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleQuantityFailure {
    pub allocation_id: String,
    pub error: AllocationError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleQualityFailure {
    pub allocation_id: String,
    pub error: QualityIntervalError,
}

/// Negative bundle result. Quantity and quality are evaluated independently so a
/// shared-capacity conflict cannot hide separate evidence/quality failures.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceBundleRejection {
    pub bundle_id: String,
    pub quantity_failure: Option<BundleQuantityFailure>,
    pub quality_failures: Vec<BundleQualityFailure>,
}

impl ResourceBundleRejection {
    pub fn quantity_failed(&self) -> bool {
        self.quantity_failure.is_some()
    }

    pub fn quality_failed(&self) -> bool {
        !self.quality_failures.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceBundleAssessment {
    Feasible(FeasibleResourceBundle),
    Rejected(ResourceBundleRejection),
}

/// Assess one bounded bundle against one exact base planning state.
///
/// Quality is evaluated for every member independently across its entire interval.
/// Quantity is then admitted into one evolving prospective book in canonical
/// allocation-ID order. Positive output therefore proves the members coexist in
/// one shared capacity state rather than merely being individually feasible.
pub fn assess_resource_bundle(
    topology: &ResourceTopology,
    capacities: &CapacitySchedule,
    base_book: &AllocationBook,
    evidence: &QualityEvidenceSet,
    mut bundle: PlannedResourceBundle,
    max_members: usize,
) -> Result<ResourceBundleAssessment, ResourceBundleError> {
    validate_bundle_structure(&bundle, max_members)?;
    bundle
        .members
        .sort_by(|left, right| left.allocation.id.cmp(&right.allocation.id));

    let mut quality_successes = Vec::with_capacity(bundle.members.len());
    let mut quality_failures = Vec::new();
    for member in &bundle.members {
        let subject = QualitySubject::Link {
            link_id: member.allocation.link_id.clone(),
        };
        match qualify_quality_interval(
            topology,
            evidence,
            &subject,
            &member.quality_requirement,
            &member.quality_policy,
            member.allocation.valid_from,
            member.allocation.valid_until,
        ) {
            Ok(quality) => quality_successes.push(QualifiedBundleMemberQuality {
                allocation_id: member.allocation.id.clone(),
                quality,
            }),
            Err(error) => quality_failures.push(BundleQualityFailure {
                allocation_id: member.allocation.id.clone(),
                error,
            }),
        }
    }

    let mut prospective_book = base_book.clone();
    let mut quantity_failure = None;
    for member in &bundle.members {
        if let Err(error) = prospective_book.admit(
            topology,
            capacities,
            member.allocation.clone(),
        ) {
            quantity_failure = Some(BundleQuantityFailure {
                allocation_id: member.allocation.id.clone(),
                error,
            });
            break;
        }
    }

    if quantity_failure.is_none() && quality_failures.is_empty() {
        debug_assert_eq!(quality_successes.len(), bundle.members.len());
        return Ok(ResourceBundleAssessment::Feasible(FeasibleResourceBundle {
            bundle_id: bundle.id,
            members: bundle.members,
            prospective_book,
            quality: quality_successes,
            max_members,
        }));
    }

    Ok(ResourceBundleAssessment::Rejected(ResourceBundleRejection {
        bundle_id: bundle.id,
        quantity_failure,
        quality_failures,
    }))
}

fn validate_bundle_structure(
    bundle: &PlannedResourceBundle,
    max_members: usize,
) -> Result<(), ResourceBundleError> {
    if bundle.id.trim().is_empty() {
        return Err(ResourceBundleError::EmptyBundleId);
    }
    if max_members == 0 {
        return Err(ResourceBundleError::ZeroMemberBound);
    }
    if bundle.members.is_empty() {
        return Err(ResourceBundleError::EmptyBundle);
    }
    if bundle.members.len() > max_members {
        return Err(ResourceBundleError::MemberBoundExceeded {
            actual: bundle.members.len(),
            maximum: max_members,
        });
    }

    let mut ids = BTreeSet::new();
    for member in &bundle.members {
        if member.allocation.id.trim().is_empty() {
            return Err(ResourceBundleError::EmptyMemberId);
        }
        if !ids.insert(member.allocation.id.clone()) {
            return Err(ResourceBundleError::DuplicateMemberId(
                member.allocation.id.clone(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourceBundleError {
    #[error("bundle id must not be empty")]
    EmptyBundleId,
    #[error("bundle member bound must be greater than zero")]
    ZeroMemberBound,
    #[error("bundle must contain at least one member")]
    EmptyBundle,
    #[error("bundle size {actual} exceeds declared member bound {maximum}")]
    MemberBoundExceeded { actual: usize, maximum: usize },
    #[error("bundle member allocation id must not be empty")]
    EmptyMemberId,
    #[error("duplicate bundle member allocation id {0}")]
    DuplicateMemberId(String),
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
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }
        schedule
    }

    fn evidence(topology: &ResourceTopology, temp_c: f64) -> QualityEvidenceSet {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        let mut evidence = QualityEvidenceSet::default();
        evidence
            .insert(
                topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link {
                        link_id: "line".into(),
                    },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        evidence
    }

    fn policy() -> QualityQualificationPolicy {
        QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        }
    }

    fn requirement(max_temp: f64) -> ResourceQualityRequirement {
        let mut requirement = ResourceQualityRequirement::new(power(1.0).key);
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(20.0),
                maximum: Some(max_temp),
            })
            .unwrap();
        requirement
    }

    fn member(id: &str, value: f64, max_temp: f64) -> PlannedBundleMember {
        PlannedBundleMember {
            allocation: PlannedAllocation {
                id: id.into(),
                link_id: "line".into(),
                valid_from: t0(),
                valid_until: t0() + Duration::hours(1),
                sent: power(value),
            },
            quality_requirement: requirement(max_temp),
            quality_policy: policy(),
        }
    }

    fn bundle(members: Vec<PlannedBundleMember>) -> PlannedResourceBundle {
        PlannedResourceBundle {
            id: "bundle".into(),
            members,
        }
    }

    #[test]
    fn jointly_feasible_members_share_one_prospective_book() {
        let topology = topology();
        let result = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(vec![member("b", 50.0, 60.0), member("a", 40.0, 60.0)]),
            8,
        )
        .unwrap();

        let ResourceBundleAssessment::Feasible(feasible) = result else {
            panic!("bundle should be feasible");
        };
        let ids: Vec<&str> = feasible
            .members()
            .iter()
            .map(|member| member.allocation.id.as_str())
            .collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(feasible.prospective_book().len(), 2);
        assert_eq!(feasible.qualified_quality().len(), 2);
    }

    #[test]
    fn individually_feasible_members_can_fail_joint_shared_capacity() {
        let topology = topology();
        let result = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(vec![member("a", 60.0, 60.0), member("b", 60.0, 60.0)]),
            8,
        )
        .unwrap();

        let ResourceBundleAssessment::Rejected(rejected) = result else {
            panic!("bundle should be rejected");
        };
        assert!(rejected.quantity_failed());
        assert!(!rejected.quality_failed());
        assert_eq!(
            rejected.quantity_failure.as_ref().unwrap().allocation_id,
            "b"
        );
    }

    #[test]
    fn per_member_quality_requirements_are_independent() {
        let topology = topology();
        let result = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(vec![member("a", 20.0, 60.0), member("b", 20.0, 35.0)]),
            8,
        )
        .unwrap();

        let ResourceBundleAssessment::Rejected(rejected) = result else {
            panic!("bundle should be rejected");
        };
        assert!(!rejected.quantity_failed());
        assert!(rejected.quality_failed());
        assert_eq!(rejected.quality_failures.len(), 1);
        assert_eq!(rejected.quality_failures[0].allocation_id, "b");
    }

    #[test]
    fn quantity_and_quality_failures_are_both_retained() {
        let topology = topology();
        let result = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(vec![member("a", 60.0, 60.0), member("b", 60.0, 35.0)]),
            8,
        )
        .unwrap();

        let ResourceBundleAssessment::Rejected(rejected) = result else {
            panic!("bundle should be rejected");
        };
        assert!(rejected.quantity_failed());
        assert!(rejected.quality_failed());
        assert_eq!(rejected.quality_failures[0].allocation_id, "b");
    }

    #[test]
    fn canonical_order_makes_quantity_failure_deterministic() {
        let topology = topology();
        let assess = |members| {
            assess_resource_bundle(
                &topology,
                &capacities(&topology),
                &AllocationBook::default(),
                &evidence(&topology, 40.0),
                bundle(members),
                8,
            )
            .unwrap()
        };
        let first = assess(vec![member("b", 60.0, 60.0), member("a", 60.0, 60.0)]);
        let second = assess(vec![member("a", 60.0, 60.0), member("b", 60.0, 60.0)]);
        let ResourceBundleAssessment::Rejected(first) = first else {
            panic!("expected rejection");
        };
        let ResourceBundleAssessment::Rejected(second) = second else {
            panic!("expected rejection");
        };
        assert_eq!(
            first.quantity_failure.as_ref().unwrap().allocation_id,
            second.quantity_failure.as_ref().unwrap().allocation_id
        );
        assert_eq!(first.quantity_failure.as_ref().unwrap().allocation_id, "b");
    }

    #[test]
    fn existing_base_load_participates_in_joint_feasibility() {
        let topology = topology();
        let capacities = capacities(&topology);
        let mut base = AllocationBook::default();
        base.admit(
            &topology,
            &capacities,
            PlannedAllocation {
                id: "existing".into(),
                link_id: "line".into(),
                valid_from: t0(),
                valid_until: t0() + Duration::hours(1),
                sent: power(40.0),
            },
        )
        .unwrap();
        let result = assess_resource_bundle(
            &topology,
            &capacities,
            &base,
            &evidence(&topology, 40.0),
            bundle(vec![member("a", 30.0, 60.0), member("b", 30.0, 60.0)]),
            8,
        )
        .unwrap();
        let ResourceBundleAssessment::Feasible(feasible) = result else {
            panic!("40 + 30 + 30 should fit exactly");
        };
        assert_eq!(feasible.prospective_book().len(), 3);
        assert_eq!(base.len(), 1);
    }

    #[test]
    fn member_bound_and_duplicate_identity_fail_before_proof_construction() {
        let topology = topology();
        let common = (
            &topology,
            capacities(&topology),
            evidence(&topology, 40.0),
        );
        let too_many = assess_resource_bundle(
            common.0,
            &common.1,
            &AllocationBook::default(),
            &common.2,
            bundle(vec![member("a", 10.0, 60.0), member("b", 10.0, 60.0)]),
            1,
        );
        assert!(matches!(
            too_many,
            Err(ResourceBundleError::MemberBoundExceeded {
                actual: 2,
                maximum: 1
            })
        ));

        let duplicate = assess_resource_bundle(
            common.0,
            &common.1,
            &AllocationBook::default(),
            &common.2,
            bundle(vec![member("same", 10.0, 60.0), member("same", 20.0, 60.0)]),
            8,
        );
        assert!(matches!(
            duplicate,
            Err(ResourceBundleError::DuplicateMemberId(id)) if id == "same"
        ));
    }

    #[test]
    fn empty_bundle_and_zero_bound_fail_closed() {
        let topology = topology();
        let empty = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(Vec::new()),
            8,
        );
        assert_eq!(empty, Err(ResourceBundleError::EmptyBundle));

        let zero = assess_resource_bundle(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            bundle(vec![member("a", 10.0, 60.0)]),
            0,
        );
        assert_eq!(zero, Err(ResourceBundleError::ZeroMemberBound));
    }
}
