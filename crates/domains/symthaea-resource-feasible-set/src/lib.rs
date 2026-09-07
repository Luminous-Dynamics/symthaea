// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic bounded feasible-set enumeration for resource planning.
//!
//! This crate evaluates a discovery-supplied, bounded universe of candidate
//! allocations through the exact `symthaea-resource-feasibility` theorem.
//! It does not rank, optimize, select, or execute candidates.
//!
//! Every candidate is assessed independently against the same exact base
//! `AllocationBook`. Therefore the returned feasible entries are alternatives
//! relative to that base state; the set does **not** claim that all feasible
//! entries may be admitted simultaneously. A future multi-allocation/bundle
//! theorem must revalidate any joint selection.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
use symthaea_resource_capacity::CapacitySchedule;
use symthaea_resource_feasibility::{
    assess_resource_candidate, FeasibleResourceCandidate, ResourceCandidateAssessment,
    ResourceCandidateRejection,
};
use symthaea_resource_quality::ResourceQualityRequirement;
use symthaea_resource_quality_evidence::QualityEvidenceSet;
use symthaea_resource_quality_qualification::QualityQualificationPolicy;
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// Semantic contract of v1 enumeration.
///
/// Each feasible candidate has been proven against the same retained base book.
/// Multiple feasible candidates are not thereby proven jointly feasible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibleSetSemantics {
    IndependentAlternativesAgainstSameBase,
}

/// Complete deterministic result for one bounded candidate universe.
///
/// The exact canonicalized candidate universe and exact base allocation book are
/// retained so downstream planners cannot detach a feasible subset from the
/// planning state and discovery universe that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct DeterministicFeasibleSet {
    base_book: AllocationBook,
    candidate_universe: Vec<PlannedAllocation>,
    feasible: Vec<FeasibleResourceCandidate>,
    rejected: Vec<ResourceCandidateRejection>,
    max_candidates: usize,
}

impl DeterministicFeasibleSet {
    pub fn semantics(&self) -> FeasibleSetSemantics {
        FeasibleSetSemantics::IndependentAlternativesAgainstSameBase
    }

    pub fn base_book(&self) -> &AllocationBook {
        &self.base_book
    }

    /// Exact candidate universe in canonical candidate-ID order.
    pub fn candidate_universe(&self) -> &[PlannedAllocation] {
        &self.candidate_universe
    }

    /// Feasible alternatives in canonical candidate-ID order.
    pub fn feasible(&self) -> &[FeasibleResourceCandidate] {
        &self.feasible
    }

    /// Rejected alternatives in canonical candidate-ID order.
    pub fn rejected(&self) -> &[ResourceCandidateRejection] {
        &self.rejected
    }

    pub fn max_candidates(&self) -> usize {
        self.max_candidates
    }

    pub fn len(&self) -> usize {
        self.candidate_universe.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidate_universe.is_empty()
    }

    pub fn feasible_candidate(&self, candidate_id: &str) -> Option<&FeasibleResourceCandidate> {
        self.feasible
            .binary_search_by(|candidate| candidate.candidate_id().cmp(candidate_id))
            .ok()
            .map(|index| &self.feasible[index])
    }

    pub fn rejection(&self, candidate_id: &str) -> Option<&ResourceCandidateRejection> {
        self.rejected
            .binary_search_by(|candidate| candidate.candidate_id.as_str().cmp(candidate_id))
            .ok()
            .map(|index| &self.rejected[index])
    }
}

/// Enumerate a bounded universe without any objective function or winner selection.
///
/// Every candidate is assessed against the exact same `base_book`. The input order
/// is intentionally erased: candidate IDs define canonical evaluation/output order.
/// Duplicate IDs fail the whole enumeration because they would make downstream
/// identity-based recommendation and diagnostics ambiguous.
pub fn enumerate_feasible_set(
    topology: &ResourceTopology,
    capacities: &CapacitySchedule,
    base_book: &AllocationBook,
    evidence: &QualityEvidenceSet,
    quality_requirement: &ResourceQualityRequirement,
    quality_policy: &QualityQualificationPolicy,
    mut candidates: Vec<PlannedAllocation>,
    max_candidates: usize,
) -> Result<DeterministicFeasibleSet, FeasibleSetError> {
    if max_candidates == 0 {
        return Err(FeasibleSetError::ZeroCandidateBound);
    }
    if candidates.len() > max_candidates {
        return Err(FeasibleSetError::CandidateBoundExceeded {
            actual: candidates.len(),
            maximum: max_candidates,
        });
    }

    let mut identities = BTreeSet::new();
    for candidate in &candidates {
        if !identities.insert(candidate.id.clone()) {
            return Err(FeasibleSetError::DuplicateCandidateId(candidate.id.clone()));
        }
    }

    candidates.sort_by(|left, right| left.id.cmp(&right.id));

    let mut feasible = Vec::new();
    let mut rejected = Vec::new();
    for candidate in candidates.iter().cloned() {
        match assess_resource_candidate(
            topology,
            capacities,
            base_book,
            evidence,
            quality_requirement,
            quality_policy,
            candidate,
        ) {
            ResourceCandidateAssessment::Feasible(candidate) => feasible.push(candidate),
            ResourceCandidateAssessment::Rejected(candidate) => rejected.push(candidate),
        }
    }

    debug_assert!(feasible
        .windows(2)
        .all(|window| window[0].candidate_id() < window[1].candidate_id()));
    debug_assert!(rejected
        .windows(2)
        .all(|window| window[0].candidate_id < window[1].candidate_id));

    Ok(DeterministicFeasibleSet {
        base_book: base_book.clone(),
        candidate_universe: candidates,
        feasible,
        rejected,
        max_candidates,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FeasibleSetError {
    #[error("candidate bound must be greater than zero")]
    ZeroCandidateBound,
    #[error("candidate universe size {actual} exceeds declared bound {maximum}")]
    CandidateBoundExceeded { actual: usize, maximum: usize },
    #[error("duplicate candidate id {0}")]
    DuplicateCandidateId(String),
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
        QualityEvidenceClass, QualityEvidenceWindow, QualitySubject,
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

    fn evidence(topology: &ResourceTopology, temp_c: f64) -> QualityEvidenceSet {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            topology,
            QualityEvidenceWindow {
                id: "quality".into(),
                subject: QualitySubject::Link {
                    link_id: "line".into(),
                },
                valid_from: t0(),
                valid_until: t0() + Duration::hours(2),
                profile,
                evidence_class: QualityEvidenceClass::Observed,
                evidence_ref: "sensor:quality".into(),
            },
        )
        .unwrap();
        set
    }

    fn allocation(id: &str, value: f64) -> PlannedAllocation {
        PlannedAllocation {
            id: id.into(),
            link_id: "line".into(),
            valid_from: t0(),
            valid_until: t0() + Duration::hours(1),
            sent: power(value),
        }
    }

    fn enumerate(
        base_book: &AllocationBook,
        candidates: Vec<PlannedAllocation>,
        temp_c: f64,
    ) -> Result<DeterministicFeasibleSet, FeasibleSetError> {
        let topology = topology();
        enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            base_book,
            &evidence(&topology, temp_c),
            &quality_requirement(),
            &quality_policy(),
            candidates,
            16,
        )
    }

    #[test]
    fn enumeration_is_canonical_independent_of_input_order() {
        let set = enumerate(
            &AllocationBook::default(),
            vec![allocation("z", 30.0), allocation("a", 20.0), allocation("m", 10.0)],
            40.0,
        )
        .unwrap();

        let universe_ids: Vec<&str> = set
            .candidate_universe()
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        let feasible_ids: Vec<&str> = set
            .feasible()
            .iter()
            .map(|candidate| candidate.candidate_id())
            .collect();
        assert_eq!(universe_ids, vec!["a", "m", "z"]);
        assert_eq!(feasible_ids, vec!["a", "m", "z"]);
        assert!(set.rejected().is_empty());
    }

    #[test]
    fn rejected_candidates_retain_typed_diagnostics() {
        let set = enumerate(
            &AllocationBook::default(),
            vec![allocation("too-large", 120.0), allocation("fits", 50.0)],
            40.0,
        )
        .unwrap();
        assert_eq!(set.feasible().len(), 1);
        assert_eq!(set.feasible()[0].candidate_id(), "fits");
        let rejection = set.rejection("too-large").unwrap();
        assert!(rejection.quantity_failed());
        assert!(!rejection.quality_failed());
    }

    #[test]
    fn duplicate_candidate_identity_fails_whole_universe() {
        let topology = topology();
        let result = enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            &quality_requirement(),
            &quality_policy(),
            vec![allocation("same", 20.0), allocation("same", 30.0)],
            16,
        );
        assert!(matches!(
            result,
            Err(FeasibleSetError::DuplicateCandidateId(id)) if id == "same"
        ));
    }

    #[test]
    fn explicit_candidate_bound_is_enforced_before_assessment() {
        let topology = topology();
        let result = enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            &quality_requirement(),
            &quality_policy(),
            vec![allocation("a", 10.0), allocation("b", 10.0)],
            1,
        );
        assert!(matches!(
            result,
            Err(FeasibleSetError::CandidateBoundExceeded {
                actual: 2,
                maximum: 1
            })
        ));
    }

    #[test]
    fn zero_candidate_bound_is_invalid_even_for_empty_universe() {
        let topology = topology();
        let result = enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology, 40.0),
            &quality_requirement(),
            &quality_policy(),
            Vec::new(),
            0,
        );
        assert_eq!(result, Err(FeasibleSetError::ZeroCandidateBound));
    }

    #[test]
    fn feasible_entries_are_independent_alternatives_not_a_joint_bundle() {
        let set = enumerate(
            &AllocationBook::default(),
            vec![allocation("a", 60.0), allocation("b", 60.0)],
            40.0,
        )
        .unwrap();

        assert_eq!(set.semantics(), FeasibleSetSemantics::IndependentAlternativesAgainstSameBase);
        assert_eq!(set.feasible().len(), 2);
        // Each candidate is feasible against the same empty base book. The two
        // prospective books are separate one-candidate states; 60 + 60 is not
        // asserted jointly feasible against a 100-unit boundary.
        assert_eq!(set.feasible_candidate("a").unwrap().prospective_book().len(), 1);
        assert_eq!(set.feasible_candidate("b").unwrap().prospective_book().len(), 1);
    }

    #[test]
    fn exact_base_load_is_retained_and_used_for_every_candidate() {
        let topology = topology();
        let capacities = capacities(&topology);
        let mut base = AllocationBook::default();
        base.admit(&topology, &capacities, allocation("existing", 40.0))
            .unwrap();

        let set = enumerate(&base, vec![allocation("a", 50.0), allocation("b", 70.0)], 40.0)
            .unwrap();

        assert_eq!(set.base_book(), &base);
        assert_eq!(set.feasible().len(), 1);
        assert_eq!(set.feasible()[0].candidate_id(), "a");
        assert_eq!(set.feasible()[0].prospective_book().len(), 2);
        assert!(set.rejection("b").unwrap().quantity_failed());
        assert_eq!(base.len(), 1);
    }

    #[test]
    fn common_quality_policy_applies_to_entire_universe() {
        let set = enumerate(
            &AllocationBook::default(),
            vec![allocation("a", 10.0), allocation("b", 20.0)],
            90.0,
        )
        .unwrap();
        assert!(set.feasible().is_empty());
        assert_eq!(set.rejected().len(), 2);
        assert!(set.rejected().iter().all(ResourceCandidateRejection::quality_failed));
    }
}
