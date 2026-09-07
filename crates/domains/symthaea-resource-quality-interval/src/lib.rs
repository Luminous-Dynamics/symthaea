// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Piecewise interval qualification for topology-bound resource quality evidence.
//!
//! Point-in-time quality qualification is insufficient for a resource allocation
//! that spans a non-zero interval. Evidence may expire, begin, disappear, or change
//! class/profile while the allocation is still active.
//!
//! This crate partitions the requested half-open interval at every relevant
//! evidence validity boundary. Because the active evidence set is constant inside
//! each resulting segment, qualifying the exact segment start proves the same
//! evidence-selection theorem for that complete segment.
//!
//! No quality value is interpolated and no gap is filled. Every segment must
//! independently qualify under the caller's exact `QualityQualificationPolicy`.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use std::collections::BTreeSet;
use symthaea_resource_quality::ResourceQualityRequirement;
use symthaea_resource_quality_evidence::{QualityEvidenceSet, QualitySubject};
use symthaea_resource_quality_qualification::{
    qualify_quality_evidence, QualifiedQualityEvidence, QualityQualificationError,
    QualityQualificationPolicy,
};
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// One non-empty half-open segment whose quality evidence independently qualified.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedQualitySegment {
    valid_from: DateTime<Utc>,
    valid_until: DateTime<Utc>,
    qualification: QualifiedQualityEvidence,
}

impl QualifiedQualitySegment {
    pub fn valid_from(&self) -> DateTime<Utc> {
        self.valid_from
    }

    pub fn valid_until(&self) -> DateTime<Utc> {
        self.valid_until
    }

    pub fn qualification(&self) -> &QualifiedQualityEvidence {
        &self.qualification
    }
}

/// Non-serializable positive witness that one exact resource-quality requirement
/// remained qualified throughout the complete requested interval.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedQualityInterval {
    subject: QualitySubject,
    valid_from: DateTime<Utc>,
    valid_until: DateTime<Utc>,
    segments: Vec<QualifiedQualitySegment>,
}

impl QualifiedQualityInterval {
    pub fn subject(&self) -> &QualitySubject {
        &self.subject
    }

    pub fn valid_from(&self) -> DateTime<Utc> {
        self.valid_from
    }

    pub fn valid_until(&self) -> DateTime<Utc> {
        self.valid_until
    }

    pub fn segments(&self) -> &[QualifiedQualitySegment] {
        &self.segments
    }
}

/// Qualify quality continuously across one exact half-open interval.
///
/// Every evidence boundary touching the target subject inside the requested
/// interval becomes a partition point. Each resulting segment is then qualified
/// at its exact start using the ordinary point-in-time qualifier.
pub fn qualify_quality_interval(
    topology: &ResourceTopology,
    evidence_set: &QualityEvidenceSet,
    subject: &QualitySubject,
    requirement: &ResourceQualityRequirement,
    policy: &QualityQualificationPolicy,
    valid_from: DateTime<Utc>,
    valid_until: DateTime<Utc>,
) -> Result<QualifiedQualityInterval, QualityIntervalError> {
    if valid_from >= valid_until {
        return Err(QualityIntervalError::InvalidInterval);
    }

    // Validate the policy even if no evidence exists so malformed policy and
    // missing evidence remain distinct failure classes.
    policy
        .validate()
        .map_err(QualityIntervalError::InvalidPolicy)?;

    let mut boundaries = BTreeSet::new();
    boundaries.insert(valid_from);
    boundaries.insert(valid_until);

    for evidence in evidence_set.iter() {
        if &evidence.subject != subject {
            continue;
        }
        // Only boundaries that can change the active set inside the requested
        // interval matter. Exact target endpoints are already present.
        if valid_from < evidence.valid_from && evidence.valid_from < valid_until {
            boundaries.insert(evidence.valid_from);
        }
        if valid_from < evidence.valid_until && evidence.valid_until < valid_until {
            boundaries.insert(evidence.valid_until);
        }
    }

    let points: Vec<_> = boundaries.into_iter().collect();
    let mut segments = Vec::with_capacity(points.len().saturating_sub(1));

    for pair in points.windows(2) {
        let segment_from = pair[0];
        let segment_until = pair[1];
        if segment_from >= segment_until {
            continue;
        }

        let qualification = qualify_quality_evidence(
            topology,
            evidence_set,
            subject,
            requirement,
            policy,
            segment_from,
        )
        .map_err(|source| QualityIntervalError::SegmentRejected {
            valid_from: segment_from,
            valid_until: segment_until,
            source,
        })?;

        segments.push(QualifiedQualitySegment {
            valid_from: segment_from,
            valid_until: segment_until,
            qualification,
        });
    }

    if segments.is_empty() {
        return Err(QualityIntervalError::InvalidInterval);
    }

    debug_assert_eq!(segments.first().map(|s| s.valid_from), Some(valid_from));
    debug_assert_eq!(segments.last().map(|s| s.valid_until), Some(valid_until));

    Ok(QualifiedQualityInterval {
        subject: subject.clone(),
        valid_from,
        valid_until,
        segments,
    })
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualityIntervalError {
    #[error("quality qualification interval must have valid_from < valid_until")]
    InvalidInterval,
    #[error("invalid quality qualification policy: {0}")]
    InvalidPolicy(QualityQualificationError),
    #[error(
        "quality evidence rejected for segment [{valid_from}, {valid_until}): {source}"
    )]
    SegmentRejected {
        valid_from: DateTime<Utc>,
        valid_until: DateTime<Utc>,
        source: QualityQualificationError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use std::collections::BTreeSet;
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

    fn subject() -> QualitySubject {
        QualitySubject::Link {
            link_id: "line".into(),
        }
    }

    fn requirement() -> ResourceQualityRequirement {
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

    fn policy() -> QualityQualificationPolicy {
        QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        }
    }

    fn evidence(
        id: &str,
        temp_c: f64,
        start_min: i64,
        end_min: i64,
    ) -> QualityEvidenceWindow {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        QualityEvidenceWindow {
            id: id.into(),
            subject: subject(),
            valid_from: t0() + Duration::minutes(start_min),
            valid_until: t0() + Duration::minutes(end_min),
            profile,
            evidence_class: QualityEvidenceClass::Observed,
            evidence_ref: format!("source:{id}"),
        }
    }

    #[test]
    fn one_evidence_window_can_cover_complete_interval() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("all", 40.0, 0, 60)).unwrap();

        let qualified = qualify_quality_interval(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy(),
            t0(),
            t0() + Duration::minutes(60),
        )
        .unwrap();
        assert_eq!(qualified.segments().len(), 1);
        assert_eq!(qualified.segments()[0].valid_from(), t0());
        assert_eq!(
            qualified.segments()[0].valid_until(),
            t0() + Duration::minutes(60)
        );
    }

    #[test]
    fn adjacent_evidence_windows_cover_without_gap_or_overlap_precedence() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("a", 40.0, 0, 30)).unwrap();
        set.insert(&topology, evidence("b", 45.0, 30, 60)).unwrap();

        let qualified = qualify_quality_interval(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy(),
            t0(),
            t0() + Duration::minutes(60),
        )
        .unwrap();
        assert_eq!(qualified.segments().len(), 2);
        assert_eq!(qualified.segments()[0].qualification().evidence_ids(), &["a".to_string()]);
        assert_eq!(qualified.segments()[1].qualification().evidence_ids(), &["b".to_string()]);
    }

    #[test]
    fn uncovered_middle_gap_fails_closed() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("a", 40.0, 0, 20)).unwrap();
        set.insert(&topology, evidence("b", 40.0, 30, 60)).unwrap();

        assert!(matches!(
            qualify_quality_interval(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy(),
                t0(),
                t0() + Duration::minutes(60),
            ),
            Err(QualityIntervalError::SegmentRejected {
                valid_from,
                valid_until,
                source: QualityQualificationError::NoAdmissibleCurrentEvidence,
            }) if valid_from == t0() + Duration::minutes(20)
                && valid_until == t0() + Duration::minutes(30)
        ));
    }

    #[test]
    fn later_incompatible_quality_rejects_whole_interval() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("good", 40.0, 0, 30)).unwrap();
        set.insert(&topology, evidence("bad", 90.0, 30, 60)).unwrap();

        assert!(matches!(
            qualify_quality_interval(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy(),
                t0(),
                t0() + Duration::minutes(60),
            ),
            Err(QualityIntervalError::SegmentRejected {
                valid_from,
                source: QualityQualificationError::IncompatibleEvidence { evidence_id, .. },
                ..
            }) if valid_from == t0() + Duration::minutes(30) && evidence_id == "bad"
        ));
    }

    #[test]
    fn evidence_boundaries_outside_requested_interval_do_not_split_it() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("wide", 40.0, -60, 120)).unwrap();

        let qualified = qualify_quality_interval(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy(),
            t0(),
            t0() + Duration::minutes(60),
        )
        .unwrap();
        assert_eq!(qualified.segments().len(), 1);
    }

    #[test]
    fn unrelated_subject_boundaries_do_not_split_target_interval() {
        let mut topology = topology();
        topology
            .add_node("other", [port("in", PortDirection::Input, 100.0)])
            .unwrap();

        let mut set = QualityEvidenceSet::default();
        set.insert(&topology, evidence("target", 40.0, 0, 60)).unwrap();

        let mut unrelated = evidence("other", 40.0, 10, 20);
        unrelated.subject = QualitySubject::Port {
            node_id: "other".into(),
            port_id: "in".into(),
        };
        set.insert(&topology, unrelated).unwrap();

        let qualified = qualify_quality_interval(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy(),
            t0(),
            t0() + Duration::minutes(60),
        )
        .unwrap();
        assert_eq!(qualified.segments().len(), 1);
    }

    #[test]
    fn invalid_zero_length_interval_is_rejected_before_evidence_lookup() {
        let topology = topology();
        let set = QualityEvidenceSet::default();
        assert!(matches!(
            qualify_quality_interval(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy(),
                t0(),
                t0(),
            ),
            Err(QualityIntervalError::InvalidInterval)
        ));
    }

    #[test]
    fn malformed_policy_remains_distinct_from_missing_evidence() {
        let topology = topology();
        let set = QualityEvidenceSet::default();
        let bad_policy = QualityQualificationPolicy {
            allowed_classes: BTreeSet::new(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        assert!(matches!(
            qualify_quality_interval(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &bad_policy,
                t0(),
                t0() + Duration::minutes(10),
            ),
            Err(QualityIntervalError::InvalidPolicy(
                QualityQualificationError::NoAllowedEvidenceClasses
            ))
        ));
    }
}
