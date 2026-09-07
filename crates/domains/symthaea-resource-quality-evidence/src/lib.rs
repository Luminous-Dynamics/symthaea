// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-, time-, and provenance-bound resource quality evidence.
//!
//! `symthaea-resource-quality` deliberately answers only whether one quality
//! profile satisfies one quality requirement. This crate binds those profiles to
//! exact physical topology subjects, explicit validity windows, and evidence
//! classes before they can participate in later feasibility reasoning.
//!
//! Evidence class is descriptive provenance, not authority. In particular:
//! `Attested` does not itself prove that an attestor is trusted, current, or
//! policy-authorized. Those facts belong in Xenia/Nixward/Mycelix adapters.
//!
//! Multiple simultaneously valid claims are retained. This crate never resolves
//! disagreement with hidden precedence such as latest-wins or attested-wins.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_resource_model::ResourceKey;
use symthaea_resource_quality::ResourceQualityProfile;
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// Exact physical subject described by one quality claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualitySubject {
    Port { node_id: String, port_id: String },
    Link { link_id: String },
}

/// Descriptive origin class for quality evidence.
///
/// There is intentionally no total-order helper. A later policy may require one
/// or more classes, but this crate does not assert that one class is universally
/// more truthful or authoritative than another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityEvidenceClass {
    Declared,
    Observed,
    Attested,
}

/// One quality claim bound to exact topology, time, and provenance metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidenceWindow {
    pub id: String,
    pub subject: QualitySubject,
    pub valid_from: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
    pub profile: ResourceQualityProfile,
    pub evidence_class: QualityEvidenceClass,
    /// Opaque exact reference to the source observation/assertion/attestation.
    ///
    /// This crate checks only that the reference is non-empty. It does not
    /// authenticate, resolve, or assign institutional authority to the reference.
    pub evidence_ref: String,
}

impl QualityEvidenceWindow {
    pub fn contains(&self, at: DateTime<Utc>) -> bool {
        self.valid_from <= at && at < self.valid_until
    }

    pub fn validate(&self, topology: &ResourceTopology) -> Result<(), QualityEvidenceError> {
        if self.id.trim().is_empty() {
            return Err(QualityEvidenceError::EmptyEvidenceId);
        }
        if self.evidence_ref.trim().is_empty() {
            return Err(QualityEvidenceError::EmptyEvidenceRef(self.id.clone()));
        }
        if self.valid_from >= self.valid_until {
            return Err(QualityEvidenceError::InvalidValidityWindow(self.id.clone()));
        }

        let expected_key = subject_resource_key(topology, &self.subject)?;
        if self.profile.key != expected_key {
            return Err(QualityEvidenceError::ResourceKeyMismatch {
                evidence_id: self.id.clone(),
                expected: expected_key,
                provided: self.profile.key,
            });
        }
        Ok(())
    }
}

/// Collection of independently retained quality claims.
///
/// The set is keyed only by exact evidence ID. Overlapping windows, different
/// provenance classes, and contradictory profile values are intentionally
/// retained. A later feasibility/policy layer must say which exact evidence is
/// sufficient for one decision and must fail closed if its own ambiguity rules
/// are not satisfied.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidenceSet {
    evidence: BTreeMap<String, QualityEvidenceWindow>,
}

impl QualityEvidenceSet {
    pub fn insert(
        &mut self,
        topology: &ResourceTopology,
        evidence: QualityEvidenceWindow,
    ) -> Result<(), QualityEvidenceError> {
        evidence.validate(topology)?;
        if self.evidence.contains_key(&evidence.id) {
            return Err(QualityEvidenceError::DuplicateEvidenceId(evidence.id));
        }
        self.evidence.insert(evidence.id.clone(), evidence);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&QualityEvidenceWindow> {
        self.evidence.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &QualityEvidenceWindow> {
        self.evidence.values()
    }

    /// Every claim for one subject that is valid at the exact half-open instant.
    ///
    /// No precedence is applied. If two active claims disagree, both are returned.
    pub fn active_for_subject<'a>(
        &'a self,
        subject: &'a QualitySubject,
        at: DateTime<Utc>,
    ) -> impl Iterator<Item = &'a QualityEvidenceWindow> + 'a {
        self.evidence
            .values()
            .filter(move |entry| &entry.subject == subject && entry.contains(at))
    }

    pub fn active_for_subject_and_class<'a>(
        &'a self,
        subject: &'a QualitySubject,
        class: QualityEvidenceClass,
        at: DateTime<Utc>,
    ) -> impl Iterator<Item = &'a QualityEvidenceWindow> + 'a {
        self.active_for_subject(subject, at)
            .filter(move |entry| entry.evidence_class == class)
    }
}

/// Resolve the exact typed resource dimension exposed by one topology subject.
pub fn subject_resource_key(
    topology: &ResourceTopology,
    subject: &QualitySubject,
) -> Result<ResourceKey, QualityEvidenceError> {
    match subject {
        QualitySubject::Port { node_id, port_id } => topology
            .port(node_id, port_id)
            .map(|port| port.capacity.key)
            .ok_or_else(|| QualityEvidenceError::UnknownPort {
                node_id: node_id.clone(),
                port_id: port_id.clone(),
            }),
        QualitySubject::Link { link_id } => topology
            .link(link_id)
            .map(|link| link.capacity.key)
            .ok_or_else(|| QualityEvidenceError::UnknownLink(link_id.clone())),
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualityEvidenceError {
    #[error("quality evidence id must not be empty")]
    EmptyEvidenceId,
    #[error("quality evidence {0} must carry a non-empty evidence reference")]
    EmptyEvidenceRef(String),
    #[error("quality evidence {0} must have valid_from < valid_until")]
    InvalidValidityWindow(String),
    #[error("duplicate quality evidence id {0}")]
    DuplicateEvidenceId(String),
    #[error("unknown topology port {node_id}/{port_id}")]
    UnknownPort { node_id: String, port_id: String },
    #[error("unknown topology link {0}")]
    UnknownLink(String),
    #[error(
        "quality evidence {evidence_id} resource key mismatch: expected {expected:?}, got {provided:?}"
    )]
    ResourceKeyMismatch {
        evidence_id: String,
        expected: ResourceKey,
        provided: ResourceKey,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_quality::QualityMetric;
    use symthaea_resource_topology::ResourceLink;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn compute(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, amount: ResourceAmount) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity: amount,
        }
    }

    fn topology() -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, power(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, power(100.0))],
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

    fn subject() -> QualitySubject {
        QualitySubject::Link {
            link_id: "line".into(),
        }
    }

    fn profile(temp_c: f64) -> ResourceQualityProfile {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        profile
    }

    fn evidence(
        id: &str,
        class: QualityEvidenceClass,
        start_min: i64,
        end_min: i64,
        temp_c: f64,
    ) -> QualityEvidenceWindow {
        QualityEvidenceWindow {
            id: id.into(),
            subject: subject(),
            valid_from: t0() + Duration::minutes(start_min),
            valid_until: t0() + Duration::minutes(end_min),
            profile: profile(temp_c),
            evidence_class: class,
            evidence_ref: format!("evidence:{id}"),
        }
    }

    #[test]
    fn port_and_link_subjects_bind_exact_resource_key() {
        let topology = topology();
        let link = evidence("link", QualityEvidenceClass::Observed, 0, 60, 30.0);
        link.validate(&topology).unwrap();

        let mut port_evidence = link.clone();
        port_evidence.id = "port".into();
        port_evidence.subject = QualitySubject::Port {
            node_id: "source".into(),
            port_id: "out".into(),
        };
        port_evidence.validate(&topology).unwrap();
    }

    #[test]
    fn wrong_resource_dimension_is_rejected() {
        let topology = topology();
        let mut bad = evidence("wrong", QualityEvidenceClass::Observed, 0, 60, 30.0);
        bad.profile = ResourceQualityProfile::new(compute(1.0).key);
        assert!(matches!(
            bad.validate(&topology),
            Err(QualityEvidenceError::ResourceKeyMismatch { .. })
        ));
    }

    #[test]
    fn half_open_validity_is_exact() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("a", QualityEvidenceClass::Observed, 0, 30, 30.0),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("b", QualityEvidenceClass::Observed, 30, 60, 31.0),
        )
        .unwrap();

        let at_boundary: Vec<_> = set
            .active_for_subject(&subject(), t0() + Duration::minutes(30))
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(at_boundary, vec!["b"]);
    }

    #[test]
    fn evidence_classes_remain_distinct_without_hidden_ordering() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("declared", QualityEvidenceClass::Declared, 0, 60, 35.0),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("observed", QualityEvidenceClass::Observed, 0, 60, 42.0),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("attested", QualityEvidenceClass::Attested, 0, 60, 39.0),
        )
        .unwrap();

        let active: Vec<_> = set
            .active_for_subject(&subject(), t0() + Duration::minutes(10))
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(active, vec!["attested", "declared", "observed"]);

        let observed: Vec<_> = set
            .active_for_subject_and_class(
                &subject(),
                QualityEvidenceClass::Observed,
                t0() + Duration::minutes(10),
            )
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(observed, vec!["observed"]);
    }

    #[test]
    fn contradictory_active_claims_are_retained_not_resolved() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("cool", QualityEvidenceClass::Observed, 0, 60, 25.0),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("hot", QualityEvidenceClass::Observed, 0, 60, 85.0),
        )
        .unwrap();

        let active: Vec<_> = set
            .active_for_subject(&subject(), t0() + Duration::minutes(15))
            .collect();
        assert_eq!(active.len(), 2);
        assert_eq!(
            active[0]
                .profile
                .numeric(QualityMetric::TemperatureCelsius)
                .unwrap(),
            25.0
        );
        assert_eq!(
            active[1]
                .profile
                .numeric(QualityMetric::TemperatureCelsius)
                .unwrap(),
            85.0
        );
    }

    #[test]
    fn duplicate_identity_is_rejected_even_if_claim_bytes_match() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        let claim = evidence("same", QualityEvidenceClass::Observed, 0, 60, 30.0);
        set.insert(&topology, claim.clone()).unwrap();
        assert!(matches!(
            set.insert(&topology, claim),
            Err(QualityEvidenceError::DuplicateEvidenceId(id)) if id == "same"
        ));
    }

    #[test]
    fn empty_evidence_reference_is_rejected() {
        let topology = topology();
        let mut claim = evidence("bad-ref", QualityEvidenceClass::Declared, 0, 60, 30.0);
        claim.evidence_ref.clear();
        assert!(matches!(
            claim.validate(&topology),
            Err(QualityEvidenceError::EmptyEvidenceRef(id)) if id == "bad-ref"
        ));
    }

    #[test]
    fn unknown_subject_is_rejected() {
        let topology = topology();
        let mut claim = evidence("unknown", QualityEvidenceClass::Observed, 0, 60, 30.0);
        claim.subject = QualitySubject::Link {
            link_id: "missing".into(),
        };
        assert!(matches!(
            claim.validate(&topology),
            Err(QualityEvidenceError::UnknownLink(id)) if id == "missing"
        ));
    }

    #[test]
    fn invalid_time_window_is_rejected() {
        let topology = topology();
        let mut claim = evidence("time", QualityEvidenceClass::Observed, 0, 60, 30.0);
        claim.valid_until = claim.valid_from;
        assert!(matches!(
            claim.validate(&topology),
            Err(QualityEvidenceError::InvalidValidityWindow(id)) if id == "time"
        ));
    }
}
