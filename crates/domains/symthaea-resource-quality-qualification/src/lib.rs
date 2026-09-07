// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed qualification of topology-bound resource quality evidence.
//!
//! This crate composes three facts without collapsing them:
//! - a consumer quality requirement;
//! - current topology-bound quality evidence; and
//! - an explicit policy describing which evidence is admissible for this decision.
//!
//! A favorable claim is never cherry-picked from contradictory current evidence.
//! The caller must state how multiple admissible claims are handled, and positive
//! qualification requires every claim admitted by that rule to satisfy the quality
//! requirement.
//!
//! Positive output remains planning evidence only. It is not evidence-source
//! authentication, institutional authority, a Mycelix lease, or execution authority.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_resource_quality::{QualityViolation, ResourceQualityRequirement};
use symthaea_resource_quality_evidence::{
    subject_resource_key, QualityEvidenceClass, QualityEvidenceSet, QualityEvidenceWindow,
    QualitySubject,
};
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// Which exact evidence identities may participate in one qualification decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceIdentityPolicy {
    Any,
    EvidenceIdAllowlist(BTreeSet<String>),
    EvidenceRefAllowlist(BTreeSet<String>),
}

/// Explicit treatment of multiple simultaneously admissible claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipleEvidenceRule {
    /// Exactly one admissible current claim must exist.
    RejectMultiple,
    /// Multiple claims are allowed, but every one must independently satisfy the
    /// quality requirement.
    RequireAllCompatible,
    /// Multiple claims are allowed only if their complete quality profiles are
    /// exactly equal, and that common profile must satisfy the requirement.
    RequireExactProfileAgreement,
}

/// Decision-local evidence policy. This is not a trust-policy proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityQualificationPolicy {
    pub allowed_classes: BTreeSet<QualityEvidenceClass>,
    pub identity_policy: EvidenceIdentityPolicy,
    pub minimum_evidence: usize,
    pub multiple_evidence_rule: MultipleEvidenceRule,
}

impl QualityQualificationPolicy {
    pub fn validate(&self) -> Result<(), QualityQualificationError> {
        if self.allowed_classes.is_empty() {
            return Err(QualityQualificationError::NoAllowedEvidenceClasses);
        }
        if self.minimum_evidence == 0 {
            return Err(QualityQualificationError::ZeroMinimumEvidence);
        }
        match &self.identity_policy {
            EvidenceIdentityPolicy::Any => {}
            EvidenceIdentityPolicy::EvidenceIdAllowlist(values)
            | EvidenceIdentityPolicy::EvidenceRefAllowlist(values) => {
                if values.is_empty() {
                    return Err(QualityQualificationError::EmptyIdentityAllowlist);
                }
                if values.iter().any(|value| value.trim().is_empty()) {
                    return Err(QualityQualificationError::BlankIdentityAllowlistEntry);
                }
            }
        }
        if self.multiple_evidence_rule == MultipleEvidenceRule::RejectMultiple
            && self.minimum_evidence > 1
        {
            return Err(QualityQualificationError::ImpossibleSingleEvidencePolicy {
                minimum_evidence: self.minimum_evidence,
            });
        }
        Ok(())
    }

    fn admits(&self, evidence: &QualityEvidenceWindow) -> bool {
        if !self.allowed_classes.contains(&evidence.evidence_class) {
            return false;
        }
        match &self.identity_policy {
            EvidenceIdentityPolicy::Any => true,
            EvidenceIdentityPolicy::EvidenceIdAllowlist(values) => values.contains(&evidence.id),
            EvidenceIdentityPolicy::EvidenceRefAllowlist(values) => {
                values.contains(&evidence.evidence_ref)
            }
        }
    }
}

/// Non-serializable positive planning witness.
///
/// It retains the exact decision time and exact evidence identities that were
/// admitted. Reconstructing a similarly shaped value from wire data grants no
/// qualification because there is no serde/public field constructor for this type.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedQualityEvidence {
    subject: QualitySubject,
    qualified_at: DateTime<Utc>,
    evidence_ids: Vec<String>,
    evidence_refs: Vec<String>,
    evidence_classes: Vec<QualityEvidenceClass>,
    multiple_evidence_rule: MultipleEvidenceRule,
}

impl QualifiedQualityEvidence {
    pub fn subject(&self) -> &QualitySubject {
        &self.subject
    }

    pub fn qualified_at(&self) -> DateTime<Utc> {
        self.qualified_at
    }

    pub fn evidence_ids(&self) -> &[String] {
        &self.evidence_ids
    }

    pub fn evidence_refs(&self) -> &[String] {
        &self.evidence_refs
    }

    pub fn evidence_classes(&self) -> &[QualityEvidenceClass] {
        &self.evidence_classes
    }

    pub fn multiple_evidence_rule(&self) -> MultipleEvidenceRule {
        self.multiple_evidence_rule
    }
}

/// Qualify one exact quality requirement at one exact instant.
pub fn qualify_quality_evidence(
    topology: &ResourceTopology,
    evidence_set: &QualityEvidenceSet,
    subject: &QualitySubject,
    requirement: &ResourceQualityRequirement,
    policy: &QualityQualificationPolicy,
    at: DateTime<Utc>,
) -> Result<QualifiedQualityEvidence, QualityQualificationError> {
    policy.validate()?;

    let subject_key = subject_resource_key(topology, subject)
        .map_err(|error| QualityQualificationError::InvalidSubject(error.to_string()))?;
    if requirement.key != subject_key {
        return Err(QualityQualificationError::RequirementResourceKeyMismatch);
    }

    let mut admissible = Vec::new();
    for evidence in evidence_set.active_for_subject(subject, at) {
        if !policy.admits(evidence) {
            continue;
        }
        evidence.validate(topology).map_err(|error| {
            QualityQualificationError::InvalidEvidence {
                evidence_id: evidence.id.clone(),
                reason: error.to_string(),
            }
        })?;
        admissible.push(evidence);
    }

    if admissible.is_empty() {
        return Err(QualityQualificationError::NoAdmissibleCurrentEvidence);
    }
    if admissible.len() < policy.minimum_evidence {
        return Err(QualityQualificationError::InsufficientEvidence {
            required: policy.minimum_evidence,
            found: admissible.len(),
        });
    }

    let evidence_ids: Vec<String> = admissible.iter().map(|entry| entry.id.clone()).collect();

    match policy.multiple_evidence_rule {
        MultipleEvidenceRule::RejectMultiple if admissible.len() != 1 => {
            return Err(QualityQualificationError::MultipleEvidenceRejected {
                evidence_ids,
            });
        }
        MultipleEvidenceRule::RequireExactProfileAgreement if admissible.len() > 1 => {
            let first = &admissible[0].profile;
            if admissible.iter().skip(1).any(|entry| &entry.profile != first) {
                return Err(QualityQualificationError::ProfileDisagreement {
                    evidence_ids,
                });
            }
        }
        _ => {}
    }

    for evidence in &admissible {
        let compatibility = requirement
            .evaluate(&evidence.profile)
            .map_err(|error| QualityQualificationError::QualityEvaluation {
                evidence_id: evidence.id.clone(),
                reason: error.to_string(),
            })?;
        if !compatibility.compatible {
            return Err(QualityQualificationError::IncompatibleEvidence {
                evidence_id: evidence.id.clone(),
                violations: compatibility.violations,
            });
        }
    }

    Ok(QualifiedQualityEvidence {
        subject: subject.clone(),
        qualified_at: at,
        evidence_ids: admissible.iter().map(|entry| entry.id.clone()).collect(),
        evidence_refs: admissible
            .iter()
            .map(|entry| entry.evidence_ref.clone())
            .collect(),
        evidence_classes: admissible
            .iter()
            .map(|entry| entry.evidence_class)
            .collect(),
        multiple_evidence_rule: policy.multiple_evidence_rule,
    })
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualityQualificationError {
    #[error("quality qualification policy must allow at least one evidence class")]
    NoAllowedEvidenceClasses,
    #[error("quality qualification minimum evidence must be non-zero")]
    ZeroMinimumEvidence,
    #[error("quality evidence identity allowlist must not be empty")]
    EmptyIdentityAllowlist,
    #[error("quality evidence identity allowlist contains a blank entry")]
    BlankIdentityAllowlistEntry,
    #[error("RejectMultiple cannot require {minimum_evidence} evidence records")]
    ImpossibleSingleEvidencePolicy { minimum_evidence: usize },
    #[error("invalid quality subject: {0}")]
    InvalidSubject(String),
    #[error("quality requirement resource key does not match the topology subject")]
    RequirementResourceKeyMismatch,
    #[error("no admissible current quality evidence exists")]
    NoAdmissibleCurrentEvidence,
    #[error("quality qualification requires {required} evidence records but found {found}")]
    InsufficientEvidence { required: usize, found: usize },
    #[error("multiple current quality evidence records are not permitted: {evidence_ids:?}")]
    MultipleEvidenceRejected { evidence_ids: Vec<String> },
    #[error("current quality evidence profiles disagree: {evidence_ids:?}")]
    ProfileDisagreement { evidence_ids: Vec<String> },
    #[error("quality evidence {evidence_id} is invalid: {reason}")]
    InvalidEvidence { evidence_id: String, reason: String },
    #[error("quality evaluation failed for evidence {evidence_id}: {reason}")]
    QualityEvaluation { evidence_id: String, reason: String },
    #[error("quality evidence {evidence_id} is incompatible: {violations:?}")]
    IncompatibleEvidence {
        evidence_id: String,
        violations: Vec<QualityViolation>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
    };
    use symthaea_resource_quality_evidence::{QualityEvidenceWindow, QualitySubject};
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

    fn evidence(
        id: &str,
        class: QualityEvidenceClass,
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
            evidence_class: class,
            evidence_ref: format!("source:{id}"),
        }
    }

    fn policy(
        classes: impl IntoIterator<Item = QualityEvidenceClass>,
        multiple_evidence_rule: MultipleEvidenceRule,
    ) -> QualityQualificationPolicy {
        QualityQualificationPolicy {
            allowed_classes: classes.into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule,
        }
    }

    #[test]
    fn exact_single_current_observation_qualifies() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("obs", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        let result = qualify_quality_evidence(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy([QualityEvidenceClass::Observed], MultipleEvidenceRule::RejectMultiple),
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert_eq!(result.evidence_ids(), &["obs".to_string()]);
        assert_eq!(result.evidence_classes(), &[QualityEvidenceClass::Observed]);
    }

    #[test]
    fn disallowed_evidence_class_cannot_qualify() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("declared", QualityEvidenceClass::Declared, 40.0, 0, 60),
        )
        .unwrap();
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy([QualityEvidenceClass::Observed], MultipleEvidenceRule::RejectMultiple),
                t0() + Duration::minutes(10),
            ),
            Err(QualityQualificationError::NoAdmissibleCurrentEvidence)
        ));
    }

    #[test]
    fn exact_evidence_id_allowlist_selects_identity_without_precedence() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("a", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("b", QualityEvidenceClass::Observed, 45.0, 0, 60),
        )
        .unwrap();

        let mut allow = BTreeSet::new();
        allow.insert("b".to_string());
        let policy = QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::EvidenceIdAllowlist(allow),
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        let result = qualify_quality_evidence(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy,
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert_eq!(result.evidence_ids(), &["b".to_string()]);
    }

    #[test]
    fn compatible_claim_cannot_hide_incompatible_current_claim() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("good", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("bad", QualityEvidenceClass::Attested, 90.0, 0, 60),
        )
        .unwrap();

        let policy = policy(
            [QualityEvidenceClass::Observed, QualityEvidenceClass::Attested],
            MultipleEvidenceRule::RequireAllCompatible,
        );
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy,
                t0() + Duration::minutes(10),
            ),
            Err(QualityQualificationError::IncompatibleEvidence { evidence_id, .. })
                if evidence_id == "bad"
        ));
    }

    #[test]
    fn reject_multiple_fails_even_when_every_claim_is_compatible() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("a", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("b", QualityEvidenceClass::Observed, 45.0, 0, 60),
        )
        .unwrap();
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy([QualityEvidenceClass::Observed], MultipleEvidenceRule::RejectMultiple),
                t0() + Duration::minutes(10),
            ),
            Err(QualityQualificationError::MultipleEvidenceRejected { .. })
        ));
    }

    #[test]
    fn exact_profile_agreement_accepts_duplicate_independent_claims() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("a", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("b", QualityEvidenceClass::Attested, 40.0, 0, 60),
        )
        .unwrap();
        let result = qualify_quality_evidence(
            &topology,
            &set,
            &subject(),
            &requirement(),
            &policy(
                [QualityEvidenceClass::Observed, QualityEvidenceClass::Attested],
                MultipleEvidenceRule::RequireExactProfileAgreement,
            ),
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert_eq!(result.evidence_ids().len(), 2);
    }

    #[test]
    fn exact_profile_agreement_rejects_distinct_compatible_values() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("a", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        set.insert(
            &topology,
            evidence("b", QualityEvidenceClass::Attested, 45.0, 0, 60),
        )
        .unwrap();
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy(
                    [QualityEvidenceClass::Observed, QualityEvidenceClass::Attested],
                    MultipleEvidenceRule::RequireExactProfileAgreement,
                ),
                t0() + Duration::minutes(10),
            ),
            Err(QualityQualificationError::ProfileDisagreement { .. })
        ));
    }

    #[test]
    fn expired_evidence_does_not_qualify() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("old", QualityEvidenceClass::Observed, 40.0, 0, 30),
        )
        .unwrap();
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy([QualityEvidenceClass::Observed], MultipleEvidenceRule::RejectMultiple),
                t0() + Duration::minutes(30),
            ),
            Err(QualityQualificationError::NoAdmissibleCurrentEvidence)
        ));
    }

    #[test]
    fn minimum_evidence_is_enforced_after_class_and_identity_filtering() {
        let topology = topology();
        let mut set = QualityEvidenceSet::default();
        set.insert(
            &topology,
            evidence("only", QualityEvidenceClass::Observed, 40.0, 0, 60),
        )
        .unwrap();
        let mut policy = policy(
            [QualityEvidenceClass::Observed],
            MultipleEvidenceRule::RequireAllCompatible,
        );
        policy.minimum_evidence = 2;
        assert!(matches!(
            qualify_quality_evidence(
                &topology,
                &set,
                &subject(),
                &requirement(),
                &policy,
                t0() + Duration::minutes(10),
            ),
            Err(QualityQualificationError::InsufficientEvidence {
                required: 2,
                found: 1
            })
        ));
    }
}
