// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-evaluable aircraft dispatch and operational-readiness gate.
//!
//! Release closure proves that a software/deployment package was qualified.
//! Dispatch readiness is narrower and time-sensitive: it binds that release to
//! the actual aircraft, current maintenance state, configuration drift,
//! operational limits, crew alerting, command security, update state, and
//! mission evidence. Missing or expired evidence cannot be converted into a
//! generic green status.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReadinessArtifactKind {
    ReleaseClosure,
    DeploymentBinding,
    AircraftConfiguration,
    FleetDriftAssessment,
    MaintenanceAssessment,
    OperationalLimits,
    EstimatorHealth,
    RealtimeHealth,
    EnvelopeConformance,
    FaultRecoveryCampaign,
    CommandSecurity,
    SecureUpdateState,
    HumanFactorsAssessment,
    MissionAuthority,
    FlightDataGovernance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessArtifactStatus {
    Verified,
    Restricted,
    Failed,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadinessArtifact {
    pub artifact_id: String,
    pub aircraft_id: String,
    pub mission_id: Option<String>,
    pub kind: ReadinessArtifactKind,
    pub status: ReadinessArtifactStatus,
    pub issued_at_ms: u64,
    pub valid_until_ms: u64,
    pub digest: String,
    pub authenticity_reference: Option<String>,
    pub restrictions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationalReadinessPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub aircraft_id: String,
    pub mission_id: String,
    pub required_kinds: Vec<ReadinessArtifactKind>,
    pub authenticity_required_for: Vec<ReadinessArtifactKind>,
    pub restriction_permitted_for: Vec<ReadinessArtifactKind>,
    pub maximum_artifact_age_ms: BTreeMap<ReadinessArtifactKind, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalReadinessStatus {
    Dispatchable,
    DispatchableRestricted,
    NoGo,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalReadinessIssue {
    EmptyIdentity,
    DuplicateArtifactId(String),
    DuplicateArtifactKind(ReadinessArtifactKind),
    MissingArtifact(ReadinessArtifactKind),
    AircraftMismatch(String),
    MissionMismatch(String),
    InvalidDigest(String),
    ArtifactNotYetValid(String),
    ArtifactExpired(String),
    ArtifactTooOld {
        artifact_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    MissingAuthenticity(String),
    ArtifactFailed(String),
    ArtifactIncomplete(String),
    RestrictionNotPermitted(String),
    EmptyRestriction(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationalReadinessReport {
    pub schema_version: String,
    pub policy_id: String,
    pub aircraft_id: String,
    pub mission_id: String,
    pub assessed_at_ms: u64,
    pub status: OperationalReadinessStatus,
    pub issues: Vec<OperationalReadinessIssue>,
    pub active_restrictions: Vec<String>,
    pub verified_kinds: Vec<ReadinessArtifactKind>,
    pub artifact_count: usize,
}

impl OperationalReadinessReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, OperationalReadinessError> {
        let mut canonical = self.clone();
        canonical.verified_kinds.sort();
        canonical.active_restrictions.sort();
        canonical.active_restrictions.dedup();
        canonical.issues.sort_by_key(issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| OperationalReadinessError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, OperationalReadinessError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalReadinessError {
    InvalidPolicy,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct OperationalReadinessGate {
    policy: OperationalReadinessPolicy,
}

impl OperationalReadinessGate {
    pub fn new(policy: OperationalReadinessPolicy) -> Result<Self, OperationalReadinessError> {
        let required: BTreeSet<_> = policy.required_kinds.iter().copied().collect();
        let authenticity: BTreeSet<_> = policy.authenticity_required_for.iter().copied().collect();
        let restricted: BTreeSet<_> = policy.restriction_permitted_for.iter().copied().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.aircraft_id.trim().is_empty()
            || policy.mission_id.trim().is_empty()
            || policy.required_kinds.is_empty()
            || required.len() != policy.required_kinds.len()
            || authenticity.len() != policy.authenticity_required_for.len()
            || restricted.len() != policy.restriction_permitted_for.len()
            || !authenticity.is_subset(&required)
            || !restricted.is_subset(&required)
            || policy
                .maximum_artifact_age_ms
                .iter()
                .any(|(kind, age)| !required.contains(kind) || *age == 0)
        {
            return Err(OperationalReadinessError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        artifacts: &[ReadinessArtifact],
        now_ms: u64,
    ) -> OperationalReadinessReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut by_kind = BTreeMap::<ReadinessArtifactKind, &ReadinessArtifact>::new();
        let mut active_restrictions = Vec::new();

        for artifact in artifacts {
            if artifact.artifact_id.trim().is_empty() || artifact.aircraft_id.trim().is_empty() {
                issues.push(OperationalReadinessIssue::EmptyIdentity);
            }
            if !ids.insert(artifact.artifact_id.clone()) {
                issues.push(OperationalReadinessIssue::DuplicateArtifactId(
                    artifact.artifact_id.clone(),
                ));
            }
            if by_kind.insert(artifact.kind, artifact).is_some() {
                issues.push(OperationalReadinessIssue::DuplicateArtifactKind(
                    artifact.kind,
                ));
            }
            if artifact.aircraft_id != self.policy.aircraft_id {
                issues.push(OperationalReadinessIssue::AircraftMismatch(
                    artifact.artifact_id.clone(),
                ));
            }
            if artifact
                .mission_id
                .as_ref()
                .is_some_and(|mission_id| mission_id != &self.policy.mission_id)
            {
                issues.push(OperationalReadinessIssue::MissionMismatch(
                    artifact.artifact_id.clone(),
                ));
            }
            if !valid_digest(&artifact.digest) {
                issues.push(OperationalReadinessIssue::InvalidDigest(
                    artifact.artifact_id.clone(),
                ));
            }
            if now_ms < artifact.issued_at_ms {
                issues.push(OperationalReadinessIssue::ArtifactNotYetValid(
                    artifact.artifact_id.clone(),
                ));
            }
            if now_ms > artifact.valid_until_ms {
                issues.push(OperationalReadinessIssue::ArtifactExpired(
                    artifact.artifact_id.clone(),
                ));
            }
            if let Some(maximum_age) = self.policy.maximum_artifact_age_ms.get(&artifact.kind) {
                let age = now_ms.saturating_sub(artifact.issued_at_ms);
                if age > *maximum_age {
                    issues.push(OperationalReadinessIssue::ArtifactTooOld {
                        artifact_id: artifact.artifact_id.clone(),
                        age_ms: age,
                        maximum_ms: *maximum_age,
                    });
                }
            }
            if self
                .policy
                .authenticity_required_for
                .contains(&artifact.kind)
                && artifact
                    .authenticity_reference
                    .as_ref()
                    .is_none_or(|reference| reference.trim().is_empty())
            {
                issues.push(OperationalReadinessIssue::MissingAuthenticity(
                    artifact.artifact_id.clone(),
                ));
            }
            match artifact.status {
                ReadinessArtifactStatus::Verified => {}
                ReadinessArtifactStatus::Restricted => {
                    if !self
                        .policy
                        .restriction_permitted_for
                        .contains(&artifact.kind)
                    {
                        issues.push(OperationalReadinessIssue::RestrictionNotPermitted(
                            artifact.artifact_id.clone(),
                        ));
                    }
                    if artifact.restrictions.is_empty()
                        || artifact
                            .restrictions
                            .iter()
                            .any(|restriction| restriction.trim().is_empty())
                    {
                        issues.push(OperationalReadinessIssue::EmptyRestriction(
                            artifact.artifact_id.clone(),
                        ));
                    } else {
                        active_restrictions.extend(artifact.restrictions.iter().cloned());
                    }
                }
                ReadinessArtifactStatus::Failed => {
                    issues.push(OperationalReadinessIssue::ArtifactFailed(
                        artifact.artifact_id.clone(),
                    ));
                }
                ReadinessArtifactStatus::Incomplete => {
                    issues.push(OperationalReadinessIssue::ArtifactIncomplete(
                        artifact.artifact_id.clone(),
                    ));
                }
            }
        }

        for required in &self.policy.required_kinds {
            if !by_kind.contains_key(required) {
                issues.push(OperationalReadinessIssue::MissingArtifact(*required));
            }
        }

        let hard_failure = issues.iter().any(|issue| {
            matches!(
                issue,
                OperationalReadinessIssue::EmptyIdentity
                    | OperationalReadinessIssue::DuplicateArtifactId(_)
                    | OperationalReadinessIssue::DuplicateArtifactKind(_)
                    | OperationalReadinessIssue::AircraftMismatch(_)
                    | OperationalReadinessIssue::MissionMismatch(_)
                    | OperationalReadinessIssue::InvalidDigest(_)
                    | OperationalReadinessIssue::ArtifactNotYetValid(_)
                    | OperationalReadinessIssue::ArtifactExpired(_)
                    | OperationalReadinessIssue::ArtifactTooOld { .. }
                    | OperationalReadinessIssue::MissingAuthenticity(_)
                    | OperationalReadinessIssue::ArtifactFailed(_)
                    | OperationalReadinessIssue::RestrictionNotPermitted(_)
                    | OperationalReadinessIssue::EmptyRestriction(_)
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                OperationalReadinessIssue::MissingArtifact(_)
                    | OperationalReadinessIssue::ArtifactIncomplete(_)
            )
        });
        active_restrictions.sort();
        active_restrictions.dedup();
        let status = if hard_failure {
            OperationalReadinessStatus::NoGo
        } else if incomplete {
            OperationalReadinessStatus::Incomplete
        } else if active_restrictions.is_empty() {
            OperationalReadinessStatus::Dispatchable
        } else {
            OperationalReadinessStatus::DispatchableRestricted
        };
        let mut verified_kinds: Vec<_> = by_kind
            .values()
            .filter(|artifact| {
                matches!(
                    artifact.status,
                    ReadinessArtifactStatus::Verified | ReadinessArtifactStatus::Restricted
                )
            })
            .map(|artifact| artifact.kind)
            .collect();
        verified_kinds.sort();
        verified_kinds.dedup();

        OperationalReadinessReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            aircraft_id: self.policy.aircraft_id.clone(),
            mission_id: self.policy.mission_id.clone(),
            assessed_at_ms: now_ms,
            status,
            issues,
            active_restrictions,
            verified_kinds,
            artifact_count: artifacts.len(),
        }
    }
}

fn valid_digest(digest: &str) -> bool {
    let Some((algorithm, value)) = digest.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && value.len() >= 16
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn issue_sort_key(issue: &OperationalReadinessIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> OperationalReadinessPolicy {
        OperationalReadinessPolicy {
            schema_version: "1".into(),
            policy_id: "dispatch".into(),
            aircraft_id: "aircraft-1".into(),
            mission_id: "mission-1".into(),
            required_kinds: vec![
                ReadinessArtifactKind::ReleaseClosure,
                ReadinessArtifactKind::MaintenanceAssessment,
                ReadinessArtifactKind::OperationalLimits,
            ],
            authenticity_required_for: vec![ReadinessArtifactKind::ReleaseClosure],
            restriction_permitted_for: vec![ReadinessArtifactKind::OperationalLimits],
            maximum_artifact_age_ms: BTreeMap::from([
                (ReadinessArtifactKind::MaintenanceAssessment, 10_000),
                (ReadinessArtifactKind::OperationalLimits, 1_000),
            ]),
        }
    }

    fn artifact(kind: ReadinessArtifactKind) -> ReadinessArtifact {
        ReadinessArtifact {
            artifact_id: format!("{kind:?}"),
            aircraft_id: "aircraft-1".into(),
            mission_id: Some("mission-1".into()),
            kind,
            status: ReadinessArtifactStatus::Verified,
            issued_at_ms: 900,
            valid_until_ms: 2_000,
            digest: "sha256:0123456789abcdef".into(),
            authenticity_reference: (kind == ReadinessArtifactKind::ReleaseClosure)
                .then(|| "signature:release".into()),
            restrictions: Vec::new(),
        }
    }

    #[test]
    fn complete_current_evidence_is_dispatchable() {
        let gate = OperationalReadinessGate::new(policy()).unwrap();
        let artifacts = vec![
            artifact(ReadinessArtifactKind::ReleaseClosure),
            artifact(ReadinessArtifactKind::MaintenanceAssessment),
            artifact(ReadinessArtifactKind::OperationalLimits),
        ];
        assert_eq!(
            gate.assess(&artifacts, 1_000).status,
            OperationalReadinessStatus::Dispatchable
        );
    }

    #[test]
    fn permitted_operational_restriction_is_preserved() {
        let gate = OperationalReadinessGate::new(policy()).unwrap();
        let mut limits = artifact(ReadinessArtifactKind::OperationalLimits);
        limits.status = ReadinessArtifactStatus::Restricted;
        limits.restrictions = vec!["day-vfr-only".into()];
        let artifacts = vec![
            artifact(ReadinessArtifactKind::ReleaseClosure),
            artifact(ReadinessArtifactKind::MaintenanceAssessment),
            limits,
        ];
        let report = gate.assess(&artifacts, 1_000);
        assert_eq!(
            report.status,
            OperationalReadinessStatus::DispatchableRestricted
        );
        assert_eq!(report.active_restrictions, vec!["day-vfr-only".to_string()]);
    }

    #[test]
    fn missing_maintenance_is_incomplete() {
        let gate = OperationalReadinessGate::new(policy()).unwrap();
        let artifacts = vec![
            artifact(ReadinessArtifactKind::ReleaseClosure),
            artifact(ReadinessArtifactKind::OperationalLimits),
        ];
        assert_eq!(
            gate.assess(&artifacts, 1_000).status,
            OperationalReadinessStatus::Incomplete
        );
    }

    #[test]
    fn expired_artifact_is_no_go() {
        let gate = OperationalReadinessGate::new(policy()).unwrap();
        let artifacts = vec![
            artifact(ReadinessArtifactKind::ReleaseClosure),
            artifact(ReadinessArtifactKind::MaintenanceAssessment),
            artifact(ReadinessArtifactKind::OperationalLimits),
        ];
        assert_eq!(
            gate.assess(&artifacts, 3_000).status,
            OperationalReadinessStatus::NoGo
        );
    }
}
