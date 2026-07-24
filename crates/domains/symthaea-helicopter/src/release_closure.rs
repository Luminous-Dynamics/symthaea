// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-evaluable release-closure gate.
//!
//! A release is not closed merely because individual checks exist. This module
//! binds the required build, qualification, model, timing, resource, evidence,
//! traceability, controllability, observability, and deployment artifacts to one
//! deployment identity and returns explicit Pass, Fail, or Incomplete semantics.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReleaseArtifactKind {
    BuildVerification,
    TestVerification,
    LintVerification,
    DeploymentManifest,
    CalibrationAssessment,
    ModelValidation,
    QualificationReport,
    RealtimeTimingReport,
    ResourceBudgetReport,
    FaultContainmentReport,
    CommonCauseReport,
    ControllabilityReport,
    ObservabilityReport,
    SafeStateReachabilityReport,
    EvidenceRetentionReport,
    RandomStreamManifest,
    QualificationEvidenceBundle,
    AssuranceTraceabilityReport,
    ClaimLedgerAssessment,
    RollbackAssessment,
    UncertaintyBudgetReport,
    FaultRecoveryCampaignReport,
    EnergyGuidanceReport,
    OperationalLimitsReport,
    RuntimeAssuranceReport,
    AdaptiveUpdateReport,
    FlightDataGovernanceReport,
    BuildProvenanceReport,
    BuildReproducibilityReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseArtifactStatus {
    Verified,
    Failed,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArtifact {
    pub artifact_id: String,
    pub deployment_id: String,
    pub kind: ReleaseArtifactKind,
    pub digest: String,
    pub status: ReleaseArtifactStatus,
    pub authenticity_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseClosurePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub deployment_id: String,
    pub required_kinds: Vec<ReleaseArtifactKind>,
    pub authenticity_required_for: Vec<ReleaseArtifactKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseClosureStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseClosureIssue {
    EmptyIdentity,
    DuplicateRequiredKind(ReleaseArtifactKind),
    DuplicateArtifactId(String),
    DuplicateArtifactKind(ReleaseArtifactKind),
    MissingArtifact(ReleaseArtifactKind),
    InvalidDigest(String),
    DeploymentMismatch(String),
    MissingAuthenticity(String),
    ArtifactFailed(String),
    ArtifactIncomplete(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseClosureReport {
    pub schema_version: String,
    pub policy_id: String,
    pub deployment_id: String,
    pub status: ReleaseClosureStatus,
    pub issues: Vec<ReleaseClosureIssue>,
    pub verified_kinds: Vec<ReleaseArtifactKind>,
    pub artifact_count: usize,
}

impl ReleaseClosureReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, ReleaseClosureError> {
        let mut canonical = self.clone();
        canonical.verified_kinds.sort();
        canonical.issues.sort_by_key(issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| ReleaseClosureError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, ReleaseClosureError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseClosureError {
    InvalidPolicy,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct ReleaseClosureGate {
    policy: ReleaseClosurePolicy,
}

impl ReleaseClosureGate {
    pub fn new(policy: ReleaseClosurePolicy) -> Result<Self, ReleaseClosureError> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.deployment_id.trim().is_empty()
            || policy.required_kinds.is_empty()
        {
            return Err(ReleaseClosureError::InvalidPolicy);
        }
        let required: BTreeSet<_> = policy.required_kinds.iter().copied().collect();
        if required.len() != policy.required_kinds.len()
            || policy
                .authenticity_required_for
                .iter()
                .any(|kind| !required.contains(kind))
        {
            return Err(ReleaseClosureError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, artifacts: &[ReleaseArtifact]) -> ReleaseClosureReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut by_kind = BTreeMap::<ReleaseArtifactKind, &ReleaseArtifact>::new();
        for artifact in artifacts {
            if artifact.artifact_id.trim().is_empty() || artifact.deployment_id.trim().is_empty() {
                issues.push(ReleaseClosureIssue::EmptyIdentity);
            }
            if !ids.insert(artifact.artifact_id.clone()) {
                issues.push(ReleaseClosureIssue::DuplicateArtifactId(
                    artifact.artifact_id.clone(),
                ));
            }
            if by_kind.insert(artifact.kind, artifact).is_some() {
                issues.push(ReleaseClosureIssue::DuplicateArtifactKind(artifact.kind));
            }
            if !valid_digest(&artifact.digest) {
                issues.push(ReleaseClosureIssue::InvalidDigest(
                    artifact.artifact_id.clone(),
                ));
            }
            if artifact.deployment_id != self.policy.deployment_id {
                issues.push(ReleaseClosureIssue::DeploymentMismatch(
                    artifact.artifact_id.clone(),
                ));
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
                issues.push(ReleaseClosureIssue::MissingAuthenticity(
                    artifact.artifact_id.clone(),
                ));
            }
            match artifact.status {
                ReleaseArtifactStatus::Verified => {}
                ReleaseArtifactStatus::Failed => issues.push(ReleaseClosureIssue::ArtifactFailed(
                    artifact.artifact_id.clone(),
                )),
                ReleaseArtifactStatus::Incomplete => issues.push(
                    ReleaseClosureIssue::ArtifactIncomplete(artifact.artifact_id.clone()),
                ),
            }
        }

        for kind in &self.policy.required_kinds {
            if !by_kind.contains_key(kind) {
                issues.push(ReleaseClosureIssue::MissingArtifact(*kind));
            }
        }

        let hard_failure = issues.iter().any(|issue| {
            matches!(
                issue,
                ReleaseClosureIssue::EmptyIdentity
                    | ReleaseClosureIssue::DuplicateArtifactId(_)
                    | ReleaseClosureIssue::DuplicateArtifactKind(_)
                    | ReleaseClosureIssue::InvalidDigest(_)
                    | ReleaseClosureIssue::DeploymentMismatch(_)
                    | ReleaseClosureIssue::MissingAuthenticity(_)
                    | ReleaseClosureIssue::ArtifactFailed(_)
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                ReleaseClosureIssue::MissingArtifact(_)
                    | ReleaseClosureIssue::ArtifactIncomplete(_)
            )
        });
        let status = if hard_failure {
            ReleaseClosureStatus::Fail
        } else if incomplete {
            ReleaseClosureStatus::Incomplete
        } else {
            ReleaseClosureStatus::Pass
        };
        let mut verified_kinds: Vec<_> = by_kind
            .values()
            .filter(|artifact| artifact.status == ReleaseArtifactStatus::Verified)
            .map(|artifact| artifact.kind)
            .collect();
        verified_kinds.sort();
        verified_kinds.dedup();

        ReleaseClosureReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            deployment_id: self.policy.deployment_id.clone(),
            status,
            issues,
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

fn issue_sort_key(issue: &ReleaseClosureIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> ReleaseClosureGate {
        ReleaseClosureGate::new(ReleaseClosurePolicy {
            schema_version: "1".into(),
            policy_id: "release-policy".into(),
            deployment_id: "aircraft-001".into(),
            required_kinds: vec![
                ReleaseArtifactKind::BuildVerification,
                ReleaseArtifactKind::QualificationReport,
                ReleaseArtifactKind::QualificationEvidenceBundle,
            ],
            authenticity_required_for: vec![ReleaseArtifactKind::QualificationEvidenceBundle],
        })
        .unwrap()
    }

    fn artifact(
        id: &str,
        kind: ReleaseArtifactKind,
        status: ReleaseArtifactStatus,
    ) -> ReleaseArtifact {
        ReleaseArtifact {
            artifact_id: id.into(),
            deployment_id: "aircraft-001".into(),
            kind,
            digest: "sha256:0123456789abcdef".into(),
            status,
            authenticity_reference: if kind == ReleaseArtifactKind::QualificationEvidenceBundle {
                Some("sig:bundle-001".into())
            } else {
                None
            },
        }
    }

    #[test]
    fn complete_verified_set_passes() {
        let report = gate().assess(&[
            artifact(
                "build",
                ReleaseArtifactKind::BuildVerification,
                ReleaseArtifactStatus::Verified,
            ),
            artifact(
                "qualification",
                ReleaseArtifactKind::QualificationReport,
                ReleaseArtifactStatus::Verified,
            ),
            artifact(
                "bundle",
                ReleaseArtifactKind::QualificationEvidenceBundle,
                ReleaseArtifactStatus::Verified,
            ),
        ]);
        assert_eq!(report.status, ReleaseClosureStatus::Pass);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn missing_artifact_is_incomplete() {
        let report = gate().assess(&[
            artifact(
                "build",
                ReleaseArtifactKind::BuildVerification,
                ReleaseArtifactStatus::Verified,
            ),
            artifact(
                "qualification",
                ReleaseArtifactKind::QualificationReport,
                ReleaseArtifactStatus::Verified,
            ),
        ]);
        assert_eq!(report.status, ReleaseClosureStatus::Incomplete);
    }

    #[test]
    fn known_failure_dominates_missing_evidence() {
        let report = gate().assess(&[artifact(
            "build",
            ReleaseArtifactKind::BuildVerification,
            ReleaseArtifactStatus::Failed,
        )]);
        assert_eq!(report.status, ReleaseClosureStatus::Fail);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            ReleaseClosureIssue::ArtifactFailed(id) if id == "build"
        )));
    }

    #[test]
    fn authenticity_is_required_when_policy_declares_it() {
        let mut bundle = artifact(
            "bundle",
            ReleaseArtifactKind::QualificationEvidenceBundle,
            ReleaseArtifactStatus::Verified,
        );
        bundle.authenticity_reference = None;
        let report = gate().assess(&[
            artifact(
                "build",
                ReleaseArtifactKind::BuildVerification,
                ReleaseArtifactStatus::Verified,
            ),
            artifact(
                "qualification",
                ReleaseArtifactKind::QualificationReport,
                ReleaseArtifactStatus::Verified,
            ),
            bundle,
        ]);
        assert_eq!(report.status, ReleaseClosureStatus::Fail);
    }
}
