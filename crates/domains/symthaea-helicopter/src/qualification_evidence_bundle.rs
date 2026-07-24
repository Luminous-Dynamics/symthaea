// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deployment-grade qualification evidence bundle completeness checks.
//!
//! Individual logs, manifests, and reports are insufficient when they are not
//! bound into one declared campaign. This module checks artifact identity,
//! required-kind coverage, duplicate/conflicting entries, and authenticity
//! references appropriate to the operating context. Digest verification and
//! signature verification remain responsibilities of the named providers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualificationArtifactKind {
    DeploymentManifest,
    Calibration,
    ScenarioManifest,
    CampaignPlan,
    QualificationReport,
    FlightEvidence,
    RuntimeSafetyReport,
    RealtimeTimingReport,
    FaultContainmentReport,
    SafeStateReachabilityReport,
    ClaimAssessment,
    RollbackCatalog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceOperatingContext {
    Simulation,
    SoftwareInLoop,
    HardwareInLoop,
    PhysicalGroundTest,
    FlightTest,
}

impl EvidenceOperatingContext {
    fn requires_authenticity(self) -> bool {
        matches!(
            self,
            Self::HardwareInLoop | Self::PhysicalGroundTest | Self::FlightTest
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationArtifactRef {
    pub artifact_id: String,
    pub kind: QualificationArtifactKind,
    pub digest: String,
    pub schema_version: String,
    pub authenticity_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationEvidenceBundle {
    pub schema_version: String,
    pub bundle_id: String,
    pub campaign_id: String,
    pub deployment_id: String,
    pub operating_context: EvidenceOperatingContext,
    pub required_kinds: Vec<QualificationArtifactKind>,
    pub artifacts: Vec<QualificationArtifactRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceBundleStatus {
    Complete,
    Incomplete,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceBundleIssue {
    MissingKind(QualificationArtifactKind),
    DuplicateKind(QualificationArtifactKind),
    DuplicateArtifactId(String),
    InvalidDigest(String),
    EmptyIdentity,
    MissingAuthenticity(String),
    ConflictingDigest(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBundleAssessment {
    pub bundle_id: String,
    pub status: EvidenceBundleStatus,
    pub issues: Vec<EvidenceBundleIssue>,
    pub artifact_count: usize,
    pub authenticated_artifact_count: usize,
    pub canonical_digest_fnv1a64: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceBundleError {
    SerializationFailed,
}

impl QualificationEvidenceBundle {
    pub fn assess(&self) -> EvidenceBundleAssessment {
        let mut issues = Vec::new();
        if self.schema_version.trim().is_empty()
            || self.bundle_id.trim().is_empty()
            || self.campaign_id.trim().is_empty()
            || self.deployment_id.trim().is_empty()
        {
            issues.push(EvidenceBundleIssue::EmptyIdentity);
        }

        let required: BTreeSet<_> = self.required_kinds.iter().copied().collect();
        if required.len() != self.required_kinds.len() {
            for kind in &self.required_kinds {
                if self
                    .required_kinds
                    .iter()
                    .filter(|candidate| *candidate == kind)
                    .count()
                    > 1
                {
                    issues.push(EvidenceBundleIssue::DuplicateKind(*kind));
                }
            }
        }

        let mut ids = BTreeSet::new();
        let mut id_to_digest = BTreeMap::new();
        let mut kinds = BTreeMap::<QualificationArtifactKind, usize>::new();
        let mut authenticated_artifact_count = 0usize;
        for artifact in &self.artifacts {
            if artifact.artifact_id.trim().is_empty() || artifact.schema_version.trim().is_empty() {
                issues.push(EvidenceBundleIssue::EmptyIdentity);
            }
            if !ids.insert(artifact.artifact_id.clone()) {
                issues.push(EvidenceBundleIssue::DuplicateArtifactId(
                    artifact.artifact_id.clone(),
                ));
            }
            if let Some(previous) =
                id_to_digest.insert(artifact.artifact_id.clone(), artifact.digest.clone())
            {
                if previous != artifact.digest {
                    issues.push(EvidenceBundleIssue::ConflictingDigest(
                        artifact.artifact_id.clone(),
                    ));
                }
            }
            if !valid_digest(&artifact.digest) {
                issues.push(EvidenceBundleIssue::InvalidDigest(
                    artifact.artifact_id.clone(),
                ));
            }
            *kinds.entry(artifact.kind).or_default() += 1;
            if artifact
                .authenticity_reference
                .as_ref()
                .is_some_and(|reference| !reference.trim().is_empty())
            {
                authenticated_artifact_count += 1;
            }
            if self.operating_context.requires_authenticity()
                && requires_artifact_authenticity(artifact.kind)
                && artifact
                    .authenticity_reference
                    .as_ref()
                    .is_none_or(|reference| reference.trim().is_empty())
            {
                issues.push(EvidenceBundleIssue::MissingAuthenticity(
                    artifact.artifact_id.clone(),
                ));
            }
        }

        for kind in &required {
            match kinds.get(kind).copied().unwrap_or(0) {
                0 => issues.push(EvidenceBundleIssue::MissingKind(*kind)),
                1 => {}
                _ => issues.push(EvidenceBundleIssue::DuplicateKind(*kind)),
            }
        }

        let rejected = issues.iter().any(|issue| {
            matches!(
                issue,
                EvidenceBundleIssue::DuplicateArtifactId(_)
                    | EvidenceBundleIssue::InvalidDigest(_)
                    | EvidenceBundleIssue::EmptyIdentity
                    | EvidenceBundleIssue::ConflictingDigest(_)
                    | EvidenceBundleIssue::MissingAuthenticity(_)
            )
        });
        let status = if rejected {
            EvidenceBundleStatus::Rejected
        } else if issues.is_empty() {
            EvidenceBundleStatus::Complete
        } else {
            EvidenceBundleStatus::Incomplete
        };
        let canonical_digest_fnv1a64 = if rejected {
            None
        } else {
            self.digest_fnv1a64().ok()
        };

        EvidenceBundleAssessment {
            bundle_id: self.bundle_id.clone(),
            status,
            issues,
            artifact_count: self.artifacts.len(),
            authenticated_artifact_count,
            canonical_digest_fnv1a64,
        }
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, EvidenceBundleError> {
        let mut canonical = self.clone();
        canonical.required_kinds.sort();
        canonical.artifacts.sort_by(|left, right| {
            left.kind
                .cmp(&right.kind)
                .then_with(|| left.artifact_id.cmp(&right.artifact_id))
        });
        serde_json::to_vec(&canonical).map_err(|_| EvidenceBundleError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, EvidenceBundleError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

fn requires_artifact_authenticity(kind: QualificationArtifactKind) -> bool {
    matches!(
        kind,
        QualificationArtifactKind::DeploymentManifest
            | QualificationArtifactKind::Calibration
            | QualificationArtifactKind::QualificationReport
            | QualificationArtifactKind::FlightEvidence
            | QualificationArtifactKind::ClaimAssessment
            | QualificationArtifactKind::RollbackCatalog
    )
}

fn valid_digest(value: &str) -> bool {
    let Some((algorithm, digest)) = value.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && digest.len() >= 8
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(kind: QualificationArtifactKind) -> QualificationArtifactRef {
        QualificationArtifactRef {
            artifact_id: format!("{kind:?}"),
            kind,
            digest: format!("sha256:{:08x}", kind as u8 + 100),
            schema_version: "v1".into(),
            authenticity_reference: Some(format!("sig-{kind:?}")),
        }
    }

    fn complete_bundle(context: EvidenceOperatingContext) -> QualificationEvidenceBundle {
        let required_kinds = vec![
            QualificationArtifactKind::DeploymentManifest,
            QualificationArtifactKind::Calibration,
            QualificationArtifactKind::QualificationReport,
            QualificationArtifactKind::FlightEvidence,
            QualificationArtifactKind::SafeStateReachabilityReport,
            QualificationArtifactKind::ClaimAssessment,
        ];
        QualificationEvidenceBundle {
            schema_version: "symthaea-helicopter-evidence-bundle-v1".into(),
            bundle_id: "bundle-1".into(),
            campaign_id: "campaign-1".into(),
            deployment_id: "deployment-1".into(),
            operating_context: context,
            artifacts: required_kinds.iter().copied().map(artifact).collect(),
            required_kinds,
        }
    }

    #[test]
    fn complete_physical_bundle_passes() {
        let assessment = complete_bundle(EvidenceOperatingContext::PhysicalGroundTest).assess();
        assert_eq!(assessment.status, EvidenceBundleStatus::Complete);
        assert!(assessment.canonical_digest_fnv1a64.is_some());
    }

    #[test]
    fn missing_required_kind_is_incomplete() {
        let mut bundle = complete_bundle(EvidenceOperatingContext::Simulation);
        bundle.artifacts.pop();
        let assessment = bundle.assess();
        assert_eq!(assessment.status, EvidenceBundleStatus::Incomplete);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| matches!(issue, EvidenceBundleIssue::MissingKind(_)))
        );
    }

    #[test]
    fn physical_evidence_without_authenticity_is_rejected() {
        let mut bundle = complete_bundle(EvidenceOperatingContext::FlightTest);
        bundle.artifacts[0].authenticity_reference = None;
        let assessment = bundle.assess();
        assert_eq!(assessment.status, EvidenceBundleStatus::Rejected);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| matches!(issue, EvidenceBundleIssue::MissingAuthenticity(_)))
        );
    }

    #[test]
    fn canonical_digest_is_order_independent() {
        let first = complete_bundle(EvidenceOperatingContext::Simulation);
        let mut second = first.clone();
        second.artifacts.reverse();
        second.required_kinds.reverse();
        assert_eq!(
            first.digest_fnv1a64().unwrap(),
            second.digest_fnv1a64().unwrap()
        );
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let mut bundle = complete_bundle(EvidenceOperatingContext::Simulation);
        bundle.artifacts[0].digest = "not-a-digest".into();
        assert_eq!(bundle.assess().status, EvidenceBundleStatus::Rejected);
    }
}
