// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification-readiness dossier assembly.
//!
//! The dossier is an engineering evidence gate, not a regulatory approval. It
//! binds the identities and validity of independently produced assurance
//! reports and preserves Pass, Fail, and Incomplete semantics.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DossierArtifactKind {
    HazardClosure,
    AssuranceTraceability,
    PartitionAssurance,
    TrustedIdentityTime,
    ModelValidation,
    QualificationCampaign,
    FaultRecoveryCampaign,
    RollbackDrill,
    EnduranceCampaign,
    OperationalLimits,
    ResourceBudget,
    BuildProvenance,
    DeploymentBinding,
    FleetSafetyAction,
    ReturnToService,
    ReleaseClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DossierArtifactStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DossierArtifact {
    pub artifact_id: String,
    pub kind: DossierArtifactKind,
    pub status: DossierArtifactStatus,
    pub aircraft_id: String,
    pub deployment_digest: String,
    pub software_digest: String,
    pub artifact_digest: String,
    pub produced_at_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub evidence_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DossierReviewRole {
    SystemSafety,
    FlightControls,
    SoftwareAssurance,
    Airworthiness,
    IndependentVerification,
    ReleaseAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DossierReviewApproval {
    pub role: DossierReviewRole,
    pub reviewer_id: String,
    pub approved: bool,
    pub approval_evidence_id: Option<String>,
    pub approved_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationDossierPolicy {
    pub required_artifact_kinds: BTreeSet<DossierArtifactKind>,
    pub required_review_roles: BTreeSet<DossierReviewRole>,
    pub maximum_artifact_age_ms: u64,
    pub require_unique_reviewers: bool,
    pub require_artifact_evidence: bool,
}

impl Default for CertificationDossierPolicy {
    fn default() -> Self {
        Self {
            required_artifact_kinds: BTreeSet::from([
                DossierArtifactKind::HazardClosure,
                DossierArtifactKind::AssuranceTraceability,
                DossierArtifactKind::PartitionAssurance,
                DossierArtifactKind::TrustedIdentityTime,
                DossierArtifactKind::ModelValidation,
                DossierArtifactKind::QualificationCampaign,
                DossierArtifactKind::FaultRecoveryCampaign,
                DossierArtifactKind::RollbackDrill,
                DossierArtifactKind::EnduranceCampaign,
                DossierArtifactKind::OperationalLimits,
                DossierArtifactKind::ResourceBudget,
                DossierArtifactKind::BuildProvenance,
                DossierArtifactKind::DeploymentBinding,
                DossierArtifactKind::ReleaseClosure,
            ]),
            required_review_roles: BTreeSet::from([
                DossierReviewRole::SystemSafety,
                DossierReviewRole::FlightControls,
                DossierReviewRole::SoftwareAssurance,
                DossierReviewRole::IndependentVerification,
                DossierReviewRole::ReleaseAuthority,
            ]),
            maximum_artifact_age_ms: 90 * 24 * 60 * 60 * 1_000,
            require_unique_reviewers: true,
            require_artifact_evidence: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificationDossierStatus {
    ReadyForExternalReview,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificationDossierIssue {
    MissingArtifact {
        kind: DossierArtifactKind,
    },
    DuplicateArtifact {
        kind: DossierArtifactKind,
    },
    FailedArtifact {
        kind: DossierArtifactKind,
        artifact_id: String,
    },
    IncompleteArtifact {
        kind: DossierArtifactKind,
        artifact_id: String,
    },
    EmptyArtifactDigest {
        artifact_id: String,
    },
    MissingArtifactEvidence {
        artifact_id: String,
    },
    AircraftIdentityMismatch {
        artifact_id: String,
    },
    DeploymentDigestMismatch {
        artifact_id: String,
    },
    SoftwareDigestMismatch {
        artifact_id: String,
    },
    ArtifactExpired {
        artifact_id: String,
        valid_until_ms: u64,
    },
    ArtifactTooOld {
        artifact_id: String,
        age_ms: u64,
    },
    MissingReview {
        role: DossierReviewRole,
    },
    DuplicateReview {
        role: DossierReviewRole,
    },
    ReviewRejected {
        role: DossierReviewRole,
    },
    MissingReviewEvidence {
        role: DossierReviewRole,
    },
    DuplicateReviewer {
        reviewer_id: String,
    },
    ReviewPredatesLatestArtifact {
        role: DossierReviewRole,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationDossierReport {
    pub status: CertificationDossierStatus,
    pub dossier_id: String,
    pub aircraft_id: String,
    pub deployment_digest: String,
    pub software_digest: String,
    pub accepted_artifact_ids: Vec<String>,
    pub latest_artifact_time_ms: Option<u64>,
    pub issues: Vec<CertificationDossierIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationDossierError {
    InvalidPolicy,
    EmptyDossierIdentity,
}

pub struct CertificationDossierAssembler {
    policy: CertificationDossierPolicy,
}

impl CertificationDossierAssembler {
    pub fn new(policy: CertificationDossierPolicy) -> Result<Self, CertificationDossierError> {
        if policy.maximum_artifact_age_ms == 0
            || policy.required_artifact_kinds.is_empty()
            || policy.required_review_roles.is_empty()
        {
            return Err(CertificationDossierError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        dossier_id: &str,
        aircraft_id: &str,
        deployment_digest: &str,
        software_digest: &str,
        artifacts: &[DossierArtifact],
        reviews: &[DossierReviewApproval],
        now_ms: u64,
    ) -> Result<CertificationDossierReport, CertificationDossierError> {
        if dossier_id.trim().is_empty()
            || aircraft_id.trim().is_empty()
            || deployment_digest.trim().is_empty()
            || software_digest.trim().is_empty()
        {
            return Err(CertificationDossierError::EmptyDossierIdentity);
        }

        let mut issues = Vec::new();
        let mut by_kind = BTreeMap::<DossierArtifactKind, Vec<&DossierArtifact>>::new();
        for artifact in artifacts {
            by_kind.entry(artifact.kind).or_default().push(artifact);
        }
        for kind in &self.policy.required_artifact_kinds {
            match by_kind.get(kind) {
                None => issues.push(CertificationDossierIssue::MissingArtifact { kind: *kind }),
                Some(matching) if matching.len() > 1 => {
                    issues.push(CertificationDossierIssue::DuplicateArtifact { kind: *kind })
                }
                Some(_) => {}
            }
        }

        let mut accepted_artifact_ids = Vec::new();
        let mut latest_artifact_time_ms = None;
        for artifact in artifacts {
            latest_artifact_time_ms = Some(
                latest_artifact_time_ms
                    .map(|current: u64| current.max(artifact.produced_at_ms))
                    .unwrap_or(artifact.produced_at_ms),
            );
            if artifact.aircraft_id != aircraft_id {
                issues.push(CertificationDossierIssue::AircraftIdentityMismatch {
                    artifact_id: artifact.artifact_id.clone(),
                });
            }
            if artifact.deployment_digest != deployment_digest {
                issues.push(CertificationDossierIssue::DeploymentDigestMismatch {
                    artifact_id: artifact.artifact_id.clone(),
                });
            }
            if artifact.software_digest != software_digest {
                issues.push(CertificationDossierIssue::SoftwareDigestMismatch {
                    artifact_id: artifact.artifact_id.clone(),
                });
            }
            if artifact.artifact_digest.trim().is_empty() {
                issues.push(CertificationDossierIssue::EmptyArtifactDigest {
                    artifact_id: artifact.artifact_id.clone(),
                });
            }
            if self.policy.require_artifact_evidence && artifact.evidence_ids.is_empty() {
                issues.push(CertificationDossierIssue::MissingArtifactEvidence {
                    artifact_id: artifact.artifact_id.clone(),
                });
            }
            match artifact.status {
                DossierArtifactStatus::Fail => {
                    issues.push(CertificationDossierIssue::FailedArtifact {
                        kind: artifact.kind,
                        artifact_id: artifact.artifact_id.clone(),
                    })
                }
                DossierArtifactStatus::Incomplete => {
                    issues.push(CertificationDossierIssue::IncompleteArtifact {
                        kind: artifact.kind,
                        artifact_id: artifact.artifact_id.clone(),
                    })
                }
                DossierArtifactStatus::Pass => {
                    accepted_artifact_ids.push(artifact.artifact_id.clone())
                }
            }
            if let Some(valid_until_ms) = artifact.valid_until_ms {
                if valid_until_ms < now_ms {
                    issues.push(CertificationDossierIssue::ArtifactExpired {
                        artifact_id: artifact.artifact_id.clone(),
                        valid_until_ms,
                    });
                }
            }
            let age_ms = now_ms.saturating_sub(artifact.produced_at_ms);
            if age_ms > self.policy.maximum_artifact_age_ms {
                issues.push(CertificationDossierIssue::ArtifactTooOld {
                    artifact_id: artifact.artifact_id.clone(),
                    age_ms,
                });
            }
        }

        let mut reviews_by_role = BTreeMap::<DossierReviewRole, Vec<&DossierReviewApproval>>::new();
        for review in reviews {
            reviews_by_role.entry(review.role).or_default().push(review);
        }
        for role in &self.policy.required_review_roles {
            match reviews_by_role.get(role) {
                None => issues.push(CertificationDossierIssue::MissingReview { role: *role }),
                Some(matching) if matching.len() > 1 => {
                    issues.push(CertificationDossierIssue::DuplicateReview { role: *role })
                }
                Some(_) => {}
            }
        }
        let mut reviewer_ids = BTreeSet::new();
        for review in reviews {
            if !review.approved {
                issues.push(CertificationDossierIssue::ReviewRejected { role: review.role });
            }
            if review
                .approval_evidence_id
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                issues.push(CertificationDossierIssue::MissingReviewEvidence { role: review.role });
            }
            if self.policy.require_unique_reviewers
                && !reviewer_ids.insert(review.reviewer_id.as_str())
            {
                issues.push(CertificationDossierIssue::DuplicateReviewer {
                    reviewer_id: review.reviewer_id.clone(),
                });
            }
            if latest_artifact_time_ms.is_some_and(|latest| review.approved_at_ms < latest) {
                issues.push(CertificationDossierIssue::ReviewPredatesLatestArtifact {
                    role: review.role,
                });
            }
        }

        accepted_artifact_ids.sort();
        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if issues.iter().any(issue_is_rejection) {
            CertificationDossierStatus::Rejected
        } else if issues.is_empty() {
            CertificationDossierStatus::ReadyForExternalReview
        } else {
            CertificationDossierStatus::Incomplete
        };
        Ok(CertificationDossierReport {
            status,
            dossier_id: dossier_id.into(),
            aircraft_id: aircraft_id.into(),
            deployment_digest: deployment_digest.into(),
            software_digest: software_digest.into(),
            accepted_artifact_ids,
            latest_artifact_time_ms,
            issues,
        })
    }
}

fn issue_is_rejection(issue: &CertificationDossierIssue) -> bool {
    matches!(
        issue,
        CertificationDossierIssue::DuplicateArtifact { .. }
            | CertificationDossierIssue::FailedArtifact { .. }
            | CertificationDossierIssue::AircraftIdentityMismatch { .. }
            | CertificationDossierIssue::DeploymentDigestMismatch { .. }
            | CertificationDossierIssue::SoftwareDigestMismatch { .. }
            | CertificationDossierIssue::ArtifactExpired { .. }
            | CertificationDossierIssue::ArtifactTooOld { .. }
            | CertificationDossierIssue::DuplicateReview { .. }
            | CertificationDossierIssue::ReviewRejected { .. }
            | CertificationDossierIssue::DuplicateReviewer { .. }
            | CertificationDossierIssue::ReviewPredatesLatestArtifact { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CertificationDossierPolicy {
        CertificationDossierPolicy {
            required_artifact_kinds: BTreeSet::from([DossierArtifactKind::HazardClosure]),
            required_review_roles: BTreeSet::from([DossierReviewRole::SystemSafety]),
            maximum_artifact_age_ms: 1_000,
            require_unique_reviewers: true,
            require_artifact_evidence: true,
        }
    }
    fn artifact() -> DossierArtifact {
        DossierArtifact {
            artifact_id: "hazards".into(),
            kind: DossierArtifactKind::HazardClosure,
            status: DossierArtifactStatus::Pass,
            aircraft_id: "A1".into(),
            deployment_digest: "D1".into(),
            software_digest: "S1".into(),
            artifact_digest: "sha256:a".into(),
            produced_at_ms: 100,
            valid_until_ms: Some(1_000),
            evidence_ids: BTreeSet::from(["ev".into()]),
        }
    }
    fn review() -> DossierReviewApproval {
        DossierReviewApproval {
            role: DossierReviewRole::SystemSafety,
            reviewer_id: "reviewer-1".into(),
            approved: true,
            approval_evidence_id: Some("approval".into()),
            approved_at_ms: 200,
        }
    }

    #[test]
    fn complete_dossier_is_ready_for_external_review() {
        let assembler = CertificationDossierAssembler::new(policy()).unwrap();
        let report = assembler
            .assemble("DOS-1", "A1", "D1", "S1", &[artifact()], &[review()], 300)
            .unwrap();
        assert_eq!(
            report.status,
            CertificationDossierStatus::ReadyForExternalReview
        );
    }

    #[test]
    fn failed_artifact_rejects_dossier() {
        let mut artifact = artifact();
        artifact.status = DossierArtifactStatus::Fail;
        let assembler = CertificationDossierAssembler::new(policy()).unwrap();
        let report = assembler
            .assemble("DOS-1", "A1", "D1", "S1", &[artifact], &[review()], 300)
            .unwrap();
        assert_eq!(report.status, CertificationDossierStatus::Rejected);
    }

    #[test]
    fn missing_review_is_incomplete() {
        let assembler = CertificationDossierAssembler::new(policy()).unwrap();
        let report = assembler
            .assemble("DOS-1", "A1", "D1", "S1", &[artifact()], &[], 300)
            .unwrap();
        assert_eq!(report.status, CertificationDossierStatus::Incomplete);
    }
}
