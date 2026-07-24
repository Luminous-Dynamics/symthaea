// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent release authorization.
//!
//! A qualified build is not self-authorizing. This module binds a release
//! candidate to its required assurance artifacts and enforces separation of
//! duties across engineering, safety, security, operations, and independent
//! verification roles.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReleaseAuthorizationRole {
    Engineering,
    Safety,
    Security,
    Operations,
    IndependentVerifier,
    ReleaseAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseApprovalDecision {
    Approve,
    Reject,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseApproval {
    pub approval_id: String,
    pub approver_id: String,
    pub organization_id: String,
    pub role: ReleaseAuthorizationRole,
    pub decision: ReleaseApprovalDecision,
    pub candidate_digest: String,
    pub approved_at_ms: u64,
    pub expires_at_ms: u64,
    pub authenticity_reference: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReleaseEvidenceKind {
    BuildProvenance,
    Qualification,
    HazardClosure,
    SafetyCaseMaintenance,
    IndependentVerification,
    Cybersecurity,
    OperationalReadiness,
    RollbackDrill,
    EnduranceCampaign,
    CertificationDossier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseEvidenceStatus {
    Verified,
    Failed,
    Missing,
    Restricted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseAuthorizationEvidence {
    pub evidence_id: String,
    pub kind: ReleaseEvidenceKind,
    pub status: ReleaseEvidenceStatus,
    pub candidate_digest: String,
    pub artifact_digest: String,
    pub issued_at_ms: u64,
    pub valid_until_ms: u64,
    pub authenticity_reference: Option<String>,
    pub restrictions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseCandidate {
    pub release_id: String,
    pub candidate_digest: String,
    pub source_digest: String,
    pub deployment_digest: String,
    pub aircraft_class: String,
    pub created_at_ms: u64,
    pub author_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentReleasePolicy {
    pub required_roles: BTreeSet<ReleaseAuthorizationRole>,
    pub required_evidence: BTreeSet<ReleaseEvidenceKind>,
    pub authenticity_required_for: BTreeSet<ReleaseEvidenceKind>,
    pub restriction_permitted_for: BTreeSet<ReleaseEvidenceKind>,
    pub minimum_organizations: usize,
    pub maximum_approval_age_ms: u64,
    pub prohibit_author_approval: bool,
    pub require_release_authority_from_independent_org: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependentReleaseStatus {
    Authorized,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependentReleaseIssue {
    EmptyIdentity,
    DuplicateApproval(String),
    DuplicateEvidenceKind(ReleaseEvidenceKind),
    CandidateDigestMismatch(String),
    MissingRole(ReleaseAuthorizationRole),
    MissingEvidence(ReleaseEvidenceKind),
    ApprovalRejected(String),
    ApprovalAbstained(String),
    ApprovalExpired(String),
    ApprovalTooOld(String),
    ApprovalNotYetValid(String),
    MissingApprovalAuthenticity(String),
    AuthorApprovedOwnRelease(String),
    InsufficientOrganizations { required: usize, observed: usize },
    ReleaseAuthorityNotIndependent,
    EvidenceFailed(String),
    EvidenceMissing(String),
    EvidenceExpired(String),
    EvidenceNotYetValid(String),
    MissingEvidenceAuthenticity(String),
    RestrictionNotPermitted(String),
    EmptyRestriction(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentReleaseReport {
    pub status: IndependentReleaseStatus,
    pub release_id: String,
    pub candidate_digest: String,
    pub verified_evidence: Vec<ReleaseEvidenceKind>,
    pub approving_roles: Vec<ReleaseAuthorizationRole>,
    pub active_restrictions: Vec<String>,
    pub issues: Vec<IndependentReleaseIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependentReleaseError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct IndependentReleaseGate {
    policy: IndependentReleasePolicy,
}

impl IndependentReleaseGate {
    pub fn new(policy: IndependentReleasePolicy) -> Result<Self, IndependentReleaseError> {
        if policy.required_roles.is_empty()
            || policy.required_evidence.is_empty()
            || policy.minimum_organizations < 2
            || policy.maximum_approval_age_ms == 0
            || !policy
                .authenticity_required_for
                .is_subset(&policy.required_evidence)
            || !policy
                .restriction_permitted_for
                .is_subset(&policy.required_evidence)
        {
            return Err(IndependentReleaseError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        candidate: &ReleaseCandidate,
        evidence: &[ReleaseAuthorizationEvidence],
        approvals: &[ReleaseApproval],
        now_ms: u64,
    ) -> IndependentReleaseReport {
        let mut issues = Vec::new();
        if [
            candidate.release_id.as_str(),
            candidate.candidate_digest.as_str(),
            candidate.source_digest.as_str(),
            candidate.deployment_digest.as_str(),
            candidate.aircraft_class.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            issues.push(IndependentReleaseIssue::EmptyIdentity);
        }

        let mut by_kind = BTreeMap::new();
        let mut verified_evidence = Vec::new();
        let mut restrictions = Vec::new();
        for artifact in evidence {
            if artifact.evidence_id.trim().is_empty() || artifact.artifact_digest.trim().is_empty()
            {
                issues.push(IndependentReleaseIssue::EmptyIdentity);
            }
            if by_kind.insert(artifact.kind, artifact).is_some() {
                issues.push(IndependentReleaseIssue::DuplicateEvidenceKind(
                    artifact.kind,
                ));
            }
            if artifact.candidate_digest != candidate.candidate_digest {
                issues.push(IndependentReleaseIssue::CandidateDigestMismatch(
                    artifact.evidence_id.clone(),
                ));
            }
            if now_ms < artifact.issued_at_ms {
                issues.push(IndependentReleaseIssue::EvidenceNotYetValid(
                    artifact.evidence_id.clone(),
                ));
            }
            if now_ms > artifact.valid_until_ms {
                issues.push(IndependentReleaseIssue::EvidenceExpired(
                    artifact.evidence_id.clone(),
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
                issues.push(IndependentReleaseIssue::MissingEvidenceAuthenticity(
                    artifact.evidence_id.clone(),
                ));
            }
            match artifact.status {
                ReleaseEvidenceStatus::Verified => verified_evidence.push(artifact.kind),
                ReleaseEvidenceStatus::Failed => issues.push(
                    IndependentReleaseIssue::EvidenceFailed(artifact.evidence_id.clone()),
                ),
                ReleaseEvidenceStatus::Missing => issues.push(
                    IndependentReleaseIssue::EvidenceMissing(artifact.evidence_id.clone()),
                ),
                ReleaseEvidenceStatus::Restricted => {
                    if !self
                        .policy
                        .restriction_permitted_for
                        .contains(&artifact.kind)
                    {
                        issues.push(IndependentReleaseIssue::RestrictionNotPermitted(
                            artifact.evidence_id.clone(),
                        ));
                    }
                    if artifact.restrictions.is_empty()
                        || artifact
                            .restrictions
                            .iter()
                            .any(|restriction| restriction.trim().is_empty())
                    {
                        issues.push(IndependentReleaseIssue::EmptyRestriction(
                            artifact.evidence_id.clone(),
                        ));
                    } else {
                        restrictions.extend(artifact.restrictions.iter().cloned());
                        verified_evidence.push(artifact.kind);
                    }
                }
            }
        }
        for kind in &self.policy.required_evidence {
            if !by_kind.contains_key(kind) {
                issues.push(IndependentReleaseIssue::MissingEvidence(*kind));
            }
        }

        let mut approval_ids = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        let mut release_authority_orgs = BTreeSet::new();
        let mut non_release_authority_orgs = BTreeSet::new();
        for approval in approvals {
            if approval.approval_id.trim().is_empty()
                || approval.approver_id.trim().is_empty()
                || approval.organization_id.trim().is_empty()
                || approval.authenticity_reference.trim().is_empty()
            {
                issues.push(IndependentReleaseIssue::EmptyIdentity);
            }
            if !approval_ids.insert(approval.approval_id.as_str()) {
                issues.push(IndependentReleaseIssue::DuplicateApproval(
                    approval.approval_id.clone(),
                ));
            }
            if approval.candidate_digest != candidate.candidate_digest {
                issues.push(IndependentReleaseIssue::CandidateDigestMismatch(
                    approval.approval_id.clone(),
                ));
            }
            if now_ms < approval.approved_at_ms {
                issues.push(IndependentReleaseIssue::ApprovalNotYetValid(
                    approval.approval_id.clone(),
                ));
            }
            if now_ms > approval.expires_at_ms {
                issues.push(IndependentReleaseIssue::ApprovalExpired(
                    approval.approval_id.clone(),
                ));
            }
            if now_ms.saturating_sub(approval.approved_at_ms) > self.policy.maximum_approval_age_ms
            {
                issues.push(IndependentReleaseIssue::ApprovalTooOld(
                    approval.approval_id.clone(),
                ));
            }
            if approval.authenticity_reference.trim().is_empty() {
                issues.push(IndependentReleaseIssue::MissingApprovalAuthenticity(
                    approval.approval_id.clone(),
                ));
            }
            if self.policy.prohibit_author_approval
                && candidate.author_ids.contains(&approval.approver_id)
            {
                issues.push(IndependentReleaseIssue::AuthorApprovedOwnRelease(
                    approval.approver_id.clone(),
                ));
            }
            match approval.decision {
                ReleaseApprovalDecision::Approve => {
                    roles.insert(approval.role);
                    organizations.insert(approval.organization_id.as_str());
                    if approval.role == ReleaseAuthorizationRole::ReleaseAuthority {
                        release_authority_orgs.insert(approval.organization_id.as_str());
                    } else {
                        non_release_authority_orgs.insert(approval.organization_id.as_str());
                    }
                }
                ReleaseApprovalDecision::Reject => issues.push(
                    IndependentReleaseIssue::ApprovalRejected(approval.approval_id.clone()),
                ),
                ReleaseApprovalDecision::Abstain => issues.push(
                    IndependentReleaseIssue::ApprovalAbstained(approval.approval_id.clone()),
                ),
            }
        }
        for role in &self.policy.required_roles {
            if !roles.contains(role) {
                issues.push(IndependentReleaseIssue::MissingRole(*role));
            }
        }
        if organizations.len() < self.policy.minimum_organizations {
            issues.push(IndependentReleaseIssue::InsufficientOrganizations {
                required: self.policy.minimum_organizations,
                observed: organizations.len(),
            });
        }
        if self.policy.require_release_authority_from_independent_org
            && release_authority_orgs
                .iter()
                .any(|organization| non_release_authority_orgs.contains(organization))
        {
            issues.push(IndependentReleaseIssue::ReleaseAuthorityNotIndependent);
        }

        verified_evidence.sort();
        verified_evidence.dedup();
        let mut approving_roles = roles.into_iter().collect::<Vec<_>>();
        approving_roles.sort();
        restrictions.sort();
        restrictions.dedup();

        let status = if issues.iter().any(is_rejection) {
            IndependentReleaseStatus::Rejected
        } else if issues.is_empty() {
            IndependentReleaseStatus::Authorized
        } else {
            IndependentReleaseStatus::Incomplete
        };
        IndependentReleaseReport {
            status,
            release_id: candidate.release_id.clone(),
            candidate_digest: candidate.candidate_digest.clone(),
            verified_evidence,
            approving_roles,
            active_restrictions: restrictions,
            issues,
        }
    }
}

fn is_rejection(issue: &IndependentReleaseIssue) -> bool {
    matches!(
        issue,
        IndependentReleaseIssue::CandidateDigestMismatch(_)
            | IndependentReleaseIssue::ApprovalRejected(_)
            | IndependentReleaseIssue::ApprovalExpired(_)
            | IndependentReleaseIssue::ApprovalTooOld(_)
            | IndependentReleaseIssue::AuthorApprovedOwnRelease(_)
            | IndependentReleaseIssue::ReleaseAuthorityNotIndependent
            | IndependentReleaseIssue::EvidenceFailed(_)
            | IndependentReleaseIssue::EvidenceExpired(_)
            | IndependentReleaseIssue::RestrictionNotPermitted(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> IndependentReleasePolicy {
        IndependentReleasePolicy {
            required_roles: BTreeSet::from([
                ReleaseAuthorizationRole::Safety,
                ReleaseAuthorizationRole::IndependentVerifier,
                ReleaseAuthorizationRole::ReleaseAuthority,
            ]),
            required_evidence: BTreeSet::from([
                ReleaseEvidenceKind::BuildProvenance,
                ReleaseEvidenceKind::HazardClosure,
                ReleaseEvidenceKind::IndependentVerification,
            ]),
            authenticity_required_for: BTreeSet::from([
                ReleaseEvidenceKind::BuildProvenance,
                ReleaseEvidenceKind::IndependentVerification,
            ]),
            restriction_permitted_for: BTreeSet::new(),
            minimum_organizations: 3,
            maximum_approval_age_ms: 2_000,
            prohibit_author_approval: true,
            require_release_authority_from_independent_org: true,
        }
    }

    fn candidate() -> ReleaseCandidate {
        ReleaseCandidate {
            release_id: "release-102".into(),
            candidate_digest: "sha256:candidate".into(),
            source_digest: "sha256:source".into(),
            deployment_digest: "sha256:deployment".into(),
            aircraft_class: "sar-helicopter".into(),
            created_at_ms: 1_000,
            author_ids: BTreeSet::from(["author".into()]),
        }
    }

    fn evidence(kind: ReleaseEvidenceKind) -> ReleaseAuthorizationEvidence {
        ReleaseAuthorizationEvidence {
            evidence_id: format!("evidence-{kind:?}"),
            kind,
            status: ReleaseEvidenceStatus::Verified,
            candidate_digest: "sha256:candidate".into(),
            artifact_digest: format!("sha256:{kind:?}"),
            issued_at_ms: 1_000,
            valid_until_ms: 5_000,
            authenticity_reference: Some(format!("sig:{kind:?}")),
            restrictions: Vec::new(),
        }
    }

    fn approval(id: &str, org: &str, role: ReleaseAuthorizationRole) -> ReleaseApproval {
        ReleaseApproval {
            approval_id: format!("approval-{id}"),
            approver_id: id.into(),
            organization_id: org.into(),
            role,
            decision: ReleaseApprovalDecision::Approve,
            candidate_digest: "sha256:candidate".into(),
            approved_at_ms: 1_500,
            expires_at_ms: 4_000,
            authenticity_reference: format!("sig:{id}"),
        }
    }

    #[test]
    fn separated_release_authority_passes() {
        let report = IndependentReleaseGate::new(policy()).unwrap().assess(
            &candidate(),
            &[
                evidence(ReleaseEvidenceKind::BuildProvenance),
                evidence(ReleaseEvidenceKind::HazardClosure),
                evidence(ReleaseEvidenceKind::IndependentVerification),
            ],
            &[
                approval("safety", "operator", ReleaseAuthorizationRole::Safety),
                approval(
                    "verifier",
                    "lab",
                    ReleaseAuthorizationRole::IndependentVerifier,
                ),
                approval(
                    "authority",
                    "authority",
                    ReleaseAuthorizationRole::ReleaseAuthority,
                ),
            ],
            2_000,
        );
        assert_eq!(report.status, IndependentReleaseStatus::Authorized);
    }

    #[test]
    fn author_self_approval_is_rejected() {
        let mut approvals = vec![
            approval("author", "operator", ReleaseAuthorizationRole::Safety),
            approval(
                "verifier",
                "lab",
                ReleaseAuthorizationRole::IndependentVerifier,
            ),
            approval(
                "authority",
                "authority",
                ReleaseAuthorizationRole::ReleaseAuthority,
            ),
        ];
        approvals[0].approver_id = "author".into();
        let report = IndependentReleaseGate::new(policy()).unwrap().assess(
            &candidate(),
            &[
                evidence(ReleaseEvidenceKind::BuildProvenance),
                evidence(ReleaseEvidenceKind::HazardClosure),
                evidence(ReleaseEvidenceKind::IndependentVerification),
            ],
            &approvals,
            2_000,
        );
        assert_eq!(report.status, IndependentReleaseStatus::Rejected);
    }
}
