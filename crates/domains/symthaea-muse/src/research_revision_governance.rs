// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance for model, protocol, and claim-relevant revisions after publication.
//!
//! A revision creates a new lineage node. It may never overwrite the source
//! confirmatory release or retroactively change the meaning of published evidence.

use crate::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, confirmatory_final_release_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RESEARCH_REVISION_GOVERNANCE_VERSION: &str =
    "symthaea-muse-research-revision-governance-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchRevisionClass {
    DocumentationOnly,
    DefectCorrection,
    ModelRevision,
    ProtocolRevision,
    OutcomeDefinitionChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevisionClaimImpact {
    NoClaimImpact,
    RequiresReanalysis,
    RequiresIndependentReplication,
    RequiresNewConfirmation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RevisionEvidenceRole {
    SourceDiff,
    BuildValidation,
    RegressionSuite,
    RootCauseAnalysis,
    MigrationPlan,
    FrozenModelCheckpoint,
    TrainingCorpus,
    HoldoutEvaluation,
    ExternalReview,
    StatisticalReview,
    HumanStudyReview,
    Preregistration,
    SyntheticDryRun,
    NewConfirmatoryPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionEvidenceBinding {
    pub role: RevisionEvidenceRole,
    pub sha256: String,
    pub public_uri: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RevisionApprovalRole {
    Maintainer,
    ReproducibilityReviewer,
    StatisticalReviewer,
    MusicTheoryReviewer,
    HumanStudyReviewer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionApproval {
    pub role: RevisionApprovalRole,
    pub reviewer_id: String,
    pub organization_id: String,
    pub independent_of_authors: bool,
    pub conflict_declaration: String,
    pub decision_sha256: String,
    pub approved_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchRevisionProposal {
    pub governance_version: String,
    pub revision_id: String,
    pub parent_final_release_sha256: String,
    pub parent_source_revision: String,
    pub proposed_semantic_version: String,
    pub revision_class: ResearchRevisionClass,
    pub claim_impact: RevisionClaimImpact,
    pub summary: String,
    pub changed_modules: Vec<String>,
    pub evidence: Vec<RevisionEvidenceBinding>,
    pub approvals: Vec<RevisionApproval>,
    pub replaces_published_evidence: bool,
    pub requires_new_release_lineage: bool,
    pub proposed_at_utc: String,
    pub proposal_sha256: String,
}

#[derive(Serialize)]
struct RevisionCommitment<'a> {
    governance_version: &'a str,
    revision_id: &'a str,
    parent_final_release_sha256: &'a str,
    parent_source_revision: &'a str,
    proposed_semantic_version: &'a str,
    revision_class: ResearchRevisionClass,
    claim_impact: RevisionClaimImpact,
    summary: &'a str,
    changed_modules: &'a [String],
    evidence: &'a [RevisionEvidenceBinding],
    approvals: &'a [RevisionApproval],
    replaces_published_evidence: bool,
    requires_new_release_lineage: bool,
    proposed_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchRevisionIssue {
    WrongVersion { found: String },
    InvalidParentRelease,
    ParentMismatch,
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidSemanticVersion,
    DuplicateChangedModule { module: String },
    DuplicateEvidenceRole { role: RevisionEvidenceRole },
    MissingEvidenceRole { role: RevisionEvidenceRole },
    DuplicateApprovalRole { role: RevisionApprovalRole },
    MissingApprovalRole { role: RevisionApprovalRole },
    ReviewerNotIndependent { role: RevisionApprovalRole },
    EmptyConflictDeclaration { role: RevisionApprovalRole },
    ClaimImpactTooWeak,
    AttemptsToReplacePublishedEvidence,
    NewLineageNotRequired,
    SerializationFailed,
    ProposalDigestMismatch,
}

pub fn research_revision_commitment(
    proposal: &ResearchRevisionProposal,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RevisionCommitment {
        governance_version: &proposal.governance_version,
        revision_id: &proposal.revision_id,
        parent_final_release_sha256: &proposal.parent_final_release_sha256,
        parent_source_revision: &proposal.parent_source_revision,
        proposed_semantic_version: &proposal.proposed_semantic_version,
        revision_class: proposal.revision_class,
        claim_impact: proposal.claim_impact,
        summary: &proposal.summary,
        changed_modules: &proposal.changed_modules,
        evidence: &proposal.evidence,
        approvals: &proposal.approvals,
        replaces_published_evidence: proposal.replaces_published_evidence,
        requires_new_release_lineage: proposal.requires_new_release_lineage,
        proposed_at_utc: &proposal.proposed_at_utc,
    })
}

pub fn seal_research_revision_proposal(
    parent: &ConfirmatoryFinalReleaseBundle,
    proposal: &mut ResearchRevisionProposal,
) -> Result<(), Vec<ResearchRevisionIssue>> {
    proposal.changed_modules.sort();
    proposal.changed_modules.dedup();
    proposal.evidence.sort_by_key(|binding| binding.role);
    proposal.approvals.sort_by_key(|approval| approval.role);
    proposal.proposal_sha256 = research_revision_commitment(proposal)
        .map_err(|_| vec![ResearchRevisionIssue::SerializationFailed])?;
    let issues = validate_research_revision_proposal(parent, proposal);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_research_revision_proposal(
    parent: &ConfirmatoryFinalReleaseBundle,
    proposal: &ResearchRevisionProposal,
) -> Vec<ResearchRevisionIssue> {
    let mut issues = Vec::new();
    if proposal.governance_version != RESEARCH_REVISION_GOVERNANCE_VERSION {
        issues.push(ResearchRevisionIssue::WrongVersion {
            found: proposal.governance_version.clone(),
        });
    }
    match confirmatory_final_release_commitment(parent) {
        Ok(digest) if digest == parent.bundle_sha256 => {}
        _ => issues.push(ResearchRevisionIssue::InvalidParentRelease),
    }
    if proposal.parent_final_release_sha256 != parent.bundle_sha256
        || proposal.parent_source_revision != parent.source_revision
    {
        issues.push(ResearchRevisionIssue::ParentMismatch);
    }
    for (field, value) in [
        ("revision_id", proposal.revision_id.as_str()),
        (
            "parent_source_revision",
            proposal.parent_source_revision.as_str(),
        ),
        ("summary", proposal.summary.as_str()),
        ("proposed_at_utc", proposal.proposed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ResearchRevisionIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "parent_final_release_sha256",
            proposal.parent_final_release_sha256.as_str(),
        ),
        ("proposal_sha256", proposal.proposal_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ResearchRevisionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if !valid_semantic_version(&proposal.proposed_semantic_version) {
        issues.push(ResearchRevisionIssue::InvalidSemanticVersion);
    }
    validate_modules(proposal, &mut issues);
    validate_evidence(proposal, &mut issues);
    validate_approvals(proposal, &mut issues);
    if !claim_impact_sufficient(proposal.revision_class, proposal.claim_impact) {
        issues.push(ResearchRevisionIssue::ClaimImpactTooWeak);
    }
    if proposal.replaces_published_evidence {
        issues.push(ResearchRevisionIssue::AttemptsToReplacePublishedEvidence);
    }
    if !proposal.requires_new_release_lineage {
        issues.push(ResearchRevisionIssue::NewLineageNotRequired);
    }
    match research_revision_commitment(proposal) {
        Ok(digest) if digest == proposal.proposal_sha256 => {}
        Ok(_) => issues.push(ResearchRevisionIssue::ProposalDigestMismatch),
        Err(_) => issues.push(ResearchRevisionIssue::SerializationFailed),
    }
    issues
}

fn validate_modules(proposal: &ResearchRevisionProposal, issues: &mut Vec<ResearchRevisionIssue>) {
    let mut modules = BTreeSet::new();
    for module in &proposal.changed_modules {
        if module.trim().is_empty() {
            issues.push(ResearchRevisionIssue::EmptyField {
                field: "changed_modules".into(),
            });
        } else if !modules.insert(module.as_str()) {
            issues.push(ResearchRevisionIssue::DuplicateChangedModule {
                module: module.clone(),
            });
        }
    }
}

fn validate_evidence(proposal: &ResearchRevisionProposal, issues: &mut Vec<ResearchRevisionIssue>) {
    let mut roles = BTreeSet::new();
    for binding in &proposal.evidence {
        if !roles.insert(binding.role) {
            issues.push(ResearchRevisionIssue::DuplicateEvidenceRole { role: binding.role });
        }
        if !is_sha256(&binding.sha256) {
            issues.push(ResearchRevisionIssue::InvalidDigest {
                field: format!("evidence.{:?}.sha256", binding.role),
            });
        }
        if binding.public_uri.trim().is_empty() {
            issues.push(ResearchRevisionIssue::EmptyField {
                field: format!("evidence.{:?}.public_uri", binding.role),
            });
        }
    }
    for role in required_evidence_roles(proposal.revision_class) {
        if !roles.contains(&role) {
            issues.push(ResearchRevisionIssue::MissingEvidenceRole { role });
        }
    }
}

fn validate_approvals(
    proposal: &ResearchRevisionProposal,
    issues: &mut Vec<ResearchRevisionIssue>,
) {
    let mut roles = BTreeSet::new();
    for approval in &proposal.approvals {
        if !roles.insert(approval.role) {
            issues.push(ResearchRevisionIssue::DuplicateApprovalRole {
                role: approval.role,
            });
        }
        for (field, value) in [
            ("reviewer_id", approval.reviewer_id.as_str()),
            ("organization_id", approval.organization_id.as_str()),
            ("approved_at_utc", approval.approved_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ResearchRevisionIssue::EmptyField {
                    field: format!("approval.{:?}.{field}", approval.role),
                });
            }
        }
        if !is_sha256(&approval.decision_sha256) {
            issues.push(ResearchRevisionIssue::InvalidDigest {
                field: format!("approval.{:?}.decision_sha256", approval.role),
            });
        }
        if approval.role != RevisionApprovalRole::Maintainer && !approval.independent_of_authors {
            issues.push(ResearchRevisionIssue::ReviewerNotIndependent {
                role: approval.role,
            });
        }
        if approval.role != RevisionApprovalRole::Maintainer
            && approval.conflict_declaration.trim().is_empty()
        {
            issues.push(ResearchRevisionIssue::EmptyConflictDeclaration {
                role: approval.role,
            });
        }
    }
    for role in required_approval_roles(proposal.revision_class) {
        if !roles.contains(&role) {
            issues.push(ResearchRevisionIssue::MissingApprovalRole { role });
        }
    }
}

fn required_evidence_roles(class: ResearchRevisionClass) -> Vec<RevisionEvidenceRole> {
    let mut roles = vec![
        RevisionEvidenceRole::SourceDiff,
        RevisionEvidenceRole::BuildValidation,
        RevisionEvidenceRole::RegressionSuite,
    ];
    match class {
        ResearchRevisionClass::DocumentationOnly => {}
        ResearchRevisionClass::DefectCorrection => {
            roles.push(RevisionEvidenceRole::RootCauseAnalysis);
            roles.push(RevisionEvidenceRole::MigrationPlan);
        }
        ResearchRevisionClass::ModelRevision => {
            roles.extend([
                RevisionEvidenceRole::MigrationPlan,
                RevisionEvidenceRole::FrozenModelCheckpoint,
                RevisionEvidenceRole::TrainingCorpus,
                RevisionEvidenceRole::HoldoutEvaluation,
                RevisionEvidenceRole::ExternalReview,
            ]);
        }
        ResearchRevisionClass::ProtocolRevision => {
            roles.extend([
                RevisionEvidenceRole::MigrationPlan,
                RevisionEvidenceRole::ExternalReview,
                RevisionEvidenceRole::StatisticalReview,
                RevisionEvidenceRole::HumanStudyReview,
                RevisionEvidenceRole::Preregistration,
                RevisionEvidenceRole::SyntheticDryRun,
            ]);
        }
        ResearchRevisionClass::OutcomeDefinitionChange => {
            roles.extend([
                RevisionEvidenceRole::MigrationPlan,
                RevisionEvidenceRole::ExternalReview,
                RevisionEvidenceRole::StatisticalReview,
                RevisionEvidenceRole::HumanStudyReview,
                RevisionEvidenceRole::Preregistration,
                RevisionEvidenceRole::SyntheticDryRun,
                RevisionEvidenceRole::NewConfirmatoryPlan,
            ]);
        }
    }
    roles
}

fn required_approval_roles(class: ResearchRevisionClass) -> Vec<RevisionApprovalRole> {
    let mut roles = vec![
        RevisionApprovalRole::Maintainer,
        RevisionApprovalRole::ReproducibilityReviewer,
    ];
    match class {
        ResearchRevisionClass::DocumentationOnly | ResearchRevisionClass::DefectCorrection => {}
        ResearchRevisionClass::ModelRevision => {
            roles.push(RevisionApprovalRole::StatisticalReviewer);
            roles.push(RevisionApprovalRole::MusicTheoryReviewer);
        }
        ResearchRevisionClass::ProtocolRevision
        | ResearchRevisionClass::OutcomeDefinitionChange => {
            roles.push(RevisionApprovalRole::StatisticalReviewer);
            roles.push(RevisionApprovalRole::MusicTheoryReviewer);
            roles.push(RevisionApprovalRole::HumanStudyReviewer);
        }
    }
    roles
}

fn claim_impact_sufficient(class: ResearchRevisionClass, impact: RevisionClaimImpact) -> bool {
    match class {
        ResearchRevisionClass::DocumentationOnly => impact == RevisionClaimImpact::NoClaimImpact,
        ResearchRevisionClass::DefectCorrection => matches!(
            impact,
            RevisionClaimImpact::RequiresReanalysis
                | RevisionClaimImpact::RequiresIndependentReplication
                | RevisionClaimImpact::RequiresNewConfirmation
        ),
        ResearchRevisionClass::ModelRevision => matches!(
            impact,
            RevisionClaimImpact::RequiresIndependentReplication
                | RevisionClaimImpact::RequiresNewConfirmation
        ),
        ResearchRevisionClass::ProtocolRevision
        | ResearchRevisionClass::OutcomeDefinitionChange => {
            impact == RevisionClaimImpact::RequiresNewConfirmation
        }
    }
}

fn valid_semantic_version(value: &str) -> bool {
    let core = value.split_once('-').map_or(value, |(core, _)| core);
    let parts = core.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_revision_requires_new_confirmation() {
        assert!(!claim_impact_sufficient(
            ResearchRevisionClass::ProtocolRevision,
            RevisionClaimImpact::RequiresReanalysis,
        ));
        assert!(claim_impact_sufficient(
            ResearchRevisionClass::ProtocolRevision,
            RevisionClaimImpact::RequiresNewConfirmation,
        ));
    }

    #[test]
    fn semantic_versions_are_strictly_numeric() {
        assert!(valid_semantic_version("1.2.3"));
        assert!(valid_semantic_version("1.2.3-rc.1"));
        assert!(!valid_semantic_version("1.2"));
        assert!(!valid_semantic_version("v1.2.3"));
    }
}
