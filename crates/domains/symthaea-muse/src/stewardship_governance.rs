// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-term stewardship charter for published Symthaea–Muse research releases.
//!
//! The charter distributes release, reproducibility, archive, security, and
//! participant-protection authority so that no single maintainer can silently
//! rewrite evidence or abandon the public record.

use crate::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, confirmatory_final_release_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STEWARDSHIP_GOVERNANCE_VERSION: &str = "symthaea-muse-stewardship-governance-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StewardshipRole {
    ReleaseMaintainer,
    ReproducibilityCustodian,
    ArchiveCustodian,
    SecurityContact,
    ParticipantProtectionOfficer,
    IndependentMethodsReviewer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StewardshipMember {
    pub member_id: String,
    pub organization_id: String,
    pub roles: Vec<StewardshipRole>,
    pub contact_commitment_sha256: String,
    pub conflict_declaration: String,
    pub independent_of_primary_author: bool,
    pub term_begins_at_utc: String,
    pub term_ends_at_utc: String,
    pub appointment_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchStewardshipCharter {
    pub charter_version: String,
    pub stewardship_id: String,
    pub source_final_release_sha256: String,
    pub members: Vec<StewardshipMember>,
    pub ordinary_quorum: u32,
    pub emergency_quorum: u32,
    pub succession_plan_sha256: String,
    pub continuity_drill_sha256: String,
    pub vulnerability_disclosure_uri: String,
    pub vulnerability_policy_sha256: String,
    pub evidence_correction_policy_sha256: String,
    pub end_of_life_policy_sha256: String,
    pub funding_conflict_policy_sha256: String,
    pub charter_public_uri: String,
    pub effective_at_utc: String,
    pub charter_sha256: String,
}

#[derive(Serialize)]
struct CharterCommitment<'a> {
    charter_version: &'a str,
    stewardship_id: &'a str,
    source_final_release_sha256: &'a str,
    members: &'a [StewardshipMember],
    ordinary_quorum: u32,
    emergency_quorum: u32,
    succession_plan_sha256: &'a str,
    continuity_drill_sha256: &'a str,
    vulnerability_disclosure_uri: &'a str,
    vulnerability_policy_sha256: &'a str,
    evidence_correction_policy_sha256: &'a str,
    end_of_life_policy_sha256: &'a str,
    funding_conflict_policy_sha256: &'a str,
    charter_public_uri: &'a str,
    effective_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StewardshipGovernanceIssue {
    WrongVersion {
        found: String,
    },
    InvalidSourceRelease,
    SourceReleaseMismatch,
    EmptyField {
        member_id: Option<String>,
        field: String,
    },
    InvalidDigest {
        member_id: Option<String>,
        field: String,
    },
    TooFewMembers {
        found: usize,
    },
    TooFewOrganizations {
        found: usize,
    },
    DuplicateMember {
        member_id: String,
    },
    DuplicateRoleForMember {
        member_id: String,
        role: StewardshipRole,
    },
    MissingRole {
        role: StewardshipRole,
    },
    CriticalRoleConcentration {
        member_id: String,
    },
    IndependentRoleMissing {
        role: StewardshipRole,
    },
    EmptyConflictDeclaration {
        member_id: String,
    },
    InvalidQuorum,
    SerializationFailed,
    CharterDigestMismatch,
}

pub fn stewardship_charter_commitment(
    charter: &ResearchStewardshipCharter,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CharterCommitment {
        charter_version: &charter.charter_version,
        stewardship_id: &charter.stewardship_id,
        source_final_release_sha256: &charter.source_final_release_sha256,
        members: &charter.members,
        ordinary_quorum: charter.ordinary_quorum,
        emergency_quorum: charter.emergency_quorum,
        succession_plan_sha256: &charter.succession_plan_sha256,
        continuity_drill_sha256: &charter.continuity_drill_sha256,
        vulnerability_disclosure_uri: &charter.vulnerability_disclosure_uri,
        vulnerability_policy_sha256: &charter.vulnerability_policy_sha256,
        evidence_correction_policy_sha256: &charter.evidence_correction_policy_sha256,
        end_of_life_policy_sha256: &charter.end_of_life_policy_sha256,
        funding_conflict_policy_sha256: &charter.funding_conflict_policy_sha256,
        charter_public_uri: &charter.charter_public_uri,
        effective_at_utc: &charter.effective_at_utc,
    })
}

pub fn seal_stewardship_charter(
    source_release: &ConfirmatoryFinalReleaseBundle,
    charter: &mut ResearchStewardshipCharter,
) -> Result<(), Vec<StewardshipGovernanceIssue>> {
    for member in &mut charter.members {
        member.roles.sort();
        member.roles.dedup();
    }
    charter
        .members
        .sort_by(|left, right| left.member_id.cmp(&right.member_id));
    charter.charter_sha256 = stewardship_charter_commitment(charter)
        .map_err(|_| vec![StewardshipGovernanceIssue::SerializationFailed])?;
    let issues = validate_stewardship_charter(source_release, charter);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_stewardship_charter(
    source_release: &ConfirmatoryFinalReleaseBundle,
    charter: &ResearchStewardshipCharter,
) -> Vec<StewardshipGovernanceIssue> {
    let mut issues = Vec::new();
    if charter.charter_version != STEWARDSHIP_GOVERNANCE_VERSION {
        issues.push(StewardshipGovernanceIssue::WrongVersion {
            found: charter.charter_version.clone(),
        });
    }
    match confirmatory_final_release_commitment(source_release) {
        Ok(digest) if digest == source_release.bundle_sha256 => {}
        _ => issues.push(StewardshipGovernanceIssue::InvalidSourceRelease),
    }
    if charter.source_final_release_sha256 != source_release.bundle_sha256 {
        issues.push(StewardshipGovernanceIssue::SourceReleaseMismatch);
    }
    for (field, value) in [
        ("stewardship_id", charter.stewardship_id.as_str()),
        (
            "vulnerability_disclosure_uri",
            charter.vulnerability_disclosure_uri.as_str(),
        ),
        ("charter_public_uri", charter.charter_public_uri.as_str()),
        ("effective_at_utc", charter.effective_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(StewardshipGovernanceIssue::EmptyField {
                member_id: None,
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "source_final_release_sha256",
            charter.source_final_release_sha256.as_str(),
        ),
        (
            "succession_plan_sha256",
            charter.succession_plan_sha256.as_str(),
        ),
        (
            "continuity_drill_sha256",
            charter.continuity_drill_sha256.as_str(),
        ),
        (
            "vulnerability_policy_sha256",
            charter.vulnerability_policy_sha256.as_str(),
        ),
        (
            "evidence_correction_policy_sha256",
            charter.evidence_correction_policy_sha256.as_str(),
        ),
        (
            "end_of_life_policy_sha256",
            charter.end_of_life_policy_sha256.as_str(),
        ),
        (
            "funding_conflict_policy_sha256",
            charter.funding_conflict_policy_sha256.as_str(),
        ),
        ("charter_sha256", charter.charter_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(StewardshipGovernanceIssue::InvalidDigest {
                member_id: None,
                field: field.into(),
            });
        }
    }
    validate_members(charter, &mut issues);
    let count = charter.members.len() as u32;
    if charter.ordinary_quorum < 2
        || charter.ordinary_quorum > count
        || charter.emergency_quorum < 2
        || charter.emergency_quorum > count
    {
        issues.push(StewardshipGovernanceIssue::InvalidQuorum);
    }
    match stewardship_charter_commitment(charter) {
        Ok(digest) if digest == charter.charter_sha256 => {}
        Ok(_) => issues.push(StewardshipGovernanceIssue::CharterDigestMismatch),
        Err(_) => issues.push(StewardshipGovernanceIssue::SerializationFailed),
    }
    issues
}

fn validate_members(
    charter: &ResearchStewardshipCharter,
    issues: &mut Vec<StewardshipGovernanceIssue>,
) {
    if charter.members.len() < 3 {
        issues.push(StewardshipGovernanceIssue::TooFewMembers {
            found: charter.members.len(),
        });
    }
    let organizations = charter
        .members
        .iter()
        .map(|member| member.organization_id.as_str())
        .collect::<BTreeSet<_>>();
    if organizations.len() < 2 {
        issues.push(StewardshipGovernanceIssue::TooFewOrganizations {
            found: organizations.len(),
        });
    }
    let mut member_ids = BTreeSet::new();
    let mut role_members = BTreeMap::<StewardshipRole, Vec<&StewardshipMember>>::new();
    for member in &charter.members {
        if !member_ids.insert(member.member_id.as_str()) {
            issues.push(StewardshipGovernanceIssue::DuplicateMember {
                member_id: member.member_id.clone(),
            });
        }
        for (field, value) in [
            ("member_id", member.member_id.as_str()),
            ("organization_id", member.organization_id.as_str()),
            ("term_begins_at_utc", member.term_begins_at_utc.as_str()),
            ("term_ends_at_utc", member.term_ends_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(StewardshipGovernanceIssue::EmptyField {
                    member_id: Some(member.member_id.clone()),
                    field: field.into(),
                });
            }
        }
        for (field, digest) in [
            (
                "contact_commitment_sha256",
                member.contact_commitment_sha256.as_str(),
            ),
            (
                "appointment_receipt_sha256",
                member.appointment_receipt_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(StewardshipGovernanceIssue::InvalidDigest {
                    member_id: Some(member.member_id.clone()),
                    field: field.into(),
                });
            }
        }
        if member.conflict_declaration.trim().is_empty() {
            issues.push(StewardshipGovernanceIssue::EmptyConflictDeclaration {
                member_id: member.member_id.clone(),
            });
        }
        let mut roles = BTreeSet::new();
        for role in &member.roles {
            if !roles.insert(*role) {
                issues.push(StewardshipGovernanceIssue::DuplicateRoleForMember {
                    member_id: member.member_id.clone(),
                    role: *role,
                });
            }
            role_members.entry(*role).or_default().push(member);
        }
        let critical_count = member
            .roles
            .iter()
            .filter(|role| {
                matches!(
                    role,
                    StewardshipRole::ReleaseMaintainer
                        | StewardshipRole::ReproducibilityCustodian
                        | StewardshipRole::ArchiveCustodian
                        | StewardshipRole::SecurityContact
                )
            })
            .count();
        if critical_count > 2 {
            issues.push(StewardshipGovernanceIssue::CriticalRoleConcentration {
                member_id: member.member_id.clone(),
            });
        }
    }
    for role in [
        StewardshipRole::ReleaseMaintainer,
        StewardshipRole::ReproducibilityCustodian,
        StewardshipRole::ArchiveCustodian,
        StewardshipRole::SecurityContact,
        StewardshipRole::ParticipantProtectionOfficer,
        StewardshipRole::IndependentMethodsReviewer,
    ] {
        let members = role_members.get(&role).cloned().unwrap_or_default();
        if members.is_empty() {
            issues.push(StewardshipGovernanceIssue::MissingRole { role });
        }
        if matches!(
            role,
            StewardshipRole::IndependentMethodsReviewer
                | StewardshipRole::ParticipantProtectionOfficer
        ) && !members
            .iter()
            .any(|member| member.independent_of_primary_author)
        {
            issues.push(StewardshipGovernanceIssue::IndependentRoleMissing { role });
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_person_cannot_hold_all_critical_roles() {
        let member = StewardshipMember {
            member_id: "one".into(),
            organization_id: "org".into(),
            roles: vec![
                StewardshipRole::ReleaseMaintainer,
                StewardshipRole::ReproducibilityCustodian,
                StewardshipRole::ArchiveCustodian,
                StewardshipRole::SecurityContact,
            ],
            contact_commitment_sha256: "a".repeat(64),
            conflict_declaration: "none".into(),
            independent_of_primary_author: false,
            term_begins_at_utc: "now".into(),
            term_ends_at_utc: "later".into(),
            appointment_receipt_sha256: "b".repeat(64),
        };
        let charter = ResearchStewardshipCharter {
            charter_version: STEWARDSHIP_GOVERNANCE_VERSION.into(),
            stewardship_id: "id".into(),
            source_final_release_sha256: "c".repeat(64),
            members: vec![member],
            ordinary_quorum: 1,
            emergency_quorum: 1,
            succession_plan_sha256: "d".repeat(64),
            continuity_drill_sha256: "e".repeat(64),
            vulnerability_disclosure_uri: "uri".into(),
            vulnerability_policy_sha256: "f".repeat(64),
            evidence_correction_policy_sha256: "1".repeat(64),
            end_of_life_policy_sha256: "2".repeat(64),
            funding_conflict_policy_sha256: "3".repeat(64),
            charter_public_uri: "uri".into(),
            effective_at_utc: "now".into(),
            charter_sha256: String::new(),
        };
        let mut issues = Vec::new();
        validate_members(&charter, &mut issues);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            StewardshipGovernanceIssue::CriticalRoleConcentration { .. }
        )));
    }
}
