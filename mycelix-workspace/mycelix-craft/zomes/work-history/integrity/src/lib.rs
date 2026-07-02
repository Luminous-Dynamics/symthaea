#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Work History Integrity Zome
//!
//! Entry types for work experience with peer verification.
//! Verification status is tracked via links, not entry mutation.

use hdi::prelude::*;

/// A work experience entry on a craft timeline.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WorkExperience {
    /// Job title / role
    pub title: String,
    /// Organization / company name
    pub organization: String,
    /// Start date (ISO 8601)
    pub start_date: String,
    /// End date (None = currently employed)
    pub end_date: Option<String>,
    /// Description of responsibilities and achievements
    pub description: String,
    /// Skills demonstrated in this role (lowercase)
    pub skills_used: Vec<String>,
    /// When this entry was created
    pub created_at: Timestamp,
}

/// Peer verification of a work experience claim.
/// Linked from the WorkExperience entry — existence of link = verified.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WorkVerification {
    /// Hash of the WorkExperience being verified
    pub experience_hash_b64: String,
    /// Agent who issued the verification
    pub verifier: String,
    /// Relationship to the verified person (e.g., "manager", "colleague")
    pub relationship: String,
    /// Rationale for verification
    pub rationale: String,
    /// When verified
    pub verified_at: Timestamp,
}

/// Anchor for organization-based lookup
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OrgAnchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    WorkExperience(WorkExperience),
    #[entry_type(visibility = "public")]
    WorkVerification(WorkVerification),
    #[entry_type(visibility = "public")]
    OrgAnchor(OrgAnchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> their work experiences
    AgentToWorkExperience,
    /// Organization anchor -> work experiences at that org
    OrgAnchorToWorkExperience,
    /// WorkExperience -> verification records (existence = verified)
    ExperienceToVerification,
}

pub fn validate_work_experience(exp: &WorkExperience) -> ExternResult<ValidateCallbackResult> {
    if exp.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Title cannot be empty".into()));
    }
    if exp.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Title cannot exceed 200 characters".into()));
    }
    if exp.organization.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Organization cannot be empty".into()));
    }
    if exp.description.len() > 5_000 {
        return Ok(ValidateCallbackResult::Invalid("Description cannot exceed 5000 characters".into()));
    }
    if exp.skills_used.len() > 30 {
        return Ok(ValidateCallbackResult::Invalid("Cannot list more than 30 skills".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::WorkExperience(exp) => validate_work_experience(&exp),
                    EntryTypes::WorkVerification(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::OrgAnchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { original_action, action, .. } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can delete this link".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. } | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. } | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can update their entries".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can delete their entries".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_experience() -> WorkExperience {
        WorkExperience {
            title: "Software Engineer".to_string(),
            organization: "Luminous Dynamics".to_string(),
            start_date: "2024-01-01".to_string(),
            end_date: None,
            description: "Building decentralized education infrastructure".to_string(),
            skills_used: vec!["rust".to_string(), "holochain".to_string()],
            created_at: Timestamp::from_micros(0),
        }
    }

    #[test]
    fn test_valid_experience() {
        assert_eq!(validate_work_experience(&valid_experience()).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_title() {
        let mut e = valid_experience();
        e.title = "".to_string();
        assert!(matches!(validate_work_experience(&e).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_organization() {
        let mut e = valid_experience();
        e.organization = "".to_string();
        assert!(matches!(validate_work_experience(&e).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_too_many_skills() {
        let mut e = valid_experience();
        e.skills_used = (0..31).map(|i| format!("skill{}", i)).collect();
        assert!(matches!(validate_work_experience(&e).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
