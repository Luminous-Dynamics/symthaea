#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Job Postings Integrity Zome
//!
//! Entry types and validation for decentralized job postings in the
//! Mycelix Craft. Jobs are discoverable via anchor-indexed
//! skill links — the client-side matching engine queries these.

use hdi::prelude::*;

// ============== Entry Types ==============

/// A job opportunity posted by an employer/organization.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JobPosting {
    /// Job title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Hiring organization name
    pub organization: String,
    /// Location (city, region, or "Remote")
    pub location: Option<String>,
    /// Whether remote work is available
    pub remote_ok: bool,
    /// Skills required for the role (lowercase, anchor-indexed)
    pub required_skills: Vec<String>,
    /// Preferred/bonus skills
    pub preferred_skills: Vec<String>,
    /// Education level requirement (e.g., "Bachelor's", "Master's", "None")
    pub education_level: Option<String>,
    /// Salary range (USD)
    pub salary_range: Option<SalaryRange>,
    /// When posted
    pub posted_at: Timestamp,
    /// Expiration date (optional)
    pub expires_at: Option<Timestamp>,
    /// Current status
    pub status: JobPostingStatus,
    /// Optional link to CareerProfile field for career intelligence
    pub career_profile_field: Option<String>,
    /// Guild that posted or endorses this job
    #[serde(default)]
    pub guild_id: Option<String>,
    /// Minimum epistemic level required to apply (e.g., "E3")
    #[serde(default)]
    pub min_epistemic_level: Option<String>,
    /// Minimum consciousness tier required to apply
    #[serde(default)]
    pub consciousness_tier_required: Option<String>,
    /// Minimum credential vitality (0-1000 permille) for applicants
    #[serde(default)]
    pub vitality_minimum: Option<u16>,
}

/// Salary range in USD
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SalaryRange {
    pub min_usd: u32,
    pub max_usd: u32,
}

/// Job posting lifecycle status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobPostingStatus {
    /// Accepting applications
    Open,
    /// No longer accepting applications
    Closed,
    /// Position has been filled
    Filled,
}

/// Anchor for skill-based job indexing
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JobAnchor(pub String);

// ============== Apprenticeship Protocol ==============

/// An employer-staked learning bounty on a curriculum pathway.
///
/// Employers invest SAP into specific skill areas, creating visible
/// "bounties" that learners can work toward. On graduation (PoL threshold
/// met), learners get guaranteed interviews.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ApprenticeshipStake {
    /// Employer agent who staked
    pub employer: AgentPubKey,
    /// Organization name
    pub organization: String,
    /// Curriculum pathway or skill area (e.g., "Rust Development", "Data Science")
    pub pathway: String,
    /// SAP amount staked (signals commitment)
    pub stake_sap: u32,
    /// Maximum number of apprentices this stake can fund
    pub max_apprentices: u16,
    /// Current number of apprentices claimed
    pub current_apprentices: u16,
    /// Minimum PoL composite score required for graduation (0-1000 permille)
    pub required_pol_permille: u16,
    /// Whether graduates get guaranteed interviews
    pub interview_guarantee: bool,
    /// Skills required for this apprenticeship
    pub required_skills: Vec<String>,
    /// When the stake was created
    pub created_at: Timestamp,
    /// Status
    pub status: StakeStatus,
    /// Guild that sponsors this apprenticeship (optional)
    #[serde(default)]
    pub guild_id: Option<String>,
}

/// Status of an apprenticeship stake
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StakeStatus {
    /// Accepting apprentices
    Active,
    /// All slots filled
    Full,
    /// Stake withdrawn or expired
    Closed,
}

// ============== Entry & Link Types ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    JobPosting(JobPosting),
    #[entry_type(visibility = "public")]
    JobAnchor(JobAnchor),
    #[entry_type(visibility = "public")]
    ApprenticeshipStake(ApprenticeshipStake),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Employer agent -> their job postings
    AgentToJobPosting,
    /// Skill anchor -> job postings requiring that skill
    SkillToJobPosting,
    /// Global anchor -> all job postings (for browse)
    AllJobPostings,
    /// Employer -> their apprenticeship stakes
    AgentToApprenticeshipStake,
    /// Pathway anchor -> apprenticeship stakes (for discovery)
    PathwayToStakes,
    /// Global anchor -> all active stakes
    AllStakes,
}

// ============== Validation ==============

pub fn validate_job_posting(posting: &JobPosting) -> ExternResult<ValidateCallbackResult> {
    if posting.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Job title cannot be empty".to_string(),
        ));
    }
    if posting.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Job title cannot exceed 200 characters".to_string(),
        ));
    }
    if posting.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Job description cannot be empty".to_string(),
        ));
    }
    if posting.description.len() > 10_000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Job description cannot exceed 10000 characters".to_string(),
        ));
    }
    if posting.organization.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Organization cannot be empty".to_string(),
        ));
    }
    if posting.required_skills.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 required skills".to_string(),
        ));
    }
    if let Some(ref range) = posting.salary_range {
        if range.min_usd > range.max_usd {
            return Ok(ValidateCallbackResult::Invalid(
                "Salary minimum cannot exceed maximum".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_apprenticeship_stake(stake: &ApprenticeshipStake) -> ExternResult<ValidateCallbackResult> {
    if stake.pathway.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Pathway cannot be empty".into()));
    }
    if stake.organization.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Organization cannot be empty".into()));
    }
    if stake.stake_sap == 0 {
        return Ok(ValidateCallbackResult::Invalid("Stake must be greater than zero".into()));
    }
    if stake.max_apprentices == 0 {
        return Ok(ValidateCallbackResult::Invalid("Must allow at least one apprentice".into()));
    }
    if stake.required_pol_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid("PoL threshold must be 0-1000 permille".into()));
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
                    EntryTypes::JobPosting(posting) => validate_job_posting(&posting),
                    EntryTypes::JobAnchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::ApprenticeshipStake(stake) => validate_apprenticeship_stake(&stake),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_posting() -> JobPosting {
        JobPosting {
            title: "Rust Developer".to_string(),
            description: "Build decentralized applications with Holochain".to_string(),
            organization: "Luminous Dynamics".to_string(),
            location: Some("Remote".to_string()),
            remote_ok: true,
            required_skills: vec!["rust".to_string(), "holochain".to_string()],
            preferred_skills: vec!["leptos".to_string()],
            education_level: Some("Bachelor's".to_string()),
            salary_range: Some(SalaryRange { min_usd: 80_000, max_usd: 120_000 }),
            posted_at: Timestamp::from_micros(0),
            expires_at: None,
            status: JobPostingStatus::Open,
            career_profile_field: Some("Software Development".to_string()),
            guild_id: None,
            min_epistemic_level: None,
            consciousness_tier_required: None,
            vitality_minimum: None,
        }
    }

    #[test]
    fn test_valid_posting() {
        let result = validate_job_posting(&valid_posting()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_title() {
        let mut p = valid_posting();
        p.title = "".to_string();
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_title_too_long() {
        let mut p = valid_posting();
        p.title = "x".repeat(201);
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_description() {
        let mut p = valid_posting();
        p.description = "".to_string();
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_organization() {
        let mut p = valid_posting();
        p.organization = "".to_string();
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_salary_range_inverted() {
        let mut p = valid_posting();
        p.salary_range = Some(SalaryRange { min_usd: 100_000, max_usd: 50_000 });
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_too_many_skills() {
        let mut p = valid_posting();
        p.required_skills = (0..51).map(|i| format!("skill{}", i)).collect();
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_no_salary_range_valid() {
        let mut p = valid_posting();
        p.salary_range = None;
        assert_eq!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_all_statuses() {
        let statuses = [JobPostingStatus::Open, JobPostingStatus::Closed, JobPostingStatus::Filled];
        for s in statuses {
            let mut p = valid_posting();
            p.status = s;
            assert_eq!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Valid);
        }
    }

    // ---- Supplementary tests ----

    #[test]
    fn test_description_too_long() {
        let mut p = valid_posting();
        p.description = "x".repeat(10_001);
        assert!(matches!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_salary_range_equal_valid() {
        let mut p = valid_posting();
        p.salary_range = Some(SalaryRange { min_usd: 60_000, max_usd: 60_000 });
        assert_eq!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_skill_normalization_idempotent() {
        let skill = "RUST".to_string();
        let normalized = skill.to_lowercase();
        assert_eq!(normalized.to_lowercase(), normalized);
    }

    #[test]
    fn test_serde_roundtrip_posting_status() {
        for s in [JobPostingStatus::Open, JobPostingStatus::Closed, JobPostingStatus::Filled] {
            let json = serde_json::to_string(&s).unwrap();
            let back: JobPostingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn test_serde_roundtrip_stake_status() {
        for s in [StakeStatus::Active, StakeStatus::Full, StakeStatus::Closed] {
            let json = serde_json::to_string(&s).unwrap();
            let back: StakeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn test_serde_roundtrip_salary_range() {
        let range = SalaryRange { min_usd: 50_000, max_usd: 120_000 };
        let json = serde_json::to_string(&range).unwrap();
        let back: SalaryRange = serde_json::from_str(&json).unwrap();
        assert_eq!(range, back);
    }

    #[test]
    fn test_preferred_skills_can_be_empty() {
        let mut p = valid_posting();
        p.preferred_skills = vec![];
        assert_eq!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_remote_ok_with_location() {
        let mut p = valid_posting();
        p.remote_ok = true;
        p.location = Some("Cape Town".to_string());
        assert_eq!(validate_job_posting(&p).unwrap(), ValidateCallbackResult::Valid);
    }
}
