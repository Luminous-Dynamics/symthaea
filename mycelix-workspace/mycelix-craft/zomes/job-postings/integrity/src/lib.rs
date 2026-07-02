// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Job Postings Integrity Zome
//!
//! Defines entry types and validation rules for workforce opportunities.

use hdi::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum JobPostingStatus {
    Open,
    Closed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SalaryRange {
    pub min: u64,
    pub max: u64,
    pub currency: String,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JobPosting {
    pub title: String,
    pub description: String,
    pub requirements: Vec<String>,
    pub organization: String,
    pub location: String,
    pub remote_ok: bool,
    pub required_skills: Vec<String>,
    pub preferred_skills: Vec<String>,
    pub education_level: String,
    pub salary_range: SalaryRange,
    pub posted_at: Timestamp,
    pub created_at: i64,
    pub status: JobPostingStatus,
    pub expires_at: Option<Timestamp>,
    pub career_profile_field: String,
    pub guild_id: Option<String>,
    pub min_epistemic_level: Option<u16>,
    pub consciousness_tier_required: Option<u16>,
    pub vitality_minimum: Option<f64>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JobAnchor(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum StakeStatus {
    Active,
    Full,
    Completed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ApprenticeshipStake {
    pub apprentice: AgentPubKey,
    pub mentor: AgentPubKey,
    pub topic: String,
    pub stake_amount: u64,
    pub employer: AgentPubKey,
    pub organization: String,
    pub pathway: String,
    pub stake_sap: u64,
    pub max_apprentices: u32,
    pub current_apprentices: u32,
    pub required_pol_permille: u16,
    pub interview_guarantee: bool,
    pub required_skills: Vec<String>,
    pub created_at: Timestamp,
    pub status: StakeStatus,
    pub guild_id: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
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
    AgentToJobPosting,
    SkillToJobPosting,
    AllJobPostings,
    AgentToApprenticeshipStake,
    PathwayToStakes,
    AllStakes,
}

// ============== Validation Functions ==============

pub fn validate_job_posting(posting: &JobPosting) -> ExternResult<ValidateCallbackResult> {
    if posting.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_apprenticeship_stake(
    stake: &ApprenticeshipStake,
) -> ExternResult<ValidateCallbackResult> {
    if stake.stake_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Stake must be greater than zero".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(entry) => match entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::JobPosting(posting) => validate_job_posting(&posting),
                EntryTypes::ApprenticeshipStake(stake) => validate_apprenticeship_stake(&stake),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
