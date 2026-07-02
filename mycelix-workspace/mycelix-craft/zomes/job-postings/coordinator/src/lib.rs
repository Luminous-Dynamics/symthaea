// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Job Postings Coordinator Zome
//!
//! Implements business logic for workforce opportunities and apprenticeship stakes.

use hdk::prelude::*;
use job_postings_integrity::*;
use mycelix_zome_helpers as _;

/// Create a new job posting.
#[hdk_extern]
pub fn create_job_posting(input: CreateJobPostingInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let salary_range = if let (Some(min), Some(max)) = (input.min_salary, input.max_salary) {
        SalaryRange {
            min,
            max,
            currency: "SAP".into(),
        }
    } else {
        SalaryRange {
            min: 0,
            max: 0,
            currency: "SAP".into(),
        }
    };

    let posting = JobPosting {
        title: input.title,
        description: input.description,
        requirements: input.required_skills.clone(),
        organization: input.organization,
        location: input.location.unwrap_or_default(),
        remote_ok: input.remote_ok,
        required_skills: input.required_skills,
        preferred_skills: input.preferred_skills.unwrap_or_default(),
        education_level: input.education_level.unwrap_or_default(),
        salary_range,
        posted_at: now,
        created_at: now.as_micros() as i64,
        status: JobPostingStatus::Open,
        expires_at: None,
        career_profile_field: input.career_profile_field.unwrap_or_default(),
        guild_id: None,
        min_epistemic_level: None,
        consciousness_tier_required: None,
        vitality_minimum: None,
    };

    let action_hash = create_entry(EntryTypes::JobPosting(posting))?;

    // Link from global anchor
    let anchor = Path::from("all_job_postings");
    create_link(
        anchor.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllJobPostings,
        (),
    )?;

    Ok(action_hash)
}

/// Get a job posting by its action hash.
#[hdk_extern]
pub fn get_job_posting(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateJobPostingInput {
    pub title: String,
    pub description: String,
    pub organization: String,
    pub location: Option<String>,
    pub remote_ok: bool,
    pub required_skills: Vec<String>,
    pub preferred_skills: Option<Vec<String>>,
    pub education_level: Option<String>,
    pub career_profile_field: Option<String>,
    pub min_salary: Option<u64>,
    pub max_salary: Option<u64>,
}

/// Create an apprenticeship stake.
#[hdk_extern]
pub fn create_apprenticeship_stake(
    input: CreateApprenticeshipStakeInput,
) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let stake = ApprenticeshipStake {
        apprentice: agent.clone(),
        mentor: input.mentor,
        topic: input.topic.clone(),
        stake_amount: input.stake_sap as u64,
        employer: agent,
        organization: input.organization,
        pathway: input.pathway,
        stake_sap: input.stake_sap as u64,
        max_apprentices: input.max_apprentices as u32,
        current_apprentices: 0,
        required_pol_permille: input.required_pol_permille,
        interview_guarantee: input.interview_guarantee,
        required_skills: input
            .required_skills
            .iter()
            .map(|s| s.to_lowercase())
            .collect(),
        created_at: now,
        status: StakeStatus::Active,
        guild_id: None,
    };

    let action_hash = create_entry(EntryTypes::ApprenticeshipStake(stake))?;

    // Link from global anchor
    let anchor = Path::from("all_stakes");
    create_link(
        anchor.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllStakes,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateApprenticeshipStakeInput {
    pub mentor: AgentPubKey,
    pub topic: String,
    pub organization: String,
    pub pathway: String,
    pub stake_sap: u32,
    pub max_apprentices: u16,
    pub required_pol_permille: u16,
    pub interview_guarantee: bool,
    pub required_skills: Vec<String>,
}
