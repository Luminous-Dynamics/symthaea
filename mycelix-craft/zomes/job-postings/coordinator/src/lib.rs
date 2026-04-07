#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Job Postings Coordinator Zome
//!
//! Business logic for creating, browsing, and searching job postings.
//! Jobs are indexed by required skills via anchor links, enabling efficient
//! client-side matching without global DHT aggregation.

use hdk::prelude::*;
use job_postings_integrity::{
    ApprenticeshipStake, EntryTypes, JobAnchor, JobPosting, JobPostingStatus,
    LinkTypes, StakeStatus,
};

// ============== Helpers ==============

fn ensure_anchor(anchor_text: &str) -> ExternResult<EntryHash> {
    let anchor = JobAnchor(anchor_text.to_string());
    create_entry(EntryTypes::JobAnchor(anchor.clone()))?;
    hash_entry(anchor)
}

fn all_jobs_anchor() -> ExternResult<EntryHash> {
    ensure_anchor("all_job_postings")
}

fn skill_anchor(skill: &str) -> ExternResult<EntryHash> {
    ensure_anchor(&format!("job_skill.{}", skill.to_lowercase()))
}

fn load_postings_from_links(links: Vec<Link>) -> ExternResult<Vec<JobPosting>> {
    let mut postings = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(posting) = JobPosting::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        postings.push(posting);
                    }
                }
            }
        }
    }
    Ok(postings)
}

// ============== Input Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateJobPostingInput {
    pub title: String,
    pub description: String,
    pub organization: String,
    pub location: Option<String>,
    pub remote_ok: bool,
    pub required_skills: Vec<String>,
    pub preferred_skills: Vec<String>,
    pub education_level: Option<String>,
    pub salary_min_usd: Option<u32>,
    pub salary_max_usd: Option<u32>,
    pub career_profile_field: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateJobPostingInput {
    pub original_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub organization: String,
    pub location: Option<String>,
    pub remote_ok: bool,
    pub required_skills: Vec<String>,
    pub preferred_skills: Vec<String>,
    pub education_level: Option<String>,
    pub salary_min_usd: Option<u32>,
    pub salary_max_usd: Option<u32>,
    pub career_profile_field: Option<String>,
}

// ============== Extern Functions ==============

/// Create a new job posting, indexed by required skills for efficient search.
#[hdk_extern]
pub fn create_job_posting(input: CreateJobPostingInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let salary_range = match (input.salary_min_usd, input.salary_max_usd) {
        (Some(min), Some(max)) => Some(job_postings_integrity::SalaryRange {
            min_usd: min,
            max_usd: max,
        }),
        _ => None,
    };

    let posting = JobPosting {
        title: input.title,
        description: input.description,
        organization: input.organization,
        location: input.location,
        remote_ok: input.remote_ok,
        required_skills: input.required_skills.iter().map(|s| s.to_lowercase()).collect(),
        preferred_skills: input.preferred_skills.iter().map(|s| s.to_lowercase()).collect(),
        education_level: input.education_level,
        salary_range,
        posted_at: now,
        expires_at: None,
        status: JobPostingStatus::Open,
        career_profile_field: input.career_profile_field,
        guild_id: None,
        min_epistemic_level: None,
        consciousness_tier_required: None,
        vitality_minimum: None,
    };

    let action_hash = create_entry(EntryTypes::JobPosting(posting.clone()))?;

    // Link: agent -> posting
    let agent_hash: AnyDhtHash = agent.into();
    create_link(
        agent_hash,
        action_hash.clone(),
        LinkTypes::AgentToJobPosting,
        vec![],
    )?;

    // Link: global anchor -> posting
    let all_anchor = all_jobs_anchor()?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllJobPostings,
        vec![],
    )?;

    // Link: each required skill anchor -> posting (for search)
    for skill in &posting.required_skills {
        let anchor = skill_anchor(skill)?;
        create_link(
            anchor,
            action_hash.clone(),
            LinkTypes::SkillToJobPosting,
            vec![],
        )?;
    }

    Ok(action_hash)
}

/// Update an existing job posting (author-only, enforced at integrity layer).
#[hdk_extern]
pub fn update_job_posting(input: UpdateJobPostingInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let salary_range = match (input.salary_min_usd, input.salary_max_usd) {
        (Some(min), Some(max)) => Some(job_postings_integrity::SalaryRange {
            min_usd: min,
            max_usd: max,
        }),
        _ => None,
    };

    let posting = JobPosting {
        title: input.title,
        description: input.description,
        organization: input.organization,
        location: input.location,
        remote_ok: input.remote_ok,
        required_skills: input.required_skills.iter().map(|s| s.to_lowercase()).collect(),
        preferred_skills: input.preferred_skills.iter().map(|s| s.to_lowercase()).collect(),
        education_level: input.education_level,
        salary_range,
        posted_at: now,
        expires_at: None,
        status: JobPostingStatus::Open,
        career_profile_field: input.career_profile_field,
        guild_id: None,
        min_epistemic_level: None,
        consciousness_tier_required: None,
        vitality_minimum: None,
    };

    update_entry(input.original_hash, &posting)
}

/// Close a job posting (set status to Closed).
#[hdk_extern]
pub fn close_job_posting(action_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Posting not found".to_string())))?;

    let mut posting: JobPosting = match record.entry().as_option() {
        Some(Entry::App(bytes)) => JobPosting::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".to_string()))),
    };

    posting.status = JobPostingStatus::Closed;
    update_entry(action_hash, &posting)
}

/// Get a single job posting by hash.
#[hdk_extern]
pub fn get_job_posting(action_hash: ActionHash) -> ExternResult<Option<JobPosting>> {
    let Some(record) = get(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            let posting = JobPosting::try_from(
                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
            )
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {}", e))))?;
            Ok(Some(posting))
        }
        _ => Ok(None),
    }
}

/// List all job postings created by the calling agent.
#[hdk_extern]
pub fn list_my_job_postings(_: ()) -> ExternResult<Vec<JobPosting>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToJobPosting)?,
        GetStrategy::Local,
    )?;

    load_postings_from_links(links)
}

/// Search job postings by a single required skill (anchor-indexed).
#[hdk_extern]
pub fn search_jobs_by_skill(skill: String) -> ExternResult<Vec<JobPosting>> {
    let anchor = skill_anchor(&skill)?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SkillToJobPosting)?,
        GetStrategy::Local,
    )?;

    // Filter to open postings only
    let postings = load_postings_from_links(links)?;
    Ok(postings
        .into_iter()
        .filter(|p| p.status == JobPostingStatus::Open)
        .collect())
}

/// Search job postings matching any of the given skills (union query).
#[hdk_extern]
pub fn search_jobs_by_skills(skills: Vec<String>) -> ExternResult<Vec<JobPosting>> {
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    for skill in skills {
        let anchor = skill_anchor(&skill)?;
        let links = get_links(
            LinkQuery::try_new(anchor, LinkTypes::SkillToJobPosting)?,
            GetStrategy::Local,
        )?;

        for link in links {
            if let Some(hash) = link.target.into_action_hash() {
                if seen.insert(hash.clone()) {
                    if let Some(record) = get(hash, GetOptions::default())? {
                        if let Some(Entry::App(bytes)) = record.entry().as_option() {
                            if let Ok(posting) = JobPosting::try_from(
                                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                            ) {
                                if posting.status == JobPostingStatus::Open {
                                    results.push(posting);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// List all open job postings (for browse page).
#[hdk_extern]
pub fn list_all_open_postings(_: ()) -> ExternResult<Vec<JobPosting>> {
    let anchor = all_jobs_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllJobPostings)?,
        GetStrategy::Local,
    )?;

    let postings = load_postings_from_links(links)?;
    Ok(postings
        .into_iter()
        .filter(|p| p.status == JobPostingStatus::Open)
        .collect())
}

// ============== Skill Intelligence ==============

/// Demand signal for a single skill.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SkillDemand {
    pub skill: String,
    pub open_job_count: u32,
}

/// Get demand counts for multiple skills (link-counting on existing anchors).
///
/// PUBLIC data — no FL needed. Counts open job postings per skill via
/// existing SkillToJobPosting anchor links.
///
/// Used by client-side gap computation: demand (this) vs supply (FL mastery).
#[hdk_extern]
pub fn get_skill_demand_counts(skills: Vec<String>) -> ExternResult<Vec<SkillDemand>> {
    let mut demands = Vec::with_capacity(skills.len());

    for skill in skills {
        let anchor = skill_anchor(&skill)?;
        let links = get_links(
            LinkQuery::try_new(anchor, LinkTypes::SkillToJobPosting)?,
            GetStrategy::Local,
        )?;

        demands.push(SkillDemand {
            skill,
            open_job_count: links.len() as u32,
        });
    }

    Ok(demands)
}

// ============== Apprenticeship Protocol ==============
//
// Employers stake SAP into curriculum pathways, creating visible "bounties."
// Learners see opportunities at the end of learning journeys.
// Graduation (PoL threshold) → guaranteed interview.

fn pathway_anchor(pathway: &str) -> ExternResult<EntryHash> {
    ensure_anchor(&format!("apprenticeship.{}", pathway.to_lowercase()))
}

fn all_stakes_anchor() -> ExternResult<EntryHash> {
    ensure_anchor("all_apprenticeship_stakes")
}

/// Input for creating an apprenticeship stake.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateStakeInput {
    pub organization: String,
    pub pathway: String,
    pub stake_sap: u32,
    pub max_apprentices: u16,
    pub required_pol_permille: u16,
    pub interview_guarantee: bool,
    pub required_skills: Vec<String>,
}

/// Create an apprenticeship stake — employer invests in a learning pathway.
///
/// Creates a visible "bounty" that learners can discover on the pathways page.
/// When a learner meets the PoL threshold and graduates, they can claim
/// an apprenticeship slot (guaranteed interview if enabled).
#[hdk_extern]
pub fn create_apprenticeship_stake(input: CreateStakeInput) -> ExternResult<ActionHash> {
    let employer = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let stake = ApprenticeshipStake {
        employer: employer.clone(),
        organization: input.organization,
        pathway: input.pathway.clone(),
        stake_sap: input.stake_sap,
        max_apprentices: input.max_apprentices,
        current_apprentices: 0,
        required_pol_permille: input.required_pol_permille,
        interview_guarantee: input.interview_guarantee,
        required_skills: input.required_skills.iter().map(|s| s.to_lowercase()).collect(),
        created_at: now,
        status: StakeStatus::Active,
        guild_id: None,
    };

    let hash = create_entry(EntryTypes::ApprenticeshipStake(stake))?;

    // Link: employer -> stake
    let employer_hash: AnyDhtHash = employer.into();
    create_link(employer_hash, hash.clone(), LinkTypes::AgentToApprenticeshipStake, vec![])?;

    // Link: pathway anchor -> stake (for discovery)
    let pw_anchor = pathway_anchor(&input.pathway)?;
    create_link(pw_anchor, hash.clone(), LinkTypes::PathwayToStakes, vec![])?;

    // Link: global anchor -> stake
    let all_anchor = all_stakes_anchor()?;
    create_link(all_anchor, hash.clone(), LinkTypes::AllStakes, vec![])?;

    Ok(hash)
}

/// List all active apprenticeship stakes (for browse page).
#[hdk_extern]
pub fn list_all_stakes(_: ()) -> ExternResult<Vec<ApprenticeshipStake>> {
    let anchor = all_stakes_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllStakes)?,
        GetStrategy::Local,
    )?;

    let mut stakes = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(stake) = ApprenticeshipStake::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if stake.status == StakeStatus::Active {
                            stakes.push(stake);
                        }
                    }
                }
            }
        }
    }
    Ok(stakes)
}

/// List stakes for a specific pathway (for pathways page bounty display).
#[hdk_extern]
pub fn list_stakes_for_pathway(pathway: String) -> ExternResult<Vec<ApprenticeshipStake>> {
    let anchor = pathway_anchor(&pathway)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::PathwayToStakes)?,
        GetStrategy::Local,
    )?;

    let mut stakes = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(stake) = ApprenticeshipStake::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if stake.status == StakeStatus::Active {
                            stakes.push(stake);
                        }
                    }
                }
            }
        }
    }
    Ok(stakes)
}

/// Claim an apprenticeship slot (learner has met PoL threshold).
///
/// Increments current_apprentices. Does NOT create the job application —
/// that's done separately via the applications zome. This just reserves
/// the slot and records the learner's claim.
#[hdk_extern]
pub fn claim_apprenticeship(stake_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(stake_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Stake not found".into())))?;

    let mut stake: ApprenticeshipStake = match record.entry().as_option() {
        Some(Entry::App(bytes)) => ApprenticeshipStake::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    if stake.status != StakeStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest("Stake is not active".into())));
    }

    if stake.current_apprentices >= stake.max_apprentices {
        return Err(wasm_error!(WasmErrorInner::Guest("All apprenticeship slots are filled".into())));
    }

    stake.current_apprentices += 1;
    if stake.current_apprentices >= stake.max_apprentices {
        stake.status = StakeStatus::Full;
    }

    update_entry(stake_hash, &stake)
}
