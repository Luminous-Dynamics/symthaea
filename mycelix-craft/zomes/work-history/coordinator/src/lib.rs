#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Work History Coordinator Zome
//!
//! Manages work experience entries with peer verification.
//! Verification uses link-based attestation (existence of link = verified).

use hdk::prelude::*;
use work_history_integrity::{
    EntryTypes, LinkTypes, OrgAnchor, WorkExperience, WorkVerification,
};

// ============== Helpers ==============

fn org_anchor(org: &str) -> ExternResult<EntryHash> {
    let anchor = OrgAnchor(org.to_lowercase());
    create_entry(EntryTypes::OrgAnchor(anchor.clone()))?;
    hash_entry(anchor)
}

fn load_experiences(links: Vec<Link>) -> ExternResult<Vec<WorkExperience>> {
    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(exp) = WorkExperience::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        items.push(exp);
                    }
                }
            }
        }
    }
    Ok(items)
}

// ============== Input Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct AddWorkExperienceInput {
    pub title: String,
    pub organization: String,
    pub start_date: String,
    pub end_date: Option<String>,
    pub description: String,
    pub skills_used: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyWorkExperienceInput {
    pub experience_hash: ActionHash,
    pub relationship: String,
    pub rationale: String,
}

// ============== Extern Functions ==============

/// Add a work experience to the agent's timeline.
#[hdk_extern]
pub fn add_work_experience(input: AddWorkExperienceInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let exp = WorkExperience {
        title: input.title,
        organization: input.organization.clone(),
        start_date: input.start_date,
        end_date: input.end_date,
        description: input.description,
        skills_used: input.skills_used.iter().map(|s| s.to_lowercase()).collect(),
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::WorkExperience(exp))?;

    // Link: agent -> experience
    let agent_hash: AnyDhtHash = agent.into();
    create_link(agent_hash, action_hash.clone(), LinkTypes::AgentToWorkExperience, vec![])?;

    // Link: org anchor -> experience
    let org_hash = org_anchor(&input.organization)?;
    create_link(org_hash, action_hash.clone(), LinkTypes::OrgAnchorToWorkExperience, vec![])?;

    Ok(action_hash)
}

/// Update an existing work experience (author-only at integrity layer).
#[hdk_extern]
pub fn update_work_experience(input: (ActionHash, AddWorkExperienceInput)) -> ExternResult<ActionHash> {
    let (original_hash, update) = input;
    let now = sys_time()?;

    let exp = WorkExperience {
        title: update.title,
        organization: update.organization,
        start_date: update.start_date,
        end_date: update.end_date,
        description: update.description,
        skills_used: update.skills_used.iter().map(|s| s.to_lowercase()).collect(),
        created_at: now,
    };

    update_entry(original_hash, &exp)
}

/// List work history for the calling agent.
#[hdk_extern]
pub fn list_my_work_history(_: ()) -> ExternResult<Vec<WorkExperience>> {
    let agent = agent_info()?.agent_initial_pubkey;
    list_work_history(agent)
}

/// List work history for any agent.
#[hdk_extern]
pub fn list_work_history(agent: AgentPubKey) -> ExternResult<Vec<WorkExperience>> {
    let agent_hash: AnyDhtHash = agent.into();
    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToWorkExperience)?,
        GetStrategy::Local,
    )?;
    load_experiences(links)
}

/// Get a single work experience by hash.
#[hdk_extern]
pub fn get_work_experience(action_hash: ActionHash) -> ExternResult<Option<WorkExperience>> {
    let Some(record) = get(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            let exp = WorkExperience::try_from(
                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
            ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?;
            Ok(Some(exp))
        }
        _ => Ok(None),
    }
}

/// Verify a work experience (peer attestation via link).
#[hdk_extern]
pub fn verify_work_experience(input: VerifyWorkExperienceInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let verification = WorkVerification {
        experience_hash_b64: input.experience_hash.to_string(),
        verifier: agent.to_string(),
        relationship: input.relationship,
        rationale: input.rationale,
        verified_at: now,
    };

    let verification_hash = create_entry(EntryTypes::WorkVerification(verification))?;

    // Link: experience -> verification (existence = verified)
    create_link(
        input.experience_hash,
        verification_hash.clone(),
        LinkTypes::ExperienceToVerification,
        vec![],
    )?;

    Ok(verification_hash)
}

/// Check if a work experience has been verified (link-counting).
#[hdk_extern]
pub fn is_experience_verified(experience_hash: ActionHash) -> ExternResult<bool> {
    let links = get_links(
        LinkQuery::try_new(experience_hash, LinkTypes::ExperienceToVerification)?,
        GetStrategy::Local,
    )?;
    Ok(!links.is_empty())
}

/// Count verifications for a work experience.
#[hdk_extern]
pub fn verification_count(experience_hash: ActionHash) -> ExternResult<u32> {
    let links = get_links(
        LinkQuery::try_new(experience_hash, LinkTypes::ExperienceToVerification)?,
        GetStrategy::Local,
    )?;
    Ok(links.len() as u32)
}
