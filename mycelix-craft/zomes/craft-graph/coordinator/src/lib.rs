// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Craft Graph Coordinator Zome
//!
//! Implements business logic for the craft/workforce graph.

use craft_graph_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers as _;

/// Create or update the agent's craft profile.
#[hdk_extern]
pub fn create_profile(input: CreateProfileInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let profile = CraftProfile {
        agent_did: agent.to_string(),
        display_name: input.display_name,
        headline: input.headline,
        bio: input.bio,
        location: input.location,
        website: input.website,
        avatar_url: input.avatar_url,
        primary_skill: input.primary_skill,
        mastery_level: input.mastery_level,
        endorsements_count: 0,
        updated_at: now,
    };

    let action_hash = create_entry(EntryTypes::CraftProfile(profile))?;

    // Link from agent to profile
    create_link(agent, action_hash.clone(), LinkTypes::AgentToProfile, ())?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProfileInput {
    pub display_name: String,
    pub headline: String,
    pub bio: String,
    pub location: String,
    pub website: String,
    pub avatar_url: String,
    pub primary_skill: String,
    pub mastery_level: u16,
}

/// Publish a credential to the craft graph.
#[hdk_extern]
pub fn publish_credential(input: PublishCredentialInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let credential = PublishedCredential {
        credential_hash: input.credential_hash.clone(),
        issuer_did: input.issuer_did.clone(),
        issuer: input.issuer_did,
        visibility: input.visibility,
        title: input.title,
        summary: input.summary,
        mastery_level_at_issue: Some(input.mastery_permille),
        last_retention_check: None,
        issued_on: format!("{:?}", now),
        expires_on: None,
        source_dna: "none".into(),
        entry_hash: "none".into(),
        action_hash: input.credential_hash.to_string(),
        vitality_permille: Some(1000),
        guild_id: input.guild_id,
        guild_name: input.guild_name,
        epistemic_code: input.epistemic_code,
        fl_model_version: input.fl_model_version,
        mastery_permille: Some(input.mastery_permille),
        verified: None,
    };

    let action_hash = create_entry(EntryTypes::PublishedCredential(credential))?;

    // Link from agent to credential
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToCredential, ())?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PublishCredentialInput {
    pub credential_hash: ActionHash,
    pub issuer_did: String,
    pub visibility: String,
    pub title: String,
    pub summary: Option<String>,
    pub mastery_permille: u16,
    pub guild_id: Option<String>,
    pub guild_name: Option<String>,
    pub epistemic_code: Option<String>,
    pub fl_model_version: Option<String>,
}

/// Create a skill endorsement.
#[hdk_extern]
pub fn create_skill_endorsement(input: CreateSkillEndorsementInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let endorsement = SkillEndorsement {
        subject_did: input.subject_did,
        endorsed_agent: input.endorsed_agent,
        skill: input.skill,
        weight: input.weight,
        rationale: input.rationale,
        evidence: input.evidence,
        timestamp: now.as_micros() as i64,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::SkillEndorsement(endorsement))?;

    // Link from agent to endorsement
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::AgentToEndorsement,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSkillEndorsementInput {
    pub subject_did: String,
    pub endorsed_agent: AgentPubKey,
    pub skill: String,
    pub weight: u16,
    pub rationale: String,
    pub evidence: String,
}

/// Create a retention check.
#[hdk_extern]
pub fn create_retention_check(input: CreateRetentionCheckInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let check = RetentionCheck {
        agent: agent_info()?.agent_initial_pubkey,
        skill: input.skill,
        credential_id: input.credential_hash.to_string(),
        retention_score_permille: input.retention_score_permille,
        questions_attempted: input.questions_attempted,
        questions_correct: input.questions_correct,
        timestamp: now.as_micros() as i64,
        checked_at: now,
    };

    let action_hash = create_entry(EntryTypes::RetentionCheck(check))?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRetentionCheckInput {
    pub skill: String,
    pub credential_hash: ActionHash,
    pub retention_score_permille: u16,
    pub questions_attempted: u16,
    pub questions_correct: u16,
}

/// Create a composite profile.
#[hdk_extern]
pub fn create_composite_profile(input: CreateCompositeProfileInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let profile = CompositeProfile {
        identity_hash: input.identity_hash,
        workforce_hash: input.workforce_hash,
        agent: agent_info()?.agent_initial_pubkey,
        archetype_name: input.archetype_name,
        credential_titles: input.credential_titles,
        coverage_permille: input.coverage_permille,
        career_profile_match: Some("Matched".into()),
        detected_at: now,
    };

    let action_hash = create_entry(EntryTypes::CompositeProfile(profile))?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCompositeProfileInput {
    pub identity_hash: ActionHash,
    pub workforce_hash: ActionHash,
    pub archetype_name: String,
    pub credential_titles: Vec<String>,
    pub coverage_permille: u16,
}
