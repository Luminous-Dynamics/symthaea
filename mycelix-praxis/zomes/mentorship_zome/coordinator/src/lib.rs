// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mentorship Matching Coordinator Zome
//!
//! Implements business logic for mentor-mentee pairing, session tracking,
//! and mentorship lifecycle management within EduNet.
//!
//! ## Consciousness Gating
//!
//! Registering as a mentor or accepting mentorships requires at least
//! Participant tier. Completing mentorships (which writes ratings) requires
//! Citizen tier.

#![deny(unsafe_code)]

use hdk::prelude::*;
use mentorship_integrity::*;

// ============== Helper Functions ==============

/// Ensure a path exists and return its entry hash.
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed_path = path.typed(link_type)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Fetch a record by ActionHash, returning a WasmError on not-found.
fn get_record(hash: ActionHash) -> ExternResult<Record> {
    get(hash.clone(), GetOptions::default())?.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Record not found: {:?}",
            hash
        )))
    })
}

/// Collect Records from a list of link targets.
fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::with_capacity(links.len());
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Link target is not an ActionHash".to_string())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============== Input Structs ==============

/// Input for proposing a new mentorship
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposeMentorshipInput {
    /// ActionHash of the mentor's profile
    pub mentor_profile_hash: ActionHash,
    /// ActionHash of the mentee's profile
    pub mentee_profile_hash: ActionHash,
    /// The skill to focus on
    pub skill_focus: String,
}

/// Input for completing a mentorship with mutual ratings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompleteMentorshipInput {
    /// ActionHash of the mentorship to complete
    pub mentorship_hash: ActionHash,
    /// Rating given by the mentee to the mentor (0.0 - 5.0)
    pub mentee_rating: Option<f64>,
    /// Rating given by the mentor to the mentee (0.0 - 5.0)
    pub mentor_rating: Option<f64>,
}

// ============== Mentor Registration ==============

/// Register the calling agent as a mentor.
///
/// Requires at least Participant consciousness tier.
#[hdk_extern]
pub fn register_as_mentor(input: MentorProfile) -> ExternResult<ActionHash> {
    // Consciousness gate: must be at least Participant
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_basic(),
        "register_as_mentor",
    )?;

    let action_hash = create_entry(EntryTypes::MentorProfile(input.clone()))?;

    // Link to "all_mentors" anchor
    let mentors_path = Path::from("all_mentors");
    let mentors_anchor = ensure_path(mentors_path, LinkTypes::AllMentors)?;
    create_link(mentors_anchor, action_hash.clone(), LinkTypes::AllMentors, ())?;

    // Link each skill to this mentor for skill-based lookup
    for skill in &input.skills {
        let skill_path = Path::from(format!("skill.{}", skill.to_lowercase()));
        let skill_anchor = ensure_path(skill_path, LinkTypes::SkillToMentor)?;
        create_link(skill_anchor, action_hash.clone(), LinkTypes::SkillToMentor, ())?;
    }

    Ok(action_hash)
}

// ============== Mentee Registration ==============

/// Register the calling agent as a mentee.
///
/// Requires at least Participant consciousness tier.
#[hdk_extern]
pub fn register_as_mentee(input: MenteeProfile) -> ExternResult<ActionHash> {
    // Consciousness gate: must be at least Participant
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_basic(),
        "register_as_mentee",
    )?;

    let action_hash = create_entry(EntryTypes::MenteeProfile(input))?;

    // Link to "all_mentees" anchor
    let mentees_path = Path::from("all_mentees");
    let mentees_anchor = ensure_path(mentees_path, LinkTypes::AllMentees)?;
    create_link(mentees_anchor, action_hash.clone(), LinkTypes::AllMentees, ())?;

    Ok(action_hash)
}

// ============== Profile Retrieval ==============

/// Get a mentor profile by its ActionHash.
#[hdk_extern]
pub fn get_mentor(hash: ActionHash) -> ExternResult<Record> {
    get_record(hash)
}

/// Get a mentee profile by its ActionHash.
#[hdk_extern]
pub fn get_mentee(hash: ActionHash) -> ExternResult<Record> {
    get_record(hash)
}

/// Find all mentors who offer a specific skill.
#[hdk_extern]
pub fn find_mentors_by_skill(skill: String) -> ExternResult<Vec<Record>> {
    let skill_path = Path::from(format!("skill.{}", skill.to_lowercase()));
    let skill_anchor = ensure_path(skill_path, LinkTypes::SkillToMentor)?;

    let links = get_links(
        LinkQuery::try_new(skill_anchor, LinkTypes::SkillToMentor)?,
        GetStrategy::Local,
    )?;

    records_from_links(links)
}

/// Get all registered mentors.
#[hdk_extern]
pub fn get_all_mentors(_: ()) -> ExternResult<Vec<Record>> {
    let mentors_path = Path::from("all_mentors");
    let mentors_anchor = ensure_path(mentors_path, LinkTypes::AllMentors)?;

    let links = get_links(
        LinkQuery::try_new(mentors_anchor, LinkTypes::AllMentors)?,
        GetStrategy::Local,
    )?;

    records_from_links(links)
}

// ============== Mentorship Lifecycle ==============

/// Propose a new mentorship between a mentor and mentee.
///
/// Creates a Mentorship entry in Proposed status and links it to both profiles.
/// Requires at least Participant consciousness tier.
#[hdk_extern]
pub fn propose_mentorship(input: ProposeMentorshipInput) -> ExternResult<ActionHash> {
    // Consciousness gate
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_basic(),
        "propose_mentorship",
    )?;

    // Fetch mentor and mentee profiles to extract DIDs
    let mentor_record = get_record(input.mentor_profile_hash.clone())?;
    let mentor_profile: MentorProfile = mentor_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize mentor profile: {}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Mentor profile entry not found".to_string())))?;

    let mentee_record = get_record(input.mentee_profile_hash.clone())?;
    let mentee_profile: MenteeProfile = mentee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize mentee profile: {}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Mentee profile entry not found".to_string())))?;

    // Check mentor capacity
    if mentor_profile.availability != MentorAvailability::Available {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Mentor is not currently available".to_string()
        )));
    }
    if mentor_profile.active_mentees >= mentor_profile.max_mentees {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Mentor has reached maximum mentee capacity".to_string()
        )));
    }

    let mentorship = Mentorship {
        mentor_did: mentor_profile.mentor_did,
        mentee_did: mentee_profile.mentee_did,
        skill_focus: input.skill_focus,
        status: MentorshipStatus::Proposed,
        started_at: None,
        completed_at: None,
        sessions_completed: 0,
        mentee_rating: None,
        mentor_rating: None,
    };

    let action_hash = create_entry(EntryTypes::Mentorship(mentorship))?;

    // Link mentor profile -> mentorship
    create_link(
        input.mentor_profile_hash,
        action_hash.clone(),
        LinkTypes::MentorToMentorship,
        (),
    )?;

    // Link mentee profile -> mentorship
    create_link(
        input.mentee_profile_hash,
        action_hash.clone(),
        LinkTypes::MenteeToMentorship,
        (),
    )?;

    Ok(action_hash)
}

/// Accept a proposed mentorship, transitioning it to Active status.
///
/// Requires at least Participant consciousness tier.
#[hdk_extern]
pub fn accept_mentorship(mentorship_hash: ActionHash) -> ExternResult<ActionHash> {
    // Consciousness gate
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_basic(),
        "accept_mentorship",
    )?;

    let record = get_record(mentorship_hash.clone())?;
    let mut mentorship: Mentorship = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize mentorship: {}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Mentorship entry not found".to_string())))?;

    if mentorship.status != MentorshipStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only proposed mentorships can be accepted".to_string()
        )));
    }

    mentorship.status = MentorshipStatus::Active;
    mentorship.started_at = Some(sys_time()?);

    let new_hash = update_entry(mentorship_hash, EntryTypes::Mentorship(mentorship))?;
    Ok(new_hash)
}

/// Record a session within an active mentorship.
///
/// Requires at least Participant consciousness tier.
#[hdk_extern]
pub fn record_session(input: MentorshipSession) -> ExternResult<ActionHash> {
    // Consciousness gate
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_basic(),
        "record_session",
    )?;

    // Verify the mentorship exists and is active
    let mentorship_record = get_record(input.mentorship_id.clone())?;
    let mentorship: Mentorship = mentorship_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize mentorship: {}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Mentorship entry not found".to_string())))?;

    if mentorship.status != MentorshipStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Sessions can only be recorded for active mentorships".to_string()
        )));
    }

    let mentorship_id = input.mentorship_id.clone();
    let action_hash = create_entry(EntryTypes::MentorshipSession(input))?;

    // Link mentorship -> session
    create_link(
        mentorship_id,
        action_hash.clone(),
        LinkTypes::MentorshipToSession,
        (),
    )?;

    Ok(action_hash)
}

/// Get all sessions for a given mentorship.
#[hdk_extern]
pub fn get_mentorship_sessions(mentorship_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(mentorship_hash, LinkTypes::MentorshipToSession)?,
        GetStrategy::Local,
    )?;

    records_from_links(links)
}

/// Complete a mentorship with optional mutual ratings.
///
/// Requires at least Citizen consciousness tier (modifies shared state with ratings).
#[hdk_extern]
pub fn complete_mentorship(input: CompleteMentorshipInput) -> ExternResult<ActionHash> {
    // Consciousness gate: Citizen tier for completing (writes ratings)
    mycelix_bridge_common::gate_civic(
        "mentorship_zome",
        &mycelix_bridge_common::civic_requirement_voting(),
        "complete_mentorship",
    )?;

    let record = get_record(input.mentorship_hash.clone())?;
    let mut mentorship: Mentorship = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize mentorship: {}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Mentorship entry not found".to_string())))?;

    if mentorship.status != MentorshipStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active mentorships can be completed".to_string()
        )));
    }

    // Validate ratings if provided
    if let Some(r) = input.mentee_rating {
        if r < 0.0 || r > 5.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Mentee rating must be between 0.0 and 5.0".to_string()
            )));
        }
    }
    if let Some(r) = input.mentor_rating {
        if r < 0.0 || r > 5.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Mentor rating must be between 0.0 and 5.0".to_string()
            )));
        }
    }

    // Count sessions
    let sessions = get_links(
        LinkQuery::try_new(input.mentorship_hash.clone(), LinkTypes::MentorshipToSession)?,
        GetStrategy::Local,
    )?;

    mentorship.status = MentorshipStatus::Completed;
    mentorship.completed_at = Some(sys_time()?);
    mentorship.sessions_completed = sessions.len() as u32;
    mentorship.mentee_rating = input.mentee_rating;
    mentorship.mentor_rating = input.mentor_rating;

    let new_hash = update_entry(input.mentorship_hash, EntryTypes::Mentorship(mentorship))?;
    Ok(new_hash)
}

/// Get all mentorships linked to the calling agent (as mentor or mentee).
///
/// Searches the agent's mentor and mentee profile links for mentorships.
#[hdk_extern]
pub fn get_my_mentorships(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let mut all_records = Vec::new();

    // Check mentor profile path
    let mentor_path = Path::from("all_mentors");
    let mentor_anchor = ensure_path(mentor_path, LinkTypes::AllMentors)?;
    let mentor_links = get_links(
        LinkQuery::try_new(mentor_anchor, LinkTypes::AllMentors)?,
        GetStrategy::Local,
    )?;

    // For each mentor profile, check if it belongs to this agent and get mentorships
    for link in mentor_links {
        let target = match link.target.into_action_hash() {
            Some(h) => h,
            None => continue,
        };
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            // Check if this profile was authored by the calling agent
            if record.action().author() == &agent {
                let mentorship_links = get_links(
                    LinkQuery::try_new(target, LinkTypes::MentorToMentorship)?,
                    GetStrategy::Local,
                )?;
                let mut mentorship_records = records_from_links(mentorship_links)?;
                all_records.append(&mut mentorship_records);
            }
        }
    }

    // Check mentee profile path
    let mentee_path = Path::from("all_mentees");
    let mentee_anchor = ensure_path(mentee_path, LinkTypes::AllMentees)?;
    let mentee_links = get_links(
        LinkQuery::try_new(mentee_anchor, LinkTypes::AllMentees)?,
        GetStrategy::Local,
    )?;

    for link in mentee_links {
        let target = match link.target.into_action_hash() {
            Some(h) => h,
            None => continue,
        };
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            if record.action().author() == &agent {
                let mentorship_links = get_links(
                    LinkQuery::try_new(target, LinkTypes::MenteeToMentorship)?,
                    GetStrategy::Local,
                )?;
                let mut mentorship_records = records_from_links(mentorship_links)?;
                all_records.append(&mut mentorship_records);
            }
        }
    }

    Ok(all_records)
}

/// Get all mentorships that are currently Active.
#[hdk_extern]
pub fn get_active_mentorships(_: ()) -> ExternResult<Vec<Record>> {
    let my_mentorships = get_my_mentorships(())?;

    let active: Vec<Record> = my_mentorships
        .into_iter()
        .filter(|record| {
            record
                .entry()
                .to_app_option::<Mentorship>()
                .ok()
                .flatten()
                .map(|m| m.status == MentorshipStatus::Active)
                .unwrap_or(false)
        })
        .collect();

    Ok(active)
}
