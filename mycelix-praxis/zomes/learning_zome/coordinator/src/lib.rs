// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Coordinator Zome
//!
//! Implements business logic for learning courses and progress tracking.
//! This zome is upgradeable - business logic can change without breaking data.

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use learning_integrity::{Course, LearnerProgress, LearningActivity, EntryTypes, LinkTypes};

/// Create a new course
#[hdk_extern]
pub fn create_course(course: Course) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Steward tier to create courses
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_constitutional(),
        "create_course",
    )?;

    let action_hash = create_entry(EntryTypes::Course(course))?;

    // Create a path anchor for listing all courses
    let path = Path::from("all_courses");
    let path_hash = ensure_path(path, LinkTypes::AllCourses)?;
    create_link(path_hash, action_hash.clone(), LinkTypes::AllCourses, ())?;

    Ok(action_hash)
}

/// Get a course by its action hash
#[hdk_extern]
pub fn get_course(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// List all courses using the path anchor
#[hdk_extern]
pub fn list_courses(_: ()) -> ExternResult<Vec<Record>> {
    // Get the path anchor for all courses
    let path = Path::from("all_courses");
    let path_hash = ensure_path(path, LinkTypes::AllCourses)?;

    // Get all links from the anchor
    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllCourses)?,
        GetStrategy::Local
    )?;

    // Fetch each course record
    let mut courses = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            courses.push(record);
        }
    }

    Ok(courses)
}

/// Update an existing course
#[hdk_extern]
pub fn update_course(input: UpdateCourseInput) -> ExternResult<ActionHash> {
    let updated_action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::Course(input.updated_course),
    )?;
    Ok(updated_action_hash)
}

/// Delete a course
#[hdk_extern]
pub fn delete_course(action_hash: ActionHash) -> ExternResult<ActionHash> {
    delete_entry(action_hash)
}

/// Record learner progress
#[hdk_extern]
pub fn update_progress(progress: LearnerProgress) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::LearnerProgress(progress))?;
    Ok(action_hash)
}

/// Get learner progress
#[hdk_extern]
pub fn get_progress(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Record a learning activity (private entry for analytics)
#[hdk_extern]
pub fn record_activity(activity: LearningActivity) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::LearningActivity(activity))?;
    Ok(action_hash)
}

/// Enroll in a course
#[hdk_extern]
pub fn enroll(course_action_hash: ActionHash) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Create bidirectional links for enrollment
    // Course -> Agent
    create_link(
        course_action_hash.clone(),
        agent.clone(),
        LinkTypes::CourseToEnrolled,
        (),
    )?;

    // Agent -> Course
    create_link(
        agent,
        course_action_hash,
        LinkTypes::EnrolledCourses,
        (),
    )?;

    Ok(())
}

/// Get all courses a learner is enrolled in
#[hdk_extern]
pub fn get_enrolled_courses(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Get all links from agent to courses
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::EnrolledCourses)?,
        GetStrategy::Local
    )?;

    // Fetch each course record
    let mut courses = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            courses.push(record);
        }
    }

    Ok(courses)
}

/// Get all learners enrolled in a course
#[hdk_extern]
pub fn get_course_enrollments(course_action_hash: ActionHash) -> ExternResult<Vec<AgentPubKey>> {
    let links = get_links(
        LinkQuery::try_new(course_action_hash, LinkTypes::CourseToEnrolled)?,
        GetStrategy::Local
    )?;

    let agents = links
        .into_iter()
        .filter_map(|link| AgentPubKey::try_from(link.target).ok())
        .collect();

    Ok(agents)
}

/// Input structure for updating a course
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCourseInput {
    pub original_action_hash: ActionHash,
    pub updated_course: Course,
}

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}
