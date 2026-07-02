// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Integrity Zome
//!
//! Defines entry types and validation rules for the learning domain.
//! This zome is immutable - it defines the data structures that cannot change.

use hdi::prelude::*;
use praxis_core::CourseId;

/// Course entry - represents a learning course
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Course {
    /// Unique identifier
    pub course_id: CourseId,

    /// Course title
    pub title: String,

    /// Description
    pub description: String,

    /// Course creator agent
    pub creator: String,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Model ID for personalization (if using FL)
    pub model_id: Option<String>,

    /// Creation timestamp
    pub created_at: i64,

    /// Last updated timestamp
    pub updated_at: i64,

    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Learner progress entry - tracks individual learner progress
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LearnerProgress {
    /// Course being studied
    pub course_id: CourseId,

    /// Learner agent
    pub learner: String,

    /// Progress percentage (0-100)
    pub progress_percent: f32,

    /// Completed modules/lessons
    pub completed_items: Vec<String>,

    /// Current learning model version
    pub model_version: Option<String>,

    /// Last activity timestamp
    pub last_active: i64,

    /// Metadata (scores, time spent, etc.)
    pub metadata: Option<serde_json::Value>,
}

/// Learning activity entry - for privacy-preserving analytics
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LearningActivity {
    /// Course context
    pub course_id: CourseId,

    /// Activity type (view, quiz, exercise, etc.)
    pub activity_type: String,

    /// Item identifier (lesson ID, quiz ID, etc.)
    pub item_id: String,

    /// Outcome (score, completion, etc.)
    pub outcome: Option<f32>,

    /// Time spent (seconds)
    pub duration_secs: u32,

    /// Timestamp
    pub timestamp: i64,
}

/// All entry types for the learning domain
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    Course(Course),
    LearnerProgress(LearnerProgress),
    #[entry_type(visibility = "private")]
    LearningActivity(LearningActivity),
}

/// Link types for course relationships
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from anchor "all_courses" to Course entries (for listing)
    AllCourses,

    /// Link from Course to enrolled agents (course -> learner)
    CourseToEnrolled,

    /// Link from agent to enrolled courses (learner -> course)
    EnrolledCourses,

    /// Link from Course to LearnerProgress (course -> progress)
    CourseToProgress,
}

/// Validation callback for all learning entries
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => validate_store_entry(store_entry),
        Op::RegisterUpdate(register_update) => validate_update_entry(register_update),
        Op::RegisterDelete(register_delete) => validate_delete_entry(register_delete),
        Op::RegisterCreateLink(register_create_link) => validate_create_link(register_create_link),
        Op::RegisterDeleteLink(register_delete_link) => validate_delete_link(register_delete_link),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_store_entry(store_entry: StoreEntry) -> ExternResult<ValidateCallbackResult> {
    let entry = store_entry.entry;

    match entry {
        Entry::App(app_entry_bytes) => {
            let bytes = SerializedBytes::from(UnsafeBytes::from(app_entry_bytes.bytes().clone()));

            // Try to deserialize as each entry type
            if let Ok(course) = Course::try_from(bytes.clone()) {
                validate_course(&course)
            } else if let Ok(progress) = LearnerProgress::try_from(bytes.clone()) {
                validate_learner_progress(&progress)
            } else if let Ok(activity) = LearningActivity::try_from(bytes) {
                validate_learning_activity(&activity)
            } else {
                Ok(ValidateCallbackResult::Invalid("Unknown entry type".to_string()))
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_update_entry(register_update: RegisterUpdate) -> ExternResult<ValidateCallbackResult> {
    let entry = register_update.new_entry;

    match entry {
        Some(Entry::App(app_entry_bytes)) => {
            let bytes = SerializedBytes::from(UnsafeBytes::from(app_entry_bytes.bytes().clone()));

            // Validate the new entry
            if let Ok(course) = Course::try_from(bytes.clone()) {
                validate_course(&course)
            } else if let Ok(progress) = LearnerProgress::try_from(bytes) {
                validate_learner_progress(&progress)
            } else {
                Ok(ValidateCallbackResult::Invalid("Cannot update activity entries".to_string()))
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_delete_entry(_register_delete: RegisterDelete) -> ExternResult<ValidateCallbackResult> {
    // Only allow deletion by the original author
    // This is enforced by Holochain's source chain validation
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_link(register_create_link: RegisterCreateLink) -> ExternResult<ValidateCallbackResult> {
    let link_type = match LinkTypes::from_type(
        register_create_link.create_link.hashed.content.zome_index,
        register_create_link.create_link.hashed.content.link_type,
    )? {
        Some(lt) => lt,
        None => {
            return Ok(ValidateCallbackResult::Invalid(
                "Invalid link type".to_string(),
            ));
        }
    };

    match link_type {
        LinkTypes::AllCourses => validate_all_courses_link(&register_create_link),
        LinkTypes::CourseToEnrolled => validate_course_to_enrolled_link(&register_create_link),
        LinkTypes::EnrolledCourses => validate_enrolled_courses_link(&register_create_link),
        LinkTypes::CourseToProgress => validate_course_to_progress_link(&register_create_link),
    }
}

fn validate_delete_link(register_delete_link: RegisterDeleteLink) -> ExternResult<ValidateCallbackResult> {
    let link_type = match LinkTypes::from_type(
        register_delete_link.create_link.zome_index,
        register_delete_link.create_link.link_type,
    )? {
        Some(lt) => lt,
        None => {
            return Ok(ValidateCallbackResult::Invalid(
                "Invalid link type".to_string(),
            ));
        }
    };

    match link_type {
        LinkTypes::AllCourses => {
            // Only course creator can remove from all courses list
            // This is enforced by source chain validation
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CourseToEnrolled | LinkTypes::EnrolledCourses => {
            // Allow unenrollment (link deletion)
            // Both sides of bidirectional link should be deleted together
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CourseToProgress => {
            // Progress links shouldn't be deleted (maintain history)
            Ok(ValidateCallbackResult::Invalid(
                "Cannot delete progress history links".to_string(),
            ))
        }
    }
}

/// Validate AllCourses link (from "all_courses" path to Course)
/// Note: ensure_path() also creates internal path links with EntryHash targets,
/// so we accept both ActionHash (course links) and EntryHash (path structure links)
fn validate_all_courses_link(link: &RegisterCreateLink) -> ExternResult<ValidateCallbackResult> {
    // Base should be an EntryHash (path entry)
    let _base_hash = EntryHash::try_from(link.create_link.hashed.content.base_address.clone())
        .map_err(|_| wasm_error!("AllCourses link base must be an EntryHash (path)"))?;

    // Target can be either:
    // 1. ActionHash - pointing to a Course entry (the intended use)
    // 2. EntryHash - path structure links created by ensure_path()
    let target = &link.create_link.hashed.content.target_address;
    let is_action = ActionHash::try_from(target.clone()).is_ok();
    let is_entry = EntryHash::try_from(target.clone()).is_ok();

    if !is_action && !is_entry {
        return Err(wasm_error!("AllCourses link target must be ActionHash or EntryHash"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CourseToEnrolled link (from Course to Agent)
fn validate_course_to_enrolled_link(link: &RegisterCreateLink) -> ExternResult<ValidateCallbackResult> {
    // Base should be ActionHash of Course entry
    let _course_hash = ActionHash::try_from(link.create_link.hashed.content.base_address.clone())
        .map_err(|_| wasm_error!("CourseToEnrolled base must be a Course ActionHash"))?;

    // Target should be AgentPubKey
    let _agent = AgentPubKey::try_from(link.create_link.hashed.content.target_address.clone())
        .map_err(|_| wasm_error!("CourseToEnrolled target must be an AgentPubKey"))?;

    // TODO: Could check for duplicate enrollments by querying existing links
    // This would require DHT access and could be added in coordinator validation

    Ok(ValidateCallbackResult::Valid)
}

/// Validate EnrolledCourses link (from Agent to Course)
fn validate_enrolled_courses_link(link: &RegisterCreateLink) -> ExternResult<ValidateCallbackResult> {
    // Base should be AgentPubKey
    let _agent = AgentPubKey::try_from(link.create_link.hashed.content.base_address.clone())
        .map_err(|_| wasm_error!("EnrolledCourses base must be an AgentPubKey"))?;

    // Target should be ActionHash of Course entry
    let _course_hash = ActionHash::try_from(link.create_link.hashed.content.target_address.clone())
        .map_err(|_| wasm_error!("EnrolledCourses target must be a Course ActionHash"))?;

    // Verify link author is the agent being enrolled
    let link_author = link.create_link.hashed.content.author.clone();
    let base_agent = AgentPubKey::try_from(link.create_link.hashed.content.base_address.clone())
        .map_err(|_| wasm_error!("Failed to extract agent from base"))?;

    if link_author != base_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Only agents can enroll themselves".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CourseToProgress link (from Course to LearnerProgress)
fn validate_course_to_progress_link(link: &RegisterCreateLink) -> ExternResult<ValidateCallbackResult> {
    // Base should be ActionHash of Course entry
    let _course_hash = ActionHash::try_from(link.create_link.hashed.content.base_address.clone())
        .map_err(|_| wasm_error!("CourseToProgress base must be a Course ActionHash"))?;

    // Target should be ActionHash of LearnerProgress entry
    let _progress_hash = ActionHash::try_from(link.create_link.hashed.content.target_address.clone())
        .map_err(|_| wasm_error!("CourseToProgress target must be a LearnerProgress ActionHash"))?;

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Course entry
fn validate_course(course: &Course) -> ExternResult<ValidateCallbackResult> {
    // Title must not be empty
    if course.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Course title cannot be empty".to_string()));
    }

    // Title must be reasonable length (1-200 chars)
    if course.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Course title too long (max 200 characters)".to_string()));
    }

    // Description must not be empty
    if course.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Course description cannot be empty".to_string()));
    }

    // Description has reasonable length (max 5000 chars)
    if course.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid("Course description too long (max 5000 characters)".to_string()));
    }

    // Updated timestamp must be >= created timestamp
    if course.updated_at < course.created_at {
        return Ok(ValidateCallbackResult::Invalid("Update time cannot be before creation time".to_string()));
    }

    // Validate tags (max 10, each max 50 chars)
    if course.tags.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid("Too many tags (max 10)".to_string()));
    }

    for tag in &course.tags {
        if tag.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid("Tags cannot be empty".to_string()));
        }
        if tag.len() > 50 {
            return Ok(ValidateCallbackResult::Invalid("Tag too long (max 50 characters)".to_string()));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a LearnerProgress entry
fn validate_learner_progress(progress: &LearnerProgress) -> ExternResult<ValidateCallbackResult> {
    // Progress percentage must be 0-100
    if !progress.progress_percent.is_finite() || progress.progress_percent < 0.0 || progress.progress_percent > 100.0 {
        return Ok(ValidateCallbackResult::Invalid("Progress must be between 0 and 100".to_string()));
    }

    // Completed items must be reasonable (max 1000)
    if progress.completed_items.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid("Too many completed items (max 1000)".to_string()));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a LearningActivity entry
fn validate_learning_activity(activity: &LearningActivity) -> ExternResult<ValidateCallbackResult> {
    // Activity type must not be empty
    if activity.activity_type.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Activity type cannot be empty".to_string()));
    }

    // Item ID must not be empty
    if activity.item_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Item ID cannot be empty".to_string()));
    }

    // Duration must be reasonable (max 24 hours = 86400 seconds)
    if activity.duration_secs > 86400 {
        return Ok(ValidateCallbackResult::Invalid("Activity duration too long (max 24 hours)".to_string()));
    }

    // Outcome (if present) should be 0-100 for scores
    if let Some(outcome) = activity.outcome {
        if !outcome.is_finite() || outcome < 0.0 || outcome > 100.0 {
            return Ok(ValidateCallbackResult::Invalid("Activity outcome must be between 0 and 100".to_string()));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

// =================================================================================
// Unit Tests
// =================================================================================

#[cfg(test)]
mod tests;
