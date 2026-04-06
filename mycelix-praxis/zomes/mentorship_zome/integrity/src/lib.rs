// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mentorship Matching Integrity Zome
//!
//! Defines entry types and validation rules for the mentorship matching system.
//!
//! ## Overview
//!
//! This zome enables peer-to-peer mentorship within EduNet by allowing experienced
//! members to register as mentors and learners to find guided support. Mentorships
//! are tracked through structured sessions with mutual ratings.
//!
//! ## Entry Types
//!
//! - **MentorProfile**: A member's mentor offering (skills, availability, capacity)
//! - **MenteeProfile**: A learner's mentee request (goals, level, schedule)
//! - **Mentorship**: A paired mentor-mentee relationship with lifecycle tracking
//! - **MentorshipSession**: An individual session log within a mentorship
//! - **Anchor**: Path-based anchor for link indexing

use hdi::prelude::*;

// ============== Enums ==============

/// Mentor availability status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MentorAvailability {
    /// Accepting new mentees
    Available,
    /// Temporarily not accepting new mentees
    Busy,
    /// Paused all mentorship activity
    OnPause,
}

impl Default for MentorAvailability {
    fn default() -> Self {
        MentorAvailability::Available
    }
}

/// Mentee's current skill level
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
}

impl Default for SkillLevel {
    fn default() -> Self {
        SkillLevel::Beginner
    }
}

/// Mentorship lifecycle status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MentorshipStatus {
    /// Proposed by mentee, awaiting mentor acceptance
    Proposed,
    /// Actively in progress
    Active,
    /// Successfully completed
    Completed,
    /// Cancelled by either party
    Cancelled,
}

impl Default for MentorshipStatus {
    fn default() -> Self {
        MentorshipStatus::Proposed
    }
}

// ============== Entry Types ==============

/// A mentor's profile advertising their expertise and availability.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MentorProfile {
    /// DID of the mentor agent
    pub mentor_did: String,
    /// Skills the mentor can teach
    pub skills: Vec<String>,
    /// Years of experience
    pub experience_years: u32,
    /// Short biography
    pub bio: String,
    /// Maximum number of concurrent mentees
    pub max_mentees: u32,
    /// Current number of active mentees
    pub active_mentees: u32,
    /// Current availability
    pub availability: MentorAvailability,
    /// Average rating from past mentees (0.0 - 5.0), stored as fixed-point (permille / 200)
    pub rating: f64,
    /// Total completed sessions across all mentorships
    pub total_sessions: u32,
}

/// A mentee's profile describing their learning goals.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MenteeProfile {
    /// DID of the mentee agent
    pub mentee_did: String,
    /// Learning goals
    pub goals: Vec<String>,
    /// Current skill level
    pub current_level: SkillLevel,
    /// Preferred meeting schedule (free-form text)
    pub preferred_schedule: String,
    /// Short biography
    pub bio: String,
}

/// A mentor-mentee pairing with lifecycle tracking.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Mentorship {
    /// DID of the mentor
    pub mentor_did: String,
    /// DID of the mentee
    pub mentee_did: String,
    /// The primary skill focus for this mentorship
    pub skill_focus: String,
    /// Current status
    pub status: MentorshipStatus,
    /// When the mentorship became active
    pub started_at: Option<Timestamp>,
    /// When the mentorship was completed
    pub completed_at: Option<Timestamp>,
    /// Number of sessions completed
    pub sessions_completed: u32,
    /// Rating given by mentee (0.0 - 5.0) upon completion
    pub mentee_rating: Option<f64>,
    /// Rating given by mentor (0.0 - 5.0) upon completion
    pub mentor_rating: Option<f64>,
}

/// A single session within a mentorship.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MentorshipSession {
    /// Hash of the parent Mentorship entry
    pub mentorship_id: ActionHash,
    /// When the session occurred
    pub session_date: Timestamp,
    /// Duration in minutes
    pub duration_minutes: u32,
    /// Topics covered during the session
    pub topics_covered: Vec<String>,
    /// Session notes
    pub notes: String,
    /// Description of mentee's progress
    pub mentee_progress: String,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 3, visibility = "public")]
    MentorProfile(MentorProfile),
    #[entry_type(required_validations = 1, visibility = "public")]
    MenteeProfile(MenteeProfile),
    #[entry_type(required_validations = 3, visibility = "public")]
    Mentorship(Mentorship),
    #[entry_type(required_validations = 1, visibility = "public")]
    MentorshipSession(MentorshipSession),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor -> all registered mentors
    AllMentors,
    /// Anchor -> all registered mentees
    AllMentees,
    /// Mentor profile -> their mentorships
    MentorToMentorship,
    /// Mentee profile -> their mentorships
    MenteeToMentorship,
    /// Skill anchor -> mentors offering that skill
    SkillToMentor,
    /// Mentorship -> its sessions
    MentorshipToSession,
}

// ============== Validation Functions ==============

/// Validate mentor profile creation
pub fn validate_mentor_profile(profile: &MentorProfile) -> ExternResult<ValidateCallbackResult> {
    if profile.skills.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mentor must list at least one skill".to_string(),
        ));
    }
    if profile.mentor_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mentor DID cannot be empty".to_string(),
        ));
    }
    if profile.max_mentees == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "max_mentees must be at least 1".to_string(),
        ));
    }
    if profile.active_mentees > profile.max_mentees {
        return Ok(ValidateCallbackResult::Invalid(
            "active_mentees cannot exceed max_mentees".to_string(),
        ));
    }
    if profile.rating < 0.0 || profile.rating > 5.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 0.0 and 5.0".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate mentee profile creation
pub fn validate_mentee_profile(profile: &MenteeProfile) -> ExternResult<ValidateCallbackResult> {
    if profile.mentee_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mentee DID cannot be empty".to_string(),
        ));
    }
    if profile.goals.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mentee must list at least one goal".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate mentorship creation
pub fn validate_mentorship(mentorship: &Mentorship) -> ExternResult<ValidateCallbackResult> {
    if mentorship.mentor_did.is_empty() || mentorship.mentee_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Both mentor and mentee DIDs are required".to_string(),
        ));
    }
    if mentorship.mentor_did == mentorship.mentee_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot mentor yourself".to_string(),
        ));
    }
    if mentorship.skill_focus.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mentorship must have a skill focus".to_string(),
        ));
    }
    if let Some(r) = mentorship.mentee_rating {
        if r < 0.0 || r > 5.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Mentee rating must be between 0.0 and 5.0".to_string(),
            ));
        }
    }
    if let Some(r) = mentorship.mentor_rating {
        if r < 0.0 || r > 5.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Mentor rating must be between 0.0 and 5.0".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate session creation
pub fn validate_session(session: &MentorshipSession) -> ExternResult<ValidateCallbackResult> {
    if session.duration_minutes == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session duration must be greater than 0".to_string(),
        ));
    }
    if session.duration_minutes > 480 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session duration cannot exceed 8 hours (480 minutes)".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate callback dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::MentorProfile(p) => validate_mentor_profile(&p),
                EntryTypes::MenteeProfile(p) => validate_mentee_profile(&p),
                EntryTypes::Mentorship(m) => validate_mentorship(&m),
                EntryTypes::MentorshipSession(s) => validate_session(&s),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::MentorProfile(p) => validate_mentor_profile(&p),
                EntryTypes::MenteeProfile(p) => validate_mentee_profile(&p),
                EntryTypes::Mentorship(m) => validate_mentorship(&m),
                EntryTypes::MentorshipSession(s) => validate_session(&s),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mentor_profile() -> MentorProfile {
        MentorProfile {
            mentor_did: "did:example:mentor1".to_string(),
            skills: vec!["Rust".to_string(), "Holochain".to_string()],
            experience_years: 5,
            bio: "Experienced developer".to_string(),
            max_mentees: 3,
            active_mentees: 0,
            availability: MentorAvailability::Available,
            rating: 0.0,
            total_sessions: 0,
        }
    }

    fn make_mentee_profile() -> MenteeProfile {
        MenteeProfile {
            mentee_did: "did:example:mentee1".to_string(),
            goals: vec!["Learn Rust".to_string()],
            current_level: SkillLevel::Beginner,
            preferred_schedule: "Weekday evenings".to_string(),
            bio: "Eager learner".to_string(),
        }
    }

    #[test]
    fn test_mentor_availability_default() {
        assert_eq!(MentorAvailability::default(), MentorAvailability::Available);
    }

    #[test]
    fn test_skill_level_default() {
        assert_eq!(SkillLevel::default(), SkillLevel::Beginner);
    }

    #[test]
    fn test_mentorship_status_default() {
        assert_eq!(MentorshipStatus::default(), MentorshipStatus::Proposed);
    }

    #[test]
    fn test_valid_mentor_profile() {
        let profile = make_mentor_profile();
        let result = validate_mentor_profile(&profile).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_mentor_no_skills() {
        let mut profile = make_mentor_profile();
        profile.skills = vec![];
        let result = validate_mentor_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentor_empty_did() {
        let mut profile = make_mentor_profile();
        profile.mentor_did = String::new();
        let result = validate_mentor_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentor_zero_max_mentees() {
        let mut profile = make_mentor_profile();
        profile.max_mentees = 0;
        let result = validate_mentor_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentor_active_exceeds_max() {
        let mut profile = make_mentor_profile();
        profile.active_mentees = 5;
        profile.max_mentees = 3;
        let result = validate_mentor_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentor_rating_out_of_range() {
        let mut profile = make_mentor_profile();
        profile.rating = 5.5;
        let result = validate_mentor_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_valid_mentee_profile() {
        let profile = make_mentee_profile();
        let result = validate_mentee_profile(&profile).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_mentee_empty_did() {
        let mut profile = make_mentee_profile();
        profile.mentee_did = String::new();
        let result = validate_mentee_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentee_no_goals() {
        let mut profile = make_mentee_profile();
        profile.goals = vec![];
        let result = validate_mentee_profile(&profile).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentorship_self_mentor() {
        let m = Mentorship {
            mentor_did: "did:example:same".to_string(),
            mentee_did: "did:example:same".to_string(),
            skill_focus: "Rust".to_string(),
            status: MentorshipStatus::Proposed,
            started_at: None,
            completed_at: None,
            sessions_completed: 0,
            mentee_rating: None,
            mentor_rating: None,
        };
        let result = validate_mentorship(&m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentorship_empty_skill_focus() {
        let m = Mentorship {
            mentor_did: "did:example:mentor".to_string(),
            mentee_did: "did:example:mentee".to_string(),
            skill_focus: String::new(),
            status: MentorshipStatus::Proposed,
            started_at: None,
            completed_at: None,
            sessions_completed: 0,
            mentee_rating: None,
            mentor_rating: None,
        };
        let result = validate_mentorship(&m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_mentorship_rating_out_of_range() {
        let m = Mentorship {
            mentor_did: "did:example:mentor".to_string(),
            mentee_did: "did:example:mentee".to_string(),
            skill_focus: "Rust".to_string(),
            status: MentorshipStatus::Completed,
            started_at: None,
            completed_at: None,
            sessions_completed: 5,
            mentee_rating: Some(6.0),
            mentor_rating: None,
        };
        let result = validate_mentorship(&m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_session_zero_duration() {
        let s = MentorshipSession {
            mentorship_id: ActionHash::from_raw_36(vec![0; 36]),
            session_date: Timestamp::from_micros(0),
            duration_minutes: 0,
            topics_covered: vec!["Rust basics".to_string()],
            notes: "Good session".to_string(),
            mentee_progress: "Making progress".to_string(),
        };
        let result = validate_session(&s).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_session_excessive_duration() {
        let s = MentorshipSession {
            mentorship_id: ActionHash::from_raw_36(vec![0; 36]),
            session_date: Timestamp::from_micros(0),
            duration_minutes: 500,
            topics_covered: vec!["Rust basics".to_string()],
            notes: "Good session".to_string(),
            mentee_progress: "Making progress".to_string(),
        };
        let result = validate_session(&s).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
