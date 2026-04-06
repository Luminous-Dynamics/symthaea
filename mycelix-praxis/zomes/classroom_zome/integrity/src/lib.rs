// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Classroom Integrity Zome
//!
//! Defines entry types and validation rules for the classroom domain.
//! Supports teacher/student/parent roles and classroom management.

#![deny(unsafe_code)]

use hdi::prelude::*;

// ============== Enums ==============

/// Roles in a classroom
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassroomRole {
    /// Primary instructor
    Teacher,
    /// Teaching assistant or co-teacher
    Assistant,
    /// Enrolled learner
    Student,
    /// Parent/guardian linked to a student
    Guardian,
}

impl Default for ClassroomRole {
    fn default() -> Self {
        ClassroomRole::Student
    }
}

/// Member status within a classroom
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberStatus {
    /// Invited but not yet accepted
    Invited,
    /// Currently active in the classroom
    Active,
    /// Temporarily inactive
    Inactive,
    /// Removed from the classroom
    Removed,
}

impl Default for MemberStatus {
    fn default() -> Self {
        MemberStatus::Active
    }
}

// ============== Entry Types ==============

/// A classroom grouping teacher + students
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Classroom {
    /// Human-readable classroom name
    pub name: String,
    /// The teacher who owns/manages this classroom
    pub teacher: AgentPubKey,
    /// Grade level (flexible string: "Grade 3", "PreK", "College 101", etc.)
    pub grade_level: String,
    /// Subject area ("Mathematics", "English Language Arts", etc.)
    pub subject: String,
    /// School year identifier ("2025-2026")
    pub school_year: String,
    /// Optional longer description
    pub description: Option<String>,
    /// Maximum number of students allowed
    pub max_students: u32,
    /// Linked learning path hashes (from learning_zome)
    pub curriculum_path_hashes: Vec<ActionHash>,
    /// Whether the classroom is currently active
    pub is_active: bool,
    /// Creation timestamp (microseconds)
    pub created_at: i64,
    /// Last modification timestamp (microseconds)
    pub modified_at: i64,
}

/// Role assignment within a classroom
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ClassroomMember {
    /// The classroom this membership belongs to
    pub classroom_hash: ActionHash,
    /// The agent who is a member
    pub agent: AgentPubKey,
    /// Their role in the classroom
    pub role: ClassroomRole,
    /// Current status
    pub status: MemberStatus,
    /// Optional display name
    pub display_name: Option<String>,
    /// When the member joined (microseconds)
    pub joined_at: i64,
}

/// Guardian-student relationship link
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct GuardianLink {
    /// The guardian agent
    pub guardian: AgentPubKey,
    /// The student agent
    pub student: AgentPubKey,
    /// Relationship type ("Parent", "Guardian", "Caregiver")
    pub relationship: String,
    /// Whether guardian can view student's learning progress
    pub can_view_progress: bool,
    /// Whether guardian can view student's assessments/grades
    pub can_view_assessments: bool,
    /// Creation timestamp (microseconds)
    pub created_at: i64,
}

/// Join code for classroom enrollment
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct JoinCode {
    /// The classroom this code grants access to
    pub classroom_hash: ActionHash,
    /// 6-character alphanumeric code
    pub code: String,
    /// What role the joiner receives
    pub role: ClassroomRole,
    /// Maximum number of times this code can be used
    pub max_uses: u32,
    /// Current number of uses
    pub uses: u32,
    /// Optional expiration timestamp (microseconds)
    pub expires_at: Option<i64>,
    /// Creation timestamp (microseconds)
    pub created_at: i64,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 3, visibility = "public")]
    Classroom(Classroom),
    #[entry_type(required_validations = 1, visibility = "public")]
    ClassroomMember(ClassroomMember),
    #[entry_type(required_validations = 1, visibility = "private")]
    GuardianLink(GuardianLink),
    #[entry_type(required_validations = 3, visibility = "public")]
    JoinCode(JoinCode),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Path anchor for all classrooms
    AllClassrooms,
    /// Teacher agent -> classrooms they own
    TeacherToClassrooms,
    /// Classroom -> its members
    ClassroomToMembers,
    /// Student agent -> classrooms they belong to
    StudentToClassrooms,
    /// Guardian agent -> students they are linked to
    GuardianToStudents,
    /// Classroom -> its join codes
    ClassroomToJoinCodes,
    /// Agent -> guardian links they participate in
    AgentToGuardianLinks,
}

// ============== Validation Functions ==============

/// Validate a Classroom entry
pub fn validate_classroom(classroom: &Classroom) -> ExternResult<ValidateCallbackResult> {
    // Name must not be empty
    if classroom.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Classroom name cannot be empty".to_string(),
        ));
    }

    // Name length limit (1-200 chars)
    if classroom.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Classroom name too long (max 200 characters)".to_string(),
        ));
    }

    // Description length limit (max 5000 chars)
    if let Some(ref desc) = classroom.description {
        if desc.len() > 5000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Classroom description too long (max 5000 characters)".to_string(),
            ));
        }
    }

    // Max students must be 1-500
    if classroom.max_students < 1 || classroom.max_students > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Max students must be between 1 and 500".to_string(),
        ));
    }

    // School year must not be empty
    if classroom.school_year.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "School year cannot be empty".to_string(),
        ));
    }

    // Curriculum paths limit (max 50)
    if classroom.curriculum_path_hashes.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many curriculum paths (max 50)".to_string(),
        ));
    }

    // Grade level must not be empty
    if classroom.grade_level.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Grade level cannot be empty".to_string(),
        ));
    }

    // Subject must not be empty
    if classroom.subject.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject cannot be empty".to_string(),
        ));
    }

    // Modified timestamp must be >= created timestamp
    if classroom.modified_at < classroom.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Modified time cannot be before creation time".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a ClassroomMember entry
pub fn validate_classroom_member(member: &ClassroomMember) -> ExternResult<ValidateCallbackResult> {
    // Display name length limit (max 200 chars)
    if let Some(ref name) = member.display_name {
        if name.len() > 200 {
            return Ok(ValidateCallbackResult::Invalid(
                "Display name too long (max 200 characters)".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a GuardianLink entry
pub fn validate_guardian_link(link: &GuardianLink) -> ExternResult<ValidateCallbackResult> {
    // Relationship must not be empty
    if link.relationship.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian relationship cannot be empty".to_string(),
        ));
    }

    // Relationship length limit (max 100 chars)
    if link.relationship.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian relationship too long (max 100 characters)".to_string(),
        ));
    }

    // Guardian and student must be different agents
    if link.guardian == link.student {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian and student must be different agents".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a JoinCode entry
pub fn validate_join_code(code: &JoinCode) -> ExternResult<ValidateCallbackResult> {
    // Code must be exactly 6 characters
    if code.code.len() != 6 {
        return Ok(ValidateCallbackResult::Invalid(
            "Join code must be exactly 6 characters".to_string(),
        ));
    }

    // Code must be alphanumeric
    if !code.code.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Join code must be alphanumeric".to_string(),
        ));
    }

    // Max uses must be at least 1
    if code.max_uses < 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Join code max uses must be at least 1".to_string(),
        ));
    }

    // Uses cannot exceed max_uses
    if code.uses > code.max_uses {
        return Ok(ValidateCallbackResult::Invalid(
            "Join code uses cannot exceed max uses".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============== Genesis & Validation Callbacks ==============

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
                EntryTypes::Classroom(classroom) => validate_classroom(&classroom),
                EntryTypes::ClassroomMember(member) => validate_classroom_member(&member),
                EntryTypes::GuardianLink(link) => validate_guardian_link(&link),
                EntryTypes::JoinCode(code) => validate_join_code(&code),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Classroom(classroom) => validate_classroom(&classroom),
                EntryTypes::ClassroomMember(member) => validate_classroom_member(&member),
                EntryTypes::GuardianLink(link) => validate_guardian_link(&link),
                EntryTypes::JoinCode(code) => validate_join_code(&code),
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

// ============== Unit Tests ==============

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a valid Classroom for testing
    fn valid_classroom() -> Classroom {
        Classroom {
            name: "Math 101".to_string(),
            teacher: AgentPubKey::from_raw_36(vec![0u8; 36]),
            grade_level: "Grade 5".to_string(),
            subject: "Mathematics".to_string(),
            school_year: "2025-2026".to_string(),
            description: Some("Introduction to fractions".to_string()),
            max_students: 30,
            curriculum_path_hashes: vec![],
            is_active: true,
            created_at: 1000000,
            modified_at: 1000000,
        }
    }

    // Helper to create a valid JoinCode for testing
    fn valid_join_code() -> JoinCode {
        JoinCode {
            classroom_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            code: "ABC123".to_string(),
            role: ClassroomRole::Student,
            max_uses: 30,
            uses: 0,
            expires_at: None,
            created_at: 1000000,
        }
    }

    // --- ClassroomRole tests ---

    #[test]
    fn test_classroom_role_default_is_student() {
        assert_eq!(ClassroomRole::default(), ClassroomRole::Student);
    }

    // --- MemberStatus tests ---

    #[test]
    fn test_member_status_default_is_active() {
        assert_eq!(MemberStatus::default(), MemberStatus::Active);
    }

    // --- Classroom validation tests ---

    #[test]
    fn test_classroom_name_empty() {
        let mut classroom = valid_classroom();
        classroom.name = "".to_string();
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_classroom_name_whitespace_only() {
        let mut classroom = valid_classroom();
        classroom.name = "   ".to_string();
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_classroom_name_too_long() {
        let mut classroom = valid_classroom();
        classroom.name = "x".repeat(201);
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_classroom_name_valid() {
        let classroom = valid_classroom();
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_classroom_name_max_length() {
        let mut classroom = valid_classroom();
        classroom.name = "x".repeat(200);
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // --- Max students tests ---

    #[test]
    fn test_max_students_zero() {
        let mut classroom = valid_classroom();
        classroom.max_students = 0;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_max_students_over_limit() {
        let mut classroom = valid_classroom();
        classroom.max_students = 501;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_max_students_valid() {
        let mut classroom = valid_classroom();
        classroom.max_students = 30;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_max_students_boundary_1() {
        let mut classroom = valid_classroom();
        classroom.max_students = 1;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_max_students_boundary_500() {
        let mut classroom = valid_classroom();
        classroom.max_students = 500;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // --- Join code tests ---

    #[test]
    fn test_join_code_too_short() {
        let mut code = valid_join_code();
        code.code = "ABC12".to_string(); // 5 chars
        let result = validate_join_code(&code).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_join_code_too_long() {
        let mut code = valid_join_code();
        code.code = "ABC1234".to_string(); // 7 chars
        let result = validate_join_code(&code).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_join_code_valid_6_chars() {
        let code = valid_join_code();
        let result = validate_join_code(&code).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_join_code_non_alphanumeric() {
        let mut code = valid_join_code();
        code.code = "ABC-12".to_string();
        let result = validate_join_code(&code).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // --- Description tests ---

    #[test]
    fn test_classroom_description_too_long() {
        let mut classroom = valid_classroom();
        classroom.description = Some("x".repeat(5001));
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_classroom_description_at_max() {
        let mut classroom = valid_classroom();
        classroom.description = Some("x".repeat(5000));
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_classroom_description_none() {
        let mut classroom = valid_classroom();
        classroom.description = None;
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // --- Curriculum paths test ---

    #[test]
    fn test_curriculum_paths_over_limit() {
        let mut classroom = valid_classroom();
        classroom.curriculum_path_hashes = (0..51)
            .map(|_| ActionHash::from_raw_36(vec![0u8; 36]))
            .collect();
        let result = validate_classroom(&classroom).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // --- GuardianLink tests ---

    #[test]
    fn test_guardian_relationship_empty() {
        let link = GuardianLink {
            guardian: AgentPubKey::from_raw_36(vec![1u8; 36]),
            student: AgentPubKey::from_raw_36(vec![2u8; 36]),
            relationship: "".to_string(),
            can_view_progress: true,
            can_view_assessments: false,
            created_at: 1000000,
        };
        let result = validate_guardian_link(&link).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_guardian_relationship_too_long() {
        let link = GuardianLink {
            guardian: AgentPubKey::from_raw_36(vec![1u8; 36]),
            student: AgentPubKey::from_raw_36(vec![2u8; 36]),
            relationship: "x".repeat(101),
            can_view_progress: true,
            can_view_assessments: false,
            created_at: 1000000,
        };
        let result = validate_guardian_link(&link).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_guardian_same_as_student() {
        let agent = AgentPubKey::from_raw_36(vec![1u8; 36]);
        let link = GuardianLink {
            guardian: agent.clone(),
            student: agent,
            relationship: "Parent".to_string(),
            can_view_progress: true,
            can_view_assessments: false,
            created_at: 1000000,
        };
        let result = validate_guardian_link(&link).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
