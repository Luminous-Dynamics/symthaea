// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Classroom Coordinator Zome
//!
//! Implements business logic for classroom management with teacher/student/parent roles.
//!
//! ## Core Functionality
//!
//! - **Classroom Lifecycle**: Create, update, deactivate classrooms
//! - **Enrollment**: Join codes, roster management
//! - **Role Management**: Teacher, assistant, student, guardian roles
//! - **Guardian Links**: Privacy-controlled parent/guardian access to student progress
//! - **Curriculum**: Link learning paths to classrooms

#![deny(unsafe_code)]

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use classroom_integrity::{
    EntryTypes, LinkTypes, Classroom, ClassroomMember, GuardianLink, JoinCode,
    ClassroomRole, MemberStatus,
};

// ============== Helper Functions ==============

/// Ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed_path = path.typed(link_type)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Convert a Timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

// ============== Input Structs ==============

/// Input for creating a classroom
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateClassroomInput {
    pub name: String,
    pub grade_level: String,
    pub subject: String,
    pub school_year: String,
    pub description: Option<String>,
    pub max_students: u32,
}

/// Input for joining a classroom via join code
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JoinClassroomInput {
    pub code: String,
    pub display_name: Option<String>,
}

/// Input for generating a join code
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateJoinCodeInput {
    pub classroom_hash: ActionHash,
    pub role: ClassroomRole,
    pub max_uses: u32,
    pub expires_at: Option<i64>,
}

/// Input for linking a guardian to a student
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkGuardianInput {
    pub student: AgentPubKey,
    pub relationship: String,
    pub can_view_progress: bool,
    pub can_view_assessments: bool,
}

/// Input for assigning curriculum to a classroom
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssignCurriculumInput {
    pub classroom_hash: ActionHash,
    pub curriculum_path_hash: ActionHash,
}

/// Input for updating a member's role
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateMemberRoleInput {
    pub member_hash: ActionHash,
    pub new_role: ClassroomRole,
}

// ============== Classroom Lifecycle ==============

/// Create a new classroom. The calling agent becomes the teacher.
#[hdk_extern]
pub fn create_classroom(input: CreateClassroomInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = timestamp_to_i64(sys_time()?);

    let classroom = Classroom {
        name: input.name,
        teacher: agent.clone(),
        grade_level: input.grade_level,
        subject: input.subject,
        school_year: input.school_year,
        description: input.description,
        max_students: input.max_students,
        curriculum_path_hashes: vec![],
        is_active: true,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::Classroom(classroom))?;

    // Link to all classrooms anchor
    let path = Path::from("all_classrooms");
    let path_hash = ensure_path(path, LinkTypes::AllClassrooms)?;
    create_link(path_hash, action_hash.clone(), LinkTypes::AllClassrooms, ())?;

    // Link teacher -> classroom
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::TeacherToClassrooms,
        (),
    )?;

    // Auto-add teacher as a member
    let teacher_member = ClassroomMember {
        classroom_hash: action_hash.clone(),
        agent: agent.clone(),
        role: ClassroomRole::Teacher,
        status: MemberStatus::Active,
        display_name: None,
        joined_at: now,
    };
    let member_hash = create_entry(EntryTypes::ClassroomMember(teacher_member))?;
    create_link(
        action_hash.clone(),
        member_hash,
        LinkTypes::ClassroomToMembers,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!("Could not find the newly created Classroom"))?;
    Ok(record)
}

/// Get a classroom by its action hash
#[hdk_extern]
pub fn get_classroom(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all classrooms the calling agent is a member of (any role)
#[hdk_extern]
pub fn get_my_classrooms(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Check teacher links
    let teacher_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::TeacherToClassrooms)?,
        GetStrategy::Local,
    )?;

    // Check student links
    let student_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::StudentToClassrooms)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for link in teacher_links.into_iter().chain(student_links.into_iter()) {
        if let Some(target) = link.target.into_action_hash() {
            if seen.insert(target.clone()) {
                if let Some(record) = get(target, GetOptions::default())? {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Generate a join code for a classroom. Only the teacher can do this.
#[hdk_extern]
pub fn generate_join_code(input: GenerateJoinCodeInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = timestamp_to_i64(sys_time()?);

    // Verify caller is the classroom teacher
    let classroom_record = get(input.classroom_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Classroom not found"))?;
    let classroom: Classroom = classroom_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Could not deserialize Classroom"))?;

    if classroom.teacher != agent {
        return Err(wasm_error!("Only the classroom teacher can generate join codes"));
    }

    // Generate a 6-character alphanumeric code from entropy
    let random_bytes = random_bytes(4)?;
    let code: String = random_bytes
        .iter()
        .flat_map(|b| {
            let chars = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
            vec![chars[(b >> 4) as usize % chars.len()] as char, chars[(b & 0x0F) as usize % chars.len()] as char]
        })
        .take(6)
        .collect();

    let join_code = JoinCode {
        classroom_hash: input.classroom_hash.clone(),
        code,
        role: input.role,
        max_uses: input.max_uses,
        uses: 0,
        expires_at: input.expires_at,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::JoinCode(join_code))?;

    // Link classroom -> join code
    create_link(
        input.classroom_hash,
        action_hash.clone(),
        LinkTypes::ClassroomToJoinCodes,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!("Could not find the newly created JoinCode"))?;
    Ok(record)
}

/// Join a classroom using a join code
#[hdk_extern]
pub fn join_classroom(input: JoinClassroomInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = timestamp_to_i64(sys_time()?);

    // Find the join code by searching all classrooms' join codes
    // In practice, the client would supply the classroom hash too,
    // but we search by code for simplicity
    let path = Path::from("all_classrooms");
    let path_hash = ensure_path(path, LinkTypes::AllClassrooms)?;
    let classroom_links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllClassrooms)?,
        GetStrategy::Local,
    )?;

    let mut found_code: Option<(JoinCode, ActionHash)> = None;

    'outer: for cl_link in &classroom_links {
        if let Some(classroom_hash) = cl_link.target.clone().into_action_hash() {
            let code_links = get_links(
                LinkQuery::try_new(classroom_hash, LinkTypes::ClassroomToJoinCodes)?,
                GetStrategy::Local,
            )?;
            for code_link in &code_links {
                if let Some(code_hash) = code_link.target.clone().into_action_hash() {
                    if let Some(record) = get(code_hash.clone(), GetOptions::default())? {
                        let join_code: JoinCode = record
                            .entry()
                            .to_app_option()
                            .map_err(|e| wasm_error!(e))?
                            .ok_or(wasm_error!("Could not deserialize JoinCode"))?;
                        if join_code.code == input.code {
                            found_code = Some((join_code, code_hash));
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    let (join_code, code_action_hash) = found_code
        .ok_or(wasm_error!("Invalid join code"))?;

    // Check code hasn't expired
    if let Some(expires_at) = join_code.expires_at {
        if now > expires_at {
            return Err(wasm_error!("Join code has expired"));
        }
    }

    // Check code hasn't exceeded max uses
    if join_code.uses >= join_code.max_uses {
        return Err(wasm_error!("Join code has reached maximum uses"));
    }

    // Increment uses on the join code
    let updated_code = JoinCode {
        uses: join_code.uses + 1,
        ..join_code.clone()
    };
    update_entry(code_action_hash, EntryTypes::JoinCode(updated_code))?;

    // Create the membership entry
    let member = ClassroomMember {
        classroom_hash: join_code.classroom_hash.clone(),
        agent: agent.clone(),
        role: join_code.role.clone(),
        status: MemberStatus::Active,
        display_name: input.display_name,
        joined_at: now,
    };

    let member_hash = create_entry(EntryTypes::ClassroomMember(member))?;

    // Link classroom -> member
    create_link(
        join_code.classroom_hash.clone(),
        member_hash.clone(),
        LinkTypes::ClassroomToMembers,
        (),
    )?;

    // Link agent -> classroom (for student/guardian lookups)
    create_link(
        agent,
        join_code.classroom_hash,
        LinkTypes::StudentToClassrooms,
        (),
    )?;

    let record = get(member_hash, GetOptions::default())?
        .ok_or(wasm_error!("Could not find the newly created ClassroomMember"))?;
    Ok(record)
}

/// Get all members of a classroom (the roster)
#[hdk_extern]
pub fn get_class_roster(classroom_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(classroom_hash, LinkTypes::ClassroomToMembers)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Link a guardian to a student. The calling agent becomes the guardian.
#[hdk_extern]
pub fn link_guardian(input: LinkGuardianInput) -> ExternResult<Record> {
    let guardian = agent_info()?.agent_initial_pubkey;
    let now = timestamp_to_i64(sys_time()?);

    let guardian_link = GuardianLink {
        guardian: guardian.clone(),
        student: input.student.clone(),
        relationship: input.relationship,
        can_view_progress: input.can_view_progress,
        can_view_assessments: input.can_view_assessments,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::GuardianLink(guardian_link))?;

    // Bidirectional links
    create_link(
        guardian.clone(),
        input.student.clone(),
        LinkTypes::GuardianToStudents,
        (),
    )?;
    create_link(
        guardian,
        action_hash.clone(),
        LinkTypes::AgentToGuardianLinks,
        (),
    )?;
    create_link(
        input.student,
        action_hash.clone(),
        LinkTypes::AgentToGuardianLinks,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!("Could not find the newly created GuardianLink"))?;
    Ok(record)
}

/// Get all guardians linked to a student
#[hdk_extern]
pub fn get_student_guardians(student: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(student, LinkTypes::AgentToGuardianLinks)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Assign a curriculum learning path to a classroom. Only the teacher can do this.
#[hdk_extern]
pub fn assign_curriculum(input: AssignCurriculumInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = timestamp_to_i64(sys_time()?);

    // Verify caller is the classroom teacher
    let classroom_record = get(input.classroom_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Classroom not found"))?;
    let mut classroom: Classroom = classroom_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Could not deserialize Classroom"))?;

    if classroom.teacher != agent {
        return Err(wasm_error!("Only the classroom teacher can assign curriculum"));
    }

    // Add the curriculum path if not already present
    if !classroom.curriculum_path_hashes.contains(&input.curriculum_path_hash) {
        classroom.curriculum_path_hashes.push(input.curriculum_path_hash);
    }
    classroom.modified_at = now;

    let updated_hash = update_entry(input.classroom_hash, EntryTypes::Classroom(classroom))?;

    let record = get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!("Could not find the updated Classroom"))?;
    Ok(record)
}
