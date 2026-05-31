// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Care Plans Integrity Zome
//! Defines entry types and validation for coordinated care plans and sessions.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of care being provided
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CareType {
    Childcare,
    Eldercare,
    DisabilitySupport,
    PostSurgery,
    MentalHealth,
    Respite,
    Other(String),
}

impl CareType {
    pub fn anchor_key(&self) -> String {
        match self {
            CareType::Childcare => "childcare".to_string(),
            CareType::Eldercare => "eldercare".to_string(),
            CareType::DisabilitySupport => "disability".to_string(),
            CareType::PostSurgery => "postsurgery".to_string(),
            CareType::MentalHealth => "mentalhealth".to_string(),
            CareType::Respite => "respite".to_string(),
            CareType::Other(s) => format!("other_{}", s.to_lowercase().replace(' ', "_")),
        }
    }
}

/// Status of a care plan
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PlanStatus {
    Draft,
    Active,
    Paused,
    Completed,
    Cancelled,
}

/// A coordinated care plan for ongoing care needs
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CarePlan {
    /// The person receiving care
    pub recipient: AgentPubKey,
    /// Title of the care plan
    pub title: String,
    /// Detailed description of care needs
    pub description: String,
    /// Type of care
    pub care_type: CareType,
    /// Schedule description (e.g. "Mon/Wed/Fri mornings")
    pub schedule: String,
    /// List of assigned caregivers
    pub caregivers: Vec<AgentPubKey>,
    /// Current plan status
    pub status: PlanStatus,
    /// When the plan was created
    pub created_at: Timestamp,
    /// When the plan was last updated
    pub updated_at: Timestamp,
    /// Estimated total hours per week
    pub hours_per_week: f32,
    /// Special instructions or notes
    pub special_instructions: String,
}

/// A logged care session within a plan
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareSession {
    /// The care plan this session belongs to
    pub plan_hash: ActionHash,
    /// The caregiver who provided care
    pub caregiver: AgentPubKey,
    /// When the session started
    pub started_at: Timestamp,
    /// When the session ended
    pub ended_at: Timestamp,
    /// Duration in hours
    pub hours: f32,
    /// Session notes
    pub notes: String,
    /// Tasks that were completed during the session
    pub tasks_completed: Vec<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CarePlan(CarePlan),
    CareSession(CareSession),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All care plans anchor
    AllPlans,
    /// Care type to plans
    TypeToPlans,
    /// Recipient to their care plans
    RecipientToPlans,
    /// Caregiver to plans they are assigned to
    CaregiverToPlans,
    /// Plan to its sessions
    PlanToSessions,
    /// Caregiver to their sessions
    CaregiverToSessions,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CarePlan(plan) => validate_create_plan(action, plan),
                EntryTypes::CareSession(session) => validate_create_session(action, session),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CarePlan(plan) => validate_update_plan(plan),
                EntryTypes::CareSession(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_plan(_action: Create, plan: CarePlan) -> ExternResult<ValidateCallbackResult> {
    if plan.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title cannot be empty".into(),
        ));
    }
    if plan.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title must be 256 characters or fewer".into(),
        ));
    }
    if plan.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan description cannot be empty".into(),
        ));
    }
    if plan.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan description must be 4096 characters or fewer".into(),
        ));
    }
    if plan.schedule.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule cannot be empty".into(),
        ));
    }
    if plan.schedule.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule must be 512 characters or fewer".into(),
        ));
    }
    if plan.caregivers.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one caregiver must be assigned".into(),
        ));
    }
    if plan.caregivers.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 caregivers".into(),
        ));
    }
    if plan.hours_per_week <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours per week must be positive".into(),
        ));
    }
    if plan.hours_per_week > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours per week cannot exceed 168".into(),
        ));
    }
    if plan.special_instructions.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Special instructions must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_plan(plan: CarePlan) -> ExternResult<ValidateCallbackResult> {
    if plan.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title cannot be empty".into(),
        ));
    }
    if plan.caregivers.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one caregiver must be assigned".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_session(
    _action: Create,
    session: CareSession,
) -> ExternResult<ValidateCallbackResult> {
    if session.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session hours must be positive".into(),
        ));
    }
    if session.hours > 24.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session cannot exceed 24 hours".into(),
        ));
    }
    if session.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session notes must be 4096 characters or fewer".into(),
        ));
    }
    if session.tasks_completed.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot list more than 50 tasks".into(),
        ));
    }
    for task in &session.tasks_completed {
        if task.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each task must be 256 characters or fewer".into(),
            ));
        }
    }
    if session.ended_at <= session.started_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Session end must be after session start".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
