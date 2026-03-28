// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Care Plans Integrity Zome
//! Defines entry types and validation for coordinated care plans and sessions.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToPlans => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToPlans link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RecipientToPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RecipientToPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CaregiverToPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CaregiverToPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PlanToSessions => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PlanToSessions link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CaregiverToSessions => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CaregiverToSessions link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_care_type_other(care_type: &CareType) -> Result<(), String> {
    if let CareType::Other(s) = care_type {
        if s.trim().is_empty() {
            return Err("Custom care type label cannot be empty".to_string());
        }
        if s.len() > 128 {
            return Err("Custom care type label must be 128 characters or fewer".to_string());
        }
    }
    Ok(())
}

fn validate_create_plan(_action: Create, plan: CarePlan) -> ExternResult<ValidateCallbackResult> {
    if plan.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title cannot be empty".into(),
        ));
    }
    if plan.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title must be 256 characters or fewer".into(),
        ));
    }
    if plan.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan description cannot be empty".into(),
        ));
    }
    if plan.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan description must be 4096 characters or fewer".into(),
        ));
    }
    if plan.schedule.trim().is_empty() {
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
    if !plan.hours_per_week.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours per week must be a finite number".into(),
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
    if let Err(msg) = validate_care_type_other(&plan.care_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_plan(plan: CarePlan) -> ExternResult<ValidateCallbackResult> {
    if plan.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title cannot be empty".into(),
        ));
    }
    if plan.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan title must be 256 characters or fewer".into(),
        ));
    }
    if plan.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plan description must be 4096 characters or fewer".into(),
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
    if plan.special_instructions.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Special instructions must be 4096 characters or fewer".into(),
        ));
    }
    if let Err(msg) = validate_care_type_other(&plan.care_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_session(
    _action: Create,
    session: CareSession,
) -> ExternResult<ValidateCallbackResult> {
    if !session.hours.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours must be a finite number".into(),
        ));
    }
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
        if task.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Task description cannot be empty".into(),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Factory Functions
    // ============================================================================

    fn valid_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn valid_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdc; 36])
    }

    fn valid_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xab; 36])
    }

    fn valid_timestamp_start() -> Timestamp {
        Timestamp::from_micros(1000000)
    }

    fn valid_timestamp_end() -> Timestamp {
        Timestamp::from_micros(2000000)
    }

    fn valid_plan() -> CarePlan {
        CarePlan {
            recipient: valid_agent(),
            title: "Daily Care Plan".to_string(),
            description: "A comprehensive care plan for daily assistance".to_string(),
            care_type: CareType::Eldercare,
            schedule: "Mon/Wed/Fri 9am-12pm".to_string(),
            caregivers: vec![valid_agent_2()],
            status: PlanStatus::Active,
            created_at: valid_timestamp_start(),
            updated_at: valid_timestamp_start(),
            hours_per_week: 20.0,
            special_instructions: "Patient prefers morning care".to_string(),
        }
    }

    fn valid_session() -> CareSession {
        CareSession {
            plan_hash: valid_action_hash(),
            caregiver: valid_agent(),
            started_at: valid_timestamp_start(),
            ended_at: valid_timestamp_end(),
            hours: 3.0,
            notes: "Session went well".to_string(),
            tasks_completed: vec![
                "Morning medication".to_string(),
                "Breakfast assistance".to_string(),
            ],
        }
    }

    fn mock_create_action() -> Create {
        Create {
            author: valid_agent(),
            timestamp: valid_timestamp_start(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    // ============================================================================
    // CarePlan Validation Tests - Create
    // ============================================================================

    #[test]
    fn test_valid_plan_passes() {
        let plan = valid_plan();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_empty_title() {
        let mut plan = valid_plan();
        plan.title = "".to_string();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_title_too_long() {
        let mut plan = valid_plan();
        plan.title = "x".repeat(257);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_title_exactly_256_chars() {
        let mut plan = valid_plan();
        plan.title = "x".repeat(256);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_title_255_chars() {
        let mut plan = valid_plan();
        plan.title = "x".repeat(255);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_empty_description() {
        let mut plan = valid_plan();
        plan.description = "".to_string();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_description_too_long() {
        let mut plan = valid_plan();
        plan.description = "x".repeat(4097);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_description_exactly_4096_chars() {
        let mut plan = valid_plan();
        plan.description = "x".repeat(4096);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_description_4095_chars() {
        let mut plan = valid_plan();
        plan.description = "x".repeat(4095);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_empty_schedule() {
        let mut plan = valid_plan();
        plan.schedule = "".to_string();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_schedule_too_long() {
        let mut plan = valid_plan();
        plan.schedule = "x".repeat(513);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_schedule_exactly_512_chars() {
        let mut plan = valid_plan();
        plan.schedule = "x".repeat(512);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_schedule_511_chars() {
        let mut plan = valid_plan();
        plan.schedule = "x".repeat(511);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_empty_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = vec![];
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_too_many_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = (0..51).map(|_| valid_agent()).collect();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_exactly_50_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = (0..50).map(|_| valid_agent()).collect();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_49_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = (0..49).map(|_| valid_agent()).collect();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_single_caregiver() {
        let mut plan = valid_plan();
        plan.caregivers = vec![valid_agent()];
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_zero_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = 0.0;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_negative_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = -5.0;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_hours_exceed_168() {
        let mut plan = valid_plan();
        plan.hours_per_week = 169.0;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_exactly_168_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = 168.0;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_167_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = 167.0;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_minimum_positive_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = 0.1;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_special_instructions_too_long() {
        let mut plan = valid_plan();
        plan.special_instructions = "x".repeat(4097);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_special_instructions_exactly_4096_chars() {
        let mut plan = valid_plan();
        plan.special_instructions = "x".repeat(4096);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_special_instructions_4095_chars() {
        let mut plan = valid_plan();
        plan.special_instructions = "x".repeat(4095);
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_empty_special_instructions() {
        let mut plan = valid_plan();
        plan.special_instructions = "".to_string();
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ============================================================================
    // CarePlan Validation Tests - Update
    // ============================================================================

    #[test]
    fn test_valid_plan_update_passes() {
        let plan = valid_plan();
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_update_empty_title() {
        let mut plan = valid_plan();
        plan.title = "".to_string();
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_update_empty_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = vec![];
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_update_single_caregiver() {
        let mut plan = valid_plan();
        plan.caregivers = vec![valid_agent()];
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_update_rejects_long_description() {
        let mut plan = valid_plan();
        plan.description = "x".repeat(4097);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_plan_update_description_at_limit() {
        let mut plan = valid_plan();
        plan.description = "x".repeat(4096);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ============================================================================
    // CareSession Validation Tests - Create
    // ============================================================================

    #[test]
    fn test_valid_session_passes() {
        let session = valid_session();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_zero_hours() {
        let mut session = valid_session();
        session.hours = 0.0;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_negative_hours() {
        let mut session = valid_session();
        session.hours = -1.0;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_hours_exceed_24() {
        let mut session = valid_session();
        session.hours = 24.1;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_exactly_24_hours() {
        let mut session = valid_session();
        session.hours = 24.0;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_23_point_9_hours() {
        let mut session = valid_session();
        session.hours = 23.9;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_minimum_positive_hours() {
        let mut session = valid_session();
        session.hours = 0.1;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_notes_too_long() {
        let mut session = valid_session();
        session.notes = "x".repeat(4097);
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_notes_exactly_4096_chars() {
        let mut session = valid_session();
        session.notes = "x".repeat(4096);
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_notes_4095_chars() {
        let mut session = valid_session();
        session.notes = "x".repeat(4095);
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_empty_notes() {
        let mut session = valid_session();
        session.notes = "".to_string();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_too_many_tasks() {
        let mut session = valid_session();
        session.tasks_completed = (0..51).map(|i| format!("Task {}", i)).collect();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_exactly_50_tasks() {
        let mut session = valid_session();
        session.tasks_completed = (0..50).map(|i| format!("Task {}", i)).collect();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_49_tasks() {
        let mut session = valid_session();
        session.tasks_completed = (0..49).map(|i| format!("Task {}", i)).collect();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_zero_tasks() {
        let mut session = valid_session();
        session.tasks_completed = vec![];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_single_task() {
        let mut session = valid_session();
        session.tasks_completed = vec!["One task".to_string()];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_task_too_long() {
        let mut session = valid_session();
        session.tasks_completed = vec!["x".repeat(257)];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_task_exactly_256_chars() {
        let mut session = valid_session();
        session.tasks_completed = vec!["x".repeat(256)];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_task_255_chars() {
        let mut session = valid_session();
        session.tasks_completed = vec!["x".repeat(255)];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_multiple_tasks_one_too_long() {
        let mut session = valid_session();
        session.tasks_completed = vec![
            "Valid task".to_string(),
            "x".repeat(257),
            "Another valid task".to_string(),
        ];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_ended_before_started() {
        let mut session = valid_session();
        session.started_at = valid_timestamp_end();
        session.ended_at = valid_timestamp_start();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_ended_equals_started() {
        let mut session = valid_session();
        session.ended_at = session.started_at;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_minimal_time_difference() {
        let mut session = valid_session();
        session.started_at = Timestamp::from_micros(1000000);
        session.ended_at = Timestamp::from_micros(1000001);
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ============================================================================
    // CareType Tests
    // ============================================================================

    #[test]
    fn test_care_type_childcare_anchor() {
        let care_type = CareType::Childcare;
        assert_eq!(care_type.anchor_key(), "childcare");
    }

    #[test]
    fn test_care_type_eldercare_anchor() {
        let care_type = CareType::Eldercare;
        assert_eq!(care_type.anchor_key(), "eldercare");
    }

    #[test]
    fn test_care_type_disability_anchor() {
        let care_type = CareType::DisabilitySupport;
        assert_eq!(care_type.anchor_key(), "disability");
    }

    #[test]
    fn test_care_type_postsurgery_anchor() {
        let care_type = CareType::PostSurgery;
        assert_eq!(care_type.anchor_key(), "postsurgery");
    }

    #[test]
    fn test_care_type_mentalhealth_anchor() {
        let care_type = CareType::MentalHealth;
        assert_eq!(care_type.anchor_key(), "mentalhealth");
    }

    #[test]
    fn test_care_type_respite_anchor() {
        let care_type = CareType::Respite;
        assert_eq!(care_type.anchor_key(), "respite");
    }

    #[test]
    fn test_care_type_other_anchor() {
        let care_type = CareType::Other("Custom Care".to_string());
        assert_eq!(care_type.anchor_key(), "other_custom_care");
    }

    #[test]
    fn test_care_type_other_anchor_with_spaces() {
        let care_type = CareType::Other("Multiple Word Type".to_string());
        assert_eq!(care_type.anchor_key(), "other_multiple_word_type");
    }

    #[test]
    fn test_care_type_other_anchor_uppercase() {
        let care_type = CareType::Other("UPPERCASE".to_string());
        assert_eq!(care_type.anchor_key(), "other_uppercase");
    }

    #[test]
    fn test_care_type_serde_childcare() {
        let care_type = CareType::Childcare;
        let serialized = serde_json::to_string(&care_type).unwrap();
        let deserialized: CareType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(care_type, deserialized);
    }

    #[test]
    fn test_care_type_serde_other() {
        let care_type = CareType::Other("Test".to_string());
        let serialized = serde_json::to_string(&care_type).unwrap();
        let deserialized: CareType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(care_type, deserialized);
    }

    // ============================================================================
    // PlanStatus Tests
    // ============================================================================

    #[test]
    fn test_plan_status_serde_draft() {
        let status = PlanStatus::Draft;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: PlanStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_plan_status_serde_active() {
        let status = PlanStatus::Active;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: PlanStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_plan_status_serde_paused() {
        let status = PlanStatus::Paused;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: PlanStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_plan_status_serde_completed() {
        let status = PlanStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: PlanStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_plan_status_serde_cancelled() {
        let status = PlanStatus::Cancelled;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: PlanStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    // ============================================================================
    // Entry Type Roundtrip Tests
    // ============================================================================

    #[test]
    fn test_care_plan_serde_roundtrip() {
        let plan = valid_plan();
        let serialized = serde_json::to_string(&plan).unwrap();
        let deserialized: CarePlan = serde_json::from_str(&serialized).unwrap();
        assert_eq!(plan, deserialized);
    }

    #[test]
    fn test_care_session_serde_roundtrip() {
        let session = valid_session();
        let serialized = serde_json::to_string(&session).unwrap();
        let deserialized: CareSession = serde_json::from_str(&serialized).unwrap();
        assert_eq!(session, deserialized);
    }

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("test_anchor".to_string());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    // ============================================================================
    // Complex Validation Scenarios
    // ============================================================================

    #[test]
    fn test_plan_with_all_care_types() {
        for care_type in vec![
            CareType::Childcare,
            CareType::Eldercare,
            CareType::DisabilitySupport,
            CareType::PostSurgery,
            CareType::MentalHealth,
            CareType::Respite,
            CareType::Other("Custom".to_string()),
        ] {
            let mut plan = valid_plan();
            plan.care_type = care_type;
            let result = validate_create_plan(mock_create_action(), plan);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_plan_with_all_statuses() {
        for status in vec![
            PlanStatus::Draft,
            PlanStatus::Active,
            PlanStatus::Paused,
            PlanStatus::Completed,
            PlanStatus::Cancelled,
        ] {
            let mut plan = valid_plan();
            plan.status = status;
            let result = validate_create_plan(mock_create_action(), plan);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_session_with_maximum_valid_tasks() {
        let mut session = valid_session();
        // Each task exactly 256 chars (largest valid size)
        session.tasks_completed = (0..50).map(|_| "x".repeat(256)).collect();
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_with_fractional_hours() {
        let mut plan = valid_plan();
        plan.hours_per_week = 20.5;
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_session_with_fractional_hours() {
        let mut session = valid_session();
        session.hours = 2.75;
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_plan_minimal_valid() {
        let plan = CarePlan {
            recipient: valid_agent(),
            title: "X".to_string(),
            description: "Y".to_string(),
            care_type: CareType::Other("Z".to_string()),
            schedule: "S".to_string(),
            caregivers: vec![valid_agent()],
            status: PlanStatus::Draft,
            created_at: valid_timestamp_start(),
            updated_at: valid_timestamp_start(),
            hours_per_week: 0.1,
            special_instructions: "".to_string(),
        };
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToPlans => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToPlans link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RecipientToPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RecipientToPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CaregiverToPlans => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CaregiverToPlans link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PlanToSessions => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PlanToSessions link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CaregiverToSessions => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CaregiverToSessions link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_tag_all_plans_at_limit() {
        let result = validate_create_link_tag(LinkTypes::AllPlans, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_all_plans_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllPlans, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_type_to_plans_at_limit() {
        let result = validate_create_link_tag(LinkTypes::TypeToPlans, vec![0u8; 512]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_type_to_plans_over_limit() {
        let result = validate_create_link_tag(LinkTypes::TypeToPlans, vec![0u8; 513]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_caregiver_to_sessions_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::CaregiverToSessions, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_caregiver_to_sessions_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::CaregiverToSessions, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::AllPlans, vec![]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_session_minimal_valid() {
        let session = CareSession {
            plan_hash: valid_action_hash(),
            caregiver: valid_agent(),
            started_at: Timestamp::from_micros(1000000),
            ended_at: Timestamp::from_micros(1000001),
            hours: 0.1,
            notes: "".to_string(),
            tasks_completed: vec![],
        };
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ============================================================================
    // CareType::Other string length validation tests
    // ============================================================================

    #[test]
    fn test_create_plan_custom_care_type_at_limit() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("a".repeat(128));
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_create_plan_custom_care_type_too_long() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("a".repeat(129));
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_create_plan_custom_care_type_empty() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("".to_string());
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_create_plan_custom_care_type_whitespace_only() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("   ".to_string());
        let result = validate_create_plan(mock_create_action(), plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ============================================================================
    // Hardened validate_update_plan tests
    // ============================================================================

    #[test]
    fn test_update_plan_title_at_limit() {
        let mut plan = valid_plan();
        plan.title = "t".repeat(256);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_update_plan_title_too_long() {
        let mut plan = valid_plan();
        plan.title = "t".repeat(257);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_update_plan_schedule_at_limit() {
        let mut plan = valid_plan();
        plan.schedule = "s".repeat(512);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_update_plan_schedule_too_long() {
        let mut plan = valid_plan();
        plan.schedule = "s".repeat(513);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_update_plan_special_instructions_at_limit() {
        let mut plan = valid_plan();
        plan.special_instructions = "i".repeat(4096);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_update_plan_special_instructions_too_long() {
        let mut plan = valid_plan();
        plan.special_instructions = "i".repeat(4097);
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_update_plan_too_many_caregivers() {
        let mut plan = valid_plan();
        plan.caregivers = (0..51).map(|_| valid_agent()).collect();
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_update_plan_50_caregivers_at_limit() {
        let mut plan = valid_plan();
        plan.caregivers = (0..50).map(|_| valid_agent()).collect();
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_update_plan_custom_care_type_too_long() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("a".repeat(129));
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_update_plan_custom_care_type_empty() {
        let mut plan = valid_plan();
        plan.care_type = CareType::Other("".to_string());
        let result = validate_update_plan(plan);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ============================================================================
    // Session task empty string validation tests
    // ============================================================================

    #[test]
    fn test_session_task_empty_string() {
        let mut session = valid_session();
        session.tasks_completed = vec!["".to_string()];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_task_whitespace_only() {
        let mut session = valid_session();
        session.tasks_completed = vec!["   ".to_string()];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_session_mixed_valid_and_empty_task() {
        let mut session = valid_session();
        session.tasks_completed = vec!["Valid task".to_string(), "".to_string()];
        let result = validate_create_session(mock_create_action(), session);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
