// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Care Plans Coordinator Zome
//! Business logic for care plan management and session logging.

use care_plans_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Create a new care plan
#[hdk_extern]
pub fn create_care_plan(plan: CarePlan) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "create_care_plan")?;
    let action_hash = create_entry(&EntryTypes::CarePlan(plan.clone()))?;

    // Link to all plans
    let all_anchor = ensure_anchor("all_care_plans")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllPlans, ())?;

    // Link by care type
    let type_anchor = ensure_anchor(&format!("care_type:{}", plan.care_type.anchor_key()))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::TypeToPlans, ())?;

    // Link recipient to plan
    let recipient_anchor = ensure_anchor(&format!("recipient_plans:{}", plan.recipient))?;
    create_link(
        recipient_anchor,
        action_hash.clone(),
        LinkTypes::RecipientToPlans,
        (),
    )?;

    // Link each caregiver to plan
    for caregiver in &plan.caregivers {
        let cg_anchor = ensure_anchor(&format!("caregiver_plans:{}", caregiver))?;
        create_link(
            cg_anchor,
            action_hash.clone(),
            LinkTypes::CaregiverToPlans,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created care plan".into()
    )))
}

/// Log a care session against a plan
#[hdk_extern]
pub fn log_session(session: CareSession) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "log_session")?;
    // Verify the plan exists
    let _plan_record = get(session.plan_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Care plan not found".into())),
    )?;

    let plan: CarePlan = _plan_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid care plan entry".into()
        )))?;

    // Verify caregiver is assigned to this plan
    if !plan.caregivers.contains(&session.caregiver) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Caregiver is not assigned to this care plan".into()
        )));
    }

    // Verify plan is active
    if plan.status != PlanStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only log sessions for active care plans".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CareSession(session.clone()))?;

    // Link plan to session
    let plan_anchor = ensure_anchor(&format!("plan_sessions:{}", session.plan_hash))?;
    create_link(
        plan_anchor,
        action_hash.clone(),
        LinkTypes::PlanToSessions,
        (),
    )?;

    // Link caregiver to session
    let cg_anchor = ensure_anchor(&format!("caregiver_sessions:{}", session.caregiver))?;
    create_link(
        cg_anchor,
        action_hash.clone(),
        LinkTypes::CaregiverToSessions,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created session".into()
    )))
}

/// Get all sessions for a care plan
#[hdk_extern]
pub fn get_plan_sessions(plan_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let plan_anchor = anchor_hash(&format!("plan_sessions:{}", plan_hash))?;
    let links = get_links(
        LinkQuery::try_new(plan_anchor, LinkTypes::PlanToSessions)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Input for updating plan status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePlanStatusInput {
    pub plan_hash: ActionHash,
    pub new_status: PlanStatus,
}

/// Update the status of a care plan
#[hdk_extern]
pub fn update_plan_status(input: UpdatePlanStatusInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_plan_status")?;
    let record = get(input.plan_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Care plan not found".into())
    ))?;

    let mut plan: CarePlan = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid care plan entry".into()
        )))?;

    let caller = agent_info()?.agent_initial_pubkey;

    // Only recipient or assigned caregivers can update status
    if caller != plan.recipient && !plan.caregivers.contains(&caller) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the recipient or assigned caregivers can update plan status".into()
        )));
    }

    let now = sys_time()?;
    plan.status = input.new_status;
    plan.updated_at = now;

    let updated_hash = update_entry(input.plan_hash, &EntryTypes::CarePlan(plan))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated care plan".into()
    )))
}

/// Get care plans where the caller is the recipient
#[hdk_extern]
pub fn get_my_care_plans(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let recipient_anchor = anchor_hash(&format!("recipient_plans:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(recipient_anchor, LinkTypes::RecipientToPlans)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get care plans where the caller is an assigned caregiver
#[hdk_extern]
pub fn get_my_caregiver_plans(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let cg_anchor = anchor_hash(&format!("caregiver_plans:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(cg_anchor, LinkTypes::CaregiverToPlans)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all sessions logged by the calling caregiver
#[hdk_extern]
pub fn get_my_sessions(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let cg_anchor = anchor_hash(&format!("caregiver_sessions:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(cg_anchor, LinkTypes::CaregiverToSessions)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all care plans
#[hdk_extern]
pub fn get_all_care_plans(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_care_plans")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllPlans)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get care plans by type
#[hdk_extern]
pub fn get_plans_by_type(care_type: CareType) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("care_type:{}", care_type.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToPlans)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Summary of total hours logged for a plan
#[derive(Serialize, Deserialize, Debug)]
pub struct PlanSessionSummary {
    pub plan_hash: ActionHash,
    pub total_sessions: u32,
    pub total_hours: f32,
    pub caregivers_active: u32,
}

/// Get a summary of sessions for a plan
#[hdk_extern]
pub fn get_plan_session_summary(plan_hash: ActionHash) -> ExternResult<PlanSessionSummary> {
    let sessions = get_plan_sessions(plan_hash.clone())?;

    let mut total_hours = 0.0f32;
    let mut caregiver_set = std::collections::HashSet::new();

    for record in &sessions {
        if let Some(session) = record
            .entry()
            .to_app_option::<CareSession>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            total_hours += session.hours;
            caregiver_set.insert(session.caregiver.to_string());
        }
    }

    Ok(PlanSessionSummary {
        plan_hash,
        total_sessions: sessions.len() as u32,
        total_hours,
        caregivers_active: caregiver_set.len() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ── UpdatePlanStatusInput serde roundtrip ────────────────────────────

    #[test]
    fn update_plan_status_input_serde_roundtrip() {
        let input = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Active,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePlanStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.plan_hash, input.plan_hash);
        assert_eq!(decoded.new_status, PlanStatus::Active);
    }

    #[test]
    fn update_plan_status_input_all_statuses() {
        for status in [
            PlanStatus::Draft,
            PlanStatus::Active,
            PlanStatus::Paused,
            PlanStatus::Completed,
            PlanStatus::Cancelled,
        ] {
            let input = UpdatePlanStatusInput {
                plan_hash: fake_action_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdatePlanStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    // ── PlanSessionSummary serde roundtrip ───────────────────────────────

    #[test]
    fn plan_session_summary_serde_roundtrip() {
        let summary = PlanSessionSummary {
            plan_hash: fake_action_hash(),
            total_sessions: 12,
            total_hours: 36.5,
            caregivers_active: 3,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: PlanSessionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.plan_hash, summary.plan_hash);
        assert_eq!(decoded.total_sessions, 12);
        assert!((decoded.total_hours - 36.5).abs() < f32::EPSILON);
        assert_eq!(decoded.caregivers_active, 3);
    }

    #[test]
    fn plan_session_summary_zero_values_roundtrip() {
        let summary = PlanSessionSummary {
            plan_hash: fake_action_hash(),
            total_sessions: 0,
            total_hours: 0.0,
            caregivers_active: 0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: PlanSessionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_sessions, 0);
        assert_eq!(decoded.total_hours, 0.0);
        assert_eq!(decoded.caregivers_active, 0);
    }

    // ── PlanStatus serde roundtrip (all variants) ───────────────────────

    #[test]
    fn plan_status_serde_all_variants() {
        let statuses = vec![
            PlanStatus::Draft,
            PlanStatus::Active,
            PlanStatus::Paused,
            PlanStatus::Completed,
            PlanStatus::Cancelled,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let decoded: PlanStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, status);
        }
    }

    // ── CareType serde roundtrip (all variants) ─────────────────────────

    #[test]
    fn care_type_serde_all_variants() {
        let types = vec![
            CareType::Childcare,
            CareType::Eldercare,
            CareType::DisabilitySupport,
            CareType::PostSurgery,
            CareType::MentalHealth,
            CareType::Respite,
            CareType::Other("Palliative".to_string()),
        ];
        for ct in &types {
            let json = serde_json::to_string(ct).unwrap();
            let decoded: CareType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, ct);
        }
    }

    // ── CareType::anchor_key pure function tests ────────────────────────

    #[test]
    fn care_type_anchor_key_known_variants() {
        assert_eq!(CareType::Childcare.anchor_key(), "childcare");
        assert_eq!(CareType::Eldercare.anchor_key(), "eldercare");
        assert_eq!(CareType::DisabilitySupport.anchor_key(), "disability");
        assert_eq!(CareType::PostSurgery.anchor_key(), "postsurgery");
        assert_eq!(CareType::MentalHealth.anchor_key(), "mentalhealth");
        assert_eq!(CareType::Respite.anchor_key(), "respite");
    }

    #[test]
    fn care_type_anchor_key_other_lowercases_and_replaces_spaces() {
        let ct = CareType::Other("Home Health Aide".to_string());
        assert_eq!(ct.anchor_key(), "other_home_health_aide");
    }

    #[test]
    fn care_type_anchor_key_other_empty_string() {
        let ct = CareType::Other(String::new());
        assert_eq!(ct.anchor_key(), "other_");
    }

    // ── CarePlan serde roundtrip ────────────────────────────────────────

    #[test]
    fn care_plan_serde_roundtrip() {
        let agent_b = AgentPubKey::from_raw_36(vec![2u8; 36]);
        let plan = CarePlan {
            recipient: fake_agent(),
            title: "Daily Care for Mom".to_string(),
            description: "Morning and evening check-ins".to_string(),
            care_type: CareType::Eldercare,
            schedule: "Mon/Wed/Fri mornings".to_string(),
            caregivers: vec![agent_b],
            status: PlanStatus::Active,
            created_at: Timestamp::from_micros(1000),
            updated_at: Timestamp::from_micros(2000),
            hours_per_week: 12.0,
            special_instructions: "Allergic to penicillin".to_string(),
        };
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: CarePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    // ── CareSession serde roundtrip ─────────────────────────────────────

    #[test]
    fn care_session_serde_roundtrip() {
        let session = CareSession {
            plan_hash: fake_action_hash(),
            caregiver: fake_agent(),
            started_at: Timestamp::from_micros(1000),
            ended_at: Timestamp::from_micros(5000),
            hours: 4.0,
            notes: "Went well".to_string(),
            tasks_completed: vec!["Medication".to_string(), "Lunch".to_string()],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CareSession = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, session);
    }

    #[test]
    fn care_session_empty_tasks_roundtrip() {
        let session = CareSession {
            plan_hash: fake_action_hash(),
            caregiver: fake_agent(),
            started_at: Timestamp::from_micros(0),
            ended_at: Timestamp::from_micros(3600),
            hours: 1.0,
            notes: String::new(),
            tasks_completed: vec![],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CareSession = serde_json::from_str(&json).unwrap();
        assert!(decoded.tasks_completed.is_empty());
        assert!(decoded.notes.is_empty());
    }

    // ── PlanStatus transition serde tests ─────────────────────────────────

    #[test]
    fn plan_status_draft_to_active_serde() {
        let input_draft = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Draft,
        };
        let input_active = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Active,
        };
        let json_draft = serde_json::to_string(&input_draft).unwrap();
        let json_active = serde_json::to_string(&input_active).unwrap();
        let decoded_draft: UpdatePlanStatusInput = serde_json::from_str(&json_draft).unwrap();
        let decoded_active: UpdatePlanStatusInput = serde_json::from_str(&json_active).unwrap();
        assert_eq!(decoded_draft.new_status, PlanStatus::Draft);
        assert_eq!(decoded_active.new_status, PlanStatus::Active);
    }

    #[test]
    fn plan_status_active_to_paused_serde() {
        let input = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Paused,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePlanStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, PlanStatus::Paused);
    }

    #[test]
    fn plan_status_active_to_completed_serde() {
        let input = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Completed,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePlanStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, PlanStatus::Completed);
    }

    #[test]
    fn plan_status_active_to_cancelled_serde() {
        let input = UpdatePlanStatusInput {
            plan_hash: fake_action_hash(),
            new_status: PlanStatus::Cancelled,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePlanStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, PlanStatus::Cancelled);
    }

    // ── CarePlan with multiple caregivers ─────────────────────────────────

    #[test]
    fn care_plan_multiple_caregivers_serde_roundtrip() {
        let cg1 = AgentPubKey::from_raw_36(vec![2u8; 36]);
        let cg2 = AgentPubKey::from_raw_36(vec![3u8; 36]);
        let cg3 = AgentPubKey::from_raw_36(vec![4u8; 36]);
        let plan = CarePlan {
            recipient: fake_agent(),
            title: "Team Care Plan".to_string(),
            description: "Multiple caregivers assigned".to_string(),
            care_type: CareType::DisabilitySupport,
            schedule: "Daily 8am-8pm".to_string(),
            caregivers: vec![cg1.clone(), cg2.clone(), cg3.clone()],
            status: PlanStatus::Active,
            created_at: Timestamp::from_micros(1000),
            updated_at: Timestamp::from_micros(2000),
            hours_per_week: 84.0,
            special_instructions: "Wheelchair accessible".to_string(),
        };
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: CarePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.caregivers.len(), 3);
        assert_eq!(decoded.caregivers[0], cg1);
        assert_eq!(decoded.caregivers[1], cg2);
        assert_eq!(decoded.caregivers[2], cg3);
    }

    // ── CarePlan status field values ──────────────────────────────────────

    #[test]
    fn care_plan_all_statuses_serde() {
        let statuses = vec![
            PlanStatus::Draft,
            PlanStatus::Active,
            PlanStatus::Paused,
            PlanStatus::Completed,
            PlanStatus::Cancelled,
        ];
        for status in statuses {
            let plan = CarePlan {
                recipient: fake_agent(),
                title: "Status Test".to_string(),
                description: "Testing status field".to_string(),
                care_type: CareType::Childcare,
                schedule: "Weekdays".to_string(),
                caregivers: vec![AgentPubKey::from_raw_36(vec![2u8; 36])],
                status: status.clone(),
                created_at: Timestamp::from_micros(0),
                updated_at: Timestamp::from_micros(0),
                hours_per_week: 10.0,
                special_instructions: String::new(),
            };
            let json = serde_json::to_string(&plan).unwrap();
            let decoded: CarePlan = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ── CarePlan hours boundary values ────────────────────────────────────

    #[test]
    fn care_plan_fractional_hours_serde() {
        let plan = CarePlan {
            recipient: fake_agent(),
            title: "Fractional Hours Plan".to_string(),
            description: "Testing edge case hours".to_string(),
            care_type: CareType::Respite,
            schedule: "As needed".to_string(),
            caregivers: vec![AgentPubKey::from_raw_36(vec![2u8; 36])],
            status: PlanStatus::Active,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
            hours_per_week: 0.25,
            special_instructions: String::new(),
        };
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: CarePlan = serde_json::from_str(&json).unwrap();
        assert!((decoded.hours_per_week - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn care_plan_max_hours_168_serde() {
        let plan = CarePlan {
            recipient: fake_agent(),
            title: "24/7 Care".to_string(),
            description: "Around the clock".to_string(),
            care_type: CareType::Eldercare,
            schedule: "24/7".to_string(),
            caregivers: vec![AgentPubKey::from_raw_36(vec![2u8; 36])],
            status: PlanStatus::Active,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
            hours_per_week: 168.0,
            special_instructions: String::new(),
        };
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: CarePlan = serde_json::from_str(&json).unwrap();
        assert!((decoded.hours_per_week - 168.0).abs() < f32::EPSILON);
    }

    // ── CareSession with many tasks ──────────────────────────────────────

    #[test]
    fn care_session_many_tasks_serde() {
        let tasks: Vec<String> = (0..10).map(|i| format!("Task {}", i)).collect();
        let session = CareSession {
            plan_hash: fake_action_hash(),
            caregiver: fake_agent(),
            started_at: Timestamp::from_micros(1000),
            ended_at: Timestamp::from_micros(5000),
            hours: 4.0,
            notes: "Productive session".to_string(),
            tasks_completed: tasks.clone(),
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CareSession = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.tasks_completed.len(), 10);
        assert_eq!(decoded.tasks_completed, tasks);
    }

    // ── PlanSessionSummary boundary values ────────────────────────────────

    #[test]
    fn plan_session_summary_large_values_roundtrip() {
        let summary = PlanSessionSummary {
            plan_hash: fake_action_hash(),
            total_sessions: u32::MAX,
            total_hours: f32::MAX,
            caregivers_active: u32::MAX,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: PlanSessionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_sessions, u32::MAX);
        assert_eq!(decoded.caregivers_active, u32::MAX);
    }

    #[test]
    fn plan_session_summary_fractional_hours_roundtrip() {
        let summary = PlanSessionSummary {
            plan_hash: fake_action_hash(),
            total_sessions: 5,
            total_hours: 12.75,
            caregivers_active: 2,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: PlanSessionSummary = serde_json::from_str(&json).unwrap();
        assert!((decoded.total_hours - 12.75).abs() < f32::EPSILON);
    }

    // ── CareType::anchor_key edge cases ──────────────────────────────────

    #[test]
    fn care_type_anchor_key_other_multiple_spaces() {
        let ct = CareType::Other("Post   Op   Care".to_string());
        let key = ct.anchor_key();
        assert_eq!(key, "other_post___op___care");
    }

    #[test]
    fn care_type_anchor_key_other_unicode() {
        let ct = CareType::Other("Soins Intensifs".to_string());
        let key = ct.anchor_key();
        assert_eq!(key, "other_soins_intensifs");
    }

    // ── CareSession hours boundary ───────────────────────────────────────

    #[test]
    fn care_session_max_hours_24_serde() {
        let session = CareSession {
            plan_hash: fake_action_hash(),
            caregiver: fake_agent(),
            started_at: Timestamp::from_micros(0),
            ended_at: Timestamp::from_micros(86_400_000_000),
            hours: 24.0,
            notes: "Full day".to_string(),
            tasks_completed: vec![],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CareSession = serde_json::from_str(&json).unwrap();
        assert!((decoded.hours - 24.0).abs() < f32::EPSILON);
    }

    #[test]
    fn care_session_fractional_hours_serde() {
        let session = CareSession {
            plan_hash: fake_action_hash(),
            caregiver: fake_agent(),
            started_at: Timestamp::from_micros(0),
            ended_at: Timestamp::from_micros(1800_000_000),
            hours: 0.5,
            notes: "Quick check-in".to_string(),
            tasks_completed: vec!["Blood pressure check".to_string()],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CareSession = serde_json::from_str(&json).unwrap();
        assert!((decoded.hours - 0.5).abs() < f32::EPSILON);
    }
}
