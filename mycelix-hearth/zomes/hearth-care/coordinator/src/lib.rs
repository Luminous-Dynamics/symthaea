//! Hearth Care Coordinator Zome
//! Business logic for care scheduling, task completion signals,
//! care swaps, and meal planning.

use hdk::prelude::*;
use hearth_coordinator_common::{decode_zome_response, get_latest_record, records_from_links, require_membership};
use hearth_care_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    GovernanceEligibility, GovernanceRequirement,
    gate_consciousness, requirement_for_basic,
};

// ============================================================================
// Consciousness Gating
// ============================================================================

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("hearth_bridge", requirement, action_name)
}

// ============================================================================
// Input Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCareScheduleInput {
    pub hearth_hash: ActionHash,
    pub care_type: CareType,
    pub title: String,
    pub description: String,
    pub assigned_to: AgentPubKey,
    pub recurrence: Recurrence,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteTaskInput {
    pub schedule_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeSwapInput {
    pub hearth_hash: ActionHash,
    pub original_schedule_hash: ActionHash,
    pub swap_date: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMealPlanInput {
    pub hearth_hash: ActionHash,
    pub week_start: Timestamp,
    pub meals: Vec<PlannedMeal>,
    pub shopper: AgentPubKey,
    pub cook: AgentPubKey,
    pub dietary_notes: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether a care schedule can be completed (must be Active).
fn is_schedule_completable(status: &CareScheduleStatus) -> bool {
    *status == CareScheduleStatus::Active
}

/// Check whether a swap is in a state that can be accepted or declined (must be Proposed).
fn is_swap_pending(status: &SwapStatus) -> bool {
    *status == SwapStatus::Proposed
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new care schedule and link it to the hearth and assigned agent.
#[hdk_extern]
pub fn create_care_schedule(input: CreateCareScheduleInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_care_schedule")?;
    require_membership(&input.hearth_hash)?;
    let schedule = CareSchedule {
        hearth_hash: input.hearth_hash.clone(),
        care_type: input.care_type,
        title: input.title,
        description: input.description,
        assigned_to: input.assigned_to.clone(),
        recurrence: input.recurrence,
        notes: input.notes,
        status: CareScheduleStatus::Active,
        completed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::CareSchedule(schedule))?;

    // Link hearth -> schedule
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToSchedules,
        (),
    )?;

    // Link agent -> schedule
    create_link(
        input.assigned_to,
        action_hash.clone(),
        LinkTypes::AgentToSchedules,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created care schedule".into()
    )))
}

/// Mark a care task as completed. Sets completed_at timestamp on the entry,
/// emits a CareTaskCompleted signal, and returns the updated record.
///
/// Auth: only the assignee or a guardian can complete a task.
/// Status: only Active schedules can be completed.
#[hdk_extern]
pub fn complete_task(input: CompleteTaskInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "complete_task")?;
    let now = sys_time()?;

    let record = get(input.schedule_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Care schedule not found".into())
    ))?;

    let schedule: CareSchedule = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid care schedule entry".into()
        )))?;

    // 1. Status check: only Active schedules can be completed
    if !is_schedule_completable(&schedule.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot complete a schedule with status {:?} (must be Active)",
            schedule.status
        ))));
    }

    let caller = agent_info()?.agent_initial_pubkey;

    // 2. Auth: caller must be the assignee OR a guardian
    if caller != schedule.assigned_to {
        let caller_role: Option<MemberRole> = decode_zome_response(
            call(
                CallTargetCell::Local,
                ZomeName::new("hearth_kinship"),
                FunctionName::new("get_caller_role"),
                None,
                schedule.hearth_hash.clone(),
            )?,
            "get_caller_role",
        )?;

        let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
            "You are not an active member of this hearth".into()
        )))?;

        if !role.is_guardian() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the assignee or a guardian can complete a care task".into()
            )));
        }
    }

    // Update the entry with completed_at timestamp
    let updated = CareSchedule {
        completed_at: Some(now),
        status: CareScheduleStatus::Completed,
        ..schedule.clone()
    };
    let updated_hash = update_entry(
        input.schedule_hash.clone(),
        &EntryTypes::CareSchedule(updated),
    )?;

    let signal = HearthSignal::CareTaskCompleted {
        assignee: caller,
        schedule_hash: input.schedule_hash,
        care_type: schedule.care_type,
    };

    emit_signal(&signal)?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated care schedule".into()
    )))
}

/// Propose a care task swap with another hearth member.
#[hdk_extern]
pub fn propose_swap(input: ProposeSwapInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "propose_swap")?;
    require_membership(&input.hearth_hash)?;
    let schedule_record =
        get(input.original_schedule_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Original care schedule not found".into())
        ))?;

    let schedule: CareSchedule = schedule_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid care schedule entry".into()
        )))?;

    let caller = agent_info()?.agent_initial_pubkey;

    let swap = CareSwap {
        hearth_hash: input.hearth_hash.clone(),
        requester: caller,
        responder: schedule.assigned_to,
        original_schedule_hash: input.original_schedule_hash.clone(),
        swap_date: input.swap_date,
        status: SwapStatus::Proposed,
    };

    let action_hash = create_entry(&EntryTypes::CareSwap(swap))?;

    // Link hearth -> swap
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToSwaps,
        (),
    )?;

    // Link schedule -> swap
    create_link(
        input.original_schedule_hash,
        action_hash.clone(),
        LinkTypes::ScheduleToSwaps,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created care swap".into()
    )))
}

/// Accept a proposed care swap.
///
/// Auth: only the swap's responder (target) or a guardian can accept.
/// Status: only Proposed swaps can be accepted.
#[hdk_extern]
pub fn accept_swap(swap_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "accept_swap")?;
    update_swap_status(swap_hash, SwapStatus::Accepted)
}

/// Decline a proposed care swap.
///
/// Auth: only the swap's responder (target) or a guardian can decline.
/// Status: only Proposed swaps can be declined.
#[hdk_extern]
pub fn decline_swap(swap_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "decline_swap")?;
    update_swap_status(swap_hash, SwapStatus::Declined)
}

/// Create a weekly meal plan for the hearth.
#[hdk_extern]
pub fn create_meal_plan(input: CreateMealPlanInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_meal_plan")?;
    require_membership(&input.hearth_hash)?;
    let plan = MealPlan {
        hearth_hash: input.hearth_hash.clone(),
        week_start: input.week_start,
        meals: input.meals,
        shopper: input.shopper,
        cook: input.cook,
        dietary_notes: input.dietary_notes,
    };

    let action_hash = create_entry(&EntryTypes::MealPlan(plan))?;

    // Link hearth -> meal plan
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToMealPlans,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created meal plan".into()
    )))
}

/// Get all care duties assigned to the calling agent.
#[hdk_extern]
pub fn get_my_care_duties(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(caller, LinkTypes::AgentToSchedules)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get the full care schedule for a hearth.
#[hdk_extern]
pub fn get_hearth_schedule(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToSchedules)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all meal plans for a hearth.
#[hdk_extern]
pub fn get_hearth_meal_plans(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMealPlans)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Create a care digest: query HearthToSchedules links, filter schedules
/// with completed_at within the epoch window, aggregate per assignee.
/// Returns Vec<CareSummary> for inclusion in the WeeklyDigest.
#[hdk_extern]
pub fn create_care_digest(input: DigestEpochInput) -> ExternResult<Vec<CareSummary>> {
    if input.epoch_start >= input.epoch_end {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "epoch_start must be before epoch_end".into()
        )));
    }

    let links = get_links(
        LinkQuery::try_new(input.hearth_hash, LinkTypes::HearthToSchedules)?,
        GetStrategy::default(),
    )?;

    // Aggregate completed tasks per assignee
    let mut assignee_stats: std::collections::HashMap<AgentPubKey, (u32, u32)> =
        std::collections::HashMap::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            let schedule: CareSchedule = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid care schedule entry".into()
                )))?;

            // Filter by completed_at within half-open epoch window [start, end)
            if let Some(completed_at) = schedule.completed_at {
                if completed_at >= input.epoch_start && completed_at < input.epoch_end {
                    let entry = assignee_stats.entry(schedule.assigned_to).or_insert((0, 0));
                    entry.0 += 1; // tasks_completed
                    entry.1 += 100; // hours_hundredths: estimate 1 hour per task
                }
            }
        }
    }

    let summaries: Vec<CareSummary> = assignee_stats
        .into_iter()
        .map(
            |(assignee, (tasks_completed, hours_hundredths))| CareSummary {
                assignee,
                tasks_completed,
                hours_hundredths,
            },
        )
        .collect();

    Ok(summaries)
}

// ============================================================================
// Internal Helpers
// ============================================================================

fn update_swap_status(swap_hash: ActionHash, new_status: SwapStatus) -> ExternResult<Record> {
    let record = get(swap_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Care swap not found".into())
    ))?;

    let mut swap: CareSwap = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid care swap entry".into()
        )))?;

    // 1. Status check: only Proposed swaps can be accepted/declined
    if !is_swap_pending(&swap.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot change a swap with status {:?} (must be Proposed)",
            swap.status
        ))));
    }

    // 2. Auth: caller must be the responder (target) OR a guardian
    let caller = agent_info()?.agent_initial_pubkey;
    if caller != swap.responder {
        let caller_role: Option<MemberRole> = decode_zome_response(
            call(
                CallTargetCell::Local,
                ZomeName::new("hearth_kinship"),
                FunctionName::new("get_caller_role"),
                None,
                swap.hearth_hash.clone(),
            )?,
            "get_caller_role",
        )?;

        let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
            "You are not an active member of this hearth".into()
        )))?;

        if !role.is_guardian() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the swap responder or a guardian can accept/decline a swap".into()
            )));
        }
    }

    let hearth_hash_for_signal = swap.hearth_hash.clone();
    let status_for_signal = new_status.clone();
    swap.status = new_status;

    let updated_hash = update_entry(swap_hash.clone(), &EntryTypes::CareSwap(swap))?;

    match status_for_signal {
        SwapStatus::Accepted => {
            emit_signal(&HearthSignal::SwapAccepted {
                swap_hash,
                hearth_hash: hearth_hash_for_signal,
            })?;
        }
        SwapStatus::Declined => {
            emit_signal(&HearthSignal::SwapDeclined {
                swap_hash,
                hearth_hash: hearth_hash_for_signal,
            })?;
        }
        _ => {}
    }

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated care swap".into()
    )))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xaa; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xbb; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa1; 36])
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    // -- Input type serde roundtrips --

    #[test]
    fn test_create_care_schedule_input_serde() {
        let input = CreateCareScheduleInput {
            hearth_hash: action_hash_1(),
            care_type: CareType::Childcare,
            title: "School run".to_string(),
            description: "Morning school drop-off".to_string(),
            assigned_to: agent_a(),
            recurrence: Recurrence::Daily,
            notes: "Pack lunches".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateCareScheduleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.title, "School run");
    }

    #[test]
    fn test_complete_task_input_serde() {
        let input = CompleteTaskInput {
            schedule_hash: action_hash_1(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CompleteTaskInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schedule_hash, action_hash_1());
    }

    #[test]
    fn test_propose_swap_input_serde() {
        let input = ProposeSwapInput {
            hearth_hash: action_hash_1(),
            original_schedule_hash: ActionHash::from_raw_36(vec![0xa2; 36]),
            swap_date: ts(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ProposeSwapInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hearth_hash, action_hash_1());
    }

    #[test]
    fn test_create_meal_plan_input_serde() {
        let input = CreateMealPlanInput {
            hearth_hash: action_hash_1(),
            week_start: ts(),
            meals: vec![PlannedMeal {
                day: "Monday".to_string(),
                meal_type: "Dinner".to_string(),
                recipe: "Tacos".to_string(),
                servings: 4,
            }],
            shopper: agent_a(),
            cook: agent_b(),
            dietary_notes: "Gluten free".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateMealPlanInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.meals.len(), 1);
        assert_eq!(back.dietary_notes, "Gluten free");
    }

    #[test]
    fn test_create_meal_plan_input_empty_meals() {
        let input = CreateMealPlanInput {
            hearth_hash: action_hash_1(),
            week_start: ts(),
            meals: vec![],
            shopper: agent_a(),
            cook: agent_b(),
            dietary_notes: "".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateMealPlanInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.meals.len(), 0);
    }

    #[test]
    fn test_create_care_schedule_input_custom_care_type() {
        let input = CreateCareScheduleInput {
            hearth_hash: action_hash_1(),
            care_type: CareType::Custom("Tutoring".to_string()),
            title: "Math tutoring".to_string(),
            description: "Help with algebra".to_string(),
            assigned_to: agent_a(),
            recurrence: Recurrence::Weekly,
            notes: "".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateCareScheduleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.care_type, CareType::Custom("Tutoring".to_string()));
    }

    #[test]
    fn test_propose_swap_input_different_hashes() {
        let input = ProposeSwapInput {
            hearth_hash: action_hash_1(),
            original_schedule_hash: ActionHash::from_raw_36(vec![0xff; 36]),
            swap_date: Timestamp::from_micros(5_000_000),
        };
        assert_ne!(input.hearth_hash, input.original_schedule_hash);
    }

    #[test]
    fn test_create_meal_plan_input_full_week() {
        let days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        let meal_types = ["Breakfast", "Lunch", "Dinner"];
        let meals: Vec<PlannedMeal> = days
            .iter()
            .flat_map(|day| {
                meal_types.iter().map(move |mt| PlannedMeal {
                    day: day.to_string(),
                    meal_type: mt.to_string(),
                    recipe: format!("{} {}", day, mt),
                    servings: 4,
                })
            })
            .collect();
        assert_eq!(meals.len(), 21);

        let input = CreateMealPlanInput {
            hearth_hash: action_hash_1(),
            week_start: ts(),
            meals,
            shopper: agent_a(),
            cook: agent_b(),
            dietary_notes: "Vegetarian".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateMealPlanInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.meals.len(), 21);
    }

    #[test]
    fn test_complete_task_input_clone() {
        let input = CompleteTaskInput {
            schedule_hash: action_hash_1(),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("schedule_hash"));
    }

    #[test]
    fn test_create_care_schedule_input_all_care_types() {
        let care_types = vec![
            CareType::Childcare,
            CareType::Eldercare,
            CareType::PetCare,
            CareType::Chore,
            CareType::MealPrep,
            CareType::Medical,
            CareType::Emotional,
        ];
        for ct in care_types {
            let input = CreateCareScheduleInput {
                hearth_hash: action_hash_1(),
                care_type: ct.clone(),
                title: "Test".to_string(),
                description: "".to_string(),
                assigned_to: agent_a(),
                recurrence: Recurrence::Daily,
                notes: "".to_string(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let back: CreateCareScheduleInput = serde_json::from_str(&json).unwrap();
            assert_eq!(back.care_type, ct);
        }
    }

    // ====================================================================
    // Pure helper: is_schedule_completable
    // ====================================================================

    #[test]
    fn schedule_completable_active() {
        assert!(is_schedule_completable(&CareScheduleStatus::Active));
    }

    #[test]
    fn schedule_not_completable_paused() {
        assert!(!is_schedule_completable(&CareScheduleStatus::Paused));
    }

    #[test]
    fn schedule_not_completable_completed() {
        assert!(!is_schedule_completable(&CareScheduleStatus::Completed));
    }

    #[test]
    fn schedule_completable_exhaustive() {
        let statuses = vec![
            (CareScheduleStatus::Active, true),
            (CareScheduleStatus::Paused, false),
            (CareScheduleStatus::Completed, false),
        ];
        for (status, expected) in statuses {
            assert_eq!(
                is_schedule_completable(&status),
                expected,
                "mismatch for {:?}",
                status
            );
        }
    }

    // ====================================================================
    // Pure helper: is_swap_pending
    // ====================================================================

    #[test]
    fn swap_pending_proposed() {
        assert!(is_swap_pending(&SwapStatus::Proposed));
    }

    #[test]
    fn swap_not_pending_accepted() {
        assert!(!is_swap_pending(&SwapStatus::Accepted));
    }

    #[test]
    fn swap_not_pending_declined() {
        assert!(!is_swap_pending(&SwapStatus::Declined));
    }

    #[test]
    fn swap_not_pending_completed() {
        assert!(!is_swap_pending(&SwapStatus::Completed));
    }

    #[test]
    fn swap_pending_exhaustive() {
        let statuses = vec![
            (SwapStatus::Proposed, true),
            (SwapStatus::Accepted, false),
            (SwapStatus::Declined, false),
            (SwapStatus::Completed, false),
        ];
        for (status, expected) in statuses {
            assert_eq!(
                is_swap_pending(&status),
                expected,
                "mismatch for {:?}",
                status
            );
        }
    }

    // ====================================================================
    // Scenario tests: care task lifecycle
    // ====================================================================

    #[test]
    fn scenario_active_schedule_can_be_completed() {
        // A new schedule starts Active and should be completable
        let status = CareScheduleStatus::Active;
        assert!(is_schedule_completable(&status));
    }

    #[test]
    fn scenario_completed_schedule_cannot_be_re_completed() {
        // Once completed, cannot complete again (idempotency guard)
        let status = CareScheduleStatus::Completed;
        assert!(!is_schedule_completable(&status));
    }

    #[test]
    fn scenario_paused_schedule_cannot_be_completed() {
        // A paused schedule must be re-activated before completion
        let status = CareScheduleStatus::Paused;
        assert!(!is_schedule_completable(&status));
    }

    // ====================================================================
    // Scenario tests: swap lifecycle
    // ====================================================================

    #[test]
    fn scenario_proposed_swap_can_be_accepted() {
        let status = SwapStatus::Proposed;
        assert!(is_swap_pending(&status));
    }

    #[test]
    fn scenario_proposed_swap_can_be_declined() {
        let status = SwapStatus::Proposed;
        assert!(is_swap_pending(&status));
    }

    #[test]
    fn scenario_accepted_swap_cannot_be_declined() {
        // Once accepted, the swap is settled
        let status = SwapStatus::Accepted;
        assert!(!is_swap_pending(&status));
    }

    #[test]
    fn scenario_declined_swap_cannot_be_accepted() {
        // Once declined, the swap is settled
        let status = SwapStatus::Declined;
        assert!(!is_swap_pending(&status));
    }

    #[test]
    fn scenario_completed_swap_is_terminal() {
        // Completed swaps are terminal state
        let status = SwapStatus::Completed;
        assert!(!is_swap_pending(&status));
    }

    // ====================================================================
    // Scenario tests: combined lifecycle flows
    // ====================================================================

    #[test]
    fn scenario_full_care_lifecycle() {
        // Create -> Active -> Complete
        let initial = CareScheduleStatus::Active;
        assert!(is_schedule_completable(&initial));

        // After completion, cannot complete again
        let final_status = CareScheduleStatus::Completed;
        assert!(!is_schedule_completable(&final_status));
    }

    #[test]
    fn scenario_full_swap_lifecycle_accept() {
        // Propose -> Proposed -> Accept -> Accepted (terminal)
        let proposed = SwapStatus::Proposed;
        assert!(is_swap_pending(&proposed));

        let accepted = SwapStatus::Accepted;
        assert!(!is_swap_pending(&accepted));
    }

    #[test]
    fn scenario_full_swap_lifecycle_decline() {
        // Propose -> Proposed -> Decline -> Declined (terminal)
        let proposed = SwapStatus::Proposed;
        assert!(is_swap_pending(&proposed));

        let declined = SwapStatus::Declined;
        assert!(!is_swap_pending(&declined));
    }

    #[test]
    fn scenario_guardian_auth_roles() {
        // Verify which roles count as guardian (used for auth fallback)
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
        assert!(!MemberRole::Ancestor.is_guardian());
    }

    // ====================================================================
    // Complex business logic scenario tests
    // ====================================================================

    /// Test the full swap lifecycle: Proposed -> Accept or Decline.
    /// Verify that Accepted, Declined, and Completed are all terminal.
    #[test]
    fn scenario_swap_lifecycle_complete() {
        // Proposed: can accept or decline
        assert!(
            is_swap_pending(&SwapStatus::Proposed),
            "Proposed swap should be actionable"
        );

        // After acceptance: terminal
        assert!(
            !is_swap_pending(&SwapStatus::Accepted),
            "Accepted swap must not be re-actionable"
        );

        // After decline: terminal
        assert!(
            !is_swap_pending(&SwapStatus::Declined),
            "Declined swap must not be re-actionable"
        );

        // Completed: also terminal
        assert!(
            !is_swap_pending(&SwapStatus::Completed),
            "Completed swap must not be re-actionable"
        );

        // Verify exhaustive: only Proposed is actionable
        let all_statuses = [
            SwapStatus::Proposed,
            SwapStatus::Accepted,
            SwapStatus::Declined,
            SwapStatus::Completed,
        ];
        let actionable_count = all_statuses.iter().filter(|s| is_swap_pending(s)).count();
        assert_eq!(
            actionable_count, 1,
            "Exactly one SwapStatus should be actionable"
        );
    }

    /// Test schedule lifecycle with pause: Active -> completable,
    /// Paused -> not completable (must reactivate), Completed -> not completable again.
    #[test]
    fn scenario_schedule_lifecycle_with_pause() {
        // Active: can be completed
        assert!(
            is_schedule_completable(&CareScheduleStatus::Active),
            "Active schedule should be completable"
        );

        // Paused: cannot be completed (must reactivate first)
        assert!(
            !is_schedule_completable(&CareScheduleStatus::Paused),
            "Paused schedule must be reactivated before completion"
        );

        // Completed: cannot be completed again (idempotency)
        assert!(
            !is_schedule_completable(&CareScheduleStatus::Completed),
            "Completed schedule must not be re-completed"
        );

        // Verify exhaustive: only Active is completable
        let all_statuses = [
            CareScheduleStatus::Active,
            CareScheduleStatus::Paused,
            CareScheduleStatus::Completed,
        ];
        let completable_count = all_statuses
            .iter()
            .filter(|s| is_schedule_completable(s))
            .count();
        assert_eq!(
            completable_count, 1,
            "Exactly one CareScheduleStatus should be completable"
        );
    }

    /// Test that a CreateMealPlanInput with 21 meals (7 days x 3 meals)
    /// serializes correctly and all fields roundtrip.
    #[test]
    fn scenario_full_meal_plan_input_validation() {
        let days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        let meal_types = ["Breakfast", "Lunch", "Dinner"];

        let meals: Vec<PlannedMeal> = days
            .iter()
            .flat_map(|day| {
                meal_types.iter().map(move |mt| PlannedMeal {
                    day: day.to_string(),
                    meal_type: mt.to_string(),
                    recipe: format!("{} {}", day, mt),
                    servings: 4,
                })
            })
            .collect();

        assert_eq!(meals.len(), 21, "Should have 7 days x 3 meals = 21 meals");

        let input = CreateMealPlanInput {
            hearth_hash: action_hash_1(),
            week_start: ts(),
            meals,
            shopper: agent_a(),
            cook: agent_b(),
            dietary_notes: "Vegetarian, nut-free".to_string(),
        };

        // Serialize and deserialize roundtrip
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateMealPlanInput = serde_json::from_str(&json).unwrap();

        // Verify all fields roundtrip correctly
        assert_eq!(back.meals.len(), 21);
        assert_eq!(back.hearth_hash, action_hash_1());
        assert_eq!(back.shopper, agent_a());
        assert_eq!(back.cook, agent_b());
        assert_eq!(back.dietary_notes, "Vegetarian, nut-free");

        // Verify each meal has the expected structure
        for meal in &back.meals {
            assert!(
                days.contains(&meal.day.as_str()),
                "Unexpected day: {}",
                meal.day
            );
            assert!(
                meal_types.contains(&meal.meal_type.as_str()),
                "Unexpected meal_type: {}",
                meal.meal_type
            );
            assert_eq!(meal.servings, 4);
            assert!(!meal.recipe.is_empty());
        }

        // Verify we have exactly 3 meals per day
        for day in &days {
            let day_meals: Vec<_> = back
                .meals
                .iter()
                .filter(|m| m.day == *day)
                .collect();
            assert_eq!(
                day_meals.len(),
                3,
                "{} should have exactly 3 meals",
                day
            );
        }
    }

    /// Test all MemberRole variants for is_guardian(), documenting which
    /// roles can manage care operations and which cannot.
    #[test]
    fn scenario_guardian_role_eligibility_for_care_ops() {
        // Guardian roles: can create schedules, complete tasks on behalf,
        // accept/decline swaps for others
        let guardian_roles = [MemberRole::Founder, MemberRole::Elder, MemberRole::Adult];
        for role in &guardian_roles {
            assert!(
                role.is_guardian(),
                "{:?} should be a guardian with care management authority",
                role
            );
        }

        // Non-guardian roles: can only manage their own tasks
        let non_guardian_roles = [
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        for role in &non_guardian_roles {
            assert!(
                !role.is_guardian(),
                "{:?} should NOT have guardian-level care management authority",
                role
            );
        }

        // Verify the split is exhaustive: 3 guardians + 4 non-guardians = 7 total
        assert_eq!(guardian_roles.len() + non_guardian_roles.len(), 7);
    }
}
