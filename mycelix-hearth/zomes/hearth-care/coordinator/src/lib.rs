//! Hearth Care Coordinator Zome
//! Business logic for care scheduling, task completion signals,
//! care swaps, and meal planning.

use hdk::prelude::*;
use hearth_care_integrity::*;
use hearth_types::*;

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

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new care schedule and link it to the hearth and assigned agent.
#[hdk_extern]
pub fn create_care_schedule(input: CreateCareScheduleInput) -> ExternResult<Record> {
    let schedule = CareSchedule {
        hearth_hash: input.hearth_hash.clone(),
        care_type: input.care_type,
        title: input.title,
        description: input.description,
        assigned_to: input.assigned_to.clone(),
        recurrence: input.recurrence,
        notes: input.notes,
        status: CareScheduleStatus::Active,
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

/// Signal that a care task has been completed (H2: signal, not DHT write).
#[hdk_extern]
pub fn complete_task(input: CompleteTaskInput) -> ExternResult<()> {
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

    let caller = agent_info()?.agent_initial_pubkey;

    let signal = HearthSignal::CareTaskCompleted {
        assignee: caller,
        schedule_hash: input.schedule_hash,
        care_type: schedule.care_type,
    };

    emit_signal(&signal)?;

    Ok(())
}

/// Propose a care task swap with another hearth member.
#[hdk_extern]
pub fn propose_swap(input: ProposeSwapInput) -> ExternResult<Record> {
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
#[hdk_extern]
pub fn accept_swap(swap_hash: ActionHash) -> ExternResult<Record> {
    update_swap_status(swap_hash, SwapStatus::Accepted)
}

/// Decline a proposed care swap.
#[hdk_extern]
pub fn decline_swap(swap_hash: ActionHash) -> ExternResult<Record> {
    update_swap_status(swap_hash, SwapStatus::Declined)
}

/// Create a weekly meal plan for the hearth.
#[hdk_extern]
pub fn create_meal_plan(input: CreateMealPlanInput) -> ExternResult<Record> {
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

/// Placeholder for H2 weekly care digest rollup.
/// Will aggregate care completion signals into a WeeklyDigest entry.
#[hdk_extern]
pub fn create_care_digest(hearth_hash: ActionHash) -> ExternResult<()> {
    // H2: Collect care completion signals from the past week,
    // aggregate into a CareSummary, and write to the WeeklyDigest.
    // For now, just emit a signal indicating the digest was requested.
    let caller = agent_info()?.agent_initial_pubkey;

    let signal = HearthSignal::CareTaskCompleted {
        assignee: caller,
        schedule_hash: hearth_hash,
        care_type: CareType::Custom("weekly-digest-request".to_string()),
    };
    emit_signal(&signal)?;

    Ok(())
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

    swap.status = new_status;

    let updated_hash = update_entry(swap_hash, &EntryTypes::CareSwap(swap))?;

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
}
