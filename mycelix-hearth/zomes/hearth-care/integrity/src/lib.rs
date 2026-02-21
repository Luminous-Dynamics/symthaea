//! Hearth Care Integrity Zome
//! Defines entry types and validation for care scheduling, task swaps,
//! and meal planning within a hearth.

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A recurring or one-time care task assigned to a hearth member.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareSchedule {
    /// The hearth this schedule belongs to.
    pub hearth_hash: ActionHash,
    /// Category of care activity.
    pub care_type: CareType,
    /// Short title for the task.
    pub title: String,
    /// Longer description of the task.
    pub description: String,
    /// The member responsible for this task.
    pub assigned_to: AgentPubKey,
    /// How often this task recurs.
    pub recurrence: Recurrence,
    /// Free-form notes (instructions, preferences, etc.).
    pub notes: String,
    /// Current status of the schedule.
    pub status: CareScheduleStatus,
}

/// A request to swap a care task between two members.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareSwap {
    /// The hearth this swap belongs to.
    pub hearth_hash: ActionHash,
    /// The member requesting the swap.
    pub requester: AgentPubKey,
    /// The member being asked to take over.
    pub responder: AgentPubKey,
    /// The care schedule being swapped.
    pub original_schedule_hash: ActionHash,
    /// The date of the swap.
    pub swap_date: Timestamp,
    /// Current status of the swap request.
    pub status: SwapStatus,
}

/// A weekly meal plan for the hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MealPlan {
    /// The hearth this meal plan belongs to.
    pub hearth_hash: ActionHash,
    /// Start of the week this plan covers.
    pub week_start: Timestamp,
    /// Individual planned meals (max 21: 3 meals/day x 7 days).
    pub meals: Vec<PlannedMeal>,
    /// The member responsible for shopping.
    pub shopper: AgentPubKey,
    /// The member responsible for cooking.
    pub cook: AgentPubKey,
    /// Dietary restrictions, allergies, or preferences.
    pub dietary_notes: String,
}

/// A single planned meal within a MealPlan.
/// This is a helper struct, NOT an entry type.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PlannedMeal {
    /// Day of the week (e.g. "Monday").
    pub day: String,
    /// Type of meal (e.g. "Breakfast", "Lunch", "Dinner").
    pub meal_type: String,
    /// Recipe name or description.
    pub recipe: String,
    /// Number of servings.
    pub servings: u32,
}

// ============================================================================
// Entry + Link Type Enums
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CareSchedule(CareSchedule),
    CareSwap(CareSwap),
    MealPlan(MealPlan),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> all care schedules in that hearth.
    HearthToSchedules,
    /// Agent -> all schedules assigned to that agent.
    AgentToSchedules,
    /// Hearth -> all care swaps in that hearth.
    HearthToSwaps,
    /// Schedule -> all swap requests for that schedule.
    ScheduleToSwaps,
    /// Hearth -> all meal plans in that hearth.
    HearthToMealPlans,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::CareSchedule(schedule) => validate_schedule(&schedule, &action),
                EntryTypes::CareSwap(swap) => validate_swap(&swap),
                EntryTypes::MealPlan(plan) => validate_meal_plan(&plan),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::CareSchedule(schedule) => validate_schedule_update(&schedule),
                EntryTypes::CareSwap(swap) => validate_swap_update(&swap),
                EntryTypes::MealPlan(plan) => validate_meal_plan_update(&plan),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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

fn validate_schedule(
    schedule: &CareSchedule,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    if schedule.title.is_empty() || schedule.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule title must be 1-256 characters".into(),
        ));
    }
    if schedule.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule description must be 0-4096 characters".into(),
        ));
    }
    if schedule.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule notes must be 0-4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_schedule_update(schedule: &CareSchedule) -> ExternResult<ValidateCallbackResult> {
    if schedule.title.is_empty() || schedule.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule title must be 1-256 characters".into(),
        ));
    }
    if schedule.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule description must be 0-4096 characters".into(),
        ));
    }
    if schedule.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care schedule notes must be 0-4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_swap(swap: &CareSwap) -> ExternResult<ValidateCallbackResult> {
    if swap.requester == swap.responder {
        return Ok(ValidateCallbackResult::Invalid(
            "Care swap requester and responder cannot be the same agent".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_swap_update(swap: &CareSwap) -> ExternResult<ValidateCallbackResult> {
    if swap.requester == swap.responder {
        return Ok(ValidateCallbackResult::Invalid(
            "Care swap requester and responder cannot be the same agent".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_meal_plan(plan: &MealPlan) -> ExternResult<ValidateCallbackResult> {
    if plan.meals.len() > 21 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meal plan cannot have more than 21 meals (3 per day x 7 days)".into(),
        ));
    }
    if plan.dietary_notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meal plan dietary notes must be 0-4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_meal_plan_update(plan: &MealPlan) -> ExternResult<ValidateCallbackResult> {
    if plan.meals.len() > 21 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meal plan cannot have more than 21 meals (3 per day x 7 days)".into(),
        ));
    }
    if plan.dietary_notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meal plan dietary notes must be 0-4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Factory helpers --

    fn agent_key_1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xaa; 36])
    }

    fn agent_key_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xbb; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa1; 36])
    }

    fn action_hash_2() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa2; 36])
    }

    fn timestamp_now() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn create_action() -> Create {
        Create {
            author: agent_key_1(),
            timestamp: timestamp_now(),
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

    fn valid_schedule() -> CareSchedule {
        CareSchedule {
            hearth_hash: action_hash_1(),
            care_type: CareType::Childcare,
            title: "Morning school run".to_string(),
            description: "Take kids to school at 8am".to_string(),
            assigned_to: agent_key_1(),
            recurrence: Recurrence::Daily,
            notes: "Pack lunches before leaving".to_string(),
            status: CareScheduleStatus::Active,
        }
    }

    fn valid_swap() -> CareSwap {
        CareSwap {
            hearth_hash: action_hash_1(),
            requester: agent_key_1(),
            responder: agent_key_2(),
            original_schedule_hash: action_hash_2(),
            swap_date: timestamp_now(),
            status: SwapStatus::Proposed,
        }
    }

    fn valid_meal() -> PlannedMeal {
        PlannedMeal {
            day: "Monday".to_string(),
            meal_type: "Dinner".to_string(),
            recipe: "Pasta Primavera".to_string(),
            servings: 4,
        }
    }

    fn valid_meal_plan() -> MealPlan {
        MealPlan {
            hearth_hash: action_hash_1(),
            week_start: timestamp_now(),
            meals: vec![valid_meal()],
            shopper: agent_key_1(),
            cook: agent_key_2(),
            dietary_notes: "No peanuts".to_string(),
        }
    }

    // -- CareSchedule validation --

    #[test]
    fn test_valid_schedule_passes() {
        let result = validate_schedule(&valid_schedule(), &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_empty_title_fails() {
        let mut schedule = valid_schedule();
        schedule.title = "".to_string();
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_schedule_title_at_limit_passes() {
        let mut schedule = valid_schedule();
        schedule.title = "a".repeat(256);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_title_over_limit_fails() {
        let mut schedule = valid_schedule();
        schedule.title = "a".repeat(257);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_schedule_empty_description_passes() {
        let mut schedule = valid_schedule();
        schedule.description = "".to_string();
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_description_at_limit_passes() {
        let mut schedule = valid_schedule();
        schedule.description = "d".repeat(4096);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_description_over_limit_fails() {
        let mut schedule = valid_schedule();
        schedule.description = "d".repeat(4097);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_schedule_notes_at_limit_passes() {
        let mut schedule = valid_schedule();
        schedule.notes = "n".repeat(4096);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_notes_over_limit_fails() {
        let mut schedule = valid_schedule();
        schedule.notes = "n".repeat(4097);
        let result = validate_schedule(&schedule, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- CareSwap validation --

    #[test]
    fn test_valid_swap_passes() {
        let result = validate_swap(&valid_swap()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_swap_same_agent_fails() {
        let mut swap = valid_swap();
        swap.responder = swap.requester.clone();
        let result = validate_swap(&swap).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- MealPlan validation --

    #[test]
    fn test_valid_meal_plan_passes() {
        let result = validate_meal_plan(&valid_meal_plan()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_meal_plan_at_21_meals_passes() {
        let mut plan = valid_meal_plan();
        plan.meals = (0..21).map(|_| valid_meal()).collect();
        let result = validate_meal_plan(&plan).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_meal_plan_over_21_meals_fails() {
        let mut plan = valid_meal_plan();
        plan.meals = (0..22).map(|_| valid_meal()).collect();
        let result = validate_meal_plan(&plan).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_meal_plan_empty_meals_passes() {
        let mut plan = valid_meal_plan();
        plan.meals = vec![];
        let result = validate_meal_plan(&plan).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_meal_plan_dietary_notes_at_limit_passes() {
        let mut plan = valid_meal_plan();
        plan.dietary_notes = "x".repeat(4096);
        let result = validate_meal_plan(&plan).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_meal_plan_dietary_notes_over_limit_fails() {
        let mut plan = valid_meal_plan();
        plan.dietary_notes = "x".repeat(4097);
        let result = validate_meal_plan(&plan).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- Serde roundtrips --

    #[test]
    fn test_care_schedule_serde_roundtrip() {
        let schedule = valid_schedule();
        let json = serde_json::to_string(&schedule).unwrap();
        let back: CareSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(schedule, back);
    }

    #[test]
    fn test_care_swap_serde_roundtrip() {
        let swap = valid_swap();
        let json = serde_json::to_string(&swap).unwrap();
        let back: CareSwap = serde_json::from_str(&json).unwrap();
        assert_eq!(swap, back);
    }

    #[test]
    fn test_meal_plan_serde_roundtrip() {
        let plan = valid_meal_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let back: MealPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn test_planned_meal_serde_roundtrip() {
        let meal = valid_meal();
        let json = serde_json::to_string(&meal).unwrap();
        let back: PlannedMeal = serde_json::from_str(&json).unwrap();
        assert_eq!(meal, back);
    }
}
