//! Hearth Resources Coordinator Zome
//!
//! Provides CRUD operations for shared resources, resource loans,
//! and budget tracking with guardian authorization and signal emission.

use hdk::prelude::*;
use hearth_coordinator_common::{decode_zome_response, get_latest_record};
use hearth_resources_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResourceInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub condition: String,
    pub location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LendResourceInput {
    pub resource_hash: ActionHash,
    pub borrower: AgentPubKey,
    pub due_date: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBudgetInput {
    pub hearth_hash: ActionHash,
    pub category: String,
    pub monthly_target_cents: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogExpenseInput {
    pub budget_hash: ActionHash,
    pub amount_cents: u64,
}

// ============================================================================
// Pure Helpers
// ============================================================================

/// Check whether a loan is in an active state (Active or Overdue both count
/// as "active" since the resource has not been returned).
fn is_loan_active(status: &LoanStatus) -> bool {
    matches!(status, LoanStatus::Active | LoanStatus::Overdue)
}

/// Check whether actual spending exceeds the monthly target.
#[allow(dead_code)]
fn is_budget_over_target(actual: u64, target: u64) -> bool {
    actual > target
}

/// Compute budget usage as basis points (0-10000).
/// Returns 0 if target is 0 and actual is 0. Caps at 10000.
#[allow(dead_code)]
fn budget_usage_pct(actual: u64, target: u64) -> u32 {
    if target == 0 {
        if actual == 0 {
            return 0;
        }
        return 10000;
    }
    ((actual as u128 * 10000) / target as u128).min(10000) as u32
}

/// Fetch the caller's role in the given hearth via cross-zome call to kinship.
/// Returns Err if the caller is not an active member.
fn require_guardian_role(hearth_hash: ActionHash) -> ExternResult<MemberRole> {
    let caller_role: Option<MemberRole> = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("get_caller_role"),
            None,
            hearth_hash,
        )?,
        "get_caller_role",
    )?;

    let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
        "You are not an active member of this hearth".into()
    )))?;

    if !role.is_guardian() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians can perform this action".into()
        )));
    }

    Ok(role)
}

/// Fetch the caller's role in the given hearth via cross-zome call to kinship.
/// Returns the role if the caller is an active member, Err otherwise.
fn get_caller_role(hearth_hash: ActionHash) -> ExternResult<MemberRole> {
    let caller_role: Option<MemberRole> = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("get_caller_role"),
            None,
            hearth_hash,
        )?,
        "get_caller_role",
    )?;

    caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
        "You are not an active member of this hearth".into()
    )))
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Register a new shared resource for a hearth.
/// Links the resource from the hearth via HearthToResources.
#[hdk_extern]
pub fn register_resource(input: RegisterResourceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "register_resource")?;
    get_caller_role(input.hearth_hash.clone())?;
    let resource = SharedResource {
        hearth_hash: input.hearth_hash.clone(),
        name: input.name,
        description: input.description,
        resource_type: input.resource_type,
        current_holder: None,
        condition: input.condition,
        location: input.location,
    };

    let resource_hash = create_entry(&EntryTypes::SharedResource(resource))?;

    create_link(
        input.hearth_hash,
        resource_hash.clone(),
        LinkTypes::HearthToResources,
        (),
    )?;

    let record = get(resource_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created SharedResource".into())
    ))?;

    Ok(record)
}

/// Lend a resource to a member. Creates a ResourceLoan with Active status.
/// Links the loan from the resource and from the borrower agent.
/// Auth: only guardians (Founder/Elder/Adult) can lend shared resources.
#[hdk_extern]
pub fn lend_resource(input: LendResourceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "lend_resource")?;
    let now = sys_time()?;

    // Get the resource to find its hearth
    let resource_record = get(input.resource_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Resource not found".into())),
    )?;
    let resource: SharedResource = resource_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize resource: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Resource entry is missing".into()
        )))?;

    // Guardian authorization
    let _role = require_guardian_role(resource.hearth_hash.clone())?;

    let loan = ResourceLoan {
        resource_hash: input.resource_hash.clone(),
        lender_hearth: resource.hearth_hash,
        borrower: input.borrower.clone(),
        due_date: input.due_date,
        status: LoanStatus::Active,
        created_at: now,
    };

    let loan_hash = create_entry(&EntryTypes::ResourceLoan(loan))?;

    create_link(
        input.resource_hash.clone(),
        loan_hash.clone(),
        LinkTypes::ResourceToLoans,
        (),
    )?;

    create_link(
        input.borrower.clone(),
        loan_hash.clone(),
        LinkTypes::AgentToLoans,
        (),
    )?;

    emit_signal(&HearthSignal::ResourceLent {
        resource_hash: input.resource_hash,
        borrower: input.borrower,
        due_date: input.due_date,
    })?;

    let record = get(loan_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created ResourceLoan".into())
    ))?;

    Ok(record)
}

/// Mark a resource loan as returned.
/// Auth: the borrower OR any guardian of the hearth can return a resource.
/// The loan must be in an active state (Active or Overdue).
#[hdk_extern]
pub fn return_resource(loan_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "return_resource")?;
    let agent = agent_info()?.agent_initial_pubkey;

    let existing = get(loan_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))?;
    let mut loan: ResourceLoan = existing
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize loan: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Loan entry is missing".into()
        )))?;

    // 1. Loan must be active (Active or Overdue)
    if !is_loan_active(&loan.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot return a loan with status {:?} (must be Active or Overdue)",
            loan.status
        ))));
    }

    // 2. Auth: borrower OR guardian
    if agent != loan.borrower {
        let role = get_caller_role(loan.lender_hearth.clone())?;
        if !role.is_guardian() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the borrower or a guardian can return a resource".into()
            )));
        }
    }

    let borrower = loan.borrower.clone();
    loan.status = LoanStatus::Returned;

    let updated_hash = update_entry(loan_hash.clone(), &loan)?;

    emit_signal(&HearthSignal::ResourceReturned {
        loan_hash,
        borrower,
    })?;

    let record = get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated ResourceLoan".into())
    ))?;

    Ok(record)
}

/// Create a new budget category for a hearth.
/// Links the budget from the hearth via HearthToBudgets.
/// Auth: only guardians (Founder/Elder/Adult) can create budget categories.
#[hdk_extern]
pub fn create_budget_category(input: CreateBudgetInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "create_budget_category")?;
    // Guardian authorization
    let _role = require_guardian_role(input.hearth_hash.clone())?;

    let budget = BudgetCategory {
        hearth_hash: input.hearth_hash.clone(),
        category: input.category,
        monthly_target_cents: input.monthly_target_cents,
        current_month_actual_cents: 0,
    };

    let budget_hash = create_entry(&EntryTypes::BudgetCategory(budget))?;

    create_link(
        input.hearth_hash,
        budget_hash.clone(),
        LinkTypes::HearthToBudgets,
        (),
    )?;

    let record = get(budget_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created BudgetCategory".into())
    ))?;

    Ok(record)
}

/// Log an expense against a budget category, incrementing current_month_actual_cents.
#[hdk_extern]
pub fn log_expense(input: LogExpenseInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "log_expense")?;
    let existing = get(input.budget_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Budget category not found".into())
    ))?;
    let mut budget: BudgetCategory = existing
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize budget: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Budget entry is missing".into()
        )))?;

    get_caller_role(budget.hearth_hash.clone())?;

    budget.current_month_actual_cents = budget
        .current_month_actual_cents
        .saturating_add(input.amount_cents);

    let updated_hash = update_entry(input.budget_hash.clone(), &budget)?;

    emit_signal(&HearthSignal::ExpenseLogged {
        budget_hash: input.budget_hash,
        amount_cents: input.amount_cents,
    })?;

    let record = get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated BudgetCategory".into())
    ))?;

    Ok(record)
}

/// Get all shared resources (inventory) for a hearth.
#[hdk_extern]
pub fn get_hearth_inventory(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToResources)?,
        GetStrategy::default(),
    )?;

    let mut resources = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get_latest_record(target)? {
            resources.push(record);
        }
    }

    Ok(resources)
}

/// Get all budget categories for a hearth.
#[hdk_extern]
pub fn get_budget_summary(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToBudgets)?,
        GetStrategy::default(),
    )?;

    let mut budgets = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get_latest_record(target)? {
            budgets.push(record);
        }
    }

    Ok(budgets)
}

/// Get all loans for a specific resource.
#[hdk_extern]
pub fn get_resource_loans(resource_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(resource_hash, LinkTypes::ResourceToLoans)?,
        GetStrategy::default(),
    )?;

    let mut loans = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get_latest_record(target)? {
            loans.push(record);
        }
    }

    Ok(loans)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Input Type Serde ----

    #[test]
    fn register_resource_input_serde_roundtrip() {
        let input = RegisterResourceInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            name: "Drill".into(),
            description: "Power drill".into(),
            resource_type: ResourceType::Tool,
            condition: "Good".into(),
            location: "Garage shelf".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RegisterResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Drill");
        assert_eq!(back.location, "Garage shelf");
    }

    #[test]
    fn lend_resource_input_serde_roundtrip() {
        let input = LendResourceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            borrower: AgentPubKey::from_raw_36(vec![0xBBu8; 36]),
            due_date: Timestamp::from_micros(2_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: LendResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.due_date, Timestamp::from_micros(2_000_000));
    }

    #[test]
    fn create_budget_input_serde_roundtrip() {
        let input = CreateBudgetInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            category: "Groceries".into(),
            monthly_target_cents: 50000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBudgetInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.category, "Groceries");
        assert_eq!(back.monthly_target_cents, 50000);
    }

    #[test]
    fn log_expense_input_serde_roundtrip() {
        let input = LogExpenseInput {
            budget_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            amount_cents: 1299,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: LogExpenseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.amount_cents, 1299);
    }

    #[test]
    fn register_resource_input_all_types() {
        let types = vec![
            ResourceType::Tool,
            ResourceType::Vehicle,
            ResourceType::Book,
            ResourceType::Kitchen,
            ResourceType::Electronics,
            ResourceType::Clothing,
            ResourceType::Custom("Garden".into()),
        ];
        for rt in types {
            let input = RegisterResourceInput {
                hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                name: "Item".into(),
                description: "desc".into(),
                resource_type: rt,
                condition: "".into(),
                location: "".into(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let _back: RegisterResourceInput = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn log_expense_zero_amount() {
        let input = LogExpenseInput {
            budget_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            amount_cents: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: LogExpenseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.amount_cents, 0);
    }

    // ========================================================================
    // Pure helper: is_loan_active
    // ========================================================================

    #[test]
    fn loan_active_is_active() {
        assert!(is_loan_active(&LoanStatus::Active));
    }

    #[test]
    fn loan_returned_is_not_active() {
        assert!(!is_loan_active(&LoanStatus::Returned));
    }

    #[test]
    fn loan_overdue_is_active() {
        // Overdue loans are still "active" — the resource hasn't been returned
        assert!(is_loan_active(&LoanStatus::Overdue));
    }

    // ========================================================================
    // Pure helper: is_budget_over_target
    // ========================================================================

    #[test]
    fn budget_under_target() {
        assert!(!is_budget_over_target(3000, 5000));
    }

    #[test]
    fn budget_exact_target() {
        assert!(!is_budget_over_target(5000, 5000));
    }

    #[test]
    fn budget_over_target() {
        assert!(is_budget_over_target(5001, 5000));
    }

    #[test]
    fn budget_zero_actual_zero_target() {
        assert!(!is_budget_over_target(0, 0));
    }

    #[test]
    fn budget_nonzero_actual_zero_target() {
        // Any spending with a zero target is over budget
        assert!(is_budget_over_target(1, 0));
    }

    // ========================================================================
    // Pure helper: budget_usage_pct
    // ========================================================================

    #[test]
    fn usage_pct_zero_zero() {
        // 0 of 0 target = 0bp (no budget, no spending)
        assert_eq!(budget_usage_pct(0, 0), 0);
    }

    #[test]
    fn usage_pct_zero_actual() {
        // 0 of 5000 target = 0bp
        assert_eq!(budget_usage_pct(0, 5000), 0);
    }

    #[test]
    fn usage_pct_half() {
        // 2500 of 5000 target = 5000bp (50%)
        assert_eq!(budget_usage_pct(2500, 5000), 5000);
    }

    #[test]
    fn usage_pct_exact() {
        // 5000 of 5000 target = 10000bp (100%)
        assert_eq!(budget_usage_pct(5000, 5000), 10000);
    }

    #[test]
    fn usage_pct_over_target_caps_at_10000() {
        // 7000 of 5000 target = capped at 10000bp
        assert_eq!(budget_usage_pct(7000, 5000), 10000);
    }

    #[test]
    fn usage_pct_way_over_caps_at_10000() {
        // u64::MAX of 1 = capped at 10000bp
        assert_eq!(budget_usage_pct(u64::MAX, 1), 10000);
    }

    #[test]
    fn usage_pct_nonzero_actual_zero_target() {
        // Any spending with 0 target = 10000bp (fully over)
        assert_eq!(budget_usage_pct(100, 0), 10000);
    }

    #[test]
    fn usage_pct_one_third() {
        // 1 of 3 target = 3333bp (33.33%)
        assert_eq!(budget_usage_pct(1, 3), 3333);
    }

    // ========================================================================
    // Scenario: lend -> return lifecycle status transitions
    // ========================================================================

    #[test]
    fn scenario_lend_return_lifecycle() {
        // A new loan starts Active
        let status = LoanStatus::Active;
        assert!(is_loan_active(&status));

        // Loan can transition to Overdue (still active, resource not returned)
        let overdue = LoanStatus::Overdue;
        assert!(is_loan_active(&overdue));

        // Loan is returned — no longer active
        let returned = LoanStatus::Returned;
        assert!(!is_loan_active(&returned));
    }

    #[test]
    fn scenario_return_only_active_loans() {
        // Can return an Active loan
        assert!(is_loan_active(&LoanStatus::Active));

        // Can return an Overdue loan (still outstanding)
        assert!(is_loan_active(&LoanStatus::Overdue));

        // Cannot "return" an already-Returned loan
        assert!(!is_loan_active(&LoanStatus::Returned));
    }

    #[test]
    fn scenario_overdue_transitions() {
        // Overdue is still active — the resource is out
        let overdue = LoanStatus::Overdue;
        assert!(is_loan_active(&overdue));

        // Once returned, it's no longer active
        let returned = LoanStatus::Returned;
        assert!(!is_loan_active(&returned));
    }

    // ========================================================================
    // Scenario: expense accumulation with saturation arithmetic
    // ========================================================================

    #[test]
    fn scenario_expense_accumulation_basic() {
        let mut actual: u64 = 0;
        let target: u64 = 50000;

        // Log 3 expenses: 15000 + 20000 + 10000 = 45000
        actual = actual.saturating_add(15000);
        actual = actual.saturating_add(20000);
        actual = actual.saturating_add(10000);
        assert_eq!(actual, 45000);

        // Under budget
        assert!(!is_budget_over_target(actual, target));
        assert_eq!(budget_usage_pct(actual, target), 9000); // 90%
    }

    #[test]
    fn scenario_expense_exceeds_target() {
        let mut actual: u64 = 0;
        let target: u64 = 50000;

        actual = actual.saturating_add(30000);
        actual = actual.saturating_add(25000);
        assert_eq!(actual, 55000);

        // Over budget
        assert!(is_budget_over_target(actual, target));
        assert_eq!(budget_usage_pct(actual, target), 10000); // Capped at 100%
    }

    #[test]
    fn scenario_expense_saturation_overflow() {
        let mut actual: u64 = u64::MAX - 10;
        let target: u64 = 50000;

        // This would overflow without saturating_add
        actual = actual.saturating_add(100);
        assert_eq!(actual, u64::MAX);

        // Over budget, capped usage
        assert!(is_budget_over_target(actual, target));
        assert_eq!(budget_usage_pct(actual, target), 10000);
    }

    #[test]
    fn scenario_expense_zero_target_any_spending() {
        let mut actual: u64 = 0;
        let target: u64 = 0;

        // No spending = not over
        assert!(!is_budget_over_target(actual, target));
        assert_eq!(budget_usage_pct(actual, target), 0);

        // Any spending = over
        actual = actual.saturating_add(1);
        assert!(is_budget_over_target(actual, target));
        assert_eq!(budget_usage_pct(actual, target), 10000);
    }

    #[test]
    fn scenario_expense_progressive_usage() {
        let target: u64 = 10000;

        // Track progressive budget usage
        assert_eq!(budget_usage_pct(0, target), 0);
        assert_eq!(budget_usage_pct(2500, target), 2500);
        assert_eq!(budget_usage_pct(5000, target), 5000);
        assert_eq!(budget_usage_pct(7500, target), 7500);
        assert_eq!(budget_usage_pct(10000, target), 10000);
        assert_eq!(budget_usage_pct(15000, target), 10000); // capped
    }
}
