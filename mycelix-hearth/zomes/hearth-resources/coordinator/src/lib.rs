//! Hearth Resources Coordinator Zome
//!
//! Provides CRUD operations for shared resources, resource loans,
//! and budget tracking.

use hdk::prelude::*;
use hearth_resources_integrity::*;
use hearth_types::*;

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
// Extern Functions
// ============================================================================

/// Register a new shared resource for a hearth.
/// Links the resource from the hearth via HearthToResources.
#[hdk_extern]
pub fn register_resource(input: RegisterResourceInput) -> ExternResult<Record> {
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
#[hdk_extern]
pub fn lend_resource(input: LendResourceInput) -> ExternResult<Record> {
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
        input.resource_hash,
        loan_hash.clone(),
        LinkTypes::ResourceToLoans,
        (),
    )?;

    create_link(
        input.borrower,
        loan_hash.clone(),
        LinkTypes::AgentToLoans,
        (),
    )?;

    let record = get(loan_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created ResourceLoan".into())
    ))?;

    Ok(record)
}

/// Mark a resource loan as returned.
#[hdk_extern]
pub fn return_resource(loan_hash: ActionHash) -> ExternResult<Record> {
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

    loan.status = LoanStatus::Returned;

    let updated_hash = update_entry(loan_hash, &loan)?;

    let record = get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated ResourceLoan".into())
    ))?;

    Ok(record)
}

/// Create a new budget category for a hearth.
/// Links the budget from the hearth via HearthToBudgets.
#[hdk_extern]
pub fn create_budget_category(input: CreateBudgetInput) -> ExternResult<Record> {
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

    budget.current_month_actual_cents = budget
        .current_month_actual_cents
        .saturating_add(input.amount_cents);

    let updated_hash = update_entry(input.budget_hash, &budget)?;

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

        if let Some(record) = get(target, GetOptions::default())? {
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

        if let Some(record) = get(target, GetOptions::default())? {
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

        if let Some(record) = get(target, GetOptions::default())? {
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
}
