// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Resources Integrity Zome
//!
//! Defines entry types and validation for shared resources, resource loans,
//! and budget categories within a hearth.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A shared resource belonging to a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SharedResource {
    /// The hearth this resource belongs to.
    pub hearth_hash: ActionHash,
    /// Name of the resource.
    pub name: String,
    /// Description of the resource.
    pub description: String,
    /// Category of resource.
    pub resource_type: ResourceType,
    /// Agent currently holding the resource (None if in common storage).
    pub current_holder: Option<AgentPubKey>,
    /// Current condition description (e.g. "good", "needs repair").
    pub condition: String,
    /// Where the resource is stored or located.
    pub location: String,
}

/// A loan of a resource to a hearth member.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResourceLoan {
    /// Hash of the resource being loaned.
    pub resource_hash: ActionHash,
    /// The hearth the resource belongs to.
    pub lender_hearth: ActionHash,
    /// The agent borrowing the resource.
    pub borrower: AgentPubKey,
    /// When the resource is due back.
    pub due_date: Timestamp,
    /// Current loan status.
    pub status: LoanStatus,
    /// When the loan was created.
    pub created_at: Timestamp,
}

/// A budget category for tracking hearth expenses.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BudgetCategory {
    /// The hearth this budget belongs to.
    pub hearth_hash: ActionHash,
    /// Name of the budget category.
    pub category: String,
    /// Monthly spending target in cents.
    pub monthly_target_cents: u64,
    /// Actual spending this month in cents.
    pub current_month_actual_cents: u64,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    SharedResource(SharedResource),
    ResourceLoan(ResourceLoan),
    BudgetCategory(BudgetCategory),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> SharedResource
    HearthToResources,
    /// SharedResource -> ResourceLoan
    ResourceToLoans,
    /// Hearth -> BudgetCategory
    HearthToBudgets,
    /// AgentPubKey -> ResourceLoan
    AgentToLoans,
}

// ============================================================================
// Genesis + Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::SharedResource(resource) => validate_resource(&resource),
            EntryTypes::ResourceLoan(loan) => validate_loan_create(&loan),
            EntryTypes::BudgetCategory(budget) => validate_budget(&budget),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry {
            app_entry,
            original_action_hash,
            ..
        }) => match app_entry {
            EntryTypes::SharedResource(resource) => {
                validate_resource(&resource)?;
                validate_resource_immutable_fields(&resource, &original_action_hash)
            }
            EntryTypes::ResourceLoan(loan) => {
                validate_loan_fields(&loan)?;
                validate_loan_immutable_fields(&loan, &original_action_hash)
            }
            EntryTypes::BudgetCategory(budget) => {
                validate_budget(&budget)?;
                validate_budget_immutable_fields(&budget, &original_action_hash)
            }
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink { tag, action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_resource(resource: &SharedResource) -> ExternResult<ValidateCallbackResult> {
    if resource.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".into(),
        ));
    }
    if resource.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name must be <= 256 characters".into(),
        ));
    }
    if resource.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource description must be <= 4096 characters".into(),
        ));
    }
    if resource.condition.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource condition must be <= 1024 characters".into(),
        ));
    }
    if resource.location.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource location must be <= 1024 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_loan_fields(loan: &ResourceLoan) -> ExternResult<ValidateCallbackResult> {
    // Structural validation shared by create and update
    match loan.status {
        LoanStatus::Active | LoanStatus::Returned | LoanStatus::Overdue => {}
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_loan_create(loan: &ResourceLoan) -> ExternResult<ValidateCallbackResult> {
    validate_loan_fields(loan)?;
    match loan.status {
        LoanStatus::Active => {}
        LoanStatus::Returned | LoanStatus::Overdue => {
            return Ok(ValidateCallbackResult::Invalid(
                "ResourceLoan cannot be created with terminal status".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a SharedResource update.
pub fn validate_resource_immutable_fields(
    new_resource: &SharedResource,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_resource: SharedResource = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original SharedResource: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original SharedResource entry is missing".into()
        )))?;

    if new_resource.hearth_hash != original_resource.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a SharedResource".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a ResourceLoan update.
pub fn validate_loan_immutable_fields(
    new_loan: &ResourceLoan,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_loan: ResourceLoan = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original ResourceLoan: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original ResourceLoan entry is missing".into()
        )))?;

    if new_loan.resource_hash != original_loan.resource_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change resource_hash on a ResourceLoan".into(),
        ));
    }
    if new_loan.lender_hearth != original_loan.lender_hearth {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change lender_hearth on a ResourceLoan".into(),
        ));
    }
    if new_loan.borrower != original_loan.borrower {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change borrower on a ResourceLoan".into(),
        ));
    }
    if new_loan.created_at != original_loan.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on a ResourceLoan".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a BudgetCategory update.
pub fn validate_budget_immutable_fields(
    new_budget: &BudgetCategory,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_budget: BudgetCategory = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original BudgetCategory: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original BudgetCategory entry is missing".into()
        )))?;

    if new_budget.hearth_hash != original_budget.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a BudgetCategory".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_budget(budget: &BudgetCategory) -> ExternResult<ValidateCallbackResult> {
    if budget.category.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Budget category cannot be empty".into(),
        ));
    }
    if budget.category.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Budget category must be <= 256 characters".into(),
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

    // ---- Helper Constructors ----

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAAu8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xABu8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_resource(name: &str, desc: &str) -> SharedResource {
        SharedResource {
            hearth_hash: fake_action_hash(),
            name: name.into(),
            description: desc.into(),
            resource_type: ResourceType::Tool,
            current_holder: None,
            condition: "Good".into(),
            location: "Garage".into(),
        }
    }

    fn make_loan(status: LoanStatus) -> ResourceLoan {
        ResourceLoan {
            resource_hash: fake_action_hash(),
            lender_hearth: fake_action_hash(),
            borrower: fake_agent(),
            due_date: Timestamp::from_micros(2_000_000),
            status,
            created_at: fake_timestamp(),
        }
    }

    fn make_budget(category: &str, target: u64) -> BudgetCategory {
        BudgetCategory {
            hearth_hash: fake_action_hash(),
            category: category.into(),
            monthly_target_cents: target,
            current_month_actual_cents: 0,
        }
    }

    // ---- SharedResource Validation ----

    #[test]
    fn valid_resource_passes() {
        let r = make_resource("Hammer", "A claw hammer");
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_empty_name_rejected() {
        let r = make_resource("", "desc");
        match validate_resource(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn resource_name_exactly_256_passes() {
        let r = make_resource(&"n".repeat(256), "desc");
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_name_257_rejected() {
        let r = make_resource(&"n".repeat(257), "desc");
        match validate_resource(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn resource_description_exactly_4096_passes() {
        let r = make_resource("Tool", &"d".repeat(4096));
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_description_4097_rejected() {
        let r = make_resource("Tool", &"d".repeat(4097));
        match validate_resource(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn resource_empty_description_passes() {
        let r = make_resource("Tool", "");
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_condition_exactly_1024_passes() {
        let mut r = make_resource("Tool", "desc");
        r.condition = "c".repeat(1024);
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_condition_1025_rejected() {
        let mut r = make_resource("Tool", "desc");
        r.condition = "c".repeat(1025);
        match validate_resource(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("condition")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn resource_location_exactly_1024_passes() {
        let mut r = make_resource("Tool", "desc");
        r.location = "l".repeat(1024);
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_location_1025_rejected() {
        let mut r = make_resource("Tool", "desc");
        r.location = "l".repeat(1025);
        match validate_resource(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("location")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn resource_empty_condition_passes() {
        let mut r = make_resource("Tool", "desc");
        r.condition = "".into();
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn resource_empty_location_passes() {
        let mut r = make_resource("Tool", "desc");
        r.location = "".into();
        assert!(matches!(
            validate_resource(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- ResourceLoan Validation ----

    #[test]
    fn valid_loan_passes() {
        let l = make_loan(LoanStatus::Active);
        assert!(matches!(
            validate_loan_fields(&l).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn loan_all_statuses_valid_for_fields() {
        let statuses = vec![
            LoanStatus::Active,
            LoanStatus::Returned,
            LoanStatus::Overdue,
        ];
        for status in statuses {
            let l = make_loan(status);
            assert!(matches!(
                validate_loan_fields(&l).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    #[test]
    fn loan_create_active_status_passes() {
        let loan = make_loan(LoanStatus::Active);
        assert!(matches!(
            validate_loan_create(&loan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn loan_create_returned_status_rejected() {
        let loan = make_loan(LoanStatus::Returned);
        match validate_loan_create(&loan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("terminal status")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn loan_create_overdue_status_rejected() {
        let loan = make_loan(LoanStatus::Overdue);
        match validate_loan_create(&loan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("terminal status")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn loan_update_returned_status_passes() {
        let loan = make_loan(LoanStatus::Returned);
        assert!(matches!(
            validate_loan_fields(&loan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn loan_update_overdue_status_passes() {
        let loan = make_loan(LoanStatus::Overdue);
        assert!(matches!(
            validate_loan_fields(&loan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- BudgetCategory Validation ----

    #[test]
    fn valid_budget_passes() {
        let b = make_budget("Groceries", 50000);
        assert!(matches!(
            validate_budget(&b).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn budget_empty_category_rejected() {
        let b = make_budget("", 50000);
        match validate_budget(&b).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_category_exactly_256_passes() {
        let b = make_budget(&"c".repeat(256), 50000);
        assert!(matches!(
            validate_budget(&b).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn budget_category_257_rejected() {
        let b = make_budget(&"c".repeat(257), 50000);
        match validate_budget(&b).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Serde Roundtrips ----

    #[test]
    fn resource_serde_roundtrip() {
        let r = make_resource("Drill", "Power drill");
        let json = serde_json::to_string(&r).unwrap();
        let back: SharedResource = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn loan_serde_roundtrip() {
        let l = make_loan(LoanStatus::Active);
        let json = serde_json::to_string(&l).unwrap();
        let back: ResourceLoan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, l);
    }

    #[test]
    fn budget_serde_roundtrip() {
        let b = make_budget("Utilities", 15000);
        let json = serde_json::to_string(&b).unwrap();
        let back: BudgetCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn resource_with_holder_serde_roundtrip() {
        let mut r = make_resource("Drill", "Power drill");
        r.current_holder = Some(fake_agent());
        let json = serde_json::to_string(&r).unwrap();
        let back: SharedResource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.current_holder, r.current_holder);
    }

    // ---- Entry / Link Type Enums ----

    #[test]
    fn entry_types_all_variants_exist() {
        let _resource = UnitEntryTypes::SharedResource;
        let _loan = UnitEntryTypes::ResourceLoan;
        let _budget = UnitEntryTypes::BudgetCategory;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _resources = LinkTypes::HearthToResources;
        let _loans = LinkTypes::ResourceToLoans;
        let _budgets = LinkTypes::HearthToBudgets;
        let _agent_loans = LinkTypes::AgentToLoans;
    }

    // ---- All ResourceType variants ----

    #[test]
    fn resource_all_types_valid() {
        let types = vec![
            ResourceType::Tool,
            ResourceType::Vehicle,
            ResourceType::Book,
            ResourceType::Kitchen,
            ResourceType::Electronics,
            ResourceType::Clothing,
            ResourceType::Custom("Garden equipment".into()),
        ];
        for rt in types {
            let mut r = make_resource("Item", "desc");
            r.resource_type = rt;
            assert!(matches!(
                validate_resource(&r).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    // ---- Immutable field tests (pure equality, no conductor) ----

    #[test]
    fn resource_immutable_hearth_hash_difference_detected() {
        let r1 = make_resource("Tool", "desc");
        let mut r2 = r1.clone();
        r2.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(r1.hearth_hash, r2.hearth_hash);
    }

    // -- ResourceLoan immutable field tests --

    #[test]
    fn loan_immutable_resource_hash_difference_detected() {
        let l1 = make_loan(LoanStatus::Active);
        let mut l2 = l1.clone();
        l2.resource_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(l1.resource_hash, l2.resource_hash);
    }

    #[test]
    fn loan_immutable_lender_hearth_difference_detected() {
        let l1 = make_loan(LoanStatus::Active);
        let mut l2 = l1.clone();
        l2.lender_hearth = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(l1.lender_hearth, l2.lender_hearth);
    }

    #[test]
    fn loan_immutable_borrower_difference_detected() {
        let l1 = make_loan(LoanStatus::Active);
        let mut l2 = l1.clone();
        l2.borrower = AgentPubKey::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(l1.borrower, l2.borrower);
    }

    #[test]
    fn loan_immutable_created_at_difference_detected() {
        let l1 = make_loan(LoanStatus::Active);
        let mut l2 = l1.clone();
        l2.created_at = Timestamp::from_micros(9_999_999);
        assert_ne!(l1.created_at, l2.created_at);
    }

    // -- BudgetCategory immutable field tests --

    #[test]
    fn budget_immutable_hearth_hash_difference_detected() {
        let b1 = make_budget("Groceries", 50000);
        let mut b2 = b1.clone();
        b2.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(b1.hearth_hash, b2.hearth_hash);
    }
}
