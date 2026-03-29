// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Work Orders Coordinator Zome
//!
//! CRUD operations and status management for manufacturing work orders.

use hdk::prelude::*;
use manufacturing_common::{WorkOrderPriority, WorkOrderStatus};
use workorders_integrity::*;

// ============================================================================
// Input types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateWorkOrderInput {
    pub product_id: String,
    pub quantity: u64,
    pub due_date: Timestamp,
    pub priority: WorkOrderPriority,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateWorkOrderStatusInput {
    pub work_order_hash: ActionHash,
    pub new_status: WorkOrderStatus,
    pub reason: Option<String>,
}

// ============================================================================
// Extern functions
// ============================================================================

/// Create a new work order in Draft status.
#[hdk_extern]
pub fn create_work_order(input: CreateWorkOrderInput) -> ExternResult<ActionHash> {
    if input.product_id.is_empty() || input.product_id.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "product_id must be 1-200 characters".to_string()
        )));
    }
    if input.quantity == 0 || input.quantity > 1_000_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "quantity must be 1-1,000,000".to_string()
        )));
    }

    let now = sys_time()?;
    let entry = WorkOrderEntry {
        product_id: input.product_id.clone(),
        quantity: input.quantity,
        due_date: input.due_date,
        status: WorkOrderStatus::Draft,
        priority: input.priority,
        notes: input.notes,
        bom_hash: None,
        routing_hash: None,
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(EntryTypes::WorkOrder(entry))?;

    // Link from "all_work_orders" anchor
    let all_path = Path::from("all_work_orders").typed(LinkTypes::AllWorkOrders)?;
    all_path.ensure()?;
    create_link(
        all_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllWorkOrders,
        (),
    )?;

    // Link from product_id anchor
    let product_path =
        Path::from(format!("product/{}", input.product_id)).typed(LinkTypes::ProductToWorkOrders)?;
    product_path.ensure()?;
    create_link(
        product_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ProductToWorkOrders,
        (),
    )?;

    Ok(action_hash)
}

/// Retrieve a work order by its action hash.
#[hdk_extern]
pub fn get_work_order(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Transition a work order to a new status, with validation.
#[hdk_extern]
pub fn update_work_order_status(input: UpdateWorkOrderStatusInput) -> ExternResult<ActionHash> {
    let record = get(input.work_order_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest(
            "Work order not found".to_string()
        )),
    )?;

    let wo: WorkOrderEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize work order".to_string()
        )))?;

    if !wo.status.can_transition_to(&input.new_status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid transition: {:?} -> {:?}. Valid targets: {:?}",
            wo.status,
            input.new_status,
            wo.status.valid_transitions()
        ))));
    }

    let now = sys_time()?;

    // Record the status update event
    let status_update = WorkOrderStatusUpdate {
        work_order_hash: input.work_order_hash.clone(),
        previous_status: wo.status.clone(),
        new_status: input.new_status.clone(),
        updated_at: now,
        reason: input.reason,
    };
    let su_hash = create_entry(EntryTypes::StatusUpdate(status_update))?;
    create_link(
        input.work_order_hash.clone(),
        su_hash,
        LinkTypes::WorkOrderToStatusUpdates,
        (),
    )?;

    // Update the work order entry itself
    let updated = WorkOrderEntry {
        status: input.new_status,
        updated_at: now,
        ..wo
    };
    update_entry(input.work_order_hash, EntryTypes::WorkOrder(updated))
}

/// List all work orders (returns action hashes via anchor links).
#[hdk_extern]
pub fn list_work_orders(_: ()) -> ExternResult<Vec<Link>> {
    let all_path = Path::from("all_work_orders").typed(LinkTypes::AllWorkOrders)?;
    get_links(
        GetLinksInputBuilder::try_new(all_path.path_entry_hash()?, LinkTypes::AllWorkOrders)?
            .build(),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_input_serde_roundtrip() {
        let input = CreateWorkOrderInput {
            product_id: "WIDGET-001".to_string(),
            quantity: 100,
            due_date: Timestamp::from_micros(1_700_000_000_000_000),
            priority: WorkOrderPriority::Normal,
            notes: Some("First run".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateWorkOrderInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.product_id, "WIDGET-001");
        assert_eq!(back.quantity, 100);
    }

    #[test]
    fn test_status_update_serde_roundtrip() {
        let input = UpdateWorkOrderStatusInput {
            work_order_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_status: WorkOrderStatus::Released,
            reason: Some("Ready for production".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateWorkOrderStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.new_status, WorkOrderStatus::Released);
    }

    #[test]
    fn test_work_order_entry_serde() {
        let entry = WorkOrderEntry {
            product_id: "TEST".to_string(),
            quantity: 50,
            due_date: Timestamp::from_micros(0),
            status: WorkOrderStatus::Draft,
            priority: WorkOrderPriority::High,
            notes: None,
            bom_hash: None,
            routing_hash: None,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"status\":\"Draft\""));
    }

    #[test]
    fn test_priority_variants() {
        for p in [
            WorkOrderPriority::Low,
            WorkOrderPriority::Normal,
            WorkOrderPriority::High,
            WorkOrderPriority::Urgent,
        ] {
            let json = serde_json::to_string(&p).unwrap();
            let back: WorkOrderPriority = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p);
        }
    }
}
