// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Maintenance Coordinator Zome
//! Business logic for maintenance requests, work orders, and inspections.

use hdk::prelude::*;
use maintenance_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Submit a new maintenance request
#[hdk_extern]
pub fn submit_request(req: MaintenanceRequest) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::MaintenanceRequest(req.clone()))?;

    // Link to open requests
    create_entry(&EntryTypes::Anchor(Anchor("open_requests".to_string())))?;
    create_link(
        anchor_hash("open_requests")?,
        action_hash.clone(),
        LinkTypes::OpenRequests,
        (),
    )?;

    // Link building to request
    create_link(
        req.building_hash,
        action_hash.clone(),
        LinkTypes::BuildingToRequest,
        (),
    )?;

    // Link reporter to request
    create_link(
        req.reported_by,
        action_hash.clone(),
        LinkTypes::ReporterToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcknowledgeRequestInput {
    pub request_hash: ActionHash,
}

/// Acknowledge a maintenance request
#[hdk_extern]
pub fn acknowledge_request(input: AcknowledgeRequestInput) -> ExternResult<Record> {
    let record = get(input.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Request not found".into())
    ))?;

    let mut req: MaintenanceRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    if req.status != MaintenanceStatus::Reported {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request must be in Reported status to acknowledge".into()
        )));
    }

    req.status = MaintenanceStatus::Acknowledged;

    let new_hash = update_entry(input.request_hash, &EntryTypes::MaintenanceRequest(req))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated request".into()
    )))
}

/// Create a work order for a maintenance request
#[hdk_extern]
pub fn create_work_order(order: WorkOrder) -> ExternResult<Record> {
    if order.assigned_to.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Assigned-to must be at most 256 characters".into()
        )));
    }

    // Update the request status to Scheduled
    let req_record = get(order.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Request not found".into())
    ))?;

    let mut req: MaintenanceRequest = req_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    req.status = MaintenanceStatus::Scheduled;
    update_entry(
        order.request_hash.clone(),
        &EntryTypes::MaintenanceRequest(req),
    )?;

    let action_hash = create_entry(&EntryTypes::WorkOrder(order.clone()))?;

    // Link request to work order
    create_link(
        order.request_hash,
        action_hash.clone(),
        LinkTypes::RequestToWorkOrder,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created work order".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteWorkOrderInput {
    pub work_order_hash: ActionHash,
    pub actual_cost_cents: Option<u64>,
    pub notes: String,
}

/// Complete a work order and mark the request as completed
#[hdk_extern]
pub fn complete_work_order(input: CompleteWorkOrderInput) -> ExternResult<Record> {
    let record = get(input.work_order_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Work order not found".into())
    ))?;

    let mut order: WorkOrder = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid work order entry".into()
        )))?;

    let now = sys_time()?;
    order.completed_date = Some(now);
    order.actual_cost_cents = input.actual_cost_cents;
    if !input.notes.is_empty() {
        order.notes = input.notes;
    }

    let new_hash = update_entry(input.work_order_hash, &EntryTypes::WorkOrder(order.clone()))?;

    // Update the maintenance request status to Completed
    let req_record = get(order.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Original request not found".into())
    ))?;

    let mut req: MaintenanceRequest = req_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    req.status = MaintenanceStatus::Completed;
    update_entry(
        order.request_hash.clone(),
        &EntryTypes::MaintenanceRequest(req),
    )?;

    // Move from open to completed
    let open_links = get_links(
        LinkQuery::try_new(anchor_hash("open_requests")?, LinkTypes::OpenRequests)?,
        GetStrategy::default(),
    )?;
    for link in open_links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == order.request_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    create_entry(&EntryTypes::Anchor(Anchor(
        "completed_requests".to_string(),
    )))?;
    create_link(
        anchor_hash("completed_requests")?,
        order.request_hash,
        LinkTypes::CompletedRequests,
        (),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated work order".into()
    )))
}

/// Schedule a building inspection
#[hdk_extern]
pub fn schedule_inspection(inspection: Inspection) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Inspection(inspection.clone()))?;

    create_link(
        inspection.building_hash,
        action_hash.clone(),
        LinkTypes::BuildingToInspection,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created inspection".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordInspectionInput {
    pub inspection_hash: ActionHash,
    pub findings: Vec<String>,
    pub passed: bool,
    pub next_due: Option<Timestamp>,
}

/// Record the results of an inspection
#[hdk_extern]
pub fn record_inspection(input: RecordInspectionInput) -> ExternResult<Record> {
    let record = get(input.inspection_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Inspection not found".into())
    ))?;

    let mut inspection: Inspection = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid inspection entry".into()
        )))?;

    inspection.findings = input.findings;
    inspection.passed = input.passed;
    inspection.next_due = input.next_due;

    let new_hash = update_entry(input.inspection_hash, &EntryTypes::Inspection(inspection))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated inspection".into()
    )))
}

/// Get all open maintenance requests
#[hdk_extern]
pub fn get_open_requests(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("open_requests")?, LinkTypes::OpenRequests)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            requests.push(record);
        }
    }

    // Sort by priority (Emergency first)
    requests.sort_by(|a, b| {
        let priority_ord = |r: &Record| -> u8 {
            r.entry()
                .to_app_option::<MaintenanceRequest>()
                .ok()
                .flatten()
                .map(|req| match req.priority {
                    MaintenancePriority::Emergency => 0,
                    MaintenancePriority::Urgent => 1,
                    MaintenancePriority::Normal => 2,
                    MaintenancePriority::Low => 3,
                    MaintenancePriority::Scheduled => 4,
                })
                .unwrap_or(255)
        };
        priority_ord(a).cmp(&priority_ord(b))
    });

    Ok(requests)
}

/// Get maintenance history for a building
#[hdk_extern]
pub fn get_building_maintenance_history(building_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(building_hash, LinkTypes::BuildingToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            requests.push(record);
        }
    }

    // Sort by reported time
    requests.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(requests)
}
