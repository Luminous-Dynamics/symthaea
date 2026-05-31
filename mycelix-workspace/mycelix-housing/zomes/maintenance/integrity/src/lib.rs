// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Maintenance Integrity Zome
//! Entry types and validation for maintenance requests, work orders, and inspections.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Category of maintenance issue
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaintenanceCategory {
    Plumbing,
    Electrical,
    HVAC,
    Structural,
    Appliance,
    Exterior,
    CommonArea,
    Safety,
    Pest,
    Other(String),
}

/// Priority of a maintenance request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaintenancePriority {
    Emergency,
    Urgent,
    Normal,
    Low,
    Scheduled,
}

/// Status of a maintenance request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaintenanceStatus {
    Reported,
    Acknowledged,
    Scheduled,
    InProgress,
    Completed,
    Deferred,
}

/// A maintenance request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MaintenanceRequest {
    pub unit_hash: Option<ActionHash>,
    pub building_hash: ActionHash,
    pub reported_by: AgentPubKey,
    pub title: String,
    pub description: String,
    pub category: MaintenanceCategory,
    pub priority: MaintenancePriority,
    pub status: MaintenanceStatus,
    pub reported_at: Timestamp,
}

/// A work order for maintenance
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WorkOrder {
    pub request_hash: ActionHash,
    pub assigned_to: String,
    pub description: String,
    pub estimated_cost_cents: Option<u64>,
    pub actual_cost_cents: Option<u64>,
    pub scheduled_date: Option<Timestamp>,
    pub completed_date: Option<Timestamp>,
    pub notes: String,
}

/// Type of building inspection
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InspectionType {
    Annual,
    Safety,
    Code,
    PreMove,
    PostMove,
}

/// A building inspection record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Inspection {
    pub building_hash: ActionHash,
    pub inspector: AgentPubKey,
    pub inspection_type: InspectionType,
    pub date: Timestamp,
    pub findings: Vec<String>,
    pub passed: bool,
    pub next_due: Option<Timestamp>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    MaintenanceRequest(MaintenanceRequest),
    WorkOrder(WorkOrder),
    Inspection(Inspection),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Open requests anchor
    OpenRequests,
    /// Building to maintenance requests
    BuildingToRequest,
    /// Request to work orders
    RequestToWorkOrder,
    /// Building to inspections
    BuildingToInspection,
    /// Reporter to their requests
    ReporterToRequest,
    /// All completed requests
    CompletedRequests,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MaintenanceRequest(req) => validate_create_request(action, req),
                EntryTypes::WorkOrder(order) => validate_create_work_order(action, order),
                EntryTypes::Inspection(inspection) => {
                    validate_create_inspection(action, inspection)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MaintenanceRequest(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WorkOrder(order) => validate_update_work_order(order),
                EntryTypes::Inspection(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::OpenRequests => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BuildingToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequestToWorkOrder => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BuildingToInspection => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ReporterToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CompletedRequests => Ok(ValidateCallbackResult::Valid),
        },
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

fn validate_create_request(
    _action: Create,
    req: MaintenanceRequest,
) -> ExternResult<ValidateCallbackResult> {
    if req.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title cannot be empty".into(),
        ));
    }
    if req.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title must be at most 256 characters".into(),
        ));
    }
    if req.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request description cannot be empty".into(),
        ));
    }
    if req.status != MaintenanceStatus::Reported {
        return Ok(ValidateCallbackResult::Invalid(
            "New requests must have Reported status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_work_order(
    _action: Create,
    order: WorkOrder,
) -> ExternResult<ValidateCallbackResult> {
    if order.assigned_to.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order must be assigned to someone".into(),
        ));
    }
    if order.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order description cannot be empty".into(),
        ));
    }
    if order.completed_date.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "New work orders cannot have a completed date".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_work_order(order: WorkOrder) -> ExternResult<ValidateCallbackResult> {
    if order.assigned_to.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order must be assigned to someone".into(),
        ));
    }
    if let (Some(actual), Some(estimated)) = (order.actual_cost_cents, order.estimated_cost_cents) {
        // Allow up to 200% of estimate without special approval
        if actual > estimated * 2 {
            return Ok(ValidateCallbackResult::Invalid(
                "Actual cost exceeds 200% of estimate; requires special approval".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_inspection(
    _action: Create,
    inspection: Inspection,
) -> ExternResult<ValidateCallbackResult> {
    if inspection.findings.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum 100 findings per inspection".into(),
        ));
    }
    for finding in &inspection.findings {
        if finding.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Inspection findings cannot be empty strings".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
