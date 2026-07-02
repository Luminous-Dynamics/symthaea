// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Maintenance Integrity Zome
//! Entry types and validation for maintenance requests, work orders, and inspections.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
            tag,
            action: _,
        } => match link_type {
            LinkTypes::OpenRequests => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OpenRequests link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuildingToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuildingToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RequestToWorkOrder => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequestToWorkOrder link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuildingToInspection => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuildingToInspection link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ReporterToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ReporterToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CompletedRequests => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CompletedRequests link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
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
    }
}

fn validate_create_request(
    _action: Create,
    req: MaintenanceRequest,
) -> ExternResult<ValidateCallbackResult> {
    if req.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title cannot be empty".into(),
        ));
    }
    if req.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title must be at most 256 characters".into(),
        ));
    }
    if req.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request description cannot be empty".into(),
        ));
    }
    if req.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request description must be at most 4096 characters".into(),
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
    if order.assigned_to.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order must be assigned to someone".into(),
        ));
    }
    if order.assigned_to.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order assigned_to must be at most 256 characters".into(),
        ));
    }
    if order.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order description cannot be empty".into(),
        ));
    }
    if order.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order description must be at most 4096 characters".into(),
        ));
    }
    if order.notes.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order notes must be at most 8192 characters".into(),
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
    if order.assigned_to.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order must be assigned to someone".into(),
        ));
    }
    if order.assigned_to.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order assigned_to must be at most 256 characters".into(),
        ));
    }
    if order.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order description must be at most 4096 characters".into(),
        ));
    }
    if order.notes.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Work order notes must be at most 8192 characters".into(),
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
        if finding.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Inspection findings cannot be empty strings".into(),
            ));
        }
        if finding.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each inspection finding must be at most 4096 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Factory Functions ====================

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_32(vec![0u8; 32])
    }

    fn test_action_hash_2() -> ActionHash {
        ActionHash::from_raw_32(vec![1u8; 32])
    }

    fn test_agent_key() -> AgentPubKey {
        AgentPubKey::from_raw_32(vec![2u8; 32])
    }

    fn test_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn test_timestamp_2() -> Timestamp {
        Timestamp::from_micros(2_000_000)
    }

    fn test_create_action() -> Create {
        Create {
            author: test_agent_key(),
            timestamp: test_timestamp(),
            action_seq: 0,
            prev_action: test_action_hash(),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_32(vec![4u8; 32]),
            weight: Default::default(),
        }
    }

    fn valid_request() -> MaintenanceRequest {
        MaintenanceRequest {
            unit_hash: Some(test_action_hash()),
            building_hash: test_action_hash_2(),
            reported_by: test_agent_key(),
            title: "Leaking faucet".to_string(),
            description: "Kitchen sink is dripping constantly".to_string(),
            category: MaintenanceCategory::Plumbing,
            priority: MaintenancePriority::Normal,
            status: MaintenanceStatus::Reported,
            reported_at: test_timestamp(),
        }
    }

    fn valid_work_order() -> WorkOrder {
        WorkOrder {
            request_hash: test_action_hash(),
            assigned_to: "John Smith".to_string(),
            description: "Fix kitchen faucet leak".to_string(),
            estimated_cost_cents: Some(5000),
            actual_cost_cents: None,
            scheduled_date: Some(test_timestamp()),
            completed_date: None,
            notes: "Bring replacement parts".to_string(),
        }
    }

    fn valid_inspection() -> Inspection {
        Inspection {
            building_hash: test_action_hash(),
            inspector: test_agent_key(),
            inspection_type: InspectionType::Annual,
            date: test_timestamp(),
            findings: vec!["All smoke detectors operational".to_string()],
            passed: true,
            next_due: Some(test_timestamp_2()),
        }
    }

    // ==================== MaintenanceRequest Tests ====================

    #[test]
    fn test_valid_request_creation() {
        let req = valid_request();
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_empty_title_rejected() {
        let mut req = valid_request();
        req.title = "".to_string();
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_title_max_length() {
        let mut req = valid_request();
        req.title = "a".repeat(256);
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_title_over_max_length_rejected() {
        let mut req = valid_request();
        req.title = "a".repeat(257);
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_title_single_char_valid() {
        let mut req = valid_request();
        req.title = "A".to_string();
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_empty_description_rejected() {
        let mut req = valid_request();
        req.description = "".to_string();
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_single_char_description_valid() {
        let mut req = valid_request();
        req.description = "X".to_string();
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_status_must_be_reported() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::Reported;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_acknowledged_status_rejected() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::Acknowledged;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_scheduled_status_rejected() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::Scheduled;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_in_progress_status_rejected() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::InProgress;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_completed_status_rejected() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::Completed;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_deferred_status_rejected() {
        let mut req = valid_request();
        req.status = MaintenanceStatus::Deferred;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_request_with_no_unit_hash_valid() {
        let mut req = valid_request();
        req.unit_hash = None;
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ==================== WorkOrder Tests ====================

    #[test]
    fn test_valid_work_order_creation() {
        let order = valid_work_order();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_empty_assigned_to_rejected() {
        let mut order = valid_work_order();
        order.assigned_to = "".to_string();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_single_char_assigned_to_valid() {
        let mut order = valid_work_order();
        order.assigned_to = "J".to_string();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_empty_description_rejected() {
        let mut order = valid_work_order();
        order.description = "".to_string();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_single_char_description_valid() {
        let mut order = valid_work_order();
        order.description = "X".to_string();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_completed_date_on_create_rejected() {
        let mut order = valid_work_order();
        order.completed_date = Some(test_timestamp());
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_no_estimated_cost_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = None;
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_no_scheduled_date_valid() {
        let mut order = valid_work_order();
        order.scheduled_date = None;
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_empty_notes_valid() {
        let mut order = valid_work_order();
        order.notes = "".to_string();
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ==================== WorkOrder Update Tests ====================

    #[test]
    fn test_valid_work_order_update() {
        let order = valid_work_order();
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_empty_assigned_to_rejected() {
        let mut order = valid_work_order();
        order.assigned_to = "".to_string();
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_actual_within_estimate_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = Some(9500);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_actual_equals_estimate_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = Some(10000);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_actual_exactly_200_percent_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = Some(20000);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_actual_201_percent_rejected() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = Some(20001);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_actual_over_200_percent_rejected() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = Some(30000);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_no_estimated_cost_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = None;
        order.actual_cost_cents = Some(50000);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_no_actual_cost_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(10000);
        order.actual_cost_cents = None;
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_zero_estimated_zero_actual_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(0);
        order.actual_cost_cents = Some(0);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_update_zero_estimated_nonzero_actual_rejected() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(0);
        order.actual_cost_cents = Some(1);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_with_completed_date_valid() {
        let mut order = valid_work_order();
        order.completed_date = Some(test_timestamp_2());
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ==================== Inspection Tests ====================

    #[test]
    fn test_valid_inspection_creation() {
        let inspection = valid_inspection();
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_no_findings_valid() {
        let mut inspection = valid_inspection();
        inspection.findings = vec![];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_max_findings_valid() {
        let mut inspection = valid_inspection();
        inspection.findings = (0..100).map(|i| format!("Finding {}", i)).collect();
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_over_max_findings_rejected() {
        let mut inspection = valid_inspection();
        inspection.findings = (0..101).map(|i| format!("Finding {}", i)).collect();
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_inspection_empty_finding_rejected() {
        let mut inspection = valid_inspection();
        inspection.findings = vec!["Valid finding".to_string(), "".to_string()];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_inspection_all_empty_findings_rejected() {
        let mut inspection = valid_inspection();
        inspection.findings = vec!["".to_string(), "".to_string()];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_inspection_single_char_finding_valid() {
        let mut inspection = valid_inspection();
        inspection.findings = vec!["X".to_string()];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_passed_true_valid() {
        let mut inspection = valid_inspection();
        inspection.passed = true;
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_passed_false_valid() {
        let mut inspection = valid_inspection();
        inspection.passed = false;
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_no_next_due_valid() {
        let mut inspection = valid_inspection();
        inspection.next_due = None;
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ==================== Enum Serde Tests ====================

    #[test]
    fn test_maintenance_category_serde_plumbing() {
        let cat = MaintenanceCategory::Plumbing;
        let serialized = serde_json::to_string(&cat).unwrap();
        let deserialized: MaintenanceCategory = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cat, deserialized);
    }

    #[test]
    fn test_maintenance_category_serde_electrical() {
        let cat = MaintenanceCategory::Electrical;
        let serialized = serde_json::to_string(&cat).unwrap();
        let deserialized: MaintenanceCategory = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cat, deserialized);
    }

    #[test]
    fn test_maintenance_category_serde_hvac() {
        let cat = MaintenanceCategory::HVAC;
        let serialized = serde_json::to_string(&cat).unwrap();
        let deserialized: MaintenanceCategory = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cat, deserialized);
    }

    #[test]
    fn test_maintenance_category_serde_other() {
        let cat = MaintenanceCategory::Other("Custom issue".to_string());
        let serialized = serde_json::to_string(&cat).unwrap();
        let deserialized: MaintenanceCategory = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cat, deserialized);
    }

    #[test]
    fn test_maintenance_priority_serde_emergency() {
        let pri = MaintenancePriority::Emergency;
        let serialized = serde_json::to_string(&pri).unwrap();
        let deserialized: MaintenancePriority = serde_json::from_str(&serialized).unwrap();
        assert_eq!(pri, deserialized);
    }

    #[test]
    fn test_maintenance_priority_serde_urgent() {
        let pri = MaintenancePriority::Urgent;
        let serialized = serde_json::to_string(&pri).unwrap();
        let deserialized: MaintenancePriority = serde_json::from_str(&serialized).unwrap();
        assert_eq!(pri, deserialized);
    }

    #[test]
    fn test_maintenance_priority_serde_normal() {
        let pri = MaintenancePriority::Normal;
        let serialized = serde_json::to_string(&pri).unwrap();
        let deserialized: MaintenancePriority = serde_json::from_str(&serialized).unwrap();
        assert_eq!(pri, deserialized);
    }

    #[test]
    fn test_maintenance_priority_serde_low() {
        let pri = MaintenancePriority::Low;
        let serialized = serde_json::to_string(&pri).unwrap();
        let deserialized: MaintenancePriority = serde_json::from_str(&serialized).unwrap();
        assert_eq!(pri, deserialized);
    }

    #[test]
    fn test_maintenance_status_serde_reported() {
        let status = MaintenanceStatus::Reported;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MaintenanceStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_maintenance_status_serde_in_progress() {
        let status = MaintenanceStatus::InProgress;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MaintenanceStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_maintenance_status_serde_completed() {
        let status = MaintenanceStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MaintenanceStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_inspection_type_serde_annual() {
        let itype = InspectionType::Annual;
        let serialized = serde_json::to_string(&itype).unwrap();
        let deserialized: InspectionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(itype, deserialized);
    }

    #[test]
    fn test_inspection_type_serde_safety() {
        let itype = InspectionType::Safety;
        let serialized = serde_json::to_string(&itype).unwrap();
        let deserialized: InspectionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(itype, deserialized);
    }

    #[test]
    fn test_inspection_type_serde_pre_move() {
        let itype = InspectionType::PreMove;
        let serialized = serde_json::to_string(&itype).unwrap();
        let deserialized: InspectionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(itype, deserialized);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_request_description_at_max_length() {
        let mut req = valid_request();
        req.description = "a".repeat(4096);
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_request_description_over_max_length_rejected() {
        let mut req = valid_request();
        req.description = "a".repeat(4097);
        let result = validate_create_request(test_create_action(), req);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_large_cost_valid() {
        let mut order = valid_work_order();
        // Use large but not overflow-inducing values (1 billion cents = $10M)
        order.estimated_cost_cents = Some(1_000_000_000);
        order.actual_cost_cents = Some(1_500_000_000);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_large_findings_valid() {
        let mut inspection = valid_inspection();
        inspection.findings = (0..50)
            .map(|i| format!("Finding {}: {}", i, "x".repeat(500)))
            .collect();
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_boundary_199_percent_valid() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(100000);
        order.actual_cost_cents = Some(199999);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_boundary_cost_one_cent_over() {
        let mut order = valid_work_order();
        order.estimated_cost_cents = Some(100);
        order.actual_cost_cents = Some(201);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ==================== Length Limit Tests ====================

    #[test]
    fn test_work_order_assigned_to_at_max_length() {
        let mut order = valid_work_order();
        order.assigned_to = "a".repeat(256);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_assigned_to_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.assigned_to = "a".repeat(257);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_description_at_max_length() {
        let mut order = valid_work_order();
        order.description = "a".repeat(4096);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_description_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.description = "a".repeat(4097);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_notes_at_max_length() {
        let mut order = valid_work_order();
        order.notes = "a".repeat(8192);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_work_order_notes_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.notes = "a".repeat(8193);
        let result = validate_create_work_order(test_create_action(), order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_assigned_to_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.assigned_to = "a".repeat(257);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_description_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.description = "a".repeat(4097);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_work_order_update_notes_over_max_length_rejected() {
        let mut order = valid_work_order();
        order.notes = "a".repeat(8193);
        let result = validate_update_work_order(order);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_inspection_finding_at_max_length() {
        let mut inspection = valid_inspection();
        inspection.findings = vec!["a".repeat(4096)];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_inspection_finding_over_max_length_rejected() {
        let mut inspection = valid_inspection();
        inspection.findings = vec!["a".repeat(4097)];
        let result = validate_create_inspection(test_create_action(), inspection);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::OpenRequests
            | LinkTypes::BuildingToRequest
            | LinkTypes::RequestToWorkOrder
            | LinkTypes::BuildingToInspection
            | LinkTypes::ReporterToRequest
            | LinkTypes::CompletedRequests => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_tag_open_requests_at_limit() {
        let result = validate_create_link_tag(LinkTypes::OpenRequests, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_open_requests_over_limit() {
        let result = validate_create_link_tag(LinkTypes::OpenRequests, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_building_to_request_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::BuildingToRequest, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_building_to_request_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::BuildingToRequest, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_completed_requests_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::CompletedRequests, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_completed_requests_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::CompletedRequests, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::OpenRequests, vec![]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
