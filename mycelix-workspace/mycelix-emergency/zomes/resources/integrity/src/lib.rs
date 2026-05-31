// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Resources Integrity Zome
//! Emergency resource tracking and deployment

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// An emergency resource
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyResource {
    pub id: String,
    pub resource_type: ResourceType,
    pub name: String,
    pub quantity: u32,
    pub unit: String,
    pub location: String,
    pub owner: AgentPubKey,
    pub status: ResourceStatus,
    pub deployed_to: Option<ActionHash>,
}

/// Types of emergency resources
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceType {
    Medical,
    Personnel,
    Equipment,
    Shelter,
    Transport,
    Communication,
    Food,
    Water,
    Power,
    Fuel,
}

/// Resource status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceStatus {
    Available,
    Deployed,
    InTransit,
    Depleted,
    Damaged,
}

/// A request for resources
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResourceRequest {
    pub disaster_hash: ActionHash,
    pub requesting_team: AgentPubKey,
    pub resource_type: ResourceType,
    pub quantity_needed: u32,
    pub urgency: UrgencyLevel,
    pub location: String,
    pub status: RequestStatus,
    pub fulfilled_by: Option<ActionHash>,
}

/// Urgency levels for resource requests
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UrgencyLevel {
    Critical,
    High,
    Medium,
    Low,
}

/// Status of a resource request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Pending,
    Approved,
    Fulfilled,
    PartiallyFulfilled,
    Denied,
    Cancelled,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    EmergencyResource(EmergencyResource),
    ResourceRequest(ResourceRequest),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllResources,
    AvailableResources,
    ResourceByType,
    DisasterToRequest,
    AgentToResource,
    RequestToResource,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyResource(resource) => {
                    validate_create_resource(action, resource)
                }
                EntryTypes::ResourceRequest(request) => validate_create_request(action, request),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyResource(resource) => validate_update_resource(resource),
                EntryTypes::ResourceRequest(request) => validate_update_request(request),
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
            LinkTypes::AllResources => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AvailableResources => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ResourceByType => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToResource => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequestToResource => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AvailableResources => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_resource(
    _action: Create,
    resource: EmergencyResource,
) -> ExternResult<ValidateCallbackResult> {
    if resource.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".into(),
        ));
    }
    if resource.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".into(),
        ));
    }
    if resource.unit.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource unit cannot be empty".into(),
        ));
    }
    if resource.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_resource(resource: EmergencyResource) -> ExternResult<ValidateCallbackResult> {
    if resource.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_request(
    _action: Create,
    request: ResourceRequest,
) -> ExternResult<ValidateCallbackResult> {
    if request.quantity_needed == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity needed must be greater than 0".into(),
        ));
    }
    if request.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(request: ResourceRequest) -> ExternResult<ValidateCallbackResult> {
    if request.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
