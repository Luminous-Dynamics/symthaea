// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Resources Coordinator Zome
//! Emergency resource registration, deployment, and request fulfillment

use hdk::prelude::*;
use resources_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new emergency resource
#[hdk_extern]
pub fn register_resource(input: RegisterResourceInput) -> ExternResult<Record> {
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Name must be 1-256 characters".into()
        )));
    }
    if input.unit.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unit cannot be empty".into()
        )));
    }
    if input.location.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Location cannot be empty".into()
        )));
    }

    let agent_info = agent_info()?;

    let resource = EmergencyResource {
        id: input.id.clone(),
        resource_type: input.resource_type.clone(),
        name: input.name,
        quantity: input.quantity,
        unit: input.unit,
        location: input.location,
        owner: agent_info.agent_initial_pubkey.clone(),
        status: ResourceStatus::Available,
        deployed_to: None,
    };

    let action_hash = create_entry(&EntryTypes::EmergencyResource(resource))?;

    // Link to all resources
    create_entry(&EntryTypes::Anchor(Anchor("all_resources".to_string())))?;
    create_link(
        anchor_hash("all_resources")?,
        action_hash.clone(),
        LinkTypes::AllResources,
        (),
    )?;

    // Link to available resources
    create_entry(&EntryTypes::Anchor(Anchor(
        "available_resources".to_string(),
    )))?;
    create_link(
        anchor_hash("available_resources")?,
        action_hash.clone(),
        LinkTypes::AvailableResources,
        (),
    )?;

    // Link by resource type
    let type_anchor = format!("resource_type:{:?}", input.resource_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::ResourceByType,
        (),
    )?;

    // Link agent to resource
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToResource,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created resource".into()
    )))
}

/// Input for registering a resource
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterResourceInput {
    pub id: String,
    pub resource_type: ResourceType,
    pub name: String,
    pub quantity: u32,
    pub unit: String,
    pub location: String,
}

/// Deploy a resource to a disaster
#[hdk_extern]
pub fn deploy_resource(input: DeployResourceInput) -> ExternResult<Record> {
    let current_record = get(input.resource_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Resource not found".into())),
    )?;

    let current_resource: EmergencyResource = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid resource entry".into()
        )))?;

    if current_resource.status != ResourceStatus::Available {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Resource is not available for deployment".into()
        )));
    }

    let updated_resource = EmergencyResource {
        status: ResourceStatus::Deployed,
        deployed_to: Some(input.disaster_hash),
        location: input.deployment_location,
        ..current_resource
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::EmergencyResource(updated_resource),
    )?;

    // Remove from available resources
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("available_resources")?,
            LinkTypes::AvailableResources,
        )?,
        GetStrategy::default(),
    )?;
    for link in links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == input.resource_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated resource".into()
    )))
}

/// Input for deploying a resource
#[derive(Serialize, Deserialize, Debug)]
pub struct DeployResourceInput {
    pub resource_hash: ActionHash,
    pub disaster_hash: ActionHash,
    pub deployment_location: String,
}

/// Request resources for a disaster
#[hdk_extern]
pub fn request_resource(input: RequestResourceInput) -> ExternResult<Record> {
    if input.quantity_needed == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quantity needed must be greater than 0".into()
        )));
    }
    if input.location.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Location cannot be empty".into()
        )));
    }

    let agent_info = agent_info()?;

    let request = ResourceRequest {
        disaster_hash: input.disaster_hash.clone(),
        requesting_team: agent_info.agent_initial_pubkey,
        resource_type: input.resource_type,
        quantity_needed: input.quantity_needed,
        urgency: input.urgency,
        location: input.location,
        status: RequestStatus::Pending,
        fulfilled_by: None,
    };

    let action_hash = create_entry(&EntryTypes::ResourceRequest(request))?;

    // Link disaster to request
    let disaster_anchor = format!("disaster_requests:{}", input.disaster_hash);
    create_entry(&EntryTypes::Anchor(Anchor(disaster_anchor.clone())))?;
    create_link(
        anchor_hash(&disaster_anchor)?,
        action_hash.clone(),
        LinkTypes::DisasterToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

/// Input for requesting a resource
#[derive(Serialize, Deserialize, Debug)]
pub struct RequestResourceInput {
    pub disaster_hash: ActionHash,
    pub resource_type: ResourceType,
    pub quantity_needed: u32,
    pub urgency: UrgencyLevel,
    pub location: String,
}

/// Fulfill a resource request
#[hdk_extern]
pub fn fulfill_request(input: FulfillRequestInput) -> ExternResult<Record> {
    let current_record = get(input.request_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Request not found".into())),
    )?;

    let current_request: ResourceRequest = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    if current_request.status != RequestStatus::Pending
        && current_request.status != RequestStatus::Approved
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request is not in a fulfillable state".into()
        )));
    }

    let updated_request = ResourceRequest {
        status: RequestStatus::Fulfilled,
        fulfilled_by: Some(input.resource_hash.clone()),
        ..current_request
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::ResourceRequest(updated_request),
    )?;

    // Link request to fulfilling resource
    create_link(
        input.request_hash,
        input.resource_hash,
        LinkTypes::RequestToResource,
        (),
    )?;

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated request".into()
    )))
}

/// Input for fulfilling a request
#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillRequestInput {
    pub request_hash: ActionHash,
    pub resource_hash: ActionHash,
}

/// Get all available resources
#[hdk_extern]
pub fn get_available_resources(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("available_resources")?,
            LinkTypes::AvailableResources,
        )?,
        GetStrategy::default(),
    )?;

    let mut resources = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            resources.push(record);
        }
    }

    Ok(resources)
}

/// Get resource requests for a disaster
#[hdk_extern]
pub fn get_resource_requests(disaster_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let disaster_anchor = format!("disaster_requests:{}", disaster_hash);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&disaster_anchor)?, LinkTypes::DisasterToRequest)?,
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

    Ok(requests)
}
