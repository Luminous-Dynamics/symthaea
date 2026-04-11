// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resources Coordinator Zome
//! Emergency resource registration, deployment, and request fulfillment

use emergency_resources_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, civic_requirement_voting,
    GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};


/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new emergency resource

#[hdk_extern]
pub fn register_resource(input: RegisterResourceInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "register_resource")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_voting(), "deploy_resource")?;
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

    get_latest_record(new_action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "request_resource")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "fulfill_request")?;
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

    get_latest_record(new_action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get_latest_record(action_hash)? {
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
        if let Some(record) = get_latest_record(action_hash)? {
            requests.push(record);
        }
    }

    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn register_resource_input_serde_roundtrip() {
        let input = RegisterResourceInput {
            id: "res-1".to_string(),
            resource_type: ResourceType::Medical,
            name: "Portable Defibrillator".to_string(),
            quantity: 5,
            unit: "units".to_string(),
            location: "Warehouse A".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "res-1");
        assert_eq!(decoded.name, "Portable Defibrillator");
        assert_eq!(decoded.quantity, 5);
        assert_eq!(decoded.resource_type, ResourceType::Medical);
    }

    #[test]
    fn deploy_resource_input_serde_roundtrip() {
        let input = DeployResourceInput {
            resource_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            disaster_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            deployment_location: "Field Hospital Zone B".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeployResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.deployment_location, "Field Hospital Zone B");
    }

    #[test]
    fn request_resource_input_serde_roundtrip() {
        let input = RequestResourceInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            resource_type: ResourceType::Water,
            quantity_needed: 1000,
            urgency: UrgencyLevel::Critical,
            location: "Evacuation Center 3".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RequestResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.resource_type, ResourceType::Water);
        assert_eq!(decoded.quantity_needed, 1000);
        assert_eq!(decoded.urgency, UrgencyLevel::Critical);
    }

    #[test]
    fn fulfill_request_input_serde_roundtrip() {
        let input = FulfillRequestInput {
            request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            resource_hash: ActionHash::from_raw_36(vec![1u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let _decoded: FulfillRequestInput = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn find_shelters_for_resource_input_serde_roundtrip() {
        let input = FindSheltersForResourceInput {
            lat: 34.05,
            lon: -118.25,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindSheltersForResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lat, 34.05);
        assert_eq!(decoded.lon, -118.25);
        assert_eq!(decoded.radius_km, 10.0);
    }

    #[test]
    fn shelter_info_serde_roundtrip() {
        let info = ShelterInfo {
            name: "Central High School".to_string(),
            address: "123 Main St".to_string(),
            capacity: 500,
            current_occupancy: 320,
            available_capacity: 180,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: ShelterInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Central High School");
        assert_eq!(decoded.capacity, 500);
        assert_eq!(decoded.current_occupancy, 320);
        assert_eq!(decoded.available_capacity, 180);
    }

    #[test]
    fn nearby_shelters_result_serde_roundtrip() {
        let result = NearbySheltersResult {
            shelters_found: 2,
            total_available_capacity: 350,
            shelters: vec![ShelterInfo {
                name: "Shelter A".to_string(),
                address: "456 Oak Ave".to_string(),
                capacity: 200,
                current_occupancy: 150,
                available_capacity: 50,
            }],
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: NearbySheltersResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.shelters_found, 2);
        assert_eq!(decoded.total_available_capacity, 350);
        assert_eq!(decoded.shelters.len(), 1);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn nearby_shelters_result_with_error_serde() {
        let result = NearbySheltersResult {
            shelters_found: 0,
            total_available_capacity: 0,
            shelters: vec![],
            error: Some("Zome not available".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: NearbySheltersResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.error, Some("Zome not available".to_string()));
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn resource_type_all_variants_serde() {
        let variants = vec![
            ResourceType::Medical,
            ResourceType::Personnel,
            ResourceType::Equipment,
            ResourceType::Shelter,
            ResourceType::Transport,
            ResourceType::Communication,
            ResourceType::Food,
            ResourceType::Water,
            ResourceType::Power,
            ResourceType::Fuel,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ResourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn resource_status_all_variants_serde() {
        let variants = vec![
            ResourceStatus::Available,
            ResourceStatus::Deployed,
            ResourceStatus::InTransit,
            ResourceStatus::Depleted,
            ResourceStatus::Damaged,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ResourceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn urgency_level_all_variants_serde() {
        let variants = vec![
            UrgencyLevel::Critical,
            UrgencyLevel::High,
            UrgencyLevel::Medium,
            UrgencyLevel::Low,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: UrgencyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn request_status_all_variants_serde() {
        let variants = vec![
            RequestStatus::Pending,
            RequestStatus::Approved,
            RequestStatus::Fulfilled,
            RequestStatus::PartiallyFulfilled,
            RequestStatus::Denied,
            RequestStatus::Cancelled,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // EmergencyResource full struct serde roundtrip
    // ========================================================================

    #[test]
    fn emergency_resource_full_struct_serde_roundtrip() {
        let resource = EmergencyResource {
            id: "res-42".to_string(),
            resource_type: ResourceType::Equipment,
            name: "Portable Generator".to_string(),
            quantity: 10,
            unit: "units".to_string(),
            location: "Staging Area C".to_string(),
            owner: AgentPubKey::from_raw_36(vec![1u8; 36]),
            status: ResourceStatus::Available,
            deployed_to: None,
        };
        let json = serde_json::to_string(&resource).unwrap();
        let decoded: EmergencyResource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "res-42");
        assert_eq!(decoded.name, "Portable Generator");
        assert_eq!(decoded.quantity, 10);
        assert_eq!(decoded.resource_type, ResourceType::Equipment);
        assert_eq!(decoded.status, ResourceStatus::Available);
        assert!(decoded.deployed_to.is_none());
    }

    #[test]
    fn emergency_resource_deployed_serde_roundtrip() {
        let resource = EmergencyResource {
            id: "res-d1".to_string(),
            resource_type: ResourceType::Medical,
            name: "Field Hospital Kit".to_string(),
            quantity: 1,
            unit: "kit".to_string(),
            location: "Disaster Zone B".to_string(),
            owner: AgentPubKey::from_raw_36(vec![2u8; 36]),
            status: ResourceStatus::Deployed,
            deployed_to: Some(ActionHash::from_raw_36(vec![3u8; 36])),
        };
        let json = serde_json::to_string(&resource).unwrap();
        let decoded: EmergencyResource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, ResourceStatus::Deployed);
        assert!(decoded.deployed_to.is_some());
    }

    #[test]
    fn emergency_resource_zero_quantity_serde() {
        let resource = EmergencyResource {
            id: "res-0".to_string(),
            resource_type: ResourceType::Water,
            name: "Water Tank".to_string(),
            quantity: 0,
            unit: "gallons".to_string(),
            location: "Empty".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            status: ResourceStatus::Depleted,
            deployed_to: None,
        };
        let json = serde_json::to_string(&resource).unwrap();
        let decoded: EmergencyResource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity, 0);
        assert_eq!(decoded.status, ResourceStatus::Depleted);
    }

    #[test]
    fn emergency_resource_max_quantity_serde() {
        let resource = EmergencyResource {
            id: "res-max".to_string(),
            resource_type: ResourceType::Food,
            name: "MRE Packs".to_string(),
            quantity: u32::MAX,
            unit: "meals".to_string(),
            location: "National Stockpile".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            status: ResourceStatus::Available,
            deployed_to: None,
        };
        let json = serde_json::to_string(&resource).unwrap();
        let decoded: EmergencyResource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity, u32::MAX);
    }

    // ========================================================================
    // ResourceRequest full struct serde roundtrip
    // ========================================================================

    #[test]
    fn resource_request_full_struct_serde_roundtrip() {
        let request = ResourceRequest {
            disaster_hash: ActionHash::from_raw_36(vec![10u8; 36]),
            requesting_team: AgentPubKey::from_raw_36(vec![11u8; 36]),
            resource_type: ResourceType::Personnel,
            quantity_needed: 25,
            urgency: UrgencyLevel::High,
            location: "Command Post Alpha".to_string(),
            status: RequestStatus::Pending,
            fulfilled_by: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: ResourceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.resource_type, ResourceType::Personnel);
        assert_eq!(decoded.quantity_needed, 25);
        assert_eq!(decoded.urgency, UrgencyLevel::High);
        assert_eq!(decoded.status, RequestStatus::Pending);
        assert!(decoded.fulfilled_by.is_none());
    }

    #[test]
    fn resource_request_fulfilled_serde_roundtrip() {
        let request = ResourceRequest {
            disaster_hash: ActionHash::from_raw_36(vec![20u8; 36]),
            requesting_team: AgentPubKey::from_raw_36(vec![21u8; 36]),
            resource_type: ResourceType::Communication,
            quantity_needed: 100,
            urgency: UrgencyLevel::Critical,
            location: "City Hall".to_string(),
            status: RequestStatus::Fulfilled,
            fulfilled_by: Some(ActionHash::from_raw_36(vec![22u8; 36])),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: ResourceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, RequestStatus::Fulfilled);
        assert!(decoded.fulfilled_by.is_some());
    }

    // ========================================================================
    // Input struct boundary conditions
    // ========================================================================

    #[test]
    fn register_resource_input_all_resource_types() {
        let types = vec![
            ResourceType::Medical,
            ResourceType::Personnel,
            ResourceType::Equipment,
            ResourceType::Shelter,
            ResourceType::Transport,
            ResourceType::Communication,
            ResourceType::Food,
            ResourceType::Water,
            ResourceType::Power,
            ResourceType::Fuel,
        ];
        for rt in types {
            let input = RegisterResourceInput {
                id: "r-1".to_string(),
                resource_type: rt.clone(),
                name: "Test Resource".to_string(),
                quantity: 1,
                unit: "ea".to_string(),
                location: "Loc".to_string(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: RegisterResourceInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.resource_type, rt);
        }
    }

    #[test]
    fn register_resource_input_zero_quantity_serde() {
        let input = RegisterResourceInput {
            id: "r-0".to_string(),
            resource_type: ResourceType::Fuel,
            name: "Empty Fuel Tank".to_string(),
            quantity: 0,
            unit: "liters".to_string(),
            location: "Depot".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity, 0);
    }

    #[test]
    fn register_resource_input_max_quantity_serde() {
        let input = RegisterResourceInput {
            id: "r-max".to_string(),
            resource_type: ResourceType::Power,
            name: "Battery Packs".to_string(),
            quantity: u32::MAX,
            unit: "packs".to_string(),
            location: "Grid".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity, u32::MAX);
    }

    #[test]
    fn request_resource_input_all_urgency_levels() {
        let levels = vec![
            UrgencyLevel::Critical,
            UrgencyLevel::High,
            UrgencyLevel::Medium,
            UrgencyLevel::Low,
        ];
        for level in levels {
            let input = RequestResourceInput {
                disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                resource_type: ResourceType::Medical,
                quantity_needed: 1,
                urgency: level.clone(),
                location: "Test".to_string(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: RequestResourceInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.urgency, level);
        }
    }

    // ========================================================================
    // ShelterInfo and NearbySheltersResult edge cases
    // ========================================================================

    #[test]
    fn shelter_info_zero_values_serde() {
        let info = ShelterInfo {
            name: "Empty".to_string(),
            address: "N/A".to_string(),
            capacity: 0,
            current_occupancy: 0,
            available_capacity: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: ShelterInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.capacity, 0);
        assert_eq!(decoded.available_capacity, 0);
    }

    #[test]
    fn shelter_info_max_capacity_serde() {
        let info = ShelterInfo {
            name: "Mega Shelter".to_string(),
            address: "1 Big Way".to_string(),
            capacity: u32::MAX,
            current_occupancy: 0,
            available_capacity: u32::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: ShelterInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.capacity, u32::MAX);
    }

    #[test]
    fn nearby_shelters_result_empty_shelters_no_error() {
        let result = NearbySheltersResult {
            shelters_found: 0,
            total_available_capacity: 0,
            shelters: vec![],
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: NearbySheltersResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.shelters_found, 0);
        assert!(decoded.shelters.is_empty());
        assert!(decoded.error.is_none());
    }

    #[test]
    fn nearby_shelters_result_multiple_shelters_serde() {
        let result = NearbySheltersResult {
            shelters_found: 3,
            total_available_capacity: 500,
            shelters: vec![
                ShelterInfo {
                    name: "A".to_string(),
                    address: "1 St".to_string(),
                    capacity: 200,
                    current_occupancy: 100,
                    available_capacity: 100,
                },
                ShelterInfo {
                    name: "B".to_string(),
                    address: "2 St".to_string(),
                    capacity: 300,
                    current_occupancy: 100,
                    available_capacity: 200,
                },
                ShelterInfo {
                    name: "C".to_string(),
                    address: "3 St".to_string(),
                    capacity: 400,
                    current_occupancy: 200,
                    available_capacity: 200,
                },
            ],
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: NearbySheltersResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.shelters.len(), 3);
        assert_eq!(decoded.total_available_capacity, 500);
    }

    // ========================================================================
    // Resource status lifecycle
    // ========================================================================

    #[test]
    fn all_resource_statuses_serde_individual() {
        // Verify each status serializes to the expected string
        let available_json = serde_json::to_string(&ResourceStatus::Available).unwrap();
        assert!(available_json.contains("Available"));

        let deployed_json = serde_json::to_string(&ResourceStatus::Deployed).unwrap();
        assert!(deployed_json.contains("Deployed"));

        let damaged_json = serde_json::to_string(&ResourceStatus::Damaged).unwrap();
        assert!(damaged_json.contains("Damaged"));
    }

    #[test]
    fn all_request_statuses_serde_individual() {
        let pending_json = serde_json::to_string(&RequestStatus::Pending).unwrap();
        assert!(pending_json.contains("Pending"));

        let denied_json = serde_json::to_string(&RequestStatus::Denied).unwrap();
        assert!(denied_json.contains("Denied"));

        let partial_json = serde_json::to_string(&RequestStatus::PartiallyFulfilled).unwrap();
        assert!(partial_json.contains("PartiallyFulfilled"));
    }
}

// ============================================================================
// Cross-domain: Find nearby shelters for resource deployment
// ============================================================================

/// Wire-compatible copy of emergency_shelters Shelter for deserialization.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct LocalShelter {
    pub id: String,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub address: String,
    pub capacity: u32,
    pub current_occupancy: u32,
    pub shelter_type: LocalShelterType,
    pub amenities: Vec<LocalAmenity>,
    pub status: LocalShelterStatus,
    pub contact: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalShelterType {
    GeneralPopulation,
    SpecialNeeds,
    Medical,
    Pet,
    Evacuation,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalAmenity {
    Beds,
    Kitchen,
    Showers,
    Laundry,
    MedicalBay,
    PetArea,
    ChildArea,
    Generator,
    WheelchairAccess,
    Wifi,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum LocalShelterStatus {
    Open,
    Full,
    Closed,
    Evacuating,
}

/// Wire-compatible copy of emergency_shelters FindNearbySheltersInput.
#[derive(Serialize, Deserialize, Debug)]
struct LocalFindNearbySheltersInput {
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindSheltersForResourceInput {
    /// Latitude of the resource depot or deployment point.
    pub lat: f64,
    /// Longitude of the resource depot or deployment point.
    pub lon: f64,
    /// Search radius in kilometers.
    pub radius_km: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ShelterInfo {
    pub name: String,
    pub address: String,
    pub capacity: u32,
    pub current_occupancy: u32,
    pub available_capacity: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NearbySheltersResult {
    pub shelters_found: u32,
    pub total_available_capacity: u32,
    pub shelters: Vec<ShelterInfo>,
    pub error: Option<String>,
}

/// Find nearby shelters that could receive deployed resources.
///
/// Cross-domain call: emergency-resources queries emergency_shelters
/// via `call(CallTargetCell::Local, ...)` to find open shelters near
/// a resource depot. This enables smart resource routing to where
/// shelter occupants need them most.
#[hdk_extern]
pub fn find_shelters_needing_resources(
    input: FindSheltersForResourceInput,
) -> ExternResult<NearbySheltersResult> {
    let search = LocalFindNearbySheltersInput {
        lat: input.lat,
        lon: input.lon,
        radius_km: input.radius_km,
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("emergency_shelters"),
        FunctionName::from("find_nearby_shelters"),
        None,
        search,
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;

            let mut shelters = Vec::new();
            let mut total_available = 0u32;

            for record in &records {
                if let Some(shelter) = record
                    .entry()
                    .to_app_option::<LocalShelter>()
                    .ok()
                    .flatten()
                {
                    let available = shelter.capacity.saturating_sub(shelter.current_occupancy);
                    total_available += available;

                    shelters.push(ShelterInfo {
                        name: shelter.name,
                        address: shelter.address,
                        capacity: shelter.capacity,
                        current_occupancy: shelter.current_occupancy,
                        available_capacity: available,
                    });
                }
            }

            // Sort by most occupied first (highest need)
            shelters.sort_by(|a, b| b.current_occupancy.cmp(&a.current_occupancy));

            Ok(NearbySheltersResult {
                shelters_found: shelters.len() as u32,
                total_available_capacity: total_available,
                shelters,
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(NearbySheltersResult {
            shelters_found: 0,
            total_available_capacity: 0,
            shelters: vec![],
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(NearbySheltersResult {
            shelters_found: 0,
            total_available_capacity: 0,
            shelters: vec![],
            error: Some("Failed to query emergency shelters zome".into()),
        }),
    }
}
