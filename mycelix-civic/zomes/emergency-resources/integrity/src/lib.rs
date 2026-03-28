// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resources Integrity Zome
//! Emergency resource tracking and deployment

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::AllResources => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AvailableResources => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ResourceByType => {
                    // Type links may store serialized type metadata
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DisasterToRequest => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AgentToResource => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::RequestToResource => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
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
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
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

fn validate_create_resource(
    _action: Create,
    resource: EmergencyResource,
) -> ExternResult<ValidateCallbackResult> {
    if resource.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".into(),
        ));
    }
    if resource.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID too long (max 256)".into(),
        ));
    }
    if resource.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".into(),
        ));
    }
    if resource.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name too long (max 256)".into(),
        ));
    }
    if resource.unit.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource unit cannot be empty".into(),
        ));
    }
    if resource.unit.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource unit too long (max 128)".into(),
        ));
    }
    if resource.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource location cannot be empty".into(),
        ));
    }
    if resource.location.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource location too long (max 4096)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_resource(resource: EmergencyResource) -> ExternResult<ValidateCallbackResult> {
    if resource.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".into(),
        ));
    }
    if resource.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID too long (max 256)".into(),
        ));
    }
    if resource.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name too long (max 256)".into(),
        ));
    }
    if resource.unit.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource unit too long (max 128)".into(),
        ));
    }
    if resource.location.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource location too long (max 4096)".into(),
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
    if request.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location cannot be empty".into(),
        ));
    }
    if request.location.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location too long (max 4096)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(request: ResourceRequest) -> ExternResult<ValidateCallbackResult> {
    if request.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location cannot be empty".into(),
        ));
    }
    if request.location.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request location too long (max 4096)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    // ========================================================================
    // CONSTRUCTION HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn make_resource() -> EmergencyResource {
        EmergencyResource {
            id: "res-001".into(),
            resource_type: ResourceType::Medical,
            name: "First Aid Kit".into(),
            quantity: 50,
            unit: "kits".into(),
            location: "Warehouse A, Richardson TX".into(),
            owner: fake_agent(),
            status: ResourceStatus::Available,
            deployed_to: None,
        }
    }

    fn make_request() -> ResourceRequest {
        ResourceRequest {
            disaster_hash: fake_action_hash(),
            requesting_team: fake_agent(),
            resource_type: ResourceType::Medical,
            quantity_needed: 10,
            urgency: UrgencyLevel::High,
            location: "Disaster Zone B, Dallas TX".into(),
            status: RequestStatus::Pending,
            fulfilled_by: None,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP - ResourceType
    // ========================================================================

    #[test]
    fn resource_type_all_variants_serde_roundtrip() {
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
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: ResourceType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP - ResourceStatus
    // ========================================================================

    #[test]
    fn resource_status_all_variants_serde_roundtrip() {
        let variants = vec![
            ResourceStatus::Available,
            ResourceStatus::Deployed,
            ResourceStatus::InTransit,
            ResourceStatus::Depleted,
            ResourceStatus::Damaged,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: ResourceStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP - UrgencyLevel
    // ========================================================================

    #[test]
    fn urgency_level_all_variants_serde_roundtrip() {
        let variants = vec![
            UrgencyLevel::Critical,
            UrgencyLevel::High,
            UrgencyLevel::Medium,
            UrgencyLevel::Low,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: UrgencyLevel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP - RequestStatus
    // ========================================================================

    #[test]
    fn request_status_all_variants_serde_roundtrip() {
        let variants = vec![
            RequestStatus::Pending,
            RequestStatus::Approved,
            RequestStatus::Fulfilled,
            RequestStatus::PartiallyFulfilled,
            RequestStatus::Denied,
            RequestStatus::Cancelled,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: RequestStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP - EmergencyResource
    // ========================================================================

    #[test]
    fn emergency_resource_serde_roundtrip() {
        let resource = make_resource();
        let json = serde_json::to_string(&resource).expect("serialize");
        let parsed: EmergencyResource = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, resource);
    }

    #[test]
    fn emergency_resource_with_deployed_to_serde_roundtrip() {
        let mut resource = make_resource();
        resource.status = ResourceStatus::Deployed;
        resource.deployed_to = Some(fake_action_hash());
        let json = serde_json::to_string(&resource).expect("serialize");
        let parsed: EmergencyResource = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, resource);
    }

    // ========================================================================
    // SERDE ROUNDTRIP - ResourceRequest
    // ========================================================================

    #[test]
    fn resource_request_serde_roundtrip() {
        let request = make_request();
        let json = serde_json::to_string(&request).expect("serialize");
        let parsed: ResourceRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, request);
    }

    #[test]
    fn resource_request_with_fulfilled_by_serde_roundtrip() {
        let mut request = make_request();
        request.status = RequestStatus::Fulfilled;
        request.fulfilled_by = Some(fake_action_hash());
        let json = serde_json::to_string(&request).expect("serialize");
        let parsed: ResourceRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, request);
    }

    // ========================================================================
    // SERDE ROUNDTRIP - Anchor
    // ========================================================================

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = Anchor("all_resources".into());
        let json = serde_json::to_string(&anchor).expect("serialize");
        let parsed: Anchor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, anchor);
    }

    // ========================================================================
    // validate_create_resource - VALID CASES
    // ========================================================================

    #[test]
    fn create_resource_valid_passes() {
        let result = validate_create_resource(fake_create(), make_resource());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_zero_quantity_passes() {
        // Validation does NOT reject quantity == 0 for resources
        let mut r = make_resource();
        r.quantity = 0;
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_max_quantity_passes() {
        let mut r = make_resource();
        r.quantity = u32::MAX;
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_deployed_status_passes() {
        let mut r = make_resource();
        r.status = ResourceStatus::Deployed;
        r.deployed_to = Some(fake_action_hash());
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_all_resource_types_pass() {
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
            let mut r = make_resource();
            r.resource_type = rt.clone();
            let result = validate_create_resource(fake_create(), r);
            assert!(is_valid(&result), "ResourceType {:?} should pass", rt);
        }
    }

    #[test]
    fn create_resource_all_statuses_pass() {
        let statuses = vec![
            ResourceStatus::Available,
            ResourceStatus::Deployed,
            ResourceStatus::InTransit,
            ResourceStatus::Depleted,
            ResourceStatus::Damaged,
        ];
        for status in statuses {
            let mut r = make_resource();
            r.status = status.clone();
            let result = validate_create_resource(fake_create(), r);
            assert!(is_valid(&result), "ResourceStatus {:?} should pass", status);
        }
    }

    #[test]
    fn create_resource_none_deployed_to_passes() {
        let mut r = make_resource();
        r.deployed_to = None;
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_resource - INVALID: empty id
    // ========================================================================

    #[test]
    fn create_resource_empty_id_rejected() {
        let mut r = make_resource();
        r.id = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource ID cannot be empty");
    }

    // ========================================================================
    // validate_create_resource - INVALID: empty name
    // ========================================================================

    #[test]
    fn create_resource_empty_name_rejected() {
        let mut r = make_resource();
        r.name = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource name cannot be empty");
    }

    // ========================================================================
    // validate_create_resource - INVALID: empty unit
    // ========================================================================

    #[test]
    fn create_resource_empty_unit_rejected() {
        let mut r = make_resource();
        r.unit = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource unit cannot be empty");
    }

    // ========================================================================
    // validate_create_resource - INVALID: empty location
    // ========================================================================

    #[test]
    fn create_resource_empty_location_rejected() {
        let mut r = make_resource();
        r.location = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource location cannot be empty");
    }

    // ========================================================================
    // validate_create_resource - EDGE CASES: unicode strings
    // ========================================================================

    #[test]
    fn create_resource_unicode_id_passes() {
        let mut r = make_resource();
        r.id = "\u{1F6D1}-\u{6551}\u{8D44}-01".into(); // Emoji + Chinese chars
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_unicode_name_passes() {
        let mut r = make_resource();
        r.name = "\u{5BFC}\u{822A}\u{8BBE}\u{5907} \u{2014} Navigation Equipment".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_unicode_location_passes() {
        let mut r = make_resource();
        r.location = "\u{6771}\u{4EAC}\u{90FD}\u{6E0B}\u{8C37}\u{533A}".into(); // Tokyo Shibuya
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_unicode_unit_passes() {
        let mut r = make_resource();
        r.unit = "\u{5355}\u{4F4D}".into(); // Chinese for "unit"
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_resource - EDGE: whitespace-only strings pass
    // (Validation only checks is_empty, not whitespace-only)
    // ========================================================================

    #[test]
    fn create_resource_whitespace_only_id_rejected() {
        let mut r = make_resource();
        r.id = "   ".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(!is_valid(&result));
    }

    #[test]
    fn create_resource_whitespace_only_name_rejected() {
        let mut r = make_resource();
        r.name = " \t\n".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(!is_valid(&result));
    }

    // ========================================================================
    // validate_create_resource - EDGE: first failing field wins
    // (validation checks id -> name -> unit -> location in order)
    // ========================================================================

    #[test]
    fn create_resource_multiple_empty_fields_reports_id_first() {
        let mut r = make_resource();
        r.id = "".into();
        r.name = "".into();
        r.unit = "".into();
        r.location = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource ID cannot be empty");
    }

    #[test]
    fn create_resource_empty_name_and_unit_reports_name_first() {
        let mut r = make_resource();
        r.name = "".into();
        r.unit = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource name cannot be empty");
    }

    #[test]
    fn create_resource_empty_unit_and_location_reports_unit_first() {
        let mut r = make_resource();
        r.unit = "".into();
        r.location = "".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource unit cannot be empty");
    }

    // ========================================================================
    // validate_update_resource TESTS
    // ========================================================================

    #[test]
    fn update_resource_valid_passes() {
        let result = validate_update_resource(make_resource());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_resource_empty_id_rejected() {
        let mut r = make_resource();
        r.id = "".into();
        let result = validate_update_resource(r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource ID cannot be empty");
    }

    #[test]
    fn update_resource_empty_name_passes() {
        // validate_update_resource only checks id, not name
        let mut r = make_resource();
        r.name = "".into();
        let result = validate_update_resource(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_resource_empty_unit_passes() {
        // validate_update_resource only checks id
        let mut r = make_resource();
        r.unit = "".into();
        let result = validate_update_resource(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_resource_empty_location_passes() {
        // validate_update_resource only checks id
        let mut r = make_resource();
        r.location = "".into();
        let result = validate_update_resource(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_resource_zero_quantity_passes() {
        let mut r = make_resource();
        r.quantity = 0;
        let result = validate_update_resource(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_resource_depleted_status_passes() {
        let mut r = make_resource();
        r.status = ResourceStatus::Depleted;
        let result = validate_update_resource(r);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_request - VALID CASES
    // ========================================================================

    #[test]
    fn create_request_valid_passes() {
        let result = validate_create_request(fake_create(), make_request());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_quantity_one_passes() {
        let mut req = make_request();
        req.quantity_needed = 1;
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_max_quantity_passes() {
        let mut req = make_request();
        req.quantity_needed = u32::MAX;
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_all_urgency_levels_pass() {
        let levels = vec![
            UrgencyLevel::Critical,
            UrgencyLevel::High,
            UrgencyLevel::Medium,
            UrgencyLevel::Low,
        ];
        for level in levels {
            let mut req = make_request();
            req.urgency = level.clone();
            let result = validate_create_request(fake_create(), req);
            assert!(is_valid(&result), "UrgencyLevel {:?} should pass", level);
        }
    }

    #[test]
    fn create_request_all_request_statuses_pass() {
        let statuses = vec![
            RequestStatus::Pending,
            RequestStatus::Approved,
            RequestStatus::Fulfilled,
            RequestStatus::PartiallyFulfilled,
            RequestStatus::Denied,
            RequestStatus::Cancelled,
        ];
        for status in statuses {
            let mut req = make_request();
            req.status = status.clone();
            let result = validate_create_request(fake_create(), req);
            assert!(is_valid(&result), "RequestStatus {:?} should pass", status);
        }
    }

    #[test]
    fn create_request_with_fulfilled_by_passes() {
        let mut req = make_request();
        req.fulfilled_by = Some(fake_action_hash());
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_none_fulfilled_by_passes() {
        let mut req = make_request();
        req.fulfilled_by = None;
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_request - INVALID: zero quantity
    // ========================================================================

    #[test]
    fn create_request_zero_quantity_rejected() {
        let mut req = make_request();
        req.quantity_needed = 0;
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Quantity needed must be greater than 0"
        );
    }

    // ========================================================================
    // validate_create_request - INVALID: empty location
    // ========================================================================

    #[test]
    fn create_request_empty_location_rejected() {
        let mut req = make_request();
        req.location = "".into();
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Request location cannot be empty");
    }

    // ========================================================================
    // validate_create_request - EDGE: first failing field wins
    // ========================================================================

    #[test]
    fn create_request_zero_quantity_and_empty_location_reports_quantity_first() {
        let mut req = make_request();
        req.quantity_needed = 0;
        req.location = "".into();
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Quantity needed must be greater than 0"
        );
    }

    // ========================================================================
    // validate_create_request - EDGE: unicode location
    // ========================================================================

    #[test]
    fn create_request_unicode_location_passes() {
        let mut req = make_request();
        req.location = "\u{0421}\u{0430}\u{043D}\u{043A}\u{0442}-\u{041F}\u{0435}\u{0442}\u{0435}\u{0440}\u{0431}\u{0443}\u{0440}\u{0433}".into(); // Sankt-Peterburg
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_request - EDGE: whitespace-only location rejected
    // ========================================================================

    #[test]
    fn create_request_whitespace_only_location_rejected() {
        let mut req = make_request();
        req.location = "   ".into();
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // validate_update_request TESTS
    // ========================================================================

    #[test]
    fn update_request_valid_passes() {
        let result = validate_update_request(make_request());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_request_empty_location_rejected() {
        let mut req = make_request();
        req.location = "".into();
        let result = validate_update_request(req);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Request location cannot be empty");
    }

    #[test]
    fn update_request_zero_quantity_passes() {
        // validate_update_request only checks location, not quantity
        let mut req = make_request();
        req.quantity_needed = 0;
        let result = validate_update_request(req);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_request_cancelled_status_passes() {
        let mut req = make_request();
        req.status = RequestStatus::Cancelled;
        let result = validate_update_request(req);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_request_unicode_location_passes() {
        let mut req = make_request();
        req.location = "\u{00C1}rea de desastre, M\u{00E9}xico".into();
        let result = validate_update_request(req);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_request_whitespace_only_location_rejected() {
        let mut req = make_request();
        req.location = " \t ".into();
        let result = validate_update_request(req);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // EDGE CASES: very long strings
    // ========================================================================

    #[test]
    fn create_resource_very_long_name_rejected() {
        let mut r = make_resource();
        r.name = "A".repeat(10_000);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_request_very_long_location_rejected() {
        let mut req = make_request();
        req.location = "B".repeat(10_000);
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // EDGE CASES: single-char strings
    // ========================================================================

    #[test]
    fn create_resource_single_char_fields_pass() {
        let mut r = make_resource();
        r.id = "X".into();
        r.name = "Y".into();
        r.unit = "Z".into();
        r.location = "W".into();
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_single_char_location_passes() {
        let mut req = make_request();
        req.location = "A".into();
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::AllResources
            | LinkTypes::AvailableResources
            | LinkTypes::DisasterToRequest
            | LinkTypes::AgentToResource
            | LinkTypes::RequestToResource => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
            LinkTypes::ResourceByType => {
                if tag_len > 512 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 512 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- AllResources (256-byte limit) boundary tests --

    #[test]
    fn link_tag_all_resources_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::AllResources, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_resources_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::AllResources, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_resources_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::AllResources, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- ResourceByType (512-byte limit) boundary tests --

    #[test]
    fn link_tag_resource_by_type_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::ResourceByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_resource_by_type_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 512]);
        let result = validate_create_link_tag(&LinkTypes::ResourceByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_resource_by_type_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 513]);
        let result = validate_create_link_tag(&LinkTypes::ResourceByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- AgentToResource (256-byte limit) boundary tests --

    #[test]
    fn link_tag_agent_to_resource_at_limit_valid() {
        let tag = LinkTag::new(vec![0xAB; 256]);
        let result = validate_create_link_tag(&LinkTypes::AgentToResource, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_agent_to_resource_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xAB; 257]);
        let result = validate_create_link_tag(&LinkTypes::AgentToResource, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::AllResources,
            LinkTypes::AvailableResources,
            LinkTypes::ResourceByType,
            LinkTypes::DisasterToRequest,
            LinkTypes::AgentToResource,
            LinkTypes::RequestToResource,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // ========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS
    // ========================================================================

    // -- Resource ID (max 256) --

    #[test]
    fn create_resource_id_at_limit_accepted() {
        let mut r = make_resource();
        r.id = "x".repeat(256);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_id_over_limit_rejected() {
        let mut r = make_resource();
        r.id = "x".repeat(257);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource ID too long (max 256)");
    }

    // -- Resource name (max 256) --

    #[test]
    fn create_resource_name_at_limit_accepted() {
        let mut r = make_resource();
        r.name = "x".repeat(256);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_name_over_limit_rejected() {
        let mut r = make_resource();
        r.name = "x".repeat(257);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource name too long (max 256)");
    }

    // -- Resource unit (max 128) --

    #[test]
    fn create_resource_unit_at_limit_accepted() {
        let mut r = make_resource();
        r.unit = "x".repeat(128);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_unit_over_limit_rejected() {
        let mut r = make_resource();
        r.unit = "x".repeat(129);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource unit too long (max 128)");
    }

    // -- Resource location (max 4096) --

    #[test]
    fn create_resource_location_at_limit_accepted() {
        let mut r = make_resource();
        r.location = "x".repeat(4096);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_resource_location_over_limit_rejected() {
        let mut r = make_resource();
        r.location = "x".repeat(4097);
        let result = validate_create_resource(fake_create(), r);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Resource location too long (max 4096)"
        );
    }

    // -- Request location (max 4096) --

    #[test]
    fn create_request_location_at_limit_accepted() {
        let mut req = make_request();
        req.location = "x".repeat(4096);
        let result = validate_create_request(fake_create(), req);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_request_location_over_limit_rejected() {
        let mut req = make_request();
        req.location = "x".repeat(4097);
        let result = validate_create_request(fake_create(), req);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Request location too long (max 4096)");
    }

    // -- Update resource string limits --

    #[test]
    fn update_resource_id_over_limit_rejected() {
        let mut r = make_resource();
        r.id = "x".repeat(257);
        let result = validate_update_resource(r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource ID too long (max 256)");
    }

    #[test]
    fn update_resource_name_over_limit_rejected() {
        let mut r = make_resource();
        r.name = "x".repeat(257);
        let result = validate_update_resource(r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource name too long (max 256)");
    }

    #[test]
    fn update_resource_unit_over_limit_rejected() {
        let mut r = make_resource();
        r.unit = "x".repeat(129);
        let result = validate_update_resource(r);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Resource unit too long (max 128)");
    }

    #[test]
    fn update_resource_location_over_limit_rejected() {
        let mut r = make_resource();
        r.location = "x".repeat(4097);
        let result = validate_update_resource(r);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Resource location too long (max 4096)"
        );
    }

    // -- Update request location limit --

    #[test]
    fn update_request_location_over_limit_rejected() {
        let mut req = make_request();
        req.location = "x".repeat(4097);
        let result = validate_update_request(req);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Request location too long (max 4096)");
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
