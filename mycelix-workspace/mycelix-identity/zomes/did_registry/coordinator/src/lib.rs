// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! DID Registry Coordinator Zome
//! Business logic for DID:mycelix operations
//!
//! Updated to use HDK 0.6 patterns

use did_registry_integrity::*;
use hdk::prelude::*;

/// Create a new DID document for the calling agent
#[hdk_extern]
pub fn create_did() -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Generate DID identifier
    let did_id = format!("did:mycelix:{}", agent_pub_key);

    // Create default verification method
    let verification_method = VerificationMethod {
        id: format!("{}#keys-1", did_id),
        type_: "Ed25519VerificationKey2020".to_string(),
        controller: did_id.clone(),
        public_key_multibase: format!("z{}", agent_pub_key),
    };

    let now = sys_time()?;
    let did_doc = DidDocument {
        id: did_id.clone(),
        controller: agent_pub_key.clone(),
        verification_method: vec![verification_method.clone()],
        authentication: vec![format!("{}#keys-1", did_id)],
        service: vec![],
        created: now,
        updated: now,
        version: 1,
    };

    let action_hash = create_entry(&EntryTypes::DidDocument(did_doc.clone()))?;

    // Link agent to DID
    create_link(
        agent_pub_key.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDid,
        (),
    )?;

    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find created DID".into())
    ))?;

    Ok(record)
}

/// Get DID document for an agent
#[hdk_extern]
pub fn get_did_document(agent_pub_key: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::AgentToDid)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the latest DID document
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        get(action_hash, GetOptions::default())
    } else {
        Ok(None)
    }
}

/// Resolve a DID to its document
#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<Option<Record>> {
    // Parse DID to extract agent pub key
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did
        .strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into())))?;

    get_did_document(agent_pub_key)
}

/// Update DID document (add service endpoints, rotate keys, etc.)
#[hdk_extern]
pub fn update_did_document(input: UpdateDidInput) -> ExternResult<Record> {
    // Input validation
    if let Some(ref methods) = input.verification_method {
        if methods.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Verification methods must not exceed 100 entries".into()
            )));
        }
        for method in methods {
            if method.id.is_empty() || method.id.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Verification method ID must be 1-256 characters".into()
                )));
            }
            if method.type_.is_empty() || method.type_.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Verification method type must be 1-256 characters".into()
                )));
            }
            if method.controller.is_empty() || method.controller.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Verification method controller must be 1-256 characters".into()
                )));
            }
            if method.public_key_multibase.is_empty() || method.public_key_multibase.len() > 4096 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Public key multibase must be 1-4096 characters".into()
                )));
            }
        }
    }
    if let Some(ref auth) = input.authentication {
        if auth.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Authentication entries must not exceed 100".into()
            )));
        }
        for a in auth {
            if a.is_empty() || a.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Authentication entry must be 1-256 characters".into()
                )));
            }
        }
    }
    if let Some(ref services) = input.service {
        if services.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Service endpoints must not exceed 100 entries".into()
            )));
        }
        for svc in services {
            if svc.id.is_empty() || svc.id.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Service endpoint ID must be 1-256 characters".into()
                )));
            }
            if svc.type_.is_empty() || svc.type_.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Service endpoint type must be 1-256 characters".into()
                )));
            }
            if svc.service_endpoint.is_empty() || svc.service_endpoint.len() > 4096 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Service endpoint URL must be 1-4096 characters".into()
                )));
            }
        }
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Get current DID document
    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID entry".into()
        )))?;

    let now = sys_time()?;

    // Build updated document
    let updated_did = DidDocument {
        id: current_did.id.clone(),
        controller: current_did.controller.clone(),
        verification_method: input
            .verification_method
            .unwrap_or(current_did.verification_method),
        authentication: input.authentication.unwrap_or(current_did.authentication),
        service: input.service.unwrap_or(current_did.service),
        created: current_did.created,
        updated: now,
        version: current_did.version + 1,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::DidDocument(updated_did),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated DID".into()
    )))
}

/// Input for updating a DID document
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDidInput {
    pub verification_method: Option<Vec<VerificationMethod>>,
    pub authentication: Option<Vec<String>>,
    pub service: Option<Vec<ServiceEndpoint>>,
}

/// Deactivate a DID
#[hdk_extern]
pub fn deactivate_did(reason: String) -> ExternResult<Record> {
    // Input validation
    if reason.is_empty() || reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-4096 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Get current DID
    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID entry".into()
        )))?;

    // Check if already deactivated
    if is_did_active(current_did.id.clone())? == false {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID is already deactivated".into()
        )));
    }

    let now = sys_time()?;

    let deactivation = DidDeactivation {
        did: current_did.id.clone(),
        reason,
        deactivated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::DidDeactivation(deactivation))?;

    // Create link from DID to deactivation record for efficient lookup
    // Use the agent pub key as the base since that's how DIDs are looked up
    create_link(
        agent_pub_key.clone(),
        action_hash.clone(),
        LinkTypes::DidToDeactivation,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find deactivation record".into()
    )))
}

/// Check if a DID is active (not deactivated)
#[hdk_extern]
pub fn is_did_active(did: String) -> ExternResult<bool> {
    // First check if DID exists
    let record = resolve_did(did.clone())?;
    if record.is_none() {
        return Ok(false);
    }

    // Parse DID to extract agent pub key for link lookup
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did
        .strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into())))?;

    // Check for deactivation links - if any exist, DID is deactivated
    let deactivation_links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::DidToDeactivation)?,
        GetStrategy::default(),
    )?;

    // If there are any deactivation links, the DID is not active
    if !deactivation_links.is_empty() {
        return Ok(false);
    }

    // No deactivation links found, DID is active
    Ok(true)
}

/// Get the deactivation record for a DID (if deactivated)
#[hdk_extern]
pub fn get_did_deactivation(did: String) -> ExternResult<Option<DidDeactivation>> {
    // Parse DID to extract agent pub key
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did
        .strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into())))?;

    // Get deactivation links
    let deactivation_links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::DidToDeactivation)?,
        GetStrategy::default(),
    )?;

    if deactivation_links.is_empty() {
        return Ok(None);
    }

    // Get the most recent deactivation record
    let latest_link = deactivation_links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let deactivation: DidDeactivation = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid deactivation entry".into()
                )))?;
            return Ok(Some(deactivation));
        }
    }

    Ok(None)
}

/// Add a service endpoint to the DID
#[hdk_extern]
pub fn add_service_endpoint(service: ServiceEndpoint) -> ExternResult<Record> {
    // Input validation
    if service.id.is_empty() || service.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service endpoint ID must be 1-256 characters".into()
        )));
    }
    if service.type_.is_empty() || service.type_.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service endpoint type must be 1-256 characters".into()
        )));
    }
    if service.service_endpoint.is_empty() || service.service_endpoint.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service endpoint URL must be 1-4096 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID entry".into()
        )))?;

    let mut services = current_did.service.clone();
    services.push(service);

    update_did_document(UpdateDidInput {
        verification_method: None,
        authentication: None,
        service: Some(services),
    })
}

/// Remove a service endpoint from the DID
#[hdk_extern]
pub fn remove_service_endpoint(service_id: String) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID entry".into()
        )))?;

    let services: Vec<ServiceEndpoint> = current_did
        .service
        .into_iter()
        .filter(|s| s.id != service_id)
        .collect();

    update_did_document(UpdateDidInput {
        verification_method: None,
        authentication: None,
        service: Some(services),
    })
}

/// Add a verification method to the DID
#[hdk_extern]
pub fn add_verification_method(method: VerificationMethod) -> ExternResult<Record> {
    // Input validation
    if method.id.is_empty() || method.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verification method ID must be 1-256 characters".into()
        )));
    }
    if method.type_.is_empty() || method.type_.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verification method type must be 1-256 characters".into()
        )));
    }
    if method.controller.is_empty() || method.controller.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verification method controller must be 1-256 characters".into()
        )));
    }
    if method.public_key_multibase.is_empty() || method.public_key_multibase.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Public key multibase must be 1-4096 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID entry".into()
        )))?;

    let mut methods = current_did.verification_method.clone();
    methods.push(method);

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: None,
        service: None,
    })
}

/// Get my DID (convenience function)
#[hdk_extern]
pub fn get_my_did(_: ()) -> ExternResult<Option<Record>> {
    let agent_info = agent_info()?;
    get_did_document(agent_info.agent_initial_pubkey)
}
