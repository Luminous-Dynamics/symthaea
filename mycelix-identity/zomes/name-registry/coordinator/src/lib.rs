use hdk::prelude::*;
use name_registry_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_voting,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("identity_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Register a mesh name (Participant+).
#[hdk_extern]
pub fn register_name(entry: MeshNameEntry) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "register_name")?;

    let action_hash = create_entry(&EntryTypes::MeshNameEntry(entry.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Create anchor for name resolution
    let anchor_str = format!("mesh_name/{}", entry.segments.join("/"));
    let name_anchor = ensure_anchor(&anchor_str)?;
    create_link(name_anchor, action_hash.clone(), LinkTypes::NamePath, ())?;

    // Link from agent
    create_link(agent, action_hash.clone(), LinkTypes::AgentToNames, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Resolve a mesh name.
#[hdk_extern]
pub fn resolve_name(canonical: String) -> ExternResult<Option<MeshNameEntry>> {
    let segments: Vec<&str> = canonical
        .strip_prefix("mycelix://")
        .unwrap_or(&canonical)
        .trim_matches('/')
        .split('/')
        .collect();
    let anchor_str = format!("mesh_name/{}", segments.join("/"));

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_str)?, LinkTypes::NamePath)?,
        GetStrategy::default(),
    )?;

    // Return the most recent registration
    for link in links.into_iter().rev() {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(entry) = record.entry().to_app_option::<MeshNameEntry>().ok().flatten() {
                    return Ok(Some(entry));
                }
            }
        }
    }
    Ok(None)
}

/// Transfer name ownership (owner only, Citizen+).
#[hdk_extern]
pub fn transfer_name(transfer: NameTransfer) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "transfer_name")?;

    // Verify caller owns the name
    let name_record = get(transfer.name_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Name not found".into())))?;
    let owner = name_record.action().author().clone();
    let caller = agent_info()?.agent_initial_pubkey;
    if owner != caller {
        return Err(wasm_error!(WasmErrorInner::Guest("Only owner can transfer".into())));
    }

    let action_hash = create_entry(&EntryTypes::NameTransfer(transfer.clone()))?;
    create_link(transfer.name_hash, action_hash.clone(), LinkTypes::NameToTransfers, ())?;

    // Link new owner
    create_link(transfer.new_owner, action_hash.clone(), LinkTypes::AgentToNames, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Renew a name (extend expiry by 1 year). Owner only.
#[hdk_extern]
pub fn renew_name(name_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "renew_name")?;

    let record = get(name_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Name not found".into())))?;
    let owner = record.action().author().clone();
    let caller = agent_info()?.agent_initial_pubkey;
    if owner != caller {
        return Err(wasm_error!(WasmErrorInner::Guest("Only owner can renew".into())));
    }

    if let Some(mut entry) = record.entry().to_app_option::<MeshNameEntry>().ok().flatten() {
        let one_year_us = 365 * 24 * 3600 * 1_000_000u64;
        entry.expires_at = entry.expires_at.saturating_add(one_year_us);
        let new_hash = update_entry(name_hash, &entry)?;
        let updated = get(new_hash, GetOptions::default())?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Updated record not found".into())))?;
        Ok(updated)
    } else {
        Err(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))
    }
}
