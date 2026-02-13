//! Food Preservation Coordinator Zome
//! Business logic for preservation batches, methods, and storage management.

use food_preservation_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// BATCH MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn start_batch(batch: PreservationBatch) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let action_hash = create_entry(&EntryTypes::PreservationBatch(batch.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_batches".to_string())))?;
    create_link(anchor_hash("all_batches")?, action_hash.clone(), LinkTypes::AllBatches, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToBatch, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created batch".into())))
}

#[hdk_extern]
pub fn complete_batch(batch_hash: ActionHash) -> ExternResult<Record> {
    let record = get(batch_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let mut batch: PreservationBatch = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid batch entry".into())))?;

    batch.status = BatchStatus::Completed;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::PreservationBatch(batch))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated batch".into())))
}

#[hdk_extern]
pub fn get_batch(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_agent_batches(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToBatch)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// METHOD REGISTRY
// ============================================================================

#[hdk_extern]
pub fn register_method(method: PreservationMethod) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::PreservationMethod(method))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_methods".to_string())))?;
    create_link(anchor_hash("all_methods")?, action_hash.clone(), LinkTypes::AllMethods, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created method".into())))
}

#[hdk_extern]
pub fn get_all_methods(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_methods")?, LinkTypes::AllMethods)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// STORAGE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn register_storage(storage: StorageUnit) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::StorageUnit(storage))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_storage".to_string())))?;
    create_link(anchor_hash("all_storage")?, action_hash.clone(), LinkTypes::AllStorage, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created storage".into())))
}

#[hdk_extern]
pub fn get_storage_inventory(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_storage")?, LinkTypes::AllStorage)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
