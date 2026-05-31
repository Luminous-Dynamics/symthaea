// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Steward Coordinator Zome
//! Business logic for watershed governance, water rights, transfers, and disputes

use hdk::prelude::*;
use steward_integrity::*;

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
// WATERSHED MANAGEMENT
// ============================================================================

/// Define a new watershed
#[hdk_extern]
pub fn define_watershed(watershed: Watershed) -> ExternResult<Record> {
    if watershed.id.is_empty() || watershed.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Watershed ID must be 1-256 characters".into()
        )));
    }
    if watershed.name.is_empty() || watershed.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Watershed name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Watershed(watershed.clone()))?;

    // Link to all watersheds
    create_entry(&EntryTypes::Anchor(Anchor("all_watersheds".to_string())))?;
    create_link(
        anchor_hash("all_watersheds")?,
        action_hash.clone(),
        LinkTypes::AllWatersheds,
        (),
    )?;

    // Link stewardship type to watershed
    let type_anchor = format!("stewardship:{:?}", watershed.stewardship_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::StewardshipTypeToWatershed,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created watershed".into()
    )))
}

/// Get all registered watersheds
#[hdk_extern]
pub fn get_all_watersheds(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_watersheds")?, LinkTypes::AllWatersheds)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// WATER RIGHTS
// ============================================================================

/// Register a new water right within a watershed
#[hdk_extern]
pub fn register_water_right(right: WaterRight) -> ExternResult<Record> {
    // Verify watershed exists
    let _ws_record = get(right.watershed_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Watershed not found".into())),
    )?;

    let action_hash = create_entry(&EntryTypes::WaterRight(right.clone()))?;

    // Link watershed to right
    create_link(
        right.watershed_hash.clone(),
        action_hash.clone(),
        LinkTypes::WatershedToRight,
        (),
    )?;

    // Link holder to right
    create_link(
        right.holder.clone(),
        action_hash.clone(),
        LinkTypes::HolderToRight,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created water right".into()
    )))
}

/// Get all water rights for a watershed
#[hdk_extern]
pub fn get_watershed_rights(watershed_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(watershed_hash, LinkTypes::WatershedToRight)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all water rights held by the calling agent
#[hdk_extern]
pub fn get_my_rights(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let links = get_links(
        LinkQuery::try_new(agent_info.agent_initial_pubkey, LinkTypes::HolderToRight)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// RIGHT TRANSFERS
// ============================================================================

/// Transfer a water right to another holder
#[hdk_extern]
pub fn transfer_right(input: TransferRightInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;

    // Fetch the water right
    let right_record = get(input.right_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Water right not found".into())
    ))?;
    let right: WaterRight = right_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid water right entry".into()
        )))?;

    // Only the holder can transfer
    if right.holder != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the right holder can initiate a transfer".into()
        )));
    }

    if !right.transferable {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This water right is not transferable".into()
        )));
    }

    if right.status != RightStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active rights can be transferred".into()
        )));
    }

    if input.volume_liters > right.volume_authorized_liters {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transfer volume exceeds authorized volume".into()
        )));
    }

    let now = sys_time()?;

    let transfer = RightTransfer {
        right_hash: input.right_hash.clone(),
        from_holder: agent_info.agent_initial_pubkey.clone(),
        to_holder: input.to_holder.clone(),
        volume_liters: input.volume_liters,
        transfer_type: input.transfer_type,
        approved_by: input.approved_by,
        transferred_at: now,
    };

    let transfer_hash = create_entry(&EntryTypes::RightTransfer(transfer))?;

    // Link right to transfer
    create_link(
        input.right_hash.clone(),
        transfer_hash.clone(),
        LinkTypes::RightToTransfer,
        (),
    )?;

    // Update the original right status if full transfer
    if input.volume_liters == right.volume_authorized_liters {
        let updated_right = WaterRight {
            watershed_hash: right.watershed_hash,
            holder: right.holder,
            right_type: right.right_type,
            volume_authorized_liters: right.volume_authorized_liters,
            priority_date: right.priority_date,
            conditions: right.conditions,
            status: RightStatus::Transferred,
            transferable: right.transferable,
        };
        update_entry(
            right_record.action_address().clone(),
            &EntryTypes::WaterRight(updated_right),
        )?;
    }

    get(transfer_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created transfer".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferRightInput {
    pub right_hash: ActionHash,
    pub to_holder: AgentPubKey,
    pub volume_liters: u64,
    pub transfer_type: TransferType,
    pub approved_by: Option<AgentPubKey>,
}

// ============================================================================
// DISPUTE RESOLUTION
// ============================================================================

/// File a water dispute within a watershed
#[hdk_extern]
pub fn file_dispute(dispute: WaterDispute) -> ExternResult<Record> {
    if dispute.description.is_empty() || dispute.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 characters".into()
        )));
    }

    // Verify watershed exists
    let _ws = get(dispute.watershed_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Watershed not found".into())
    ))?;

    let action_hash = create_entry(&EntryTypes::WaterDispute(dispute.clone()))?;

    // Link watershed to dispute
    create_link(
        dispute.watershed_hash.clone(),
        action_hash.clone(),
        LinkTypes::WatershedToDispute,
        (),
    )?;

    // Link complainant to dispute
    create_link(
        dispute.complainant.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDispute,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created dispute".into()
    )))
}

/// Resolve a water dispute
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    let record = get(input.dispute_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dispute not found".into())
    ))?;
    let mut dispute: WaterDispute = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid dispute entry".into()
        )))?;

    if dispute.status == DisputeStatus::Resolved || dispute.status == DisputeStatus::Dismissed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute is already resolved or dismissed".into()
        )));
    }

    dispute.status = input.new_status;
    dispute.resolution = Some(input.resolution_text);

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::WaterDispute(dispute),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated dispute".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_hash: ActionHash,
    pub new_status: DisputeStatus,
    pub resolution_text: String,
}

/// Get all disputes for a watershed
#[hdk_extern]
pub fn get_watershed_disputes(watershed_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(watershed_hash, LinkTypes::WatershedToDispute)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get disputes filed by the calling agent
#[hdk_extern]
pub fn get_my_disputes(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let links = get_links(
        LinkQuery::try_new(agent_info.agent_initial_pubkey, LinkTypes::AgentToDispute)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
