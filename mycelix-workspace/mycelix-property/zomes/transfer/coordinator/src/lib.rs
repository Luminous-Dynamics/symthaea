// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Transfer Coordinator Zome
use hdk::prelude::*;
use transfer_integrity::*;

/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn initiate_transfer(input: InitiateTransferInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let transfer = Transfer {
        id: format!("transfer:{}:{}", input.property_id, now.as_micros()),
        property_id: input.property_id.clone(),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        transfer_type: input.transfer_type,
        price: input.price,
        currency: input.currency,
        conditions: input.conditions,
        status: TransferStatus::Initiated,
        initiated: now,
        completed: None,
    };

    let action_hash = create_entry(&EntryTypes::Transfer(transfer))?;
    create_link(
        anchor_hash(&input.property_id)?,
        action_hash.clone(),
        LinkTypes::PropertyToTransfers,
        (),
    )?;
    create_link(
        anchor_hash(&input.from_did)?,
        action_hash.clone(),
        LinkTypes::SellerToTransfers,
        (),
    )?;
    create_link(
        anchor_hash(&input.to_did)?,
        action_hash.clone(),
        LinkTypes::BuyerToTransfers,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateTransferInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TransferType,
    pub price: Option<f64>,
    pub currency: Option<String>,
    pub conditions: Vec<TransferCondition>,
}

#[hdk_extern]
pub fn create_escrow(input: CreateEscrowInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let escrow = Escrow {
        id: format!("escrow:{}:{}", input.transfer_id, now.as_micros()),
        transfer_id: input.transfer_id.clone(),
        escrow_agent_did: input.escrow_agent_did,
        amount: input.amount,
        currency: input.currency,
        funded: false,
        release_conditions: input.release_conditions,
        created: now,
        released: None,
    };

    let action_hash = create_entry(&EntryTypes::Escrow(escrow))?;
    create_link(
        anchor_hash(&input.transfer_id)?,
        action_hash.clone(),
        LinkTypes::TransferToEscrow,
        (),
    )?;

    // Update transfer status
    update_transfer_status(&input.transfer_id, TransferStatus::InEscrow)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateEscrowInput {
    pub transfer_id: String,
    pub escrow_agent_did: Option<String>,
    pub amount: f64,
    pub currency: String,
    pub release_conditions: Vec<String>,
}

fn update_transfer_status(transfer_id: &str, new_status: TransferStatus) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == transfer_id {
                let updated = Transfer {
                    status: new_status,
                    ..transfer
                };
                update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(updated),
                )?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

/// Complete a transfer and update ownership in the registry
///
/// This function:
/// 1. Verifies all conditions are satisfied
/// 2. Calls registry zome to transfer ownership (creates new TitleDeed)
/// 3. Updates the Transfer status to Completed
/// 4. Broadcasts ownership change event via bridge
#[hdk_extern]
pub fn complete_transfer(transfer_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == transfer_id {
                // Verify all conditions are satisfied
                for condition in &transfer.conditions {
                    if !condition.satisfied {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Not all conditions satisfied".into()
                        )));
                    }
                }

                // Determine transfer type string for registry
                let transfer_type_str = match &transfer.transfer_type {
                    TransferType::Sale => "Sale",
                    TransferType::Inheritance => "Inheritance",
                    TransferType::Gift => "Gift",
                    TransferType::CourtOrder => "CourtOrder",
                    TransferType::Exchange => "Exchange",
                    TransferType::Other => "Other",
                };

                // Call registry zome to transfer ownership and create new deed
                let registry_input = TransferOwnershipInput {
                    property_id: transfer.property_id.clone(),
                    from_did: transfer.from_did.clone(),
                    to_did: transfer.to_did.clone(),
                    transfer_type: transfer_type_str.to_string(),
                    transfer_id: Some(transfer.id.clone()),
                };

                // Cross-zome call to registry
                let response = call(
                    CallTargetCell::Local,
                    "registry",
                    "transfer_ownership".into(),
                    None,
                    registry_input,
                )?;
                let result: TransferOwnershipResult = match response {
                    ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
                    })?,
                    other => {
                        return Err(wasm_error!(WasmErrorInner::Guest(format!(
                            "Zome call failed: {:?}",
                            other
                        ))));
                    }
                };

                let now = sys_time()?;
                let completed = Transfer {
                    status: TransferStatus::Completed,
                    completed: Some(now),
                    ..transfer.clone()
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(completed),
                )?;

                // Broadcast ownership change via bridge zome
                let bridge_input = BroadcastOwnershipChangeInput {
                    property_id: transfer.property_id.clone(),
                    from_did: transfer.from_did.clone(),
                    to_did: transfer.to_did.clone(),
                    transfer_type: transfer_type_str.to_string(),
                    new_deed_id: result.new_deed_id,
                };

                // Best effort bridge notification (don't fail transfer if bridge fails)
                let _ = call(
                    CallTargetCell::Local,
                    "bridge",
                    "broadcast_ownership_change".into(),
                    None,
                    bridge_input,
                );

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

/// Input for cross-zome call to registry
#[derive(Serialize, Deserialize, Debug)]
struct TransferOwnershipInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: String,
    pub transfer_id: Option<String>,
}

/// Result from registry transfer_ownership call
#[derive(Serialize, Deserialize, Debug)]
struct TransferOwnershipResult {
    pub property_action_hash: ActionHash,
    pub new_deed_id: String,
    pub deed_action_hash: ActionHash,
    pub previous_deed_id: String,
    pub encumbrances_carried: u32,
}

/// Input for cross-zome call to bridge
#[derive(Serialize, Deserialize, Debug)]
struct BroadcastOwnershipChangeInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: String,
    pub new_deed_id: String,
}

#[hdk_extern]
pub fn satisfy_condition(input: SatisfyConditionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(mut transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == input.transfer_id {
                if let Some(condition) = transfer.conditions.get_mut(input.condition_index) {
                    condition.satisfied = true;
                    condition.verified_by = Some(input.verifier_did);
                }
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(transfer),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SatisfyConditionInput {
    pub transfer_id: String,
    pub condition_index: usize,
    pub verifier_did: String,
}

/// Get a specific transfer by ID
#[hdk_extern]
pub fn get_transfer(transfer_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == transfer_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all transfers where DID is seller
#[hdk_extern]
pub fn get_seller_transfers(did: String) -> ExternResult<Vec<Record>> {
    let mut transfers = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::SellerToTransfers)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            transfers.push(record);
        }
    }
    Ok(transfers)
}

/// Get all transfers where DID is buyer
#[hdk_extern]
pub fn get_buyer_transfers(did: String) -> ExternResult<Vec<Record>> {
    let mut transfers = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::BuyerToTransfers)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            transfers.push(record);
        }
    }
    Ok(transfers)
}

/// Get all transfers for a property
#[hdk_extern]
pub fn get_property_transfers(property_id: String) -> ExternResult<Vec<Record>> {
    let mut transfers = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&property_id)?, LinkTypes::PropertyToTransfers)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            transfers.push(record);
        }
    }
    Ok(transfers)
}

/// Cancel a transfer (only initiator can cancel, only before completion)
#[hdk_extern]
pub fn cancel_transfer(input: CancelTransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == input.transfer_id {
                // Only seller can cancel
                if transfer.from_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only seller can cancel transfer".into()
                    )));
                }

                // Cannot cancel completed transfers
                if transfer.status == TransferStatus::Completed {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot cancel completed transfer".into()
                    )));
                }

                let cancelled = Transfer {
                    status: TransferStatus::Cancelled,
                    ..transfer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(cancelled),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelTransferInput {
    pub transfer_id: String,
    pub requester_did: String,
}

/// Accept a transfer (buyer accepts)
#[hdk_extern]
pub fn accept_transfer(input: AcceptTransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == input.transfer_id {
                // Only buyer can accept
                if transfer.to_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only buyer can accept transfer".into()
                    )));
                }

                if transfer.status != TransferStatus::Initiated
                    && transfer.status != TransferStatus::AwaitingAcceptance
                {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Transfer not in acceptable state".into()
                    )));
                }

                let new_status = if transfer.conditions.is_empty() {
                    TransferStatus::ConditionsPending
                } else {
                    TransferStatus::ConditionsPending
                };

                let accepted = Transfer {
                    status: new_status,
                    ..transfer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(accepted),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcceptTransferInput {
    pub transfer_id: String,
    pub requester_did: String,
}

/// Fund an escrow
#[hdk_extern]
pub fn fund_escrow(escrow_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Escrow,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(escrow) = record.entry().to_app_option::<Escrow>().ok().flatten() {
            if escrow.id == escrow_id {
                if escrow.funded {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Escrow already funded".into()
                    )));
                }

                let funded = Escrow {
                    funded: true,
                    ..escrow
                };
                let action_hash =
                    update_entry(record.action_address().clone(), &EntryTypes::Escrow(funded))?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Escrow not found".into()
    )))
}

/// Release escrow to seller
#[hdk_extern]
pub fn release_escrow(escrow_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Escrow,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(escrow) = record.entry().to_app_option::<Escrow>().ok().flatten() {
            if escrow.id == escrow_id {
                if !escrow.funded {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Escrow not funded".into()
                    )));
                }
                if escrow.released.is_some() {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Escrow already released".into()
                    )));
                }

                let now = sys_time()?;
                let released = Escrow {
                    released: Some(now),
                    ..escrow
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Escrow(released),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Escrow not found".into()
    )))
}

/// Get escrow for a transfer
#[hdk_extern]
pub fn get_transfer_escrow(transfer_id: String) -> ExternResult<Option<Record>> {
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&transfer_id)?, LinkTypes::TransferToEscrow)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

/// Mark transfer as disputed
#[hdk_extern]
pub fn dispute_transfer(input: DisputeTransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == input.transfer_id {
                // Either party can dispute
                if transfer.from_did != input.requester_did
                    && transfer.to_did != input.requester_did
                {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only parties can dispute transfer".into()
                    )));
                }

                if transfer.status == TransferStatus::Completed
                    || transfer.status == TransferStatus::Cancelled
                {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot dispute completed or cancelled transfer".into()
                    )));
                }

                let disputed = Transfer {
                    status: TransferStatus::Disputed,
                    ..transfer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(disputed),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisputeTransferInput {
    pub transfer_id: String,
    pub requester_did: String,
}

/// Add a new condition to transfer (before acceptance)
#[hdk_extern]
pub fn add_condition(input: AddConditionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.id == input.transfer_id {
                if transfer.status != TransferStatus::Initiated {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only add conditions before acceptance".into()
                    )));
                }

                let new_condition = TransferCondition {
                    condition_type: input.condition_type,
                    description: input.description,
                    satisfied: false,
                    verified_by: None,
                };

                let mut conditions = transfer.conditions.clone();
                conditions.push(new_condition);

                let updated = Transfer {
                    conditions,
                    ..transfer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Transfer not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddConditionInput {
    pub transfer_id: String,
    pub condition_type: ConditionType,
    pub description: String,
}

/// Get transfers by status
#[hdk_extern]
pub fn get_transfers_by_status(status: TransferStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Transfer,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(transfer) = record.entry().to_app_option::<Transfer>().ok().flatten() {
            if transfer.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}
