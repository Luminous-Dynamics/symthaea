//! Property Transfer Coordinator Zome
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_proposal, requirement_for_voting, GovernanceEligibility,
    GovernanceRequirement,
};
use property_transfer_integrity::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn initiate_transfer(input: InitiateTransferInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "initiate_transfer")?;

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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = require_consciousness(&requirement_for_voting(), "complete_transfer")?;

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
                        ))))
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

                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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

                let new_status = TransferStatus::ConditionsPending;

                let accepted = Transfer {
                    status: new_status,
                    ..transfer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Transfer(accepted),
                )?;
                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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
                return get_latest_record(action_hash)?
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

// ============================================================================
// Cross-domain: Check water rights before completing property transfer
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckWaterRightsInput {
    /// The property being transferred.
    pub property_id: String,
    /// Watershed ActionHash to check — caller provides the relevant watershed.
    pub watershed_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WaterRightsCheckResult {
    pub has_water_rights: bool,
    pub water_rights_count: u32,
    pub warning: Option<String>,
    pub error: Option<String>,
}

/// Wire-compatible copy of water steward WaterRight for deserialization.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct LocalWaterRight {
    pub watershed_hash: ActionHash,
    pub holder: AgentPubKey,
    pub right_type: LocalRightType,
    pub volume_authorized_liters: u64,
    pub priority_date: Timestamp,
    pub conditions: Vec<String>,
    pub status: LocalRightStatus,
    pub transferable: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalRightType {
    Riparian,
    Appropriative,
    Prescriptive,
    Groundwater,
    Recycled,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalRightStatus {
    Active,
    Suspended,
    Revoked,
    Transferred,
    Expired,
}

/// Check if a property's watershed has associated water rights before
/// completing a transfer.
///
/// Cross-domain call: property-transfer queries water_steward via
/// `call(CallTargetCell::Local, ...)` to check for water rights in
/// the watershed. If water rights exist, the buyer should be informed
/// that rights may need separate transfer.
#[hdk_extern]
pub fn check_water_rights_before_transfer(
    input: CheckWaterRightsInput,
) -> ExternResult<WaterRightsCheckResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("water_steward"),
        FunctionName::from("get_watershed_rights"),
        None,
        input.watershed_hash,
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;

            let count = records.len() as u32;
            let has_rights = count > 0;

            Ok(WaterRightsCheckResult {
                has_water_rights: has_rights,
                water_rights_count: count,
                warning: if has_rights {
                    Some(format!(
                        "Property '{}' has {} water right(s) in this watershed. \
                         Water rights may need separate transfer.",
                        input.property_id, count
                    ))
                } else {
                    None
                },
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(WaterRightsCheckResult {
            has_water_rights: false,
            water_rights_count: 0,
            warning: None,
            error: Some(format!("Network error querying water rights: {}", err)),
        }),
        _ => Ok(WaterRightsCheckResult {
            has_water_rights: false,
            water_rights_count: 0,
            warning: None,
            error: Some("Failed to query water steward zome".into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initiate_transfer_input_serde_roundtrip() {
        let input = InitiateTransferInput {
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6Mk001".to_string(),
            to_did: "did:key:z6Mk002".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(250_000.0),
            currency: Some("USD".to_string()),
            conditions: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: InitiateTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-001");
        assert_eq!(decoded.price, Some(250_000.0));
    }

    #[test]
    fn create_escrow_input_serde_roundtrip() {
        let input = CreateEscrowInput {
            transfer_id: "transfer-001".to_string(),
            escrow_agent_did: Some("did:key:z6MkAgent".to_string()),
            amount: 250_000.0,
            currency: "USD".to_string(),
            release_conditions: vec!["Inspection passed".to_string(), "Title cleared".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateEscrowInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.amount - 250_000.0).abs() < f64::EPSILON);
        assert_eq!(decoded.release_conditions.len(), 2);
    }

    #[test]
    fn satisfy_condition_input_serde_roundtrip() {
        let input = SatisfyConditionInput {
            transfer_id: "transfer-001".to_string(),
            condition_index: 0,
            verifier_did: "did:key:z6MkInspector".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SatisfyConditionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.condition_index, 0);
        assert_eq!(decoded.verifier_did, "did:key:z6MkInspector");
    }

    #[test]
    fn cancel_transfer_input_serde_roundtrip() {
        let input = CancelTransferInput {
            transfer_id: "transfer-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CancelTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.transfer_id, "transfer-001");
    }

    #[test]
    fn accept_transfer_input_serde_roundtrip() {
        let input = AcceptTransferInput {
            transfer_id: "transfer-001".to_string(),
            requester_did: "did:key:z6Mk002".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AcceptTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.requester_did, "did:key:z6Mk002");
    }

    #[test]
    fn dispute_transfer_input_serde_roundtrip() {
        let input = DisputeTransferInput {
            transfer_id: "transfer-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DisputeTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.transfer_id, "transfer-001");
    }

    #[test]
    fn add_condition_input_serde_roundtrip() {
        let input = AddConditionInput {
            transfer_id: "transfer-001".to_string(),
            condition_type: ConditionType::InspectionComplete,
            description: "Professional home inspection".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddConditionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description, "Professional home inspection");
    }

    #[test]
    fn check_water_rights_input_serde_roundtrip() {
        let input = CheckWaterRightsInput {
            property_id: "prop-001".to_string(),
            watershed_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CheckWaterRightsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-001");
    }

    #[test]
    fn water_rights_check_result_serde_roundtrip() {
        let result = WaterRightsCheckResult {
            has_water_rights: true,
            water_rights_count: 2,
            warning: Some("Property has 2 water right(s)".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: WaterRightsCheckResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.has_water_rights);
        assert_eq!(decoded.water_rights_count, 2);
        assert!(decoded.warning.is_some());
        assert!(decoded.error.is_none());
    }

    #[test]
    fn transfer_type_all_variants_serialize() {
        let variants = vec![
            TransferType::Sale,
            TransferType::Gift,
            TransferType::Inheritance,
            TransferType::CourtOrder,
            TransferType::Exchange,
            TransferType::Other,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: TransferType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn transfer_status_all_variants_serialize() {
        let statuses = vec![
            TransferStatus::Initiated,
            TransferStatus::AwaitingAcceptance,
            TransferStatus::InEscrow,
            TransferStatus::ConditionsPending,
            TransferStatus::Completed,
            TransferStatus::Cancelled,
            TransferStatus::Disputed,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: TransferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn condition_type_all_variants_serialize() {
        let types = vec![
            ConditionType::PaymentReceived,
            ConditionType::InspectionComplete,
            ConditionType::TitleClear,
            ConditionType::DocumentsSigned,
            ConditionType::TaxesPaid,
            ConditionType::Custom("Zoning approval".to_string()),
        ];
        for ct in types {
            let json = serde_json::to_string(&ct).unwrap();
            let decoded: ConditionType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, ct);
        }
    }

    #[test]
    fn initiate_transfer_with_conditions() {
        let input = InitiateTransferInput {
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6MkSeller".to_string(),
            to_did: "did:key:z6MkBuyer".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(500_000.0),
            currency: Some("USD".to_string()),
            conditions: vec![
                TransferCondition {
                    condition_type: ConditionType::PaymentReceived,
                    description: "Full payment".to_string(),
                    satisfied: false,
                    verified_by: None,
                },
                TransferCondition {
                    condition_type: ConditionType::InspectionComplete,
                    description: "Home inspection".to_string(),
                    satisfied: true,
                    verified_by: Some("did:key:z6MkInspector".to_string()),
                },
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: InitiateTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.conditions.len(), 2);
        assert!(!decoded.conditions[0].satisfied);
        assert!(decoded.conditions[1].satisfied);
    }

    #[test]
    fn water_rights_result_no_rights() {
        let result = WaterRightsCheckResult {
            has_water_rights: false,
            water_rights_count: 0,
            warning: None,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: WaterRightsCheckResult = serde_json::from_str(&json).unwrap();
        assert!(!decoded.has_water_rights);
        assert_eq!(decoded.water_rights_count, 0);
    }

    #[test]
    fn water_rights_result_with_error() {
        let result = WaterRightsCheckResult {
            has_water_rights: false,
            water_rights_count: 0,
            warning: None,
            error: Some("Network error querying water rights".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: WaterRightsCheckResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.error.is_some());
    }

    // ── Transfer entry full serde roundtrip ─────────────────────────

    #[test]
    fn transfer_entry_full_serde_roundtrip() {
        let transfer = Transfer {
            id: "transfer:prop-001:1000".to_string(),
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6MkSeller".to_string(),
            to_did: "did:key:z6MkBuyer".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(350_000.0),
            currency: Some("EUR".to_string()),
            conditions: vec![
                TransferCondition {
                    condition_type: ConditionType::PaymentReceived,
                    description: "Bank transfer confirmed".to_string(),
                    satisfied: false,
                    verified_by: None,
                },
                TransferCondition {
                    condition_type: ConditionType::Custom("Zoning approval".to_string()),
                    description: "City zoning board approval".to_string(),
                    satisfied: true,
                    verified_by: Some("did:key:z6MkZoningOfficer".to_string()),
                },
            ],
            status: TransferStatus::ConditionsPending,
            initiated: Timestamp::from_micros(1_000_000),
            completed: None,
        };
        let json = serde_json::to_string(&transfer).unwrap();
        let decoded: Transfer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, transfer.id);
        assert_eq!(decoded.from_did, "did:key:z6MkSeller");
        assert_eq!(decoded.to_did, "did:key:z6MkBuyer");
        assert_eq!(decoded.conditions.len(), 2);
        assert!(!decoded.conditions[0].satisfied);
        assert!(decoded.conditions[1].satisfied);
        assert_eq!(decoded.status, TransferStatus::ConditionsPending);
        assert!(decoded.completed.is_none());
    }

    // ── Escrow entry full serde roundtrip ───────────────────────────

    #[test]
    fn escrow_entry_full_serde_roundtrip() {
        let escrow = Escrow {
            id: "escrow:transfer-001:2000".to_string(),
            transfer_id: "transfer-001".to_string(),
            escrow_agent_did: Some("did:key:z6MkEscrowAgent".to_string()),
            amount: 250_000.50,
            currency: "USD".to_string(),
            funded: true,
            release_conditions: vec![
                "Title search clear".to_string(),
                "Inspection passed".to_string(),
                "Loan approved".to_string(),
            ],
            created: Timestamp::from_micros(2_000_000),
            released: Some(Timestamp::from_micros(3_000_000)),
        };
        let json = serde_json::to_string(&escrow).unwrap();
        let decoded: Escrow = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, escrow.id);
        assert!(decoded.funded);
        assert!((decoded.amount - 250_000.50).abs() < f64::EPSILON);
        assert_eq!(decoded.release_conditions.len(), 3);
        assert!(decoded.released.is_some());
    }

    // ── TransferCondition serde with all condition types ────────────

    #[test]
    fn transfer_condition_all_types_serde() {
        let types_and_descs = vec![
            (ConditionType::PaymentReceived, "Full payment"),
            (ConditionType::InspectionComplete, "Home inspection"),
            (ConditionType::TitleClear, "Title search"),
            (ConditionType::DocumentsSigned, "All documents"),
            (ConditionType::TaxesPaid, "Property taxes current"),
            (
                ConditionType::Custom("Environmental review".to_string()),
                "Environmental review",
            ),
        ];
        for (ct, desc) in types_and_descs {
            let condition = TransferCondition {
                condition_type: ct.clone(),
                description: desc.to_string(),
                satisfied: false,
                verified_by: None,
            };
            let json = serde_json::to_string(&condition).unwrap();
            let decoded: TransferCondition = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.condition_type, ct);
            assert_eq!(decoded.description, desc);
            assert!(!decoded.satisfied);
        }
    }

    // ── Transfer status lifecycle transitions ───────────────────────

    #[test]
    fn transfer_status_initiated_to_conditions_pending_serde() {
        // Simulate the status transition that accept_transfer produces
        let transfer_before = Transfer {
            id: "transfer:lifecycle:1".to_string(),
            property_id: "prop-lc-01".to_string(),
            from_did: "did:key:z6MkSeller".to_string(),
            to_did: "did:key:z6MkBuyer".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(100_000.0),
            currency: Some("USD".to_string()),
            conditions: vec![],
            status: TransferStatus::Initiated,
            initiated: Timestamp::from_micros(1000),
            completed: None,
        };
        // Transition to ConditionsPending (as accept_transfer does)
        let transfer_after = Transfer {
            status: TransferStatus::ConditionsPending,
            ..transfer_before.clone()
        };
        let json = serde_json::to_string(&transfer_after).unwrap();
        let decoded: Transfer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, TransferStatus::ConditionsPending);
        // Verify all other fields unchanged
        assert_eq!(decoded.id, transfer_before.id);
        assert_eq!(decoded.from_did, transfer_before.from_did);
    }

    #[test]
    fn transfer_status_completed_with_timestamp_serde() {
        let completed_ts = Timestamp::from_micros(5_000_000);
        let transfer = Transfer {
            id: "transfer:complete:1".to_string(),
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6MkSeller".to_string(),
            to_did: "did:key:z6MkBuyer".to_string(),
            transfer_type: TransferType::Inheritance,
            price: None,
            currency: None,
            conditions: vec![],
            status: TransferStatus::Completed,
            initiated: Timestamp::from_micros(1_000_000),
            completed: Some(completed_ts),
        };
        let json = serde_json::to_string(&transfer).unwrap();
        let decoded: Transfer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, TransferStatus::Completed);
        assert_eq!(decoded.completed, Some(completed_ts));
        assert!(decoded.price.is_none());
        assert!(decoded.currency.is_none());
    }

    #[test]
    fn transfer_cancelled_preserves_all_fields_serde() {
        let transfer = Transfer {
            id: "transfer:cancel:1".to_string(),
            property_id: "prop-002".to_string(),
            from_did: "did:key:z6MkSeller".to_string(),
            to_did: "did:key:z6MkBuyer".to_string(),
            transfer_type: TransferType::Exchange,
            price: Some(1.0),
            currency: Some("BTC".to_string()),
            conditions: vec![TransferCondition {
                condition_type: ConditionType::PaymentReceived,
                description: "Crypto transfer".to_string(),
                satisfied: false,
                verified_by: None,
            }],
            status: TransferStatus::Cancelled,
            initiated: Timestamp::from_micros(1000),
            completed: None,
        };
        let json = serde_json::to_string(&transfer).unwrap();
        let decoded: Transfer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, TransferStatus::Cancelled);
        assert_eq!(decoded.conditions.len(), 1);
        assert!(!decoded.conditions[0].satisfied);
    }

    // ── Cross-domain internal type serde roundtrips ──────────────────

    #[test]
    fn transfer_ownership_input_serde_roundtrip() {
        let input = TransferOwnershipInput {
            property_id: "prop-xdomain-01".to_string(),
            from_did: "did:key:z6MkOldOwner".to_string(),
            to_did: "did:key:z6MkNewOwner".to_string(),
            transfer_type: "Sale".to_string(),
            transfer_id: Some("transfer:xdomain:1".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-xdomain-01");
        assert_eq!(decoded.transfer_type, "Sale");
        assert!(decoded.transfer_id.is_some());
    }

    #[test]
    fn transfer_ownership_result_serde_roundtrip() {
        let result = TransferOwnershipResult {
            property_action_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
            new_deed_id: "deed:new:001".to_string(),
            deed_action_hash: ActionHash::from_raw_36(vec![0xbb; 36]),
            previous_deed_id: "deed:old:001".to_string(),
            encumbrances_carried: 3,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: TransferOwnershipResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_deed_id, "deed:new:001");
        assert_eq!(decoded.previous_deed_id, "deed:old:001");
        assert_eq!(decoded.encumbrances_carried, 3);
    }

    #[test]
    fn broadcast_ownership_change_input_serde_roundtrip() {
        let input = BroadcastOwnershipChangeInput {
            property_id: "prop-broadcast-01".to_string(),
            from_did: "did:key:z6MkFrom".to_string(),
            to_did: "did:key:z6MkTo".to_string(),
            transfer_type: "Gift".to_string(),
            new_deed_id: "deed:gift:001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: BroadcastOwnershipChangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-broadcast-01");
        assert_eq!(decoded.transfer_type, "Gift");
        assert_eq!(decoded.new_deed_id, "deed:gift:001");
    }

    // ── Edge cases: gift transfer without price ─────────────────────

    #[test]
    fn initiate_gift_transfer_no_price_no_currency_serde() {
        let input = InitiateTransferInput {
            property_id: "prop-gift-01".to_string(),
            from_did: "did:key:z6MkGifter".to_string(),
            to_did: "did:key:z6MkRecipient".to_string(),
            transfer_type: TransferType::Gift,
            price: None,
            currency: None,
            conditions: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: InitiateTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.transfer_type, TransferType::Gift);
        assert!(decoded.price.is_none());
        assert!(decoded.currency.is_none());
        assert!(decoded.conditions.is_empty());
    }

    // ── Edge case: escrow without agent ──────────────────────────────

    #[test]
    fn create_escrow_input_no_agent_serde() {
        let input = CreateEscrowInput {
            transfer_id: "transfer-no-agent".to_string(),
            escrow_agent_did: None,
            amount: 50_000.0,
            currency: "GBP".to_string(),
            release_conditions: vec!["All conditions met".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateEscrowInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.escrow_agent_did.is_none());
        assert_eq!(decoded.currency, "GBP");
    }

    // ── Edge case: satisfy condition at large index ──────────────────

    #[test]
    fn satisfy_condition_large_index_serde() {
        let input = SatisfyConditionInput {
            transfer_id: "transfer-many-conds".to_string(),
            condition_index: usize::MAX,
            verifier_did: "did:key:z6MkVerifier".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SatisfyConditionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.condition_index, usize::MAX);
    }
}
