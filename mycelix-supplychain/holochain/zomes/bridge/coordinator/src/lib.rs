//! Bridge Coordinator Zome
//!
//! Provides CRUD operations for cross-hApp bridge records.

use hdk::prelude::*;
use bridge_integrity::*;

/// Input for creating a bridge record
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateBridgeInput {
    pub source_happ: String,
    pub target_happ: String,
    pub record_hash: String,
    pub bridge_type: String,
}

/// Create a new bridge record
#[hdk_extern]
pub fn create_bridge(input: CreateBridgeInput) -> ExternResult<Record> {
    // Input validation
    if input.source_happ.is_empty() || input.source_happ.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source hApp must be 1-100 characters".to_string()
        )));
    }
    if input.target_happ.is_empty() || input.target_happ.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Target hApp must be 1-100 characters".to_string()
        )));
    }
    if input.record_hash.is_empty() || input.record_hash.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Record hash must be 1-200 characters".to_string()
        )));
    }
    if input.bridge_type.is_empty() || input.bridge_type.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bridge type must be 1-50 characters".to_string()
        )));
    }

    let bridge = BridgeRecord {
        source_happ: input.source_happ.clone(),
        target_happ: input.target_happ.clone(),
        record_hash: input.record_hash,
        bridge_type: input.bridge_type,
    };

    let action_hash = create_entry(EntryTypes::BridgeRecord(bridge.clone()))?;

    // Link from source hApp
    let source_hash = hash_identifier(&input.source_happ)?;
    create_link(
        source_hash,
        action_hash.clone(),
        LinkTypes::SourceToBridge,
        (),
    )?;

    // Link from target hApp
    let target_hash = hash_identifier(&input.target_happ)?;
    create_link(
        target_hash,
        action_hash.clone(),
        LinkTypes::TargetToBridge,
        (),
    )?;

    // Link to all bridges anchor
    let all_anchor = all_bridges_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllBridges, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created bridge record".to_string()
    )))
}

/// Get a bridge record by its action hash
#[hdk_extern]
pub fn get_bridge(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all bridge records from a source hApp
#[hdk_extern]
pub fn get_bridges_from_source(source_happ: String) -> ExternResult<Vec<Record>> {
    let source_hash = hash_identifier(&source_happ)?;
    let links = get_links(
        LinkQuery::try_new(source_hash, LinkTypes::SourceToBridge)?,
        GetStrategy::default(),
    )?;

    let mut bridges = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                bridges.push(record);
            }
        }
    }

    Ok(bridges)
}

/// Get all bridge records to a target hApp
#[hdk_extern]
pub fn get_bridges_to_target(target_happ: String) -> ExternResult<Vec<Record>> {
    let target_hash = hash_identifier(&target_happ)?;
    let links = get_links(
        LinkQuery::try_new(target_hash, LinkTypes::TargetToBridge)?,
        GetStrategy::default(),
    )?;

    let mut bridges = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                bridges.push(record);
            }
        }
    }

    Ok(bridges)
}

/// Get all bridge records
#[hdk_extern]
pub fn get_all_bridges(limit: u32) -> ExternResult<Vec<Record>> {
    if limit == 0 || limit > 1000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Limit must be between 1 and 1000".to_string()
        )));
    }

    let anchor = all_bridges_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllBridges)?,
        GetStrategy::default(),
    )?;

    let mut bridges = Vec::new();
    for link in links.into_iter().take(limit as usize) {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                bridges.push(record);
            }
        }
    }

    Ok(bridges)
}

// ============================================================================
// Cross-hApp Query Functions
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SupplyChainQueryInput {
    pub item_id: Option<String>,
    pub po_hash: Option<String>,
    pub provider: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SupplyChainQueryResult {
    pub claims: Vec<String>,
    pub verifications: Vec<String>,
    pub shipments: Vec<String>,
    pub payments: Vec<String>,
}

/// Query supply chain data across multiple domains
#[hdk_extern]
pub fn query_supply_chain(input: SupplyChainQueryInput) -> ExternResult<SupplyChainQueryResult> {
    // This is a placeholder for cross-hApp queries
    // In production, would use Holochain's bridge API to query other DNAs

    let mut result = SupplyChainQueryResult {
        claims: Vec::new(),
        verifications: Vec::new(),
        shipments: Vec::new(),
        payments: Vec::new(),
    };

    // Query claims if item_id provided
    if let Some(item_id) = input.item_id {
        // Would bridge to claims zome
        result.claims.push(format!("Claims for item: {}", item_id));
    }

    // Query PO-related data if po_hash provided
    if let Some(po_hash) = input.po_hash {
        result.shipments.push(format!("Shipments for PO: {}", po_hash));
        result.payments.push(format!("Payments for PO: {}", po_hash));
    }

    // Query provider data if provider provided
    if let Some(provider) = input.provider {
        result.claims.push(format!("Provider claims: {}", provider));
    }

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastEventInput {
    pub event_type: String,
    pub data: String,
    pub target_happs: Vec<String>,
}

/// Broadcast an event to multiple hApps
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<Vec<String>> {
    // Input validation
    if input.event_type.is_empty() || input.event_type.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event type must be 1-100 characters".to_string()
        )));
    }
    if input.data.len() > 10_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event data cannot exceed 10KB".to_string()
        )));
    }
    if input.target_happs.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one target hApp is required".to_string()
        )));
    }

    let source_happ = "mycelix-supplychain".to_string();
    let mut broadcasted_to = Vec::new();

    // Create bridge records for each target hApp
    for target_happ in input.target_happs {
        let bridge_input = CreateBridgeInput {
            source_happ: source_happ.clone(),
            target_happ: target_happ.clone(),
            record_hash: format!("event:{}", input.event_type),
            bridge_type: "event_broadcast".to_string(),
        };

        match create_bridge(bridge_input) {
            Ok(_) => broadcasted_to.push(target_happ),
            Err(_) => continue,
        }
    }

    Ok(broadcasted_to)
}

/// Get bridge connections for a specific hApp
#[hdk_extern]
pub fn get_happ_connections(happ_name: String) -> ExternResult<(Vec<Record>, Vec<Record>)> {
    if happ_name.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "hApp name is required".to_string()
        )));
    }

    let outgoing = get_bridges_from_source(happ_name.clone())?;
    let incoming = get_bridges_to_target(happ_name)?;

    Ok((outgoing, incoming))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a deterministic hash from a string identifier
fn hash_identifier(identifier: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", identifier).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get the anchor for all bridges
fn all_bridges_anchor() -> ExternResult<EntryHash> {
    hash_identifier("all_bridges")
}
