// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Coordinator Zome
//!
//! Implements the Anticipatory Repair Loop and cross-hApp integration:
//! - Property hApp: Digital twin wear prediction
//! - Knowledge hApp: Safety verification
//! - Supply Chain hApp: Material sourcing
//! - HEARTH: Local economy funding
//! - Marketplace: Design trading

use bridge_integrity::*;
use fabrication_common::*;
use hdk::prelude::*;

// =============================================================================
// ANTICIPATORY REPAIR SYSTEM
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRepairPredictionInput {
    pub property_asset_hash: ActionHash,
    pub asset_model: String,
    pub predicted_failure_component: String,
    pub failure_probability: f32,
    pub estimated_failure_date: Timestamp,
    pub confidence_interval_days: u32,
    pub sensor_data_summary: String,
}

#[hdk_extern]
pub fn create_repair_prediction(input: CreateRepairPredictionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let recommended_action = if input.failure_probability > 0.8 {
        RepairAction::PrintReplacement
    } else if input.failure_probability > 0.6 {
        RepairAction::CreateDesign
    } else {
        RepairAction::Monitor
    };

    let prediction = RepairPrediction {
        property_asset_hash: input.property_asset_hash.clone(),
        asset_model: input.asset_model,
        predicted_failure_component: input.predicted_failure_component,
        failure_probability: input.failure_probability,
        estimated_failure_date: input.estimated_failure_date,
        confidence_interval_days: input.confidence_interval_days,
        sensor_data_summary: input.sensor_data_summary,
        recommended_action,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let entry = RepairPredictionEntry { prediction };
    let hash = create_entry(EntryTypes::RepairPrediction(entry))?;

    create_link(
        input.property_asset_hash,
        hash.clone(),
        LinkTypes::AssetToPredictions,
        (),
    )?;

    // Auto-create workflow if probability is high enough
    if input.failure_probability > 0.7 {
        create_repair_workflow(hash.clone())?;
    }

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Create a repair workflow from a prediction
#[hdk_extern]
pub fn create_repair_workflow(prediction_hash: ActionHash) -> ExternResult<Record> {
    let now = sys_time()?;

    let workflow = RepairWorkflowEntry {
        prediction_hash: prediction_hash.clone(),
        status: RepairWorkflowStatus::Predicted,
        design_hash: None,
        printer_hash: None,
        hearth_funding_hash: None,
        print_job_hash: None,
        property_installation_hash: None,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        completed_at: None,
    };

    let hash = create_entry(EntryTypes::RepairWorkflow(workflow))?;
    create_link(
        prediction_hash,
        hash.clone(),
        LinkTypes::PredictionToWorkflow,
        (),
    )?;

    let active_anchor = active_workflows_anchor()?;
    create_link(active_anchor, hash.clone(), LinkTypes::ActiveWorkflows, ())?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Update repair workflow status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateWorkflowInput {
    pub workflow_hash: ActionHash,
    pub status: RepairWorkflowStatus,
    pub design_hash: Option<ActionHash>,
    pub printer_hash: Option<ActionHash>,
    pub hearth_funding_hash: Option<ActionHash>,
    pub print_job_hash: Option<ActionHash>,
}

#[hdk_extern]
pub fn update_repair_workflow(input: UpdateWorkflowInput) -> ExternResult<Record> {
    let record = get(input.workflow_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Workflow not found".into())
    ))?;

    let mut workflow: RepairWorkflowEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Parse error".into())))?;

    workflow.status = input.status.clone();
    if let Some(h) = input.design_hash {
        workflow.design_hash = Some(h);
    }
    if let Some(h) = input.printer_hash {
        workflow.printer_hash = Some(h);
    }
    if let Some(h) = input.hearth_funding_hash {
        workflow.hearth_funding_hash = Some(h);
    }
    if let Some(h) = input.print_job_hash {
        workflow.print_job_hash = Some(h);
    }

    if matches!(
        input.status,
        RepairWorkflowStatus::Installed | RepairWorkflowStatus::Cancelled
    ) {
        let now = sys_time()?;
        workflow.completed_at = Some(Timestamp::from_micros(now.as_micros() as i64));
    }

    let new_hash = update_entry(input.workflow_hash, EntryTypes::RepairWorkflow(workflow))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Get active repair workflows
#[hdk_extern]
pub fn get_active_workflows(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = active_workflows_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ActiveWorkflows)?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(wf) = record
                    .entry()
                    .to_app_option::<RepairWorkflowEntry>()
                    .ok()
                    .flatten()
                {
                    // Only include non-completed workflows
                    if wf.completed_at.is_none() {
                        results.push(record);
                    }
                }
            }
        }
    }
    Ok(results)
}

// =============================================================================
// FABRICATION EVENTS & QUERIES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct EmitEventInput {
    pub event_type: FabEventType,
    pub design_id: Option<ActionHash>,
    pub payload: String,
}

#[hdk_extern]
pub fn emit_fabrication_event(input: EmitEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = FabricationEventEntry {
        event_type: input.event_type,
        design_id: input.design_id,
        payload: input.payload,
        source_happ: "fabrication".to_string(),
        timestamp: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::FabricationEvent(event))?;

    let events_anchor = recent_events_anchor()?;
    create_link(events_anchor, hash.clone(), LinkTypes::RecentEvents, ())?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_recent_events(since: Option<Timestamp>) -> ExternResult<Vec<Record>> {
    let anchor = recent_events_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(event) = record
                    .entry()
                    .to_app_option::<FabricationEventEntry>()
                    .ok()
                    .flatten()
                {
                    if let Some(since_ts) = since {
                        if event.timestamp.as_micros() >= since_ts.as_micros() {
                            results.push(record);
                        }
                    } else {
                        results.push(record);
                    }
                }
            }
        }
    }
    Ok(results)
}

// =============================================================================
// MARKETPLACE INTEGRATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ListDesignInput {
    pub design_hash: ActionHash,
    pub price: Option<u64>,
    pub listing_type: ListingType,
}

#[hdk_extern]
pub fn list_design_on_marketplace(input: ListDesignInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let listing = MarketplaceListingEntry {
        design_hash: input.design_hash.clone(),
        marketplace_listing_hash: None, // Would be set by marketplace callback
        price: input.price,
        listing_type: input.listing_type,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::MarketplaceListing(listing))?;
    create_link(
        input.design_hash.clone(),
        hash.clone(),
        LinkTypes::DesignToListings,
        (),
    )?;

    // Emit event for marketplace
    emit_fabrication_event(EmitEventInput {
        event_type: FabEventType::DesignPublished,
        design_id: Some(input.design_hash),
        payload: "{}".to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

// =============================================================================
// SUPPLY CHAIN INTEGRATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct LinkSupplierInput {
    pub material_hash: ActionHash,
    pub supplier_did: String,
    pub supplychain_item_hash: Option<ActionHash>,
}

#[hdk_extern]
pub fn link_material_to_supplier(input: LinkSupplierInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let link_entry = SupplyChainLinkEntry {
        material_hash: input.material_hash.clone(),
        supplychain_item_hash: input.supplychain_item_hash,
        supplier_did: input.supplier_did,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::SupplyChainLink(link_entry))?;
    create_link(
        input.material_hash,
        hash.clone(),
        LinkTypes::MaterialToSuppliers,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

// =============================================================================
// SUPPLY CHAIN INTEGRATION (Cross-cluster via OtherRole)
// =============================================================================

/// Circuit breaker state for cross-cluster calls.
static CIRCUIT_FAILURES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;

fn circuit_open() -> bool {
    CIRCUIT_FAILURES.load(std::sync::atomic::Ordering::Relaxed) >= CIRCUIT_BREAKER_THRESHOLD
}

fn circuit_record_success() {
    CIRCUIT_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
}

fn circuit_record_failure() {
    CIRCUIT_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Query inventory for a specific material type via supply chain cluster.
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryInventoryInput {
    pub material_type: String,
}

#[hdk_extern]
pub fn query_inventory_for_material(input: QueryInventoryInput) -> ExternResult<Vec<u8>> {
    if circuit_open() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circuit breaker open — supply chain unavailable".into()
        )));
    }

    let payload = ExternIO(
        SerializedBytes::try_from(serde_json::json!({
            "category": input.material_type,
        }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .bytes()
        .to_vec(),
    );

    match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("inventory"),
        FunctionName::from("get_items_by_category"),
        None,
        payload,
    ) {
        Ok(response) => match response {
            ZomeCallResponse::Ok(data) => {
                circuit_record_success();
                Ok(data.into_inner())
            }
            ZomeCallResponse::NetworkError(e) => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Supply chain network error: {}",
                    e
                ))))
            }
            other => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Supply chain call rejected: {:?}",
                    other
                ))))
            }
        },
        Err(e) => {
            circuit_record_failure();
            Err(e)
        }
    }
}

/// Check stock level for a specific SKU via supply chain.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckStockInput {
    pub sku: String,
}

#[hdk_extern]
pub fn check_stock_level(input: CheckStockInput) -> ExternResult<Vec<u8>> {
    if circuit_open() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circuit breaker open — supply chain unavailable".into()
        )));
    }

    let payload = ExternIO(
        SerializedBytes::try_from(serde_json::json!({
            "sku": input.sku,
        }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .bytes()
        .to_vec(),
    );

    match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("inventory"),
        FunctionName::from("get_available_stock"),
        None,
        payload,
    ) {
        Ok(response) => match response {
            ZomeCallResponse::Ok(data) => {
                circuit_record_success();
                Ok(data.into_inner())
            }
            other => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Stock check failed: {:?}",
                    other
                ))))
            }
        },
        Err(e) => {
            circuit_record_failure();
            Err(e)
        }
    }
}

/// Create a provenance claim for a fabricated item via supply chain.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProvenanceInput {
    pub item_id: ActionHash,
    pub claim_data: String,
}

#[hdk_extern]
pub fn create_provenance_claim(input: CreateProvenanceInput) -> ExternResult<Vec<u8>> {
    if circuit_open() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circuit breaker open — supply chain unavailable".into()
        )));
    }

    let payload = ExternIO(
        SerializedBytes::try_from(serde_json::json!({
            "item_hash": input.item_id.to_string(),
            "claim_type": "FABRICATED",
            "claim_data": input.claim_data,
        }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .bytes()
        .to_vec(),
    );

    match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("claims"),
        FunctionName::from("create_claim"),
        None,
        payload,
    ) {
        Ok(response) => match response {
            ZomeCallResponse::Ok(data) => {
                circuit_record_success();
                Ok(data.into_inner())
            }
            other => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Provenance claim failed: {:?}",
                    other
                ))))
            }
        },
        Err(e) => {
            circuit_record_failure();
            Err(e)
        }
    }
}

// =============================================================================
// FINANCE INTEGRATION (Cross-cluster via OtherRole)
// =============================================================================

/// Settle payment for a completed print job via finance cluster.
#[derive(Serialize, Deserialize, Debug)]
pub struct SettlePaymentInput {
    pub job_hash: ActionHash,
    pub amount: u64,
    pub currency: String,
}

#[hdk_extern]
pub fn settle_print_payment(input: SettlePaymentInput) -> ExternResult<Vec<u8>> {
    if circuit_open() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circuit breaker open — finance unavailable".into()
        )));
    }

    let payload = ExternIO(
        SerializedBytes::try_from(serde_json::json!({
            "reference": input.job_hash.to_string(),
            "amount": input.amount,
            "currency": input.currency,
            "memo": "fabrication_print_settlement",
        }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .bytes()
        .to_vec(),
    );

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("payments"),
        FunctionName::from("send_payment"),
        None,
        payload,
    ) {
        Ok(response) => match response {
            ZomeCallResponse::Ok(data) => {
                circuit_record_success();
                Ok(data.into_inner())
            }
            other => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Payment settlement failed: {:?}",
                    other
                ))))
            }
        },
        Err(e) => {
            circuit_record_failure();
            Err(e)
        }
    }
}

/// Distribute PoGF rewards to quality contributors via finance recognition.
#[derive(Serialize, Deserialize, Debug)]
pub struct DistributeRewardsInput {
    pub certificate_hash: ActionHash,
    pub recipients: Vec<AgentPubKey>,
    pub reason: String,
}

#[hdk_extern]
pub fn distribute_pog_rewards(input: DistributeRewardsInput) -> ExternResult<Vec<u8>> {
    if circuit_open() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circuit breaker open — finance unavailable".into()
        )));
    }

    let payload = ExternIO(
        SerializedBytes::try_from(serde_json::json!({
            "certificate": input.certificate_hash.to_string(),
            "recipients": input.recipients.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
            "reason": input.reason,
            "recognition_type": "POG_FABRICATION",
        }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .bytes()
        .to_vec(),
    );

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("recognition"),
        FunctionName::from("create_recognition"),
        None,
        payload,
    ) {
        Ok(response) => match response {
            ZomeCallResponse::Ok(data) => {
                circuit_record_success();
                Ok(data.into_inner())
            }
            other => {
                circuit_record_failure();
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "PoGF reward distribution failed: {:?}",
                    other
                ))))
            }
        },
        Err(e) => {
            circuit_record_failure();
            Err(e)
        }
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes =
        SerializedBytes::from(UnsafeBytes::from(format!("anchor:{}", name).into_bytes()));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn active_workflows_anchor() -> ExternResult<EntryHash> {
    make_anchor("active_workflows")
}

fn recent_events_anchor() -> ExternResult<EntryHash> {
    make_anchor("recent_events")
}
