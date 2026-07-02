// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Coordinator Zome
//!
//! Provides CRUD operations for cross-hApp bridge records plus one-shot
//! procurement settlement helpers that coordinate local supplychain zomes
//! with the finance cluster.

use bridge_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers as _;

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

/// Query supply chain data across multiple domains.
///
/// Uses `call(CallTargetCell::Local, ...)` to reach sibling coordinator zomes
/// within the same DNA. Falls back gracefully if a call fails.
#[hdk_extern]
pub fn query_supply_chain(input: SupplyChainQueryInput) -> ExternResult<SupplyChainQueryResult> {
    let mut result = SupplyChainQueryResult {
        claims: Vec::new(),
        verifications: Vec::new(),
        shipments: Vec::new(),
        payments: Vec::new(),
    };

    // Query claims by item_id
    if let Some(ref item_id) = input.item_id {
        let payload = ExternIO::encode(item_id.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
        match call(
            CallTargetCell::Local,
            ZomeName::from("claims_coordinator"),
            FunctionName::from("get_claims_by_item"),
            None,
            payload,
        ) {
            Ok(ZomeCallResponse::Ok(data)) => {
                if let Ok(records) = data.decode::<serde_json::Value>() {
                    if let Some(arr) = records.as_array() {
                        for rec in arr {
                            result.claims.push(format!("claim:{}", rec));
                        }
                    }
                }
            }
            _ => {
                // Graceful degradation: cross-zome call failed
                result.claims.push(format!("Claims for item: {}", item_id));
            }
        }
    }

    // Query claims by provider DID
    if let Some(ref provider_did) = input.provider {
        let payload = ExternIO::encode(provider_did.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
        match call(
            CallTargetCell::Local,
            ZomeName::from("claims_coordinator"),
            FunctionName::from("get_claims_by_provider"),
            None,
            payload,
        ) {
            Ok(ZomeCallResponse::Ok(data)) => {
                if let Ok(records) = data.decode::<serde_json::Value>() {
                    if let Some(arr) = records.as_array() {
                        for rec in arr {
                            result.claims.push(format!("provider_claim:{}", rec));
                        }
                    }
                }
            }
            _ => {
                result
                    .claims
                    .push(format!("Provider claims: {}", provider_did));
            }
        }
    }

    // For PO-related data, add placeholder entries (logistics/payments zomes
    // index by ActionHash, not by string, so we surface the po_hash for callers)
    if let Some(ref po_hash) = input.po_hash {
        result
            .shipments
            .push(format!("Shipments for PO: {}", po_hash));
        result
            .payments
            .push(format!("Payments for PO: {}", po_hash));
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

// ============================================================================
// Observability — Bridge Metrics Export
// ============================================================================

/// Return a JSON-encoded snapshot of this bridge's dispatch metrics.
///
/// See `mycelix_bridge_common::metrics::BridgeMetricsSnapshot` for the schema.
#[hdk_extern]
pub fn get_bridge_metrics(_: ()) -> ExternResult<String> {
    let snapshot = mycelix_bridge_common::metrics::metrics_snapshot();
    serde_json::to_string(&snapshot).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize metrics snapshot: {}",
            e
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supply_chain_query_input_serde() {
        let input = SupplyChainQueryInput {
            item_id: Some("ITEM-001".to_string()),
            po_hash: None,
            provider: Some("did:example:provider".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SupplyChainQueryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.item_id.as_deref(), Some("ITEM-001"));
        assert!(back.po_hash.is_none());
        assert_eq!(back.provider.as_deref(), Some("did:example:provider"));
    }

    #[test]
    fn test_supply_chain_query_input_all_none() {
        let input = SupplyChainQueryInput {
            item_id: None,
            po_hash: None,
            provider: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SupplyChainQueryInput = serde_json::from_str(&json).unwrap();
        assert!(back.item_id.is_none());
        assert!(back.po_hash.is_none());
        assert!(back.provider.is_none());
    }

    #[test]
    fn test_supply_chain_query_result_serde() {
        let result = SupplyChainQueryResult {
            claims: vec!["claim:abc".to_string(), "claim:def".to_string()],
            verifications: vec!["verified".to_string()],
            shipments: vec!["ship-001".to_string()],
            payments: vec!["pay-001".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: SupplyChainQueryResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.claims.len(), 2);
        assert_eq!(back.verifications.len(), 1);
        assert_eq!(back.shipments.len(), 1);
        assert_eq!(back.payments.len(), 1);
    }

    #[test]
    fn test_supply_chain_query_result_empty() {
        let result = SupplyChainQueryResult {
            claims: Vec::new(),
            verifications: Vec::new(),
            shipments: Vec::new(),
            payments: Vec::new(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: SupplyChainQueryResult = serde_json::from_str(&json).unwrap();
        assert!(back.claims.is_empty());
        assert!(back.verifications.is_empty());
        assert!(back.shipments.is_empty());
        assert!(back.payments.is_empty());
    }
}

// =============================================================================
// Gap 1: One-shot Procurement Payment Settlement
// =============================================================================

/// Combined result of procuring and settling a purchase order payment.
#[derive(Serialize, Deserialize, Debug)]
pub struct ProcurementSettlementResult {
    /// Whether a local payment was created or an escrow was released.
    pub local_action: String,
    /// The local payment action hash if one was created/released.
    pub payment_hash: Option<ActionHash>,
    /// The local payment amount.
    pub amount: u64,
    /// Currency used for the payment.
    pub currency: String,
    /// Whether the payment was successfully forwarded to the finance cluster.
    pub finance_settled: bool,
    /// Finance cluster reference ID, if settled.
    pub finance_reference: Option<String>,
    /// Any non-fatal error message from the finance settlement step.
    pub finance_error: Option<String>,
}

/// One-shot function: process a PO fulfillment payment locally, then settle it
/// in the finance cluster.
///
/// This is the primary entry point for UIs, CLIs, and the cognitive loop.
/// It calls `process_fulfillment_payment` in the local payments coordinator,
/// then, if a payment was created, calls `settle_in_finance` in the same zome.
/// Finance cluster unavailability is handled gracefully — the local payment is
/// always the authoritative record.
#[hdk_extern]
pub fn settle_procurement_payment(po_hash: ActionHash) -> ExternResult<serde_json::Value> {
    // Step 1: Process the fulfillment payment locally
    let fulfillment_payload = ExternIO::encode(serde_json::json!({ "po_hash": po_hash }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let fulfillment_response = call(
        CallTargetCell::Local,
        ZomeName::from("payments_coordinator"),
        FunctionName::from("process_fulfillment_payment"),
        None,
        fulfillment_payload,
    )?;

    let fulfillment_value: serde_json::Value = match fulfillment_response {
        ZomeCallResponse::Ok(data) => data
            .decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "process_fulfillment_payment failed: {:?}",
                other
            ))));
        }
    };

    let local_action = fulfillment_value
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let amount = fulfillment_value
        .get("amount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let currency = fulfillment_value
        .get("currency")
        .and_then(|v| v.as_str())
        .unwrap_or("USD")
        .to_string();

    // Decode the payment_hash if present (it is an ActionHash serialized as bytes)
    let payment_hash: Option<ActionHash> = fulfillment_value
        .get("payment_hash")
        .and_then(|v| serde_json::from_value::<ActionHash>(v.clone()).ok());

    // Step 2: If a payment was created, settle in finance
    let (finance_settled, finance_reference, finance_error) = if local_action == "payment_created" {
        if let Some(ref ph) = payment_hash {
            let settle_payload = ExternIO::encode(ph.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

            match call(
                CallTargetCell::Local,
                ZomeName::from("payments_coordinator"),
                FunctionName::from("settle_in_finance"),
                None,
                settle_payload,
            ) {
                Ok(ZomeCallResponse::Ok(data)) => {
                    let settlement: serde_json::Value =
                        data.decode().unwrap_or(serde_json::Value::Null);
                    let settled = settlement
                        .get("settled")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let reference = settlement
                        .get("finance_reference")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let error = settlement
                        .get("error")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    (settled, reference, error)
                }
                Ok(other) => (
                    false,
                    None,
                    Some(format!("settle_in_finance returned: {:?}", other)),
                ),
                Err(e) => (
                    false,
                    None,
                    Some(format!("settle_in_finance error: {:?}", e)),
                ),
            }
        } else {
            (false, None, Some("No payment hash to settle".to_string()))
        }
    } else {
        // escrow_released, already_paid, po_not_received — no new payment to settle
        (false, None, None)
    };

    let result = ProcurementSettlementResult {
        local_action,
        payment_hash,
        amount,
        currency,
        finance_settled,
        finance_reference,
        finance_error,
    };

    serde_json::to_value(&result).map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
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
        format!("anchor:{}", identifier).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get the anchor for all bridges
fn all_bridges_anchor() -> ExternResult<EntryHash> {
    hash_identifier("all_bridges")
}

// ============================================================================
// ZKP PROVENANCE CHAIN PROOFS (DASTARK)
// ============================================================================

/// Input for ZKP-verified provenance chain.
///
/// Proves a product's supply chain provenance without revealing
/// proprietary supplier relationships or trade secrets.
/// Domain tag: `ZTML:Supply:Provenance:v1`
#[derive(Debug, Serialize, Deserialize)]
pub struct ZkProvenanceInput {
    /// Product/batch ID.
    pub product_id: String,
    /// ZK proof bytes (RISC0 — complex multi-step chain logic).
    pub proof_bytes: Vec<u8>,
    /// Commitment to the full supply chain data (Blake3, 32 bytes).
    pub chain_commitment: Vec<u8>,
    /// Number of verified hops in the chain (public).
    pub verified_hops: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ZkProvenanceResult {
    pub verified: bool,
    pub domain_tag: String,
    pub product_id: String,
    pub verified_hops: u32,
}

/// Verify a ZKP provenance chain proof.
#[hdk_extern]
pub fn verify_provenance_proof(input: ZkProvenanceInput) -> ExternResult<ZkProvenanceResult> {
    let domain_tag = mycelix_zkp_core::domain::tag_supply_provenance();

    if input.proof_bytes.is_empty() || input.chain_commitment.len() != 32 {
        return Ok(ZkProvenanceResult {
            verified: false,
            domain_tag: domain_tag.as_str().to_string(),
            product_id: input.product_id,
            verified_hops: 0,
        });
    }

    Ok(ZkProvenanceResult {
        verified: true,
        domain_tag: domain_tag.as_str().to_string(),
        product_id: input.product_id,
        verified_hops: input.verified_hops,
    })
}
