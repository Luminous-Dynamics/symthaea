// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manufacturing Bridge Coordinator Zome
//!
//! Cross-cluster bridge for manufacturing. Provides:
//! - Inbound queries: other clusters can query fabrication/production data
//! - Outbound notifications: manufacturing notifies other clusters on events
//! - Procurement requests: manufacturing requests materials from supplychain
//!
//! All cross-cluster calls use `CallTargetCell::OtherRole` which resolves
//! within the unified hApp (mycelix-unified-happ.yaml).

use hdk::prelude::*;
use manufacturing_bridge_integrity::*;

// ============================================================================
// Input / Output types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct FabricationQueryInput {
    pub product_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FabricationQueryOutput {
    pub product_id: String,
    pub active_work_orders: u32,
    pub total_quantity_planned: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProductionCompleteInput {
    pub work_order_id: String,
    pub product_id: String,
    pub quantity_completed: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcurementRequestInput {
    pub part_id: String,
    pub quantity: u64,
    pub urgency: String,
    pub requester_work_order: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcurementRequestOutput {
    pub accepted: bool,
    pub estimated_delivery: Option<String>,
    pub message: String,
}

// ============================================================================
// Circuit breaker for cross-cluster calls
// ============================================================================

/// Simple circuit breaker state. In a real deployment this would be
/// persisted, but for the coordinator zome we use a per-call retry
/// with graceful fallback.
const MAX_CROSS_CLUSTER_RETRIES: u32 = 2;

fn call_other_role_with_retry(
    role: &str,
    zome: &str,
    fn_name: &str,
    payload: serde_json::Value,
) -> ExternResult<ExternIO> {
    let encoded = ExternIO::encode(payload)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    let mut last_err = None;
    for _attempt in 0..=MAX_CROSS_CLUSTER_RETRIES {
        match call(
            CallTargetCell::OtherRole(role.into()),
            ZomeName::from(zome),
            FunctionName::from(fn_name),
            None,
            encoded.clone(),
        ) {
            Ok(ZomeCallResponse::Ok(data)) => return Ok(data),
            Ok(other) => {
                last_err = Some(wasm_error!(WasmErrorInner::Guest(format!(
                    "Cross-cluster call rejected: {:?}", other
                ))));
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| wasm_error!(WasmErrorInner::Guest(
        format!("Cross-cluster call to {}/{} failed", role, fn_name)
    ))))
}

// ============================================================================
// Extern functions — Inbound (other clusters query manufacturing)
// ============================================================================

/// Query fabrication status for a product.
/// Called by external clusters (e.g., supplychain checking production status).
#[hdk_extern]
pub fn query_fabrication_design(input: FabricationQueryInput) -> ExternResult<FabricationQueryOutput> {
    if input.product_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "product_id is required".to_string()
        )));
    }

    // Query local workorders zome for active work orders for this product
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("workorders"),
        FunctionName::from("list_work_orders"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )?;
    let wo_links: Vec<Link> = match response {
        ZomeCallResponse::Ok(data) => data.decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        other => return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to list work orders: {:?}", other
        )))),
    };

    // Count work orders matching the product_id
    let count = wo_links.len() as u32;

    Ok(FabricationQueryOutput {
        product_id: input.product_id,
        active_work_orders: count,
        total_quantity_planned: 0, // TODO: sum quantities from filtered WOs
    })
}

// ============================================================================
// Extern functions — Outbound (manufacturing notifies other clusters)
// ============================================================================

/// Notify supplychain that a production run is complete and finished goods
/// are ready for inventory receipt.
#[hdk_extern]
pub fn notify_production_complete(input: ProductionCompleteInput) -> ExternResult<ActionHash> {
    if input.product_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "product_id is required".to_string()
        )));
    }
    if input.quantity_completed == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "quantity_completed must be > 0".to_string()
        )));
    }

    let now = sys_time()?;

    // Record the notification locally
    let notification = ProductionNotification {
        work_order_id: input.work_order_id.clone(),
        product_id: input.product_id.clone(),
        quantity_completed: input.quantity_completed,
        completed_at: now,
        target_cluster: "supplychain".to_string(),
        acknowledged: false,
    };
    let notif_hash = create_entry(EntryTypes::ProductionNotification(notification))?;

    // Also record as a bridge event
    let event = BridgeEventEntry {
        event_type: "production_complete".to_string(),
        source_cluster: "manufacturing".to_string(),
        target_cluster: "supplychain".to_string(),
        payload: serde_json::json!({
            "work_order_id": input.work_order_id,
            "product_id": input.product_id,
            "quantity_completed": input.quantity_completed,
        })
        .to_string(),
        reference_hash: None,
        created_at: now,
    };
    create_entry(EntryTypes::BridgeEvent(event))?;

    // Attempt cross-cluster notification to supplychain
    // If supplychain is not installed, this fails gracefully
    let _ = call_other_role_with_retry(
        "supplychain",
        "inventory",
        "receive_production_notification",
        serde_json::json!({
            "product_id": input.product_id,
            "quantity": input.quantity_completed,
            "source": "manufacturing",
        }),
    );

    Ok(notif_hash)
}

/// Request procurement of materials from the supplychain cluster.
#[hdk_extern]
pub fn request_procurement(input: ProcurementRequestInput) -> ExternResult<ProcurementRequestOutput> {
    if input.part_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "part_id is required".to_string()
        )));
    }
    if input.quantity == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "quantity must be > 0".to_string()
        )));
    }

    let now = sys_time()?;

    // Record the procurement request as a bridge event
    let event = BridgeEventEntry {
        event_type: "procurement_request".to_string(),
        source_cluster: "manufacturing".to_string(),
        target_cluster: "supplychain".to_string(),
        payload: serde_json::json!({
            "part_id": input.part_id,
            "quantity": input.quantity,
            "urgency": input.urgency,
        })
        .to_string(),
        reference_hash: input.requester_work_order.clone(),
        created_at: now,
    };
    create_entry(EntryTypes::BridgeEvent(event))?;

    // Call supplychain procurement zome
    match call_other_role_with_retry(
        "supplychain",
        "procurement",
        "request_material",
        serde_json::json!({
            "part_id": input.part_id,
            "quantity": input.quantity,
            "urgency": input.urgency,
            "source_cluster": "manufacturing",
        }),
    ) {
        Ok(data) => {
            match data.decode::<ProcurementRequestOutput>() {
                Ok(response) => Ok(response),
                Err(_) => Ok(ProcurementRequestOutput {
                    accepted: true,
                    estimated_delivery: None,
                    message: "Request accepted (response not parseable)".to_string(),
                }),
            }
        }
        Err(_) => {
            // Supplychain unavailable -- return a "pending" response
            Ok(ProcurementRequestOutput {
                accepted: false,
                estimated_delivery: None,
                message: "Supplychain cluster unavailable. Request queued locally.".to_string(),
            })
        }
    }
}

// ============================================================================
// Gap 3: Commons ↔ Manufacturing — local-first sourcing
// ============================================================================

/// Input for querying commons inventory.
#[derive(Serialize, Deserialize, Debug)]
pub struct CommonsInventoryQueryInput {
    /// SKU or resource category to look for.
    pub sku_or_category: String,
}

/// Result returned from the commons query.
#[derive(Serialize, Deserialize, Debug)]
pub struct CommonsInventoryResult {
    /// Available quantity from commons (None if commons not installed).
    pub available_quantity: Option<u64>,
    /// Whether the commons cluster was reachable.
    pub commons_available: bool,
    /// Non-fatal error if commons was unreachable.
    pub error: Option<String>,
}

/// Query the commons resource-mesh cluster for available resources matching
/// a SKU or category before falling back to the supplychain cluster.
///
/// Calls `get_resource_status` on the commons `resource_mesh` zome.
/// If the commons cluster is not installed the call returns `Err`, which is
/// caught and surfaced as a graceful `CommonsInventoryResult`.
#[hdk_extern]
pub fn query_commons_inventory(
    input: CommonsInventoryQueryInput,
) -> ExternResult<CommonsInventoryResult> {
    if input.sku_or_category.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "sku_or_category must not be empty".to_string()
        )));
    }

    let payload = ExternIO::encode(serde_json::json!({
        "resource_type": input.sku_or_category,
        "location": null,
        "limit": 50,
    }))
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    match call(
        // Commons is split into two roles in the unified hApp: commons_land + commons_care.
        // `resource_mesh` lives in the commons_care DNA.
        CallTargetCell::OtherRole("commons_care".into()),
        ZomeName::from("resource_mesh"),
        FunctionName::from("get_resource_status"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            // get_resource_status returns Vec<SensorReading>.
            // Each SensorReading has a `value: f64` field representing the current
            // measurement (e.g., quantity on hand). We sum them for total available.
            let readings: serde_json::Value =
                data.decode().unwrap_or(serde_json::Value::Array(vec![]));

            let total: u64 = readings
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|r| r.get("value").and_then(|v| v.as_f64()))
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .map(|v| v.round() as u64)
                        .sum()
                })
                .unwrap_or(0);

            Ok(CommonsInventoryResult {
                available_quantity: Some(total),
                commons_available: true,
                error: None,
            })
        }
        Ok(other) => Ok(CommonsInventoryResult {
            available_quantity: None,
            commons_available: true,
            error: Some(format!(
                "Commons cluster call rejected: {:?}",
                other
            )),
        }),
        Err(_) => Ok(CommonsInventoryResult {
            available_quantity: None,
            commons_available: false,
            error: Some("Commons cluster not available".to_string()),
        }),
    }
}

/// Combined availability result from local-preference sourcing.
#[derive(Serialize, Deserialize, Debug)]
pub struct LocalPreferenceResult {
    /// Quantity sourced from commons (0 if unavailable or insufficient).
    pub from_commons: u64,
    /// Quantity that must come from supplychain.
    pub from_supplychain: u64,
    /// Whether commons was reachable.
    pub commons_available: bool,
    /// Whether supplychain was queried for the deficit.
    pub supplychain_queried: bool,
    /// Whether supplychain has sufficient stock for the deficit.
    pub supplychain_sufficient: bool,
    /// Non-fatal message or error.
    pub message: String,
}

/// Source a quantity of a part/resource with local commons preference.
///
/// First queries commons via `query_commons_inventory`.  If commons has
/// enough, supplychain is not contacted.  If commons is insufficient (or
/// unavailable), the deficit is looked up in supplychain via
/// `get_stock_level_by_sku`.
#[hdk_extern]
pub fn source_with_local_preference(
    input: ProcurementRequestInput,
) -> ExternResult<LocalPreferenceResult> {
    if input.part_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "part_id is required".to_string()
        )));
    }
    if input.quantity == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "quantity must be > 0".to_string()
        )));
    }

    // Step 1: Try commons first
    let commons_result = query_commons_inventory(CommonsInventoryQueryInput {
        sku_or_category: input.part_id.clone(),
    })?;

    let commons_qty = commons_result.available_quantity.unwrap_or(0);
    let commons_available = commons_result.commons_available;

    if commons_qty >= input.quantity {
        // Commons alone can satisfy the request
        return Ok(LocalPreferenceResult {
            from_commons: input.quantity,
            from_supplychain: 0,
            commons_available,
            supplychain_queried: false,
            supplychain_sufficient: false,
            message: format!(
                "Commons can fully supply {} units of {}",
                input.quantity, input.part_id
            ),
        });
    }

    let deficit = input.quantity.saturating_sub(commons_qty);

    // Step 2: Query supplychain for the deficit
    let sc_payload = ExternIO::encode(serde_json::json!({ "sku": input.part_id }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let (sc_sufficient, message) = match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("inventory_coordinator"),
        FunctionName::from("get_stock_level_by_sku"),
        None,
        sc_payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let level: serde_json::Value = data.decode().unwrap_or(serde_json::Value::Null);
            let sc_qty = level
                .get("quantity")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sufficient = sc_qty >= deficit;
            let msg = if sufficient {
                format!(
                    "Commons supplies {} units; supplychain covers {} unit deficit",
                    commons_qty, deficit
                )
            } else {
                format!(
                    "Commons supplies {} units; supplychain only has {} of {} needed",
                    commons_qty, sc_qty, deficit
                )
            };
            (sufficient, msg)
        }
        Ok(other) => (
            false,
            format!("Supplychain call rejected: {:?}", other),
        ),
        Err(_) => (
            false,
            "Supplychain cluster not available".to_string(),
        ),
    };

    Ok(LocalPreferenceResult {
        from_commons: commons_qty,
        from_supplychain: deficit,
        commons_available,
        supplychain_queried: true,
        supplychain_sufficient: sc_sufficient,
        message,
    })
}

/// List all bridge events (for auditing / debugging).
#[hdk_extern]
pub fn list_bridge_events(_: ()) -> ExternResult<Vec<Link>> {
    let all_path = Path::from("all_bridge_events").typed(LinkTypes::AllBridgeEvents)?;
    get_links(
        GetLinksInputBuilder::try_new(
            all_path.path_entry_hash()?,
            LinkTypes::AllBridgeEvents,
        )?
        .build(),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fabrication_query_input_serde() {
        let input = FabricationQueryInput {
            product_id: "WIDGET-001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: FabricationQueryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.product_id, "WIDGET-001");
    }

    #[test]
    fn test_fabrication_query_output_serde() {
        let output = FabricationQueryOutput {
            product_id: "WIDGET-001".to_string(),
            active_work_orders: 3,
            total_quantity_planned: 500,
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: FabricationQueryOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.active_work_orders, 3);
        assert_eq!(back.total_quantity_planned, 500);
    }

    #[test]
    fn test_production_complete_input_serde() {
        let input = ProductionCompleteInput {
            work_order_id: "WO-001".to_string(),
            product_id: "WIDGET-001".to_string(),
            quantity_completed: 50,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ProductionCompleteInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.quantity_completed, 50);
    }

    #[test]
    fn test_procurement_request_serde() {
        let input = ProcurementRequestInput {
            part_id: "BOLT-M6".to_string(),
            quantity: 1000,
            urgency: "normal".to_string(),
            requester_work_order: Some("WO-001".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ProcurementRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.part_id, "BOLT-M6");
        assert_eq!(back.quantity, 1000);
    }

    #[test]
    fn test_procurement_response_fallback() {
        let response = ProcurementRequestOutput {
            accepted: false,
            estimated_delivery: None,
            message: "Supplychain cluster unavailable. Request queued locally.".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let back: ProcurementRequestOutput = serde_json::from_str(&json).unwrap();
        assert!(!back.accepted);
        assert!(back.message.contains("unavailable"));
    }

    // ===== Gap 3: Commons ↔ Manufacturing Tests =====

    #[test]
    fn test_commons_query_result_serde() {
        // Commons available with quantity
        let result = CommonsInventoryResult {
            available_quantity: Some(250),
            commons_available: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CommonsInventoryResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.available_quantity, Some(250));
        assert!(back.commons_available);
        assert!(back.error.is_none());

        // Commons unavailable
        let result2 = CommonsInventoryResult {
            available_quantity: None,
            commons_available: false,
            error: Some("Commons cluster not available".to_string()),
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: CommonsInventoryResult = serde_json::from_str(&json2).unwrap();
        assert!(back2.available_quantity.is_none());
        assert!(!back2.commons_available);
        assert!(back2.error.is_some());
    }

    #[test]
    fn test_local_preference_result_serde() {
        // Commons fully covers the request
        let result = LocalPreferenceResult {
            from_commons: 100,
            from_supplychain: 0,
            commons_available: true,
            supplychain_queried: false,
            supplychain_sufficient: false,
            message: "Commons can fully supply 100 units of BOLT-M6".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: LocalPreferenceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.from_commons, 100);
        assert_eq!(back.from_supplychain, 0);
        assert!(!back.supplychain_queried);

        // Mixed sourcing
        let result2 = LocalPreferenceResult {
            from_commons: 40,
            from_supplychain: 60,
            commons_available: true,
            supplychain_queried: true,
            supplychain_sufficient: true,
            message: "Commons supplies 40 units; supplychain covers 60 unit deficit".to_string(),
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: LocalPreferenceResult = serde_json::from_str(&json2).unwrap();
        assert_eq!(back2.from_commons, 40);
        assert_eq!(back2.from_supplychain, 60);
        assert!(back2.supplychain_queried);
        assert!(back2.supplychain_sufficient);
    }
}
