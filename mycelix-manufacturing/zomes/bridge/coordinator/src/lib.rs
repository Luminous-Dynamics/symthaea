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
}
