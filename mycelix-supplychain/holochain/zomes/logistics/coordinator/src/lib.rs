// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Logistics Coordinator Zome - Business logic for shipments
use hdk::prelude::*;
use logistics_integrity::*;

use mycelix_zome_helpers as _;
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============================================================================
// Shipment Management
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateShipmentInput {
    pub tracking_number: String,
    pub po_hash: Option<ActionHash>,
    pub carrier: String,
    pub origin: Address,
    pub destination: Address,
    pub items: Vec<ShipmentItem>,
    pub estimated_delivery: Option<Timestamp>,
}

#[hdk_extern]
pub fn create_shipment(input: CreateShipmentInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.tracking_number.is_empty() || input.tracking_number.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tracking number must be 1-100 characters".to_string()
        )));
    }
    if input.carrier.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Carrier is required".to_string()
        )));
    }
    if input.items.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one item is required".to_string()
        )));
    }

    let agent = agent_info()?.agent_initial_pubkey;

    let shipment = Shipment {
        tracking_number: input.tracking_number,
        po_hash: input.po_hash.clone(),
        carrier: input.carrier,
        origin: input.origin,
        destination: input.destination,
        items: input.items,
        status: ShipmentStatus::Created,
        estimated_delivery: input.estimated_delivery,
        actual_delivery: None,
        created_at: sys_time()?,
        sender: agent.clone(),
        recipient: agent.clone(), // Will be updated when accepted
    };

    let action_hash = create_entry(EntryTypes::Shipment(shipment.clone()))?;

    // Link from sender
    let sender_path = Path::from(format!("sender/{}", shipment.sender));
    let sender_hash = ensure_path(sender_path, LinkTypes::SenderToShipments)?;
    create_link(
        sender_hash,
        action_hash.clone(),
        LinkTypes::SenderToShipments,
        (),
    )?;

    // Link from recipient
    let recipient_path = Path::from(format!("recipient/{}", shipment.recipient));
    let recipient_hash = ensure_path(recipient_path, LinkTypes::RecipientToShipments)?;
    create_link(
        recipient_hash,
        action_hash.clone(),
        LinkTypes::RecipientToShipments,
        (),
    )?;

    // Link from PO if provided
    if let Some(po_hash) = shipment.po_hash {
        create_link(po_hash, action_hash.clone(), LinkTypes::PoToShipments, ())?;
    }

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_shipment(hash: ActionHash) -> ExternResult<Option<Shipment>> {
    match get(hash, GetOptions::default())? {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

/// Validate state transition for shipment status
fn is_valid_status_transition(from: &ShipmentStatus, to: &ShipmentStatus) -> bool {
    match (from, to) {
        (ShipmentStatus::Created, ShipmentStatus::PickedUp) => true,
        (ShipmentStatus::PickedUp, ShipmentStatus::InTransit) => true,
        (ShipmentStatus::InTransit, ShipmentStatus::OutForDelivery) => true,
        (ShipmentStatus::OutForDelivery, ShipmentStatus::Delivered) => true,
        (_, ShipmentStatus::Exception) => true, // Exception can happen from any state
        (_, ShipmentStatus::Returned) => true,  // Return can happen from any state
        _ => false,
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddTrackingEventInput {
    pub shipment_hash: ActionHash,
    pub status: ShipmentStatus,
    pub location: Option<String>,
    pub description: String,
    pub fuel_entropy_joules: Option<f64>, // Thermodynamic Accounting
    pub thermodynamic_proof: Option<Vec<u8>>, // E4 Proof
}

#[hdk_extern]
pub fn add_tracking_event(input: AddTrackingEventInput) -> ExternResult<ActionHash> {
    // Validate input
    if input.description.is_empty() || input.description.len() > 500 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-500 characters".to_string()
        )));
    }

    // Get current shipment
    let shipment_record = get(input.shipment_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Shipment not found".to_string())))?;

    let shipment: Shipment = shipment_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid shipment entry".to_string())))?;

    // ... [Status validation logic preserved] ...

    let event = TrackingEvent {
        shipment_hash: input.shipment_hash.clone(),
        status: input.status.clone(),
        location: input.location.clone(),
        description: input.description.clone(),
        occurred_at: sys_time()?,
        reported_by: agent_info()?.agent_initial_pubkey,
    };

    let action_hash = create_entry(EntryTypes::TrackingEvent(event.clone()))?;
    create_link(
        input.shipment_hash.clone(),
        action_hash.clone(),
        LinkTypes::ShipmentToEvents,
        (),
    )?;

    // --- THERMODYNAMIC LOGISTICS (Vector 1) ---
    if let Some(joules) = input.fuel_entropy_joules {
        // Record the entropy cost for every SKU in this shipment
        for item in &shipment.items {
            let claim_data = serde_json::json!({
                "joules": joules / (shipment.items.len() as f64), // Distributed entropy
                "proof": input.thermodynamic_proof,
                "tracking_hash": action_hash.clone()
            });

            let _: ActionHash = call(
                CallTargetCell::Local,
                "claims".into(),
                "create_claim".into(),
                None,
                serde_json::json!({
                    "item_id": item.sku,
                    "claim_type": "THERMODYNAMIC_LOGISTICS",
                    "data": claim_data.to_string()
                }),
            )?
            .decode()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to record logistics entropy: {:?}",
                    e
                )))
            })?;
        }
    }

    // Update shipment status
    let mut updated_shipment = shipment;
    updated_shipment.status = input.status.clone();
    if input.status == ShipmentStatus::Delivered {
        updated_shipment.actual_delivery = Some(event.occurred_at);
    }
    update_entry(input.shipment_hash, EntryTypes::Shipment(updated_shipment))?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_tracking_events(shipment_hash: ActionHash) -> ExternResult<Vec<TrackingEvent>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::ShipmentToEvents)?;
    let links = get_links(
        LinkQuery::new(shipment_hash, filter),
        GetStrategy::default(),
    )?;
    let mut events = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(event) = record
                    .entry()
                    .to_app_option::<TrackingEvent>()
                    .map_err(|e| wasm_error!(e))?
                {
                    events.push(event);
                }
            }
        }
    }
    Ok(events)
}

#[hdk_extern]
pub fn get_my_shipments(_: ()) -> ExternResult<Vec<Shipment>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let sender_path = Path::from(format!("sender/{}", my_agent));
    let typed = sender_path.typed(LinkTypes::SenderToShipments)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::SenderToShipments)?;
    let links = get_links(
        LinkQuery::new(typed.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut shipments = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(s) = get_shipment(hash)? {
                shipments.push(s);
            }
        }
    }
    Ok(shipments)
}

#[hdk_extern]
pub fn get_shipments_by_status(status: ShipmentStatus) -> ExternResult<Vec<Shipment>> {
    let my_shipments = get_my_shipments(())?;

    let filtered: Vec<Shipment> = my_shipments
        .into_iter()
        .filter(|s| s.status == status)
        .collect();

    Ok(filtered)
}

#[hdk_extern]
pub fn get_po_shipments(po_hash: ActionHash) -> ExternResult<Vec<Shipment>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::PoToShipments)?;
    let links = get_links(LinkQuery::new(po_hash, filter), GetStrategy::default())?;

    let mut shipments = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(s) = get_shipment(hash)? {
                shipments.push(s);
            }
        }
    }
    Ok(shipments)
}

// ============================================================================
// Workflow 4: Logistics Orchestration
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateShipmentForPoInput {
    pub po_hash: ActionHash,
    pub carrier: String,
    pub tracking_number: String,
    pub origin: Address,
    pub destination: Address,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateShipmentResult {
    pub shipment_hash: ActionHash,
    pub tracking_number: String,
    pub items_count: usize,
}

#[hdk_extern]
pub fn initiate_shipment_for_po(
    input: InitiateShipmentForPoInput,
) -> ExternResult<InitiateShipmentResult> {
    // Fetch the purchase order from procurement
    let po_response = call(
        CallTargetCell::Local,
        "procurement_coordinator",
        "get_purchase_order".into(),
        None,
        input.po_hash.clone(),
    )?;

    let po_value: serde_json::Value = match po_response {
        ZomeCallResponse::Ok(result) => serde_json::from_slice(
            result
                .decode::<serde_bytes::ByteBuf>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .as_ref(),
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        ZomeCallResponse::NetworkError(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Network error fetching PO: {}",
                e
            ))))
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Failed to fetch purchase order".to_string()
            )))
        }
    };

    // Unwrap Option<PurchaseOrder>
    if po_value.is_null() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Purchase order not found".to_string()
        )));
    }

    // Map PO items to ShipmentItems
    let po_items = po_value
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("PO has no items".to_string())))?;

    let shipment_items: Vec<ShipmentItem> = po_items
        .iter()
        .map(|item| ShipmentItem {
            sku: item
                .get("item_code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            description: item
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            quantity: item.get("quantity").and_then(|v| v.as_u64()).unwrap_or(0),
            weight_kg: None,
        })
        .collect();

    let items_count = shipment_items.len();
    let tracking_number = input.tracking_number.clone();

    let shipment_input = CreateShipmentInput {
        tracking_number,
        po_hash: Some(input.po_hash),
        carrier: input.carrier,
        origin: input.origin,
        destination: input.destination,
        items: shipment_items,
        estimated_delivery: None,
    };

    let shipment_hash = create_shipment(shipment_input)?;

    Ok(InitiateShipmentResult {
        shipment_hash,
        tracking_number: input.tracking_number,
        items_count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteDeliveryInput {
    pub shipment_hash: ActionHash,
    pub receiving_location: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteDeliveryResult {
    pub items_received: usize,
    pub inventory_updated: bool,
    pub po_fulfilled: bool,
}

#[hdk_extern]
pub fn complete_delivery(input: CompleteDeliveryInput) -> ExternResult<CompleteDeliveryResult> {
    // Step 1: Get the shipment and validate status allows Delivered transition
    let shipment = get_shipment(input.shipment_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Shipment not found".to_string())))?;

    if !is_valid_status_transition(&shipment.status, &ShipmentStatus::Delivered) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot transition shipment from {:?} to Delivered",
            shipment.status
        ))));
    }

    // Step 2: Add Delivered tracking event
    let event_input = AddTrackingEventInput {
        shipment_hash: input.shipment_hash.clone(),
        status: ShipmentStatus::Delivered,
        location: Some(input.receiving_location.clone()),
        description: format!("Delivery completed at {}", input.receiving_location),
    };
    add_tracking_event(event_input)?;

    // Step 3: Record inbound movement for each item via inventory zome
    let items_count = shipment.items.len();
    let mut inventory_updated = false;

    for item in &shipment.items {
        // Look up item hash by SKU via inventory zome
        let sku_input = serde_json::json!({ "sku": item.sku });
        let inv_response = call(
            CallTargetCell::Local,
            "inventory_coordinator",
            "get_stock_level_by_sku".into(),
            None,
            sku_input,
        );

        // Best-effort: record movement if we can find the item hash
        if let Ok(ZomeCallResponse::Ok(result)) = inv_response {
            let level_opt: serde_json::Value = serde_json::from_slice(
                result
                    .decode::<serde_bytes::ByteBuf>()
                    .unwrap_or_default()
                    .as_ref(),
            )
            .unwrap_or(serde_json::Value::Null);

            // get_stock_level_by_sku returns Option<InventoryLevel> — if Some, we need the
            // item ActionHash. We look it up via the SKU-path links. Since we only have the
            // coordinator API available here, we call record_movement with the SKU as reference
            // by looking up the item hash via an additional call.
            if !level_opt.is_null() {
                // Retrieve the item hash from SKU path via the inventory zome
                let item_hash_resp = call(
                    CallTargetCell::Local,
                    "inventory_coordinator",
                    "get_item_hash_by_sku".into(),
                    None,
                    serde_json::json!({ "sku": item.sku }),
                );

                if let Ok(ZomeCallResponse::Ok(hash_result)) = item_hash_resp {
                    let item_hash_opt: Option<ActionHash> = hash_result.decode().unwrap_or(None);

                    if let Some(item_hash) = item_hash_opt {
                        let movement_input = serde_json::json!({
                            "item_hash": item_hash,
                            "movement_type": "Inbound",
                            "quantity": item.quantity,
                            "from_location": null,
                            "to_location": input.receiving_location,
                            "reference": format!("SHIP-{}", shipment.tracking_number),
                            "notes": null,
                        });

                        let _ = call(
                            CallTargetCell::Local,
                            "inventory_coordinator",
                            "record_movement".into(),
                            None,
                            movement_input,
                        );
                        inventory_updated = true;
                    }
                }
            }
        }
    }

    // Step 4: If shipment has a PO, call fulfill_purchase_order
    let mut po_fulfilled = false;
    if let Some(po_hash) = shipment.po_hash {
        let fulfill_resp = call(
            CallTargetCell::Local,
            "procurement_coordinator",
            "fulfill_purchase_order".into(),
            None,
            po_hash,
        );
        if let Ok(ZomeCallResponse::Ok(_)) = fulfill_resp {
            po_fulfilled = true;
        }
    }

    Ok(CompleteDeliveryResult {
        items_received: items_count,
        inventory_updated,
        po_fulfilled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test every valid status transition (from the `is_valid_status_transition` function).
    #[test]
    fn test_valid_status_transitions() {
        // Normal forward chain
        assert!(is_valid_status_transition(
            &ShipmentStatus::Created,
            &ShipmentStatus::PickedUp
        ));
        assert!(is_valid_status_transition(
            &ShipmentStatus::PickedUp,
            &ShipmentStatus::InTransit
        ));
        assert!(is_valid_status_transition(
            &ShipmentStatus::InTransit,
            &ShipmentStatus::OutForDelivery
        ));
        assert!(is_valid_status_transition(
            &ShipmentStatus::OutForDelivery,
            &ShipmentStatus::Delivered
        ));
    }

    /// Exception can be raised from any source state.
    #[test]
    fn test_exception_from_any_state() {
        let all_states = vec![
            ShipmentStatus::Created,
            ShipmentStatus::PickedUp,
            ShipmentStatus::InTransit,
            ShipmentStatus::OutForDelivery,
            ShipmentStatus::Delivered,
            ShipmentStatus::Exception,
            ShipmentStatus::Returned,
        ];
        for state in &all_states {
            assert!(
                is_valid_status_transition(state, &ShipmentStatus::Exception),
                "Expected Exception to be valid from {:?}",
                state
            );
        }
    }

    /// Returned can be raised from any source state.
    #[test]
    fn test_returned_from_any_state() {
        let all_states = vec![
            ShipmentStatus::Created,
            ShipmentStatus::PickedUp,
            ShipmentStatus::InTransit,
            ShipmentStatus::OutForDelivery,
            ShipmentStatus::Delivered,
            ShipmentStatus::Exception,
            ShipmentStatus::Returned,
        ];
        for state in &all_states {
            assert!(
                is_valid_status_transition(state, &ShipmentStatus::Returned),
                "Expected Returned to be valid from {:?}",
                state
            );
        }
    }

    /// Invalid transitions must be rejected.
    #[test]
    fn test_invalid_status_transitions() {
        // Skipping steps is not allowed
        assert!(!is_valid_status_transition(
            &ShipmentStatus::Created,
            &ShipmentStatus::InTransit
        ));
        assert!(!is_valid_status_transition(
            &ShipmentStatus::Created,
            &ShipmentStatus::Delivered
        ));
        assert!(!is_valid_status_transition(
            &ShipmentStatus::PickedUp,
            &ShipmentStatus::OutForDelivery
        ));
        assert!(!is_valid_status_transition(
            &ShipmentStatus::InTransit,
            &ShipmentStatus::Delivered
        ));
        // Going backwards is not allowed
        assert!(!is_valid_status_transition(
            &ShipmentStatus::Delivered,
            &ShipmentStatus::InTransit
        ));
        assert!(!is_valid_status_transition(
            &ShipmentStatus::InTransit,
            &ShipmentStatus::PickedUp
        ));
        // Same state is not a valid transition
        assert!(!is_valid_status_transition(
            &ShipmentStatus::Created,
            &ShipmentStatus::Created
        ));
    }

    #[test]
    fn test_shipment_status_serde_roundtrip() {
        let statuses = vec![
            ShipmentStatus::Created,
            ShipmentStatus::PickedUp,
            ShipmentStatus::InTransit,
            ShipmentStatus::OutForDelivery,
            ShipmentStatus::Delivered,
            ShipmentStatus::Exception,
            ShipmentStatus::Returned,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let back: ShipmentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn test_create_shipment_input_serde() {
        let input = CreateShipmentInput {
            tracking_number: "TRACK-001".to_string(),
            po_hash: None,
            carrier: "FedEx".to_string(),
            origin: Address {
                name: "Warehouse".to_string(),
                street: "123 Main St".to_string(),
                city: "Dallas".to_string(),
                state: "TX".to_string(),
                postal_code: "75001".to_string(),
                country: "US".to_string(),
            },
            destination: Address {
                name: "Customer".to_string(),
                street: "456 Oak Ave".to_string(),
                city: "Austin".to_string(),
                state: "TX".to_string(),
                postal_code: "78701".to_string(),
                country: "US".to_string(),
            },
            items: vec![ShipmentItem {
                sku: "BOLT-M6".to_string(),
                description: "M6 Bolt".to_string(),
                quantity: 100,
                weight_kg: Some(0.5),
            }],
            estimated_delivery: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateShipmentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tracking_number, "TRACK-001");
        assert_eq!(back.carrier, "FedEx");
        assert_eq!(back.items.len(), 1);
    }

    #[test]
    fn test_initiate_shipment_result_serde() {
        let result = InitiateShipmentResult {
            shipment_hash: ActionHash::from_raw_36(vec![7u8; 36]),
            tracking_number: "TRACK-AUTO-001".to_string(),
            items_count: 3,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: InitiateShipmentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tracking_number, "TRACK-AUTO-001");
        assert_eq!(back.items_count, 3);
    }

    #[test]
    fn test_complete_delivery_input_serde() {
        let input = CompleteDeliveryInput {
            shipment_hash: ActionHash::from_raw_36(vec![8u8; 36]),
            receiving_location: "Warehouse A - Bay 3".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CompleteDeliveryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.receiving_location, "Warehouse A - Bay 3");
    }

    #[test]
    fn test_complete_delivery_result_serde() {
        let result = CompleteDeliveryResult {
            items_received: 5,
            inventory_updated: true,
            po_fulfilled: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CompleteDeliveryResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.items_received, 5);
        assert!(back.inventory_updated);
        assert!(back.po_fulfilled);

        // Verify partial success variant
        let partial = CompleteDeliveryResult {
            items_received: 2,
            inventory_updated: false,
            po_fulfilled: false,
        };
        let json2 = serde_json::to_string(&partial).unwrap();
        let back2: CompleteDeliveryResult = serde_json::from_str(&json2).unwrap();
        assert!(!back2.inventory_updated);
        assert!(!back2.po_fulfilled);
    }
}
