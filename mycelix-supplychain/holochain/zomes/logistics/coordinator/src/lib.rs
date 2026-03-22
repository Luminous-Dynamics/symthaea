//! Logistics Coordinator Zome - Business logic for shipments
use hdk::prelude::*;
use logistics_integrity::*;

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
    create_link(sender_hash, action_hash.clone(), LinkTypes::SenderToShipments, ())?;

    // Link from recipient
    let recipient_path = Path::from(format!("recipient/{}", shipment.recipient));
    let recipient_hash = ensure_path(recipient_path, LinkTypes::RecipientToShipments)?;
    create_link(recipient_hash, action_hash.clone(), LinkTypes::RecipientToShipments, ())?;

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
        (_, ShipmentStatus::Returned) => true, // Return can happen from any state
        _ => false,
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddTrackingEventInput {
    pub shipment_hash: ActionHash,
    pub status: ShipmentStatus,
    pub location: Option<String>,
    pub description: String,
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
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Shipment not found".to_string()
        )))?;

    let shipment: Shipment = shipment_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Invalid shipment entry".to_string()
        )))?;

    // Validate state transition
    if !is_valid_status_transition(&shipment.status, &input.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid status transition from {:?} to {:?}", shipment.status, input.status)
        )));
    }

    let event = TrackingEvent {
        shipment_hash: input.shipment_hash.clone(),
        status: input.status.clone(),
        location: input.location,
        description: input.description,
        occurred_at: sys_time()?,
        reported_by: agent_info()?.agent_initial_pubkey,
    };

    let action_hash = create_entry(EntryTypes::TrackingEvent(event.clone()))?;
    create_link(input.shipment_hash.clone(), action_hash.clone(), LinkTypes::ShipmentToEvents, ())?;

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
    let links = get_links(LinkQuery::new(shipment_hash, filter), GetStrategy::default())?;
    let mut events = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(event) = record.entry().to_app_option::<TrackingEvent>().map_err(|e| wasm_error!(e))? {
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
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Test every valid status transition (from the `is_valid_status_transition` function).
    #[test]
    fn test_valid_status_transitions() {
        // Normal forward chain
        assert!(is_valid_status_transition(&ShipmentStatus::Created, &ShipmentStatus::PickedUp));
        assert!(is_valid_status_transition(&ShipmentStatus::PickedUp, &ShipmentStatus::InTransit));
        assert!(is_valid_status_transition(&ShipmentStatus::InTransit, &ShipmentStatus::OutForDelivery));
        assert!(is_valid_status_transition(&ShipmentStatus::OutForDelivery, &ShipmentStatus::Delivered));
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
                "Expected Exception to be valid from {:?}", state
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
                "Expected Returned to be valid from {:?}", state
            );
        }
    }

    /// Invalid transitions must be rejected.
    #[test]
    fn test_invalid_status_transitions() {
        // Skipping steps is not allowed
        assert!(!is_valid_status_transition(&ShipmentStatus::Created, &ShipmentStatus::InTransit));
        assert!(!is_valid_status_transition(&ShipmentStatus::Created, &ShipmentStatus::Delivered));
        assert!(!is_valid_status_transition(&ShipmentStatus::PickedUp, &ShipmentStatus::OutForDelivery));
        assert!(!is_valid_status_transition(&ShipmentStatus::InTransit, &ShipmentStatus::Delivered));
        // Going backwards is not allowed
        assert!(!is_valid_status_transition(&ShipmentStatus::Delivered, &ShipmentStatus::InTransit));
        assert!(!is_valid_status_transition(&ShipmentStatus::InTransit, &ShipmentStatus::PickedUp));
        // Same state is not a valid transition
        assert!(!is_valid_status_transition(&ShipmentStatus::Created, &ShipmentStatus::Created));
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
}
