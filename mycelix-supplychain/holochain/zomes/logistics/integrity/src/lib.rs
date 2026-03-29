// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Logistics Integrity Zome - Entry types for shipments and tracking
use hdi::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ShipmentStatus { Created, PickedUp, InTransit, OutForDelivery, Delivered, Exception, Returned }

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Shipment {
    pub tracking_number: String,
    pub po_hash: Option<ActionHash>,
    pub carrier: String,
    pub origin: Address,
    pub destination: Address,
    pub items: Vec<ShipmentItem>,
    pub status: ShipmentStatus,
    pub estimated_delivery: Option<Timestamp>,
    pub actual_delivery: Option<Timestamp>,
    pub created_at: Timestamp,
    pub sender: AgentPubKey,
    pub recipient: AgentPubKey,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Address {
    pub name: String,
    pub street: String,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub country: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShipmentItem {
    pub sku: String,
    pub description: String,
    pub quantity: u64,
    pub weight_kg: Option<f64>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrackingEvent {
    pub shipment_hash: ActionHash,
    pub status: ShipmentStatus,
    pub location: Option<String>,
    pub description: String,
    pub occurred_at: Timestamp,
    pub reported_by: AgentPubKey,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Shipment(Shipment),
    TrackingEvent(TrackingEvent),
}

#[hdk_link_types]
pub enum LinkTypes { SenderToShipments, RecipientToShipments, ShipmentToEvents, PoToShipments }

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::Shipment(s) => {
                    if s.tracking_number.is_empty() { return Ok(ValidateCallbackResult::Invalid("Tracking number required".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
