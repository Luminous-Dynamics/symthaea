// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe types for the Supply Chain cluster frontend.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ItemStatus { Active, InTransit, Delivered, Recalled, Archived }
impl ItemStatus {
    pub fn label(&self) -> &'static str {
        match self { Self::Active => "Active", Self::InTransit => "In Transit", Self::Delivered => "Delivered", Self::Recalled => "Recalled", Self::Archived => "Archived" }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryItemView {
    pub id: String,
    pub name: String,
    pub sku: String,
    pub quantity: u32,
    pub location: String,
    pub status: ItemStatus,
    pub last_movement: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderStatus { Draft, Submitted, Approved, Shipped, Delivered, Cancelled }
impl OrderStatus {
    pub fn label(&self) -> &'static str {
        match self { Self::Draft => "Draft", Self::Submitted => "Submitted", Self::Approved => "Approved", Self::Shipped => "Shipped", Self::Delivered => "Delivered", Self::Cancelled => "Cancelled" }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PurchaseOrderView {
    pub id: String,
    pub supplier_did: String,
    pub buyer_did: String,
    pub items: Vec<String>,
    pub total_amount: f64,
    pub currency: String,
    pub status: OrderStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShipmentView {
    pub id: String,
    pub order_id: String,
    pub origin: String,
    pub destination: String,
    pub carrier: String,
    pub status: String,
    pub estimated_arrival: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupplierTrustView {
    pub did: String,
    pub name: String,
    pub trust_score: f64,
    pub verified: bool,
    pub total_orders: u32,
    pub on_time_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceClaimView {
    pub id: String,
    pub item_id: String,
    pub claim_type: String,
    pub origin: String,
    pub verified: bool,
    pub certifications: Vec<String>,
}
