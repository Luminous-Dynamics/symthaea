// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Supply Chain.

use leptos::prelude::*;
use supplychain_leptos_types::*;

#[derive(Clone)]
pub struct SupplychainCtx {
    pub inventory: RwSignal<Vec<InventoryItemView>>,
    pub orders: RwSignal<Vec<PurchaseOrderView>>,
    pub shipments: RwSignal<Vec<ShipmentView>>,
    pub suppliers: RwSignal<Vec<SupplierTrustView>>,
    pub provenance: RwSignal<Vec<ProvenanceClaimView>>,
}

pub fn provide_supplychain_context() {
    let state = SupplychainCtx {
        inventory: RwSignal::new(vec![
            InventoryItemView { id: "ITM-001".into(), name: "Solar Panel 400W".into(), sku: "SP-400W".into(), quantity: 250, location: "Cape Town Warehouse".into(), status: ItemStatus::Active, last_movement: 1714000000 },
            InventoryItemView { id: "ITM-002".into(), name: "Wind Turbine Blade 45m".into(), sku: "WTB-45M".into(), quantity: 12, location: "Saldanha Port".into(), status: ItemStatus::InTransit, last_movement: 1714200000 },
            InventoryItemView { id: "ITM-003".into(), name: "Battery Cell LFP 280Ah".into(), sku: "BC-LFP-280".into(), quantity: 1500, location: "Joburg Distribution".into(), status: ItemStatus::Active, last_movement: 1714100000 },
        ]),
        orders: RwSignal::new(vec![
            PurchaseOrderView { id: "PO-001".into(), supplier_did: "did:mycelix:supplier-001".into(), buyer_did: "did:mycelix:user-001".into(), items: vec!["SP-400W x50".into()], total_amount: 125_000.0, currency: "ZAR".into(), status: OrderStatus::Approved },
            PurchaseOrderView { id: "PO-002".into(), supplier_did: "did:mycelix:supplier-002".into(), buyer_did: "did:mycelix:user-001".into(), items: vec!["BC-LFP-280 x500".into()], total_amount: 950_000.0, currency: "ZAR".into(), status: OrderStatus::Shipped },
        ]),
        shipments: RwSignal::new(vec![
            ShipmentView { id: "SH-001".into(), order_id: "PO-002".into(), origin: "Shenzhen, CN".into(), destination: "Joburg, ZA".into(), carrier: "Maersk".into(), status: "In Transit".into(), estimated_arrival: 1715000000 },
        ]),
        suppliers: RwSignal::new(vec![
            SupplierTrustView { did: "did:mycelix:supplier-001".into(), name: "SA Solar Co".into(), trust_score: 0.92, verified: true, total_orders: 47, on_time_rate: 0.95 },
            SupplierTrustView { did: "did:mycelix:supplier-002".into(), name: "Global Battery Corp".into(), trust_score: 0.78, verified: true, total_orders: 23, on_time_rate: 0.87 },
        ]),
        provenance: RwSignal::new(vec![
            ProvenanceClaimView { id: "PV-001".into(), item_id: "ITM-001".into(), claim_type: "Fair Trade".into(), origin: "SA Solar Co, Western Cape".into(), verified: true, certifications: vec!["ISO 14001".into(), "B-Corp".into()] },
        ]),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[SupplyChain] Conductor connected — loading real data...".into());

            // Load inventory items
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "inventory", "get_all_items", &()
            ).await {
                let items: Vec<InventoryItemView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !items.is_empty() {
                    web_sys::console::log_1(&format!("[SupplyChain] Loaded {} inventory items", items.len()).into());
                    state.inventory.set(items);
                }
            }

            // Load purchase orders
            let owner = "did:mycelix:user-001".to_string();
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "inventory", "get_orders_by_buyer", &owner
            ).await {
                let orders: Vec<PurchaseOrderView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !orders.is_empty() {
                    web_sys::console::log_1(&format!("[SupplyChain] Loaded {} orders", orders.len()).into());
                    state.orders.set(orders);
                }
            }

            // Load shipments
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "provenance", "get_shipments_by_buyer", &owner
            ).await {
                let shipments: Vec<ShipmentView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !shipments.is_empty() {
                    web_sys::console::log_1(&format!("[SupplyChain] Loaded {} shipments", shipments.len()).into());
                    state.shipments.set(shipments);
                }
            }

            // Load supplier trust scores
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "provenance", "get_all_suppliers", &()
            ).await {
                let suppliers: Vec<SupplierTrustView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !suppliers.is_empty() {
                    web_sys::console::log_1(&format!("[SupplyChain] Loaded {} suppliers", suppliers.len()).into());
                    state.suppliers.set(suppliers);
                }
            }

            // Load provenance claims
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "provenance", "get_all_claims", &()
            ).await {
                let claims: Vec<ProvenanceClaimView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !claims.is_empty() {
                    web_sys::console::log_1(&format!("[SupplyChain] Loaded {} provenance claims", claims.len()).into());
                    state.provenance.set(claims);
                }
            }
        }
    });
}

/// Extract the entry from a Holochain Record JSON value.
/// Records have structure: { "entry": { "Present": { ...fields... } } }
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_supplychain_context() -> SupplychainCtx {
    expect_context::<SupplychainCtx>()
}
