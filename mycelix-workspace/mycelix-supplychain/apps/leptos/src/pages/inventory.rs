// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_supplychain_context;

#[component]
pub fn InventoryPage() -> impl IntoView {
    let ctx = use_supplychain_context();

    view! {
        <div class="page-inventory">
            <h1>"Inventory"</h1>
            <div class="item-list">
                {move || ctx.inventory.get().iter().map(|item| {
                    let name = item.name.clone();
                    let sku = item.sku.clone();
                    let qty = item.quantity;
                    let loc = item.location.clone();
                    let status = item.status.label();
                    view! {
                        <div class="item-card">
                            <div class="item-header">
                                <h3 class="item-name">{name}</h3>
                                <span class="badge">{status}</span>
                            </div>
                            <div class="item-body">
                                <span>{format!("SKU: {sku} | Qty: {qty}")}</span>
                                <span class="item-location">{loc}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
