// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_supplychain_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_supplychain_context();

    view! {
        <div class="page-home">
            <h1>"Supply Chain"</h1>
            <p class="subtitle">"Provenance tracking, procurement, and trust."</p>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Inventory Items"</span>
                    <span class="dash-value">{move || ctx.inventory.get().iter().map(|i| i.quantity as u64).sum::<u64>().to_string()}</span>
                    <span class="dash-sub">{move || format!("{} SKUs", ctx.inventory.get().len())}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Active Orders"</span>
                    <span class="dash-value">{move || ctx.orders.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"In Transit"</span>
                    <span class="dash-value">{move || ctx.shipments.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Verified Suppliers"</span>
                    <span class="dash-value">{move || ctx.suppliers.get().iter().filter(|s| s.verified).count().to_string()}</span>
                </div>
            </div>
        </div>
    }
}
