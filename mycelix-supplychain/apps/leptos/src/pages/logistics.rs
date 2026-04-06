// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_supplychain_context;

#[component]
pub fn LogisticsPage() -> impl IntoView {
    let ctx = use_supplychain_context();

    view! {
        <div class="page-logistics">
            <h1>"Logistics"</h1>
            <p class="subtitle">"Shipment tracking and delivery."</p>

            {move || ctx.shipments.get().iter().map(|s| {
                let origin = s.origin.clone();
                let dest = s.destination.clone();
                let carrier = s.carrier.clone();
                let status = s.status.clone();
                let order = s.order_id.clone();
                view! {
                    <div class="shipment-card">
                        <div class="shipment-route">
                            <span class="shipment-origin">{origin}</span>
                            <span class="shipment-arrow">" \u{2192} "</span>
                            <span class="shipment-dest">{dest}</span>
                        </div>
                        <div class="shipment-meta">
                            <span>{format!("Carrier: {carrier}")}</span>
                            <span>{format!("Order: {order}")}</span>
                            <span class="badge">{status}</span>
                        </div>
                    </div>
                }
            }).collect_view()}
        </div>
    }
}
