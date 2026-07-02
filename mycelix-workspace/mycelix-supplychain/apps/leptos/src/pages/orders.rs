// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_supplychain_context;

#[component]
pub fn OrdersPage() -> impl IntoView {
    let ctx = use_supplychain_context();

    view! {
        <div class="page-orders">
            <h1>"Purchase Orders"</h1>
            <div class="order-list">
                {move || ctx.orders.get().iter().map(|o| {
                    let id = o.id.clone();
                    let amount = o.total_amount;
                    let currency = o.currency.clone();
                    let status = o.status.label();
                    let items = o.items.join(", ");
                    view! {
                        <div class="order-card">
                            <div class="order-header">
                                <span class="order-id">{id}</span>
                                <span class="badge">{status}</span>
                            </div>
                            <span class="order-items">{items}</span>
                            <span class="order-amount">{format!("{currency} {amount:.0}")}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
