// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_supplychain_context;

#[component]
pub fn TrustPage() -> impl IntoView {
    let ctx = use_supplychain_context();

    view! {
        <div class="page-trust">
            <h1>"Supplier Trust"</h1>
            <p class="subtitle">"Reputation scores and verification status."</p>

            <div class="supplier-list">
                {move || ctx.suppliers.get().iter().map(|s| {
                    let name = s.name.clone();
                    let score = s.trust_score;
                    let verified = s.verified;
                    let orders = s.total_orders;
                    let on_time = s.on_time_rate * 100.0;
                    view! {
                        <div class="supplier-card">
                            <div class="supplier-header">
                                <h3 class="supplier-name">{name}</h3>
                                {verified.then(|| view! { <span class="badge badge-success">"Verified"</span> })}
                            </div>
                            <div class="supplier-stats">
                                <span>{format!("Trust: {score:.2}")}</span>
                                <span>{format!("{orders} orders")}</span>
                                <span>{format!("{on_time:.0}% on-time")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
