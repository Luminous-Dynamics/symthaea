// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::consciousness::use_consciousness;
use mycelix_leptos_core::SovereignRadar;
use crate::context::use_supplychain_context;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let ctx = use_supplychain_context();

    view! {
        <div class="page-profile">
            <h1>"Supply Chain Profile"</h1>
            <div class="profile-section">
                <h2>"Sovereign Profile"</h2>
                <SovereignRadar />
            </div>
            <div class="profile-section">
                <h2>"Activity"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Inventory SKUs"</span>
                        <span class="dash-value">{move || ctx.inventory.get().len().to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Orders"</span>
                        <span class="dash-value">{move || ctx.orders.get().len().to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Provenance Claims"</span>
                        <span class="dash-value">{move || ctx.provenance.get().len().to_string()}</span>
                    </div>
                </div>
            </div>
        </div>
    }
}
