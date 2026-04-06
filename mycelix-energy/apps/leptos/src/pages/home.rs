// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::ConnectionStatus;
use mycelix_leptos_core::holochain_provider;
use mycelix_leptos_core::consciousness::use_consciousness;
use crate::context::use_energy_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let hc = holochain_provider::use_holochain();
    let consciousness = use_consciousness();
    let ctx = use_energy_context();

    view! {
        <div class="page-home">
            <h1>"Energy"</h1>
            <p class="subtitle">"Welcome to the Energy cluster."</p>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Status"</span>
                    <span class="dash-value">
                        {move || match hc.status.get() {
                            ConnectionStatus::Connected => "Connected",
                            ConnectionStatus::Mock => "Mock mode",
                            ConnectionStatus::Connecting => "Connecting...",
                            ConnectionStatus::Disconnected => "Disconnected",
                        }}
                    </span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Tier"</span>
                    <span class="dash-value">
                        {move || consciousness.tier.get().label()}
                    </span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Items"</span>
                    <span class="dash-value">
                        {move || ctx.items.get().len().to_string()}
                    </span>
                    <span class="dash-sub">"mock data"</span>
                </div>
            </div>
        </div>
    }
}
