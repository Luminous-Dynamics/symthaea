// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::ConnectionStatus;
use mycelix_leptos_core::holochain_provider;
use crate::context::use_energy_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let hc = holochain_provider::use_holochain();
    let ctx = use_energy_context();

    let total_mw = move || {
        ctx.projects.get().iter().map(|p| p.capacity_mw).sum::<f64>()
    };

    view! {
        <div class="page-home">
            <h1>"Energy"</h1>
            <p class="subtitle">"Renewable energy projects, P2P trading, and regenerative ownership."</p>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Total Capacity"</span>
                    <span class="dash-value">{move || format!("{:.0} MW", total_mw())}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Projects"</span>
                    <span class="dash-value">{move || ctx.projects.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Active Offers"</span>
                    <span class="dash-value">{move || ctx.offers.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Investments"</span>
                    <span class="dash-value">{move || ctx.investments.get().len().to_string()}</span>
                </div>
            </div>

            <span class="status-footer">{move || match hc.status.get() {
                ConnectionStatus::Mock => "Mock mode",
                ConnectionStatus::Connected => "Connected",
                _ => "",
            }}</span>
        </div>
    }
}
