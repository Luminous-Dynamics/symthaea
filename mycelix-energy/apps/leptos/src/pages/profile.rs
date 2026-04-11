// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::consciousness::use_consciousness;
use mycelix_leptos_core::SovereignRadar;
use crate::context::use_energy_context;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let ctx = use_energy_context();

    let total_invested = move || {
        ctx.investments.get().iter().map(|i| i.amount).sum::<f64>()
    };

    view! {
        <div class="page-profile">
            <h1>"Energy Profile"</h1>

            <div class="profile-section">
                <h2>"Sovereign Profile"</h2>
                <SovereignRadar />
            </div>

            <div class="profile-section">
                <h2>"Portfolio"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Total Invested"</span>
                        <span class="dash-value">{move || format!("R {:.0}", total_invested())}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Positions"</span>
                        <span class="dash-value">{move || ctx.investments.get().len().to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Projects"</span>
                        <span class="dash-value">{move || ctx.projects.get().len().to_string()}</span>
                    </div>
                </div>
            </div>
        </div>
    }
}
