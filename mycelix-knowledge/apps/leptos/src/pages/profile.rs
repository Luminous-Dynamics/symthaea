// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::consciousness::use_consciousness;
use mycelix_leptos_core::SovereignRadar;
use crate::context::use_knowledge_context;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let ctx = use_knowledge_context();

    view! {
        <div class="page-profile">
            <h1>"Knowledge Profile"</h1>

            <div class="profile-section">
                <h2>"Sovereign Profile"</h2>
                <SovereignRadar />
            </div>

            <div class="profile-section">
                <h2>"Contributions"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Claims Submitted"</span>
                        <span class="dash-value">{move || ctx.claims.get().len().to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Fact Checks"</span>
                        <span class="dash-value">{move || ctx.fact_checks.get().len().to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Inferences"</span>
                        <span class="dash-value">{move || ctx.inferences.get().len().to_string()}</span>
                    </div>
                </div>
            </div>
        </div>
    }
}
