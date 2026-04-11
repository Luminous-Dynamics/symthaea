// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;
use mycelix_leptos_core::consciousness::use_consciousness;
use mycelix_leptos_core::SovereignRadar;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let ctx = use_finance_context();

    view! {
        <div class="page-profile">
            <h1>"Finance Profile"</h1>

            <div class="profile-section">
                <h2>"Identity"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Civic Tier"</span>
                        <span class="dash-value">{move || consciousness.tier.get().label()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"MYCEL Tier"</span>
                        <span class="dash-value">{move || ctx.mycel_score.get().tier.label()}</span>
                    </div>
                </div>
            </div>

            <div class="profile-section">
                <h2>"Sovereign Profile"</h2>
                <SovereignRadar />
            </div>

            <div class="profile-section">
                <h2>"Balances"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"TEND"</span>
                        <span class="dash-value">{move || format!("{:+}h", ctx.tend_balance.get().balance)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"SAP"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.sap_balance.get().display_balance())}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"MYCEL"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.mycel_score.get().score)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Staked"</span>
                        <span class="dash-value">{move || {
                            let total: u64 = ctx.stakes.get().iter().map(|s| s.sap_amount).sum();
                            format!("{:.2} SAP", total as f64 / 1_000_000.0)
                        }}</span>
                    </div>
                </div>
            </div>
        </div>
    }
}
