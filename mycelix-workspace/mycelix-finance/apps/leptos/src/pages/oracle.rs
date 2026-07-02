// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn OraclePage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-oracle">
            <h1>"Oracle"</h1>
            <p class="subtitle">"Network vitality and credit limits."</p>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Vitality"</span>
                    <span class="dash-value">{move || ctx.oracle_state.get().vitality.to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Tier"</span>
                    <span class=move || format!("dash-value {}", ctx.oracle_state.get().tier.css_class())>
                        {move || ctx.oracle_state.get().tier.label()}
                    </span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Credit Limit"</span>
                    <span class="dash-value">{move || format!("{} TEND", ctx.oracle_state.get().tier.credit_limit())}</span>
                </div>
            </div>
        </div>
    }
}
