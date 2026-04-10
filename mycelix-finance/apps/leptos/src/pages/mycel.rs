// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn MycelPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-mycel">
            <h1>"MYCEL Score"</h1>
            <p class="subtitle">"Soulbound reputation \u{2014} you cannot buy this."</p>

            <div class="mycel-display">
                <span class=move || format!("mycel-score {}", ctx.mycel_score.get().tier.css_class())>
                    {move || format!("{:.2}", ctx.mycel_score.get().score)}
                </span>
                <span class="mycel-tier">{move || ctx.mycel_score.get().tier.label()}</span>
            </div>

            <div class="mycel-breakdown">
                <h2>"Breakdown"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Participation"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.mycel_score.get().participation)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Recognition"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.mycel_score.get().recognition)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Validation"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.mycel_score.get().validation)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Longevity"</span>
                        <span class="dash-value">{move || format!("{:.2}", ctx.mycel_score.get().longevity)}</span>
                    </div>
                </div>
                <span class="mycel-months">{move || format!("{} active months", ctx.mycel_score.get().active_months)}</span>
            </div>
        </div>
    }
}
