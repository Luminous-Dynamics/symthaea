// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use crate::context::use_attribution_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_attribution_context();

    view! {
        <div class="page-home">
            <section class="hero">
                <h1>"Attribution"</h1>
                <p class="hero-subtitle">"Honor every dependency. Reciprocate value."</p>
                <div class="hero-cta">
                    <A href="/registry" attr:class="btn btn-primary">"Browse Registry"</A>
                    <A href="/reciprocity" attr:class="btn btn-ghost">"Make a Pledge"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"📦"</span>
                        <h3>"Register"</h3>
                        <p>"Track your dependencies"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"✅"</span>
                        <h3>"Attest"</h3>
                        <p>"Prove your usage"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🔁"</span>
                        <h3>"Reciprocate"</h3>
                        <p>"Give back to maintainers"</p>
                    </div>
                </div>
            </section>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Dependencies"</span>
                    <span class="dash-value">{move || ctx.dependencies.get().len().to_string()}</span>
                    <span class="dash-sub">"registered"</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Pledges"</span>
                    <span class="dash-value">{move || ctx.pledges.get().len().to_string()}</span>
                    <span class="dash-sub">"reciprocity"</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Usage Receipts"</span>
                    <span class="dash-value">{move || ctx.receipts.get().len().to_string()}</span>
                    <span class="dash-sub">"attested"</span>
                </div>
            </div>
        </div>
    }
}
