// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use crate::context::use_space_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_space_context();

    let high_risk = move || ctx.conjunctions.get().iter().filter(|c| c.risk_level == space_leptos_types::RiskLevel::High || c.risk_level == space_leptos_types::RiskLevel::Critical).count();

    view! {
        <div class="page-home">
            <section class="hero">
                <h1>"Space"</h1>
                <p class="hero-subtitle">"Decentralized space situational awareness."</p>
                <div class="hero-cta">
                    <A href="/catalog" attr:class="btn btn-primary">"View Catalog"</A>
                    <A href="/conjunctions" attr:class="btn btn-ghost">"Check Conjunctions"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"🛰️"</span>
                        <h3>"Track"</h3>
                        <p>"Orbital object catalog"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"⚠️"</span>
                        <h3>"Predict"</h3>
                        <p>"Collision probability screening"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🤝"</span>
                        <h3>"Coordinate"</h3>
                        <p>"Multi-party traffic negotiation"</p>
                    </div>
                </div>
            </section>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Tracked Objects"</span>
                    <span class="dash-value">{move || ctx.objects.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Conjunctions"</span>
                    <span class="dash-value">{move || ctx.conjunctions.get().len().to_string()}</span>
                    <span class="dash-sub">{move || format!("{} high risk", high_risk())}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Active Bounties"</span>
                    <span class="dash-value">{move || ctx.bounties.get().iter().filter(|b| b.status == space_leptos_types::BountyStatus::Open).count().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Proposals"</span>
                    <span class="dash-value">{move || ctx.proposals.get().len().to_string()}</span>
                </div>
            </div>
        </div>
    }
}
