// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_core::ConnectionStatus;
use mycelix_leptos_core::holochain_provider;
use mycelix_leptos_core::consciousness::use_consciousness;
use crate::context::use_climate_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let hc = holochain_provider::use_holochain();
    let consciousness = use_consciousness();
    let ctx = use_climate_context();

    view! {
        <div class="page-home">
            <section class="hero">
                <h1>"Climate"</h1>
                <p class="hero-subtitle">"Track carbon. Trade credits. Restore the earth."</p>
                <div class="hero-cta">
                    <A href="/projects" attr:class="btn btn-primary">"Explore Projects"</A>
                    <A href="/emissions" attr:class="btn btn-ghost">"Record Footprint"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"📏"</span>
                        <h3>"Measure"</h3>
                        <p>"Record Scope 1/2/3 emissions"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🌱"</span>
                        <h3>"Reduce"</h3>
                        <p>"Fund climate projects"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"♻️"</span>
                        <h3>"Offset"</h3>
                        <p>"Retire carbon credits"</p>
                    </div>
                </div>
            </section>

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
                    <span class="dash-label">"Carbon Credits"</span>
                    <span class="dash-value">"0"</span>
                    <span class="dash-sub">"mock data"</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Projects"</span>
                    <span class="dash-value">
                        {move || ctx.projects.get().len().to_string()}
                    </span>
                    <span class="dash-sub">"active"</span>
                </div>
            </div>
        </div>
    }
}
