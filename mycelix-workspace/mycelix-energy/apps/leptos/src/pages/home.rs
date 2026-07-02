// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
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
            <section class="hero">
                <h1>"Energy"</h1>
                <p class="hero-subtitle">"Community-owned renewable energy for South Africa."</p>
                <div class="hero-cta">
                    <A href="/projects" attr:class="btn btn-primary">"See Projects"</A>
                    <A href="/grid" attr:class="btn btn-ghost">"Trade Energy"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"⚡"</span>
                        <h3>"Generate"</h3>
                        <p>"Solar, wind, hydro projects"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🔄"</span>
                        <h3>"Trade"</h3>
                        <p>"P2P energy marketplace"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🏘️"</span>
                        <h3>"Own"</h3>
                        <p>"Regenerative community ownership"</p>
                    </div>
                </div>
            </section>

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
