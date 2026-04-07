// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_core::ConnectionStatus;
use mycelix_leptos_core::holochain_provider;
use crate::context::use_knowledge_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let hc = holochain_provider::use_holochain();
    let ctx = use_knowledge_context();

    view! {
        <div class="page-home">
            <section class="hero">
                <h1>"Knowledge"</h1>
                <p class="hero-subtitle">"Epistemic commons \u{2014} claims verified by the community."</p>
                <div class="hero-cta">
                    <A href="/browse" attr:class="btn btn-primary">"Browse Claims"</A>
                    <A href="/fact-check" attr:class="btn btn-ghost">"Fact Check"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"📝"</span>
                        <h3>"Claim"</h3>
                        <p>"Submit knowledge claims with sources"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🔍"</span>
                        <h3>"Verify"</h3>
                        <p>"Community fact-checking"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🕸️"</span>
                        <h3>"Connect"</h3>
                        <p>"Build the knowledge graph"</p>
                    </div>
                </div>
            </section>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <span class="dash-label">"Claims"</span>
                    <span class="dash-value">{move || ctx.claims.get().len().to_string()}</span>
                </div>
                <div class="dash-card">
                    <span class="dash-label">"Relationships"</span>
                    <span class="dash-value">{move || ctx.graph_stats.get().relationship_count.to_string()}</span>
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

            <div class="recent-section">
                <h2>"Recent Claims"</h2>
                {move || ctx.claims.get().iter().take(3).map(|c| {
                    let content = c.content.clone();
                    let e = c.classification.empirical.label();
                    let n = c.classification.normative.label();
                    let m = c.classification.materiality.label();
                    let ct = c.claim_type.label();
                    view! {
                        <div class="claim-card-mini">
                            <span class="claim-type-badge">{ct}</span>
                            <p class="claim-content">{content}</p>
                            <div class="epistemic-tags">
                                <span class="e-tag">{format!("E: {e}")}</span>
                                <span class="n-tag">{format!("N: {n}")}</span>
                                <span class="m-tag">{format!("M: {m}")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>

            <span class="status-footer">{move || match hc.status.get() {
                ConnectionStatus::Mock => "Mock mode",
                ConnectionStatus::Connected => "Connected",
                _ => "",
            }}</span>
        </div>
    }
}
