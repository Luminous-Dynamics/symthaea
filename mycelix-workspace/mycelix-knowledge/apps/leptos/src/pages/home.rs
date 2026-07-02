// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_core::{
    ActivityFeed, ActivityFeedItem, AvailabilityState, AvailabilityStateKind, ConnectionStatus,
    FreshnessBadge, FreshnessLevel,
};
use mycelix_leptos_core::holochain_provider;
use crate::context::use_knowledge_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let hc = holochain_provider::use_holochain();
    let ctx = use_knowledge_context();
    let has_content = Memo::new(move |_| {
        !ctx.claims.get().is_empty()
            || !ctx.fact_checks.get().is_empty()
            || !ctx.inferences.get().is_empty()
    });
    let activity_feed = move || {
        let mut items = Vec::new();

        items.extend(
            ctx.claims
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, claim)| ActivityFeedItem {
                    id: format!("knowledge-claim-{index}-{}", claim.id),
                    domain_label: "Claim".into(),
                    description: claim.content,
                    emphasis_class: Some("activity-feed-live".into()),
                }),
        );

        items.extend(
            ctx.fact_checks
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, fact_check)| ActivityFeedItem {
                    id: format!("knowledge-factcheck-{index}-{}", fact_check.id),
                    domain_label: "Fact Check".into(),
                    description: format!("{} [{}]", fact_check.statement, fact_check.verdict.label()),
                    emphasis_class: Some("activity-feed-warning".into()),
                }),
        );

        items.extend(
            ctx.inferences
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, inference)| ActivityFeedItem {
                    id: format!("knowledge-inference-{index}-{}", inference.id),
                    domain_label: "Inference".into(),
                    description: inference.conclusion,
                    emphasis_class: Some(
                        if inference.verified {
                            "activity-feed-success"
                        } else {
                            "activity-feed-live"
                        }
                        .into(),
                    ),
                }),
        );

        items.truncate(6);
        items
    };

    view! {
        <div class="page-home">
            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || match hc.status.get() {
                    ConnectionStatus::Mock => {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Mock knowledge posture" /> }.into_any()
                    }
                    ConnectionStatus::Connected => {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Live knowledge posture without timestamped freshness yet" /> }.into_any()
                    }
                    _ => {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Knowledge connection still establishing" /> }.into_any()
                    }
                }}
            </div>

            {move || match hc.status.get() {
                ConnectionStatus::Mock => {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Knowledge Posture"
                            description="Knowledge is using the shared shell contract, but claims and graph signals are still drawn from mock data rather than a live conductor."
                            action={None}
                        />
                    }.into_any()
                }
                ConnectionStatus::Connected if !has_content.get() => {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Empty
                            title="Live Knowledge, Empty Graph"
                            description="The knowledge shell is connected, but no claims, fact checks, or inferences have been loaded into the current posture yet."
                            action={Some(view! {
                                <A href="/browse" attr:class="btn btn-primary">"Browse Claims"</A>
                            }.into_any())}
                        />
                    }.into_any()
                }
                _ => view! { <></> }.into_any(),
            }}

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

            <section class="recent-section">
                <h2>"Recent Knowledge Activity"</h2>
                <ActivityFeed items=activity_feed() />
            </section>
        </div>
    }
}
