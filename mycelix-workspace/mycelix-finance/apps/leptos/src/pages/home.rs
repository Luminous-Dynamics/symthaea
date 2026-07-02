// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_core::{
    ActivityFeed, ActivityFeedItem, AvailabilityState, AvailabilityStateKind, FreshnessBadge,
    FreshnessLevel,
};
use mycelix_leptos_core::holochain_provider::use_holochain;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_finance_context();
    let hc = use_holochain();
    let freshness_hc = hc.clone();
    let availability_hc = hc.clone();
    let has_activity = Memo::new(move |_| {
        !ctx.sap_payments.get().is_empty()
            || !ctx.tend_exchanges.get().is_empty()
            || !ctx.recognitions.get().is_empty()
    });
    let activity_feed = move || {
        let mut items = Vec::new();

        items.extend(
            ctx.sap_payments
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, payment)| ActivityFeedItem {
                    id: format!("finance-payment-{index}-{}", payment.id),
                    domain_label: "Payments".into(),
                    description: format!(
                        "{} {} micro-SAP [{}]",
                        payment.memo.clone().unwrap_or_else(|| "Payment".into()),
                        payment.amount,
                        payment.status.label()
                    ),
                    emphasis_class: Some("activity-feed-live".into()),
                }),
        );

        items.extend(
            ctx.tend_exchanges
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, exchange)| ActivityFeedItem {
                    id: format!("finance-exchange-{index}-{}", exchange.id),
                    domain_label: "TEND".into(),
                    description: format!(
                        "{}h {} [{}]",
                        exchange.hours,
                        exchange.service_description,
                        exchange.status.label()
                    ),
                    emphasis_class: Some("activity-feed-live".into()),
                }),
        );

        items.extend(
            ctx.recognitions
                .get()
                .into_iter()
                .take(2)
                .enumerate()
                .map(|(index, recognition)| ActivityFeedItem {
                    id: format!("finance-recognition-{index}-{}", recognition.hash),
                    domain_label: "Recognition".into(),
                    description: format!(
                        "{} recognition from {}",
                        recognition.contribution_type.label(),
                        recognition.recognizer_did
                    ),
                    emphasis_class: Some("activity-feed-success".into()),
                }),
        );

        items.truncate(6);
        items
    };

    view! {
        <div class="page-home">
            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || {
                    if freshness_hc.is_mock() {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Mock finance posture" /> }.into_any()
                    } else if has_activity.get() {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Live finance posture without timestamped freshness yet" /> }.into_any()
                    } else {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="No finance activity loaded yet" /> }.into_any()
                    }
                }}
            </div>

            {move || {
                if availability_hc.is_mock() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Finance Posture"
                            description="Finance is rendering shared shell posture, but the current balances and flows are still sourced from local mock data."
                            action={None}
                        />
                    }.into_any()
                } else if !has_activity.get() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Empty
                            title="Live Finance, Empty Activity"
                            description="The finance shell is connected to a live conductor, but member-scoped payments, exchanges, and recognitions have not populated yet."
                            action={Some(view! {
                                <A href="/tend" attr:class="btn btn-primary">"Open TEND"</A>
                            }.into_any())}
                        />
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <section class="hero">
                <h1>"Finance"</h1>
                <p class="hero-subtitle">"Three currencies. One commons. No extraction."</p>
                <div class="hero-cta">
                    <A href="/tend" attr:class="btn btn-primary">"View Balance"</A>
                    <A href="/tend" attr:class="btn btn-ghost">"Give Care"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"🤝"</span>
                        <h3>"TEND"</h3>
                        <p>"Mutual credit \u{2014} 1 hour = 1 TEND"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"💧"</span>
                        <h3>"SAP"</h3>
                        <p>"Transferable value with demurrage"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🍄"</span>
                        <h3>"MYCEL"</h3>
                        <p>"Soulbound reputation"</p>
                    </div>
                </div>
            </section>

            <div class="currency-grid">
                <div class="currency-card tend-card">
                    <span class="currency-icon">"🤝"</span>
                    <span class="currency-name">"TEND"</span>
                    <span class="currency-value">{move || format!("{} hours", ctx.tend_balance.get().balance)}</span>
                    <span class="currency-desc">{move || ctx.tend_balance.get().equilibrium_label()}</span>
                </div>
                <div class="currency-card sap-card">
                    <span class="currency-icon">"💧"</span>
                    <span class="currency-name">"SAP"</span>
                    <span class="currency-value">{move || format!("{:.2}", ctx.sap_balance.get().display_balance())}</span>
                    <span class="currency-desc">"transferable with 2%/yr demurrage"</span>
                </div>
                <div class="currency-card mycel-card">
                    <span class="currency-icon">"🍄"</span>
                    <span class="currency-name">"MYCEL"</span>
                    <span class="currency-value">{move || format!("{:.2}", ctx.mycel_score.get().score)}</span>
                    <span class="currency-desc">{move || ctx.mycel_score.get().tier.label()}</span>
                </div>
            </div>

            <section class="recent-section">
                <h2>"Recent Finance Activity"</h2>
                <ActivityFeed items=activity_feed() />
            </section>
        </div>
    }
}
