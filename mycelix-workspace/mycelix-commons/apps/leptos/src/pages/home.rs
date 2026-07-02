// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::{
    ActivityFeed, ActivityFeedItem, AvailabilityState, AvailabilityStateKind, FreshnessBadge,
    FreshnessLevel,
};

use crate::contexts::commons_context::use_commons;

#[component]
pub fn HomePage() -> impl IntoView {
    let commons = use_commons();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    let hc_for_freshness = hc.clone();
    let hc_for_availability = hc.clone();
    let latest_event = Memo::new(move |_| commons.events.get().into_iter().map(|event| event.start_time).max());
    let feed_items = move || {
        commons
            .events
            .get()
            .into_iter()
            .take(4)
            .map(|event| ActivityFeedItem {
                id: event.hash,
                domain_label: event.category,
                description: format!("{} — {}", event.title, event.description),
                emphasis_class: None,
            })
            .collect::<Vec<_>>()
    };

    view! {
        <div class="page home-page">
            <section class="hero">
                <h1 class="hero-title">"The Commons"</h1>
                <p class="hero-subtitle">
                    "Community-owned resources managed on Holochain. "
                    "Property, housing, care, water, food, and transport — coordinated without middlemen."
                </p>
            </section>

            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || {
                    if hc_for_freshness.is_mock() {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="Mock commons posture" /> }.into_any()
                    } else {
                        latest_event.get()
                            .map(|timestamp| view! { <FreshnessBadge level=freshness_from_secs(timestamp) detail=format!("Events {}", format_relative_secs(timestamp)) /> }.into_any())
                            .unwrap_or_else(|| view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="No commons events yet" /> }.into_any())
                    }
                }}
            </div>

            {move || {
                if hc_for_availability.is_mock() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Commons Mesh"
                            description="Commons is rendering simulated community resource data while conductor-backed mesh endpoints continue to come online."
                            action={None}
                        />
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <section class="dashboard-grid">
                <div class="stat-card">
                    <span class="stat-value">"2,847"</span>
                    <span class="stat-label">"Properties Registered"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"156"</span>
                    <span class="stat-label">"Housing Cooperatives"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"1,203"</span>
                    <span class="stat-label">"Care Commitments Active"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"89%"</span>
                    <span class="stat-label">"Resource Mesh Uptime"</span>
                </div>
            </section>

            <section class="resource-overview">
                <h2>"Recent Commons Activity"</h2>
                <ActivityFeed items=feed_items() />
            </section>

            <section class="resource-overview">
                <h2>"Resource Mesh Status"</h2>
                <div class="mesh-grid">
                    <div class="mesh-card water">
                        <h3>"Water"</h3>
                        <p>"12 community wells monitored"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card food">
                        <h3>"Food"</h3>
                        <p>"34 community gardens, 8 food banks"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card transport">
                        <h3>"Transport"</h3>
                        <p>"67 shared vehicles, 12 routes"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card mutual-aid">
                        <h3>"Mutual Aid"</h3>
                        <p>"423 active commitments"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                </div>
            </section>

            <section class="consciousness-info">
                <h2>"Consciousness-Gated Access"</h2>
                <p>
                    "Resource access is weighted by your participation tier. "
                    "Contribute to the commons to deepen your engagement level."
                </p>
                <div class="tier-cards">
                    <div class="tier observer">"Observer — View resources"</div>
                    <div class="tier participant">"Participant — Request resources"</div>
                    <div class="tier citizen">"Citizen — Manage allocations"</div>
                    <div class="tier steward">"Steward — Govern policies"</div>
                    <div class="tier guardian">"Guardian — Constitutional changes"</div>
                </div>
            </section>
        </div>
    }
}

fn freshness_from_secs(timestamp_secs: i64) -> FreshnessLevel {
    let now_secs = (js_sys::Date::now() / 1000.0) as i64;
    let age_minutes = now_secs.saturating_sub(timestamp_secs) / 60;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn format_relative_secs(timestamp_secs: i64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64((timestamp_secs * 1000) as f64));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}
