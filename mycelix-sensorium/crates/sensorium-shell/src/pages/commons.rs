// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commons domain — excellent: resource overview, mutual aid, water, housing.

use domain_commons::types::*;
use leptos::prelude::*;
use sensorium_viz::{bar_chart::Bar, pie_chart::Segment, BarChart, PieChart};

#[derive(Clone, Copy, PartialEq)]
enum CommonsView {
    Overview,
    Housing,
    MutualAid,
    Water,
    Food,
}

#[component]
pub fn CommonsOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(CommonsView::Overview);
    let aid_requests: RwSignal<Vec<AidRequest>> = RwSignal::new(vec![
        AidRequest {
            id: "ar-1".into(),
            requester_did: "did:maria".into(),
            request_type: AidRequestType::Childcare,
            description: "Need childcare coverage Tuesday-Thursday while recovering from surgery."
                .into(),
            urgency: Urgency::High,
            location: Some("Block 3".into()),
            amount_needed: None,
            fulfilled_amount: 0,
            status: AidRequestStatus::Open,
            created_at: 1711900800,
        },
        AidRequest {
            id: "ar-2".into(),
            requester_did: "did:james".into(),
            request_type: AidRequestType::Food,
            description: "Family of 5, temporary food insecurity after job transition.".into(),
            urgency: Urgency::Medium,
            location: Some("Block 7".into()),
            amount_needed: Some(200),
            fulfilled_amount: 120,
            status: AidRequestStatus::PartiallyFulfilled,
            created_at: 1711814400,
        },
        AidRequest {
            id: "ar-3".into(),
            requester_did: "did:community".into(),
            request_type: AidRequestType::Transportation,
            description: "Van needed for elderly transport to health center (3x/week).".into(),
            urgency: Urgency::Medium,
            location: None,
            amount_needed: Some(500),
            fulfilled_amount: 500,
            status: AidRequestStatus::Fulfilled,
            created_at: 1711728000,
        },
    ]);

    fn urgency_color(u: &Urgency) -> &'static str {
        match u {
            Urgency::Critical => "#ef4444",
            Urgency::High => "#f59e0b",
            Urgency::Medium => "#60a5fa",
            Urgency::Low => "#6b7280",
        }
    }
    fn status_label(s: &AidRequestStatus) -> &'static str {
        match s {
            AidRequestStatus::Open => "OPEN",
            AidRequestStatus::PartiallyFulfilled => "PARTIAL",
            AidRequestStatus::Fulfilled => "FULFILLED",
            AidRequestStatus::Cancelled => "CANCELLED",
        }
    }

    view! {
        <div class="commons-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == CommonsView::Overview { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(CommonsView::Overview)>"Overview"</button>
                <button class=move || if active_view.get() == CommonsView::Housing { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(CommonsView::Housing)>"Housing"</button>
                <button class=move || if active_view.get() == CommonsView::MutualAid { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(CommonsView::MutualAid)>"Mutual Aid"</button>
                <button class=move || if active_view.get() == CommonsView::Water { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(CommonsView::Water)>"Water"</button>
                <button class=move || if active_view.get() == CommonsView::Food { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(CommonsView::Food)>"Food"</button>
            </div>

            <Show when=move || active_view.get() == CommonsView::Overview>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #059669">"HOUSING"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"142"</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Units (12 buildings)"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #06b6d4">"WATER"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"8"</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Active sources"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #f59e0b">"AID"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || aid_requests.get().iter().filter(|r| r.status == AidRequestStatus::Open).count().to_string()}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Open requests"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"FOOD"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"3"</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Active markets"</p>
                    </div>
                </div>
                <h3 class="section-title">"Resource Distribution"</h3>
                <PieChart segments=vec![
                    Segment { label: "Housing".into(), value: 42.0, color: "#059669".into() },
                    Segment { label: "Water".into(), value: 18.0, color: "#06b6d4".into() },
                    Segment { label: "Food".into(), value: 25.0, color: "#f59e0b".into() },
                    Segment { label: "Transport".into(), value: 15.0, color: "#8b5cf6".into() },
                ] />
            </Show>

            <Show when=move || active_view.get() == CommonsView::MutualAid>
                <div class="thought-list">
                    {move || { aid_requests.get().iter().map(|r| {
                        let desc = r.description.clone();
                        let rtype = format!("{:?}", r.request_type);
                        let ucolor = urgency_color(&r.urgency);
                        let status = status_label(&r.status);
                        let fulfilled = r.fulfilled_amount;
                        let needed = r.amount_needed;
                        let location = r.location.clone().unwrap_or_default();
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {ucolor};")>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {ucolor}")>{format!("{:?}", r.urgency)}</span>
                                    <span class="thought-domain">{rtype}</span>
                                    <span class="thought-confidence">{status}</span>
                                </div>
                                <p class="thought-content">{desc}</p>
                                <div style="font-size: 0.7rem; color: var(--text-muted); display: flex; gap: 0.5rem;">
                                    {(!location.is_empty()).then(|| view! { <span>{location}</span> })}
                                    {needed.map(|n| view! { <span>{format!("{fulfilled}/{n} SAP fulfilled")}</span> })}
                                </div>
                                {(r.status == AidRequestStatus::Open).then(|| view! {
                                    <button class="form-submit" style="margin-top: 0.5rem;">"Offer Help"</button>
                                })}
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == CommonsView::Housing>
                <div class="thought-card">
                    <h3 class="section-title">"Housing Units by Type"</h3>
                    <BarChart data=vec![
                        Bar { label: "1BR".into(), value: 35.0, color: "#059669".into() },
                        Bar { label: "2BR".into(), value: 48.0, color: "#34D399".into() },
                        Bar { label: "3BR".into(), value: 32.0, color: "#059669".into() },
                        Bar { label: "Studio".into(), value: 15.0, color: "#6ee7b7".into() },
                        Bar { label: "Access.".into(), value: 12.0, color: "#10b981".into() },
                    ] width=350.0 height=180.0 />
                    <p style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;">"8 units available \u{00b7} 3 under maintenance \u{00b7} 131 occupied"</p>
                </div>
            </Show>

            <Show when=move || active_view.get() == CommonsView::Water>
                <div class="thought-card">
                    <h3 class="section-title">"Water Sources"</h3>
                    <BarChart data=vec![
                        Bar { label: "Municipal".into(), value: 85.0, color: "#06b6d4".into() },
                        Bar { label: "Well".into(), value: 60.0, color: "#22d3ee".into() },
                        Bar { label: "Rainwater".into(), value: 40.0, color: "#67e8f9".into() },
                        Bar { label: "Recycled".into(), value: 25.0, color: "#a5f3fc".into() },
                    ] width=350.0 height=160.0 />
                    <p style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;">"Total capacity: 450,000 L/day \u{00b7} Current usage: 78%"</p>
                </div>
            </Show>

            <Show when=move || active_view.get() == CommonsView::Food>
                <div class="thought-card">
                    <h3 class="section-title">"Food Distribution Markets"</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">"3 active markets \u{00b7} 47 listings \u{00b7} 12 producers"</p>
                    <BarChart data=vec![
                        Bar { label: "Farmers".into(), value: 23.0, color: "#f59e0b".into() },
                        Bar { label: "CSA".into(), value: 15.0, color: "#fbbf24".into() },
                        Bar { label: "Co-op".into(), value: 9.0, color: "#fcd34d".into() },
                    ] width=280.0 height=140.0 />
                </div>
            </Show>
        </div>
    }
}
